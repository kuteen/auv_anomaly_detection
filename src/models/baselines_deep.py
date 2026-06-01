"""Neural-network anomaly detectors used in the benchmark.

This module intentionally groups every deep-learning model in one place.
Classical detectors live elsewhere, here we collect only the deep models
and the building blocks they share.

The file is laid out in the following sections.

* Shared reconstruction scoring, used by the autoencoder-style detectors.
* Positional encodings.
* LSTM autoencoder.
* 1-D CNN autoencoder.
* TranAD, an adversarial transformer reconstructor.
* STGNN spatial operators (graph convolution, attention, SAGE, GATv2).
* STGNN temporal operators (transformer, RNN, GRU, LSTM).
* The STGNN model itself.

Every detector wraps an inner ``nn.Module`` and exposes the common
``AnomalyDetector`` interface (``fit``, ``score``, ``save``, ``load``).
Most detectors are trained to reconstruct the input window, the squared
reconstruction error then serves as the anomaly score. The simpler
autoencoders defer to the shared trainer, whereas TranAD and the STGNN
own their training loops because they need extra state, the adversarial
epoch schedule and the channel adjacency respectively.

The STGNN is deliberately modular. Its spatial operator and its temporal
operator are both selectable, so the same model covers a family of
graph and sequence backbones from one config.

Tensor-shape conventions used throughout.

* ``B`` batch size.
* ``T`` or ``K`` window length in timesteps.
* ``C`` or ``m`` number of channels, equivalently graph nodes.
* ``N`` total number of scored windows across a loader.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.base import AnomalyDetector, ScoreResult
from models.losses import adversarial_loss, reconstruction_mse
from training.early_stopping import EarlyStopping
from utils.runtime import configure_runtime
from utils.terminal import ProgressBar

logger = logging.getLogger(__name__)


# ── Shared Reconstruction Scoring ───────────────────────────────────────


def _torch_load_checkpoint(path: str) -> Any:
    """Load a torch checkpoint while preferring weights-only deserialisation."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _score_reconstruction(module: nn.Module, loader: DataLoader) -> ScoreResult:
    """Run an autoencoder over a loader and turn reconstruction error into scores.

    The module is expected to map a window ``[B, T, C]`` to a reconstruction
    of the same shape. Per-element squared error is averaged over time to give
    a per-channel score and averaged again to give a single global score.

    Returns
    -------
    ScoreResult
        ``scores_time_channel`` is ``[N, C]``, ``scores_time_global`` is
        ``[N]`` and ``reconstruction`` is the time-averaged reconstruction.
    """
    module.eval()
    device = next(module.parameters()).device
    all_scores, all_recon = [], []
    non_blocking = device.type == "cuda"
    with torch.inference_mode():
        for batch in loader:
            # Loaders may yield bare tensors or ``(input, label)`` tuples.
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device, non_blocking=non_blocking)
            recon = module(batch)
            err = (batch - recon).pow(2)
            # Collapse the time axis (dim 1) to keep a per-channel error vector.
            all_scores.append(err.mean(dim=1).cpu().numpy())
            all_recon.append(recon.mean(dim=1).cpu().numpy())
    scores_tc = np.concatenate(all_scores, axis=0)
    recon_tc = np.concatenate(all_recon, axis=0)
    scores_global = scores_tc.mean(axis=1)
    return ScoreResult(
        scores_time_channel=scores_tc,
        scores_time_global=scores_global,
        reconstruction=recon_tc,
    )


# ── LSTM Autoencoder ────────────────────────────────────────────────────


class _LSTMAEModule(nn.Module):
    """Inner sequence-to-sequence LSTM autoencoder.

    The encoder LSTM compresses the window, its final hidden and cell states
    seed the decoder LSTM, and a linear head projects back to the channel
    space. Reconstruction error of the channel signals drives the score.

    Parameters
    ----------
    n_channels : int
        Number of input channels ``C``.
    hidden_dim : int
        Width of the LSTM hidden state.
    n_layers : int
        Stacked LSTM layers. Dropout between layers is only active when
        ``n_layers > 1`` because PyTorch ignores it for a single layer.
    dropout : float
        Inter-layer dropout probability.
    """

    def __init__(self, n_channels: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.encoder = nn.LSTM(
            n_channels,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.decoder = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input window. ``x`` is ``[B, T, C]``, output matches."""
        enc_out, (h, c) = self.encoder(x)
        # Re-use the encoder's final states to prime the decoder.
        dec_out, _ = self.decoder(enc_out, (h, c))
        return self.fc(dec_out)


class LSTMAutoencoder(AnomalyDetector):
    """LSTM autoencoder detector exposing the benchmark interface.

    Wraps :class:`_LSTMAEModule` and scores anomalies by reconstruction error.
    The constructor stores defaults, the actual module is rebuilt from the
    run config in :meth:`fit` so saved configs are the single source of truth.
    """

    name = "lstm_ae"

    def __init__(
        self,
        n_channels: int = 19,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        self._n_channels = n_channels
        self._module = _LSTMAEModule(n_channels, hidden_dim, n_layers, dropout)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Rebuild the module from the config and train it, returning train stats."""
        from training.train_loop import train_model

        self._module = _LSTMAEModule(
            self._n_channels,
            config["model"].get("hidden_dim", 64),
            config["model"].get("n_layers", 2),
            config["model"].get("dropout", 0.1),
        )
        return train_model(self._module, train_loader, val_loader, config)

    def score(self, test_loader: DataLoader, config: Dict[str, Any]) -> ScoreResult:
        """Score the loader by per-channel and global reconstruction error."""
        return _score_reconstruction(self._module, test_loader)

    def save(self, path: str) -> None:
        """Save the inner module state dict to ``path``."""
        torch.save(self._module.state_dict(), path)

    def load(self, path: str) -> None:
        """Load the inner module state dict from ``path``."""
        self._module.load_state_dict(_torch_load_checkpoint(path))


# ── 1-D CNN Autoencoder ────────────────────────────────────────────────


class _CNNAEModule(nn.Module):
    """Inner 1-D convolutional autoencoder over the channel axis.

    The encoder applies two padded convolutions that keep the time length
    fixed while squeezing the channel width down to a bottleneck, the decoder
    mirrors them back to the channel space. Reconstruction error drives the
    score.

    Parameters
    ----------
    n_channels : int
        Number of input channels ``C``, used as the convolution channel count.
    hidden_dim : int
        Width of the first convolutional feature map, halved at the bottleneck.
    """

    def __init__(self, n_channels: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, n_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input window. ``x`` is ``[B, T, C]``, output matches."""
        # Conv1d expects channels on dim 1, so swap time and channel axes,
        # run the encoder/decoder, then swap back to the [B, T, C] convention.
        z = self.encoder(x.transpose(1, 2))
        return self.decoder(z).transpose(1, 2)


class CNNAutoencoder(AnomalyDetector):
    """1-D CNN autoencoder detector exposing the benchmark interface.

    Wraps :class:`_CNNAEModule` and scores anomalies by reconstruction error.
    As with the other autoencoders the module is rebuilt from the run config
    in :meth:`fit` so the saved config remains the single source of truth.
    """

    name = "cnn_ae"

    def __init__(self, n_channels: int = 19, hidden_dim: int = 64):
        self._n_channels = n_channels
        self._module = _CNNAEModule(n_channels, hidden_dim)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Rebuild the module from the config and train it, returning train stats."""
        from training.train_loop import train_model

        self._module = _CNNAEModule(
            self._n_channels,
            config["model"].get("hidden_dim", 64),
        )
        return train_model(self._module, train_loader, val_loader, config)

    def score(self, test_loader: DataLoader, config: Dict[str, Any]) -> ScoreResult:
        """Score the loader by per-channel and global reconstruction error."""
        return _score_reconstruction(self._module, test_loader)

    def save(self, path: str) -> None:
        """Save the inner module state dict to ``path``."""
        torch.save(self._module.state_dict(), path)

    def load(self, path: str) -> None:
        """Load the inner module state dict from ``path``."""
        self._module.load_state_dict(_torch_load_checkpoint(path))


# ── TranAD ─────────────────────────────────────────────────────────────


class _TranADPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as used by the TranAD reference model.

    Follows the TranAD convention where the sine and cosine terms are summed
    into a single table rather than interleaved, the table is registered as a
    non-trainable buffer of shape ``[1, max_len, d_model]``.

    Parameters
    ----------
    d_model : int
        Embedding width the encoding is added to.
    dropout : float
        Dropout applied after the position is added.
    max_len : int
        Longest sequence the table covers, capping the supported window length.
    """

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor, pos: int = 0) -> torch.Tensor:
        """Add the encoding to ``x`` ``[B, T, d_model]`` from offset ``pos``."""
        # Slice the cached table to the current sequence length before adding.
        return self.dropout(x + self.pe[:, pos : pos + x.size(1), :])


class _TranADEncoderLayer(nn.Module):
    """Pre-built transformer encoder layer with batch-first self-attention.

    A hand-rolled equivalent of :class:`torch.nn.TransformerEncoderLayer` that
    fixes the signature TranAD expects, self-attention then a feedforward block,
    each wrapped in a residual connection and post layer norm.

    Parameters
    ----------
    d_model : int
        Token embedding width.
    n_heads : int
        Number of attention heads.
    dim_feedforward : int
        Hidden width of the feedforward block.
    dropout : float
        Dropout applied to attention and feedforward outputs.
    """

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Self-attend over ``src`` then apply the feedforward block.

        ``src`` is ``[B, T, d_model]`` and the output keeps that shape. The mask
        arguments mirror the PyTorch signature so the layer is interchangeable.
        """
        attn_output = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )[0]
        src = self.norm1(src + self.dropout1(attn_output))
        ffn_output = self.ffn(src)
        src = self.norm2(src + self.dropout2(ffn_output))
        return src


class _TranADDecoderLayer(nn.Module):
    """Pre-built transformer decoder layer with batch-first attention.

    A hand-rolled equivalent of :class:`torch.nn.TransformerDecoderLayer`,
    masked self-attention over the target, then cross-attention into the encoder
    memory, then a feedforward block, each residual with post layer norm.

    Parameters
    ----------
    d_model : int
        Token embedding width.
    n_heads : int
        Number of attention heads for both attention blocks.
    dim_feedforward : int
        Hidden width of the feedforward block.
    dropout : float
        Dropout applied to each sub-block output.
    """

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.multihead_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Decode ``tgt`` against encoder ``memory``.

        ``tgt`` and ``memory`` are ``[B, T, d_model]`` and the output keeps the
        target shape. The mask arguments mirror the PyTorch signature so the
        layer is interchangeable.
        """
        masked_attn_output = self.self_attn(
            tgt,
            tgt,
            tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )[0]
        tgt = self.norm1(tgt + self.dropout1(masked_attn_output))
        attn_output = self.multihead_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )[0]
        tgt = self.norm2(tgt + self.dropout2(attn_output))
        ffn_output = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout3(ffn_output))
        return tgt


class _TranADModule(nn.Module):
    """Inner TranAD adversarial transformer reconstructor.

    A shared encoder feeds two decoders. The first decoder produces a coarse
    reconstruction, its squared error becomes a focus score that is fed back as
    context for a second encode-decode pass, which yields the refined output
    used for scoring. Adversarial training pushes the two decoders apart during
    early epochs and together later.

    Parameters
    ----------
    n_channels : int
        Number of input channels ``C``, also the embedding scale factor.
    hidden_dim : int
        Transformer model width.
    n_heads : int
        Attention heads per layer.
    n_layers : int
        Number of stacked encoder and decoder layers.
    window_length : int
        Window length ``T``, used to size the positional encoding table.
    dropout : float
        Dropout shared across attention, feedforward, and positional encoding.
    output_min, output_max : float
        Bounds the sigmoid output is rescaled into, matching the data
        normalisation range so reconstructions live in the input domain.
    """

    def __init__(
        self,
        n_channels: int,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        window_length: int,
        dropout: float,
        output_min: float,
        output_max: float,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.output_min = float(output_min)
        self.output_max = float(output_max)
        self.embedding = nn.Linear(n_channels, hidden_dim)
        # Combiner fuses the window embedding with its focus context, hence 2x width in.
        self.combiner = nn.Linear(2 * hidden_dim, hidden_dim)
        self.pos_encoder = _TranADPositionalEncoding(hidden_dim, dropout, max_len=window_length)
        self.encoder_layers = nn.ModuleList(
            [
                _TranADEncoderLayer(hidden_dim, n_heads, hidden_dim * 4, dropout)
                for _ in range(n_layers)
            ]
        )
        self.decoder1_layers = nn.ModuleList(
            [
                _TranADDecoderLayer(hidden_dim, n_heads, hidden_dim * 4, dropout)
                for _ in range(n_layers)
            ]
        )
        self.decoder2_layers = nn.ModuleList(
            [
                _TranADDecoderLayer(hidden_dim, n_heads, hidden_dim * 4, dropout)
                for _ in range(n_layers)
            ]
        )
        self.output_proj = nn.Linear(hidden_dim, n_channels)
        self.output_activation = nn.Sigmoid()

    def encode(
        self,
        src: torch.Tensor,
        context: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed and encode a window, returning the target embedding and memory.

        ``src``, ``context`` and ``target`` are all ``[B, T, C]``. The window and
        its focus context are embedded, concatenated and combined, then run
        through the encoder stack to give the memory the decoders attend to.
        """
        src_emb = self.embedding(src)
        context_emb = self.embedding(context)
        # Fuse the window with its focus context along the feature axis.
        combined = torch.cat([src_emb, context_emb], dim=-1)
        src_hidden = self.combiner(combined)
        # Scale embeddings before adding positions, the TranAD scaling convention.
        src_hidden = self.pos_encoder(src_hidden * math.sqrt(self.n_channels))
        memory = src_hidden
        for layer in self.encoder_layers:
            memory = layer(memory)
        target_emb = self.embedding(target)
        return target_emb, memory

    def _decode(self, decoder_layers: nn.ModuleList, target_emb: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Run a decoder stack and rescale its sigmoid output into the data range."""
        output = target_emb
        for layer in decoder_layers:
            output = layer(output, memory)
        output = self.output_proj(output)
        output = self.output_activation(output)
        # Map the [0, 1] sigmoid back onto the normalised input domain.
        return self.output_min + ((self.output_max - self.output_min) * output)

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Two-pass adversarial reconstruction of the window ``src`` ``[B, T, C]``.

        Returns
        -------
        tuple of torch.Tensor
            ``x1`` and ``x2`` are the first-pass reconstructions from the two
            decoders, ``o2`` is the refined second-pass output used for scoring,
            and ``c2`` is the focus score fed back between passes. All are
            ``[B, T, C]``.
        """
        target = src
        # First pass starts with an all-zero focus context.
        context1 = torch.zeros_like(src)
        target_phase1, memory1 = self.encode(src, context1, target)
        x1 = self._decode(self.decoder1_layers, target_phase1, memory1)
        x2 = self._decode(self.decoder2_layers, target_phase1, memory1)
        # Squared first-pass error becomes the focus context for the second pass.
        c2 = (x1 - src).pow(2)
        target_phase2, memory2 = self.encode(src, c2, target)
        o2 = self._decode(self.decoder2_layers, target_phase2, memory2)
        return x1, x2, o2, c2


class TranADWrapper(AnomalyDetector):
    """TranAD detector exposing the benchmark interface.

    Wraps :class:`_TranADModule` and owns its own adversarial training loop
    rather than the shared trainer, since TranAD needs the epoch index to
    schedule its adversarial weighting. Anomalies are scored by the squared
    error of the refined second-pass reconstruction.
    """

    name = "tranad"

    def __init__(
        self,
        n_channels: int = 19,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        window_length: int = 10,
        dropout: float = 0.1,
        output_min: float = 0.0,
        output_max: float = 1.0,
    ):
        self._n_channels = n_channels
        self._output_min = float(output_min)
        self._output_max = float(output_max)
        self._module = _TranADModule(
            n_channels,
            hidden_dim,
            n_heads,
            n_layers,
            window_length,
            dropout,
            self._output_min,
            self._output_max,
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Rebuild the module from the config and run the adversarial training loop.

        Returns
        -------
        dict
            ``best_val_loss`` and ``last_epoch``, the best validation loss seen
            and the epoch the loop finished on, early stopping included.
        """
        cfg_m = config["model"]
        cfg_t = config["training"]
        self._module = _TranADModule(
            self._n_channels,
            cfg_m.get("hidden_dim", 64),
            cfg_m.get("n_heads", 4),
            cfg_m.get("n_layers", 2),
            config["windowing"]["window_length"],
            cfg_m.get("dropout", 0.1),
            float(config["data_processing"].get("normalisation_min", self._output_min)),
            float(config["data_processing"].get("normalisation_max", self._output_max)),
        )
        runtime = configure_runtime(config)
        device = runtime.device
        self._module.to(device)
        optimiser = torch.optim.Adam(
            self._module.parameters(),
            lr=cfg_t.get("learning_rate", 1e-3),
            weight_decay=cfg_t.get("weight_decay", 1e-4),
        )
        es = EarlyStopping(cfg_t.get("patience", 10), cfg_t.get("min_delta", 1e-4))
        n_epochs = cfg_t.get("epochs", 50)
        best_loss = float("inf")
        progress = ProgressBar(total=n_epochs, desc="tranad epochs", unit="epoch", leave=False)

        for epoch in range(1, n_epochs + 1):
            self._module.train()
            epoch_loss = 0.0
            for batch in train_loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(device, non_blocking=runtime.non_blocking)
                x1, _x2, o2, _c2 = self._module(batch)
                # Adversarial weighting between the two passes is scheduled by epoch.
                loss = adversarial_loss(x1, o2, batch, epoch, n_epochs)
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._module.parameters(), cfg_t.get("gradient_clip", 1.0))
                optimiser.step()
                epoch_loss += loss.item()

            epoch_loss /= max(len(train_loader), 1)
            val_loss = self._validate(val_loader, device, epoch, n_epochs) if val_loader else epoch_loss
            if val_loss < best_loss:
                best_loss = val_loss
            progress.set_postfix_str(
                f"epoch={epoch}/{n_epochs} train={epoch_loss:.4f} val={val_loss:.4f}"
            )
            progress.update(1)
            if es.step(val_loss):
                progress.write(f"  TranAD early stop at epoch {epoch}")
                break

        progress.close()
        logger.info(
            "TranAD training complete: best_val_loss=%.6f after %d epoch(s)",
            best_loss,
            epoch,
        )

        return {"best_val_loss": best_loss, "last_epoch": epoch}

    def score(self, test_loader: DataLoader, config: Dict[str, Any]) -> ScoreResult:
        """Score the loader by squared error of the refined second-pass output.

        Returns a :class:`ScoreResult` with per-channel scores ``[N, C]``, a
        global score ``[N]`` and the time-averaged reconstruction.
        """
        self._module.eval()
        device = next(self._module.parameters()).device
        all_scores, all_recon = [], []
        non_blocking = device.type == "cuda"
        with torch.inference_mode():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(device, non_blocking=non_blocking)
                # Only the refined output o2 is used for scoring, the rest are training signals.
                _x1, _x2, o2, _c2 = self._module(batch)
                err = (batch - o2).pow(2).mean(dim=1)
                all_scores.append(err.cpu().numpy())
                all_recon.append(o2.mean(dim=1).cpu().numpy())
        scores_tc = np.concatenate(all_scores, axis=0)
        scores_global = scores_tc.mean(axis=1)
        recon = np.concatenate(all_recon, axis=0)
        return ScoreResult(scores_time_channel=scores_tc, scores_time_global=scores_global, reconstruction=recon)

    def _validate(
        self,
        loader: Optional[DataLoader],
        device: torch.device,
        epoch: int,
        n_epochs: int,
    ) -> float:
        """Return the mean adversarial loss over ``loader``, or infinity if absent."""
        if loader is None:
            return float("inf")
        self._module.eval()
        total = 0.0
        non_blocking = device.type == "cuda"
        with torch.inference_mode():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(device, non_blocking=non_blocking)
                x1, _x2, o2, _c2 = self._module(batch)
                total += adversarial_loss(x1, o2, batch, epoch, n_epochs).item()
        return total / max(len(loader), 1)

    def save(self, path: str) -> None:
        """Save the inner module state dict to ``path``."""
        torch.save(self._module.state_dict(), path)

    def load(self, path: str) -> None:
        """Load the inner module state dict from ``path``."""
        self._module.load_state_dict(_torch_load_checkpoint(path))


# ── STGNN Spatial Operators ─────────────────────────────────────────────



class GraphConv(nn.Module):
    """First-order spectral graph convolution (Kipf and Welling).

    Each node is linearly projected, then features are mixed across nodes by
    the normalised adjacency, the standard ``A_norm @ (x @ W)`` propagation.

    Parameters
    ----------
    in_features : int
        Input feature width per node.
    out_features : int
        Output feature width per node.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.empty(in_features, out_features))
        self.b = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        """Propagate node features ``x`` ``[B, N, in]`` over ``A_norm`` ``[N, N]``.

        Returns the convolved features ``[B, N, out]``.
        """
        # Project per node first, then spread the projected features over neighbours.
        support = torch.matmul(x, self.W)
        return torch.matmul(A_norm, support) + self.b


class GraphAttention(nn.Module):
    """Single-head graph attention layer (Velickovic et al.).

    Attention coefficients are learned over each ordered node pair, masked to
    the graph edges, normalised with a softmax over neighbours, then used to
    aggregate the projected node features.

    Parameters
    ----------
    in_features : int
        Input feature width per node.
    out_features : int
        Output feature width per node.
    dropout : float
        Dropout applied to the attention weights.
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.empty(2 * out_features, 1))
        nn.init.xavier_uniform_(self.a)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        """Attend over neighbours given ``x`` ``[B, N, in]`` and ``A_norm`` ``[N, N]``.

        Returns the attended features ``[B, N, out]``.
        """
        batch_size, n_nodes, _ = x.shape
        h = self.W(x)
        # Build every ordered (i, j) feature pair by broadcasting along two new axes.
        a_input = torch.cat(
            [
                h.unsqueeze(2).expand(-1, -1, n_nodes, -1),
                h.unsqueeze(1).expand(-1, n_nodes, -1, -1),
            ],
            dim=-1,
        )
        e = self.leaky_relu(torch.matmul(a_input, self.a).squeeze(-1))
        # Non-edges get -inf so the softmax assigns them zero attention.
        e = e.masked_fill((A_norm == 0).unsqueeze(0), float("-inf"))
        alpha = F.softmax(e, dim=-1)
        alpha = self.dropout(alpha)
        # Isolated nodes have an all -inf row, softmax gives NaN there, so zero it.
        alpha = torch.nan_to_num(alpha, 0.0)
        return torch.matmul(alpha, h)


class GraphSAGE(nn.Module):
    """GraphSAGE convolution with a configurable neighbour aggregator.

    Projects node features, aggregates neighbour features by mean, sum or max
    over the adjacency, then adds a self connection. With no adjacency it
    degrades to a plain per-node projection.

    Parameters
    ----------
    in_features : int
        Input feature width per node.
    out_features : int
        Output feature width per node.
    aggregator : str
        Neighbour aggregation, one of ``"mean"``, ``"sum"`` or ``"max"``.
        Any other value falls back to the matrix-multiply aggregation.
    dropout : float
        Dropout layer width, retained for parity with the other operators.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        aggregator: str = "mean",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.aggregator = aggregator
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None, A_norm: torch.Tensor = None) -> torch.Tensor:
        """Aggregate neighbours of ``x`` ``[B, N, in]`` and add the self term.

        ``edge_index`` is accepted for interface parity and unused, propagation
        uses the dense ``A_norm`` ``[N, N]``. Returns features ``[B, N, out]``.
        """
        batch_size, n_nodes, channels = x.shape
        h = self.W(x)
        if A_norm is None:
            return h
        if self.aggregator in {"mean", "sum"}:
            out = torch.matmul(A_norm.unsqueeze(0), h)
        elif self.aggregator == "max":
            # Mask non-neighbours to -inf so they cannot win the elementwise max.
            h_expanded = h.unsqueeze(1).expand(-1, n_nodes, -1, -1)
            mask = (A_norm == 0).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, channels)
            h_expanded = h_expanded.masked_fill(mask, float("-inf"))
            out, _ = h_expanded.max(dim=2)
        else:
            out = torch.matmul(A_norm.unsqueeze(0), h)
        # Residual self connection keeps each node's own features.
        return out + h


class GATv2(nn.Module):
    """Multi-head GATv2 attention layer (Brody et al.) with a residual.

    GATv2 applies the non-linearity before the attention scoring vector, which
    makes the attention dynamic rather than static. Heads are computed in
    parallel and concatenated, and a linear residual is added to the output.

    Parameters
    ----------
    in_features : int
        Input feature width per node.
    out_features : int
        Output feature width per node, must divide evenly by ``heads``.
    heads : int
        Number of attention heads.
    dropout : float
        Dropout applied to the attention weights.
    concat : bool
        Kept for interface parity, the heads are always concatenated here.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.concat = concat
        assert out_features % heads == 0, "out_features must be divisible by heads"
        self.head_dim = out_features // heads
        self.W = nn.Linear(in_features, heads * self.head_dim, bias=False)
        self.att = nn.Linear(2 * self.head_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.residual = nn.Linear(in_features, out_features if concat else self.head_dim)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.att.weight)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor = None) -> torch.Tensor:
        """Multi-head attend over ``x`` ``[B, N, in]``, optionally masked by ``A_norm``.

        Returns the attended features plus residual ``[B, N, out]``.
        """
        batch_size, n_nodes, _ = x.shape
        # Project then split the feature axis into separate attention heads.
        h = self.W(x).view(batch_size, n_nodes, self.heads, self.head_dim)
        # Form every ordered (i, j) head-wise pair for the GATv2 scoring function.
        h_i = h.unsqueeze(2).expand(-1, -1, n_nodes, -1, -1)
        h_j = h.unsqueeze(1).expand(-1, n_nodes, -1, -1, -1)
        h_cat = torch.cat([h_i, h_j], dim=-1)
        e = self.leaky_relu(self.att(h_cat).squeeze(-1))
        if A_norm is not None:
            # Mask non-edges per head before the softmax over source nodes.
            mask = (A_norm == 0).unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, self.heads)
            e = e.masked_fill(mask, float("-inf"))
        # Softmax over dim 2 normalises attention across the source node j.
        alpha = F.softmax(e, dim=2)
        alpha = self.dropout(alpha)
        # Isolated nodes give all -inf rows, softmax yields NaN, so zero them.
        alpha = torch.nan_to_num(alpha, 0.0)
        h_expanded = h.unsqueeze(2).expand(-1, -1, n_nodes, -1, -1)
        # Weight neighbour features, sum over source, then flatten the heads back.
        out = (alpha.unsqueeze(-1) * h_expanded).sum(dim=2).reshape(batch_size, n_nodes, -1)
        return out + self.residual(x)


# ── STGNN Temporal Operators ────────────────────────────────────────────


class _STGNNPosEnc(nn.Module):
    """Interleaved sinusoidal positional encoding for the STGNN temporal stack.

    Uses the classic transformer layout where even feature indices carry the
    sine term and odd indices the cosine term, the table is a non-trainable
    buffer of shape ``[1, max_len, d_model]``.

    Parameters
    ----------
    d_model : int
        Embedding width the encoding is added to.
    max_len : int
        Longest sequence the table covers, capping the window length.
    dropout : float
        Dropout applied after the position is added.
    """

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the encoding to ``x`` ``[B, T, d_model]``, trimmed to its length."""
        return self.dropout(x + self.pe[:, : x.size(1)])


# ── STGNN Model ─────────────────────────────────────────────────────────


class _STGNNModule(nn.Module):
    """Inner spatio-temporal graph autoencoder.

    Each timestep is treated as a graph over the channels. A spatial operator
    mixes features across channels at every timestep, then a temporal operator
    models the sequence per channel, and a decoder reconstructs the scalar
    signal. The spatial and temporal operators are both selectable.

    Parameters
    ----------
    n_channels : int
        Number of channels ``m``, equivalently graph nodes.
    window_length : int
        Window length ``K`` in timesteps.
    hidden_dim : int
        Shared hidden width across the spatial, temporal and decoder blocks.
    n_heads : int
        Attention heads, used by the GATv2 and transformer operators.
    n_layers : int
        Depth of the temporal operator.
    dropout : float
        Dropout shared across the operators and positional encoding.
    graph_conv : str
        Spatial operator, one of ``"gcn"``, ``"gat"``, ``"gatv2"`` or
        ``"graphsage"``. Unknown values fall back to ``"gcn"``.
    temporal_mode : str
        Temporal operator, one of ``"transformer"``, ``"rnn"``, ``"gru"`` or
        ``"lstm"``.

    Raises
    ------
    ValueError
        If ``temporal_mode`` is not one of the supported values.
    """

    def __init__(
        self,
        n_channels: int,
        window_length: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        graph_conv: str = "gcn",
        temporal_mode: str = "transformer",
    ) -> None:
        super().__init__()
        self.m = n_channels
        self.K = window_length
        self.hidden_dim = hidden_dim
        self.temporal_mode = temporal_mode
        self.spatial_in = nn.Linear(1, hidden_dim)
        # Select the spatial operator, defaulting to plain graph convolution.
        if graph_conv == "gat":
            self.gcn = GraphAttention(hidden_dim, hidden_dim, dropout)
        elif graph_conv == "gatv2":
            self.gcn = GATv2(hidden_dim, hidden_dim, n_heads, dropout)
        elif graph_conv == "graphsage":
            self.gcn = GraphSAGE(hidden_dim, hidden_dim, aggregator="mean", dropout=dropout)
        else:
            self.gcn = GraphConv(hidden_dim, hidden_dim)
        self.pos_enc = _STGNNPosEnc(hidden_dim, max_len=window_length, dropout=dropout)

        if temporal_mode == "transformer":
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.temporal_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        elif temporal_mode == "rnn":
            self.temporal_enc = nn.RNN(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                nonlinearity="tanh",
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
        elif temporal_mode == "gru":
            self.temporal_enc = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
        elif temporal_mode == "lstm":
            self.temporal_enc = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
        else:
            raise ValueError(
                f"Unsupported temporal_mode '{temporal_mode}'. "
                "Choose from ['transformer', 'rnn', 'gru', 'lstm']."
            )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        """Reconstruct the window ``x`` ``[B, K, m]`` over adjacency ``A_norm`` ``[m, m]``.

        The input is folded so that the spatial operator sees every timestep as a
        batched graph, then refolded so the temporal operator sees every channel
        as a batched sequence. Returns the reconstruction ``[B, K, m]``.
        """
        batch_size, window_length, n_nodes = x.shape
        # Flatten batch and time so each timestep is an independent graph of m nodes.
        x_flat = x.reshape(batch_size * window_length, n_nodes, 1)
        h = F.relu(self.spatial_in(x_flat))
        # Spatial message passing mixes features across channels at each timestep.
        h = F.relu(self.gcn(h, A_norm))
        h = h.reshape(batch_size, window_length, n_nodes, self.hidden_dim)
        # Move nodes ahead of time and flatten so each channel is its own sequence.
        h = h.permute(0, 2, 1, 3).reshape(batch_size * n_nodes, window_length, self.hidden_dim)
        if self.temporal_mode == "transformer":
            # The transformer is order blind, so inject positions first.
            h = self.pos_enc(h)
            h = self.temporal_enc(h)
        else:
            # The recurrent operators return (output, hidden), keep only the output.
            h, _ = self.temporal_enc(h)
        out = self.decoder(h).squeeze(-1)
        # Undo the channel-major fold to restore the [B, K, m] window layout.
        return out.reshape(batch_size, n_nodes, window_length).permute(0, 2, 1)


class SpatioTemporalGNN(AnomalyDetector):
    """Spatio-temporal GNN detector exposing the benchmark interface.

    Wraps :class:`_STGNNModule` and owns its own training loop so the channel
    adjacency can be threaded through every forward pass. Anomalies are scored
    by reconstruction error. When no adjacency is set before fitting a fully
    connected, normalised graph is used by default.
    """

    name = "graph_stgnn"

    def __init__(
        self,
        n_channels: int = 19,
        window_length: int = 10,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        graph_conv: str = "gcn",
        temporal_mode: str = "transformer",
    ) -> None:
        self._n_channels = n_channels
        self._graph_conv = graph_conv
        self._temporal_mode = temporal_mode
        self._module = _STGNNModule(
            n_channels,
            window_length,
            hidden_dim,
            n_heads,
            n_layers,
            dropout,
            graph_conv=graph_conv,
            temporal_mode=temporal_mode,
        )
        self._A_norm: Optional[torch.Tensor] = None

    def set_adjacency(self, A_norm: np.ndarray) -> None:
        """Set the normalised channel adjacency ``[m, m]`` used by every forward pass."""
        self._A_norm = torch.tensor(A_norm, dtype=torch.float32)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Rebuild the module from the config, resolve the adjacency, then train.

        Falls back to a fully connected normalised graph when no adjacency was
        set. Returns ``best_val_loss`` and ``last_epoch`` from the training loop.
        """
        cfg_m = config["model"]
        graph_conv = cfg_m.get("graph_conv", "gcn")
        temporal_mode = cfg_m.get("temporal_mode", "transformer")
        self._graph_conv = graph_conv
        self._temporal_mode = temporal_mode
        self._module = _STGNNModule(
            self._n_channels,
            config["windowing"]["window_length"],
            cfg_m.get("hidden_dim", 64),
            cfg_m.get("n_heads", 4),
            cfg_m.get("n_layers", 2),
            cfg_m.get("dropout", 0.1),
            graph_conv=graph_conv,
            temporal_mode=temporal_mode,
        )
        runtime = configure_runtime(config)
        device = runtime.device
        self._module.to(device)
        if self._A_norm is not None:
            self._A_norm = self._A_norm.to(device)
        else:
            # No graph supplied, default to a fully connected normalised adjacency.
            A = torch.ones(self._n_channels, self._n_channels, device=device)
            from data.graph_builders import normalise_adjacency
            self._A_norm = torch.tensor(
                normalise_adjacency(A.cpu().numpy()),
                dtype=torch.float32,
                device=device,
            )
        return self._train(train_loader, val_loader, config, device)

    def score(self, test_loader: DataLoader, config: Dict[str, Any]) -> ScoreResult:
        """Score the loader by per-channel and global reconstruction error.

        Returns a :class:`ScoreResult` with per-channel scores ``[N, m]``, a
        global score ``[N]`` and the time-averaged reconstruction.
        """
        self._module.eval()
        device = next(self._module.parameters()).device
        adjacency = self._A_norm.to(device)
        all_scores, all_recon = [], []
        non_blocking = device.type == "cuda"
        with torch.inference_mode():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(device, non_blocking=non_blocking)
                recon = self._module(batch, adjacency)
                err = (batch - recon).pow(2).mean(dim=1)
                all_scores.append(err.cpu().numpy())
                all_recon.append(recon.mean(dim=1).cpu().numpy())
        scores_tc = np.concatenate(all_scores, axis=0)
        scores_global = scores_tc.mean(axis=1)
        recon = np.concatenate(all_recon, axis=0)
        return ScoreResult(scores_time_channel=scores_tc, scores_time_global=scores_global, reconstruction=recon)

    def save(self, path: str) -> None:
        """Save the module state dict and the adjacency together to ``path``."""
        torch.save({"module": self._module.state_dict(), "A_norm": self._A_norm}, path)

    def load(self, path: str) -> None:
        """Load the module state dict and the adjacency from ``path``."""
        ckpt = _torch_load_checkpoint(path)
        self._module.load_state_dict(ckpt["module"])
        self._A_norm = ckpt["A_norm"]

    def _train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: Dict[str, Any],
        device: torch.device,
    ) -> Dict[str, Any]:
        """Run the reconstruction training loop with early stopping.

        Returns ``best_val_loss`` and ``last_epoch``, the best validation loss
        seen and the epoch the loop finished on.
        """
        cfg_t = config["training"]
        optimiser = torch.optim.Adam(
            self._module.parameters(),
            lr=cfg_t.get("learning_rate", 1e-3),
            weight_decay=cfg_t.get("weight_decay", 1e-4),
        )
        es = EarlyStopping(cfg_t.get("patience", 10), cfg_t.get("min_delta", 1e-4))
        n_epochs = cfg_t.get("epochs", 50)
        adjacency = self._A_norm.to(device)
        best_loss = float("inf")
        progress = ProgressBar(total=n_epochs, desc="stgnn epochs", unit="epoch", leave=False)

        for epoch in range(1, n_epochs + 1):
            self._module.train()
            total = 0.0
            for batch in train_loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(device, non_blocking=device.type == "cuda")
                recon = self._module(batch, adjacency)
                loss = reconstruction_mse(batch, recon)
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._module.parameters(), cfg_t.get("gradient_clip", 1.0))
                optimiser.step()
                total += loss.item()
            total /= max(len(train_loader), 1)

            val_loss = self._eval_loss(val_loader, device, adjacency) if val_loader else total
            if val_loss < best_loss:
                best_loss = val_loss
            progress.set_postfix_str(
                f"epoch={epoch}/{n_epochs} train={total:.4f} val={val_loss:.4f}"
            )
            progress.update(1)
            if es.step(val_loss):
                progress.write(f"  STGNN early stop at epoch {epoch}")
                break

        progress.close()
        logger.info(
            "STGNN training complete: best_val_loss=%.6f after %d epoch(s)",
            best_loss,
            epoch,
        )

        return {"best_val_loss": best_loss, "last_epoch": epoch}

    def _eval_loss(
        self,
        loader: Optional[DataLoader],
        device: torch.device,
        adjacency: torch.Tensor,
    ) -> float:
        """Return the mean reconstruction loss over ``loader``, or infinity if absent."""
        if loader is None:
            return float("inf")
        self._module.eval()
        total = 0.0
        non_blocking = device.type == "cuda"
        with torch.inference_mode():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(device, non_blocking=non_blocking)
                recon = self._module(batch, adjacency)
                total += reconstruction_mse(batch, recon).item()
        return total / max(len(loader), 1)
