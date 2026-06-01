"""Generic PyTorch training loop for reconstruction-based models."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.early_stopping import EarlyStopping
from models.losses import reconstruction_mse
from utils.runtime import configure_runtime
from utils.terminal import ProgressBar

logger = logging.getLogger(__name__)


def train_model(
    module: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Standard training loop with early stopping and checkpoint saving.

    Parameters
    ----------
    module : nn.Module
        Must accept ``x`` of shape ``[B, K, m]`` and return the same shape.
    train_loader, val_loader : DataLoader
    config : dict
        Must contain ``training`` and ``output`` sections.

    Returns
    -------
    dict  –  ``{"best_val_loss": float, "last_epoch": int}``
    """
    cfg = config["training"]
    runtime = configure_runtime(config)
    device = runtime.device
    module.to(device)

    optimiser = torch.optim.Adam(
        module.parameters(),
        lr=cfg.get("learning_rate", 1e-3),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    es = EarlyStopping(cfg.get("patience", 10), cfg.get("min_delta", 1e-4))
    n_epochs = cfg.get("epochs", 50)
    clip = cfg.get("gradient_clip", 1.0)
    best_loss = float("inf")
    model_name = str(config.get("model", {}).get("name", "model"))
    progress = ProgressBar(total=n_epochs, desc=f"{model_name} epochs", unit="epoch", leave=False)

    for epoch in range(1, n_epochs + 1):
        module.train()
        epoch_loss = 0.0
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device, non_blocking=runtime.non_blocking)
            recon = module(batch)
            loss = reconstruction_mse(batch, recon)
            optimiser.zero_grad()
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(module.parameters(), clip)
            optimiser.step()
            epoch_loss += loss.item()
        epoch_loss /= max(len(train_loader), 1)

        # Validation
        val_loss = epoch_loss
        if val_loader is not None:
            module.eval()
            vl = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, (list, tuple)):
                        batch = batch[0]
                    batch = batch.to(device, non_blocking=runtime.non_blocking)
                    recon = module(batch)
                    vl += reconstruction_mse(batch, recon).item()
            val_loss = vl / max(len(val_loader), 1)

        if val_loss < best_loss:
            best_loss = val_loss

        progress.set_postfix_str(
            f"epoch={epoch}/{n_epochs} train={epoch_loss:.4f} val={val_loss:.4f}"
        )
        progress.update(1)

        if es.step(val_loss):
            progress.write(f"  Early stop at epoch {epoch} (val_loss={val_loss:.6f})")
            break

    progress.close()
    logger.info(
        "%s training complete: best_val_loss=%.6f after %d epoch(s)",
        model_name,
        best_loss,
        epoch,
    )

    return {"best_val_loss": best_loss, "last_epoch": epoch}
