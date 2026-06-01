"""Loss functions shared across models.

Reconstruction MSE backs the autoencoder baselines, the KL term backs the
VAE variant, and the evolutionary adversarial loss backs the TranAD
wrapper. All operate on batched windows of shape ``[B, K, m]``, batch by
window length by channels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruction_mse(
    x: torch.Tensor, x_hat: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """Per-channel MSE reconstruction loss.

    Parameters
    ----------
    x, x_hat : torch.Tensor  [B, K, m]
    reduction : ``"mean"`` | ``"none"``
        If ``"none"`` returns shape ``[B, m]``.
    """
    err = (x - x_hat).pow(2)
    if reduction == "none":
        return err.mean(dim=1)  # average over time -> [B, m]
    return err.mean()


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Standard KL term for VAEs: -0.5 * sum(1 + log(σ²) - μ² - σ²)."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def adversarial_loss(
    o1: torch.Tensor, o2: torch.Tensor, x: torch.Tensor, epoch: int, n_epochs: int
) -> torch.Tensor:
    """TranAD-style evolutionary adversarial loss.

    Early epochs weight reconstruction more; later epochs weight the
    adversarial component.

    Parameters
    ----------
    o1, o2 : torch.Tensor  [B, K, m]
        First-phase and second-phase decoder outputs.
    x : torch.Tensor  [B, K, m]
        Input window the outputs reconstruct.
    epoch, n_epochs : int
        Current epoch and total epoch budget, drive the schedule weight.
    """
    # alpha ramps from 0 to 1 over the first half of training, then saturates.
    alpha = min(epoch / max(n_epochs // 2, 1), 1.0)
    l1 = (1 - alpha) * reconstruction_mse(x, o1) + alpha * reconstruction_mse(x, o2)
    return l1
