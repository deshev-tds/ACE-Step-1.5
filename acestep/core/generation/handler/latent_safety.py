"""Numerical safety helpers for latent tensors."""

from __future__ import annotations

import torch


def count_non_finite(latents: torch.Tensor) -> int:
    """Return the number of NaN/Inf elements in ``latents``."""
    return int((~torch.isfinite(latents)).sum().item())


def sanitize_non_finite_latents(latents: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Replace NaN/Inf elements with zeros and return the cleaned tensor.

    Args:
        latents: Tensor containing latent values.

    Returns:
        A tuple ``(cleaned_latents, replaced_count)``.
    """
    replaced_count = count_non_finite(latents)
    if replaced_count == 0:
        return latents, 0
    cleaned = torch.nan_to_num(latents, nan=0.0, posinf=0.0, neginf=0.0)
    return cleaned, replaced_count


def all_non_finite_sample_mask(latents: torch.Tensor) -> torch.Tensor:
    """Return a per-sample mask for batch items that contain no finite values.

    Args:
        latents: Batched latent tensor shaped ``[B, ...]``.

    Returns:
        Boolean tensor of shape ``[B]`` where ``True`` marks samples that are
        entirely non-finite.
    """
    if latents.dim() == 0:
        return torch.tensor([not bool(torch.isfinite(latents).item())], device=latents.device)
    if latents.dim() == 1:
        finite_any = torch.isfinite(latents).any().unsqueeze(0)
        return ~finite_any
    finite_any = torch.isfinite(latents).reshape(latents.shape[0], -1).any(dim=1)
    return ~finite_any
