"""Latent-length normalization helpers for batch preparation."""

import math

import torch


def _as_2d_latent(latent: torch.Tensor, *, name: str) -> torch.Tensor:
    """Return a ``[T, D]`` latent tensor from ``[T, D]`` or ``[1, T, D]`` input."""
    if latent.dim() == 2:
        return latent
    if latent.dim() == 3 and latent.shape[0] == 1:
        return latent[0]
    raise ValueError(f"{name} must have shape [T, D] or [1, T, D], got {tuple(latent.shape)}")


def build_silence_latent(
    silence_latent: torch.Tensor,
    length: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build a silence latent of exact ``length`` by slicing/repeating a base latent."""
    if length < 0:
        raise ValueError(f"length must be >= 0, got {length}")

    silence_2d = _as_2d_latent(silence_latent, name="silence_latent")
    base_length = silence_2d.shape[0]
    if base_length == 0:
        raise ValueError("silence_latent must have at least one time step")

    if length == 0:
        output = silence_2d[:0]
    elif length <= base_length:
        output = silence_2d[:length]
    else:
        repeat_count = math.ceil(length / base_length)
        output = silence_2d.repeat(repeat_count, 1)[:length]

    if device is not None or dtype is not None:
        output = output.to(device=device if device is not None else output.device, dtype=dtype or output.dtype)
    return output


def normalize_latent_length(
    latent: torch.Tensor,
    target_length: int,
    silence_latent: torch.Tensor,
) -> torch.Tensor:
    """Crop/pad latent to ``target_length`` using repeated silence padding."""
    if target_length < 0:
        raise ValueError(f"target_length must be >= 0, got {target_length}")

    latent_2d = _as_2d_latent(latent, name="latent")
    if latent_2d.shape[0] >= target_length:
        return latent_2d[:target_length]

    pad_length = target_length - latent_2d.shape[0]
    pad = build_silence_latent(
        silence_latent,
        pad_length,
        device=latent_2d.device,
        dtype=latent_2d.dtype,
    )
    if pad.shape[-1] != latent_2d.shape[-1]:
        raise ValueError(
            "silence_latent feature size mismatch: "
            f"latent has D={latent_2d.shape[-1]}, silence has D={pad.shape[-1]}"
        )
    return torch.cat([latent_2d, pad], dim=0)
