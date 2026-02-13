"""ROCm compatibility helpers for DiT service initialization.

This module centralizes runtime checks that differ between CUDA and ROCm
so initialization code can stay small and deterministic.
"""

from typing import List

import torch


def is_rocm_cuda_device(device: str) -> bool:
    """Return ``True`` when the selected device is CUDA backed by ROCm.

    Args:
        device: Device string such as ``"cuda"``, ``"cuda:0"``, or ``"cpu"``.

    Returns:
        ``True`` if the device resolves to CUDA and the active PyTorch build
        exposes a HIP runtime version.
    """
    device_type = str(device).split(":", 1)[0]
    return device_type == "cuda" and getattr(torch.version, "hip", None) is not None


def choose_service_dtype(device: str) -> torch.dtype:
    """Choose a stable default dtype for service initialization.

    Args:
        device: Device string selected for DiT inference.

    Returns:
        Preferred dtype for that runtime backend.

    Notes:
        ROCm uses ``float16`` by default here because ``bfloat16`` model load
        can be unstable on some ROCm stacks.
    """
    device_type = str(device).split(":", 1)[0]
    if device_type == "cuda":
        return torch.float16 if is_rocm_cuda_device(device) else torch.bfloat16
    if device_type == "xpu":
        return torch.bfloat16
    return torch.float32


def build_attention_candidates(
    use_flash_attention: bool,
    flash_attention_available: bool,
    is_rocm_cuda: bool,
) -> List[str]:
    """Build ordered attention implementation candidates for model loading.

    Args:
        use_flash_attention: Whether the user requested FlashAttention.
        flash_attention_available: Whether FlashAttention is available.
        is_rocm_cuda: Whether the runtime is CUDA-on-ROCm.

    Returns:
        Ordered list of attention implementations to try.
    """
    if use_flash_attention and flash_attention_available:
        candidates = ["flash_attention_2"]
    elif is_rocm_cuda:
        # ROCm path: prefer eager first, then SDPA fallback.
        candidates = ["eager"]
    else:
        candidates = ["sdpa"]

    if is_rocm_cuda:
        for impl in ("eager", "sdpa"):
            if impl not in candidates:
                candidates.append(impl)
    else:
        for impl in ("sdpa", "eager"):
            if impl not in candidates:
                candidates.append(impl)

    return candidates


def should_rocm_direct_model_load(
    is_rocm_cuda: bool,
    offload_to_cpu: bool,
    offload_dit_to_cpu: bool,
) -> bool:
    """Return ``True`` if DiT should be loaded directly onto ROCm device.

    Args:
        is_rocm_cuda: Whether runtime is CUDA-on-ROCm.
        offload_to_cpu: Global CPU offload flag.
        offload_dit_to_cpu: DiT-specific CPU offload flag.

    Returns:
        ``True`` when ROCm is active and DiT is intended to stay resident on
        GPU after initialization.
    """
    if not is_rocm_cuda:
        return False
    if not offload_to_cpu:
        return True
    return not offload_dit_to_cpu
