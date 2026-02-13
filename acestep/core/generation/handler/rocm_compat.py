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


def force_rocm_quantizer_project_out_fp32(model: torch.nn.Module) -> bool:
    """Force tokenizer quantizer ``project_out`` to ``float32`` when present.

    On some ROCm stacks, ``ResidualFSQ.forward`` can emit ``float32`` activations
    while ``project_out`` stays in ``float16``, causing ``mat1/mat2`` dtype
    mismatch at runtime. This helper upgrades that linear layer to ``float32``.

    Args:
        model: Loaded DiT model instance.

    Returns:
        ``True`` if dtype was changed, ``False`` when no change was needed or
        the expected tokenizer path does not exist.
    """
    root_model = getattr(model, "_orig_mod", model)
    tokenizer = getattr(root_model, "tokenizer", None)
    quantizer = getattr(tokenizer, "quantizer", None)
    project_out = getattr(quantizer, "project_out", None)
    weight = getattr(project_out, "weight", None)

    if project_out is None or weight is None:
        return False
    if weight.dtype == torch.float32:
        return False

    project_out.to(dtype=torch.float32)
    return True


def install_rocm_detokenizer_input_cast_hook(model: torch.nn.Module) -> bool:
    """Install a hook that casts detokenizer input to its floating parameter dtype.

    After forcing tokenizer ``project_out`` to ``float32``, some model paths feed
    ``float32`` tensors into a ``float16`` detokenizer and fail in Linear layers.
    This hook keeps detokenizer inputs aligned with its own parameter dtype.

    Args:
        model: Loaded DiT model instance.

    Returns:
        ``True`` if a new hook was installed, ``False`` otherwise.
    """
    root_model = getattr(model, "_orig_mod", model)
    detokenizer = getattr(root_model, "detokenizer", None)
    if detokenizer is None or not isinstance(detokenizer, torch.nn.Module):
        return False
    if hasattr(detokenizer, "_acestep_rocm_input_cast_hook"):
        return False

    target_dtype = None
    for parameter in detokenizer.parameters():
        if parameter.is_floating_point():
            target_dtype = parameter.dtype
            break
    if target_dtype is None:
        return False

    def _cast_input(_module, inputs):
        if not inputs:
            return None
        first = inputs[0]
        if (
            torch.is_tensor(first)
            and first.is_floating_point()
            and first.dtype != target_dtype
        ):
            return (first.to(dtype=target_dtype), *inputs[1:])
        return None

    hook_handle = detokenizer.register_forward_pre_hook(_cast_input)
    detokenizer._acestep_rocm_input_cast_hook = hook_handle
    return True
