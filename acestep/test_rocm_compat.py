"""Unit tests for ROCm compatibility helpers used by service init."""

import unittest
from unittest.mock import patch

import torch

from acestep.core.generation.handler.rocm_compat import (
    build_attention_candidates,
    choose_service_dtype,
    is_rocm_cuda_device,
    should_rocm_direct_model_load,
)


class TestRocmCompat(unittest.TestCase):
    """Validate ROCm-specific defaults and fallbacks."""

    @patch("acestep.core.generation.handler.rocm_compat.torch.version.hip", "6.4.0")
    def test_is_rocm_cuda_device_true_on_cuda(self):
        """CUDA device should be treated as ROCm when HIP runtime exists."""
        self.assertTrue(is_rocm_cuda_device("cuda"))
        self.assertTrue(is_rocm_cuda_device("cuda:0"))

    @patch("acestep.core.generation.handler.rocm_compat.torch.version.hip", None)
    def test_is_rocm_cuda_device_false_without_hip(self):
        """CUDA device is not ROCm when HIP runtime is absent."""
        self.assertFalse(is_rocm_cuda_device("cuda"))
        self.assertFalse(is_rocm_cuda_device("cpu"))

    @patch("acestep.core.generation.handler.rocm_compat.torch.version.hip", "6.4.0")
    def test_choose_service_dtype_rocm_uses_float16(self):
        """ROCm CUDA path should default to float16 for stability."""
        self.assertEqual(choose_service_dtype("cuda"), torch.float16)

    @patch("acestep.core.generation.handler.rocm_compat.torch.version.hip", None)
    def test_choose_service_dtype_non_rocm_cuda_uses_bfloat16(self):
        """CUDA path without ROCm should keep bfloat16 default."""
        self.assertEqual(choose_service_dtype("cuda"), torch.bfloat16)

    def test_choose_service_dtype_non_cuda_backends(self):
        """Non-CUDA backends should preserve existing defaults."""
        self.assertEqual(choose_service_dtype("xpu"), torch.bfloat16)
        self.assertEqual(choose_service_dtype("cpu"), torch.float32)
        self.assertEqual(choose_service_dtype("mps"), torch.float32)

    def test_build_attention_candidates_rocm_without_flash(self):
        """ROCm without flash should try eager before SDPA."""
        result = build_attention_candidates(
            use_flash_attention=False,
            flash_attention_available=False,
            is_rocm_cuda=True,
        )
        self.assertEqual(result, ["eager", "sdpa"])

    def test_build_attention_candidates_non_rocm_without_flash(self):
        """Non-ROCm without flash should keep SDPA-first behavior."""
        result = build_attention_candidates(
            use_flash_attention=False,
            flash_attention_available=False,
            is_rocm_cuda=False,
        )
        self.assertEqual(result, ["sdpa", "eager"])

    def test_build_attention_candidates_flash_requested_but_unavailable(self):
        """Flash request without availability should follow backend fallback order."""
        rocm_result = build_attention_candidates(
            use_flash_attention=True,
            flash_attention_available=False,
            is_rocm_cuda=True,
        )
        cuda_result = build_attention_candidates(
            use_flash_attention=True,
            flash_attention_available=False,
            is_rocm_cuda=False,
        )
        self.assertEqual(rocm_result, ["eager", "sdpa"])
        self.assertEqual(cuda_result, ["sdpa", "eager"])

    def test_build_attention_candidates_flash_available(self):
        """Flash path should still include deterministic fallbacks."""
        rocm_result = build_attention_candidates(
            use_flash_attention=True,
            flash_attention_available=True,
            is_rocm_cuda=True,
        )
        cuda_result = build_attention_candidates(
            use_flash_attention=True,
            flash_attention_available=True,
            is_rocm_cuda=False,
        )
        self.assertEqual(rocm_result, ["flash_attention_2", "eager", "sdpa"])
        self.assertEqual(cuda_result, ["flash_attention_2", "sdpa", "eager"])

    def test_should_rocm_direct_model_load(self):
        """Direct load should be enabled only for ROCm + resident DiT setups."""
        self.assertTrue(
            should_rocm_direct_model_load(
                is_rocm_cuda=True,
                offload_to_cpu=False,
                offload_dit_to_cpu=False,
            )
        )
        self.assertTrue(
            should_rocm_direct_model_load(
                is_rocm_cuda=True,
                offload_to_cpu=True,
                offload_dit_to_cpu=False,
            )
        )
        self.assertFalse(
            should_rocm_direct_model_load(
                is_rocm_cuda=True,
                offload_to_cpu=True,
                offload_dit_to_cpu=True,
            )
        )
        self.assertFalse(
            should_rocm_direct_model_load(
                is_rocm_cuda=False,
                offload_to_cpu=False,
                offload_dit_to_cpu=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
