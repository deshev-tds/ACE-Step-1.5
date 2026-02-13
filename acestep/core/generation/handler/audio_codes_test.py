"""Unit tests for audio-code decoding helpers."""

import contextlib
import types
import unittest
from unittest.mock import patch

import torch

from acestep.core.generation.handler.audio_codes import AudioCodesMixin


class _MismatchQuantizer(torch.nn.Module):
    """Minimal quantizer that reproduces Float/Half linear mismatch."""

    def __init__(self):
        super().__init__()
        self.project_out = torch.nn.Linear(6, 2048, bias=True).to(dtype=torch.float16)

    def get_output_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Return quantized vectors from indices."""
        batch_size, seq_len, _ = indices.shape
        codes_summed = torch.ones(batch_size, seq_len, 6, device=indices.device, dtype=torch.float32)
        return self.project_out(codes_summed)


class _AlwaysFailQuantizer(torch.nn.Module):
    """Quantizer that raises a non-dtype runtime error."""

    def __init__(self):
        super().__init__()
        self.project_out = torch.nn.Linear(6, 6, bias=False)

    def get_output_from_indices(self, _indices: torch.Tensor) -> torch.Tensor:
        """Raise a non-recoverable error."""
        raise RuntimeError("non-dtype failure")


class _Host(AudioCodesMixin):
    """Host stub for exercising ``AudioCodesMixin`` methods."""

    def __init__(self, quantizer: torch.nn.Module):
        self.device = "cpu"
        self.dtype = torch.float16
        self.model = types.SimpleNamespace(
            tokenizer=types.SimpleNamespace(quantizer=quantizer),
            detokenizer=torch.nn.Identity(),
        )
        self.vae = None
        self.silence_latent = None

    @contextlib.contextmanager
    def _load_model_context(self, _name: str):
        """No-op context manager for tests."""
        yield


class AudioCodesMixinTests(unittest.TestCase):
    """Tests for robust audio-code decoding behavior."""

    def test_decode_audio_codes_retries_with_fp32_project_out_on_dtype_mismatch(self):
        """Dtype mismatch in quantizer should be recovered by fp32 fallback."""
        quantizer = _MismatchQuantizer()
        host = _Host(quantizer)

        with patch("acestep.core.generation.handler.audio_codes.logger.warning") as warning:
            latents = host._decode_audio_codes_to_latents("<|audio_code_1|><|audio_code_2|>")

        self.assertIsNotNone(latents)
        self.assertEqual(latents.dtype, torch.float16)
        self.assertEqual(quantizer.project_out.weight.dtype, torch.float32)
        warning.assert_called_once()

    def test_decode_audio_codes_propagates_non_dtype_runtime_error(self):
        """Non-dtype runtime errors should not be swallowed by fallback logic."""
        host = _Host(_AlwaysFailQuantizer())

        with self.assertRaisesRegex(RuntimeError, "non-dtype failure"):
            host._decode_audio_codes_to_latents("<|audio_code_1|>")


if __name__ == "__main__":
    unittest.main()
