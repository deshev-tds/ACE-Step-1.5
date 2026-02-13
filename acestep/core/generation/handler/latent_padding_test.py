"""Unit tests for latent padding helpers."""

import unittest

import torch

from acestep.core.generation.handler.latent_padding import (
    build_silence_latent,
    normalize_latent_length,
)


class LatentPaddingTests(unittest.TestCase):
    """Tests for deterministic latent length normalization."""

    def test_build_silence_latent_repeats_to_requested_length(self):
        """Silence builder should repeat base latent when requested length is larger."""
        silence = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=torch.float32)
        out = build_silence_latent(silence, 8)

        expected = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(out, expected))

    def test_normalize_latent_length_pads_with_silence_and_preserves_dtype(self):
        """Padding should extend latent to target length using silence on latent dtype/device."""
        latent = torch.tensor([[10.0, 20.0], [11.0, 21.0]], dtype=torch.float16)
        silence = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=torch.float32)

        out = normalize_latent_length(latent, 7, silence)

        expected_tail = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float16,
        )
        self.assertEqual(out.shape, (7, 2))
        self.assertEqual(out.dtype, torch.float16)
        self.assertTrue(torch.equal(out[:2], latent))
        self.assertTrue(torch.equal(out[2:], expected_tail))

    def test_normalize_latent_length_crops_when_input_is_longer(self):
        """Normalization should crop overlong latents to target length."""
        latent = torch.arange(18, dtype=torch.float32).reshape(9, 2)
        silence = torch.ones(1, 2, 2, dtype=torch.float32)

        out = normalize_latent_length(latent, 4, silence)
        self.assertEqual(out.shape, (4, 2))
        self.assertTrue(torch.equal(out, latent[:4]))

    def test_normalize_latent_length_supports_large_padding_gap(self):
        """Large target gaps should still produce exact lengths for downstream stacking."""
        silence = torch.randn(1, 750, 64, dtype=torch.float32)
        latent_a = torch.randn(20095, 64, dtype=torch.float32)
        latent_b = torch.randn(20160, 64, dtype=torch.float32)
        target_len = 20160

        out_a = normalize_latent_length(latent_a, target_len, silence)
        out_b = normalize_latent_length(latent_b, target_len, silence)
        stacked = torch.stack([out_a, out_b])

        self.assertEqual(out_a.shape[0], target_len)
        self.assertEqual(out_b.shape[0], target_len)
        self.assertEqual(stacked.shape, (2, target_len, 64))


if __name__ == "__main__":
    unittest.main()
