"""Unit tests for latent numerical safety helpers."""

import unittest

import torch

from acestep.core.generation.handler.latent_safety import (
    all_non_finite_sample_mask,
    count_non_finite,
    sanitize_non_finite_latents,
)


class LatentSafetyTests(unittest.TestCase):
    """Tests for NaN/Inf detection and sanitization behavior."""

    def test_count_non_finite_counts_nan_and_inf(self):
        """Non-finite counter should include NaN and both Inf signs."""
        latents = torch.tensor([0.0, float("nan"), float("inf"), float("-inf")], dtype=torch.float32)
        self.assertEqual(count_non_finite(latents), 3)

    def test_sanitize_non_finite_latents_replaces_with_zero(self):
        """Sanitizer should replace non-finite values while keeping finite ones."""
        latents = torch.tensor([[1.0, float("nan")], [float("inf"), -2.0]], dtype=torch.float32)
        cleaned, replaced = sanitize_non_finite_latents(latents)
        expected = torch.tensor([[1.0, 0.0], [0.0, -2.0]], dtype=torch.float32)
        self.assertEqual(replaced, 2)
        self.assertTrue(torch.equal(cleaned, expected))

    def test_all_non_finite_sample_mask_marks_fully_invalid_batch_items(self):
        """Mask should flag only samples that have no finite values at all."""
        latents = torch.tensor(
            [
                [[float("nan"), float("inf")], [float("-inf"), float("nan")]],
                [[0.5, 1.0], [float("nan"), 2.0]],
            ],
            dtype=torch.float32,
        )
        mask = all_non_finite_sample_mask(latents)
        self.assertTrue(torch.equal(mask, torch.tensor([True, False])))


if __name__ == "__main__":
    unittest.main()
