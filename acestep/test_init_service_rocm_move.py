"""Unit tests for ROCm-safe model movement in InitServiceMixin."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from acestep.core.generation.handler.init_service import InitServiceMixin


class _DummyInitService(InitServiceMixin):
    """Minimal harness for testing InitServiceMixin helpers."""

    def __init__(self):
        self.device = "cpu"

    def _synchronize(self):
        """Override accelerator synchronization for unit tests."""
        return


class TestInitServiceRocmMove(unittest.TestCase):
    """Validate ROCm-specific safe path in _recursive_to_device."""

    def test_recursive_to_device_skips_model_to_on_rocm(self):
        """ROCm path should avoid model.to() and use manual parameter movement."""
        helper = _DummyInitService()
        helper._is_on_target_device = MagicMock(return_value=True)
        helper._move_model_parameters_individually = MagicMock()
        helper._move_module_recursive = MagicMock()

        model = MagicMock()
        model.named_parameters.return_value = []
        model.to = MagicMock(side_effect=AssertionError("model.to must be skipped on ROCm"))

        with patch("acestep.core.generation.handler.init_service.torch.version.hip", "6.4.0"):
            helper._recursive_to_device(model, "cuda", torch.float16)

        helper._move_model_parameters_individually.assert_called_once()
        helper._move_module_recursive.assert_not_called()
        model.to.assert_not_called()

    def test_recursive_to_device_uses_model_to_without_rocm(self):
        """Non-ROCm path should keep the standard model.to() fast path."""
        helper = _DummyInitService()
        helper._is_on_target_device = MagicMock(return_value=True)
        helper._move_model_parameters_individually = MagicMock()
        helper._move_module_recursive = MagicMock()

        model = MagicMock()
        model.named_parameters.return_value = []
        model.to = MagicMock()

        with patch("acestep.core.generation.handler.init_service.torch.version.hip", None):
            helper._recursive_to_device(model, "cuda", torch.float16)

        model.to.assert_any_call(torch.device("cuda"))
        model.to.assert_any_call(torch.float16)
        helper._move_module_recursive.assert_called_once()


if __name__ == "__main__":
    unittest.main()
