"""Audio-code parsing and conversion helpers for handler decomposition."""

import re
import traceback
from typing import List, Optional

import torch
from loguru import logger


class AudioCodesMixin:
    """Mixin containing audio-code parsing and latent conversion helpers.

    Depends on host members:
    - Attributes: ``model``, ``vae``, ``device``, ``dtype``, ``silence_latent``.
    - Methods: ``_load_model_context``, ``process_src_audio``, ``is_silence``,
      ``_encode_audio_to_latents``.
    """

    def _parse_audio_code_string(self, code_str: str) -> List[int]:
        """Extract integer audio codes from tokens like ``<|audio_code_123|>``."""
        if not code_str:
            return []
        try:
            max_audio_code = 63999
            codes = []
            clamped_count = 0
            for x in re.findall(r"<\|audio_code_(\d+)\|>", code_str):
                code_value = int(x)
                clamped_value = max(0, min(code_value, max_audio_code))
                if clamped_value != code_value:
                    clamped_count += 1
                    logger.warning(
                        f"[_parse_audio_code_string] Clamped audio code value from {code_value} to {clamped_value}"
                    )
                codes.append(clamped_value)
            if clamped_count > 0:
                logger.warning(
                    f"[_parse_audio_code_string] Clamped {clamped_count} audio code value(s) "
                    f"to valid range [0, {max_audio_code}]"
                )
            return codes
        except Exception as e:
            logger.debug(f"[_parse_audio_code_string] Failed to parse audio code string: {e}")
            return []

    @staticmethod
    def _is_linear_dtype_mismatch(error: RuntimeError) -> bool:
        """Return ``True`` when the runtime error indicates matmul dtype mismatch."""
        return "mat1 and mat2 must have the same dtype" in str(error)

    def _decode_quantizer_indices(self, quantizer, indices: torch.Tensor) -> torch.Tensor:
        """Decode quantizer indices with a mixed-precision fallback.

        Some ROCm stacks surface ``Float`` input + ``Half`` ``project_out`` weight
        mismatches inside ``ResidualFSQ.get_output_from_indices``. In that case we
        upcast ``project_out`` to ``float32`` and retry once.
        """
        try:
            return quantizer.get_output_from_indices(indices)
        except RuntimeError as error:
            if not self._is_linear_dtype_mismatch(error):
                raise

            project_out = getattr(quantizer, "project_out", None)
            weight = getattr(project_out, "weight", None)
            if project_out is None or weight is None:
                raise

            if weight.dtype != torch.float32:
                logger.warning(
                    "[_decode_audio_codes_to_latents] Detected quantizer dtype mismatch; "
                    "switching tokenizer.quantizer.project_out to float32 for decode compatibility."
                )
                project_out.to(dtype=torch.float32)

            return quantizer.get_output_from_indices(indices)

    def _decode_audio_codes_to_latents(self, code_str: str) -> Optional[torch.Tensor]:
        """Convert serialized audio-code string into 25Hz latents."""
        if self.model is None or not hasattr(self.model, "tokenizer") or not hasattr(self.model, "detokenizer"):
            return None

        code_ids = self._parse_audio_code_string(code_str)
        if len(code_ids) == 0:
            return None

        with self._load_model_context("model"):
            quantizer = self.model.tokenizer.quantizer
            detokenizer = self.model.detokenizer
            indices = torch.tensor(code_ids, device=self.device, dtype=torch.long)
            indices = indices.unsqueeze(0).unsqueeze(-1)

            quantized = self._decode_quantizer_indices(quantizer, indices)
            if quantized.dtype != self.dtype:
                quantized = quantized.to(self.dtype)
            lm_hints_25hz = detokenizer(quantized)
            return lm_hints_25hz

    def convert_src_audio_to_codes(self, audio_file) -> str:
        """Convert uploaded source audio into serialized audio code tokens."""
        if audio_file is None:
            return "❌ Please upload source audio first"
        if self.model is None or self.vae is None:
            return "❌ Model not initialized. Please initialize the service first."

        try:
            processed_audio = self.process_src_audio(audio_file)
            if processed_audio is None:
                return "❌ Failed to process audio file"

            with torch.inference_mode():
                with self._load_model_context("vae"):
                    if self.is_silence(processed_audio.unsqueeze(0)):
                        return "❌ Audio file appears to be silent"
                    latents = self._encode_audio_to_latents(processed_audio)

                attention_mask = torch.ones(latents.shape[0], dtype=torch.bool, device=self.device)
                with self._load_model_context("model"):
                    hidden_states = latents.unsqueeze(0)
                    _, indices, _ = self.model.tokenize(
                        hidden_states, self.silence_latent, attention_mask.unsqueeze(0)
                    )
                    indices_flat = indices.flatten().cpu().tolist()
                    codes_string = "".join([f"<|audio_code_{idx}|>" for idx in indices_flat])
                    logger.info(f"[convert_src_audio_to_codes] Generated {len(indices_flat)} audio codes")
                    return codes_string
        except Exception as e:
            error_msg = f"❌ Error converting audio to codes: {str(e)}\n{traceback.format_exc()}"
            logger.exception("[convert_src_audio_to_codes] Error converting audio to codes")
            return error_msg
