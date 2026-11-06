# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch
from transformers import PreTrainedConfig

from vllm.model_executor.layers.rotary_embedding.common import (
    yarn_find_correction_range,
    yarn_linear_ramp_mask,
)
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
    DeepseekScalingRotaryEmbedding,
)
from vllm.triton_utils import tl, triton

_FUSED_ROTATE = os.getenv("VLLM_MISTRAL_FUSED_ROPE", "1") != "0"


@triton.jit(
    do_not_specialize=[
        "cs_stride", "a_stride_t", "a_stride_h", "a_heads",
        "b_stride_t", "b_stride_h", "b_heads",
    ]
)  # fmt: skip
def _rope_gptj_bf16_kernel(
    pos_ptr,
    cs_ptr,  # cos_sin_cache [max_pos, 2 * HALF] bf16: cos then sin
    cs_stride,
    a_ptr,  # [tokens, heads, 2 * HALF] bf16, rotated in place (program axis 1 = 0)
    a_stride_t,
    a_stride_h,
    a_heads,
    b_ptr,  # the same for program axis 1 = 1
    b_stride_t,
    b_stride_h,
    b_heads,
    HALF: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # forward_native rounds every bf16 tensor op (x * cos, rotate(x) * sin and
    # their sum) to bf16 once from fp32; rounding at the same points gives the
    # same bits.
    t = tl.program_id(0).to(tl.int64)
    if tl.program_id(1) == 0:
        ptr, stride_t, stride_h, heads = a_ptr, a_stride_t, a_stride_h, a_heads
    else:
        ptr, stride_t, stride_h, heads = b_ptr, b_stride_t, b_stride_h, b_heads
    pos = tl.load(pos_ptr + t).to(tl.int64)
    j = tl.arange(0, HALF)
    cos = tl.load(cs_ptr + pos * cs_stride + j).to(tl.float32)[None, :]
    sin = tl.load(cs_ptr + pos * cs_stride + HALF + j).to(tl.float32)[None, :]
    h = tl.arange(0, BLOCK_H)
    mask = (h < heads)[:, None]
    even = ptr + t * stride_t + h[:, None] * stride_h + 2 * j[None, :]
    x1 = tl.load(even, mask=mask).to(tl.float32)
    x2 = tl.load(even + 1, mask=mask).to(tl.float32)
    out1 = (x1 * cos).to(tl.bfloat16).to(tl.float32) + (-x2 * sin).to(
        tl.bfloat16
    ).to(tl.float32)
    out2 = (x2 * cos).to(tl.bfloat16).to(tl.float32) + (x1 * sin).to(
        tl.bfloat16
    ).to(tl.float32)
    tl.store(even, out1.to(tl.bfloat16), mask=mask)
    tl.store(even + 1, out2.to(tl.bfloat16), mask=mask)


class MistralRotaryEmbedding(DeepseekScalingRotaryEmbedding):
    """Interleaved (GPT-J style) YaRN RoPE with optional LeRoPE.

    LeRoPE multiplies the extrapolated and interpolated inverse frequencies by
    ``exp(alpha)``, where ``alpha`` is a checkpoint parameter shared by all
    layers. The YaRN ramp keeps using the unscaled frequencies. The cos/sin
    cache is rebuilt by :meth:`set_lerope_alpha` once the weights are loaded.
    """

    def __init__(self, config: PreTrainedConfig, rotary_dim: int) -> None:
        rope_parameters = config.rope_parameters
        is_yarn = rope_parameters.get("rope_type", "default") == "yarn"
        # YaRN magnitude scaling (`apply_scale`) is not used by Mistral models.
        assert not is_yarn or rope_parameters.get("attention_factor") == 1.0
        self.lerope_alpha: torch.Tensor | None = None
        super().__init__(
            head_size=rotary_dim,
            rotary_dim=rotary_dim,
            max_position_embeddings=(
                rope_parameters["original_max_position_embeddings"]
                if is_yarn
                else config.max_position_embeddings
            ),
            base=rope_parameters["rope_theta"],
            is_neox_style=False,
            scaling_factor=rope_parameters["factor"] if is_yarn else 1.0,
            dtype=torch.get_default_dtype(),
            beta_fast=rope_parameters.get("beta_fast", 32),
            beta_slow=rope_parameters.get("beta_slow", 1),
            mscale=rope_parameters.get("mscale", 1),
            mscale_all_dim=rope_parameters.get("mscale_all_dim", 0),
        )
        # `apply_scale: false` checkpoints must not rescale cos/sin.
        assert not is_yarn or self.mscale == 1.0

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        pos_freqs = self.base ** (
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)
        if self.lerope_alpha is not None:
            lerope_scale = torch.exp(self.lerope_alpha.to(pos_freqs))
            inv_freq_extrapolation = inv_freq_extrapolation * lerope_scale
            inv_freq_interpolation = inv_freq_interpolation * lerope_scale
        # The YaRN ramp is defined on the frequencies without LeRoPE.
        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
        )
        extrapolation_mask = 1 - yarn_linear_ramp_mask(
            low, high, self.rotary_dim // 2, dtype=torch.float
        )
        return (
            inv_freq_interpolation * (1 - extrapolation_mask)
            + inv_freq_extrapolation * extrapolation_mask
        )

    @torch.no_grad()
    def set_lerope_alpha(self, lerope_alpha: torch.Tensor) -> None:
        assert lerope_alpha.shape == (self.rotary_dim // 2,)
        self.lerope_alpha = lerope_alpha.detach().float()
        cache = self._compute_cos_sin_cache()
        self.cos_sin_cache.copy_(cache.to(self.cos_sin_cache.dtype))

    def rotate_(
        self,
        positions: torch.Tensor,
        query: torch.Tensor | None,
        key: torch.Tensor,
    ) -> bool:
        """``forward_native(positions, query, key)`` in place, bitwise, as one
        launch; without a query only the key rotates.

        Returns False, having done nothing, when the launch does not apply
        (not bf16 [tokens, heads, rotary_dim] GPU tensors and cache); the
        caller then runs ``forward``.
        """
        cache = self.cos_sin_cache
        tensors = [x for x in (query, key) if x is not None]
        if (
            not _FUSED_ROTATE
            or self.is_neox_style
            or cache.dtype != torch.bfloat16
            or not cache.is_cuda
            or cache.stride(-1) != 1
            or positions.dim() != 1
            or any(
                x.dtype != torch.bfloat16
                or x.dim() != 3
                or x.shape[0] != positions.shape[0]
                or x.shape[-1] != self.rotary_dim
                or x.stride(-1) != 1
                for x in tensors
            )
        ):
            return False
        if positions.shape[0] == 0:
            return True
        a, b = tensors[0], tensors[-1]
        _rope_gptj_bf16_kernel[(positions.shape[0], len(tensors))](
            positions,
            cache,
            cache.stride(0),
            a,
            a.stride(0),
            a.stride(1),
            a.shape[1],
            b,
            b.stride(0),
            b.stride(1),
            b.shape[1],
            HALF=self.rotary_dim // 2,
            BLOCK_H=triton.next_power_of_2(max(a.shape[1], b.shape[1])),
        )
        return True
