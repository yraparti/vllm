# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 weights for the Mistral Large 4 draft path.

``VLLM_ML4_DRAFT_FP8=1``: at 3-4 rows the draft path streams about 1.6 GB of
bf16 weights a verify step on each GPU: the draft's lm_head call (the
target's vocabulary shard), the Markov bias head (replicated, once per draft
step) and target_projection (replicated). Those GEMVs run at HBM bandwidth,
so their bytes set their time. ``gemv_fp8`` gives a weight an FP8 E4M3 copy
with one per-tensor fp32 scale and runs vLLM's FP8 skinny GEMM (wvSplitKQ)
on it, the activation quantized per tensor on the fly: half the weight bytes.

``VLLM_ML4_DRAFT_Q128=1``: the draft layers' bf16 linear weights rounded
through block FP8 at load (``fake_quant_draft_layers``), the rounding the
fused draft-layer kernel's weights carry (rocke_draft.py), so its acceptance
cost can be measured on vLLM's own layers.

Only the draft reads these weights. The draft tokens can change; the
target's output cannot, since the rejection sampler checks every draft token
against the target's own logits.
"""

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.getenv("VLLM_ML4_DRAFT_FP8", "0") != "0"
# The draft layers' linear weights rounded through block FP8 at load.
Q128 = os.getenv("VLLM_ML4_DRAFT_Q128", "0") != "0"
FP8_MAX = 448.0
# vLLM's FP8 skinny GEMM (wvSplitKQ) takes at most this many rows.
SKINNY_ROWS = 4

_fp8_weights: dict = {}


def fp8_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """A per-tensor FP8 E4M3 copy of the bf16 weight ``w`` and its fp32 scale,
    built the first time ``w`` is seen outside CUDA-graph capture (the
    profiling run, before capture): a copy first built inside a capture
    would be requantized on every replay. None for a weight first seen
    while capturing; that call keeps its bf16 GEMV."""
    key = (w.data_ptr(), tuple(w.shape))
    got = _fp8_weights.get(key)
    if got is None:
        if torch.cuda.is_current_stream_capturing():
            logger.warning_once("ML4 draft FP8: a weight first seen in graph capture keeps its bf16 GEMV")
            return None
        scale = (w.abs().amax().float().clamp_min(1e-12) / FP8_MAX).reshape(1)
        q = torch.empty(w.shape, dtype=torch.float8_e4m3fn, device=w.device)
        for r in range(0, w.shape[0], 8192):
            q[r : r + 8192] = (w[r : r + 8192].float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        got = _fp8_weights[key] = (q, scale)
        logger.info("ML4 draft FP8: %s weight -> FP8 copy (%.0f MB)", tuple(w.shape), q.numel() / 2**20)
    return got


def gemv_fp8(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor | None:
    """``x @ w.T`` (bf16 out) through the FP8 copy of ``w``: the activation
    quantized per tensor on the fly, vLLM's FP8 skinny GEMM (wvSplitKQ).
    None when the call keeps its bf16 GEMV: switch off, more than
    SKINNY_ROWS rows, a weight that is not bf16, or no copy yet."""
    if not ENABLED or x.dim() != 2 or x.shape[0] > SKINNY_ROWS or w.dtype != torch.bfloat16:
        return None
    if x.dtype != torch.bfloat16 or w.shape[0] % 16 or w.shape[1] % 16:
        return None
    got = fp8_weight(w)
    if got is None:
        return None
    from vllm import _custom_ops as ops
    from vllm.utils.platform_utils import num_compute_units

    q, s_w = got
    xq, s_x = ops.scaled_fp8_quant(x.contiguous(), None)
    return ops.wvSplitKQ(q, xq, torch.bfloat16, s_x, s_w, num_compute_units())


def _q128_roundtrip(w: torch.Tensor, block_cols: int = 128) -> torch.Tensor:
    """``w`` through E4M3 with one fp32 scale per 128 x ``block_cols`` block
    (amax / 448), back in ``w``'s dtype."""
    r, c = w.shape
    wp = torch.nn.functional.pad(w.float(), (0, (-c) % block_cols, 0, (-r) % 128))
    b = wp.view(wp.shape[0] // 128, 128, wp.shape[1] // block_cols, block_cols)
    s = b.abs().amax((1, 3), keepdim=True).clamp_min(1e-12) / FP8_MAX
    q = (b / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * s
    return q.view_as(wp)[:r, :c].to(w.dtype)


def fake_quant_draft_layers(layers) -> None:
    """Round each draft layer's bf16 linear weights through block FP8 in
    place, in the blockings of the fused draft-layer kernel (the target's
    mono weight build: 128 x 128, W_UK^T 128 x 64). Must run before the MLA
    layers derive their absorbed W_UK / W_UV from ``kv_b_proj``."""
    n = 0
    for layer in layers:
        attn, mlp = layer.self_attn, layer.mlp
        lins = (attn.q_proj, attn.kv_a_proj_with_mqa, attn.o_proj, mlp.gate_up_proj, mlp.down_proj)
        for lin in lins:
            w = lin.weight.data
            if w.dtype == torch.bfloat16 and w.dim() == 2:
                w.copy_(_q128_roundtrip(w))
                n += 1
        w = attn.kv_b_proj.weight.data
        if w.dtype == torch.bfloat16:
            nh, nope, vd = attn.num_local_heads, attn.qk_nope_head_dim, attn.v_head_dim
            kvb = w.view(nh, nope + vd, -1)
            uk = kvb[:, :nope, :].transpose(1, 2).reshape(-1, nope)
            uk = _q128_roundtrip(uk, 64).view(nh, -1, nope).transpose(1, 2)
            uv = _q128_roundtrip(kvb[:, nope:, :].reshape(nh * vd, -1)).view(nh, vd, -1)
            kvb[:, :nope, :].copy_(uk)
            kvb[:, nope:, :].copy_(uv)
            n += 1
    logger.info("ML4 draft Q128: %d draft linear weights rounded through block FP8", n)
