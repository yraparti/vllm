# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy DSpark Markov sampling of the Mistral Large 4 draft with fewer launches.

vLLM's ``DSparkSpeculator._sample_sequential`` runs about 17 kernels for each
of the 3 draft steps of the recurrent (``rnn``) Markov head: the anchor
gather, the Markov embedding, the state zeros and the concat that feed
``joint_proj``, seven elementwise kernels of the gated state update, the
``markov_w2`` bias GEMV, the add to the base logits, the argmax (a memset and
a reduction) and the copy into ``draft_tokens``.

Here a step is the two GEMVs, unchanged, and three launches around them:

* ``_joint_input_kernel``: the previous token (the anchor at step 0, else the
  argmax over the previous step's block maxima, which it also stores into
  ``draft_tokens``), its ``markov_w1`` row, and the ``joint_proj`` input row
  ``[state, embedding, draft hidden]``;
* ``_gate_kernel``: the gated state update and the output embedding, rounding
  to bf16 after every op as the bf16 tensor ops do;
* ``_bias_argmax_kernel``: ``base + bias`` rounded to bf16 and the first
  maximum of each vocabulary block.

The GEMVs read the same bf16 inputs as vLLM's loop and the argmax keeps the
first of equal maxima, as ``torch.argmax`` does, so the draft tokens are the
same.
"""

import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, tldevice, triton

logger = init_logger(__name__)

ENABLED = os.getenv("VLLM_ML4_FUSED_MARKOV", "1") != "0"
# Vocabulary-parallel steps (TP > 1): each rank keeps its lm_head shard's
# base logits (no all-gather; the soft cap in the argmax launch), multiplies
# only its rows of markov_w2, and the ranks trade one (max, token) pair a row
# per step: each writes its slot of a zeroed buffer and a sum all-reduce
# hands every rank all of them. Ranks hold the vocabulary in order, so the
# first maximum over the ranks is torch.argmax's over the vocabulary.
VP = os.getenv("VLLM_ML4_VP_MARKOV", "1") != "0"
# Vocabulary tokens per program of the bias/argmax launch.
BLOCK = 1024
# The launches unroll their loop over the requests of a step.
MAX_ROWS = 8


@triton.jit
def _joint_input_kernel(
    prev_max_ptr,  # [MAX_ROWS, num_blocks] fp32, the previous step's block maxima
    prev_arg_ptr,  # [MAX_ROWS, num_blocks] int64, their token ids
    input_ids_ptr,
    anchor_idx_ptr,  # step 0's previous token: input_ids[anchor_idx[r]]
    tokens_ptr,  # draft_tokens [max_num_reqs, n_spec] int64
    tokens_stride,
    w1_ptr,  # markov_w1 [vocab, RANK] bf16
    state_ptr,  # [MAX_ROWS, RANK] bf16, the recurrent state
    hidden_ptr,  # this step's draft hidden of row r at r * hidden_stride
    hidden_stride,
    joint_ptr,  # [MAX_ROWS, 2 * RANK + HIDDEN] bf16, joint_proj's input
    num_blocks,
    STEP: tl.constexpr,
    RANK: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCKS_POW2: tl.constexpr,
    HBLOCK: tl.constexpr,
    PAIRS: tl.constexpr = False,  # VP: prev_max_ptr holds [rows, 2, num_blocks] pairs
):
    r = tl.program_id(0)
    if STEP == 0:
        prev = tl.load(input_ids_ptr + tl.load(anchor_idx_ptr + r)).to(tl.int64)
    else:
        prev = _best_token(prev_max_ptr, prev_arg_ptr, r, num_blocks, BLOCKS_POW2, PAIRS)
        tl.store(tokens_ptr + r * tokens_stride + (STEP - 1), prev)
    k = tl.arange(0, RANK)
    row = joint_ptr + r * (2 * RANK + HIDDEN)
    if STEP == 0:
        tl.store(row + k, tl.zeros([RANK], dtype=tl.bfloat16))
    else:
        tl.store(row + k, tl.load(state_ptr + r * RANK + k))
    tl.store(row + RANK + k, tl.load(w1_ptr + prev * RANK + k))
    for h0 in tl.static_range(0, HIDDEN, HBLOCK):
        h = h0 + tl.arange(0, HBLOCK)
        x = tl.load(hidden_ptr + r * hidden_stride + h, mask=h < HIDDEN)
        tl.store(row + 2 * RANK + h, x, mask=h < HIDDEN)


@triton.jit
def _gate_kernel(
    joint_ptr,  # joint_proj output [rows, 3 * RANK] bf16: gate, candidate, output
    joint_stride,
    state_ptr,  # [MAX_ROWS, RANK] bf16, updated in place
    embed_ptr,  # [MAX_ROWS, RANK] bf16, the step's Markov embedding
    STEP: tl.constexpr,
    RANK: tl.constexpr,
):
    # next = sigmoid(g) * state + (1 - sigmoid(g)) * tanh(c); out = tanh(o),
    # each op rounded to bf16 as markov_step's bf16 tensor ops are
    r = tl.program_id(0)
    k = tl.arange(0, RANK)
    g = tl.load(joint_ptr + r * joint_stride + k).to(tl.float32)
    c = tl.load(joint_ptr + r * joint_stride + RANK + k).to(tl.float32)
    o = tl.load(joint_ptr + r * joint_stride + 2 * RANK + k).to(tl.float32)
    if STEP == 0:
        state = tl.zeros([RANK], dtype=tl.float32)
    else:
        state = tl.load(state_ptr + r * RANK + k).to(tl.float32)
    gate = tl.math.div_rn(1.0, 1.0 + tldevice.exp(-g)).to(tl.bfloat16).to(tl.float32)
    kept = (gate * state).to(tl.bfloat16).to(tl.float32)
    rest = (1.0 - gate).to(tl.bfloat16).to(tl.float32)
    cand = tldevice.tanh(c).to(tl.bfloat16).to(tl.float32)
    new = (rest * cand).to(tl.bfloat16).to(tl.float32)
    tl.store(state_ptr + r * RANK + k, (kept + new).to(tl.bfloat16))
    tl.store(embed_ptr + r * RANK + k, tldevice.tanh(o).to(tl.bfloat16))


@triton.jit
def _bias_argmax_kernel(
    base_ptr,  # this step's base logits of row r at r * base_stride + v, bf16
    base_stride,
    bias_ptr,  # [rows, vocab] bf16
    bias_stride,
    out_max_ptr,  # [MAX_ROWS, num_blocks] fp32
    out_arg_ptr,  # [MAX_ROWS, num_blocks] int64
    vocab,
    num_blocks,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    v = b * BLOCK + tl.arange(0, BLOCK)
    vmask = v < vocab
    for r in tl.static_range(ROWS):
        base = tl.load(base_ptr + r * base_stride + v, mask=vmask, other=0.0)
        bias = tl.load(bias_ptr + r * bias_stride + v, mask=vmask, other=0.0)
        # a bf16 tensor add: one rounding of the fp32 sum
        logits = (base.to(tl.float32) + bias.to(tl.float32)).to(tl.bfloat16)
        logits = tl.where(vmask, logits.to(tl.float32), float("-inf"))
        idx = tl.argmax(logits, axis=0, tie_break_left=True)
        tl.store(out_max_ptr + r * num_blocks + b, tl.max(logits, axis=0))
        tl.store(out_arg_ptr + r * num_blocks + b, (b * BLOCK + idx).to(tl.int64))


@triton.jit
def _last_token_kernel(
    prev_max_ptr,
    prev_arg_ptr,
    tokens_ptr,
    tokens_stride,
    num_blocks,
    STEP: tl.constexpr,
    BLOCKS_POW2: tl.constexpr,
    PAIRS: tl.constexpr = False,
):
    r = tl.program_id(0)
    prev = _best_token(prev_max_ptr, prev_arg_ptr, r, num_blocks, BLOCKS_POW2, PAIRS)
    tl.store(tokens_ptr + r * tokens_stride + STEP, prev)


@triton.jit
def _best_token(max_ptr, arg_ptr, r, num_blocks, BLOCKS_POW2: tl.constexpr, PAIRS: tl.constexpr):
    """Row r's token: the first of the largest block maxima ([rows, blocks]
    fp32 maxima, int64 tokens), or (PAIRS) of the ranks' pairs ([rows, 2,
    ranks] fp32: maxima, then tokens)."""
    j = tl.arange(0, BLOCKS_POW2)
    if PAIRS:
        row = max_ptr + r * 2 * num_blocks
        m = tl.load(row + j, mask=j < num_blocks, other=float("-inf"))
        best = tl.argmax(m, axis=0, tie_break_left=True)
        tok = tl.load(row + num_blocks + best).to(tl.int64)
    else:
        m = tl.load(max_ptr + r * num_blocks + j, mask=j < num_blocks, other=float("-inf"))
        best = tl.argmax(m, axis=0, tie_break_left=True)
        tok = tl.load(arg_ptr + r * num_blocks + best)
    return tok


@triton.jit
def _vp_bias_argmax_kernel(
    base_ptr,  # this rank's base logits of row r at r * base_stride + v, bf16, before the soft cap
    base_stride,
    bias_ptr,  # [rows, n_local] bf16, this rank's rows of the Markov bias
    bias_stride,
    out_max_ptr,  # [MAX_ROWS, num_blocks] fp32
    out_arg_ptr,  # [MAX_ROWS, num_blocks] int64, vocabulary token ids
    n_local,
    vocab_start,
    inv_cap,
    cap,
    num_blocks,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    SOFT_CAP: tl.constexpr,
):
    b = tl.program_id(0)
    v = b * BLOCK + tl.arange(0, BLOCK)
    vmask = v < n_local
    for r in tl.static_range(ROWS):
        base = tl.load(base_ptr + r * base_stride + v, mask=vmask, other=0.0).to(tl.float32)
        if SOFT_CAP:
            # LogitsProcessor's bf16 ops: / cap (a multiply by the fp32
            # reciprocal), tanh, * cap, each rounded to bf16
            x = (base * inv_cap).to(tl.bfloat16).to(tl.float32)
            t = tldevice.tanh(x).to(tl.bfloat16).to(tl.float32)
            base = (t * cap).to(tl.bfloat16).to(tl.float32)
        bias = tl.load(bias_ptr + r * bias_stride + v, mask=vmask, other=0.0).to(tl.float32)
        logits = (base + bias).to(tl.bfloat16)
        logits = tl.where(vmask, logits.to(tl.float32), float("-inf"))
        idx = tl.argmax(logits, axis=0, tie_break_left=True)
        tl.store(out_max_ptr + r * num_blocks + b, tl.max(logits, axis=0))
        tl.store(out_arg_ptr + r * num_blocks + b, (vocab_start + b * BLOCK + idx).to(tl.int64))


@triton.jit
def _vp_pair_kernel(
    blk_max_ptr,  # [MAX_ROWS, num_blocks] fp32
    blk_arg_ptr,  # [MAX_ROWS, num_blocks] int64
    pair_ptr,  # [rows, 2, TP] fp32: this rank's slot, zeros elsewhere
    num_blocks,
    rank,
    TP: tl.constexpr,
    BLOCKS_POW2: tl.constexpr,
):
    r = tl.program_id(0)
    j = tl.arange(0, BLOCKS_POW2)
    m = tl.load(blk_max_ptr + r * num_blocks + j, mask=j < num_blocks, other=float("-inf"))
    best = tl.argmax(m, axis=0, tie_break_left=True)
    mx = tl.max(m, axis=0)
    tok = tl.load(blk_arg_ptr + r * num_blocks + best).to(tl.float32)
    t = tl.arange(0, TP)
    mine = t == rank
    tl.store(pair_ptr + r * 2 * TP + t, tl.where(mine, mx, 0.0))
    tl.store(pair_ptr + r * 2 * TP + TP + t, tl.where(mine, tok, 0.0))


_buffers: dict = {}


def _scratch(device, vocab: int, rank: int, hidden: int, tp: int = 1) -> dict:
    """Allocated once, before graph capture, so captured graphs keep the
    addresses."""
    key = (device, vocab, rank, hidden, tp)
    if key not in _buffers:
        nb = triton.cdiv(vocab, BLOCK)
        _buffers[key] = {
            # VP: each rank's (max, token) slot per row, for the sum all-reduce
            "pairs": torch.zeros(MAX_ROWS, 2, tp, dtype=torch.float32, device=device),
            "joint": torch.empty(
                MAX_ROWS, 2 * rank + hidden, dtype=torch.bfloat16, device=device
            ),
            "state": torch.empty(MAX_ROWS, rank, dtype=torch.bfloat16, device=device),
            "embed": torch.empty(MAX_ROWS, rank, dtype=torch.bfloat16, device=device),
            "max": [
                torch.empty(MAX_ROWS, nb, dtype=torch.float32, device=device)
                for _ in range(2)
            ],
            "arg": [
                torch.empty(MAX_ROWS, nb, dtype=torch.int64, device=device)
                for _ in range(2)
            ],
        }
    return _buffers[key]


def _why_not(spec, model, num_reqs: int) -> str | None:
    """Why this step takes vLLM's loop, or None when the fused launches compute
    the same draft tokens."""
    if not ENABLED:
        return "VLLM_ML4_FUSED_MARKOV=0"
    if not getattr(model, "has_recurrent_markov_head", False):
        return "no recurrent Markov head"
    if spec.draft_logits is not None:
        return "probabilistic drafts"
    if spec.draft_watermarker is not None:
        return "a draft watermarker"
    if spec._d2t_scatter_index is not None:
        return "a reduced draft vocabulary"
    if spec.use_confidence_head:
        return "the confidence head"
    if spec.acceptance_estimator is not None:
        return "the acceptance estimator"
    if hasattr(model, "apply_output_logit_transform"):
        return "an output logit transform"
    if not 1 <= num_reqs <= MAX_ROWS:
        return f"{num_reqs} requests"
    head = model.model.markov_head
    w1, w2 = head.markov_w1.weight, head.markov_w2.weight
    if w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        return "Markov weights not bf16"
    if not w1.is_contiguous():
        return "markov_w1 layout"
    if head.markov_w2.quant_method.__class__.__name__ != "UnquantizedEmbeddingMethod":
        return "a quantized Markov head"
    return None


def _vp_why_not(model) -> str | None:
    """Why the steps keep the gathered logits, or None for vocabulary-parallel."""
    if not VP:
        return "VLLM_ML4_VP_MARKOV=0"
    head = model.lm_head
    if head is None or getattr(head, "tp_size", 1) <= 1:
        return "no vocabulary-parallel lm_head"
    lp = model.logits_processor
    if lp.scale != 1.0 or lp.org_vocab_size != lp.vocab_size:
        return "a logit scale or a padded vocabulary"
    w2 = model.model.markov_head.markov_w2.weight
    if w2.shape[0] != lp.org_vocab_size:
        return "a Markov head over another vocabulary"
    return None


def _sample_vp(spec, model, num_reqs: int, sample_hidden: torch.Tensor) -> bool:
    """The greedy steps over this rank's vocabulary shard (see VP)."""
    from vllm.distributed import tensor_model_parallel_all_reduce
    from vllm.model_executor.layers.utils import dispatch_unquantized_gemm

    from . import draft_fp8

    head = model.lm_head
    lp = model.logits_processor
    si = head.shard_indices
    v0, n_local = si.org_vocab_start_index, si.num_org_elements
    tp, rank = head.tp_size, head.tp_rank
    # this rank's base logits, before the soft cap (applied in the argmax launch)
    local = draft_fp8.gemv_fp8(sample_hidden, head.weight)
    if local is None:
        local = head.quant_method.apply(head, sample_hidden)
    n_spec = spec.num_speculative_steps
    base = local.view(num_reqs, n_spec, local.shape[-1])
    hidden = sample_hidden.view(num_reqs, n_spec, -1)
    mh = model.model.markov_head
    w1 = mh.markov_w1.weight
    w2_rows = mh.markov_w2.weight[v0 : v0 + n_local]
    gemm = dispatch_unquantized_gemm()
    rank_dim, width = w1.shape[1], hidden.shape[-1]
    buf = _scratch(local.device, n_local, rank_dim, width, tp)
    nb = triton.cdiv(n_local, BLOCK)
    nb2 = triton.next_power_of_2(nb)
    tp2 = triton.next_power_of_2(tp)
    cap = lp.soft_cap
    inv_cap = float(torch.tensor(1.0, dtype=torch.float32) / torch.tensor(cap or 1.0, dtype=torch.float32))
    tokens = spec.draft_tokens
    joint_in = buf["joint"][:num_reqs]
    embed = buf["embed"][:num_reqs]
    pairs = buf["pairs"][:num_reqs]
    logger.info_once(
        "ML4 DSpark fused Markov sampling, vocabulary-parallel: %d steps, %d of %d tokens on rank %d",
        n_spec, n_local, lp.org_vocab_size, rank,
    )  # fmt: skip
    reduced = None
    for step in range(n_spec):
        _joint_input_kernel[(num_reqs,)](
            reduced if reduced is not None else pairs, buf["arg"][0],
            spec.input_buffers.input_ids, spec._anchor_idx,
            tokens, tokens.stride(0),
            w1, buf["state"],
            hidden[:, step], hidden.stride(0),
            buf["joint"], tp,
            STEP=step, RANK=rank_dim, HIDDEN=width, BLOCKS_POW2=tp2, HBLOCK=1024, PAIRS=True,
        )  # fmt: skip
        joint = model.model.joint_proj(joint_in)[0]
        _gate_kernel[(num_reqs,)](
            joint, joint.stride(0), buf["state"], buf["embed"], STEP=step, RANK=rank_dim
        )
        bias = gemm(mh.markov_w2, embed, w2_rows)
        _vp_bias_argmax_kernel[(nb,)](
            base[:, step], base.stride(0),
            bias, bias.stride(0),
            buf["max"][0], buf["arg"][0],
            n_local, v0, inv_cap, float(cap or 1.0), nb,
            ROWS=num_reqs, BLOCK=BLOCK, SOFT_CAP=cap is not None,
        )  # fmt: skip
        _vp_pair_kernel[(num_reqs,)](
            buf["max"][0], buf["arg"][0], pairs, nb, rank, TP=tp, BLOCKS_POW2=nb2,
        )  # fmt: skip
        reduced = tensor_model_parallel_all_reduce(pairs)
    _last_token_kernel[(num_reqs,)](
        reduced, buf["arg"][0], tokens, tokens.stride(0), tp,
        STEP=n_spec - 1, BLOCKS_POW2=tp2, PAIRS=True,
    )  # fmt: skip
    return True


def sample_sequential(spec, model, num_reqs: int, head_hidden: torch.Tensor) -> bool:
    """``DSparkSpeculator._sample_sequential`` (greedy) with the fused launches.
    Returns False, having done nothing, when the step takes vLLM's loop."""
    why = _why_not(spec, model, num_reqs)
    if why is not None:
        logger.debug_once("ML4 DSpark fused Markov sampling off: %s", why)
        return False
    n_spec = spec.num_speculative_steps
    num_sample = num_reqs * n_spec
    sample_hidden = head_hidden[spec.sample_indices[:num_sample]]
    vp_why = _vp_why_not(model)
    if vp_why is None:
        return _sample_vp(spec, model, num_reqs, sample_hidden)
    logger.debug_once("ML4 DSpark vocabulary-parallel Markov sampling off: %s", vp_why)
    base_logits = model.compute_draft_logits(sample_hidden)
    if base_logits.dtype != torch.bfloat16 or not base_logits.is_contiguous():
        logger.warning_once(
            "ML4 DSpark fused Markov sampling: %s base logits, using vLLM's loop "
            "from them",
            base_logits.dtype,
        )
        _loop_from_base_logits(spec, model, num_reqs, base_logits, sample_hidden)
        return True
    vocab = base_logits.shape[-1]
    base = base_logits.view(num_reqs, n_spec, vocab)
    hidden = sample_hidden.view(num_reqs, n_spec, -1)
    head = model.model.markov_head
    w1 = head.markov_w1.weight
    rank, width = w1.shape[1], hidden.shape[-1]
    buf = _scratch(base_logits.device, vocab, rank, width)
    nb = triton.cdiv(vocab, BLOCK)
    nb2 = triton.next_power_of_2(nb)
    tokens = spec.draft_tokens
    joint_in = buf["joint"][:num_reqs]
    embed = buf["embed"][:num_reqs]
    logger.info_once(
        "ML4 DSpark fused Markov sampling: %d steps, vocabulary %d, rank %d",
        n_spec,
        vocab,
        rank,
    )
    for step in range(n_spec):
        prev_max, prev_arg = buf["max"][(step + 1) % 2], buf["arg"][(step + 1) % 2]
        out_max, out_arg = buf["max"][step % 2], buf["arg"][step % 2]
        _joint_input_kernel[(num_reqs,)](
            prev_max, prev_arg,
            spec.input_buffers.input_ids, spec._anchor_idx,
            tokens, tokens.stride(0),
            w1, buf["state"],
            hidden[:, step], hidden.stride(0),
            buf["joint"], nb,
            STEP=step, RANK=rank, HIDDEN=width, BLOCKS_POW2=nb2, HBLOCK=1024,
        )  # fmt: skip
        joint = model.model.joint_proj(joint_in)[0]
        _gate_kernel[(num_reqs,)](
            joint, joint.stride(0), buf["state"], buf["embed"], STEP=step, RANK=rank
        )
        bias = model.markov_bias(embed)
        _bias_argmax_kernel[(nb,)](
            base[:, step], base.stride(0),
            bias, bias.stride(0),
            out_max, out_arg, vocab, nb,
            ROWS=num_reqs, BLOCK=BLOCK,
        )  # fmt: skip
    last = (n_spec - 1) % 2
    _last_token_kernel[(num_reqs,)](
        buf["max"][last], buf["arg"][last], tokens, tokens.stride(0), nb,
        STEP=n_spec - 1, BLOCKS_POW2=nb2,
    )  # fmt: skip
    return True


def _loop_from_base_logits(spec, model, num_reqs, base_logits, sample_hidden) -> None:
    """vLLM's loop of _sample_sequential after compute_draft_logits, so a step
    computes the base logits only once."""
    n_spec = spec.num_speculative_steps
    base_logits = base_logits.view(num_reqs, n_spec, base_logits.shape[-1])
    hidden = sample_hidden.view(num_reqs, n_spec, -1)
    prev = spec.input_buffers.input_ids[spec._anchor_idx[:num_reqs]]
    state = None
    for i in range(n_spec):
        markov_embed, state = model.markov_step(prev, hidden[:, i], state)
        logits = base_logits[:, i] + model.markov_bias(markov_embed)
        draft = model.map_draft_to_target(logits.argmax(dim=-1))
        spec.draft_tokens[:num_reqs, i] = draft
        prev = draft
