# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark draft layers as rocKE launches (``VLLM_ML4_DRAFT_ROCKE=1``).

A draft forward of one request (its 4-row block) runs each draft layer as one
persistent launch of the rocKE ML4 layer family's ``draft`` kind, mapping the
residual stream entering the layer to the stream leaving it: input RMSNorm,
the latent attention over the layer's sliding window (the block's own rows
included, non-causal), both TP all-reduces and the dense MLP, inside the
kernel. The kernels read and write vLLM's bf16 MLA latent pages through the
layer's block table and slot mapping; the window positions go to the kernel
as its key list. Any other forward runs vLLM's layers.

The kernel's weights are the draft layers' bf16 linears rounded through block
FP8 (128 x 128 blocks, W_UK^T 128 x 64), built after loading together with
the launcher, scratch and peer buffer (a collective of the TP group). The
draft tokens can change; the target's output cannot, since the rejection
sampler checks every draft token against the target's own logits.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.getenv("VLLM_ML4_DRAFT_ROCKE", "0") != "0"
# one request's draft block
ROWS = 4
HEADS, NOPE, PE, KVL, VD, HIDDEN, INTER = 16, 64, 64, 512, 128, 6144, 6144
QK = KVL + PE
# the kernel's key-list width (split's chunks); the window must fit it
TOPK = 2048
# fp32 parameter table: SSMax s and b by head (s = 0, b = 1: no SSMax)
P_SS_S = 2 * HIDDEN
P_SS_B = P_SS_S + 64
FP8_MAX = 448.0
ORDER = ("qkv_a", "q_norm", "q_b", "cache", "uk", "split", "uv", "o", "mlp_ug", "mlp_down")
# schedule knobs of the draft kind's build (``Ml4LayerSpec``): the activation
# from up/gate's epilogue (interleaved gate / up rows), and split gathering
# its keys before its queries arrive (no top-k to wait on)
KNOBS: dict = {"mlp_act": True, "split_q_first": False}
PLAN = (
    ("qkv_a", QK // 16, 0), ("q_norm", ROWS, 36), ("cache", 1, 40), ("q_b", HEADS * 128 // 16, 41),
    ("uk", HEADS * KVL // 128, 169), ("split", ROWS * TOPK // 64, 0), ("uv", ROWS * HEADS * VD // 64, 128),
    ("o", HIDDEN // 32, 0), ("mlp_ug", 2 * INTER // 48, 0), ("mlp_down", HIDDEN // 32, 64),
)  # fmt: skip


def _q128(w: torch.Tensor, bc: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [r, c] -> (E4M3 [r, c], fp32 scale per 128 x bc block, amax / 448)."""
    r, c = w.shape
    wp = F.pad(w.float(), (0, (-c) % bc, 0, (-r) % 128))
    blk = wp.view(wp.shape[0] // 128, 128, wp.shape[1] // bc, bc)
    s = blk.abs().amax((1, 3)).clamp_min(1e-12) / FP8_MAX
    q = (blk / s[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.view(wp.shape)[:r, :c].contiguous(), s.contiguous()


def _interleave(ug: torch.Tensor) -> torch.Tensor:
    """[gate; up] rows -> 16-row tiles of 8 gate rows and the same 8 up rows
    (the build's ``mlp_act``)."""
    n = ug.shape[0] // 2
    return torch.stack([ug[:n].view(n // 8, 8, -1), ug[n:].view(n // 8, 8, -1)], 1).reshape(2 * n, -1)


def refusal(model) -> str | None:
    """Why the draft model cannot run the kernels (None: it can)."""
    if get_tensor_model_parallel_world_size() != 4:
        return "the kernel is built for TP 4"
    for layer in model.layers:
        attn, mlp = layer.self_attn, layer.mlp
        if getattr(attn, "spec", None) is not None and attn.spec.shares_kv:
            return "a KV-sharing draft layer"
        if (attn.num_local_heads, attn.qk_nope_head_dim, attn.v_head_dim) != (HEADS, NOPE, VD):
            return "attention geometry"
        if mlp.gate_up_proj.weight.shape != (2 * INTER, HIDDEN) or mlp.down_proj.weight.shape != (HIDDEN, INTER):
            return "MLP geometry"
        weights = (attn.q_proj, attn.kv_a_proj_with_mqa, attn.kv_b_proj, attn.o_proj, mlp.gate_up_proj, mlp.down_proj)
        if any(lin.weight.dtype != torch.bfloat16 for lin in weights):
            return "the draft linears are not bf16"
    window = getattr(model.config, "sliding_window", None)
    if not window or window > TOPK:
        return f"window {window}"
    return None


class RockeDraft:
    """Kernel weights, launcher and buffers of the draft layers (one rank)."""

    def __init__(self, model, max_model_len: int, block_size: int) -> None:
        from aiter.ops.flydsl.kernels.mono.mistral_large4.gfx950 import layout, rocke_layer
        from aiter.ops.flydsl.mono.runtime.peer_memory import SymmetricPeerBuffer
        from aiter.ops.flydsl.mono.weights.packing import pack_fp8

        root = rocke_layer.rocke_root()
        if root is None:
            raise RuntimeError(f"VLLM_ML4_DRAFT_ROCKE needs the rocKE tree in {rocke_layer.ENV}")
        self.model = model
        self.window = int(model.config.sliding_window)
        self.rank = get_tensor_model_parallel_rank()
        self.npes = get_tensor_model_parallel_world_size()
        self.layer_names = [layer.self_attn.attn.layer_name for layer in model.layers]
        dev = model.final_norm.weight.device
        self.layers = []
        for layer in model.layers:
            attn, mlp = layer.self_attn, layer.mlp
            t = {}
            for name, lin in (("q_b", attn.q_proj), ("qkv_a", attn.kv_a_proj_with_mqa), ("o", attn.o_proj),
                              ("ug", mlp.gate_up_proj), ("dn", mlp.down_proj)):  # fmt: skip
                w = lin.weight.data
                if name == "ug" and KNOBS.get("mlp_act"):
                    w = _interleave(w)
                q, s = _q128(w)
                t["w_" + name], t["s_" + name] = pack_fp8(q), s
            kvb = attn.kv_b_proj.weight.data.float().view(HEADS, NOPE + VD, KVL)
            q, t["s_uk"] = _q128(kvb[:, :NOPE, :].transpose(1, 2).reshape(HEADS * KVL, NOPE), 64)
            t["w_uk"] = pack_fp8(q)
            q, t["s_uv"] = _q128(kvb[:, NOPE:, :].reshape(HEADS * VD, KVL))
            t["w_uv"] = pack_fp8(q)
            t["g_in"] = layer.input_layernorm.weight.data.to(torch.bfloat16).contiguous()
            t["g_kv"] = attn.kv_a_layernorm.weight.data.to(torch.bfloat16).contiguous()
            t["g_post"] = layer.post_attention_layernorm.weight.data.to(torch.bfloat16).contiguous()
            self.layers.append(t)
        self.g_q = torch.zeros(P_SS_B + 64, dtype=torch.float32, device=dev)
        self.g_q[P_SS_B : P_SS_B + HEADS] = 1.0
        cache = model.rotary_emb.cos_sin_cache[: max_model_len + ROWS]
        half = cache.shape[-1] // 2
        self.cos = cache[:, :half].float().contiguous()
        self.sin = cache[:, half:].float().contiguous()
        self.dummy = torch.zeros(1024, dtype=torch.float32, device=dev)

        self.page = block_size
        self.bt_stride = -(-max_model_len // block_size)
        scr, sym = layout.layout(ROWS, HEADS, self.npes, TOPK, with_indexer=False, inter=INTER)
        spec = dict(
            kind="draft", rows=ROWS, heads=HEADS, npes=self.npes, inter=INTER, stages=list(ORDER),
            plan=[list(p) for p in PLAN],
            scratch=[[n, int(o)] for n, o in scr.items() if not n.startswith("_")],
            sym=[[n, int(o)] for n, o in sym.items() if isinstance(o, int)],
            launches_per_step=len(self.layers), index_page_size=block_size,
            index_bt_stride=self.bt_stride, index_request_width=ROWS,
            name=f"ml4_draft_r{ROWS}_tp{self.npes}", **KNOBS,
        )  # fmt: skip
        self.launch = rocke_layer.RockeLauncher(spec, root)
        self.scratch = torch.zeros(scr["_bytes"], dtype=torch.uint8, device=dev)
        self.peers = SymmetricPeerBuffer(max(sym["_bytes"], 64), rank=self.rank, npes=self.npes,
                                         group=get_tp_group().cpu_group)  # fmt: skip
        self.table = torch.zeros(16, dtype=torch.int64, device=dev)
        self.step = torch.zeros(1, dtype=torch.int32, device=dev)
        self.indices = torch.zeros(ROWS, TOPK, dtype=torch.int32, device=dev)
        self.indptr = torch.zeros(ROWS + 1, dtype=torch.int32, device=dev)
        self.k = torch.arange(TOPK, dtype=torch.int64, device=dev)
        self.rows = torch.arange(ROWS + 1, dtype=torch.int32, device=dev)
        self.x = [torch.empty(ROWS, HIDDEN, dtype=torch.bfloat16, device=dev) for _ in range(2)]
        self.refused: set[str] = set()
        logger.info("ML4 draft: %d draft layers as rocKE launches (%s, page %d, window %d)",
                    len(self.layers), self.launch.resources, block_size, self.window)  # fmt: skip

    def _refusal(self, positions: torch.Tensor) -> str | None:
        """Host state alike on every TP rank (the kernels reduce in-kernel: a
        rank that ran them alone would wait on its peers)."""
        if positions.shape[0] != ROWS or positions.dtype != torch.int64:
            return f"{positions.shape[0]} rows"
        if not is_forward_context_available():
            return "no forward context"
        ctx = get_forward_context()
        md, slots = ctx.attn_metadata, ctx.slot_mapping
        if not isinstance(md, dict) or not isinstance(slots, dict):
            return "no per-layer metadata"
        for layer, name in zip(self.model.layers, self.layer_names):
            kv = layer.self_attn.attn.kv_cache
            if (not isinstance(kv, torch.Tensor) or kv.dim() != 3 or kv.shape[1:] != (self.page, QK)
                    or kv.dtype != torch.bfloat16 or not kv.is_contiguous()):  # fmt: skip
                return "latent cache not bound (bf16 [blocks, page, 576])"
            m = md.get(name)
            if m is None or slots.get(name) is None or getattr(m, "num_prefills", 1):
                return "not a decode step"
            bt = getattr(getattr(m, "decode", None), "block_table", None)
            if bt is None:
                bt = getattr(m, "block_table", None)
            if bt is None or bt.dtype != torch.int32 or bt.stride(0) != self.bt_stride or bt.stride(1) != 1:
                return "block table layout"
        return None

    def __call__(self, positions: torch.Tensor, embeds: torch.Tensor) -> torch.Tensor | None:
        """The stream leaving the last draft layer, or None (vLLM's layers run)."""
        why = self._refusal(positions)
        if why is not None:
            if why not in self.refused:
                self.refused.add(why)
                logger.info("ML4 draft: a forward on vLLM's layers (%s)", why)
            return None
        ctx = get_forward_context()
        # the block's window: the last `window` positions up to its end
        end = positions[ROWS - 1] + 1
        start = torch.clamp(end - self.window, min=0)
        n = end - start
        self.indices.copy_(torch.where(self.k < n, start + self.k, 0).to(torch.int32).expand(ROWS, TOPK))
        self.indptr.copy_(self.rows * n.to(torch.int32))
        x = self.x
        stream = torch.cuda.current_stream()
        h = embeds
        for i, (t, name) in enumerate(zip(self.layers, self.layer_names)):
            m = ctx.attn_metadata[name]
            bt = getattr(getattr(m, "decode", None), "block_table", None)
            if bt is None:
                bt = m.block_table
            kv = self.model.layers[i].self_attn.attn.kv_cache
            out = x[i % 2]
            p = lambda v: v.data_ptr()  # noqa: E731
            self.launch(
                p(h), p(out), p(bt), p(positions), p(ctx.slot_mapping[name]), p(self.indptr),
                p(kv), p(kv), p(self.indices), p(self.cos), p(self.sin),
                p(t["g_in"]), p(self.g_q), p(t["g_kv"]), p(t["g_post"]),
                p(t["w_qkv_a"]), p(t["s_qkv_a"]), p(t["w_q_b"]), p(t["s_q_b"]),
                p(t["w_uk"]), p(t["s_uk"]), p(t["w_uv"]), p(t["s_uv"]),
                p(t["w_o"]), p(t["s_o"]), p(self.dummy), p(self.dummy),
                p(t["w_ug"]), p(t["s_ug"]), p(t["w_dn"]), p(t["s_dn"]),
                p(self.scratch), self.peers.local_address, p(self.peers.addresses), p(self.table),
                p(self.step), self.rank, i,
                stream=stream,
            )  # fmt: skip
            h = out
        self.step.add_(1)
        return h

    def close(self) -> None:
        if self.peers is not None:
            self.peers.close()
            self.peers = None
