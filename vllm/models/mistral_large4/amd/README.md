# Mistral Large 4 on ROCm: the DSpark draft path

`dspark.py` is the DSpark drafter (the ROCm counterpart of `../nvidia/dspark.py`).
Three modules here cut the time of its work in a decode verify step. Each is a
switch, and each changes at most which draft tokens are proposed, never the
target's output: the rejection sampler checks every draft token against the
target's own logits. The target's decode kernels are documented in
[`specialized_decode/README.md`](specialized_decode/README.md).

| module | switch (default) | what it does |
|---|---|---|
| `fused_markov.py` | `VLLM_ML4_FUSED_MARKOV` (on) | greedy sampling of the recurrent (`rnn`) Markov head: each draft step's two GEMVs as they are and three launches around them, instead of about 17 kernels |
| | `VLLM_ML4_VP_MARKOV` (on) | the same, vocabulary-parallel at TP > 1: each rank soft-caps its own shard of the base logits, adds its shard of the Markov bias and takes its shard's best token, and the ranks exchange one (value, token) pair each instead of gathering the full-vocabulary logits; the tokens are the gathered loop's, bit for bit |
| `draft_fp8.py` | `VLLM_ML4_DRAFT_FP8` (off) | FP8 copies (per-tensor scales) of the draft's lm_head shard and of target_projection, read by vLLM's FP8 skinny GEMM: half their weight bytes; the draft tokens can change |
| | `VLLM_ML4_DRAFT_Q128` (off) | the draft layers' linear weights rounded through block FP8 at load: the fused draft-layer kernel's weights on vLLM's own layers, to measure their acceptance cost |
| `rocke_draft.py` | `VLLM_ML4_DRAFT_ROCKE` (off) | each draft layer of a 4-row decode block as one persistent rocKE launch: attention over the layer's sliding window, the dense MLP and both TP all-reduces in one kernel |

## Fused draft layers (`VLLM_ML4_DRAFT_ROCKE`)

The kernel is the rocKE ML4 layer family's `draft` kind
(`library/kernels/gfx950/ml4_mono/layer` in the rocKE tree, launched through
aiter's `rocke_layer.RockeLauncher`); `AITER_ML4_ROCKE` names the rocKE tree.
After the draft weights load, `RockeDraft` rounds the draft layers' bf16
linears through block FP8 (128 x 128 blocks, `W_UK^T` 128 x 64) into the
kernel's tile layouts, builds the kernel, and allocates its scratch and the
symmetric peer buffer of its TP reductions (a collective of the TP group),
before vLLM's memory profiling.

A forward then runs the three layers as three launches when it is a decode
step of one request's 4-row block: the window positions of the block (the
last `sliding_window` positions up to its end) go to the kernel as its key
list, and the kernels read and write vLLM's bf16 MLA latent pages through
each layer's block table and slot mapping. The result is the residual stream
itself (already TP-reduced), so `final_norm` follows directly. Any other
forward runs vLLM's layers, the reason logged once: a different row count, a
prefill, unbound caches, or a block table of another width. The build is
refused (vLLM's layers throughout) at a TP size other than 4, another
attention or MLP geometry, non-bf16 draft weights, or a window wider than
the kernel's key list.

`aiter/op_tests/flydsl_tests/check_rocke_ml4_draft.py` checks the kernel
against a PyTorch model of these layers on one GPU or TP4.
