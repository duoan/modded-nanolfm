# Track MoE — LFM-hybrid MoE speedrun (small variant)

Same backbone and (roughly) same active per-token compute as
[Track Dense](../track_dense/), but the SwiGLU MLP in every layer
is replaced with a **top-2 routed Mixture-of-Experts FFN**. The
goal is to measure what a sparse FFN buys at fixed active-compute
when everything else (data, tokens, optimizer, attention/conv
structure) is held constant.

**Status:** spec-only. `src/model.py` currently implements the
dense backbone; the MoE FFN + router still need to land before R00
can run. The architecture below is frozen in
[`src/configs.py::MOE_SMALL_SPEC`](../../src/configs.py) so the
future R00 has an unambiguous build target.

## Architecture (proposed)

| | |
|--|--|
| Layers | 12 (same as Dense) |
| `d_model` | 768 |
| Attention | 12 heads / 4 KV (GQA, same as Dense) |
| Conv kernel | 3 (same as Dense) |
| Layer pattern | conv / conv / attn (same as Dense) |
| **Experts per FFN** | **8** |
| **Top-k routing** | **top-2** |
| Per-expert FF dim | 1 024 (so active FF dim = top-2 × 1024 = 2 048, matches Dense) |
| Router | tiny linear `(d_model → n_experts)`, softmax + top-k |
| Load-balancing loss | aux loss weight `0.01` (Switch-Transformer style; TBD) |
| Vocab | 50 304 (GPT-2 BPE) |
| Tied embeddings | yes |
| **Total params** | **~235 M** |
| **Active params / token** | **~122 M** (≈ Dense) |
| Sparsity ratio | ~2× |

Per-token compute is roughly equal to Dense (top-2 of 8 experts = 2/8 of
the FF FLOPs), so steps should run at similar wall-clock to Dense R00.
The extra parameters (~110 M) are pure capacity, traded for either lower
loss at the same step count, or fewer steps to the same loss.

## Data budget

Same data layout as Dense (FineWeb-GPT2). MoE typically needs a few more
tokens to fully utilise the expert pool, so we set the **token budget
at 7 B** (vs Dense's 5 B). On 8 × H100 this is ~13 350 iters at
batch = 512 × 1024.

## Reference / target

There is no canonical "MoE-LFM-hybrid at this scale" baseline yet.
**R00 will set the target val CE** (and wall-clock) for the track.
The relevant comparison is then:

| Run | Active params | Total params | Tokens | Wall-clock | Val CE |
|-----|--------------:|-------------:|-------:|-----------:|-------:|
| Track Dense R00 | 122 M | 122 M | 5 B | 21.7 min | 3.3148 |
| Track MoE R00 (pending) | ~122 M | ~235 M | 7 B | TBD | TBD |

If MoE R00 hits a meaningfully lower val CE at similar wall-clock /
similar active params, the track is interesting. If not, MoE-small at
this scale isn't worth the implementation complexity.

## Records

| #   | Wall-clock | Val CE | Description | Date | Log | Contributor |
|-----|-----------:|-------:|-------------|------|-----|-------------|
| —   | —          | —      | track not active yet (MoE FFN not implemented) | — | — | — |
