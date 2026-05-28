# Track Dense — LFM-hybrid AdamW speedrun

Same scale and data budget as
[modded-nanogpt's GPT-2-small AdamW baseline](https://github.com/KellerJordan/modded-nanogpt),
swapping the backbone for an **LFM2-style hybrid** (gated short
causal convolutions interleaved with grouped-query attention,
QK-Norm, Gemma-style RoPE, SwiGLU MLP, tied embeddings).

## Architecture (fixed for this track)

| | |
|--|--|
| Layers | 12 |
| `d_model` | 768 |
| FFN dim | 2 048 (SwiGLU) |
| Attention | 12 heads / 4 KV (GQA) |
| Conv kernel | 3 |
| Layer pattern | conv / conv / attn (repeating; same as the 122 M pipeline shape) |
| Vocab | 50 304 (GPT-2 BPE) |
| Tied embeddings | yes |
| Parameter count | **~122 M** |

This shape is what `TRACK=dense` (or unset `TRACK=`) resolves to in
`train_lfm.py` via `src.configs.dense_baseline()`.

## Reference: modded-nanogpt baseline

modded-nanogpt's record #2 (AdamW baseline on 124 M GPT-2-small):

| | |
|--|--|
| Params | 124 M |
| Tokens | ~5 B (FineWeb-GPT2) |
| Hardware | 8 × H100 |
| Wall-clock | **31.4 min** |
| Final val CE | **3.276** |

A new record on this track is **accepted** iff it hits val CE ≤ **3.276**
in **less wall-clock** than the prior record on 8 × H100.

## Records

| #   | Wall-clock | Val CE  | Description | Date | Log | Contributor |
|-----|-----------:|--------:|-------------|------|-----|-------------|
| R00 | 21.7 min   | 3.3148 (above target) | [LFM2-hybrid + AdamW, modded-nanogpt schedule](R00_AdamW_8xH100/) | 2026-05-27 | [log](R00_AdamW_8xH100/f0372c77-938c-4258-a614-6f2038259e16.txt) | initial |

R00 is faster wall-clock than GPT-2 AdamW (~30% per step on the same
hardware) but the loss curve plateaus ~0.04 above the target. The
target-hit + faster combo is what R01+ needs to deliver. See
[`R00_AdamW_8xH100/README.md`](R00_AdamW_8xH100/) for the loss curve
and per-checkpoint breakdown.
