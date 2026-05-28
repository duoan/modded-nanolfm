# R00 — LFM2-dense baseline + AdamW

**Status:** snapshot frozen. **Smoke run passed** on RTX PRO 6000 Blackwell
(`compile=True`, 30 timed iters, 1 FineWeb-10B shard). Full 5 B-token run not
yet executed (see "Projected full-run timing" below).

This is the cleanest possible LFM2-flavoured starting point for the speedrun,
analogous to modded-nanogpt's pre-Muon AdamW baseline (which itself reached
3.276 val loss in 9536 iters on 8 × H100).

## What's in this record

```
R00_baseline_lfm2/
├── train_lfm.py        # single-file trainer (this snapshot is what produced the log)
├── src/
│   ├── model.py        # LFM2-dense backbone (RMSNorm, RoPE, GQA+QKNorm, ShortConv, SwiGLU)
│   ├── optimizer.py    # build_optimizers -> [fused AdamW] with decay/no-decay groups
│   ├── kernels.py      # stub (no Triton kernels yet)
│   └── __init__.py
└── data/
    └── cached_fineweb10B.py
```

## Architecture

| Field                  | Value                                                                  |
|------------------------|------------------------------------------------------------------------|
| Vocab                  | 50304 (GPT-2 BPE padded to multiple of 128)                            |
| Hidden size            | 768                                                                    |
| Layers                 | 12                                                                     |
| Layer pattern          | `c c c c c a c c c a c c` (10 ShortConv + 2 Attention, attn at 5, 9)   |
| Attn heads             | 12 query, 4 KV (GQA 3:1)                                               |
| Head dim               | 64                                                                     |
| QK-Norm                | per-head RMSNorm on Q and K                                            |
| ShortConv kernel       | depthwise causal Conv1d, K=3                                           |
| FFN                    | SwiGLU, intermediate 2048                                              |
| RoPE base              | 10 000                                                                  |
| Embedding tying        | yes (`embed_tokens.weight is lm_head.weight`)                          |
| **Parameter count**    | **122.04 M** (decay: 83.4 M, no-decay: 38.6 M ≈ tied embedding)        |

Faithful subset of HuggingFace
[`Lfm2`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2/modular_lfm2.py)
with all inference / KV-cache / fast-path plumbing removed.

## Optimisation (aligned with modded-nanogpt AdamW baseline)

| Field                    | Value                                              |
|--------------------------|----------------------------------------------------|
| Optimiser                | `torch.optim.AdamW(fused=True)`                    |
| Betas                    | (0.9, 0.95)                                        |
| Eps                      | 1e-10                                              |
| Weight decay             | **0.0** (matches modded-nanogpt; later records can sweep) |
| LR (peak)                | **1.8e-3**                                          |
| LR schedule              | Trapezoidal — linear warmup → plateau → linear warmdown |
| Warmup / warmdown iters  | **250 / 2000**                                      |
| Grad clip                | 1.0                                                |
| Precision                | bf16 activations / fp32 RMSNorm gains              |
| `torch.compile`          | on (`mode='default'`)                              |
| DDP                      | optional (`broadcast_buffers=False`)               |

## Training schedule (same token budget as modded-nanogpt AdamW baseline)

| Field                  | Value                                              |
|------------------------|----------------------------------------------------|
| Seq len                | 1024                                               |
| Device batch size      | 32 sequences (configurable; fits on 1× Blackwell)  |
| Global batch (tokens)  | 512 seqs × 1024 = **2¹⁹ = 524 288 tokens / step**  |
| Grad accumulation      | 512 / (device_B × world)                            |
| **Iterations**         | **9 536**                                            |
| **Total train tokens** | **9 536 × 2¹⁹ ≈ 5.00 B**                            |
| Val tokens per check   | 10 485 760 (10 M)                                   |
| Val every              | 128 steps (≈ 74 checks per run)                     |

## Smoke run (verified)

Smoke mode (`SMOKE=1`) collapses the budget to 30–50 steps so the full
end-to-end pipeline can be validated in minutes. Captured here as `smoke.log`.

| Step | val_loss | train_loss | wall-clock (post-warmup) |
|-----:|---------:|-----------:|--------------------------|
| 0    | 10.9863  | 10.9863    | 0 ms (init)              |
| 30   | 6.8508   | 6.7732     | 33.7 s (compile-on)      |

Per-step wall-clock after the 10-step kernel-warmup, with `torch.compile`:
**1.68 s / step** at 524 288 tokens / step (≈ 312 K tokens / s, ≈ 30 % of
Blackwell's bf16 peak). Peak memory **14.6 GiB / 96 GiB** — plenty of slack to
push `device_batch_size` higher in R01 if it helps utilisation.

For reference, the modded-nanogpt AdamW baseline at step 32 (~equivalent
token count) had `train_loss=7.555`. Our LFM-style architecture lands at
`6.77` at the same point — the conv blocks are doing useful work early.

### Projected full-run timing on the same hardware

| Quantity                | Estimate                                 |
|-------------------------|------------------------------------------|
| Train steps × per-step  | 9 536 × 1.68 s ≈ **16 000 s ≈ 4.5 h**    |
| Val checks × per-check  | 74 × ~14 s ≈ 17 min                       |
| Compile + warmup        | ~30 s                                     |
| **Total wall-clock**    | **≈ 4.7 h** on 1 × Blackwell              |

The first measured loss target is **≤ 3.28 val CE on FineWeb** (same as
modded-nanogpt). The AdamW baseline reached 3.276 there with the same token
budget; on this LFM architecture we expect to land in a similar neighbourhood.

## Reproduce

```bash
uv sync

# Smoke (~3 min, ~400 MB of data) — validates the pipeline end-to-end.
uv run python data/cached_fineweb10B.py 1
SMOKE=1 ./run.sh

# Full R00 (~5 h, ~10 GB of data) — produces a comparable val curve.
uv run python data/cached_fineweb10B.py 9
./run.sh
```

Logs land in `logs/<uuid>.txt` with the trainer source as a header. Promote
the canonical log into this directory once the full run completes:

```bash
cp logs/<uuid>.txt records/R00_baseline_lfm2/log.txt
```

## Known limitations (each becomes a later record)

- Eager `nn.Conv1d` for ShortConv (no document-boundary `cu_seqlens`). → **R03**
- `F.scaled_dot_product_attention` with default dispatcher (no explicit FA3). → **R04**
- Full `[B, S, V]` logits materialised by `lm_head` + `F.cross_entropy`. → **R07**
- Fixed sequence length (no warmup 2K → 8K → 16K). → **R05**
- Logits not soft-capped. → **R02**
- AdamW does all the work; Muon's spectral updates arrive in **R01**.
- ~30 % of Blackwell bf16 peak — most kernels still untuned.
