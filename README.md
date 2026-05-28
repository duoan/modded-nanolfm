# modded-nanolfm

> Speedrun an **LFM (Liquid Foundation Model)** hybrid backbone on a single
> Blackwell node. Methodology mirrors
> [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt):
> one self-contained trainer at HEAD (`train_lfm.py`), every accepted record
> frozen as an immutable snapshot under `records/`, README maintains the world
> record table.

## Goal

Train an LFM-style hybrid model (RMSNorm + RoPE + GQA/QK-Norm + ShortConv +
SwiGLU, layer pattern à la [HuggingFace `Lfm2`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2/modular_lfm2.py))
to **≤3.28 validation cross-entropy on FineWeb** in the minimum wall-clock,
on a single RTX PRO 6000 Blackwell (96 GB GDDR7).

The first record is the cleanest possible LFM2-dense + AdamW baseline. Every
subsequent record must beat the prior one on wall-clock (and preserve
≤3.28 val loss).

## Quickstart

```bash
# 1. install
uv sync

# 2. download data
uv run python data/cached_fineweb10B.py 1       # smoke: ~400 MB
uv run python data/cached_fineweb10B.py 9       # full R00: ~9 GB

# 3. run the current HEAD trainer
SMOKE=1 ./run.sh               # ~3 min smoke (50 iters, no compile, 1 shard)
./run.sh                       # full R00 (~4.5 h projected on 1× Blackwell)
NPROC=8 ./run.sh               # multi-GPU if you have it
```

`run.sh` invokes `torchrun --standalone --nproc_per_node=$NPROC train_lfm.py`.
The trainer logs everything to `logs/<uuid>.txt`, including a copy of its own
source as the header — every log is a perfect reproduction artifact of the
code that produced it.

Useful env overrides (no need to edit `train_lfm.py` for ad-hoc experiments):

| Env                 | Effect                                                       |
|---------------------|--------------------------------------------------------------|
| `SMOKE=1`           | 50 iters, no compile, val_tokens = 1 batch — finishes in min |
| `USE_COMPILE=0`     | force `torch.compile` off (debug / first-bring-up)           |
| `USE_COMPILE=1`     | force `torch.compile` on (overrides `SMOKE=1`'s default)     |
| `NUM_ITERATIONS=N`  | override iteration count                                     |
| `LEARNING_RATE=X`   | override peak LR                                             |
| `DEVICE_BATCH_SIZE` | override per-GPU sequences (default 32)                      |

## Layout

```
modded-nanolfm/
├── train_lfm.py            # HEAD trainer (single file, current record)
├── run.sh                  # canonical launcher
├── pyproject.toml          # uv-managed deps
├── data/
│   └── cached_fineweb10B.py    # downloads GPT-2 pre-tokenised shards
├── src/
│   ├── model.py            # LFM2-style backbone (RMSNorm / RoPE / GQA+QKN / ShortConv / SwiGLU)
│   ├── optimizer.py        # build_optimizers(...) -> list[Optimizer]
│   └── kernels.py          # custom Triton kernels (stubbed; populated by later records)
└── records/
    └── R00_baseline_lfm2/  # frozen snapshot of the R00 trainer + src/
```

## Speedrun trajectory

The board below is the **plan**; numbers fill in as records land and reproduce
on the same hardware. The first row is the only one that already exists.

| #     | Wall-clock | Description                                                                                  | Status |
|-------|-----------:|----------------------------------------------------------------------------------------------|--------|
| R00   | ≈4.5 h proj. | LFM2-dense baseline (122M, depth 12, hidden 768, GQA 12→4, K=3 ShortConv) + fused AdamW (5 B-token budget, matches modded-nanogpt AdamW baseline) | smoke verified; full run pending |
| R01   | —          | **Muon** for 2D `weight_projection` matrices; AdamW keeps embeddings/biases/norms/conv weights | planned |
| R02   | —          | Architectural modernisation: zero-init out_proj, ReLU² toggle, **logit softcap = 30**, pad embedding | planned |
| R03   | —          | Causal `causal_conv1d` Triton kernel (replace eager `nn.Conv1d` in ShortConv)                | planned |
| R04   | —          | **Flash Attention 3** backend (Blackwell SM_100); FA3-friendly head_dim                      | planned |
| R05   | —          | Sequence-length warmup (2K → 8K → 16K) per spec; document-boundary `cu_seqlens`              | planned |
| R06   | —          | **FP8 LM head** (Blackwell mx-fp8 GEMM)                                                       | planned |
| R07   | —          | **Fused LM head + cross-entropy** Triton kernel (no `[B, S, V]` materialisation in HBM)      | planned |
| R08   | —          | Embedding / value-residual skip connections                                                  | planned |
| R09   | —          | `torch.compile(fullgraph=True)` + asymmetric logit rescale                                   | planned |
| R10   | —          | NorMuon / Polar Express NS iteration                                                         | planned |
| R11   | —          | Online Top-K distillation from a frozen Qwen2.5-72B teacher (vLLM)                           | planned |
| R12+  | —          | **Fused ShortConv + RHT + MXFP4 (G=32)** Triton kernel with on-chip dequant on backward      | planned |

(See `src/kernels.py` for the kernel-side trajectory.)

## Architecture (R00)

`src/model.py` mirrors HuggingFace `Lfm2` (dense) with the inference plumbing
removed:

- **Layer pattern (`layer_types`)**: layers 0-1 anchored as `ShortConv`; from
  layer 2 onward, every 4th layer is `Attention`. Spec calls this 3:1 conv:attn.
  Override via `LFMConfig.layer_types`.
- **Attention**: `q_proj` / `k_proj` / `v_proj` / `out_proj`, GQA, per-head
  RMSNorm on Q and K (QK-Norm), Gemma-style RoPE, `F.scaled_dot_product_attention`.
- **ShortConv** (= "Gated-Conv"): `in_proj(D → 3D)` → `chunk(B, C, x)` →
  `B * x` → causal depthwise `nn.Conv1d(K=3)` → `C * conv_out` → `out_proj`.
- **MLP**: SwiGLU (`w1 / w3 / w2`, `silu(w1(x)) * w3(x)`).
- **Norms**: RMSNorm in fp32 (stability guardrail), everything else bf16.
- **Tied embeddings** by default (`tie_word_embeddings=True`).

## Optimiser (R00)

`src/optimizer.build_optimizers` returns `list[Optimizer]` — a single fused
AdamW with two param groups:

- `decay` (wd = 0.1): all 2D+ projections / conv kernels.
- `no_decay` (wd = 0): embeddings, biases, RMSNorm gains.

The naming-convention hooks for Muon (`weight_projection` / `weight_conv`) are
documented in `src/__init__.py` so R01 can flip them on without touching the
trainer.

## Rules (mirroring modded-nanogpt)

A new record is accepted iff:

1. It hits **≤3.28 mean val CE on FineWeb** with enough runs to give `p < 0.01`.
2. It runs **faster wall-clock** than the prior record on the same hardware.
3. It does **not** alter the train/val token streams (batch shape and seq-len
   schedule are fair game).
4. It does **not** flip global `torch._inductor.config` or
   `torch.compile(...)` flags that materially affect compile time.

References:
- HF `Lfm2`: https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2/modular_lfm2.py
- HF `Lfm2MoE`: https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2_moe/modular_lfm2_moe.py
- modded-nanogpt: https://github.com/KellerJordan/modded-nanogpt
