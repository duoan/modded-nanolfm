# modded-nanolfm

This repository hosts the *LFM speedrun*: we (collaboratively | competitively)
search for the fastest single-node algorithm to train an **LFM-hybrid**
(Liquid Foundation Model: short causal convolutions interleaved with
grouped-query attention) at **~124 M params on ~5 B FineWeb-GPT2 tokens**,
mirroring the framing of
[KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
exactly so the only thing that varies between us and them is the
*architecture* (LFM-hybrid vs GPT-2 Transformer).

The architecture follows the
[LFM2 Technical Report](https://arxiv.org/abs/2511.23404) and the HF
[`Lfm2`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2/modular_lfm2.py)
reference: gated short causal convolutions interleaved with grouped-query
attention (QK-Norm + Gemma-style RoPE), with a SwiGLU MLP. The MoE track
swaps the MLP for a top-2 routed Mixture-of-Experts (HF
[`Lfm2MoE`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2_moe/modular_lfm2_moe.py)-style
but at the 124 M-active scale instead of 1.5 B).

We run **2 tracks**, both at modded-nanogpt scale:

| Track | Variant | Architecture | Total params | Active per token | Status |
|-------|---------|--------------|-------------:|-----------------:|--------|
| [Dense](records/track_dense/) | LFM-hybrid + SwiGLU MLP | 12 L · d=768 · FF=2048 · 12Q / 4KV | ~122 M | ~122 M | **active (R00 set)** |
| [MoE](records/track_moe/)     | LFM-hybrid + 8-expert top-2 MoE FFN | 12 L · d=768 · FF<sub>moe</sub>=1024 · 12Q / 4KV | ~235 M | ~122 M | spec-only (MoE FFN not yet implemented) |

Both tracks share the same trainer at HEAD (`train_lfm.py`); the FFN
variant is chosen by the `TRACK` env var (`TRACK=dense` ≡ unset;
`TRACK=moe` will be wired once MoE lands in `src/model.py`). Per-track
targets, records, and architecture details live under
[`records/track_*/`](records/).

### Reference baseline (what we're racing)

| | modded-nanogpt record #2 | this repo, Track Dense R00 |
|--|--|--|
| Backbone | GPT-2 Transformer (124 M) | LFM-hybrid (122 M) |
| Optimizer | AdamW | AdamW (same fused impl) |
| Schedule | trapezoidal 250 / 7 286 / 2 000 | identical |
| Tokens | ~5 B FineWeb-GPT2 | identical |
| Hardware | 8 × H100 | identical |
| Wall-clock | 31.4 min | **21.7 min** (–30%) |
| Final val CE | **3.276** | 3.3148 (above target by 0.04) |

R00 already beats wall-clock; missing the val target is what R01+ closes.

---

## Running the current record

```bash
git clone <this-repo> modded-nanolfm && cd modded-nanolfm
uv sync
uv run python data/cached_fineweb10B.py 50    # ~5 B tokens of FineWeb-GPT2
TRACK=dense ./run.sh                          # → records/track_dense/R00 (current record)
```

Add `torchrun` to `PATH` if `./run.sh` errors with `torchrun: command not found`.

> **Note**: `torch.compile` adds ~30 s of latency on the first run.

Official records are timed on **8 × H100** (same canonical hardware as
modded-nanogpt). The local dev path uses an RTX PRO 6000 Blackwell
(96 GB GDDR7) as a single-GPU smoke rig; logs land in
`records/R00_baseline_lfm2/` (preserved as the original local sanity
check) and don't overwrite the canonical 8 × H100 record.

### Useful env overrides

```bash
TRACK=dense              ./run.sh   # Track Dense (≡ unset TRACK)
TRACK=moe                ./run.sh   # Track MoE (raises until MoE FFN is implemented)

SMOKE=1                  ./run.sh   # 50 iters, no compile -- pipeline check
NPROC=8                  ./run.sh   # 8-GPU local run (DDP via torchrun)

NUM_ITERATIONS=20000 LEARNING_RATE=2e-3 ./run.sh    # ad-hoc tweak
```

See [`train_lfm.py`](train_lfm.py) `_apply_env_overrides` for the full set
(USE_COMPILE, NO_COMPILE, DEVICE_BATCH_SIZE, BATCH_SIZE, SEQUENCE_LENGTH,
WEIGHT_DECAY, GRAD_CLIP, VAL_LOSS_EVERY, VAL_TOKENS).

---

## Running on cloud (Modal: H100 / B200)

For canonical timing we use [Modal](https://modal.com). One-time setup:

```bash
uv sync
uv run modal setup                                              # OAuth
uv run modal run modal_app.py::download_data --num-chunks 50    # ~5 B tokens, persists
```

Then launch the track of your choice on the appropriate GPU profile:

```bash
# Track Dense -- canonical 8 × H100 timing
TRACK=dense  scripts/launch_modal.sh h100x8 dense_R01     # next-record attempt
TRACK=dense  scripts/launch_modal.sh b200x8 dense_b200    # 8 × B200 throughput (off-canonical)

# Track MoE  -- once src/model.py implements MoE FFN
TRACK=moe    scripts/launch_modal.sh h100x8 moe_R00

# Single-GPU dev iteration
TRACK=dense  scripts/launch_modal.sh b200x1 dense_dev
```

The second arg is a **TAG**; `launch_modal.sh` auto-prepends a UTC
`YYYYMMDD_HHMM_` prefix to produce a unique `RUN_NAME` (the dir name
on the Modal volume). So `TRACK=dense scripts/launch_modal.sh h100x8
dense_R01` actually creates `logs/20260528_0455_dense_R01/`. Run the
same command 2 minutes later and you get `..._0457_dense_R01/` — no
collision. This is what makes simultaneous launches safe.

`scripts/launch_modal.sh` uses `modal run --detach` so the run survives your
SSH dropping. To inspect / pull / persist:

```bash
uv run modal app list                # find active app ids
uv run modal app logs <app-id>       # stream live stdout/stderr from Modal

scripts/sync_modal_logs.sh           # pull on-disk artifacts → ./logs/modal/
scripts/sync_modal_logs.sh --watch   # poll every 60s; safe to leave running

# When a run is worth keeping forever, promote it into the git-tracked records/.
# Auto-generates a curve.png and auto-derives the destination from the
# timestamped log dir name (e.g. logs/modal/20260528_0455_dense_R01/ ->
# records/track_dense/20260528_0455_dense_R01/).
scripts/promote_record.sh                                # latest log, auto-everything
scripts/promote_record.sh track_dense/R01_AdamW          # explicit DEST (any name you want)

# (Manual plotting, e.g. for a still-running local run:)
uv run scripts/plot_run.py logs/<uuid>.txt   # writes logs/curve.png by default
```

### Storage tiers (read this once — it matters)

| Tier                  | Path                              | Persistence | Cost |
|-----------------------|-----------------------------------|-------------|------|
| Modal Volume          | `nanolfm-logs` (remote)           | until **you** delete the Volume to save money | $$ |
| Local staging         | `./logs/modal/<run_name>/`        | until you `rm -rf logs/`, machine dies, or you re-init | free |
| **Git-tracked record**| `./records/track_*/R<NN>_<name>/` | **forever**, in commit history | free |

> The first two tiers are **transient**. The only way a run survives long-term
> is `scripts/promote_record.sh` → `git commit`. Get into the habit of
> promoting accepted runs the same day you sync them — otherwise the Volume
> eviction (or `rm -rf logs/`) takes them out.

Each Modal run produces a run-named directory on the volume with everything
needed to reproduce it, so promotion is essentially `cp -r`:

```
logs/modal/<run_name>/
├── snapshot/                  # frozen at run-start, bit-exact code that ran
│   ├── train_lfm.py
│   ├── src/{model,configs,optimizer,kernels,__init__}.py
│   └── data/cached_fineweb10B.py
├── meta.txt                   # run_name, track, gpu, start/end, exit_code, modal_app_id
└── <uuid>.txt                 # trainer log (own source as header + per-step lines)
```

See [`modal_app.py`](modal_app.py) for the full set of profiles
(`train_h100x8`, `train_b200x8`, `train_h100x1`, `train_b200x1`,
`download_data`) and the volume layout.

---

## Track Dense

LFM-hybrid backbone (gated short convs + GQA + QK-Norm + Gemma RoPE) with a
vanilla SwiGLU MLP. Direct apples-to-apples comparison with modded-nanogpt's
GPT-2-small AdamW baseline. Full spec, target, and per-record narrative:
[`records/track_dense/README.md`](records/track_dense/README.md).

| #   | Wall-clock | Val CE | Description | Date | Record | Contributor |
| --- | ---------: | -----: | ----------- | ---- | ------ | ----------- |
| R00 | **21.7 min** | 3.3148 (above target) | LFM2-hybrid + AdamW + modded-nanogpt schedule | 2026-05-27 | [20260528_0335_dense_R00_AdamW_8xH100](records/track_dense/20260528_0335_dense_R00_AdamW_8xH100/) | initial |

R00 sets the wall-clock target at 21.7 min on 8 × H100; subsequent records
have to hit val ≤ **3.276** (modded-nanogpt's GPT-2 number) in *less*
wall-clock to be accepted.

[![Track Dense R00 learning curve](records/track_dense/20260528_0335_dense_R00_AdamW_8xH100/curve.png)](records/track_dense/20260528_0335_dense_R00_AdamW_8xH100/)

Smoothed train (navy) and val (orange) overlap throughout — that's the real
learning curve; the light-blue cloud is per-step train noise. See
[the record's README](records/track_dense/20260528_0335_dense_R00_AdamW_8xH100/)
for the loss-vs-wall-clock breakdown.

---

## Track MoE

Same backbone as Track Dense, but the FFN in every layer is replaced with a
top-2 routed Mixture-of-Experts (8 experts, per-expert FF=1024). Per-token
active compute matches Dense (~122 M active params), but total capacity grows
to ~235 M. Tests what a sparse FFN buys at fixed active-compute.

**Spec-only for now** — `src/model.py` implements the dense backbone, not
the MoE FFN + router. The full architectural spec is frozen in
[`src/configs.py::MOE_SMALL_SPEC`](src/configs.py) so the future R00 has an
unambiguous build target. Full track scope:
[`records/track_moe/README.md`](records/track_moe/README.md).

| #   | Wall-clock | Val CE | Description | Date | Record | Contributor |
| --- | ---------: | -----: | ----------- | ---- | ------ | ----------- |
| —   | —          | —      | track not yet active (MoE FFN not implemented) | — | — | — |

---

## Rules

A new record on any track is accepted iff:

1. It attains **≤ that track's target val CE** on the FineWeb val stream
   (Dense: 3.276, matching modded-nanogpt; MoE: TBD at R00). Submissions
   should provide enough run logs to achieve `p < 0.01` statistical
   significance; for systems-only speedups that don't touch the ML, this
   requirement is waived.
2. It runs **faster wall-clock** than the prior record on the canonical
   **8 × H100** hardware (both tracks).
3. It does **not** modify the train / val token streams. (Batch size,
   sequence length, attention pattern within the architecture, etc. are fair
   game; tokens are not.)
4. It does **not** flip global `torch._inductor.config` or `torch.compile(...)`
   flags that materially affect compile time.

Discretionary reasons a PR may be rejected:

1. Disproportionately degrades codebase readability. A 200-line kernel that
   drops 300 ms is worth it; 500 lines of optimizer plumbing for 50 ms is not.
2. Substantially consumes the val-loss buffer (we keep the current record's
   mean loss ~0.001-0.002 below target to make validation simpler).

---

## Layout

```
modded-nanolfm/
├── train_lfm.py            # HEAD trainer (single file, current record)
├── run.sh                  # local launcher (uses TRACK env var)
├── modal_app.py            # Modal app: H100/B200 single- or 8-GPU profiles
├── pyproject.toml          # uv-managed deps
├── data/
│   └── cached_fineweb10B.py
├── src/
│   ├── model.py            # LFM2-style backbone (RMSNorm / RoPE / GQA+QKN / ShortConv / SwiGLU)
│   ├── configs.py          # dense_baseline() + MOE_SMALL_SPEC + TRACK registry
│   ├── optimizer.py        # build_optimizers(...) -> list[Optimizer]
│   └── kernels.py          # custom Triton kernels (stubbed; populated by later records)
├── scripts/
│   ├── launch.sh           # detached local launcher (setsid + nohup)
│   ├── launch_modal.sh     # detached Modal launcher (--detach)
│   ├── sync_modal_logs.sh  # pull Modal logs volume → ./logs/modal/  (transient)
│   ├── promote_record.sh   # promote a synced run dir → ./records/   (permanent)
│   ├── plot_run.py         # parse trainer log → curve.png (auto-run by promote)
│   ├── status.sh           # local run status
│   └── stop.sh             # local run kill switch
└── records/
    ├── R00_baseline_lfm2/  # original local single-GPU smoke (RTX PRO 6000); kept as sanity
    ├── track_dense/        # Track Dense records; current = 20260528_0335_dense_R00_AdamW_8xH100/
    └── track_moe/          # Track MoE spec + records (spec-only until MoE FFN lands)
```

## Architecture (shared backbone)

`src/model.py` mirrors HuggingFace `Lfm2` (dense) with the inference plumbing
stripped. Both tracks share the same backbone; only the FFN differs.

- **Shape (both tracks)**: 12 layers, `d_model=768`, 12 attention heads /
  4 KV heads (GQA), conv kernel 3, vocab 50 304 (GPT-2 BPE padded), tied
  embeddings.
- **Layer pattern (`layer_types`)**: alternating short-conv / attention
  (every layer has one of the two; same pattern modded-nanolfm has used
  since the 122 M pipeline).
- **Attention**: `q_proj` / `k_proj` / `v_proj` / `out_proj`, GQA (12Q / 4KV),
  per-head RMSNorm on Q and K (QK-Norm), Gemma-style RoPE (`theta = 1e6`),
  `F.scaled_dot_product_attention`.
- **ShortConv** ("Gated-Conv"): `in_proj(D → 3D)` → split into `(B, C, x)` →
  `B * x` → causal depthwise `nn.Conv1d(K=3)` → `C * conv_out` → `out_proj`.
- **FFN (Track Dense)**: SwiGLU (`w1 / w3 / w2`, `silu(w1(x)) * w3(x)`),
  `intermediate_size = 2048`.
- **FFN (Track MoE, not yet implemented)**: 8 experts, top-2 routed,
  per-expert SwiGLU at `intermediate_size = 1024`. Tiny linear router with
  Switch-style aux load-balancing loss. Spec frozen in
  [`src/configs.py::MOE_SMALL_SPEC`](src/configs.py).
- **Norms**: RMSNorm in fp32 (stability guardrail), everything else bf16.

## Optimiser (R00)

`src/optimizer.build_optimizers` returns `list[Optimizer]` — a single fused
AdamW with two param groups:

- `decay` (wd = 0.1): all 2D+ projections / conv kernels.
- `no_decay` (wd = 0): embeddings, biases, RMSNorm gains.

The naming-convention hooks for Muon (`weight_projection` / `weight_conv`) are
documented in `src/__init__.py` so a Muon record can flip them on without
touching the trainer.

## Optimization sub-track (planned)

Modeled on
[modded-nanogpt's track 3](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_3_optimization):
fix architecture / data / batch size, **minimize step count** with unlimited
wall-clock budget. This isolates the optimizer's contribution from systems
work. Will launch first on Track Dense (Muon vs AdamW at the 122 M shape).

---

## References

- LFM2 Technical Report: https://arxiv.org/abs/2511.23404
- HF `Lfm2`: https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2/modular_lfm2.py
- HF `Lfm2MoE`: https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2_moe/modular_lfm2_moe.py
- modded-nanogpt: https://github.com/KellerJordan/modded-nanogpt
- The Muon optimizer: https://kellerjordan.github.io/posts/muon/

## Citation

```
@misc{modded_nanolfm_2026,
  title        = {modded-nanolfm: Speedrunning an LFM-hybrid at the modded-nanogpt scale},
  year         = {2026},
  note         = {Methodology mirrors KellerJordan/modded-nanogpt; architecture
                  follows Liquid AI's LFM2 Technical Report (arXiv:2511.23404).
                  124 M-param dense and 235 M-param top-2 MoE variants, both
                  at ~122 M active params, trained on ~5 B FineWeb-GPT2 tokens.}
}
```
