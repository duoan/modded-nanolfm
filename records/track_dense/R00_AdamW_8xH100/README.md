# Track Dense — R00 (AdamW baseline on 8 × H100)

**Status:** complete; sets the wall-clock target for Track Dense at
**21.7 min** on 8 × H100. Val target (≤ 3.276, matching modded-nanogpt)
**not yet hit** — R00 plateaus at 3.3148, so R01+ needs to hit ≤ 3.276
in less wall-clock to be accepted.

This is the first apples-to-apples 8 × H100 data point we have against
modded-nanogpt's GPT-2-small AdamW baseline: same param count (~124 M),
same data (FineWeb-GPT2, 5 B tokens), same optimizer (fused AdamW), same
schedule (trapezoidal 250 / 7 286 / 2 000), only the model architecture
differs (LFM-hybrid vs GPT-2 Transformer).

## Result

| Metric                | Value       |
|-----------------------|-------------|
| Final val loss        | **3.3148**  |
| Total train time      | **21.7 min** (1 302 397 ms over 9 536 steps) |
| Step avg              | 136.7 ms    |
| Peak memory / GPU     | 14 754 MiB  |
| Hardware              | 8 × NVIDIA H100 80 GB HBM3 |
| Modal app id          | `ap-emHOmhpN5w3zU36Oxrc6tq` |


![Learning curve — R00 baseline 8xH100](curve.png)

Three series, log-scale y:
- **light-blue cloud** — per-step `train_loss` (one ~64 K-token batch each)
- **dark-navy line** — `train_loss` smoothed with a 128-step rolling mean
- **orange line** — `val_loss` (held-out, every 128 steps, ~16 M tokens)
- **red dashed** — the speedrun target (3.28)

The smoothed train and val basically overlap — that is the *real* learning
curve. Don't read the per-step cloud as model quality.

### Why per-step `train_loss` lies

The trainer logs raw single-batch losses; with 8 GPUs × ~8 K tokens that's
~64 K tokens per sample. The variance is huge — in the **final 128 steps**
alone the per-step train_loss spans **2.84 → 5.20**. Lucky batches read
3.13; unlucky ones read 5.0+. They average out to **3.32**, which is what
val_loss measures correctly with ~16 M tokens per checkpoint.

| Quantity (final 128-step window) | Value  |
|----------------------------------|-------:|
| Smoothed `train_loss` (128-step EMA) | **3.318** |
| `val_loss` (held-out)            | **3.315** |
| Per-step `train_loss` **min**    | 2.84   |
| Per-step `train_loss` **max**    | 5.20   |
| Per-step `train_loss` **stddev** | ≈ 0.27 |

### Loss-vs-wall-clock (smoothed train ≈ val throughout)

| Wall-clock | Step    | train (128-EMA) | val_loss | Δ to 3.28 |
|-----------:|--------:|----------------:|---------:|----------:|
|    0.2 s   |       0 | —               | 10.9864  | —         |
|    1.7 min |     768 | 4.028           |  3.9710  | 0.69      |
|    4.6 min |   2 048 | 3.679           |  3.6448  | 0.36      |
|    **8.4 min** |   3 712 | 3.545 | **3.5172**  | 0.24      |
|   14.3 min |   6 272 | 3.434           |  3.4370  | 0.16      |
|   17.0 min |   7 424 | 3.422           |  3.4133  | 0.13  ← warmdown begins |
|   19.5 min |   8 576 | 3.379           |  3.3442  | 0.064     |
|   21.7 min |   9 536 | **3.318**       |  **3.3148** | 0.035  |

(Per-step `train_loss` first ducks below 3.28 at step **1145 / 2.6 min**,
but that's noise — the smoothed train at that point is **3.86**, miles
above target.)

The bulk of the descent happens **fast**: we're past val 3.50 by 8.4 min and
past 3.40 by 14.3 min. The remaining 7 min (32 % of wall-clock) only buys
another 0.09 of loss — the curve is flattening to an asymptote a bit above
the 3.28 target. More training on the same schedule **probably will not**
close the 0.035 gap; that needs Muon / arch tweaks (Tracks R01+).

### vs reference

| Run                                                  | Wall-clock | Val loss |
|------------------------------------------------------|-----------:|---------:|
| modded-nanogpt AdamW baseline (record #2, 124 M GPT-2) | 31.4 min  | 3.276   |
| **this run** (122 M LFM2-hybrid, same data/optimizer/schedule) | **21.7 min** | **3.3148** |

Same param count (~124 M), same token budget (~5 B), same AdamW schedule
(trapezoidal: 250 warmup / 7 286 plateau / 2 000 cooldown), same FineWeb-GPT2
tokens. The LFM2-hybrid backbone is **~30 % cheaper wall-clock per step**
(136.7 ms vs ~200 ms on the same hardware), but its loss curve **plateaus
~0.035 above** GPT-2's at this depth/width. Whether that gap is structural
to the hybrid backbone or just an artifact of unoptimised hyperparameters /
schedule is what later records will tell us.

## What's in this record

```
R00_AdamW_8xH100/
├── f0372c77-938c-4258-a614-6f2038259e16.txt   # full trainer log (per-step loss + source header)
├── meta.txt                                    # run metadata (cmd, gpu, timing, modal id)
├── curve.png                                   # auto-plotted learning curve (scripts/plot_run.py)
└── snapshot/                                   # the exact code that ran
    ├── train_lfm.py                           # extracted bit-exact from the log header
    ├── src/{__init__,model,configs,optimizer,kernels}.py
    └── data/cached_fineweb10B.py
```

The `snapshot/train_lfm.py` was extracted from the log file's source-self-read
header. The `snapshot/src/` files are byte-identical to the working tree at the
time the run started (mtimes confirm no edits after 20:35 PDT), so this
snapshot is a faithful reproduction artifact.

## Historical note (run name vs current track)

The run was launched as `scripts/launch_modal.sh h100x8 d350m_R00` back when
we briefly experimented with a 5-track structure indexed by official LFM2
release sizes (D-350M / D-700M / ...). At that point, `TRACK` env passthrough
was not yet wired into `launch_modal.sh`, so the trainer fell back to
`LFMConfig()` (122 M params) regardless. We later collapsed the 5-track
structure down to **Dense / MoE** at modded-nanogpt scale -- the run name in
`meta.txt` (`d350m_R00`) and the `intended_track=d350m` field are vestiges
of that earlier framing. The actual config the run trained is the canonical
Track Dense baseline (`src/configs.dense_baseline()`), which is why it lives
here.
