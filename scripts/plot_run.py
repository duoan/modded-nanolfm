#!/usr/bin/env python3
"""Plot the train + val loss curves from a modded-nanolfm trainer log.

The trainer writes per-step lines of the form::

    step:NNNN/TOTAL train_loss:F.FFFF train_time:Nms step_avg:Fms
    step:NNNN/TOTAL val_loss:F.FFFF   train_time:Nms step_avg:Fms

This script parses those lines and produces a single PNG with two panels:
loss vs step (left) and loss vs wall-clock minutes (right). Train loss is
drawn faint (per-step is noisy); val loss is the bold curve. A dashed line
marks the speedrun target (default 3.28). The point where wall-clock
crosses the 80th percentile of total time is annotated so readers can
spot the asymptote.

Usage::

    scripts/plot_run.py LOG_FILE [-o OUTPUT.png] [--title TITLE] [--target 3.28]

Defaults output to ``<log_dir>/curve.png`` so it sits next to the log it
was generated from. Used by ``scripts/promote_record.sh`` to populate
records.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# step:NNNN/TOTAL (train_loss|val_loss):F.FFFF train_time:Nms step_avg:Fms
STEP_RE = re.compile(
    r"^step:(?P<step>\d+)/(?P<total>\d+)\s+"
    r"(?P<kind>train_loss|val_loss):(?P<loss>[\d.]+)\s+"
    r"train_time:(?P<ms>\d+)ms"
)


def parse_log(log_path: Path) -> dict:
    """Return ``{'train': [(step, loss, ms), ...], 'val': [...], 'total_steps': N}``."""
    train: list[tuple[int, float, int]] = []
    val: list[tuple[int, float, int]] = []
    total_steps = 0
    with open(log_path) as f:
        for line in f:
            m = STEP_RE.match(line)
            if not m:
                continue
            step = int(m["step"])
            total_steps = int(m["total"])
            loss = float(m["loss"])
            ms = int(m["ms"])
            (train if m["kind"] == "train_loss" else val).append((step, loss, ms))
    return {"train": train, "val": val, "total_steps": total_steps}


def _rolling_mean(values: list[float], window: int) -> list[float]:
    """Right-aligned simple moving average. window=128 = same cadence as val."""
    out = []
    acc = 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        out.append(acc / min(i + 1, window))
    return out


def plot(data: dict, out_path: Path, title: str, target: float | None,
         smooth: int = 128) -> None:
    train = data["train"]
    val = data["val"]
    if not train and not val:
        sys.exit("plot_run: no step:N/M loss lines parsed")

    fig, (ax_step, ax_time) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    # Pre-compute smoothed train (only meaningful if we have enough samples)
    train_smooth = None
    if train and len(train) >= smooth // 4:
        train_smooth = _rolling_mean([r[1] for r in train], smooth)

    # -- Step axis (left) ----------------------------------------------------
    if train:
        ts = [r[0] for r in train]
        tl = [r[1] for r in train]
        ax_step.plot(ts, tl, color="#4a8dbf", alpha=0.18, linewidth=0.4,
                     label="train (per step, noisy)")
        if train_smooth:
            ax_step.plot(ts, train_smooth, color="#1f4e79", alpha=0.9,
                         linewidth=1.8, label=f"train ({smooth}-step EMA)")
    if val:
        vs = [r[0] for r in val]
        vl = [r[1] for r in val]
        ax_step.plot(vs, vl, "o-", color="#e07b39", linewidth=1.8,
                     markersize=3.5, label="val")
    if target is not None:
        ax_step.axhline(target, color="#c0392b", linestyle="--", alpha=0.55,
                        linewidth=1.0, label=f"target {target}")
    ax_step.set_xlabel("step")
    ax_step.set_ylabel("cross-entropy loss (log scale)")
    ax_step.set_yscale("log")
    ax_step.set_title("loss vs step")
    ax_step.grid(True, alpha=0.25, which="both")
    ax_step.legend(loc="upper right", framealpha=0.9, fontsize=8)

    # -- Wall-clock axis (right) --------------------------------------------
    if train:
        tm = [r[2] / 60_000 for r in train]
        tl = [r[1] for r in train]
        ax_time.plot(tm, tl, color="#4a8dbf", alpha=0.18, linewidth=0.4)
        if train_smooth:
            ax_time.plot(tm, train_smooth, color="#1f4e79", alpha=0.9, linewidth=1.8)
    if val:
        vm = [r[2] / 60_000 for r in val]
        vl = [r[1] for r in val]
        ax_time.plot(vm, vl, "o-", color="#e07b39", linewidth=1.8, markersize=3.5)
    if target is not None:
        ax_time.axhline(target, color="#c0392b", linestyle="--", alpha=0.55, linewidth=1.0)
    ax_time.set_xlabel("wall-clock (min)")
    ax_time.set_yscale("log")
    ax_time.set_title("loss vs wall-clock")
    ax_time.grid(True, alpha=0.25, which="both")

    # -- Annotate final + target gap ----------------------------------------
    if val:
        v_step, v_loss, v_ms = val[-1]
        v_min = v_ms / 60_000
        ax_time.annotate(
            f"final: {v_loss:.4f} @ {v_min:.1f} min",
            xy=(v_min, v_loss), xytext=(8, 12), textcoords="offset points",
            fontsize=9, color="#e07b39",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#e07b39", alpha=0.85),
        )

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    print(f"plot_run: wrote {out_path} ({out_path.stat().st_size // 1024} KiB)", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log", type=Path, help="trainer log file (logs/<uuid>.txt etc.)")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="output PNG path (default: <log_dir>/curve.png)")
    ap.add_argument("--title", default=None,
                    help="plot title (default: log file stem)")
    ap.add_argument("--target", type=float, default=3.28,
                    help="horizontal target line at this val loss (set <=0 to hide)")
    ap.add_argument("--smooth", type=int, default=128,
                    help="rolling-mean window for train_loss (steps); set 0 to disable")
    args = ap.parse_args()

    if not args.log.is_file():
        sys.exit(f"plot_run: log file not found: {args.log}")

    out = args.output or args.log.parent / "curve.png"
    title = args.title or args.log.stem
    target = args.target if args.target > 0 else None
    plot(parse_log(args.log), out, title, target, smooth=max(1, args.smooth))


if __name__ == "__main__":
    main()
