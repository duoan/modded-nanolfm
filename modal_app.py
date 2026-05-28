"""Modal app for running modded-nanolfm on cloud H100s / B200s.

Why this exists
---------------
The modded-nanogpt speedrun is timed on **8 × H100** on PrimeIntellect. For our
records to be apples-to-apples comparable, we need to run the same
``train_lfm.py`` on identical hardware. This Modal app provides that path, and
also exposes B200 configs for the spec's "8 × B200 nodes" target.

Setup (once per machine)
------------------------
::

    uv sync
    uv run modal setup       # one-time auth
    # (optional) attach a HuggingFace token for higher download rate limits:
    uv run modal secret create huggingface HF_TOKEN=hf_xxx

Workflow
--------
::

    # 1. Download FineWeb shards into the Modal Volume (once; persists).
    uv run modal run modal_app.py::download_data --num-chunks 50

    # 2. Pick a GPU profile and launch training.
    #    --detach makes the run survive your laptop dying / SSH dropping.
    uv run modal run --detach modal_app.py::train_h100x8 --run-name R00
    uv run modal run --detach modal_app.py::train_b200x8 --run-name R00_b200
    uv run modal run --detach modal_app.py::train_h100x1 --run-name R00_h100x1
    uv run modal run --detach modal_app.py::train_b200x1 --run-name R00_b200x1

    # 3. Monitor / pull back logs.
    uv run modal app list
    uv run modal app logs <app-id>       # live stdout/stderr stream
    scripts/sync_modal_logs.sh           # one-shot pull of nanolfm-logs volume
    scripts/sync_modal_logs.sh --watch   # poll every 60s (good for long runs)

Env overrides recognised by ``train_lfm.py`` (NUM_ITERATIONS, LEARNING_RATE,
USE_COMPILE, SMOKE, etc.) are passed through Modal's process env -- set them
in the calling shell, or pass them on the ``modal run`` CLI via
``--env KEY=VALUE``.

Profile cheat-sheet
-------------------
- ``train_h100x8``: official modded-nanogpt speedrun timing reference.
- ``train_b200x8``: spec's flagship config; biggest single-node throughput.
- ``train_h100x1`` / ``train_b200x1``: cheap single-GPU iteration on cloud
  hardware (mirrors our local Blackwell setup but with someone else's GPU).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime

import modal

# ---------------------------------------------------------------------------
# App + Image
# ---------------------------------------------------------------------------
APP_NAME = "modded-nanolfm"
PROJECT_DIR = "/root/modded-nanolfm"

app = modal.App(APP_NAME)

# Pin torch to roughly what we develop against locally. Keep the version
# expression loose so Modal can resolve a wheel for the chosen CUDA.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch>=2.10",
        "numpy>=2.0",
        "huggingface-hub>=1.0",
        "tqdm>=4.66",
    )
    # Mount our source tree. Any change here invalidates the image layer.
    # NOTE: do *not* `add_local_dir("data", ...)` -- the local `data/fineweb10B/`
    # holds multi-GB shards we don't want baked into the image.
    .add_local_dir("src", f"{PROJECT_DIR}/src")
    .add_local_file("data/cached_fineweb10B.py", f"{PROJECT_DIR}/data/cached_fineweb10B.py")
    .add_local_file("train_lfm.py", f"{PROJECT_DIR}/train_lfm.py")
    .add_local_file("pyproject.toml", f"{PROJECT_DIR}/pyproject.toml")
)


# ---------------------------------------------------------------------------
# Volumes: persist data shards + training logs across runs.
# ---------------------------------------------------------------------------
data_volume = modal.Volume.from_name("nanolfm-fineweb10B", create_if_missing=True)
logs_volume = modal.Volume.from_name("nanolfm-logs", create_if_missing=True)

VOLUMES = {
    f"{PROJECT_DIR}/data/fineweb10B": data_volume,
    f"{PROJECT_DIR}/logs": logs_volume,
}

# If you hit HuggingFace rate limits during downloads, create a Modal Secret
# with your token and append it to the relevant functions::
#
#     modal secret create huggingface HF_TOKEN=hf_xxx
#     # then add  secrets=[modal.Secret.from_name("huggingface")]  to download_data
#
# We don't reference it by default so the app loads cleanly without the secret.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _snapshot_code(snapshot_dir: str) -> None:
    """Freeze trainer + helpers into ``snapshot_dir``.

    After the run, this snapshot lives alongside the log file inside the run
    dir, so ``logs/<run_name>/`` is record-ready -- promoting an accepted
    record is just ``mv logs/modal/<run_name> records/track_<X>/R<NN>_<name>/``.
    """
    os.makedirs(snapshot_dir, exist_ok=True)
    shutil.copy("train_lfm.py", os.path.join(snapshot_dir, "train_lfm.py"))
    src_dst = os.path.join(snapshot_dir, "src")
    if os.path.exists(src_dst):
        shutil.rmtree(src_dst)
    shutil.copytree("src", src_dst)
    data_dst = os.path.join(snapshot_dir, "data")
    os.makedirs(data_dst, exist_ok=True)
    shutil.copy("data/cached_fineweb10B.py", os.path.join(data_dst, "cached_fineweb10B.py"))


def _write_meta(meta_path: str, **fields: object) -> None:
    """Append ``key=value`` lines to ``meta_path`` (one per kwarg)."""
    with open(meta_path, "a") as f:
        for k, v in fields.items():
            f.write(f"{k}={v}\n")


def _run_torchrun(world_size: int, run_name: str, track: str = "") -> None:
    """Spawn torchrun as a subprocess so PyTorch's re-entrant entrypoint
    doesn't collide with Modal's container entrypoint.

    ``train_lfm.py`` already handles ``WORLD_SIZE>1`` via DDP. The trainer
    also honours ``RUN_NAME`` (puts its log file under ``logs/<run_name>/``)
    and ``TRACK`` (selects the canonical ``LFMConfig`` for that track).

    On the volume each run produces this layout, ready to be moved into
    ``records/track_<X>/R<NN>_<name>/`` if the run is accepted::

        logs/<run_name>/
            snapshot/
                train_lfm.py
                src/
                data/cached_fineweb10B.py
            meta.txt           # run_name, track, gpu, start/end, exit code
            <uuid>.txt         # trainer log (own source as header + per-step lines)

    A periodic log-volume commit runs in the background so the log file is
    pullable even mid-training (and survives OOM / timeout / SIGKILL).
    """
    import threading

    os.chdir(PROJECT_DIR)

    # -- Set up the per-run dir on the volume -----------------------------
    run_dir = os.path.join(PROJECT_DIR, "logs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    snapshot_dir = os.path.join(run_dir, "snapshot")
    meta_path = os.path.join(run_dir, "meta.txt")

    _snapshot_code(snapshot_dir)

    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    gpu_name = "(nvidia-smi unavailable)"
    try:
        gpu_name = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=False,
        ).stdout.strip().split("\n")[0] or gpu_name
    except FileNotFoundError:
        pass

    # Truncate any old meta from a previous run with the same name.
    open(meta_path, "w").close()
    _write_meta(
        meta_path,
        run_name=run_name,
        track=track or "(default-pipeline-baseline-122M)",
        world_size=world_size,
        gpu=gpu_name,
        started_at=started_at,
        modal_app_id=os.environ.get("MODAL_TASK_ID", "unknown"),
    )

    # -- Env passthrough to the trainer ------------------------------------
    env = os.environ.copy()
    env["RUN_NAME"] = run_name
    if track:
        env["TRACK"] = track

    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={world_size}",
        "train_lfm.py",
    ]
    print(f"[modal] run_dir={run_dir}", flush=True)
    print(f"[modal] $ {' '.join(cmd)}  (run_name={run_name}, track={track or '(none)'}, world_size={world_size})", flush=True)

    # -- Periodic log volume commits ---------------------------------------
    # Without this, all log writes stay container-local and a hard kill would
    # lose them. Commit every 60 s so a mid-run `sync_modal_logs.sh` is useful.
    stop = threading.Event()

    def _commit_loop() -> None:
        while not stop.wait(60.0):
            try:
                logs_volume.commit()
            except Exception as e:  # noqa: BLE001 -- best-effort, never fatal
                print(f"[modal] periodic logs_volume.commit failed: {e}", flush=True)

    commit_thread = threading.Thread(target=_commit_loop, name="logs-commit", daemon=True)
    commit_thread.start()

    # First commit so the snapshot + meta are pullable before training is done.
    logs_volume.commit()

    # -- Run training, with guaranteed completion record on ANY exit -------
    exit_code: int | str = "unknown"
    try:
        result = subprocess.run(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr, check=False)
        exit_code = result.returncode
        if exit_code != 0:
            raise RuntimeError(f"torchrun exited {exit_code}")
        print("[modal] training completed cleanly", flush=True)
    finally:
        stop.set()
        commit_thread.join(timeout=5)
        _write_meta(
            meta_path,
            completed_at=datetime.now(UTC).isoformat(timespec="seconds"),
            exit_code=exit_code,
        )
        try:
            logs_volume.commit()
            print("[modal] final logs_volume.commit OK", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[modal] final logs_volume.commit FAILED: {e}", flush=True)


def _hours(h: float) -> int:
    return int(h * 3600)


# ---------------------------------------------------------------------------
# Data download (one-off; writes to the data volume)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    timeout=_hours(2),
    volumes=VOLUMES,
    cpu=4,
    memory=8 * 1024,
)
def download_data(num_chunks: int = 50) -> None:
    """Download ``num_chunks`` train shards + 1 val shard to the data volume.

    Default 50 chunks → 5 B tokens → enough for one R00 epoch without cycling.
    """
    os.chdir(PROJECT_DIR)
    print(f"[modal] downloading {num_chunks} shards to {PROJECT_DIR}/data/fineweb10B/", flush=True)
    subprocess.run(
        [sys.executable, "data/cached_fineweb10B.py", str(num_chunks)],
        check=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    data_volume.commit()
    print("[modal] data volume committed", flush=True)


# ---------------------------------------------------------------------------
# Training entry points (one per GPU profile)
# ---------------------------------------------------------------------------
# 6h timeout fits the projected R00 wall-clock (~4.7h on 1× Blackwell) with
# slack for compile + val. Bump if you crank NUM_ITERATIONS via env.
_DEFAULT_TIMEOUT = _hours(6)


@app.function(
    image=image,
    gpu="H100:8",
    timeout=_DEFAULT_TIMEOUT,
    volumes=VOLUMES,
)
def train_h100x8(run_name: str = "R00", track: str = "") -> None:
    """8 × H100. The official modded-nanogpt timing rig."""
    _run_torchrun(world_size=8, run_name=run_name, track=track)


@app.function(
    image=image,
    gpu="B200:8",
    timeout=_DEFAULT_TIMEOUT,
    volumes=VOLUMES,
)
def train_b200x8(run_name: str = "R00", track: str = "") -> None:
    """8 × B200. Spec's flagship single-node config."""
    _run_torchrun(world_size=8, run_name=run_name, track=track)


@app.function(
    image=image,
    gpu="H100:1",
    timeout=_DEFAULT_TIMEOUT,
    volumes=VOLUMES,
)
def train_h100x1(run_name: str = "R00", track: str = "") -> None:
    """1 × H100 for cheap single-GPU iteration on cloud hardware."""
    _run_torchrun(world_size=1, run_name=run_name, track=track)


@app.function(
    image=image,
    gpu="B200:1",
    timeout=_DEFAULT_TIMEOUT,
    volumes=VOLUMES,
)
def train_b200x1(run_name: str = "R00", track: str = "") -> None:
    """1 × B200 -- closest cloud analogue to our local RTX PRO 6000 Blackwell."""
    _run_torchrun(world_size=1, run_name=run_name, track=track)


# ---------------------------------------------------------------------------
# Local entrypoint (so ``modal run modal_app.py`` without ::func works too)
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(action: str = "status", num_chunks: int = 50, run_name: str = "R00") -> None:
    """Convenience dispatcher: ``modal run modal_app.py -- --action download``.

    For the actual training launches, call the per-GPU functions directly so
    Modal picks up the right ``gpu=`` argument::

        modal run --detach modal_app.py::train_h100x8 --run-name R00
    """
    if action == "download":
        download_data.remote(num_chunks=num_chunks)
    elif action == "status":
        print("Modal training functions:")
        print("  train_h100x8 / train_b200x8 / train_h100x1 / train_b200x1")
        print()
        print("Recommended launch:")
        print(f"  modal run --detach modal_app.py::train_h100x8 --run-name {run_name}")
        print()
        print("Data download:")
        print(f"  modal run modal_app.py::download_data --num-chunks {num_chunks}")
    else:
        raise ValueError(f"unknown action: {action!r} (expected 'download' or 'status')")
