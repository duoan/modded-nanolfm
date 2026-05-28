"""modded-nanolfm R00 baseline trainer.

A single-file pretraining loop, in the spirit of ``modded-nanogpt``'s
``train_gpt.py``. Architecture and optimiser live in :mod:`src` so future
records can swap them without touching this file's stepping logic.

Launch (single GPU, simplest path)::

    uv run torchrun --standalone --nproc_per_node=1 train_lfm.py

Multi-GPU (when we have it)::

    uv run torchrun --standalone --nproc_per_node=8 train_lfm.py

The trainer logs every step to stdout and to ``logs/<uuid>.txt``, and copies
its own source as the file header (so each log is a perfect, reproducible
snapshot of the code that produced it).
"""
# ruff: noqa: E402  -- the source-self-read must happen *before* any imports.

from __future__ import annotations

import os
import sys

with open(sys.argv[0]) as _self:
    _self_src = _self.read()  # captured immediately so logs are reproducible

import glob
import subprocess
import time
import uuid
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.model import LFM, LFMConfig
from src.optimizer import build_optimizers, summarize_param_groups

# =============================================================================
# Distributed Data Loader (binary uint16 shards, identical to modded-nanogpt)
# =============================================================================

_HEADER_MAGIC = 20240520
_HEADER_VERSION = 1
_HEADER_INTS = 256


def _peek_data_shard(filename: str) -> int:
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(_HEADER_INTS * 4), dtype=np.int32)
    if header[0] != _HEADER_MAGIC:
        raise RuntimeError(f"bad magic in {filename}: got {header[0]}, expected {_HEADER_MAGIC}")
    if header[1] != _HEADER_VERSION:
        raise RuntimeError(f"unsupported version in {filename}: {header[1]}")
    return int(header[2])


def _load_data_shard(filename: str) -> np.ndarray:
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(_HEADER_INTS * 4), dtype=np.int32)
        assert header[0] == _HEADER_MAGIC and header[1] == _HEADER_VERSION
        ntok = int(header[2])
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, f"length mismatch in {filename}: {len(tokens)} != {ntok}"
    return tokens


class DistributedDataLoader:
    """Round-robins tokens across DDP ranks, advancing through shards on disk."""

    def __init__(self, filename_pattern: str, B: int, T: int, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.B = B
        self.T = T
        self.files = sorted(glob.glob(filename_pattern))
        if not self.files:
            raise FileNotFoundError(
                f"no files matched {filename_pattern!r}. Did you run `python data/cached_fineweb10B.py`?"
            )
        self.ntok_total = sum(_peek_data_shard(f) for f in self.files)
        self._tokens: np.ndarray | None = None
        self._shard_idx = 0
        self._pos = 0
        self.reset()

    def reset(self) -> None:
        self._shard_idx = 0
        self._pos = self.rank * self.B * self.T
        self._tokens = _load_data_shard(self.files[self._shard_idx])

    def _advance_shard(self) -> None:
        self._shard_idx = (self._shard_idx + 1) % len(self.files)
        self._pos = self.rank * self.B * self.T
        self._tokens = _load_data_shard(self.files[self._shard_idx])

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = self.B, self.T
        assert self._tokens is not None
        buf = self._tokens[self._pos : self._pos + B * T + 1]
        # Promote to int32 first (uint16 -> int32 is faster than uint16 -> int64).
        buf_t = torch.from_numpy(buf.astype(np.int32, copy=False)).long()
        x = buf_t[:-1].view(B, T)
        y = buf_t[1:].view(B, T)
        self._pos += B * T * self.world_size
        if self._pos + (B * T * self.world_size + 1) > len(self._tokens):
            self._advance_shard()
        return x.to("cuda", non_blocking=True), y.to("cuda", non_blocking=True)


# =============================================================================
# Hyperparameters (the only knobs this baseline exposes)
# =============================================================================


@dataclass
class Hyperparameters:
    """R00 budget: matches modded-nanogpt's AdamW baseline (5.00 B tokens).

    Token math: 512 seqs × 1024 tok = 524 288 (= 2**19) tokens / step.
    9536 steps × 524 288 ≈ 5.00 B tokens. Mirrors
    ``records/track_1_short/2024-06-06_AdamW`` (final val 3.276).
    """

    # -- data
    input_bin: str = "data/fineweb10B/fineweb_train_*.bin"
    input_val_bin: str = "data/fineweb10B/fineweb_val_*.bin"
    # -- model (R00: 122 M params; matches LFMConfig() defaults)
    vocab_size: int = 50304
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    conv_kernel_size: int = 3
    # -- optimisation (modded-nanogpt AdamW-baseline settings)
    sequence_length: int = 1024
    device_batch_size: int = 32           # per-GPU sequences (fits comfortably in 96 GB)
    batch_size: int = 512                 # global sequences; → grad_accum = 512 / (B * world)
    num_iterations: int = 9536            # 9536 * 2**19 ≈ 5.00 B tokens
    learning_rate: float = 1.8e-3
    weight_decay: float = 0.0             # AdamW baseline used no decay
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-10
    warmup_iters: int = 250
    warmdown_iters: int = 2_000           # trapezoidal: 250 / 7286 / 2000 steps
    grad_clip: float = 1.0
    # -- evaluation / logging
    val_loss_every: int = 128             # matches modded-nanogpt val cadence
    val_tokens: int = 10_485_760          # 10 M tokens for a low-variance val estimate
    save_every: int = 0                   # 0 = only at the end (we generally don't save in R00)
    log_dir: str = "logs"
    # -- runtime knobs
    use_compile: bool = True


def _apply_env_overrides(args: Hyperparameters) -> Hyperparameters:
    """Cheap overrides so we don't fork the trainer for smoke-runs / sweeps.

    SMOKE=1 collapses the budget to ~50 steps so the full pipeline can be
    validated in minutes, not hours.
    """
    if os.environ.get("SMOKE", "0") == "1":
        args.num_iterations = 50
        args.warmup_iters = 5
        args.warmdown_iters = 10
        args.val_loss_every = 25
        args.val_tokens = 524_288           # exactly one global batch
        args.use_compile = False            # save ~30 s of cold compile
    # Manual single-knob overrides (numeric envs) for quick experiments.
    for field_name in (
        "num_iterations", "device_batch_size", "batch_size", "sequence_length",
        "val_loss_every", "val_tokens", "warmup_iters", "warmdown_iters",
    ):
        env_name = field_name.upper()
        if env_name in os.environ:
            setattr(args, field_name, int(os.environ[env_name]))
    for field_name in ("learning_rate", "weight_decay", "grad_clip"):
        env_name = field_name.upper()
        if env_name in os.environ:
            setattr(args, field_name, float(os.environ[env_name]))
    if "USE_COMPILE" in os.environ:
        args.use_compile = os.environ["USE_COMPILE"] not in ("0", "false", "False", "")
    elif os.environ.get("NO_COMPILE", "0") == "1":
        args.use_compile = False
    return args


args = _apply_env_overrides(Hyperparameters())

# =============================================================================
# DDP bootstrap (works for nproc=1 too, when launched via torchrun)
# =============================================================================

assert torch.cuda.is_available(), "this trainer requires CUDA"

_is_torchrun = "RANK" in os.environ and "WORLD_SIZE" in os.environ
if _is_torchrun:
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
else:
    # Allow plain ``python train_lfm.py`` for fast smoke runs.
    rank = local_rank = 0
    world_size = 1

device = f"cuda:{local_rank}"
torch.cuda.set_device(device)
master = (rank == 0)
if master:
    print(f"[ddp] rank={rank} world_size={world_size} device={device}")
    is_smoke = os.environ.get("SMOKE", "0") == "1"
    print(
        f"[run] mode={'SMOKE' if is_smoke else 'FULL'}  "
        f"iters={args.num_iterations}  lr={args.learning_rate}  wd={args.weight_decay}  "
        f"warmup/down={args.warmup_iters}/{args.warmdown_iters}  compile={args.use_compile}"
    )

# Cheap, deterministic-enough seeds (the data loader is the source of variance).
torch.manual_seed(0x1F1F + rank)


# =============================================================================
# Batch-shape sanity
# =============================================================================

B, T = args.device_batch_size, args.sequence_length
assert args.val_tokens % (B * T * world_size) == 0, "val_tokens must divide B*T*world_size cleanly"
val_steps = args.val_tokens // (B * T * world_size)
assert args.batch_size % (B * world_size) == 0, "global batch_size must divide B*world_size cleanly"
grad_accum_steps = args.batch_size // (B * world_size)
if master:
    print(
        f"[batch] device_B={B} T={T} world={world_size} grad_accum={grad_accum_steps} "
        f"global_tokens_per_step={B * T * world_size * grad_accum_steps:,}"
    )


# =============================================================================
# Data
# =============================================================================

train_loader = DistributedDataLoader(args.input_bin, B, T, rank, world_size)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, rank, world_size)
if master:
    print(
        f"[data] train {train_loader.ntok_total / 1e9:.2f}B tokens across {len(train_loader.files)} shards"
    )
    print(
        f"[data] val   {val_loader.ntok_total / 1e6:.1f}M tokens across {len(val_loader.files)} shards"
    )


# =============================================================================
# Model
# =============================================================================

cfg = LFMConfig(
    vocab_size=args.vocab_size,
    hidden_size=args.hidden_size,
    intermediate_size=args.intermediate_size,
    num_hidden_layers=args.num_hidden_layers,
    num_attention_heads=args.num_attention_heads,
    num_key_value_heads=args.num_key_value_heads,
    conv_kernel_size=args.conv_kernel_size,
)
model = LFM(cfg).to(device=device, dtype=torch.bfloat16)
# Keep RMSNorm weights in fp32 to satisfy the spec's stability guardrail.
for m in model.modules():
    if m.__class__.__name__ == "RMSNorm":
        m.weight.data = m.weight.data.float()

if master:
    print(f"[model] {model.num_parameters() / 1e6:.2f}M parameters  cfg={cfg}")
    print(summarize_param_groups(model, weight_decay=args.weight_decay))

if args.use_compile:
    model = torch.compile(model)

if world_size > 1:
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
    raw_model = model.module
else:
    raw_model = model

ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)


# =============================================================================
# Optimiser + LR schedule
# =============================================================================

optimizers, _opt_param_lists = build_optimizers(
    raw_model,
    lr=args.learning_rate,
    betas=args.betas,
    eps=args.eps,
    weight_decay=args.weight_decay,
)


def get_lr(it: int) -> float:
    """Trapezoidal schedule: linear warmup, plateau, linear warmdown.

    Same shape as modded-nanogpt's R00 schedule (so loss curves are
    apples-to-apples comparable across the two repos).
    """
    assert 0 <= it <= args.num_iterations
    if it < args.warmup_iters:
        return (it + 1) / max(1, args.warmup_iters)
    plateau_end = args.num_iterations - args.warmdown_iters
    if it < plateau_end:
        return 1.0
    return max(0.0, (args.num_iterations - it) / max(1, args.warmdown_iters))


schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]


# =============================================================================
# Logging
# =============================================================================

if master:
    os.makedirs(args.log_dir, exist_ok=True)
    run_id = str(uuid.uuid4())
    logfile = os.path.join(args.log_dir, f"{run_id}.txt")
    with open(logfile, "w") as f:
        f.write("=" * 100 + "\n")
        f.write(_self_src)
        f.write("=" * 100 + "\n")
        f.write(f"torch {torch.__version__}  cuda {torch.version.cuda}\n")
        try:
            smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False).stdout
            f.write(smi + "\n")
        except FileNotFoundError:
            pass
        f.write("=" * 100 + "\n")

    def log(line: str) -> None:
        print(line, flush=True)
        with open(logfile, "a") as fh:
            fh.write(line + "\n")
else:
    def log(line: str) -> None:
        pass


# =============================================================================
# Training loop
# =============================================================================

training_time_ms = 0.0
torch.cuda.synchronize()
t0 = time.time()
train_loader.reset()

x, y = train_loader.next_batch()

for step in range(args.num_iterations + 1):
    last_step = step == args.num_iterations
    # untimed warmup: torch.compile + first allocator pass eats the first ~10 steps
    if step == 10:
        training_time_ms = 0.0
        torch.cuda.synchronize()
        t0 = time.time()
    timed_steps = float("nan") if step <= 11 else (step - 10) + 1

    # ----- validation -------------------------------------------------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        model.eval()
        val_loader.reset()
        val_loss = torch.zeros((), device=device)
        with torch.no_grad():
            for _ in range(val_steps):
                xv, yv = val_loader.next_batch()
                with ctx:
                    _, loss = model(xv, yv, return_logits=False)
                val_loss += loss.detach()
        if world_size > 1:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        log(
            f"step:{step}/{args.num_iterations} val_loss:{val_loss.item():.4f} "
            f"train_time:{training_time_ms:.0f}ms "
            f"step_avg:{training_time_ms / max(1, timed_steps - 1):.2f}ms"
        )
        torch.cuda.synchronize()
        t0 = time.time()

    if last_step:
        break

    # ----- train step -------------------------------------------------------
    model.train()
    for ms_i in range(1, grad_accum_steps + 1):
        with ctx:
            _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()
        x, y = train_loader.next_batch()  # overlap with backward
        is_last_accum = ms_i == grad_accum_steps
        if world_size > 1 and not is_last_accum:
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()

    if grad_accum_steps > 1:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.div_(grad_accum_steps)

    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)

    for opt, sched in zip(optimizers, schedulers, strict=True):
        opt.step()
        sched.step()
    model.zero_grad(set_to_none=True)

    if master:
        approx_time_ms = training_time_ms + 1000 * (time.time() - t0)
        log(
            f"step:{step + 1}/{args.num_iterations} train_loss:{train_loss.item():.4f} "
            f"train_time:{approx_time_ms:.0f}ms step_avg:{approx_time_ms / timed_steps:.2f}ms"
        )

if master:
    log(f"peak memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

if _is_torchrun:
    dist.destroy_process_group()
