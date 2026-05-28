#!/usr/bin/env bash
# modded-nanolfm runner. Override $NPROC for multi-GPU (defaults to 1 since
# the target hardware is a single Blackwell node).
set -euo pipefail
NPROC=${NPROC:-1}
exec uv run torchrun --standalone --nproc_per_node="${NPROC}" train_lfm.py
