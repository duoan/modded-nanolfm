#!/usr/bin/env bash
# scripts/sync_modal_logs.sh -- pull Modal log volume to ./logs/modal/.
#
# Usage:
#   scripts/sync_modal_logs.sh                # one-shot pull (idempotent)
#   scripts/sync_modal_logs.sh --watch        # poll every 60s until killed
#   scripts/sync_modal_logs.sh --watch 300    # poll every 300s
#
# Notes:
# - Pulls everything from the `nanolfm-logs` Volume (the trainer's per-run
#   `logs/<uuid>.txt` files, the source-code header it dumps at startup,
#   etc.) into `./logs/modal/`. Existing files are overwritten by `modal
#   volume get` (each pull is a fresh snapshot).
# - For stdout/stderr (the live print stream), prefer `modal app logs <id>`.
#   This script is for the on-disk training logs, not the runtime console.
# - In `--watch` mode the loop is safe to leave running over SSH; it just
#   re-pulls periodically so files appear locally as they get committed by
#   the Modal-side periodic commit (every 60 s).

set -euo pipefail
cd "$(dirname "$0")/.."

DEST="logs/modal"
VOLUME="nanolfm-logs"
mkdir -p "$DEST"

pull() {
    local ts
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${ts}] pulling ${VOLUME} -> ${DEST}/"
    if uv run modal volume get --force "${VOLUME}" / "${DEST}/" 2>&1 | tail -3; then
        echo "[${ts}] OK ($(find "${DEST}" -type f 2>/dev/null | wc -l) local files)"
    else
        echo "[${ts}] pull failed -- volume may be empty, or Modal auth needed"
        return 1
    fi
}

if [[ "${1:-}" == "--watch" ]]; then
    INTERVAL=${2:-60}
    echo "watching ${VOLUME} every ${INTERVAL}s (Ctrl-C to stop)"
    while true; do
        pull || true
        sleep "${INTERVAL}"
    done
else
    pull
fi
