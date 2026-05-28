#!/usr/bin/env bash
# scripts/launch.sh -- fully detached training launcher.
#
# Usage:
#   scripts/launch.sh [RUN_NAME]    # default RUN_NAME=R00
#
# The launched process survives Cursor session closure, SSH disconnect, and
# parent-shell death because we:
#   1. setsid -f          → start a new session, fully detached from any TTY
#   2. nohup              → ignore SIGHUP
#   3. < /dev/null        → no stdin tied to a terminal
#   4. > stdout 2> stderr → all output goes to files on disk
#   5. .venv/bin/torchrun → bypass `uv run` (its TTY assumptions broke things)
#
# Status / stop:
#   scripts/status.sh [RUN_NAME]
#   scripts/stop.sh   [RUN_NAME]

set -euo pipefail

cd "$(dirname "$0")/.."
RUN_NAME=${1:-R00}
mkdir -p logs

PIDFILE="logs/${RUN_NAME}.pid"
STDOUT="logs/${RUN_NAME}.stdout"
STDERR="logs/${RUN_NAME}.stderr"
META="logs/${RUN_NAME}.meta"

# -- Pre-flight: don't double-launch ----------------------------------------
if [[ -f "$PIDFILE" ]]; then
    OLD_PID=$(cat "$PIDFILE" 2>/dev/null || true)
    if [[ -n "${OLD_PID:-}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "ERROR: ${RUN_NAME} already running (PID ${OLD_PID})" >&2
        echo "       run  scripts/stop.sh ${RUN_NAME}  first, or pick another RUN_NAME" >&2
        exit 1
    fi
    rm -f "$PIDFILE"
fi

# -- Pre-flight: env sanity --------------------------------------------------
TORCHRUN=".venv/bin/torchrun"
if [[ ! -x "$TORCHRUN" ]]; then
    echo "ERROR: $TORCHRUN not found. Run \`uv sync\` first." >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not on PATH; no GPU?" >&2
    exit 1
fi
if ! nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi failed; driver/GPU not healthy" >&2
    exit 1
fi

# -- Capture run metadata for later forensics --------------------------------
{
    echo "run_name=${RUN_NAME}"
    echo "started_at=$(date -Iseconds)"
    echo "hostname=$(hostname)"
    echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    if git diff --quiet 2>/dev/null && git diff --quiet --cached 2>/dev/null; then
        echo "git_dirty=no"
    else
        echo "git_dirty=yes"
    fi
} > "$META"

# -- Detach. setsid -f forks; the child is a new session leader.  ------------
# `nohup` is belt-and-suspenders; redirection of all three streams kills the
# last tie to any controlling terminal.
setsid -f nohup "$TORCHRUN" --standalone --nproc_per_node=1 train_lfm.py \
    > "$STDOUT" 2> "$STDERR" < /dev/null

# setsid -f doesn't tell us the child PID directly. The detached torchrun
# spawns a Python worker; we discover both by greping for our STDOUT file
# in the open-file table.
sleep 2
PID=$(ps -eo pid,cmd | awk -v t="train_lfm.py" '
    $0 ~ t && $0 ~ /torchrun/ && $0 !~ /awk/ { print $1; exit }
')
if [[ -z "${PID:-}" ]]; then
    # fall back: try fuser on the stdout file
    PID=$(fuser "$STDOUT" 2>/dev/null | awk '{ print $1 }' | head -1 || true)
fi
if [[ -z "${PID:-}" ]] || ! kill -0 "$PID" 2>/dev/null; then
    echo "ERROR: launch failed -- no torchrun process found." >&2
    echo "Last 60 lines of stderr:" >&2
    tail -60 "$STDERR" >&2 || true
    exit 1
fi
echo "$PID" > "$PIDFILE"

# -- Wait until the trainer prints its own UUID log filename ----------------
# train_lfm.py creates logs/<uuid>.txt as its first action. Find it so we can
# tell the user what to tail.
UUID_LOG=""
for _ in $(seq 1 60); do
    UUID_LOG=$(ls -t logs/*.txt 2>/dev/null | head -1 || true)
    [[ -n "$UUID_LOG" ]] && break
    sleep 1
done

echo
echo "============================================================"
echo "  ${RUN_NAME} LAUNCHED"
echo "============================================================"
echo "  launcher pid:  ${PID}"
echo "  stdout:        ${STDOUT}"
echo "  stderr:        ${STDERR}"
echo "  trainer log:   ${UUID_LOG:-(not yet created)}"
echo "  meta:          ${META}"
echo
echo "  monitor:       tail -f ${STDOUT}"
echo "  status:        scripts/status.sh ${RUN_NAME}"
echo "  stop:          scripts/stop.sh   ${RUN_NAME}"
echo "============================================================"
