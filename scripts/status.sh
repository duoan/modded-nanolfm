#!/usr/bin/env bash
# scripts/status.sh -- one-shot status report for a launched training run.
set -euo pipefail

cd "$(dirname "$0")/.."
RUN_NAME=${1:-R00}
PIDFILE="logs/${RUN_NAME}.pid"
STDOUT="logs/${RUN_NAME}.stdout"
META="logs/${RUN_NAME}.meta"

if [[ ! -f "$PIDFILE" ]]; then
    echo "${RUN_NAME}: no pid file at ${PIDFILE} -- nothing launched, or already finished."
    [[ -f "$STDOUT" ]] && { echo; echo "Last 15 lines of ${STDOUT}:"; tail -15 "$STDOUT"; }
    exit 0
fi

PID=$(cat "$PIDFILE")
if kill -0 "$PID" 2>/dev/null; then
    STATE="RUNNING"
    UPTIME=$(ps -o etime= -p "$PID" | tr -d ' ' || echo unknown)
    GPU_LINE=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "?")
else
    STATE="DEAD (PID $PID not running)"
    UPTIME="-"
    GPU_LINE="-"
fi

echo "============================================================"
echo "  ${RUN_NAME}: ${STATE}"
echo "============================================================"
[[ -f "$META" ]] && cat "$META" | sed 's/^/  /'
echo "  pid=${PID}  uptime=${UPTIME}  gpu(mem_MiB,util%)=${GPU_LINE}"
echo

if [[ -f "$STDOUT" ]]; then
    SIZE=$(stat -c%s "$STDOUT")
    echo "  stdout: ${STDOUT}  (${SIZE} bytes)"
    LAST_STEP=$(grep -oE 'step:[0-9]+/[0-9]+' "$STDOUT" | tail -1 || echo none)
    LAST_VAL=$(grep -oE 'val_loss:[0-9]+\.[0-9]+' "$STDOUT" | tail -1 || echo none)
    LAST_TRAIN=$(grep -oE 'train_loss:[0-9]+\.[0-9]+' "$STDOUT" | tail -1 || echo none)
    echo "  last step:        ${LAST_STEP}"
    echo "  last val_loss:    ${LAST_VAL}"
    echo "  last train_loss:  ${LAST_TRAIN}"
    echo
    echo "  tail (last 15 lines):"
    tail -15 "$STDOUT" | sed 's/^/    /'
fi
