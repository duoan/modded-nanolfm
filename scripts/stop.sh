#!/usr/bin/env bash
# scripts/stop.sh -- gracefully terminate a launched training run.
set -euo pipefail

cd "$(dirname "$0")/.."
RUN_NAME=${1:-R00}
PIDFILE="logs/${RUN_NAME}.pid"

if [[ ! -f "$PIDFILE" ]]; then
    echo "${RUN_NAME}: no pid file -- nothing to stop."
    exit 0
fi

PID=$(cat "$PIDFILE")
if ! kill -0 "$PID" 2>/dev/null; then
    echo "${RUN_NAME}: PID $PID not running; removing stale pid file."
    rm -f "$PIDFILE"
    exit 0
fi

echo "${RUN_NAME}: sending SIGTERM to PID $PID..."
kill -TERM "$PID" || true

# torchrun + python need a couple of seconds to wind down.
for _ in $(seq 1 10); do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "${RUN_NAME}: stopped."
        rm -f "$PIDFILE"
        exit 0
    fi
    sleep 1
done

echo "${RUN_NAME}: still alive after 10s, escalating to SIGKILL..."
kill -KILL "$PID" || true
sleep 1
rm -f "$PIDFILE"
echo "${RUN_NAME}: killed."

# Also nuke any orphaned worker python processes that torchrun spawned
pkill -f "train_lfm.py" 2>/dev/null || true
