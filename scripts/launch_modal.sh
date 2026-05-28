#!/usr/bin/env bash
# scripts/launch_modal.sh -- launch training on Modal (cloud H100 / B200).
#
# Usage:
#   scripts/launch_modal.sh [PROFILE] [RUN_NAME]
#
# Examples:
#   scripts/launch_modal.sh h100x8 R00            # 8 x H100, pipeline-baseline 122M
#   TRACK=d350m scripts/launch_modal.sh h100x8 d350m_R00     # ← canonical D-350M launch
#   TRACK=d1_2b scripts/launch_modal.sh b200x8 d1_2b_R00     # 8 x B200, LFM2-1.2B shape
#   TRACK=d350m scripts/launch_modal.sh h100x1 d350m_dev     # 1 x H100 dev iteration
#
# This launches with --detach so the run survives your SSH dying. Monitor with:
#   modal app list
#   modal app logs <app-id>
#   scripts/sync_modal_logs.sh [--watch]
#
# Volume layout after the run:
#   logs/modal/<RUN_NAME>/
#       snapshot/{train_lfm.py, src/, data/cached_fineweb10B.py}
#       meta.txt
#       <uuid>.txt

set -euo pipefail
cd "$(dirname "$0")/.."

PROFILE=${1:-h100x8}
RUN_NAME=${2:-R00}
TRACK=${TRACK:-}

case "$PROFILE" in
    h100x8|b200x8|h100x1|b200x1) ;;
    *)
        echo "ERROR: unknown profile '${PROFILE}'" >&2
        echo "       expected one of: h100x8 b200x8 h100x1 b200x1" >&2
        exit 1
        ;;
esac

FUNC="train_${PROFILE}"

# Friendly nudge if RUN_NAME implies a track but TRACK env wasn't set.
if [[ -z "${TRACK}" && "${RUN_NAME}" =~ ^d(350m|700m|1_2b|2_6b)_ ]]; then
    inferred="${BASH_REMATCH[1]}"
    inferred="d${inferred}"
    echo "WARNING: RUN_NAME=${RUN_NAME} looks like a Track-${inferred} run, but"
    echo "         TRACK env is not set. The trainer will fall back to the 122M"
    echo "         pipeline-baseline config." >&2
    echo "         Did you mean:  TRACK=${inferred} $0 ${PROFILE} ${RUN_NAME}" >&2
    echo
fi

echo "============================================================"
echo "  MODAL LAUNCH: ${FUNC}"
echo "============================================================"
echo "  run_name:     ${RUN_NAME}"
echo "  track:        ${TRACK:-(pipeline-baseline 122M)}"
echo "  data volume:  nanolfm-fineweb10B"
echo "  logs volume:  nanolfm-logs   →  logs/<run_name>/"
echo "  sync logs:    scripts/sync_modal_logs.sh [--watch]"
echo
exec uv run modal run --detach \
    "modal_app.py::${FUNC}" \
    --run-name "${RUN_NAME}" \
    --track "${TRACK}"
