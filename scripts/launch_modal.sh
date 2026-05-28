#!/usr/bin/env bash
# scripts/launch_modal.sh -- launch training on Modal (cloud H100 / B200).
#
# Usage:
#   scripts/launch_modal.sh [PROFILE] [RUN_NAME]
#
# Examples:
#   TRACK=dense scripts/launch_modal.sh h100x8 dense_R00     # ← canonical Track Dense
#   TRACK=dense scripts/launch_modal.sh h100x1 dense_dev     # 1 x H100 dev iteration
#   TRACK=moe   scripts/launch_modal.sh h100x8 moe_R00       # Track MoE (once MoE lands in src/model.py)
#                                                            # NOTE: ``TRACK=moe`` currently raises until
#                                                            # the MoE FFN is implemented.
#
# Unset ``TRACK`` is equivalent to ``TRACK=dense`` (both resolve to the
# 122M LFM-hybrid baseline shape).
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
if [[ -z "${TRACK}" && "${RUN_NAME}" =~ ^(dense|moe)_ ]]; then
    inferred="${BASH_REMATCH[1]}"
    echo "WARNING: RUN_NAME=${RUN_NAME} looks like a Track-${inferred} run, but"
    echo "         TRACK env is not set. (Unset TRACK == TRACK=dense; explicit is" >&2
    echo "         better. For Track MoE the env MUST be set.)" >&2
    echo "         Did you mean:  TRACK=${inferred} $0 ${PROFILE} ${RUN_NAME}" >&2
    echo
fi

echo "============================================================"
echo "  MODAL LAUNCH: ${FUNC}"
echo "============================================================"
echo "  run_name:     ${RUN_NAME}"
echo "  track:        ${TRACK:-dense (default; unset TRACK)}"
echo "  data volume:  nanolfm-fineweb10B"
echo "  logs volume:  nanolfm-logs   →  logs/<run_name>/"
echo "  sync logs:    scripts/sync_modal_logs.sh [--watch]"
echo
exec uv run modal run --detach \
    "modal_app.py::${FUNC}" \
    --run-name "${RUN_NAME}" \
    --track "${TRACK}"
