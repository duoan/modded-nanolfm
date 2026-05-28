#!/usr/bin/env bash
# scripts/launch_modal.sh -- launch training on Modal (cloud H100 / B200).
#
# Usage:
#   scripts/launch_modal.sh [PROFILE] [TAG]
#
# The on-volume directory name is auto-prefixed with the launch time at
# minute resolution -- e.g. ``20260528_0455_dense_R01`` -- so parallel
# launches (and re-launches with the same TAG) never collide on the
# Modal volume. If TAG already starts with a ``YYYYMMDD_HHMM_`` prefix
# (e.g. you're re-running a script that built one), the prefix is kept
# as-is rather than double-prefixed.
#
# Examples:
#   TRACK=dense scripts/launch_modal.sh h100x8 dense_R01
#     ->  RUN_NAME = 20260528_0455_dense_R01
#
#   TRACK=dense scripts/launch_modal.sh h100x8 dense_R01     # again, 2 min later
#     ->  RUN_NAME = 20260528_0457_dense_R01     (different dir, no collision)
#
#   TRACK=dense scripts/launch_modal.sh h100x1 dense_dev     # 1 x H100 dev iteration
#   TRACK=moe   scripts/launch_modal.sh h100x8 moe_R00       # Track MoE
#                                                            #   (raises until the MoE FFN
#                                                            #   lands in src/model.py)
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
TAG=${2:-R00}
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

# Auto-prepend a minute-resolution UTC timestamp so two simultaneous
# launches with the same TAG don't collide on logs/<RUN_NAME>/. If TAG
# already has such a prefix, leave it alone (idempotent for retries that
# pass the previous full RUN_NAME back in).
if [[ "${TAG}" =~ ^[0-9]{8}_[0-9]{4}_ ]]; then
    RUN_NAME="${TAG}"
else
    RUN_NAME="$(date -u +%Y%m%d_%H%M)_${TAG}"
fi

# Friendly nudge if TAG implies a track but TRACK env wasn't set.
if [[ -z "${TRACK}" && "${TAG}" =~ ^(dense|moe)_ ]]; then
    inferred="${BASH_REMATCH[1]}"
    echo "WARNING: TAG=${TAG} looks like a Track-${inferred} run, but"
    echo "         TRACK env is not set. (Unset TRACK == TRACK=dense; explicit is" >&2
    echo "         better. For Track MoE the env MUST be set.)" >&2
    echo "         Did you mean:  TRACK=${inferred} $0 ${PROFILE} ${TAG}" >&2
    echo
fi

echo "============================================================"
echo "  MODAL LAUNCH: ${FUNC}"
echo "============================================================"
echo "  run_name:     ${RUN_NAME}    (volume dir; auto-timestamped)"
echo "  tag:          ${TAG}"
echo "  track:        ${TRACK:-dense (default; unset TRACK)}"
echo "  data volume:  nanolfm-fineweb10B"
echo "  logs volume:  nanolfm-logs   →  logs/${RUN_NAME}/"
echo "  sync logs:    scripts/sync_modal_logs.sh [--watch]"
echo
exec uv run modal run --detach \
    "modal_app.py::${FUNC}" \
    --run-name "${RUN_NAME}" \
    --track "${TRACK}"
