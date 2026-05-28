#!/usr/bin/env bash
# scripts/promote_record.sh -- copy a synced Modal run dir into records/ for
# permanent (git-tracked) storage.
#
# Usage:
#   scripts/promote_record.sh [DEST] [SOURCE]
#
# Both args are optional. The rules:
#   * SOURCE defaults to the most recently modified subdir of logs/modal/.
#   * DEST is relative to records/. If omitted OR if it's just a track
#     directory (``track_dense``, ``track_moe``), the SOURCE basename is
#     appended automatically -- which means the timestamped Modal run dir
#     name (e.g. ``20260528_0455_dense_R01``) is preserved into records/,
#     guaranteeing uniqueness across parallel jobs.
#   * If DEST is fully qualified (``track_dense/R01_my_thing``), it's used as-is.
#
# Examples:
#   # Zero-arg: take latest log, infer track from RUN_NAME prefix
#   scripts/promote_record.sh
#       -> records/track_dense/20260528_0455_dense_R01/
#
#   # Explicit track only -- preserve the timestamped run name
#   scripts/promote_record.sh track_dense
#       -> records/track_dense/20260528_0455_dense_R01/
#
#   # Fully qualified path -- exact name used
#   scripts/promote_record.sh track_dense/R01_my_thing
#       -> records/track_dense/R01_my_thing/
#
#   # Both args specified
#   scripts/promote_record.sh track_dense/R01_AdamW logs/modal/20260528_0455_dense_R01
#
# Why this exists:
#   `logs/` is gitignored, and Modal Volume storage costs money -- when you
#   delete the volume to save costs, any run that hasn't been promoted to
#   `records/` is gone for good. This script makes promotion a one-liner so
#   it actually happens.
#
# What it does:
#   1. Refuses if records/DEST already exists (no silent overwrite).
#   2. `cp -r SOURCE/. records/DEST/` (preserves snapshot/, meta.txt, log).
#   3. Auto-runs scripts/plot_run.py to produce curve.png next to the log.
#   4. If records/DEST/README.md doesn't exist, scaffolds a stub.
#   5. Prints the `git add && git commit` command to actually persist it.
#
# It deliberately does NOT auto-commit -- you should review the snapshot,
# write a real README, and commit when you're ready.

set -euo pipefail
cd "$(dirname "$0")/.."

DEST="${1:-}"
SOURCE="${2:-}"

# -- Resolve source ----------------------------------------------------------
if [[ -z "${SOURCE}" ]]; then
    SOURCE=$(find logs/modal -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null \
             | sort -nr | head -1 | awk '{print $2}')
    if [[ -z "${SOURCE}" ]]; then
        echo "ERROR: no subdirs in logs/modal/. Run scripts/sync_modal_logs.sh first." >&2
        exit 1
    fi
    echo "[promote] auto-picked SOURCE=${SOURCE} (most-recent subdir of logs/modal/)"
fi

if [[ ! -d "${SOURCE}" ]]; then
    echo "ERROR: SOURCE=${SOURCE} does not exist or is not a directory" >&2
    exit 1
fi

SOURCE_NAME=$(basename "${SOURCE}")

# -- Resolve DEST -----------------------------------------------------------
# Track inference: strip any leading ``YYYYMMDD_HHMM_`` and look for
# dense_/moe_ prefix.
STRIPPED="${SOURCE_NAME#[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9]_}"
case "${STRIPPED}" in
    dense_*|dense)  INFERRED_TRACK="track_dense" ;;
    moe_*|moe)      INFERRED_TRACK="track_moe" ;;
    *)              INFERRED_TRACK="" ;;
esac

if [[ -z "${DEST}" ]]; then
    if [[ -z "${INFERRED_TRACK}" ]]; then
        echo "ERROR: cannot infer track from SOURCE=${SOURCE} (expected name to start" >&2
        echo "       with dense_/moe_ after any timestamp prefix). Pass DEST explicitly:" >&2
        echo "       Usage: scripts/promote_record.sh DEST [SOURCE]" >&2
        exit 1
    fi
    DEST="${INFERRED_TRACK}/${SOURCE_NAME}"
    echo "[promote] auto-derived DEST=${DEST} (inferred track + source name)"
elif [[ "${DEST}" == "track_dense" || "${DEST}" == "track_moe" || "${DEST}" == "track_dense/" || "${DEST}" == "track_moe/" ]]; then
    DEST="${DEST%/}/${SOURCE_NAME}"
    echo "[promote] DEST was just a track dir; promoted as ${DEST}"
fi

DEST_FULL="records/${DEST}"
if [[ -e "${DEST_FULL}" ]]; then
    echo "ERROR: ${DEST_FULL} already exists. Delete it first if you really want to re-promote." >&2
    exit 1
fi

# -- Preflight: warn about git state -----------------------------------------
if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "WARNING: not in a git repo; promotion will copy files but nothing will be tracked." >&2
fi

# -- Copy --------------------------------------------------------------------
mkdir -p "${DEST_FULL}"
cp -r "${SOURCE}/." "${DEST_FULL}/"

# -- Auto-plot the learning curve --------------------------------------------
# Find the trainer log (the trainer writes ``<uuid>.txt`` in the run dir).
LOG_FILE=$(find "${DEST_FULL}" -maxdepth 1 -name "*.txt" -not -name "meta.txt" -type f | head -1)
if [[ -n "${LOG_FILE}" ]]; then
    if uv run scripts/plot_run.py "${LOG_FILE}" \
        --title "$(basename "${DEST}")" 2>&1 | tail -1; then
        echo "[promote] learning curve → ${DEST_FULL}/curve.png"
    else
        echo "[promote] WARN: plot_run.py failed; record/ entry has no curve.png" >&2
    fi
else
    echo "[promote] no <uuid>.txt log in source; skipping plot generation"
fi

# -- Scaffold README.md if missing ------------------------------------------
if [[ ! -f "${DEST_FULL}/README.md" ]]; then
    NAME=$(basename "${DEST}")
    META="${DEST_FULL}/meta.txt"
    if [[ -f "${META}" ]]; then
        RUN_NAME=$(grep -E '^run_name=' "${META}" | head -1 | cut -d= -f2- || echo "${NAME}")
        TRACK=$(grep -E '^(actual_track|track)=' "${META}" | head -1 | cut -d= -f2- || echo "(unknown)")
        GPU=$(grep -E '^gpu=' "${META}" | head -1 | cut -d= -f2- || echo "(unknown)")
        WS=$(grep -E '^world_size=' "${META}" | head -1 | cut -d= -f2- || echo "?")
        FVL=$(grep -E '^final_val_loss=' "${META}" | head -1 | cut -d= -f2- || echo "?")
        TRT=$(grep -E '^total_train_time_min=' "${META}" | head -1 | cut -d= -f2- || echo "?")
    else
        RUN_NAME="${NAME}"; TRACK="(unknown)"; GPU="(unknown)"; WS="?"; FVL="?"; TRT="?"
    fi
    CURVE_LINE=""
    if [[ -f "${DEST_FULL}/curve.png" ]]; then
        CURVE_LINE=$'\n![Learning curve](curve.png)\n\nGenerated by `scripts/plot_run.py`.\n'
    fi
    cat > "${DEST_FULL}/README.md" <<EOF
# ${NAME}

Auto-generated stub by \`scripts/promote_record.sh\`. **Edit me** with the
real story before committing.

## Run

| Field            | Value             |
|------------------|-------------------|
| run_name         | ${RUN_NAME}       |
| track / config   | ${TRACK}          |
| hardware         | ${WS} × ${GPU}    |
| final val loss   | ${FVL}            |
| wall-clock (min) | ${TRT}            |
${CURVE_LINE}
## Promoted from

\`\`\`
${SOURCE}/
\`\`\`

## What's in this record

\`\`\`
$(find "${DEST_FULL}" -mindepth 1 -not -name README.md | sort | sed "s|${DEST_FULL}/||" | head -20)
\`\`\`

See \`meta.txt\` for full run metadata.
EOF
    echo "[promote] scaffolded ${DEST_FULL}/README.md (please edit before committing)"
fi

# -- Summary -----------------------------------------------------------------
echo
echo "============================================================"
echo "  PROMOTED  →  ${DEST_FULL}"
echo "============================================================"
echo "  files: $(find "${DEST_FULL}" -type f | wc -l)"
echo "  size:  $(du -sh "${DEST_FULL}" | cut -f1)"
echo
echo "Next steps:"
echo "  1. Review:           ls -R ${DEST_FULL}"
echo "  2. Edit README:      \$EDITOR ${DEST_FULL}/README.md"
echo "  3. (Optional) wipe local staging:  rm -rf ${SOURCE}"
echo "  4. Commit:           git add ${DEST_FULL} && git commit -m 'Promote ${DEST}'"
echo
echo "Note: SOURCE (${SOURCE}) was COPIED, not moved. Delete it once you're"
echo "confident the records/ entry is correct."
