#!/usr/bin/env bash
#
# scripts/siesta-gpu-rebuild.sh
# ----------------------------
# Wipe sentinels + build dirs for a named component (or all components)
# in an already-installed molbuilder-siesta-gpu env, then re-run the
# build phase.  The conda env itself is left intact.
#
# Why this exists separately from siesta-gpu-bootstrap.sh: the bootstrap
# script does conda create + pip + build, while this one ONLY touches
# the build phase.  Users iterating on a SIESTA patch or testing a
# newer ELPA tag (via MOLBUILDER_ELPA_TAG) want the build-only path so
# the conda env solve doesn't re-run.
#
# Usage:
#   bash scripts/siesta-gpu-rebuild.sh siesta        # rebuild siesta + downstream
#   bash scripts/siesta-gpu-rebuild.sh elpa          # rebuild elpa + elsi + siesta
#   bash scripts/siesta-gpu-rebuild.sh all           # rebuild everything
#   bash scripts/siesta-gpu-rebuild.sh --dry-run all # plan only
#
# Valid components (in dependency order; rebuilding an earlier one
# implicitly rebuilds later ones):
#
#   elpa     ELPA eigensolver (CUDA-enabled)
#   elsi     Electronic Structure Library Interface (links ELPA)
#   siesta   SIESTA + TranSiesta + TBtrans (links ELSI)
#
# Bottom-up note: --rebuild=all also re-clones from upstream, picking
# up any new commits on whichever ref the recipe (or
# MOLBUILDER_*_TAG override) requests.  --rebuild=<component> preserves
# the src/ clones to skip re-fetch on slow networks.
#
# See docs/engines/siesta-gpu.md § 4 for the phase + sentinel model.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<EOF
Usage: bash scripts/siesta-gpu-rebuild.sh [--dry-run] <component>

  <component>: elpa | elsi | siesta | all

EOF
    exit 2
}

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    shift
fi

COMP="${1:-}"
case "$COMP" in
    elpa|elsi|siesta|all) ;;
    "") echo "Error: component required." >&2; usage ;;
    *)  echo "Error: unknown component '$COMP'." >&2; usage ;;
esac

if [[ -n "$DRY_RUN" ]]; then
    exec bash "${SCRIPT_DIR}/install-env.sh" --dry-run molbuilder-siesta-gpu
fi

# Explain destructive intent BEFORE the Python layer's own confirm.
# Specifically, --rebuild=<comp> wipes the install dir + build dir for
# the named component and everything downstream of it.  The Python
# layer will surface another confirm with the exact paths.
case "$COMP" in
    siesta)
        echo "Will rebuild SIESTA only (ELPA + ELSI install dirs preserved)."
        ;;
    elsi)
        echo "Will rebuild ELSI + SIESTA (ELPA install dir preserved)."
        ;;
    elpa)
        echo "Will rebuild ELPA + ELSI + SIESTA (everything downstream)."
        ;;
    all)
        echo "Will rebuild ELPA + ELSI + SIESTA (all install + build dirs"
        echo "wiped; src/ clones preserved to skip re-fetch).  Pass --dry-run"
        echo "to see the plan without touching disk."
        ;;
esac
echo "Python layer will surface a final confirmation with detected toolchain."
echo

exec bash "${SCRIPT_DIR}/install-env.sh" "--rebuild=${COMP}" molbuilder-siesta-gpu
