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
#   bash scripts/siesta-gpu-rebuild.sh siesta        # rebuild siesta only
#   bash scripts/siesta-gpu-rebuild.sh elpa          # rebuild elpa + siesta
#   bash scripts/siesta-gpu-rebuild.sh all           # rebuild everything
#   bash scripts/siesta-gpu-rebuild.sh --dry-run all # plan only
#
# Valid components (in dependency order; rebuilding an earlier one
# implicitly rebuilds later ones).  Two-component architecture per
# SIESTA 5.4 INSTALL.md -- ELSI + libfdf + libpsml + xmlf90 +
# libgridxc are SIESTA submodules, compiled on the fly inside SIESTA:
#
#   elpa     ELPA eigensolver (CUDA-enabled; built externally because
#            conda-forge ELPA isn't built with CUDA support)
#   siesta   SIESTA + TranSiesta + TBtrans (clones with
#            --recurse-submodules so External/{libfdf,libpsml,xmlf90,
#            libgridxc,ELSI-project,libxc} populate; SIESTA's cmake
#            compiles them on the fly)
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
    elpa|siesta|all) ;;
    elsi) echo "Note: ELSI is a SIESTA submodule -- rebuild 'siesta' instead." >&2
          COMP=siesta ;;
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
        echo "Will rebuild SIESTA only (ELPA install dir preserved)."
        echo "Note: SIESTA submodules (ELSI, libfdf, libpsml, xmlf90,"
        echo "libgridxc) are rebuilt as part of SIESTA's cmake."
        ;;
    elpa)
        echo "Will rebuild ELPA + SIESTA (everything downstream)."
        ;;
    all)
        echo "Will rebuild ELPA + SIESTA (all install + build dirs wiped;"
        echo "src/ clones preserved to skip re-fetch).  Pass --dry-run to"
        echo "see the plan without touching disk."
        ;;
esac
echo "Python layer will surface a final confirmation with detected toolchain."
echo

exec bash "${SCRIPT_DIR}/install-env.sh" "--rebuild=${COMP}" molbuilder-siesta-gpu
