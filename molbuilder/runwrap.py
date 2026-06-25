"""Shell-wrapper emission for ``molbuilder run``.

Each generated script (``.fdf`` or ``.py``) gets a sibling
``<basename>.run.sh`` that activates the right conda env and executes
the tool.  The user runs the ``.sh`` manually (foreground / background
/ cluster scheduler -- their call); molbuilder does **not** manage
processes.

The wrapper is intentionally small and human-readable:

* A user can read it to understand what command they're about to run.
* They can edit it to add custom flags (MPI options, env vars, ulimit).
* They can copy chunks into SLURM / PBS / GNU parallel scripts.

The wrapper is regenerated freshly each time ``molbuilder run`` runs
(it's per-invocation output, not state); edits between regenerations
are lost.

Testing hook: tests inject a synthetic Capabilities via
:func:`molbuilder.diagnostics.set_capabilities`.  Production call
sites pass only the script path + optional ``env`` / ``mpi_np``
overrides.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .diagnostics import EXTENSION_TO_CATEGORY, get_capabilities


def _build_locate_and_activate_block(target_env: str,
                                       autodetect: bool,
                                       preactivate_populated: bool = False) -> str:
    """Return the ``Locate conda + activate`` portion of the wrapper.

    Two distinct shapes, gated by ``script_generation.autodetect_conda``
    in the effective config (docs/config.md § 3.6):

      * autodetect=False (the DEFAULT): trust that ``preactivate`` and
        ``MOLBUILDER_PREACTIVATE_CMDS`` arranged the env -- emit only
        path 1 (already-in-env skip) + a direct ``conda activate``.
        Failure prints the actionable hint + the available envs +
        exits 1.  No silent fallback discovery, no source step the
        wrapper has to guess at.  Fail loud.

      * autodetect=True: emit the full 6-path discovery + the source
        step + ``conda activate`` -- the permissive shape suitable
        for laptop dev, debugging an unknown machine, or sites where
        the operator doesn't want to mandate ``preactivate`` setup.

    Both branches log every step to the per-run log file so a
    failure on the cluster yields a self-contained diagnostic.
    """
    head = (
        f"# --- Locate conda + activate ({target_env}) ------------\n"
        f'_target_env="{target_env}"\n'
        f"# Path 1: already in the right env -- nothing to do.\n"
        f'if [ "${{CONDA_DEFAULT_ENV:-}}" = "$_target_env" ]; then\n'
        f"    _log STAGE \"path 1: already in '$_target_env'; "
        f"skipping activation\"\n"
        f"else\n"
    )
    activate_call = (
        f'    _log STAGE "conda activate $_target_env"\n'
        f"    if ! conda activate \"$_target_env\" 2>&1; then\n"
        f"        _log ERROR \"conda activate failed\"\n"
        f"        cat >&2 <<EOM\n"
        f"\n"
        f"ERROR: ``conda activate $_target_env`` failed.  Three common causes:\n"
        f"  * conda / mamba not on PATH      -> put ``module load mamba`` (or\n"
        f"                                     the equivalent for your site)\n"
        f"                                     in script_generation.preactivate\n"
        f"                                     (docs/config.md § 3.6)\n"
        f"  * ``conda activate`` undefined   -> the shell function is loaded\n"
        f"                                     by sourcing conda.sh.  Either\n"
        f"                                     run from an interactive shell\n"
        f"                                     where ``conda init`` has fired,\n"
        f"                                     OR add to preactivate:\n"
        f"                                       source \"$(mamba info --base)/etc/profile.d/conda.sh\"\n"
        f"                                     OR set\n"
        f"                                       script_generation.autodetect_conda: true\n"
        f"                                     for the permissive 6-path discovery mode.\n"
        f"  * env ``$_target_env`` missing   -> install it:\n"
        f"                                       bash scripts/install-env.sh install $_target_env\n"
        f"\n"
        f"Available envs visible to this shell:\n"
        f"EOM\n"
        f"        conda env list 2>/dev/null | tail -n +3 \\\n"
        f"            | awk '{{print \"  \" $1}}' >&2 \\\n"
        f"            || echo \"  (conda env list unavailable)\" >&2\n"
        f"        exit 1\n"
        f"    fi\n"
        f"fi\n"
    )
    if not autodetect:
        # Fail-loud mode (the default).  ``preactivate`` is expected
        # to have put conda on PATH and made ``conda activate`` a
        # valid command (modern ``module load mamba`` does both via
        # the conda-init shell hooks the module ships).  If it didn't,
        # the bare ``conda activate`` call will fail with a clear
        # error -- which is the documented intent of autodetect=false.
        if preactivate_populated:
            stage_msg = (
                "autodetect_conda=false; trusting baked-in preactivate "
                "to have put conda on PATH"
            )
        else:
            stage_msg = (
                "autodetect_conda=false + preactivate empty; relying on "
                "inherited shell state to have conda on PATH"
            )
        body = (
            f"    # script_generation.autodetect_conda is FALSE.  The\n"
            f"    # wrapper trusts that the preactivate block above (or\n"
            f"    # MOLBUILDER_PREACTIVATE_CMDS) arranged the env.  No\n"
            f"    # 6-path fallback emitted -- if the env isn't reachable\n"
            f"    # now, the ``conda activate`` call below fails loud.\n"
            f'    _log STAGE "{stage_msg}"\n'
        )
        return head + body + activate_call
    # Autodetect=True: 6-path discovery + source step + activate.
    body = (
        f'    _baked_conda_base=\'{_BAKED_PLACEHOLDER}\'\n'
        f"    _conda_base=\"\"   # filled in by whichever path succeeds\n"
        f"    # Path 2: baked-in generator-time conda base.\n"
        f'    if [ -n "$_baked_conda_base" ] \\\n'
        f'         && [ -f "$_baked_conda_base/etc/profile.d/'
        f'conda.sh" ]; then\n'
        f'        _conda_base="$_baked_conda_base"\n'
        f'        _log STAGE "path 2: using baked-in conda base"\n'
        f'        _log INFO "_conda_base=$_conda_base"\n'
        f"    fi\n"
        f"    # Path 3: mamba on PATH -- preferred over conda.\n"
        f'    if [ -z "$_conda_base" ] && '
        f'command -v mamba >/dev/null 2>&1; then\n'
        f'        _cb="$(mamba info --base 2>/dev/null)" || true\n'
        f'        if [ -n "$_cb" ] && '
        f'[ -f "$_cb/etc/profile.d/conda.sh" ]; then\n'
        f'            _conda_base="$_cb"\n'
        f'            _log STAGE "path 3: ``mamba info --base`` '
        f'-> $_conda_base"\n'
        f"        fi\n"
        f"    fi\n"
        f"    # Path 4: conda on PATH -- fallback.\n"
        f'    if [ -z "$_conda_base" ] && '
        f'command -v conda >/dev/null 2>&1; then\n'
        f'        _cb="$(conda info --base 2>/dev/null)" || true\n'
        f'        if [ -n "$_cb" ] && '
        f'[ -f "$_cb/etc/profile.d/conda.sh" ]; then\n'
        f'            _conda_base="$_cb"\n'
        f'            _log STAGE "path 4: ``conda info --base`` '
        f'-> $_conda_base"\n'
        f"        fi\n"
        f"    fi\n"
        f"    # Path 5: ``module load`` then re-probe.\n"
        f'    if [ -z "$_conda_base" ] && '
        f'command -v module >/dev/null 2>&1; then\n'
        f"        for _mod in mamba miniforge3 miniforge mambaforge"
        f" \\\n"
        f"                    miniconda3 miniconda anaconda3 "
        f"anaconda; do\n"
        f'            _log INFO "trying ``module load $_mod``"\n'
        f'            if module load "$_mod" 2>/dev/null; then\n'
        f"                for _bin in mamba conda; do\n"
        f"                    command -v $_bin >/dev/null 2>&1 "
        f"|| continue\n"
        f'                    _cb="$($_bin info --base 2>/dev/null)"'
        f' || true\n'
        f'                    if [ -n "$_cb" ] \\\n'
        f'                         && [ -f "$_cb/etc/profile.d/'
        f'conda.sh" ]; then\n'
        f'                        _conda_base="$_cb"\n'
        f'                        _log STAGE "path 5: module '
        f'load $_mod -> $_conda_base"\n'
        f"                        break 2\n"
        f"                    fi\n"
        f"                done\n"
        f"            fi\n"
        f"        done\n"
        f"    fi\n"
        f"    # Path 6: common-location filesystem probe.\n"
        f'    if [ -z "$_conda_base" ]; then\n'
        f"        for _cb in \"$HOME/miniforge3\" \"$HOME/mambaforge\""
        f" \\\n"
        f"                   \"$HOME/miniconda3\" \"$HOME/anaconda3\""
        f" \\\n"
        f"                   /opt/miniforge3 /opt/mambaforge \\\n"
        f"                   /opt/miniconda3 /opt/anaconda3; do\n"
        f'            if [ -f "$_cb/etc/profile.d/conda.sh" ]; then\n'
        f'                _conda_base="$_cb"\n'
        f'                _log STAGE "path 6: filesystem '
        f'probe -> $_conda_base"\n'
        f"                break\n"
        f"            fi\n"
        f"        done\n"
        f"    fi\n"
        f"    # All six paths failed: error with actionable advice.\n"
        f'    if [ -z "$_conda_base" ]; then\n'
        f'        _log ERROR "could not locate conda/mamba on '
        f'this system"\n'
        f"        cat >&2 <<EOM\n"
        f"\n"
        f"ERROR: this wrapper needs the '$_target_env' conda env, but no\n"
        f"conda / mamba installation was found.  Tried (in order):\n"
        f"  (1) CONDA_DEFAULT_ENV already == '$_target_env'\n"
        f"  (2) baked-in generator-time conda base: "
        f"${{_baked_conda_base:-<unset>}}\n"
        f"  (3) ``mamba info --base`` (mamba on PATH -- preferred)\n"
        f"  (4) ``conda info --base`` (conda on PATH -- fallback)\n"
        f"  (5) ``module load {{mamba,miniforge,...}}`` then re-probe\n"
        f"  (6) common locations: \\$HOME/miniforge3 etc.\n"
        f"\n"
        f"Fix one of:\n"
        f"  * install conda / mamba (docs/README_install.md)\n"
        f"  * populate script_generation.preactivate in molbuilder.json\n"
        f"    (docs/config.md § 3.6)\n"
        f"  * set MOLBUILDER_PREACTIVATE_CMDS in your shell config\n"
        f"\n"
        f"See $_runwrap_log for the full per-path trace.\n"
        f"EOM\n"
        f"        exit 1\n"
        f"    fi\n"
        f"    # Source + activate.\n"
        f'    _log STAGE "sourcing $_conda_base/etc/profile.d/'
        f'conda.sh"\n'
        f"    # shellcheck disable=SC1091\n"
        f'    source "$_conda_base/etc/profile.d/conda.sh"\n'
    )
    return head + body + activate_call


# Placeholder for the baked-in conda base; replaced at wrapper-render
# time via str.replace so the helper doesn't need to thread the path.
_BAKED_PLACEHOLDER = "__MOLBUILDER_BAKED_CONDA_BASE__"


def _detect_generate_time_conda_base() -> Optional[str]:
    """Return the conda root visible to the generator process, or None.

    Used to BAKE the generator-machine's conda base path into the
    emitted wrapper as a hint.  The wrapper's runtime activation block
    tries the baked-in path FIRST (works perfectly when generate ==
    target machine, which is the common HPC case where the user runs
    ``molbuilder run`` after ``conda activate molbuilder`` on the same
    cluster).  Falls back to runtime ``conda info --base`` and a
    common-location probe if the baked-in path doesn't exist on the
    target machine (the cross-machine case, e.g. generate on laptop +
    rsync to cluster).

    Resolution order:
      1. ``$CONDA_PREFIX`` walked to its base (strip ``/envs/<name>``).
      2. ``mamba info --base`` (preferred -- faster, and many HPCs
         ship only mamba, e.g. ASU sc002 via ``module load mamba``).
      3. ``conda info --base`` (fallback for conda-only installs).
      4. None -- the wrapper's runtime probe is the only remaining
         path.
    """
    cp = os.environ.get("CONDA_PREFIX")
    if cp:
        cp_path = Path(cp)
        # Walk up if we're inside <base>/envs/<name>.
        if cp_path.parent.name == "envs":
            cp_path = cp_path.parent.parent
        if (cp_path / "etc" / "profile.d" / "conda.sh").is_file():
            return str(cp_path)
    for binary in ("mamba", "conda"):
        if not shutil.which(binary):
            continue
        try:
            r = subprocess.run(
                [binary, "info", "--base"],
                capture_output=True, text=True, timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode == 0 and r.stdout.strip():
            cand = Path(r.stdout.strip())
            if (cand / "etc" / "profile.d" / "conda.sh").is_file():
                return str(cand)
    return None


# Shell-safety guard for wrapper emission.  The wrapper interpolates
# ``basename`` and ``script_name`` (the script's filename stem and
# the filename itself) into many bash strings -- some inside ``"..."``,
# some unquoted in glob lists, some inside ``$(...)`` substitutions.
# A filename containing ``"``, ``$``, ``` ` ```, ``\\``, ``;``, ``|``,
# ``&``, newlines, etc. would either break the wrapper or execute
# arbitrary code at run time.
#
# Rather than escape every interpolation (fragile + makes the bash
# unreadable for the user, who is meant to be able to read and edit
# the wrapper), we restrict the input to a safe alphabet at
# emission time.  Matches the ``_BASENAME_RE`` rule in
# ``molbuilder/config/siesta.py`` (SystemLabel validation) and
# ``molbuilder/config/pyscf.py`` (job_name): letters, digits, dot,
# underscore, hyphen.  The dot is allowed because PySCF emits files
# like ``<basename>.molwatch.log`` and SIESTA's user-named
# SystemLabel often contains dots.
_SAFE_WRAPPER_NAME_RE = re.compile(r"^[A-Za-z0-9._\-]+$")


class WrapperError(Exception):
    """Wrapper cannot be generated -- unsupported extension, no env
    routing, missing script file, ..."""


def _run_index_resolver(basename: str, ext: str = ".out") -> str:
    """Bash block that resolves ``_out_file`` to
    ``{basename}-runN{ext}``.

    ``ext`` is the output-file suffix (with leading dot).  Defaults
    to ``.out`` for SIESTA wrappers; PySCF wrappers pass
    ``.pyscf.log`` so the Results-tab dispatcher can tell PySCF
    output apart from SIESTA's (Phase C, 2026-06-07).

    Honours two shell variables that the caller (the engine-specific
    args block) is expected to set:

      ``_continue`` (0/1)  -- ``--continue`` was passed; advance to
        next free N.  Without ``--continue`` we refuse to overwrite
        an existing -runN{ext} unless ``--force`` was also passed.
      ``_force``    (0/1)  -- ``--force`` was passed; allow restart
        from -run0 even if prior runs exist (they are preserved on
        disk; we just don't continue from them).

    The resolver is shared by SIESTA + PySCF wrappers so the run-index
    semantics are identical across engines; only the suffix differs.
    ``basename`` is the script stem (e.g. ``siesta-hemeC-gas-stage3``)
    baked in at generation time -- the bash itself doesn't try to
    derive it.
    """
    return (
        f"# --- Run index resolution ------------------------------\n"
        f"# Outputs are ``{basename}-runN{ext}``.  First run produces\n"
        f"# -run0; ``--continue`` scans existing -runN files and uses\n"
        f"# max(N)+1.  Without --continue, refuse to overwrite an\n"
        f"# existing -run0 unless --force was passed (otherwise the\n"
        f"# user would silently lose the previous result).\n"
        f"_existing_max=-1\n"
        f'shopt -s nullglob 2>/dev/null || true\n'
        f'for _f in "{basename}-run"*{ext}; do\n'
        f'    _n=${{_f#{basename}-run}}\n'
        f'    _n=${{_n%{ext}}}\n'
        f'    case "$_n" in\n'
        f"        ''|*[!0-9]*) continue ;;\n"
        f"    esac\n"
        f'    if [ "$_n" -gt "$_existing_max" ]; then\n'
        f'        _existing_max=$_n\n'
        f"    fi\n"
        f"done\n"
        f"\n"
        f'if [ "$_continue" = "1" ]; then\n'
        f'    if [ "$_existing_max" -ge 0 ]; then\n'
        f"        _run_n=$((_existing_max + 1))\n"
        f"    else\n"
        f'        echo "[molbuilder] --continue requested but no prior '
        f'-runN{ext} found; starting fresh as -run0" >&2\n'
        f"        _run_n=0\n"
        f"    fi\n"
        f"else\n"
        f'    if [ "$_existing_max" -ge 0 ] && [ "$_force" != "1" ]; then\n'
        f'        echo "ERROR: previous output exists '
        f'(``{basename}-run${{_existing_max}}{ext}``)." >&2\n'
        f'        echo "  Use --continue to add '
        f'``{basename}-run$((_existing_max + 1)){ext}`` '
        f'(resume from last state)." >&2\n'
        f'        echo "  Use --force to start over from -run0 '
        f'(overwrites the existing run0)." >&2\n'
        f"        exit 1\n"
        f"    fi\n"
        f"    _run_n=0\n"
        f"fi\n"
        f'_out_file="{basename}-run${{_run_n}}{ext}"\n'
        f'echo "[molbuilder] run index: $_run_n  ->  $_out_file"\n'
        f"\n"
    )


def _continue_force_args_parser(name_for_usage: str) -> str:
    """Bash snippet declaring + parsing ``--continue`` / ``-c`` /
    ``--force`` / ``-f`` / ``--cold`` / ``--from-scratch``.

    Three orthogonal flags:

      * ``--continue`` / ``-c``: advance run-index AND let the
        engine warm-start from prior state files (SIESTA: .DM, .CG,
        .XV; PySCF: .chk).
      * ``--force`` / ``-f``: start a fresh run-index sequence
        (-run0) even when prior outputs exist.  Does NOT touch the
        engine warm-start files -- the engine still loads them.
      * ``--cold`` / ``--from-scratch``: move prior warm-start
        files (SIESTA .DM/.CG/.XV/.LWF/.ZM; PySCF .chk) into a
        timestamped backup directory before running, so the engine
        starts strictly from the .fdf / .py coordinates and
        conditions.  Distinct from ``--force`` which only resets
        the run-index; ``--cold`` resets the engine state too.

    Added 2026-06-14 after the BDT-stage-2 incident where stage 2
    ran without the user's frozen-atom constraints (contract bug
    in the form, fixed in the same release); the now-corrupt
    warm-start files would have contaminated every subsequent run
    until ``--cold`` was provided.

    Caller is responsible for the eventual ``--help`` text.
    """
    return (
        f"# --- Continuation flags (shared SIESTA / PySCF) --------\n"
        f"# ``--continue`` / ``-c``: advance run-index AND warm-\n"
        f"#                          start from prior .DM/.CG/.XV\n"
        f"#                          (SIESTA) or .chk (PySCF).\n"
        f"# ``--force``    / ``-f``: reset run-index to -run0; prior\n"
        f"#                          warm-start files remain on disk\n"
        f"#                          and the engine still loads them.\n"
        f"# ``--cold`` / ``--from-scratch``:\n"
        f"#                          move warm-start files (.DM/.CG/\n"
        f"#                          .XV/.LWF/.ZM/.chk) into a\n"
        f"#                          timestamped backup dir BEFORE\n"
        f"#                          running, so the engine starts\n"
        f"#                          purely from the .fdf/.py.  Use\n"
        f"#                          when the prior run was bad and\n"
        f"#                          its restart files would corrupt\n"
        f"#                          the next run.\n"
        f"_continue=0\n"
        f"_force=0\n"
        f"_cold=0\n"
        f"# We strip --continue / --force / --cold from $@ here,\n"
        f"# leaving the rest for the engine-specific arg loop below\n"
        f"# (-np for SIESTA; nothing for PySCF).\n"
        f"_argv_remaining=()\n"
        f'while [ $# -gt 0 ]; do\n'
        f'    case "$1" in\n'
        f"        --continue|-c)        _continue=1; shift ;;\n"
        f"        --force|-f)           _force=1;    shift ;;\n"
        f"        --cold|--from-scratch) _cold=1;    shift ;;\n"
        f'        *)                    _argv_remaining+=("$1"); shift ;;\n'
        f"    esac\n"
        f"done\n"
        f'set -- "${{_argv_remaining[@]+\"${{_argv_remaining[@]}}\"}}"\n'
        f"\n"
    )


def _cold_restart_aside_block(basename: str, *, engine: str) -> str:
    """Bash snippet that moves engine warm-start files aside when
    ``_cold=1``.  Idempotent: no-op when ``_cold=0`` or when no
    warm-start files exist.

    Engine extensions handled:

      * ``siesta``: ``.DM``, ``.CG``, ``.XV``, ``.LWF``, ``.ZM``,
        ``.Bonds``, ``.PARTIAL``, ``.EIG`` (anything SIESTA's
        ``DM.UseSaveDM`` / ``MD.UseSaveCG`` / ``MD.UseSaveXV``
        family auto-loads).  We move them all so a partial set
        can't trigger a half-warm-start.
      * ``pyscf``: ``.chk`` (SCF DM init guess), ``_optimized.xyz``
        (geometry warm-restart hook, task #539), ``_geom_optim.xyz``
        / ``_geom_optim.tmp`` / ``_geom.tmp`` (geomeTRIC trajectory
        + checkpoints).  Anything the generator's auto-resume
        branches (or geomeTRIC's own append mode) would pick up.

    Backups land in
    ``<basename>-restart-aside-<UTC-timestamp>/`` so the user can
    inspect / recover the prior state if needed.  Deleting that
    directory is a manual user action -- we never auto-delete the
    user's prior results.
    """
    if engine == "siesta":
        # SIESTA warm-start file extensions.  Every one of these
        # carries SCF / geometry / electronic state that a fresh
        # run would otherwise pick up via ioxv / restart code paths,
        # silently contaminating the "cold" run.
        #
        #   DM, CG, XV, LWF, ZM, Bonds, PARTIAL, EIG
        #     -- canonical relaxation restart set (density matrix,
        #        CG geometry checkpoint, coordinates+velocities,
        #        Wannier functions, Z-matrix, bond cache, partial
        #        sums, eigenvalues).
        #   HSX     -- Hamiltonian + overlap matrices.  Loaded by
        #              TranSIESTA NEGF on restart.  Missing this in
        #              the first --cold ship was a B.3 transport
        #              hazard: a fresh run could pick up the prior
        #              run's H/S matrices and silently reuse them.
        #   WFSX    -- saved wavefunctions (SaveWaveFunctions /
        #              post-processing).  Loaded on TS.SaveBias /
        #              transmission calculations.
        #   STRUCT_NEXT_ITER -- next-iteration geometry checkpoint
        #              written by SIESTA at the end of every CG /
        #              FIRE step.  SIESTA reads it on restart with
        #              ``MD.UseStructFile T`` (default true for
        #              relaxations); without removal a stage-2
        #              cold run reuses the stage-1 geometry.
        #   TSHS    -- TranSIESTA self-energy Hamiltonian; loaded on
        #              electrode reuse.  Critical for transport.
        #   TSDE    -- TranSIESTA density matrix (NEGF-specific
        #              counterpart of .DM).  Same restart hazard.
        #
        # Intentionally OMITTED: .PSF (pseudopotential cache;
        # regenerable, and a stale cache may carry the user's
        # custom pseudo which would be destructive to remove).
        exts = (
            "DM", "CG", "XV", "LWF", "ZM", "Bonds", "PARTIAL", "EIG",
            "HSX", "WFSX", "STRUCT_NEXT_ITER",
            "TSHS", "TSDE",
        )
    elif engine == "pyscf":
        # PySCF warm-start file SUFFIXES (NOT bare extensions -- the
        # generator names per-stage trajectory files
        # ``<JOB>_geom_optim.xyz`` and the optimized-geometry hook
        # ``<JOB>_optimized.xyz``, neither of which a plain
        # ``{ext}`` glob would catch).  The runwrap must move ALL of
        # these aside on ``--cold`` so a fresh run can't silently
        # warm-restart from a partial prior state.
        #
        # Inventory (synced with docs/protocols/script-execution.md
        # § "Warm-restart file inventory" -> PySCF table):
        #
        #   .chk              -- SCF density matrix.  Auto-loaded by
        #                        the ``mf.chkfile`` + ``if exists ->
        #                        init_guess="chkfile"`` block in the
        #                        generated script.
        #   _optimized.xyz    -- last converged geometry.  Auto-loaded
        #                        by the ``_atom_block`` warm-restart
        #                        hook in the generated script (task
        #                        #539 generator side).
        #   _geom_optim.xyz   -- geomeTRIC trajectory frames.  Not
        #                        auto-loaded by molbuilder but
        #                        geomeTRIC's own append mode picks
        #                        them up if present + can corrupt the
        #                        new trajectory.
        #   _geom_optim.tmp,
        #   _geom.tmp         -- geomeTRIC checkpoints.  geomeTRIC
        #                        resumes from these on certain failure
        #                        modes; leaving them in place defeats
        #                        ``--cold``.
        #
        # Suffix-keyed, not extension-keyed: ``.chk`` is just
        # "``.chk``" but the others end in ``_optimized.xyz`` etc.
        # The glob template below interpolates each suffix verbatim
        # against ``$_warm_label`` + the wrapper basename.
        suffixes = (
            ".chk",
            "_optimized.xyz",
            "_geom_optim.xyz",
            "_geom_optim.tmp",
            "_geom.tmp",
        )
    else:                                  # pragma: no cover
        raise WrapperError(f"unknown engine for cold-restart: {engine!r}")
    # 2026-06-14 fix: SIESTA names its warm-start files after the
    # ``SystemLabel`` from inside the .fdf, NOT after the .fdf's
    # filename basename.  E.g. an .fdf named ``foo-stage2.fdf``
    # whose ``SystemLabel`` line says ``foo`` writes ``foo.DM`` /
    # ``foo.XV`` / ``foo.CG`` -- NOT ``foo-stage2.DM``.  The first
    # ``--cold`` ship missed this and emitted a glob against the
    # wrapper basename only, which silently missed every staged-
    # relaxation project (basename ``foo-stage2`` vs SystemLabel
    # ``foo``).  At runtime we read the SystemLabel from the .fdf
    # for SIESTA; for PySCF we read the ``JOB`` assignment from
    # the .py script which is the equivalent.
    # The glob also covers the wrapper-basename case as a fallback
    # because some users name SystemLabel == basename and we still
    # want to clean those up.
    if engine == "siesta":
        label_extract = (
            f'# Read SystemLabel from the .fdf so the warm-start glob\n'
            f"# matches the files SIESTA will look for at startup\n"
            f"# (not what the wrapper's filename happens to be).\n"
            f"# Robustness notes:\n"
            f"#   * Lower-case the line before matching instead of\n"
            f"#     using gawk's IGNORECASE (mawk / BusyBox awk /\n"
            f"#     BSD awk silently ignore IGNORECASE).\n"
            f"#   * Strip surrounding ``\"...\"`` quotes from the\n"
            f"#     extracted token -- SIESTA accepts quoted labels.\n"
            f"#   * ``|| true`` keeps awk's non-zero exit (e.g.\n"
            f"#     missing .fdf) from aborting the wrapper under\n"
            f"#     ``set -euo pipefail``.\n"
            f"#   * The :- default guards against ``set -u`` when\n"
            f"#     awk produced no output at all.\n"
            f'_warm_label=$(awk \''
            f'tolower($1) == "systemlabel" '
            f'{{ gsub(/"/, "", $2); print $2; exit }}'
            f'\' "{basename}.fdf" 2>/dev/null || true)\n'
            f'_warm_label="${{_warm_label:-{basename}}}"\n'
            # J1 2026-06-14 (defense in depth): sanitize the
            # SystemLabel string read from the .fdf to the same
            # charset the wrapper basename already enforces
            # ([A-Za-z0-9._-]; ``_SAFE_WRAPPER_NAME_RE``).  Bash
            # double-quotes already block command-substitution
            # on ``$_warm_label`` -- this is belt + suspenders for
            # the case where a future emitter forgets the quotes
            # and ``rm -rf /`` lives in a SystemLabel.  ``tr`` is
            # POSIX so we don't drag in awk or grep for the
            # sanitiser; everything matching the safe charset
            # passes through, everything else is dropped.
            'case "$_warm_label" in\n'
            '    *[!A-Za-z0-9._-]*)\n'
            f'        echo "[molbuilder] warning: SystemLabel in '
            f'{basename}.fdf contained disallowed characters; '
            f'falling back to basename" >&2\n'
            f'        _warm_label="{basename}"\n'
            '        ;;\n'
            'esac\n'
        )
        # Two-label glob: SystemLabel-keyed AND wrapper-basename-keyed
        # so we catch both naming styles.
        glob_pieces = " ".join(
            f'"$_warm_label.{ext}" "$_warm_label".*.{ext} '
            f"{basename}.{ext} {basename}.*.{ext}"
            for ext in exts
        )
    else:
        # PySCF: extract the JOB assignment from the script.
        # Anchored to a literal ``JOB =`` (whitespace tolerant) on
        # a single line.  Same set-u guard as SIESTA.
        # The -F char class needs to match both ``"`` and ``'``.
        # We use awk's octal ``\047`` for ``'`` so the bash-side SQ
        # delimiter never has to be escape-broken.  The malformed
        # ``-F'["\\'"]'`` pattern that shipped pre-2026-06-20 left
        # bash with an unterminated DQ; gated now by `bash -n` self-
        # check in :func:`_validate_rendered_wrapper` + an L2 test on
        # the PySCF render path (test_runwrap_pyscf_bash_n).
        label_extract = (
            f'_warm_label=$(awk -F\'["\\047]\' \''
            f'/^[[:space:]]*JOB[[:space:]]*=/ '
            f'{{print $2; exit}}'
            f'\' "{basename}.py" 2>/dev/null || true)\n'
            f'_warm_label="${{_warm_label:-{basename}}}"\n'
            # Mirror of the SIESTA-side sanitizer above (J1 2026-
            # 06-14): the PySCF JOB string is user-controlled too
            # (read from the user's script).  Sanitize to the same
            # safe charset.
            'case "$_warm_label" in\n'
            '    *[!A-Za-z0-9._-]*)\n'
            f'        echo "[molbuilder] warning: JOB string in '
            f'{basename}.py contained disallowed characters; '
            f'falling back to basename" >&2\n'
            f'        _warm_label="{basename}"\n'
            '        ;;\n'
            'esac\n'
        )
        # Suffix-based glob template: each entry expands to BOTH the
        # JOB-keyed name (from inside the .py) AND the wrapper-
        # basename-keyed fallback (for users whose JOB happens to
        # match the wrapper basename).  Both forms are emitted so a
        # mismatch between JOB and basename can't slip a warm-start
        # file past the move-aside step.
        #
        # CRITICAL: brace the var expansion as ``${_warm_label}``,
        # NOT bare ``$_warm_label``.  Suffixes here START with ``_``
        # (e.g. ``_optimized.xyz``), and bash absorbs trailing
        # ``[A-Za-z0-9_]+`` into the variable name -- so
        # ``"$_warm_label_optimized.xyz"`` would dereference the
        # variable ``_warm_label_optimized`` (unset -> trips
        # ``set -u``) instead of the intended concatenation.  Braces
        # terminate the variable name explicitly.  The pre-#539 glob
        # template happened to be safe because every suffix started
        # with ``.`` which terminates the var name naturally.
        glob_pieces = " ".join(
            f'"${{_warm_label}}{suffix}" {basename}{suffix}'
            for suffix in suffixes
        )
    return (
        f"# --- Cold-restart: move engine warm-start files aside ---\n"
        f'if [ "$_cold" = "1" ]; then\n'
        f"    # UTC timestamp keeps multiple cold runs from\n"
        f"    # colliding when they fire within a second of each\n"
        f"    # other.\n"
        f'    _aside="{basename}-restart-aside-$(date -u +%Y%m%dT%H%M%SZ)"\n'
        f"    _moved=0\n"
        f"    shopt -s nullglob 2>/dev/null || true\n"
        f"    {label_extract}"
        f"    echo \"[molbuilder] --cold: scanning for "
        f"\\\"$_warm_label\\\".* and \\\"{basename}\\\".* warm-start files\" >&2\n"
        f"    for _f in {glob_pieces}; do\n"
        f'        if [ -e "$_f" ]; then\n'
        f'            if [ "$_moved" = "0" ]; then\n'
        f'                mkdir -p "$_aside"\n'
        f"                _moved=1\n"
        f"            fi\n"
        f'            mv "$_f" "$_aside/"\n'
        f'            echo "[molbuilder] --cold: moved $_f" >&2\n'
        f"        fi\n"
        f"    done\n"
        f'    if [ "$_moved" = "1" ]; then\n'
        f'        echo "[molbuilder] --cold: moved warm-start files into $_aside/" >&2\n'
        f"    else\n"
        f'        echo "[molbuilder] --cold: no warm-start files to move; already a clean start" >&2\n'
        f"    fi\n"
        f"fi\n"
        f"\n"
    )


def _runtime_status_block(
    basename: str,
    *,
    engine: str,
    script_name: str,
) -> str:
    """Bash snippet that detects and emits the execution status banner.

    Prints, at script launch time, AFTER cold/run-index resolution but
    BEFORE the engine launches:

      * **Mode** -- one of:
        - ``COLD`` (``--cold`` was passed; warm-start files moved aside)
        - ``WARM-RESUME`` (``--continue`` with prior state on disk)
        - ``WARM-RESUME REQUESTED but no prior state`` (``--continue``
          with nothing to resume from -- the user probably intended
          this to be a fresh start)
        - ``WARM-RESTART`` (no ``--continue``, but warm-start files
          exist on disk; the engine will silently load them.  This is
          the silent-failure mode pre-2026-06-14: a stage-2 run with
          bad constraints contaminated stage-3's restart files and the
          user couldn't see why their numbers were wrong.  The flag
          now makes this case loudly visible.)
        - ``initial-run (clean state)`` (no prior state; no flags).

      * **Constraints** -- engine-specific:
        - SIESTA: counts ``position`` lines + total listed indices in
          ``%block Geometry.Constraints``; ``(none)`` when the block
          is absent or empty.
        - PySCF: counts indices from the ``# Source: Structure.
          frozen_atoms = [...]`` comment the generator emits.

    The detection logic reads the actual on-disk script at runtime
    (NOT the values baked in at emit time), so a user-edited .fdf
    or .py shows the EDITED values -- the script ``you see is
    what runs`` contract that the 2026-06-14 fix landed.

    Added 2026-06-14 as part of the BDT-stage-2 incident response:
    silently-warm-restarting from contaminated restart files would
    have re-contaminated every downstream run; users have to be able
    to see at a glance whether their constraints are honored and
    where the SCF / geometry will start from.
    """
    if engine == "siesta":
        # Warm-start file extensions matching the cold-restart block.
        # 2026-06-14 fix: SIESTA writes warm-start files keyed on the
        # ``SystemLabel`` from inside the .fdf -- NOT on the .fdf's
        # filename basename.  So a stage2 .fdf with
        # ``SystemLabel  foo`` writes ``foo.DM`` etc., regardless of
        # the script being named ``foo-stage2.fdf``.  Test both label
        # patterns (SystemLabel-keyed AND basename-keyed) for the
        # warm-start detection so the Mode line is accurate.
        # Match the full SIESTA warm-start ext tuple used by the
        # cold-restart aside below — covers transport (.HSX/.TSHS/
        # .TSDE/.WFSX) and the geometry-checkpoint case
        # (STRUCT_NEXT_ITER) too.  Used only to detect whether the
        # banner should report "Mode: hot/warm" vs "Mode: cold".
        warmstart_exts = (
            "DM", "CG", "XV", "LWF", "ZM",
            "HSX", "WFSX", "STRUCT_NEXT_ITER",
            "TSHS", "TSDE",
        )
        warmstart_test_pieces = []
        for ext in warmstart_exts:
            warmstart_test_pieces.append(f'[ -e "$_warm_label.{ext}" ]')
            warmstart_test_pieces.append(f'[ -e "{basename}.{ext}" ]')
        warmstart_test = " || ".join(warmstart_test_pieces)
        warmstart_listing = " ".join(
            f"{basename}.{ext}" for ext in warmstart_exts
        )
        # Awk that:
        #   * lower-cases the first token for case-insensitive
        #     ``%block`` / ``%endblock`` / ``position`` matching
        #     (portable; replaces gawk's IGNORECASE).
        #   * accepts BOTH ``Geometry.Constraints`` and
        #     ``Geometry_Constraints`` for the block name (SIESTA
        #     treats ``.`` and ``_`` interchangeably).
        # Two passes so ``_ncon_lines`` and ``_ncon_indices`` come
        # from the same logical scan.  ``|| true`` keeps awk's exit
        # code from aborting under ``set -euo pipefail``.
        _awk_count_program = (
            r'{ k = tolower($1) } '
            r'k == "%block" && '
            r'(tolower($2) == "geometry.constraints" || '
            r'tolower($2) == "geometry_constraints") '
            r'{ in_b = 1; next } '
            r'k == "%endblock" && '
            r'(tolower($2) == "geometry.constraints" || '
            r'tolower($2) == "geometry_constraints") '
            r'{ in_b = 0; next } '
            r'in_b && tolower($1) == "position"'
        )
        constraint_detection = (
            f'_constraints="(no Geometry.Constraints block -- all atoms free)"\n'
            f'_fdf_path="{script_name}"\n'
            # Grep gate: case-insensitive AND tolerant of both
            # `.` and `_` in the block name.  Same dialect as the
            # awk below.
            f'if [ -e "$_fdf_path" ] && grep -qiE '
            f'\'^[[:space:]]*%block[[:space:]]+Geometry[._]Constraints\' '
            f'"$_fdf_path"; then\n'
            f'    _ncon_lines=$(awk \'BEGIN{{in_b=0;n=0}} '
            f'{_awk_count_program} '
            f'{{ n++ }} END{{ print n }}\' "$_fdf_path" 2>/dev/null '
            f'|| echo 0)\n'
            f'    _ncon_indices=$(awk \'BEGIN{{in_b=0;n=0}} '
            f'{_awk_count_program} '
            f'{{ for (i=2;i<=NF;i++) if ($i ~ /^[0-9]+$/) n++ }} '
            f'END{{ print n }}\' "$_fdf_path" 2>/dev/null '
            f'|| echo 0)\n'
            f'    if [ -z "$_ncon_lines" ] || [ "$_ncon_lines" = "0" ]; then\n'
            f'        _constraints="Geometry.Constraints block present but EMPTY -- all atoms free"\n'
            f'    else\n'
            f'        _constraints="$_ncon_lines position line(s), $_ncon_indices listed indices (range expansion not counted)"\n'
            f"    fi\n"
            f"fi\n"
        )
        warm_files_label = "DM/CG/XV/LWF/ZM"
    elif engine == "pyscf":
        # PySCF's ``mf.chkfile`` is keyed on ``JOB`` (a Python
        # variable in the .py script), same naming-mismatch risk as
        # SIESTA's SystemLabel.  Mirror the dual test.
        warmstart_test = (
            f'[ -e "$_warm_label.chk" ] || [ -e "{basename}.chk" ]'
        )
        warmstart_listing = f"{basename}.chk"
        # PySCF embeds the canonical frozen-atom list as a single-line
        # comment.  Counting digits in that comment is sufficient
        # since the indices are comma-separated inside ``[...]``.
        constraint_detection = (
            f'_constraints="(no frozen_atoms -- all atoms free)"\n'
            f'_py_path="{script_name}"\n'
            f'if [ -e "$_py_path" ]; then\n'
            f'    _frozen_line=$(grep -E \'^[[:space:]]*#[[:space:]]*Source:[[:space:]]+Structure\\.frozen_atoms\' "$_py_path" | head -1 || true)\n'
            f'    if [ -n "$_frozen_line" ]; then\n'
            f'        _ncon_indices=$(printf %s "$_frozen_line" | grep -oE \'[0-9]+\' | wc -l)\n'
            f'        if [ "$_ncon_indices" = "0" ]; then\n'
            f'            _constraints="frozen_atoms comment present but lists 0 indices -- all atoms free"\n'
            f"        else\n"
            f'            _constraints="frozen_atoms: $_ncon_indices listed indices (geomeTRIC constraints file)"\n'
            f"        fi\n"
            f"    fi\n"
            f"fi\n"
        )
        warm_files_label = "chk"
    else:                                  # pragma: no cover
        raise WrapperError(f"unknown engine for status block: {engine!r}")

    # Extract the engine's canonical label (SystemLabel for SIESTA,
    # JOB for PySCF) UNCONDITIONALLY so the warmstart_test below has
    # ``$_warm_label`` in scope even when ``--cold`` was NOT passed
    # (the cold block also extracts it but inside its own ``if``).
    # The wrapper runs under ``set -euo pipefail``; an unbound
    # variable would otherwise abort the run.
    # Same robustness rules as the cold block's extraction:
    # portable case-insensitive matching (no gawk IGNORECASE),
    # quote-stripping for SystemLabel, ``|| true`` under set-e,
    # ``:-`` default under set-u.
    if engine == "siesta":
        label_extract_unconditional = (
            f'_warm_label=$(awk \''
            f'tolower($1) == "systemlabel" '
            f'{{ gsub(/"/, "", $2); print $2; exit }}'
            f'\' "{script_name}" 2>/dev/null || true)\n'
            f'_warm_label="${{_warm_label:-{basename}}}"\n'
        )
    else:                              # pyscf
        # Same octal-\047 escape rationale as in label_extract above.
        label_extract_unconditional = (
            f'_warm_label=$(awk -F\'["\\047]\' \''
            f'/^[[:space:]]*JOB[[:space:]]*=/ '
            f'{{print $2; exit}}'
            f'\' "{script_name}" 2>/dev/null || true)\n'
            f'_warm_label="${{_warm_label:-{basename}}}"\n'
        )
    return (
        f"# --- Runtime status banner --------------------------\n"
        f"# Reads the actual on-disk script + state files at run\n"
        f"# time and reports the resulting MODE + CONSTRAINTS so\n"
        f"# the user can see what's about to happen BEFORE the\n"
        f"# engine starts.  See _runtime_status_block docstring.\n"
        + label_extract_unconditional
        + f'_warmstart_present=0\n'
        + f"if {warmstart_test}; then _warmstart_present=1; fi\n"
        f'_mode="initial-run (clean state)"\n'
        f'if [ "$_cold" = "1" ]; then\n'
        f'    _mode="COLD (--cold; warm-start files moved aside)"\n'
        f'elif [ "$_continue" = "1" ]; then\n'
        f'    if [ "$_warmstart_present" = "1" ]; then\n'
        f'        _mode="WARM-RESUME (--continue; engine will load {warm_files_label})"\n'
        f"    else\n"
        f'        _mode="WARM-RESUME REQUESTED but no prior state found -- starting cold by necessity"\n'
        f"    fi\n"
        f'elif [ "$_warmstart_present" = "1" ]; then\n'
        f'    _mode="WARM-RESTART (silent; engine will load existing {warm_files_label}.  '
        f'Pass --cold to discard them.)"\n'
        f"fi\n"
        f"{constraint_detection}"
        f"\n"
    )


def _probe_gpu0_numa() -> Optional[int]:
    """Resolve the NUMA node that GPU 0 is attached to.

    Two paths, tried in order:

    1. NVML via the ``nvidia-ml-py`` package (importable as
       ``pynvml`` -- same as ``web/blueprints/system_load.py`` and
       the project's pinned dependency in ``pyproject.toml``).
       Returns the GPU's PCI bus ID as bytes
       (``b"00000000:65:00.0"``) which we feed to step 2.
    2. Kernel sysfs at ``/sys/bus/pci/devices/<id>/numa_node``.  The
       stable ABI the Linux kernel itself uses for NUMA-aware
       allocation.  A single integer; "-1" for "no NUMA / single
       node system".

    Returns the int, or ``None`` when:
      * pynvml isn't importable (CPU-only host, no NVIDIA stack)
      * NVML init or device enumeration fails
      * the sysfs file isn't readable
      * the sysfs value is "-1" (no NUMA affinity)

    Called once per wrapper render, at script-generation time.  The
    answer is baked into the generated bash as a literal -- no
    string-parsing of ``nvidia-smi`` output at runtime.
    """
    try:
        import pynvml  # type: ignore[import-untyped]
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:  # noqa: BLE001 -- many distinct NVML failure types
        return None
    try:
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:  # noqa: BLE001 -- no GPUs / permission denied
            return None
        try:
            pci = pynvml.nvmlDeviceGetPciInfo(h)
        except Exception:  # noqa: BLE001
            return None
        # busId is ``bytes`` in older pynvml, ``str`` in newer; coerce.
        bus_id = getattr(pci, "busId", None)
        if isinstance(bus_id, bytes):
            bus_id = bus_id.decode("ascii", errors="replace")
        if not isinstance(bus_id, str) or not bus_id:
            return None
        bus_id = bus_id.strip().lower()
        # nvidia-smi / NVML format: ``00000000:65:00.0`` (8-hex domain).
        # sysfs path:             ``0000:65:00.0``    (4-hex domain).
        # Strip the leading 4 hex digits of the domain.
        import re
        m = re.match(r"^[0-9a-f]{8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]$", bus_id)
        if not m:
            return None
        sysfs_id = bus_id[4:]
        try:
            with open(f"/sys/bus/pci/devices/{sysfs_id}/numa_node", "r") as fh:
                txt = fh.read().strip()
        except OSError:
            return None
        try:
            val = int(txt)
        except ValueError:
            return None
        if val < 0:
            return None  # "-1" = no NUMA affinity / single-node system
        return val
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


def _baked_numa_literal_line() -> str:
    """Format the ``_gpu_numa="${MOLBUILDER_GPU_NUMA:-<probed>}"``
    bash line with the probed value baked in.  Factored out so the
    "0 is a valid NUMA node, don't truthy-fallback it" logic stays
    in one obvious place."""
    probed = _probe_gpu0_numa()
    value = "unknown" if probed is None else str(probed)
    return f'_gpu_numa="${{MOLBUILDER_GPU_NUMA:-{value}}}"\n'


def _gpu_runtime_defaults_block(n_atoms: Optional[int]) -> str:
    """Bash that probes hardware and computes GPU-mode MPI/OMP defaults.

    Encodes the GPU-mode placement policy for our ELPA-CUDA build
    (single workstation, 1 GPU, no NCCL).  Inputs come from
    ``lscpu`` / NVML / kernel sysfs; outputs are shell variables
    consumed by the rest of the SIESTA wrapper template:

      * ``_gpu_mpi_np_default``: with MPS, ``phys_cores // 4`` capped
        at 4 (ELPA 2024.05 release notes report 4 ranks/GPU as the
        no-NCCL throughput optimum; BSC MareNostrum5 SIESTA-ACC report
        confirms on V100/A100/H100); without MPS, 2 dual-socket or
        ``cps >= 16``, else 1.  Clamped <= n_atoms.
      * ``_gpu_numa``: NUMA node the GPU is attached to.  Baked at
        script-generation time by :func:`_probe_gpu0_numa` (NVML
        ``nvmlDeviceGetPciInfo`` + kernel sysfs
        ``/sys/bus/pci/devices/<id>/numa_node``); overridable at run
        time via ``MOLBUILDER_GPU_NUMA``.  Either an integer string
        ("0", "1", ...) or the literal "unknown" when no GPU was
        present at generation time / NVML failed / sysfs reports
        "-1".  Stable: not subject to nvidia-smi tabular-layout drift.
      * ``_gpu_budget``: phys-core budget the OMP arithmetic divides.
        Two regimes:
          - NUMA-pinned (``_gpu_numa != "unknown"`` AND
            ``_n_sockets >= 2`` AND numactl is on PATH): ``_cps`` --
            the full GPU-proximate socket.  The other socket sits
            idle, leaving plenty of room for the kernel + ELPA-GPU
            host driver thread without reserving a core.
          - Single-socket OR GPU-NUMA unknown: ``_phys_cores - 1``
            -- the whole box minus 1 core for the driver thread,
            because there is no other socket to absorb it.
      * ``_omp_default``: ``_gpu_budget // mpi_np`` -- divide the
        budget across ranks.  Policy revised 2026-06-16 after the
        OMP-per-rank correction: OMP threads DO accelerate ELPA's
        host-side eigensolver stages and SIESTA's non-solver host
        code even when ``Diag.ELPA.GPU`` is on.  The "GPU choice at
        runtime not compatible with OpenMP" ELPA docs sentence
        applies to the ``elpa_setup_gpu`` runtime-switch API
        (2023.11+), NOT to OpenMP threading within a rank that uses
        ``elpa_set("nvidia-gpu", 1)`` at SCF-setup time (which is
        the path SIESTA takes when ``Diag.ELPA.GPU .true.`` is set).
      * Override knobs (read here so the same banner can name them):
        ``MOLBUILDER_MPI_NP`` and ``MOLBUILDER_OMP_NUM_THREADS``.

    The block also prints a kubectl-context-switch-style one-line
    banner to stderr so the user sees the mode change without scrolling
    through the full wrapper banner.  A second line surfaces the
    derived "chosen X ranks × Y threads = Z of phys_cores" arithmetic
    and the GPU-NUMA proximity so the user can decide whether to
    override or run ``molbuilder envs advise siesta-gpu`` for a guided
    pick.
    """
    n_atoms_lit = "" if n_atoms is None else str(int(n_atoms))
    return (
        "# --- GPU mode: ELPA-CUDA defaults (no NCCL in our build) ---\n"
        "# Policy researched 2026-06-15; sources cited in runwrap.py.\n"
        "# Override anywhere via MOLBUILDER_MPI_NP /\n"
        "# MOLBUILDER_OMP_NUM_THREADS / wrapper '-np N' / MB_NP env.\n"
        '# Hardware probe -- prefer lscpu because it gives PHYSICAL\n'
        '# cores (HT-aware); nproc returns logical, which counts HT\n'
        '# siblings and would make _cps too large -> over-binding.\n'
        '_phys_cores=$(LANG=C lscpu -p=Core,Socket 2>/dev/null '
        '| grep -v "^#" | sort -u | wc -l 2>/dev/null)\n'
        'if [ -z "$_phys_cores" ] || [ "$_phys_cores" -lt 1 ]; then\n'
        '    # Fallback: nproc / 2 (assume 2-way HT, conservative\n'
        '    # estimate; bare nproc would over-count on HT boxes).\n'
        '    _logical=$(nproc --all 2>/dev/null '
        '|| getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)\n'
        '    _phys_cores=$(( _logical / 2 ))\n'
        '    [ "$_phys_cores" -lt 1 ] && _phys_cores=$_logical\n'
        'fi\n'
        '_n_sockets=$(LANG=C lscpu -p=Socket 2>/dev/null '
        '| grep -v "^#" | sort -u | wc -l 2>/dev/null)\n'
        'if [ -z "$_n_sockets" ] || [ "$_n_sockets" -lt 1 ]; '
        'then _n_sockets=1; fi\n'
        '_cps=$(( _phys_cores / _n_sockets ))\n'
        '[ "$_cps" -lt 1 ] && _cps=1\n'
        # ---- MPS availability ----
        # NVIDIA Multi-Process Service: a daemon that lets multiple
        # CUDA client processes share one GPU CONCURRENTLY via Hyper-Q
        # instead of serialising through the driver context.  Without
        # MPS, our 2 MPI ranks queue their CUDA calls sequentially --
        # one rank's ELPA-GPU diag must finish before the other's can
        # start.  With MPS, both ranks' kernels run on the GPU at the
        # same time.  The binary lives on the HOST DRIVER side
        # (nvidia-cuda-mps-control); not a conda package.
        '_have_mps=0\n'
        'if command -v nvidia-cuda-mps-control >/dev/null 2>&1; '
        'then _have_mps=1; fi\n'
        # User overrides: --mps / --no-mps flag wins; env var second;
        # default ON when MPS is available AND the run will use >= 2
        # ranks (single-rank MPS adds overhead with no concurrency
        # benefit).  Decided AFTER mpi_np resolves below.
        '_use_mps_default="${MOLBUILDER_USE_MPS:-$_have_mps}"\n'
        '# Echo what we detected so the user can sanity-check it.\n'
        '# Format ``mps_available=yes/no`` to make it obvious that\n'
        '# this is a binary capability flag, not a count (no matter\n'
        '# how many MPI ranks you run, there\'s exactly ONE MPS\n'
        '# daemon per GPU -- the ranks share it).\n'
        '_have_mps_str="no"; '
        '[ "$_have_mps" = "1" ] && _have_mps_str="yes"\n'
        'echo "molbuilder: detected phys_cores=$_phys_cores, '
        'n_sockets=$_n_sockets, cores_per_socket=$_cps, '
        'mps_available=$_have_mps_str" >&2\n'
        # ---- MPI rank policy ----
        # With MPS: target ~4 ranks/GPU (ELPA User Guide §"ELPA -
        # Usability" reports 4 ranks/GPU as the sweet spot without
        # NCCL on 2024.05; our older 2021.11.001 build runs in the
        # same regime).  Cap so each rank gets >= 4 cores -- ranks
        # crammed onto fewer cores lose to MPI overhead.
        # Without MPS: ranks serialise on the GPU, so 2 is the
        # practical ceiling (the prior 2026-06-15 policy).
        'if [ "$_use_mps_default" = "1" ]; then\n'
        '    _gpu_mpi_np_default=$(( _phys_cores / 4 ))\n'
        '    [ "$_gpu_mpi_np_default" -gt 4 ] && _gpu_mpi_np_default=4\n'
        '    [ "$_gpu_mpi_np_default" -lt 1 ] && _gpu_mpi_np_default=1\n'
        'else\n'
        '    if [ "$_n_sockets" -ge 2 ]; then\n'
        '        _gpu_mpi_np_default=2\n'
        '    elif [ "$_cps" -ge 16 ]; then\n'
        '        _gpu_mpi_np_default=2\n'
        '    else\n'
        '        _gpu_mpi_np_default=1\n'
        '    fi\n'
        'fi\n'
        + (
            # n_atoms clamp -- same rationale as CPU path (avoid empty
            # propor blocks); only emit when generation-time parser
            # found the value.
            f'# n_atoms clamp (auto-parsed from .fdf): {n_atoms_lit}\n'
            f'if [ "$_gpu_mpi_np_default" -gt {n_atoms_lit} ]; then\n'
            f'    _gpu_mpi_np_default={n_atoms_lit}\n'
            f'fi\n'
            if n_atoms_lit else ""
        ) +
        # ---- GPU NUMA proximity (probed at generation time) ----
        # Resolved by the Python generator via ``_probe_gpu0_numa()``
        # using NVML (the official NVIDIA library, already imported
        # by the load-monitor blueprint) + the kernel sysfs ABI
        # (``/sys/bus/pci/devices/<id>/numa_node``).  No string-
        # scraping of ``nvidia-smi``'s tabular output -- that was
        # the failure mode of the 2026-06-16 run where libnuma
        # rejected ``--cpunodebind=N/A`` because we misread the
        # "GPU NUMA ID" column as NUMA Affinity.
        #
        # Baked literal:
        #   * an integer string ("0", "1", ...) when NVML + sysfs
        #     report a real NUMA assignment for GPU 0
        #   * "unknown" when no GPU present at generation time, NVML
        #     can't bind, sysfs says "-1" (no affinity), or the host
        #     is single-NUMA
        # The runtime override ``MOLBUILDER_GPU_NUMA=N`` wins over
        # the baked value (useful for moving the wrapper between
        # boxes or when sysfs lies).
        # NB: explicit ``is None`` check; ``probe() or "unknown"`` is
        # WRONG because 0 is a valid NUMA node (very common on
        # single-GPU boxes where GPU 0 sits on socket 0) and would
        # be silently swallowed by the truthy fallback.  Caught
        # 2026-06-16 by smoke after the libnuma N/A regression.
        _baked_numa_literal_line() +
        '# Defence in depth: even if the baked / override value got\n'
        '# garbage, refuse to use it as a NUMA target unless it parses\n'
        '# as a non-negative integer.\n'
        'case "$_gpu_numa" in\n'
        '    ""|*[!0-9]*) _gpu_numa="unknown" ;;\n'
        'esac\n'
        # ---- numactl availability + NUMA-pin decision ----
        # numactl is the canonical tool for restricting a child
        # process tree to one NUMA node.  We wrap mpirun (not each
        # rank) so the cpuset is inherited uniformly by every rank
        # OpenMPI forks.  Three conditions must all hold:
        #   1. dual-socket+ box (single-socket has no NUMA penalty)
        #   2. GPU NUMA known (else we don't know which socket)
        #   3. numactl on PATH (otherwise the wrap would no-op)
        '_numa_wrap_gpu=""\n'
        '_numa_pinned=0\n'
        'if [ "$_gpu_numa" != "unknown" ] && '
        '[ "$_n_sockets" -ge 2 ] && '
        'command -v numactl >/dev/null 2>&1; then\n'
        '    _numa_wrap_gpu="numactl --cpunodebind=$_gpu_numa '
        '--membind=$_gpu_numa"\n'
        '    _numa_pinned=1\n'
        'fi\n'
        # ---- Phys-core budget for the OMP arithmetic ----
        # NUMA-pinned: only the GPU socket is usable, but its full
        # capacity is available (the OTHER socket sits idle, so the
        # kernel + ELPA-GPU host driver thread can use it -- no need
        # to reserve a core on the GPU socket).
        # Single-socket / NUMA unknown: whole box is in play but the
        # driver thread shares this socket, so leave 1 core.
        'if [ "$_numa_pinned" = 1 ]; then\n'
        '    _gpu_budget=$_cps\n'
        'else\n'
        '    _gpu_budget=$(( _phys_cores - 1 ))\n'
        '    [ "$_gpu_budget" -lt 1 ] && _gpu_budget=1\n'
        'fi\n'
        # OMP policy (2026-06-16): fill the budget -- one OMP thread
        # per available core, divided across ranks.  OMP threads
        # accelerate BOTH ELPA's host-side eigensolver stages
        # (tridiag, back-transform) AND SIESTA's non-solver host code
        # (H_matrix_setup, grid, nlefsm) even when Diag.ELPA.GPU is
        # on.  See the docstring for the policy correction note.
        '_omp_default=$(( _gpu_budget / _gpu_mpi_np_default ))\n'
        '[ "$_omp_default" -lt 1 ] && _omp_default=1\n'
        '# Apply env-var overrides (precedence: env > policy)\n'
        '_gpu_mpi_np_default="${MOLBUILDER_MPI_NP:-$_gpu_mpi_np_default}"\n'
        '_omp_default="${MOLBUILDER_OMP_NUM_THREADS:-$_omp_default}"\n'
        '# Stringify the MPS boolean for the banner.  ``mps=on`` /\n'
        '# ``mps=off`` reads unambiguously next to the numeric\n'
        '# ``mpi_np=N`` / ``OMP=M`` -- no risk of the reader thinking\n'
        '# ``mps=1`` means "1 MPS instance" (there is no count to\n'
        '# choose: one MPS daemon per GPU, all ranks share it).\n'
        '_use_mps_str="off"; '
        '[ "$_use_mps_default" = "1" ] && _use_mps_str="on"\n'
        '_numa_str="off"; [ "$_numa_pinned" = 1 ] && '
        '_numa_str="on (socket $_gpu_numa)"\n'
        '_total_cores_used=$(( _gpu_mpi_np_default * _omp_default ))\n'
        # Banner: 3 lines on stderr, kubectl-style.  Line 1 is the
        # mode summary; line 2 is the derived arithmetic + NUMA pin
        # state + budget shape; line 3 is the advisor / override hint.
        'echo "molbuilder: GPU mode (ELPA-CUDA, no NCCL) -- '
        'mpi_np=$_gpu_mpi_np_default, OMP=$_omp_default, '
        'mps=$_use_mps_str, numa-pin=$_numa_str" >&2\n'
        'echo "molbuilder: chosen $_gpu_mpi_np_default ranks '
        '× $_omp_default threads = $_total_cores_used '
        'of $_gpu_budget budget cores (GPU0 NUMA=$_gpu_numa, '
        'phys=$_phys_cores, cps=$_cps)" >&2\n'
        'echo "molbuilder: tune via \'molbuilder envs advise '
        'siesta-gpu\' or MOLBUILDER_MPI_NP / '
        'MOLBUILDER_OMP_NUM_THREADS / MOLBUILDER_USE_MPS / '
        '-np / -omp / --mps / --no-mps" >&2\n'
        "\n"
    )


def _fdf_requests_gpu(fdf_path: Path) -> bool:
    """Whether the .fdf has ``Diag.ELPA.GPU`` set true.

    The SIESTA 5.4.2 source (Src/diag_option.F90:138-139) accepts two
    keyword spellings that toggle the same internal ``elpa_use_gpu``
    flag: ``Diag.ELPA.UseGPU`` (older) and ``Diag.ELPA.GPU`` (newer).
    Either turning the value true means the job needs to run in the
    ``molbuilder-siesta-gpu`` env (the CPU env's SIESTA is linked
    against a non-CUDA ELPA and silently ignores the flag).

    Match defensively: SIESTA's FDF parser is whitespace- and case-
    insensitive on labels; the value may be ``.true.``, ``true``,
    ``yes``, ``T``, ``Y``, or ``1`` (the canonical truthy set fdf_get
    accepts).  Returns False on any read error -- the routing
    fall-through is the CPU env, which is the safe default.
    """
    import re
    try:
        text = fdf_path.read_text()
    except OSError:
        return False
    pat = re.compile(
        r"(?im)^\s*Diag\.ELPA\.(?:Use)?GPU\b\s+(\S+)"
    )
    truthy = {".true.", "true", "yes", "t", "y", "1"}
    # FDF re-reads the same keyword: last occurrence wins (matches
    # SIESTA's read_options.F90 semantics).
    last_value: Optional[str] = None
    for m in pat.finditer(text):
        last_value = m.group(1).strip().lower()
    return last_value in truthy if last_value is not None else False


def _parse_fdf_n_atoms(fdf_path: Path) -> Optional[int]:
    """Read the ``NumberOfAtoms`` line from a SIESTA .fdf, or None.

    The SIESTA wrapper auto-mpi path needs to know n_atoms so it can
    clamp ``mpi_np <= n_atoms`` -- a rank count exceeding the atom
    count makes the propor IMAX=0 abort unfixable regardless of
    BlockSize.  Parsing the .fdf at install time keeps the wrapper
    self-contained (the .fdf IS the source of truth for what SIESTA
    will see) and avoids plumbing n_atoms through every caller.
    Returns None if the file can't be read or the line isn't found;
    callers fall back to the un-clamped behaviour in that case.
    """
    import re
    try:
        text = fdf_path.read_text()
    except OSError:
        return None
    # SIESTA FDF parsing is whitespace-insensitive + case-insensitive
    # on labels.  Match defensively.
    m = re.search(r"(?im)^\s*NumberOfAtoms\b\s+(\d+)", text)
    return int(m.group(1)) if m else None


def render_run_wrapper(script_path: Path, *,
                        env: Optional[str] = None,
                        mpi_np: Optional[int] = None,
                        omp_threads: Optional[int] = None,
                        max_memory_mb: Optional[int] = None,
                        n_atoms: Optional[int] = None,
                        autodetect_override: Optional[bool] = None) -> str:
    """Return the bash text for a wrapper running ``script_path``.

    Routing by file extension:

    * ``.fdf``  → SIESTA.  Uses ``mpirun -np <N>`` when ``mpi_np`` is
                  given and ≥ 2; redirects stdout to the dynamic
                  ``<basename>-runN.out`` (the run index N is resolved
                  by the wrapper at run time; first run is -run0,
                  ``--continue`` advances to next free N).
    * ``.py``   → PySCF.  Runs ``python <script>`` with the same
                  ``-runN`` redirect, but the suffix is
                  ``.pyscf.log`` instead of ``.out`` (Phase C
                  rename, 2026-06-07) so the Results-tab inspector
                  dispatcher can distinguish PySCF stdout from
                  SIESTA's.  The inlined ``_MolwatchEmitter`` handles
                  its own log files independently.

    Both wrappers accept ``--continue`` / ``-c`` and ``--force`` /
    ``-f``.  See the wrapper's ``-h`` for the full flag inventory.

    Args:
      script_path: the ``.fdf`` or ``.py`` to wrap.
      env: override the routed env name for this invocation.  Default
        is whatever ``Capabilities.env_for_category(<category>)`` returns.
      mpi_np: SIESTA MPI rank count.  Ignored for ``.py`` scripts.
      n_atoms: SIESTA atom count.  Used to clamp the auto-mpi default
        (``resolved_mpi = min(physical_cores, n_atoms)``) -- otherwise
        a small molecule on a many-core box gets mpi_np > n_atoms and
        SIESTA aborts at propor IMAX=0 with no possible BlockSize fix.
        Auto-parsed from the .fdf by ``write_run_wrapper`` when
        omitted; pass None to keep the un-clamped legacy behaviour.
    """
    script_path = Path(script_path)
    suffix = script_path.suffix.lower()
    category = EXTENSION_TO_CATEGORY.get(suffix)
    if category is None:
        raise WrapperError(
            f"`{script_path.name}`: unsupported script extension "
            f"`{suffix}`.  Supported: "
            f"{', '.join(sorted(EXTENSION_TO_CATEGORY))}."
        )

    # SIESTA-GPU routing: the .fdf is the ground truth for which env
    # to run in.  When ``Diag.ELPA.GPU`` is set true at generate time,
    # the job needs ``molbuilder-siesta-gpu`` (whose ELPA was built
    # with --enable-nvidia-gpu) -- the CPU env would silently ignore
    # the keyword and run on CPU.  Inspecting the fdf here keeps the
    # config -> runwrap path stateless: there's no parallel routing
    # metadata to keep in sync with the file.
    #
    # IMPORTANT: ``category`` drives every downstream ``if category ==
    # "siesta":`` branch in this module (MPI launch, .out filename,
    # log extension, runtime-status block).  We must NOT change it
    # here -- only the env LOOKUP needs to differ.  The earlier shape
    # of this fix mutated ``category`` to ``"siesta-gpu"`` and silently
    # disabled the entire SIESTA branch, which leaked a ``.pyscf.log``
    # filename and an unbalanced-quote template into the wrapper.
    env_lookup_category = category
    if (category == "siesta" and env is None
            and _fdf_requests_gpu(script_path)):
        env_lookup_category = "siesta-gpu"

    caps = get_capabilities()
    target_env = (env if env is not None
                  else caps.env_for_category(env_lookup_category))
    if target_env is None:
        raise WrapperError(
            f"category `{category}`: no env name registered.  Pass "
            f"env=... explicitly or add a default to "
            f"molbuilder.diagnostics.DEFAULT_ENV_NAMES."
        )

    basename = script_path.stem
    script_name = script_path.name
    # Shell-safety: both basename and script_name are interpolated
    # raw into bash f-strings throughout this module (inside
    # ``"..."``, inside glob lists, inside ``$(...)``, etc.).
    # Reject anything outside ``_SAFE_WRAPPER_NAME_RE`` to prevent
    # shell injection via a malicious filename.  The same alphabet
    # the SIESTA SystemLabel / PySCF job_name validators enforce
    # in ``molbuilder/config/*``.
    if not _SAFE_WRAPPER_NAME_RE.fullmatch(basename):
        raise WrapperError(
            f"unsafe script basename for wrapper emission: "
            f"{basename!r}.  Allowed characters: letters, digits, "
            f"``.``, ``_``, ``-``.  Rename the script before "
            f"running ``molbuilder run``."
        )
    if not _SAFE_WRAPPER_NAME_RE.fullmatch(script_name):
        # script_name is basename + suffix; if basename passed but
        # script_name fails, the suffix carries the offending char
        # (shouldn't be reachable since suffix is fixed to .fdf/.py,
        # but defence in depth).
        raise WrapperError(
            f"unsafe script filename for wrapper emission: "
            f"{script_name!r}."
        )

    # Pre-command env exports.  Shared anti-oversubscription recipe
    # with PySCF / spectra (see molbuilder/runtime_info.py): BLAS is
    # ALWAYS pinned to 1 thread per rank so OMP * BLAS doesn't
    # multiply.  OMP defaults differ by engine:
    #
    #   * SIESTA: ``OMP_NUM_THREADS = 1`` (mainline SIESTA is not
    #     reliably OMP-aware; pure MPI is the standard recipe).  User
    #     overrides via cfg.omp_threads only when running an
    #     OMP-compiled SIESTA build (hybrid MPI+OMP).
    #
    #   * PySCF: handled in-script by molbuilder.runtime_info, which
    #     sets OMP_NUM_THREADS = physical_cores (NOT physical_cores
    #     // mpi_np -- PySCF doesn't use MPI, only OMP).  The
    #     wrapper deliberately leaves OMP_NUM_THREADS unset so the
    #     in-script setdefault wins.
    env_prefix = ""
    if category == "siesta":
        # GPU mode is detected from the .fdf (the single source of
        # truth -- see _fdf_requests_gpu).  When on, the wrapper
        # switches to the ELPA-CUDA defaults policy researched
        # 2026-06-15:
        #   * mpi_np = 2 if dual-socket OR cores/socket >= 16,
        #     else 1 (clamped <= n_atoms)
        #   * OMP = cores_per_socket // 2, forced even
        #   * mpirun --bind-to core --map-by package:PE=$OMP
        # Sources: ELPA User Guide §"ELPA - Usability" (Raven A100,
        # 2024.05 benchmarks); SIESTA performance-options doc; OpenMPI
        # 5.0 mpirun(1) ("socket" is an alias for "package").
        # NCCL/RCCL is NOT compiled into our ELPA 2021.11.001 build,
        # so we sit firmly in the "multi-rank-per-GPU" regime; the
        # 1-rank-per-GPU best case only applies when NCCL is on.
        gpu_mode = (script_path.suffix.lower() == ".fdf"
                    and _fdf_requests_gpu(script_path))
        # Resolve MPI rank count.  SIESTA is fundamentally an MPI
        # code; even single-host execution is launched via mpirun.
        # When the user leaves mpi_np blank we default to ALL physical
        # cores -- that matches user expectation ("the wrapper should
        # use MPI") instead of silently emitting a bare ``siesta``
        # invocation that ignores all but one core.
        #
        # Clamp: mpi_np > n_atoms is mathematically impossible to
        # serve without trailing-rank crashes (propor IMAX=0) --
        # SIESTA's per-atom distribution leaves the last (mpi_np -
        # ceil(n_atoms / BlockSize)) ranks empty regardless of
        # BlockSize choice.  When n_atoms is known (auto-parsed from
        # the .fdf by write_run_wrapper) we clamp the AUTO path; the
        # USER-SET path is honoured verbatim (sovereign override) but
        # tagged with a runtime warning so the user sees what's about
        # to crash.
        from .runtime_info import physical_core_count
        phys = physical_core_count()
        clamp_note = ""
        if mpi_np is None or int(mpi_np) < 1:
            raw = max(1, phys)
            if n_atoms is not None and raw > int(n_atoms):
                resolved_mpi = max(1, int(n_atoms))
                mpi_source = (
                    f"auto: physical_cores ({phys}) clamped to "
                    f"n_atoms ({n_atoms}) -- mpi_np > n_atoms would "
                    f"abort SIESTA at propor IMAX=0"
                )
                clamp_note = (
                    f"# auto-mpi clamped from {raw} (physical cores) "
                    f"to {resolved_mpi} (n_atoms) so trailing ranks "
                    f"aren't empty\n"
                )
            else:
                resolved_mpi = raw
                mpi_source = f"auto: physical_cores ({phys})"
        else:
            resolved_mpi = int(mpi_np)
            if n_atoms is not None and resolved_mpi > int(n_atoms):
                mpi_source = (
                    f"user-set; WARNING mpi_np ({resolved_mpi}) > "
                    f"n_atoms ({n_atoms}) -- propor IMAX=0 expected"
                )
                clamp_note = (
                    f"# WARNING: user-set mpi_np={resolved_mpi} > "
                    f"n_atoms={n_atoms}; SIESTA will abort at propor "
                    f"IMAX=0 regardless of BlockSize.  Lower mpi_np "
                    f"to <= {n_atoms} to fix.\n"
                )
            else:
                mpi_source = "user-set"

        # OMP threads.  SIESTA mainline is mostly NOT OMP-aware;
        # pure MPI + OMP=1 is the standard SIESTA recipe.  User can
        # explicitly request hybrid by setting omp_threads > 1 (only
        # meaningful with an OMP-compiled SIESTA build).
        if omp_threads is None:
            resolved_omp = 1
            omp_source   = "default; SIESTA isn't reliably OMP-aware"
        else:
            resolved_omp = max(1, int(omp_threads))
            omp_source   = "user-set"

        # NOTE: the actual launch command is computed at RUN time by
        # the probe block below, NOT here -- the wrapper picks
        # ``mpirun -np $_mpi_np siesta`` vs bare ``siesta`` based on
        # what ``siesta --version`` reports for the currently-installed
        # binary AND the runtime $_mpi_np value (from -np / MB_NP /
        # generation-time default).  ``inner`` is the post-probe
        # shell expression; the launch_block at the bottom of this
        # function wraps it in ``set +e`` + propor-detection.  The
        # ``description`` string here is for the wrapper file header
        # only -- the user's actual -np at run time may differ.
        inner = f"$_launch_cmd {script_name} > $_out_file"
        description = f"SIESTA run, default -np {resolved_mpi}"

        # ---- Argument-parsing prelude (SIESTA only) ----
        # The wrapper accepts ``-np N`` (or env var ``MB_NP=N``) so
        # users can experiment with MPI rank counts WITHOUT
        # regenerating.  Background: SIESTA can crash with
        # ``propor: ERROR: IMAX = 0`` at startup for certain
        # mpi_np / molecule combinations.  The crash is data-
        # dependent on the ProcessorY x ProcessorX grid SIESTA auto-
        # picks for that rank count; predicting it from rank count
        # alone is not robust.  Allowing a runtime override means
        # the user can try ``./run.sh -np 8`` after a crash with
        # ``-np 15`` without regenerating the .fdf or the wrapper.
        # The post-run diagnostic at the bottom of this wrapper
        # catches the crash and prints retry suggestions.
        # Two-stage argument parsing:
        #
        #   1. Shared --continue / --force consumption (engine-
        #      agnostic; strips those flags and leaves the rest in $@).
        #   2. SIESTA-specific -np / -h.
        #
        # ORDER matters: --continue/--force are recognised first so
        # callers can combine them with -np in any order
        # (``--continue -np 8`` and ``-np 8 --continue`` both work).
        # GPU-mode default expressions for ``_mpi_np_default`` /
        # ``_omp_threads_default``.  Precedence in GPU mode:
        #
        #   1. User explicitly set ``mpi_np`` / ``omp_threads`` on the
        #      form (the values arrive in this function as non-None
        #      kwargs) -- bake them into the wrapper as literal ints.
        #   2. User left them auto (kwargs are None) AND gpu_mode is
        #      on -- defer to the runtime-probed GPU policy via
        #      ``$_gpu_mpi_np_default`` / ``$_omp_default`` (set by
        #      _gpu_runtime_defaults_block injected earlier in
        #      env_prefix).
        #   3. CPU mode -- bake the resolved integer (existing path).
        #
        # The pre-2026-06-15 shape skipped step 1 and always used the
        # bash shell var in GPU mode, silently dropping the user's
        # form choices.
        user_set_mpi = mpi_np is not None
        user_set_omp = omp_threads is not None
        if gpu_mode and not user_set_mpi:
            _mpi_np_default_expr = "$_gpu_mpi_np_default"
        else:
            _mpi_np_default_expr = str(resolved_mpi)
        if gpu_mode and not user_set_omp:
            _omp_threads_default_expr = "$_omp_default"
        else:
            _omp_threads_default_expr = str(resolved_omp)
        siesta_args_block = (
            _continue_force_args_parser("SIESTA wrapper")
            + f"# --- SIESTA-specific argument parsing -----------\n"
            f"# Override the generation-time defaults with: ``-np N`` /\n"
            f"# ``-omp N`` flags, or ``MB_NP`` / ``OMP_NUM_THREADS`` env\n"
            f"# vars.  Useful for retrying after a propor crash (see the\n"
            f"# diagnostic at the bottom of this wrapper) or for bench\n"
            f"# sweeps WITHOUT regenerating the .fdf / wrapper.\n"
            f"_mpi_np_default={_mpi_np_default_expr}\n"
            f"_omp_threads_default={_omp_threads_default_expr}\n"
            f'_mpi_np="${{MB_NP:-$_mpi_np_default}}"\n'
            # OMP precedence: -omp flag > OMP_NUM_THREADS env > policy
            # default.  Honoring a user-set OMP_NUM_THREADS matches the
            # standard OMP-toolchain convention; the prior wrapper
            # unconditionally clobbered it, which surprised users
            # benching with ``OMP_NUM_THREADS=8 ./run.sh``.
            f'_omp_threads="${{OMP_NUM_THREADS:-$_omp_threads_default}}"\n'
            f'while [ $# -gt 0 ]; do\n'
            f'    case "$1" in\n'
            f"        -np|--np)\n"
            f'            if [ $# -lt 2 ]; then\n'
            f'                echo "ERROR: -np requires a value" >&2\n'
            f"                exit 1\n"
            f"            fi\n"
            f'            _mpi_np="$2"; shift 2 ;;\n'
            f"        -omp|--omp|-t|--threads)\n"
            f'            if [ $# -lt 2 ]; then\n'
            f'                echo "ERROR: -omp requires a value" >&2\n'
            f"                exit 1\n"
            f"            fi\n"
            f'            _omp_threads="$2"; shift 2 ;;\n'
            # MPS toggle.  Default state is decided in the GPU runtime
            # defaults block based on (a) ``nvidia-cuda-mps-control``
            # binary presence and (b) the MOLBUILDER_USE_MPS env var.
            # These flags ALWAYS win.  Single-rank runs auto-disable
            # below (MPS has overhead with no concurrency benefit when
            # only one process touches the GPU).
            + (
                f"        --mps)\n"
                f'            _use_mps_default=1; shift ;;\n'
                f"        --no-mps)\n"
                f'            _use_mps_default=0; shift ;;\n'
                if gpu_mode else ""
            ) +
            f"        -h|--help)\n"
            f'            cat <<USAGE\n'
            f'Usage: bash $(basename "$0") [--continue|-c] [--force|-f] [--cold] '
            f"[-np N] [-omp N] [-h]\n"
            f"\n"
            f"  --continue, -c   resume from prior run.  Scans existing\n"
            f"                   -runN.out files and writes -run(N+1).\n"
            f"                   SIESTA reads .DM/.CG/.XV automatically\n"
            f"                   when present (generator emits the\n"
            f"                   ``DM.UseSaveDM`` / ``MD.UseSaveCG`` /\n"
            f"                   ``MD.UseSaveXV`` flags by default).\n"
            f"  --force, -f      start over from -run0 even if prior\n"
            f"                   runs exist.  Old files are NOT deleted;\n"
            f"                   the existing -run0.out is overwritten.\n"
            f"                   Prior .DM/.CG/.XV warm-start files STAY\n"
            f"                   on disk -- SIESTA will still load them.\n"
            f"  --cold,\n"
            f"  --from-scratch   move .DM/.CG/.XV/.LWF/.ZM warm-start\n"
            f"                   files into a timestamped backup dir\n"
            f"                   BEFORE running.  Use when a prior run\n"
            f"                   was bad (e.g. wrong constraints) and its\n"
            f"                   restart files would corrupt this run.\n"
            f"                   Combine with -f to also restart the\n"
            f"                   run-index sequence at -run0.\n"
            f"  -np N            override MPI rank count.  Default at\n"
            f"                   generation time was $_mpi_np_default.\n"
            f"  -omp N,          override OpenMP threads per MPI rank.\n"
            f"  -t N, --threads  Aliased.  Default was $_omp_threads_default.\n"
            f"                   (For SIESTA: pure MPI w/ OMP=1 is the\n"
            f"                   typical CPU recipe; GPU mode auto-picks\n"
            f"                   half-the-cores-per-package, even.)\n"
            f"  -h               this help.\n"
            f"\n"
            f"Environment variables:\n"
            f"  MB_NP=N            same as -np N (useful for SLURM/PBS:\n"
            f"                     ``export MB_NP=\\$SLURM_NTASKS``).\n"
            f"  OMP_NUM_THREADS=N  same as -omp N (standard OMP toolchain\n"
            f"                     convention; honored if set in env).\n"
            f"  MOLBUILDER_MPI_NP=N\n"
            f"  MOLBUILDER_OMP_NUM_THREADS=N\n"
            f"                     GPU-mode policy overrides applied\n"
            f"                     BEFORE -np / -omp.  Useful when a\n"
            f"                     workstation has unusual topology.\n"
            f"\n"
            f"On 'propor: ERROR: IMAX = 0' crashes at startup, retry\n"
            f"with a smaller -np.  See the diagnostic the wrapper prints\n"
            f"on failure for specific suggestions.\n"
            f"USAGE\n"
            f"            exit 0 ;;\n"
            f"        *)\n"
            f'            echo "ERROR: unknown argument: $1 (use -h)" >&2\n'
            f"            exit 1 ;;\n"
            f"    esac\n"
            f"done\n"
            f'if ! printf %s "$_mpi_np" | grep -qE \'^[1-9][0-9]*$\'; then\n'
            f'    echo "ERROR: -np must be a positive integer; got: '
            f'\'$_mpi_np\'" >&2\n'
            f"    exit 1\n"
            f"fi\n"
            f'if ! printf %s "$_omp_threads" | grep -qE \'^[1-9][0-9]*$\'; then\n'
            f'    echo "ERROR: -omp must be a positive integer; got: '
            f'\'$_omp_threads\'" >&2\n'
            f"    exit 1\n"
            f"fi\n"
            f"\n"
            + _run_index_resolver(basename)
            + _cold_restart_aside_block(basename, engine="siesta")
            + _runtime_status_block(basename, engine="siesta",
                                     script_name=script_name)
        )

        env_prefix = (
            (_gpu_runtime_defaults_block(n_atoms) if gpu_mode else "")
            + siesta_args_block
            + f"# MPI rank count: $_mpi_np (default: $_mpi_np_default, "
            f"source: {mpi_source})\n"
            f"{clamp_note}"
            f"# Thread / BLAS pinning.\n"
            f"#   * OMP_NUM_THREADS ({omp_source}): SIESTA mainline is\n"
            f"#     mostly not OMP-aware, so pure MPI with OMP=1 is the\n"
            f"#     standard recipe.  Bump only with an OMP-compiled\n"
            f"#     SIESTA build (hybrid MPI+OMP).  In GPU mode the\n"
            f"#     default is the hardware-derived policy value (see\n"
            f"#     the GPU-mode block above); override with -omp N or\n"
            f"#     by exporting OMP_NUM_THREADS before invoking.\n"
            f"#   * BLAS pinned to 1 per rank so OMP * BLAS doesn't\n"
            f"#     oversubscribe.\n"
            f"export OMP_NUM_THREADS=$_omp_threads\n"
            f"export MKL_NUM_THREADS=1\n"
            + (
                # Hybrid MPI+OMP needs the OMP runtime told to bind --
                # ``mpirun --bind-to core`` only binds the rank
                # (cpuset), not the threads inside it.  Without these
                # two env vars SIESTA prints "OpenMP NOT bound (please
                # bind threads!)" and the OMP runtime is free to
                # migrate threads across cores, causing cache thrash
                # + cross-package traffic that defeats the binding
                # we set on mpirun.  ``close`` keeps threads near the
                # rank's master; ``cores`` says "one place per core".
                "export OMP_PROC_BIND=close\n"
                "export OMP_PLACES=cores\n"
                if gpu_mode else ""
            )
            + f""
            f"export OPENBLAS_NUM_THREADS=1\n"
            # MPS setup: enable Hyper-Q GPU sharing when (a) the binary
            # is on PATH, (b) the user hasn't opted out, and (c) we'll
            # actually have multiple ranks (single-rank MPS is pure
            # overhead).  Per-job pipe / log directories so concurrent
            # molbuilder runs don't trample each other's MPS daemon.
            # Trap on EXIT cleans up after siesta returns.
            + (
                'if [ "$_use_mps_default" = "1" ] '
                '&& [ "$_mpi_np" -ge 2 ]; then\n'
                '    export CUDA_MPS_PIPE_DIRECTORY="/tmp/mb-mps-$$"\n'
                '    export CUDA_MPS_LOG_DIRECTORY="/tmp/mb-mps-$$-log"\n'
                '    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" '
                '"$CUDA_MPS_LOG_DIRECTORY" 2>/dev/null\n'
                '    # Start MPS only if the control socket for THIS\n'
                '    # pipe dir is not already present (a global daemon\n'
                '    # started outside this run uses a different dir,\n'
                '    # so this check is independent).\n'
                '    if [ ! -S "$CUDA_MPS_PIPE_DIRECTORY/control" ]; then\n'
                '        nvidia-cuda-mps-control -d 2>/dev/null\n'
                '        # Daemon readiness signal: the control UNIX\n'
                '        # SOCKET file appears in the pipe directory.\n'
                '        # 2026-06-16 audit fix: the prior probe polled\n'
                '        # ``echo get_server_list | nvidia-cuda-mps-control\n'
                '        # | grep -q .`` -- BUT MPS servers are spawned\n'
                '        # by the daemon only when a CLIENT FIRST\n'
                '        # CONNECTS.  Pre-launch, ``get_server_list``\n'
                '        # returns an empty string regardless of daemon\n'
                '        # health, so the loop always timed out at 5 s\n'
                '        # and falsely reported "daemon failed to bind"\n'
                '        # on perfectly healthy hosts.  The control\n'
                '        # socket appears as soon as the daemon binds\n'
                '        # (typically <100 ms) -- that is the correct\n'
                '        # readiness signal.\n'
                '        _mps_wait=0\n'
                '        while [ ! -S "$CUDA_MPS_PIPE_DIRECTORY/control" ]; do\n'
                '            sleep 0.1\n'
                '            _mps_wait=$((_mps_wait + 1))\n'
                '            if [ "$_mps_wait" -gt 50 ]; then\n'
                '                echo "molbuilder: MPS control socket '
                'did not appear within 5 s ('
                '$CUDA_MPS_PIPE_DIRECTORY/control); falling back '
                'to no-MPS." >&2\n'
                '                _use_mps_default=0\n'
                '                break\n'
                '            fi\n'
                '        done\n'
                '    fi\n'
                '    # Cleanup: stop daemon + remove per-job dirs on\n'
                '    # ANY exit (success, error, signal).  Trap is\n'
                '    # armed regardless of whether the daemon actually\n'
                '    # bound -- a partially-started daemon still wants\n'
                '    # the directories cleaned.\n'
                '    trap \'echo quit | nvidia-cuda-mps-control '
                '>/dev/null 2>&1; '
                'rm -rf "$CUDA_MPS_PIPE_DIRECTORY" '
                '"$CUDA_MPS_LOG_DIRECTORY"\' EXIT\n'
                # Gate the "MPS enabled" message on the daemon-bind
                # result.  Before this gate the readiness-poll fallback
                # at the loop above would print "MPS daemon failed to
                # bind ... falling back to no-MPS" AND THEN the line
                # below would print "MPS enabled (pipe=...)" -- two
                # contradictory messages, with the run continuing
                # without MPS but the banner claiming otherwise.
                '    if [ "$_use_mps_default" = "1" ]; then\n'
                '        echo "molbuilder: MPS enabled '
                '(pipe=$CUDA_MPS_PIPE_DIRECTORY)" >&2\n'
                '    fi\n'
                'else\n'
                '    if [ "$_use_mps_default" = "1" ] '
                '&& [ "$_mpi_np" -lt 2 ]; then\n'
                '        echo "molbuilder: MPS auto-disabled '
                '(single-rank run; MPS adds overhead with no '
                'concurrency benefit)" >&2\n'
                '    fi\n'
                'fi\n'
                if gpu_mode else ""
            )
        )
        if max_memory_mb is not None and int(max_memory_mb) > 0:
            kb = int(max_memory_mb) * 1024
            env_prefix += (
                f"# Memory cap (cfg.max_memory_mb): {max_memory_mb} MB\n"
                f"ulimit -v {kb} || true  # soft cap; ignored if shell can't set it\n"
            )
        env_prefix += "\n"

        # Runtime SIESTA build probe + launcher selection.
        #
        # ``siesta --version`` (5.x +) self-reports the parallelisation
        # the binary was compiled with.  Example for a typical conda-
        # forge build:
        #
        #   Version         : 5.4.2
        #   Parallelisations: MPI
        #
        # We parse the ``Parallelisations:`` line and pick the launcher
        # accordingly:
        #
        #   MPI present      ->  mpirun -np <N> siesta   (always)
        #   OMP present      ->  bare siesta             (OMP env vars take effect)
        #   both             ->  mpirun -np <N> siesta   (hybrid)
        #   probe failed     ->  mpirun -np <N> siesta   (safe default for
        #                                                 MPI-compiled binaries)
        #   serial build     ->  bare siesta
        #
        # The probe runs ONCE per wrapper invocation and prints what
        # it found before exec, so the user sees the actual build
        # capability + the launcher choice in the log.  This adapts
        # automatically if you rebuild SIESTA with different flags --
        # no need to regenerate the wrapper.
        env_prefix += (
            f"# --- Probe SIESTA build at runtime ---\n"
            f'_siesta_bin_path="$(command -v siesta || echo \"\")"\n'
            f'if [ -z "$_siesta_bin_path" ]; then\n'
            f"    echo \"ERROR: 'siesta' not on PATH after activating "
            f"'{target_env}'.  Is SIESTA installed in this env?\" >&2\n"
            f"    exit 1\n"
            f"fi\n"
            f'_siesta_version_out="$(siesta --version 2>/dev/null || true)"\n'
            f'_siesta_ver="$(printf %s \"$_siesta_version_out\" '
            f"| awk -F': *' '/^Version/ {{print $2; exit}}')\"\n"
            f'_siesta_par="$(printf %s \"$_siesta_version_out\" '
            f"| awk -F': *' '/^Parallelisations/ {{print $2; exit}}')\"\n"
            f"# Decide launcher from probe.  Default to mpirun (safe\n"
            f"# for any MPI-compiled binary) when the probe can't\n"
            f"# tell us anything.\n"
            f'_has_mpi=0; _has_omp=0\n'
            f'# Word-boundary match.  ``*MPI*`` alone would falsely\n'
            f'# catch ``NoMPI`` / ``pre-MPI`` / ``nompi``.  Strategy:\n'
            f'# focus on CONTENT, not formatting -- normalise ANY\n'
            f'# whitespace (space, tab, vertical-tab, ...) AND the\n'
            f'# comma/semicolon list-separators to single spaces, then\n'
            f'# space-anchor the token match.  ``tr "[:space:]" " "``\n'
            f'# rewrites every POSIX whitespace char; ``tr ",;" "  "``\n'
            f'# absorbs the list separators; ``tr -s " "`` collapses\n'
            f'# runs.  Robust against any SIESTA build that prints the\n'
            f'# Parallelisations line with tabs / extra spaces /\n'
            f'# mixed list separators -- the value semantics carry.\n'
            f'_par_norm=" $(echo \"$_siesta_par\" '
            f'| tr "[:space:]" " " | tr ",;" "  " | tr -s " ") "\n'
            f'case "$_par_norm" in *" MPI "*) _has_mpi=1 ;; esac\n'
            f'case "$_par_norm" in *" OMP "*|*" OpenMP "*) _has_omp=1 ;; esac\n'
            # GPU mode: pin threads to cores in the same package so
            # OpenMP doesn't spill across sockets (SIESTA performance-
            # options guidance), and tell OpenMPI to place ranks one
            # per package.  On OpenMPI 5.x "package" is canonical;
            # "socket" still works as an alias.
            + (
                # PE counting hazard caught 2026-06-16 in a live
                # 212-atom Au-BDT run: the previous
                # ``ppr:K:package:PE=$_omp`` form, on Intel HT boxes,
                # allocated PE=2 *processing units* (PUs) per rank
                # mapped as HT-sibling pairs of ONE physical core.
                # Observed binding: rank 0 cpus={0,20} (core 0
                # threads), rank 1 cpus={2,22}, etc.  So 4 ranks x
                # PE=2 used only 4 physical cores (not 8), with each
                # rank's 2 OMP threads sharing one core's execution
                # units -- socket 0 idle at 20% while it should have
                # been driving 80%.
                #
                # Replace with the canonical "N physical cores per
                # rank, packed onto packages" form:
                #
                #   --map-by package:PE=$_omp_threads
                #     map ranks across packages, PE counts physical
                #     CORES (not PUs) by default in OpenMPI 5 without
                #     ``--use-hwthread-cpus``.  When the cpuset
                #     restricts to one package (numactl wrap),
                #     OpenMPI packs all ranks onto that single
                #     visible package -- correct.
                #   --bind-to core
                #     bind each rank to its PE cores (one cpuset
                #     per rank covering all its cores; OS scheduler
                #     places OMP threads on those cores).
                #
                # NB: NO ``--rank-by core``.  OpenMPI 4.x rejects it
                # ("Valid directives: slot:node:fill:span"); rank
                # ordering defaults to map order which is already
                # deterministic for our use.  Caught 2026-06-16 by
                # bench rc=213 on the user's box.
                '_mpirun_bind="--bind-to core --map-by '
                'package:PE=$_omp_threads"\n'
                if gpu_mode else
                f'_mpirun_bind=""\n'
            )
            # ``_numa_wrap_gpu`` is set by _gpu_runtime_defaults_block
            # in GPU mode; default to empty here so CPU-mode wrappers
            # (which never inject that block) still see a defined var
            # in the launch_cmd interpolation below.  This is also the
            # safe-default branch when GPU mode runs on single-socket
            # boxes / boxes where numactl isn't installed -- the block
            # leaves the var empty in those cases.
            + '_numa_wrap_gpu="${_numa_wrap_gpu:-}"\n'
            + f'if [ "$_has_mpi" = 1 ]; then\n'
            f'    _launch_cmd="$_numa_wrap_gpu mpirun -np $_mpi_np $_mpirun_bind siesta"\n'
            f'    if [ "$_has_omp" = 1 ]; then\n'
            f'        _launch_note="hybrid MPI+OMP ($_mpi_np ranks x $_omp_threads OMP threads)"\n'
            f'    else\n'
            f'        _launch_note="pure MPI ($_mpi_np ranks; OMP setting irrelevant to this binary)"\n'
            f'    fi\n'
            f'elif [ "$_has_omp" = 1 ]; then\n'
            f'    _launch_cmd="siesta"\n'
            f'    _launch_note="OMP-only build ($_omp_threads threads)"\n'
            f'elif [ -z "$_siesta_par" ]; then\n'
            f'    _launch_cmd="$_numa_wrap_gpu mpirun -np $_mpi_np $_mpirun_bind siesta"\n'
            f'    _launch_note="MPI fallback (probe inconclusive; safe default for MPI-compiled SIESTA)"\n'
            f'else\n'
            f'    _launch_cmd="siesta"\n'
            f'    _launch_note="serial build (no parallelisation compiled in)"\n'
            f"fi\n"
            f"\n"
        )

        # Human-readable banner printed at run time so the user sees
        # the rank count / threading / cwd / command + BUILD probe
        # results before SIESTA spends 30 seconds reading the .fdf.
        env_prefix += (
            f'echo "===== molbuilder SIESTA run-wrapper ====="\n'
            f'echo "  Date          : $(date -Iseconds)"\n'
            f'echo "  Host          : $(hostname)"\n'
            f'echo "  Cwd           : $(pwd)"\n'
            f'echo "  Conda env     : ${{CONDA_DEFAULT_ENV:-?}}"\n'
            f'echo "  SIESTA binary : $_siesta_bin_path"\n'
            f'echo "  SIESTA version: ${{_siesta_ver:-unknown}}"\n'
            f'echo "  Build paral.  : ${{_siesta_par:-unknown}}"\n'
            f'echo "  Launch mode   : $_launch_note"\n'
            f'echo "  Threading     : OMP_NUM_THREADS=$_omp_threads, '
            f'OPENBLAS=1, MKL=1"\n'
            # GPU mode: print a brief monitoring hint so the user has
            # nvidia-smi commands at hand when they start the run.
            + ((
                # IMPORTANT: keep the command on its own line so the
                # user can copy-paste it directly into a shell.  An
                # earlier banner shape put ``(sm%, mem%, ...)`` after
                # the command on the same line and bash interpreted
                # the ``(`` as a subshell open + ``%`` as a format op
                # when the user pasted it -- "syntax error near
                # unexpected token \`(\`".  Annotation goes on the
                # NEXT line, prefixed with ``# `` so even if it's
                # included accidentally in a paste the shell treats
                # it as a comment.
                'echo "  GPU monitor   : nvidia-smi dmon -s pucvmet -d 1"\n'
                'echo "                # columns: sm%, mem%, clk, temp, power"\n'
                'echo "                # if sm% bounces 0->100->0 across ranks, MPS may help"\n'
                'echo "                # (this ELPA build has no NCCL; multi-rank-per-GPU benefits from MPS)"\n'
            ) if gpu_mode else "")
            # ---- Mode + constraints (post-cold, post-run-index) ----
            # Surfaces the silent-warm-restart class explicitly so the
            # user can see whether the engine is starting clean,
            # resuming from prior state, or being asked to honor
            # frozen-atom constraints.  Both lines are read from the
            # on-disk script + restart files at runtime so a manual
            # .fdf edit shows the EDITED state, not the generation-
            # time snapshot (the "what you see is what runs" rule).
            + f'echo "  Mode          : $_mode"\n'
            + f'echo "  Constraints   : $_constraints"\n'
            + f'echo "  Command       : $_launch_cmd {script_name} > $_out_file"\n'
            + f'echo "  Stdout        : $_out_file (live; tail -f to follow)"\n'
            f'echo "========================================="\n'
            f"\n"
        )
    else:                                          # pyscf
        inner = f"python {script_name} > $_out_file 2>&1"
        description = "PySCF run"
        # PySCF: the inline ``runtime_info`` block in the emitted
        # .py sets OMP_NUM_THREADS / OPENBLAS_NUM_THREADS = 1 via
        # ``os.environ.setdefault`` BEFORE numpy import.  We don't
        # set them in the wrapper too -- doing so would override
        # the env-respect (the script honors a pre-export) AND
        # mask the in-script auto-detect that picks physical cores.

        # Argument parsing: PySCF gets --continue / --force +
        # a -h.  No engine-specific flags (PySCF doesn't have an
        # MPI rank knob; threading is auto-detected in the script).
        pyscf_args_block = (
            _continue_force_args_parser("PySCF wrapper")
            + f"# --- PySCF wrapper argument parsing -------------\n"
            f'while [ $# -gt 0 ]; do\n'
            f'    case "$1" in\n'
            f"        -h|--help)\n"
            f'            cat <<USAGE\n'
            f'Usage: bash $(basename "$0") [--continue|-c] [--force|-f] [--cold] [-h]\n'
            f"\n"
            f"  --continue, -c   resume from prior run.  Scans existing\n"
            f"                   -runN.pyscf.log files and writes\n"
            f"                   -run(N+1).pyscf.log.  PySCF loads the\n"
            f"                   SCF density matrix from ``<JOB>.chk``\n"
            f"                   automatically when present (the\n"
            f"                   generator emits ``mf.chkfile`` by\n"
            f"                   default and the chkfile-init-guess\n"
            f"                   shim auto-loads it on continuation).\n"
            f"  --force, -f      start over from -run0 even if prior\n"
            f"                   runs exist.  Old files are NOT deleted;\n"
            f"                   the existing -run0.pyscf.log is\n"
            f"                   overwritten.  Prior ``.chk`` warm-start\n"
            f"                   files STAY on disk -- PySCF still loads.\n"
            f"  --cold,\n"
            f"  --from-scratch   move ``.chk`` warm-start files into a\n"
            f"                   timestamped backup dir BEFORE running.\n"
            f"                   Use when the prior run was bad and its\n"
            f"                   chkfile would corrupt this run.\n"
            f"  -h               this help.\n"
            f"USAGE\n"
            f"            exit 0 ;;\n"
            f"        *)\n"
            f'            echo "ERROR: unknown argument: $1 (use -h)" >&2\n'
            f"            exit 1 ;;\n"
            f"    esac\n"
            f"done\n"
            f"\n"
            # PySCF uses ``.pyscf.log`` instead of ``.out`` so the
            # Results-tab inspector dispatcher can tell PySCF output
            # apart from SIESTA's (which keeps ``.out``).  Per
            # docs/tabs/architecture.md § 7 (Phase C, 2026-06-07).
            + _run_index_resolver(basename, ext=".pyscf.log")
            + _cold_restart_aside_block(basename, engine="pyscf")
            + _runtime_status_block(basename, engine="pyscf",
                                     script_name=script_name)
        )

        # Same human-readable banner pattern as SIESTA -- the script
        # itself logs its own runtime info but the wrapper covers
        # the "did it even start" window before Python imports.
        env_prefix = (
            pyscf_args_block
            + f'echo "===== molbuilder PySCF run-wrapper ====="\n'
            f'echo "  Date        : $(date -Iseconds)"\n'
            f'echo "  Host        : $(hostname)"\n'
            f'echo "  Cwd         : $(pwd)"\n'
            f'echo "  Conda       : ${{CONDA_DEFAULT_ENV:-?}}"\n'
            # ---- Mode + constraints (mirrors SIESTA; see siesta
            # banner above for the rationale). ----
            f'echo "  Mode        : $_mode"\n'
            f'echo "  Constraints : $_constraints"\n'
            f'echo "  Command     : python {script_name} > $_out_file"\n'
            f'echo "  Stdout      : $_out_file"\n'
            f'echo "  Logs        : see <basename>.molwatch.log (script writes its own)"\n'
            f'echo "========================================"\n'
            f"\n"
        )

    # Build a self-contained activation block + per-run log file.
    # The wrapper must run as a fully standalone job script under
    # SLURM/PBS, with NO requirement that the caller has conda on
    # PATH or has activated anything.
    #
    # Per docs/config.md (the unified configuration design), the
    # wrapper is composed of THREE env-handling layers in this order:
    #
    #   1. Baked-in ``preactivate`` from script_generation.preactivate
    #      in the effective config (server-wide ∪ project, both files
    #      under docs/config.md § 2).  Each block emitted with a
    #      sentinel comment naming its source scope so a user reading
    #      the wrapper sees exactly where each line came from.
    #   2. ``MOLBUILDER_PREACTIVATE_CMDS`` env var (runtime escape).
    #   3. 6-path conda detection -- ONLY when
    #      script_generation.autodetect_conda is true (opt-in;
    #      defaults to false so a typo in ``preactivate`` fails loud
    #      instead of being silently rescued by the fallback).
    #
    # Then: ``conda activate <target_env>``.  Failure is logged with
    # the available-envs list and exits 1 (no silent fall-through).
    #
    # ``project_dir`` for the config read is ``script_path.parent``,
    # matching the convention every other subsystem already uses for
    # project context.
    from . import runtime_config as _rc
    _project_dir = script_path.parent if script_path.parent.exists() else None
    _sg = _rc.get_script_generation(project_dir=_project_dir)
    _preactivate_text = _sg["preactivate"]
    _preactivate_scopes = _sg.get("_preactivate_scopes", [])
    # autodetect_override is the explicit test/programmatic override;
    # otherwise honour the effective config (default false per
    # docs/config.md § 3.6 -- fail-loud-by-default).
    _autodetect = (autodetect_override
                    if autodetect_override is not None
                    else bool(_sg["autodetect_conda"]))

    _baked_conda_base = _detect_generate_time_conda_base() or ""
    # Quote for safe bash embedding -- conda base paths usually have
    # no spaces / quotes, but we shouldn't assume.
    _baked_q = _baked_conda_base.replace("'", "'\\''")

    # Render the baked-in preactivate block with per-scope sentinels.
    # ``_preactivate_scopes`` carries scope labels in concatenation
    # order; split _preactivate_text back into per-scope chunks for
    # the sentinel-comment header.  Each chunk separated by "\n\n"
    # (see get_script_generation in runtime_config.py).
    #
    # PER-COMMAND TRACING: bash xtrace (``set -x``) is enabled around
    # the user's preactivate so each command + its expansion lands in
    # the log file via the stderr tee.  Disabled immediately after so
    # the verbose tracing doesn't leak into the rest of the wrapper.
    # If any command in the block returns non-zero, ``set -euo pipefail``
    # (already set at the wrapper top) aborts the script with the
    # offending stderr captured in the log -- the user can find which
    # exact preactivate line failed without re-running.
    _preactivate_populated = bool(_preactivate_text)
    if _preactivate_populated:
        _scope_labels = {
            "server":  "SERVER-WIDE PREACTIVATION (from molbuilder.json)",
            "project": "PROJECT PREACTIVATION (from .molbuilder.json)",
        }
        _chunks = _preactivate_text.split("\n\n")
        _rendered_chunks = []
        for label, chunk in zip(_preactivate_scopes, _chunks):
            header = _scope_labels.get(label, label.upper())
            _rendered_chunks.append(
                f"# === {header} ===\n{chunk}\n"
            )
        _preactivate_block = (
            "# --- Baked-in preactivate block (see "
            "docs/config.md § 3.6) ---\n"
            "_log STAGE \"running baked-in preactivate block\"\n"
            "# Trace each preactivate command via bash xtrace so the\n"
            "# log captures the exact line that ran (and the exact one\n"
            "# that failed, if any).  Restore quiet mode after.\n"
            "set -x\n"
            + "\n".join(_rendered_chunks)
            + "{ set +x; } 2>/dev/null\n"
            "\n"
        )
    else:
        _preactivate_block = (
            "# --- Baked-in preactivate block (empty -- see "
            "docs/config.md § 3.6 to populate) ---\n"
            "# (no script_generation.preactivate configured)\n"
            "\n"
        )
    env_activation = (
        f"# --- Per-run log file (vigorous logging of all wrapper steps) -\n"
        f"# Captured: pre-activation hooks, conda detection paths tried,\n"
        f"# module-load attempts, post-activation env state, MPI/OMP/GPU\n"
        f"# pre-launch state, the launch command itself, exit code +\n"
        f"# wall time.  One log file per invocation, timestamped so\n"
        f"# multiple runs don't trample.  Look at it FIRST when a job\n"
        f"# fails on a remote scheduler -- it has everything you need\n"
        f"# to diagnose without re-submitting.\n"
        f'_runwrap_log="{basename}.runwrap-$(date +%Y%m%d-%H%M%S).log"\n'
        f"# Tee both stdout AND stderr into the log file while keeping\n"
        f"# them on the original streams (so sbatch's stdout/stderr\n"
        f"# capture still works AND the standalone log file exists).\n"
        f'exec > >(tee -a "$_runwrap_log") 2> >(tee -a "$_runwrap_log" >&2)\n'
        f"\n"
        f"# Structured log helper -- used at every load-bearing step.\n"
        f"_log() {{\n"
        f"    # $1=tag (STAGE|INFO|WARN|ERROR), $2=message\n"
        f"    printf '[%s] [%-5s] %s\\n' \"$(date '+%H:%M:%S%z')\" \"$1\" \"$2\" >&2\n"
        f"}}\n"
        f"_log_run() {{\n"
        f"    # $1=command-to-trace, runs + logs stdout/stderr inline\n"
        f"    _log INFO \"$ $1\"\n"
        f"    eval \"$1\"\n"
        f"}}\n"
        f"\n"
        f"_log STAGE \"===== molbuilder wrapper start =====\"\n"
        f'_log INFO "timestamp:  $(date \'+%Y-%m-%d %H:%M:%S %Z\')"\n'
        f'_log INFO "hostname:   $(hostname)"\n'
        f'_log INFO "user:       ${{USER:-?}}"\n'
        f'_log INFO "cwd:        $(pwd)"\n'
        f'_log INFO "script:     $0"\n'
        f'_log INFO "argv:       $0 $*"\n'
        f'_log INFO "log file:   $_runwrap_log"\n'
        f"# Scheduler context -- only emit if the var is set.\n"
        f'for _v in SLURM_JOB_ID SLURM_NTASKS SLURM_CPUS_PER_TASK \\\n'
        f"          SLURM_JOB_NODELIST SLURM_GPUS SLURM_JOB_GPUS \\\n"
        f"          PBS_JOBID PBS_NP PBS_NODEFILE \\\n"
        f"          MOLBUILDER_PREACTIVATE_CMDS; do\n"
        f"    _v_val=\"${{!_v:-}}\"\n"
        f'    [ -n "$_v_val" ] && _log INFO "$_v=$_v_val"\n'
        f"done\n"
        f"\n"
        f"{_preactivate_block}"
        f"# --- Runtime pre-activation hook ----------------------------\n"
        f"# The env var below is the runtime escape hatch for the\n"
        f"# baked-in preactivate block above.  Use it for one-off\n"
        f"# overrides (a different module name on a fresh cluster, a\n"
        f"# temporary debug ``set -x``).  Permanent cluster setup\n"
        f"# belongs in molbuilder.json's script_generation.preactivate\n"
        f"# (docs/config.md § 3.6) so it ships with every generated\n"
        f"# wrapper instead of relying on .bashrc state.\n"
        f'if [ -n "${{MOLBUILDER_PREACTIVATE_CMDS:-}}" ]; then\n'
        f"    _log STAGE \"running MOLBUILDER_PREACTIVATE_CMDS\"\n"
        f'    _log_run "$MOLBUILDER_PREACTIVATE_CMDS" \\\n'
        f"        || _log WARN \"preactivate cmds returned non-zero (continuing)\"\n"
        f"fi\n"
        f"\n"
        + _build_locate_and_activate_block(
              target_env, _autodetect,
              preactivate_populated=_preactivate_populated,
          ).replace(_BAKED_PLACEHOLDER, _baked_q)
        + f"\n"
        + f'_log INFO "CONDA_DEFAULT_ENV=${{CONDA_DEFAULT_ENV:-<unset>}}"\n'
        + f'_log INFO "CONDA_PREFIX=${{CONDA_PREFIX:-<unset>}}"\n'
        + f'_log INFO "which python: $(command -v python 2>/dev/null || echo \'(not on PATH)\')"\n'
        + f"\n"
    )

    # Launch + diagnostics.  For SIESTA we run the command (not
    # exec) so we can inspect the .out for ``propor: ERROR: IMAX = 0``
    # on failure and print a targeted retry hint.  Layer-on-top
    # cost: one extra bash process for the wrapper's lifetime; cheap.
    # For PySCF the original exec is preserved -- no diagnostic
    # surface there yet.
    if category == "siesta":
        launch_block = (
            f"# --- Launch SIESTA + capture exit -----------------------\n"
            f"# `set +e` lets us inspect the exit code; the diagnostic\n"
            f"# below reads the .out for ``propor: ERROR`` and prints a\n"
            f"# retry suggestion.  Then we re-exit with SIESTA's code.\n"
            f"set +e\n"
            f"$_launch_cmd {script_name} > $_out_file\n"
            f"_siesta_exit=$?\n"
            f"set -e\n"
            f"\n"
            f'if [ "$_siesta_exit" -ne 0 ]; then\n'
            f"    echo \"\"\n"
            f'    echo "===== SIESTA exited with code $_siesta_exit =====" >&2\n'
            f'    if grep -aq "propor: ERROR" "$_out_file" 2>/dev/null; then\n'
            f"        cat <<HINT >&2\n"
            f"\n"
            f"SIESTA crashed with 'propor: ERROR: IMAX = 0' during startup.\n"
            f"\n"
            f"This is a SIESTA-side issue with how it distributes work\n"
            f"across MPI ranks for this specific molecule + rank-count\n"
            f"combination.  Some mpi_np values leave some ranks empty\n"
            f"in matel_table's radial-function pipeline and the internal\n"
            f"proportionality check then fails.  It is NOT a configuration\n"
            f"bug in your .fdf -- the same .fdf works at a different -np.\n"
            f"\n"
            f"Your -np was $_mpi_np.  Retry WITHOUT regenerating:\n"
            f"\n"
            f"  bash {basename}.run.sh -np 8     # powers of 2 are usually safest\n"
            f"  bash {basename}.run.sh -np 4     # if 8 also fails\n"
            f"  bash {basename}.run.sh -np 2     # last resort\n"
            f"\n"
            f"Empirically (probed on hemeC, 81 atoms, Fe + S + organic):\n"
            f"  works: 2, 4, 6, 8, 12, 14\n"
            f"  fails: 9, 10, 11, 13, 15, 16\n"
            f"For pure-organic systems the safe range is usually wider;\n"
            f"larger heteroatom mixes are more conservative.\n"
            f"\n"
            f"Full SIESTA output: $_out_file\n"
            f"HINT\n"
            f'    elif grep -aq "ERROR\\|aborted\\|Stopping" '
            f'"$_out_file" 2>/dev/null; then\n'
            f'        echo "Other SIESTA error detected; check '
            f'$_out_file for details." >&2\n'
            f"    fi\n"
            f'    exit "$_siesta_exit"\n'
            f"fi\n"
            f"\n"
            f'echo "SIESTA completed: $_launch_cmd {script_name} -> '
            f'$_out_file"\n'
        )
    else:
        launch_block = f"exec {inner}\n"

    # Engine-specific output suffix.  SIESTA's wrapper writes
    # ``-runN.out``; PySCF's writes ``-runN.pyscf.log`` (Phase C
    # rename, 2026-06-07).  The banner below shows the suffix the
    # user will actually see so they don't go hunting for the
    # wrong filename after the first run.  BOMB-6 fix.
    _ext = ".pyscf.log" if suffix == ".py" else ".out"
    # ----- Script-contract PROVENANCE block -----
    # See docs/protocols/script-contract.md.  PROVENANCE only for the
    # wrapper -- no BENCH-MARKS (wrapper-side parameters are overridden
    # via existing env vars per the contract) and no ATOM-METADATA
    # (lives in the engine input file, not the wrapper).
    #
    # For PySCF (.py) wrappers, mpi_np is meaningless (PySCF is OMP-
    # only) and must not surface as a per-call value -- the
    # test_render_pyscf_ignores_mpi_np invariant (same wrapper text
    # regardless of mpi_np input) is the contract here.
    from . import script_emit as _sc
    _is_pyscf = suffix == ".py"
    _resolved_defaults = {
        "target_env":    target_env,
        "omp_threads": (
            "auto" if omp_threads is None else str(omp_threads)
        ),
        "max_memory_mb": (
            "n/a" if max_memory_mb is None else str(max_memory_mb)
        ),
    }
    if _is_pyscf:
        _resolved_defaults["mpi_np"] = "n/a (PySCF is OMP-only)"
    else:
        _resolved_defaults["mpi_np"] = (
            "auto" if mpi_np is None else str(mpi_np)
        )
    _provenance = _sc.emit_provenance(
        generator_version=_sc.molbuilder_git_sha(),
        generated_at=_sc.generated_at_now(),
        resolved_defaults=_resolved_defaults,
    )
    _user_custom = _sc.emit_user_custom_placeholder()
    return (
        f"#!/usr/bin/env bash\n"
        f"{_provenance}\n"
        f"#\n"
        f"# molbuilder run-wrapper -- {description}\n"
        f"# Script: {script_name}\n"
        f"# Target env: {target_env}\n"
        f"#\n"
        f"# Generated by `molbuilder run`.  Edit freely; molbuilder will\n"
        f"# not regenerate this file unless `molbuilder run` is invoked\n"
        f"# again on the same script.  Run directly:\n"
        f"#\n"
        f"#     bash {basename}.run.sh              # first run -> -run0{_ext}\n"
        f"#     bash {basename}.run.sh --continue   # resume -> -run1, -run2, ...\n"
        f"#     bash {basename}.run.sh --force      # restart from -run0 (overwrite)\n"
        f"#     bash {basename}.run.sh -np 8        # override mpi_np (SIESTA only)\n"
        f"#     MB_NP=8 bash {basename}.run.sh      # same via env var (SLURM/PBS)\n"
        f"#     nohup ./{basename}.run.sh &         # background, detached\n"
        f"#\n"
        f"# Continuation contract:\n"
        f"#  * SIESTA: ``DM.UseSaveDM`` / ``MD.UseSaveCG`` / ``MD.UseSaveXV``\n"
        f"#    are emitted by the generator by default; SIESTA auto-loads\n"
        f"#    .DM/.CG/.XV from the previous run.\n"
        f"#  * PySCF: ``mf.chkfile`` is set by default; the script's startup\n"
        f"#    shim auto-loads the prior SCF density via ``mf.init_guess =\n"
        f"#    \"chkfile\"`` when the .chk file exists.\n"
        f"#\n"
        f"# IMPORTANT: this wrapper chdirs to the script directory\n"
        f"# before launching, so ALL output artefacts (.log, .chk,\n"
        f"# trajectory XYZ, .molwatch.log, .spectra.json, ...) land\n"
        f"# right next to the script.  Do NOT add ``cd`` lines into\n"
        f"# this wrapper or modify the working directory inside the\n"
        f"# generated .py / .fdf -- the Python side resolves every\n"
        f"# output path through ``_mb_outfile()`` against the script\n"
        f"# directory, but other code (PySCF's mol.log open(),\n"
        f"# geomeTRIC's optimize prefix) you might layer on top would\n"
        f"# write into the wrong place if you chdir to elsewhere.\n"
        f"#\n"
        f"set -euo pipefail\n"
        f"cd \"$(dirname \"$0\")\"\n"
        f"\n"
        f"{env_activation}"
        f"{env_prefix}"
        f"{launch_block}"
        f"\n{_user_custom}\n"
    )


def write_run_wrapper(script_path: Path, *,
                       env: Optional[str] = None,
                       mpi_np: Optional[int] = None,
                       omp_threads: Optional[int] = None,
                       max_memory_mb: Optional[int] = None,
                       autodetect_override: Optional[bool] = None) -> Path:
    """Render + write ``<basename>.run.sh`` next to ``script_path``.

    Returns the wrapper's path.  Sets executable bit (0o755) so the
    user can ``./my-job.run.sh`` directly.  Overwrites any existing
    wrapper.

    For ``.fdf`` scripts the file is parsed for ``NumberOfAtoms`` and
    that value is threaded into ``render_run_wrapper`` so the auto-mpi
    path can clamp ``mpi_np <= n_atoms`` (the propor IMAX=0 lower
    bound).  Parse-failure is treated as "unknown" and falls back to
    the unclamped behaviour rather than refusing to render.
    """
    script_path = Path(script_path).resolve()
    if not script_path.is_file():
        raise WrapperError(f"script not found: {script_path}")
    n_atoms = None
    if script_path.suffix.lower() == ".fdf":
        n_atoms = _parse_fdf_n_atoms(script_path)
    text = render_run_wrapper(
        script_path,
        env=env, mpi_np=mpi_np,
        omp_threads=omp_threads,
        max_memory_mb=max_memory_mb,
        n_atoms=n_atoms,
        autodetect_override=autodetect_override,
    )
    # Defense-in-depth: every rendered wrapper goes through ``bash -n``
    # (parse-only, no execution) before we hand it back.  Catches the
    # whole class of "shipped malformed shell" bugs that the
    # 2026-06-20 pentanedithiol incident exposed (an awk -F char-
    # class with a broken SQ-escape produced an unterminated DQ; the
    # user only found out when they tried to run the script).  Both
    # the writer-side L2 tests and this in-line gate must agree:
    # ``bash -n`` rejects → caller never sees a broken file on disk.
    _validate_rendered_wrapper(text, script_path)
    # Use stem + ".run.sh" rather than ``.with_suffix(".run.sh")``: the
    # latter REPLACES only the last suffix, so ``job.spectra.py`` would
    # become ``job.run.sh`` and lose the "spectra" tag.  We want
    # ``job.spectra.run.sh``.
    wrapper_path = script_path.parent / (script_path.stem + ".run.sh")
    wrapper_path.write_text(text)
    wrapper_path.chmod(0o755)
    return wrapper_path


def _validate_rendered_wrapper(text: str, script_path: Path) -> None:
    """Run ``bash -n`` (parse-only) on the rendered wrapper text.
    Raises :exc:`WrapperError` if bash rejects it as malformed shell.

    Writes the text to a tempfile (in the same dir so a quirky FS
    can't surprise us) and runs ``bash -n`` against it.  No execution
    happens; bash only checks shell-syntax validity.

    Cheap: a few ms per render; the user's wait is dominated by the
    upstream form-submit roundtrip anyway.  The alternative — only
    finding out at run time — costs the user a full re-render cycle
    (or worse, a confused "why doesn't my script run?" support
    request like the 2026-06-20 PDT incident).
    """
    import subprocess
    import tempfile
    parent = script_path.parent
    fd, tmp = tempfile.mkstemp(
        prefix=".runwrap-syntax-check-",
        suffix=".sh",
        dir=str(parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        cp = subprocess.run(
            ["bash", "-n", tmp],
            capture_output=True, text=True, timeout=15,
        )
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    if cp.returncode != 0:
        raise WrapperError(
            f"generator produced malformed shell for {script_path.name}; "
            f"bash -n rejected the rendered wrapper.  This is a "
            f"molbuilder bug -- the wrapper template emitted invalid "
            f"syntax.  bash stderr:\n{cp.stderr}"
        )


__all__ = [
    "WrapperError",
    "render_run_wrapper",
    "write_run_wrapper",
]
