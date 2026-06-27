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
from typing import Any, List, Mapping, Optional, Tuple

from .diagnostics import EXTENSION_TO_CATEGORY, get_capabilities

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

      ``_continue`` (0/1)  -- ``--continue`` was passed; in addition
        to the default index advance it asks the engine to warm-start
        from prior state (.DM/.CG/.XV).
      ``_force``    (0/1)  -- ``--force`` was passed; restart the
        sequence at -run0 (overwriting it) instead of advancing.

    DEFAULT (2026-06-26): when a prior ``-runN{ext}`` exists the
    resolver **auto-advances** to ``max(N)+1`` -- re-running NEVER
    errors and NEVER overwrites a prior result.  The old behaviour
    (refuse-with-exit-1 unless ``--continue``/``--force``) was the
    single biggest papercut in the iterative resubmit loop.

    The resolver is shared by SIESTA + PySCF wrappers so the run-index
    semantics are identical across engines; only the suffix differs.
    ``basename`` is the script stem (e.g. ``siesta-hemeC-gas-stage3``)
    baked in at generation time -- the bash itself doesn't try to
    derive it.
    """
    return (
        f"# --- Run index resolution ------------------------------\n"
        f"# Outputs are ``{basename}-runN{ext}``.  First run produces\n"
        f"# -run0; any later run AUTO-ADVANCES to max(N)+1 by default so\n"
        f"# re-running never errors and never clobbers a prior result.\n"
        f"# --force restarts the sequence at -run0; --continue adds an\n"
        f"# engine warm-start on top of the (default) index advance.\n"
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
        f'if [ "$_force" = "1" ]; then\n'
        f"    # --force: explicitly restart the sequence at -run0 (SIESTA's\n"
        f"    # redirect clobbers the existing -run0).\n"
        f"    _run_n=0\n"
        f'elif [ "$_existing_max" -ge 0 ]; then\n'
        f"    # DEFAULT auto-continue (2026-06-26): a prior -runN exists, so\n"
        f"    # advance to the next free index.  Re-running the script NEVER\n"
        f"    # errors and NEVER overwrites a prior result -- the #1 papercut\n"
        f"    # in the iterative HPC loop (resubmit after OOM / propor / a\n"
        f"    # walltime hit) was the old refuse-with-exit-1 gate.\n"
        f"    # (--continue additionally asks the engine to warm-start from\n"
        f"    # .DM/.CG/.XV; the run-index advances either way.)\n"
        f"    _run_n=$((_existing_max + 1))\n"
        f'    if [ "$_continue" != "1" ]; then\n'
        f'        echo "[molbuilder] prior output present; auto-continuing '
        f'as -run${{_run_n}}{ext} (use --force to restart at -run0, '
        f'--cold to also drop warm-start state)." >&2\n'
        f"    fi\n"
        f"else\n"
        f"    _run_n=0   # first run\n"
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


def _bash_numa_from_gpu(gpu_expr: str, out_var: str,
                        bus_var: str = "_bus", indent: str = "") -> str:
    """Emit bash that resolves the NUMA node of the GPU at index
    ``gpu_expr`` into ``out_var`` (``-1`` when unknown).

    Single source for the sysfs lookup so the per-rank launcher (§ 7.5.1)
    and the ``--dry-run`` placement report don't each carry their own
    copy of the subtle bit: ``nvidia-smi`` prints an **8-hex-digit** PCI
    domain (``00000000:02:00.0``) but the kernel sysfs path uses **4**
    (``0000:02:00.0``), so we strip the leading 4 chars with
    ``${bus:4}``.  Lowercased + space-stripped because sysfs is
    lowercase.  Any read failure leaves ``out_var=-1``.  ``indent`` is a
    leading-whitespace string applied to every line so the snippet lands
    aligned inside an indented block (e.g. the dry-run loop).
    """
    return (
        f'{indent}{bus_var}=$(nvidia-smi --id={gpu_expr} '
        f'--query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null '
        f"| tr 'A-Z' 'a-z' | tr -d ' ')\n"
        f'{indent}{out_var}=-1\n'
        f'{indent}[ -n "${bus_var}" ] && {out_var}=$(cat '
        f'/sys/bus/pci/devices/${{{bus_var}:4}}/numa_node 2>/dev/null '
        f'|| echo -1)\n'
    )


def _gpu_loadbalance_block() -> str:
    """Bash that derives the rank<->GPU load balance at RUN time.

    GOAL: turn the resolved rank count into a ranks-per-GPU figure so the
    rest of the wrapper can (a) gate MPS on actual GPU sharing and (b)
    let the per-rank launcher block-distribute ranks across GPUs.

    READS  ``$_mpi_np``  (rank count; set by the args block).
    SETS   ``$_ngpu``           -- visible GPU count (0 if none),
           ``$_ranks_per_gpu``  -- ``mpi_np / ngpu`` (clamped >= 1).
    EMITS  one stderr line ``molbuilder: GPU load-balance -- N ranks over
           G GPU(s) = K rank(s)/GPU`` -- the human-checkable confirmation
           that K matches the allocation (validation pinpoint for the
           § 11 benchmark sweep).
    WHY: K is the knob the benchmark sizes; it must come from the ACTUAL
    runtime allocation (SLURM may grant a different GPU count than
    generation assumed).  1 GPU degenerates to all-ranks-on-GPU0 (the
    existing single-GPU behavior) with no code fork.

    ORDERING: emit AFTER ``$_mpi_np`` resolves and BEFORE the MPS block
    (which reads ``$_ranks_per_gpu``).  slurm-integration.md § 7.5.1.
    """
    return (
        "# --- GPU load-balance: rank <-> GPU matching "
        "(slurm-integration.md § 7.5.1) ---\n"
        "# Count ALLOCATED GPUs now and split ranks across them.  The\n"
        "# per-rank launcher maps rank -> GPU as\n"
        "# local_rank*ngpu/localsize, so K=mpi_np/ngpu ranks share\n"
        "# each GPU via MPS.  1 GPU -> all ranks on GPU0 (the existing\n"
        "# behavior); N GPUs -> block-distributed, no code fork.\n"
        "# Prefer CUDA_VISIBLE_DEVICES (SLURM's allocated set) over\n"
        "# nvidia-smi -L (which over-counts on a shared node without\n"
        "# device-cgroup isolation) -- must agree with the per-rank\n"
        "# launcher's own count.\n"
        'if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then\n'
        '    _ngpu=$(printf %s "$CUDA_VISIBLE_DEVICES" '
        '| tr , "\\n" | grep -c .)\n'
        'else\n'
        '    _ngpu=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU " || true)\n'
        'fi\n'
        'case "$_ngpu" in ""|*[!0-9]*) _ngpu=0 ;; esac\n'
        'if [ "$_ngpu" -ge 1 ]; then\n'
        '    _ranks_per_gpu=$(( _mpi_np / _ngpu ))\n'
        '    [ "$_ranks_per_gpu" -lt 1 ] && _ranks_per_gpu=1\n'
        'else\n'
        '    _ranks_per_gpu=$_mpi_np\n'
        'fi\n'
        'echo "molbuilder: GPU load-balance -- $_mpi_np ranks over '
        '$_ngpu GPU(s) = $_ranks_per_gpu rank(s)/GPU '
        '(MPS when >=2)" >&2\n'
        "\n"
    )


def _gpu_socket_affinity_block() -> str:
    """Bash (rank-time, inside the quoted heredoc) implementing the
    GPU↔CPU socket co-location framework fix (slurm-integration.md
    § 7.5.2).

    Each rank already knows its GPU's NUMA node (``$_numa``).  This block
    resolves that NUMA node's **socket** (``physical_package_id``) and, by
    walking the rank's own ``Cpus_allowed_list``, the set of sockets it
    owns cores on AND the GPU-socket NUMA nodes it actually owns
    (``$_pin_numas``), then:

      * **pins** (``numactl --cpunodebind/--membind`` to the **owned**
        GPU-socket NUMA nodes only) when the cpuset spans MORE than one
        socket -- the whole-node / ``--exclusive`` (or partial multi-
        socket) case where a rank would otherwise roam cross-socket;
      * **no-ops** when the cpuset is already confined to the GPU's socket
        (SLURM co-located us -- nothing to do);
      * **warns, never fails** when we own NO cores on the GPU's socket
        (a shared cross-socket allocation we cannot fix from here).

    Binding only to OWNED NUMA nodes is the B-1 fix (audit 2026-06-27): a
    partial multi-NUMA allocation on the 8-NUMA Sol GPU nodes must never
    ``--membind`` to a socket-mate NUMA node it wasn't allocated.

    Sets ``$_pin`` (empty or the ``numactl`` prefix) for the ``exec``.
    All sysfs reads are guarded; missing topology/``numactl`` -> no pin.
    """
    return (
        '# --- GPU<->CPU socket co-location (slurm-integration.md 7.5.2) ---\n'
        '_pin=""\n'
        '_gpu_sock=""\n'
        'if [ "${_numa:--1}" -ge 0 ] && '
        '[ -r "/sys/devices/system/node/node${_numa}/cpulist" ]; then\n'
        '    _gcpu=$(sed "s/[,-].*//" '
        '"/sys/devices/system/node/node${_numa}/cpulist")\n'
        '    _gpu_sock=$(cat '
        '"/sys/devices/system/cpu/cpu${_gcpu}/topology/physical_package_id" '
        '2>/dev/null)\n'
        'fi\n'
        '# Walk this rank\'s OWNED cpus: collect the sockets we span and the\n'
        '# GPU-socket NUMA nodes we actually own.  Sample both endpoints of\n'
        '# each range so a contiguous "0-47" is seen as two sockets.\n'
        '_my_socks=""; _pin_numas=""\n'
        '_cl=$(awk "/Cpus_allowed_list/{print \\$2}" /proc/self/status '
        '2>/dev/null)\n'
        '_oldifs=$IFS; IFS=,\n'
        'for _rng in $_cl; do\n'
        '    _a=${_rng%%-*}; _b=${_rng##*-}\n'
        '    for _e in "$_a" "$_b"; do\n'
        '        case "$_e" in ""|*[!0-9]*) continue ;; esac\n'
        '        _ps=$(cat '
        '"/sys/devices/system/cpu/cpu${_e}/topology/physical_package_id" '
        '2>/dev/null)\n'
        '        [ -n "$_ps" ] || continue\n'
        '        case ",$_my_socks," in *",$_ps,"*) : ;; '
        '*) _my_socks="${_my_socks:+$_my_socks,}$_ps" ;; esac\n'
        '        # If this owned cpu is on the GPU\'s socket, record its\n'
        '        # NUMA node (the node* symlink under the cpu dir).\n'
        '        if [ -n "$_gpu_sock" ] && [ "$_ps" = "$_gpu_sock" ]; then\n'
        '            for _nd in '
        '/sys/devices/system/cpu/cpu${_e}/node[0-9]*; do\n'
        '                _nn=${_nd##*node}\n'
        '                case "$_nn" in ""|*[!0-9]*) continue ;; esac\n'
        '                case ",$_pin_numas," in *",$_nn,"*) : ;; '
        '*) _pin_numas="${_pin_numas:+$_pin_numas,}$_nn" ;; esac\n'
        '                break\n'
        '            done\n'
        '        fi\n'
        '    done\n'
        'done\n'
        'IFS=$_oldifs\n'
        'if [ -n "$_gpu_sock" ] && [ -n "$_my_socks" ]; then\n'
        '    case ",$_my_socks," in\n'
        '      *",$_gpu_sock,"*)\n'
        '        case "$_my_socks" in\n'
        '          *,*)\n'
        '            if command -v numactl >/dev/null 2>&1 && '
        '[ -n "$_pin_numas" ]; then\n'
        '                _pin="numactl --cpunodebind=$_pin_numas '
        '--membind=$_pin_numas"\n'
        '                echo "molbuilder[rank ${_lr}/${_ls}]: socket-pin '
        '-> GPU socket $_gpu_sock, owned numa $_pin_numas via numactl '
        '(was multi-socket)" >&2\n'
        '            fi\n'
        '            ;;\n'
        '        esac\n'
        '        ;;\n'
        '      *)\n'
        '        echo "molbuilder[rank ${_lr}/${_ls}]: WARN cross-socket '
        '-- GPU numa=$_numa is on socket $_gpu_sock but this rank owns '
        'cores only on socket(s) $_my_socks; host<->device + ELPA OpenMP '
        'pay a remote hop. Request --exclusive for clean GPU timing '
        '(slurm-integration.md 7.5.2)." >&2\n'
        '        ;;\n'
        '    esac\n'
        'fi\n'
    )


def _gpu_per_rank_launcher_block() -> str:
    """Bash that writes the per-rank GPU launcher + picks the CPU-bind
    policy.  Emit ONLY in GPU mode.

    GOAL: give every MPI rank its own GPU and (under SLURM) defer CPU/mem
    placement to the scheduler.  Implements the general load-balance
    model -- 1 GPU degenerates to "all ranks on GPU0", N GPUs
    block-distribute -- with no code fork on GPU count (§ 7.5.1).

    WRITES a runtime helper ``.mb-rank-launch-$$.sh`` (``trap``-removed on
    EXIT) in which each rank computes ``gpu = local_rank*ngpu/localsize``,
    resolves that GPU's NUMA node, sets ``CUDA_VISIBLE_DEVICES``, logs its
    GPU+NUMA+cpuset, then ``exec``s SIESTA.  The heredoc is QUOTED
    (``'HELPEREOF'``) so the rank-time variables resolve when the rank
    runs, not when the wrapper writes the file.
    SETS  ``$_siesta_target="bash $_rank_helper"`` (the launch target the
          ``_launch_cmd`` assembly interpolates in place of bare siesta).
    UNDER SLURM (``$SLURM_JOB_ID`` set): clears ``$_numa_wrap_gpu`` and
          ``$_mpirun_bind`` so we do NOT double-bind against SLURM's
          cgroup cpuset (P1 in § 7.5.1.b; the benchmark logs the actual
          cpuset to confirm SLURM bound near the GPU).
    WHY a helper FILE (not ``bash -c``): the per-rank logic can't survive
    as a word-split ``_launch_cmd`` string -- the ``bash -c`` quoting
    breaks under the later unquoted expansion -- so a tiny temp script is
    the robust idiom.
    """
    return (
        '_rank_helper=".mb-rank-launch-$$.sh"\n'
        "cat > \"$_rank_helper\" <<'HELPEREOF'\n"
        '#!/bin/bash\n'
        '_lr=${OMPI_COMM_WORLD_LOCAL_RANK:-${PMIX_RANK:-0}}\n'
        '_ls=${OMPI_COMM_WORLD_LOCAL_SIZE:-${SLURM_NTASKS:-1}}\n'
        '[ "${_ls:-0}" -lt 1 ] && _ls=1\n'
        '# Allocated GPU list: prefer CUDA_VISIBLE_DEVICES (SLURM sets it\n'
        '# to the allocated set -- robust whether or not device-cgroups\n'
        '# isolate the node, and it carries the real physical indices);\n'
        '# fall back to all visible GPUs only when unset (local / no\n'
        '# scheduler).  nvidia-smi -L alone would over-count on a shared\n'
        '# node without cgroup isolation.\n'
        'if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then\n'
        '    IFS=, read -ra _gpus <<< "$CUDA_VISIBLE_DEVICES"\n'
        'else\n'
        '    _n=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU " || true)\n'
        '    case "$_n" in ""|*[!0-9]*) _n=1 ;; esac\n'
        '    [ "$_n" -lt 1 ] && _n=1\n'
        '    _gpus=(); _i=0\n'
        '    while [ "$_i" -lt "$_n" ]; do _gpus+=("$_i"); _i=$((_i+1)); done\n'
        'fi\n'
        '_ngpu=${#_gpus[@]}\n'
        '[ "$_ngpu" -lt 1 ] && { _gpus=(0); _ngpu=1; }\n'
        '_idx=$(( _lr * _ngpu / _ls ))\n'
        '_gpu=${_gpus[$_idx]}\n'
        # Resolve NUMA from the chosen GPU index BEFORE narrowing
        # CUDA_VISIBLE_DEVICES (keeps the sysfs lookup unambiguous).
        + _bash_numa_from_gpu("$_gpu", "_numa")
        + 'export CUDA_VISIBLE_DEVICES=$_gpu\n'
        "_cpus=$(awk '/Cpus_allowed_list/{print $2}' "
        '/proc/self/status 2>/dev/null)\n'
        'echo "molbuilder[rank ${_lr}/${_ls}]: '
        'CUDA_VISIBLE_DEVICES=$_gpu (numa=$_numa) '
        'cpus=${_cpus:-?}" >&2\n'
        + _gpu_socket_affinity_block()
        + 'exec $_pin siesta "$@"\n'
        'HELPEREOF\n'
        'chmod +x "$_rank_helper"\n'
        # Cleanup is handled by the single unified EXIT trap (_mb_cleanup,
        # set near the top): it rm's $_rank_helper.  A local ``trap ...
        # EXIT`` here would CLOBBER the MPS cleanup trap (bash keeps only
        # the last EXIT trap) -- hence the centralised function.
        '_siesta_target="bash $_rank_helper"\n'
        # Under SLURM, the scheduler's cgroup cpuset (--gres-flags=
        # enforce-binding + -c) governs CPU/memory placement.  Do NOT
        # also numactl-wrap / --map-by -- that double-binds and can fight
        # SLURM.  P1 in slurm-integration.md § 7.5.1.b.
        'if [ -n "${SLURM_JOB_ID:-}" ]; then\n'
        '    _numa_wrap_gpu=""\n'
        '    _mpirun_bind=""\n'
        '    echo "molbuilder: under SLURM (job $SLURM_JOB_ID) -- '
        'trusting scheduler cpuset for CPU/mem binding '
        '(no manual numactl)" >&2\n'
        'fi\n'
    )


def _siesta_resolved_log_block(script_name: str, gpu_mode: bool) -> str:
    """Bash that records the fully-resolved launch command + placement
    into the wrapper log -- ALWAYS, on every run.

    GOAL (user request 2026-06-26): the log must carry the exact command
    that ran and the GPU/NUMA context, so a post-mortem never has to
    guess what was launched.  These ``_log INFO`` lines ARE the audit
    trail that pins what the run did -- the validation anchor when a run
    is slow or wrong.

    READS ``$_launch_cmd`` / ``$_launch_note`` / ``$_mpi_np`` /
    ``$_omp_threads`` and (GPU mode) ``$_ngpu`` / ``$_ranks_per_gpu`` /
    ``$_numa_wrap_gpu``.
    EMITS ``_log INFO`` lines (resolved launch / launch mode / ranks-omp /
    [gpu placement]).
    """
    return (
        f'# --- Record resolved launch command + placement (log) ---\n'
        f'_log INFO "resolved launch : $_launch_cmd {script_name} '
        f'> $_out_file"\n'
        f'_log INFO "launch mode     : $_launch_note"\n'
        f'_log INFO "ranks / omp     : $_mpi_np ranks x $_omp_threads '
        f'OMP threads"\n'
        + (
            f'_log INFO "gpu placement   : ${{_ngpu:-0}} GPU(s), '
            f'${{_ranks_per_gpu:-?}} rank(s)/GPU, '
            f'numa-wrap=\'${{_numa_wrap_gpu:-none}}\'"\n'
            if gpu_mode else ""
        )
    )


def _siesta_mem_audit_block(params: Mapping[str, Any]) -> str:
    """Bash that, at runtime, RE-estimates peak memory for the ACTUAL
    resolved rank count and compares it to the SLURM allocation -- the
    runtime half of the ``--mem`` story (user request 2026-06-27).

    The launcher runs in the backend env (no molbuilder/numpy), so the
    memory MODEL coefficients are baked here at generate time and the
    estimate is recomputed in ``awk`` against ``$_mpi_np``.  This is what
    catches the dangerous case the generate-time ``--mem`` cannot: the
    user overriding ``-n`` or ``--mem`` on the ``sbatch`` CLI (the np=64
    OOM-at-240G lesson).  Mirrors
    :func:`molbuilder.siesta.memory.estimate_siesta_memory`:
    ``ceil(safety*(fixed + c_rank*n)) + ceil(extra)``, floored, capped.

    READS ``$_mpi_np`` and SLURM's ``SLURM_MEM_PER_NODE`` /
    ``SLURM_MEM_PER_CPU`` (+ ``/proc/meminfo`` when not under SLURM).
    EMITS one ``_log INFO`` line (+ ``_log WARN`` if the estimate exceeds
    the allocation).
    """
    fixed  = float(params["fixed_gb"])
    perrk  = float(params["per_rank_gb"])
    safety = float(params["safety"])
    extra  = float(params["extra_gb"])
    floor  = float(params["floor_gb"])
    cap    = float(params.get("cap_gb") or 0.0)
    return (
        "# --- Memory: estimate (for the resolved rank count) vs SLURM "
        "allocation ---\n"
        f"_mb_mem_est=$(awk -v n=\"$_mpi_np\" -v f={fixed:g} -v pr={perrk:g} "
        f"-v s={safety:g} -v e={extra:g} -v fl={floor:g} -v cap={cap:g} "
        "'BEGIN{ raw=s*(f+pr*n); est=int(raw); if(est<raw)est++; "
        "ex=int(e); if(ex<e)ex++; est+=ex; if(est<fl)est=fl; "
        "if(cap>0 && est>cap)est=cap; print est }')\n"
        "_mb_mem_alloc_mb=\"\"\n"
        "if [ -n \"${SLURM_MEM_PER_NODE:-}\" ]; then\n"
        "    _mb_mem_alloc_mb=$SLURM_MEM_PER_NODE\n"
        "elif [ -n \"${SLURM_MEM_PER_CPU:-}\" ]; then\n"
        "    _mb_mem_alloc_mb=$(( SLURM_MEM_PER_CPU * "
        "${SLURM_NTASKS:-1} * ${SLURM_CPUS_PER_TASK:-1} ))\n"
        "fi\n"
        "if [ -n \"$_mb_mem_alloc_mb\" ]; then\n"
        "    _mb_mem_alloc=$(( _mb_mem_alloc_mb / 1024 ))\n"
        "    _log INFO \"memory          : estimated ~${_mb_mem_est}G "
        "($_mpi_np ranks, model) vs allocated ~${_mb_mem_alloc}G (SLURM)\"\n"
        "    if [ \"$_mb_mem_est\" -gt \"$_mb_mem_alloc\" ]; then\n"
        "        _log WARN \"memory          : estimate (~${_mb_mem_est}G) "
        "EXCEEDS allocation (~${_mb_mem_alloc}G) -- OOM risk; raise --mem "
        "or lower -n (slurm-integration.md mem model)\"\n"
        "    fi\n"
        "else\n"
        "    _mb_mem_phys=$(awk '/MemTotal/{printf \"%d\", $2/1048576}' "
        "/proc/meminfo 2>/dev/null)\n"
        "    _log INFO \"memory          : estimated ~${_mb_mem_est}G "
        "($_mpi_np ranks, model); node RAM ~${_mb_mem_phys:-?}G "
        "(no SLURM --mem)\"\n"
        "fi\n"
    )


def _siesta_dry_run_block(script_name: str, gpu_mode: bool) -> str:
    """Bash implementing ``--dry-run``: print the resolved command and,
    in GPU mode, the per-rank GPU/NUMA mapping that WOULD be used, then
    ``exit 0`` WITHOUT launching SIESTA.

    GOAL (user request 2026-06-26): let a user ``sbatch job.sbatch
    --dry-run`` (or run locally) and read the log to confirm the
    generated command matches the SLURM-allocated resources, WITHOUT
    spending a real run.  This is the cheap pre-flight validation of the
    whole resolution chain.

    READS the same resolved vars as the launch.  In GPU mode it loops
    ranks ``0..mpi_np-1`` and prints ``rank R -> GPU G (numa=N)`` -- the
    block-distribution the per-rank launcher will apply (divisor is
    ``$_mpi_np`` here since LOCAL_SIZE isn't set without mpirun; equal on
    single-node v1).
    SIDE-EFFECT-FREE: the MPS daemon start is separately gated on
    ``_dry_run != 1`` so a dry run touches no GPU state.
    EXPECTED OUTCOME: a ``DRY RUN`` banner + mapping table on stdout/log,
    then ``exit 0`` -- nothing else runs.
    """
    return (
        f'if [ "$_dry_run" = 1 ]; then\n'
        f'    echo ""\n'
        f'    echo "===== molbuilder DRY RUN (no SIESTA launch) ====="\n'
        f'    echo "  Resolved cmd : $_launch_cmd {script_name} '
        f'> $_out_file"\n'
        f'    echo "  Launch mode  : $_launch_note"\n'
        f'    echo "  MPI ranks    : $_mpi_np"\n'
        f'    echo "  OMP threads  : $_omp_threads"\n'
        f'    echo "  SLURM        : job=${{SLURM_JOB_ID:-<none>}} '
        f'ntasks=${{SLURM_NTASKS:-?}} cpus/task=${{SLURM_CPUS_PER_TASK:-?}} '
        f'gpus=${{SLURM_JOB_GPUS:-${{SLURM_GPUS:-?}}}}"\n'
        + (
            f'    echo "  Visible GPUs : ${{_ngpu:-0}}"\n'
            f'    echo "  Ranks/GPU    : ${{_ranks_per_gpu:-?}} '
            f'(MPS ${{_use_mps_str:-?}})"\n'
            f'    echo "  Rank -> GPU mapping (block-distributed):"\n'
            f'    _dr_ng=${{_ngpu:-0}}\n'
            f'    if [ "$_dr_ng" -lt 1 ]; then\n'
            f'        echo "      (no GPUs visible here -- on the '
            f'compute node each rank maps as rank*ngpu/ntasks)"\n'
            f'    else\n'
            f'        _dr_r=0\n'
            f'        while [ "$_dr_r" -lt "$_mpi_np" ]; do\n'
            f'            _dr_g=$(( _dr_r * _dr_ng / _mpi_np ))\n'
            + _bash_numa_from_gpu("$_dr_g", "_dr_numa",
                                  bus_var="_dr_bus", indent="            ")
            + f'            echo "      rank $_dr_r -> GPU $_dr_g '
            f'(numa=$_dr_numa)"\n'
            f'            _dr_r=$(( _dr_r + 1 ))\n'
            f'        done\n'
            f'    fi\n'
            if gpu_mode else ""
        )
        + f'    echo "================================================"\n'
        f'    _log INFO "dry-run complete; no SIESTA launched"\n'
        f'    exit 0\n'
        f'fi\n'
        f"\n"
    )


def _siesta_scf_timing_func() -> str:
    """Bash defining ``_mb_scf_tee`` — the SCF per-iteration timing
    instrument (slurm-integration.md § 11.0b).

    GOAL: SIESTA emits **no usable per-iteration wall time** (the ``scf:``
    lines carry energies + dDmax but no time; ``timer: IterSCF`` is
    cumulative). This filter IS the benchmark's measurement instrument:
    it tees SIESTA stdout to the ``.out`` AND timestamps every ``scf:``
    iteration line into a per-run ``.scf-timing.log`` as
    ``<epoch.ns> <iter#> <full scf line>``, so per-iteration wall time =
    consecutive-stamp delta (report mean of iters 3–5, § 11.0).

    EXPECTED OUTPUT: `<basename>-runN.scf-timing.log`, one line per SCF
    iteration; subtracting adjacent epochs gives the steady-state
    per-iteration time — the number the CPU-vs-GPU / K-sweep comparison
    ranks on.

    WHY this shape: a single ``awk`` process copies every line at C speed
    (``fflush(out)`` keeps the ``.out`` live for ``tail -f``); ``date
    +%s.%N`` is spawned only on the infrequent ``scf:`` lines. Portable
    across gawk/mawk (getline-from-command + close + fflush). The caller
    pipes ``$_launch_cmd … | _mb_scf_tee "$_out_file" "$_scf_timing_log"``
    and reads ``${PIPESTATUS[0]}`` for SIESTA's exit (awk never masks it).
    """
    return (
        "# --- SCF per-iteration timing instrument "
        "(slurm-integration.md § 11.0b) ---\n"
        "# Tees SIESTA stdout to the .out AND stamps each scf: iteration\n"
        "# line into the per-run .scf-timing.log so per-iter wall time =\n"
        "# consecutive-epoch delta (SIESTA prints no per-iter time).\n"
        "_mb_scf_tee() {\n"
        "    awk -v out=\"$1\" -v tlog=\"$2\" '\n"
        "        { print > out; fflush(out) }\n"
        "        /^[ \\t]*scf:[ \\t]*[0-9]/ {\n"
        "            _cmd=\"date +%s.%N\"; _cmd | getline _ts; close(_cmd)\n"
        "            _l=$0; sub(/^[ \\t]*scf:[ \\t]*/, \"\", _l)\n"
        "            split(_l, _f, /[ \\t]+/)\n"
        "            print _ts, _f[1], $0 > tlog; fflush(tlog)\n"
        "        }\n"
        "    '\n"
        "}\n"
    )


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
                        mem_audit: Optional[Mapping[str, Any]] = None) -> str:
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

    # GPU-env presence gate.  When the .fdf opted into GPU diagonalization
    # at generate time but ``molbuilder-siesta-gpu`` isn't installed,
    # ``source activate molbuilder-siesta-gpu`` would fail at run time
    # with a conda-side error -- the user-facing message would not point
    # at the real fix (install the env or turn the toggle off).  Raise
    # here with the install hint so the wrong env is caught at script-
    # generation time.  Only fires when we AUTO-routed (env not user-
    # passed) AND the host snapshot lists at least one env (an empty
    # snapshot means the conda probe never ran -- can't gate on that).
    if (env is None
            and env_lookup_category == "siesta-gpu"
            and caps.conda_envs
            and not caps.env_available(target_env)):
        raise WrapperError(
            f"`{script_path.name}` requests GPU diagonalization "
            f"(``Diag.ELPA.GPU .true.``) but the ``{target_env}`` env "
            f"is not installed.  Install it with "
            f"``python -m molbuilder envs install {target_env}`` "
            f"(source build, takes ~10 minutes), or turn off the "
            f"SIESTA ``Use GPU`` toggle and regenerate the .fdf to "
            f"run on the precompiled CPU SIESTA."
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
        # CPU-branch defaults: resolved_mpi already reflects user choice
        # or auto-MPI clamp -- bake it verbatim (preserves the existing
        # CPU-mode contract).
        cpu_mpi_default = str(resolved_mpi)
        cpu_omp_default = str(resolved_omp)
        # GPU-branch defaults: honour an explicit user-set choice, else
        # use the runtime-probed GPU policy variables emitted by
        # _gpu_runtime_defaults_block.  When the wrapper is generated
        # in CPU mode (no GPU defaults block emitted) but the user
        # later toggles Diag.ELPA.GPU .true. in the .fdf, those vars
        # don't exist -- fall back to safe hardcoded 4 ranks / 1 OMP
        # (the ELPA 2024.05 no-NCCL throughput optimum on 1 GPU).
        if user_set_mpi:
            gpu_mpi_default = str(resolved_mpi)
        elif gpu_mode:
            gpu_mpi_default = "$_gpu_mpi_np_default"
        else:
            gpu_mpi_default = "4"
        if user_set_omp:
            gpu_omp_default = str(resolved_omp)
        elif gpu_mode:
            gpu_omp_default = "$_omp_default"
        else:
            gpu_omp_default = "1"
        # Runtime selector: re-read the .fdf at LAUNCH (not at
        # generation) so a user who manually toggles ``Diag.ELPA.GPU``
        # in the .fdf after the wrapper is emitted gets the right
        # rank count automatically.  Before this (2026-06-26 fix),
        # gpu_mode was baked at gen time -- a CPU-mode bake left
        # _mpi_np_default=20 in place, which OOM'd the GPU when the
        # user later set Diag.ELPA.GPU .true.  See task #36.
        _mpi_np_default_assignment = (
            f'# Default mpi_np / omp_threads -- re-evaluated at LAUNCH\n'
            f'# from the current .fdf so toggling Diag.ELPA.GPU after\n'
            f'# generation picks up the right rank count (task #36 fix).\n'
            f'if grep -qiE \'^[[:space:]]*Diag\\.ELPA\\.GPU[[:space:]]+'
            f'\\.true\\.\' "{script_name}" 2>/dev/null; then\n'
            f'    _mpi_np_default={gpu_mpi_default}\n'
            f'    _omp_threads_default={gpu_omp_default}\n'
            f'    echo "molbuilder: .fdf has Diag.ELPA.GPU .true. -> '
            f'default mpi_np=$_mpi_np_default, omp=$_omp_threads_default" >&2\n'
            f'else\n'
            f'    _mpi_np_default={cpu_mpi_default}\n'
            f'    _omp_threads_default={cpu_omp_default}\n'
            f'fi\n'
        )
        siesta_args_block = (
            _continue_force_args_parser("SIESTA wrapper")
            + f"# --- SIESTA-specific argument parsing -----------\n"
            f"# Override the generation-time defaults with: ``-np N`` /\n"
            f"# ``-omp N`` flags, or ``MB_NP`` / ``OMP_NUM_THREADS`` env\n"
            f"# vars.  Useful for retrying after a propor crash (see the\n"
            f"# diagnostic at the bottom of this wrapper) or for bench\n"
            f"# sweeps WITHOUT regenerating the .fdf / wrapper.\n"
            + _mpi_np_default_assignment
            + f"# MPI rank-count precedence (highest first):\n"
            f"#   1. ``-np N`` flag on the wrapper invocation\n"
            f"#   2. ``MB_NP`` env var (manual override)\n"
            f"#   3. ``SLURM_NTASKS`` (scheduler-allocated under sbatch)\n"
            f"#   4. ``PBS_NP`` (scheduler-allocated under qsub)\n"
            f"#   5. generation-time default ($_mpi_np_default)\n"
            f"# Per docs/config.md § 1.5: reading scheduler env vars for\n"
            f"# launch tuning is part of the wrapper contract -- the user\n"
            f"# reserved ``--ntasks=N`` from SLURM, the wrapper honors it.\n"
            f'_mpi_np="${{MB_NP:-${{SLURM_NTASKS:-${{PBS_NP:-$_mpi_np_default}}}}}}"\n'
            # OMP precedence: -omp flag > OMP_NUM_THREADS env >
            # SLURM_CPUS_PER_TASK (the sbatch ``-c`` allocation) > policy
            # default.  Honoring a user-set OMP_NUM_THREADS matches the
            # standard OMP-toolchain convention; the prior wrapper
            # unconditionally clobbered it, which surprised users
            # benching with ``OMP_NUM_THREADS=8 ./run.sh``.  Under sbatch
            # the scheduler reserved ``-c`` cores/rank (the OMP width per
            # the slurm-integration.md § 7.5.1 sizing) -- honor it so the
            # Sol allocation drives OMP automatically without a manual
            # -omp (config.md § 1.5: reading scheduler env for launch
            # tuning is part of the wrapper contract).
            f'_omp_threads="${{OMP_NUM_THREADS:-'
            f'${{SLURM_CPUS_PER_TASK:-$_omp_threads_default}}}}"\n'
            f'_dry_run=0\n'
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
            f"        --dry-run|--dryrun)\n"
            f"            # Resolve + LOG the launch command and the\n"
            f"            # rank<->GPU/NUMA placement, then exit WITHOUT\n"
            f"            # running SIESTA.  Lets you sbatch a preview and\n"
            f"            # read the log to confirm the command matches\n"
            f"            # the allocation (slurm-integration.md § 7.5.1).\n"
            f'            _dry_run=1; shift ;;\n'
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
            f"  --dry-run        resolve + log the launch command and the\n"
            f"                   rank->GPU/NUMA placement for the current\n"
            f"                   allocation, then exit WITHOUT running\n"
            f"                   SIESTA.  Use to preview/validate a job\n"
            f"                   (e.g. ``sbatch job.sbatch --dry-run``).\n"
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
            # GPU load-balance: derive ranks-per-GPU from the resolved
            # rank count so MPS gates on real sharing + the per-rank
            # launcher can block-distribute.  Must sit AFTER the args
            # block (needs $_mpi_np) and BEFORE the MPS block (reads
            # $_ranks_per_gpu).  See _gpu_loadbalance_block.__doc__.
            + (_gpu_loadbalance_block() if gpu_mode else "")
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
                # ``_dry_run != 1`` guard: a --dry-run must not start the
                # MPS daemon (a real GPU side-effect).  The dry-run report
                # still shows the would-be MPS state from _use_mps_str.
                'if [ "$_use_mps_default" = "1" ] '
                '&& [ "$_ranks_per_gpu" -ge 2 ] '
                '&& [ "${_dry_run:-0}" != "1" ]; then\n'
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
                '    # Mark MPS as started so the SINGLE unified EXIT trap\n'
                '    # (_mb_cleanup, set near the top) stops the daemon +\n'
                '    # removes the per-job dirs on ANY exit (success, error,\n'
                '    # signal).  Set regardless of whether the daemon bound\n'
                '    # -- a partially-started daemon still wants cleanup.\n'
                '    # NB: a local ``trap ... EXIT`` here would be CLOBBERED\n'
                '    # by the per-rank launcher trap; the unified function\n'
                '    # is why teardown is centralised.\n'
                '    _mps_started=1\n'
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
                '&& [ "$_ranks_per_gpu" -lt 2 ]; then\n'
                '        echo "molbuilder: MPS auto-disabled '
                '(1 rank/GPU; MPS adds overhead with no '
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
            # Default launch target is the bare binary; GPU mode swaps in
            # the per-rank launcher (assigns each rank its GPU + picks the
            # CPU-bind policy).  See _gpu_per_rank_launcher_block.__doc__.
            + '_siesta_target="siesta"\n'
            + (_gpu_per_rank_launcher_block() if gpu_mode else "")
            + f'if [ "$_has_mpi" = 1 ]; then\n'
            f'    _launch_cmd="$_numa_wrap_gpu mpirun -np $_mpi_np $_mpirun_bind $_siesta_target"\n'
            f'    if [ "$_has_omp" = 1 ]; then\n'
            f'        _launch_note="hybrid MPI+OMP ($_mpi_np ranks x $_omp_threads OMP threads)"\n'
            f'    else\n'
            f'        _launch_note="pure MPI ($_mpi_np ranks; OMP setting irrelevant to this binary)"\n'
            f'    fi\n'
            f'elif [ "$_has_omp" = 1 ]; then\n'
            f'    _launch_cmd="siesta"\n'
            f'    _launch_note="OMP-only build ($_omp_threads threads)"\n'
            f'elif [ -z "$_siesta_par" ]; then\n'
            f'    _launch_cmd="$_numa_wrap_gpu mpirun -np $_mpi_np $_mpirun_bind $_siesta_target"\n'
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
            f'_dry_run=0\n'
            f'while [ $# -gt 0 ]; do\n'
            f'    case "$1" in\n'
            f"        --dry-run|--dryrun)\n"
            f'            _dry_run=1; shift ;;\n'
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

    # Per docs/config.md (v2 rewrite, 2026-06-24): the wrapper is a
    # self-contained shell script.  At generate time the generator reads
    # ``script_generation.preamble`` and ``script_generation.activation``
    # from molbuilder.json (server-wide) + .molbuilder.json (project,
    # optional) and bakes them VERBATIM into the wrapper.  At runtime
    # the wrapper does no discovery, no probing, no config-file reads,
    # no env-var-driven behaviour switching.  If anything fails,
    # ``set -euo pipefail`` aborts with the real bash error.
    #
    # Per § 2 of the design doc, ``activation`` has no default -- the
    # generator refuses to emit a wrapper when it isn't set.  The
    # ``require_activation`` helper raises RuntimeConfigError with a
    # operator-facing message + the canonical molbuilder.json snippet.
    from . import runtime_config as _rc
    _project_dir = script_path.parent if script_path.parent.exists() else None
    _sg = _rc.get_script_generation(project_dir=_project_dir)
    _preamble_chunks = _sg["preamble_chunks"]
    _activation_form = _rc.require_activation(project_dir=_project_dir)

    # Render preamble with per-scope sentinel comments so a user
    # reading the wrapper sees which scope contributed which lines.
    # No xtrace, no fancy framing -- just the user's bash, baked.
    _scope_labels = {
        "server":  "SERVER PREAMBLE (from molbuilder.json)",
        "project": "PROJECT ADDITIONS (from .molbuilder.json)",
    }
    if _preamble_chunks:
        _rendered_chunks = [
            f"# === {_scope_labels.get(scope, scope.upper())} ===\n{text}\n"
            for scope, text in _preamble_chunks
        ]
        _preamble_block = (
            "# --- Baked preamble (verbatim from molbuilder.json) ---\n"
            "_log STAGE \"running baked preamble\"\n"
            + "\n".join(_rendered_chunks)
            + "\n"
        )
    else:
        _preamble_block = (
            "# --- Baked preamble (none configured) ---\n"
            "# (no script_generation.preamble in any scope)\n"
            "\n"
        )

    env_activation = (
        f"# --- Per-run log file (current directory; see docs/config.md § 1) -\n"
        f'_runwrap_log="{basename}.runwrap-$(date +%Y%m%d-%H%M%S).log"\n'
        f'exec > >(tee -a "$_runwrap_log") 2> >(tee -a "$_runwrap_log" >&2)\n'
        f"\n"
        f"# Structured log helper.\n"
        f"_log() {{\n"
        f"    printf '[%s] [%-5s] %s\\n' \"$(date '+%H:%M:%S%z')\" \"$1\" \"$2\" >&2\n"
        f"}}\n"
        f"\n"
        f"# Single unified EXIT cleanup (one trap -- a second ``trap ...\n"
        f"# EXIT`` would REPLACE the first, so all teardown lives here).\n"
        f"# No-ops unless the relevant vars were set, so it is safe for\n"
        f"# CPU / PySCF / non-MPS runs.  Cleans (a) the per-rank GPU\n"
        f"# launcher temp file and (b) the MPS daemon + its pipe/log dirs.\n"
        f"_mb_cleanup() {{\n"
        f'    [ -n "${{_monitor_pid:-}}" ] && kill "$_monitor_pid" '
        f"2>/dev/null\n"
        f'    [ -n "${{_rank_helper:-}}" ] && rm -f "$_rank_helper" 2>/dev/null\n'
        f'    if [ "${{_mps_started:-0}}" = "1" ]; then\n'
        f"        echo quit | nvidia-cuda-mps-control >/dev/null 2>&1\n"
        f'        rm -rf "${{CUDA_MPS_PIPE_DIRECTORY:-}}" '
        f'"${{CUDA_MPS_LOG_DIRECTORY:-}}" 2>/dev/null\n'
        f"    fi\n"
        f"}}\n"
        f"trap _mb_cleanup EXIT\n"
        f"\n"
        f"_log STAGE \"===== molbuilder wrapper start =====\"\n"
        f'_log INFO "timestamp:  $(date \'+%Y-%m-%d %H:%M:%S %Z\')"\n'
        f'_log INFO "hostname:   $(hostname)"\n'
        f'_log INFO "user:       ${{USER:-?}}"\n'
        f'_log INFO "cwd:        $(pwd)"\n'
        f'_log INFO "script:     $0"\n'
        f'_log INFO "argv:       $0 $*"\n'
        f'_log INFO "log file:   $_runwrap_log"\n'
        f"# Scheduler context -- only emit if the var is set.  These are\n"
        f"# read for diagnostic logging + launch tuning (see § 1.5 of\n"
        f"# docs/config.md); they do NOT alter activation or preamble.\n"
        f'for _v in SLURM_JOB_ID SLURM_NTASKS SLURM_CPUS_PER_TASK \\\n'
        f"          SLURM_JOB_NODELIST SLURM_GPUS SLURM_JOB_GPUS \\\n"
        f"          PBS_JOBID PBS_NP PBS_NODEFILE; do\n"
        f"    _v_val=\"${{!_v:-}}\"\n"
        f'    [ -n "$_v_val" ] && _log INFO "$_v=$_v_val"\n'
        f"done\n"
        f"\n"
        f"# The env bootstrap (preamble ``module load``, conda/mamba\n"
        f"# ``activate``) sources EXTERNAL scripts not under our control.\n"
        f"# Conda activate.d hooks (e.g. cuda-nvcc's, which references an\n"
        f"# unset NVCC_PREPEND_FLAGS) abort under the wrapper's ``set -u``.\n"
        f"# Disable nounset for the bootstrap ONLY; restore it before our\n"
        f"# own logic (where -u still catches real bugs).\n"
        f"set +u\n"
        f"{_preamble_block}"
        f"# --- Activation (verbatim from script_generation.activation) ---\n"
        f'_log STAGE "{_activation_form} {target_env}"\n'
        f"{_activation_form} {target_env}\n"
        f"set -u\n"
        f"\n"
        f'_log INFO "CONDA_DEFAULT_ENV=${{CONDA_DEFAULT_ENV:-<unset>}}"\n'
        f'_log INFO "CONDA_PREFIX=${{CONDA_PREFIX:-<unset>}}"\n'
        f'_log INFO "which python: $(command -v python 2>/dev/null || echo \'(not on PATH)\')"\n'
        f"\n"
    )

    # Launch + diagnostics.  For SIESTA we run the command (not
    # exec) so we can inspect the .out for ``propor: ERROR: IMAX = 0``
    # on failure and print a targeted retry hint.  Layer-on-top
    # cost: one extra bash process for the wrapper's lifetime; cheap.
    # For PySCF the original exec is preserved -- no diagnostic
    # surface there yet.
    if category == "siesta":
        # Always-on launch-command audit log + the --dry-run preview, both
        # extracted into named block-emitters (see their docstrings for
        # the goal/contract).  Order: log the resolved command, then the
        # dry-run guard (exits before launch), then the real launch.
        launch_block = (
            _siesta_resolved_log_block(script_name, gpu_mode)
            + (_siesta_mem_audit_block(mem_audit) if mem_audit else "")
            + _siesta_dry_run_block(script_name, gpu_mode)
            + _siesta_scf_timing_func()
            + f"# --- Launch SIESTA + capture exit -----------------------\n"
            f"# `set +e` lets us inspect the exit code; the diagnostic\n"
            f"# below reads the .out for ``propor: ERROR`` and prints a\n"
            f"# retry suggestion.  Then we re-exit with SIESTA's code.\n"
            f"# stdout is piped through _mb_scf_tee, which writes the .out\n"
            f"# AND the per-iteration .scf-timing.log (§ 11.0b); SIESTA's\n"
            f"# stderr stays on the wrapper's stderr (runwrap log).  We read\n"
            f"# ${{PIPESTATUS[0]}} so awk never masks SIESTA's exit code.\n"
            f'_scf_timing_log="${{_out_file%.out}}.scf-timing.log"\n'
            f'_log INFO "scf timing  : per-iteration stamps -> '
            f'$_scf_timing_log"\n'
            # --- Background job monitor (PoC; § 11.0b) -------------------
            # The monitor is the SELF-CONTAINED, stdlib-only ``mb_monitor.py``
            # shipped next to this wrapper (a copy of molbuilder/monitor.py).
            # It runs with the JOB's OWN python from the working dir -- NO
            # molbuilder install, NO numpy, NO repo on PATH, NO separate env
            # (the backend siesta env has none of those).  Each interval it
            # parses .out + .scf-timing.log, appends status to
            # <basename>.monitor.log, and fires notifier hooks.  It blocks on
            # time.sleep() (0 CPU while idle) and runs at `nice -n 19` (+ a
            # self-nice) so it never competes with the compute ranks.  Opt
            # out with MB_MONITOR=0.  Killed by _mb_cleanup; also self-exits
            # when this wrapper's PID ($$) disappears.
            f'_monitor_pid=""\n'
            f'if [ "${{MB_MONITOR:-1}}" = "1" ] '
            f'&& command -v nice >/dev/null 2>&1 '
            f'&& [ -f mb_monitor.py ]; then\n'
            f'    nice -n 19 python mb_monitor.py '
            f'--out "$_out_file" --timing "$_scf_timing_log" '
            f'--log "{basename}.monitor.log" '
            f'--util "{basename}.util.csv" '
            f'--interval "${{MB_MONITOR_INTERVAL:-5}}" '
            f'--stall-heartbeat "${{MB_MONITOR_STALL_HEARTBEAT:-600}}" '
            f'--watch-pid $$ >/dev/null 2>&1 &\n'
            f'    _monitor_pid=$!\n'
            f'    _log INFO "monitor: pid=$_monitor_pid (nice 19, interval '
            f'${{MB_MONITOR_INTERVAL:-5}}s, quiet-when-stalled, util-sampling, '
            f'self-contained mb_monitor.py) '
            f'-> {basename}.monitor.log + {basename}.util.csv"\n'
            f'else\n'
            f'    _log INFO "monitor: not started (set MB_MONITOR=1; needs '
            f'nice + mb_monitor.py beside the job)"\n'
            f'fi\n'
            f"_t_start=$(date +%s.%N)\n"
            f"set +e\n"
            f'$_launch_cmd {script_name} | _mb_scf_tee "$_out_file" '
            f'"$_scf_timing_log"\n'
            f"_siesta_exit=${{PIPESTATUS[0]}}\n"
            f"set -e\n"
            f"_t_end=$(date +%s.%N)\n"
            f'_siesta_wall=$(awk -v a="$_t_start" -v b="$_t_end" '
            f"'BEGIN{{printf \"%.1f\", b-a}}')\n"
            # Reliable per-iteration metric = total wall / N_iters.  SIESTA's
            # OWN per-scf time in the .out is NOT trustworthy (it effectively
            # records only the first iteration), and the external per-line
            # stamps in .scf-timing.log are subject to Fortran stdout
            # buffering -- so the headline benchmark number is total/N
            # (slurm-integration.md § 11.0).  N = scf: iteration lines.
            f'if [ -f "$_scf_timing_log" ]; then\n'
            f'    _n_scf=$(wc -l < "$_scf_timing_log" | tr -d " ")\n'
            f'else\n'
            f'    _n_scf=0   # SIESTA crashed before any scf: output\n'
            f'fi\n'
            f'case "$_n_scf" in ""|*[!0-9]*) _n_scf=0 ;; esac\n'
            f'if [ "$_n_scf" -ge 1 ]; then\n'
            f'    _per_iter=$(awk -v t="$_siesta_wall" -v n="$_n_scf" '
            f"'BEGIN{{printf \"%.2f\", t/n}}')\n"
            f'    _log INFO "benchmark: SIESTA wall ${{_siesta_wall}}s / '
            f'${{_n_scf}} SCF iters = ${{_per_iter}}s/iter '
            f'(total/N -- the reliable metric)"\n'
            f'else\n'
            f'    _log INFO "benchmark: SIESTA wall ${{_siesta_wall}}s '
            f'(no SCF iterations parsed from $_scf_timing_log)"\n'
            f'fi\n'
            f"\n"
            f'if [ "$_siesta_exit" -ne 0 ]; then\n'
            f"    echo \"\"\n"
            f'    echo "===== SIESTA exited with code $_siesta_exit =====" >&2\n'
            f'    if grep -aq "propor: ERROR" "$_out_file" "$_runwrap_log" '
            f'2>/dev/null; then\n'
            f"        cat <<HINT >&2\n"
            f"\n"
            f"SIESTA crashed with 'propor: ERROR: IMAX = 0' during startup.\n"
            f"\n"
            f"IMAX=0 means one of SIESTA's per-orbital / per-rank tables\n"
            f"came out EMPTY.  It has three common causes -- check them in\n"
            f"this order, most fundamental first.  Do NOT just lower -np\n"
            f"reflexively: that masks a defective pseudopotential and gives\n"
            f"silently-wrong physics.\n"
            f"\n"
            f"1) DEFECTIVE / MISMATCHED PSEUDOPOTENTIAL  (check this FIRST)\n"
            f"   A pseudo with a null Kleinman-Bylander projector (ekb=0 on\n"
            f"   a whole l-channel) trips IMAX=0.  In $_out_file find each\n"
            f"   species' 'PSML: Kleinman-Bylander projectors' table and look\n"
            f"   for 'rc=  0.000010   Ekb=  0.000000' rows -- a valence\n"
            f"   element missing its p or d channel is broken.  Fix: replace\n"
            f"   that .psml with a vetted one (PseudoDojo; match the rest of\n"
            f"   your set's generator version + XC).  Screen the whole set:\n"
            f"     python -m molbuilder pseudo check <pseudo-dir>\n"
            f"\n"
            f"2) MPI RANK COUNT  (np IS a legitimate tunable here)\n"
            f"   If the pseudos are clean, some -np values leave trailing\n"
            f"   ranks empty in the orbital/projector distribution.  Retry\n"
            f"   at a different -np (lower; powers of 2 are safest):\n"
            f"     bash {basename}.run.sh -np 8\n"
            f"     bash {basename}.run.sh -np 4\n"
            f"   The same clean .fdf can fail at one -np and pass at another.\n"
            f"\n"
            f"3) ZERO / MISSING NET SPIN on an open-shell metal\n"
            f"   SpinPolarized with Spin.Total unset or 0 on a d/f-shell\n"
            f"   metal also triggers IMAX=0; set an initial Spin.Total.\n"
            f"   (molbuilder's preflight normally catches this pre-run.)\n"
            f"\n"
            f"Full SIESTA output: $_out_file\n"
            f"HINT\n"
            f'    elif grep -aq "ERROR\\|aborted\\|Stopping" '
            f'"$_out_file" "$_runwrap_log" 2>/dev/null; then\n'
            f'        echo "Other SIESTA error detected; check '
            f'$_out_file / $_runwrap_log for details." >&2\n'
            f"    fi\n"
            f'    exit "$_siesta_exit"\n'
            f"fi\n"
            f"\n"
            f'echo "SIESTA completed: $_launch_cmd {script_name} -> '
            f'$_out_file"\n'
        )
    else:
        launch_block = (
            f'_log INFO "resolved launch : {inner}"\n'
            f'if [ "$_dry_run" = 1 ]; then\n'
            f'    echo ""\n'
            f'    echo "===== molbuilder DRY RUN (no PySCF launch) ====="\n'
            f'    echo "  Resolved cmd : {inner}"\n'
            f'    echo "  Conda env    : ${{CONDA_DEFAULT_ENV:-?}}"\n'
            f'    echo "==============================================="\n'
            f'    _log INFO "dry-run complete; no PySCF launched"\n'
            f'    exit 0\n'
            f'fi\n'
            f"exec {inner}\n"
        )

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
        f"# IMPORTANT: this wrapper does NOT change cwd (see\n"
        f"# docs/config.md § 1).  Output artefacts (the runwrap log,\n"
        f"# -runN.out, .chk, trajectory XYZ, .molwatch.log,\n"
        f"# .spectra.json, ...) land in the CALLER'S current working\n"
        f"# directory.  Under sbatch this is ``SLURM_SUBMIT_DIR`` (the\n"
        f"# dir you ran sbatch from); under direct invocation it's\n"
        f"# wherever you ``cd``'d before running the wrapper.\n"
        f"#\n"
        f"# Convention: invoke from the project directory (the dir\n"
        f"# holding the .fdf / .py and where you want outputs to\n"
        f"# accumulate).  Do NOT add ``cd`` lines or modify the cwd\n"
        f"# from inside the generated .py / .fdf -- the Python side\n"
        f"# resolves every output path through ``_mb_outfile()``\n"
        f"# against the SCRIPT directory, but other code (PySCF's\n"
        f"# mol.log open(), geomeTRIC's optimize prefix) you might\n"
        f"# layer on top would write into the wrong place if you\n"
        f"# chdir to elsewhere.\n"
        f"#\n"
        f"set -euo pipefail\n"
        f"# Per docs/config.md § 1: the wrapper does NOT change cwd.\n"
        f"# SLURM lands the job in SLURM_SUBMIT_DIR by default; direct\n"
        f"# callers ``cd`` to the project dir before invoking.  The\n"
        f"# caller's cwd is the contract -- outputs (log, -runN.out)\n"
        f"# land where the wrapper was invoked.\n"
        f"\n"
        f"{env_activation}"
        f"{env_prefix}"
        f"{launch_block}"
        f"\n{_user_custom}\n"
    )


def _ship_monitor_script(dest_dir: Path) -> Path:
    """Copy the stdlib-only monitor (``molbuilder/monitor.py``) into
    ``dest_dir`` as ``mb_monitor.py``.

    GOAL: make the background monitor runnable by a generated job with
    the JOB's OWN python, from the working directory -- molbuilder is
    never installed and the backend env has no numpy/molbuilder, so the
    monitor cannot be reached as ``python -m molbuilder monitor``.  A
    verbatim copy of the stdlib-only module solves it (§ 11.0b, item F).
    Overwrites any existing copy so it stays in sync with the package.
    """
    from . import monitor as _monitor
    src = Path(_monitor.__file__)
    dst = Path(dest_dir) / "mb_monitor.py"
    try:
        shutil.copyfile(src, dst)
    except OSError as exc:
        raise WrapperError(
            f"could not ship mb_monitor.py to {dest_dir}: {exc}") from None
    return dst


def _build_mem_audit(script_path: Path, *,
                     gres: Optional[str],
                     env: Optional[str]) -> Optional[dict]:
    """Build the baked memory-model coefficients for the runtime
    estimate-vs-allocation audit (:func:`_siesta_mem_audit_block`).

    Returns None (no audit line) unless this is a **CPU** SIESTA ``.fdf``
    whose system actually parses: GPU jobs size memory from ``gpu.mem``,
    not this model.  The coefficients are ntasks-independent (the launcher
    plugs in the runtime rank count), so the estimate is computed once at
    ``ntasks=1`` purely to read the np-independent component sizes.
    Best-effort: any failure -> None (the wrapper just omits the line).
    """
    if script_path.suffix.lower() != ".fdf":
        return None
    if gres is not None:
        return None  # GPU job (explicit --gres)
    try:
        if env is None and _fdf_requests_gpu(script_path):
            return None  # GPU job (fdf requests ELPA-CUDA)
        from . import runtime_config as _rc
        project_dir = script_path.parent if script_path.parent.exists() else None
        scheduler = _rc.get_scheduler(project_dir=project_dir)
        mem_cfg = scheduler.get("mem_model") if scheduler else None
        from .siesta.memory import estimate_siesta_memory, MemModel
        model = MemModel.from_config(mem_cfg)
        est = estimate_siesta_memory(
            script_path, 1, model=model, psml_lib=script_path.parent)
        if est.n_orb <= 0:
            return None
        return {
            "fixed_gb": est.fixed_raw_gb,   # unrounded base+dense+mesh (B-4)
            "per_rank_gb": model.c_rank,
            "safety": model.safety,
            "extra_gb": model.extra_gb,
            "floor_gb": model.floor_gb,
            "cap_gb": model.node_mem_gb or 0.0,
        }
    except Exception:
        return None


def write_run_wrapper(script_path: Path, *,
                       env: Optional[str] = None,
                       mpi_np: Optional[int] = None,
                       omp_threads: Optional[int] = None,
                       max_memory_mb: Optional[int] = None,
                       time: Optional[str] = None,
                       gres: Optional[str] = None,
                       mem: Optional[str] = None,
                       cpus_per_task: Optional[int] = None,
                       exclusive: Optional[bool] = None,
                       emit_sbatch: bool = True) -> Path:
    """Render + write ``<basename>.run.sh`` next to ``script_path``.

    Returns the wrapper's path.  Sets executable bit (0o755) so the
    user can ``./my-job.run.sh`` directly.  Overwrites any existing
    wrapper.

    For ``.fdf`` scripts the file is parsed for ``NumberOfAtoms`` and
    that value is threaded into ``render_run_wrapper`` so the auto-mpi
    path can clamp ``mpi_np <= n_atoms`` (the propor IMAX=0 lower
    bound).  Parse-failure is treated as "unknown" and falls back to
    the unclamped behaviour rather than refusing to render.

    **SLURM submission layer** (slurm-integration.md § 15 B): when a
    ``scheduler`` block is configured (``get_scheduler`` non-None) and
    ``emit_sbatch`` is True, ALSO writes ``<basename>.sbatch`` -- the
    thin submission wrapper that ``sbatch``'s this same ``.run.sh``.
    The ``time``/``gres``/``mem``/``cpus_per_task``/``exclusive`` knobs
    feed that header (§ 6 value-source matrix); they are no-ops when no
    scheduler is configured (local/laptop users just get the ``.run.sh``,
    § 10).
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
        mem_audit=_build_mem_audit(script_path, gres=gres, env=env),
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

    # Ship the self-contained background monitor next to SIESTA jobs so
    # the wrapper can run it with the JOB's own python (the backend env
    # has no molbuilder/numpy; molbuilder is never installed -- it runs
    # from the repo dir only).  ``mb_monitor.py`` is a verbatim copy of
    # the stdlib-only molbuilder/monitor.py (§ 11.0b, item F).
    if script_path.suffix.lower() == ".fdf":
        _ship_monitor_script(script_path.parent)

    # --- SLURM submission layer (slurm-integration.md § 15 B) ---------
    # Emit <basename>.sbatch iff a scheduler is configured.  Resolution
    # of the per-job header values (ntasks/cpus/gpu) lives here because
    # only this layer knows both the .fdf (GPU request, n_atoms) and the
    # CLI overrides.  Skipped silently when no scheduler block exists
    # (today's behaviour for local/laptop users, § 10).
    if emit_sbatch:
        _maybe_write_sbatch(
            script_path, wrapper_path,
            mpi_np=mpi_np, time=time, gres=gres, mem=mem,
            cpus_per_task=cpus_per_task, exclusive=exclusive,
            env=env,
        )
    return wrapper_path


def _maybe_write_sbatch(script_path: Path,
                        wrapper_path: Path,
                        *,
                        mpi_np: Optional[int],
                        time: Optional[str],
                        gres: Optional[str],
                        mem: Optional[str],
                        cpus_per_task: Optional[int],
                        exclusive: Optional[bool],
                        env: Optional[str]) -> Optional[Path]:
    """Resolve the per-job header values and write ``<basename>.sbatch``
    when a ``scheduler`` block is configured; else return None.

    Resolution rules (slurm-integration.md § 6):
      * ``-n`` (ntasks) = the **MPI rank count** (``mpi_np``) for BOTH CPU
        and GPU jobs.  For GPU jobs ``--gres`` carries the **GPU count**,
        which is INDEPENDENT of the rank count: under the K-ranks-per-GPU
        load-balance model (§ 7.5.1), ranks may exceed GPUs (e.g. 8 ranks
        sharing 1 A100 via MPS -> ``-n 8 --gres=gpu:a100:1``).
      * Unset ``mpi_np`` falls back to 1 (the launcher still resolves the
        runtime rank count from ``SLURM_NTASKS``, § 7.3).
    """
    from . import runtime_config as _rc
    project_dir = script_path.parent if script_path.parent.exists() else None
    scheduler = _rc.get_scheduler(project_dir=project_dir)
    if scheduler is None:
        return None  # § 10: no scheduler -> emit only .run.sh

    suffix = script_path.suffix.lower()
    is_siesta = suffix == ".fdf"

    # Is this a GPU job?  Only SIESTA .fdf can be; honour an explicit
    # --env override that points away from GPU (mirrors the run-wrapper's
    # env_lookup_category logic).
    gpu = bool(is_siesta and env is None and _fdf_requests_gpu(script_path))
    gpu_type: Optional[str] = None
    gpu_count: Optional[int] = None
    if gres is not None:
        gpu_type, gpu_count = _parse_gres(gres)
        gpu = True  # an explicit --gres forces a GPU header

    if gpu:
        # ``--gres`` carries the GPU COUNT; ``-n`` (ntasks) is the MPI
        # RANK count.  These are INDEPENDENT -- under the K-ranks-per-GPU
        # load-balance model (§ 7.5.1) ranks may exceed GPUs (e.g. 8 ranks
        # sharing 1 A100 via MPS).  So ntasks = mpi_np, NOT the GPU count.
        if gpu_count is None:
            gpu_count = 1   # default to 1 GPU when --gres count not given
        ntasks = mpi_np if (mpi_np and mpi_np >= 1) else gpu_count
    else:
        # CPU job: ntasks = mpi_np (the rank count).  When unset, 1 is a
        # safe header floor -- under sbatch the launcher reads
        # SLURM_NTASKS (§ 7.3), so the user controls scale via -n.
        ntasks = mpi_np if (mpi_np and mpi_np >= 1) else 1

    return write_sbatch(
        script_path, scheduler,
        ntasks=ntasks,
        cpus_per_task=cpus_per_task,
        time=time,
        gpu=gpu,
        gpu_count=gpu_count,
        gpu_type=gpu_type,
        mem=mem,
        exclusive=exclusive,
    )


# --------------------------------------------------------------------- #
#  SLURM .sbatch submission layer (docs/protocols/slurm-integration.md)  #
# --------------------------------------------------------------------- #


_GRES_RE = re.compile(r"^(?:gpu:)?(?P<type>[A-Za-z0-9_.]+):(?P<count>\d+)$")


def _parse_gres(gres: str) -> Tuple[Optional[str], int]:
    """Parse a ``--gres`` spec into ``(gpu_type, count)``.

    Accepts ``gpu:a100:2``, ``a100:2``, or a bare count ``2`` (=> type
    unspecified, caller falls back to ``scheduler.gpu.default_type``).
    Raises :exc:`WrapperError` on anything else so a typo'd CLI value
    fails at generate time, not after a job queues.
    """
    g = gres.strip()
    if g.isdigit():
        return None, int(g)
    m = _GRES_RE.match(g)
    if not m:
        raise WrapperError(
            f"invalid --gres value {gres!r}; expected "
            f"``[gpu:]<type>:<count>`` (e.g. ``gpu:a100:2``) or a bare "
            f"count."
        )
    return m.group("type"), int(m.group("count"))


def render_sbatch(script_path: Path,
                  scheduler: Mapping[str, Any],
                  *,
                  ntasks: int,
                  cpus_per_task: Optional[int] = None,
                  time: Optional[str] = None,
                  gpu: bool = False,
                  gpu_count: Optional[int] = None,
                  gpu_type: Optional[str] = None,
                  mem: Optional[str] = None,
                  exclusive: Optional[bool] = None) -> str:
    """Render the ``<basename>.sbatch`` submission script.

    The thin two-layer model (slurm-integration.md § 3, § 5): an
    ``#SBATCH`` header that allocates resources, then a one-line body
    that delegates to the UNCHANGED launcher
    ``bash <basename>.run.sh "$@"``.  The launcher still owns env
    activation + the ``mpirun`` launch; the ``.sbatch`` never
    re-implements ``module load`` / ``source activate`` (§ 2 principle 3).

    Value sourcing (§ 6): stable site directives come from ``scheduler``;
    per-job values (``ntasks``/``cpus_per_task``/``time``/``mem``/GPU
    type+count/``exclusive``) are resolved by the CALLER (CLI flag →
    ``.fdf`` → config default) and passed in here already resolved.

    Args:
      scheduler: the resolved block from
        :func:`runtime_config.get_scheduler` (``directives`` carry a
        validated ``partition``+``qos``).
      ntasks: ``-n``.  CPU jobs: the MPI rank count.  GPU jobs: 1 rank
        per GPU (§ 7.5.1) -- pass ``gpu_count``.  Under sbatch the
        launcher reads ``SLURM_NTASKS`` (runwrap line ~1367), so this
        ``-n`` and ``mpirun -np`` agree by construction (§ 7.3).
      gpu: emit the ``--gres`` + ``--gres-flags=enforce-binding`` lines
        and route to ``scheduler.gpu.partition`` (§ 7.4, § 8).
      exclusive: ``--exclusive``.  Defaults to ``scheduler.gpu.exclusive``
        for GPU jobs (None => use the config value); always off for CPU.
    """
    if not isinstance(ntasks, int) or ntasks < 1:
        raise WrapperError(
            f"render_sbatch: ntasks must be a positive int; got {ntasks!r}."
        )

    basename = Path(script_path).stem
    if not _SAFE_WRAPPER_NAME_RE.fullmatch(basename):
        raise WrapperError(
            f"unsafe script basename for sbatch emission: {basename!r}."
        )

    directives = dict(scheduler.get("directives") or {})
    gpu_cfg    = dict(scheduler.get("gpu") or {})
    defaults   = dict(scheduler.get("defaults") or {})

    partition = directives.get("partition")
    qos       = directives.get("qos")
    # Refuse-to-emit (§ 10): get_scheduler already guarantees these, but
    # render_sbatch may be called directly -- never emit a header that
    # won't allocate.
    if not partition or not qos:
        raise WrapperError(
            "render_sbatch: scheduler.directives.partition + qos are "
            "required (slurm-integration.md § 10).  Use "
            "runtime_config.get_scheduler() which enforces this."
        )

    if gpu:
        # GPU jobs route to gpu.partition when set; else the same
        # partition (on Sol `public` carries the GPU nodes -- § 7.4).
        partition = gpu_cfg.get("partition") or partition
        gpu_type  = gpu_type or gpu_cfg.get("default_type")
        if not gpu_type:
            raise WrapperError(
                "render_sbatch: GPU job but no gpu type resolved; set "
                "scheduler.gpu.default_type or pass --gres <type>:<n> "
                "(slurm-integration.md § 6)."
            )
        if gpu_count is None:
            gpu_count = ntasks  # 1 rank per GPU (§ 7.5.1)
        if exclusive is None:
            exclusive = bool(gpu_cfg.get("exclusive", False))
    else:
        exclusive = False  # CPU jobs never request a whole node here

    # Per-job values: caller arg wins, else config default.
    cpus = cpus_per_task if cpus_per_task is not None \
        else defaults.get("cpus_per_task")
    walltime = time if time is not None else defaults.get("time")
    # mem: caller arg wins; else for GPU jobs prefer ``gpu.mem`` (Sol's GPU
    # default is a tight 24 GB/GPU), else ``defaults.mem``.  CPU jobs use
    # ``defaults.mem`` (null => the generous partition default -- do NOT cap
    # a 64-rank CPU job at a small total).
    mem_comment: Optional[str] = None
    if mem is not None:
        memory = mem
    elif gpu:
        memory = gpu_cfg.get("mem") or defaults.get("mem")
    else:
        memory = defaults.get("mem")
        # CPU SIESTA job with no explicit / config mem: estimate it from
        # the problem size so a large job doesn't inherit the tiny
        # partition default and OOM (the np=64 BDT-Au lesson -- 64 ranks
        # needed ~250 GB but the bare default killed it).  Best-effort:
        # estimation NEVER blocks emission; on any failure we fall back
        # to the partition default.  See slurm-integration.md (mem model)
        # and molbuilder/siesta/memory.py.
        if memory is None and Path(script_path).suffix.lower() == ".fdf":
            try:
                from .siesta.memory import (estimate_siesta_memory,
                                            MemModel)
                _model = MemModel.from_config(scheduler.get("mem_model"))
                _est = estimate_siesta_memory(
                    script_path, ntasks, model=_model,
                    psml_lib=Path(script_path).parent)
                # Only emit when we actually parsed a system; a stub/
                # unparseable .fdf yields n_orb=0 -> floor, which is
                # meaningless.  Fall back to the partition default there.
                if _est.n_orb > 0:
                    memory = f"{_est.request_gb}G"
                    mem_comment = (
                        f"# --mem auto-estimated from problem size "
                        f"({_est.breakdown_str()}); tune via "
                        f"scheduler.mem_model or override with --mem.")
            except Exception:
                memory = None   # fall back to the partition default

    site = "asu-sol" if partition == "public" else "custom"
    lines: List[str] = [
        "#!/bin/bash",
        f"# === molbuilder sbatch header (scheduler: slurm; site: {site}) ===",
        "# Generated by `molbuilder run` from the `scheduler` config block.",
        "# Authoritative design: docs/protocols/slurm-integration.md.",
        "# Submit with:  cd <projdir>; sbatch "
        f"{basename}.sbatch   (NOT bash -- § 7.8)",
        "#",
        f"#SBATCH -J {basename}",
        "#SBATCH -N 1",
        f"#SBATCH -n {ntasks}",
    ]
    if cpus is not None:
        lines.append(f"#SBATCH -c {cpus}")
    if walltime:
        lines.append(f"#SBATCH -t {walltime}")
    lines.append(f"#SBATCH -p {partition}")
    lines.append(f"#SBATCH -q {qos}")
    if gpu:
        lines.append(f"#SBATCH --gres=gpu:{gpu_type}:{gpu_count}")
        # Nudge SLURM toward co-locating the CPU cores with the GPU's
        # NUMA node; the launcher still does the authoritative per-rank
        # runtime bind (§ 7.5.1).
        lines.append("#SBATCH --gres-flags=enforce-binding")
    if exclusive:
        lines.append("#SBATCH --exclusive")
    if memory:
        if mem_comment:
            lines.append(mem_comment)
        lines.append(f"#SBATCH --mem={memory}")
    lines.append("#SBATCH -o slurm.%j.out")
    lines.append("#SBATCH -e slurm.%j.err")
    if directives.get("mail_type"):
        lines.append(f"#SBATCH --mail-type={directives['mail_type']}")
    if directives.get("mail_user"):
        lines.append(f"#SBATCH --mail-user=\"{directives['mail_user']}\"")
    if directives.get("export"):
        lines.append(f"#SBATCH --export={directives['export']}")

    body = (
        "\n"
        "# SLURM lands us in SLURM_SUBMIT_DIR = the project dir; the\n"
        "# launcher never cd's (config.md § 1).  --export=NONE means a\n"
        "# clean env, so the launcher's `module load mamba` + activation\n"
        "# are load-bearing (§ 7.1).  \"$@\" forwards --cold / --continue.\n"
        f"bash {basename}.run.sh \"$@\"\n"
    )
    return "\n".join(lines) + "\n" + body


def write_sbatch(script_path: Path,
                 scheduler: Mapping[str, Any],
                 **kwargs: Any) -> Path:
    """Render + write ``<basename>.sbatch`` next to ``script_path``.

    Returns the path.  Mode 0o644 (a submission script, not directly
    executed -- you ``sbatch`` it, you don't ``./`` it).  Validated
    through ``bash -n`` like the wrapper before it hits disk.
    """
    script_path = Path(script_path).resolve()
    text = render_sbatch(script_path, scheduler, **kwargs)
    _validate_rendered_wrapper(text, script_path)
    sbatch_path = script_path.parent / (script_path.stem + ".sbatch")
    sbatch_path.write_text(text)
    sbatch_path.chmod(0o644)
    return sbatch_path


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
    "render_sbatch",
    "write_sbatch",
]
