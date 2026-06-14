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

import re
from pathlib import Path
from typing import Optional

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
      * ``pyscf``: ``.chk`` (PySCF's chkfile + the geomeTRIC
        chkpoint when present).

    Backups land in
    ``<basename>-restart-aside-<UTC-timestamp>/`` so the user can
    inspect / recover the prior state if needed.  Deleting that
    directory is a manual user action -- we never auto-delete the
    user's prior results.
    """
    if engine == "siesta":
        exts = ("DM", "CG", "XV", "LWF", "ZM", "Bonds", "PARTIAL", "EIG")
    elif engine == "pyscf":
        exts = ("chk",)
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
            f'_warm_label=$(awk \'BEGIN{{IGNORECASE=1}} '
            f'/^[[:space:]]*SystemLabel[[:space:]]+/ '
            f'{{print $2; exit}}\' "{basename}.fdf" 2>/dev/null)\n'
            f'[ -z "$_warm_label" ] && _warm_label="{basename}"\n'
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
        label_extract = (
            f'_warm_label=$(awk -F\'["\\\'"]\' \'/^JOB[[:space:]]*=/ '
            f'{{print $2; exit}}\' "{basename}.py" 2>/dev/null)\n'
            f'[ -z "$_warm_label" ] && _warm_label="{basename}"\n'
        )
        glob_pieces = " ".join(
            f'"$_warm_label.{ext}" {basename}.{ext}'
            for ext in exts
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
        warmstart_exts = ("DM", "CG", "XV", "LWF", "ZM")
        warmstart_test_pieces = []
        for ext in warmstart_exts:
            warmstart_test_pieces.append(f'[ -e "$_warm_label.{ext}" ]')
            warmstart_test_pieces.append(f'[ -e "{basename}.{ext}" ]')
        warmstart_test = " || ".join(warmstart_test_pieces)
        warmstart_listing = " ".join(
            f"{basename}.{ext}" for ext in warmstart_exts
        )
        constraint_detection = (
            f'_constraints="(no Geometry.Constraints block -- all atoms free)"\n'
            f'_fdf_path="{script_name}"\n'
            f'if [ -e "$_fdf_path" ] && grep -qiE \'^[[:space:]]*%block[[:space:]]+Geometry\\.Constraints\' "$_fdf_path"; then\n'
            f'    _ncon_lines=$(awk \'BEGIN{{IGNORECASE=1;in_b=0;n=0}} /^[[:space:]]*%block[[:space:]]+Geometry\\.Constraints/{{in_b=1;next}} /^[[:space:]]*%endblock[[:space:]]+Geometry\\.Constraints/{{in_b=0;next}} in_b && /^[[:space:]]*position/{{n++}} END{{print n}}\' "$_fdf_path")\n'
            f'    _ncon_indices=$(awk \'BEGIN{{IGNORECASE=1;in_b=0;n=0}} /^[[:space:]]*%block[[:space:]]+Geometry\\.Constraints/{{in_b=1;next}} /^[[:space:]]*%endblock[[:space:]]+Geometry\\.Constraints/{{in_b=0;next}} in_b && /^[[:space:]]*position/{{for(i=2;i<=NF;i++) if($i~/^[0-9]+$/) n++}} END{{print n}}\' "$_fdf_path")\n'
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
    if engine == "siesta":
        label_extract_unconditional = (
            f'_warm_label=$(awk \'BEGIN{{IGNORECASE=1}} '
            f'/^[[:space:]]*SystemLabel[[:space:]]+/ '
            f'{{print $2; exit}}\' "{script_name}" 2>/dev/null)\n'
            f'[ -z "$_warm_label" ] && _warm_label="{basename}"\n'
        )
    else:                              # pyscf
        label_extract_unconditional = (
            f'_warm_label=$(awk -F\'["\\\'"]\' \'/^JOB[[:space:]]*=/ '
            f'{{print $2; exit}}\' "{script_name}" 2>/dev/null)\n'
            f'[ -z "$_warm_label" ] && _warm_label="{basename}"\n'
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
                        n_atoms: Optional[int] = None) -> str:
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

    caps = get_capabilities()
    target_env = env if env is not None else caps.env_for_category(category)
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
        siesta_args_block = (
            _continue_force_args_parser("SIESTA wrapper")
            + f"# --- SIESTA-specific argument parsing -----------\n"
            f"# Override the generation-time -np with: ``-np N`` or\n"
            f"# ``MB_NP=N``.  Useful for retrying after a propor crash\n"
            f"# (see diagnostic at the bottom of this wrapper).\n"
            f"_mpi_np_default={resolved_mpi}\n"
            f'_mpi_np="${{MB_NP:-$_mpi_np_default}}"\n'
            f'while [ $# -gt 0 ]; do\n'
            f'    case "$1" in\n'
            f"        -np|--np)\n"
            f'            if [ $# -lt 2 ]; then\n'
            f'                echo "ERROR: -np requires a value" >&2\n'
            f"                exit 1\n"
            f"            fi\n"
            f'            _mpi_np="$2"; shift 2 ;;\n'
            f"        -h|--help)\n"
            f'            cat <<USAGE\n'
            f'Usage: bash $(basename "$0") [--continue|-c] [--force|-f] [--cold] [-np N] [-h]\n'
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
            f"  -h               this help.\n"
            f"\n"
            f"Environment variables:\n"
            f"  MB_NP=N  same as -np N (useful for SLURM/PBS scripts:\n"
            f"           ``export MB_NP=\\$SLURM_NTASKS`` then bash this).\n"
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
            f"\n"
            + _run_index_resolver(basename)
            + _cold_restart_aside_block(basename, engine="siesta")
            + _runtime_status_block(basename, engine="siesta",
                                     script_name=script_name)
        )

        env_prefix = (
            siesta_args_block
            + f"# MPI rank count: $_mpi_np (default: $_mpi_np_default, "
            f"source: {mpi_source})\n"
            f"{clamp_note}"
            f"# Thread / BLAS pinning.\n"
            f"#   * OMP_NUM_THREADS ({omp_source}): SIESTA mainline is\n"
            f"#     mostly not OMP-aware, so pure MPI with OMP=1 is the\n"
            f"#     standard recipe.  Bump only with an OMP-compiled\n"
            f"#     SIESTA build (hybrid MPI+OMP).\n"
            f"#   * BLAS pinned to 1 per rank so OMP * BLAS doesn't\n"
            f"#     oversubscribe.\n"
            f"export OMP_NUM_THREADS={resolved_omp}\n"
            f"export MKL_NUM_THREADS=1\n"
            f"export OPENBLAS_NUM_THREADS=1\n"
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
            f'if [ "$_has_mpi" = 1 ]; then\n'
            f'    _launch_cmd="mpirun -np $_mpi_np siesta"\n'
            f'    if [ "$_has_omp" = 1 ]; then\n'
            f'        _launch_note="hybrid MPI+OMP ($_mpi_np ranks x {resolved_omp} OMP threads)"\n'
            f'    else\n'
            f'        _launch_note="pure MPI ($_mpi_np ranks; OMP setting irrelevant to this binary)"\n'
            f'    fi\n'
            f'elif [ "$_has_omp" = 1 ]; then\n'
            f'    _launch_cmd="siesta"\n'
            f'    _launch_note="OMP-only build ({resolved_omp} threads)"\n'
            f'elif [ -z "$_siesta_par" ]; then\n'
            f'    _launch_cmd="mpirun -np $_mpi_np siesta"\n'
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
            f'echo "  Threading     : OMP_NUM_THREADS={resolved_omp}, '
            f'OPENBLAS=1, MKL=1"\n'
            # ---- Mode + constraints (post-cold, post-run-index) ----
            # Surfaces the silent-warm-restart class explicitly so the
            # user can see whether the engine is starting clean,
            # resuming from prior state, or being asked to honor
            # frozen-atom constraints.  Both lines are read from the
            # on-disk script + restart files at runtime so a manual
            # .fdf edit shows the EDITED state, not the generation-
            # time snapshot (the "what you see is what runs" rule).
            f'echo "  Mode          : $_mode"\n'
            f'echo "  Constraints   : $_constraints"\n'
            f'echo "  Command       : $_launch_cmd {script_name} > $_out_file"\n'
            f'echo "  Stdout        : $_out_file (live; tail -f to follow)"\n'
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

    # Conda env activation block.  Three paths so the wrapper Just
    # Works in the common cases:
    #
    # 1. Already in the right env (CONDA_DEFAULT_ENV == target_env):
    #    skip activation, run directly.  Lets the user activate
    #    interactively + invoke the wrapper without double-init.
    # 2. conda on PATH: source the conda.sh hook + activate.  Full
    #    env-setup (PATH, LD_LIBRARY_PATH, env-specific hooks like
    #    CUDA bootstraps) -- more robust than `conda run` for MPI
    #    launchers that can mishandle the `--no-capture-output`
    #    pipe redirection.
    # 3. conda not on PATH: print a clear error message naming the
    #    target env + how to install conda; exit 1.
    #
    # This is the "hybrid" pattern: catches the common cases, gives
    # a real error message instead of a cryptic "command not found".
    env_activation = (
        f"# --- Activate conda env ({target_env}) ----------------------\n"
        f'if [ "${{CONDA_DEFAULT_ENV:-}}" = "{target_env}" ]; then\n'
        f"    : # already in the target env -- nothing to do\n"
        f"elif command -v conda >/dev/null 2>&1; then\n"
        f'    _conda_base="$(conda info --base 2>/dev/null)"\n'
        f'    if [ -z "$_conda_base" ] || [ ! -f "$_conda_base/etc/profile.d/conda.sh" ]; then\n'
        f"        echo \"ERROR: conda is on PATH but conda.sh not found; \"\\\n"
        f"             \"reinstall conda or set CONDA_PREFIX manually.\" >&2\n"
        f"        exit 1\n"
        f"    fi\n"
        f'    # shellcheck disable=SC1091\n'
        f'    source "$_conda_base/etc/profile.d/conda.sh"\n'
        f"    conda activate {target_env}\n"
        f"else\n"
        f"    echo \"ERROR: conda not on PATH; this wrapper needs the \"\\\n"
        f"         \"'{target_env}' env activated.  Either:\" >&2\n"
        f"    echo \"  * install Miniconda + create the env: \"\\\n"
        f"         \"see docs/README_install.md\" >&2\n"
        f"    echo \"  * or pre-activate it: \"\\\n"
        f"         \"conda activate {target_env} && bash $0\" >&2\n"
        f"    exit 1\n"
        f"fi\n"
        f"\n"
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
    return (
        f"#!/usr/bin/env bash\n"
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
    )


def write_run_wrapper(script_path: Path, *,
                       env: Optional[str] = None,
                       mpi_np: Optional[int] = None,
                       omp_threads: Optional[int] = None,
                       max_memory_mb: Optional[int] = None) -> Path:
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
    )
    # Use stem + ".run.sh" rather than ``.with_suffix(".run.sh")``: the
    # latter REPLACES only the last suffix, so ``job.spectra.py`` would
    # become ``job.run.sh`` and lose the "spectra" tag.  We want
    # ``job.spectra.run.sh``.
    wrapper_path = script_path.parent / (script_path.stem + ".run.sh")
    wrapper_path.write_text(text)
    wrapper_path.chmod(0o755)
    return wrapper_path


__all__ = [
    "WrapperError",
    "render_run_wrapper",
    "write_run_wrapper",
]
