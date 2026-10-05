"""The molwatch log's lines -- header, step block, footer -- and ONE reader each.

The ``.molwatch.log v1`` format is molbuilder's own: `trajectory_log`
writes it (the PySCF deck imports its writer), and this is the grammar every
reader reads it with -- the reading pass (`molwatch_reader`, which the
registered parser builds Frames from and the monitor reads a running job
with), the PySCF parser's sibling-log enrichment, and the cheap ending scan
(`_run_ending`).

**Stdlib only, and travels** (`configuration.md`'s rule for the monitor's
modules): it goes beside every job (`runwrap.MONITOR_COMPANIONS`), so the
monitor reads a PySCF run's progress with the lines the Results tab reads it
with, and what it imports of ours -- the physical constants, the PySCF end
lines -- travels with it and is imported two ways.  It is `molwatch.py`'s grammar
half, split out on 2026-09-26 for that reason -- the SIESTA family's table
(`siesta_grammar`) was split from its parser the same way.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

try:                                        # inside molbuilder
    from ...constants import HARTREE_BOHR_EV_ANGSTROM_ASE, HARTREE_EV
    from ...pyscf.end_lines import FOOTER_CONCLUDED, FOOTER_ERROR
except ImportError:                         # beside a job, in mb_monitor.pyz
    from constants import HARTREE_BOHR_EV_ANGSTROM_ASE, HARTREE_EV
    from end_lines import FOOTER_CONCLUDED, FOOTER_ERROR

# ---- The header ---------------------------------------------------------------
HEADER = re.compile(r"^#\s*molwatch\s+trajectory\s+log", re.IGNORECASE)
ENGINE = re.compile(r"^#\s*engine:\s*(\S+)", re.IGNORECASE)
RUNTIME = re.compile(r"^#\s*runtime\.([a-zA-Z_][a-zA-Z0-9_]*):\s*(.*)$")
#: One segment of a convergence key -- DIGIT-FIRST INCLUDED, because the
#: nested form's first segment is a stage token and those lead with the
#: zero-padded ordinal (``01_coarse``, `job-contracts.md` § 6.3).  The
#: identifier-shaped ``[a-zA-Z_]`` head that stood in the parser was written
#: against the ``stage1.<leaf>`` example in the emitter's comment, so every
#: real staged header parsed to an EMPTY target dict and the Results card said
#: "not found in source" over eight present lines (2026-08-19).  The ONE
#: spelling of the key grammar: the parser's section rule uses it too.
CONV_KEY = r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)?"
#: ``# convergence.<leaf>: <v>`` (single-stage) or
#: ``# convergence.<stage>.<leaf>: <v>`` (staged, #534).
CONVERGENCE = re.compile(r"^#\s*convergence\.(" + CONV_KEY + r"):\s*(.*)$")
#: ``# frozen_atoms: 0 3`` -- the atoms the run holds, 0-based in the
#: structure's own order.  The progress log is the run's own output, so it
#: states them as a SIESTA run's ``.out`` does in its constraints echo,
#: and its reader reads them as its own content (`model/parse.md`
#: § 5.3).  Written by prep's preview and by the script's writer
#: (`trajectory_log`); absent when the run holds nothing.
FROZEN_ATOMS = re.compile(r"^#\s*frozen_atoms:\s*(.*)$")

# ---- The step block -------------------------------------------------------------
BLOCK_BEGIN = re.compile(r"====\s*molwatch\s+step\s+(\d+)\s+begin\s*====")
BLOCK_END = re.compile(r"====\s*molwatch\s+step\s+(\d+)\s+end\s*====")
#: The block's one-value lines, as `trajectory_log.emitter` writes them.
STEP_INDEX = "step_index:"
KIND = "kind:"
ENERGY = "energy (eV):"
MAX_FORCE = "max_force (eV/Ang):"
WALL_TIME = "wall_time:"
SCF_HISTORY_BEGIN = "scf_history begin"
SCF_HISTORY_END = "scf_history end"
#: The ``kind`` of the step-0 block the writer puts down before the run
#: starts (`trajectory_log`: prep seeds it, the emitter writes it on
#: construction) -- a preview of the input, not a step the run took.
PREVIEW_KIND = "initial_preview"

# ---- The footer -----------------------------------------------------------------
#: Rendered from the writer's own spelling (`pyscf/end_lines`).
ERROR = re.compile("^" + re.escape(FOOTER_ERROR) + r"\s*(.+)$", re.IGNORECASE)
CONCLUDED = re.compile("^" + re.escape(FOOTER_CONCLUDED) + r"\s*(.+)$",
                       re.IGNORECASE)


def maybe_float(token: str) -> Optional[float]:
    """A token as a float; ``None`` for the literal ``None`` / ``null``, and
    for anything that is not a number."""
    if token == "None" or token == "null":
        return None
    try:
        return float(token)
    except ValueError:
        return None


def field_value(line: str) -> Optional[float]:
    """The number after a block line's ``<name>:`` -- ``None`` when absent."""
    head, sep, rest = line.strip().partition(":")
    return maybe_float(rest.strip()) if sep else None


#: The one SCF phase of a run this log records -- the key its
#: :func:`scf_criteria` are stated under, beside SIESTA's ``periodic`` and
#: ``negf`` (`web/trajectory.md` § 3).
SCF_PHASE = "scf"


def scf_criteria(runtime_info: Dict[str, Any]) -> Dict[str, Any]:
    """What the SCF had to reach, in the shape every engine states it in
    (`web/trajectory.md` § 3): ``{SCF_PHASE: {residual: {tolerance, unit,
    required}}}`` -- ``{}`` when the log states neither tolerance.

    PySCF converges when the energy change is below ``conv_tol`` AND the
    orbital-gradient norm below ``conv_tol_grad`` (``scf.hf.kernel``), so
    both are required.  They are the values the deck read back off the
    solver (``# runtime.scf_conv_tol`` / ``scf_conv_tol_grad``, in Hartree),
    and are stated here in eV, the unit of the ``dE`` and ``|g|`` the step
    blocks carry.
    """
    def number(key: str) -> Optional[float]:
        try:
            return float(runtime_info[key])
        except (KeyError, TypeError, ValueError):
            return None

    energy, gradient = number("scf_conv_tol"), number("scf_conv_tol_grad")
    # AN UNSET GRADIENT TOLERANCE IS sqrt(conv_tol) -- PySCF's own rule --
    # and the log says so on its source line.  Logs written before
    # 2026-09-27 state the value beside that line as 0 or None: the deck's
    # parameters record wrote over the value it had read back.  A 0 with no
    # source line is the configuration's "unset" too -- no norm is below 0.
    derived = str(runtime_info.get("scf_conv_tol_grad_source") or ""
                  ).startswith("derived")
    if energy is not None and (derived or (gradient is not None
                                           and gradient <= 0)):
        gradient = energy ** 0.5
    out: Dict[str, Any] = {}
    for residual, tolerance in (("dE", energy), ("|g|", gradient)):
        if tolerance is not None:
            # PySCF states its SCF tolerances in Hartree, and the step blocks
            # carry the SCF's residuals in eV.
            out[residual] = {"tolerance": tolerance * HARTREE_EV,
                             "unit": "eV", "required": True}
    return {SCF_PHASE: out} if out else {}


def parse_convergence_line(line: str, targets: Dict[str, Any]) -> bool:
    """Apply one ``# convergence.<key>: <value>`` header line to
    ``targets`` and return whether the line matched.

    THE one reader of the convergence-header grammar.  The molwatch parser's
    reading pass (`molwatch_reader`) delegates here, and so does the PySCF
    trajectory parser's sibling-log enrichment -- the private copy the
    PySCF parser once kept "to avoid coupling" was letter-first and flat-only,
    so a staged header read as EMPTY on one path while the other read it fine
    (2026-08-19).  Coupling to the format's owner is the point.

    Both header shapes land as the callers expect (#534):

      * flat   ``convergence.<leaf>``          -> ``targets[<leaf>]``
      * nested ``convergence.<stage>.<leaf>``  -> ``targets[<stage>][<leaf>]``

    Stamps ``source = "molwatch_header"`` on the first hit.
    """
    m = CONVERGENCE.match(line)
    if not m:
        return False
    full_key, val = m.group(1), m.group(2).strip()
    targets.setdefault("source", "molwatch_header")

    def _coerce(s):
        if s == "None" or s == "null":
            return None
        if s in ("True", "False"):
            return s == "True"
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            pass
        return s

    if "." in full_key:
        stage_name, leaf_key = full_key.split(".", 1)
        stage_bucket = targets.setdefault(stage_name, {})
        if not isinstance(stage_bucket, dict):
            # Defensive: a flat-shape run that reused a stage-name as a
            # leaf key (hand-edited file) is not silently clobbered.
            return True
        stage_bucket[leaf_key] = _coerce(val)
        return True
    targets[full_key] = _coerce(val)
    return True


def parse_frozen_atoms_line(line: str) -> Optional[List[int]]:
    """The atoms a ``# frozen_atoms:`` header line says the run holds,
    0-based and sorted -- or ``None`` for any other line."""
    m = FROZEN_ATOMS.match(line)
    if m is None:
        return None
    try:
        return sorted(int(t) for t in m.group(1).split())
    except ValueError:
        return None


def parse_runtime_line(line: str, runtime_info: Dict[str, Any]) -> bool:
    """Apply one ``# runtime.<key>: <value>`` header line to
    ``runtime_info`` and return whether the line matched.

    **THE one reader of the runtime-header grammar**, and the sibling of
    :func:`parse_convergence_line` -- same shape, same reason.  The FORMAT is
    shared on purpose: the /spectra script writers and the Build SIESTA
    writers use IDENTICAL lines (`molbuilder.runtime_info` owns the write
    side), so a `.molwatch.log` and a `.out` carrying the same header are read
    by this one function.

    Coercion, in order: ``None`` -> ``None``; ``True`` / ``False`` ->
    ``bool``; anything ``int()`` accepts -> ``int``; otherwise the raw
    string.  Floats stay strings, which is the behaviour both former copies
    had.
    """
    m = RUNTIME.match(line)
    if not m:
        return False
    key, val = m.group(1), m.group(2).strip()
    if val == "None":
        runtime_info[key] = None
    elif val in ("True", "False"):
        runtime_info[key] = (val == "True")
    else:
        try:
            runtime_info[key] = int(val)
        except ValueError:
            runtime_info[key] = val
    return True


def parse_conclusion_line(line: str, out: Dict[str, Any]) -> bool:
    """Apply one ``# error:`` / ``# concluded:`` FOOTER line to ``out``
    and return whether the line matched.

    **THE one reader of the conclusion-footer grammar**, and the third sibling
    of :func:`parse_convergence_line` and :func:`parse_runtime_line`.

    **Error outranks concluded, and the LAST error wins.**  A log is appended
    across attempts, so a later ``# error:`` describes a later attempt; a
    ``# concluded:`` after an error does not un-fail the run, because the
    error is the more specific claim.  Callers scan in file order and let this
    decide.
    """
    m = ERROR.match(line)
    if m:
        out["run_state"] = "stopped"
        out["error_message"] = m.group(1).strip()
        return True
    if CONCLUDED.match(line):
        if out.get("run_state") != "stopped":
            out["run_state"] = "ended"
        return True
    return False


def read_conclusion(path) -> Dict[str, Any]:
    """What a molwatch log's FOOTER states -- ``run_state`` and, after an
    error, ``error_message`` -- read over the whole file in order through
    :func:`parse_conclusion_line`; ``{}`` when it carries no footer."""
    out: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line[:1] == "#":
                parse_conclusion_line(line.rstrip("\n"), out)
    return out


def scan_conclusion(path) -> str:
    """The run-state a molwatch log's FOOTER states, without building a
    Trajectory -- ``"running"`` when it carries no footer.

    The cheap door onto :func:`parse_conclusion_line`, for a caller that
    wants only how the run ended (`model/parse.md` § 2b) -- what
    ``_run_ending.ending_of`` asks of a progress log.  A status probe over a
    whole directory must not full-parse to reach one string: doing that is
    what made ``run_status`` create and grow a ``.parse.log`` beside every
    molwatch log it looked at, on every Watch poll.

    **Every line, in file order**, because the grammar's rule is *error
    outranks concluded and the LAST error wins* -- a tail-only read would
    answer ``"ended"`` for a log whose earlier attempt failed.  Only lines
    opening with ``#`` are offered, which is all either pattern can match.
    """
    return read_conclusion(path).get("run_state") or "running"


#: The ``scf_history`` header of a log whose orbital-gradient norm was
#: written as norm x Hartree/Bohr->eV/Ang, and the factor that makes it eV:
#: the norm is an energy (Hartree over dimensionless orbital rotations), so
#: eV is norm x Hartree->eV = old x (HARTREE_EV / HARTREE_BOHR_EV_ANGSTROM_ASE),
#: both from `constants`, which travels in the bundle beside this module.
OLD_GNORM_HEADER = "gnorm(eV/Ang)"
OLD_GNORM_TO_EV = HARTREE_EV / HARTREE_BOHR_EV_ANGSTROM_ASE


def scf_history_row(line: str) -> Optional[Dict[str, Any]]:
    """One ``scf_history`` row -- ``cycle energy delta_E gnorm ddm
    [wall_time]`` -- as a dict, or ``None`` when the line is not one.

    The optional sixth column is the emitter's epoch when the cycle finished;
    older logs (before 2026-06-20) have five.  It keeps its on-disk name and
    is read as ``wall_clock_s``, because it is an epoch where SIESTA's
    same-position value is an elapsed time (`model/parse.md` § 2a).
    """
    parts = line.split()
    if len(parts) < 5 or line.lstrip().startswith("#"):
        return None
    try:
        cycle = int(parts[0])
        energy = float(parts[1])
        delta_e = float(parts[2])
    except ValueError:
        return None
    return {"cycle": cycle, "energy": energy, "delta_E": delta_e,
            "gnorm": maybe_float(parts[3]), "ddm": maybe_float(parts[4]),
            "wall_clock_s": (maybe_float(parts[5])
                             if len(parts) >= 6 else None)}
