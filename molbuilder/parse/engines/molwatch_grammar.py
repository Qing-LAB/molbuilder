"""The molwatch log's lines -- header, step block, footer -- and ONE reader each.

The ``.molwatch.log v1`` format is molbuilder's own: `trajectory_log`
writes it (the PySCF deck inlines its emitter), and this is the grammar every
reader reads it with -- the reading pass (`molwatch_reader`, which the
registered parser builds Frames from and the monitor reads a running job
with), the PySCF parser's sibling-log enrichment, and the cheap ending scan
(`_run_ending`).

**Stdlib only, and nothing of ours**: it travels beside every job
(`runwrap.MONITOR_COMPANIONS`), so the monitor reads a PySCF run's progress
with the lines the Results tab reads it with.  It is `molwatch.py`'s grammar
half, split out on 2026-09-26 for that reason -- the SIESTA family's table
(`siesta_grammar`) was split from its parser the same way.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

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
ERROR = re.compile(r"^#\s*error:\s*(.+)$", re.IGNORECASE)
CONCLUDED = re.compile(r"^#\s*concluded:\s*(.+)$", re.IGNORECASE)


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


def scan_conclusion(path) -> str:
    """The run-state a molwatch log's FOOTER states, without building a
    Trajectory -- ``"running"`` when it carries no footer.

    The cheap door onto :func:`parse_conclusion_line`, for a caller that
    wants only how the run ended (`model/parse.md` § 2b), and the sibling of
    ``_run_ending.scan_ending`` on the ``.out`` side.  A status probe over a
    whole directory must not full-parse to reach one string: doing that is
    what made ``run_status`` create and grow a ``.parse.log`` beside every
    molwatch log it looked at, on every Watch poll.

    **Every line, in file order**, because the grammar's rule is *error
    outranks concluded and the LAST error wins* -- a tail-only read would
    answer ``"ended"`` for a log whose earlier attempt failed.  Only lines
    opening with ``#`` are offered, which is all either pattern can match.
    """
    out: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line[:1] == "#":
                parse_conclusion_line(line, out)
    return out.get("run_state") or "running"


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
