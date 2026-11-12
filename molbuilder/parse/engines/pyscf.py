"""PySCF / geomeTRIC trajectory FileParser.

When molbuilder generates a PySCF script it hands geomeTRIC a prefix
of ``JOB`` plus the stage token plus ``_geom`` (`job-contracts.md`
§ 2.2a: the token sits after the label, never inside the role), so
geomeTRIC streams a multi-frame XYZ to ``<JOB>_<NN>_<stage>_geom_optim.xyz``
-- or ``<JOB>_geom_optim.xyz`` for a run with no stage -- one frame per
accepted geom step.  Either way the ROLE is ``_geom_optim.xyz``, which is
what `runfiles.parse` reports and what `warm-files.toml` declares.

Frame format::

    {N}
    Iteration {K} Energy {E:.8f}
    {El}  {x:14.8f}  {y:14.8f}  {z:14.8f}
    ...

Energies are stored in Hartree on disk; we convert to eV (matches
the SIESTA parser) so the energy plot is unit-consistent across
formats.  geomeTRIC's ``_optim.xyz`` carries no per-frame forces, and
the deck asks geomeTRIC for no ``qdata`` file, so a PySCF frame states
no force.
"""

from __future__ import annotations

import os

from ..errors import ParseError
from ...runfiles import compose as _rf_compose, stem_before as _rf_stem_before
from ...runfiles import named_beside as _rf_named_beside
from ...runfiles import role_of as _rf_role_of
from ...runfiles import (ROLE_GEOM_TRAJ as _ROLE_GEOM_TRAJ,
                         ROLE_MOLWATCH as _ROLE_MOLWATCH,
                         ROLE_PYSCF_LOG as _ROLE_PYSCF_LOG)
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from molbuilder.frame import Frame, Trajectory
from molbuilder.parse.base import FileParser
from molbuilder.parse.types import TrajectoryResult
from molbuilder.structure import Structure

from ._helpers import wrap_trajectory
from .molwatch_reader import read_log
from .xyz import is_xyz, read_frames


# Hartree -> eV
from molbuilder.constants import HARTREE_EV as _HARTREE_TO_EV


_COMMENT_RE = re.compile(
    r"Iteration\s+(\d+)\s+Energy\s+(-?[\d.eE+-]+)",
    re.IGNORECASE,
)


# Matches a PySCF SCF iteration line, e.g.:
#   cycle= 1 E= -5005.99362145001  delta_E= 33.1  |g|= 13.4  |ddm|= 23.1
_SCF_LINE_RE = re.compile(
    r"cycle\s*=\s*(\d+)\s+"
    r"E\s*=\s*(-?[\d.eE+-]+)\s+"
    r"delta_E\s*=\s*(-?[\d.eE+-]+)\s+"
    r"\|g\|\s*=\s*([\d.eE+-]+)\s+"
    r"\|ddm\|\s*=\s*([\d.eE+-]+)"
)

# End-of-one-SCF-run marker.
_SCF_CONVERGED_RE = re.compile(
    r"converged SCF energy\s*=\s*(-?[\d.eE+-]+)"
)



def _run_stem_of(traj_path: str) -> Optional[str]:
    """The stem of the run whose geomeTRIC trajectory ``traj_path`` is --
    ``H2_01_coarse`` for ``H2_01_coarse_geom_optim.xyz`` -- or ``None`` for
    any other XYZ this reader is handed: the trajectory names its run's
    other files itself, and a file that names no run reads only itself
    (`model/parse.md` § 5.3).  The role and the cut are the catalogue's
    (`runfiles.ROLE_GEOM_TRAJ`, `runfiles.stem_before`)."""
    return _rf_stem_before(traj_path, _ROLE_GEOM_TRAJ)


def _companion(traj_path: str, role: str) -> Optional[str]:
    """The run's file in ``role`` beside geomeTRIC's trajectory -- named on
    the run's stem (:func:`_run_stem_of`) by the one composer
    (`runfiles.compose`, which refuses a stem no label of ours is) -- or
    ``None`` where it is not there, or for an XYZ that names no run, which
    reads only itself (`model/parse.md` § 5.3)."""
    stem = _run_stem_of(traj_path)
    if stem is None:
        return None
    path = os.path.join(os.path.dirname(traj_path) or ".",
                        _rf_compose(stem, role))
    return path if os.path.isfile(path) else None


#: THE RUN'S FACTS THE TRAJECTORY TAKES FROM ITS PROGRESS LOG (§ 5a) --
#: what geomeTRIC's file does not state and the log does.  Not the log's
#: atom index map, which is the log's own order record: on the trajectory's
#: reading it would be taken as the trajectory's (`atom_index.of_reading`).
_LOG_FACTS = ("convergence_targets", "frozen_atoms", "scf_criteria",
              "scf_residuals")


def _read_molwatch_metadata(traj_path: str) -> Dict[str, object]:
    """What the run's progress log beside the trajectory states of the run,
    read through the log's one read (`molwatch_reader.read_log`,
    `model/parse.md` § 4a) as its own parser reads it: ``runtime_info`` --
    the :data:`_LOG_FACTS` it states, as it states them -- ``run_state`` and
    ``error_message`` (its footer, § 2b: ``running`` for a log that states
    none).

    Empty for a trajectory with no progress log beside it; for one that
    does not read -- a file the system will not read, or a stated atom
    index map that does not, which the read refuses -- the trajectory reads
    as itself alone (§ 5a, rule 2).  WHICH
    LOG, OR WHY NONE (§ 5a, rule 4): ``molwatch_source`` names the log the
    facts came from, ``molwatch_error`` the log that did not read and why."""
    log_path = _companion(traj_path, _ROLE_MOLWATCH)
    if log_path is None:
        return {}
    name = os.path.basename(log_path)
    try:
        read = read_log(log_path).finish()
    except (OSError, ValueError) as exc:
        return {"runtime_info": {"molwatch_error": f"{name}: {exc}"}}
    info = read["runtime_info"]
    facts = {k: info[k] for k in _LOG_FACTS if k in info}
    facts["molwatch_source"] = name
    return {"runtime_info": facts,
            "run_state": read["run_state"],
            "error_message": read["error_message"]}


def _read_scf_history(
    traj_path: str,
) -> Tuple[List[List[Dict[str, float]]], Dict[str, str]]:
    """``(runs, said)``: the per-cycle SCF data of ``<stem>.log``, PySCF's
    own log -- a list of runs, one per geom-opt step, each a list of
    per-cycle dicts; none for a trajectory with no log beside it, or one
    that does not read -- and WHICH LOG, OR WHY NONE (`model/parse.md` § 5a,
    rule 4): ``pyscf_log_source`` the log read, ``pyscf_log_error`` the log
    that did not read and why."""
    log_path = _companion(traj_path, _ROLE_PYSCF_LOG)
    if log_path is None:
        return [], {}
    name = os.path.basename(log_path)

    runs: List[List[Dict[str, float]]] = []
    current: List[Dict[str, float]] = []
    prev_cycle: Optional[int] = None

    try:
        with open(log_path, "r", errors="replace") as fh:
            for raw in fh:
                m = _SCF_LINE_RE.search(raw)
                if m:
                    cycle, e, de, g, ddm = m.groups()
                    cy = int(cycle)
                    # New-run boundary: any cycle number that is NOT
                    # strictly greater than the previous one signals
                    # a new SCF run (PySCF first-cycle is sometimes
                    # 0, sometimes 1 depending on version).
                    is_boundary = (
                        current and prev_cycle is not None
                        and cy <= prev_cycle
                    )
                    if is_boundary:
                        runs.append(current)
                        current = []
                    try:
                        current.append({
                            "cycle":   cy,
                            "energy":  float(e)  * _HARTREE_TO_EV,
                            "delta_E": float(de) * _HARTREE_TO_EV,
                            # |g| is PySCF's orbital-gradient norm, an
                            # ENERGY (Hartree over dimensionless orbital
                            # rotations, `web/trajectory.md` § 3), converted
                            # as the progress log's writer converts it.
                            "gnorm":   float(g)  * _HARTREE_TO_EV,
                            "ddm":     float(ddm),
                        })
                        prev_cycle = cy
                    except ValueError:
                        continue
                elif _SCF_CONVERGED_RE.search(raw):
                    if current:
                        runs.append(current)
                        current = []
                        prev_cycle = None
            if current:
                runs.append(current)
    except OSError as exc:
        return [], {"pyscf_log_error": f"{name}: {exc}"}
    return runs, {"pyscf_log_source": name}


def _parse_pyscf_xyz(path: str) -> Trajectory:
    """Parse geomeTRIC's ``<stem>_geom_optim.xyz`` into a Trajectory: its
    frames, read as every XYZ is (`xyz.read_frames`), each comment line's
    iteration and energy, and its run's files beside it.  See module
    docstring for the format."""
    frames_raw = read_frames(path)
    energies: List[Optional[float]] = []
    iterations: List[int] = []
    for comment, _symbols, _positions in frames_raw:
        m = _COMMENT_RE.search(comment)
        iterations.append(int(m.group(1)) if m else len(iterations))
        energies.append(float(m.group(2)) * _HARTREE_TO_EV if m else None)

    # THE RUN'S TWO LOGS beside it (§ 5a): geomeTRIC's file states no
    # targets, no atoms held, no SCF criteria and no ending -- the progress
    # log does -- and no step's SCF -- PySCF's own log does.
    mw_meta = _read_molwatch_metadata(path)
    scf_history, scf_said = _read_scf_history(path)

    frames: List[Frame] = []
    for i, (_comment, elements, positions) in enumerate(frames_raw):
        struct = Structure(elements=elements, positions=positions)
        scf_for_step = (scf_history[i]
                        if i < len(scf_history) and scf_history[i]
                        else None)
        frames.append(Frame(
            structure   = struct,
            step_index  = iterations[i],
            energy      = energies[i],
            forces      = None,
            max_force   = None,
            max_force_constrained = None,
            scf_history = scf_for_step,
        ))

    runtime_info = dict(mw_meta.get("runtime_info", {}), **scf_said)
    if "pyscf_log_source" in scf_said:
        # HOW MANY FRAMES THE LOG CHANGED (§ 5a, rule 4): the steps it gave
        # an SCF history.
        runtime_info["pyscf_log_scf_upgraded"] = sum(
            1 for f in frames if f.scf_history is not None)
    return Trajectory(
        source_format = "pyscf",
        frames        = frames,
        lattice       = None,           # geomeTRIC traj has no cell
        run_state     = mw_meta.get("run_state", "unknown"),
        error_message = mw_meta.get("error_message"),
        runtime_info  = runtime_info,
    )



# PySCF's own report of itself, near the top of the log its logger writes
# (``pyscf/lib/misc.py`` ``format_sys_info``): ``System: ... Threads <n>``,
# ``Python <v>``, ``numpy <v>  scipy <v>  h5py <v>``, ``PySCF version <v>``.
# The run record's engine facts (`model/parse.md` § 5d.2): a deck is written
# before the run and cannot know the version that ran it.
_SYS_THREADS = re.compile(r"^System:.*\bThreads\s+(\d+)")
_SYS_PYTHON = re.compile(r"^Python\s+(\S+)")
_SYS_NUMPY = re.compile(r"^numpy\s+(\S+)\s+scipy\s+(\S+)")
_SYS_VERSION = re.compile(r"^PySCF version\s+(\S+)")


def read_pyscf_sys_info(text: str) -> Dict[str, Any]:
    """``{version, python, numpy, scipy, threads}`` from a PySCF log -- each
    absent when the log does not state it."""
    out: Dict[str, Any] = {}
    for line in (text or "").splitlines():
        for key, pat in (("version", _SYS_VERSION), ("python", _SYS_PYTHON)):
            m = pat.match(line)
            if m:
                out.setdefault(key, m.group(1))
        m = _SYS_NUMPY.match(line)
        if m:
            out.setdefault("numpy", m.group(1))
            out.setdefault("scipy", m.group(2))
        m = _SYS_THREADS.match(line)
        if m:
            out.setdefault("threads", int(m.group(1)))
        if "version" in out and "threads" in out and "numpy" in out:
            break
    return out


def record_fields(f) -> Dict[str, Any]:
    """WHAT PYSCF'S OWN LOG ANSWERS OF THE RUN RECORD -- the engine's report
    of itself, from its logger's file: what the log's catalogue row names for
    ``record`` (`runfiles.WRITTEN`; `model/parse.md` § 5d.1b).  ``f`` is the
    run's files as the run record found them (`parse.dirs.record.RunFiles`)."""
    info = read_pyscf_sys_info(f.head(f.of(_ROLE_PYSCF_LOG)))
    if not info:
        return {}
    engine = {"program": f.engine}
    for key in ("version", "python"):
        if key in info:
            engine[key] = info[key]
    comp: Dict[str, Any] = {"engine": engine}
    if "threads" in info:
        comp["launch"] = {"threads_engine": info["threads"]}
    return {"computation": comp}


#: THE RUN RECORD'S FIELDS THIS READER ANSWERS, and no others
#: (`parse.dirs.record`, § 5d.1b's one source per quantity).
record_fields.fields = (
    "computation.engine.program", "computation.engine.version",
    "computation.engine.python", "computation.launch.threads_engine")


class PySCFOutFileParser(FileParser):
    """Parse a PySCF + geomeTRIC trajectory file
    (``<job>_geom_optim.xyz``).  Returns a :class:`TrajectoryResult`
    with one Frame per geomeTRIC step + per-step SCF history
    (extracted from the companion ``.log`` when present)."""

    name   = "pyscf"

    @classmethod
    def footgun_hint_for(cls, filename: str):
        # THE PYSCF RUN-TIME LOG'S ROLE, by the catalogue (`runfiles.role_of`:
        # the longest declared role wins, so a ``.molwatch.log`` -- the
        # MolwatchLogFileParser's -- is not this foot-gun).
        if _rf_role_of(filename) != _ROLE_PYSCF_LOG:
            return None
        traj = _rf_named_beside(filename, _ROLE_GEOM_TRAJ)
        return (
            f"PySCF runs write the run-time log to {filename} but "
            f"the trajectory lives in the file whose role is "
            f"`{_ROLE_GEOM_TRAJ}`"
            + (f", {traj}, beside it" if traj else "")
            + ". Point molbuilder at that instead."
        )
    label  = "geomeTRIC trajectory (a PySCF relaxation's steps)"
    hint   = ("geomeTRIC's <job>[_<NN>_<stage>]_geom_optim.xyz, one frame "
              "per accepted step (NOT the PySCF .log); energies from its "
              "`Iteration K Energy E` comment lines")
    output = TrajectoryResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        # GEOMETRIC'S TRAJECTORY, by its name: the one XYZ this reader is
        # for, read with its run's files beside it.  A PySCF deck's two
        # geometries are `pyscf-geom`'s, a pair `structure-pair`'s, every
        # other XYZ `xyz`'s -- so `detect()`, exactly-one-or-raise, opens
        # each (`model/parse.md` § 4).
        return _run_stem_of(str(path)) is not None and is_xyz(str(path))

    @classmethod
    def parse(cls, path: Path) -> TrajectoryResult:
        # A PARSER RAISES `ParseError`, and this is the one place that can
        # promise it for every name-composing site inside.
        #
        # The body looks for companions -- the progress log and PySCF's own
        # log -- by COMPOSING their names from this file's stem, and
        # `runfiles` refuses a stem that is not a legal label (§ 2.1: a
        # dotted label cannot be read back out of a filename) with a
        # `RunFileError` -- a `ValueError`, not a `ParseError`.
        #
        # Wrapped HERE rather than at each compose site: the boundary is the
        # one place the promise can be kept.
        from molbuilder.runfiles import RunFileError
        try:
            traj = _parse_pyscf_xyz(str(path))
        except RunFileError as exc:
            raise ParseError(
                f"{Path(path).name}: this is a readable XYZ, but its name is "
                f"not one molbuilder composes, so the companions it would be "
                f"read with cannot be named -- {exc}") from exc
        return wrap_trajectory(traj, cls.name, path)
