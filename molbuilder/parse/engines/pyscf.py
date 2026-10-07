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
formats.  geomeTRIC's ``_optim.xyz`` carries no per-frame forces;
when the companion ``<prefix>.qdata`` is present we additionally pull
the maximum force per step (Hartree/Bohr -> eV/Ang) + a constrained
variant that masks out the atoms the run holds, as its progress log's
``# frozen_atoms:`` line states them (SIESTA analog of
``MD.MaxForceTol`` semantics).
"""

from __future__ import annotations

import math
import os

from ..errors import ParseError
from ...runfiles import compose as _rf_compose
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from molbuilder.frame import Frame, Trajectory
from molbuilder.parse.base import FileParser
from molbuilder.parse.types import TrajectoryResult
from molbuilder.structure import Structure

from ._helpers import wrap_trajectory


# Hartree -> eV
from molbuilder.constants import HARTREE_EV as _HARTREE_TO_EV
# Hartree/Bohr -> eV/Ang, the ASE convention.  The same forces are
# converted by `trajectory_log.emitter` and the thresholds drawn over them
# by `pyscf.input`, both with this one; deriving it here instead
# (HARTREE_EV / BOHR_ANGSTROM) gives a number 0.36 ppm away, so a force
# read back would not equal the force emitted.
from molbuilder.constants import (
    HARTREE_BOHR_EV_ANGSTROM_ASE as _HA_BOHR_TO_EV_ANG)


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



# Bounded sanity-check on line-0 atom count when detecting an XYZ.
_MAX_PLAUSIBLE_ATOMS = 1_000_000


def _can_parse_xyz(path: str) -> bool:
    """Structural XYZ check, not banner-matching.  Verifies the
    canonical XYZ invariant: positive integer N on line 0, a free-text
    comment on line 1, and N atom lines of ``element x y z`` form.
    Samples at most the first 3 atom lines."""
    try:
        with open(path, "r", errors="replace") as fh:
            line0 = fh.readline().strip()
            if not line0.isdigit():
                return False
            n_atoms = int(line0)
            if n_atoms <= 0 or n_atoms > _MAX_PLAUSIBLE_ATOMS:
                return False
            fh.readline()  # comment line; any content accepted
            for _ in range(min(n_atoms, 3)):
                parts = fh.readline().split()
                if len(parts) < 4:
                    return False
                # COLUMN 0 IS AN ELEMENT: SIESTA's `.FA` is `N` then
                # `index fx fy fz` -- same count, same three floats, an
                # integer where the symbol goes.
                #
                # "NOT A NUMBER" rather than a periodic-table lookup, and
                # that is deliberate: a dummy atom (`X`), a ghost (`Bq`) or
                # an isotope label is a legal XYZ column 0 and belongs to
                # whoever writes it, while an integer index never is.
                # Measured over all 114 real `.xyz` in the tree: 114 carry a
                # symbol, 0 carry a number.
                try:
                    float(parts[0])
                except ValueError:
                    pass          # a symbol -- which is what an XYZ has
                else:
                    return False  # a number -- this is somebody's data file
                try:
                    float(parts[1])
                    float(parts[2])
                    float(parts[3])
                except ValueError:
                    return False
            return True
    except OSError:
        return False


#: geomeTRIC's trajectory, named on its run's stem --
#: ``<label>_<token>_geom_optim.xyz`` (`pyscf/input.py` ``ROLE_GEOM_TRAJ``,
#: the role's home, which this reader sits below and does not import).  What
#: precedes it is the stem every other file of the run is named on.
_GEOM_TRAJ = "_geom_optim.xyz"


def _run_stem_of(traj_path: str) -> Optional[str]:
    """The stem of the run whose geomeTRIC trajectory ``traj_path`` is --
    ``H2_01_coarse`` for ``H2_01_coarse_geom_optim.xyz`` -- or ``None`` for
    any other XYZ this reader is handed: the trajectory names its run's
    other files itself, and a file that names no run reads only itself
    (`model/parse.md` § 5.3)."""
    name = os.path.basename(traj_path)
    if len(name) <= len(_GEOM_TRAJ) or not name.endswith(_GEOM_TRAJ):
        return None
    return name[:-len(_GEOM_TRAJ)]


def _sibling_molwatch_log(traj_path: str) -> Optional[str]:
    """The run's progress log, ``<stem>.molwatch.log`` beside the
    trajectory (:func:`_run_stem_of`), or None when there is none.

    Used by :func:`_read_molwatch_metadata` to surface
    convergence_targets + run_state + error_message onto PySCF-parser
    Trajectories — which the molwatch parser already extracts but is
    otherwise lost when the user is viewing the trajectory file
    instead of the .molwatch.log.
    """
    stem = _run_stem_of(traj_path)
    if stem is None:
        return None
    base = os.path.dirname(traj_path) or "."
    candidate = os.path.join(base, _rf_compose(stem, ".molwatch.log"))
    return candidate if os.path.isfile(candidate) else None


# Marker regexes scoped to the header / footer scan.  Header lines
# all start with ``#`` and appear before the first ``==== molwatch
# step N begin ====`` block; footer markers (``# concluded:`` /
# ``# error:``) appear after the last ``==== ... end ====`` block.
# The step-begin marker and the footer grammar belong to the molwatch
# format, so they are read through its grammar, `molwatch_grammar`.
from . import molwatch_grammar as _MG   # noqa: E402
# The convergence-header grammar has ONE reader --
# ``molwatch_grammar.parse_convergence_line``.


def _read_molwatch_metadata(traj_path: str) -> Dict[str, object]:
    """Read the sibling ``.molwatch.log``'s header (for
    ``convergence_targets``) and footer markers (``# concluded:`` /
    ``# error:`` for run_state) without parsing the per-step blocks.

    Returns a dict that may contain:
      * ``"convergence_targets"`` — dict with the molwatch-emitter
        keys (max_force_tol_eV_per_A, scf_energy_tol, etc.) plus a
        ``"source": "molwatch_header"`` stamp matching what the
        molwatch parser surfaces.
      * ``"frozen_atoms"``  — the atoms the run holds, as its log's
        ``# frozen_atoms:`` line states them (`model/parse.md` § 5.3).
      * ``"run_state"``     — "ended" | "stopped" when the
        corresponding marker is present.
      * ``"error_message"`` — when ``# error:`` is present.

    Returns empty dict when there is no sibling .molwatch.log.
    Header read is bounded to the lines before the first step
    begin; the footer is read over the whole file in order.

    Mirrors the molwatch parser semantics so the PySCF parser can
    surface the same fields when the user is viewing the geomeTRIC
    trajectory (which has no header) instead of the molwatch.log.
    """
    log_path = _sibling_molwatch_log(traj_path)
    if log_path is None:
        return {}
    out: Dict[str, object] = {}
    convergence: Dict[str, object] = {}
    runtime: Dict[str, object] = {}

    # Header scan: read until the first ``==== molwatch step N
    # begin ====`` line (exclusive).  Header is normally <50 lines.
    try:
        with open(log_path, "r", errors="replace") as fh:
            for line in fh:
                if _MG.BLOCK_BEGIN.search(line):
                    break
                held = _MG.parse_frozen_atoms_line(line.rstrip("\n"))
                if held is not None:
                    out["frozen_atoms"] = held
                    continue
                (_MG.parse_convergence_line(line.rstrip("\n"), convergence)
                 or _MG.parse_runtime_line(line.rstrip("\n"), runtime))
    except OSError:
        return {}
    if len(convergence) > 1:      # more than the "source" stamp alone
        out["convergence_targets"] = convergence
    # THE SCF'S CRITERIA, as the molwatch parser states them for the same
    # log (`molwatch_grammar.scf_criteria`, `web/trajectory.md` § 3).
    crit = _MG.scf_criteria(runtime)
    if crit:
        out["scf_criteria"] = crit

    # THE FOOTER, through its one reader, over the whole file in order: a
    # log is appended across attempts, and an earlier attempt's error
    # outranks a later `# concluded:`.
    try:
        out.update(_MG.read_conclusion(log_path))
    except OSError:
        pass
    return out


def _read_qdata_forces(
    traj_path: str, n_frames: int, frozen=(),
) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """Read ``<prefix>.qdata{,.txt}`` and return per-step (max,
    max_excluding_frozen) force magnitudes in eV/Ang.

    The constrained variant exists because ``MD.MaxForceTol``-style
    convergence thresholds apply to FREE atoms only — a forever-
    pinned frozen atom keeps the unconstrained max above threshold
    and the user can't tell when their run converged.  ``frozen`` is
    the run's held atoms, as its own progress log states them
    (:func:`_read_molwatch_metadata`); ``None`` for a step when it holds
    none or the qdata entry is missing.
    """
    base, fname = os.path.split(traj_path)
    stem = fname
    if stem.endswith("_optim.xyz"):
        stem = stem[: -len("_optim.xyz")]
    candidates = [
        os.path.join(base, f"{stem}.qdata.txt"),
        os.path.join(base, f"{stem}.qdata"),
    ]
    qpath = next((p for p in candidates if os.path.isfile(p)), None)
    if qpath is None:
        return [None] * n_frames, [None] * n_frames

    frozen_set = set(frozen or ())

    max_forces:             List[Optional[float]] = []
    max_forces_constrained: List[Optional[float]] = []
    try:
        with open(qpath, "r", errors="replace") as fh:
            step_max:         Optional[float] = None
            step_max_constr:  Optional[float] = None
            in_frame = False
            for raw in fh:
                s = raw.strip()
                if s.startswith("ENERGY"):
                    if in_frame:
                        max_forces.append(step_max)
                        max_forces_constrained.append(step_max_constr)
                    in_frame = True
                    step_max = None
                    step_max_constr = None
                elif s.startswith("GRADIENT"):
                    try:
                        comps = [float(x) for x in s.split()[1:]]
                    except ValueError:
                        continue
                    if len(comps) >= 3:
                        per_atom = []
                        for atom_idx, i in enumerate(
                                range(0, len(comps) - 2, 3)):
                            mag = math.sqrt(
                                comps[i]**2 + comps[i+1]**2
                                + comps[i+2]**2)
                            per_atom.append((atom_idx, mag))
                        if per_atom:
                            step_max = max(
                                m for _, m in per_atom
                            ) * _HA_BOHR_TO_EV_ANG
                            if frozen_set:
                                free = [m for ai, m in per_atom
                                        if ai not in frozen_set]
                                if free:
                                    step_max_constr = (
                                        max(free) * _HA_BOHR_TO_EV_ANG)
            if in_frame:
                max_forces.append(step_max)
                max_forces_constrained.append(step_max_constr)
    except OSError:
        return [None] * n_frames, [None] * n_frames

    if len(max_forces) < n_frames:
        max_forces.extend([None] * (n_frames - len(max_forces)))
    if len(max_forces_constrained) < n_frames:
        max_forces_constrained.extend(
            [None] * (n_frames - len(max_forces_constrained)))
    return max_forces[:n_frames], max_forces_constrained[:n_frames]


def _read_scf_history(
    traj_path: str,
) -> List[List[Dict[str, float]]]:
    """Parse ``<prefix>.log`` for per-cycle SCF data.  Returns a list
    of runs (one per geom-opt step); each run is a list of per-cycle
    dicts.  Empty list when no log is present."""
    base = os.path.dirname(traj_path) or "."
    # ``<stem>.log`` -- the pyscf stdout carries the run's stem
    # (`pyscf/input.py`: ``ROLE_LOG``).
    stem = _run_stem_of(traj_path)
    if stem is None:
        return []
    log_path = os.path.join(base, stem + ".log")
    if not os.path.isfile(log_path):
        return []

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
    except OSError:
        return []
    return runs


def _parse_pyscf_xyz(path: str) -> Trajectory:
    """Parse a PySCF/geomeTRIC ``*_optim.xyz`` (or any XYZ) into a
    Trajectory.  See module docstring for the format and the qdata
    + .log sibling-file conventions."""
    frames_raw: List[List[List[Any]]] = []
    energies: List[Optional[float]] = []
    iterations: List[int] = []

    with open(path, "r", errors="replace") as fh:
        while True:
            header = fh.readline()
            if not header:
                break                     # clean EOF
            header = header.strip()
            if not header.isdigit():
                # Probably a torn write at EOF; bail out cleanly.
                break
            n_atoms = int(header)
            comment = fh.readline()
            if not comment:
                break                     # torn frame
            m = _COMMENT_RE.search(comment)
            if m:
                step_idx = int(m.group(1))
                energy_ha = float(m.group(2))
                energy_eV: Optional[float] = energy_ha * _HARTREE_TO_EV
            else:
                step_idx = len(frames_raw)
                energy_eV = None

            atoms: List[List[Any]] = []
            torn = False
            for _ in range(n_atoms):
                line = fh.readline()
                if not line:
                    torn = True
                    break
                parts = line.split()
                if len(parts) < 4:
                    torn = True
                    break
                try:
                    atoms.append([
                        parts[0],
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                    ])
                except ValueError:
                    torn = True
                    break
            if torn or len(atoms) != n_atoms:
                # Last frame is mid-write; drop it.
                break
            frames_raw.append(atoms)
            energies.append(energy_eV)
            iterations.append(step_idx)

    # THE RUN'S OWN PROGRESS LOG, read once: its held atoms decide the
    # free-atom force below, and its targets and ending enrich the frames.
    mw_meta = _read_molwatch_metadata(path)
    max_forces, max_forces_constrained = _read_qdata_forces(
        path, len(frames_raw), mw_meta.get("frozen_atoms") or ())
    scf_history = _read_scf_history(path)

    if not iterations:
        iterations = list(range(len(frames_raw)))
    frames: List[Frame] = []
    for i, atoms in enumerate(frames_raw):
        elements  = [row[0] for row in atoms]
        positions = np.array([row[1:4] for row in atoms], dtype=float)
        struct = Structure(elements=elements, positions=positions)
        scf_for_step = (scf_history[i]
                        if i < len(scf_history) and scf_history[i]
                        else None)
        frames.append(Frame(
            structure   = struct,
            step_index  = iterations[i],
            energy      = energies[i],
            forces      = None,
            max_force   = max_forces[i] if i < len(max_forces) else None,
            max_force_constrained = (
                max_forces_constrained[i]
                if i < len(max_forces_constrained) else None),
            scf_history = scf_for_step,
        ))

    # THE ATOMS THE RUN HOLDS, as its own progress log states them -- the
    # field the SIESTA and molwatch parsers fill from their own files
    # (`model/parse.md` § 5.3).
    runtime_info: Dict[str, object] = {}
    if mw_meta.get("frozen_atoms"):
        runtime_info["frozen_atoms"] = list(mw_meta["frozen_atoms"])

    # Sibling-log enrichment.  The user may load the geomeTRIC
    # ``_geom_optim.xyz`` directly; that file carries no convergence
    # targets and no run-state markers.  When the molbuilder-generated
    # ``<stem>.molwatch.log`` is present next to it, pull
    # convergence_targets (for the threshold lines on the Results-tab
    # force/energy plots) and run_state (so the badge says "Finished"
    # / "Error" instead of "Ongoing").  Symmetric with how the
    # molwatch parser surfaces these when the user loads the .log
    # directly — same data, same field names, just sourced via the
    # sibling file (read once, above).
    for key in ("convergence_targets", "scf_criteria"):
        if key in mw_meta:
            runtime_info[key] = mw_meta[key]
    run_state = mw_meta.get("run_state", "unknown")
    error_message = mw_meta.get("error_message")

    return Trajectory(
        source_format = "pyscf",
        frames        = frames,
        lattice       = None,           # geomeTRIC traj has no cell
        run_state     = run_state,
        error_message = error_message,
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


class PySCFParser:
    """Body parser for PySCF + geomeTRIC ``*_optim.xyz`` trajectories.
    Returns legacy :class:`Trajectory` (dict-shaped via the JSON
    round-trip); use :class:`PySCFOutFileParser` for a typed
    :class:`TrajectoryResult`.

    ``can_parse(path) -> bool`` + ``parse(path) -> Trajectory``,
    plus ``name`` / ``label`` / ``hint`` class attributes."""

    name  = "pyscf"
    label = "XYZ trajectory (PySCF / geomeTRIC / generic multi-frame XYZ)"
    hint  = ("a multi-frame XYZ trajectory -- e.g., geomeTRIC's "
             "<job>_geom_optim.xyz (NOT the PySCF .log).  Generic XYZ "
             "with any comment-line format is also accepted; energies "
             "are extracted only when the comment matches the geomeTRIC "
             "`Iteration K Energy E` pattern.")

    @classmethod
    def can_parse(cls, path):
        return PySCFOutFileParser.can_parse(Path(path))

    parse = staticmethod(_parse_pyscf_xyz)


class PySCFOutFileParser(FileParser):
    """Parse a PySCF + geomeTRIC trajectory file
    (``<job>_geom_optim.xyz``).  Returns a :class:`TrajectoryResult`
    with one Frame per geomeTRIC step + per-step SCF history
    (extracted from the companion ``.log`` when present)."""

    name   = "pyscf"

    @classmethod
    def footgun_hint_for(cls, filename: str):
        lower = filename.lower()
        if not lower.endswith(".log") or "geom_optim" in lower:
            return None
        # Strip both ``.molwatch.log`` and bare ``.log``; molwatch is
        # the MolwatchLogFileParser's filename, this hint is for the
        # PySCF run-time .log foot-gun.
        if lower.endswith(".molwatch.log"):
            return None
        stem = filename[:-len(".log")]
        return (
            f"PySCF runs write the run-time log to {filename} but "
            f"the trajectory lives in the file whose role is "
            f"`_geom_optim.xyz` -- {_rf_compose(stem, '_geom_optim.xyz')} for "
            f"a single run, or the same with this rung's token after the "
            f"label for a ladder (job-contracts.md § 2.2a). Point molbuilder "
            f"at that instead."
        )
    label  = "XYZ trajectory (PySCF / geomeTRIC / generic multi-frame XYZ)"
    hint   = ("a multi-frame XYZ trajectory -- e.g., geomeTRIC's "
              "<job>[_<NN>_<stage>]_geom_optim.xyz (NOT the PySCF .log).  "
              "Generic XYZ "
              "with any comment-line format is also accepted; energies "
              "are extracted only when the comment matches the geomeTRIC "
              "`Iteration K Energy E` pattern.")
    output = TrajectoryResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        # `<job>_optimized.xyz` IS a valid XYZ, and it belongs to
        # `pyscf-geom`; were both parsers to claim it, `detect()` -- which
        # is exactly-one-or-raise -- would refuse it.
        #
        # The split is right the way it is documented: `_optimized.xyz`
        # is ONE converged geometry (`pyscf/warm-files.toml`: *"latest
        # converged geometry"*), so a `StructureResult` is what it is;
        # this parser answers `TrajectoryResult` and has nothing to say
        # about a single frame that the sibling does not say better.
        if path.name.endswith("_optimized.xyz"):
            return False
        return _can_parse_xyz(str(path))

    @classmethod
    def parse(cls, path: Path) -> TrajectoryResult:
        # A PARSER RAISES `ParseError`, and this is the one place that can
        # promise it for every name-composing site inside.
        #
        # The body looks for companions -- the geomeTRIC log, the molwatch
        # log, the pyscf stdout -- by COMPOSING their names from this file's
        # stem, and `runfiles` refuses a stem that is not a legal label
        # (§ 2.1: a dotted label cannot be read back out of a filename) with
        # a `RunFileError` -- a `ValueError`, not a `ParseError`.
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
