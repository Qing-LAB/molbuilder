"""PySCF / geomeTRIC trajectory FileParser.

H1 of parse-module.md migration (was Phase C wrapper around
``molbuilder.parsers.pyscf.PySCFParser``): absorbed the legacy parse
body directly so this module no longer imports from
``molbuilder.parsers``.  The legacy module stays in place until H4;
consumers (web blueprints) still use it until H3.

When molbuilder generates a PySCF script with `prefix=JOB+'_geom'`
on the optimize() call (the default), geomeTRIC streams a multi-frame
XYZ to ``<JOB>_geom_optim.xyz`` -- one frame per accepted geom step.

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
variant that masks out the indices listed in the sidecar's
``frozen_atoms`` field (SIESTA analog of ``MD.MaxForceTol`` semantics).
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from molbuilder.frame import Frame, Trajectory
from molbuilder.parse.base import FileParser
from molbuilder.parse.types import TrajectoryResult
from molbuilder.structure import Structure

from ._helpers import wrap_trajectory
from ._sidecar import read_frozen_atoms


# Hartree -> eV
_HARTREE_TO_EV = 27.211386245988
# Hartree/Bohr -> eV/Ang
_HA_BOHR_TO_EV_ANG = _HARTREE_TO_EV / 0.5291772108


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

# geomeTRIC progress line: ``Step    0 : Gradient = ... Energy = -1028.231``
_GEOMETRIC_STEP_RE = re.compile(
    r"^Step\s+(\d+)\s*:[^\n]*?Energy\s*=\s*(-?[\d.eE+-]+)",
    re.MULTILINE,
)
# ANSI escape codes geomeTRIC uses to color its progress lines.
_ANSI_ESC_RE = re.compile(r"\x1b\[[0-9;]*m")


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
                try:
                    float(parts[1])
                    float(parts[2])
                    float(parts[3])
                except ValueError:
                    return False
            return True
    except OSError:
        return False


def _read_initial_energy_from_log(traj_path: str) -> Optional[float]:
    """Return the geomeTRIC ``Step 0`` energy (eV) from the sibling
    ``.pyscf.log`` / ``-run<N>.pyscf.log`` file, or None when no log
    exists / no Step-0 line is found.

    Strips ANSI escapes; picks the LAST Step-0 match (multiple
    geom-opt restarts within one log re-emit Step 0).  Highest-N
    ``-run<N>.pyscf.log`` wins; bare ``<stem>.pyscf.log`` fallback.
    """
    base, fname = os.path.split(traj_path)
    if not base:
        base = "."
    stem = fname
    for suffix in ("_geom_optim.xyz", "_initial.xyz",
                   "_optimized.xyz", ".xyz"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    candidates: List[str] = []
    try:
        for entry in os.listdir(base):
            if entry.startswith(stem + "-run") \
                    and entry.endswith(".pyscf.log"):
                candidates.append(os.path.join(base, entry))
    except OSError:
        pass
    candidates.sort()
    bare = os.path.join(base, stem + ".pyscf.log")
    if os.path.isfile(bare):
        candidates.append(bare)

    for log_path in reversed(candidates):
        try:
            with open(log_path, "r", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        text = _ANSI_ESC_RE.sub("", text)
        last = None
        for m in _GEOMETRIC_STEP_RE.finditer(text):
            if int(m.group(1)) == 0:
                last = m
        if last is None:
            continue
        try:
            ha = float(last.group(2))
        except (TypeError, ValueError):
            continue
        return ha * _HARTREE_TO_EV
    return None


def _read_qdata_forces(
    traj_path: str, n_frames: int,
) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """Read ``<prefix>.qdata{,.txt}`` and return per-step (max,
    max_excluding_frozen) force magnitudes in eV/Ang.

    The constrained variant exists because ``MD.MaxForceTol``-style
    convergence thresholds apply to FREE atoms only — a forever-
    pinned frozen atom keeps the unconstrained max above threshold
    and the user can't tell when their run converged.  ``None`` for
    a step when the sidecar isn't present or the qdata entry is
    missing.
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

    frozen_set = read_frozen_atoms(traj_path)

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
    base, fname = os.path.split(traj_path)
    stem = fname
    if stem.endswith("_optim.xyz"):
        stem = stem[: -len("_optim.xyz")]
    if stem.endswith("_geom"):
        stem = stem[: -len("_geom")]
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
                            "gnorm":   float(g)  * _HA_BOHR_TO_EV_ANG,
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

    max_forces, max_forces_constrained = _read_qdata_forces(
        path, len(frames_raw))
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

    # Early-window fallback: a single frame with no energy may be the
    # ``<JOB>_initial.xyz`` of a fresh run.  Pull the geomeTRIC Step 0
    # energy from the sibling .pyscf.log so the plot has a data point.
    if (len(frames) == 1 and frames[0].energy is None):
        fallback_energy = _read_initial_energy_from_log(path)
        if (fallback_energy is not None
                and math.isfinite(fallback_energy)):
            f = frames[0]
            frames[0] = Frame(
                structure   = f.structure,
                step_index  = f.step_index,
                energy      = fallback_energy,
                forces      = f.forces,
                max_force   = f.max_force,
                max_force_constrained = f.max_force_constrained,
                scf_history = f.scf_history,
            )

    # Surface the sidecar's frozen_atoms list (same contract as
    # the SIESTA + molwatch parsers).
    runtime_info: Dict[str, object] = {}
    frozen = sorted(read_frozen_atoms(path))
    if frozen:
        runtime_info["frozen_atoms"] = frozen

    return Trajectory(
        source_format = "pyscf",
        frames        = frames,
        lattice       = None,           # geomeTRIC traj has no cell
        runtime_info  = runtime_info,
    )


class PySCFOutFileParser(FileParser):
    """Parse a PySCF + geomeTRIC trajectory file
    (``<job>_geom_optim.xyz``).  Returns a :class:`TrajectoryResult`
    with one Frame per geomeTRIC step + per-step SCF history
    (extracted from the companion ``.log`` when present)."""

    name   = "pyscf-out"
    label  = "XYZ trajectory (PySCF / geomeTRIC / generic multi-frame XYZ)"
    hint   = ("a multi-frame XYZ trajectory -- e.g., geomeTRIC's "
              "<job>_geom_optim.xyz (NOT the PySCF .log).  Generic XYZ "
              "with any comment-line format is also accepted; energies "
              "are extracted only when the comment matches the geomeTRIC "
              "`Iteration K Energy E` pattern.")
    output = TrajectoryResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        return _can_parse_xyz(str(path))

    @classmethod
    def parse(cls, path: Path) -> TrajectoryResult:
        traj = _parse_pyscf_xyz(str(path))
        return wrap_trajectory(traj, cls.name, path)
