"""PySCF / geomeTRIC trajectory parser.

When molbuilder generates a PySCF script with `prefix=JOB+'_geom'` on
the optimize() call (the default), geomeTRIC streams a multi-frame XYZ
to ``<JOB>_geom_optim.xyz`` -- one frame per accepted geometry step.

Frame format:

    {N}
    Iteration {K} Energy {E:.8f}
    {El}  {x:14.8f}  {y:14.8f}  {z:14.8f}
    ...

Energy is in Hartree.  We convert to eV (matches what the SIESTA parser
emits) so the energy plot is unit-consistent across formats.

geomeTRIC's `_optim.xyz` doesn't include force info per frame; if the
companion ``<prefix>.qdata`` file is present alongside, we additionally
pull the maximum force per step from there (Hartree/Bohr -> eV/Ang).
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..frame import Frame, Trajectory
from ..structure import Structure
from .base import TrajectoryParser


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

# "converged SCF energy = ..." marks the end of one SCF run (one
# geom-opt step's electronic problem).  Used to delimit run boundaries
# robustly even when cycle= 0 also appears in mid-run extra-cycle blocks.
_SCF_CONVERGED_RE = re.compile(
    r"converged SCF energy\s*=\s*(-?[\d.eE+-]+)"
)


class PySCFParser(TrajectoryParser):
    name  = "pyscf"
    label = "XYZ trajectory (PySCF / geomeTRIC / generic multi-frame XYZ)"
    hint  = ("a multi-frame XYZ trajectory -- e.g., geomeTRIC's "
             "<job>_geom_optim.xyz (NOT the PySCF .log).  Generic XYZ "
             "with any comment-line format is also accepted; energies "
             "are extracted only when the comment matches the geomeTRIC "
             "`Iteration K Energy E` pattern.")

    # Maximum atom count we'll accept in line 0 before declaring the
    # file isn't an XYZ.  Far above any realistic chemistry use case
    # but bounded to reject obvious garbage (e.g., a CSV row count of
    # millions parsed as atom count).
    _MAX_PLAUSIBLE_ATOMS = 1_000_000

    @classmethod
    def can_parse(cls, path: str) -> bool:
        """Structural XYZ check, not banner-matching.

        An XYZ file -- regardless of which tool produced it -- has the
        invariant:

            line 0:  positive integer N      (atom count)
            line 1:  arbitrary comment       (free text)
            lines 2..N+1:  atom lines        (element symbol + 3 floats)

        We verify exactly that, with no requirement on the comment line's
        content.  geomeTRIC's `Iteration K Energy E` is one possible
        comment format; ASE writes a different one; user scripts may
        write any string.  ``parse()`` extracts an energy when the
        comment matches the geomeTRIC pattern and falls back to
        ``energy=None`` otherwise -- so accepting any well-formed XYZ
        here is safe.

        We sample at most the first 3 atom lines (or all of them if N<3)
        for the structural check -- enough to reject CSVs / namelists /
        other text files that happen to start with an integer, while
        keeping the detector cheap.
        """
        try:
            with open(path, "r", errors="replace") as fh:
                line0 = fh.readline().strip()
                if not line0.isdigit():
                    return False
                n_atoms = int(line0)
                if n_atoms <= 0 or n_atoms > cls._MAX_PLAUSIBLE_ATOMS:
                    return False
                fh.readline()  # comment line; any content accepted
                # Verify the atom lines parse: element token + 3 floats.
                # Sample at most 3 (or fewer if n_atoms < 3).
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

    @classmethod
    def parse(cls, path: str) -> Trajectory:
        # The inner loop still builds parallel lists for parsing
        # convenience; we zip them into Frame objects just before
        # returning, so the per-frame construction lives in one place.
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
                    # Comment line we don't recognise; record the frame
                    # but with unknown step / energy.
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
                    # Last frame is mid-write; drop it so the JS slider
                    # never sees a torn frame.
                    break
                frames_raw.append(atoms)
                energies.append(energy_eV)
                iterations.append(step_idx)

        # Optional: pull max-force per step from the companion .qdata.
        # geomeTRIC writes `<prefix>.qdata`, where this file is named
        # `<prefix>_optim.xyz`.  Same prefix, so derive it:
        # 2026-06-12: now returns a 2-tuple — (max_over_all_atoms,
        # max_excluding_frozen_atoms).  The second component is the
        # PySCF analog of SIESTA's "Max <val> constrained" line.
        # Computed by masking out the indices listed in the sidecar's
        # ``frozen_atoms`` field when building the per-atom magnitudes
        # from qdata's GRADIENT block.  When no sidecar is present or
        # ``frozen_atoms`` is empty, the constrained list is filled
        # with None — same trivial-case signal SIESTA's parser
        # surfaces (and the ``trajectory_to_legacy_dict`` collapse
        # rule below converts to ``[]`` so the plot stays single-
        # trace).
        max_forces, max_forces_constrained = cls._read_qdata_forces(
            path, len(frames_raw))

        # PySCF's main .log has the SCF iteration tables (one block per
        # geom-opt step's electronic problem).  Surface this as
        # scf_history so molwatch can show progress within the current
        # geom-opt step.
        scf_history = cls._read_scf_history(path)

        # Zip the parallel lists into Frame objects.  geomeTRIC
        # trajectories carry no per-atom forces (Frame.forces=None) and
        # no cell (Trajectory.lattice=None).
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

        return Trajectory(
            source_format = cls.name,
            frames        = frames,
            lattice       = None,           # geomeTRIC traj has no cell
        )

    # ------------------------------------------------------------- #
    #  qdata-companion helper                                        #
    # ------------------------------------------------------------- #
    @classmethod
    def _read_qdata_forces(
        cls, traj_path: str, n_frames: int,
    ) -> "Tuple[List[Optional[float]], List[Optional[float]]]":
        """Try `<prefix>.qdata` next to the trajectory.  Returns a
        2-tuple of length-``n_frames`` lists (eV/Ang):

          ``(max_forces, max_forces_constrained)``

        where ``max_forces[i]`` is the max per-atom force magnitude
        across ALL atoms (including any constrained ones) and
        ``max_forces_constrained[i]`` is the same statistic EXCLUDING
        the indices listed in the sidecar's ``frozen_atoms`` field.

        2026-06-12: the constrained variant exists because
        ``MD.MaxForceTol``-style convergence thresholds apply to the
        FREE atoms only — a forever-pinned frozen atom keeps the
        unconstrained max above the threshold and the user can't
        tell from the plot when their run actually converged.  See
        SIESTA's ``Max <val> constrained`` line for the engine's
        own version of the same idea.

        ``max_forces_constrained[i]`` falls back to ``None`` when
        the sidecar isn't present, has an empty ``frozen_atoms``,
        or the qdata entry for that step is missing — the JSON
        layer collapses an all-``None`` list to ``[]`` so consumers
        can detect "no constraints in this run" via
        ``arr.length === 0``.

        Each list entry is ``None`` if the qdata file isn't there
        or the entry is missing for that step.
        """
        base, fname = os.path.split(traj_path)
        # geomeTRIC's pair: <prefix>_optim.xyz <-> <prefix>.qdata.txt
        # But qdata extension varies across versions; try a couple.
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

        frozen_set = cls._read_sidecar_frozen_atoms(traj_path)

        # Per-step: ENERGY starts a frame, GRADIENT (for THAT frame)
        # follows.  We flush the current frame's max only on the NEXT
        # ENERGY line (or at EOF), so the gradient is bound to the
        # right frame.
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
                        # GRADIENT line lists 3N gradient components in
                        # Hartree/Bohr.  Convention compatibility:
                        # SIESTA's "Max" is the largest per-atom force
                        # MAGNITUDE (sqrt(fx^2+fy^2+fz^2)), not the
                        # largest scalar component.  Match it here so
                        # the energy/force plots overlay sensibly
                        # across formats.
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

        # Pad / truncate to align with frames.
        if len(max_forces) < n_frames:
            max_forces.extend([None] * (n_frames - len(max_forces)))
        if len(max_forces_constrained) < n_frames:
            max_forces_constrained.extend(
                [None] * (n_frames - len(max_forces_constrained)))
        return max_forces[:n_frames], max_forces_constrained[:n_frames]

    @classmethod
    def _read_sidecar_frozen_atoms(cls, traj_path: str) -> "set[int]":
        """Look for ``<base>.molstruct.json`` next to ``<base>_optim
        .xyz`` and return the ``frozen_atoms`` set (0-based ints).

        Empty set when:
          * no sidecar is present
          * sidecar has no ``frozen_atoms`` field
          * sidecar fails to parse (the constrained-max field stays
            ``None`` for the run; no error surfaces to the user)
        """
        base, fname = os.path.split(traj_path)
        stem = fname
        if stem.endswith("_optim.xyz"):
            stem = stem[: -len("_optim.xyz")]
        # Try a couple of conventions: the geomeTRIC pair often
        # has a stripped ``_geom`` suffix on the sidecar too.
        candidates = [
            os.path.join(base, f"{stem}.molstruct.json"),
        ]
        if stem.endswith("_geom"):
            candidates.append(
                os.path.join(base, stem[:-5] + ".molstruct.json"))
        sidecar_path = next(
            (p for p in candidates if os.path.isfile(p)), None)
        if sidecar_path is None:
            return set()
        try:
            import json as _json
            with open(sidecar_path, "r", errors="replace") as fh:
                data = _json.load(fh)
        except (OSError, ValueError):
            return set()
        frozen = data.get("frozen_atoms")
        if not isinstance(frozen, list):
            return set()
        return {int(i) for i in frozen if isinstance(i, int)}

    # ------------------------------------------------------------- #
    #  PySCF-log SCF-iteration helper                                #
    # ------------------------------------------------------------- #
    @classmethod
    def _read_scf_history(cls, traj_path: str) -> List[List[Dict[str, float]]]:
        """Try ``<prefix>.log`` next to the trajectory.

        Returns a list of SCF runs, where each run is a list of
        per-cycle dicts:

            [
              [   # geom-opt step 0
                {"cycle": 0, "energy": <eV>, "delta_E": <eV>,
                 "gnorm":  <eV/Ang>, "ddm": <dimensionless>},
                ...
              ],
              [...],   # step 1
              ...
            ]

        molwatch uses the LAST entry as "the current step's SCF" --
        it's the one most useful for live monitoring.

        Returns an empty list if the .log file isn't present (or
        can't be opened); the front-end then hides the SCF-progress
        panel.

        Filename derivation:
          * traj is `<base>_geom_optim.xyz` (molbuilder convention)
          * pyscf log is `<base>.log` (no `_geom`)
        We strip the suffix(es) accordingly.
        """
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
                        # New SCF run boundary detection (SP-D).
                        # PySCF's first SCF cycle is sometimes ``0`` and
                        # sometimes ``1`` depending on version.  Use the
                        # robust signal: any cycle number that is NOT
                        # strictly greater than the previous one is a
                        # new-run marker.  ``_SCF_CONVERGED_RE`` below
                        # also flushes -- this catches the case where a
                        # run diverged before the converged line.
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
                        # End of an SCF run.  Flush only if we
                        # collected anything; consecutive converged
                        # lines without intervening cycles can happen
                        # in pathological logs.
                        if current:
                            runs.append(current)
                            current = []
                            prev_cycle = None
                if current:
                    runs.append(current)
        except OSError:
            return []
        return runs
