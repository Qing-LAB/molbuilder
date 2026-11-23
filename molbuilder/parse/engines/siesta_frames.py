"""A SIESTA output's steps, made Frames -- the arrays the reading pass leaves
out (`model/parse.md` § 4a).

MODULE  parse.engines.siesta_frames (numpy, the parser's types --
        `frame`, `structure` -- and nothing else; it travels beside the job
        in ``mb_vibration.pyz``, `runwrap.members`)
ROLE    the one conversion of the reading pass's records
        (`siesta_reader.SiestaReader.finish`) into the parser's types: a
        step's ``[label, x, y, z]`` rows a :class:`Structure`
        (:func:`structure_of`), and the whole reading a :class:`Trajectory`
        (:func:`trajectory_of`)
USED-BY the registered parser (`parse.engines.siesta`); a SIESTA vibration's
        steps beside the job (`spectra.siesta_vibration`): the `relax`
        call's relaxation record (§ 5b.1), of this trajectory, and the
        geometries its steps take -- the force-constant run's reference,
        the last geometry a call reached, a call's coordinates

One conversion, so the relaxation record a vibration's record step writes
into its run's index and the one a viewer makes of the same output count the
same steps.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

import numpy as np

# TWO WAYS, because this module travels beside the job in ``mb_vibration.pyz``.
try:                                        # inside molbuilder
    from ...frame import Frame, ParseWarning, Trajectory
    from ...structure import Structure
except ImportError:                         # beside a job, in a bundle
    from frame import Frame, ParseWarning, Trajectory  # type: ignore[no-redef]
    from structure import Structure  # type: ignore[no-redef]


def structure_of(coords: Sequence[Sequence[Any]]) -> Structure:
    """A step's coordinates -- the reading pass's ``[label, x, y, z]`` rows,
    in Å -- as the structure they describe, each atom's label as SIESTA
    printed it."""
    return Structure(elements=[str(row[0]) for row in coords],
                     positions=np.array([row[1:4] for row in coords],
                                        dtype=float))


def trajectory_of(read: Mapping[str, Any], *, source_format: str,
                  parse_warnings: Optional[List[ParseWarning]] = None
                  ) -> Trajectory:
    """THE READING, MADE A TRAJECTORY: a Frame per step the pass committed,
    numbered in the order it read them, and -- for a run still going with
    SCF cycles and no coordinates to attach them to yet -- a Frame FLAGGED
    ``in_progress``, so the Results tab draws the SCF chart at once.  That
    frame's geometry is the last committed frame's (SIESTA holds geometry
    constant within an SCF cycle), or a 1-atom placeholder the inspector
    never renders.  The atoms the run held are the pass's to state
    (``runtime_info["frozen_atoms"]``)."""
    frames: List[Frame] = []
    for step in read["steps"]:
        frames.append(Frame(
            structure   = structure_of(step["coords"]),
            step_index  = len(frames),
            energy      = step["energy"],
            forces      = (np.asarray(step["forces"], dtype=float)
                           if step["forces"] else None),
            max_force   = step["max_force"],
            max_force_constrained = step["max_force_constrained"],
            charges     = step["charges"],
            stress_voigt_kbar = step["stress_voigt_kbar"],
            scf_history = step["scf_history"],
            in_progress = step["in_progress"],
            cut_short   = step["cut_short"],
        ))
    live = read["live_scf"]
    if live is not None:
        frames.append(Frame(
            structure   = (frames[-1].structure if frames else Structure(
                elements=["X"], positions=np.zeros((1, 3), dtype=float))),
            step_index  = len(frames),
            energy      = live["energy"],
            scf_history = live["scf_history"],
            in_progress = True,
        ))
    return Trajectory(
        source_format  = source_format,
        frames         = frames,
        lattice        = read["lattice"],
        run_state      = read["run_state"],
        # P-S2: reported, never a verdict.  `None` when no SCF block was
        # seen at all -- a correct final answer, not a hole.
        scf_converged  = read["scf_converged"],
        error_message  = read["error_message"],
        cause          = read["cause"],
        runtime_info   = read["runtime_info"],
        parse_warnings = list(parse_warnings or []),
    )


__all__ = ["structure_of", "trajectory_of"]


#: WHAT THIS MODULE NEEDS BESIDE IT in a job's bundle -- the modules it
#: imports there by their bare names (`runwrap.bundle_members`).
TRAVELS_WITH = ("molbuilder.frame",
                "molbuilder.structure")
