"""A MODE'S FRAME SET FROM THE COMMAND LINE AND THE PAGE -- one vibration
mode, thermal at a temperature, written as one multi-frame pair a transport
calculation cites (`engines/vibration.md` § 5.10 ③), through the motion
toolkit's run door and one writer (`vibration_frames.vibration_of_run`,
`vibration_frames.write_frame_set`; `engines/vibration-frames.md` § 5).
To be retired in R3d2b2b2b2, the doors writing a mode's own set in any
state through the toolkit itself.

MODULE  spectra.mode_frames (L2; the vibration frame sets' toolkit
        (`vibration_frames`), the vibration's mechanics (`vibration`), the
        structure's codec, transport's electrode check)
ROLE    :func:`frameset_from_mode` -- the doors' frame set, and
        :class:`ModeFrames`, what it wrote and what it says;
        :func:`outermost_move` and :func:`at_temperature` -- what a
        temperature means for a mode row before its set is written: its
        mean number of quanta, its spread and the outermost frames' largest
        atom move for each rule (`vibration.outermost_move_ang`, the one
        rule)
USED-BY the command line (`molbuilder frameset from-mode`) and the
        spectrum viewer's doors (`POST /api/spectra/frameset`; what a
        temperature means, at `POST /api/spectra/load`)

**Its extent is the person's: a temperature** (the user, 2026-10-28) --
the junction's, or for a mode pumped out of equilibrium its effective one.
It sets the mode's mean number of quanta ``n`` and the spread of the
Gaussian the frames sample, ``sigma = Q_zp sqrt(2n + 1)``; each frame sits
at a node of the state's own Gauss rule with the rule's own weight, and the
outermost frames' largest atom move is measured and reported -- the check
a person reads against a bond length.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# THE TOOLKIT'S REFUSAL, which this module's callers catch.
from ..vibration_frames import FrameSetError

#: The rules the generator writes -- three frames by default, five or seven
#: on request, each the check of the one before
#: (`science/vibrational-averaging.md` § 5.1, § 5.3).
NODES = (3, 5, 7)



@dataclass(frozen=True)
class ModeFrames:
    """What :func:`frameset_from_mode` wrote: the pair's ``.xyz`` (``path``),
    the mode's mean number of quanta at the temperature and its spread
    against its zero-point amplitude (``mean_quanta``,
    ``sigma_over_zero_point``), each frame's position in units of the
    spread, largest atom move and weight (``frames``), and what the
    generator says of the set (``said``): the electrode atoms that move, a
    set with no settings to cite."""
    path: Path
    mean_quanta: float
    sigma_over_zero_point: float
    frames: Tuple[Dict[str, float], ...]
    said: Tuple[str, ...]

    def to_wire(self, root: Optional[Path] = None) -> Dict[str, Any]:
        """As a page is answered: the pair's path from ``root`` (the tree),
        each frame, and what is said."""
        from ..projects import address
        path = (address(self.path, root) if root is not None
                else str(self.path))
        return {"path": path, "frames": [dict(f) for f in self.frames],
                "said": list(self.said)}


def _checked_nodes(nodes: int) -> int:
    """``nodes`` when it is one of the rules the generator writes, refused
    by name otherwise."""
    if nodes not in NODES:
        raise FrameSetError(f"the frame set is written at "
                            f"{', '.join(str(n) for n in NODES[:-1])} or "
                            f"{NODES[-1]} nodes, not {nodes}")
    return nodes


def _checked_temperature(temperature_k) -> float:
    """The temperature the frames sample, by the rule every door shares
    (`vibration.checked_temperature_k`) -- stated every time, never
    defaulted -- refused as the generator refuses."""
    from ..vibration import checked_temperature_k
    try:
        return checked_temperature_k(temperature_k)
    except ValueError:
        raise FrameSetError(
            f"the temperature is the one the frames sample -- the "
            f"junction's, or a pumped mode's effective one -- a number of "
            f"kelvin, 0 or more; {temperature_k!r} is not one") from None


def outermost_move(mode: Dict[str, Any], temperature_k: float,
                   nodes: int) -> Optional[float]:
    """THE OUTERMOST MOVE of a mode row at ``temperature_k`` for the
    ``nodes``-point rule, in Angstrom: the largest atom's move at the
    outermost frames, ``|q_out| max_A |L_A|`` for the thermal state at that
    temperature (`science/vibrational-averaging.md` § 5) -- the one rule
    (`vibration.outermost_move_ang`) a written set's
    ``max_displacement_ang`` row takes.  ``None`` for a mode with no
    zero-point amplitude (an imaginary one)."""
    from ..vibration import outermost_move_ang, state
    nodes = _checked_nodes(nodes)
    q_zp = mode.get("zero_point_amplitude_amu12_ang")
    if q_zp is None:
        return None
    return outermost_move_ang(
        state(float(q_zp), float(mode["frequency_cm1"]),
              temperature_k=float(temperature_k)),
        mode["eigenvector_canonical"], nodes)


def at_temperature(mode: Dict[str, Any],
                   temperature_k: float) -> Optional[Dict[str, Any]]:
    """WHAT ``temperature_k`` MEANS FOR A MODE ROW before its set is
    written (`web/spectra.md` § 4.3): its mean number of quanta, its spread
    and the outermost move for each rule the generator writes --
    ``{mean_quanta, sigma_amu12_ang, sigma_over_zero_point,
    outermost_move_ang: {"3": …, …}}`` -- from the derivations the
    generator writes by.  ``None`` for a mode with no zero-point
    amplitude."""
    from ..vibration import mean_quanta, thermal_spread_of
    sigma = thermal_spread_of(mode, temperature_k)
    if sigma is None:
        return None
    return {"mean_quanta": mean_quanta(mode["frequency_cm1"], temperature_k),
            "sigma_amu12_ang": float(sigma),
            "sigma_over_zero_point": float(sigma) / float(
                mode["zero_point_amplitude_amu12_ang"]),
            "outermost_move_ang": {str(n): outermost_move(mode, temperature_k,
                                                          n)
                                   for n in NODES}}


def frameset_from_mode(run_folder, *, mode: int, temperature_k: float,
                       nodes: int = NODES[0], out) -> ModeFrames:
    """THE COMMAND LINE'S AND THE PAGE'S FRAME SET (`engines/vibration.md`
    § 5.10 ③): vibration run ``run_folder``'s mode ``mode`` -- counted as its
    result counts it, ``index_1based`` -- thermal at ``temperature_k``,
    its frames the ``nodes``-point rule of that state, written as one pair
    at ``out`` (its ``.xyz``) by the motion toolkit's one writer
    (`vibration_frames.write_frame_set`), its run read by the toolkit's
    run door (`vibration_frames.vibration_of_run`).

    Refused, naming what to do (:class:`FrameSetError`): what the run door
    and the writer refuse, a mode the result does not hold or an imaginary
    one, a temperature that is not a number of kelvin, 0 or more, another
    rule than :data:`NODES`."""
    from ..structure import Structure
    from ..transport.compose import frames_moving_the_leads
    from ..vibration_frames import (_checked_target, _mode_of,
                                    vibration_of_run, write_frame_set)
    from ..workingcopy_structure import StructureCodec
    nodes = _checked_nodes(nodes)
    temperature_k = _checked_temperature(temperature_k)
    _checked_target(out)
    v = vibration_of_run(run_folder)
    m = _mode_of(v, mode)
    if m.has_imag:
        raise FrameSetError(
            f"mode {mode} is imaginary ({m.frequency_cm1:g} cm-1): it has no "
            f"amplitude, and the geometry is no minimum along it")
    s = m.state(temperature_k=temperature_k)
    f = s.frames(nodes)
    written = write_frame_set(out, f)
    sigma = float(s.spread_amu12_ang)
    frames = tuple({"q_sigma": float(q) / sigma,
                    "max_atom_displacement_ang": float(move),
                    "weight": float(w)}
                   for q, move, w in zip(f.q_amu12_ang[:, 0],
                                         f.max_atom_displacement_ang,
                                         f.weights))
    # AN ELECTRODE ATOM THE FRAMES MOVE -- said in the words of the one
    # check a transport citation asks of every frame
    # (`compose.frames_moving_the_leads`; `engines/transport.md` § 2a.9),
    # until R3d2b2b2b2 leaves it to the citation alone.
    built: Structure = StructureCodec().load(written.path, frames=True)
    moving = frames_moving_the_leads(built, written.path)
    return ModeFrames(path=written.path, mean_quanta=float(s.mean_quanta),
                      sigma_over_zero_point=sigma / float(
                          m.zero_point_amplitude_amu12_ang),
                      frames=frames,
                      said=tuple(written.said) + ((moving,) if moving else ()))


__all__ = ["NODES", "FrameSetError", "ModeFrames",
           "at_temperature", "frameset_from_mode", "outermost_move"]
