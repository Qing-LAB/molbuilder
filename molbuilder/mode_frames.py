"""A MODE'S FRAME SET -- one vibration mode written as one multi-frame
structure pair a transport calculation cites (`engines/vibration.md`
§ 5.10 ③; its rows `model/structure.md` § 2.2f; its rule
`science/vibrational-averaging.md` § 5).

MODULE  mode_frames (L2; the run door, the vibration result's reader, the
        structure and its codec, numpy)
ROLE    :func:`frameset_from_mode` -- the one generator, and
        :class:`ModeFrames`, what it wrote and what it says;
        :func:`outermost_move` and :func:`at_temperature` -- what a
        temperature means for a mode before its set is written: its mean
        number of quanta, its spread and the outermost frames' largest atom
        move for each rule (`science/vibrational-averaging.md` § 5)
USED-BY the command line (`molbuilder frameset from-mode`) and the spectrum
        viewer's doors (`POST /api/spectra/frameset`; what a temperature
        means, at `POST /api/spectra/load`)

**Its extent is the person's: a temperature** (the user, 2026-10-28) --
the junction's, or for a mode pumped out of equilibrium its effective one.
It sets the mode's mean number of quanta ``n`` and the spread of the
Gaussian the frames sample, ``sigma = Q_zp sqrt(2n + 1)``; each frame sits
at a Gauss-Hermite node of it with the rule's own weight, and the
outermost frames' largest atom move is measured and reported -- the check
a person reads against a bond length.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .persist import sha256_of

#: The rules the generator writes -- three frames by default, five or seven
#: on request, each the check of the one before
#: (`science/vibrational-averaging.md` § 5.1, § 5.3).
NODES = (3, 5, 7)

#: How its frames sample the mode, as its ``sampling_rule`` row states it
#: (`model/structure.md` § 2.2f).
SAMPLING_RULE = "gauss-hermite"


class FrameSetError(ValueError):
    """The frame set cannot be written -- the message names what is missing
    and where it is read from."""


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
        path = (str(self.path.relative_to(root)) if root is not None
                else str(self.path))
        return {"path": path, "frames": [dict(f) for f in self.frames],
                "said": list(self.said)}


def _ordered_nodes(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """The ``n``-point Gauss-Hermite rule's nodes ``x`` and weights
    ``w / sqrt(pi)`` -- frame 0 the node at 0, then each pair of nodes
    nearest first, its positive side first (`engines/vibration.md`
    § 5.10 ③)."""
    x, w = np.polynomial.hermite.hermgauss(n)
    order = sorted(range(n), key=lambda i: (round(abs(x[i]), 12), -x[i]))
    return x[order], w[order] / math.sqrt(math.pi)


def _relaxation_record(run) -> Tuple[Optional[Dict[str, Any]], str]:
    """THE RECORD OF THE RELAXATION a vibration run's geometry came from
    (`engines/vibration.md` § 5.10 ③) and, where there is none, why: the
    run's own `relax` call relaxed it at the one template's settings,
    so its own deck's record (`runs.view_of`); a structure stated already
    relaxed (its deck's `vibration` block says so) states its own -- the
    calculation's structure beside it, its ``info.calculation``."""
    from .deck_record import extract_vibration_record
    from .jobset.prep import structure_path
    from .runs import view_of
    from .workingcopy_structure import StructureCodec
    from .parse.fdf import read_deck
    deck = run.deck
    record = (extract_vibration_record(read_deck(deck))
              if deck is not None else None)
    if not (record or {}).get("already_relaxed"):
        own = (view_of(run).info or {}).get("calculation")
        return own, ("" if own is not None else
                     f"{run.label}'s deck records no settings (a PySCF deck "
                     f"states none), so this set states no info.calculation "
                     f"and is no transport citation")
    path = structure_path(run.place.task, run.place.root)
    stated = (StructureCodec().load(path).info or {}).get("calculation") \
        if path is not None else None
    if stated is None:
        return None, (
            f"the structure {run.label} measured was stated already relaxed "
            f"and states no record of the relaxation it came from, so this "
            f"set states no info.calculation and is no transport citation -- "
            f"save the structure from its relaxation run (the Results tab, or "
            f"`molbuilder xv2xyz --from-run`) and run the vibration from it")
    return stated, ""


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
    (`spectra.derived.checked_temperature_k`) -- stated every time, never
    defaulted -- refused as the generator refuses."""
    from .spectra.derived import checked_temperature_k
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
    outermost frames, ``√2 · x_out · σ · max_A |L_A|`` with ``σ`` the
    mode's spread at that temperature (`science/vibrational-averaging.md`
    § 5) -- what :func:`frameset_from_mode` measures on the frames it
    writes, by the same node rule.  ``None`` for a mode with no zero-point
    amplitude (an imaginary one)."""
    from .spectra.derived import thermal_spread_of
    from .spectra.normal_modes import largest_move
    nodes = _checked_nodes(nodes)
    sigma = thermal_spread_of(mode, temperature_k)
    if sigma is None:
        return None
    largest = largest_move(mode["eigenvector_canonical"])
    x_out = float(np.abs(_ordered_nodes(nodes)[0]).max())
    return math.sqrt(2.0) * x_out * float(sigma) * largest


def at_temperature(mode: Dict[str, Any],
                   temperature_k: float) -> Optional[Dict[str, Any]]:
    """WHAT ``temperature_k`` MEANS FOR A MODE ROW before its set is
    written (`web/spectra.md` § 4.3): its mean number of quanta, its spread
    and the outermost move for each rule the generator writes --
    ``{mean_quanta, sigma_amu12_ang, sigma_over_zero_point,
    outermost_move_ang: {"3": …, …}}`` -- from the derivations the
    generator writes by.  ``None`` for a mode with no zero-point
    amplitude."""
    from .spectra.derived import mean_quanta, thermal_spread_of
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


def _relative(path: Path, root: Optional[Path]) -> str:
    """``path`` from the projects tree ``root`` where it lies in it, else
    as it is -- where a person finds the run again."""
    path = Path(path).resolve()
    if root is not None:
        try:
            return str(path.relative_to(Path(root).resolve()))
        except ValueError:
            pass
    return str(path)


def frameset_from_mode(run_folder, *, mode: int, temperature_k: float,
                       nodes: int = NODES[0], out,
                       root: Optional[Path] = None) -> ModeFrames:
    """THE ONE GENERATOR (`engines/vibration.md` § 5.10 ③): vibration run
    ``run_folder``'s mode ``mode`` -- counted as its result counts it,
    ``index_1based`` -- written as one multi-frame pair at ``out`` (its
    ``.xyz``), its frames at the ``nodes`` Gauss-Hermite nodes of the
    Gaussian the mode spreads over at ``temperature_k``.  ``root`` is the
    projects tree the run's folder is stated from in ``info.vibration``.

    Refused, naming what to do: no run of ours -- or a calculation whose
    description does not read, in its reader's words -- no result in it, a
    result stating no masses, a mode it does not hold or an imaginary one,
    a temperature that is not a number of kelvin, 0 or more, another rule
    than :data:`NODES`, a pair already at ``out``."""
    from .chemistry import MASS_SOURCE
    from .frame_average import INFO_KEY, MASS_CHANNEL, ROWS
    from .runfiles import ROLE_SPECTRA
    from .runs import on_label_at, place_of, run_of
    from .siesta.vibration_phases import force_constants_call
    from .sidecars.molstruct import sidecar_path_for
    from .sidecars.spectra import SpectraJsonError, parse_spectra_json
    from .spectra.derived import (mean_quanta, thermal_spread_amu12_ang,
                                  zero_point_amplitude_amu12_ang)
    from .structure import AtomChannel, Structure
    from .workingcopy_structure import StructureCodec

    nodes = _checked_nodes(nodes)
    temperature_k = _checked_temperature(temperature_k)
    out = Path(out)
    if out.suffix != ".xyz":
        raise FrameSetError(f"{out.name}: the pair is written as its .xyz "
                            f"and its .molstruct.json -- name the .xyz")
    taken = [p for p in (out, sidecar_path_for(out)) if p.exists()]
    if taken:
        raise FrameSetError(
            f"{', '.join(p.name for p in taken)} already "
            f"{'exists' if len(taken) == 1 else 'exist'} -- a pair is never "
            f"written over, since a transport calculation citing it is "
            f"pinned to it: choose another name")

    # THE RUN, AND ITS RESULT, through the one door for a result on the
    # label (`runs.on_label_at`): a run's own folder, or a run of points'
    # root, where its finish writes it (`engines/vibration.md` § 5.11 § 2).
    place = place_of(Path(run_folder))
    if not place.ours or place.problem:
        # WHY IT IS NO RUN, in the run door's words: a calculation whose
        # description does not read says so, naming its way out.
        raise FrameSetError(
            f"{run_folder}: its calculation's description does not read -- "
            f"{place.problem}" if place.problem else
            f"{run_folder}: no run of ours -- name the folder of the "
            f"vibration run whose mode is written")
    result = on_label_at(Path(run_folder), ROLE_SPECTRA)
    if result is None:
        raise FrameSetError(
            f"{run_folder}: the run holds no vibration result "
            f"({place.task.label}{ROLE_SPECTRA}) -- one whose finish has "
            f"written it")
    # THE RUN WHOSE DECK MEASURED THE FORCE CONSTANTS: the folder named --
    # PySCF's one run -- or, a run of points' root, its force-constant call
    # (`vibration_phases.force_constants_call`).
    run = run_of(Path(run_folder)) or run_of(force_constants_call(run_folder))
    if run is None:
        raise FrameSetError(
            f"{run_folder}: holds {result.name} but no run that measured it")
    try:
        res = parse_spectra_json(result)
    except SpectraJsonError as exc:
        raise FrameSetError(f"{result.name}: {exc}") from exc
    if res.equilibrium_masses_amu is None:
        raise FrameSetError(
            f"{result.name} states no masses (a result before schema 7): "
            f"the mode's coordinate is weighted by them -- run the "
            f"vibration again with this molbuilder")
    found = [m for m in res.modes if int(m.index_1based) == int(mode)]
    if not found:
        raise FrameSetError(
            f"{result.name} holds modes 1 to {len(res.modes)}; there is no "
            f"mode {mode}")
    m = found[0]
    q_zp = zero_point_amplitude_amu12_ang(m.frequency_cm1, m.has_imag)
    if q_zp is None:
        raise FrameSetError(
            f"mode {mode} is imaginary ({m.frequency_cm1:g} cm-1): it has no "
            f"amplitude, and the geometry is no minimum along it")
    stated = res.structure_stated()
    if stated is None:
        raise FrameSetError(f"{result.name} states no geometry yet")
    base = Structure.from_dict(stated)

    # THE MODE OVER EVERY ATOM: the canonical eigenvector on the free atoms,
    # zero on every held one -- so no held atom moves.
    # THE SPREAD AT THE TEMPERATURE -- n quanta, sigma = Q_zp sqrt(2n + 1)
    # (`science/vibrational-averaging.md` § 5) -- and each frame at a node,
    # q_j = sqrt(2) sigma x_j, by THE ONE DISPLACEMENT RULE
    # (`spectra.normal_modes`): R + q·L, what a Raman phase's plan applies
    # at ±δQ too (§ 5.11 § 3).
    from .spectra.normal_modes import displaced, mode_over_atoms
    L = mode_over_atoms(m.eigenvector_canonical, res.free_atom_idxs,
                        base.n_atoms)
    n_quanta = mean_quanta(m.frequency_cm1, temperature_k)
    sigma = thermal_spread_amu12_ang(q_zp, m.frequency_cm1, temperature_k)
    x, weights = _ordered_nodes(nodes)
    qs = [math.sqrt(2.0) * sigma * float(xj) for xj in x]
    r0 = np.asarray(base.positions, dtype=float)
    built = base.with_frames(np.stack([displaced(r0, L, q) for q in qs]))

    # FRAME 0'S SETTINGS -- the record of the relaxation its geometry came
    # from, the one a transport citation takes its settings from
    # (`engines/vibration.md` § 5.10 ③, `transport.md` § 3.1).
    said: List[str] = []
    calculation, why = _relaxation_record(run)
    if calculation is not None:
        built.set_info("calculation", calculation)
    else:
        said.append(why)

    # THE ROWS, in `model/structure.md` § 2.2f's vocabulary: each with its
    # unit and its meaning, the row's note, as the vocabulary states them.
    masses = [float(v) for v in res.equilibrium_masses_amu]
    built.set_channel(MASS_CHANNEL, AtomChannel(
        "value", {i: v for i, v in enumerate(masses)}))
    vocabulary = {r.name: r for r in ROWS}

    def write(name: str, value: Any, frame: Optional[int] = None) -> None:
        built.set_customized(name, value, unit=vocabulary[name].unit,
                             note=vocabulary[name].means, frame=frame)

    frames = []
    for q, w in zip(qs, weights):
        frames.append({"q_sigma": q / sigma,
                       "max_atom_displacement_ang":
                           float(np.linalg.norm(q * L, axis=1).max()),
                       "weight": float(w)})
    # THE OUTERMOST MOVE, measured on the frames written -- what the person
    # reads against a bond length.
    outermost = max(f["max_atom_displacement_ang"] for f in frames)
    stated = {"mode_index_1based": int(m.index_1based),
              "frequency_cm1": float(m.frequency_cm1),
              "temperature_k": float(temperature_k),
              "mean_quanta": float(n_quanta),
              "zero_point_amplitude_amu12_ang": float(q_zp),
              "max_displacement_ang": outermost,
              "sigma_amu12_ang": float(sigma),
              "sampling_rule": SAMPLING_RULE,
              "sampling_nodes": int(nodes),
              "mass_source": MASS_SOURCE}
    # IN THE VOCABULARY'S ORDER (`model/structure.md` § 2.2f).
    for r in ROWS:
        if r.where == "structure":
            write(r.name, stated[r.name])
    for f, (q, row) in enumerate(zip(qs, frames)):
        measured = {"displacement_ang": float(np.linalg.norm(q * L)),
                    "max_atom_displacement_ang":
                        row["max_atom_displacement_ang"],
                    "q_amu12_ang": q,
                    "node_sigma": row["q_sigma"],
                    "weight": row["weight"]}
        for r in ROWS:
            if r.where == "frame":
                write(r.name, measured[r.name], frame=f)
    built.set_info(INFO_KEY, {
        "run": _relative(Path(run_folder), root),
        "result_sha256": sha256_of(result)})

    # AN ELECTRODE ATOM THE FRAMES MOVE -- written, and said in the words
    # of the one check a transport citation asks of every frame
    # (`compose._frames_keep_the_leads`, within `wizard.FROZEN_TOL_ANG`;
    # `engines/transport.md` § 2a.9), so what the person is told is what
    # the citation does.
    from .transport.compose import ComposeError, _frames_keep_the_leads
    try:
        _frames_keep_the_leads(built, out)
    except ComposeError as exc:
        said.append(str(exc))
    StructureCodec().write(built, out)
    return ModeFrames(path=out, mean_quanta=float(n_quanta),
                      sigma_over_zero_point=float(sigma / q_zp),
                      frames=tuple(frames), said=tuple(said))


__all__ = ["NODES", "SAMPLING_RULE", "FrameSetError", "ModeFrames",
           "at_temperature", "frameset_from_mode", "outermost_move"]
