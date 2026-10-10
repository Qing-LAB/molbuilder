"""The transport composite's RECORD — `engines/transport.md` § 2a.12,
build step P6.

TBtrans's own outputs are the truth this module reads: the k-averaged
transmission file (``<label>.TBT.AVTRANS_<L>-<R>``, two columns E vs T)
and the current line its ``.out`` prints (the binary's own Landauer
integral — parsed, never recomputed).  Both formats were pinned against
a REAL 5.4.2 run (the carbon-chain live walk, 2026-08-29).

What lands on disk is ONE file at the calculation root,
``<label>.transport.json`` (``molbuilder/transport-result@5``): T(E)
per point -- a frame and a voltage -- the I–V table per frame, the mode's
average over a frame set (`average.mode_average`), the junction's TOTAL
current, with the
figure TBtrans printed and the factor between them (:data:`CURRENT_MEANS`)
-- and the provenance that says which junction built it — the citation (from ``slot-provenance.json``) and
the atom-permutation reference, so every downstream index can be
mapped back to the relaxation's identities.

Reading is ASYNCHRONOUS by design, the same doctrine as the bench
summarizer: a point whose transmission has not run yet reads as
*pending*, never as a failure of the set, and one whose run ended without
its transmission reads as *failed*, in its run's own words — `summarize`
is a reader, and nothing is produced on a host that has produced nothing.

**COMPOSED ON READ** (`engines/transport.md` § 2a.12): every rung's state is
the one status door's (`runstatus.jobset_status`, what `jobset status` and
the Results tab's ladder say), and every point's its run's (`run_status`);
the Results tab asks `/api/transport/record`, which composes the record
each time it is opened, so a rung that ran since `summarize` is never shown
as it was.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ..atom_permutation import PERMUTATION_FILE
from ..constants import BOLTZMANN_EV_K, CONDUCTANCE_QUANTUM_S

if TYPE_CHECKING:
    from .stages import Point

#: ``@5``: the ``average`` carries no slope or curvature, and states the
#: mode's definition by its own names -- the structure's rows of a mode's
#: frame set (`model/structure.md` § 2.2f) -- its weights checked at the
#: citation door.
#: ``@4``: a point is a frame and a voltage -- every point, pending and
#: failed entry carries its ``frame`` (``None`` on a calculation with no
#: frame axis) and its folder ``tokens``, a point its frame's
#: ``customized`` rows whole; the I-V carries a ``frame`` column; the
#: ``average`` block is the mode's average over a frame set (§ 2a.12).
#: ``@3``: each rung's ``state`` is the status door's word with its
#: ``detail``; a point without a transmission is ``pending`` or ``failed``;
#: ``current_a`` is the junction's TOTAL current, TBtrans's printed figure
#: beside it.  An older record is refused by its version, and `summarize
#: task` writes it again.
TRANSPORT_RESULT_SCHEMA = "molbuilder/transport-result@5"

#: What the record's current IS, said in the record (`engines/transport.md`
#: § 2a.4; user, 2026-10-03, Q5: "make sure the result presentation, data
#: record and the summary/comments clearly explain what is what").  TBtrans's
#: Landauer integral is one spin channel's -- I = (e/h)∫T, no factor 2 for
#: spin (`m_tbt_save.F90`) -- while the conductance beside it is in
#: G0 = 2e²/h, both channels.
CURRENT_MEANS = {
    "non-polarized": (
        "current_a is the junction's total current, both spin channels: "
        "TBtrans prints one spin channel's (its Landauer integral has no "
        "factor 2 for spin), and the two channels of a non-polarized "
        "calculation carry the same current, so the total is twice the "
        "printed figure (current_a_printed, as TBtrans printed it)."),
    "polarized": (
        "current_a is the junction's total current, the sum of its two spin "
        "channels' currents as TBtrans printed each (empty when it printed "
        "only one); current_a_printed is the channel it printed first."),
}

#: THE CAVEAT that goes with any DFT-NEGF conductance (`engines/transport.md`
#: § 1 and § 2a.12), carried on the result so every reader says it.
CAVEAT = (
    "A DFT-NEGF conductance with a plain GGA functional places the "
    "molecule's levels too close to the Fermi level, and so overestimates a "
    "molecular junction's conductance, often by one to two orders of "
    "magnitude.  Read T(E) for where the levels are and how they couple, "
    "not for an absolute conductance.")

#: The SCF row fields the report draws (`web/results.md` § 2.5) -- the
#: SIESTA reader's own names; a NEGF row carries its charge too.
_SCF_FIELDS = ("cycle", "energy", "delta_E", "dDmax", "dHmax", "ef",
               "phase", "dq", "vha_ev", "charges")

#: HOW FAR PAST THE BIAS WINDOW the transmission must reach for a current
#: computed from T(E, 0) to be whole: the Fermi functions' tails, in kT
#: (`engines/transport.md` § 2a.10; the window gate, plan § 5x P2).
WINDOW_TAILS_KT = 5.0


class RecordError(Exception):
    """The record cannot be built — the message names what to run or
    fix first, ready to surface verbatim."""


def _one_frame_or_refuse(base: Path) -> None:
    """THE PDOS OF A SELECTION AT A FRAME IS NOT BUILT YET (plan § 5z
    Q17-f, with the Results tab's frame bar): its door names a voltage and
    no frame, so a calculation citing a set of more than one frame is
    refused here by name, never read at another frame's point."""
    from .stages import frames_of
    n = frames_of(base)
    if n > 1:
        raise RecordError(
            f"this calculation cites a frame set of {n} frames, and the PDOS "
            f"of a selection names no frame yet (plan § 5z Q17-f, with the "
            f"Results tab's frame bar).  Each frame's DOS and PDOS by region "
            f"are in the record's points.")


def record_path(base_dir, label: str) -> Path:
    """The ONE spelling of the record's location -- composed, not concatenated.

    `runfiles.WRITTEN` declares `.transport.json` and its module docstring says
    *"Nothing composes or splits a run-file name inline."*  Composing also
    inherits the label rule -- `compose` refuses `my.relax`, a name
    `runfiles.parse` cannot read back.
    """
    from ..runfiles import compose as _rf
    return Path(base_dir) / _rf(label, ".transport.json")


def parse_avtrans(text: str) -> Tuple[List[float], List[float]]:
    """``<label>.TBT.AVTRANS_<L>-<R>`` → ``(energies_ev, transmission)``.

    The format (pinned live, 5.4.2): ``#`` comment lines, then two
    columns — E in eV, and the k-averaged T(E).

    E is relative to E_F -- TBtrans's own convention for its contour and
    its output (`engines/transport.md` § 2a.12: T(E) is measured relative
    to the leads' E_F), which is what the record states as
    ``energies_relative_to_ef``.
    """
    energies: List[float] = []
    trans: List[float] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        try:
            e, t = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        energies.append(e)
        trans.append(t)
    if not energies:
        raise RecordError("no transmission rows parsed -- not an "
                          "AVTRANS file?")
    return energies, trans


def deck_spin(run_dir) -> str:
    """The spin the run's own deck states -- ``non-polarized`` or
    ``polarized`` -- read from the deck in its run directory, the input the
    run used (every SIESTA deck molbuilder writes states ``Spin``).  No ``Spin`` line is SIESTA's own default, non-polarized."""
    from ..parse.fdf import _parse_fdf
    from ..runfiles import find_by_role
    # THE FRAMEWORK'S SEARCH for the deck (`project-layout.md` § 4.5), not a
    # glob of its suffix here.
    for deck in find_by_role(run_dir, ".fdf"):
        scalars, _blocks = _parse_fdf(deck.read_text(encoding="utf-8",
                                                     errors="replace"))
        said = (scalars.get("spin") or [None])[0]
        if said:
            return ("polarized" if str(said).strip().lower() == "polarized"
                    else "non-polarized")
    return "non-polarized"


def point_transmission(where, label: str, spin: str) -> Dict:
    """What TBtrans wrote at one point, read through the family's one reader
    (`parse.engines.tbtrans.transmission_files`): ``energy_ev`` and
    ``transmission`` -- **per spin channel, so G = G0 · T(E_F)** with
    G0 = 2e²/h: the unpolarized file's T as it is; a polarized point's two
    channels averaged, (T↑ + T↓)/2, each kept whole in ``channels`` -- and
    ``transmission_file``.  :class:`RecordError` names what is missing: a
    polarized deck whose run wrote one channel is not done
    (`engines/transport.md` § 2a.12)."""
    from ..parse.engines.tbtrans import transmission_files
    files = transmission_files(where, label)
    if spin == "polarized":
        missing = [c for c in ("up", "down") if not files.get(c)]
        if missing:
            raise RecordError(
                f"the deck is spin-polarized and the run wrote no "
                f"{' or '.join(missing)} channel transmission")
        e_up, t_up = parse_avtrans(files["up"][0].read_text())
        e_dn, t_dn = parse_avtrans(files["down"][0].read_text())
        if e_up != e_dn:
            raise RecordError("the two spin channels' energy grids differ")
        return {"energy_ev": e_up,
                "transmission": [(a + b) / 2.0 for a, b in zip(t_up, t_dn)],
                "channels": {"up": t_up, "down": t_dn},
                "transmission_file": [files["up"][0].name,
                                      files["down"][0].name]}
    if not files.get("unpolarized"):
        raise RecordError("the run wrote no transmission file")
    energies, trans = parse_avtrans(files["unpolarized"][0].read_text())
    return {"energy_ev": energies, "transmission": trans,
            "transmission_file": files["unpolarized"][0].name}


def point_currents(where, token: str, spin: str) -> Dict:
    """The current TBtrans printed at one point -- its own Landauer integral,
    parsed, never recomputed -- as the record says it
    (:data:`CURRENT_MEANS`): ``current_a_printed``, the first figure it
    printed (one spin channel's); ``current_a``, the junction's total --
    twice it for a non-polarized run, the two channels' sum for a polarized
    one -- and the run's ``k_points`` / ``k_method``.  Read from the point's
    newest output through the family's reader
    (`parse.engines.tbtrans.read_tbtrans_out`); ``None`` where the run
    printed none."""
    from ..parse.engines.tbtrans import read_tbtrans_out
    out: Dict = {"current_a": None, "current_a_printed": None}
    for path in _outs_newest_first(where, token):
        facts = read_tbtrans_out(path.read_text(errors="replace"))
        rows = facts.get("currents") or []
        for k in ("k_points", "k_method"):
            if k in facts:
                out[k] = facts[k]
        if not rows:
            continue
        # ONE ELECTRODE PAIR -- the first printed; a channel per pass.
        pair = (rows[0]["from"], rows[0]["to"])
        mine = [r for r in rows if (r["from"], r["to"]) == pair
                and r.get("current_a") is not None]
        if not mine:
            continue
        out["current_a_printed"] = mine[0]["current_a"]
        if spin == "polarized":
            by = {r.get("channel"): r["current_a"] for r in mine}
            if "up" in by and "down" in by:
                out["current_a"] = by["up"] + by["down"]
        else:
            out["current_a"] = 2.0 * mine[0]["current_a"]
        break
    return out


def deck_temperature_k(run_dir) -> float:
    """The electronic temperature the point's own deck states, in kelvin
    (`units.temperature_k`: ``300 K`` or an energy), SIESTA's default 300 K
    when the deck states none."""
    from ..parse.fdf import _parse_fdf
    from ..runfiles import find_by_role
    from ..units import temperature_k
    for deck in find_by_role(run_dir, ".fdf"):
        scalars, _blocks = _parse_fdf(deck.read_text(encoding="utf-8",
                                                     errors="replace"))
        said = scalars.get("electronictemperature")
        if said:
            return temperature_k(float(said[0]),
                                 said[1] if len(said) > 1 else "K")
    return 300.0


def window_short_of(emin_ev: float, emax_ev: float, volts: float,
                    kt_ev: float) -> Optional[str]:
    """ONE RULE for the transmission window against a bias
    (`engines/transport.md` § 2a.10, P2): the window must reach
    ±(|V|/2 + :data:`WINDOW_TAILS_KT`·kT) -- both leads' Fermi tails at
    μ = ±V/2 -- or the current integrated over it is cut short without a
    word from TBtrans.  ``None`` when it reaches; else the sentence naming
    the reach and the fix, said alike by the record (a voltage it does not
    integrate), the transmission deck's gate and the description's
    preflight."""
    half = abs(float(volts)) / 2.0
    reach = half + WINDOW_TAILS_KT * float(kt_ev)
    lo, hi = float(emin_ev), float(emax_ev)
    if reach > hi or -reach < lo:
        return (f"the transmission window [{lo:g}, {hi:g}] eV does not reach "
                f"±{reach:.3f} eV for {float(volts):g} V (V/2 + "
                f"{WINDOW_TAILS_KT:g} kT at kT = {float(kt_ev):.4f} eV); "
                f"widen it -- transmission_emin_ev / transmission_emax_ev")
    return None


def linear_response_iv(energies: List[float], trans: List[float],
                       voltages: List[float], kt_ev: float) -> Dict:
    """**I(V) = (2e/h) ∫ T(E, 0) [f(E − μ_L) − f(E − μ_R)] dE**, μ = ±eV/2 --
    the low-bias approximation's current, computed by the record from the
    one zero-bias slice for each listed voltage (`engines/transport.md`
    § 2a.10): G0 times the integral in eV, ``trans`` per spin channel as
    :func:`point_transmission` gives it.  A voltage whose window plus the
    Fermi tails (:data:`WINDOW_TAILS_KT`) reaches past the transmission's
    energy window gets ``None`` and a note naming the reach -- the record
    never integrates a slice it does not have."""
    import numpy as np
    e = np.asarray(energies, dtype=float)
    t = np.asarray(trans, dtype=float)
    lo, hi = float(e.min()), float(e.max())
    currents: List[Optional[float]] = []
    notes: Dict[str, str] = {}
    for v in voltages:
        half = abs(float(v)) / 2.0
        short = window_short_of(lo, hi, v, kt_ev)
        if short:
            currents.append(None)
            notes[f"{v:g}"] = short
            continue
        with np.errstate(over="ignore"):
            f_l = 1.0 / (1.0 + np.exp((e - half) / kt_ev))
            f_r = 1.0 / (1.0 + np.exp((e + half) / kt_ev))
        currents.append(float(CONDUCTANCE_QUANTUM_S
                              * np.trapezoid(t * (f_l - f_r), e)))
    return {"voltages_v": [float(v) for v in voltages],
            "current_a": currents,
            "computed": "linear-response", "kt_ev": kt_ev,
            "window_ev": [lo, hi], "notes": notes}


def conductance_g0(energies: List[float], trans: List[float]
                   ) -> Optional[float]:
    """T interpolated at E = 0 (the grid is E − E_F) — G(E_F) in G0."""
    import numpy as np
    e = np.asarray(energies, dtype=float)
    t = np.asarray(trans, dtype=float)
    if e.min() > 0 or e.max() < 0:
        return None                    # window does not straddle E_F
    return float(np.interp(0.0, e, t))


def _outs_newest_first(where: Path, token: str) -> List[Path]:
    """A rung's engine outputs in ``where``, its NEWEST RUN first -- the
    run door's (`runs.Run.outputs`), by the run number every one carries,
    never by file time (plan W38 M4): a copy, a touch or a restored folder
    reorders the times and not the runs.  ``[]`` where no run of ours is."""
    from ..runs import run_of
    run = run_of(where, stage=token)
    return run.outputs if run is not None else []


def result_folders(base: Path, task, stage: str, run: Optional[Path] = None
                   ) -> List[Tuple["Point", Optional[Path], Optional[Path]]]:
    """``(point, folder, run)`` of each result ``stage`` holds now: a rung
    that runs per point (`stages.rung_points`: a frame, a voltage, or both),
    its latest run's point folders; any other rung's latest run, as the one
    point ``Point()`` at 0 V -- ``folder`` ``None`` while no run is open
    (`engines/transport.md` § 2a.11).  ``run`` is where the launch record
    lies; given, that run is read instead of the latest (the device run
    the transmission gathered, § 2a.12)."""
    from ..jobset.materialize import latest_attempt, stage_home
    from .stages import Point, frames_of, points_in, rung_points
    if run is None:
        run = latest_attempt(stage_home(base, task, stage).dir)
    pts = rung_points(task, stage, frames=frames_of(base))
    if pts:
        if run is None:
            return [(pt, None, None) for pt in pts]
        return [(pt, p, run)
                for p, pt in points_in(run, task, stage, base=base)]
    return [(Point(), run, run)]


def _stage_facts(base: Path, task, label: str) -> List[Dict]:
    """One entry per rung of the ladder: where it stands, and its own answer.

    **A transport result is FIVE calculations, and the record says so**, so
    a reader can tell a run that has not started from one stalled at the
    device (`engines/transport.md` § 1).

    **WHERE EACH RUNG STANDS is the one status door's answer**
    (`runstatus.jobset_status` -- what `jobset status` prints and the
    Results tab's ladder draws): its ``state`` and ``detail`` in that
    door's words, a bias sweep's rung speaking from its first point not
    finished.  Nothing here reads a run's state a second way.

    **Each rung's own key fact, which is not the same fact**, read from the
    run's engine output once it has one:

    * *seed* / *device* -- did the SCF converge, and at what energy.  The seed
      hands the device a density; the device hands TBtrans a Hamiltonian.
    * *electrode_L* / *electrode_R* -- **the lead's Fermi level**, and this is
      the number the whole junction is referenced to: a lead is a periodic
      BULK run whose *"E_F is the reference energy"* (§ 2a.13), T(E) is
      measured relative to it, and `G = G0 * T(E_F)` is evaluated at it.  Two
      leads that disagree is a defect nothing else on the Results tab shows.
    * *transmission* -- has its own `points` / `pending` / `failed` blocks.
    """
    from ..jobset.materialize import ladder_homes
    from ..jobset.model import FILENAME as JOBSET_FILENAME, JobSet
    from ..jobset.runstatus import jobset_status
    from .stages import STAGE_FACT, TRANSPORT_STAGES, frames_of, rung_points

    jpath = base / JOBSET_FILENAME
    status = jobset_status(JobSet.load(jpath) if jpath.is_file() else None,
                           base)
    by_name = {s.name: s for s in status.stages}
    tokens = {h.name: h.token for h in ladder_homes(base, task)}
    # THE DEVICE RUN THE TRANSMISSION READ, when it has one -- the device
    # facts shown beside T(E) are that run's, never the newest by itself
    # (`engines/transport.md` § 2a.12).
    device_run = gathered_device_run(base, task)
    out: List[Dict] = []
    for name in TRANSPORT_STAGES:
        s = by_name.get(name)
        if s is None:                          # the description omits it
            out.append({"stage": name, "token": None,
                        "state": "not_described"})
            continue
        token = tokens.get(name)
        fact: Dict = {"stage": name, "token": token, "state": s.state,
                      "detail": s.detail}
        run = None
        if name == "device" and device_run is not None:
            run = device_run
            fact["attempt"] = str(run.relative_to(base))
            fact["gathered_by"] = "transmission"
        elif s.attempt and s.dir:
            fact["attempt"] = f"{s.dir}/{s.attempt}"
            run = base / s.dir / s.attempt
        # WHICH QUESTION THIS RUNG ANSWERS -- a column, not a branch on the
        # name (`stages.STAGE_FACT`).  The transmission's answer is its
        # points; TBtrans converges nothing and reports no total energy.
        answers = STAGE_FACT.get(name, "scf")
        # EVERY SWEPT RUNG'S POINTS, as the status door reads them: the
        # folder each ran in, what it started from and what it alone took
        # (§ 2a.11, § 2a.12) -- the provenance chain's per-point lines, for
        # a product rung (the transmission) as for the device.  THE RUN THE
        # TRANSMISSION GATHERED answers for its own points when it is not
        # the stage's newest (§ 2a.12): the same per-point reader
        # (`runstatus.point_rows`), asked for that run.
        points = s.points
        if run is not None and s.attempt and s.dir \
                and run != base / s.dir / s.attempt and s.points:
            from ..jobset.runstatus import point_rows
            from ..runfiles import RunNames
            from ..runrecord import launch_record
            from .stages import products_of
            _names = RunNames.of(task.label, token, task.shape)
            try:
                points, _ = point_rows(
                    run, _names, task, name, launch=launch_record(run, _names),
                    products=products_of(name, task.label, base_dir=base),
                    base=base)
            except Exception as exc:              # noqa: BLE001 -- said, never hidden
                fact["unreadable"] = f"{run.relative_to(base)}: {exc}"
                points = []
        if points:
            fact["taken"] = [
                {"frame": p.get("frame"), "bias_v": p.get("bias_v"),
                 "attempt": (f"{s.dir}/{p['folder']}" if s.dir and p.get("folder")
                             else None),
                 "started_from": p.get("started_from"),
                 "took": p.get("took") or []}
                for p in points]
        if rung_points(task, name, frames=frames_of(base)) \
                and answers != "product":
            # A RUNG THAT RUNS PER POINT, POINT BY POINT -- the report's
            # convergence card follows the selected point (`web/results.md`
            # § 2.5); the rung's own facts are its first point's.
            fact["by_point"] = _rung_points(base, task, name, token, answers,
                                            run=run)
            # WHAT EACH POINT STARTED FROM AND TOOK -- the status door's own
            # reading of the point's `.continued-from` and `.gathered-from`
            # (`StageStatus.points`; § 2a.12: the provenance of a sweep's
            # points), joined by the point -- its frame and its voltage;
            # never a second reader here.
            said = {(p.get("frame"), float(p["bias_v"])): p
                    for p in (points or ()) if p.get("bias_v") is not None}
            for p in fact["by_point"]:
                sp = said.get((p.get("frame"), float(p["bias_v"])))
                if sp is not None:
                    p["started_from"] = sp.get("started_from")
                    p["took"] = sp.get("took") or []
            # THE RUN THE TRANSMISSION GATHERED answers for itself when it is
            # not the stage's newest (§ 2a.12: the device facts beside T(E)
            # are that run's): its points' states, and a detail naming both.
            if run is not None and s.attempt and s.dir \
                    and run != base / s.dir / s.attempt:
                states = [p.get("state") for p in fact["by_point"]]
                fact["state"] = ("finished" if states
                                 and all(st == "finished" for st in states)
                                 else next((st for st in states
                                            if st != "finished"), "unknown"))
                fact["detail"] = (f"the run the transmission gathered "
                                  f"({fact['attempt']}); the stage's newest "
                                  f"run is {s.dir}/{s.attempt}: {s.detail}")
            first = next((p for p in fact["by_point"]
                          if p.get("energy_ev") is not None), None)
            if first is not None:
                fact.update({k: first[k] for k in first
                             if k not in ("bias_v", "frame", "point",
                                          "attempt", "state", "detail",
                                          "started_from", "took")})
                fact["facts_at_v"] = first["bias_v"]
                fact["facts_at_frame"] = first["frame"]
                fact["facts_at"] = first["point"]
            out.append(fact)
            continue
        outs = (_outs_newest_first(run, token)
                if answers != "product" and run is not None else [])
        if not outs:
            out.append(fact)
            continue
        fact.update(_science(outs[0], answers))
        out.append(fact)
    return out


def gathered_device_run(base: Path, task) -> Optional[Path]:
    """The device run the transmission's newest run gathered its
    Hamiltonians from -- read from the transmission's first point's own
    ``.gathered-from`` (`runrecord.read_gathered_from`), the run above the
    point named there -- or ``None`` while the transmission has gathered
    nothing (`engines/transport.md` § 2a.12: the provenance is what was
    gathered, never the newest run by file time)."""
    from ..runrecord import read_gathered_from
    from .stages import frames_of, rung_points
    device = rung_points(task, "device", frames=frames_of(base))
    # A DEVICE THAT RUNS PER POINT names its point: the run is as many
    # folders above it as the point has levels (`f001/v0.2`: two).
    levels = len(Path(device[0].rel).parts) if device else 0
    for _pt, where, _run in result_folders(base, task, "transmission"):
        if where is None:
            continue
        for g in read_gathered_from(where):
            if g["file"].endswith(".TS.HSX"):
                src = base / g["from"]
                for _ in range(levels):
                    src = src.parent
                return src
    return None


def _science(out: Path, answers: str) -> Dict:
    """A rung run's own answer, read from its engine output: its SCF as it
    ran (every row of its last step, phase-tagged, with each phase's
    criteria and how it ended), whether it converged and at what energy,
    a NEGF phase's own figures, a lead's E_F -- or ``unreadable`` with the
    parser's first sentence."""
    from ..parse import detect
    fact: Dict = {}
    try:
        res = detect(str(out)).parse(str(out))
    except Exception as exc:                   # a refusal is an ANSWER here
        # THE FIRST SENTENCE, not the essay: `UnknownFormatError` lists
        # every registered parser, which buries the other four rungs.
        fact["unreadable"] = str(exc).split(". ")[0].strip() or str(exc)
        return fact
    fact["scf_converged"] = res.scf_converged
    frames = res.frames or []
    info = res.runtime_info or {}
    # THE RUN'S SCF, AS IT RAN (`web/results.md` § 2.5): every row of the
    # last step, phase-tagged -- a device's periodic start and its NEGF
    # loop apart -- with what each phase had to reach and how it ended.
    hist = (frames[-1].scf_history or []) if frames else []
    fact["scf"] = [{k: c[k] for k in _SCF_FIELDS if k in c} for c in hist]
    if info.get("scf_criteria"):
        fact["scf_criteria"] = info["scf_criteria"]
    if info.get("scf_phases"):
        fact["scf_phases"] = info["scf_phases"]
    # THE NEGF PHASE'S OWN FIGURES -- never the periodic start's
    # (`engines/transport.md` § 2a.12): its last row's E_F and charge, and
    # how many cycles it ran.
    negf = [c for c in hist if c.get("phase") == "negf"]
    if negf:
        last = negf[-1]
        # `vha_ev` is TranSIESTA's Hartree potential on the boundary plane,
        # the shift between the device's frame and the seed's (§ 2a.12).
        fact["negf"] = {"cycles": len(negf),
                        **{k: last[k] for k in ("ef", "dq", "charges",
                                                "energy", "vha_ev")
                           if k in last}}
        # THE CONTOUR THE DENSITY WAS INTEGRATED ON (§ 2a.12: "the contour
        # and pole count TranSIESTA used"), from the engine's own start-up
        # echo (`siesta_reader` keeps it under `runtime_info.transiesta`):
        # the equilibrium contour's pole count, and the non-equilibrium
        # line's window and points when the run had a bias.
        contour = _contour_facts((info.get("transiesta") or {})
                                 .get("contours") or {})
        if contour:
            fact["negf"]["contour"] = contour
    if frames:
        fact["energy_ev"] = frames[-1].energy
        # THE RUN'S FERMI LEVEL, from the last SCF cycle of the last frame
        # -- the converged one: a lead's is the reference energy the whole
        # junction is measured against, the seed's is the periodic
        # junction's, the device's its NEGF loop's (the last row is NEGF).
        # Asked by the COLUMN, so adding a third lead one day is a table
        # row and not a third name here.
        for cyc in reversed(hist):
            if cyc.get("ef") is not None:
                fact["fermi_ev"] = cyc["ef"]
                break
    return fact


def _contour_facts(contours: Dict) -> Dict:
    """The contour TranSIESTA echoed, in the record's words: the pole count
    of its equilibrium (continued-fraction) contour -- one number, every
    chemical potential's segment states it -- and, under a bias, the
    non-equilibrium line's window and points.  ``{}`` when the echo holds
    neither."""
    out: Dict = {}
    poles = []
    for segs in contours.values():
        for seg in segs:
            n = seg.get("Number of poles")
            if n is not None:
                try:
                    poles.append(int(str(n).split()[0]))
                except ValueError:
                    pass
            if "line contour points" in seg:
                try:
                    out["neq_line"] = {
                        "emin_ev": float(str(seg.get("line contour E_min", "")).split()[0]),
                        "emax_ev": float(str(seg.get("line contour E_max", "")).split()[0]),
                        "points": int(str(seg["line contour points"]).split()[0])}
                except (ValueError, IndexError):
                    pass
    if poles:
        out["poles"] = poles[0]
    return out


def _rung_points(base: Path, task, name: str, token: str,
                 answers: str, run: Optional[Path] = None) -> List[Dict]:
    """``[{frame, bias_v, point, attempt, state, detail, ...science}]`` --
    each point of a rung that runs per point, its state the run door's
    (`run_status`) and its own answer (:func:`_science`); ``run`` names the
    run to read instead of the latest."""
    from ..parse.dirs import run_status
    from ..runfiles import RunNames
    from ..runrecord import LaunchRecordError, launch_record
    names = RunNames.of(task.label, token, task.shape)
    out: List[Dict] = []
    for pt, where, run in result_folders(base, task, name, run=run):
        entry: Dict = {"frame": pt.frame, "bias_v": pt.bias_v,
                       "point": pt.words()}
        if where is None or not where.is_dir():
            from ..jobset.runstatus import MISSING
            entry.update(state=MISSING[0], detail=MISSING[1])
            out.append(entry)
            continue
        entry["attempt"] = str(where.relative_to(base))
        try:
            st = run_status(where, names.stem,
                            launch=launch_record(run, names))
            entry.update(state=st.state, detail=st.detail)
        except LaunchRecordError as exc:
            entry.update(state="unreadable", detail=str(exc))
        outs = _outs_newest_first(where, token)
        if outs:
            entry.update(_science(outs[0], answers))
        out.append(entry)
    return out


def device_regions(base: Path) -> Dict[str, List[int]]:
    """The device's regions molbuilder owns -- L-electrode, bridge,
    R-electrode -- as atom indices (0-based, the composed junction's order,
    which is every rung's deck's), read from the calculation's composed
    junction (`compose`'s copy and its sidecar).  ``{}`` before it is
    composed."""
    from ..runfiles import JUNCTION_FILE
    from ..workingcopy_structure import StructureCodec
    from .sort import ELECTRODE_LABELS, REGION_BRIDGE
    path = Path(base) / JUNCTION_FILE
    if not path.is_file():
        return {}
    regions = StructureCodec().load(path).regions or {}
    return {label: [int(i) for i in regions.get(label, ())]
            for label in (ELECTRODE_LABELS[0], REGION_BRIDGE,
                          ELECTRODE_LABELS[1])
            if regions.get(label)}


def _chain(base: Path, stages: List[Dict]) -> List[Dict]:
    """``[{stage, attempt, gathered: [{file, from}]}]`` -- what each rung's
    attempt was gathered from, read from its `.gathered-from`
    (`runrecord.read_gathered_from`); a rung that gathered nothing is
    listed with ``[]``."""
    from ..runrecord import read_gathered_from
    out = []
    for st in stages:
        if not st.get("attempt"):
            continue
        row = {"stage": st["stage"], "attempt": st["attempt"],
               "gathered": read_gathered_from(base / st["attempt"])}
        # A SWEPT RUN'S POINTS, each with what it started from and what it
        # alone took (the status door's reading, carried by the rung's
        # facts) -- a transmission point's device Hamiltonian, a device
        # point's start (§ 2a.11, § 2a.12).
        if st.get("taken"):
            row["points"] = list(st["taken"])
        out.append(row)
    return out


def collect_record(base_dir, task, *, partial: bool = False) -> Dict:
    """Walk the transmission attempts and build the record dict.

    **A MODE'S FRAME SET IS READ THROUGH ITS ONE DOOR** (`frameset.read`,
    checked at the citation door, `engines/transport.md` § 2a.9): its record
    carries the mode's average at each voltage, at every frame's stated
    weight, the mode's definition beside it (`average.mode_average`).

    Reads each point's LATEST attempt, its state the run door's
    (`parse.dirs.run_status`): a point with its transmission is in
    ``points``; one not launched, queued or running is ``pending`` -- never
    a failure; one whose run failed, stopped, or ended without its
    transmission is ``failed``, saying which (the displacement sweep's
    split, `spectra.displacement_sweep`).  Raises :class:`RecordError` when
    no point has its transmission -- an empty record would say less than
    the refusal -- unless ``partial``: the Results tab's read, where a
    ladder in progress has a report (`engines/transport.md` § 2a.12).
    """
    from ..jobset.materialize import stage_home
    from ..parse.dirs import run_status
    from ..runfiles import JUNCTION_FILE, RunNames
    from ..runrecord import LaunchRecordError, launch_record
    from ..frameset import FrameSetError, read as read_frame_set
    from .average import AverageError, mode_average
    from .compose import PROVENANCE_FILE
    from .stages import frames_of, rung_points
    from .tbtnc import (ORBITAL_NOTE, ORBITAL_TYPES, TbtError, point_dos,
                        tbt_file)

    base = Path(base_dir)
    junction = composed_junction(base)
    try:
        frame_set = (read_frame_set(junction) if junction is not None
                     else None)
    except FrameSetError as exc:
        raise RecordError(str(exc)) from exc
    points_out: List[Dict] = []
    pending: List[Dict] = []
    failed: List[Dict] = []
    token = stage_home(base, task, "transmission").token
    names = RunNames.of(task.label, token, task.shape)
    regions = device_regions(base)
    stages = _stage_facts(base, task, task.label)
    # A POINT NOT OPENED stands where its rung does -- the ready door's
    # `ready` / `waiting` and what for, on the rung's row (prep opens every
    # point at once).
    rung = next(f for f in stages if f["stage"] == "transmission")
    opened = False                        # an attempt open: it is prepared
    for pt, where, run in result_folders(base, task, "transmission"):
        v = pt.bias_v
        # WHICH POINT, on every entry: its frame (``None`` with no frame
        # axis), its folder below the run, and in words.
        at = {"frame": pt.frame, "bias_v": v, "tokens": pt.rel,
              "point": pt.words()}
        opened = opened or where is not None
        if where is None:
            pending.append({**at, "state": rung["state"],
                            "why": rung.get("detail") or ""})
            continue
        rel = str(where.relative_to(base))
        # THE POINT'S STATE, the run door's -- asked as every reader asks
        # it, with its run's launch record; one that does not read is said.
        try:
            launch = launch_record(run, names)
        except LaunchRecordError as exc:
            failed.append({**at, "attempt": rel,
                           "state": "unreadable", "why": str(exc)})
            continue
        st = run_status(where, names.stem, launch=launch)
        if st.state != "finished":
            entry = {**at, "attempt": rel, "state": st.state,
                     "why": st.detail}
            (pending if st.state in ("pending", "queued", "running")
             else failed).append(entry)
            continue
        spin = deck_spin(where)
        # WHAT TBTRANS WROTE, through the family's one reader -- a point
        # that does not read is that point's failure, said in its words; the
        # other points still read.
        try:
            read = point_transmission(where, task.label, spin)
        except (OSError, ValueError, RecordError) as exc:
            failed.append({**at, "attempt": rel, "state": "finished",
                           "why": f"the run finished without its "
                                  f"transmission: {exc} ({st.detail})"})
            continue
        # THE DOS, ITS PARTS AND THE EIGENCHANNELS, from the point's own
        # `.TBT.nc` (`tbtnc.point_dos`; `web/results.md` § 2.5).
        nc = tbt_file(where, task.label)
        dos = None
        dos_why = None
        if nc is None:
            dos_why = f"{task.label}.TBT.nc is not in {rel}"
        else:
            try:
                dos = point_dos(nc, regions)
            except TbtError as exc:
                dos_why = str(exc)
        energies = read["energy_ev"]
        points_out.append({
            **at,
            # ITS FRAME'S ROWS, whole -- what the frame is, as its writer
            # stated it (`model/structure.md` § 2.2d).
            "customized": (junction.customized_rows(
                frame=pt.frame if pt.frame is not None else 0)
                if junction is not None else []),
            "attempt": rel,
            **read,
            "conductance_g0": conductance_g0(energies, read["transmission"]),
            # THE TOTAL, and the figure it came from (`CURRENT_MEANS`), with
            # the run's k-sampling.
            **point_currents(where, token, spin),
            # THE WINDOW AND ITS POINTS, as written (`engines/transport.md`
            # § 2a.12).
            "window_ev": [min(energies), max(energies)],
            "n_energies": len(energies),
            "spin": spin,
            **({"dos": dos} if dos is not None else {"dos_why": dos_why}),
        })
    if not points_out and not partial:
        # THE WAY ON, by what the stage's state says: prepared (an attempt is
        # open), it is launched or let finish -- a prepared stage refuses a
        # second prep (`job-system.md` § 5.0); else prepared, then launched.
        from ..jobset.commands import block, launch_lines, run_first
        raise RecordError(
            "no transmission point has its transmission yet -- "
            + ("launch the transmission stage, or let it finish:\n"
               + block(launch_lines("task", "transmission", base=base_dir))
               if opened else
               "run the transmission stage first:\n"
               + block(run_first("transmission", base=base_dir)))
            + "\n"
            + "".join(f"  ({what}: "
                      + "; ".join(f"{p['point'] or 'the one point'}, "
                                  f"{p['state']} ({p['why']})"
                                  for p in got) + ")\n"
                      for what, got in (("pending", pending),
                                        ("failed", failed)) if got))

    try:
        average = mode_average(
            junction.n_frames if junction is not None else None, frame_set,
            points_out, pending + failed)
    except AverageError as exc:
        raise RecordError(str(exc)) from exc
    provenance = None
    prov_file = base / PROVENANCE_FILE
    if prov_file.is_file():
        provenance = json.loads(prov_file.read_text())
    # THE DESCRIPTION'S, as stated (`Task.treatment`): a list of several
    # voltages states the switch; one is a single bias.  Never inferred
    # from how many points ran.
    treatment = task.treatment
    record: Dict = {
        "schema": TRANSPORT_RESULT_SCHEMA,
        "label": task.label,
        "energies_relative_to_ef": True,
        # THE LADDER IS THE RESULT'S STRUCTURE, so the record carries it.
        "stages": stages,
        # WHAT THE RESULT IS ENTITLED TO BE CALLED (§ 2a.10, ruled
        # 2026-09-16).  The mechanism is identical either way; what differs is
        # how many device SCFs were paid for, and therefore what may be
        # claimed.  One slice is T(E), and an I-V read off it is the
        # LINEAR-RESPONSE approximation -- integrating a zero-bias curve
        # cannot reproduce a resonance entering the window, nor that
        # resonance moving under the field.  A self-consistent sweep is T(E, V)
        # and its I-V carries no such caveat.
        #
        # Recorded here, not decided in the browser: § 2a.12 requires the
        # treatment NAMED BESIDE THE CURVE and not in metadata, because "the
        # two kinds of I-V are different claims and look identical on a plot".
        "treatment": treatment,
        "treatment_note": TREATMENT_NOTE[treatment],
        # THE TWO LEADS' FERMI LEVELS, compared here once -- and every
        # rung's E_F said in its frame (`fermi_frames`).
        "leads": leads_agreement(stages),
        "fermi_frames": fermi_frames(stages),
        # THE COMPOSED JUNCTION the device was built from, by its catalogue
        # name (`runfiles.JUNCTION_FILE`) -- what the report draws.
        "junction_file": JUNCTION_FILE,
        "points": points_out,
        # THE I-V: each point's own TBtrans current -- or, under the low-bias
        # approximation, computed by the record from the one 0 V slice for
        # every listed voltage (`linear_response_iv`, § 2a.10).
        "iv": _iv(base, task, treatment, points_out,
                  [pt.frame for pt in rung_points(
                      task, "transmission", frames=frames_of(base))]
                  or [None]),
        # THE MODE'S AVERAGE over a frame set, at each voltage (§ 2a.12).
        "average": average,
        # WHAT THE CURRENT IS, in the record's own words -- for each spin the
        # points were run with (one, for a junction decided once).
        "current_means": {s: CURRENT_MEANS[s]
                          for s in sorted({p["spin"] for p in points_out})},
        "provenance": {
            "slot": provenance,
            "atom_permutation": PERMUTATION_FILE,
            # WHAT EACH RUNG WAS GATHERED FROM -- its attempt's
            # `.gathered-from`, never the newest attempt by file time
            # (`engines/transport.md` § 2a.12).
            "chain": _chain(base, stages),
        },
        "caveat": CAVEAT,
        # THE ORBITAL MENU of the PDOS of a selection, and what it cannot
        # form (`tbtnc`; `web/results.md` § 2.5).
        "pdos_orbitals": {"types": ["all", *ORBITAL_TYPES],
                          "note": ORBITAL_NOTE},
    }
    # EVERY KEY, EVERY RECORD: an empty list where there is none.
    record["pending"] = pending
    record["failed"] = failed
    return record


def composed_junction(base: Path):
    """The calculation's composed junction, every frame
    (`compose.write_compose_record`, read through the codec), or ``None``
    before its first prep composed it."""
    from ..runfiles import JUNCTION_FILE
    from ..workingcopy_structure import StructureCodec
    path = base / JUNCTION_FILE
    return StructureCodec().load(path, frames=True) if path.is_file() else None


def selection_pdos(base_dir, task, bias_v: float, atoms, orbitals: str
                   ) -> Dict:
    """The PDOS of ``atoms`` at the bias point ``bias_v``, narrowed to one
    orbital type (`tbtnc.selection_pdos`; `web/results.md` § 2.5): read from
    that point's transmission run's ``.TBT.nc``, each orbital's type from the
    device run's ``.ORB_INDX`` at the same point -- the two runs share the
    composed junction's atom order.  :class:`RecordError` names what is
    missing."""
    from .tbtnc import TbtError, selection_pdos as _pdos, tbt_file
    base = Path(base_dir)
    _one_frame_or_refuse(base)

    def _at(stage: str) -> Optional[Path]:
        # A RUNG THAT DOES NOT SWEEP has one result, at 0 V, which serves
        # every voltage -- a single bias's, and the low-bias approximation's
        # (`stages.rung_points`).
        found = result_folders(base, task, stage)
        if len(found) == 1:
            return found[0][1]
        return next((w for pt, w, _r in found
                     if abs(pt.bias_v - float(bias_v)) < 1e-9), None)

    if not any(abs(float(v) - float(bias_v)) < 1e-9
               for v in (task.bias or (0.0,))):
        raise RecordError(f"no transmission point at {bias_v:g} V")
    where = _at("transmission")
    if where is None:
        raise RecordError(f"no transmission point at {bias_v:g} V")
    nc = tbt_file(where, task.label)
    if nc is None:
        raise RecordError(f"the transmission at {bias_v:g} V has written "
                          f"no {task.label}.TBT.nc")
    dev = _at("device")
    orb = (dev / f"{task.label}.ORB_INDX") if dev is not None else None
    try:
        return _pdos(nc, orb, atoms, orbitals)
    except (TbtError, OSError, ValueError) as exc:
        # A FILE THAT DOES NOT READ is a refusal in its words, as the record
        # says every other one.
        raise RecordError(f"{nc.name}: {exc}") from exc


def _iv(base: Path, task, treatment: str, points_out: List[Dict],
        frames: List[Optional[int]]) -> Dict:
    """The record's I-V block -- PER FRAME (`engines/transport.md` § 2a.12):
    columns ``frame`` (each row's frame, ``None`` on a calculation with no
    frame axis), ``voltages_v``, ``current_a``, ``current_a_printed``.
    ``computed`` says where the currents came from: ``tbtrans`` -- each
    point's own printed integral -- or ``linear-response`` -- the record's
    integral of each frame's own 0 V slice for every listed voltage, with
    the Fermi tails at the electronic temperature that point's deck states
    (§ 2a.10).  The frames share one transmission deck, so ``kt_ev``,
    ``window_ev`` and the ``notes`` keyed by voltage are the block's; a
    frame whose 0 V point has not run is noted under its token (``all``
    with no frame axis)."""
    from ..structure import frame_words
    from .stages import frame_token
    if treatment == "low-bias-approximation":
        out: Dict = {"frame": [], "voltages_v": [], "current_a": [],
                     "current_a_printed": [], "computed": "linear-response",
                     "kt_ev": None, "window_ev": None, "notes": {}}
        for f in frames:
            zero = next((p for p in points_out if p["frame"] == f
                         and abs(p["bias_v"]) < 1e-9), None)
            if zero is None:
                for v in task.bias:
                    out["frame"].append(f)
                    out["voltages_v"].append(float(v))
                    out["current_a"].append(None)
                    out["current_a_printed"].append(None)
                out["notes"][frame_token(f) if f is not None else "all"] = (
                    f"the 0 V transmission of {frame_words(f, len(frames))} "
                    f"has not run" if f is not None
                    else "the 0 V transmission has not run")
                continue
            kt = BOLTZMANN_EV_K * deck_temperature_k(base / zero["attempt"])
            iv = linear_response_iv(zero["energy_ev"], zero["transmission"],
                                    list(task.bias), kt)
            for v, i in zip(iv["voltages_v"], iv["current_a"]):
                out["frame"].append(f)
                out["voltages_v"].append(v)
                out["current_a"].append(i)
                out["current_a_printed"].append(
                    zero["current_a_printed"] if abs(v) < 1e-9 else None)
            out["notes"].update(iv["notes"])
            if out["kt_ev"] is None:
                out["kt_ev"], out["window_ev"] = iv["kt_ev"], iv["window_ev"]
        return out
    return {"frame": [p["frame"] for p in points_out],
            "voltages_v": [p["bias_v"] for p in points_out],
            "current_a": [p["current_a"] for p in points_out],
            "current_a_printed": [p["current_a_printed"] for p in points_out],
            "computed": "tbtrans"}


def write_record(base_dir, record: Dict) -> Path:
    from ..persist import write_json
    out = record_path(base_dir, record["label"])
    write_json(out, record)
    return out


#: HOW FAR THE TWO LEADS' FERMI LEVELS MAY DIFFER, in eV, before the record
#: says they disagree: both are bulk runs of the same lead, and T(E) is
#: measured relative to E_F (`engines/transport.md` § 2a.12).
LEAD_EF_TOLERANCE_EV = 0.05


def leads_agreement(stages: List[Dict]) -> Optional[Dict]:
    """``{fermi_ev: {lead: E_F}, differ_ev, agree, tolerance_ev}`` once both
    leads have said their Fermi level, else ``None``."""
    efs = {st["stage"]: st["fermi_ev"] for st in stages
           if st.get("stage") in ("electrode_L", "electrode_R")
           and st.get("fermi_ev") is not None}
    if len(efs) != 2:
        return None
    differ = abs(efs["electrode_L"] - efs["electrode_R"])
    return {"fermi_ev": efs, "differ_ev": differ,
            "agree": differ <= LEAD_EF_TOLERANCE_EV,
            "tolerance_ev": LEAD_EF_TOLERANCE_EV}


def fermi_frames(stages: List[Dict]) -> Optional[Dict]:
    """THE FERMI LEVELS, EACH IN ITS RUN'S OWN FRAME, said once for every
    reader (`engines/transport.md` § 2a.12; the 2026-10-08 road walk showed
    the device's 5.17 eV beside the leads' −1.92 eV with no word).  The
    seed's and the leads' E_F are in their periodic cells, where the
    cell-average potential is zero.  TranSIESTA fixes the Hartree potential
    on the boundary plane instead and reports the device's E_F in that
    frame: ``device_ef + vha`` is the device's E_F back in the seed's frame,
    and the NEGF loop holds E_F fixed while the charge floats, so it equals
    the seed's.  The leads' Hamiltonians are aligned to μ = E_F ± V/2
    inside TranSIESTA; T(E) is measured relative to the leads' E_F.  Nothing
    printed tests the physical alignment of the device's electrode region
    against the bulk.  ``None`` until the device has a NEGF E_F."""
    by = {st["stage"]: st for st in stages}
    dev = (by.get("device") or {}).get("negf") or {}
    if dev.get("ef") is None:
        return None
    out: Dict = {"device_ef_ev": dev["ef"], "frame": "transiesta"}
    seed_ef = (by.get("seed") or {}).get("fermi_ev")
    leads = {n: by[n].get("fermi_ev") for n in ("electrode_L", "electrode_R")
             if n in by and by[n].get("fermi_ev") is not None}
    if seed_ef is not None:
        out["seed_ef_ev"] = seed_ef
    if leads:
        out["lead_ef_ev"] = leads
    vha = dev.get("vha_ev")
    # WHICH POINT THESE DEVICE FIGURES ARE -- a rung that runs per point
    # has its first point's facts (`_stage_facts`), and the sentence says
    # which, in the point's words.
    at_v = (by.get("device") or {}).get("facts_at_v")
    at = (by.get("device") or {}).get("facts_at")
    where = f" at {at}" if at else ""
    note = ("Fermi levels are in each run's own frame: the seed's and the "
            "leads' in their periodic cells (cell-average potential zero); "
            "the device's after TranSIESTA fixes the Hartree potential on "
            "the boundary plane")
    if vha is not None:
        out["vha_ev"] = vha
        if at_v is not None:
            out["at_v"] = at_v
            out["at"] = at
        out["device_ef_in_seed_frame_ev"] = dev["ef"] + vha
        note += (f" (ts-Vha {vha:+.3f} eV{where}) -- in the seed's frame "
                 f"it is {dev['ef'] + vha:.3f} eV")
        if seed_ef is not None:
            out["device_minus_seed_ev"] = dev["ef"] + vha - seed_ef
            note += (f", against the seed's {seed_ef:.3f} eV"
                     + (" (equal: the NEGF loop holds E_F and lets the "
                        "charge float)"
                        if abs(out["device_minus_seed_ev"]) < 0.01 else
                        f" ({out['device_minus_seed_ev']:+.3f} eV apart)"))
    note += ("; the leads' Hamiltonians are aligned to E_F ± V/2 inside "
             "TranSIESTA and T(E) is measured relative to the leads' E_F.")
    out["note"] = note
    return out


#: WHAT EACH TREATMENT'S I-V IS ENTITLED TO BE CALLED, beside the curve
#: (`engines/transport.md` § 2a.10, § 2a.12) -- the one wording, which the
#: record carries and both the Results report and `summarize` print.
TREATMENT_NOTE = {
    "single-bias": ("One device SCF. An I-V derived from this curve is the "
                    "LINEAR-RESPONSE approximation: integrating a zero-bias "
                    "slice cannot reproduce a resonance entering the bias "
                    "window, nor that resonance moving under the field."),
    "low-bias-approximation": (
        "LOW-BIAS (linear-response) approximation: the device SCF converged "
        "once, at 0 V, and each voltage's current is the zero-bias "
        "transmission T(E, 0) integrated over that voltage's window -- a "
        "resonance entering the window is not re-converged, nor moved by "
        "the field."),
    "self-consistent": ("The device SCF was converged at every voltage, so "
                        "each slice is its own solution and the I-V carries "
                        "no approximation beyond the method."),
}


def iv_table_text(record: Dict) -> str:
    """The printed deliverable: one row per point -- its frame's token,
    G(E_F), the junction's total current, and the figure TBtrans printed --
    then what the current is, in the record's words (:data:`CURRENT_MEANS`),
    and the mode's average at each voltage, or why there is none."""
    lines = [f"transport record — {record['label']}: "
             f"{len(record['points'])} point(s)"
             + (f", {len(record['pending'])} pending"
                if record.get("pending") else "")
             + (f", {len(record['failed'])} failed"
                if record.get("failed") else "")]
    lines.append(f"  {'frame':>5}  {'V [V]':>8}  {'G(E_F) [G0]':>12}  "
                 f"{'I total [A]':>12}  {'I printed [A]':>13}")

    def _frame(p) -> str:
        return (p["tokens"].split("/")[0] if p.get("frame") is not None
                else "--")
    for p in record["points"]:
        g = p["conductance_g0"]
        i, i0 = p["current_a"], p.get("current_a_printed")
        lines.append(
            f"  {_frame(p):>5}  {p['bias_v']:>8.3f}  "
            + (f"{g:>12.4f}" if g is not None else f"{'--':>12}")
            + "  "
            + (f"{i:>12.4e}" if i is not None else f"{'--':>12}")
            + "  "
            + (f"{i0:>13.4e}" if i0 is not None else f"{'--':>13}"))
    for what in ("pending", "failed"):
        for p in record.get(what, ()):
            lines.append(f"  {_frame(p):>5}  {p['bias_v']:>8.3f}  "
                         f"{what:>12}  {p['state']:>12}  ({p['why']})")
    iv = record.get("iv") or {}
    if iv.get("computed") == "linear-response":
        # THE RECORD'S OWN I-V, from each frame's 0 V slice
        # (`linear_response_iv`).
        kt = iv.get("kt_ev")
        lines.append("  I(V) computed from T(E, 0)"
                     + (f", kT = {kt:.4f} eV, window {iv.get('window_ev')} eV"
                        if kt is not None else "")
                     + ":")
        from .stages import frame_token
        for f, v, i in zip(iv.get("frame", ()), iv.get("voltages_v", ()),
                           iv.get("current_a", ())):
            notes = iv.get("notes") or {}
            note = notes.get(f"{v:g}") or (
                notes.get(frame_token(f)) if f is not None
                else notes.get("all"))
            lines.append(f"  {frame_token(f) if f is not None else '--':>5}  "
                         f"{v:>8.3f}  {'':>12}  "
                         + (f"{i:>12.4e}" if i is not None else f"{'--':>12}")
                         + (f"  ({note})" if note else ""))
    for spin, means in sorted((record.get("current_means") or {}).items()):
        lines.append(f"  {spin}: {means}")
    lines.append(f"  {record['treatment']}: {record['treatment_note']}")
    lines.extend(_average_lines(record.get("average") or {}))
    frames = record.get("fermi_frames")
    if frames and frames.get("note"):
        lines.append(f"  E_F: {frames['note']}")
    return "\n".join(lines)


def _average_lines(avg: Dict) -> List[str]:
    """The mode's average, as `summarize` prints it: the mode and its
    weights, at each voltage the averaged conductance beside the base
    frame's and its change in per cent -- or why there is none."""
    if not avg.get("frames") or avg["frames"] < 2:
        return []
    if avg.get("why"):
        return [f"  the frames' average: {avg['why']}"]
    head = (f"  the mode's average over {avg['frames']} frames -- mode "
            f"{avg.get('mode_index_1based')} at "
            f"{avg.get('frequency_cm1'):g} cm-1, {avg.get('temperature_k'):g} K, "
            f"sigma {avg.get('sigma_amu12_ang'):.6g} amu^1/2 A")
    lines = [head + f", weights {avg['weights']} (sum "
             f"{avg['weight_sum']!r}, within {avg['tolerance']:g} of 1):"]
    for e in avg.get("at", ()):
        if e.get("why"):
            lines.append(f"  {e['bias_v']:>8.3f} V  {e['why']}")
            continue
        g, d = e.get("conductance_g0"), e.get("conductance_change_percent")
        g0 = e.get("base_conductance_g0")
        lines.append(
            f"  {e['bias_v']:>8.3f} V  <G> = "
            + (f"{g:.6f} G0" if g is not None else "--")
            + " against the base frame's "
            + (f"{g0:.6f} G0" if g0 is not None else "--")
            + ", change "
            + (f"{d:+.4f} %" if d is not None
               else f"-- ({e.get('conductance_why')})"))
    lines.append("  it assumes: " + " ".join(avg.get("assumptions") or ()))
    return lines
