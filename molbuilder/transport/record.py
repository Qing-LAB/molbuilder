"""The transport composite's RECORD — `engines/transport.md` § 2a.12,
build step P6.

TBtrans's own outputs are the truth this module reads: the k-averaged
transmission file (``<label>.TBT.AVTRANS_<L>-<R>``, two columns E vs T)
and the current line its ``.out`` prints (the binary's own Landauer
integral — parsed, never recomputed; a point that shares energies with a
finished calculation's has its current the record's, over its joined grid,
by the binary's own rule, :func:`joined_currents`).  Both formats were pinned against
a REAL 5.4.2 run (the carbon-chain live walk, 2026-08-29).

What lands on disk is ONE file at the calculation root,
``<label>.transport.json`` (``molbuilder/transport-result@18``): T(E)
per point -- a frame and a voltage -- the I–V table per frame, the frames'
average, their post-process's (`frame_average.average_block`), the junction's TOTAL
current, with the
figure TBtrans printed and the factor between them (:data:`CURRENT_MEANS`)
-- and the provenance that says which junction built it — the citation (from
``slot-provenance.json``).  The junction it names (``junction_file``)
carries its own order record in its sidecar (`model/overview.md` § 2.2a), so
every index in the engine's order goes back to the user order.

Reading is ASYNCHRONOUS by design, the same doctrine as the bench
summarizer: a point whose transmission has not run yet reads as
*pending*, never as a failure of the set, and one whose run ended without
its transmission reads as *failed*, in its run's own words — `summarize`
is a reader, and nothing is produced on a host that has produced nothing.

**COMPOSED ON READ** (`engines/transport.md` § 2a.12): every rung's state is
the one status door's (`runstatus.jobset_status`, what `jobset status` and
the Results tab's ladder say), and every point's its run's row there
(`runs.concluded` for a run it has no rows for);
the Results tab asks `/api/transport/record`, which composes the record
each time it is opened, so a rung that ran since `summarize` is never shown
as it was.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from ..calcdirs import RunPlace
from ..constants import BOLTZMANN_EV_K, CONDUCTANCE_QUANTUM_S
from ..persist import json_safe
from ..identity import COMPUTES_PRODUCT
from ..task import LOW_BIAS_APPROXIMATION, SELF_CONSISTENT, SINGLE_BIAS
from ..units import volts_words
from ..runfiles import (CHANNEL_DOWN, CHANNEL_UNPOLARIZED, CHANNEL_UP,
                        SPIN_NON_POLARIZED, SPIN_POLARIZED,
                        TBT_SPIN_CHANNELS)
from .conductance import at_fermi, per_spin_channel, spin_of
from .energies import window_short_of
from .stages import DEVICE_RUNG, SEED_RUNG
from .window import window_rung

if TYPE_CHECKING:
    from .stages import Point

#: What the record's current IS, said in the record (`engines/transport.md`
#: § 2a.4; user, 2026-10-03, Q5: "make sure the result presentation, data
#: record and the summary/comments clearly explain what is what").  TBtrans's
#: Landauer integral is one spin channel's -- I = (e/h)∫T, no factor 2 for
#: spin (`m_tbt_save.F90`) -- while the conductance beside it is in
#: G0 = 2e²/h, both channels.
CURRENT_MEANS = {
    SPIN_NON_POLARIZED: (
        "The total current is the junction's, both spin channels: "
        "TBtrans prints one spin channel's (its Landauer integral has no "
        "factor 2 for spin), and the two channels of a non-polarized "
        "calculation carry the same current, so the total is twice the "
        "figure TBtrans printed."),
    SPIN_POLARIZED: (
        "The total current is the junction's, the sum of its two spin "
        "channels' currents as TBtrans printed each (empty when it printed "
        "only one)."),
    # A POINT THAT SHARES ENERGIES with a finished calculation's point
    # (`engines/transport.md` § 2a.16): TBtrans printed its own run's
    # energies' current alone.
    "joined": (
        "A point sharing energies with a finished calculation's point: its "
        "current is summed over the joined grid -- each energy's transmission, times the grid's "
        "spacing, times the two leads' Fermi functions' difference, G0 times "
        "the sum (TBtrans's own rule) -- and no figure TBtrans printed "
        "is its current, TBtrans having printed its own run's energies' "
        "current alone."),
}

#: THE CAVEAT that goes with any DFT-NEGF conductance (`engines/transport.md`
#: § 1 and § 2a.12), carried on the result so every reader says it.
CAVEAT = (
    "A DFT-NEGF conductance with a plain GGA functional places the "
    "molecule's levels too close to the Fermi level, and so overestimates a "
    "molecular junction's conductance, often by one to two orders of "
    "magnitude.  Read T(E) for where the levels are and how they couple, "
    "not for an absolute conductance.")

def _scf_fields() -> tuple:
    """The SCF row fields the report keeps (`web/results.md` § 2.5): every
    key the engine's rows hold, by its one declaration (the engine table's
    ``scf_columns``, for the kind's engine), and a NEGF row's charges."""
    from ..engine_table import scf_columns
    from .stages import ENGINE
    return tuple(c[0] for c in scf_columns(ENGINE)) + ("charges",)


class RecordError(Exception):
    """The record cannot be built — the message names what to run or
    fix first, ready to surface verbatim."""


def record_path(base_dir, label: str) -> Path:
    """The ONE spelling of the record's location -- composed, not concatenated.

    `runfiles.WRITTEN` declares `.transport.json` and its module docstring says
    *"Nothing composes or splits a run-file name inline."*  Composing also
    inherits the label rule -- `compose` refuses `my.relax`, a name
    `runfiles.parse` cannot read back.
    """
    from ..runfiles import ROLE_TRANSPORT_RECORD, compose as _rf
    return Path(base_dir) / _rf(label, ROLE_TRANSPORT_RECORD)


def _ran(run_dir):
    """What the deck a run ran states (`parse.fdf.parse_fdf_params`) -- its
    deck found through the run door (`runs.run_of`, `Run.deck`), the input
    the run used.  A run with no deck of ours is a :class:`RecordError`,
    never a default: what it ran cannot be said."""
    from ..parse.fdf import parse_fdf_params
    from ..runs import run_of
    run = run_of(run_dir)
    deck = run.deck if run is not None else None
    if deck is None:
        raise RecordError(f"{run_dir} holds no deck of this calculation's "
                          f"run, so what it ran cannot be read")
    from ..parse.fdf import read_deck
    return parse_fdf_params(read_deck(deck), source=deck.name)


def point_transmission(where, label: str, spin: str, *,
                       av: Optional[Dict] = None) -> Dict:
    """What TBtrans wrote at one point, read through the SIESTA family's
    parser (`tbtrans_nc.point_averages`, `model/parse.md` § 5e): its
    k-averaged transmission on the data file's energies, at full precision
    (`engines/transport.md` § 2a.16) -- ``energy_ev`` and ``transmission``
    **per spin channel, so G = G0 · T(E_F)** with G0 = 2e²/h: the
    unpolarized file's T as it is; a polarized point's two channels
    averaged, (T↑ + T↓)/2, each kept whole in ``channels`` and their sum in
    ``transmission_total`` -- and ``transmission_file``.  :class:`RecordError`
    names what is missing: a polarized deck whose run wrote one channel is
    not done (`engines/transport.md` § 2a.12).  ``av`` is the point's
    averages where the caller read them already (:func:`point_averages_of`,
    one read for the transmission and the DOS)."""
    if av is None:
        try:
            av = point_averages_of(where, label)
        except (OSError, ValueError) as exc:
            raise RecordError(str(exc)) from exc
    from .energies import same_grid
    curves = {c: a["transmission"] for c, a in av.items()
              if "transmission" in a}
    if (spin == SPIN_POLARIZED and CHANNEL_UP in curves
            and CHANNEL_DOWN in curves
            and not same_grid(av[CHANNEL_UP]["energy_ev"],
                              av[CHANNEL_DOWN]["energy_ev"])):
        raise RecordError("the two spin channels' energy grids differ")
    # PER SPIN CHANNEL, the one rule (`conductance.per_spin_channel`), which a
    # transmission run's milestone reads by too (`run-reports.md` § 2.9).
    try:
        transmission = per_spin_channel(curves, spin)
    except ValueError as exc:
        raise RecordError(str(exc)) from None
    if spin == SPIN_POLARIZED:
        up, dn = av[CHANNEL_UP], av[CHANNEL_DOWN]
        t_up, t_dn = up["transmission"], dn["transmission"]
        return {"energy_ev": up["energy_ev"],
                "transmission": transmission,
                "channels": {CHANNEL_UP: t_up, CHANNEL_DOWN: t_dn},
                # THE TOTAL, T↑ + T↓, what the report draws beside the two
                # (`web/results.md` § 2.5).
                "transmission_total": [a + b for a, b in zip(t_up, t_dn)],
                "transmission_file": [up["files"]["transmission"],
                                      dn["files"]["transmission"]]}
    one = av[CHANNEL_UNPOLARIZED]
    return {"energy_ev": one["energy_ev"],
            "transmission": transmission,
            "transmission_file": one["files"]["transmission"]}


def point_averages_of(where, label: str) -> Dict:
    """What TBtrans averaged at one point, read once through the SIESTA
    family's parser (`tbtrans_nc.point_averages`, `model/parse.md` § 5e) --
    the pair from the leads' own block names (`TS.Elecs`), left to right:
    the file TBtrans writes for it, never the first of a glob.  Raises
    what the parser raises."""
    from ..parse.engines.tbtrans_nc import point_averages
    from .transiesta import LEAD_NAMES, LEAD_PAIR
    return point_averages(where, label, pair=LEAD_PAIR,
                          leads=tuple(LEAD_NAMES))


def point_currents(now: Optional[Dict], spin: str) -> Dict:
    """The current TBtrans printed at one point -- its own Landauer integral,
    parsed, never recomputed -- as the record says it
    (:data:`CURRENT_MEANS`): ``printed_a``, every figure it printed by its
    channel's name (`runfiles.TBT_SPIN_CHANNELS`) -- a non-polarized run's
    one, ``unpolarized``; a polarized run's ``up`` and ``down``, each also
    ``current_by_spin_a``; ``current_a``, the junction's total -- twice the
    one for a non-polarized run, the two channels' sum for a polarized one
    -- and the
    run's ``k_points`` / ``k_method`` and its
    ``loop_elapsed_s``, its passes' loops summed as TBtrans measured them
    (`model/parse.md` § 5c.1).  Read from ``now``, the point's output's
    reading through the family's one read -- the caller's
    (`collect_record`: one read per point) -- and its account of TBtrans's
    loop (`siesta_reader.read_output`, ``now()["loop"]``,
    `model/parse.md` § 5d.5): each pass's currents beside its channel;
    ``current_a`` ``None`` and no ``printed_a`` where the run printed none,
    or the point has no reading -- no output, or one that does not read."""
    out: Dict = {"current_a": None}
    if now is None:
        return out
    loop = now.get("loop") or {}
    for k in ("k_points", "k_method"):
        if k in loop:
            out[k] = loop[k]
    passes = loop.get("passes") or []
    took = [p["loop_elapsed_s"] for p in passes if "loop_elapsed_s" in p]
    if took:
        out["loop_elapsed_s"] = sum(took)
    rows = [(p.get("channel"), r) for p in passes
            for r in p.get("currents") or []]
    if not rows:
        return out
    # ONE ELECTRODE PAIR -- the first printed; a channel per pass.
    pair = (rows[0][1]["from"], rows[0][1]["to"])
    mine = [(ch, r) for ch, r in rows if (r["from"], r["to"]) == pair
            and r.get("current_a") is not None]
    if not mine:
        return out
    out["printed_a"] = (
        {ch: r["current_a"] for ch, r in mine} if spin == SPIN_POLARIZED
        else {TBT_SPIN_CHANNELS[spin][0]: mine[0][1]["current_a"]})
    if spin == SPIN_POLARIZED:
        by = out["printed_a"]
        if all(c in by for c in TBT_SPIN_CHANNELS[SPIN_POLARIZED]):
            out["current_a"] = by[CHANNEL_UP] + by[CHANNEL_DOWN]
            # EACH SPIN'S, as TBtrans printed it -- the I-V draws them
            # beside the total (`web/results.md` § 2.5).
            out["current_by_spin_a"] = {
                c: by[c] for c in TBT_SPIN_CHANNELS[SPIN_POLARIZED]}
    else:
        out["current_a"] = 2.0 * mine[0][1]["current_a"]
    return out


def temperature_k_of(params) -> float:
    """The electronic temperature a deck's parameters state
    (`parse.fdf.parse_fdf_params`), in kelvin (``300 K`` or an energy), or
    SIESTA's own default when the deck states none -- the catalogue's
    ``engine_default`` (`siesta.engine_input.engine_defaults`): the one
    rule the record's current and the scratch folder's estimate read it
    by."""
    said = params.electronic_temperature_k
    if said is not None:
        return float(said)
    from ..parse.fdf import fdf_key
    from ..siesta.engine_input import engine_defaults
    from ..siesta.layout import keyword as siesta_keyword
    from ..units import temperature_k
    keyword = siesta_keyword("electronic_temperature")
    toks = engine_defaults()[fdf_key(keyword)]
    return temperature_k(float(toks[0]), toks[1] if len(toks) > 1 else None,
                         what=keyword, source="SIESTA's default")


def deck_temperature_k(run_dir) -> float:
    """The electronic temperature the point's own deck states
    (:func:`temperature_k_of`)."""
    return temperature_k_of(_ran(run_dir))


def landauer_current(energies: List[float], trans: List[float],
                     volts: float, kt_ev: float, spacing_ev: float) -> float:
    """**I(V) = (2e/h) ∫ T(E) [f(E − μ_L) − f(E − μ_R)] dE**, μ_L,R = ±V/2,
    V signed -- THE RECORD'S ONE RULE for a current it computes, TBtrans's
    own (`m_tbt_save.F90`): each energy's transmission, times its weight
    on the mid-rule grid -- the window's spacing, ``spacing_ev``
    (`energies.spacing`) -- times the two leads' Fermi functions'
    difference, G0 times the sum in eV (`engines/transport.md` § 2a.10,
    § 2a.16).  ``trans`` per spin channel, as :func:`point_transmission`
    gives it."""
    import numpy as np
    e = np.asarray(energies, dtype=float)
    # SIGNED: a negative voltage carries a negative current, as TBtrans's
    # own does.
    half = float(volts) / 2.0
    with np.errstate(over="ignore"):
        f_l = 1.0 / (1.0 + np.exp((e - half) / kt_ev))
        f_r = 1.0 / (1.0 + np.exp((e + half) / kt_ev))
    return float(CONDUCTANCE_QUANTUM_S * float(spacing_ev)
                 * np.sum(np.asarray(trans, dtype=float) * (f_l - f_r)))


def channel_current(energies: List[float], trans: List[float],
                    volts: float, kt_ev: float, spacing_ev: float) -> float:
    """ONE SPIN CHANNEL'S CURRENT, (e/h) ∫ T(E) [f_L − f_R] dE -- what
    TBtrans prints for a channel (`m_tbt_save.F90`): half
    :func:`landauer_current`, whose G0 is both channels', the one rule a
    polarized point's channel is computed by."""
    return landauer_current(energies, trans, volts, kt_ev, spacing_ev) / 2.0


def low_bias_iv(energies: List[float], trans: List[float],
                voltages: List[float], kt_ev: float,
                spacing_ev: float, *, current=landauer_current) -> Dict:
    """The low-bias approximation's I-V: ``current`` -- the junction's,
    :func:`landauer_current`, or one spin channel's,
    :func:`channel_current` -- of the one zero-bias slice, T(E, 0), at each
    listed voltage (`engines/transport.md` § 2a.10), on the grid of spacing
    ``spacing_ev`` the slice's deck states.  A voltage whose window plus
    the Fermi tails (`energies.WINDOW_TAILS_KT`) reaches past the
    transmission's energies gets ``None`` and a note naming the reach --
    the record never integrates a slice it does not have."""
    lo, hi = float(min(energies)), float(max(energies))
    currents: List[Optional[float]] = []
    notes: Dict[str, str] = {}
    for v in voltages:
        short = window_short_of(lo, hi, v, kt_ev)
        if short:
            currents.append(None)
            # KEYED BY THE ROW (the voltage's index in `voltages_v`), never
            # by a spelling of the number a page would have to reproduce.
            notes[str(len(currents) - 1)] = short
            continue
        currents.append(current(energies, trans, v, kt_ev, spacing_ev))
    return {"voltages_v": [float(v) for v in voltages],
            "current_a": currents,
            "computed": LOW_BIAS_APPROXIMATION, "kt_ev": kt_ev,
            "window_ev": [lo, hi], "notes": notes}


def joined_currents(read: Dict, volts: float, kt_ev: float,
                    spacing_ev: float) -> Dict:
    """A POINT THAT SHARES ENERGIES -- its current the record's, over its
    joined grid (`engines/transport.md` § 2a.16), TBtrans's printed one
    covering its own run's energies alone: :func:`landauer_current` of
    ``read`` (:func:`point_transmission`'s, joined) on its deck's grid; a
    polarized point's each channel beside its total.  It printed no
    figure of its current (``printed_a`` empty)."""
    e = read["energy_ev"]
    out: Dict = {"current_a": landauer_current(e, read["transmission"],
                                               volts, kt_ev, spacing_ev),
                 "printed_a": {}, "current_from": "joined"}
    chans = read.get("channels") or {}
    if all(c in chans for c in TBT_SPIN_CHANNELS[SPIN_POLARIZED]):
        out["current_by_spin_a"] = {
            k: channel_current(e, chans[k], volts, kt_ev, spacing_ev)
            for k in TBT_SPIN_CHANNELS[SPIN_POLARIZED]}
    return out


def iv_rows(iv: Dict) -> List[Tuple[Optional[int], float]]:
    """THE I-V TABLE'S ROWS (:func:`_iv`'s), each ``(frame, voltage)`` --
    the frame ``None`` for one structure: the one reading of the record's
    ``iv`` by its rows."""
    volts = list(iv.get("voltages_v") or ())
    frames = list(iv.get("frame") or [None] * len(volts))
    return list(zip(frames, volts))


def _output_of(where: Path) -> Optional[Path]:
    """A rung's engine output in ``where`` -- the run door's
    (`runs.Run.stdout`); ``None`` where no run of ours is, or it printed
    none."""
    from ..runs import run_of
    run = run_of(where)
    return run.stdout if run is not None else None


def result_folders(calc, stage: str, run: Optional[Path] = None
                   ) -> List[Tuple["Point", Optional[Path], Optional[Path]]]:
    """``(point, folder, run)`` of each result ``stage`` of ``calc`` -- the
    calculation whole -- holds now: a rung that runs per point
    (`stages.rung_points`: a frame, a voltage, or both), its latest run's
    point folders; any other rung's latest run, as the one
    point ``Point()`` at 0 V -- ``folder`` ``None`` while no run is open
    (`engines/transport.md` § 2a.11).  ``run`` is where the launch record
    lies; given, that run is read instead of the latest (the device run
    the transmission gathered, § 2a.12)."""
    from ..jobset.materialize import latest_attempt
    from ..jobset.naming import stage_home
    from .rung import frames_had
    from .stages import Point, points_in, rung_points
    base, task = Path(calc.base), calc.task
    if run is None:
        run = latest_attempt(stage_home(base, task, stage).dir)
    pts = rung_points(task, stage, frames=frames_had(calc))
    if pts:
        if run is None:
            return [(pt, None, None) for pt in pts]
        return [(pt, p, run) for p, pt in points_in(
            calc.task, stage, run, frames=frames_had(calc))]
    return [(Point(), run, run)]


def _stage_facts(calc, *, status) -> List[Dict]:
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
    * *electrode_L* / *electrode_R* -- **the lead's Fermi level**, as its
      run computed it: a lead is a periodic BULK run, whose E_F places its
      bands in the device; T(E)'s energies are measured from the device's
      E_F, and `G = G0 * T(E_F)` is evaluated there (§ 2a.12).
    * *transmission* -- has its own `points` / `pending` / `failed` blocks.

    ``status`` is that door's answer, read once by the caller
    (:func:`collect_record`) for every part of the record.
    """
    from ..jobset.naming import ladder_homes
    from .rung import frames_had
    from .stages import STAGE_FACT, TRANSPORT_STAGES, rung_points

    base, task = Path(calc.base), calc.task
    by_name = {s.name: s for s in status.stages}
    tokens = {h.name: h.token for h in ladder_homes(base, task)}
    # THE DEVICE RUNS THE TRANSMISSION READ, when it has read any -- the
    # device facts shown beside T(E) are each point's own, never the newest
    # by itself (`engines/transport.md` § 2a.12); the rung's run its first
    # point's.
    device_runs, device_unread = gathered_device_runs(calc)
    device_run = gathered_device_run(device_runs)
    out: List[Dict] = []
    for name in TRANSPORT_STAGES:
        s = by_name.get(name)
        if s is None:                          # the description omits it
            out.append({"stage": name, "token": None,
                        "state": "not_described"})
            continue
        token = tokens.get(name)
        fact: Dict = {"stage": name, "token": token, "state": s.state,
                      "detail": s.detail,
                      # THE RUN IT IS A COPY OF, a run copied in from the
                      # finished calculation this one reuses
                      # (`engines/transport.md` § 2a.16).
                      **({"copied_from": s.copied_from}
                         if s.copied_from else {})}
        run = None
        if name == DEVICE_RUNG and device_run is not None:
            run = device_run
            fact["attempt"] = str(run.relative_to(base))
            fact["gathered_by"] = window_rung()
        elif s.attempt and s.dir:
            fact["attempt"] = f"{s.dir}/{s.attempt}"
            run = base / s.dir / s.attempt
        # WHICH QUESTION THIS RUNG ANSWERS -- a column, not a branch on the
        # name (`stages.STAGE_FACT`).  The transmission's answer is its
        # points; TBtrans converges nothing and reports no total energy.
        answers = STAGE_FACT[name]
        # THE RUN THE TRANSMISSION GATHERED answers for itself when it is not
        # the stage's newest (§ 2a.12: the device facts beside T(E) are that
        # run's): its state in its own words, and a detail naming both.
        newest = base / s.dir / s.attempt if s.dir and s.attempt else None
        another = run is not None and newest is not None and run != newest
        per_point = bool(rung_points(task, name, frames=frames_had(calc))
                         and answers != COMPUTES_PRODUCT)
        several = len(set(device_runs.values())) > 1

        def gathered_said(said: str) -> str:
            whose = ("the run the transmission's first point gathered"
                     if several else "the run the transmission gathered")
            return (f"{said}; {whose} ({fact['attempt']}); the stage's "
                    f"newest run is {s.dir}/{s.attempt}: {s.detail}")
        if another:
            # THE RUN IT IS A COPY OF, the gathered run's own -- never the
            # newest run's beside the other's figures (§ 2a.16).
            from ..runrecord import LaunchRecordError, copied_of
            fact.pop("copied_from", None)
            try:
                copied = copied_of(base, run)
            except LaunchRecordError as exc:
                copied = None
                _say_unreadable(fact, str(exc))
            if copied:
                fact["copied_from"] = copied
        # A POINT WHOSE ``.gathered-from`` DOES NOT READ, said on the rung.
        if name == DEVICE_RUNG and device_unread:
            _say_unreadable(fact, *device_unread)
        # EVERY SWEPT RUNG'S POINTS, as the status door reads them: the
        # folder each ran in, what it started from and what it alone took
        # (§ 2a.11, § 2a.12) -- the provenance chain's per-point lines, for
        # a product rung (the transmission) as for the device.  THE RUN THE
        # TRANSMISSION GATHERED answers for its own points when it is not
        # the stage's newest (§ 2a.12): the same per-point reader
        # (`continuation.point_rows`), asked for that run.
        points = s.points
        points_run = newest
        gathered = None
        if another and per_point:
            # ITS POINTS AND ITS STATE BY THE ONE READING of a run of points
            # (:func:`_points_read`) -- whatever the newest run's say.
            points_run = run
            got = _points_read(calc, name, run, fact)
            gathered = ((got.state, got.detail, list(got.rows))
                        if got is not None
                        else ("unreadable", "its records do not read", []))
            points = gathered[2]
        if points and not per_point:
            fact["taken"] = [
                _taken(p, str(RunPlace.of(points_run).point(p["rel"])
                              .relative_to(base))
                       if points_run is not None and p.get("rel") else None)
                for p in points]
        if per_point:
            # A RUNG THAT RUNS PER POINT, POINT BY POINT -- the report's
            # convergence card follows the selected point (`web/results.md`
            # § 2.5); the rung's own facts are its first point's.
            # EACH POINT, ITS STATE AND WHAT IT STARTED FROM AND TOOK, from
            # its own run's rows (`_rung_points`).
            fact["by_point"] = _rung_points(
                calc, name, run=run, rung=fact, rows=points,
                runs=device_runs if name == DEVICE_RUNG else None)
            _bridge_charge_from_0v(fact["by_point"])
            # THE PROVENANCE CHAIN'S PER-POINT LINES, each point's own run's.
            if points:
                fact["taken"] = [_taken(p, p.get("attempt"))
                                 for p in fact["by_point"]]
            if gathered is not None:
                # ITS STATE, as the one reading said it for this run
                # (`continuation.read_points`, above).
                fact["state"] = gathered[0]
                fact["detail"] = gathered_said(gathered[1])
            first = next((p for p in fact["by_point"]
                          if p.get("energy_ev") is not None), None)
            if first is not None:
                fact.update({k: first[k] for k in first
                             if k not in ("bias_v", "frame", "of", "point",
                                          "attempt", "state", "detail",
                                          *_PROVENANCE)})
                fact["facts_at_v"] = first["bias_v"]
                fact["facts_at_frame"] = first["frame"]
                fact["facts_at"] = first["point"]
                if name == DEVICE_RUNG:
                    # ITS OWN PERIODIC START, as read; else where its chain
                    # began (:func:`_periodic_shift`).
                    fact["periodic"] = (
                        {**first["periodic"], "from": first["attempt"]}
                        if first.get("periodic") else
                        _periodic_shift(base, base / first["started_from"])
                        if first.get("started_from") else None)
            # A TRANSMISSION RUN WHOSE POINTS GATHERED FROM SEVERAL DEVICE
            # RUNS -- points taken over beside points run again -- says so;
            # each point's device facts are its own run's (above).
            if name == DEVICE_RUNG and several:
                gathered_from = sorted({str(r.relative_to(base))
                                        for r in device_runs.values()})
                fact["detail"] = (
                    f"{fact['detail']}; the transmission's points gathered "
                    f"from {len(gathered_from)} device runs: "
                    f"{', '.join(gathered_from)}")
            out.append(fact)
            continue
        if another and answers != COMPUTES_PRODUCT:
            # A RUN-ONCE RUNG'S GATHERED RUN, by the one door for a run's
            # state (`runs.concluded`) -- its facts below are that run's.
            from ..jobset.naming import stage_home
            from ..runs import concluded
            st = concluded(run, stage_home(base, task, name).names,
                           base=base, launched_in=run)
            fact["state"], fact["detail"] = st.state, gathered_said(st.detail)
        output = (_output_of(run)
                  if answers != COMPUTES_PRODUCT and run is not None else None)
        if output is None:
            out.append(fact)
            continue
        science = _science(output)
        unread = science.pop("unreadable", None)
        fact.update(science)
        if unread:
            _say_unreadable(fact, unread)
        if name == DEVICE_RUNG:
            fact["periodic"] = (
                {**fact["periodic"], "from": fact.get("attempt")}
                if fact.get("periodic") else
                _periodic_shift(base, run))
            # A DEVICE THAT RUNS ONCE ran at 0 V -- a single bias's list is
            # [0.0], the low-bias approximation's slice 0 V (`Task.bias`) --
            # so it is its own 0 V point.
            _bridge_charge_from_0v([fact], ran_at=0.0)
        out.append(fact)
    return out


def _periodic_shift(base: Path, folder: Path) -> Optional[Dict]:
    """THE SHIFT BETWEEN THE DEVICE'S ZERO AND THE SEED'S, where it was
    printed -- ``{vha_ev, from}``: the periodic start's `ts-Vha` of
    ``folder``'s own output, else of the run it continued from, back along
    the chain of runs to where it began (TranSIESTA carries the E_F over in
    the density file a run continues from, so the shift printed there is
    this run's; `engines/transport.md` § 2a.12) -- or ``None`` when no run
    of the chain is in the folder to say it."""
    from ..runrecord import LaunchRecordError, read_continued_from
    seen: set = set()
    while folder is not None and folder.is_dir() and folder not in seen:
        seen.add(folder)
        output = _output_of(folder)
        shift = (_science(output).get("periodic")
                 if output is not None else None)
        if shift:
            return {**shift, "from": str(folder.relative_to(base))}
        try:
            src = read_continued_from(folder)
        except LaunchRecordError:
            return None
        folder = src.path(base) if src is not None else None
    return None


def gathered_device_runs(calc) -> Tuple[Dict[str, Path], List[str]]:
    """THE DEVICE RUN EACH TRANSMISSION POINT GATHERED, by the device
    point's folder below its run (``v0.2``, ``f001/v0.2``) -- read from each
    point's own ``.gathered-from`` (`runrecord.read_gathered_from`), the run
    above the device point named there (`engines/transport.md` § 2a.12: the
    provenance is what was gathered, never the newest run by file time).  A
    point a launch took over keeps the device run it was computed from,
    whatever the new run gathered (§ 2a.11, *Launched again*), so one
    transmission run's points may name two device runs.  Empty while the
    transmission has gathered nothing; beside it, a sentence for each point
    whose ``.gathered-from`` does not read -- said on the device rung, never
    the whole record refused."""
    from ..runrecord import LaunchRecordError, read_gathered_from
    from .stages import inputs_of
    base = Path(calc.base)
    # THE DEVICE'S FILE, as the DAG names it (`stage_inputs`), and THE RUN
    # its folder lies in -- a point's folder lies below it
    # (`RunRef.run_folder`).
    device = {fn for up, fn in inputs_of(calc.task, window_rung())
              if up == DEVICE_RUNG}
    out: Dict[str, Path] = {}
    unread: List[str] = []
    for pt, where, _run in result_folders(calc, window_rung()):
        if where is None:
            continue
        try:
            got = read_gathered_from(where)
        except LaunchRecordError as exc:
            unread.append(f"{pt.words()}: {exc}")
            continue
        for src, fn in got:
            if fn in device:
                run = src.run_folder().path(base)
                out[str(src.path(base).relative_to(run))] = run
                break
    return out, unread


def gathered_device_run(runs: Dict[str, Path]) -> Optional[Path]:
    """THE DEVICE RUNG'S OWN RUN on the record: the one the transmission's
    first point whose ``.gathered-from`` reads gathered, of ``runs``
    (:func:`gathered_device_runs`) -- ``None`` while the transmission has
    gathered nothing."""
    return next(iter(runs.values()), None)


def readings_kept(readings: Optional[Sequence[Dict]]) -> bool:
    """Whether every reading of a point's data files is kept -- its own and
    those in its ``shared/``, the point's ``digest`` (`model/parse.md`
    § 5e): the one answer to *its DOS per orbital is whole*, asked once a
    point as the record is composed -- its ``per_orbital_unread``, which
    the digests it names, the data file's ``dos_per_orbital_reading``, the
    export's missing-DOS sentence and the page read."""
    return bool(readings) and all(d.get("state") == "kept"
                                  for d in readings)


def _science(out: Path) -> Dict:
    """A rung run's own answer, read from its engine output: its SCF as it
    ran (every row of its last step, phase-tagged, with each phase's
    criteria and how it ended), whether it converged and at what energy,
    a NEGF phase's own figures, a lead's E_F -- or ``unreadable`` with the
    parser's first sentence."""
    from ..parse import detect
    from ..parse.engines.siesta_reader import engine_version
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
    # THE ENGINE THAT RAN IT (`engines/transport.md` § 2a.12).
    version = engine_version(info)
    if version:
        fact["engine_version"] = version
    # THE RUN'S SCF, AS IT RAN (`web/results.md` § 2.5): every row of the
    # last step, phase-tagged -- a device's periodic start and its NEGF
    # loop apart -- with what each phase had to reach and how it ended.
    # AN ENGINE'S UNREADABLE NUMBER IS NOT STATED (`persist.json_safe`): a
    # field SIESTA overflowed reads as NaN, and every figure below -- E_F,
    # the frames' note -- is reasoned on, so it
    # enters the record as None, never as a NaN that compares and prints.
    hist = json_safe(frames[-1].scf_history or []) if frames else []
    keep = _scf_fields()
    fact["scf"] = [{k: c[k] for k in keep if k in c} for c in hist]
    if info.get("scf_criteria"):
        fact["scf_criteria"] = info["scf_criteria"]
    if info.get("scf_residuals"):
        fact["scf_residuals"] = info["scf_residuals"]
    if info.get("scf_phases"):
        fact["scf_phases"] = info["scf_phases"]
    # THE PERIODIC START'S SHIFT (`engines/transport.md` § 2a.12): the
    # `ts-Vha` TranSIESTA printed in the device cell's SCF before its NEGF
    # cycles -- the shift between the device's zero and the seed's.  A run
    # continued from a density file has no periodic start
    # (:func:`_periodic_shift` follows its chain).
    from ..parse.engines.siesta_grammar import (PHASE_NEGF, PHASE_PERIODIC,
                                                rows_of_phase)
    periodic = [c for c in rows_of_phase(hist, PHASE_PERIODIC)
                if c.get("vha_ev") is not None]
    if periodic:
        fact["periodic"] = {"vha_ev": periodic[-1]["vha_ev"]}
    # THE NEGF PHASE'S OWN FIGURES -- never the periodic start's
    # (`engines/transport.md` § 2a.12): its last row's E_F and charge, and
    # how many cycles it ran.
    negf = rows_of_phase(hist, PHASE_NEGF)
    if negf:
        last = negf[-1]
        # `vha_ev` is the last NEGF cycle's `ts-Vha`, as TranSIESTA printed
        # it -- a value, not the shift between the two zeros (§ 2a.12).
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
        # -- the converged one: a lead's its bulk's, the seed's the periodic
        # junction's, the device's its NEGF loop's (the last row is NEGF) --
        # the one T(E)'s energies are measured from (§ 2a.12).
        # Asked by the COLUMN, so adding a third lead one day is a table
        # row and not a third name here.
        for cyc in reversed(hist):
            if cyc.get("ef") is not None:
                fact["fermi_ev"] = cyc["ef"]
                break
    return fact


def _bridge_charge_from_0v(points: List[Dict],
                           ran_at: Optional[float] = None) -> None:
    """Each device point's ``negf.d_change_from_0v_e``: ``D(V) - D(0 V)``,
    the bridge's net Mulliken population at its voltage less its frame's at
    0 V, both the last NEGF cycle's (`engines/transport.md` § 2a.12, *The
    device's charges*) -- set where both are stated and the 0 V point is
    done, its run finished and its SCF converged; a 0 V point that is not
    is no reference.  A frame's later voltages start from the previous
    voltage's density, so 0 V is the one reference every voltage of a
    frame shares.  ``ran_at``: the voltage of a device that runs once, whose
    one entry states none."""
    def volts(p) -> Optional[float]:
        v = p.get("bias_v") if ran_at is None else ran_at
        return None if v is None else float(v)
    at_0v = {p.get("frame"): ((p.get("negf") or {}).get("charges") or {})
             .get("D") for p in points
             if volts(p) == 0.0 and p.get("state") == "finished"
             and p.get("scf_converged") is True}
    for p in points:
        d = ((p.get("negf") or {}).get("charges") or {}).get("D")
        d0 = at_0v.get(p.get("frame"))
        if d is not None and d0 is not None:
            p["negf"]["d_change_from_0v_e"] = float(d) - float(d0)


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


def _points_read(calc, name: str, run: Path, rung: Dict):
    """THE ONE READING of a run of points (`continuation.read_points`): its
    points' rows and its state -- ``None`` where its records do not read or
    the reader fails, said on the rung (:func:`_say_unreadable`)."""
    from ..jobset.continuation import read_points
    where = run.relative_to(Path(calc.base))
    try:
        got = read_points(calc, name, run)
    except Exception as exc:                      # noqa: BLE001 -- said, never hidden
        _say_unreadable(rung, f"{where}: {exc}")
        return None
    if got.state == "unreadable":
        _say_unreadable(rung, f"{where}: {got.detail}")
        return None
    return got


def _say_unreadable(fact: Dict, *said: str) -> None:
    """RECORDS THAT DO NOT READ, said on the rung (``unreadable``) -- each
    sentence beside those said before, never one over another, and never
    the whole record refused (`engines/transport.md` § 2a.12)."""
    fact["unreadable"] = "; ".join(
        [*([fact["unreadable"]] if fact.get("unreadable") else []), *said])


#: WHAT A POINT STARTED FROM AND TOOK, as its run's row states it -- the
#: status door's reading of its ``.continued-from`` and ``.gathered-from``
#: (`StageStatus.points`; `engines/transport.md` § 2a.12): why its start is
#: not known where that does not read, and the point it is a copy of
#: (§ 2a.16).
_PROVENANCE = ("started_from", "start_unread", "copied_from", "took")


def _taken(p: Dict, attempt: Optional[str]) -> Dict:
    """A point's line of the provenance chain: which point, the folder it
    ran in (``attempt``), and :data:`_PROVENANCE` as its row states it."""
    return {"frame": p.get("frame"), "of": p.get("of"),
            "bias_v": p.get("bias_v"), "attempt": attempt,
            "started_from": p.get("started_from"),
            **{k: p[k] for k in ("start_unread", "copied_from") if p.get(k)},
            "took": p.get("took") or []}


def _rung_points(calc, name: str, run: Optional[Path] = None, *, rung: Dict,
                 rows=(), runs: Optional[Dict[str, Path]] = None
                 ) -> List[Dict]:
    """``[{frame, bias_v, point, attempt, state, detail, ...science}]`` --
    each point of ``calc``'s rung ``name`` that runs per point (``calc``
    the calculation whole), its state the status door's row (``rows``,
    read once per answer: `architecture.md` § 3.2) -- the
    run door's (`runs.concluded`) only for a run the status door has no
    rows for -- and its own answer (:func:`_science`); ``run`` names the
    run to read instead of the latest.  A point with no folder yet -- no run
    opened, the rung described or prepared and not launched -- is the
    rung's own row (``rung``: its state and detail, the status door's).
    A point the ended walk never reached is *not run*, as the rows say it
    (`continuation.never_reached`, `engines/transport.md` § 2a.11) --
    never one still to come, never failed.  ``runs``: a point's own run by
    its folder below it, where it is another -- the device run a
    transmission point gathered (:func:`gathered_device_runs`) -- read
    there, its state and :data:`_PROVENANCE` from that run's rows
    (`continuation.read_points`).  Each entry carries its row's
    :data:`_PROVENANCE`."""
    from ..jobset.naming import stage_home
    from ..runs import concluded
    base = Path(calc.base)
    names = stage_home(base, calc.task, name).names
    said = {r["folder"]: r for r in rows or ()}
    read_runs: set = set()
    out: List[Dict] = []
    for pt, where, run in result_folders(calc, name, run=run):
        own = (runs or {}).get(pt.rel)
        if own is not None and run is not None and own != run:
            where, run = RunPlace.of(own).point(pt.rel), own
            if own not in read_runs:
                # ITS RUN'S ROWS, read once (:func:`_points_read`) -- a run
                # whose records do not read leaves its point to the run
                # door's own read.
                read_runs.add(own)
                got = _points_read(calc, name, own, rung)
                if got is not None:
                    said.update({r["folder"]: r for r in got.rows})
        entry: Dict = {**pt.to_wire(), "point": pt.words()}
        if where is None or not where.is_dir():
            entry.update(state=rung["state"], detail=rung["detail"])
            out.append(entry)
            continue
        entry["attempt"] = str(where.relative_to(base))
        # THE POINT'S STATE: the status door's row, as the ladder reads it
        # (`runstatus.every_run`); the one door's own read (`runs.concluded`,
        # with its run's launch record) only where no row was handed in.
        row = (said.get(RunPlace.of(run).walked(pt.rel))
               if run is not None else None)
        if row is not None:
            entry.update(state=row["state"],
                         detail=row["detail"] if row["done"] else row["why"])
            entry.update({k: row[k] for k in _PROVENANCE if row.get(k)})
        else:
            st = concluded(where, names, base=base, launched_in=run)
            entry.update(state=st.state, detail=st.detail)
        output = _output_of(where)
        if output is not None:
            entry.update(_science(output))
        out.append(entry)
    return out


def device_regions(junction) -> Dict[str, List[int]]:
    """The device's regions molbuilder owns -- L-electrode, bridge,
    R-electrode (`sort.DEVICE_REGIONS`) -- as atom indices in ``junction``'s
    own order: the composed junction's, the decks' order, for the record's
    sums; the restored one's, the user order, for the data file's
    ``atom_region``.  ``{}`` before a junction is composed (``None``)."""
    from .sort import DEVICE_REGIONS
    if junction is None:
        return {}
    regions = junction.regions or {}
    return {label: [int(i) for i in regions.get(label, ())]
            for label in DEVICE_REGIONS if regions.get(label)}


def _chain(base: Path, stages: List[Dict]) -> List[Dict]:
    """``[{stage, attempt, gathered: [{file, from}]}]`` -- what each rung's
    attempt was gathered from, read from its `.gathered-from`
    (`runrecord.read_gathered_from`); a rung that gathered nothing is
    listed with ``[]``, one whose record does not read with why
    (``unreadable``) -- never the whole record refused."""
    from ..runrecord import LaunchRecordError, read_gathered_from
    out = []
    for st in stages:
        if not st.get("attempt"):
            continue
        try:
            got = read_gathered_from(base / st["attempt"])
            unread = None
        except LaunchRecordError as exc:
            got, unread = [], str(exc)
        row = {"stage": st["stage"], "attempt": st["attempt"],
               "gathered": [{"file": fn, "from": str(src)}
                            for src, fn in got],
               **({"unreadable": unread} if unread else {})}
        # A SWEPT RUN'S POINTS, each with what it started from and what it
        # alone took (the status door's reading, carried by the rung's
        # facts) -- a transmission point's device Hamiltonian, a device
        # point's start (§ 2a.11, § 2a.12).
        if st.get("taken"):
            row["points"] = list(st["taken"])
        out.append(row)
    return out


def not_transport(calc) -> Optional[str]:
    """Why the calculation ``calc`` (`calculation.open`) answers no transport
    door -- it is another kind -- or ``None``: the one sentence the
    transport routes refuse with (an export is picked by the kind table,
    `kind_table.export_of`)."""
    if calc.task.calculation == "transport":
        return None
    return (f"{Path(calc.base).name} is a {calc.task.calculation} "
            f"calculation, not a transport one")


def collect_record(calc, *, partial: bool = False) -> Dict:
    """Walk the transmission attempts and build the record dict.

    **THE FRAMES' ROWS ARE READ THROUGH THEIR ONE DOOR** (`frameset.read`):
    the record carries the frames' average at each voltage, at every frame's
    stated weight, where every frame states one, the structure's rows beside
    it as written -- the frames' post-process, `frame_average.average_block`.

    Reads each point's LATEST attempt, its state the status door's row
    (`runstatus.jobset_status`), the run door's (`runs.concluded`) for a
    run it has no rows for: a point with its transmission is in
    ``points``; one not launched, queued or running is ``pending`` -- never
    a failure; one whose run failed, stopped, or ended without its
    transmission is ``failed``, saying which (the displacement sweep's
    split, `spectra.displacement_sweep`).  Raises :class:`RecordError` when
    no point has its transmission -- an empty record would say less than
    the refusal -- unless ``partial``: the Results tab's read, where a
    ladder in progress has a report (`engines/transport.md` § 2a.12).
    """
    from ..parse.sidecars.transport import TRANSPORT_RESULT_SCHEMA
    from ..jobset.naming import stage_home
    from ..parse.dirs.job import FINAL_STATES, NOT_ENDED_STATES
    from ..parse.engines.tbtrans import loop_record
    from ..runpoints import NOT_RUN
    from ..parse.engines._run_ending import keep_reading, version_of
    from ..parse.engines.siesta_reader import engine_version, read_output
    from ..runfiles import JUNCTION_FILE
    from ..frame_average import (AverageError, average_block, average_said,
                                 citation_block)
    from ..frameset import read as read_frame_set
    from ..deck_record import DeckRecordUnreadable
    from .compose import read_provenance, recorded_junction
    from .rung import compose_refusals
    from .rung import frames_had
    from .stages import rung_points, same_volts
    from .energies import joined, sharing_of, spacing
    from .tbtnc import (DOS_CURVES, NO_CURVE, ORBITAL_NOTE,
                        ORBITAL_TYPES, TbtError, point_dos,
                        readings)

    base_dir = calc.base
    base, task = Path(base_dir), calc.task
    # THE COMPOSED JUNCTION, or why it does not read -- said, never the
    # whole report refused (`engines/transport.md` § 2a.12; F12's review,
    # E-D1: an HTML 500 on Results, a traceback from `summarize`).
    try:
        recorded, junction_unread = recorded_junction(base), None
    except compose_refusals() as exc:
        recorded, junction_unread = None, str(exc)
    junction = recorded.structure if recorded is not None else None
    frame_set = (read_frame_set(junction) if junction is not None
                 else None)
    points_out: List[Dict] = []
    pending: List[Dict] = []
    not_run: List[Dict] = []
    failed: List[Dict] = []
    home = stage_home(base, task, window_rung())
    names = home.names
    regions = device_regions(junction)
    # THE JUNCTION'S ATOM INDEX MAP, as the transmission's deck writes it
    # (`runs.stage_atom_map`) -- which each point's kept reading must
    # state (`model/overview.md` § 2.2b, F7).
    from ..runs import stage_atom_map
    junction_atoms = (stage_atom_map(junction.n_atoms, junction.atom_order,
                                     task.calculation, window_rung())
                      if junction is not None else None)
    # WHERE EACH RUNG STANDS, the status door's one answer, read once for
    # every part of the record (`runstatus.jobset_status`).
    from ..jobset.runstatus import jobset_status
    status = jobset_status(calc)
    stages = _stage_facts(calc, status=status)
    # A POINT NOT OPENED stands where its rung does -- the ready door's
    # `ready` / `waiting` and what for, on the rung's row (prep opens every
    # point at once).
    rung = next(f for f in stages if f["stage"] == window_rung())
    opened = False                        # an attempt open: it is prepared
    # THE TRANSMISSION'S POINTS AS THE STATUS DOOR READ THEM -- each one's
    # state, what it is a copy of -- one per point folder below its run
    # (`continuation.points_state`): a point the ended walk never reached
    # is *not run* there -- never one still to come, never failed
    # (`engines/transport.md` § 2a.11).
    rows = {r["folder"]: r for s in status.stages
            if s.name == window_rung() for r in (s.points or ())}
    # WHAT TBTRANS AVERAGED AT EACH FOLDER, read once for the transmission
    # and the DOS (`point_averages_of`).
    averaged: Dict[Path, Dict] = {}

    def _averages(folder) -> Dict:
        key = Path(folder)
        if key not in averaged:
            averaged[key] = point_averages_of(folder, task.label)
        return averaged[key]
    for pt, where, run in result_folders(calc, window_rung()):
        # WHICH POINT, on every entry: its frame (``None`` with no frame
        # axis), its folder below the run, and in words.
        at = {**pt.to_wire(), "tokens": pt.rel, "point": pt.words()}
        opened = opened or where is not None
        if where is None:
            pending.append({**at, "state": rung["state"],
                            "why": rung.get("detail") or ""})
            continue
        rel = str(where.relative_to(base))
        # THE POINT'S OUTPUT, READ ONCE (`model/parse.md` § 5d.5): its
        # reading serves the run door (`keep_reading`: an output that stated
        # its own end is not read again), the loop's words and the currents
        # -- three readers, one read.
        output = _output_of(where)
        now = None
        version = None
        if output is not None:
            try:
                reader = read_output(output)
                now = reader.now()
                # THE TBTRANS THAT RAN IT -- the family's one read
                # (`engines/transport.md` § 2a.12).
                version = engine_version(reader.runtime_info)
                keep_reading(output, version_of(output), reader.finish())
            except OSError:
                now = None
        # THE POINT'S STATE: the status door's row (``rows``, read once
        # for the record), as the ladder reads it (`runstatus.every_run`);
        # the one door's own read (`runs.concluded`, with its run's launch
        # record) only for a run the status door has no rows for.  One
        # that does not read is said.
        row = (rows.get(RunPlace.of(run).walked(pt.rel))
               if run is not None else None)
        if row is not None:
            state = row["state"]
            why = row["detail"] if row["done"] else row["why"]
        else:
            from ..runs import concluded
            st = concluded(where, names, base=base, launched_in=run)
            state, why = st.state, st.detail
        if state == "unreadable":
            failed.append({**at, "attempt": rel,
                           "state": "unreadable", "why": why})
            continue
        # WHAT ITS RUN COMPUTES, beside what it shares -- the energies of
        # its grid its own run computes, and those the finished point holds
        # (`engines/transport.md` § 2a.16): its deck's record, said beside
        # its loop, which counts its own run's energies.
        try:
            sharing = sharing_of(where)
        except DeckRecordUnreadable as exc:
            # ITS DECK'S RECORD OF WHAT IT SHARES DOES NOT READ: said on this
            # point alone, never the whole record refused (§ 2a.12) -- one
            # that finished is not done, in the words status gives its row
            # (`continuation.point_done`); one not ended is said once it
            # ends, its progress said now.
            if state == "finished":
                failed.append({**at, "attempt": rel, "state": "finished",
                               "why": str(exc)})
                continue
            sharing = None
        shares = ({"shared": {"from": sharing.source, "held": sharing.held,
                              "computed": sharing.computed}}
                  if sharing is not None else {})
        if state != "finished":
            entry = {**at, "attempt": rel, "state": state, "why": why,
                     **shares,
                     **({"engine_version": version} if version else {})}
            # HOW FAR ITS LOOP IS -- running, or where it stopped -- in the
            # one sentence the point's run record states it in
            # (`tbtrans.loop_record`, `model/parse.md` § 5d.1b, § 5d.5).
            progress = loop_record(output, over=state in FINAL_STATES,
                                    now=now)
            if progress:
                entry["progress"] = {
                    "said": progress["progress"],
                    **({"estimated_at": progress["estimated_at"]}
                       if "estimated_at" in progress else {})}
            (pending if state in NOT_ENDED_STATES
             else not_run if state == NOT_RUN
             else failed).append(entry)
            continue
        # WHAT TBTRANS WROTE, through the family's one reader -- a point
        # that does not read is that point's failure, said in its words; the
        # other points still read.
        try:
            # ITS DECK, read once: its spin and its temperature
            # (`spin_of`, `temperature_k_of`).
            params = _ran(where)
            spin = spin_of(params)
            # ITS GRID, JOINED -- its own run's energies and those it shares
            # with a finished calculation's point (`energies.joined`,
            # `engines/transport.md` § 2a.16).
            read = joined(where, lambda f: point_transmission(
                f, task.label, spin, av=_averages(f)))
        except (OSError, ValueError, RecordError) as exc:
            failed.append({**at, "attempt": rel, "state": "finished",
                           "why": f"the run finished without its "
                                  f"transmission: {exc} ({why})"})
            continue
        # THE DOS, ITS PARTS AND THE EIGENCHANNELS -- TBtrans's averages,
        # and each region's from the data file's kept reading -- a polarized
        # point's two channels summed with each spin's own beside
        # (`tbtnc.point_dos`; `web/results.md` § 2.5).
        def _dos(folder, spin=spin):
            return point_dos(folder, task.label, regions, spin,
                             junction_atoms, av=_averages(folder))
        dos = None
        dos_why = None
        try:
            dos = joined(where, _dos)
        except TbtError as exc:
            dos_why = str(exc)
        if dos is not None and not any(k in dos for k in DOS_CURVES):
            # A BLOCK OF NO CURVE is no DOS, said.
            dos, dos_why = None, NO_CURVE
        if dos is not None and "device_atoms" in dos:
            # THE ATOMS ITS DOS COVERS, in the user order, as the record
            # names every atom (`model/overview.md` § 2.2a, I1) -- never
            # by engine index under the record's name for them.
            engine = dos.pop("device_atoms")
            if junction is not None:
                dos["device_atoms"] = sorted(
                    junction.map_to_user_index(engine))
        # WHAT IS KEPT OF EACH DATA FILE'S READING -- its own and those in
        # its `shared/` (`model/parse.md` § 5e) -- and, ONE DECISION the data
        # file and the export both read, the digests the point names: only
        # where every reading is kept and the DOS was read from each, never a
        # reading in part -- one spin, one side of a shared grid.
        digest = [k.to_wire(base) for k in readings(where, task.label)]
        # NOT READ YET -- a data file whose reading is not whole; one with
        # none has nothing to read: the one answer the report, the data file
        # and the export ask.
        unread = bool(digest) and not readings_kept(digest)
        named = (dos or {}).get("per_orbital") or ()
        if dos is not None and "per_orbital" in dos and not (
                not unread and digest and len(named) == len(digest)):
            dos.pop("per_orbital")
        for src in (dos or {}).get("per_orbital") or ():
            # EACH DIGEST BY ITS PATH FROM THE CALCULATION'S FOLDER, where
            # the data file lies (`record.record_path`).
            src["digest"] = str(Path(src["digest"]).relative_to(base))
        # THE POINT IT IS A COPY OF -- taken over from, or copied in with
        # its run from, the finished calculation this one reuses (§ 2a.16):
        # the run that computed it (`runrecord.copied_of`).
        from ..runrecord import LaunchRecordError, copied_of
        if row is not None:
            copied = row.get("copied_from")
        else:
            try:
                copied = copied_of(base, where)
            except LaunchRecordError as exc:
                # SAID ON THE RUNG, never the whole record refused (§ 2a.12).
                copied = None
                _say_unreadable(rung, f"{pt.words()}: {exc}")
        # ITS CURRENT, and which it is (`engines/transport.md` § 2a.16): the
        # printed figures' total, or -- for a point that shares energies --
        # the record's over its joined grid.
        currents = point_currents(now, spin)
        currents.update(
            joined_currents(read, pt.volts or 0.0,
                            BOLTZMANN_EV_K * temperature_k_of(params),
                            spacing(sharing.window))
            if sharing is not None else {"current_from": "tbtrans"})
        energies = read["energy_ev"]
        points_out.append({
            **at,
            # ITS FRAME'S ROWS, whole -- what the frame is, as its writer
            # stated it (`model/structure.md` § 2.2d).
            "customized": ([dict(r) for r in frame_set.frames[
                pt.frame if pt.frame is not None else 0]]
                if frame_set is not None else []),
            "attempt": rel,
            # ITS STATE, as a pending or failed point states its own
            # (`engines/transport.md` § 2a.12).
            "state": state,
            **read,
            # G/G0 = T(E_F), the one rule (`conductance.at_fermi`), which a
            # transmission run's milestone says too (`run-reports.md` § 2.9).
            "conductance_g0": at_fermi(energies, read["transmission"]),
            # THE TOTAL, and the figure it came from (`CURRENT_MEANS`), with
            # the run's k-sampling.
            **currents,
            # THE WINDOW AND ITS POINTS, as written (`engines/transport.md`
            # § 2a.12).
            "window_ev": [min(energies), max(energies)],
            "n_energies": len(energies),
            **shares,
            **({"copied_from": copied} if copied else {}),
            **({"engine_version": version} if version else {}),
            "spin": spin,
            **({"dos": dos} if dos is not None else {"dos_why": dos_why}),
            # kept, missing, out of date or being read, with its size and
            # its claim (above).
            "digest": digest,
            "per_orbital_unread": unread,
        })
    if not points_out and not partial:
        # THE WAY ON, by what the stage's state says: prepared (an attempt is
        # open), it is launched or let finish -- a prepared stage refuses a
        # second prep (`job-system.md` § 5.0); else prepared, then launched.
        from ..jobset.commands import block, launch_lines, run_first
        raise RecordError(
            "no transmission point has its transmission yet -- "
            + ("launch the transmission stage, or let it finish:\n"
               + block(launch_lines("task", window_rung(), base=base_dir))
               if opened else
               "run the transmission stage first:\n"
               + block(run_first(window_rung(), calc=calc)))
            + "\n"
            + "".join(f"  ({what}: "
                      + "; ".join(f"{p['point']}, "
                                  f"{p['state']} ({p['why']})"
                                  for p in got) + ")\n"
                      for what, got in (("pending", pending),
                                        ("not run", not_run),
                                        ("failed", failed)) if got))

    try:
        # A PAGE STILL OPENS (`engines/transport.md` § 3.8.1): the Results
        # tab's read draws a refused frame set where the average would be;
        # the record `summarize` writes is refused by name.
        average = (average_said if partial else average_block)(
            frames_had(calc) if junction is not None or junction_unread
            else None, frame_set,
            points_out, pending + not_run + failed, same_volts)
    except AverageError as exc:
        raise RecordError(str(exc)) from exc
    provenance = read_provenance(base)
    # THE CITED RUN'S VERDICT AS DATA (`engines/transport.md`, *What is said
    # about how it ended*): its kept words read back by their writer
    # (`_run_ending.verdict_read`), so the report decides on the verdict and
    # never on words -- a record composed before the field was served
    # included.
    relaxed = (provenance or {}).get("relaxation")
    if isinstance(relaxed, dict):
        from ..parse.engines._run_ending import verdict_read
        verdict = verdict_read(relaxed.get("converged"))
        relaxed["converged_ok"] = verdict[1] if verdict else None
    # THE DESCRIPTION'S, as stated (`Task.treatment`): a list of several
    # voltages states the switch; one is a single bias.  Never inferred
    # from how many points ran.
    treatment = task.treatment
    # A SPIN-POLARIZED CALCULATION, the one answer the I-V, the data file
    # and the report read: a point whose deck is polarized.
    polarized = any(p.get("spin") == SPIN_POLARIZED for p in points_out)
    record: Dict = {
        "schema": TRANSPORT_RESULT_SCHEMA,
        "label": task.label,
        "energies_relative_to_ef": True,
        # THE LADDER IS THE RESULT'S STRUCTURE, so the record carries it.
        "stages": stages,
        # WHAT THE RESULT IS ENTITLED TO BE CALLED (§ 2a.10, ruled
        # 2026-09-16).  The mechanism is identical either way; what differs is
        # how many device SCFs were paid for, and therefore what may be
        # claimed.  One slice serving every voltage is the LOW-BIAS
        # approximation -- T(E, V) taken as T(E, 0), which cannot reproduce
        # a resonance entering the window, nor that resonance moving under
        # the field.  A self-consistent sweep is T(E, V), and a single bias
        # its own SCF at its voltage: neither carries that caveat.
        #
        # Recorded here, not decided in the browser: § 2a.12 requires the
        # treatment NAMED BESIDE THE CURVE and not in metadata, because "the
        # two kinds of I-V are different claims and look identical on a plot".
        "treatment": treatment,
        # WHETHER THE LOW-BIAS APPROXIMATION WAS MADE, the flag itself and
        # its words (`engines/transport.md` § 2a.10, *The labelling rule*).
        "low_bias_approximation": treatment == LOW_BIAS_APPROXIMATION,
        "low_bias_said": LOW_BIAS_SAID[treatment == LOW_BIAS_APPROXIMATION],
        "treatment_note": TREATMENT_NOTE[treatment],
        # EVERY RUNG'S E_F, said in its frame (`fermi_frames`).
        "fermi_frames": fermi_frames(stages),
        # THE COMPOSED JUNCTION the device was built from, by its catalogue
        # name (`runfiles.JUNCTION_FILE`) -- what the report draws; ``None``
        # until a prep has composed it.
        "junction_file": JUNCTION_FILE if recorded is not None else None,
        # ...OR WHY IT DOES NOT READ, in its reader's words, where the
        # report would draw it; ``None`` when it reads or none is composed.
        "junction_unread": junction_unread,
        # WHEN ITS ORDER IS NOT RECORDED -- a junction whose sidecar states
        # none (I9) -- the sentence the report says beside the drawing, which
        # then lists the atoms as TranSIESTA does; ``None`` otherwise.
        "order_unstated": order_unstated_said(base, recorded),
        # ITS WAY ON ALONE, for the report, whose drawing states the fact
        # itself (MolView's notice); ``None`` where the order is recorded.
        "order_way": order_way_said(base, recorded),
        "polarized": polarized,
        "points": points_out,
        # THE I-V: each point's own TBtrans current -- or, under the low-bias
        # approximation, computed by the record from the one 0 V slice for
        # every listed voltage (`low_bias_iv`, § 2a.10).
        "iv": _iv(base, task, treatment, points_out,
                  [pt.frame for pt in rung_points(
                      task, window_rung(), frames=frames_had(calc))]
                  or [None], polarized),
        # THE FRAMES' AVERAGE at each voltage -- their post-process's
        # (`frame_average`, § 2a.12) -- and the frame set it is taken over,
        # cited whole: its rows with their meanings, the vibration, the
        # masses in the user order (`model/overview.md` § 2.2a), the pair
        # and its sha256; null for one structure.
        "average": average,
        "frame_set": citation_block(
            junction.restored() if junction is not None else None,
            provenance),
        # WHAT THE CURRENT IS, in the record's own words -- for each spin the
        # points were run with (one, for a junction decided once).
        "current_means": {s: CURRENT_MEANS[s]
                          for s in sorted({p["spin"] for p in points_out}
                                          | {p["current_from"]
                                             for p in points_out
                                             if p.get("current_from")
                                             == "joined"})},
        "provenance": {
            "slot": provenance,
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
    # THE RUNG SAYS ITS RUNNING POINT'S LOOP, naming the point, how many
    # points are still to come after it, and the last finished point's
    # loop -- what a point still to come is judged by; nothing projects it
    # (`engines/transport.md` § 2a.12).
    at_i = next((i for i, p in enumerate(pending)
                 if p.get("state") == "running" and p.get("progress")), None)
    if at_i is not None:
        running = pending[at_i]
        rung["progress"] = {
            **running["progress"],
            **{k: running.get(k) for k in ("frame", "of", "bias_v")},
            # EVERY OTHER POINT NOT DONE is still to come, whatever order
            # the walk runs them in (round by round, `submit._points_walk`).
            "after": len(pending) - 1}
        measured = next((p for p in reversed(points_out)
                         if p.get("loop_elapsed_s")), None)
        if measured is not None:
            rung["progress"]["measured"] = {
                k: measured.get(k)
                for k in ("frame", "of", "bias_v", "loop_elapsed_s")}
    record["not_run"] = not_run
    record["failed"] = failed
    return record


def order_unstated_said(base: Path, recorded) -> Optional[str]:
    """What the record, its data file and a selection's PDOS say of a
    junction whose sidecar states no order record (`model/overview.md`
    § 2.2a, I9) -- the fact (`compose.RecordedJunction.order_unstated`) and
    its way on, `prep task` (`engines/transport.md` § 1.1) -- or ``None``.
    ``recorded`` is `compose.recorded_junction`'s answer, ``None`` before
    anything was composed."""
    way = order_way_said(base, recorded)
    return None if way is None else f"{recorded.order_unstated}.  {way}"


def order_way_said(base: Path, recorded) -> Optional[str]:
    """The way on alone for a junction whose sidecar states no order
    record -- `prep task` (`jobset.commands.order_recorded`) -- or
    ``None`` where the order is recorded."""
    if recorded is None or not recorded.order_unstated:
        return None
    from ..jobset.commands import order_recorded
    return order_recorded(base=base)


def device_orbital_index(calc) -> Optional[Path]:
    """The device run's ``<label>.ORB_INDX`` -- each atom's basis, in its
    order -- of the device run the transmission gathered, at the first point
    holding one: one junction, one basis at every point
    (`engines/transport.md` § 2a.12, *A PDOS of the atoms you choose, from
    the files*); ``None`` where no device run holds one."""
    from ..runfiles import SIESTA_ORB_INDX
    from ..runs import run_of
    for _pt, folder, _run in result_folders(
            calc, DEVICE_RUNG, run=gathered_device_run(
                gathered_device_runs(calc)[0])):
        if folder is None:
            continue
        run = run_of(folder)
        orb = run.on_label(SIESTA_ORB_INDX) if run is not None else None
        if orb is not None and Path(orb).is_file():
            return Path(orb)
    return None


def selection_pdos(calc, bias_v: float, atoms, orbitals: str,
                   frame: Optional[int] = None) -> Dict:
    """The PDOS of ``atoms`` at the point (``frame``, ``bias_v``) of the
    transport calculation ``calc`` (`calculation.open`), narrowed to one
    orbital type (`tbtnc.selection_pdos`; `web/results.md` § 2.5): read from
    that point's kept reading of its transmission run's data file, each
    orbital's type from the device run's ``.ORB_INDX``
    (:func:`device_orbital_index`: one junction, one basis at every point,
    the reading's orbital counts held against it) -- the two runs share the
    composed junction's atom order.  ``atoms`` are
    named in the user order, the order the report draws the junction in,
    and so are the answer's ``atoms`` and ``outside_device``
    (`model/overview.md` § 2.2a, I1, I3); an index that is not one of the
    junction's atoms is refused by name.  ``frame`` is the frame's index,
    named for a frame set and for nothing else.  :class:`RecordError` names
    what is missing."""
    from ..atom_permutation import AtomIndexError
    from .rung import frames_had
    from .stages import Point, same_volts
    from ..parse.engines.tbtrans_nc import data_files
    from .tbtnc import TbtError, selection_pdos as _pdos
    base, task = Path(calc.base), calc.task
    n = frames_had(calc)
    # THE POINT IS A FRAME AND A VOLTAGE (`engines/transport.md` § 2a.9):
    # a frame set's names its frame, one structure's names none.
    if n > 1 and frame is None:
        raise RecordError(f"this calculation cites a frame set of {n} "
                          f"frames: name the frame (its index, 0 … {n - 1})")
    if n == 1 and frame is not None:
        raise RecordError(f"this calculation cites one structure, so a "
                          f"point names no frame: leave out "
                          f"frame index {int(frame)}")
    if frame is not None and (isinstance(frame, bool)
                              or int(frame) != frame):
        raise RecordError(f"a frame is named by its index, a whole number: "
                          f"{frame!r}")
    if frame is not None and not 0 <= int(frame) < n:
        raise RecordError(f"frame index {int(frame)} is not one of this "
                          f"calculation's {n} frame(s): 0 … {n - 1}")
    want = Point(frame=None if frame is None else int(frame), volts=bias_v,
                 of=n if frame is not None else None)
    said = want.words()

    def _transmission_at():
        # THE POINT BY EACH AXIS THE RUNG VARIES (`stages.rung_points`): an
        # axis a rung does not vary is ``None`` on its points, and its one
        # result serves every value of it -- a rung that runs once serves
        # every point, one that runs per frame every voltage (a single
        # bias, the low-bias approximation).  ``(point, folder)``.
        return next(((pt, w) for pt, w, _r in result_folders(
                         calc, window_rung())
                     if (pt.frame is None or pt.frame == want.frame)
                     and (pt.volts is None
                          or same_volts(pt.bias_v, want.bias_v))),
                    (None, None))

    if not any(same_volts(v, bias_v)
               for v in task.bias):
        raise RecordError(f"no transmission point at {volts_words(bias_v)}")
    # THE SELECTION CROSSES INTO THE JUNCTION'S ORDER through its own order
    # record, and the answer's atoms back (`model/overview.md` § 2.2a, I3):
    # the runs list the atoms as the composed junction does.  A record that
    # states no order names no atom in the user order (I9).
    from .compose import recorded_junction
    from .rung import compose_refusals
    try:
        recorded = recorded_junction(base)
    except compose_refusals() as exc:
        # A JUNCTION THAT DOES NOT READ names no atom: the selection's
        # refusal, in its reader's words (F12's review, E-D1).
        raise RecordError(str(exc)) from None
    if recorded is None:
        from ..runfiles import JUNCTION_FILE
        raise RecordError(f"the composed junction, {JUNCTION_FILE}, is not in "
                          f"the calculation's folder, so no atom of it can "
                          f"be named")
    unstated = order_unstated_said(base, recorded)
    if unstated:
        raise RecordError(unstated)
    read_at, where = _transmission_at()
    if where is None:
        raise RecordError(f"no transmission point at {said}")
    ncs = data_files(where, task.label)
    if not ncs:
        raise RecordError(f"the transmission at {said} has written no data "
                          f"file")
    junction = recorded.structure if recorded is not None else None
    # THE JUNCTION'S BASIS, the one orbital index the data file's table is
    # read from too (:func:`device_orbital_index`): the device run the
    # transmission gathered -- one junction, one basis at every point; the
    # reading's own orbital counts are held against it (`tbtnc._orbitals_of`).
    orb = device_orbital_index(calc)
    # THE SELECTION, each atom one of the junction's -- refused by name
    # otherwise, never wrapped (`model/overview.md` § 2.2a).
    try:
        mine = junction.map_from_user_index(atoms)
    except AtomIndexError as exc:
        raise RecordError(str(exc)) from exc
    # THE JUNCTION'S ATOM INDEX MAP, which each reading's must be (F7).
    from ..runs import stage_atom_map
    junction_atoms = stage_atom_map(junction.n_atoms, junction.atom_order,
                                    calc.task.calculation, window_rung())
    try:
        # THE POINT IT WAS READ AT, stated beside the answer: under the
        # low-bias approximation a voltage's PDOS is its frame's 0 V slice.
        from .energies import joined
        got = joined(where, lambda f: _pdos(f, task.label, orb, mine,
                                            orbitals,
                                            expected=junction_atoms))
        return {**got,
                "atoms": sorted(junction.map_to_user_index(got["atoms"])),
                "outside_device": sorted(
                    junction.map_to_user_index(got["outside_device"])),
                "point": Point(frame=read_at.frame, volts=read_at.volts,
                               of=want.of).to_wire()}
    except TbtError as exc:
        # THE READER'S REFUSAL, in its words -- naming its file where a
        # file is what it is about.
        raise RecordError(str(exc)) from exc
    except (OSError, ValueError) as exc:
        # A FILE THAT DOES NOT READ is a refusal in its words, as the record
        # says every other one.
        raise RecordError(f"{', '.join(p.name for p in ncs.values())}: "
                          f"{exc}") from exc


def _iv(base: Path, task, treatment: str, points_out: List[Dict],
        frames: List[Optional[int]], polarized: bool) -> Dict:
    """The record's I-V block -- PER FRAME (`engines/transport.md` § 2a.12):
    columns ``frame`` (each row's frame, ``None`` on a calculation with no
    frame axis), ``voltages_v``, ``current_a``.
    ``computed`` says where the currents came from: ``tbtrans`` -- each
    point's own printed integral -- or ``low-bias-approximation`` -- the record's
    integral of each frame's own 0 V slice for every listed voltage, with
    the Fermi tails at the electronic temperature that point's deck states
    (§ 2a.10).  The frames share one transmission deck, so ``kt_ev``,
    ``window_ev`` and the ``notes`` are the block's: each ``{frame, bias_v,
    note}`` -- a voltage whose window reaches past the slice's, or a frame
    whose 0 V point has not run, its ``bias_v`` ``None``: one form
    for the page, the table `summarize` prints and the data file, which has
    no row axis.  A POLARIZED calculation adds
    ``current_up_a`` and ``current_down_a``, each spin's current beside the
    total
    (`web/results.md` § 2.5): TBtrans's printed figures, or under the
    approximation each spin's own integral, :func:`low_bias_iv` of that
    spin's T(E, 0) by :func:`channel_current`."""
    from ..structure import frame_words
    from .energies import spacing, window_of
    from .stages import same_volts
    if treatment == LOW_BIAS_APPROXIMATION:
        out: Dict = {"frame": [], "voltages_v": [], "current_a": [],
                     "computed": LOW_BIAS_APPROXIMATION,
                     "kt_ev": None, "window_ev": None, "notes": []}
        if polarized:
            for spin in TBT_SPIN_CHANNELS[SPIN_POLARIZED]:
                out[f"current_{spin}_a"] = []
        for f in frames:
            zero = next((p for p in points_out if p["frame"] == f
                         and same_volts(p["bias_v"], 0.0)), None)
            if zero is None:
                for v in task.bias:
                    out["frame"].append(f)
                    out["voltages_v"].append(float(v))
                    out["current_a"].append(None)
                    if polarized:
                        for spin in TBT_SPIN_CHANNELS[SPIN_POLARIZED]:
                            out[f"current_{spin}_a"].append(None)
                out["notes"].append({
                    "frame": f, "bias_v": None,
                    "note": (f"the 0 V transmission of "
                             f"{frame_words(f, len(frames))} has not run"
                             if f is not None
                             else "the 0 V transmission has not run")})
                continue
            kt = BOLTZMANN_EV_K * deck_temperature_k(base / zero["attempt"])
            # THE SLICE'S GRID, as its deck states it -- its spacing each
            # energy's weight (`landauer_current`).
            window = window_of(base / zero["attempt"])
            if window is None:
                raise RecordError(
                    f"{zero['attempt']}: its deck states no transmission "
                    f"window (TBT.Contour.window)")
            iv = low_bias_iv(zero["energy_ev"], zero["transmission"],
                             list(task.bias), kt, spacing(window))
            if polarized:
                chans = zero.get("channels") or {}
                for spin in TBT_SPIN_CHANNELS[SPIN_POLARIZED]:
                    out[f"current_{spin}_a"].extend(
                        low_bias_iv(zero["energy_ev"], chans[spin],
                                    list(task.bias), kt, spacing(window),
                                    current=channel_current)["current_a"]
                        if spin in chans else [None] * len(iv["current_a"]))
            for v, i in zip(iv["voltages_v"], iv["current_a"]):
                out["frame"].append(f)
                out["voltages_v"].append(v)
                out["current_a"].append(i)
            # EACH NOTE OF THE FRAME'S OWN LIST, keyed by its row there
            # (`low_bias_iv`), placed by the frame and the voltage.
            out["notes"].extend(
                {"frame": f, "bias_v": iv["voltages_v"][int(k)], "note": note}
                for k, note in (iv.get("notes") or {}).items())
            if out["kt_ev"] is None:
                out["kt_ev"], out["window_ev"] = iv["kt_ev"], iv["window_ev"]
        return out
    out = {"frame": [p["frame"] for p in points_out],
           "voltages_v": [p["bias_v"] for p in points_out],
           "current_a": [p["current_a"] for p in points_out],
           "computed": "tbtrans", "notes": []}
    if polarized:
        for spin in TBT_SPIN_CHANNELS[SPIN_POLARIZED]:
            out[f"current_{spin}_a"] = [
                (p.get("current_by_spin_a") or {}).get(spin)
                for p in points_out]
    return out


def write_record(base_dir, record: Dict) -> Path:
    from ..persist import write_json
    out = record_path(base_dir, record["label"])
    write_json(out, record)
    return out


def fermi_frames(stages: List[Dict]) -> Optional[Dict]:
    """THE FERMI LEVELS, EACH IN ITS RUN'S OWN FRAME, said once for every
    reader (`engines/transport.md` § 2a.12).  The seed's and the leads' E_F
    are in their periodic cells, where the cell-average potential is zero.
    TranSIESTA fixes the Hartree potential on the boundary plane instead and
    reports the device's E_F against that: the `ts-Vha` it printed in its
    periodic start -- the device cell's SCF before the NEGF cycles -- is the
    shift between the two zeros, so ``device_ef + vha`` is the device's E_F
    back in the seed's frame, beside the seed's own, their difference a
    number and no verdict.  A run continued from a density file has no
    periodic start; the shift is the one printed where its chain began, that
    run named (`_periodic_shift`), and with none the device's E_F is given
    alone, saying why.  Nothing here enters T(E), G or the I-V.  ``None``
    until the device has a NEGF E_F."""
    by = {st["stage"]: st for st in stages}
    device = by.get(DEVICE_RUNG) or {}
    dev = device.get("negf") or {}
    if dev.get("ef") is None:
        return None
    out: Dict = {"device_ef_ev": dev["ef"], "frame": "transiesta"}
    seed_ef = (by.get(SEED_RUNG) or {}).get("fermi_ev")
    # THE LEADS, the kind table's (`CalculationKind.leads`).
    from ..kind_table import kind
    leads = {n: by[n].get("fermi_ev") for n in kind("transport").leads
             if n in by and by[n].get("fermi_ev") is not None}
    if seed_ef is not None:
        out["seed_ef_ev"] = seed_ef
    if leads:
        out["lead_ef_ev"] = leads
    shift = device.get("periodic") or {}
    # WHICH POINT THESE DEVICE FIGURES ARE -- a rung that runs per point
    # has its first point's facts (`_stage_facts`), and the sentence says
    # which, in the point's words.
    at_v = device.get("facts_at_v")
    at = device.get("facts_at")
    where = f" at {at}" if at else ""
    note = ("Fermi levels are in each run's own frame: the seed's and the "
            "leads' in their periodic cells (cell-average potential zero); "
            "the device's after TranSIESTA fixes the Hartree potential on "
            "the boundary plane")
    if at_v is not None:
        out["at_v"] = at_v
        out["at"] = at
    vha = shift.get("vha_ev")
    if vha is not None:
        out["vha_ev"] = vha
        out["vha_from"] = shift.get("from")
        out["device_ef_in_seed_frame_ev"] = dev["ef"] + vha
        note += (f" (the periodic start's ts-Vha {vha:+.6f} eV, printed in "
                 f"{shift.get('from')}) -- the device's E_F{where} in the "
                 f"seed's frame is {dev['ef'] + vha:.6f} eV")
        if seed_ef is not None:
            out["device_minus_seed_ev"] = dev["ef"] + vha - seed_ef
            note += (f", the seed's {seed_ef:.6f} eV, a difference of "
                     f"{out['device_minus_seed_ev']:+.6f} eV")
    else:
        note += (f"; no run of the device's chain{where} that printed its "
                 f"periodic start's ts-Vha is in the folder, so its E_F, "
                 f"{dev['ef']:.6f} eV, stays in TranSIESTA's frame")
    note += ("; the leads' Hamiltonians are aligned to E_F ± V/2 inside "
             "TranSIESTA, and T(E) is measured from the device's E_F, which "
             "TBtrans subtracts as it reads the device's Hamiltonian -- under "
             "a bias the leads sit at ± V/2 from it.")
    out["note"] = note
    return out


#: WHETHER THE LOW-BIAS APPROXIMATION WAS MADE, in the words every surface
#: shows beside the curve (`engines/transport.md` § 2a.10, *The labelling
#: rule*; the user, 2026-10-25): yes only where one 0 V slice served every
#: voltage -- a single voltage ran its device and transmission at it.
LOW_BIAS_SAID = {
    True: "low-bias approximation: yes, the 0 V transmission integrated "
          "over each voltage's window",
    False: "low-bias approximation: no, each voltage its own device SCF",
}


#: WHAT EACH TREATMENT'S I-V IS ENTITLED TO BE CALLED, beside the curve
#: (`engines/transport.md` § 2a.10, § 2a.12) -- the one wording, which the
#: record carries and both the Results report and `summarize` print.
TREATMENT_NOTE = {
    SINGLE_BIAS: ("One device SCF and one transmission, at the one "
                    "voltage: T(E) there, with no approximation beyond the "
                    "method."),
    LOW_BIAS_APPROXIMATION: (
        "The low-bias approximation: the device SCF is converged once, at "
        "0 V, and each voltage's current is the zero-bias transmission "
        "T(E, 0) integrated over that voltage's window \u2014 the "
        "transmission's dependence on bias neglected; a resonance entering "
        "the window is not re-converged, nor moved by the field."),
    SELF_CONSISTENT: ("The device SCF is converged at every voltage, so "
                        "each slice is its own solution and the I-V carries "
                        "no approximation beyond the method."),
}


def printed_said(p: Dict) -> str:
    """What TBtrans printed at a point (``printed_a``), as the I-V table
    shows it: a non-polarized point's one figure, a polarized point's each
    channel's by name, ``--`` where it printed none -- a point that shares
    energies (:data:`CURRENT_MEANS`)."""
    got = p.get("printed_a") or {}
    if list(got) == list(TBT_SPIN_CHANNELS[SPIN_NON_POLARIZED]):
        return f"{next(iter(got.values())):>13.4e}"
    if got:
        return "  ".join(f"{k} {v:.4e}" for k, v in got.items())
    return f"{'--':>13}"


def iv_table_text(record: Dict) -> str:
    """The printed deliverable: one row per point -- its frame's token,
    G(E_F), the junction's total current, and the figure TBtrans printed --
    then what the current is, in the record's words (:data:`CURRENT_MEANS`),
    and the frames' average at each voltage, or why there is none (their
    post-process's, `frame_average.summary_lines`)."""
    lines = [f"transport record — {record['label']}: "
             f"{len(record['points'])} point(s)"
             + (f", {len(record['pending'])} not ended"
                if record.get("pending") else "")
             + (f", {len(record['not_run'])} not run"
                if record.get("not_run") else "")
             + (f", {len(record['failed'])} failed"
                if record.get("failed") else "")]
    from ..structure import frame_words
    lines.append(f"  {'':>8}  {'V [V]':>8}  {'G(E_F) [G0]':>12}  "
                 f"{'I total [A]':>12}  {'I printed [A]':>13}")

    def _frame(f) -> str:
        return frame_words(f) if f is not None else "--"
    for p in record["points"]:
        g = p["conductance_g0"]
        i = p["current_a"]
        lines.append(
            f"  {_frame(p.get('frame')):>8}  {p['bias_v']:>8.3f}  "
            + (f"{g:>12.4f}" if g is not None else f"{'--':>12}")
            + "  "
            + (f"{i:>12.4e}" if i is not None else f"{'--':>12}")
            + "  " + printed_said(p))
    for what, said in (("pending", "not ended"), ("not_run", "not run"),
                       ("failed", "failed")):
        for p in record.get(what, ()):
            lines.append(f"  {_frame(p.get('frame')):>8}  {p['bias_v']:>8.3f}  "
                         f"{said:>12}  {p['state']:>12}  ({p['why']})")
    iv = record.get("iv") or {}
    if iv.get("computed") == LOW_BIAS_APPROXIMATION:
        # THE RECORD'S OWN I-V, from each frame's 0 V slice
        # (`low_bias_iv`).
        kt = iv.get("kt_ev")
        lines.append("  I(V) computed from T(E, 0)"
                     + (f", kT = {kt:.4f} eV, window {iv.get('window_ev')} eV"
                        if kt is not None else "")
                     + ":")
        notes = iv.get("notes") or ()
        for (f, v), i in zip(iv_rows(iv), iv.get("current_a", ())):
            # A VOLTAGE'S NOTE, ELSE ITS FRAME'S, as the record places them.
            note = next((n["note"] for n in notes if n["frame"] == f
                         and n["bias_v"] == v), None) or next(
                (n["note"] for n in notes if n["frame"] == f
                 and n["bias_v"] is None), None)
            lines.append(f"  {_frame(f):>8}  "
                         f"{v:>8.3f}  {'':>12}  "
                         + (f"{i:>12.4e}" if i is not None else f"{'--':>12}")
                         + (f"  ({note})" if note else ""))
    for spin, means in sorted((record.get("current_means") or {}).items()):
        lines.append(f"  {spin}: {means}")
    lines.append(f"  {record['low_bias_said']}")
    lines.append(f"  {record['treatment']}: {record['treatment_note']}")
    from ..frame_average import summary_lines
    lines.extend(summary_lines(record.get("average") or {}))
    frames = record.get("fermi_frames")
    if frames and frames.get("note"):
        lines.append(f"  E_F: {frames['note']}")
    return "\n".join(lines)
