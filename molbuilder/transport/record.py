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
``<label>.transport.json`` (``molbuilder/transport-result@14``): T(E)
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
the Results tab's ladder say), and every point's its run's (`run_status`);
the Results tab asks `/api/transport/record`, which composes the record
each time it is opened, so a rung that ran since `summarize` is never shown
as it was.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ..constants import BOLTZMANN_EV_K, CONDUCTANCE_QUANTUM_S
from ..persist import json_safe

if TYPE_CHECKING:
    from .stages import Point

#: ``@14``: ``frame_set`` is ``null`` where the citation is no frame set --
#: one structure -- never a block naming a run as one.
#: ``@13``: ``frame_set`` -- the frame set the average is taken over, cited
#: whole: the citation and its files' sha256, the structure's rows as
#: written with their meanings, ``info.vibration`` and every atom's
#: ``mass_amu`` in the user order (`frame_average.citation_block`).
#: ``@10``: each finished point's ``digest`` -- what is kept of each data
#: file's reading, its own and its ``shared/``'s: its state, size and claim
#: (`model/parse.md` § 5e); the DOS by region from the kept readings alone.
#: ``@9``: ``current_from`` on every point -- ``tbtrans`` (the junction's
#: total from the printed figures) or ``joined`` (a point that shares energies
#: with a finished calculation's, its current the record's over the joined
#: grid, its printed figure none; `engines/transport.md` § 2a.16).
#: ``@7``: ``order_unstated`` -- the sentence a junction whose sidecar states
#: no order record is said with, and its way on (I9), or ``None``.
#: ``@6``: the provenance names no ``atom_permutation`` file -- the junction
#: (``junction_file``) carries its order record in its sidecar.
#: ``@5``: the ``average`` carries no slope or curvature, and states the
#: structure's rows of a mode's frame set by their own names
#: (`model/structure.md` § 2.2f).
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
TRANSPORT_RESULT_SCHEMA = "molbuilder/transport-result@14"

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
        "figure TBtrans printed."),
    "polarized": (
        "current_a is the junction's total current, the sum of its two spin "
        "channels' currents as TBtrans printed each (empty when it printed "
        "only one)."),
    # A POINT THAT SHARES ENERGIES with a finished calculation's point
    # (`engines/transport.md` § 2a.16): TBtrans printed its own run's
    # energies' current alone.
    "joined": (
        "a point whose current_from is joined shares energies with a "
        "finished calculation's point: its current_a is the record's, over "
        "its joined grid -- each energy's transmission, times the grid's "
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


def record_path(base_dir, label: str) -> Path:
    """The ONE spelling of the record's location -- composed, not concatenated.

    `runfiles.WRITTEN` declares `.transport.json` and its module docstring says
    *"Nothing composes or splits a run-file name inline."*  Composing also
    inherits the label rule -- `compose` refuses `my.relax`, a name
    `runfiles.parse` cannot read back.
    """
    from ..runfiles import compose as _rf
    return Path(base_dir) / _rf(label, ".transport.json")


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


def deck_spin(run_dir) -> str:
    """The spin the run's own deck states -- ``non-polarized`` or
    ``polarized`` -- in any word SIESTA accepts for it (`parse.fdf`), SIESTA's
    own default, non-polarized, where the deck states none."""
    return ("polarized" if _ran(run_dir).spin_treatment == "unrestricted"
            else "non-polarized")


def point_transmission(where, label: str, spin: str) -> Dict:
    """What TBtrans wrote at one point, read through the SIESTA family's
    parser (`tbtrans_nc.point_averages`, `model/parse.md` § 5e): its
    k-averaged transmission on the data file's energies, at full precision
    (`engines/transport.md` § 2a.16) -- ``energy_ev`` and ``transmission``
    **per spin channel, so G = G0 · T(E_F)** with G0 = 2e²/h: the
    unpolarized file's T as it is; a polarized point's two channels
    averaged, (T↑ + T↓)/2, each kept whole in ``channels`` and their sum in
    ``transmission_total`` -- and ``transmission_file``.  :class:`RecordError`
    names what is missing: a polarized deck whose run wrote one channel is
    not done (`engines/transport.md` § 2a.12)."""
    from ..parse.engines.tbtrans_nc import point_averages
    from .transiesta import LEAD_NAMES, LEAD_PAIR
    # THE PAIR, from the leads' own block names (`TS.Elecs`), left to
    # right: the file TBtrans writes for it, never the first of a glob.
    try:
        av = point_averages(where, label, pair=LEAD_PAIR,
                            leads=tuple(LEAD_NAMES))
    except (OSError, ValueError) as exc:
        raise RecordError(str(exc)) from exc
    have = {c for c, a in av.items() if "transmission" in a}
    if spin == "polarized":
        missing = [c for c in ("up", "down") if c not in have]
        if missing:
            raise RecordError(
                f"the deck is spin-polarized and the run wrote no "
                f"{' or '.join(missing)} channel transmission")
        up, dn = av["up"], av["down"]
        if up["energy_ev"] != dn["energy_ev"]:
            raise RecordError("the two spin channels' energy grids differ")
        t_up, t_dn = up["transmission"], dn["transmission"]
        return {"energy_ev": up["energy_ev"],
                "transmission": [(a + b) / 2.0 for a, b in zip(t_up, t_dn)],
                "channels": {"up": t_up, "down": t_dn},
                # THE TOTAL, T↑ + T↓, what the report draws beside the two
                # (`web/results.md` § 2.5).
                "transmission_total": [a + b for a, b in zip(t_up, t_dn)],
                "transmission_file": [up["files"]["transmission"],
                                      dn["files"]["transmission"]]}
    if "unpolarized" not in have:
        raise RecordError("the run wrote no transmission file")
    one = av["unpolarized"]
    return {"energy_ev": one["energy_ev"],
            "transmission": one["transmission"],
            "transmission_file": one["files"]["transmission"]}


def point_currents(where, token: str, spin: str, *,
                   now: Optional[Dict] = None) -> Dict:
    """The current TBtrans printed at one point -- its own Landauer integral,
    parsed, never recomputed -- as the record says it
    (:data:`CURRENT_MEANS`): ``current_a_printed``, the first figure it
    printed (one spin channel's); ``current_a``, the junction's total --
    twice it for a non-polarized run, the two channels' sum for a polarized
    one -- and the run's ``k_points`` / ``k_method`` and its
    ``loop_elapsed_s``, its passes' loops summed as TBtrans measured them
    (`model/parse.md` § 5c.1).  Read from the point's newest
    output through the family's one read and its account of TBtrans's loop
    (`siesta_reader.read_output`, ``now()["tbtrans"]``, `model/parse.md`
    § 5d.5): each pass's currents beside its channel; ``None`` where the
    run printed none.  ``now`` is the newest output's reading where the
    caller holds it already (`collect_record`: one read per point); an
    older output is read only when the newest printed no current."""
    from ..parse.engines.siesta_reader import read_output
    out: Dict = {"current_a": None, "current_a_printed": None}
    for i, path in enumerate(_outs_newest_first(where, token)):
        reading = now if (i == 0 and now is not None) else read_output(path).now()
        loop = reading.get("tbtrans") or {}
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
            continue
        # ONE ELECTRODE PAIR -- the first printed; a channel per pass.
        pair = (rows[0][1]["from"], rows[0][1]["to"])
        mine = [(ch, r) for ch, r in rows if (r["from"], r["to"]) == pair
                and r.get("current_a") is not None]
        if not mine:
            continue
        out["current_a_printed"] = mine[0][1]["current_a"]
        if spin == "polarized":
            by = {ch: r["current_a"] for ch, r in mine}
            if "up" in by and "down" in by:
                out["current_a"] = by["up"] + by["down"]
                # EACH SPIN'S, as TBtrans printed it -- the I-V draws them
                # beside the total (`web/results.md` § 2.5).
                out["current_by_spin_a"] = {"up": by["up"],
                                            "down": by["down"]}
        else:
            out["current_a"] = 2.0 * mine[0][1]["current_a"]
        break
    return out


def deck_temperature_k(run_dir) -> float:
    """The electronic temperature the point's own deck states, in kelvin
    (`parse.fdf`: ``300 K`` or an energy), SIESTA's default 300 K when the
    deck states none."""
    said = _ran(run_dir).electronic_temperature_k
    return 300.0 if said is None else float(said)


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
        return (f"the transmission's sampled window [{lo:g}, {hi:g}] eV "
                f"does not reach "
                f"±{reach:.3f} eV for {float(volts):g} V (V/2 + "
                f"{WINDOW_TAILS_KT:g} kT at kT = {float(kt_ev):.4f} eV); "
                f"widen it -- transmission_emin_ev / transmission_emax_ev")
    return None


def sampled_window(emin_ev: float, emax_ev: float,
                   n_points: int) -> Tuple[float, float]:
    """The first and last energy TBtrans samples on a window of ``n_points``
    -- the mid-rule's points, half a spacing inside each edge
    (`m_integrate.f90`; the deck's ``method mid-rule``): what
    :func:`window_short_of` is asked with by the transmission deck's gate
    and the description's preflight, as the record asks it with the energies
    the run wrote -- one rule, on the energies the current is integrated
    over."""
    from .energies import spacing
    lo, hi = float(emin_ev), float(emax_ev)
    h = spacing((lo, hi, n_points))
    return lo + h / 2.0, hi - h / 2.0


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


def low_bias_iv(energies: List[float], trans: List[float],
                voltages: List[float], kt_ev: float,
                spacing_ev: float) -> Dict:
    """The low-bias approximation's I-V: :func:`landauer_current` of the
    one zero-bias slice, T(E, 0), at each listed voltage
    (`engines/transport.md` § 2a.10), on the grid of spacing
    ``spacing_ev`` the slice's deck states.  A voltage whose window plus
    the Fermi tails (:data:`WINDOW_TAILS_KT`) reaches past the
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
        currents.append(landauer_current(energies, trans, v, kt_ev,
                                         spacing_ev))
    return {"voltages_v": [float(v) for v in voltages],
            "current_a": currents,
            "computed": "low-bias-approximation", "kt_ev": kt_ev,
            "window_ev": [lo, hi], "notes": notes}


def joined_currents(read: Dict, volts: float, kt_ev: float,
                    spacing_ev: float) -> Dict:
    """A POINT THAT SHARES ENERGIES -- its current the record's, over its
    joined grid (`engines/transport.md` § 2a.16), TBtrans's printed one
    covering its own run's energies alone: :func:`landauer_current` of
    ``read`` (:func:`point_transmission`'s, joined) on its deck's grid; a
    polarized point's each channel beside its total.  Its printed figure
    is none."""
    e = read["energy_ev"]

    def total(trans) -> float:
        return landauer_current(e, trans, volts, kt_ev, spacing_ev)
    out: Dict = {"current_a": total(read["transmission"]),
                 "current_a_printed": None, "current_from": "joined"}
    chans = read.get("channels") or {}
    if "up" in chans and "down" in chans:
        out["current_by_spin_a"] = {k: total(chans[k]) / 2.0
                                    for k in ("up", "down")}
    return out


def iv_notes_placed(iv: Dict) -> List[Dict]:
    """The I-V's notes, each with the frame and the voltage it is about --
    a voltage's (keyed by its row of the table, :func:`low_bias_iv`)
    with both, a frame's own (keyed by the frame: its 0 V transmission not
    run) with ``bias_v`` ``None``: how a file with no row axis states them
    (`transport/datafile.py`, `engines/transport.md` § 2a.12)."""
    from ..paths import frame_token
    notes = iv.get("notes") or {}
    volts = list(iv.get("voltages_v") or ())
    frames = list(iv.get("frame") or [None] * len(volts))
    out = [{"frame": f, "bias_v": v, "note": notes[str(k)]}
           for k, (f, v) in enumerate(zip(frames, volts))
           if str(k) in notes]
    for f in dict.fromkeys(frames):
        key = frame_token(f) if f is not None else "all"
        if key in notes:
            out.append({"frame": f, "bias_v": None, "note": notes[key]})
    return out


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
    from ..jobset.materialize import latest_attempt
    from ..jobset.naming import stage_home
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


def _stage_facts(calc, label: str) -> List[Dict]:
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
      measured relative to it, and `G = G0 * T(E_F)` is evaluated at it.
    * *transmission* -- has its own `points` / `pending` / `failed` blocks.
    """
    from ..jobset.naming import ladder_homes
    from ..jobset.runstatus import jobset_status
    from .stages import STAGE_FACT, TRANSPORT_STAGES, frames_of, rung_points

    base, task = Path(calc.base), calc.task
    status = jobset_status(calc)
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
                      "detail": s.detail,
                      # THE RUN IT IS A COPY OF, a run copied in from the
                      # finished calculation this one reuses
                      # (`engines/transport.md` § 2a.16).
                      **({"copied_from": s.copied_from}
                         if s.copied_from else {})}
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
        # (`continuation.point_rows`), asked for that run.
        points = s.points
        if run is not None and s.attempt and s.dir \
                and run != base / s.dir / s.attempt and s.points:
            from ..jobset.continuation import point_rows
            from ..runrecord import launch_record
            from ..jobset.naming import stage_home
            _names = stage_home(base, task, name).names(task.shape)
            try:
                points, _ = point_rows(
                    run, _names, task, name, launch=launch_record(run, _names),
                    base=base)
            except Exception as exc:              # noqa: BLE001 -- said, never hidden
                fact["unreadable"] = f"{run.relative_to(base)}: {exc}"
                points = []
        if points:
            fact["taken"] = [
                {"frame": p.get("frame"), "of": p.get("of"),
                 "bias_v": p.get("bias_v"),
                 "attempt": (f"{s.dir}/{p['folder']}" if s.dir and p.get("folder")
                             else None),
                 "started_from": p.get("started_from"),
                 # THE POINT IT IS A COPY OF, the finished calculation's
                 # (`engines/transport.md` § 2a.16).
                 **({"copied_from": p["copied_from"]}
                    if p.get("copied_from") else {}),
                 "took": p.get("took") or []}
                for p in points]
        if rung_points(task, name, frames=frames_of(base)) \
                and answers != "product":
            # A RUNG THAT RUNS PER POINT, POINT BY POINT -- the report's
            # convergence card follows the selected point (`web/results.md`
            # § 2.5); the rung's own facts are its first point's.
            fact["by_point"] = _rung_points(base, task, name, token, answers,
                                            run=run)
            _bridge_charge_from_0v(fact["by_point"])
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
                    and run != base / s.dir / s.attempt and points:
                # ITS STATE BY THE STATUS DOOR'S ONE RULE for a run of
                # points (`continuation.points_verdict`), on the rows that
                # door read for this run (`point_rows`, above), never a
                # second rule.
                from ..jobset.continuation import points_verdict
                from .stages import points_summary
                fact["state"], said = points_verdict(points,
                                                     points_summary(points))
                fact["detail"] = (f"{said}; the run the transmission "
                                  f"gathered ({fact['attempt']}); the "
                                  f"stage's newest run is "
                                  f"{s.dir}/{s.attempt}: {s.detail}")
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
                if name == "device":
                    # ITS OWN PERIODIC START, as read; else where its chain
                    # began (:func:`_periodic_shift`).
                    fact["periodic"] = (
                        {**first["periodic"], "from": first["attempt"]}
                        if first.get("periodic") else
                        _periodic_shift(base, task, name, token,
                                        base / first["started_from"])
                        if first.get("started_from") else None)
            out.append(fact)
            continue
        outs = (_outs_newest_first(run, token)
                if answers != "product" and run is not None else [])
        if not outs:
            out.append(fact)
            continue
        fact.update(_science(outs[0], answers))
        if name == "device":
            fact["periodic"] = (
                {**fact["periodic"], "from": fact.get("attempt")}
                if fact.get("periodic") else
                _periodic_shift(base, task, name, token, run))
        out.append(fact)
    return out


def _periodic_shift(base: Path, task, stage: str, token: str,
                    folder: Path) -> Optional[Dict]:
    """THE SHIFT BETWEEN THE DEVICE'S ZERO AND THE SEED'S, where it was
    printed -- ``{vha_ev, from}``: the periodic start's `ts-Vha` of
    ``folder``'s own output, else of the run it continued from, back along
    the chain of runs to where it began (TranSIESTA carries the E_F over in
    the density file a run continues from, so the shift printed there is
    this run's; `engines/transport.md` § 2a.12) -- or ``None`` when no run
    of the chain is in the folder to say it."""
    from ..jobset.naming import stage_home
    from ..runrecord import LaunchRecordError, read_continued_from
    names = stage_home(base, task, stage).names(task.shape)
    seen: set = set()
    while folder is not None and folder.is_dir() and folder not in seen:
        seen.add(folder)
        outs = _outs_newest_first(folder, token)
        shift = _science(outs[0], "scf").get("periodic") if outs else None
        if shift:
            return {**shift, "from": str(folder.relative_to(base))}
        try:
            src = read_continued_from(folder, names, 0)
        except LaunchRecordError:
            return None
        folder = src.path(base) if src is not None else None
    return None


def gathered_device_run(base: Path, task) -> Optional[Path]:
    """The device run the transmission's newest run gathered its
    Hamiltonians from -- read from the transmission's first point's own
    ``.gathered-from`` (`runrecord.read_gathered_from`), the run above the
    point named there -- or ``None`` while the transmission has gathered
    nothing (`engines/transport.md` § 2a.12: the provenance is what was
    gathered, never the newest run by file time)."""
    from .stages import inputs_of
    # THE DEVICE'S FILE, as the DAG names it (`stage_inputs`), and THE RUN
    # its folder lies in -- a point's folder lies below it
    # (`RunRef.run_folder`).
    device = {fn for up, fn in inputs_of(task, "transmission")
              if up == "device"}
    for _pt, where, _run in result_folders(base, task, "transmission"):
        if where is None:
            continue
        for src, fn in _gathered(where):
            if fn in device:
                return src.run_folder().path(base)
    return None


def _gathered(folder):
    """What ``folder``'s run gathered (`runrecord.read_gathered_from`) -- a
    record that does not read is the record's refusal, a
    :class:`RecordError` naming it, as every reader of the transport record
    says one."""
    from ..runrecord import LaunchRecordError, read_gathered_from
    try:
        return read_gathered_from(folder)
    except LaunchRecordError as exc:
        raise RecordError(str(exc)) from None


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
    # AN ENGINE'S UNREADABLE NUMBER IS NOT STATED (`persist.json_safe`): a
    # field SIESTA overflowed reads as NaN, and every figure below -- E_F,
    # the frames' note -- is reasoned on, so it
    # enters the record as None, never as a NaN that compares and prints.
    hist = json_safe(frames[-1].scf_history or []) if frames else []
    fact["scf"] = [{k: c[k] for k in _SCF_FIELDS if k in c} for c in hist]
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
    periodic = [c for c in hist if c.get("phase") == "periodic"
                and c.get("vha_ev") is not None]
    if periodic:
        fact["periodic"] = {"vha_ev": periodic[-1]["vha_ev"]}
    # THE NEGF PHASE'S OWN FIGURES -- never the periodic start's
    # (`engines/transport.md` § 2a.12): its last row's E_F and charge, and
    # how many cycles it ran.
    negf = [c for c in hist if c.get("phase") == "negf"]
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


def _bridge_charge_from_0v(points: List[Dict]) -> None:
    """Each device point's ``negf.d_change_from_0v_e``: ``D(V) - D(0 V)``,
    the bridge's net Mulliken population at its voltage less its frame's at
    0 V, both the last NEGF cycle's (`engines/transport.md` § 2a.12, *The
    device's charges*) -- set where both are stated, never otherwise.  A
    frame's later voltages start from the previous voltage's density, so 0 V
    is the one reference every voltage of a frame shares."""
    at_0v = {p.get("frame"): ((p.get("negf") or {}).get("charges") or {})
             .get("D") for p in points
             if p.get("bias_v") is not None and float(p["bias_v"]) == 0.0}
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


def _rung_points(base: Path, task, name: str, token: str,
                 answers: str, run: Optional[Path] = None) -> List[Dict]:
    """``[{frame, bias_v, point, attempt, state, detail, ...science}]`` --
    each point of a rung that runs per point, its state the run door's
    (`run_status`) and its own answer (:func:`_science`); ``run`` names the
    run to read instead of the latest."""
    from ..jobset.naming import stage_home
    from ..runs import concluded
    names = stage_home(base, task, name).names(task.shape)
    out: List[Dict] = []
    for pt, where, run in result_folders(base, task, name, run=run):
        entry: Dict = {**pt.to_wire(), "point": pt.words()}
        if where is None or not where.is_dir():
            from ..jobset.runstatus import MISSING
            entry.update(state=MISSING[0], detail=MISSING[1])
            out.append(entry)
            continue
        entry["attempt"] = str(where.relative_to(base))
        # THE POINT'S STATE, the one door's, with its run's launch record
        # (`runs.concluded`, given the run's folder: the scheduler asked
        # once where it can, `running-a-job.md` § 4.2).
        st = concluded(where, names, base=base, launched_in=run)
        entry.update(state=st.state, detail=st.detail)
        outs = _outs_newest_first(where, token)
        if outs:
            entry.update(_science(outs[0], answers))
        out.append(entry)
    return out


def device_regions(junction) -> Dict[str, List[int]]:
    """The device's regions molbuilder owns -- L-electrode, bridge,
    R-electrode -- as atom indices (0-based, the composed junction's order,
    which is the seed's, the device's and the transmission's decks' -- each
    electrode rung lists a cell of its own), from the calculation's composed
    junction
    (`compose.recorded_junction`).  ``{}`` before it is composed
    (``None``)."""
    from .sort import ELECTRODE_LABELS, REGION_BRIDGE
    if junction is None:
        return {}
    regions = junction.regions or {}
    return {label: [int(i) for i in regions.get(label, ())]
            for label in (ELECTRODE_LABELS[0], REGION_BRIDGE,
                          ELECTRODE_LABELS[1])
            if regions.get(label)}


def _chain(base: Path, stages: List[Dict]) -> List[Dict]:
    """``[{stage, attempt, gathered: [{file, from}]}]`` -- what each rung's
    attempt was gathered from, read from its `.gathered-from`
    (`runrecord.read_gathered_from`); a rung that gathered nothing is
    listed with ``[]``."""
    out = []
    for st in stages:
        if not st.get("attempt"):
            continue
        row = {"stage": st["stage"], "attempt": st["attempt"],
               "gathered": [{"file": fn, "from": str(src)} for src, fn
                            in _gathered(base / st["attempt"])]}
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
    transport routes and an export refuse with."""
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
    from ..jobset.naming import stage_home
    from ..parse.dirs.job import FINAL_STATES
    from ..parse.dirs.record import tbtrans_loop
    from ..parse.engines._run_ending import keep_reading, version_of
    from ..parse.engines.siesta_reader import read_output
    from ..runfiles import JUNCTION_FILE
    from ..frame_average import (AverageError, average_block, average_said,
                                 citation_block)
    from ..frameset import read as read_frame_set
    from .compose import read_provenance, recorded_junction
    from .stages import frames_of, rung_points, same_volts
    from .energies import joined, sharing_of, spacing
    from .tbtnc import (DOS_CURVES, NO_CURVE, ORBITAL_NOTE,
                        ORBITAL_TYPES, TbtError, point_dos,
                        readings)

    base_dir = calc.base
    base, task = Path(base_dir), calc.task
    recorded = recorded_junction(base)
    junction = recorded.structure if recorded is not None else None
    frame_set = (read_frame_set(junction) if junction is not None
                 else None)
    points_out: List[Dict] = []
    pending: List[Dict] = []
    failed: List[Dict] = []
    home = stage_home(base, task, "transmission")
    token, names = home.token, home.names(task.shape)
    regions = device_regions(junction)
    # THE JUNCTION'S ATOM INDEX MAP -- every junction rung's deck written from
    # it -- which each point's kept reading must state (`model/overview.md`
    # § 2.2b, F7).
    from ..atom_permutation import AtomIndexMap
    junction_atoms = (AtomIndexMap.of_engine_record(junction.atom_order,
                                                    junction.n_atoms)
                      if junction is not None else None)
    stages = _stage_facts(calc, task.label)
    # A POINT NOT OPENED stands where its rung does -- the ready door's
    # `ready` / `waiting` and what for, on the rung's row (prep opens every
    # point at once).
    rung = next(f for f in stages if f["stage"] == "transmission")
    opened = False                        # an attempt open: it is prepared
    # WHETHER EACH RUN OF POINTS HAS ENDED, by the one rule the status door
    # asks (`continuation.points_state`) -- read once a run: a point a
    # stopped walk never reached is no point still to come.
    ended_runs: Dict[str, bool] = {}

    def _run_ended(run) -> bool:
        key = str(run)
        if key not in ended_runs:
            from ..jobset.continuation import stage_run_state
            try:
                state, _said = stage_run_state(calc, "transmission", run)
            except (OSError, ValueError, KeyError):
                state = "unreadable"
            ended_runs[key] = state in ("finished", "failed")
        return ended_runs[key]
    for pt, where, run in result_folders(base, task, "transmission"):
        # WHICH POINT, on every entry: its frame (``None`` with no frame
        # axis), its folder below the run, and in words.
        at = {**pt.to_wire(), "tokens": pt.rel, "point": pt.words()}
        opened = opened or where is not None
        if where is None:
            pending.append({**at, "state": rung["state"],
                            "why": rung.get("detail") or ""})
            continue
        rel = str(where.relative_to(base))
        # THE POINT'S NEWEST OUTPUT, READ ONCE (`model/parse.md` § 5d.5):
        # its reading serves the run door (`keep_reading`: an output that
        # stated its own end is not read again), the loop's words and the
        # currents -- three readers, one read.
        outs = _outs_newest_first(where, token)
        now = None
        if outs:
            try:
                reader = read_output(outs[0])
                now = reader.now()
                keep_reading(outs[0], version_of(outs[0]), reader.finish())
            except OSError:
                now = None
        # THE POINT'S STATE, the one door's -- asked as every reader asks
        # it, with its run's launch record (`runs.concluded`, given the
        # run's folder: the scheduler asked once where it can); one that
        # does not read is said.
        from ..runs import concluded
        st = concluded(where, names, base=base, launched_in=run)
        if st.state == "unreadable":
            failed.append({**at, "attempt": rel,
                           "state": "unreadable", "why": st.detail})
            continue
        # WHAT ITS RUN COMPUTES, beside what it shares -- the energies of
        # its grid its own run computes, and those the finished point holds
        # (`engines/transport.md` § 2a.16): its deck's record, said beside
        # its loop, which counts its own run's energies.
        sharing = sharing_of(where)
        shares = ({"shared": {"from": sharing.source, "held": sharing.held,
                              "computed": sharing.computed}}
                  if sharing is not None else {})
        if st.state != "finished":
            unreached = (st.state in ("pending", "queued")
                         and _run_ended(run))
            entry = {**at, "attempt": rel,
                     "state": "failed" if unreached else st.state,
                     "why": ("the walk ended before it reached this point"
                             if unreached else st.detail), **shares}
            # HOW FAR ITS LOOP IS -- running, or where it stopped -- in the
            # one sentence the point's run record states it in
            # (`tbtrans_loop`, `model/parse.md` § 5d.1b, § 5d.5).
            progress = tbtrans_loop(
                outs[0] if outs else None,
                over=st.state in FINAL_STATES, now=now)
            if progress:
                entry["progress"] = {
                    "said": progress["progress"],
                    **({"estimated_at": progress["estimated_at"]}
                       if "estimated_at" in progress else {})}
            (pending if st.state in ("pending", "queued", "running")
             and not unreached else failed).append(entry)
            continue
        # WHAT TBTRANS WROTE, through the family's one reader -- a point
        # that does not read is that point's failure, said in its words; the
        # other points still read.
        try:
            spin = deck_spin(where)
            # ITS GRID, JOINED -- its own run's energies and those it shares
            # with a finished calculation's point (`energies.joined`,
            # `engines/transport.md` § 2a.16).
            read = joined(where, lambda f: point_transmission(
                f, task.label, spin))
        except (OSError, ValueError, RecordError) as exc:
            failed.append({**at, "attempt": rel, "state": "finished",
                           "why": f"the run finished without its "
                                  f"transmission: {exc} ({st.detail})"})
            continue
        # THE DOS, ITS PARTS AND THE EIGENCHANNELS -- TBtrans's averages,
        # and each region's from the data file's kept reading -- a polarized
        # point's two channels summed with each spin's own beside
        # (`tbtnc.point_dos`; `web/results.md` § 2.5).
        def _dos(folder, spin=spin):
            return point_dos(folder, task.label, regions, spin,
                             junction_atoms)
        dos = None
        dos_why = None
        try:
            dos = joined(where, _dos)
        except TbtError as exc:
            dos_why = str(exc)
        if dos is not None and not any(k in dos for k in DOS_CURVES):
            # A BLOCK OF NO CURVE is no DOS, said.
            dos, dos_why = None, NO_CURVE
        # THE POINT IT IS A COPY OF -- taken over from, or copied in with
        # its run from, the finished calculation this one reuses (§ 2a.16):
        # the run that computed it (`runrecord.copied_of`).
        from ..runrecord import copied_of
        copied = copied_of(base, where)
        # ITS CURRENT, and which it is (`engines/transport.md` § 2a.16): the
        # printed figures' total, or -- for a point that shares energies --
        # the record's over its joined grid.
        currents = point_currents(where, token, spin, now=now)
        currents.update(
            joined_currents(read, pt.volts or 0.0,
                            BOLTZMANN_EV_K * deck_temperature_k(where),
                            spacing(sharing.window))
            if sharing is not None else {"current_from": "tbtrans"})
        energies = read["energy_ev"]
        points_out.append({
            **at,
            # ITS FRAME'S ROWS, whole -- what the frame is, as its writer
            # stated it (`model/structure.md` § 2.2d).
            "customized": (junction.customized_rows(
                frame=pt.frame if pt.frame is not None else 0)
                if junction is not None else []),
            "attempt": rel,
            # ITS STATE, the run door's word, as a pending or failed point
            # states its own (`engines/transport.md` § 2a.12).
            "state": st.state,
            **read,
            "conductance_g0": conductance_g0(energies, read["transmission"]),
            # THE TOTAL, and the figure it came from (`CURRENT_MEANS`), with
            # the run's k-sampling.
            **currents,
            # THE WINDOW AND ITS POINTS, as written (`engines/transport.md`
            # § 2a.12).
            "window_ev": [min(energies), max(energies)],
            "n_energies": len(energies),
            **shares,
            **({"copied_from": copied} if copied else {}),
            "spin": spin,
            **({"dos": dos} if dos is not None else {"dos_why": dos_why}),
            # WHAT IS KEPT OF EACH DATA FILE'S READING -- its own and those
            # in its `shared/` -- the DOS by region's (`model/parse.md`
            # § 5e): kept, missing, out of date or being read, with its
            # size and its claim.
            "digest": [k.to_wire(base) for k in readings(where, task.label)],
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
               + block(run_first("transmission", calc=calc)))
            + "\n"
            + "".join(f"  ({what}: "
                      + "; ".join(f"{p['point']}, "
                                  f"{p['state']} ({p['why']})"
                                  for p in got) + ")\n"
                      for what, got in (("pending", pending),
                                        ("failed", failed)) if got))

    try:
        # A PAGE STILL OPENS (`engines/transport.md` § 3.8.1): the Results
        # tab's read draws a refused frame set where the average would be;
        # the record `summarize` writes is refused by name.
        average = (average_said if partial else average_block)(
            frames_of(base) if junction is not None else None, frame_set,
            points_out, pending + failed, same_volts)
    except AverageError as exc:
        raise RecordError(str(exc)) from exc
    provenance = read_provenance(base)
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
        "low_bias_approximation": treatment == "low-bias-approximation",
        "low_bias_said": LOW_BIAS_SAID[treatment == "low-bias-approximation"],
        "treatment_note": TREATMENT_NOTE[treatment],
        # EVERY RUNG'S E_F, said in its frame (`fermi_frames`).
        "fermi_frames": fermi_frames(stages),
        # THE COMPOSED JUNCTION the device was built from, by its catalogue
        # name (`runfiles.JUNCTION_FILE`) -- what the report draws.
        "junction_file": JUNCTION_FILE,
        # WHEN ITS ORDER IS NOT RECORDED -- a junction whose sidecar states
        # none (I9) -- the sentence the report says beside the drawing, which
        # then lists the atoms as TranSIESTA does; ``None`` otherwise.
        "order_unstated": order_unstated_said(base, recorded),
        "points": points_out,
        # THE I-V: each point's own TBtrans current -- or, under the low-bias
        # approximation, computed by the record from the one 0 V slice for
        # every listed voltage (`low_bias_iv`, § 2a.10).
        "iv": _iv(base, task, treatment, points_out,
                  [pt.frame for pt in rung_points(
                      task, "transmission", frames=frames_of(base))]
                  or [None]),
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
    record["failed"] = failed
    return record


def order_unstated_said(base: Path, recorded) -> Optional[str]:
    """What the record, its data file and a selection's PDOS say of a
    junction whose sidecar states no order record (`model/overview.md`
    § 2.2a, I9) -- the fact (`compose.RecordedJunction.order_unstated`) and
    its way on, `prep task` (`engines/transport.md` § 1.1) -- or ``None``.
    ``recorded`` is `compose.recorded_junction`'s answer, ``None`` before
    anything was composed."""
    if recorded is None or not recorded.order_unstated:
        return None
    from ..jobset.commands import order_recorded
    return f"{recorded.order_unstated}.  {order_recorded(base=base)}"


def selection_pdos(calc, bias_v: float, atoms, orbitals: str,
                   frame: Optional[int] = None) -> Dict:
    """The PDOS of ``atoms`` at the point (``frame``, ``bias_v``) of the
    transport calculation ``calc`` (`calculation.open`), narrowed to one
    orbital type (`tbtnc.selection_pdos`; `web/results.md` § 2.5): read from
    that point's kept reading of its transmission run's data file, each
    orbital's type from the device run's ``.ORB_INDX`` at the same point --
    the two runs share the composed junction's atom order.  ``atoms`` are
    named in the user order, the order the report draws the junction in,
    and so are the answer's ``atoms`` and ``outside_device``
    (`model/overview.md` § 2.2a, I1, I3); an index that is not one of the
    junction's atoms is refused by name.  ``frame`` is the frame's index,
    named for a frame set and for nothing else.  :class:`RecordError` names
    what is missing."""
    from ..atom_permutation import AtomIndexError
    from ..structure import frame_words
    from .stages import Point, frames_of, same_volts
    from ..parse.engines.tbtrans_nc import data_files
    from .tbtnc import TbtError, selection_pdos as _pdos
    base, task = Path(calc.base), calc.task
    n = frames_of(base)
    # THE POINT IS A FRAME AND A VOLTAGE (`engines/transport.md` § 2a.9):
    # a frame set's names its frame, one structure's names none.
    if n > 1 and frame is None:
        raise RecordError(f"this calculation cites a frame set of {n} "
                          f"frames: name the frame (its index, 0 … {n - 1})")
    if n == 1 and frame is not None:
        raise RecordError(f"this calculation cites one structure, so a "
                          f"point names no frame: leave out "
                          f"{frame_words(int(frame))}")
    if frame is not None and (isinstance(frame, bool)
                              or int(frame) != frame):
        raise RecordError(f"a frame is named by its index, a whole number: "
                          f"{frame!r}")
    if frame is not None and not 0 <= int(frame) < n:
        raise RecordError(f"{frame_words(int(frame))} is not one of this "
                          f"calculation's {n} frame(s): 0 … {n - 1}")
    want = Point(frame=None if frame is None else int(frame), volts=bias_v,
                 of=n if frame is not None else None)
    said = want.words()

    def _at(stage: str, run: Optional[Path] = None):
        # THE POINT BY EACH AXIS THE RUNG VARIES (`stages.rung_points`): an
        # axis a rung does not vary is ``None`` on its points, and its one
        # result serves every value of it -- a rung that runs once serves
        # every point, one that runs per frame every voltage (a single
        # bias, the low-bias approximation).  ``(point, folder)``.
        return next(((pt, w) for pt, w, _r in result_folders(
                         base, task, stage, run=run)
                     if (pt.frame is None or pt.frame == want.frame)
                     and (pt.volts is None
                          or same_volts(pt.bias_v, want.bias_v))),
                    (None, None))

    if not any(same_volts(v, bias_v)
               for v in task.bias):
        raise RecordError(f"no transmission point at {bias_v:g} V")
    # THE SELECTION CROSSES INTO THE JUNCTION'S ORDER through its own order
    # record, and the answer's atoms back (`model/overview.md` § 2.2a, I3):
    # the runs list the atoms as the composed junction does.  A record that
    # states no order names no atom in the user order (I9).
    from .compose import recorded_junction
    recorded = recorded_junction(base)
    unstated = order_unstated_said(base, recorded)
    if unstated:
        raise RecordError(unstated)
    read_at, where = _at("transmission")
    if where is None:
        raise RecordError(f"no transmission point at {said}")
    ncs = data_files(where, task.label)
    if not ncs:
        raise RecordError(f"the transmission at {said} has written no data "
                          f"file")
    junction = recorded.structure if recorded is not None else None
    # THE DEVICE RUN THE TRANSMISSION GATHERED, as every device fact beside
    # a transmission is (`engines/transport.md` § 2a.12) -- never a newer one.
    _dev_at, dev = _at("device", run=gathered_device_run(base, task))
    # ITS ORBITAL INDEX, the engine's own file on the label, through the run
    # door (`runs.Run.on_label`).
    from ..runs import run_of
    dev_run = run_of(dev) if dev is not None else None
    orb = dev_run.on_label(".ORB_INDX") if dev_run is not None else None
    # THE SELECTION, each atom one of the junction's -- refused by name
    # otherwise, never wrapped (`model/overview.md` § 2.2a).
    try:
        mine = junction.map_from_user_index(atoms)
    except AtomIndexError as exc:
        raise RecordError(str(exc)) from exc
    # THE JUNCTION'S ATOM INDEX MAP, which each reading's must be (F7).
    from ..atom_permutation import AtomIndexMap
    junction_atoms = AtomIndexMap.of_engine_record(junction.atom_order,
                                                   junction.n_atoms)
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
        frames: List[Optional[int]]) -> Dict:
    """The record's I-V block -- PER FRAME (`engines/transport.md` § 2a.12):
    columns ``frame`` (each row's frame, ``None`` on a calculation with no
    frame axis), ``voltages_v``, ``current_a``, ``current_a_printed``.
    ``computed`` says where the currents came from: ``tbtrans`` -- each
    point's own printed integral -- or ``low-bias-approximation`` -- the record's
    integral of each frame's own 0 V slice for every listed voltage, with
    the Fermi tails at the electronic temperature that point's deck states
    (§ 2a.10).  The frames share one transmission deck, so ``kt_ev``,
    ``window_ev`` and the ``notes`` keyed by voltage are the block's; a
    frame whose 0 V point has not run is noted under its token (``all``
    with no frame axis).  A POLARIZED calculation adds ``current_up_a`` and
    ``current_down_a``, each spin's current beside the total
    (`web/results.md` § 2.5): TBtrans's printed figures, or under the
    approximation each spin's own integral, half of
    :func:`low_bias_iv` of that spin's T(E, 0) -- the integral is
    the two-channel G₀'s."""
    from ..structure import frame_words
    from .energies import spacing, window_of
    from ..paths import frame_token
    from .stages import same_volts
    polarized = any(p.get("spin") == "polarized" for p in points_out)
    if treatment == "low-bias-approximation":
        out: Dict = {"frame": [], "voltages_v": [], "current_a": [],
                     "current_a_printed": [], "computed": "low-bias-approximation",
                     "kt_ev": None, "window_ev": None, "notes": {}}
        if polarized:
            out["current_up_a"], out["current_down_a"] = [], []
        for f in frames:
            zero = next((p for p in points_out if p["frame"] == f
                         and same_volts(p["bias_v"], 0.0)), None)
            if zero is None:
                for v in task.bias:
                    out["frame"].append(f)
                    out["voltages_v"].append(float(v))
                    out["current_a"].append(None)
                    out["current_a_printed"].append(None)
                    if polarized:
                        out["current_up_a"].append(None)
                        out["current_down_a"].append(None)
                out["notes"][frame_token(f) if f is not None else "all"] = (
                    f"the 0 V transmission of {frame_words(f, len(frames))} "
                    f"has not run" if f is not None
                    else "the 0 V transmission has not run")
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
                for spin in ("up", "down"):
                    got = (low_bias_iv(zero["energy_ev"], chans[spin],
                                       list(task.bias), kt, spacing(window))
                           ["current_a"] if spin in chans
                           else [None] * len(iv["current_a"]))
                    out[f"current_{spin}_a"].extend(
                        None if i is None else i / 2.0 for i in got)
            # THE NOTES, KEYED BY THE ROW OF THE WHOLE TABLE: this frame's
            # rows start at `row0`, and a note keyed by its row within the
            # frame's own list is re-keyed by that offset.
            row0 = len(out["voltages_v"])
            for v, i in zip(iv["voltages_v"], iv["current_a"]):
                out["frame"].append(f)
                out["voltages_v"].append(v)
                out["current_a"].append(i)
                out["current_a_printed"].append(
                    zero["current_a_printed"] if same_volts(v, 0.0)
                    else None)
            for k, note in (iv.get("notes") or {}).items():
                out["notes"][str(row0 + int(k))] = note
            if out["kt_ev"] is None:
                out["kt_ev"], out["window_ev"] = iv["kt_ev"], iv["window_ev"]
        return out
    out = {"frame": [p["frame"] for p in points_out],
           "voltages_v": [p["bias_v"] for p in points_out],
           "current_a": [p["current_a"] for p in points_out],
           "current_a_printed": [p["current_a_printed"] for p in points_out],
           "computed": "tbtrans"}
    if polarized:
        for spin in ("up", "down"):
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
    device = by.get("device") or {}
    dev = device.get("negf") or {}
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
             "TranSIESTA and T(E) is measured relative to the leads' E_F.")
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
    "single-bias": ("One device SCF and one transmission, at the one "
                    "voltage: T(E) there, with no approximation beyond the "
                    "method."),
    "low-bias-approximation": (
        "The low-bias approximation: the device SCF converged once, at 0 V, "
        "and each voltage's current is the zero-bias transmission T(E, 0) "
        "integrated over that voltage's window -- the transmission's "
        "dependence on bias neglected; a resonance entering the window is "
        "not re-converged, nor moved by the field."),
    "self-consistent": ("The device SCF was converged at every voltage, so "
                        "each slice is its own solution and the I-V carries "
                        "no approximation beyond the method."),
}


def iv_table_text(record: Dict) -> str:
    """The printed deliverable: one row per point -- its frame's token,
    G(E_F), the junction's total current, and the figure TBtrans printed --
    then what the current is, in the record's words (:data:`CURRENT_MEANS`),
    and the frames' average at each voltage, or why there is none (their
    post-process's, `frame_average.summary_lines`)."""
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
    if iv.get("computed") == "low-bias-approximation":
        # THE RECORD'S OWN I-V, from each frame's 0 V slice
        # (`low_bias_iv`).
        kt = iv.get("kt_ev")
        lines.append("  I(V) computed from T(E, 0)"
                     + (f", kT = {kt:.4f} eV, window {iv.get('window_ev')} eV"
                        if kt is not None else "")
                     + ":")
        from ..paths import frame_token
        for k, (f, v, i) in enumerate(zip(iv.get("frame", ()),
                                          iv.get("voltages_v", ()),
                                          iv.get("current_a", ()))):
            notes = iv.get("notes") or {}
            # A VOLTAGE'S NOTE IS KEYED BY ITS ROW (`low_bias_iv`).
            note = notes.get(str(k)) or (
                notes.get(frame_token(f)) if f is not None
                else notes.get("all"))
            lines.append(f"  {frame_token(f) if f is not None else '--':>5}  "
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
