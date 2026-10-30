"""The vibration kind on SIESTA, through the whole described road: one stage,
one run in phases.

``molbuilder describe --engine siesta --calculation vibration`` → ``prep task``
→ ``launch task --mode direct`` on this workstation, and nothing after it: the
road's CO2 (`support.road.CO2`), listed O O C with the carbon held.  The one
stage `freq` runs as ONE RUN IN PHASES (`engines/vibration.md` § 5.11): its
`relax` call, its `fc` call nudging the two oxygens at the geometry the
relaxation reached, and -- the two stretches picked -- its optical calls, each a
SIESTA call in a folder of its own below the run, walked by one job; the
modes derived BY THE JOB -- the `fc` call's record step, `mb_vibration.pyz`,
run by each call's run script with the job's own python, which cannot import
molbuilder -- through the one path both engines share (§ 5.5, I22, I23).
The carbon is LAST in the input, so the held-first order the decks are
written in is a 3-cycle (`support.road.CO2_HELD_FIRST`), and the result comes
back in your order through the order record its deck states
(`model/overview.md` § 2.2a, I5, I7).

WHAT THE PHYSICS SAYS OF IT.  The molecule lies along z.  Six free
coordinates; the carbon held on the axis leaves two whole-body motions -- the
turns about it, about x and about y -- so four modes (`science/normal-modes.md`
§ 3.1): a bend pair, degenerate by the molecule's symmetry, moving the
oxygens in the (x, y) plane, and two stretches along z.

MUTATIONS THIS MUST FAIL AGAINST: a run script that does not run its
record step (the launch then ends with no spectrum, and no optical call is
planned); a bundle missing a member (the job's python cannot import it, and
the job fails); a plan step that writes the force-constant call's
coordinates from the structure rather than the relaxation (the geometry in
the result is the input's); a record step reading no order record from its
deck (the result's oxygens then stand where the carbon is); a finish that
does not send the run's `finish` (the notification record lists `start`
alone).

The roads: one per state of the person's one box (`engines/vibration.md`
§ 2.2) -- unticked, the run relaxes first (the shared run,
`support.real_runs.vibration`, which the relaxation's and the monitor's
e2e read too), its `fc` call at the geometry `relax` reached, one mode's
optical calls after it; ticked, the run is the `fc` call alone, at the
geometry as given -- then a displacement sweep (§ 5.9), on the unticked
calculation; and the unticked run stopped by the person once its force
constants and its first optical call ended, then launched again warm
(§ 5.11 § 6), on a copy of it -- and, on another copy, launched again warm
under a wall already passed, each optical call it runs failing alike
(`run-reports.md` § 2.9).
The optical calls' data file is read beside the index and the result
(§ 5.11 § 8).  The ticked road's result is then opened on the Results tab,
in a browser.

Needs the ``molbuilder-siesta`` env + conda hook on this machine; skipped
cleanly anywhere else.  Each force-constant call is thirteen single points of
CO2 on one rank, about a minute and a half here, each optical call some 25 s,
and the module launches five runs beside the shared one -- the ticked, the
sweep's second stage, the stopped, and the two launched after it, warm and
under the wall (`process/testing.md` tier 3: each run made once a pass).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from _road import conda_hook, env_available, set_template_value
from support.real_runs import notified, told
from support.road import (CO2, CO2_HELD_FIRST, _all_written, group_words,
                          ladder_edited)

pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(
        not (conda_hook().is_file() and env_available("molbuilder-siesta")),
        reason="needs the molbuilder-siesta env + a detectable conda hook"),
]

#: q0² · ν̃ for every mode: the zero-point amplitude in mass-weighted units is
#: q0 = sqrt(ħ / 2ω), so q0² ν̃ = ħ / (4π c) -- CODATA 2018's ħ, c and atomic
#: mass unit, in amu Å² cm⁻¹ (`engines/vibration.md` § 6.6).
_ZERO_POINT = 1.054571817e-34 / (4 * np.pi * 2.99792458e10
                                 * 1.66053906660e-27 * 1e-20)

#: The modes the shared run's optical phase is asked for -- the two
#: stretches, modes 3 and 4 of four, 1-based (`support.real_runs.CO2_OPTICAL`).
_PICKED_MODES = [3, 4]
#: Its optical calls' folders: frame 0, then each mode's two sides.
_OPTICAL_CALLS = ("f000", "f001", "f002", "f003", "f004")


@pytest.fixture(scope="module")
def unticked(real_co2_vibration, tmp_path_factory):
    """The box unticked: a copy of the shared run's calculation
    (`support.real_runs.copy_of`), on which, in order -- `summarize` asked
    of one stage; `freq` launched again, warm, with every call done and the
    result written; a displacement sweep, `freq_half` at 0.02 Bohr added
    beside `freq` in a group of its own (a launched group is fixed,
    `engines/stages.md` § 2), prepared, the sweep summarized (``early``),
    `freq_half` launched and summarized (``record``), and summarized again
    under a tolerance (``flagged``).  What each step answered is written
    down by its name."""
    from support.real_runs import _road_at, _taken, _typed, copy_of

    def on_the_run(run):
        record = run.bundle / "CO2.fc-sweep.json"
        with _road_at(run.tree.parent, tmp_path_factory):
            _typed(run, "summarize", "summarize", "task")
            _typed(run, "launch freq again", "launch", "task", "--stage",
                   "freq", "--mode", "direct", "--foreground", "--yes")
            ladder_edited({"added": [{"name": "freq_half",
                                      "overrides": {"fc_displacement": 0.02}}]},
                          run.bundle)
            _taken(run, "prep freq_half", "prep", "task",
                   *group_words(run.bundle, "freq_half"), "--target", "this")
            _taken(run, "summarize early", "summarize", "task")
            run.early = json.loads(record.read_text())
            _taken(run, "launch freq_half", "launch", "task", "--stage",
                   "freq_half", "--mode", "direct", "--foreground", "--yes")
            _taken(run, "summarize the sweep", "summarize", "task")
            run.record = json.loads(record.read_text())
            _taken(run, "summarize with a tolerance", "summarize", "task",
                   "--tolerance-cm1", "1e-6")
            run.flagged = json.loads(record.read_text())
    return copy_of(real_co2_vibration, "unticked", on_the_run)


@pytest.fixture(scope="module")
def ticked(tmp_path_factory):
    """The box ticked, on the road: the template's `already_relaxed` true
    and its temperature 350 K, `freq` prepared and launched -- one run, its
    `fc` call alone -- each answer written down."""
    from support.real_runs import (RealRun, _on_the_road, _taken,
                                   co2_vibration)
    with _on_the_road(tmp_path_factory, "ticked") as tree:
        run = RealRun(tree=tree, bundle=co2_vibration(tree))
        template = run.bundle / "CO2.template.toml"
        set_template_value(template, "already_relaxed", "true")
        set_template_value(template, "temperature_K", "350.0")
        _taken(run, "prep freq", "prep", "task",
               *group_words(run.bundle, "freq"), "--target", "this")
        _taken(run, "launch freq", "launch", "task", "--stage", "freq",
               "--mode", "direct", "--foreground", "--yes")
    return run


def _deck_of(call):
    """The deck the SIESTA call in ``call`` read, through the run door
    (`runs.run_of`), as the engine read it -- its coordinates block inlined
    from the file beside it (`parse.fdf.read_deck`): ``(scalars, blocks,
    run)``."""
    from molbuilder.parse.fdf import read_deck, read_fdf
    from molbuilder.runs import run_of
    run = run_of(call)
    return (*read_fdf(read_deck(run.deck)), run)


def _said(deck, key):
    """What the deck (:func:`_deck_of`) sets ``key`` to, as fdf reads it."""
    from molbuilder.parse.fdf import fdf_key
    return deck[0].get(fdf_key(key), [])


def _deck_coordinates(deck):
    """The deck's own coordinate block (:func:`_deck_of`), as SIESTA reads
    it."""
    from molbuilder.parse.fdf import fdf_key
    rows = deck[1][fdf_key("AtomicCoordinatesAndAtomicSpecies")]
    return np.array([[float(x) for x in r[:3]] for r in rows])


def _the_run(run, stage: str = "freq", n: int = 0):
    """Run ``n`` of ``stage`` as `jobset status` reads it -- the one status
    door (`support.real_runs.stage_status`), which reads the stage's newest
    run, so ``n`` must be it: finished, every call done and its finish
    ended; and its folder."""
    from support.real_runs import stage_status
    status = stage_status(run.bundle, stage)
    folder = run.attempt(stage, n)
    assert status.attempt == folder.name, (
        f"the status door read {status.attempt}, not {folder.name}")
    assert status.state == "finished", (status.state, status.detail)
    assert all(p["done"] for p in status.points), status.points
    return status, folder


def _the_result(run, stage: str = "freq", n: int = 0):
    """The spectrum the launch left at the run's root -- written by the
    job's own record step, so its run finished only with it
    (`engines/vibration.md` § 5.11 § 3: the step's failure is the job's) --
    the run read through the one status door (:func:`_the_run`), its
    result through the one door for a result on the label
    (`runs.on_label_at`), read by its reader (`sidecars.spectra.parse_spectra_json`, which
    refuses a derived value its inputs contradict), as it states itself."""
    from molbuilder.runfiles import ROLE_SPECTRA
    from molbuilder.runs import on_label_at
    from molbuilder.sidecars.spectra import parse_spectra_json
    _status, folder = _the_run(run, stage, n)
    out = on_label_at(folder, ROLE_SPECTRA)
    assert out is not None and out.is_file(), (
        "the launch ended without the spectrum at the run's root: the job "
        "did not finish its own calculation")
    return parse_spectra_json(str(out)).to_dict()


def _index_of(run, stage: str = "freq", n: int = 0) -> dict:
    """The run's index -- its calls, what each is, how each ended, the
    finish -- the file of its role among the run's own, read by its one
    reader (`siesta.vibration_phases.read_index`)."""
    from molbuilder.runfiles import ROLE_PHASES, find
    from molbuilder.siesta.vibration_phases import read_index
    from molbuilder.task import of as task_of
    (path, _rf), = find(run.attempt(stage, n), task_of(run.bundle).label,
                        role=ROLE_PHASES)
    return read_index(path)


def _the_optical_record(run, computed_in, n: int = 0):
    """THE RAMAN PHASE'S RECORD of run ``n`` of `freq`, and the three records
    of it agreeing (`engines/vibration.md` § 5.11 § 8, § 6.2): the run's
    data file, at its root -- read by its reader
    (`spectra.optical_data.read_optical_data`) and by `xarray.load_dataset`,
    as a person's script reads it -- lists the index's optical calls in the
    index's order, each frame finished, its mode, side and `δQ` the
    index's, the run that computed it ``computed_in[<call>]``, and its
    spectrum the call's own `.EPSIMG` as the parse registry reads it
    (`postprocess.parse`); the result (:func:`_the_result`) cites the file
    by its bytes' sha256, each frame's state, the selection, the light's
    direction and `δ` as the file states them.  Returns ``(result, data,
    calls)``."""
    import xarray
    from molbuilder.persist import sha256_of
    from molbuilder.postprocess import parse as through_the_registry
    from molbuilder.runfiles import ROLE_OPTICAL
    from molbuilder.runs import on_label_at, run_of
    from molbuilder.siesta.vibration_phases import OPTICAL, calls_of, entry_of
    from molbuilder.spectra.optical_data import read_optical_data
    d = _the_result(run, n=n)
    folder = run.attempt("freq", n)
    index = _index_of(run, n=n)
    calls = [c for c in calls_of(index) if c.phase == OPTICAL]
    path = on_label_at(folder, ROLE_OPTICAL)
    assert path is not None, \
        "the run left no data file of its optical calls at its root"
    cited = d["optical"]
    assert (cited["data_file"], cited["data_sha256"]) == (
        path.name, sha256_of(path)), cited
    data = read_optical_data(path)
    frames = data["frames"]
    assert [f["call"] for f in frames] == [c.rel for c in calls], frames
    grid = np.asarray(data["photon_energy_ev"])
    ds = xarray.load_dataset(path, group="frames")
    assert ds["eps2"].dims == ("frame", "photon_energy_ev", "spin")
    for k, (f, c) in enumerate(zip(frames, calls)):
        row = entry_of(index, c)
        assert f["state"] == "finished", (c.rel, f["state"])
        assert (f["mode"], f["side"], f["dq_amu12_ang"]) == (
            c.mode, int(c.side or 0), row.get("dq_amu12_ang")), (f, row)
        assert f["computed_in"] == computed_in[c.name], (c.rel, f)
        got = through_the_registry(run_of(folder / c.rel).on_label(".EPSIMG"))
        assert got.result_kind == "dielectric" and list(got.spin) == \
            data["spin"], got.spin
        assert np.array_equal(got.energy_ev, grid), c.rel
        assert np.array_equal(f["eps2"], got.eps2), c.rel
        assert np.array_equal(ds["eps2"].values[k], got.eps2.T), c.rel
    settings = data["settings"]
    assert (cited["direction"], cited["selection"], cited["displacement_ang"]
            ) == (settings["direction"], settings["selection"],
                  settings["delta_ang"]), (cited, settings)
    assert cited["reference"] == "finished", cited
    assert cited["modes"] == [
        {"mode": f["mode"], "frequency_cm1": f["frequency_cm1"],
         "dq_amu12_ang": f["dq_amu12_ang"], "plus": "finished",
         "minus": "finished"} for f in frames if f["side"] == 1], cited
    return d, data, calls


def _common_assertions(d, *, optical: bool):
    from molbuilder.spectra.results import SCHEMA_VERSION
    assert d["engine"] == "siesta" and d["schema_version"] == SCHEMA_VERSION
    # THE USER ORDER: the oxygens atoms 0 and 1, the held carbon atom 2.
    assert d["structure"]["elements"] == CO2["elements"]
    assert d["structure"]["metadata"]["regions"]["frozen_atoms"] == [2]
    assert d["free_atom_idxs"] == [0, 1] and "frozen_atom_idxs" not in d
    assert d["hessian_scope"] == "free" and d["n_atoms_in_hessian"] == 2
    # the two turns about the held carbon, about x and about y; four modes
    # left
    assert d["removed_motions"]["count"] == 2
    _a_bend_pair_and_two_stretches(d["modes"])
    assert d["equilibrium"]["mo_energies_eh"] is None
    assert d["thermo"]["regime"] == "vibrational-only"
    # the frequencies are this route's whole answer; the probe was never
    # asked of it (vibration.md § 4.9); Raman is the optical phase's, where
    # it ran -- complete once every optical call ended, its step δ the
    # force constants' own, the `.FC` header's (§ 6.2; § 5.11 § 3)
    assert d["phase_frequencies"] == "complete"
    assert d["phase_es"] == "not requested"
    assert d["raman_route"] == ("optical" if optical else "none"), d
    if optical:
        assert d["phase_raman"] == "complete", d["phase_raman"]
        assert d["raman_fd_step_ang"] == \
            d["engine_metadata"]["fc_displacement_ang"], d["raman_fd_step_ang"]
    else:
        assert d["phase_raman"] == "not requested"
        assert d["raman_fd_step_ang"] is None and d["optical"] is None
        assert d["modes"][0]["raman_activity_a4_amu"] is None
    assert "Head1997" in d["bibliography_keys"]
    # R5 on this route: the reference-step forces judged by the tolerance
    # the geometry is relaxed to (vibration.md § 5.5) -- the stage's own,
    # the template's when ticked: the kind's recommended 0.01 eV/A
    assert d["engine_metadata"]["reference_force_criterion_ev_ang"] == 0.01
    # the mass-calibrated displacement a transport step displaces along
    for m in d["modes"]:
        assert (m["zero_point_amplitude_amu12_ang"] ** 2
                * m["frequency_cm1"]) == pytest.approx(_ZERO_POINT,
                                                       rel=1e-3), m
    _states_its_masses_and_derived_values(d)
    # no strength is computed from the force constants alone, so no mode
    # can be placed in a channel (`engines/vibration.md` § 6.5)
    assert {m["activity_class"] for m in d["modes"]} == {"partial"}
    _states_its_charges_and_their_infrared(d)


def _states_its_charges_and_their_infrared(d):
    """`engines/vibration.md` § 5.10 ②, § 6.2: the result states every
    atom's charge at every force-constant step, by both partitions -- the
    reference, then each oxygen along x, y, z, side -1 before +1, by user
    index (the deck is a 3-cycle of the input) -- each step's charges
    summing to zero, the cell being neutral, and at the reference the
    oxygens negative and the carbon positive under both.  The infrared
    derived from them (`science/vibrational-optical-response.md` § 2) is
    the free atoms' -- the two oxygens, the carbon held (`web/spectra.md`
    § 3.2) -- and reads what holds exactly for any geometry along the axis,
    relaxed or not: a field along z sees no bend, a field along x no
    stretch; each stretch's dipole change along z opposes the oxygens' net
    motion along it -- negative charge moved; and the oxygens' charge
    changes only as the held carbon's does, the cell neutral: no bend moves
    any, a plane through the axis turning each bend into its own negative,
    while the stretches move charge between the oxygens and the carbon.
    (How silent the stretch whose oxygens move apart reads depends on the
    mesh, which tells the two oxygens apart -- 0.5 % of the other at this
    test's 150 Ry, 6.5e-8 at 500 Ry, the method's § 7 -- so it is not
    asserted.)

    MUTATIONS THIS MUST FAIL AGAINST: the finish's steps out of SIESTA's
    order -- an axis swapped lights a forbidden band, a side swapped turns
    each stretch's sign -- or a step's charges left in the deck's order,
    which puts the carbon's charge on an oxygen; the view taken over the
    whole molecule, which, neutral, moves no charge in any mode."""
    from molbuilder.spectra.infrared import infrared_view
    from molbuilder.spectra.results import SpectraResults
    pc = d["partial_charges"]
    assert pc is not None and sorted(pc["charges_e"]) == ["hirshfeld",
                                                          "voronoi"]
    assert [(s["atom"], s["axis"], s["side"]) for s in pc["steps"]] == (
        [(None, None, 0)] + [(a, axis, side) for a in (0, 1)
                             for axis in "xyz" for side in (-1, 1)])
    for name, rows in pc["charges_e"].items():
        assert len(rows) == 13 and all(len(r) == 3 for r in rows), name
        assert max(abs(sum(r)) for r in rows) < 1e-3, name
        assert rows[0][0] < 0 and rows[0][1] < 0 and rows[0][2] > 0, \
            (name, rows[0])
    result = SpectraResults.from_dict(d)
    oxygens_z = [float(np.asarray(m["eigenvector_canonical"])[:, 2].sum())
                 for m in d["modes"]]
    for partition in ("hirshfeld", "voronoi"):
        view = infrared_view(result, direction="z", partition=partition)
        assert (view.atoms_label, view.atoms) == ("the free atoms", (0, 1))
        along_z = view.modes
        along_x = infrared_view(result, direction="x",
                                partition=partition).modes
        stretch = max(m.intensity_km_mol for m in along_z[2:])
        bend = max(m.intensity_km_mol for m in along_x[:2])
        assert stretch > 0 and bend > 0, (partition, along_z, along_x)
        assert all(m.intensity_km_mol < 1e-6 * stretch
                   for m in along_z[:2]), (partition, along_z)
        assert all(m.intensity_km_mol < 1e-6 * bend
                   for m in along_x[2:]), (partition, along_x)
        for k in (2, 3):
            assert along_z[k].dmu_dq_e_per_amu12[2] * oxygens_z[k] < 0, \
                (partition, k, along_z[k], oxygens_z[k])
        moved = [m.charge_moved_e_per_amu12_ang for m in along_z]
        assert all(abs(q) < 1e-3 for q in moved[:2]), (partition, moved)
        assert max(abs(q) for q in moved[2:]) > 1e-3, (partition, moved)


def _a_bend_pair_and_two_stretches(modes):
    """The module's head: the two lowest modes a degenerate pair moving the
    oxygens in (x, y) alone, the two highest moving them along z alone."""
    assert len(modes) == 4, [m["frequency_cm1"] for m in modes]
    freqs = [m["frequency_cm1"] for m in modes]
    assert freqs == sorted(freqs)
    assert freqs[1] - freqs[0] < 0.01 * freqs[0], freqs
    assert freqs[2] > freqs[1], freqs
    for k, m in enumerate(modes):
        L = np.asarray(m["eigenvector_canonical"])
        along_z = float((L[:, 2] ** 2).sum() / (L ** 2).sum())
        assert (along_z < 0.01 if k < 2 else along_z > 0.99), (k, along_z)


def _states_its_masses_and_derived_values(d):
    """`engines/vibration.md` § 6.2, § 6.6, I3: the result states every
    atom's mass -- `chemistry.atomic_mass`'s, the table every deck's masses
    are stated from -- and its modes are weighted by those masses (each
    canonical vector `Σ m|L|² = 1` in them); each mode's derived values sit
    beside it -- which its reader took (:func:`_the_result`)."""
    from molbuilder.chemistry import atomic_mass
    from molbuilder.spectra.derived import DERIVED_MODE_KEYS
    eq = d["equilibrium"]
    # The elements are the structure's (schema 9, § 6.10.1); the masses
    # stay beside the geometry's energies, every atom's.
    assert eq["masses_amu"] == [atomic_mass(e)
                                for e in d["structure"]["elements"]], eq
    stated = np.asarray(eq["masses_amu"])[d["free_atom_idxs"]]
    for m in d["modes"]:
        assert all(k in m for k in DERIVED_MODE_KEYS), sorted(m)
        L = np.asarray(m["eigenvector_canonical"])
        assert abs(float((stated[:, None] * L * L).sum()) - 1.0) < 1e-6, m


def _relaxed_geometry(call):
    """Where the `relax` call in ``call`` ended, in the user order -- the run
    door's one read of a run's geometry (`runs.steps_of`)."""
    from molbuilder.runs import openable, run_of, steps_of
    return np.asarray(steps_of(run_of(call), Path(openable(call)[0]))
                      .last.positions, dtype=float)


def test_unticked_the_run_relaxes_first_and_measures_at_the_relaxed_geometry(
        real_co2_vibration, unticked):
    """The box unticked (the default `describe` writes): the one run's
    calls are the relaxation, then the force constants at the geometry it
    reached -- in the held-first order its deck states, written beside that
    deck by the run's own plan step -- then the two picked stretches' five
    optical calls, each planned by the `fc` call's record step; the modes come out
    in the user order with the reference forces within the stage's own
    tolerance (`engines/vibration.md` § 5.11 §§ 1–3, § 5.5); the index
    lists every call with how it ended, the selection as asked and applied,
    and the finish; the run sent its start, a milestone after the
    relaxation and after the force constants, each saying its phase's
    outcome, and its finish (§ 7a, `run-reports.md` § 2.9); and
    `status` counts the run in its phases' words (§ 7).  Launched again warm
    with every call done and the result written, a launch has nothing to
    run and says so, naming `--cold` (§ 6)."""
    from molbuilder.parse.fdf import fdf_key
    from molbuilder.runs import declared
    from molbuilder.siesta.vibration_phases import FC, OPTICAL, RELAX
    run = real_co2_vibration
    status, folder = _the_run(run)
    n = len(_OPTICAL_CALLS)
    assert status.detail == f"relax done · fc done · optical {n} of {n}", \
        status.detail
    calls = {(p["phase"], p["call"]): p for p in status.points}
    assert [k[0] for k in calls] == [RELAX, FC] + [OPTICAL] * n, list(calls)
    # the ordinary relaxation deck, from the same held-first copy: Broyden
    # to the kind's recommended 0.01 eV/A, the held carbon -- SIESTA's atom 1
    # after the sort -- constrained
    relax = _deck_of(run.call("freq", RELAX))
    assert _said(relax, "MD.TypeOfRun") == ["Broyden"]
    assert _said(relax, "MD.MaxForceTol")[:1] == ["0.01"]
    held = [["position", "1"]]
    assert relax[1][fdf_key("Geometry.Constraints")] == held
    # the force-constant deck: the oxygens SIESTA's atoms 2 and 3, the
    # carbon held; the run states the held-first order record, read by the
    # run door's reader (I5); and its coordinates are where the relaxation
    # ended, in that order (§ 5.11 § 3)
    fc = _deck_of(run.call("freq", FC))
    assert (_said(fc, "MD.TypeOfRun"), _said(fc, "FC.First"),
            _said(fc, "FC.Last")) == (["FC"], ["2"], ["3"])
    assert fc[1][fdf_key("Geometry.Constraints")] == held
    assert declared(fc[2]).atom_order.to_dict() == CO2_HELD_FIRST
    relaxed = _relaxed_geometry(run.call("freq", RELAX))
    assert np.allclose(_deck_coordinates(fc),
                       relaxed[CO2_HELD_FIRST["sorted_to_original"]],
                       atol=1e-6)
    fc_file = fc[2].on_label(".FC")
    assert fc_file is not None and fc_file.is_file(), \
        "the force-constant call must leave its .FC"
    # THE INDEX: every call, how it ended, the selection, the finish.
    index = _index_of(run)
    assert index["phases"] == [RELAX, FC, OPTICAL]
    assert [(c["phase"], c["exit"]) for c in index["calls"]] == \
        [(RELAX, 0), (FC, 0)] + [(OPTICAL, 0)] * n, index["calls"]
    assert index["selection"]["applied"] == _PICKED_MODES, index["selection"]
    assert index["finish"]["exit"] == 0, index["finish"]
    # the optical calls: the equilibrium, then each mode's two sides,
    # each a folder below the run's `optical/` holding the optical deck and
    # what SIESTA left (`vibration_phases.calls_of`, `Call.rel`)
    from molbuilder.siesta.vibration_phases import calls_of
    optical_calls = [c for c in calls_of(index) if c.phase == OPTICAL]
    assert [(c.mode, c.side) for c in optical_calls] == [(None, 0)] + [
        (m, side) for m in _PICKED_MODES for side in (1, -1)], optical_calls
    for c in optical_calls:
        optical = _deck_of(folder / c.rel)
        assert _said(optical, "OpticalCalculation") == [".true."], c
        assert optical[2].on_label(".EPSIMG") is not None, c
    # THE RUN REPORTED AS ONE RUN (§ 7a, `run-reports.md` § 2.9): its
    # start; a milestone at each phase it went on after -- the relaxation,
    # the force constants; its finish.
    assert notified(run, "freq") == ["start", "milestone", "milestone", "finish"]
    # THE LAUNCH ENDS WITH THE RESULT: the record step wrote it (§ 5.11 § 3).
    d = _the_result(run)
    _common_assertions(d, optical=True)
    # EACH MILESTONE SAYS ITS PHASE'S OUTCOME, from the record its step
    # wrote, and the run's count after it (`run-reports.md` § 2.9).
    after_relax, after_fc = told(run, "milestone", "freq")
    rec = index["calls"][0]["relaxation"]
    assert after_relax.startswith(
        f"running (the relaxation "
        f"{'converged' if rec['converged'] else 'did not converge'} in "
        f"{rec['n_steps']} move"), after_relax
    assert f"(tol {rec['force_tolerance_ev_ang']:g}))" in after_relax, \
        after_relax
    assert "| the run: relax done · fc pending · optical: after fc |" \
        in after_relax, after_relax
    imaginary = sum(bool(m["has_imag"]) for m in d["modes"])
    assert after_fc.startswith(
        f"running (the force constants: {len(d['modes'])} modes, "
        + (f"{imaginary} imaginary" if imaginary else "none imaginary")), \
        after_fc
    stationary = {True: "a stationary point",
                  False: "not a stationary point"}[
                      d["relaxation"]["converged"]]
    assert (f"-- {stationary} at this level of theory; {n} optical calls "
            f"planned)") in after_fc, after_fc
    assert f"| the run: relax done · fc done · optical 0 of {n} |" \
        in after_fc, after_fc
    # HOW IT WAS COMPUTED, the force-constant deck cited (§ 6.2): by its
    # sha256, every keyword and block as the engine read them but the
    # geometry, and each species' pseudopotential SIESTA opened beside it.
    from molbuilder.parse.fdf import GEOMETRY_KEYS
    from molbuilder.persist import sha256_of
    cited = d["config"]["deck"]
    assert (cited["name"], cited["sha256"]) == (
        fc[2].deck.name, sha256_of(fc[2].deck)), cited
    assert {fdf_key(k): v.split() for k, v in cited["keywords"].items()} \
        == {k: v for k, v in fc[0].items() if k not in GEOMETRY_KEYS}
    assert {fdf_key(k): [r.split() for r in rows]
            for k, rows in cited["blocks"].items()} \
        == {k: v for k, v in fc[1].items() if k not in GEOMETRY_KEYS}
    assert "kgrid_Monkhorst_Pack" in cited["blocks"], sorted(cited["blocks"])
    species = [r[2] for r in fc[1][fdf_key("ChemicalSpeciesLabel")]]
    assert cited["pseudopotentials"] == [
        {"species": s, "name": f"{s}.psml",
         "sha256": sha256_of(fc[2].deck.parent / f"{s}.psml")}
        for s in species], cited["pseudopotentials"]
    # ...each by the spelling the deck writes (vibration_deck: the run).
    assert cited["keywords"]["MD.TypeOfRun"] == "FC", cited["keywords"]
    assert d["relaxation"]["converged"] is True
    assert d["relaxation"]["already_relaxed"] is False
    assert d["phase_relaxation"] == "complete"
    assert d["relaxation"]["enabled"] is True and d["relaxation"]["n_steps"] >= 1
    # the geometry in the file is the one the force constants were taken
    # at -- where the relaxation ended, in the user order
    assert np.allclose(d["structure"]["positions"], relaxed, atol=1e-5)

    # ...and summarize derives nothing: with one stage there is no sweep to
    # summarize (§ 5.9, I22).
    said = unticked.said["summarize"]
    assert said.exit_code != 0 and "a sweep compares two or more" in \
        said.output, said.output
    # LAUNCHED AGAIN with every call done and the finish written: refused,
    # the cold launch named, and no run opened (§ 5.11 § 6).
    said = unticked.said["launch freq again"]
    assert said.exit_code != 0 and "--cold" in said.output, said.output
    assert not unticked.attempt("freq", 1).exists(), (
        "a refused launch opened a run")


def test_an_optical_calls_monitor_says_where_its_run_is(real_co2_vibration):
    """`execution/run-reports.md` § 2.8: the reading pass says SIESTA's
    optical calculation underway from its opening line -- read by the
    output's parts table -- until the run ends; a member monitor of an
    optical call says its run's count and its own call from the run's
    index, the pace of the phase's finished calls and the calls left, and
    the run's clock from its first call beside the call's own.

    THE SILENT FAILURE (the user's Raman run on Sol, 2026-10-28): two hours
    into SIESTA's optical calculation the card said *periodic SCF iteration
    6 … 5.43 s/iter … elapsed 2h* -- a converged SCF as the row in flight,
    the call's clock unnamed as the run's, nothing of the 121 calls.  Only a
    real optical call's output carries the part's opening line, and only a
    real run's index its calls' starts and ends."""
    import time
    from molbuilder.jobset.naming import stage_home
    from molbuilder.monitor import WatchedRun
    from molbuilder.parse.engines.siesta_grammar import output_part
    from molbuilder.parse.engines.siesta_reader import SiestaReader
    from molbuilder.runfiles import FIRST_TRY, find
    from molbuilder.siesta import vibration_phases as vp
    from molbuilder.task import of as task_of
    from molbuilder.runrecord import stamp_time
    run = real_co2_vibration
    task = task_of(run.bundle)
    folder = run.attempt("freq")
    # THE RUN'S OWN CALLS, from its index; the second optical call.
    index = _index_of(run)
    optical = [c for c in vp.calls_of(index) if c.phase == vp.OPTICAL]
    call = folder / optical[1].rel
    (out, _rf), = find(call, task.label, role=".out")
    # THE READING PASS, fed the output as the monitor's wakes feed it: the
    # SCF speaks until the part's opening line -- found by the grammar's
    # own declaration -- the part from it, and no part once the run ends.
    begins = output_part("optical").begins
    lines = out.read_text(encoding="utf-8", errors="replace").split("\n")
    at = next(i for i, ln in enumerate(lines) if begins in ln.lower())
    reader = SiestaReader()
    for ln in lines[:at]:
        reader.feed(ln)
    before = reader.now()
    assert before.get("part") is None, before.get("part")
    assert "SCF iteration" in (before.get("doing") or ""), before.get("doing")
    reader.feed(lines[at])
    during = reader.now()
    assert during.get("part") == "optical", during.get("part")
    assert during.get("doing") == ("SIESTA's optical calculation -- it "
                                   "prints nothing until it ends"), during
    for ln in lines[at + 1:]:
        reader.feed(ln)
    assert reader.now().get("part") is None, reader.now().get("part")
    # THE MEMBER MONITOR, as the call's run script starts it -- the run's
    # folder its `--launched-in` -- read once the call has ended.
    names = stage_home(run.bundle, task, "freq").names
    watched = WatchedRun(names=names, run=FIRST_TRY, directory=call,
                         launched_in=folder)
    now = time.time()
    st = watched.read(now - 1.0, now)
    text = st.as_text()
    n = len(_OPTICAL_CALLS)
    assert st.run_words and f"optical {n} of {n}" in st.run_words, text
    assert st.call_words and st.call_words.startswith(
        f"mode {_PICKED_MODES[0]} "), text
    assert "SIESTA's optical calculation" not in text, text
    assert (st.units_done, st.units_left) == (n, 0), text
    assert st.unit_s is not None and st.unit_s > 0, text
    assert "each optical call takes " in text, text
    assert st.call_elapsed_s is not None and "this call" in text, text
    # THE RUN'S CLOCK, from its first call's start in its index.
    rows = vp.rows_of(index, folder, task.label)
    first = min(stamp_time(r["started"]).timestamp() for r in rows
                if r.get("started"))
    assert abs(st.elapsed_s - (now - first)) < 1.0, (st.elapsed_s, now - first)


@pytest.mark.parametrize("cited", ["the stage's folder", "its run"])
def test_a_vibration_cited_as_a_junction_names_its_frame_sets(
        real_co2_vibration, cited):
    """`engines/transport.md` § 3.1: a transport calculation takes a
    vibration through one of its modes -- a frame set, written from its
    result by the Results tab's *Write frame set* (`web/spectra.md` § 4.3)
    -- never through the vibration's own folder, which the citation door
    refuses; its refusal names that way.

    THE SILENT FAILURE (the user, on Sol, 2026-10-28): the extension's
    frame-set chooser, given a vibration's stage folder, refused it and named
    one way only -- relax the junction -- a road away from the frames the
    person asked for."""
    from molbuilder.transport.compose import ComposeError, classify_citation
    run = real_co2_vibration.attempt("freq", 0)
    with pytest.raises(ComposeError) as got:
        classify_citation(run.parent if cited == "the stage's folder"
                          else run)
    said = str(got.value)
    assert "Write frame set" in said and "cite the .xyz it saves" in said, \
        said


def test_the_optical_calls_are_one_data_file_the_result_cites(
        real_co2_vibration):
    """THE RAMAN PHASE'S RECORD (`engines/vibration.md` § 5.11 § 8, § 6.2):
    the shared run's five optical calls -- the reference geometry, then each
    picked mode's two sides -- in one data file at the run's root that the
    index and the result agree with (:func:`_the_optical_record`), each
    computed in this run; the light along the optical deck's
    `Optical.Vector`; the selection as asked and applied; and each frame at
    the geometry the displacement rule gives (§ 5.11 § 3): the reference
    the result's own, in the user order, and each side the reference moved
    by ±δQ along the mode's vector -- zero on the held carbon -- its
    largest atom's move `δ`, the force constants' step.

    MUTATIONS THIS MUST FAIL AGAINST: a frame's geometry left in the deck's
    held-first order (the carbon's row then moves); a frame's spectrum from
    another call's file; the result citing the data file as the force
    constants' record step wrote it, before the optical calls ran (its
    sha256 and every state pending)."""
    from molbuilder.parse.fdf import fdf_key
    run = real_co2_vibration
    d, data, calls = _the_optical_record(
        run, computed_in=dict.fromkeys(_OPTICAL_CALLS, "01_freq/run-0"))
    assert data["settings"]["selection"]["applied"] == _PICKED_MODES
    optical = _deck_of(run.attempt("freq") / calls[0].rel)
    vector = optical[1][fdf_key("Optical.Vector")]
    assert data["settings"]["direction"] == [float(x) for x in vector[0][:3]]
    reference, *sides = data["frames"]
    r0 = np.asarray(reference["position_ang"])
    assert np.allclose(r0, d["structure"]["positions"], atol=1e-5)
    delta = d["engine_metadata"]["fc_displacement_ang"]
    assert [f["mode"] for f in sides] == [m for m in _PICKED_MODES
                                          for _side in (1, -1)], sides
    for f in sides:
        mode = d["modes"][f["mode"] - 1]
        assert f["frequency_cm1"] == mode["frequency_cm1"], (f, mode)
        vec = np.zeros_like(r0)
        vec[d["free_atom_idxs"]] = np.asarray(mode["eigenvector_canonical"])
        move = np.asarray(f["position_ang"]) - r0
        assert np.allclose(move, f["side"] * f["dq_amu12_ang"] * vec,
                           atol=1e-5), (f["mode"], f["side"], move)
        assert np.linalg.norm(move, axis=1).max() == pytest.approx(
            delta, abs=1e-5), move


#: The laser a Raman view is asked at, nm -- a helium-neon's.
_LASER_NM = 633.0


def _stretches(d):
    """The two picked stretches by how their oxygens move along z, the
    molecule's axis: ``(apart, together)`` -- the mode whose oxygens move in
    opposite directions, both bonds lengthening at once, and the one whose
    oxygens move the same way, one bond lengthening as the other shortens."""
    by = {}
    for m in _PICKED_MODES:
        oxygens = np.asarray(d["modes"][m - 1]["eigenvector_canonical"])
        by["apart" if oxygens[0][2] * oxygens[1][2] < 0 else "together"] = m
    return by["apart"], by["together"]


def test_the_raman_at_a_laser_holds_co2s_symmetry_and_exports_whole(
        real_co2_vibration, monkeypatch, tmp_path):
    """THE RAMAN AT A LASER (`engines/vibration.md` § 5.11 § 10, § 14): the
    load door's view of the shared run at 633 nm, broadened to the run's own
    width -- **CO2's symmetry**: the held carbon is the molecule's centre of
    inversion, so the polarizability is the same at +dQ and -dQ of the
    stretch whose oxygens move together and its strength is ~0, against the
    one whose oxygens move apart, a band; the script's door
    (`postprocess.raman_view`) gives the same view.  **The export at that
    laser** through `jobset export`, NetCDF and CSV: the run's two data
    files byte for byte and the view composed now, read back through
    `postprocess.read_raman_data` and `xarray` equal to the door's, citing
    the result and the optical calls' data file by their sha256; nothing is
    written into the calculation.

    MUTATIONS THIS MUST FAIL AGAINST: a mode's derivative taken from
    another mode's side (the together stretch then reads a band); an export
    holding a number the page does not show; a view citing a file by a
    checksum that is not its bytes'; an export writing into the
    calculation's folder."""
    import csv
    import xarray
    from molbuilder.persist import sha256_of
    from molbuilder.postprocess import raman_view as script_view
    from molbuilder.postprocess import read_raman_data
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.runfiles import ROLE_OPTICAL, ROLE_SPECTRA
    from molbuilder.runs import on_label_at
    from molbuilder.spectra.activity import CHANNEL_PRESENT_FLOOR_RAMAN_A4_AMU
    from molbuilder.task import of as task_of
    from molbuilder.web.app import create_app
    from support.road import jobset
    run = real_co2_vibration
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(run.tree))
    d = _the_result(run)
    folder = run.attempt("freq")
    result = on_label_at(folder, ROLE_SPECTRA)
    optical = on_label_at(folder, ROLE_OPTICAL)
    label = task_of(run.bundle).label

    # THE LOAD DOOR, asked as the page asks it.
    got = create_app(config={}).test_client().post(
        "/api/spectra/load",
        json={"path": str(result), "raman": {"laser_nm": _LASER_NM}}
    ).get_json()
    assert got["ok"] and "raman_said" not in got["results"], got
    view = got["results"]["raman"]
    assert view["laser_nm"] == _LASER_NM, view
    assert view["broaden_ev"] == view["run_broaden_ev"], view
    assert view["direction_name"] == "z", view
    rows = {m["index_1based"]: m for m in view["modes"]}
    assert sorted(rows) == _PICKED_MODES, rows
    assert {rows[m]["state"] for m in _PICKED_MODES} == {"finished"}, rows
    apart, together = _stretches(d)
    for key in ("strength_laser_a4_amu", "strength_static_a4_amu"):
        assert rows[apart][key] > CHANNEL_PRESENT_FLOOR_RAMAN_A4_AMU, rows
        assert rows[together][key] < 1e-2 * rows[apart][key], rows
    assert rows[apart]["relative"] == pytest.approx(1.0), rows
    assert script_view(result, laser_nm=_LASER_NM).to_wire() == view

    # THE EXPORT AT THAT LASER, both forms, outside the calculation.
    before = _all_written(run.bundle)
    for fmt in ("netcdf", "csv"):
        said = jobset("export", "--bundle", run.bundle, "--format", fmt,
                      "--to", tmp_path, "--laser-nm", _LASER_NM)
        assert said.exit_code == 0, said.output
    assert _all_written(run.bundle) == before, "nothing written in it"
    spectra_nc = result.with_name(f"{label}.spectra.nc")
    out = tmp_path / f"{label}.vibration-netcdf"
    assert sorted(f.name for f in out.iterdir()) == sorted(
        [spectra_nc.name, optical.name, f"{label}.raman.nc"])
    for kept in (spectra_nc, optical):
        assert (out / kept.name).read_bytes() == kept.read_bytes(), kept
    # THE OPTICAL CALLS' FILE, exported, reads through the script's door
    # with its atom index map (§ 10's table): the run's own frames.
    from molbuilder.postprocess import read_optical_data
    from molbuilder.spectra.optical_data import read_optical_data as own
    got = read_optical_data(out / optical.name)
    assert [f["call"] for f in got["frames"]] == [
        f["call"] for f in own(optical)["frames"]]
    assert got["atom_index_map"] is not None
    data = read_raman_data(out / f"{label}.raman.nc")
    assert data["derived_from"] == [
        {"name": result.name, "sha256": sha256_of(result)},
        {"name": optical.name, "sha256": sha256_of(optical)}]
    assert (data["laser_nm"], data["broaden_ev"]) == (
        view["laser_nm"], view["broaden_ev"])
    keys = ("strength_laser_a4_amu", "strength_static_a4_amu",
            "dalpha_dq_laser_real_a2_per_amu12",
            "dalpha_dq_laser_imag_a2_per_amu12",
            "dalpha_dq_static_a2_per_amu12")
    for row in data["modes"]:
        for key in keys:
            assert row[key] == pytest.approx(rows[row["mode"]][key],
                                             rel=1e-12), (row, key)
    with xarray.open_dataset(out / f"{label}.raman.nc", group="modes") as ds:
        assert [rows[int(m)]["strength_laser_a4_amu"]
                for m in ds["mode"].values] == pytest.approx(
            ds["strength_laser_a4_amu"].values.tolist(), rel=1e-12)
    tables = tmp_path / f"{label}.vibration-csv" / f"{label}.raman"
    assert (tables / "README.txt").is_file()
    with (tables / "modes" / "mode.csv").open() as fh:
        table = list(csv.DictReader(fh))
    column = next(k for k in table[0]
                  if k.startswith("strength_laser_a4_amu"))
    assert {int(r["mode"]): float(r[column]) for r in table} == \
        pytest.approx({m: rows[m]["strength_laser_a4_amu"]
                       for m in _PICKED_MODES}, rel=1e-12)


def test_ticked_the_run_measures_at_the_geometry_as_given(ticked):
    """The box ticked: no relaxation -- the run's one call is the force
    constants at the geometry as given, the reference forces judged by the
    template's tolerance -- the road's CO2, one C=O long, is no stationary
    point, and the result says so.

    The thermochemistry is summed at the TEMPLATE'S temperature, which is
    one meaning on both engines (§ 3.1): set to 350 K here, it reaches the
    result through the deck's `vibration` block -- and a vibrational-only
    result records no pressure, since none enters it (§ 4.7).

    MUTATION THIS MUST FAIL AGAINST: prep leaving the temperature out of the
    block -- the record step then refuses the deck and the launch fails.
    """
    from molbuilder.constants import HARTREE_BOHR_EV_ANGSTROM_ASE
    from molbuilder.siesta.vibration_phases import FC
    status, _folder = _the_run(ticked)
    assert status.detail == "fc done", status.detail
    assert _index_of(ticked)["phases"] == [FC]
    assert notified(ticked, "freq") == ["start", "finish"]
    d = _the_result(ticked)
    # THE FORCE CONSTANTS END THE RUN, so the finish says their outcome
    # (`run-reports.md` § 2.9), with the result's own stationarity verdict.
    finish, = told(ticked, "finish", "freq")
    assert finish.startswith(
        f"finished (the force constants: {len(d['modes'])} modes, "), finish
    stationary = {True: "a stationary point",
                  False: "not a stationary point"}[
                      d["relaxation"]["converged"]]
    assert f"-- {stationary} at this level of theory)" in finish, finish
    assert "| the run: fc done |" in finish, finish
    _common_assertions(d, optical=False)
    assert d["thermo"]["temperature_K"] == 350.0, d["thermo"]["temperature_K"]
    assert d["thermo"]["pressure_atm"] is None
    assert 350.0 in d["thermo"]["grid"]["temperatures_K"]
    assert d["relaxation"]["already_relaxed"] is True
    # nothing relaxed: the force-constant call itself never does
    assert d["phase_relaxation"] == "not requested"
    assert d["relaxation"]["enabled"] is False
    # at the geometry as given, in the user order, placed in its box as the deck
    # states it -- design + applied_offset (`model/structure-periodicity.md`
    # § 6.0) -- and judged off a stationary point by the template's 0.01 eV/A
    from molbuilder.deck_record import extract_engine_offset
    from molbuilder.runs import run_of
    deck = run_of(ticked.call("freq", FC)).deck.read_text()
    placed = (np.asarray(CO2["positions"], dtype=float)
              + extract_engine_offset(deck)["applied_offset"])
    assert np.allclose(d["structure"]["positions"], placed, atol=1e-6)
    assert d["relaxation"]["converged"] is False
    assert d["relaxation"]["max_force_eh_bohr"] \
        > 0.01 / HARTREE_BOHR_EV_ANGSTROM_ASE


#: named: a hidden Plotly node resized on load logs this, and nothing else
#: may.
_PLOTLY_HIDDEN_RESIZE = "Resize must be passed a displayed plot div"


def _shown(page, selector):
    """Whether ``selector`` takes up space on the page -- its computed
    display, up its ancestors, never its attribute."""
    return page.evaluate(
        "(sel) => { const e = document.querySelector(sel);"
        "  return !!e && e.getClientRects().length > 0; }", selector)


def test_the_results_tab_shows_a_siesta_vibrations_free_atoms_infrared(
        ticked, page, monkeypatch):
    """**The ticked road's result, looked at** (`web/spectra.md` § 2.1,
    § 3.2), opened the way a person opens it, through the tab's dropdown:

    * the free atoms' infrared shown when it opens -- the oxygens, the
      carbon held -- along z by Hirshfeld charges: a spectrum, the column
      titled with its note's mark, a number in every cell, its note under the table
      naming the free atoms, the light and the charges, the run summary
      saying the same; no choice of atoms anywhere;
    * the bar above the chart holding *Light along* and *Charges*, no Raman
      boxes -- the run asked for none -- and *Update* disabled;
    * a change in the bar asks nothing: the line says it waits, the column
      and its note stay as they were; *Update* asks the load door under the
      page busy fence, and the view, its note and the summary follow;
    * the modes table without the columns the route cannot compute, and the
      run summary saying the Raman was *not requested* and the probe *not
      computed on this route* (§ 9b.3); no phase dot for Raman or the probe
      (§ 3);
    * the electronic-structure tab naming the planned projected density of
      states, never PySCF's ``es_mode_selection`` (§ 9b.3);
    * the thermochemistry labelled as vibrational contributions, no
      pressure, its bars the file's own numbers -- summing to ``F_vib`` (§ 3;
      `engines/vibration.md` § 4.7).

    Every hiding is read from the COMPUTED style, never the attribute or the
    class: a ``display`` rule of the element's own beats ``[hidden]``.  The
    silent failure: an infrared of another set of atoms than the free ones,
    a bar change applied without Update, or Update asking outside the fence.
    """
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.runfiles import ROLE_SPECTRA
    from molbuilder.runs import on_label_at
    from support.live_server import serve
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(ticked.tree))
    d = _the_result(ticked)
    attempt = ticked.attempt("freq")
    result = on_label_at(attempt, ROLE_SPECTRA)

    def summary():
        return page.evaluate("""() => {
            const dl = document.getElementById("results-summary-list");
            const out = {};
            dl.querySelectorAll("dt").forEach(dt => {
                out[dt.textContent] = dt.nextElementSibling.textContent; });
            return out; }""")

    def cells():
        return page.eval_on_selector_all(
            "#modes-tbody td.ir-col", "ts => ts.map(t => t.textContent)")

    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: (errors.append(m.text)
                                  if m.type == "error" else None))
    with serve() as base:
        page.add_init_script(
            "try { sessionStorage.setItem('molbuilder.current_dir', "
            f"{json.dumps(str(attempt))}); }} catch (_) {{}}")
        page.goto(f"{base}/results")
        page.wait_for_function(
            "(want) => [...document.querySelectorAll("
            "  '#results-file-picker-select option')]"
            "  .some(o => o.value === want)", arg=str(result), timeout=20000)
        page.select_option("#results-file-picker-select", value=str(result))
        page.wait_for_selector("#modes-tbody tr", state="attached",
                               timeout=30000)

        # THE FREE ATOMS' INFRARED ON OPEN (§ 3.2): a spectrum, the column
        # titled short with its note, a number in every cell, the summary
        # naming the view -- and no choice of atoms.
        page.wait_for_function(
            "() => document.getElementById('ir-col-head').textContent"
            "  === 'IR* (km/mol)'", timeout=20000)
        assert page.inner_text("#spectrum-heading") == "Spectrum"
        assert _shown(page, "#broadening-fwhm")
        assert not _shown(page, "#spectrum-absent")
        assert page.locator("#modes-tbody tr").count() == len(d["modes"])
        along_z = cells()
        assert len(along_z) == len(d["modes"]) and "—" not in along_z, \
            along_z
        note = page.inner_text("#note-ir")
        assert note == (
            "* IR — how strongly this vibration absorbs infrared light "
            "polarized along z. It is the square of how much the free atoms' "
            "(2) dipole along z changes as the vibration moves them, the "
            "dipole built from each free atom's Hirshfeld charge. Held atoms "
            "are not counted. Zero means light along z does not see this "
            "vibration."), note
        assert page.inner_text("#charge-col-head") == \
            "Charge moved† (e/(√amu·Å))"
        assert page.inner_text("#note-charge").startswith(
            "† Charge moved — how much electric charge the free atoms gain "
            "or lose as this vibration moves them, per unit of vibration "
            "(Hirshfeld charges). Positive: they lose electrons. Negative: "
            "they gain electrons."), \
            page.inner_text("#note-charge")
        assert _shown(page, "#ir-explain")
        assert summary()["IR intensities"] == (
            "from the run's charges — the free atoms (2), along z, "
            "Hirshfeld"), summary()["IR intensities"]
        assert page.locator("#ir-atoms").count() == 0

        # THE BAR (§ 2.1): the light and the charges, no Raman -- the run
        # asked for none -- and nothing to update.
        assert _shown(page, "#view-bar")
        assert _shown(page, "#ir-direction") and _shown(page, "#ir-partition")
        assert not _shown(page, "#raman-controls")
        assert not _shown(page, "#ir-absent")
        assert page.is_disabled("#view-update")

        # A CHANGE ASKS NOTHING UNTIL UPDATE: the line says it waits, the
        # view on screen stays the one its note names.
        page.select_option("#ir-direction", value="x")
        assert page.inner_text("#view-status") == \
            "changed — Update to show it"
        assert not page.is_disabled("#view-update")
        assert page.inner_text("#note-ir").startswith(
            "* IR — how strongly this vibration absorbs infrared light "
            "polarized along z."), \
            page.inner_text("#note-ir")
        assert cells() == along_z

        # UPDATE, UNDER THE FENCE (`ui-contract.md` § 10): each claim's
        # reason the bar's.
        page.evaluate("""() => {
            const fence = window.molbuilder.pageBusy, claim = fence.claim;
            window.__claims = [];
            fence.claim = function (reason, cancelers) {
                window.__claims.push(reason);
                return claim.call(fence, reason, cancelers); }; }""")
        # HELD, THEN CANCELLED (§ 2.1): while Update's ask is out the window
        # is covered; Cancel aborts it and nothing is committed -- the view
        # stays along z and the bar still reads *changed*.
        held = []

        def hold(route):
            held.append(route)
        page.route("**/api/spectra/load", hold)
        page.click("#view-update")
        page.wait_for_function(
            "() => window.molbuilder.pageBusy.isClaimed()", timeout=20000)
        assert _shown(page, ".page-busy-cover")
        page.click(".page-busy-cancel")
        page.wait_for_function(
            "() => !window.molbuilder.pageBusy.isClaimed()", timeout=20000)
        page.unroute("**/api/spectra/load")
        assert len(held) == 1, held
        assert page.inner_text("#note-ir").startswith(
            "* IR — how strongly this vibration absorbs infrared light "
            "polarized along z."), \
            page.inner_text("#note-ir")
        assert cells() == along_z
        # the bar drawn again by a touch of another choice, there and back:
        # still x against the z applied
        page.select_option("#ir-partition", value="voronoi")
        page.select_option("#ir-partition", value="hirshfeld")
        assert page.inner_text("#view-status") == \
            "changed — Update to show it"
        assert not page.is_disabled("#view-update")
        # ...AND LET THROUGH: the view, its note and the summary along x.
        page.click("#view-update")
        page.wait_for_function(
            "() => document.getElementById('note-ir').textContent"
            "  .startsWith('* IR — how strongly this vibration absorbs '"
            "  + 'infrared light polarized along x.')",
            timeout=20000)
        assert page.evaluate("() => window.__claims") == [
            "Updating the spectrum…"] * 2
        assert not page.evaluate("() => window.molbuilder.pageBusy"
                                 ".isClaimed()")
        assert summary()["IR intensities"] == (
            "from the run's charges — the free atoms (2), along x, "
            "Hirshfeld"), summary()["IR intensities"]
        along_x = cells()
        assert "—" not in along_x and along_x != along_z, along_x
        assert page.is_disabled("#view-update")
        assert page.inner_text("#view-status") == ""

        # THE TABLE AND THE SUMMARY BY ROUTE (§ 9b.3): what SIESTA cannot
        # compute has no column, and the summary says so for each.
        for col in ("raman_activity_a4_amu", "has_es", "gap_eq_ev"):
            assert not _shown(page, f'#modes-table th[data-col="{col}"]'), col
        assert _shown(page, '#modes-table th[data-col="frequency_cm1"]')
        assert _shown(page, '#modes-table th[data-col="ir_intensity_km_mol"]')
        assert _shown(page, '#modes-table th[data-col="charge_moved"]')
        said = summary()
        assert said["Raman"] == "not requested", said["Raman"]
        assert said["Per-mode orbital energies"].startswith(
            "not computed on this route"), said["Per-mode orbital energies"]

        # A DOT FOR EACH PHASE THE RUN HAS, BY ROUTE (§ 3;
        # `engines/vibration.md` § 5.11 § 8): no Raman -- no optical calls
        # were asked -- and no probe; the infrared complete, its charges
        # stored (`ir_route` `charges`).
        assert _shown(page, '.spectra-phase-dot[data-phase="frequencies"]')
        assert not _shown(page, '.spectra-phase-dot[data-phase="raman"]')
        assert not _shown(page, '.spectra-phase-dot[data-phase="es"]')
        assert _shown(page, '.spectra-phase-dot[data-phase="ir"]')
        assert "spectra-phase-complete" in page.get_attribute(
            '.spectra-phase-dot[data-phase="ir"]', "class")

        # THE ELECTRONS' RESPONSE: planned, and named as what it will be.
        page.click("#mode-tabbtn-es")
        es = page.inner_text("#es-bar-diagram")
        assert "projected density of states" in es, es
        assert "es_mode_selection" not in es and "Mode selection" not in es, es

        # THE THERMOCHEMISTRY BY ITS REGIME (§ 3): vibrational
        # contributions, and no pressure -- none entered them.
        assert d["thermo"]["regime"] == "vibrational-only"
        page.click("#mode-tabbtn-thermo")
        note = page.inner_text("#thermo-note")
        assert "vibrational contributions only" in note, note
        assert "F_vib" in note and "S_vib" in note, note
        assert "P =" not in note and "atm" not in note, note
        # ...and its bars are the file's: SIESTA carries no electronic
        # energy, so the reference is zero, the parts add up to the last
        # bar and the last bar is the file's own F_vib (§ 3).
        from molbuilder import constants as C
        kcal = C.HARTREE_EV / C.KCAL_MOL_EV
        bars = page.wait_for_function(
            "() => { const d = document.getElementById('thermo-decomp');"
            "  return d && d.data && d.data[0]"
            "  ? {x: d.data[0].x, y: d.data[0].y} : null; }",
            timeout=20000).json_value()
        assert bars["x"] == ["ZPE", "U_vib", "−T·S_vib", "F_vib"], bars["x"]
        y = bars["y"]
        assert d["equilibrium"]["scf_energy_eh"] is None
        assert sum(y[:-1]) == pytest.approx(y[-1], abs=1e-9), y
        assert y[-1] == pytest.approx(d["thermo"]["g_eh"] * kcal, abs=1e-6)
        assert y[0] == pytest.approx(d["thermo"]["zpe_eh"] * kcal, abs=1e-6)

    unexpected = [e for e in errors if _PLOTLY_HIDDEN_RESIZE not in e]
    assert unexpected == [], f"the page reported JS errors: {unexpected}"


def test_the_ladder_names_each_call_of_the_run_in_its_words(
        real_co2_vibration, page, monkeypatch):
    """**The calculation's ladder, looked at** (`web/results.md` § 2.4, § 2.5
    *one wording*; `engines/vibration.md` § 5.11 § 7): opened at the
    calculation's root, the Results tab lists every call of the shared run
    under its stage, each row named by its run and by the call in the page's
    words -- *the relaxation*, *the force constants*, *the equilibrium*,
    each picked mode's two sides at its frequency
    (`lib/results/point-words.js`).

    THE SILENT FAILURE: every call's row reading ``run-0`` alone -- eight
    rows of one name, a person unable to tell the relaxation from an
    optical call without opening each.

    MUTATION THIS MUST FAIL AGAINST: the ladder naming a row from its
    voltage alone (a transport point's words only)."""
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.siesta.vibration_phases import FC, RELAX, calls_of
    from support.live_server import serve
    run = real_co2_vibration
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(run.tree))

    # WHAT EACH ROW IS NAMED BY, the contract's words for each call the
    # run's index lists, read by its one reader.
    def said(c):
        if c.phase == RELAX:
            return "the relaxation"
        if c.phase == FC:
            return "the force constants"
        if c.mode is None:
            return "the equilibrium"
        return (f"mode {c.mode} ({round(c.frequency_cm1)} cm\u207b\u00b9), "
                + ("+\u03b4Q" if c.side > 0 else "\u2212\u03b4Q"))
    calls = calls_of(_index_of(run))
    assert {c.phase for c in calls} >= {RELAX, FC, "optical"}, calls
    want = sorted(f"run-0 \u00b7 {said(c)}" for c in calls)

    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    with serve() as base:
        page.add_init_script(
            "try { sessionStorage.setItem('molbuilder.current_dir', "
            f"{json.dumps(str(run.bundle))}); }} catch (_) {{}}")
        page.goto(f"{base}/results")
        page.locator(".results-ladder").wait_for(state="visible",
                                                  timeout=30000)
        page.locator(".results-ladder-table tr.is-run").first.wait_for(
            timeout=30000)
        names = page.eval_on_selector_all(
            ".results-ladder-table tr.is-run",
            "rows => rows.map(r => r.children[1].textContent)")
    assert sorted(names) == want, (names, want)
    assert errors == [], errors


def test_the_page_asks_the_raman_at_a_laser_and_exports_it(
        real_co2_vibration, page, monkeypatch):
    """**The shared run's Raman, on its page** (`web/spectra.md` § 3.3),
    opened as a person opens it, through the Results tab's dropdown:

    * before a laser is applied, no Raman number -- the Raman column titled
      *Raman strength* with a dash in every cell, the bar's line asking for
      the laser, the run summary saying the same;
    * the laser typed, 633 nm, and *Update* (§ 2.1): the column titled
      with its note's mark, the note naming what was asked (the light's
      direction, the laser, the width -- the run's own), a number for each
      picked mode and a dash for the others, the static strength beside it,
      the bar's line saying the run's own width, the run summary naming the
      route, the laser and the width;
    * the Raman dot complete -- the route has the phase;
    * Export as NetCDF hands over one zip holding the run's two data files
      and the view at that laser.

    The silent failure: a page showing a number before the laser is named,
    titling SIESTA's strength as PySCF's activity, or exporting another view
    than it shows."""
    import io
    import zipfile
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.runfiles import ROLE_OPTICAL, ROLE_SPECTRA
    from molbuilder.runs import on_label_at
    from molbuilder.task import of as task_of
    from support.live_server import serve
    run = real_co2_vibration
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(run.tree))
    d = _the_result(run)
    attempt = run.attempt("freq")
    result = on_label_at(attempt, ROLE_SPECTRA)
    optical = on_label_at(attempt, ROLE_OPTICAL)
    label = task_of(run.bundle).label

    def column(selector):
        return page.eval_on_selector_all(
            f"#modes-tbody td{selector}", "ts => ts.map(t => t.textContent)")

    def summary(line):
        return page.evaluate("""(line) => {
            const dt = [...document.querySelectorAll(
                "#results-summary-list dt")].find(x => x.textContent === line);
            return dt ? dt.nextElementSibling.textContent : null; }""", line)

    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: (errors.append(m.text)
                                  if m.type == "error" else None))
    with serve() as base:
        page.add_init_script(
            "try { sessionStorage.setItem('molbuilder.current_dir', "
            f"{json.dumps(str(attempt))}); }} catch (_) {{}}")
        page.goto(f"{base}/results")
        page.wait_for_function(
            "(want) => [...document.querySelectorAll("
            "  '#results-file-picker-select option')]"
            "  .some(o => o.value === want)", arg=str(result), timeout=20000)
        page.select_option("#results-file-picker-select", value=str(result))
        page.wait_for_selector("#modes-tbody tr", state="attached",
                               timeout=30000)

        # NO NUMBER BEFORE THE LASER IS NAMED; the bar's line asks for it.
        assert _shown(page, "#view-bar")
        assert _shown(page, "#raman-laser") and _shown(page, "#raman-broaden")
        assert page.inner_text("#raman-col-head") == "Raman strength (Å⁴/amu)"
        assert set(column(".raman-col:not(.raman-static-col)")) == {"—"}
        assert not _shown(page, "#raman-static-col-head")
        assert not _shown(page, "#note-raman")
        assert page.inner_text("#view-status") == (
            "Raman: type the laser's wavelength — the strength depends on it")
        assert summary("Raman").endswith(
            "type the laser's wavelength — the strength depends on it"), \
            summary("Raman")
        assert _shown(page, '.spectra-phase-dot[data-phase="raman"]')
        assert "spectra-phase-complete" in page.get_attribute(
            '.spectra-phase-dot[data-phase="raman"]', "class")

        # THE LASER TYPED, then Update (§ 2.1): the door answers, the page
        # draws its answer.
        page.fill("#raman-laser", str(int(_LASER_NM)))
        assert page.inner_text("#view-status").startswith(
            "changed — Update to show it"), page.inner_text("#view-status")
        page.click("#view-update")
        page.wait_for_function(
            "() => document.getElementById('raman-col-head').textContent"
            "  === 'Raman strength‡ (Å⁴/amu)'", timeout=20000)
        # THE RUN'S OWN WIDTH, said on the bar's line, names both notes.
        said = page.inner_text("#view-status")
        head = "Raman: the run's own broadening: "
        assert said.startswith(head) and said.endswith(" eV"), said
        width = said[len(head):-len(" eV")]
        assert page.inner_text("#raman-static-col-head") == \
            "Raman strength, static§ (Å⁴/amu)"
        assert page.inner_text("#note-raman") == (
            "‡ Raman strength — how strongly this vibration scatters laser "
            "light of 633 nm polarized along z. It is the square of how much "
            "the structure's polarizability along z changes as the vibration "
            "moves it, from SIESTA's optical calculation with the atoms "
            "pushed slightly each way, the absorption spectrum smoothed by "
            f"{width} eV."), page.inner_text("#note-raman")
        assert page.inner_text("#note-raman-static") == (
            "§ Raman strength, static — the same, in the limit of zero light "
            "frequency, so it does not depend on the laser (absorption "
            f"spectrum smoothed by {width} eV). When this limit is "
            "meaningful: the method, § 3.1."), \
            page.inner_text("#note-raman-static")
        laser = column(".raman-col:not(.raman-static-col)")
        static = column(".raman-static-col")
        assert _shown(page, "#raman-static-col-head")
        picked = [m["index_1based"] in _PICKED_MODES for m in d["modes"]]
        assert [c != "—" for c in laser] == picked, laser
        assert [c != "—" for c in static] == picked, static
        assert summary("Raman").startswith(
            "SIESTA's optical calculation at ±δQ along each picked mode"), \
            summary("Raman")
        assert "; at 633 nm, broadened to " in summary("Raman"), \
            summary("Raman")

        # EXPORT: the run's data files and the view the page shows.
        with page.expect_download() as caught:
            page.click("#spectra-export-netcdf")
        download = caught.value
        assert download.suggested_filename == \
            f"{label}.vibration-netcdf.zip", download.suggested_filename
        folder = f"{label}.vibration-netcdf"
        got = Path(download.path()).parent / "unzipped"
        with zipfile.ZipFile(io.BytesIO(Path(download.path()).read_bytes())) \
                as z:
            names = sorted(z.namelist())
            z.extract(f"{folder}/{label}.raman.nc", got)
        assert names == sorted(f"{folder}/{n}" for n in (
            f"{label}.spectra.nc", optical.name, f"{label}.raman.nc")), names
        # ...THE VIEW ON SCREEN, read back through its reader: the laser
        # applied, the run's own width.
        from molbuilder.postprocess import read_raman_data
        data = read_raman_data(got / folder / f"{label}.raman.nc")
        assert data["laser_nm"] == _LASER_NM, data["laser_nm"]
        assert data["broaden_ev"] == pytest.approx(float(width), rel=1e-5), \
            (data["broaden_ev"], width)

    unexpected = [e for e in errors if _PLOTLY_HIDDEN_RESIZE not in e]
    assert unexpected == [], f"the page reported JS errors: {unexpected}"


def test_a_displacement_sweep_compares_every_stages_force_constants(unticked):
    """A displacement sweep (`engines/vibration.md` § 5.9): two stages --
    `freq` at the template's 0.04 Bohr and `freq_half` at 0.02 -- each one
    run in phases, each at a relaxed geometry; each stage's job writes its
    own spectrum; and `summarize task` compares them into
    `<label>.fc-sweep.json` at the root, naming each stage's files rather
    than copying them (I25), each stage's nudged atoms in SIESTA's
    numbering.  Before `freq_half` is launched the record lists it as
    pending, in its attempt's own words.

    MUTATION THIS MUST FAIL AGAINST: the second stage's record step not
    writing its own result at its run's root -- the summary then lists it
    failed, and compares nothing.
    """
    from molbuilder.siesta.vibration_phases import FC
    bundle, early, rec, rec2 = (unticked.bundle, unticked.early,
                                unticked.record, unticked.flagged)
    assert [s["name"] for s in early["stages"]] == ["freq"], early["stages"]
    (waiting,) = early["pending"]
    assert (waiting["stage"], waiting["state"]) == ("freq_half", "pending"), \
        waiting
    assert early["failed"] == []

    half = _deck_of(unticked.call("freq_half", FC))
    assert _said(half, "FC.Displacement")[:1] == ["0.02"]
    a1, a2 = (_the_result(unticked, "freq"),
              _the_result(unticked, "freq_half"))
    for d in (a1, a2):
        assert d["phase_relaxation"] == "complete" and \
            d["relaxation"]["converged"] is True

    assert rec["schema"] == "molbuilder/fc-displacement-sweep@2"
    assert rec["pending"] == [] and rec["failed"] == []  # every stage has one
    # the Results tab offers only what the parse registry reads
    # (`model/parse.md` § 5.5): the record has its own reader
    from molbuilder.parse import detect
    kind = detect(str(bundle / "CO2.fc-sweep.json"))
    assert kind.name == "fc-sweep-json", kind
    assert kind.parse(bundle / "CO2.fc-sweep.json").schema == "fc-sweep/v1"
    assert [s["name"] for s in rec["stages"]] == ["freq", "freq_half"]
    # I25: each stage's files are NAMED where its run wrote them, and are its
    # own -- the record carries paths, not copies.
    for s in rec["stages"]:
        for key in ("spectrum", "fc_file"):
            assert (bundle / s[key]).is_file(), (key, s)
    for s in rec["stages"]:
        where = unticked.attempt(s["name"]).relative_to(bundle)
        assert Path(s["spectrum"]).parent == where, (s["spectrum"], where)
        assert Path(s["fc_file"]).parent == where / FC, (s["fc_file"], where)
    assert not list(bundle.glob("*.FC")) and not list(
        bundle.glob("*.spectra.json")), "the summary copied a stage's files"
    # the oxygens nudged, SIESTA's atoms 2 and 3 in the held-first deck
    assert [s["fc_range_engine_1based"] for s in rec["stages"]] == [[2, 3]] * 2
    # the displacement each stage USED, from its own spectrum
    d1, d2 = (s["fc_displacement_ang"] for s in rec["stages"])
    assert abs(d2 / d1 - 0.5) < 1e-6, (d1, d2)
    assert rec["stages"][1]["varies"] == {"fc_displacement": 0.02}
    assert rec["stages"][1]["varies_units"] == {"fc_displacement": "Bohr"}
    # every mode at both displacements; the stretches matched by shape -- a
    # degenerate bend pair's shapes are any two in their plane, so theirs
    # may mix between the stages
    assert len(rec["modes"]) == 4
    for m in rec["modes"]:
        assert set(m["frequency_cm1"]) == {"freq", "freq_half"}
        assert m["spread_cm1"] == pytest.approx(
            abs(m["frequency_cm1"]["freq"] - m["frequency_cm1"]["freq_half"]))
        assert m["flagged"] is None
    for m in rec["modes"][2:]:
        assert min(m["overlap"].values()) > 0.999, m
    assert rec["tolerance_cm1"] is None
    (fc,) = rec["force_constants"]
    assert fc["against"] == "freq" and fc["max_abs_change_ev_ang2"] > 0.0
    assert all(m["flagged"] is True for m in rec2["modes"])
    assert rec2["tolerance_cm1"] == 1e-6


@pytest.fixture(scope="module")
def stopped(real_co2_vibration, tmp_path_factory):
    """THE PERSON STOPS THE RUN ONCE ITS FORCE CONSTANTS AND ITS FIRST
    OPTICAL CALL ENDED (`engines/vibration.md` § 5.11 § 6): a copy of the
    shared run's calculation (`support.real_runs.copy_of`), `freq` launched
    again cold in the background -- a launch's default (`job-system.md`
    § 6.0) -- so run-1 relaxes and takes its force constants anew; once
    `jobset status` reads an optical call of run-1 done and run-1's index
    lists the next one started (an optical call takes some 25 s here), the
    run stopped with the line that launch printed and let end; the run as
    the stop left it kept -- its status (``stopped``) and its index
    (``stopped_index``).  Each road launched after the stop is a copy of
    this one."""
    from molbuilder.siesta.vibration_phases import OPTICAL
    from support.real_runs import (_road_at, _stopped_by_its_line, _taken,
                                   _until, copy_of, stage_status)

    def stop(c):
        with _road_at(c.tree.parent, tmp_path_factory):
            run1 = c.attempt("freq", 1).name
            launched = _taken(c, "launch freq cold", "launch", "task",
                              "--stage", "freq", "--mode", "direct", "--yes",
                              "--cold")

            def one_done_and_the_next_started():
                # RUN-1'S CALLS, by the done rule launch takes them over by
                # -- the status door -- and the next call's start, its index.
                s = stage_status(c.bundle, "freq")
                if s.attempt != run1:
                    return False          # run-1 not opened yet
                done = {p["call"] for p in s.points
                        if p["phase"] == OPTICAL and p["done"]}
                if not done:
                    return False
                index = _index_of(c, "freq", 1)
                return any(r["phase"] == OPTICAL and r.get("started")
                           and r["name"] not in done for r in index["calls"])
            _until(one_done_and_the_next_started, within=900,
                   what="an optical call of run-1 done, the next started")
            _stopped_by_its_line(c, "stop as its launch said", launched)
            c.stopped = stage_status(c.bundle, "freq")
            c.stopped_index = _index_of(c, "freq", 1)
    return copy_of(real_co2_vibration, "stopped", stop)


@pytest.fixture(scope="module")
def stopped_after_fc(stopped, tmp_path_factory):
    """THE STOPPED RUN LAUNCHED AGAIN (:func:`stopped`): a copy of it, `freq`
    launched again warm, in the foreground -- run-2 -- the stop's status and
    index kept beside it."""
    from support.real_runs import _road_at, _typed, copy_of

    def warm(c):
        c.stopped, c.stopped_index = stopped.stopped, stopped.stopped_index
        with _road_at(c.tree.parent, tmp_path_factory):
            _typed(c, "launch freq warm", "launch", "task", "--stage",
                   "freq", "--mode", "direct", "--foreground", "--yes")
    return copy_of(stopped, "warm", warm)


def test_launched_again_warm_after_a_stop_runs_only_the_optical_calls(
        stopped_after_fc):
    """LAUNCHED AGAIN WARM after the person stopped the run once its force
    constants and its first optical call ended (`engines/vibration.md`
    § 5.11 § 6): run-2 takes over, from run-1 -- the run that computed
    them -- the relaxation, the force constants and every optical call
    run-1 finished, each call's `.continued-from` naming run-1's, as
    `jobset status` reads it, and its index entry naming it too, its start
    and end run-1's, so none ran again; it runs the optical calls run-1
    left unfinished, from the coordinates run-1's force constants planned,
    and nothing else -- the launch says so in its words; `status` counts
    the calls taken over; the run reports as one run (`start` once,
    `finish` once); and its result and data file are whole
    (:func:`_the_optical_record`), each frame computed where it was --
    run-1's for a frame taken over -- with no force-constant call of its
    own, the record of each optical call stands on the one it took over.

    MUTATIONS THIS MUST FAIL AGAINST: launch not taking a done call over
    (the relaxation runs again: its start is run-2's); a done optical call
    walked again; an optical call's record step reading what only a run
    that took its own force constants writes (the run then fails); the
    take-over naming the run it copied from rather than the one that
    computed the call; a taken-over frame's data file row naming its call
    folder rather than its run."""
    from molbuilder.siesta.vibration_phases import FC, OPTICAL, RELAX
    run = stopped_after_fc
    # THE PREMISE, by the status door launch decides by: the stop landed
    # once the force constants and at least one optical call were done,
    # and before every optical call was.
    ended = {p["call"]: p for p in run.stopped.points}
    assert ended[RELAX]["done"] and ended[FC]["done"], run.stopped.points
    finished = {p["call"] for p in run.stopped.points
                if p["phase"] == OPTICAL and p["done"]}
    assert 0 < len(finished) < len(_OPTICAL_CALLS), run.stopped.points
    before = {r["name"]: r for r in run.stopped_index["calls"]}
    said = run.said["launch freq warm"]
    assert said.exit_code == 0, said.output
    assert ("takes over the relaxation, the force constants, frame 0, the "
            "equilibrium" in said.output
            and f"done in run-1; runs the optical calls below run-2/optical/ "
                f"({len(_OPTICAL_CALLS) - len(finished)} to run again)"
            in said.output), \
        said.output
    status, _folder = _the_run(run, n=2)
    n = len(_OPTICAL_CALLS)
    assert status.detail == (f"relax done · fc done · optical {n} of {n} "
                             f"({len(finished)} taken over)"), status.detail
    run1 = run.attempt("freq", 1).relative_to(run.bundle).as_posix()
    run2 = run.attempt("freq", 2).relative_to(run.bundle).as_posix()
    index = {r["name"]: r for r in _index_of(run, n=2)["calls"]}
    for p in status.points:
        name = p["call"]
        taken = p["phase"] in (RELAX, FC) or name in finished
        rel = f"{run1}/{p['phase']}" + (f"/{name}" if p["phase"] == OPTICAL
                                         else "")
        assert p["started_from"] == (rel if taken else None), p
        assert index[name].get("taken_from") == (rel if taken else None), \
            index[name]
        if taken:
            assert (index[name]["started"], index[name]["ended"]) == (
                before[name]["started"], before[name]["ended"]), \
                (index[name], before[name])
    # THE CALLS TAKEN OVER LEAVE NO RECORD STEP OF THIS RUN, so no
    # milestone; its finish counts the run as the status door does.
    assert notified(run, "freq", n=2) == ["start", "finish"]
    finish, = told(run, "finish", "freq", n=2)
    assert f"| the run: {status.detail} |" in finish, finish
    d, _data, _calls = _the_optical_record(
        run, n=2, computed_in={name: (run1 if name in finished else run2)
                               for name in _OPTICAL_CALLS})
    _common_assertions(d, optical=True)


@pytest.fixture(scope="module")
def failed_parts(stopped, tmp_path_factory):
    """THE STOPPED RUN LAUNCHED AGAIN UNDER A WALL ALREADY PASSED
    (:func:`stopped`): a copy of it, its run card given a wall of 20 s
    (`execution.time`, `job-request.md`), prepared again for that job
    request -- what it computes kept (`job-system.md` § 4, *launched*) --
    and `freq` launched again warm, in the foreground: run-2 takes over
    what run-1 finished, and each optical call it runs is told what is left
    of the run's wall, so SIESTA stops at its first check
    (`running-a-job.md` § 3.6; `siesta.md` § 8.9) -- every call it runs
    ending alike, not done."""
    from support.road import read_request, write_request
    from support.real_runs import _road_at, _taken, _typed, copy_of

    def walled(c):
        request = read_request(c.bundle)
        request["execution"] = {**request.get("execution", {}),
                                "time": "00:00:20"}
        write_request(c.bundle, request)
        with _road_at(c.tree.parent, tmp_path_factory):
            _taken(c, "prep freq for the wall", "prep", "task", "--group",
                   "1", "--target", "this")
            _typed(c, "launch freq warm", "launch", "task", "--stage",
                   "freq", "--mode", "direct", "--foreground", "--yes")
    return copy_of(stopped, "walled", walled)


def test_a_failed_part_is_said_at_once_and_once_per_reason(failed_parts):
    """`run-reports.md` § 2.9: a call that ends not done while the run has
    calls left to run is said at once -- the call in its words, why, and
    the run's count, which counts it failed -- and a later call failing for
    the same reason is not said again; the finish names the first call not
    done and how many are, its count the failed calls.
    On the real engine: every optical call of a run launched under a wall
    already passed stops at once, alike.

    THE SILENT FAILURES: a run's failed call never reported until its
    finish, hours later; a run whose calls all fail alike -- its wall
    reached, a file missing -- sending one message per call; and a finish
    naming one failed call of many, its count nowhere; a count that says
    "1 of 5" without saying whether the other 4 failed or wait."""
    from molbuilder.siesta.vibration_phases import OPTICAL, rows_of
    from molbuilder.task import of as task_of
    run = failed_parts
    rows = rows_of(_index_of(run, n=2), run.attempt("freq", 2),
                   task_of(run.bundle).label)
    optical = [r for r in rows if r["phase"] == OPTICAL]
    taken = [r for r in optical if r["started_from"]]
    ran = sorted((r for r in optical if not r["started_from"]),
                 key=lambda r: r["ended_at"])
    # THE PREMISE, by the one done rule: run-2 ran two calls or more, and
    # each ended not done for the one reason.
    assert len(ran) >= 2, rows
    assert not any(r["done"] for r in ran), ran
    assert len({r["why"] for r in ran}) == 1, ran
    sent = notified(run, "freq", n=2)
    assert sent[0] == "start" and sent[-1] == "finish", sent
    assert sent.count("part_failed") == 1, sent
    # ...the wall's warning, which a member monitor may send once before
    # its call stops, the only other word.
    assert set(sent) <= {"start", "wall_near", "part_failed", "finish"}, sent
    assert sent.count("wall_near") <= 1, sent
    first = ran[0]
    said, = told(run, "part_failed", "freq", n=2)
    assert said.startswith(
        f"running ({first['words']}: not done -- {first['why']}) | the "
        f"run: relax done · fc done · optical {len(taken)} of "
        f"{len(optical)} ({len(taken)} taken over, 1 failed) |"), said
    left = [r for r in rows if not r["done"]]
    finish, = told(run, "finish", "freq", n=2)
    assert finish.startswith(
        f"failed ({left[0]['words']}: not done -- {left[0]['why']}; "
        f"{len(left)} of {len(rows)} calls not done) | the run: relax done "
        f"· fc done · optical {len(taken)} of {len(optical)} "
        f"({len(taken)} taken over, {len(ran)} failed) |"), finish
