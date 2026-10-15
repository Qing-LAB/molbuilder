"""The vibration kind on SIESTA, through the whole described road.

``jobset init --engine siesta --calculation vibration`` → ``prep task`` →
``launch task --mode direct`` on this workstation, and nothing after it: the
road's CO2 (`support.road.CO2`), listed O O C with the carbon held, the
force-constant run nudging the two oxygens only, and the modes derived BY THE
JOB -- its finish, ``mb_vibration.pyz``, run by the wrapper after SIESTA with
the job's own python, which cannot import molbuilder -- through the one path
both engines share (`engines/vibration.md` § 5.5, I22, I23).  The carbon is
LAST in the input, so the held-first order the decks are written in is a
3-cycle (`support.road.CO2_HELD_FIRST`), and the result comes back in your
order through the order record its deck states (`model/overview.md` § 2.2a,
I5, I7).

WHAT THE PHYSICS SAYS OF IT.  The molecule lies along z.  Six free
coordinates; the carbon held on the axis leaves two whole-body motions -- the
turns about it, about x and about y -- so four modes (`science/normal-modes.md`
§ 3.1): a bend pair, degenerate by the molecule's symmetry, moving the
oxygens in the (x, y) plane, and two stretches along z.

MUTATIONS THIS MUST FAIL AGAINST: a wrapper that does not run the finish
(the launch then ends with no spectrum); a finish bundle missing a member
(the job's python cannot import it, and the job fails); a finish that reads
no order record from its deck (the result's oxygens then stand where the
carbon is).

The roads: one per state of the person's one box (`engines/vibration.md`
§ 2.2) -- unticked, the ladder `init` writes relaxes first (the shared
relaxation, `support.real_runs.relaxed_for_vibration`), and `freq` is written
at the geometry `relax` reached (§ 5.2a); ticked, `freq` alone measures at
the geometry as given -- then a displacement sweep (§ 5.9).  A flat
calculation's refusal of a sweep needs no engine: `test_siesta_vibration_deck`.

Needs the ``molbuilder-siesta`` env + conda hook on this machine; skipped
cleanly anywhere else.  Each force-constant run is thirteen single points of
CO2 on one rank, about a minute here, and the module launches five, with one
relaxation of its own.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from _road import conda_hook, env_available, set_template_value
from support.road import CO2, CO2_HELD_FIRST, group_words, ladder_edited

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


@pytest.fixture(scope="module")
def ladder(real_relaxed_for_vibration, tmp_path_factory):
    """The box unticked: the shared relaxation's calculation with its `freq`
    stage prepared and launched, `summarize` asked, and `freq` launched again
    (`support.real_runs.freq_launched_on`)."""
    from support.real_runs import freq_launched_on
    return freq_launched_on(real_relaxed_for_vibration, tmp_path_factory)


@pytest.fixture(scope="module")
def ticked(tmp_path_factory):
    """The box ticked, on the road: `freq` asked before `relax` has run; the
    ladder cut to `freq` alone and asked again unticked; then ticked, the
    template's temperature set to 350 K, prepared and launched -- each answer
    written down."""
    from support.real_runs import (RealRun, _on_the_road, _taken, _typed,
                                   co2_vibration)
    with _on_the_road(tmp_path_factory, "ticked") as tree:
        run = RealRun(tree=tree, bundle=co2_vibration(tree))
        _typed(run, "prep freq before relax", "prep", "task",
               *group_words(run.bundle, "freq"), "--target", "this")
        ladder_edited({"removed": ["relax"]}, run.bundle)
        _typed(run, "prep freq unticked", "prep", "task",
               *group_words(run.bundle, "freq"), "--target", "this")
        template = run.bundle / "CO2.template.toml"
        set_template_value(template, "already_relaxed", "true")
        set_template_value(template, "temperature_K", "350.0")
        _taken(run, "prep freq", "prep", "task",
               *group_words(run.bundle, "freq"), "--target", "this")
        _taken(run, "launch freq", "launch", "task", "--stage", "freq",
               "--mode", "direct", "--yes")
    return run


@pytest.fixture(scope="module")
def sweep(tmp_path_factory):
    """A displacement sweep on the road: `freq_half` at 0.02 Bohr beside
    `freq`, in a group of its own -- `freq` is launched before it is
    prepared, and a launched group is fixed (`engines/stages.md` § 2);
    `relax` and `freq` launched, `freq_half` prepared and the sweep
    summarized (``early``), `freq_half` launched and summarized
    (``record``), and summarized again under a tolerance (``flagged``)."""
    from support.real_runs import RealRun, _on_the_road, _taken, co2_vibration
    with _on_the_road(tmp_path_factory, "sweep") as tree:
        run = RealRun(tree=tree, bundle=co2_vibration(tree))
        ladder_edited({"added": [{"name": "freq_half",
                                  "overrides": {"fc_displacement": 0.02}}]},
                      run.bundle)
        record = run.bundle / "CO2.fc-sweep.json"
        for stage in ("relax", "freq"):
            _taken(run, f"prep {stage}", "prep", "task",
                   *group_words(run.bundle, stage), "--target", "this")
            _taken(run, f"launch {stage}", "launch", "task", "--stage",
                   stage, "--mode", "direct", "--yes")
        _taken(run, "prep freq_half", "prep", "task",
               *group_words(run.bundle, "freq_half"), "--target", "this")
        _taken(run, "summarize early", "summarize", "task")
        run.early = json.loads(record.read_text())
        _taken(run, "launch freq_half", "launch", "task", "--stage",
               "freq_half", "--mode", "direct", "--yes")
        _taken(run, "summarize", "summarize", "task")
        run.record = json.loads(record.read_text())
        _taken(run, "summarize with a tolerance", "summarize", "task",
               "--tolerance-cm1", "1e-6")
        run.flagged = json.loads(record.read_text())
    return run


def _deck_of(attempt):
    """The deck the run in ``attempt`` read, through the run door
    (`runs.run_of`), as fdf reads it (`parse.fdf.read_fdf`): ``(scalars,
    blocks, run)``."""
    from molbuilder.parse.fdf import read_fdf
    from molbuilder.runs import run_of
    run = run_of(attempt)
    return (*read_fdf(run.deck.read_text()), run)


def _said(deck, key):
    """What the deck (:func:`_deck_of`) sets ``key`` to, as fdf reads it."""
    from molbuilder.parse.fdf import fdf_key
    return deck[0].get(fdf_key(key), [])


def _the_result(attempt):
    """The spectrum the launch left in ``attempt`` -- written by the job
    itself, so its run finished only with it (`engines/vibration.md` § 5.5:
    the finish's failure is the job's) -- asked through the run door (the
    one status door, `runs.run_answer`; the run's result on its label,
    `Run.on_label`) and read by its reader (`sidecars.spectra
    .parse_spectra_json`, which refuses a derived value its inputs
    contradict), as it states itself."""
    from molbuilder.runs import run_answer, run_of
    from molbuilder.sidecars.spectra import parse_spectra_json
    said = run_answer(attempt)
    assert (said["state"], said["detail"]) == ("finished", "job_completed"), \
        said
    out = run_of(attempt).on_label(".spectra.json")
    assert out is not None and out.is_file(), (
        "the launch ended without the spectrum: the job did not finish its "
        "own calculation")
    return parse_spectra_json(str(out)).to_dict()


def _common_assertions(d):
    from molbuilder.spectra.results import SCHEMA_VERSION
    assert d["engine"] == "siesta" and d["schema_version"] == SCHEMA_VERSION
    # YOUR ORDER: the oxygens atoms 0 and 1, the held carbon atom 2.
    assert d["equilibrium"]["elements"] == CO2["elements"]
    assert d["free_atom_idxs"] == [0, 1] and d["frozen_atom_idxs"] == [2]
    assert d["hessian_scope"] == "free" and d["n_atoms_in_hessian"] == 2
    # the two turns about the held carbon, about x and about y; four modes
    # left
    assert d["removed_motions"]["count"] == 2
    _a_bend_pair_and_two_stretches(d["modes"])
    assert d["equilibrium"]["mo_energies_eh"] is None
    assert d["modes"][0]["raman_activity_a4_amu"] is None
    assert d["thermo"]["regime"] == "vibrational-only"
    # the frequencies are this route's whole answer; the strengths and the
    # probe were never asked of it (vibration.md § 4.9)
    assert d["phase_frequencies"] == "complete"
    assert (d["phase_raman"], d["phase_es"]) == ("not requested",) * 2
    assert "Head1997" in d["bibliography_keys"]
    # R5 on this route: the reference-step forces judged by the template's
    # own tolerance -- the kind's recommended 0.01 eV/A
    assert d["engine_metadata"]["reference_force_criterion_ev_ang"] == 0.01
    # the mass-calibrated displacement a transport step displaces along
    for m in d["modes"]:
        assert (m["zero_point_amplitude_amu12_ang"] ** 2
                * m["frequency_cm1"]) == pytest.approx(_ZERO_POINT,
                                                       rel=1e-3), m
    _states_its_masses_and_derived_values(d)
    # no strength is computed on this route, so no mode can be placed in a
    # channel (`engines/vibration.md` § 6.5)
    assert {m["activity_class"] for m in d["modes"]} == {"partial"}


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
    assert eq["masses_amu"] == [atomic_mass(e) for e in eq["elements"]], eq
    stated = np.asarray(eq["masses_amu"])[d["free_atom_idxs"]]
    for m in d["modes"]:
        assert all(k in m for k in DERIVED_MODE_KEYS), sorted(m)
        L = np.asarray(m["eigenvector_canonical"])
        assert abs(float((stated[:, None] * L * L).sum()) - 1.0) < 1e-6, m


def _relaxed_geometry(attempt):
    """Where the `relax` run in ``attempt`` ended, in your order -- the run
    door's one read of a run's geometry (`runs.steps_of`)."""
    from molbuilder.runs import openable, run_of, steps_of
    return np.asarray(steps_of(run_of(attempt), openable(attempt)[0])
                      .last.positions, dtype=float)


def test_unticked_the_ladder_relaxes_first_and_freq_measures_at_the_relaxed_geometry(
        ladder):
    """The box unticked (the default `init` writes): the ladder is `relax`
    then `freq`; after `relax`, `freq` is written at the geometry `relax`
    reached, in the held-first order its deck states, and the modes come out
    in your order with the reference forces within the template's own
    tolerance (vibration.md § 2.2, § 5.2a, § 5.5).  And a force-constant
    stage launched again starts over: its kind's rerun carries nothing of
    its own (`warm-files.toml` [vibration] `resumes`, `job-system.md` § 5.4,
    *A stage launched again*)."""
    from molbuilder.parse.fdf import fdf_key
    from molbuilder.runs import declared
    run = ladder
    task = json.loads((run.bundle / "task.json").read_text())
    assert [s["name"] for s in task["stages"]] == ["relax", "freq"]
    # the ordinary relaxation deck, from the same held-first copy: Broyden
    # to the kind's recommended 0.01 eV/A, the held carbon -- SIESTA's atom 1
    # after the sort -- constrained
    relax = _deck_of(run.attempt("relax"))
    assert _said(relax, "MD.TypeOfRun") == ["Broyden"]
    assert _said(relax, "MD.MaxForceTol")[:1] == ["0.01"]
    held = [["position", "1"]]
    assert relax[1][fdf_key("Geometry.Constraints")] == held
    # the force-constant deck: the oxygens SIESTA's atoms 2 and 3, the
    # carbon held; and the run states the held-first order record, read by
    # the run door's reader (I5)
    fc = _deck_of(run.attempt("freq"))
    assert (_said(fc, "MD.TypeOfRun"), _said(fc, "FC.First"),
            _said(fc, "FC.Last")) == (["FC"], ["2"], ["3"])
    assert fc[1][fdf_key("Geometry.Constraints")] == held
    assert declared(fc[2]).atom_order.to_dict() == CO2_HELD_FIRST
    attempt = run.attempt("freq")
    fc_file = fc[2].on_label(".FC")
    assert fc_file is not None and fc_file.is_file(), \
        "the force-constant run must leave its .FC"
    # THE LAUNCH ENDS WITH THE RESULT: the job's finish wrote it (§ 5.5).
    d = _the_result(attempt)
    # ...and summarize derives nothing: with one force-constant stage there
    # is no sweep to summarize (§ 5.9, I22).
    said = run.said["summarize"]
    assert said.exit_code != 0 and "a sweep compares two or more" in \
        said.output, said.output
    _common_assertions(d)
    assert d["relaxation"]["converged"] is True
    # the ladder relaxed first, and the artifact says so from the relax
    # stage's own record (vibration.md § 4.9)
    assert d["relaxation"]["already_relaxed"] is False
    assert d["phase_relaxation"] == "complete"
    assert d["relaxation"]["enabled"] is True and d["relaxation"]["n_steps"] >= 1
    # the geometry in the file is the one the force constants were taken
    # at -- where `relax` ended, in your order
    assert np.allclose(d["equilibrium"]["positions_ang"],
                       _relaxed_geometry(run.attempt("relax")), atol=1e-5)

    # LAUNCHED AGAIN, the force constants start over -- nothing handed on.
    assert ("launched again into run-1: it starts over -- nothing is handed "
            "on from a run of its own") in run.said["launch freq again"].output
    again = run.attempt("freq", 1)
    assert again.is_dir() and not (again / ".continued-from").exists()


def test_ticked_freq_alone_measures_at_the_geometry_as_given(ticked):
    """The box ticked: no relaxation, the force constants at the geometry as
    given, the reference forces judged by the template's tolerance -- the
    road's CO2, one C=O long, is no stationary point, and the result says so.
    Before it, the ladder's own order and the contradiction refused: `freq`
    before `relax` has run names the stage to run first (`continuation`,
    § 5.2a's table), and unticked with no relax stage in the ladder is
    neither state, so prep names both ways out (§ 5.2a).

    The thermochemistry is summed at the TEMPLATE'S temperature, which is
    one meaning on both engines (§ 3.1): set to 350 K here, it reaches the
    result through the deck's `vibration` block -- and a vibrational-only
    result records no pressure, since none enters it (§ 4.7).

    MUTATION THIS MUST FAIL AGAINST: prep leaving the temperature out of the
    block -- the finish then refuses the deck and the launch fails.
    """
    from molbuilder.constants import HARTREE_BOHR_EV_ANGSTROM_ASE
    said = ticked.said["prep freq before relax"]
    assert said.exit_code != 0 and "`freq` builds on `relax`" in said.output \
        and "Run it first" in said.output, said.output
    said = ticked.said["prep freq unticked"]
    assert said.exit_code != 0 and "already_relaxed = true" in said.output \
        and "`relax` stage" in said.output, said.output
    d = _the_result(ticked.attempt("freq"))
    _common_assertions(d)
    assert d["thermo"]["temperature_K"] == 350.0, d["thermo"]["temperature_K"]
    assert d["thermo"]["pressure_atm"] is None
    assert 350.0 in d["thermo"]["grid"]["temperatures_K"]
    assert d["relaxation"]["already_relaxed"] is True
    # nothing relaxed: the force-constant run itself never does
    assert d["phase_relaxation"] == "not requested"
    assert d["relaxation"]["enabled"] is False
    # at the geometry as given, in your order, placed in its box as the deck
    # states it -- design + applied_offset (`model/structure-periodicity.md`
    # § 6.0) -- and judged off a stationary point by the template's 0.01 eV/A
    from molbuilder.deck_record import extract_engine_offset
    from molbuilder.runs import run_of
    deck = run_of(ticked.attempt("freq")).deck.read_text()
    placed = (np.asarray(CO2["positions"], dtype=float)
              + extract_engine_offset(deck)["applied_offset"])
    assert np.allclose(d["equilibrium"]["positions_ang"], placed, atol=1e-6)
    assert d["relaxation"]["converged"] is False
    assert d["relaxation"]["max_force_eh_bohr"] \
        > 0.01 / HARTREE_BOHR_EV_ANGSTROM_ASE


def _deck_coordinates(deck):
    """The deck's own coordinate block (:func:`_deck_of`), as SIESTA reads
    it."""
    from molbuilder.parse.fdf import fdf_key
    rows = deck[1][fdf_key("AtomicCoordinatesAndAtomicSpecies")]
    return np.array([[float(x) for x in r[:3]] for r in rows])


def test_a_displacement_sweep_measures_every_stage_at_the_relaxed_geometry(
        sweep):
    """A displacement sweep (`engines/vibration.md` § 5.9): after `relax`,
    two force-constant stages -- `freq` at the template's 0.04 Bohr and
    `freq_half` at 0.02 -- BOTH measure at the relaxed geometry, whatever
    their names (I24); each stage's job writes its own spectrum; and
    `summarize task` compares them into `<label>.fc-sweep.json` at the root,
    naming each stage's files rather than copying them (I25), each stage's
    nudged atoms in SIESTA's numbering.  Before `freq_half` is launched the
    record lists it as pending, in its attempt's own words.

    MUTATION THIS MUST FAIL AGAINST: only the stage named `freq` taking the
    relaxed geometry -- `freq_half` then measures the unrelaxed input.
    """
    bundle, early, rec, rec2 = (sweep.bundle, sweep.early, sweep.record,
                                sweep.flagged)
    assert [s["name"] for s in early["stages"]] == ["freq"], early["stages"]
    (waiting,) = early["pending"]
    assert (waiting["stage"], waiting["state"]) == ("freq_half", "pending"), \
        waiting
    assert early["failed"] == []

    decks = {stage: _deck_of(sweep.attempt(stage))
             for stage in ("freq", "freq_half")}
    # I24: one geometry for every force-constant stage -- the relaxed one.
    assert np.allclose(_deck_coordinates(decks["freq"]),
                       _deck_coordinates(decks["freq_half"]), atol=1e-8), \
        "the second force-constant stage was not written at the relaxed geometry"
    assert _said(decks["freq_half"], "FC.Displacement")[:1] == ["0.02"]
    a1, a2 = (_the_result(sweep.attempt("freq")),
              _the_result(sweep.attempt("freq_half")))
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
        where = sweep.attempt(s["name"]).relative_to(bundle)
        assert Path(s["spectrum"]).parent == where, (s["spectrum"], where)
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
