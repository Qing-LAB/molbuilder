"""The vibration kind on SIESTA, through the whole described road.

``jobset init --engine siesta --calculation vibration`` → ``prep run`` →
``launch run --mode direct`` on this workstation, and nothing after it: a
two-atom molecule with one atom held, the force-constant run nudging the
free atom only, and the modes derived BY THE JOB -- its finish,
``mb_vibration.pyz``, run by the wrapper after SIESTA with the job's own
python, which cannot import molbuilder -- through the one path both engines
share (`engines/vibration.md` § 5.5, I22, I23).  The held atom is LAST in
the input, so the held-first sort really reorders the copy the deck is
written from and the answer has to come back through the recorded
permutation.

MUTATIONS THIS MUST FAIL AGAINST: a wrapper that does not run the finish
(the launch then ends with no spectrum); a finish bundle missing a member
(the job's python cannot import it, and the job fails).

The roads: one per state of the person's one box (`engines/vibration.md`
§ 2.2) -- unticked, the ladder `init` writes relaxes first, a `relax` stage
whose geometry the `freq` stage is written at (§ 5.2a), and the finish
judges the reference-step forces by the template's own tolerance; ticked,
`freq` alone measures at the geometry as given -- then a displacement sweep
(§ 5.9) and its refusals, and the finish's own failures: a finish that
fails, and one that cannot load, which stops the job before SIESTA.

Needs the ``molbuilder-siesta`` env + conda hook on this machine; skipped
cleanly anywhere else.  Each launch is a short Broyden relaxation or seven
single points of H2 -- about 15-30 s here -- and the module launches a dozen
(about five minutes in all).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from _road import conda_hook, env_available, env_bin

FIXTURES = Path(__file__).resolve().parent / "fixtures"


CONDA_SH = conda_hook()

pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(
        not (CONDA_SH.is_file()
             and env_available("molbuilder-siesta")),
        reason="needs the molbuilder-siesta env + a detectable conda hook"),
]


def _describe(tree, monkeypatch, positions, shape="hierarchical"):
    """The calculation the way a person makes it: the structure in the
    projects tree, the free atom FIRST and the held atom second (so the
    held-first sort really reorders), `init` on the SIESTA vibration kind."""
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec

    (tree / "P" / "structure").mkdir(parents=True)
    (tree / "pseudopotential").mkdir()
    shutil.copy(FIXTURES / "psml" / "H.psml", tree / "pseudopotential" / "H.psml")
    s = Structure(elements=["H", "H"], positions=np.asarray(positions, float),
                  cell=np.diag([10.0, 10.0, 10.0]),
                  axis_kind=("isolated",) * 3)
    s.frozen_atoms = [1]
    StructureCodec().write(s, tree / "P" / "structure" / "h2.xyz")
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tree))
    monkeypatch.chdir(tree.parent)
    # The wrapper looks the engine up BY NAME after activating the env --
    # the road itself.  The suite's stub toolchain (conftest) sits ahead of
    # the env's bin so that no test reaches the host's engine; this test
    # wants the env's REAL engine on that same road, so the env's own bin
    # goes in front of the stubs -- the binary the product finds when the
    # person launches, and never /usr/local/bin's.
    bin_ = env_bin("molbuilder-siesta")
    assert (bin_ / "siesta").is_file(), bin_
    monkeypatch.setenv("PATH", f"{bin_}{os.pathsep}{os.environ['PATH']}")

    r = CliRunner().invoke(jobset_group, [
        "init", "--structure", "P/structure/h2.xyz",
        "--bundle", "P/frequency/F", "--engine", "siesta",
        "--shape", shape, "--calculation", "vibration",
        "--name", "H2", "--psml-lib", "pseudopotential"])
    assert r.exit_code == 0, r.output
    bundle = tree / "P" / "frequency" / "F"
    # How a shell enters conda HERE is this machine's record's to say -- the
    # activation the generator reads (`configuration.md` § 4).
    from conftest import write_machine_record
    write_machine_record(env_init={
        "activation": "conda activate", "preamble": f"source {CONDA_SH}"})
    task = json.loads((bundle / "task.json").read_text())
    # Two atoms, two ranks, one thread: the run states its shape
    # (`architecture.md` § 5.2).
    task["execution"] = {**task.get("execution", {}), "mpi_np": 2,
                         "omp_threads": 1}
    (bundle / "task.json").write_text(json.dumps(task, indent=2))
    return bundle


def _jobset(*args):
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    return CliRunner().invoke(jobset_group, list(args))


def _kw(deck, key):
    """The value tokens of the first line of ``deck`` that sets ``key``."""
    for line in deck.splitlines():
        parts = line.split()
        if parts and parts[0] == key:
            return parts[1:]
    return []


def _tick_already_relaxed(bundle):
    tmpl = bundle / "H2.template.toml"
    text = tmpl.read_text()
    head, _, tail = text.partition("[item.already_relaxed]")
    assert tail, text
    tail = tail.replace("value = false", "value = true", 1)
    tmpl.write_text(head + "[item.already_relaxed]" + tail)


def _set_template_value(bundle, item, old, new):
    """Change one item's value in the calculation's template, the way a
    person edits it -- the ``value = <old>`` line under ``[item.<item>]``."""
    tmpl = bundle / "H2.template.toml"
    head, _, tail = tmpl.read_text().partition(f"[item.{item}]")
    assert tail, item
    assert f"value = {old}" in tail, (item, tail[:400])
    tmpl.write_text(head + f"[item.{item}]"
                    + tail.replace(f"value = {old}", f"value = {new}", 1))


def _the_monitor_closed(attempt, stage, step_words):
    """What the monitor beside a real run said at its end
    (`run-reports.md` § 2.1a, § 2.3): how the run ended in the Results tab's
    words, the step in SIESTA's own, every SCF phase converged -- and a
    utilisation basis that is the JOB's: a run started directly is its
    process tree over the two ranks it was launched on, and a CPU run is
    judged on no GPU."""
    from molbuilder.runfiles import compose
    log = (attempt / compose("H2", ".monitor.log", stage, run=0)).read_text()
    closing = [ln for ln in log.splitlines() if "[STATUS]" in ln][-1]
    assert "finished" in closing and step_words in closing, closing
    assert "converged: periodic yes" in closing, closing
    basis = next(ln for ln in log.splitlines() if "[UTIL-BASIS]" in ln)
    assert ("cpu% of 2 core(s) [launched on]" in basis
            and "cpu time [process tree]" in basis
            and "mem [process tree" in basis), basis
    summary = next(ln for ln in log.splitlines() if "[UTIL-SUMMARY]" in ln)
    assert "gpu" not in summary.lower(), summary
    # a fraction of the two cores it holds -- two busy ranks, not 87445%
    # (the first sample once took its rate over two microseconds)
    import re
    mean = float(re.search(r"cpu mean=(\d+)%", summary).group(1))
    assert 20.0 <= mean <= 120.0, summary


def _the_result(attempt, stage):
    """The spectrum the launch left in ``attempt`` -- written by the job
    itself, so the attempt concluded 0 only with it (`engines/vibration.md`
    § 5.5: the finish's failure is the job's)."""
    from molbuilder.runfiles import compose
    out = attempt / "H2.spectra.json"
    concluded = (attempt / compose("H2", ".concluded", stage, run=0)).read_text()
    assert out.is_file(), (
        "the launch ended without the spectrum: the job did not finish its "
        f"own calculation ({concluded.strip()})")
    assert concluded.startswith("rc=0"), concluded
    return json.loads(out.read_text())


def _common_assertions(d):
    assert d["engine"] == "siesta" and d["schema_version"] >= 6
    # The input order: the free atom is atom 0, the held one atom 1.
    assert d["free_atom_idxs"] == [0] and d["frozen_atom_idxs"] == [1]
    assert d["removed_motions"]["count"] == 2
    assert d["hessian_scope"] == "free" and d["n_atoms_in_hessian"] == 1
    assert d["equilibrium"]["mo_energies_eh"] is None
    assert d["modes"][0]["raman_activity_a4_amu"] is None
    assert d["thermo"]["regime"] == "vibrational-only"
    # the frequencies are this route's whole answer; the strengths and the
    # probe were never asked of it (vibration.md § 4.9)
    assert d["phase_frequencies"] == "complete"
    assert (d["phase_raman"], d["phase_es"]) == ("not requested",) * 2
    assert "Head1997" in d["bibliography_keys"]
    # R5 on this route: the reference-step forces read back and judged by
    # the template's own tolerance -- the kind's recommended 0.01 eV/A
    rx = d["relaxation"]
    assert d["engine_metadata"]["reference_force_criterion_ev_ang"] == 0.01
    assert rx["converged"] is True
    # the mass-calibrated displacement a transport step displaces along:
    # one hydrogen near 3000 cm-1 swings about 0.075 A at its zero point
    m0 = d["modes"][0]
    assert 0.06 < m0["zero_point_amplitude_amu12_ang"] < 0.09


def test_unticked_the_ladder_relaxes_first_and_freq_measures_at_the_relaxed_bond(
        tmp_path, monkeypatch):
    """The box unticked (the default `init` writes): the ladder is `relax`
    then `freq`; `freq` is refused before `relax` has concluded; after it,
    `freq` is written at the relaxed geometry and the modes come out at
    the relaxed bond's frequency with the reference forces within the
    template's own tolerance (vibration.md § 2.2, § 5.2a, § 5.5)."""
    from molbuilder.atom_permutation import read_permutation

    tree = tmp_path / "projects"
    # The experimental bond, 0.741 A -- NOT the stationary point at this
    # level of theory (measured 2026-09-24 on this road: 1.27 eV/A there,
    # and 3358 cm-1 where the relaxed bond gives 3022).
    bundle = _describe(tree, monkeypatch, [[5.0, 5.0, 5.741], [5.0, 5.0, 5.0]])
    task = json.loads((bundle / "task.json").read_text())
    assert [s["name"] for s in task["stages"]] == ["relax", "freq"]

    # THE JOB SET'S OWN ORDER: freq before relax has run is refused by the
    # one hand-over door, naming the stage to run first and how
    # (`continuation`, vibration.md § 5.2a's table).
    r = _jobset("prep", "run", "freq", "--bundle", str(bundle), "--target", "this")
    assert r.exit_code != 0 and "`freq` builds on `relax`" in r.output \
        and "Run it first" in r.output, r.output

    r = _jobset("prep", "run", "relax", "--bundle", str(bundle), "--target", "this")
    assert r.exit_code == 0, r.output
    relax_deck = next((bundle / "01_relax").glob("*.fdf")).read_text()
    # the ordinary relaxation deck, from the same sorted copy: Broyden to
    # the kind's recommended 0.01 eV/A, the held atom (first after the
    # sort) constrained
    assert _kw(relax_deck, "MD.TypeOfRun") == ["Broyden"], relax_deck
    assert _kw(relax_deck, "MD.MaxForceTol")[:1] == ["0.01"], relax_deck
    assert "position 1" in relax_deck
    r = _jobset("launch", "run", "relax", "--bundle", str(bundle),
                "--mode", "direct", "--yes")
    assert r.exit_code == 0, r.output
    _the_monitor_closed(bundle / "01_relax" / "run-0", "01_relax",
                        "Broyden opt. move")

    r = _jobset("prep", "run", "freq", "--bundle", str(bundle), "--target", "this")
    assert r.exit_code == 0, r.output
    perm = read_permutation(bundle)
    assert perm.key == "held-first" and perm.sorted_to_original == (1, 0)
    fc_deck = next((bundle / "02_freq").glob("*.fdf")).read_text()
    assert "MD.TypeOfRun      FC" in fc_deck and "FC.First          2" in fc_deck \
        and "FC.Last           2" in fc_deck and "position 1" in fc_deck
    r = _jobset("launch", "run", "freq", "--bundle", str(bundle),
                "--mode", "direct", "--yes")
    assert r.exit_code == 0, r.output
    attempt = bundle / "02_freq" / "run-0"
    assert (attempt / "H2.FC").is_file(), "the force-constant run must leave H2.FC"
    _the_monitor_closed(attempt, "02_freq", "FC step")
    # THE LAUNCH ENDS WITH THE RESULT: the job's finish wrote it (§ 5.5).
    d = _the_result(attempt, "02_freq")
    # ...and summarize derives nothing: with one force-constant stage there
    # is no sweep to summarize (§ 5.9, I22).
    r = _jobset("summarize", "run", "--bundle", str(bundle))
    assert r.exit_code != 0 and "a sweep compares two or more" in r.output, \
        r.output
    _common_assertions(d)
    # the ladder relaxed first, and the artifact says so from the relax
    # stage's own record (vibration.md § 4.9)
    assert d["relaxation"]["already_relaxed"] is False
    assert d["phase_relaxation"] == "complete"
    assert d["relaxation"]["enabled"] is True and d["relaxation"]["n_steps"] >= 1
    freqs = [m["frequency_cm1"] for m in d["modes"]]
    # the relaxed bond's frequency (3022-3024 across the tolerances measured
    # 2026-09-24), not the unrelaxed 3358
    assert len(freqs) == 1 and 2950.0 < freqs[0] < 3100.0, freqs
    # the geometry in the file is the one the force constants were taken
    # at -- the RELAXED bond, read from the run's own reference step
    pos = np.asarray(d["equilibrium"]["positions_ang"])
    bond = abs(pos[0, 2] - pos[1, 2])
    assert 0.76 < bond < 0.79, bond


def test_ticked_freq_alone_measures_at_the_geometry_as_given(tmp_path,
                                                             monkeypatch):
    """The box ticked: no relaxation, the force constants at the geometry as
    given, the reference forces judged by the template's tolerance.  And
    the contradiction refused first: unticked with no relax stage in the
    ladder is neither state, and prep names both ways out (§ 5.2a).

    The thermochemistry is summed at the TEMPLATE'S temperature, which is
    one meaning on both engines (§ 3.1): set to 350 K here, it reaches the
    result through the deck's `vibration` block -- and a vibrational-only
    result records no pressure, since none enters it (§ 4.7).

    MUTATION THIS MUST FAIL AGAINST: prep leaving the temperature out of the
    block -- the finish then refuses the deck and the launch fails.
    """
    tree = tmp_path / "projects"
    # the relaxed bond (H-H 0.7745 A, measured 2026-09-24 on this road)
    bundle = _describe(tree, monkeypatch, [[5.0, 5.0, 5.77446], [5.0, 5.0, 5.0]])
    task = json.loads((bundle / "task.json").read_text())
    task["stages"] = [s for s in task["stages"] if s["name"] == "freq"]
    (bundle / "task.json").write_text(json.dumps(task, indent=2))

    r = _jobset("prep", "run", "freq", "--bundle", str(bundle), "--target", "this")
    assert r.exit_code != 0 and "already_relaxed = true" in r.output \
        and "`relax` stage" in r.output, r.output
    _tick_already_relaxed(bundle)
    _set_template_value(bundle, "temperature_K", "298.15", "350.0")
    r = _jobset("prep", "run", "freq", "--bundle", str(bundle), "--target", "this")
    assert r.exit_code == 0, r.output
    r = _jobset("launch", "run", "freq", "--bundle", str(bundle),
                "--mode", "direct", "--yes")
    assert r.exit_code == 0, r.output
    attempt = bundle / "01_freq" / "run-0"
    d = _the_result(attempt, "01_freq")
    _common_assertions(d)
    assert d["thermo"]["temperature_K"] == 350.0, d["thermo"]["temperature_K"]
    assert d["thermo"]["pressure_atm"] is None
    assert 350.0 in d["thermo"]["grid"]["temperatures_K"]
    assert d["relaxation"]["already_relaxed"] is True
    # nothing relaxed: the force-constant run itself never does
    assert d["phase_relaxation"] == "not requested"
    assert d["relaxation"]["enabled"] is False
    freqs = [m["frequency_cm1"] for m in d["modes"]]
    assert len(freqs) == 1 and 2950.0 < freqs[0] < 3100.0, freqs
    from molbuilder.constants import HARTREE_BOHR_EV_ANGSTROM_ASE
    assert d["relaxation"]["max_force_eh_bohr"] < 0.01 / HARTREE_BOHR_EV_ANGSTROM_ASE


def _deck_coordinates(deck_text):
    """The deck's own coordinate block, as SIESTA reads it."""
    from molbuilder.parse.fdf import _norm, _parse_fdf
    rows = _parse_fdf(deck_text)[1][_norm("AtomicCoordinatesAndAtomicSpecies")]
    return np.array([[float(x) for x in r[:3]] for r in rows])


def test_a_displacement_sweep_measures_every_stage_at_the_relaxed_bond(
        tmp_path, monkeypatch):
    """A displacement sweep (`engines/vibration.md` § 5.9): after `relax`,
    two force-constant stages -- `freq` at the template's 0.04 Bohr and
    `freq_half` at 0.02 -- BOTH measure at the relaxed geometry, whatever
    their names (I24); each stage's job writes its own spectrum; and
    `summarize run` compares them into `<label>.fc-sweep.json` at the root,
    naming each stage's files rather than copying them (I25).  Before
    `freq_half` is launched the record lists it as pending, in its attempt's
    own words.

    MUTATION THIS MUST FAIL AGAINST: only the stage named `freq` taking the
    relaxed geometry -- `freq_half` then measures the unrelaxed input bond.
    """
    tree = tmp_path / "projects"
    bundle = _describe(tree, monkeypatch, [[5.0, 5.0, 5.741], [5.0, 5.0, 5.0]])
    task = json.loads((bundle / "task.json").read_text())
    # a stage overrides only what the description varies (stages.md § 6.2)
    task["varies"] = sorted(set(task.get("varies") or []) | {"fc_displacement"})
    task["stages"].append({"name": "freq_half", "enabled": True,
                           "overrides": {"fc_displacement": 0.02}})
    (bundle / "task.json").write_text(json.dumps(task, indent=2))

    def _prep_and_launch(stage):
        r = _jobset("prep", "run", stage, "--bundle", str(bundle),
                    "--target", "this")
        assert r.exit_code == 0, r.output
        r = _jobset("launch", "run", stage, "--bundle", str(bundle),
                    "--mode", "direct", "--yes")
        assert r.exit_code == 0, r.output

    _prep_and_launch("relax")
    _prep_and_launch("freq")
    r = _jobset("prep", "run", "freq_half", "--bundle", str(bundle),
                "--target", "this")
    assert r.exit_code == 0, r.output
    # A STAGE STILL TO COME is pending, never a failure, in the words its
    # attempt's `run_status` gives (prepped, not launched).
    r = _jobset("summarize", "run", "--bundle", str(bundle))
    assert r.exit_code == 0, r.output
    early = json.loads((bundle / "H2.fc-sweep.json").read_text())
    assert [s["name"] for s in early["stages"]] == ["freq"], early["stages"]
    (waiting,) = early["pending"]
    assert (waiting["stage"], waiting["state"]) == ("freq_half", "pending"), \
        waiting
    assert early["failed"] == []
    r = _jobset("launch", "run", "freq_half", "--bundle", str(bundle),
                "--mode", "direct", "--yes")
    assert r.exit_code == 0, r.output

    decks = {tok: next((bundle / tok).glob("*.fdf")).read_text()
             for tok in ("02_freq", "03_freq_half")}
    # I24: one geometry for every force-constant stage -- the relaxed one.
    assert np.allclose(_deck_coordinates(decks["02_freq"]),
                       _deck_coordinates(decks["03_freq_half"]), atol=1e-8), \
        "the second force-constant stage was not written at the relaxed bond"
    assert _kw(decks["03_freq_half"], "FC.Displacement")[:1] == ["0.02"]
    a1, a2 = (_the_result(bundle / "02_freq" / "run-0", "02_freq"),
              _the_result(bundle / "03_freq_half" / "run-0", "03_freq_half"))
    for d in (a1, a2):
        assert d["phase_relaxation"] == "complete" and \
            d["relaxation"]["converged"] is True

    r = _jobset("summarize", "run", "--bundle", str(bundle))
    assert r.exit_code == 0, r.output
    rec = json.loads((bundle / "H2.fc-sweep.json").read_text())
    assert rec["schema"] == "molbuilder/fc-displacement-sweep@1"
    assert rec["pending"] == [] and rec["failed"] == []  # every stage has one
    # the Results tab offers only what the parse registry reads
    # (`model/parse.md` § 5.5): the record has its own reader
    from molbuilder.parse import detect
    kind = detect(str(bundle / "H2.fc-sweep.json"))
    assert kind.name == "fc-sweep-json", kind
    assert kind.parse(bundle / "H2.fc-sweep.json").schema == "fc-sweep/v1"
    assert [s["name"] for s in rec["stages"]] == ["freq", "freq_half"]
    # I25: each stage's files are NAMED where its run wrote them, and are its
    # own -- the record carries paths, not copies.
    for s in rec["stages"]:
        for key in ("spectrum", "fc_file"):
            assert (bundle / s[key]).is_file(), (key, s)
    assert rec["stages"][0]["spectrum"].startswith("02_freq/run-0/")
    assert rec["stages"][1]["spectrum"].startswith("03_freq_half/run-0/")
    assert not list(bundle.glob("*.FC")) and not list(
        bundle.glob("*.spectra.json")), "the summary copied a stage's files"
    # the displacement each stage USED, from its own spectrum
    d1, d2 = (s["fc_displacement_ang"] for s in rec["stages"])
    assert abs(d2 / d1 - 0.5) < 1e-6, (d1, d2)
    assert rec["stages"][1]["varies"] == {"fc_displacement": 0.02}
    assert rec["stages"][1]["varies_units"] == {"fc_displacement": "Bohr"}
    # the one mode, matched by shape, at both displacements
    (m,) = rec["modes"]
    assert set(m["frequency_cm1"]) == {"freq", "freq_half"}
    assert min(m["overlap"].values()) > 0.999, m
    assert m["spread_cm1"] == pytest.approx(
        abs(m["frequency_cm1"]["freq"] - m["frequency_cm1"]["freq_half"]))
    assert m["flagged"] is None and rec["tolerance_cm1"] is None
    (fc,) = rec["force_constants"]
    assert fc["against"] == "freq" and fc["max_abs_change_ev_ang2"] > 0.0

    # a tolerance the person gives flags; the numbers are unchanged
    r = _jobset("summarize", "run", "--bundle", str(bundle),
                "--tolerance-cm1", "1e-6")
    assert r.exit_code == 0, r.output
    rec2 = json.loads((bundle / "H2.fc-sweep.json").read_text())
    assert rec2["modes"][0]["flagged"] is True and rec2["tolerance_cm1"] == 1e-6


def test_a_flat_calculation_refuses_a_second_force_constant_stage(
        tmp_path, monkeypatch):
    """In the flat layout every stage writes the same `<label>.FC` and
    `<label>.spectra.json`, so two force-constant stages would overwrite each
    other's result: `prep` refuses the description before any sort,
    permutation record or deck is written, at whichever stage is prepped
    first (`engines/vibration.md` § 5.9) -- counting the stages the
    description runs: a disabled one is never prepped (`engines/stages.md`
    § 6.2).

    MUTATION THIS MUST FAIL AGAINST: the check not made.
    """
    tree = tmp_path / "projects"
    bundle = _describe(tree, monkeypatch, [[5.0, 5.0, 5.741], [5.0, 5.0, 5.0]],
                       shape="flat")
    task = json.loads((bundle / "task.json").read_text())
    # a stage overrides only what the description varies (stages.md § 6.2)
    task["varies"] = sorted(set(task.get("varies") or []) | {"fc_displacement"})
    task["stages"].append({"name": "freq_half", "enabled": True,
                           "overrides": {"fc_displacement": 0.02}})
    (bundle / "task.json").write_text(json.dumps(task, indent=2))
    r = _jobset("prep", "run", "relax", "--bundle", str(bundle),
                "--target", "this")
    assert r.exit_code != 0 and "hierarchical layout" in r.output, r.output
    for written in ("*.fdf", "atom-permutation.json", "job-set.json"):
        assert not list(bundle.glob(written)), f"{written} before refusing"
