"""`jobset prep bench` — step 6 u2: the grid through the ONE builder.

Contract: `project-layout.md` § 2.3.1a (benchmarking is `prep` whose
parameters are a set) · § 2.3.2 (trials relabelled + forced cold; submission
one trial per invocation) · `generator.md` §§ 2, 5 · `template.md` § 8.1
(rebuild and render, never splice — the pins land as schema values).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from molbuilder import describe as D
from molbuilder.config.siesta import SiestaConfig
from molbuilder.environment import Environment, Topology
from molbuilder.jobset._cli import _bench_inputs
from molbuilder.jobset.model import Resources
from molbuilder.jobset.prep import prep_calculation
from molbuilder.siesta.stages import default_siesta_stages
from molbuilder.structure import Structure


@pytest.fixture
def calc(tmp_path):
    """A described calculation on a machine whose probe found one a100."""
    struct = Structure(elements=["H", "H"],
                       positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
                       vacuum=(10.0, 10.0, 10.0))
    (tmp_path / "h2.xyz").write_text(struct.to_xyz())
    dest = tmp_path / "calc"
    D.write_description(
        D.build_description(struct, SiestaConfig(system_label="JOB"),
                            default_siesta_stages("publishable"),
                            engine="siesta", shape="hierarchical", name="JOB",
                            source=str(tmp_path / "h2.xyz")),
        dest)
    (dest / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": "true"}}))
    # The probe's answer, pre-seeded: resolve_target early-returns on an
    # existing environment.json, so the grid enumerates THIS topology.
    (dest / "environment.json").write_text(
        Environment(scheduler="workstation",
                    topology=Topology(sockets=1, cores_per_socket=4,
                                      gpus_per_node=1,
                                      gpu_type="a100")).to_json() + "\n")
    return dest


def _prep_bench(calc):
    sweep, pins, translation = _bench_inputs(calc)
    prep_calculation(calc, "coarse",
                     allocation=Resources(mpi_np=8, cpus_per_task=8),
                     sweep=sweep, pins=pins, translation=translation,
                     emit_sbatch=False)
    # The sweep's OWN record lives in the stage's bench/ container
    # (job-contracts.md § 6.3); the root job-set.json is the RUN plan.
    return json.loads(
        (calc / "01_coarse" / "bench" / "job-set.json").read_text())


def test_the_grid_becomes_a_sweep_jobset_of_relabelled_trials(calc):
    js = _prep_bench(calc)
    assert js["kind"] == "sweep"
    names = [j["name"] for j in js["jobs"]]
    assert len(names) == len(set(names)) and len(names) >= 2
    assert all(n.startswith("G1K") and "C" in n for n in names)
    # one deck per trial, named by the TRIAL's label + the stage token
    import re
    for n in names:
        deck = calc / f"JOB-{n}_01_coarse.fdf"
        assert deck.is_file()
        assert re.search(rf"^SystemLabel\s+JOB-{n}\s*$",
                         deck.read_text(), re.M)


def test_the_pins_land_as_rendered_schema_values_not_splices(calc):
    """What `transform_fdf` spliced is now resolved and rendered: the capped
    SCF, the single point, the eigensolver — readable in the deck."""
    import re
    js = _prep_bench(calc)
    deck = (calc / f"JOB-{js['jobs'][0]['name']}_01_coarse.fdf").read_text()
    assert re.search(r"^MaxSCFIterations\s+5\s*$", deck, re.M)
    assert re.search(r"^Diag\.Algorithm\s+ELPA-1STAGE\s*$", deck, re.M)
    assert re.search(r"^Diag\.ELPA\.GPU\s+\.true\.\s*$", deck, re.M)
    assert re.search(r"^MD\.NumCGsteps\s+0\s*$", deck, re.M)


def test_each_trials_resources_carry_its_own_coordinate(calc):
    """G·K ranks, C cores, the machine's GPU type in gres — translated once,
    per element, never re-derived downstream (generator.md § 5)."""
    js = _prep_bench(calc)
    for job in js["jobs"]:
        g = int(job["name"][1])
        k = int(job["name"][job["name"].index("K") + 1:job["name"].index("C")])
        c = int(job["name"][job["name"].index("C") + 1:])
        r = job["resources"]
        assert r["mpi_np"] == g * k
        assert r["cpus_per_task"] == c
        assert r["gres"] == f"gpu:a100:{g}"


def test_trials_nest_inside_the_stage_they_measure(calc):
    """`job-contracts.md` § 6.3: ``<NN>_<stage>/bench/bench-<point>/`` — the
    trial in the stage's ``bench/`` CONTAINER, and its links reach the
    bundle root through a COMPUTED depth, not a hardcoded hop."""
    js = _prep_bench(calc)
    name = js["jobs"][0]["name"]
    d = calc / "01_coarse" / "bench" / f"bench-{name}"
    assert d.is_dir()
    script = d / f"JOB-{name}_01_coarse.fdf"
    assert script.is_symlink() and script.resolve().is_file()
    wrapper = d / f"JOB-{name}_01_coarse.run.sh"
    assert wrapper.is_symlink() and wrapper.resolve().is_file()
    # nothing landed at the old flat location
    assert not (calc / f"bench-{name}").exists()
    assert not (calc / f"point-{name}").exists()


def test_a_machine_without_gpus_is_refused_by_name(calc):
    import click
    (calc / "environment.json").write_text(
        Environment(scheduler="workstation",
                    topology=Topology(sockets=1,
                                      cores_per_socket=4)).to_json() + "\n")
    with pytest.raises(click.ClickException, match=r"no GPU topology"):
        _bench_inputs(calc)


def test_cli_prep_bench_requires_a_stage(calc):
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    r = CliRunner().invoke(jobset_group,
                           ["prep", "bench", "--bundle", str(calc)])
    assert r.exit_code != 0
    assert "name it" in r.output


def test_cli_prep_bench_end_to_end_lists_trials_not_attempts(calc):
    """The whole verb through the CLI: trials prepped and listed, and NO
    attempt machinery -- a sweep's jobs are named by coordinate, so the
    run-kind's stage/attempt tail must not run."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    r = CliRunner().invoke(jobset_group,
                           ["prep", "bench", "coarse", "--bundle", str(calc),
                            "--np", "8", "--cpus-per-task", "8",
                            "--no-sbatch"])
    assert r.exit_code == 0, r.output
    assert "trial dir(s) for stage 'coarse'" in r.output
    assert "one trial per invocation" in r.output
    assert "config:" in r.output          # provenance rides every prep


def test_cli_summarize_bench_reads_trials_by_data(calc):
    """u4: discovery keyed by job-set.json's data, results through the same
    artifacts as any run, ASYNC — a trial with no output yet reports
    ``incomplete`` rather than failing the set (user, 2026-08-12)."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    js = _prep_bench(calc)
    name = js["jobs"][0]["name"]
    d = calc / "01_coarse" / "bench" / f"bench-{name}"
    (d / f"JOB-{name}_01_coarse-run0.out").write_text(
        "banner\nsiesta: Final energy (eV) = -1.0\n")
    r = CliRunner().invoke(jobset_group, ["summarize", "bench", "coarse",
                                          "--bundle", str(calc)])
    assert r.exit_code == 0, r.output
    assert (calc / "01_coarse" / "bench" / "bench-result.json").is_file()
    assert not (calc / "bench-result.json").exists()
    assert name in r.output
    assert "completed" in r.output       # the trial with the finished .out
    assert "unknown" in r.output         # siblings with no output yet
    # A run's outputs are the calculation's, not a benchmark's to rank.
    r = CliRunner().invoke(jobset_group,
                           ["summarize", "run", "--bundle", str(calc)])
    assert r.exit_code != 0


def _finished_trial_and_verdict(calc):
    """prep bench + one completed trial + summarize -> bench-result.json."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    js = _prep_bench(calc)
    name = js["jobs"][0]["name"]
    d = calc / "01_coarse" / "bench" / f"bench-{name}"
    (d / f"JOB-{name}_01_coarse-run0.out").write_text(
        "x\n>> End of run:\n")
    # epoch-per-line format: consecutive deltas are the per-iter durations
    (d / f"JOB-{name}_01_coarse-run0.scf-timing.log").write_text(
        "100.0 scf 1\n104.0 scf 2\n108.0 scf 3\n112.0 scf 4\n")
    r = CliRunner().invoke(jobset_group, ["summarize", "bench", "coarse",
                                          "--bundle", str(calc)])
    assert r.exit_code == 0, r.output
    return name


def test_prep_run_offers_the_verdict_and_silence_is_no(calc):
    """§ 2.3.2: it asks; it does not just take it — and a non-interactive
    shell's silence is No, so nothing is ever applied by default."""
    import re
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    _finished_trial_and_verdict(calc)
    r = CliRunner().invoke(jobset_group,
                           ["prep", "run", "coarse", "--bundle", str(calc),
                            "--no-sbatch"])
    assert r.exit_code == 0, r.output
    assert "a benchmark result exists" in r.output
    assert "use it?" in r.output
    assert "not applied" in r.output
    deck = (calc / "JOB_01_coarse.fdf").read_text()
    assert not re.search(r"^Diag\.ELPA\.GPU\s+\.true\.", deck, re.M)


def test_prep_run_applies_an_accepted_verdict_but_flags_win(calc):
    """On yes, the measured machine half fills only what the user did NOT
    state, and the winner's eigensolver arrives as pins."""
    import re
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    from molbuilder.parse.scripts.bench_marks import _extract_bench_marks_dict
    name = _finished_trial_and_verdict(calc)
    g = int(name[1])
    k = int(name[name.index("K") + 1:name.index("C")])
    r = CliRunner().invoke(jobset_group,
                           ["prep", "run", "coarse", "--bundle", str(calc),
                            "--cpus-per-task", "3", "--no-sbatch"],
                           input="y\n")
    assert r.exit_code == 0, r.output
    assert "applied:" in r.output
    deck = (calc / "JOB_01_coarse.fdf").read_text()
    marks = _extract_bench_marks_dict(deck)
    assert marks.get("mpi_np") == g * k          # measured, user said nothing
    assert re.search(r"^Diag\.Algorithm\s+ELPA-1STAGE\s*$", deck, re.M)
    assert re.search(r"^Diag\.ELPA\.GPU\s+\.true\.\s*$", deck, re.M)
    # the flag the user DID state beat the verdict
    js = json.loads((calc / "job-set.json").read_text())
    assert js["jobs"][0]["resources"]["cpus_per_task"] == 3


def test_cli_submit_bench_is_one_trial_per_invocation(calc):
    """`submit bench <stage> [<trial>]` (§ 2.3.2, decided 2026-08-12): no
    stage refuses by name; a bare stage picks the NEXT UNLAUNCHED trial and
    says how many remain; naming a trial launches THAT one.  Either way ONE
    launch per invocation -- direct mode stays exempt (not submission)."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    js = _prep_bench(calc)
    trial = js["jobs"][0]["name"]
    runner = CliRunner()
    r = runner.invoke(jobset_group, ["submit", "bench", "--bundle", str(calc),
                                     "--mode", "submit", "--dry-run"])
    assert r.exit_code != 0 and "name it" in r.output
    r = runner.invoke(jobset_group, ["submit", "bench", "coarse",
                                     "--bundle", str(calc),
                                     "--mode", "submit", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "next unlaunched trial" in r.output
    assert f"({len(js['jobs'])} of {len(js['jobs'])} remain" in r.output
    assert r.output.count("WOULD run") == 1
    r = runner.invoke(jobset_group, ["submit", "bench", "coarse", trial,
                                     "--bundle", str(calc),
                                     "--mode", "submit", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert r.output.count("WOULD run") == 1
    # a trial name is a bench concept: run refuses it
    r = runner.invoke(jobset_group, ["submit", "run", "coarse", trial,
                                     "--bundle", str(calc), "--dry-run"])
    assert r.exit_code != 0 and "TRIAL names a benchmark point" in r.output


def test_prep_run_of_a_second_stage_merges_the_root_plan(calc):
    """The root ``job-set.json`` is the RUN plan and MERGES per stage
    (`job-contracts.md` § 6.1): prepping `medium` must not erase `coarse` --
    erasing it broke the status rollup and silently withheld the
    cross-stage ``.CG`` carry."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    r = CliRunner()
    for stage in ("coarse", "medium"):
        res = r.invoke(jobset_group, ["prep", "run", stage,
                                      "--bundle", str(calc), "--no-sbatch"])
        assert res.exit_code == 0, res.output
    js = json.loads((calc / "job-set.json").read_text())
    assert js["kind"] == "ladder"
    assert [j["name"] for j in js["jobs"]] == ["coarse", "medium"]
    # and re-prepping a stage REPLACES its own entry, never duplicates it
    res = r.invoke(jobset_group, ["prep", "run", "coarse",
                                  "--bundle", str(calc), "--no-sbatch"])
    assert res.exit_code == 0, res.output
    js = json.loads((calc / "job-set.json").read_text())
    assert [j["name"] for j in js["jobs"]] == ["coarse", "medium"]


def test_a_sweeps_record_never_touches_the_root_plan(calc):
    """Per-kind persistence (§ 6.1): `prep bench` writes the sweep's OWN
    ``job-set.json`` into the stage's container and leaves the root plan
    alone -- a run prepped afterwards is a ladder, not a sweep."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    _prep_bench(calc)
    assert not (calc / "job-set.json").exists()
    r = CliRunner().invoke(jobset_group, ["prep", "run", "coarse",
                                          "--bundle", str(calc),
                                          "--no-sbatch"])
    assert r.exit_code == 0, r.output
    js = json.loads((calc / "job-set.json").read_text())
    assert js["kind"] == "ladder"
    assert [j["name"] for j in js["jobs"]] == ["coarse"]
    sweep = json.loads(
        (calc / "01_coarse" / "bench" / "job-set.json").read_text())
    assert sweep["kind"] == "sweep"


def test_every_verb_records_its_decisions_in_the_ledger(calc):
    """The bundle's decision ledger (user rule 2026-08-12): what each verb
    decided -- and from which inputs -- is READ BACK from the bundle, in
    order, when the terminal is long gone."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    from molbuilder.jobset.ledger import LEDGER_FILE
    r = CliRunner()
    # through the VERB: the ledger is the surface's record of what it
    # decided -- the library returns decision data and never logs
    res = r.invoke(jobset_group,
                   ["prep", "bench", "coarse", "--bundle", str(calc),
                    "--np", "8", "--cpus-per-task", "8", "--no-sbatch"])
    assert res.exit_code == 0, res.output
    res = r.invoke(jobset_group, ["submit", "bench", "coarse",
                                  "--bundle", str(calc),
                                  "--mode", "submit", "--dry-run"])
    assert res.exit_code == 0, res.output
    res = r.invoke(jobset_group, ["summarize", "bench", "coarse",
                                  "--bundle", str(calc)])
    assert res.exit_code == 0, res.output
    lines = [json.loads(l) for l in
             (calc / LEDGER_FILE).read_text().splitlines()]
    got = [(e["verb"], e["decision"]) for e in lines]
    assert got == [("prep", "prepped"),
                   ("submit", "trial-picked"),
                   ("submit", "launched"),
                   ("summarize", "verdict-written")]
    prep = lines[0]
    assert prep["kind"] == "bench" and prep["stage"] == "coarse"
    assert "provenance" in prep            # WHERE each setting came from
    pick = lines[1]
    assert pick["picked_by"].startswith("next unlaunched")
    assert pick["total"] == pick["remaining"] == len(
        json.loads((calc / "01_coarse" / "bench"
                    / "job-set.json").read_text())["jobs"])
    launch = lines[2]
    assert launch["mode"] == "submit"
    assert launch["mode_source"] == "--mode flag"
    assert launch["jobs"][0]["status"] == "planned"


def test_the_choice_names_its_winner_and_its_mechanism(calc):
    """U13: the verdict is consumable as DATA -- the winner's label is a
    field (not a sentence to parse), the knobs speak the job-set's own
    exchange vocabulary, and the MECHANISM is read from the winning
    trial's own deck, never re-derived from `engine == "gpu"`."""
    _finished_trial_and_verdict(calc)
    res = json.loads((calc / "01_coarse" / "bench"
                      / "bench-result.json").read_text())
    choice = res["choice"]
    assert choice["label"] and choice["label"] in (
        j["name"] for j in json.loads(
            (calc / "01_coarse" / "bench" / "job-set.json").read_text()
        )["jobs"])
    assert "mpi_np" in choice["knobs"]          # exchange, not "ranks"
    assert "ranks" not in choice["knobs"]
    mech = choice["mechanism"]
    assert mech["enable_gpu"] is True           # the deck's gpu_mode
    assert mech["diag_algorithm"] == "ELPA-1STAGE"   # the deck's own line
    assert res["generated_at"]                  # the offer reads this key


def test_prep_over_a_launched_attempt_asks_and_no_stops_it(calc):
    """U14/A3: a re-prep where a run already HAPPENED (a launched
    attempt's run.json) says what is there and ASKS.  'n' stops before
    anything is written; 'y' proceeds; § 6's floor holds either way --
    warm files untouched, nothing renamed."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    from molbuilder.jobset.materialize import write_run_launch
    r = CliRunner()
    res = r.invoke(jobset_group, ["prep", "run", "coarse",
                                  "--bundle", str(calc), "--no-sbatch"])
    assert res.exit_code == 0, res.output
    attempt = calc / "01_coarse" / "run-0"
    assert attempt.is_dir()
    write_run_launch(attempt, mode="direct", command=["bash", "x"])
    res = r.invoke(jobset_group, ["prep", "run", "coarse",
                                  "--bundle", str(calc), "--no-sbatch"],
                   input="n\n")
    assert res.exit_code != 0
    assert "already under way" in res.output
    assert "run-0/ was launched" in res.output
    assert "NOT touched" in res.output
    assert "stopped at your request" in res.output
    res = r.invoke(jobset_group, ["prep", "run", "coarse",
                                  "--bundle", str(calc), "--no-sbatch"],
                   input="y\n")
    assert res.exit_code == 0, res.output


def test_prep_underway_with_no_answer_proceeds_and_says_so(calc):
    """§ 6 warns, it does not refuse: non-interactively (EOF at the
    prompt) the re-prep PROCEEDS and says so -- the inverse of the
    verdict offer's silence-is-no, deliberately: no one else's numbers
    are being applied, and a scripted re-prep must not die on a question
    it cannot hear."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    from molbuilder.jobset.materialize import write_run_launch
    r = CliRunner()
    res = r.invoke(jobset_group, ["prep", "run", "coarse",
                                  "--bundle", str(calc), "--no-sbatch"])
    assert res.exit_code == 0, res.output
    write_run_launch(calc / "01_coarse" / "run-0",
                     mode="direct", command=["bash", "x"])
    res = r.invoke(jobset_group, ["prep", "run", "coarse",
                                  "--bundle", str(calc), "--no-sbatch"])
    assert res.exit_code == 0, res.output
    assert "no answer (non-interactive): proceeding" in res.output
    # and the decision is in the ledger
    from molbuilder.jobset.ledger import LEDGER_FILE
    lines = [json.loads(l) for l in
             (calc / LEDGER_FILE).read_text().splitlines()]
    asks = [e for e in lines if e["decision"] == "underway-ask"]
    assert asks and asks[-1]["answer"].startswith("no answer")


def test_a_trial_is_cold_by_construction(calc):
    """§ 2.3.2's forced-cold half, pinned at its mechanism (U20): the
    RELABEL is what makes a trial cold -- its deck's SystemLabel names
    warm files that never exist, and nothing links the real run's warm
    state into a trial's directory.  A benchmark that warm-started from
    the production run would measure the wrong thing silently."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    r = CliRunner().invoke(jobset_group, ["prep", "run", "coarse",
                                          "--bundle", str(calc),
                                          "--no-sbatch"])
    assert r.exit_code == 0, r.output
    # the production run leaves warm state under the BASE label
    (calc / "01_coarse" / "JOB.DM").write_text("warm density")
    (calc / "01_coarse" / "JOB.XV").write_text("warm coords")
    js = _prep_bench(calc)
    for j in js["jobs"]:
        d = calc / "01_coarse" / "bench" / f"bench-{j['name']}"
        # no file or link under the BASE label reaches the trial
        assert not list(d.glob("JOB.*")), list(d.iterdir())
        # and the deck's own label is the trial's, so SIESTA's UseSave*
        # looks for JOB-<point>.DM -- which never exists
        import re
        deck = (d / f"JOB-{j['name']}_01_coarse.fdf").read_text()
        assert re.search(rf"^SystemLabel\s+JOB-{j['name']}\s*$", deck, re.M)


def test_each_trials_wrapper_carries_its_own_translated_launch(calc):
    """The STOMP's true regression pin (U1 fixed it at the job-set level;
    this pins the WRAPPER TEXT, where the bug actually bit): every
    trial's .run.sh bakes ITS OWN G*K rank default, not the base
    allocation's -- a stomped wrapper launches every trial at the same
    np and the benchmark measures nothing."""
    import re
    js = _prep_bench(calc)
    seen = set()
    for j in js["jobs"]:
        name = j["name"]
        g = int(name[1])
        k = int(name[name.index("K") + 1:name.index("C")])
        w = (calc / "01_coarse" / "bench" / f"bench-{name}"
             / f"JOB-{name}_01_coarse.run.sh").resolve().read_text()
        # ANCHORED to the assignment lines: the GPU policy block's
        # ``_gpu_mpi_np_default=4`` contains this name as a substring,
        # which is exactly what an unanchored search matched first.
        vals = re.findall(r"^\s*_mpi_np_default=(\d+)$", w, re.M)
        assert vals, "wrapper carries no baked rank default"
        assert all(int(v) == g * k for v in vals), (name, vals)
        seen.add(g * k)
    assert len(seen) > 1, "all wrappers share one rank count -- the stomp"


def test_stage_plan_records_the_config_provenance(calc):
    """STAGE-PLAN.md is the reviewable record; the provenance block
    (which file supplied each setting) is what makes a machine
    difference debuggable from the plan alone (user rule 2026-08-12)."""
    _prep_bench(calc)
    plan = (calc / "01_coarse" / "bench" / "STAGE-PLAN.md").read_text()
    assert "config:" in plan
    assert ".molbuilder.json" in plan


def test_the_two_stage_sequence_carries_the_geometry_forward(calc):
    """The cross-stage story end-to-end: prep+launch-record coarse, then
    prep medium --from coarse's attempt -- the carried warm files land
    in medium's attempt, which is what the root plan's MERGE (U1) keeps
    verifiable."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    r = CliRunner()
    res = r.invoke(jobset_group, ["prep", "run", "coarse",
                                  "--bundle", str(calc), "--no-sbatch"])
    assert res.exit_code == 0, res.output
    attempt = calc / "01_coarse" / "run-0"
    (attempt / "JOB.XV").write_text("relaxed coords")
    res = r.invoke(jobset_group, ["prep", "run", "medium",
                                  "--bundle", str(calc), "--no-sbatch",
                                  "--from", "01_coarse/run-0"])
    assert res.exit_code == 0, res.output
    carried = list((calc / "02_medium").rglob("JOB.XV"))
    assert carried, "the geometry did not carry"
    assert carried[0].read_text() == "relaxed coords"
    js = json.loads((calc / "job-set.json").read_text())
    assert [j["name"] for j in js["jobs"]] == ["coarse", "medium"]


def test_a_stageless_calculation_runs_end_to_end(tmp_path):
    """R1 (review-4 keystone): `engines/stages.md` § 6.5's single-
    parameter-set calculation is a FIRST-CLASS run.  Until 2026-08-12 the
    tokenless fallback filed it under ``bench-<label>/`` (a directory
    named for a benchmark), the hint said "prep run <stage>" over a
    ladder with no stages, and submit was unreachable -- the whole form
    was dead end-to-end.  The calculation IS its own one rung: deck,
    wrapper and attempts at the bundle root, bare verbs acting on it."""
    from click.testing import CliRunner
    from molbuilder import describe as D
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.jobset._cli import jobset_group
    from molbuilder.structure import Structure
    struct = Structure(elements=["H", "H"],
                       positions=np.array([[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.74]]),
                       vacuum=(10.0, 10.0, 10.0))
    (tmp_path / "h2.xyz").write_text(struct.to_xyz())
    dest = tmp_path / "calc"
    D.write_description(
        D.build_description(struct, SiestaConfig(system_label="JOB"), (),
                            engine="siesta", shape="hierarchical",
                            name="JOB", source=str(tmp_path / "h2.xyz")),
        dest)
    (dest / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": "true"}}))
    r = CliRunner()
    res = r.invoke(jobset_group, ["prep", "run", "--bundle", str(dest),
                                  "--no-sbatch"])
    assert res.exit_code == 0, res.output
    assert (dest / "run-0").is_dir()          # the attempt, at the root
    assert not (dest / "bench-JOB").exists()  # the old misfiling
    assert "prep run <stage>" not in res.output   # no circular hint
    assert "submit run --mode" in res.output
    js = json.loads((dest / "job-set.json").read_text())
    assert js["kind"] == "ladder" and len(js["jobs"]) == 1
    res = r.invoke(jobset_group, ["submit", "run", "--bundle", str(dest),
                                  "--mode", "direct", "--dry-run"])
    assert res.exit_code == 0, res.output
    assert res.output.count("WOULD run") == 1
    res = r.invoke(jobset_group, ["status", "--bundle", str(dest)])
    assert res.exit_code == 0, res.output
    # a hand-built SWEEP's tokenless jobs keep their bench-<name> homes
    from molbuilder.jobset.materialize import job_dir_names
    from molbuilder.jobset.model import Job, JobSet, Resources
    sweep = JobSet(name="X", engine="siesta", kind="sweep",
                   jobs=[Job(name="G1K1C4", script="job-gpu.fdf",
                             resources=Resources())])
    assert job_dir_names(sweep)["G1K1C4"] == "bench-G1K1C4"


def test_a_launched_trial_refuses_relaunch_and_selection_skips_it(calc):
    """R2: § 1.5's immutability holds for trials AT THE SEAM -- a named
    relaunch is refused by the library naming run.json and the next
    verbs, and the bare form's next-unlaunched pick SKIPS the launched
    trial (the first test anywhere to exercise selection with a launched
    trial present)."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    from molbuilder.jobset.materialize import write_run_launch
    js = _prep_bench(calc)
    first = js["jobs"][0]["name"]
    second = js["jobs"][1]["name"]
    write_run_launch(calc / "01_coarse" / "bench" / f"bench-{first}",
                     mode="submit", command=["sbatch", "x"], job_id="42")
    r = CliRunner()
    res = r.invoke(jobset_group, ["submit", "bench", "coarse", first,
                                  "--bundle", str(calc),
                                  "--mode", "submit", "--dry-run"])
    assert res.exit_code != 0
    assert "already launched" in res.output
    assert "summarize" in res.output
    res = r.invoke(jobset_group, ["submit", "bench", "coarse",
                                  "--bundle", str(calc),
                                  "--mode", "submit", "--dry-run"])
    assert res.exit_code == 0, res.output
    assert f"next unlaunched trial: {second}" in res.output
