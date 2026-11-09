"""`jobset prep bench` — step 6 u2: the grid through the ONE builder.

Contract: `project-layout.md` § 2.3.1a (benchmarking is `prep` whose
parameters are a set) · § 2.3.2 (trials relabelled + forced cold; submission
grouped per resource shelf, a named trial alone) · `generator.md` §§ 2, 5 · `template.md` § 8.1
(rebuild and render, never splice — the pins land as schema values).
"""
from __future__ import annotations

import pathlib
import re

from conftest import write_machine_record
from molbuilder.calculation import open as open_calculation
from molbuilder.jobset import ledger
from molbuilder.jobset.materialize import bench_owner, latest_attempt
from molbuilder.jobset.model import JobSet
from molbuilder.jobset.summarize import sweep_view
from molbuilder.scheduler import Domain, Topology
from support.machines import SLURM
from support.road import (RUN_CARD, declare_bench, described, group_words,
                          jobset, read_request, road_tree, shelf_headers,
                          write_request)

import pytest

#: This machine, as its probe found it: four cores and one a100 on a
#: workstation; `_on_a_queue` is the same node behind a scheduler.
_NODE = Topology(sockets=1, cores_per_socket=4, gpus_per_node=1,
                 gpu_type="a100")


def _artifacts(calc, point, stage="01_coarse"):
    """Where a trial's files ARE — its attempt.

    A trial keeps attempts (`project-layout.md` § 1.5a), so its deck and
    wrapper sit in ``run-<n>`` rather than in the trial directory -- the
    one reader rule, `materialize.latest_attempt`.
    """
    c = pathlib.Path(calc) / stage / "bench" / f"bench-{point}"
    return latest_attempt(c)


def _described(tmp_path, monkeypatch, **card):
    """A calculation, described, on a machine whose probe found one a100.
    A RUN STATES ITS LAUNCH SHAPE (`architecture.md` § 5.2): the road's run
    card (`support.road.RUN_CARD`), ``card`` over it -- the calculation's,
    for every rung that does not say otherwise.  And the wall and memory a
    job handed to a scheduler states, which the benchmark's trials share:
    these tests launch them in submit mode.  WHAT TO MEASURE IS STATED
    (`generator.md` § 4.3a): three shapes the machine's one GPU node
    holds."""
    write_machine_record(topology=_NODE)
    return described(road_tree(tmp_path, monkeypatch), name="JOB",
                     execution={**RUN_CARD["siesta"], **card},
                     allocation={"time": "0-01:00:00", "mem": "8G"},
                     bench={"mpi_np": [1, 2, 4], "omp_threads": [1],
                            "gpu_count": [1]})


@pytest.fixture
def calc(tmp_path, monkeypatch):
    """A GPU calculation, described, on a machine whose probe found one a100.

    ``use_gpu=True`` is stated here rather than assumed, because it is the
    run card that decides whether this is a GPU benchmark —
    `web/task-setup.md` § 6.2, *"use GPU or not is set up only at the Job
    Prep UI"*.  Every test below is about the G × K × C grid.
    """
    # A GPU run states how many GPUs (`execution/gpu.md` G5) -- the GPU and
    # the solver with it.  The bench takes the GPU from it; its shapes and
    # GPU counts are its own grid's.
    return _described(tmp_path, monkeypatch, use_gpu=True,
                      diag_algorithm="ELPA-1STAGE", gpu_count=1)


@pytest.fixture
def cpu_calc(tmp_path, monkeypatch):
    """The same calculation, its run card on the CPU, for the tests of a
    declared grid.  A CPU run states no GPU count: one stated for a run that
    does not use the GPU is refused at prep (`execution/gpu.md` G5)."""
    return _described(tmp_path, monkeypatch, use_gpu=False,
                      diag_algorithm="ELPA-1STAGE")


#: `prep bench coarse`, as typed: this prep's ask bounding the cells.
_PREP_BENCH = ("prep", "bench", "coarse", "--np", "8", "--cpus-per-task", "8")


def _prep_bench(calc):
    """`prep bench coarse`, its trials to run where it is launched -- and the
    sweep's OWN record, in the stage's bench/ container (job-contracts.md
    § 6.3); the root job-set.json is the RUN plan."""
    r = jobset(*_PREP_BENCH, "--bundle", calc, "--no-sbatch")
    assert r.exit_code == 0, r.output
    return _plan_of(calc / "01_coarse" / "bench")


def _plan_of(folder) -> dict:
    """The folder's plan, through its door (`JobSet.of`)."""
    return JobSet.of(folder).to_dict()


def _on_a_queue(calc):
    """This machine's record names a queue -- a cluster's, the topology the
    grid is enumerated on kept, one GPU queue its node fits -- and the
    description names that queue.  A benchmark launched to a scheduler is
    planned whole, its shelves' headers with it, so a launch the machine
    could not take is refused, its dry run too (`job-system.md` § 6.0); on
    the workstation `calc` describes, `--mode submit` is."""
    write_machine_record(
        scheduler=SLURM, topology=_NODE,
        domains=[Domain.from_row(
            {"name": "gpu", "partition": "gpu", "qos": "public",
             "max_time": "1-00:00:00", "max_cores": 4, "gpu": {"a100": 1},
             "node_types": [{"cores": 4, "nodes": 1, "gpu": {"a100": 1}}]})])
    req = read_request(calc)
    req["allocation"]["domain"] = "gpu"
    write_request(calc, req)


def test_the_grid_becomes_a_sweep_jobset_of_relabelled_trials(calc):
    """Each trial nests in the stage it measures, `<NN>_<stage>/bench/
    bench-<point>/` (`job-contracts.md` § 6.3): its deck and run script real
    files there, never links (`project-layout.md` § 1.5), and nothing
    rendered at the calculation's root (L1, roadmap 7.10).  Its deck carries
    the TRIAL's own identity line (§ 2.3.2's relabel, which keys its warm
    files away from the run's), and the sweep's record points at that very
    deck."""
    js = _prep_bench(calc)
    assert js["kind"] == "sweep"
    names = [j["name"] for j in js["jobs"]]
    assert len(names) == len(set(names)) and len(names) >= 2
    assert all(n.startswith("G1N") and "C" in n for n in names)
    for job in js["jobs"]:
        n = job["name"]
        deck = _artifacts(calc, n) / f"JOB-{n}_01_coarse.fdf"
        assert deck.is_file() and not deck.is_symlink()
        assert job["script"] == deck.name
        assert not (calc / deck.name).exists()
        assert re.search(rf"^SystemLabel\s+JOB-{n}\s*$",
                         deck.read_text(), re.M)
        script = deck.with_name(f"JOB-{n}_01_coarse.run.sh")
        assert script.is_file() and not script.is_symlink()


def test_the_pins_land_as_rendered_schema_values_not_splices(calc):
    """The pins are resolved and rendered: the capped SCF, the single
    point, the eigensolver — readable in the deck."""
    js = _prep_bench(calc)
    _n0 = js['jobs'][0]['name']
    deck = (_artifacts(calc, _n0)
            / f"JOB-{_n0}_01_coarse.fdf").read_text()
    assert re.search(r"^MaxSCFIterations\s+3\s*$", deck, re.M)
    assert re.search(r"^Diag\.Algorithm\s+ELPA-1STAGE\s*$", deck, re.M)
    # running-a-job.md § 3.5: a trial's capped SCF ends clean, never aborts.
    assert re.search(r"^SCF\.MustConverge\s+\.false\.\s*$", deck, re.M)
    assert re.search(r"^Diag\.ELPA\.GPU\s+\.true\.\s*$", deck, re.M)
    assert re.search(r"^MD\.Steps\s+0\s*$", deck, re.M)


def test_each_trials_resources_carry_its_own_coordinate(calc):
    """N ranks, C cores, G GPUs in gres -- a count, no card (scheduler.md
    R2a) -- translated once, per element, never re-derived downstream
    (generator.md § 5); the coordinate read from the trial's own point,
    never cut out of its name (job-contracts.md § 6.3)."""
    js = _prep_bench(calc)
    for job in js["jobs"]:
        pt = job["point"]
        r = job["resources"]
        assert r["mpi_np"] == pt["N"]
        assert r["cpus_per_task"] == pt["C"]
        assert r["gres"] == f"gpu:{pt['G']}"


def test_cli_prep_bench_end_to_end_lists_trials_not_attempts(calc):
    """The whole verb through the CLI: trials prepared and listed, each
    named by its path from the calculation's folder -- every trial holds
    its run, `run-0`, as a stage does (`project-layout.md` § 1.5a), so a
    bare folder name would name nobody."""
    r = jobset(*_PREP_BENCH, "--bundle", calc, "--no-sbatch")
    assert r.exit_code == 0, r.output
    assert "trial dir(s) for stage 'coarse'" in r.output
    # the hint teaches the REAL grammar + the launcher (E-J2 fix,
    # 2026-08-21): grouped submission, then summarize -> run-config.
    assert "molbuilder jobset launch bench" in r.output
    # the exact hint phrase -- "per side" alone would match the vacuum
    # warning's "8 Å per side"
    assert "one job per resource shelf" in r.output
    assert "summarize bench" in r.output
    assert "config:" in r.output          # provenance rides every prep

    # AND EACH TRIAL IS NAMED BY ITS PATH FROM THE BUNDLE (O5, user yes
    # 2026-08-28).  With the attempt layer every trial's directory ends in
    # `run-<n>`, so a listing of bare names reads "run-0, run-0" and names
    # nobody.  The listing is the indented block right after its header.
    # Selected STRUCTURALLY and not by matching `bench-`: a filter that
    # looks for what correct output contains cannot see the regression.
    lines = r.output.splitlines()
    head = next(i for i, l in enumerate(lines)
                if "trial dir(s) for stage" in l)
    listed = []
    for l in lines[head + 1:]:
        if not l.startswith("  "):
            break
        listed.append(l.strip())
    assert len(listed) == 3, (
        "three trials were prepared, so three are listed; got: " + repr(listed))
    assert all("/" in l for l in listed), (
        "each trial must be named by its path from the bundle, not the bare "
        "attempt name; got: " + repr(listed))
    assert len(set(listed)) == len(listed), (
        "two trials printed the same name, which names nobody: "
        + repr(listed))


def test_a_trials_deck_prints_the_launch_of_that_trial(calc):
    """A trial's deck tells a person how to run it, as a run's does -- and
    what it names is the trial's own launch, `launch bench <stage> <trial>`
    (`job-system.md` § 5.3; plan § 5w K12).  It printed the run's line,
    which launches the stage's run, not this trial.  Typed back, the line
    plans exactly this trial.

    MUTATION THIS MUST FAIL AGAINST: prep not telling the deck it is a trial
    (the header prints `launch task --stage coarse`)."""
    _on_a_queue(calc)
    r = jobset(*_PREP_BENCH, "--bundle", calc)
    assert r.exit_code == 0, r.output
    js = _plan_of(calc / "01_coarse" / "bench")
    trial = js["jobs"][0]["name"]
    deck = (_artifacts(calc, trial) / f"JOB-{trial}_01_coarse.fdf").read_text()
    line = re.search(r"molbuilder jobset launch (bench \S+ \S+)$",
                     deck, re.M).group(1).split()
    assert line == ["bench", "coarse", trial], line
    r = jobset("launch", *line, "--bundle", calc, "--mode", "submit",
               "--dry-run", "--yes")
    assert r.exit_code == 0, r.output
    plans = [ln for ln in r.output.splitlines() if "WOULD run" in ln]
    assert len(plans) == 1 and trial in plans[0], plans


def test_cli_submit_bench_groups_the_sweep_by_shelf(calc):
    """`launch bench <stage>` under submit mode: one grouped job per
    RESOURCE SHELF (user 2026-08-21 -- an exact-fit allocation per group,
    so a narrow trial never idles a wide envelope).  The probed grid's every point has its
    own width here, so #groups == #shelves -- still one LAUNCH ACT per
    shelf, never a queue flood; the value-axis cartesian is what shares
    shelves in real matrices."""
    _on_a_queue(calc)
    r = jobset(*_PREP_BENCH, "--bundle", calc)
    assert r.exit_code == 0, r.output
    js = _plan_of(calc / "01_coarse" / "bench")
    trial = js["jobs"][0]["name"]
    r = jobset("launch", "bench", "--bundle", calc, "--mode", "submit",
               "--dry-run", "--yes")
    assert r.exit_code != 0 and "name one" in r.output
    r = jobset("launch", "bench", "coarse", "--bundle", calc,
               "--mode", "submit", "--dry-run", "--yes")
    assert r.exit_code == 0, r.output
    # one sbatch per SHELF, each an exact fit -- and the shelves submit
    # widest first, so the first planned group asks the widest -n
    shelves = {(j["resources"].get("mpi_np") or 0,
                j["resources"].get("cpus_per_task") or 0,
                j["resources"].get("gres") or "") for j in js["jobs"]}
    plans = [l for l in r.output.splitlines() if "WOULD run" in l]
    assert len(plans) == len(shelves)
    assert "bench-group" in r.output
    assert ".sbatch" in plans[0]
    # TIME IS NEVER INVENTED (user dictation, 2026-08-24): the wall is the
    # one the description states -- `allocation.time`, on every shelf's
    # header -- and one stated nowhere is refused, not sent
    # (`architecture.md` § 5.2).  A --time at launch is the wall instead,
    # restated on the line.
    headers = list(shelf_headers(calc, "coarse").values())
    assert all("#SBATCH -t 0-01:00:00" in h for h in headers), (
        f"the stated wall did not ride every shelf-job: {headers}")
    r2 = jobset("launch", "bench", "coarse", "--bundle", calc,
                "--mode", "submit", "--dry-run", "--yes", "--time", "4h")
    assert r2.exit_code == 0, r2.output
    plans2 = [l for l in r2.output.splitlines() if "WOULD run" in l]
    assert all(" -t 0-04:00:00 " in pl for pl in plans2), (
        f"--time 4h must ride every shelf-job: {plans2}")
    # widest = the CORE footprint (ranks x cores-per-rank), not ranks
    wid = max(js["jobs"], key=lambda j: (j["resources"].get("mpi_np") or 1)
              * (j["resources"].get("cpus_per_task") or 1))["resources"]
    assert (f"#SBATCH -n {wid.get('mpi_np') or 1}" in headers[0]
            and f"#SBATCH -c {wid.get('cpus_per_task')}" in headers[0]), (
        f"the widest shelf must submit first: {headers[0]}"
    )
    assert r.output.count("rides the group") == len(js["jobs"])
    assert "next unlaunched trial" not in r.output
    r = jobset("launch", "bench", "coarse", trial, "--bundle", calc,
               "--mode", "submit", "--dry-run", "--yes")
    assert r.exit_code == 0, r.output
    assert r.output.count("WOULD run") == 1
    assert "bench-group" not in r.output


def test_prep_run_of_a_second_stage_merges_the_root_plan(calc):
    """The root ``job-set.json`` is the RUN plan and MERGES per stage
    (`job-contracts.md` § 6.1): prepping `medium` must not erase `coarse` --
    erasing it broke the status rollup and silently withheld the
    cross-stage ``.CG`` carry."""
    # `medium` before `coarse` has run: it starts from the structure, said
    # out loud (`job-system.md` § 5.4) -- what this is about is the plan
    for stage in ("coarse", "medium"):
        res = jobset("prep", "task", *group_words(calc, stage),
                     "--bundle", calc, "--no-sbatch")
        assert res.exit_code == 0, res.output
    js = _plan_of(calc)
    assert js["kind"] == "ladder"
    assert [j["name"] for j in js["jobs"]] == ["coarse", "medium"]


def test_a_sweeps_record_never_touches_the_root_plan(calc):
    """Per-kind persistence (§ 6.1): `prep bench` writes the sweep's OWN
    ``job-set.json`` into the stage's container and leaves the root plan
    alone -- a run prepared afterwards is a ladder, not a sweep."""
    _prep_bench(calc)
    assert JobSet.of(calc) is None
    r = jobset("prep", "task", "--group", "1", "--bundle", calc, "--no-sbatch")
    assert r.exit_code == 0, r.output
    js = _plan_of(calc)
    assert js["kind"] == "ladder"
    assert [j["name"] for j in js["jobs"]] == ["coarse"]
    sweep = _plan_of(calc / "01_coarse" / "bench")
    assert sweep["kind"] == "sweep"


def test_every_verb_records_its_decisions_in_the_ledger(calc):
    """The bundle's decision ledger (user rule 2026-08-12): what each verb
    decided -- and from which inputs -- is READ BACK from the bundle, in
    order, when the terminal is long gone."""
    # through the VERB: the ledger is the surface's record of what it
    # decided -- the library returns decision data and never logs
    _prep_bench(calc)
    res = jobset("launch", "bench", "coarse", "--bundle", calc,
                 "--mode", "direct", "--dry-run")
    assert res.exit_code == 0, res.output
    # ...and sent to a queue this workstation has none of: refused, and the
    # refusal written down with the mode and where it came from
    res = jobset("launch", "bench", "coarse", "--bundle", calc,
                 "--mode", "submit", "--yes")
    assert res.exit_code != 0 and "no scheduler header" in res.output, \
        res.output
    res = jobset("summarize", "bench", "coarse", "--bundle", calc)
    assert res.exit_code == 0, res.output
    lines = ledger.read(calc)
    got = [(e["verb"], e["decision"]) for e in lines]
    # A DRY RUN WRITES NOTHING, the ledger included (`job-system.md` § 6.0,
    # step 3).  The state prep saved first comes first
    # (`checkpointing.md` § 9).
    assert got == [("prep", "saved"),
                   ("prep", "prepared"),
                   ("launch", "refused"),
                   ("summarize", "verdict-written")]
    assert lines[0]["new"] is True and " · before prep bench coarse" in \
        lines[0]["note"], lines[0]
    prep = lines[1]
    assert prep["kind"] == "bench" and prep["stage"] == "coarse"
    assert "provenance" in prep            # WHERE each setting came from
    launch = lines[2]
    assert launch["mode"] == "submit"
    assert launch["mode_source"] == "--mode flag"
    assert "no scheduler header" in launch["reason"], launch


def test_each_trials_wrapper_carries_its_own_translated_launch(calc):
    """The STOMP's true regression pin (U1 fixed it at the job-set level;
    this pins the WRAPPER TEXT, where the bug actually bit): every
    trial's .run.sh bakes ITS OWN rank default, N, not the base
    allocation's -- a stomped wrapper launches every trial at the same
    np and the benchmark measures nothing."""
    js = _prep_bench(calc)
    seen = set()
    for j in js["jobs"]:
        name = j["name"]
        n = j["point"]["N"]
        w = (_artifacts(calc, name)
             / f"JOB-{name}_01_coarse.run.sh").resolve().read_text()
        # ANCHORED to the assignment lines: the GPU policy block's
        # ``_gpu_mpi_np_default=4`` contains this name as a substring,
        # which is exactly what an unanchored search matched first.
        vals = re.findall(r"^\s*_mpi_np_default=(\d+)$", w, re.M)
        assert vals, "wrapper carries no baked rank default"
        assert all(int(v) == n for v in vals), (name, vals)
        seen.add(n)
    assert len(seen) > 1, "all wrappers share one rank count -- the stomp"


class TestTheRunsOwnCondition:
    """`stages.md` § 6.8d — `execution` is the ONE condition a run uses, and
    it is INDEPENDENT of `bench`.

    *(user, 2026-09-02: "run parameter is independent from bench grid or bench
    result.  User can run without bench… bench and run share the same
    framework to choose/set parameters but run does not produce grid but a
    single user defined condition to run."  And: "bench is bench and run is
    run.  They are structurally separated in their dir and they are
    functionally different as the user decides to use either.")*

    A design that made a one-point `bench` row the run's shape shipped on
    2026-09-01 and was wrong for one reason that no test then covered: it
    made the two share a block, so `{mpi_np: [4, 8, 16]}` could not coexist
    with *"run at 8"* — narrowing the row to say what the run uses DESTROYS
    the plan to measure.  The first test here is that case.

    Driven through the CLI, because the producer is the CLI's.
    """

    BENCH = {"mpi_np": [4, 8, 16], "omp_threads": [1, 2]}

    @staticmethod
    def _write(calc, **doc):
        req = read_request(calc)
        req["allocation"] = {"time": "0-01:00:00"}
        req.update(doc)
        write_request(calc, req)

    @staticmethod
    def _run(calc, stage="coarse", *extra):
        """Prep one stage's run and return THAT stage's job.

        The root plan MERGES across stages (`test_prep_run_of_a_second_stage_
        merges_the_root_plan`), so `jobs[0]` is whichever stage was prepared
        first -- which is how a per-stage assertion silently reads the
        previous rung's numbers."""
        r = jobset("prep", "task", *group_words(calc, stage), "--bundle", calc,
                   "--no-sbatch", *extra)
        assert r.exit_code == 0, r.output
        jobs = _plan_of(calc)["jobs"]
        mine = [j for j in jobs if stage in j["name"]]
        assert len(mine) == 1, [j["name"] for j in jobs]
        return mine[0]

    def test_a_grid_and_a_run_condition_COEXIST(self, calc):
        """The case the conflated design could not express at all."""
        self._write(calc, bench=self.BENCH,
                    execution={"mpi_np": 2, "omp_threads": 2,
                               "use_gpu": True, "gpu_count": 1})
        r = self._run(calc)["resources"]
        assert (r["mpi_np"], r["cpus_per_task"]) == (2, 2)
        # AND THE PLAN TO MEASURE SURVIVES, byte for byte
        assert read_request(calc)["bench"] == self.BENCH

    def test_a_run_needs_no_bench_at_all(self, calc):
        """*"User can run without bench."*  Not executed, not summarized,
        not even declared."""
        req = read_request(calc)
        del req["bench"]
        write_request(calc, req)
        self._write(calc, execution={"mpi_np": 2, "omp_threads": 2,
                                     "use_gpu": True, "gpu_count": 1})
        r = self._run(calc)["resources"]
        assert (r["mpi_np"], r["cpus_per_task"]) == (2, 2)

    def test_a_bench_alone_sizes_no_run(self, calc):
        """The other half of the separation: a grid is a plan to measure and
        says nothing about the run, whatever its arity.  With no condition
        the run's shape is stated nowhere, and prep refuses it -- the grid
        does not stand in (`architecture.md` § 5.2)."""
        for bench in (self.BENCH, {"mpi_np": [8], "omp_threads": [2]}):
            # ...the run on the GPU, its launch shape stated nowhere
            self._write(calc, bench=bench, execution={"use_gpu": True})
            # The GPU count on the prep: this calculation runs on a GPU, and
            # a GPU run states how many (`gpu.md` G5) -- a count, not a shape.
            r = jobset("prep", "task", "--group", "1", "--bundle", calc,
                       "--no-sbatch", "--gpus", "1")
            assert r.exit_code != 0 and "ranks (mpi_np)" in r.output, (
                bench, r.output)

    def test_a_stage_lays_its_own_over_the_calculations(self, calc):
        """Both scopes, FIELD BY FIELD: a rung naming only `mpi_np` keeps the
        calculation's thread count."""
        stages = read_request(calc)["stages"]
        first_name, second_name = list(stages)[:2]
        stages[first_name]["execution"] = {"mpi_np": 2}
        self._write(calc, execution={"mpi_np": 1, "omp_threads": 2,
                                     "use_gpu": True, "gpu_count": 1},
                    stages=stages)
        first = self._run(calc, first_name)["resources"]
        assert (first["mpi_np"], first["cpus_per_task"]) == (2, 2)
        # the second before the first has run: from the structure, said out
        # loud (`job-system.md` § 5.4)
        second = self._run(calc, second_name)["resources"]
        assert (second["mpi_np"], second["cpus_per_task"]) == (1, 2)

    def test_the_whole_condition_reaches_the_launch_and_the_deck(self, calc):
        """One block, two destinations, split by the catalogue's own answer:
        machine items become the launch shape, the rest pin the template."""
        self._write(calc, execution={"mpi_np": 2, "omp_threads": 2,
                                     "gpu_count": 1, "use_gpu": True,
                                     "diag_algorithm": "ELPA-2STAGE"})
        job = self._run(calc)
        r = job["resources"]
        assert (r["mpi_np"], r["cpus_per_task"]) == (2, 2)
        # a COUNT -- no card is named (`scheduler.md` R2a)
        assert r["gres"] == "gpu:1", r["gres"]
        deck = next((calc / "01_coarse").glob("*.fdf")).read_text()
        assert "Diag.Algorithm     ELPA-2STAGE" in deck

    def test_a_surface_with_no_flags_passes_an_EMPTY_ask(self, calc):
        """§ 6.0a's contract, and the crash it was written after.

        The browser has no flags, so it hands the assembly nothing -- and
        `None` is not nothing, it is the absence of the object the verdict is
        folded UNDER field by field.  The tab's prep button died with
        ``'NoneType' object has no attribute 'mpi_np'`` on 2026-09-02 for
        exactly this.  Asked of the tab's own door; the terminal's empty ask
        is every `_run` here."""
        from molbuilder.web.app import create_app
        self._write(calc, execution={"mpi_np": 2, "omp_threads": 2,
                                     "use_gpu": False})
        answer = create_app(config={}).test_client().post(
            "/api/task-setup/prep",
            json={"dest": str(calc), "kind": "task", "groups": [1],
                  "plan": True})
        assert answer.status_code == 200, answer.get_data(as_text=True)
        said = answer.get_json()["answers"][0]["launch"]["run_script"]
        assert said == [{"name": "mpi_np", "value": "2"},
                        {"name": "omp_threads", "value": "2"}], said

    def test_the_RUN_owns_its_wall_and_queue(
            self, calc):
        """`stages.md` § 6.8e -- the one rung `time` and `domain` gained.

        `allocation` is folded by the shared prep path, so `prep bench` and
        `prep task` read the SAME wall.  One number cannot serve both: four
        hours kills a two-day run, and two days makes every ten-minute trial
        queue behind everything -- so a benchmark, the thing you run to save
        time, waits a day to start.

        Three claims, because each can break alone:
          * the run takes `execution`'s wall and queue
          * a rung with no `execution` keeps the calculation's
          * `mem` is NOT among them and stays the calculation's
        """

        req = read_request(calc)
        req["allocation"] = {"time": "0-04:00:00", "domain": "debug",
                             "mem": "256G"}
        req["stages"]["coarse"]["execution"] = {"mpi_np": 64,
                                                "time": "2-00:00:00",
                                                "domain": "public"}
        write_request(calc, req)

        # THE ALLOCATION EACH JOB IS WRITTEN WITH -- its row in the plan,
        # `job-set.json`, the stage resolved and its allocation folded once
        # (`prep._resolve_stage`).
        run = self._run(calc, "coarse")["resources"]
        assert run["time"] == "2-00:00:00", f"the run lost its wall: {run}"
        assert run["domain"] == "public", f"the run lost its queue: {run}"
        assert run["mem"] == "256G", "mem does not ride `execution`"

        other = self._run(calc, "medium")["resources"]
        assert other["time"] == "0-04:00:00", "a rung with no `execution` "\
            "of its own must keep the calculation's"

    def test_a_flag_still_wins_field_by_field(self, calc):
        """`generator.md` § 4.3a: template < a bench pin < run-config <
        `execution` < a flag.  A person typing a number now is answering
        about now -- and must not erase the device ask nobody mentioned."""
        self._write(calc, execution={"mpi_np": 2, "omp_threads": 2,
                                     "gpu_count": 1, "use_gpu": True})
        r = self._run(calc, "coarse", "--cpus-per-task", "16")["resources"]
        assert r["cpus_per_task"] == 16, "the flag lost to the description"
        assert r["gres"] == "gpu:1", "the flag erased the device ask"

    def test_prep_writes_a_big_ask_and_does_not_second_guess_it(self, calc):
        """`generator.md` § 4.3: what you asked for is written, or refused --
        never trimmed.  Where a header is written, it is placed on the queue
        it names and its fit said (`scheduler.md` R6); with no header, as
        here (`--no-sbatch`), nothing is placed, and

        `prep` writes what you asked for, exactly as it does for `--np 999`.
        It refused here for one afternoon on 2026-09-02, because the run's
        shape was being produced BY the sweep's enumerator, whose job is to
        leave cells out -- so a run inherited a bench's verdict on its own
        numbers, in the bench's words, for a legal `max_memory_mb` among
        others."""
        self._write(calc, execution={"mpi_np": 64, "omp_threads": 8,
                                     "use_gpu": False})
        r = self._run(calc)["resources"]
        assert (r["mpi_np"], r["cpus_per_task"]) == (64, 8), (
            "prep second-guessed a number it was handed")

    def test_every_machine_item_is_carried_not_just_the_grids_three(self, calc):
        """The sweep resolves `mpi_np`, `omp_threads` and `gpu_count` and
        refuses anything else as an axis it does not know.  A CONDITION is not
        a sweep: every machine-answered catalogue item with a `Resources`
        field is carried, `max_memory_mb` among them -- it is settable as
        `--max-memory-mb`, so a description may say it too."""
        self._write(calc, execution={"mpi_np": 2, "omp_threads": 1,
                                     "max_memory_mb": 4000,
                                     "use_gpu": False})
        r = self._run(calc)["resources"]
        assert r["mpi_np"] == 2
        assert r["max_memory_mb"] == 4000


def test_stage_plan_records_the_config_provenance(calc):
    """STAGE-PLAN.md is the reviewable record; the provenance block
    (which file supplied each setting) is what makes a machine
    difference debuggable from the plan alone (user rule 2026-08-12)."""
    _prep_bench(calc)
    plan = (calc / "01_coarse" / "bench" / "STAGE-PLAN.md").read_text()
    assert "config:" in plan
    assert "molbuilder.json" in plan


def test_a_one_stage_calculation_is_one_named_stage(tmp_path, monkeypatch):
    """R1 (review-4 keystone): the single-parameter-set calculation is a
    FIRST-CLASS run, one NAMED, tokened stage (§ 6.5): prep, launch's plan
    and status all reach it, one rung deep, its run holding real files.
    What a launch writes as its run starts -- its launch
    record -- is read off a real run (plan § 5y)."""
    dest = described(road_tree(tmp_path, monkeypatch), name="JOB",
                     stage_strategy="")
    # § 6.5: the bare verb does NOT guess the lone stage.  A rule that held
    # only at length one would stop holding the moment a second stage was
    # added -- so one rung refuses exactly as three do, listing the ladder.
    res = jobset("prep", "task", "--bundle", dest, "--no-sbatch")
    assert res.exit_code != 0, res.output
    assert "coarse" in res.output, res.output
    assert not (dest / "01_coarse").exists(), \
        "the bare verb guessed the lone stage and prepared it"

    res = jobset("prep", "task", "--group", "1", "--bundle", dest,
                 "--no-sbatch")
    assert res.exit_code == 0, res.output
    rung = dest / "01_coarse"                 # the one rung, tokened
    assert (rung / "run-0").is_dir()          # its first run
    assert not (dest / "run-0").exists()      # never at the root
    # ITS NEXT LINE, the launch with nothing named (`job-system.md`, *The
    # task*): it sends the group that can go.
    assert "launch task --mode" in res.output
    # THE RUN HOLDS REAL FILES (`project-layout.md` § 1.5: every file goes
    # in as a real file, never a link).
    entries = sorted((rung / "run-0").iterdir())
    assert entries, "the run is empty"
    assert all(p.is_file() and not p.is_symlink() for p in entries), [
        p.name for p in entries if not p.is_file() or p.is_symlink()]
    js = _plan_of(dest)
    assert js["kind"] == "ladder" and len(js["jobs"]) == 1
    res = jobset("launch", "task", "--stage", "coarse", "--bundle", dest,
                 "--mode", "direct", "--dry-run", "--yes")
    assert res.exit_code == 0, res.output
    assert res.output.count("WOULD start") == 1
    res = jobset("status", "--bundle", dest)
    assert res.exit_code == 0, res.output


def test_a_one_stage_calculation_can_be_benchmarked(tmp_path, monkeypatch):
    """A4 (redo 2026-08-12): a one-stage calculation can be benchmarked,
    and it is benchmarked the way every rung is -- ``prep bench <stage>``,
    trials in THAT stage's container."""
    dest = described(road_tree(tmp_path, monkeypatch), name="JOB",
                     stage_strategy="",
                     bench={"mpi_np": [1, 2], "omp_threads": [1]})
    res = jobset("prep", "bench", "coarse", "--bundle", dest, "--no-sbatch")
    assert res.exit_code == 0, res.output
    assert "None" not in res.output, res.output
    bench = dest / "01_coarse" / "bench"
    js = _plan_of(bench)
    assert js["kind"] == "sweep" and len(js["jobs"]) >= 2
    # trial decks carry the rung's token like any other stage's trials
    for j in js["jobs"]:
        assert (_artifacts(dest, j["name"]) / j["script"]).is_file()
        assert "01_coarse" in j["script"], j["script"]
    # A-2 (2026-08-13): the trials live IN their stage's bench container,
    # beside the record just read -- not at the root, where the
    # underway-ask never looked and a re-prep silently re-rendered the
    # decks a queued trial's links point at.
    for j in js["jobs"]:
        assert (bench / f"bench-{j['name']}").is_dir()
        assert not (dest / f"bench-{j['name']}").exists()
    # a NAMED trial submits alone (how a single point is re-run); an
    # unnamed bench submits one grouped job per resource shelf -- one
    # LAUNCH ACT per shelf.
    t0, t1 = js["jobs"][0]["name"], js["jobs"][1]["name"]
    res = jobset("launch", "bench", "coarse", t1, "--bundle", dest,
                 "--mode", "direct", "--dry-run", "--yes")
    assert res.exit_code == 0, res.output
    would = [l for l in res.output.splitlines() if "WOULD start" in l]
    assert len(would) == 1, res.output
    assert t1 in would[0] and t0 not in would[0], res.output
    # an unknown stage is refused with the ladder listed, not ignored
    res = jobset("summarize", "bench", "not-a-stage", "--bundle", dest)
    assert res.exit_code != 0 and "not-a-stage" in res.output
    # ...and with its one stage's bench prepared, there is none to offer
    # (a prepared stage is not prepared again, `job-system.md` § 5.0)
    res = jobset("prep", "bench", "--bundle", dest, "--no-sbatch")
    assert res.exit_code != 0, res.output
    assert "every stage is prepared already" in res.output, res.output


# --------------------------------------------------------------------- #
#  The declared grid drives the sweep (roadmap § 0.1 B1)                 #
# --------------------------------------------------------------------- #
# generator.md § 4.3a: the job request's `bench` DECLARES what to measure —
# portable points, resolved here.


def test_the_declared_grid_is_the_sweep(cpu_calc):
    """Declared axes produce exactly those points — nothing enumerated."""
    declare_bench(cpu_calc, {"mpi_np": [1, 2], "omp_threads": [1]})
    assert [j["point"] for j in _prep_bench(cpu_calc)["jobs"]] == [
        {"G": 0, "N": 1, "C": 1}, {"G": 0, "N": 2, "C": 1}]


def test_a_one_point_declaration_is_a_pin_and_decides_the_grid(cpu_calc):
    """The override lane's ONE-point half (user rule, 2026-08-20): a CPU
    description with `bench: {use_gpu: [true]}` enumerates the GPU grid
    -- the declaration overrides the template's answer -- and the value
    rides every trial as a pin, under the measurement pins: executed, not
    assumed, the pinned eigensolver lands in the rendered trial deck,
    replacing the run card's (`ELPA-1STAGE`)."""
    declare_bench(cpu_calc, {"use_gpu": [True],
                              "diag_algorithm": ["ELPA-2STAGE"],
                              "mpi_np": [4], "omp_threads": [1],
                              "gpu_count": [1]})
    jobs = _prep_bench(cpu_calc)["jobs"]
    assert [j["point"] for j in jobs] == [{"G": 1, "N": 4, "C": 1}], (
        "a declared use_gpu=true must enumerate the GPU grid")
    name = jobs[0]["name"]
    deck = (_artifacts(cpu_calc, name)
            / f"JOB-{name}_01_coarse.fdf").read_text()
    # The VALUE lines, not the catalogue help comments -- those name every
    # choice, so a substring match passes with the pin broken.
    assert re.search(r"^Diag\.ELPA\.GPU\s+\.true\.\s*$", deck, re.M)
    assert re.search(r"^Diag\.Algorithm\s+ELPA-2STAGE\s*$", deck, re.M)
    assert re.search(r"^MaxSCFIterations\s+3\s*$", deck, re.M), (
        "the measurement pins still ride")


# --------------------------------------------------------------------- #
#  The summary CONNECTS to the run (roadmap § 0.1 B3/B5) and the offer   #
#  explains an empty verdict (B4)                                        #
# --------------------------------------------------------------------- #


def test_a_verdictless_summarize_prints_no_report(calc):
    """No verdict, no report — a report would recommend a shape with nothing
    measured behind it, and a person reading one cannot tell that from a
    measurement."""
    _prep_bench(calc)
    r = jobset("summarize", "bench", "coarse", "--bundle", calc)
    assert r.exit_code == 0, r.output
    # `generator.md` § 4.3a, *The summary*: no winner is said, with why
    assert "no winner: no finished, timed trial to rank (" in r.output
    assert "bench recommendation" not in r.output


def test_a_bench_row_never_reaches_the_run_deck(cpu_calc):
    """A one-point NON-machine `bench` declaration is what the TRIALS run
    with, and only them (user, 2026-09-30, narrowing the 2026-08-20 rule;
    `generator.md` § 4.3a, `stages.md` § 6.8d): the run's values are its own
    run card's.  Until 2026-09-30 `prep task` rendered the row into the run's
    deck -- a second home for a value the card states.
    """
    declare_bench(cpu_calc, {"diag_algorithm": ["ELPA-2STAGE"]})
    r = jobset("prep", "task", "--group", "1", "--bundle", cpu_calc)
    assert r.exit_code == 0, r.output
    deck = next((cpu_calc / "01_coarse").glob("run-*/JOB*.fdf"))
    assert not re.search(r"^Diag\.Algorithm\s+ELPA-2STAGE",
                         deck.read_text(), re.M), (
        "the bench row reached the run deck (the value line, not the "
        "help comments)")


def _bench_dir(calc):
    return calc / "01_coarse" / "bench"


def _load_sweep(calc):
    """The jobset, and the BUNDLE it belongs to -- resolved the way the
    route does it (`materialize.bench_owner`), which is NOT the file's own
    directory."""
    container = _bench_dir(calc)
    jobset = JobSet.of(container)
    return jobset, bench_owner(container)[0]


def test_sweep_view_has_one_trial_per_job_in_the_job_sets_order(calc):
    js = _prep_bench(calc)
    jobset, bundle = _load_sweep(calc)
    view = sweep_view(open_calculation(bundle), jobset)
    assert view["n_trials"] == len(js["jobs"])
    assert [t["label"] for t in view["trials"]] == [j["name"] for j in js["jobs"]]


def test_sweep_view_reports_where_the_RUN_is(calc):
    """A trial's state is the run door's -- `queued/running/finished/
    failed`, the one rule every surface says (`model/parse.md` § 2b) -- and
    the view carries that one word.  Until 2026-10-18 a second word read off
    the output alone (`artifacts`: `completed` once `>> End of run` was
    printed) stood beside it, and the ranking read that one, so a trial
    that failed after its output ended could be crowned.
    """
    _prep_bench(calc)
    jobset, bundle = _load_sweep(calc)
    view = sweep_view(open_calculation(bundle), jobset)
    t = view["trials"][0]
    assert "state" in t and "artifacts" not in t
    # Nothing has been launched, so the RUN has not started: the door's
    # word, never `unknown` and never `completed` -- and the page's count of
    # trials running is 0, never a benchmark called *still running* for not
    # having finished.
    assert t["state"] == "pending", t["state"]
    assert view["n_running"] == view["n_queued"] == 0


def test_sweep_view_names_the_coordinate_the_sweep_varied(calc):
    _prep_bench(calc)
    jobset, bundle = _load_sweep(calc)
    view = sweep_view(open_calculation(bundle), jobset)
    # every trial declares a coordinate, and the sweep varied at least one
    assert all(isinstance(t["point"], dict) for t in view["trials"])
    assert view["varied"], "a sweep that varied something must say which"
    for k in view["varied"]:
        seen = {repr(t["point"].get(k)) for t in view["trials"]}
        assert len(seen) > 1, f"{k!r} is listed as varied but never changes"


def test_sweep_view_never_writes_the_record_or_the_proposal(calc):
    """Safe to call while the sweep is still running -- which is exactly
    when someone watches it, and the page polls every 15 s (B4).

    `run_summarize_jobset` is the verb that WRITES: ``bench-result.json``
    and, beside it, ``run-config.toml`` -- and that proposal is the USER's
    file, kept rather than overwritten once it exists.  A view that wrote
    either from a poll would be publishing a record nobody asked for, and
    could race the very file the user is editing.

    It is NOT "touches no bytes at all": ``jobset_status`` decodes each run
    directory, and every parser appends its documented ``<input>.parse.log``
    sidecar (``parse/_log.py``: default ON, append mode, *"re-parses (e.g.
    /api/watch/data polls) accumulate history"*).  The trajectory viewer's
    own polling already does that; pinning "no new files at all" would pin
    a promise this stack does not make.
    """
    _prep_bench(calc)
    jobset, bundle = _load_sweep(calc)
    bench = _bench_dir(calc)
    assert not (bench / "bench-result.json").exists()
    sweep_view(open_calculation(bundle), jobset)
    assert not (bench / "bench-result.json").exists(), (
        "the view published a record; that is run_summarize_jobset's job")
    assert not (bench / "run-config.toml").exists(), (
        "the view wrote a proposal -- that file is the user's")


def test_the_terminal_says_each_warning_once_across_a_sweep(calc):
    """O5 (user yes, 2026-08-28): the science gate FIRES per trial — every
    trial's own `.validation.txt` carries its findings — but the terminal
    said the same three lines once per trial, sixteen times at 16 trials.
    One line per unique finding now, with the suppression said out loud."""
    said = jobset(*_PREP_BENCH, "--bundle", calc, "--no-sbatch").output
    warn_lines = [l.strip() for l in said.splitlines()
                  if l.strip().startswith(("warn", "info"))]
    assert warn_lines, ("the bench pins (SCF capped, steps zeroed) must "
                        "produce warnings, or this test tests nothing")
    assert len(warn_lines) == len(set(warn_lines)), (
        "the same finding printed more than once:\n" +
        "\n".join(l for l in warn_lines if warn_lines.count(l) > 1))
    assert "each warning shown once" in said, (
        "suppression must be SAID, not silent")
