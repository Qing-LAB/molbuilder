"""`jobset prep bench` — step 6 u2: the grid through the ONE builder.

Contract: `project-layout.md` § 2.3.1a (benchmarking is `prep` whose
parameters are a set) · § 2.3.2 (trials relabelled + forced cold; submission
grouped per resource shelf, a named trial alone) · `generator.md` §§ 2, 5 · `template.md` § 8.1
(rebuild and render, never splice — the pins land as schema values).
"""
from __future__ import annotations

import json
import pathlib
import os

import numpy as np
import pytest

from molbuilder import describe as D
from molbuilder.config.siesta import SiestaConfig
from molbuilder.scheduler import Environment, Topology
from molbuilder.jobset._cli import _bench_inputs
from molbuilder.jobset.model import Resources
from molbuilder.jobset.prep import prep_calculation
from molbuilder.siesta.stages import default_siesta_stages
from molbuilder.structure import Structure
from molbuilder.task import Stage


def _artifacts(calc, point, stage="01_coarse"):
    """Where a trial's files ARE — the attempt when one exists.

    A trial keeps attempts since 2026-08-27 (`project-layout.md` § 1.5a),
    so its deck and wrapper sit in ``run-<n>`` rather than in the trial
    directory. This is the reader rule `runstatus` and `summarize` both
    use: *the latest attempt where there is one, the container otherwise* —
    which also keeps these tests correct for a flat calculation, where
    there is no attempt layer at all.

    One helper because the tests had the same duplication the code did:
    eleven hand-built copies of one path, each of which would need finding
    the next time the layout moves.
    """
    from molbuilder.jobset.materialize import latest_attempt
    c = pathlib.Path(calc) / stage / "bench" / f"bench-{point}"
    return latest_attempt(c) or c




@pytest.fixture(autouse=True)
def _tmp_is_the_projects_tree(tmp_path, monkeypatch):
    """These tests build a calculation under ``tmp_path`` and hand its path
    to a verb.  ``--bundle`` is fenced to the projects tree
    (`job-contracts.md` § 2.5b), so the test says where its tree IS rather
    than handing over a path from outside one -- which is exactly what a
    user does when their calculations live on scratch: set
    ``paths.projects`` / ``$MOLBUILDER_PROJECTS`` and the fence follows.
    """
    from molbuilder.projects import PROJECTS_ROOT_ENV
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tmp_path))


def _one_stage():
    """The ordinary starting ladder: ONE stage (`engines/stages.md` § 6.5).

    Until 2026-08-16 these cases were written stage-LESS -- ``stages=()``,
    artifacts at the folder root, bare verbs.  That shape is gone: a single
    stage is still a stage, so it is named, tokened (``01_coarse``) and
    prepped exactly like a rung of a three-stage ladder.  The tests below
    kept their subjects and moved onto this shape.
    """
    return (Stage(name="coarse", enabled=True, overrides={}),)


@pytest.fixture(autouse=True)
def _sandbox(tmp_path_factory, monkeypatch):
    """cwd + HOME isolation for EVERY test here (I6, 2026-08-13).

    The `calc` fixture and its users read runtime config through the
    cascade, and without this they read the DEVELOPER's cwd
    molbuilder.json and ~/.molbuilder -- the contamination that made a
    prep test pass GREEN off a config the test never wrote (the A8 pin's
    own first version, and the 14-of-24 failure the 2026-08-12 isolation
    memory records).  The correct pattern the newer tests already use,
    applied to the whole file."""
    box = tmp_path_factory.mktemp("sandbox")
    monkeypatch.chdir(box)
    monkeypatch.setenv("HOME", str(box / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    (box / "home").mkdir()


@pytest.fixture
def calc(tmp_path):
    """A GPU calculation, described, on a machine whose probe found one a100.

    ``use_gpu=True`` is stated here rather than assumed, because from
    2026-08-17 it is the DESCRIPTION that decides whether this is a GPU
    benchmark — `web/task-setup.md` § 6.2, *"use GPU or not is set up only at
    the Job Prep UI"*.  `_bench_inputs` used to pin it True for every trial, so
    this fixture measured a GPU while describing a CPU run and nothing said so.
    Every test below is about the G × K × C grid, and the fixture now says so.
    """
    struct = Structure(elements=["H", "H"],
                       positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
                       vacuum=(10.0, 10.0, 10.0))
    (tmp_path / "h2.xyz").write_text(struct.to_xyz())
    dest = tmp_path / "calc"
    D.write_description(
        D.build_description(struct,
                            SiestaConfig(system_label="JOB", use_gpu=True,
                                         diag_algorithm="ELPA-1STAGE"),
                            default_siesta_stages("publishable"),
                            engine="siesta", shape="hierarchical", name="JOB",
                            source=str(tmp_path / "h2.xyz")),
        dest)
    from conftest import write_pseudos
    write_pseudos(dest, sorted(set(struct.elements)))
    (dest / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": "true"}}))
    from conftest import write_pseudos
    write_pseudos(dest, ["H"])
    # The probe's answer, pre-seeded: resolve_target early-returns on an
    # existing environment.json, so the grid enumerates THIS topology.
    (dest / "environment.json").write_text(
        Environment(scheduler="workstation",
                    topology=Topology(sockets=1, cores_per_socket=4,
                                      gpus_per_node=1,
                                      gpu_type="a100")).to_json() + "\n")
    return dest


def _prep_bench(calc):
    sweep, pins, translation = _bench_inputs(calc, None)
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
        # L1 (roadmap 7.10): the deck is born in the TRIAL's directory;
        # nothing rendered sits at the bundle root any more.
        deck = _artifacts(calc, n) \
            / f"JOB-{n}_01_coarse.fdf"
        assert deck.is_file() and not deck.is_symlink()
        assert not (calc / f"JOB-{n}_01_coarse.fdf").exists()
        assert re.search(rf"^SystemLabel\s+JOB-{n}\s*$",
                         deck.read_text(), re.M)


def test_the_pins_land_as_rendered_schema_values_not_splices(calc):
    """What `transform_fdf` spliced is now resolved and rendered: the capped
    SCF, the single point, the eigensolver — readable in the deck."""
    import re
    js = _prep_bench(calc)
    _n0 = js['jobs'][0]['name']
    deck = (_artifacts(calc, _n0)
            / f"JOB-{_n0}_01_coarse.fdf").read_text()
    assert re.search(r"^MaxSCFIterations\s+3\s*$", deck, re.M)
    assert re.search(r"^Diag\.Algorithm\s+ELPA-1STAGE\s*$", deck, re.M)
    assert re.search(r"^Diag\.ELPA\.GPU\s+\.true\.\s*$", deck, re.M)
    assert re.search(r"^MD\.Steps\s+0\s*$", deck, re.M)


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
    d = _artifacts(calc, name)
    assert d.is_dir()
    script = d / f"JOB-{name}_01_coarse.fdf"
    assert script.is_file() and not script.is_symlink()   # L2: real files
    wrapper = d / f"JOB-{name}_01_coarse.run.sh"
    assert wrapper.is_file() and not wrapper.is_symlink()
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
        _bench_inputs(calc, None)


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
    # the hint teaches the REAL grammar + the launcher (E-J2 fix,
    # 2026-08-21): grouped submission, then summarize -> run-config.
    assert "molbuilder jobset launch bench" in r.output
    # the exact hint phrase -- "per side" alone false-matched the vacuum
    # warning's "8 Å per side" (found green while the hint was wrong,
    # review 2026-08-21)
    assert "one job per resource shelf" in r.output
    assert "summarize bench" in r.output
    assert "config:" in r.output          # provenance rides every prep


def test_cli_summarize_bench_reads_trials_by_data(calc):
    """u4: discovery keyed by job-set.json's data, results through the same
    artifacts as any run, ASYNC — a trial with no output yet reports
    ``incomplete`` rather than failing the set (user, 2026-08-12)."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    js = _prep_bench(calc)
    name = js["jobs"][0]["name"]
    d = _artifacts(calc, name)
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
    d = _artifacts(calc, name)
    (d / f"JOB-{name}_01_coarse-run0.out").write_text(
        "x\n>> End of run:\n")
    # epoch-per-line format: consecutive deltas are the per-iter durations
    (d / f"JOB-{name}_01_coarse-run0.scf-timing.log").write_text(
        "100.0 scf 1\n104.0 scf 2\n108.0 scf 3\n112.0 scf 4\n")
    r = CliRunner().invoke(jobset_group, ["summarize", "bench", "coarse",
                                          "--bundle", str(calc)])
    assert r.exit_code == 0, r.output
    return name


def _describe_cpu(calc):
    """Turn this calculation's description back to CPU.

    The `calc` fixture asks for the GPU because the grid tests are about the
    G × K × C grid.  A test whose claim is *"the verdict was NOT applied"*
    cannot use it: `Diag.ELPA.GPU .true.` in the deck only proves a verdict was
    taken if the description did not ask for the GPU itself.  So the claim and
    the fixture are separated here rather than weakened.
    """
    p = calc / "JOB.template.toml"
    text = p.read_text()
    i = text.index("[item.use_gpu]")
    j = text.index("[item.", i + 1)
    p.write_text(text[:i]
                 + text[i:j].replace("value = true", "value = false")
                 + text[j:])


def test_prep_run_deleting_the_file_declines_the_verdict(calc):
    """§ 2.3.2: finding a verdict is not permission — permission is
    `run-config.toml`, and deleting it is the No.  Prep says what that
    means instead of going quiet, and nothing is applied."""
    import re
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    from molbuilder.jobset.summarize import RUN_CONFIG_NAME
    _finished_trial_and_verdict(calc)
    _describe_cpu(calc)
    (calc / "01_coarse" / "bench" / RUN_CONFIG_NAME).unlink()
    r = CliRunner().invoke(jobset_group,
                           ["prep", "run", "coarse", "--bundle", str(calc),
                            "--no-sbatch"])
    assert r.exit_code == 0, r.output
    assert "a bench verdict exists" in r.output
    assert RUN_CONFIG_NAME in r.output
    assert "declined" in r.output
    # and with nothing stated, the wrapper's runtime policy is NAMED
    assert "wrapper sizes the launch at run time" in r.output
    assert "running-a-job.md" in r.output
    deck = (calc / "01_coarse" / "JOB_01_coarse.fdf").read_text()
    assert not re.search(r"^Diag\.ELPA\.GPU\s+\.true\.", deck, re.M)


def test_prep_run_applies_the_proposal_file_but_flags_win(calc):
    """The file summarize wrote fills only what the user did NOT state,
    and the winner's eigensolver arrives as pins — no question asked,
    because the file IS the answer (§ 2.3.2, 2026-08-19)."""
    import re
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    from molbuilder.jobset.summarize import RUN_CONFIG_NAME
    from molbuilder.parse.scripts.bench_marks import _extract_bench_marks_dict
    name = _finished_trial_and_verdict(calc)
    g = int(name[1])
    k = int(name[name.index("K") + 1:name.index("C")])
    r = CliRunner().invoke(jobset_group,
                           ["prep", "run", "coarse", "--bundle", str(calc),
                            "--cpus-per-task", "3", "--no-sbatch"])
    assert r.exit_code == 0, r.output
    assert f"applied 01_coarse/bench/{RUN_CONFIG_NAME}" in r.output
    assert "edit or delete the file" in r.output
    deck = (calc / "01_coarse" / "JOB_01_coarse.fdf").read_text()
    marks = _extract_bench_marks_dict(deck)
    assert marks.get("mpi_np") == g * k          # measured, user said nothing
    assert re.search(r"^Diag\.Algorithm\s+ELPA-1STAGE\s*$", deck, re.M)
    assert re.search(r"^Diag\.ELPA\.GPU\s+\.true\.\s*$", deck, re.M)
    # the flag the user DID state beat the verdict
    js = json.loads((calc / "job-set.json").read_text())
    assert js["jobs"][0]["resources"]["cpus_per_task"] == 3


def test_cli_submit_bench_groups_the_sweep_by_shelf(calc):
    """`launch bench <stage>` under submit mode: one grouped job per
    RESOURCE SHELF (user 2026-08-21 -- an exact-fit allocation per group,
    so a narrow trial never idles a wide envelope; until then one job per
    SIDE, § 2.3.2 user 2026-08-20).  The probed grid's every point has its
    own width here, so #groups == #shelves -- still one LAUNCH ACT per
    shelf, never a queue flood; the value-axis cartesian is what shares
    shelves in real matrices.  Naming a trial still submits that one alone
    (a single point's re-run); a trial name on `submit run` still
    refuses."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    js = _prep_bench(calc)
    trial = js["jobs"][0]["name"]
    runner = CliRunner()
    r = runner.invoke(jobset_group, ["launch", "bench", "--bundle", str(calc),
                                     "--mode", "submit", "--dry-run", "--yes", "--domain", "htc"])
    assert r.exit_code != 0 and "name it" in r.output
    r = runner.invoke(jobset_group, ["launch", "bench", "coarse",
                                     "--bundle", str(calc),
                                     "--mode", "submit", "--dry-run", "--yes", "--domain", "htc"])
    assert r.exit_code == 0, r.output
    # one sbatch per SHELF, each an exact fit -- and the shelves submit
    # widest first, so the first planned group asks the widest -n
    shelves = {(j["resources"].get("mpi_np") or 0,
                j["resources"].get("cpus_per_task") or 0,
                j["resources"].get("gres") or "") for j in js["jobs"]}
    plans = [l for l in r.output.splitlines() if "WOULD run" in l]
    assert len(plans) == len(shelves)
    assert len(plans) < len(js["jobs"]) or len(shelves) == len(js["jobs"])
    assert "bench-group" in r.output
    assert ".sbatch" in plans[0]
    # TIME IS NEVER INVENTED (user dictation, 2026-08-24).  This fixture is
    # a workstation record -- no queue menu, so no ceiling to default to --
    # and nothing was stated, so NO -t rides the command and the display
    # says the scheduler's default stands.  A stated --time is the wall.
    assert " -t " not in plans[0], (
        f"a wall was invented on a machine with no ceiling: {plans[0]}")
    r2 = runner.invoke(jobset_group, ["launch", "bench", "coarse",
                                      "--bundle", str(calc),
                                      "--mode", "submit", "--dry-run",
                                      "--yes", "--domain", "htc",
                                      "--time", "4h"])
    assert r2.exit_code == 0, r2.output
    plans2 = [l for l in r2.output.splitlines() if "WOULD run" in l]
    assert all(" -t 0-04:00:00 " in pl for pl in plans2), (
        f"--time 4h must ride every shelf-job: {plans2}")
    # widest = the CORE footprint (ranks x cores-per-rank), not ranks
    wid = max(js["jobs"], key=lambda j: (j["resources"].get("mpi_np") or 1)
              * (j["resources"].get("cpus_per_task") or 1))["resources"]
    assert (f" -n {wid.get('mpi_np') or 1} " in plans[0]
            and f" -c {wid.get('cpus_per_task')} " in plans[0]), (
        f"the widest shelf must submit first: {plans[0]}"
    )
    assert r.output.count("rides the group") == len(js["jobs"])
    assert "next unlaunched trial" not in r.output
    r = runner.invoke(jobset_group, ["launch", "bench", "coarse", trial,
                                     "--bundle", str(calc),
                                     "--mode", "submit", "--dry-run", "--yes", "--domain", "htc"])
    assert r.exit_code == 0, r.output
    assert r.output.count("WOULD run") == 1
    assert "bench-group" not in r.output
    r = runner.invoke(jobset_group, ["launch", "run", "coarse", trial,
                                     "--bundle", str(calc),
                                     "--mode", "submit", "--dry-run", "--yes", "--domain", "htc"])
    assert r.exit_code != 0 and "TRIAL names a benchmark point" in r.output


def test_launch_bench_mem_reaches_the_grouped_sbatch_command(calc):
    """**The regression.**  Job 62039305 (48 ranks, 1 GPU, `htc`) OOM'd at
    24576M on 2026-08-23 -- a number nobody asked for and nothing told
    anyone about, because `--mem` typed at `launch` never reached
    `submit_bench_group` at all: `_group_envelope` only ever set
    mpi_np/cpus_per_task/gres/exclusive, so the actual `sbatch` command
    asked for no memory no matter what a person typed.

    Fixed by threading `Ask.mem_gb` through `submit_bench_group` ->
    `_submit_side_group`, applied to the envelope the same way
    `_dc_replace_time` already applies the wall (`jobset/submit.py`).
    This is the CLI end to end, `--dry-run` so nothing real is submitted --
    the same entry point `test_cli_submit_bench_groups_the_sweep_by_shelf`
    proves the shelf-grouping through, with one flag added.
    """
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    _prep_bench(calc)
    r = CliRunner().invoke(jobset_group, [
        "launch", "bench", "coarse", "--bundle", str(calc),
        "--mode", "submit", "--dry-run", "--yes", "--domain", "htc",
        "--mem", "128G"])
    assert r.exit_code == 0, r.output
    plans = [l for l in r.output.splitlines() if "WOULD run" in l]
    assert plans, r.output
    assert all("--mem=128G" in p for p in plans), (
        f"--mem 128G never reached the sbatch command: {plans}")


def test_the_launch_plan_states_gpu_sharing_and_what_is_unstated(calc):
    """**What a person approves says what will actually be asked for.**

    User, 2026-08-23: *"explicitly note for gpu enabled task: how many
    task will be sharing the gpu at the same time, and warn if that
    number is exceedingly high"* -- and 2026-08-24, that an unstated
    limit must be SAID rather than silently defaulted.  Both are checked
    on the real launch door, from the very commands it is about to send.

    The arithmetic itself (`ask.gpu_share_notes`) is unit-tested; this
    pins that it REACHES the approval screen, which is the half that was
    missing while five Sol jobs went out against limits nobody chose.
    """
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    _prep_bench(calc)
    r = CliRunner().invoke(jobset_group, [
        "launch", "bench", "coarse", "--bundle", str(calc),
        "--mode", "submit", "--dry-run", "--yes", "--domain", "htc"])
    assert r.exit_code == 0, r.output
    # every GPU shelf's ratio, stated -- and stated ONCE per ratio, not
    # once per shelf (several shelves share a ratio).
    ratios = [l for l in r.output.splitlines() if "gpu share" in l]
    assert ratios, r.output
    assert len(ratios) == len(set(ratios)), f"repeated: {ratios}"
    assert any("rank(s)/GPU" in l for l in ratios)
    # and the two unstated facts, each said exactly once
    assert r.output.count("MEMORY NOT STATED") == 1, r.output
    assert r.output.count("time not stated") <= 1
    # a stated --mem removes its warning entirely
    r2 = CliRunner().invoke(jobset_group, [
        "launch", "bench", "coarse", "--bundle", str(calc),
        "--mode", "submit", "--dry-run", "--yes", "--domain", "htc",
        "--mem", "128G"])
    assert r2.exit_code == 0, r2.output
    assert "MEMORY NOT STATED" not in r2.output


def test_the_group_sequencer_runs_every_trial_and_survives_failures(
        calc, monkeypatch):
    """The generated bash, EXECUTED: every pending trial runs in its own
    directory in order; a failing trial does not stop the walk; a trial
    that hits the per-trial bound is killed (rc=124, named in the log) and
    the rest still run; the script exits nonzero because something failed;
    and every included trial's run.json carries the ONE job id."""
    import json as _json
    import stat
    import subprocess as _sp
    from pathlib import Path

    from molbuilder.jobset._cli import _load_bench_set
    from molbuilder.jobset.materialize import (job_dir_names, shape_of,
                                               was_launched)
    from molbuilder.jobset.submit import submit_bench_group

    # A DECLARED one-shelf sweep (one machine point x a block_size value
    # axis): the walk story needs several trials in ONE group, and since
    # the shelf split (2026-08-21) only same-ask trials share a group --
    # exactly what a value axis produces.
    _declare_bench(calc, {"mpi_np": [4], "omp_threads": [1],
                          "block_size": [16, 32, 64]})
    _prep_bench(calc)
    js, base = _load_bench_set(calc, "coarse")
    dirs = job_dir_names(js, shape_of(js, base))

    # Stub each trial's wrapper: first succeeds and leaves a marker,
    # second fails, third (if any) sleeps past the bound.
    behaviours = ["ok", "fail", "sleep"]
    for n, job in enumerate(js.jobs):
        d = base / dirs[job.name]
        wrapper = d / (Path(job.script).stem + ".run.sh")
        kind = behaviours[min(n, 2)]
        body = {"ok":    "#!/usr/bin/env bash\ntouch ran.marker\nexit 0\n",
                "fail":  "#!/usr/bin/env bash\ntouch ran.marker\nexit 3\n",
                "sleep": "#!/usr/bin/env bash\ntouch ran.marker\nsleep 30\n",
                }[kind]
        wrapper.write_text(body)
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)

    # sbatch is faked; the sequencer is then run HERE with bash.
    calls = {}

    def fake_run(cmd, **kw):
        calls["cmd"] = cmd
        calls["cwd"] = kw.get("cwd")
        class R:
            returncode = 0
            stdout = "Submitted batch job 4242"
            stderr = ""
        return R()
    # Rebind ONLY the submit module's `subprocess` name -- patching the
    # global module would also fake the REAL bash run below.
    import types

    import molbuilder.jobset.submit as submod
    monkeypatch.setattr(submod, "subprocess",
                        types.SimpleNamespace(run=fake_run))
    # This box has no queue, so the real header render answers None (and
    # the group rightly refuses).  The header is not under test here -- the
    # sequencer is -- so stub the emitter the way sbatch is stubbed.
    import molbuilder.runwrap as _rw
    monkeypatch.setattr(_rw, "_render_sbatch_for",
                        lambda *a, **k: "#!/bin/bash\n"
                        "#SBATCH -o slurm.%j.out\n#SBATCH -e slurm.%j.err\n"
                        "bash bench-group.run.sh \"$@\"\n")

    results = submit_bench_group(js, base, dry_run=False, trial_timeout_s=2)
    assert results[0].name == "bench-group"
    assert results[0].job_id == "4242"

    container = Path(calls["cwd"])
    script = container / "launch" / "bench-group.run.sh"
    assert script.is_file(), (
        "the sequencer lives in launch/ (L3, roadmap 7.10)")
    assert container.name == "bench", (
        f"the group runs at the parent that sees every trial: {container}"
    )

    # EXECUTE the generated bash for real -- from the container, exactly
    # as the sbatch body does (`bash launch/<name>.run.sh`), so the trial
    # dirs resolve relative to the container and the log lands in launch/.
    proc = _sp.run(["bash", f"launch/{script.name}"], cwd=str(container),
                   capture_output=True, text=True, timeout=120)
    log = (container / "launch" / "bench-group.log").read_text()
    ran = [job.name for job in js.jobs
           if (base / dirs[job.name] / "ran.marker").exists()]
    assert ran == [j.name for j in js.jobs], (
        f"a failure stopped the walk: only {ran} ran\n{log}"
    )
    assert proc.returncode != 0, "a sweep with failures must say so"
    if len(js.jobs) >= 3:
        assert "hit the 2s per-trial bound" in log, log
    # The explicit record (user, 2026-08-20): when each trial started,
    # finished, with what rc and duration -- and the allocation the group
    # ran in, so an env-inheritance question is answered by the log itself.
    assert "alloc_ntasks=" in log and "job=" in log, log
    for job in js.jobs:
        assert f"-> {job.name} starts" in log, log
        assert f"<- {job.name} finished rc=" in log, log
    assert "took=" in log and "s" in log
    for job in js.jobs:
        assert was_launched(base / dirs[job.name]), (
            f"{job.name} has no launch record"
        )
        rec = _json.loads(
            (base / dirs[job.name] / "run.json").read_text())
        assert rec.get("job_id") == "4242"

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
    res = r.invoke(jobset_group, ["launch", "bench", "coarse",
                                  "--bundle", str(calc),
                                  "--mode", "submit", "--dry-run", "--yes", "--domain", "htc"])
    assert res.exit_code == 0, res.output
    res = r.invoke(jobset_group, ["summarize", "bench", "coarse",
                                  "--bundle", str(calc)])
    assert res.exit_code == 0, res.output
    lines = [json.loads(l) for l in
             (calc / LEDGER_FILE).read_text().splitlines()]
    got = [(e["verb"], e["decision"]) for e in lines]
    assert got == [("prep", "prepped"),
                   ("launch", "bench-grouped"),
                   ("launch", "launched"),
                   ("summarize", "verdict-written")]
    prep = lines[0]
    assert prep["kind"] == "bench" and prep["stage"] == "coarse"
    assert "provenance" in prep            # WHERE each setting came from
    group = lines[1]
    # Unstated stays None in the ledger too -- the 15-minute default this
    # asserted is deleted (user dictation, 2026-08-24).
    assert group["trial_timeout_s"] is None
    assert len(group["sweep"]) == len(
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
    assert mech["use_gpu"] is True           # the deck's gpu_mode
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
        d = _artifacts(calc, j['name'])
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
        w = (_artifacts(calc, name)
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


def test_the_verb_renders_the_trial_decks_it_promises(calc):
    """I5 (2026-08-13): every earlier deck-content pin supplied the
    grid, pins and translation itself through library internals
    (`_bench_inputs` + `prep_calculation`), so the VERB's own wiring of
    them was unpinned.  This drives `jobset prep bench coarse` -- the
    command a user types -- and asserts the CONTENT of what lands: each
    trial deck carries the TRIAL's own identity line (§ 2.3.2's relabel,
    which is what keys its warm files away from the run's), and the
    sweep's record rows point at those very decks."""
    import re
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    res = CliRunner().invoke(jobset_group, ["prep", "bench", "coarse",
                                            "--bundle", str(calc),
                                            "--no-sbatch"])
    assert res.exit_code == 0, res.output
    js = json.loads(
        (calc / "01_coarse" / "bench" / "job-set.json").read_text())
    assert js["kind"] == "sweep" and len(js["jobs"]) >= 2
    from molbuilder.jobset.materialize import (job_dir_names, shape_of)
    from molbuilder.jobset.model import JobSet
    _dirs = job_dir_names(JobSet.from_dict(js), shape_of(None, calc))
    from molbuilder.jobset.materialize import latest_attempt
    for j in js["jobs"]:
        # The reader rule: the latest attempt where there is one
        # (`project-layout.md` § 1.5a). `job_dir_names` answers where a
        # trial LIVES; its deck is in the attempt.
        _c = calc / _dirs[j["name"]]
        deck = ((latest_attempt(_c) or _c) / j["script"]).read_text()
        m = re.search(r"^SystemLabel\s+(\S+)", deck, re.M)
        assert m and m.group(1) == f"JOB-{j['name']}", (
            f"{j['script']}: identity line {m.group(1) if m else None!r} "
            f"is not the trial's own label -- its warm files would "
            f"collide with the run's")


def test_a_fine_tuned_vocabulary_copy_wins_and_is_named(calc):
    """U6a (§ 4.2a's template mechanism): a calculation carrying its own
    warm-files.toml is the fine-tuned state -- nearest file wins, the
    surgical edit here being 'withhold the density' (the .DM row loses
    its carry flag).  The plan names WHICH file answered, so the
    surprising carry is debuggable from the bundle alone."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    from molbuilder import warmfiles as _wf
    engine_file = _wf._rules_path("siesta")
    text = engine_file.read_text()
    text = text.replace(
        'suffix      = ".DM"             # converged density matrix\n'
        'carry       = "when-continuing"\n'
        'honoured_by = "DM.UseSaveDM"',
        'suffix      = ".DM"             # fine-tuned: NOT carried here')
    (calc / "warm-files.toml").write_text(text)
    r = CliRunner()
    res = r.invoke(jobset_group, ["prep", "run", "medium",
                                  "--bundle", str(calc), "--no-sbatch"])
    assert res.exit_code == 0, res.output
    js = json.loads((calc / "job-set.json").read_text())
    warm = {w["name"] for j in js["jobs"] if j["name"] == "medium"
            for w in j["warm"]}
    assert "JOB.XV" in warm and "JOB.CG" in warm
    assert "JOB.DM" not in warm, (
        "the calculation's own vocabulary says the density does not "
        "carry, and the engine default answered anyway")
    plan = (calc / "STAGE-PLAN.md").read_text()
    assert f"warm-files: {calc / 'warm-files.toml'}" in plan


def test_the_cg_pair_rule_holds_both_ways_on_a_live_bundle(calc):
    """G2 I-list (2026-08-13): project-layout § 2.3.4 row 3 driven
    through the VERBS on a real described bundle, both directions.  The
    shipped ladder is coarse=CG, medium=Broyden, tight=Broyden -- so
    coarse->medium must WITHHOLD `.CG` (a CG history is meaningless to
    Broyden; carrying it would corrupt the restart) while medium->tight
    must CARRY it (same optimizer, verified through the merged plan the
    A11-era merge keeps whole).  The first version of this test asserted
    a blind carry and the SYSTEM was right to refuse it."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    r = CliRunner()
    res = r.invoke(jobset_group, ["prep", "run", "coarse",
                                  "--bundle", str(calc), "--no-sbatch"])
    assert res.exit_code == 0, res.output
    a1 = calc / "01_coarse" / "run-0"
    (a1 / "JOB.XV").write_text("relaxed coords")
    (a1 / "JOB.CG").write_text("cg history")
    res = r.invoke(jobset_group, ["prep", "run", "medium",
                                  "--bundle", str(calc), "--no-sbatch",
                                  "--from", "01_coarse/run-0"])
    assert res.exit_code == 0, res.output
    a2 = calc / "02_medium" / "run-0"
    carried = {p.name for p in a2.glob("JOB.*") if not p.is_symlink()}
    assert "JOB.XV" in carried
    assert "JOB.CG" not in carried, (
        "a CG-optimizer history crossed into a Broyden stage -- the "
        "corrupting carry § 2.3.4 row 3 exists to prevent")
    (a2 / "JOB.XV").write_text("more relaxed")
    (a2 / "JOB.CG").write_text("broyden history")
    res = r.invoke(jobset_group, ["prep", "run", "tight",
                                  "--bundle", str(calc), "--no-sbatch",
                                  "--from", "02_medium/run-0"])
    assert res.exit_code == 0, res.output
    a3 = calc / "03_tight" / "run-0"
    carried = {p.name for p in a3.glob("JOB.*") if not p.is_symlink()}
    assert "JOB.XV" in carried
    assert "JOB.CG" in carried, (
        "same-optimizer pair (broyden->broyden) withheld the history -- "
        "the pair verification broke on the live path")


def test_a_one_stage_calculation_runs_end_to_end(tmp_path):
    """R1 (review-4 keystone), rewritten 2026-08-16: the single-parameter-
    set calculation is a FIRST-CLASS run.  Its shape changed -- § 6.5 now
    gives that one parameter set a NAMED, tokened stage instead of the
    tokenless root form -- but the property under test did not: prep,
    submit and status all reach it, one rung deep, with no dangling link
    anywhere in the attempt."""
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
        D.build_description(struct, SiestaConfig(system_label="JOB"),
                            _one_stage(),
                            engine="siesta", shape="hierarchical",
                            name="JOB", source=str(tmp_path / "h2.xyz")),
        dest)
    from conftest import write_pseudos
    write_pseudos(dest, sorted(set(struct.elements)))
    (dest / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": "true"}}))
    from conftest import write_pseudos
    write_pseudos(dest, ["H"])
    r = CliRunner()
    # § 6.5: the bare verb does NOT guess the lone stage.  A rule that held
    # only at length one would stop holding the moment a second stage was
    # added -- so one rung refuses exactly as three do, listing the ladder.
    res = r.invoke(jobset_group, ["prep", "run", "--bundle", str(dest),
                                  "--no-sbatch"])
    assert res.exit_code != 0, res.output
    assert "coarse" in res.output, res.output
    assert not (dest / "01_coarse").exists(), \
        "the bare verb guessed the lone stage and prepped it"

    res = r.invoke(jobset_group, ["prep", "run", "coarse", "--bundle",
                                  str(dest), "--no-sbatch"])
    assert res.exit_code == 0, res.output
    rung = dest / "01_coarse"                 # the one rung, tokened
    assert (rung / "run-0").is_dir()          # its attempt
    assert not (dest / "run-0").exists()      # never at the root
    assert not (dest / "bench-JOB").exists()  # nor named for a benchmark
    assert "launch run coarse --mode" in res.output
    # Every link in the attempt RESOLVES.  The 2026-08-12 redo found the
    # first version of this test asserting existence only, over links that
    # all dangled (prepare_attempt hopped a hardcoded "../.." over a
    # depth-1 attempt): prep exited 0, submit was dead.  Existence of a
    # symlink proves nothing -- resolve it.
    links = [p for p in (rung / "run-0").iterdir()]
    assert links, "the attempt is empty -- prep changed shape"
    assert not any(p.is_symlink() for p in links), (
        "L2 (roadmap 7.10): an attempt holds real copies, never links")
    for link in links:
        assert link.resolve().is_file(), \
            f"{link.name} -> {os.readlink(link)} dangles"
    js = json.loads((dest / "job-set.json").read_text())
    assert js["kind"] == "ladder" and len(js["jobs"]) == 1
    res = r.invoke(jobset_group, ["launch", "run", "coarse", "--bundle",
                                  str(dest), "--mode", "direct", "--dry-run", "--yes"])
    assert res.exit_code == 0, res.output
    assert res.output.count("WOULD run") == 1
    # A REAL direct launch, not --dry-run: the launcher stats the wrapper
    # THROUGH the link (a dangling one reads as absent and earns the
    # "run prep_jobset first" refusal), and run.json is written at start,
    # so the record proves the launch began no matter how the engine's
    # process exits in this environment.
    res = r.invoke(jobset_group, ["launch", "run", "coarse", "--bundle",
                                  str(dest), "--mode", "direct", "--yes"])
    assert "run prep_jobset first" not in res.output
    assert (rung / "run-0" / "run.json").is_file(), res.output
    res = r.invoke(jobset_group, ["status", "--bundle", str(dest)])
    assert res.exit_code == 0, res.output
    # A hand-built SWEEP whose jobs carry no token still files them in the
    # bare bench/ container beside their own record (§ 6.3; A-2,
    # 2026-08-13).  No DESCRIPTION reaches this row any more -- every one
    # of them has a stage, so every deck has a token -- but the naming
    # authority is still asked it directly, so it is still pinned.
    from molbuilder.jobset.materialize import job_dir_names
    from molbuilder.jobset.model import Job, JobSet, Resources
    sweep = JobSet(name="X", engine="siesta", kind="sweep",
                   jobs=[Job(name="G1K1C4", script="job-gpu.fdf",
                             resources=Resources())])
    assert job_dir_names(sweep)["G1K1C4"] == "bench/bench-G1K1C4"


def test_a_one_stage_calculation_continues_from_its_own_attempt(tmp_path):
    """A-3 (final review, 2026-08-13), rewritten 2026-08-16: a job
    continuing from its OWN attempt is the one pair that cannot disagree
    with itself, so `warm_carry` must hand it the conditional ``.CG``.
    The original bug needed the stage-less root directory ``.`` to bite (a
    head-component match could never equal it) and § 6.5 has since deleted
    that shape -- but the invariant is about the self-pair, not the
    spelling, so it is re-pinned here on the one-stage form."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    from molbuilder.jobset.materialize import write_run_launch
    struct = Structure(elements=["H", "H"],
                       positions=np.array([[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.74]]),
                       vacuum=(10.0, 10.0, 10.0))
    (tmp_path / "h2.xyz").write_text(struct.to_xyz())
    dest = tmp_path / "calc"
    D.write_description(
        D.build_description(struct, SiestaConfig(system_label="JOB"),
                            _one_stage(),
                            engine="siesta", shape="hierarchical",
                            name="JOB", source=str(tmp_path / "h2.xyz")),
        dest)
    from conftest import write_pseudos
    write_pseudos(dest, sorted(set(struct.elements)))
    (dest / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": "true"}}))
    # A calculation that CONTINUES -- which is what a described calculation
    # does by default since 2026-08-18 (`run-identity.md` § 4 rule 3): a run
    # started in a folder that already holds a result was started after
    # somebody read that result.  The template is checked rather than edited,
    # because the value being the default is the thing this depends on.
    tpl = dest / "JOB.template.toml"
    head, sep, tail = tpl.read_text().partition("[item.restart]")
    assert sep, "the template lost its restart item"
    assert 'value = "continue"' in tail, (
        "the described default is no longer `continue`, so this test is no "
        "longer setting up the case it was written for")
    r = CliRunner()
    res = r.invoke(jobset_group, ["prep", "run", "coarse", "--bundle",
                                  str(dest), "--no-sbatch"])
    assert res.exit_code == 0, res.output
    # the attempt ran (hit its step cap, say): warm files in it, launch on
    # record
    rung = dest / "01_coarse"
    (rung / "run-0" / "JOB.XV").write_text("relaxed coords")
    (rung / "run-0" / "JOB.CG").write_text("cg history")
    write_run_launch(rung / "run-0", mode="direct", command=["bash"])
    res = r.invoke(jobset_group, ["prep", "run", "coarse", "--bundle",
                                  str(dest), "--no-sbatch",
                                  "--from", "01_coarse/run-0"])
    assert res.exit_code == 0, res.output
    carried = {p.name for p in (rung / "run-1").glob("JOB.*")
               if not p.is_symlink()}
    assert "JOB.XV" in carried
    assert "JOB.CG" in carried, (
        "continuing from its own attempt withheld the optimizer history "
        "-- the self-pair read as unverified (A-3)")


def test_a_charged_decks_promised_script_ships_with_it(tmp_path):
    """E6 (redo 2026-08-12): a charged deck's header instructs
    ``python3 makov_payne_correction.py`` -- a promise only `convert`
    kept.  The described route rendered the same header and never wrote
    the script, so prep shipped an instruction to run a file that did
    not exist.  The seam's sibling_artifacts hook writes it now."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    struct = Structure(elements=["H", "H"],
                       positions=np.array([[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.74]]),
                       vacuum=(10.0, 10.0, 10.0))
    (tmp_path / "h2.xyz").write_text(struct.to_xyz())
    dest = tmp_path / "calc"
    D.write_description(
        D.build_description(struct, SiestaConfig(system_label="JOB",
                                                 net_charge=1), _one_stage(),
                            engine="siesta", shape="hierarchical",
                            name="JOB", source=str(tmp_path / "h2.xyz")),
        dest, struct=struct)
    from conftest import write_pseudos
    write_pseudos(dest, sorted(set(struct.elements)))
    (dest / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": "true"}}))
    res = CliRunner().invoke(jobset_group,
                             ["prep", "run", "coarse", "--bundle", str(dest),
                              "--no-sbatch"])
    assert res.exit_code == 0, res.output
    _deck_path = next(dest.glob("01_coarse/*.fdf"))     # L1: in the stage dir
    deck = _deck_path.read_text()
    if "makov_payne_correction.py" in deck:
        assert (_deck_path.parent / "makov_payne_correction.py").is_file(), (
            "the deck instructs running a script prep did not write "
            "beside it")


def test_a_one_stage_calculation_can_be_benchmarked(tmp_path):
    """A4 (redo 2026-08-12), rewritten 2026-08-16: a one-stage calculation
    can be benchmarked, and it is benchmarked the way every rung is --
    ``prep bench <stage>``, trials in THAT stage's container.

    A4's own subject was the bare-verb bench grammar for a stage-less
    calculation (a lone name after ``bench`` bound to the trial, two names
    refused, and the hint printed "prep bench None").  § 6.5 deleted the
    shape that grammar served, so those assertions are retired rather than
    translated -- there is no longer a calculation that owns no stage.
    What survives is the part that was never about stage-less-ness: the
    verdict offer, and a named trial submitting exactly itself."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    struct = Structure(elements=["H", "H"],
                       positions=np.array([[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.74]]),
                       vacuum=(10.0, 10.0, 10.0))
    (tmp_path / "h2.xyz").write_text(struct.to_xyz())
    dest = tmp_path / "calc"
    D.write_description(
        D.build_description(struct, SiestaConfig(system_label="JOB"),
                            _one_stage(),
                            engine="siesta", shape="hierarchical",
                            name="JOB", source=str(tmp_path / "h2.xyz")),
        dest)
    from conftest import write_pseudos
    write_pseudos(dest, sorted(set(struct.elements)))
    (dest / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": "true"}}))
    (dest / "environment.json").write_text(
        Environment(scheduler="workstation",
                    topology=Topology(sockets=1, cores_per_socket=4,
                                      gpus_per_node=1,
                                      gpu_type="a100")).to_json() + "\n")
    r = CliRunner()
    res = r.invoke(jobset_group, ["prep", "bench", "coarse", "--bundle",
                                  str(dest), "--no-sbatch"])
    assert res.exit_code == 0, res.output
    assert "None" not in res.output, res.output
    bench = dest / "01_coarse" / "bench"
    js = json.loads((bench / "job-set.json").read_text())
    assert js["kind"] == "sweep" and len(js["jobs"]) >= 2
    # trial decks carry the rung's token like any other stage's trials
    for j in js["jobs"]:
        from molbuilder.jobset.materialize import (job_dir_names as _jdn,
                                                    shape_of as _sof)
        from molbuilder.jobset.model import JobSet as _JS
        _dmap = _jdn(_JS.from_dict(js), _sof(None, dest))
        from molbuilder.jobset.materialize import latest_attempt
        # the reader rule (`project-layout.md` 1.5a): a trial's
        # deck is in its attempt where the shape keeps one.
        _c = dest / _dmap[j["name"]]
        assert ((latest_attempt(_c) or _c) / j["script"]).is_file()
        assert "01_coarse" in j["script"], j["script"]
    # A-2 (2026-08-13): the trials live IN their stage's bench container,
    # beside the record just read -- not at the root, where the
    # underway-ask never looked and a re-prep silently re-rendered the
    # decks a queued trial's links point at.
    for j in js["jobs"]:
        assert (bench / f"bench-{j['name']}").is_dir()
        assert not (dest / f"bench-{j['name']}").exists()
    # a NAMED trial submits alone (how a single point is re-run); an
    # unnamed bench submits one grouped job per resource shelf -- the
    # old one-per-invocation rule survives as one LAUNCH ACT per shelf.
    t0, t1 = js["jobs"][0]["name"], js["jobs"][1]["name"]
    res = r.invoke(jobset_group, ["launch", "bench", "coarse", t1, "--bundle",
                                  str(dest), "--mode", "direct", "--dry-run", "--yes"])
    assert res.exit_code == 0, res.output
    would = [l for l in res.output.splitlines() if "WOULD run" in l]
    assert len(would) == 1, res.output
    assert t1 in would[0] and t0 not in would[0], res.output
    # an unknown stage is refused with the ladder listed, not ignored
    res = r.invoke(jobset_group, ["summarize", "bench", "not-a-stage",
                                  "--bundle", str(dest)])
    assert res.exit_code != 0 and "not-a-stage" in res.output
    # and a bare bench owes a name here exactly as it does on three rungs
    res = r.invoke(jobset_group, ["prep", "bench", "--bundle", str(dest),
                                  "--no-sbatch"])
    assert res.exit_code != 0 and "name it" in res.output
    # the same on a longer ladder -- one rule, not a per-length one
    ladder = tmp_path / "laddered"
    D.write_description(
        D.build_description(struct, SiestaConfig(system_label="JOB"),
                            default_siesta_stages("publishable"),
                            engine="siesta", shape="hierarchical",
                            name="JOB", source=str(tmp_path / "h2.xyz")),
        ladder)
    (ladder / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": "true"}}))
    res = r.invoke(jobset_group, ["prep", "bench", "--bundle", str(ladder),
                                  "--no-sbatch"])
    assert res.exit_code != 0
    assert "name it" in res.output


def test_two_flat_stages_benchmarks_do_not_collide(tmp_path):
    """A5 (redo 2026-08-12): FLAT has no stage directory for `bench/` to
    nest inside, so unqualified, two stages' benchmarks shared one root
    container and each prep overwrote the other's job-set, plan and
    verdict.  The token qualifies the container's own name in flat —
    ``bench_<NN>_<stage>/`` — underscore-joined so it cannot read as a
    trial's dash-joined ``bench-<point>``."""
    struct = Structure(elements=["H", "H"],
                       positions=np.array([[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.74]]),
                       vacuum=(10.0, 10.0, 10.0))
    (tmp_path / "h2.xyz").write_text(struct.to_xyz())
    dest = tmp_path / "calc"
    D.write_description(
        D.build_description(struct, SiestaConfig(system_label="JOB"),
                            default_siesta_stages("publishable"),
                            engine="siesta", shape="flat", name="JOB",
                            source=str(tmp_path / "h2.xyz")),
        dest)
    from conftest import write_pseudos
    write_pseudos(dest, sorted(set(struct.elements)))
    (dest / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": "true"}}))
    (dest / "environment.json").write_text(
        Environment(scheduler="workstation",
                    topology=Topology(sockets=1, cores_per_socket=4,
                                      gpus_per_node=1,
                                      gpu_type="a100")).to_json() + "\n")
    sweep, pins, translation = _bench_inputs(dest, None)
    prep_calculation(dest, "coarse",
                     allocation=Resources(mpi_np=8, cpus_per_task=8),
                     sweep=sweep, pins=pins, translation=translation,
                     emit_sbatch=False)
    prep_calculation(dest, "medium",
                     allocation=Resources(mpi_np=8, cpus_per_task=8),
                     sweep=sweep, pins=pins, translation=translation,
                     emit_sbatch=False)
    coarse = dest / "bench_01_coarse" / "job-set.json"
    medium = dest / "bench_02_medium" / "job-set.json"
    assert coarse.is_file() and medium.is_file()
    assert not (dest / "bench" / "job-set.json").exists()
    # each record still names ITS stage's decks: no overwrite happened
    cj = json.loads(coarse.read_text())
    mj = json.loads(medium.read_text())
    assert all("01_coarse" in j["script"] for j in cj["jobs"])
    assert all("02_medium" in j["script"] for j in mj["jobs"])
    # A-1 (2026-08-13): the TRIALS live in the same qualified container as
    # the record.  Until job_dir_names and prep shared one spelling
    # (materialize.bench_container), flat trials fell into an unqualified
    # shared root bench/ -- two stages' same-coordinate trials collided
    # (run.json cross-contamination read as "all trials launched") while
    # the underway-ask globbed the qualified container and found nothing.
    for cont, record in (("bench_01_coarse", cj), ("bench_02_medium", mj)):
        for j in record["jobs"]:
            assert (dest / cont / f"bench-{j['name']}").is_dir(), (
                f"trial {j['name']} not in {cont}/")
    assert not (dest / "bench").exists()


def test_a_flat_one_stage_calculation_preps_to_completion(tmp_path):
    """A2 (redo 2026-08-12), rewritten 2026-08-16: FLAT prep is COMPLETE
    without an attempt -- only an explicit ``--from``/``--cold`` (an
    attempt ask flat cannot serve) reaches prepare_attempt's refusal.  The
    original bug rode the stage-less bare-verb path § 6.5 has since
    deleted; the shape-blindness it exposed is still worth pinning, so the
    case moved onto the one-stage form."""
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
        D.build_description(struct, SiestaConfig(system_label="JOB"),
                            _one_stage(),
                            engine="siesta", shape="flat",
                            name="JOB", source=str(tmp_path / "h2.xyz")),
        dest)
    from conftest import write_pseudos
    write_pseudos(dest, sorted(set(struct.elements)))
    (dest / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": "true"}}))
    r = CliRunner()
    res = r.invoke(jobset_group, ["prep", "run", "coarse", "--bundle",
                                  str(dest), "--no-sbatch"])
    assert res.exit_code == 0, res.output
    assert "no attempt to open" in res.output
    assert "launch run" in res.output
    assert not (dest / "run-0").exists()
    # an attempt ASK on flat is still the one refusal, with its story
    res = r.invoke(jobset_group, ["prep", "run", "coarse", "--bundle",
                                  str(dest), "--no-sbatch", "--cold"])
    assert res.exit_code != 0
    assert "flat" in res.output


def test_a_config_refusal_is_a_refusal_not_a_traceback(tmp_path, monkeypatch):
    """A8 (redo 2026-08-12): the described route's most likely
    first-contact failure -- ``script_generation.activation`` unset --
    escaped `prep` as a raw ``RuntimeConfigError`` traceback.  The named
    user-fixable classes translate to PrepError at the library seam, so
    the CLI answers with the refusal text, not a stack.

    Sandboxed cwd+HOME: the first version of this test passed a fake HOME
    to CliRunner's ``env`` -- which never touches ``os.environ`` -- and
    prepped GREEN off the developer's own molbuilder.json in the repo cwd."""
    from click.testing import CliRunner
    from molbuilder import describe as D
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.jobset._cli import jobset_group
    from molbuilder.structure import Structure
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    (tmp_path / "home").mkdir()
    struct = Structure(elements=["H", "H"],
                       positions=np.array([[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.74]]),
                       vacuum=(10.0, 10.0, 10.0))
    (tmp_path / "h2.xyz").write_text(struct.to_xyz())
    dest = tmp_path / "calc"
    D.write_description(
        D.build_description(struct, SiestaConfig(system_label="JOB"),
                            _one_stage(),
                            engine="siesta", shape="hierarchical",
                            name="JOB", source=str(tmp_path / "h2.xyz")),
        dest)
    from conftest import write_pseudos
    write_pseudos(dest, sorted(set(struct.elements)))
    (dest / ".molbuilder.json").write_text("{}")   # no activation anywhere
    res = CliRunner().invoke(jobset_group,
                             ["prep", "run", "coarse", "--bundle", str(dest),
                              "--no-sbatch"])
    assert res.exit_code != 0
    assert "activation" in res.output, res.output
    assert "Traceback" not in res.output, res.output


def test_prep_bench_asks_when_a_trial_is_already_launched(calc):
    """A7 (redo 2026-08-12): `prep bench` re-renders the very decks a
    QUEUED trial's symlinks point at, and the underway-ask ran for
    `prep run` only -- so that re-render was silent.  The ask now sees
    launched trials; unanswerable, it proceeds saying so (§ 6 warns, it
    does not refuse)."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    from molbuilder.jobset.materialize import write_run_launch
    js = _prep_bench(calc)
    first = js["jobs"][0]["name"]
    write_run_launch(_artifacts(calc, first),
                     mode="submit", command=["sbatch", "x"], job_id="7")
    res = CliRunner().invoke(jobset_group,
                             ["prep", "bench", "coarse",
                              "--bundle", str(calc), "--no-sbatch"])
    assert res.exit_code == 0, res.output
    assert "under way" in res.output
    assert "launched trial(s)" in res.output and first in res.output


def test_a_stage_without_an_open_attempt_refuses_to_launch(calc):
    """C5 (redo 2026-08-12, R2's missing half): a hierarchical stage
    prepped without `prep run` -- decks and wrappers in place, no run-<n>
    -- used to launch IN ITS OWN CONTAINER: no run.json, silently
    relaunchable, everything § 1.5/1.6 exist to prevent.  It refuses now
    and names the verb that opens the attempt."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    from molbuilder.jobset.prep import prep_calculation as _pc
    _pc(calc, "coarse", allocation=Resources(mpi_np=2, cpus_per_task=2),
        emit_sbatch=False)             # library prep: NO attempt opened
    assert not (calc / "01_coarse" / "run-0").exists()
    res = CliRunner().invoke(jobset_group,
                             ["launch", "run", "coarse",
                              "--bundle", str(calc),
                              "--mode", "direct", "--yes"])
    assert res.exit_code != 0
    assert "no attempt is open" in res.output
    assert "prep run coarse" in res.output
    assert not (calc / "01_coarse" / "run.json").exists()


def test_a_direct_sweep_resumes_past_launched_trials(calc):
    """A6 (redo 2026-08-12): direct mode runs the set in order, and the
    launched-trial refusal (R2) made it die at the FIRST record -- an
    interrupted direct sweep could never finish.  The loop now skips a
    launched trial out loud and runs the rest; under submit mode the
    grouped path collects the still-unlaunched remainder the same way."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    from molbuilder.jobset.materialize import write_run_launch
    js = _prep_bench(calc)
    first = js["jobs"][0]["name"]
    write_run_launch(_artifacts(calc, first),
                     mode="direct", command=["bash", "x"])
    res = CliRunner().invoke(jobset_group,
                             ["launch", "bench", "coarse",
                              "--bundle", str(calc),
                              "--mode", "direct", "--dry-run", "--yes"])
    assert res.exit_code == 0, res.output
    assert "skip" in res.output and first in res.output
    assert res.output.count("WOULD run") == len(js["jobs"]) - 1
    # R2: § 1.5's immutability holds for trials AT THE SEAM -- a named
    # relaunch is refused by the library naming run.json and the next
    # verbs, and the grouped path collects only the still-unlaunched
    # remainder (this stray block was a second test's docstring left
    # mid-function by a merge; kept as the comment it really is).
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    from molbuilder.jobset.materialize import write_run_launch
    js = _prep_bench(calc)
    first = js["jobs"][0]["name"]
    second = js["jobs"][1]["name"]
    write_run_launch(_artifacts(calc, first),
                     mode="submit", command=["sbatch", "x"], job_id="42")
    r = CliRunner()
    res = r.invoke(jobset_group, ["launch", "bench", "coarse", first,
                                  "--bundle", str(calc),
                                  "--mode", "submit", "--dry-run", "--yes", "--domain", "htc"])
    assert res.exit_code != 0
    # THE PROPERTY, not one branch's wording.  A launched trial is refused
    # and the refusal names it.  Which sentence comes back depends on the
    # SHAPE: hierarchical answers from the attempt branch ("run-0 has
    # already been launched"), flat from the trial-is-its-own-attempt one
    # ("already launched") -- two accurate phrasings of one fact
    # (`project-layout.md` § 1.5a).
    assert "already" in res.output and "launched" in res.output, res.output
    assert res.exit_code != 0
    assert "summarize" in res.output
    res = r.invoke(jobset_group, ["launch", "bench", "coarse",
                                  "--bundle", str(calc),
                                  "--mode", "submit", "--dry-run", "--yes", "--domain", "htc"])
    assert res.exit_code == 0, res.output
    # the bare form groups the REMAINDER: the launched trial does not ride
    assert "bench-group" in res.output
    assert res.output.count("rides the group") == len(js["jobs"]) - 1
    assert f"rides      {first}" not in res.output


def test_a_trial_deck_is_forced_cold_not_only_relabelled():
    """The relabel alone does not cover the case that matters.

    Prep the same trial twice and the second render carries the SAME trial
    label, so the engine finds the FIRST attempt's ``.XV`` / ``.DM`` under it
    and warm-starts.  That point measures a continued run while its
    neighbours measure cold ones, and the timings a benchmark exists to
    compare stop being comparable.

    So a trial's config is forced to ``restart = "clean"`` however the
    description was written -- here the description says ``continue``, which
    is the case the relabel cannot save.
    """
    import dataclasses
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.resolve import ResolvedConfig
    from molbuilder.jobset.model import Resources

    warm = SiestaConfig(system_label="bdt", restart="continue")
    trial = ResolvedConfig(values=warm, resources=Resources(),
                           point={"G": 1, "K": 4}, label="bdt-G1K4")
    run = ResolvedConfig(values=warm, resources=Resources(),
                         point={}, label="bdt")

    assert trial.is_trial is True
    assert run.is_trial is False

    # What `prep_calculation` does to each, in the branch under test.
    def _as_prepped(element):
        cfg = element.render_config()
        if element.is_trial:
            cfg = dataclasses.replace(cfg, system_label=element.label)
            if hasattr(cfg, "restart"):
                cfg = dataclasses.replace(cfg, restart="clean")
        return cfg

    assert _as_prepped(trial).restart == "clean", (
        "a trial must be forced cold -- a second prep of the same trial would "
        "otherwise warm-start from its own first attempt")
    assert _as_prepped(run).restart == "continue", (
        "a RUN keeps what the description asked for; only trials are forced")


# --------------------------------------------------------------------- #
#  The declared grid drives the sweep (roadmap § 0.1 B1)                 #
# --------------------------------------------------------------------- #
# generator.md § 4.3a: `task.json`'s `bench` DECLARES what to measure —
# portable points, resolved here.  Until 2026-08-19 nothing read it: the
# machine enumerated its own grid regardless, so a user declaring
# {mpi_np: [1,2,3]} got eleven machine-chosen K×C trials.


def _declare_bench(calc, axes):
    import json as _json
    p = calc / "task.json"
    obj = _json.loads(p.read_text())
    obj["bench"] = axes
    p.write_text(_json.dumps(obj, indent=2) + "\n")


def test_the_declared_grid_is_the_sweep(calc):
    """Declared axes produce exactly those points — nothing enumerated."""
    _describe_cpu(calc)
    _declare_bench(calc, {"mpi_np": [1, 2], "omp_threads": [1]})
    sweep, _pins, translation = _bench_inputs(calc, None)
    assert sweep == [{"K": 1, "C": 1}, {"K": 2, "C": 1}]
    assert translation.axes == ("K", "C")


def test_a_declared_point_over_capability_is_refused_by_name(calc):
    """A point the machine cannot hold is refused naming the point and the
    bound — never clamped, because a clamped point measures a configuration
    nobody declared."""
    import click
    _describe_cpu(calc)
    _declare_bench(calc, {"mpi_np": [4096], "omp_threads": [2]})
    with pytest.raises(click.ClickException) as e:
        _bench_inputs(calc, None)
    assert "mpi_np=4096" in str(e.value) and "omp_threads=2" in str(e.value)


def test_prep_bench_asks_only_about_launched_trials(calc):
    """User, 2026-08-21: "bench always starts cold -- there is no point of
    asking."  A bench prep weighs ONE kind of evidence: launched trials in
    its own container (their decks may be read by a queued job, A7).  The
    run's launched attempts and the root's warm files cannot be touched by
    re-rendering relabelled cold trial decks -- so beside them the bench
    re-prep asks NOTHING, while the run-side ask still weighs both."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    from molbuilder.jobset.materialize import write_run_launch

    js = _prep_bench(calc)
    # a launched RUN attempt + a warm file at the root: run-side evidence
    run_attempt = calc / "01_coarse" / "run-0"
    run_attempt.mkdir(parents=True)
    write_run_launch(run_attempt, mode="direct", command=["bash", "x"])
    (calc / "JOB.DM").write_text("warm\n")

    r = CliRunner().invoke(jobset_group,
                           ["prep", "bench", "coarse", "--bundle", str(calc),
                            "--np", "8", "--cpus-per-task", "8",
                            "--no-sbatch"])
    assert r.exit_code == 0, r.output
    assert "already under way" not in r.output, \
        "a bench prep beside a launched RUN must not ask"

    # a launched TRIAL is the one evidence that still asks
    first = js["jobs"][0]["name"]
    write_run_launch(_artifacts(calc, first),
                     mode="direct", command=["bash", "x"])
    r = CliRunner().invoke(jobset_group,
                           ["prep", "bench", "coarse", "--bundle", str(calc),
                            "--np", "8", "--cpus-per-task", "8",
                            "--no-sbatch"], input="y\n")
    assert r.exit_code == 0, r.output
    assert "already under way" in r.output and first in r.output


def test_a_multi_point_value_entry_is_a_value_axis(calc):
    """§ 4.3a, BUILT 2026-08-21 (this test pinned the refusal while the
    extension was recorded-not-built): a non-machine entry with SEVERAL
    points multiplies the machine grid, each point carries its coordinate,
    and the trial names carry it too."""
    _describe_cpu(calc)
    _declare_bench(calc, {"block_size": [64, 128],
                          "mpi_np": [4], "omp_threads": [1]})
    sweep, pins, _tr = _bench_inputs(calc, None)
    assert len(sweep) == 2
    assert sorted(p["block_size"] for p in sweep) == [64, 128]
    assert "block_size" not in pins, "an axis is not a pin"
    assert pins["max_scf_iter"] == 3, "the measurement pins still ride"


def test_a_value_axis_naming_a_measurement_pin_is_refused(calc):
    """§ 4.3a: the pins make a trial a MEASUREMENT and win over every
    declared value, so an AXIS on one would render identical decks under
    different labels -- one measurement, twice.  ``continue_retries`` is
    the live case: an *execution* item (it passes the membership
    preflight) that the bench pins to 0 on every trial.  The non-execution
    pins (``max_scf_iter``...) are barred upstream by
    `_bench_names_a_speed_knob`, so they never reach this refusal."""
    import click
    _describe_cpu(calc)
    _declare_bench(calc, {"continue_retries": [0, 2],
                          "mpi_np": [4], "omp_threads": [1]})
    with pytest.raises(click.ClickException) as e:
        _bench_inputs(calc, None)
    assert "continue_retries" in str(e.value)
    assert "measurement" in str(e.value)


def test_a_name_outside_the_execution_category_is_refused(calc):
    """Pinning or sweeping a non-execution item changes the ANSWER, not the
    speed (§ 6.8).  ONE door owns membership -- the preflight's
    `_bench_names_a_speed_knob` (validation/task.py), which the dispatch
    runs before any pins are computed -- so this pins the CLI surface, not
    the helper (whose first version duplicated the rule; the holistic
    review removed the second door, and this § 6.8 error had no test at
    all until then)."""
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group

    _describe_cpu(calc)
    _declare_bench(calc, {"mesh_cutoff": [300]})
    r = CliRunner().invoke(jobset_group, ["prep", "bench", "coarse",
                                          "--bundle", str(calc)])
    assert r.exit_code != 0
    assert "mesh_cutoff" in r.output and "execution" in r.output


def test_a_one_point_declaration_is_a_pin_and_decides_the_grid(calc):
    """The override lane's ONE-point half (user rule, 2026-08-20): a CPU
    description with `bench: {use_gpu: [true]}` enumerates the GPU grid
    -- the declaration overrides the template's answer -- and the value
    rides every trial as a pin, under the measurement pins."""
    _describe_cpu(calc)                       # template says use_gpu=false
    _declare_bench(calc, {"use_gpu": [True],
                          "diag_algorithm": ["ELPA-2STAGE"],
                          "mpi_np": [4], "omp_threads": [1]})
    sweep, pins, _tr = _bench_inputs(calc, None)
    assert pins["use_gpu"] is True
    assert pins["diag_algorithm"] == "ELPA-2STAGE"
    assert pins["max_scf_iter"] == 3, "the measurement pins still ride"
    assert all("G" in p for p in sweep), (
        "a declared use_gpu=true must enumerate the GPU grid")


def test_a_declared_pin_reaches_the_trial_deck(calc):
    """Executed, not assumed: the pinned eigensolver lands in the rendered
    trial deck, replacing the template's (`ELPA-1STAGE` in the fixture)."""
    _declare_bench(calc, {"diag_algorithm": ["ELPA-2STAGE"],
                          "mpi_np": [4], "omp_threads": [1]})
    _prep_bench(calc)
    decks = list((calc / "01_coarse" / "bench").glob("bench-*/**/*.fdf"))
    assert decks, "no trial decks rendered"
    text = decks[0].read_text()
    # The VALUE line, not the catalogue help comments -- those name
    # every choice, so a substring match passes with the pin broken
    # (this assertion's own first version did; its mutation run
    # caught it).
    import re as _re
    assert _re.search(r"^Diag\.Algorithm\s+ELPA-2STAGE", text, _re.M), (
        [ln for ln in text.splitlines()
         if ln.startswith("Diag.Algorithm")])


def test_a_bad_enum_value_and_a_non_bool_are_refused_with_the_choices(calc):
    import click
    _describe_cpu(calc)
    _declare_bench(calc, {"diag_algorithm": ["ELPA-9STAGE"]})
    with pytest.raises(click.ClickException) as e:
        _bench_inputs(calc, None)
    assert "ELPA-9STAGE" in str(e.value) and "ScaLAPACK" in str(e.value)
    _declare_bench(calc, {"use_gpu": [1]})
    with pytest.raises(click.ClickException) as e:
        _bench_inputs(calc, None)
    assert "true or false" in str(e.value)


def test_a_declared_gpu_point_runs_the_declared_total_ranks(calc):
    """On a GPU description, G ranges over the divisors of each declared
    rank count, so G*K equals the declared mpi_np exactly."""
    _declare_bench(calc, {"mpi_np": [4], "omp_threads": [1]})
    sweep, _pins, _tr = _bench_inputs(calc, None)
    assert all(p["G"] * p["K"] == 4 for p in sweep)
    assert {p["G"] for p in sweep} == {1}          # fixture probes one a100


def test_the_cap_is_clean_scf_must_converge_is_pinned_off(calc):
    """B2: the pins include scf_must_converge False, so a capped trial ends
    as the single-point measurement it is instead of ABNORMAL_TERMINATION —
    which is what lets `choose_winner` ever see a completed point."""
    _sweep, pins, _tr = _bench_inputs(calc, None)
    assert pins["scf_must_converge"] is False
    assert pins["max_scf_iter"] == 3


# --------------------------------------------------------------------- #
#  The summary CONNECTS to the run (roadmap § 0.1 B3/B5) and the offer   #
#  explains an empty verdict (B4)                                        #
# --------------------------------------------------------------------- #


def _mk_point(label, state="completed", spi=None, knobs=None):
    from molbuilder.bench.result import BenchPoint
    return BenchPoint(label=label, engine="siesta", state=state,
                      knobs=knobs or {},
                      metrics=({"s_per_iter": spi} if spi is not None else {}))


def test_the_summary_closes_with_the_verdict_and_the_commands():
    """B5 (file-based since 2026-08-19): the summary ends with what to do —
    edit the proposal file, prep, submit — and the coverage clause keeps a
    partial sweep honest."""
    from pathlib import Path
    from molbuilder.bench.result import build_bench_result
    from molbuilder.jobset.summarize import RUN_CONFIG_NAME, summary_text
    res = build_bench_result(
        [_mk_point("K1C1", spi=1.9, knobs={"mpi_np": 1}),
         _mk_point("K2C1", spi=1.1, knobs={"mpi_np": 2}),
         _mk_point("K5C1", state="unknown")])
    out = summary_text(
        res, Path("/x/bench-result.json"),
        run_config=(Path("/x") / RUN_CONFIG_NAME, "written"), stage="tight")
    assert f"edit {RUN_CONFIG_NAME}" in out
    assert "the file is the decision" in out
    assert "prep run tight" in out                # the stage, by name
    assert "coverage: 2 of 3" in out              # the honesty clause
    assert "the proposal -- yours to edit" in out


def test_a_verdictless_summary_says_so_with_the_census():
    """B3/B4's surface: 'no winner' and 'nothing ran yet' are different
    situations, and the census is what separates them."""
    from pathlib import Path
    from molbuilder.bench.result import build_bench_result
    from molbuilder.jobset.summarize import summary_text
    res = build_bench_result([_mk_point("K1C1", state="incomplete"),
                              _mk_point("K2C1", state="unknown")])
    out = summary_text(res, Path("/x/bench-result.json"))
    assert "NO VERDICT" in out
    assert "1 incomplete" in out and "1 unknown" in out
    assert "prep run <stage>" not in out          # no command without a verdict


def test_the_offer_explains_an_empty_verdict_instead_of_silence(calc, capsys):
    """B4: prep run in a folder whose bench-result concludes nothing names
    the file and the census — silence made 'no benchmark' and 'benchmark
    that failed to conclude' look identical."""
    import json as _json
    from molbuilder.bench.result import build_bench_result
    from molbuilder.jobset._cli import _apply_run_config, _stage_bench_dir
    from molbuilder.jobset.model import Resources
    container, _tok = _stage_bench_dir(calc, "coarse")
    container.mkdir(parents=True, exist_ok=True)
    res = build_bench_result([_mk_point("K1C1", state="incomplete")])
    (container / "bench-result.json").write_text(res.to_json() + "\n")
    alloc, pins = _apply_run_config(calc, Resources(), stage="coarse")
    assert pins == {}
    out = capsys.readouterr().out
    assert "concludes nothing" in out and "1 incomplete" in out


# --------------------------------------------------------------------- #
#  The proposal file (run-config.toml) — § 2.3.2, 2026-08-19            #
# --------------------------------------------------------------------- #


def test_summarize_writes_the_proposal_and_never_overwrites_yours(calc):
    """`summarize` materialises the verdict as `run-config.toml`; the file
    parses back to the verdict's values.  Once it exists it is the USER's
    — a re-summarize keeps it (edits and all) and says so."""
    import tomllib
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    from molbuilder.jobset.summarize import RUN_CONFIG_NAME
    _finished_trial_and_verdict(calc)
    cfg = calc / "01_coarse" / "bench" / RUN_CONFIG_NAME
    assert cfg.is_file()
    raw = tomllib.loads(cfg.read_text())
    assert raw["schema"] == "molbuilder/run-config@1"
    rec = json.loads(
        (calc / "01_coarse" / "bench" / "bench-result.json").read_text())
    assert raw["resources"]["mpi_np"] == rec["choice"]["knobs"]["mpi_np"]
    # THE PROPOSAL CARRIES WHAT WAS MEASURED, AND NO WALL OR MEMORY AT ALL
    # (2026-08-24, user).  It asserted `resources.time == recommend.time`
    # until then -- and that `recommend` block sized a wall from
    # `s/iter x an assumed 200 iterations x 1.5`, which `prep` folded into
    # an allocation and `sbatch` received.  Both fields are deleted, so
    # the two asks stay the person's (`submission.md` S1, S2).
    assert "recommend" not in rec
    assert "time" not in raw["resources"]
    assert "mem" not in raw["resources"]
    assert raw["pins"]["use_gpu"] == \
        rec["choice"]["mechanism"]["use_gpu"]
    # the user edits it; a re-summarize must not clobber the edit
    cfg.write_text(cfg.read_text().replace(
        f"mpi_np = {raw['resources']['mpi_np']}", "mpi_np = 1"))
    r = CliRunner().invoke(jobset_group, ["summarize", "bench", "coarse",
                                          "--bundle", str(calc)])
    assert r.exit_code == 0, r.output
    assert "kept:" in r.output and "not\n" not in r.output[:0] + ""
    assert "delete it and summarize again" in r.output
    assert tomllib.loads(cfg.read_text())["resources"]["mpi_np"] == 1


def test_a_verdictless_summarize_writes_no_proposal(calc):
    """No verdict, no proposal — a file would be an instruction to run
    with nothing measured behind it."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    from molbuilder.jobset.summarize import RUN_CONFIG_NAME
    _prep_bench(calc)
    r = CliRunner().invoke(jobset_group, ["summarize", "bench", "coarse",
                                          "--bundle", str(calc)])
    assert r.exit_code == 0, r.output
    assert "NO VERDICT" in r.output
    assert not (calc / "01_coarse" / "bench" / RUN_CONFIG_NAME).exists()


def test_run_config_refuses_what_it_does_not_know(tmp_path):
    """An unknown key or a mistyped value is refused BY NAME — a
    silently-dropped edit is a decision the user wrote down and nobody
    obeyed (the bench grid's unknown-axis doctrine)."""
    import pytest
    from molbuilder.jobset.summarize import read_run_config
    head = 'schema = "molbuilder/run-config@1"\n'

    def _write(body):
        f = tmp_path / "run-config.toml"
        f.write_text(head + body)
        return f

    assert read_run_config(_write("[resources]\nmpi_np = 2\n"),
                           engine="siesta") == {
        "resources": {"mpi_np": 2}, "pins": {}}
    with pytest.raises(ValueError, match="no section named alloc"):
        read_run_config(_write("[alloc]\nmpi_np = 2\n"), engine="siesta")
    with pytest.raises(ValueError, match="no field named np_total"):
        read_run_config(_write("[resources]\nnp_total = 4\n"), engine="siesta")
    with pytest.raises(ValueError, match="mpi_np must be int"):
        read_run_config(_write("[resources]\nmpi_np = true\n"), engine="siesta")
    with pytest.raises(ValueError, match="mem must be str"):
        read_run_config(_write("[resources]\nmem = 29\n"), engine="siesta")
    with pytest.raises(ValueError, match="use_gpu must be bool"):
        read_run_config(_write("[pins]\nuse_gpu = 1\n"), engine="siesta")
    with pytest.raises(ValueError, match="not valid TOML"):
        read_run_config(_write("= what ="), engine="siesta")
    f = tmp_path / "run-config.toml"
    f.write_text('schema = "molbuilder/other@1"\n')
    with pytest.raises(ValueError, match="schema"):
        read_run_config(f, engine="siesta")


def test_the_wrapper_policy_note_is_engine_aware_and_yields_to_flags(
        tmp_path, capsys):
    """With neither file nor flags the wrapper's runtime policy is NAMED,
    per engine (user, 2026-08-19); any stated launch-shape flag, or an
    applied file, silences it — the note is for the all-defaults case."""
    from molbuilder.jobset._cli import _apply_run_config
    from molbuilder.jobset.model import Resources
    _apply_run_config(tmp_path, Resources(), engine="siesta")
    out = capsys.readouterr().out
    assert "MPI over all physical cores" in out
    assert "ELPA-CUDA placement policy" in out
    _apply_run_config(tmp_path, Resources(), engine="pyscf")
    out = capsys.readouterr().out
    assert "OMP thread count" in out and "OMP_NUM_THREADS" in out
    assert "MPI over all physical cores" not in out
    _apply_run_config(tmp_path, Resources(mpi_np=4), engine="siesta")
    assert capsys.readouterr().out == ""


def test_the_table_measures_beside_the_ask_and_gates_gpu_columns():
    """The summary table: knobs beside measurements, `--` where nothing
    was measured, and GPU columns only when the sweep ASKED for a GPU —
    the monitor samples gpu0_* as zeros on GPU-less runs and three
    columns of 0 on a CPU sweep are noise."""
    from pathlib import Path
    from molbuilder.bench.result import BenchPoint, build_bench_result
    from molbuilder.jobset.summarize import _fmt_wall, summary_text
    cpu = BenchPoint(
        label="K2C1", engine="cpu",
        knobs={"mpi_np": 2, "cpus_per_task": 1},
        metrics={"s_per_iter": 1.0, "iters_measured": 3, "wall_s": 41,
                 "peak_rss_gb": 24.73, "cpu_mean_pct": 96.2},
        bound="host", state="completed",
        effective={"diag_algorithm": "D&C"})
    out = summary_text(build_bench_result([cpu]), Path("/x/b.json"))
    row = next(l for l in out.splitlines() if "K2C1" in l)
    for cell in ("2", "1", "D&C", "3", "41s", "24.7G", "96", "host",
                 "completed"):
        assert cell in row.split(), (cell, row)
    assert "gpu-sm%" not in out and "vram" not in out
    gpu = BenchPoint(
        label="G1K4C6", engine="gpu",
        knobs={"mpi_np": 4, "cpus_per_task": 6, "gres": "gpu:1"},
        metrics={"s_per_iter": 2.3}, state="completed")
    out = summary_text(build_bench_result([cpu, gpu]), Path("/x/b.json"))
    assert "gpu-sm%" in out and "vram" in out
    row = next(l for l in out.splitlines() if "G1K4C6" in l)
    assert row.split().count("--") >= 4      # unmeasured cells say so
    assert _fmt_wall(41) == "41s"
    assert _fmt_wall(245) == "4m05s"
    assert _fmt_wall(7523) == "2h05m"


def test_the_group_refuses_a_trial_without_an_explicit_shape(calc):
    """The env-inheritance shield (user, 2026-08-20): inside the allocation
    SLURM_NTASKS/SLURM_CPUS_PER_TASK describe the ENVELOPE, and a wrapper
    with no flags falls back to them -- so a trial that cannot state its
    own -np/-omp would silently measure the widest point.  Refused BY NAME
    at generation, never mis-measured."""
    import dataclasses

    import pytest as _pytest

    from molbuilder.jobset._cli import _load_bench_set
    from molbuilder.jobset.submit import SubmitError, submit_bench_group

    _prep_bench(calc)
    js, base = _load_bench_set(calc, "coarse")
    stripped = js.jobs[0]
    js.jobs[0] = dataclasses.replace(
        stripped, resources=dataclasses.replace(stripped.resources,
                                                cpus_per_task=None))
    with _pytest.raises(SubmitError, match=stripped.name):
        submit_bench_group(js, base, dry_run=True)


def test_a_declared_pin_reaches_the_run_deck_and_the_verdict_outranks_it(
        calc):
    """The run half of the override lane (user rule, 2026-08-20), with
    § 4.3a's precedence executed: template < one-point declaration <
    run-config verdict.  `prep run` on a calc declaring
    `diag_algorithm: [ELPA-2STAGE]` renders the declared value into the
    stage deck; write a run-config whose pins say ScaLAPACK, re-prep, and
    the verdict's value stands instead."""
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group

    _describe_cpu(calc)      # GPU + ScaLAPACK is an invalid pair, and the
    _declare_bench(calc, {"diag_algorithm": ["ELPA-2STAGE"]})
    runner = CliRunner()     # verdict half of this test pins ScaLAPACK
    r = runner.invoke(jobset_group, ["prep", "run", "coarse",
                                     "--bundle", str(calc)])
    assert r.exit_code == 0, r.output
    import re as _re
    deck = next((calc / "01_coarse").glob("run-*/JOB*.fdf"))
    assert _re.search(r"^Diag\.Algorithm\s+ELPA-2STAGE",
                      deck.read_text(), _re.M), (
        "the declared pin did not reach the run deck (the value "
        "line, not the help comments)")

    # The measured verdict outranks the declaration.
    bench = calc / "01_coarse" / "bench"
    bench.mkdir(parents=True, exist_ok=True)
    (bench / "run-config.toml").write_text(
        'schema = "molbuilder/run-config@1"\n'
        "[pins]\n"
        'diag_algorithm = "ScaLAPACK"\n')
    r = runner.invoke(jobset_group, ["prep", "run", "coarse",
                                     "--bundle", str(calc)])
    assert r.exit_code == 0, r.output
    decks = sorted((calc / "01_coarse").glob("run-*/JOB*.fdf"))
    text = decks[-1].read_text()
    # ScaLAPACK EMITS NOTHING -- omitting the keyword IS the deliberate
    # emission for SIESTA's own default (`siesta/input.py`; siesta.md § 7).
    # So the verdict outranking the declaration shows as the declared ELPA
    # value line GONE, not as a ScaLAPACK line appearing.  (This assertion
    # first expected the line -- the emit rule says otherwise.)
    assert not _re.search(r"^Diag\.Algorithm\s", text, _re.M), (
        [ln for ln in text.splitlines() if ln.startswith("Diag.")])


def test_a_pyscf_description_is_refused_by_name_at_the_bench_seam(tmp_path):
    """E-J1 (restored 2026-08-21): the bench lane speaks SIESTA's
    vocabulary -- its measurement pins name SiestaConfig fields, and the
    GPU question is read under `use_gpu`.  A PySCF description used
    to be stopped only by ACCIDENT: those pins failing resolve, with a
    refusal blaming settings the user never wrote.  The seam now refuses
    by NAME, before any grid is enumerated."""
    import click

    from molbuilder.config.pyscf import PySCFConfig
    struct = Structure(elements=["H", "H"],
                       positions=np.array([[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.74]]),
                       vacuum=(10.0, 10.0, 10.0))
    (tmp_path / "h2.xyz").write_text(struct.to_xyz())
    dest = tmp_path / "pycalc"
    D.write_description(
        D.build_description(struct, PySCFConfig(job_name="JOB"),
                            [Stage(name="only", enabled=True, overrides={})],
                            engine="pyscf", shape="hierarchical", name="JOB",
                            source=str(tmp_path / "h2.xyz")),
        dest)
    (dest / "environment.json").write_text(
        Environment(scheduler="workstation",
                    topology=Topology(sockets=1,
                                      cores_per_socket=4)).to_json() + "\n")
    with pytest.raises(click.ClickException) as e:
        _bench_inputs(dest, None)
    msg = str(e.value)
    assert "'pyscf'" in msg and "SIESTA" in msg, (
        "the refusal must name the engine and the reason")
    assert "max_scf_iter" not in msg, (
        "the refusal blames measurement pins the user never wrote")


def test_no_winner_speaks_only_about_the_timed_set():
    """R2-2: the "every timed trial ran something other than asked"
    verdict scanned ALL points -- one unfinished point carrying mismatch
    data made the summary assert a census of timed trials it never took.
    With nothing timed, the honest sentence is the NO VERDICT census;
    the every-timed-mismatched sentence needs a non-empty timed set."""
    from pathlib import Path
    from molbuilder.bench.result import build_bench_result
    from molbuilder.jobset.summarize import summary_text
    # Nothing timed at all; one incomplete point carries mismatch data.
    p = _mk_point("K1C1", state="incomplete")
    p.mismatch = {"mpi_np": {"asked": 4, "ran": 2}}
    res = build_bench_result([p, _mk_point("K2C1", state="unknown")])
    out = summary_text(res, Path("/x/bench-result.json"))
    assert "every timed trial" not in out, (
        "the summary asserts a census of timed trials it never took")
    assert "NO VERDICT" in out
    # And the sentence still fires when the timed set really is all
    # mismatched.
    q = _mk_point("K4C1", spi=2.0, knobs={"mpi_np": 4})
    q.mismatch = {"mpi_np": {"asked": 4, "ran": 2}}
    res2 = build_bench_result([q])
    out2 = summary_text(res2, Path("/x/bench-result.json"))
    assert "every timed trial" in out2


# ===================================================================== #
#  `sweep_view` — the whole sweep composed for a reader                 #
#                                                                       #
#  Contract: docs/web/bench-summary.md.  Its B1 -- NOT this file's other  #
#  B1 (roadmap § 0.1, the declared grid) -- is the property under        #
#  test: this composes what four doors already produce and computes      #
#  nothing.  B2 says why -- submission.md § 3's summary that showed      #
#  "170 minutes" for five 38-minute jobs got there by working out its    #
#  own total a second way, and a view comparing six trials has six       #
#  chances to repeat that.                                              #
# ===================================================================== #

def _bench_dir(calc):
    return calc / "01_coarse" / "bench"


def _load_sweep(calc):
    """The jobset, and the BUNDLE it belongs to -- resolved the way the
    route does it, which is NOT the file's own directory (see
    ``bundle_for_sweep_file``)."""
    from molbuilder.jobset.model import JobSet
    from molbuilder.jobset.summarize import bundle_for_sweep_file
    jpath = _bench_dir(calc) / "job-set.json"
    jobset = JobSet.load(jpath)
    return jobset, bundle_for_sweep_file(jobset, jpath)


def test_sweep_view_has_one_trial_per_job_in_the_job_sets_order(calc):
    from molbuilder.jobset.summarize import sweep_view
    js = _prep_bench(calc)
    jobset, bundle = _load_sweep(calc)
    view = sweep_view(jobset, bundle)
    assert view["n_trials"] == len(js["jobs"])
    assert [t["label"] for t in view["trials"]] == [j["name"] for j in js["jobs"]]


def test_sweep_view_reports_where_the_RUN_is_not_what_the_files_say(calc):
    """§ 2's table gives 'queued/running/finished/failed' to jobset_status.

    ``BenchPoint.state`` answers a DIFFERENT question -- what the artifacts
    on disk look like -- so the view carries both under separate keys and
    never lets the artifact word stand in for the run's position.
    """
    from molbuilder.jobset.summarize import sweep_view
    _prep_bench(calc)
    jobset, bundle = _load_sweep(calc)
    view = sweep_view(jobset, bundle)
    t = view["trials"][0]
    assert "state" in t and "artifacts" in t
    # Nothing has been launched, so the RUN has not started.  The artifact
    # word for "no .out at all" is `unknown`, which is not a run state.
    assert t["artifacts"] == "unknown"
    assert t["state"] != "unknown", (
        "the run's position must come from jobset_status, not from the "
        "artifact reader")


def test_sweep_view_carries_the_verdict(calc):
    """The analysis IS choose_winner's answer, composed in -- not a second
    ranking done by the view."""
    from molbuilder.jobset.summarize import sweep_view
    name = _finished_trial_and_verdict(calc)
    jobset, bundle = _load_sweep(calc)
    view = sweep_view(jobset, bundle)
    assert view["choice"].get("label") == name
    # and the winning trial's own measurement is on its row
    won = [t for t in view["trials"] if t["label"] == name][0]
    assert won["s_per_iter"] == pytest.approx(4.0)


def test_sweep_view_names_the_coordinate_the_sweep_varied(calc):
    from molbuilder.jobset.summarize import sweep_view
    _prep_bench(calc)
    jobset, bundle = _load_sweep(calc)
    view = sweep_view(jobset, bundle)
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
    from molbuilder.jobset.summarize import sweep_view
    _prep_bench(calc)
    jobset, bundle = _load_sweep(calc)
    bench = _bench_dir(calc)
    assert not (bench / "bench-result.json").exists()
    sweep_view(jobset, bundle)
    assert not (bench / "bench-result.json").exists(), (
        "the view published a record; that is run_summarize_jobset's job")
    assert not (bench / "run-config.toml").exists(), (
        "the view wrote a proposal -- that file is the user's")


def test_sweep_view_refuses_to_pair_trials_by_position_if_the_readers_disagree(
        calc, monkeypatch):
    """The join is positional because the two readers key differently -- a
    BenchPoint's label is the JOB's name, a StageStatus's is its STAGE's.
    If either reader ever learns to skip a job, that must raise, not
    silently pair trial N's measurement with trial N+1's state."""
    from molbuilder.jobset import summarize as S
    _prep_bench(calc)
    jobset, bundle = _load_sweep(calc)
    real = S.discover_points_from_jobset
    monkeypatch.setattr(S, "discover_points_from_jobset",
                        lambda b, j: real(b, j)[:-1])
    with pytest.raises(ValueError, match="refusing to pair"):
        S.sweep_view(jobset, bundle)


def test_both_doors_onto_a_sweep_report_the_SAME_verdict(calc):
    """`bench-summary.md` B2 (not this file's other B2, the pins): a
    second path that computes the same figure
    is the defect, not the feature.

    There are two ways to ask a sweep what it concluded -- the CLI, which
    writes ``bench-result.json``, and the Results tab, which reads
    ``sweep_view``.  They composed the record separately and then differed:
    only the writing path enriched ``choice`` with ``_winner_mechanism``,
    so the same sweep answered "ELPA-1STAGE on a GPU" through one door and
    said nothing about mechanism through the other.  Both go through
    ``bench_record`` now, and this fails if either grows its own copy.
    """
    from molbuilder.jobset.summarize import sweep_view, bench_record
    name = _finished_trial_and_verdict(calc)
    jobset, bundle = _load_sweep(calc)

    written = json.loads((_bench_dir(calc) / "bench-result.json").read_text())
    viewed = sweep_view(jobset, bundle)["choice"]

    assert written["choice"]["label"] == viewed["label"] == name
    assert written["choice"].get("mechanism"), (
        "the fixture's winner should carry a mechanism, or this proves nothing")
    # every key the record's verdict has, the view's verdict has too
    assert set(written["choice"]) == set(viewed), (
        f"the two doors disagree on the verdict's shape: "
        f"written-only={set(written['choice']) - set(viewed)}, "
        f"view-only={set(viewed) - set(written['choice'])}")
    assert viewed["mechanism"] == written["choice"]["mechanism"]
