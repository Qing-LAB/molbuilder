"""The launch door -- through the road: `molbuilder describe`, `prep`, `launch`.

PINS: ``docs/execution/job-system.md`` § 6.0 (launch sends the header its
prep wrote, each stage its own; `-J <calculation>/<job>`);
``docs/execution/architecture.md`` § 5.2 (every launch value is stated, at
prep); ``docs/execution/submission.md`` S4 (nothing is submitted unseen --
every door); ``docs/execution/running-a-job.md`` § 5.5 (a flag the launch
would not read is refused).

PREVENTS: the queue one stage's prep baked routing every stage of the
ladder; a direct run without the ranks and threads its prep stated; a send
over a folder changed since its plan.

The scheduler's submit command is a stub where the machine's record says it
is, off PATH, that queues nothing and writes every call down
(`support.road.a_machine_with_queues`).  Nothing here launches an engine.
"""
from __future__ import annotations

import json

import pytest
from molbuilder.calculation import open as open_calculation

from support.road import (read_request, write_request,
                          a_machine_with_queues, calls_made, describe_calculation,
                          group_words, jobset, stage_file)


def _queues(gpu: bool = False):
    from molbuilder.scheduler import Domain
    rows = [Domain(name="debug", partition="htc", qos="debug",
                   max_time="0-00:15:00"),
            Domain(name="htc", partition="htc", qos="public",
                   max_time="0-04:00:00")]
    if gpu:
        rows.append(Domain(name="gpu", partition="gpu", qos="public",
                           max_time="0-04:00:00",
                           gpu={"a100": 4}))
    return rows


def _states_its_wall_and_memory(bundle, **more):
    """The calculation states its wall and its memory -- its job request's
    `allocation` -- as a run on a scheduler does, or prep refuses
    (`architecture.md` § 5.2).  Its QUEUE each prep names for its own stage
    (`--domain`), which is what this file is about -- or, given here, the
    job request names it for every stage."""
    request = read_request(bundle)
    request["allocation"] = {"time": "0-01:00:00", "mem": "8G", **more}
    write_request(bundle, request)
    return bundle


@pytest.fixture
def cluster(tmp_path, monkeypatch):
    """A machine whose record names two queues -- `debug` (15 minutes) and
    `htc` (4 hours) -- an `sbatch` that writes down each call and refuses
    it (no scheduler text in the basic tests), and the H2 ladder described
    on it, stating its wall and memory: ``(bundle, calls)``."""
    calls = a_machine_with_queues(tmp_path, monkeypatch, _queues())
    return _states_its_wall_and_memory(describe_calculation(tmp_path, monkeypatch)), \
        calls


def _prep(bundle, stage="coarse", *more):
    r = jobset("prep", "task", *group_words(bundle, stage), "--bundle", bundle,
               "--target", "this",
               *more)
    assert r.exit_code == 0, r.output
    return r


from support.road import sbatch_line as _line


def _flag(argv, flag):
    return argv[argv.index(flag) + 1]


def _ledger(bundle):
    from molbuilder.jobset.ledger import LEDGER_FILE
    return [json.loads(ln) for ln in
            (bundle / LEDGER_FILE).read_text().splitlines()]


def test_the_queue_prep_baked_belongs_to_its_own_stage(cluster):
    """`prep task --stage coarse --domain debug` names coarse's queue -- not
    medium's.  Medium, prepared naming none, is refused at its prep with the
    record's queues listed (a run on a scheduler names its queue,
    `architecture.md` § 5.2); named, its header carries its own -- and
    launch sends each stage's own header.

    MUTATION THIS MUST FAIL AGAINST: a header rendered from another stage's
    placement (medium's on coarse's `debug`)."""
    bundle, calls = cluster
    _prep(bundle, "coarse", "--domain", "debug", "--time", "10m")
    r = jobset("prep", "task", "--group", "2", "--bundle", bundle,
               "--target", "this")
    assert r.exit_code != 0, r.output
    for said in ("the target's record lists:", "htc/debug", "htc/public"):
        assert said in r.output, r.output
    _prep(bundle, "medium", "--domain", "htc")
    for stage, qos in (("coarse", "debug"), ("medium", "public")):
        header = stage_file(bundle, stage, ".sbatch")
        assert f"#SBATCH -q {qos}" in header.read_text().splitlines(), stage
        # MEDIUM'S FIRST LAUNCH from the structure (`--cold`): coarse has
        # not run, and what medium would take from it is not this test's
        # question.
        r = jobset("launch", "task", "--stage", stage, "--bundle", bundle,
                   "--mode", "submit", "--dry-run",
                   *(("--cold",) if stage == "medium" else ()))
        assert r.exit_code == 0, r.output
        assert header.name in _line(r.output), r.output


def test_direct_runs_what_was_typed_and_refuses_what_it_would_not_read(
        cluster):
    """`--mode direct` runs the stage's own wrapper here, the ranks and
    threads its prep stated as arguments; a submitted job's wall (`--time`)
    beside it is refused by name.
    A bundle whose config says `launch.mode: submit` launches that way when
    nothing is typed -- sending the header its prep wrote -- and `--mode
    direct` typed over it runs here.

    MUTATION THIS MUST FAIL AGAINST: a direct run without -np/-omp."""
    bundle, _calls = cluster
    _prep(bundle, "coarse", "--np", "4", "--cpus-per-task", "2",
          "--domain", "htc")
    r = jobset("launch", "task", "--stage", "coarse", "--bundle", bundle,
               "--mode", "direct", "--dry-run")
    assert r.exit_code == 0, r.output
    run = next(ln.split() for ln in r.output.splitlines()
               if "WOULD start" in ln)
    assert run[run.index("bash") + 1] == "H2_01_coarse.run.sh", run
    assert (_flag(run, "-np"), _flag(run, "-omp")) == ("4", "2"), run
    r = jobset("launch", "task", "--stage", "coarse", "--bundle", bundle,
               "--mode", "direct", "--time", "1h")
    assert r.exit_code != 0 and "--time" in r.output, r.output

    from molbuilder.runtime_config import write_config_scope
    write_config_scope({"launch": {"mode": "submit"}})
    r = jobset("launch", "task", "--stage", "coarse", "--bundle", bundle, "--dry-run")
    assert r.exit_code == 0, r.output
    assert (stage_file(bundle, "coarse", ".sbatch").name
            in _line(r.output)), r.output
    r = jobset("launch", "task", "--stage", "coarse", "--bundle", bundle,
               "--mode", "direct", "--dry-run")
    assert r.exit_code == 0, r.output


def test_a_send_over_a_folder_changed_since_its_plan_is_refused(cluster):
    """`job-system.md` § 6.0, step 4: the send checks the folder against the
    one its plan was made from -- a file the plan read, written since, is
    refused, saying to launch again; nothing is sent, and the refusal is
    written down.

    API-LEVEL because the road cannot reach it: one `launch` makes its plan
    and sends it in one command, and the folder changes in between only
    while the person reads the question.  The plan and the send are the
    entry's own two calls (`submit.plan_launch`, `submit.send_launch`); the
    change between them is a person's.

    MUTATION THIS MUST FAIL AGAINST: the send comparing nothing."""
    import os

    from molbuilder.jobset.submit import SubmitError, plan_launch, send_launch
    from molbuilder.jobset.ask import Said
    bundle, calls = cluster
    _prep(bundle, "coarse", "--domain", "debug", "--time", "10m")
    attempt = bundle / "01_coarse" / "run-0"

    def plan():                  # the queue its prep placed it on: `debug`
        return plan_launch(open_calculation(bundle), mode="submit",
                           only="coarse",
                           told=dict(kind="task", stage="coarse", trial=None,
                                     mode="submit", flags=[]))

    shown = plan()
    # THE FILE THE PLAN READ -- prep's deck in the stage's folder, which the
    # run is filled from at the send (`_Member.read_from`).
    deck = next((bundle / "01_coarse").glob("*.fdf"))
    assert any(deck.name in r for r in shown.reads), shown.reads
    st = deck.stat()
    os.utime(deck, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))
    with pytest.raises(SubmitError) as e:
        send_launch(shown, said=Said(True, "yes (--yes)"))
    said = str(e.value)
    assert "the folder changed since this launch was planned" in said, said
    assert deck.name in said and "Launch again" in said, said
    assert calls_made(calls) == [] and not (attempt / "run.json").exists()
    assert _ledger(bundle)[-1]["decision"] == "refused"


def _steps_run(plan, text: str, cwd):
    """Every script a walk's steps run, and their walks' in turn -- the
    folder each step goes to from ``cwd``, where the walk runs, and the
    script it runs there with bash -- each as the plan leaves it."""
    import shlex
    out = []
    for line in text.splitlines():
        if 'run_member "' not in line:
            continue
        words = shlex.split(line.strip())
        at = words.index("run_member")
        assert words[at + 3] == "bash", line
        folder = cwd / words[at + 2]
        script = folder / words[at + 4]
        out.append(script)
        if plan.writes.is_file(script):
            out += _steps_run(plan, plan.writes.read_text(script), folder)
    return out


def test_a_groups_run_in_phases_is_walked_by_its_own_walk(tmp_path,
                                                          monkeypatch):
    """`job-system.md` § 6.0, step 1 -- a group's one job runs each member's
    run: "a run of points, its points".  A SIESTA displacement sweep
    (`engines/vibration.md` § 5.9) -- a vibration with a second stage of
    its own displacement, added on Task setup and put in the first stage's
    group -- prepared as group 1 and its launch planned: every step of the
    group's walk runs a script the launch leaves -- each stage's own walk,
    written by the plan, whose steps run the run's calls.

    API-LEVEL: the plan is the launch verb's own (`submit.plan_launch`); a
    send would start the engine.

    THE SILENT FAILURE: the group's job stepping into each stage's run for
    a run script a run in phases does not have there -- every stage
    failing at once, its calls never run.  MUTATION THIS MUST FAIL
    AGAINST: `_plan_group` walking every member as a single run
    (`_walk_of([m], base)` for a run of points too)."""
    from molbuilder.jobset.submit import plan_launch
    from support.road import save_both
    bundle = describe_calculation(tmp_path, monkeypatch,
                                  calculation="vibration")
    task = json.loads((bundle / "task.json").read_text())
    request = read_request(bundle)
    task["stages"].append({"name": "fine",
                           "overrides": {"fc_displacement": 0.02}})
    task["varies"] = sorted(set(task.get("varies") or [])
                            | {"fc_displacement"})
    request["stages"]["fine"] = {"group": 1}
    r = save_both(bundle, task, request)
    assert r.status_code == 200, r.get_json()
    r = jobset("prep", "task", "--group", "1", "--bundle", bundle,
               "--target", "this")
    assert r.exit_code == 0, r.output

    plan = plan_launch(open_calculation(bundle), mode="direct", group=1,
                       record=False,
                       told=dict(kind="task", stage=None, trial=None,
                                 mode="direct", flags=[]))
    (sub,) = plan.submissions
    walk = sub.cwd / sub.command[-1]
    ran = _steps_run(plan, plan.writes.read_text(walk), sub.cwd)
    # Each stage's walk, then its run's two calls: relax, fc.
    assert len(ran) == 6, ran
    assert all(plan.writes.is_file(s) for s in ran), [
        s.relative_to(bundle) for s in ran if not plan.writes.is_file(s)]
    assert {s.relative_to(bundle).parts[0] for s in ran} == {
        "01_freq", "02_fine"}, ran
