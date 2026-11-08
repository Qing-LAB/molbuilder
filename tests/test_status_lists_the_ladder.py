"""`jobset status` lists the description's ladder -- through the road:
`molbuilder describe`, the Task-setup save, `prep`, `status`, the Results tab.

PINS: ``docs/execution/job-system.md`` § 5.3 (the table is the description's
ladder: every stage with its number from the moment `molbuilder describe`
writes it, the ones not prepared yet as `ready` or `waiting`, a disabled
one never the stage to resume from; `status <stage>` is a stage in full --
its deck, what it declares, its resources) and ``docs/web/results.md``
§ 2.4 (the Results tab's ladder is `jobset_status`'s answer).

PREVENTS, each read in the code before 2026-10-01:

* `status` refusing a described calculation until its first prep, and listing
  only the stages prepared so far -- while the Results tab listed all of them,
  having composed the ladder itself;
* a stage disabled in the description named as the stage to resume from;
* the deck, carry set and resources behind a second verb, `plan`, that
  listed the same ladder from the same file.

Nothing here launches an engine.
"""
from __future__ import annotations

import json



def _row(output, name):
    """The status table's row for ``name``: ``[seq, stage, group, attempt, state,
    ...]``."""
    return next(ln.split() for ln in output.splitlines()
                if ln.split()[1:2] == [name])


def _steps(output):
    """`status`'s next step, one entry a group: ``[group 1's words and
    lines, group 2's, ...]``."""
    steps = output.split("each group, and what is next:\n", 1)[1]
    return steps.split("\n  group ")


def test_status_answers_what_it_cannot_read_and_where_it_was_asked(
        tmp_path, monkeypatch):
    """A template prep would refuse is said in the table's last line, never
    a traceback; a stage renamed in case only keeps its prepared job; asked
    from inside one of its stage folders, `status` names the calculation it
    belongs to.

    MUTATIONS THIS MUST FAIL AGAINST: the continuation door raising (a
    traceback); stages joined to jobs by exact name; a stage folder told to
    run `molbuilder describe`."""
    from support.road import (described, jobset, read_request, road_tree,
                              save_both)
    bundle = described(road_tree(tmp_path, monkeypatch))

    template = next(bundle.glob("*.template.toml"))
    kept = template.read_text()
    template.write_text("this is not a template [\n")
    r = jobset("status", "--bundle", bundle)
    assert r.exit_code == 0 and r.exception is None, r.output
    assert "group 1  coarse" in r.output, r.output
    assert "waits before it can be prepared" in r.output, r.output
    assert "continues from cannot be read" in r.output, r.output
    template.write_text(kept)

    assert jobset("prep", "task", "--group", "1", "--bundle", bundle,
                  "--target", "this").exit_code == 0

    task = json.loads((bundle / "task.json").read_text())
    task["stages"][0]["name"] = "COARSE"
    r = save_both(bundle, task, read_request(bundle))
    assert r.status_code == 200, r.get_json()
    r = jobset("status", "--bundle", bundle)
    assert r.exit_code == 0, r.output
    assert _row(r.output, "COARSE")[4] == "pending", r.output

    r = jobset("status", "--bundle", bundle / "01_coarse")
    assert r.exit_code != 0, r.output
    assert "is a folder of the calculation at" in r.output, r.output
    assert "molbuilder describe" not in r.output, r.output


def test_status_ends_with_each_groups_next_step(tmp_path, monkeypatch,
                                               web_client):
    """`job-system.md`, *The task*, *The next step*: `status` ends with every
    group, where it stands and the line that moves it, the first group not
    finished the next.  A described ladder: its first group not prepared,
    its line `prep task --group 1`; prepared, ready to launch, its line the
    launch of group 1 -- the groups after it not prepared, with no line;
    its run's folder then removed by hand -- against the design -- *missing*,
    naming the stage (*prepared, its run not on disk*), on its row, as its
    step, on Task setup's ladder and in its group's lines and the prep offer,
    the line its group's prep, which opens its run anew.  The silent
    failures: a person told no next line after a prep, or the line of a
    group whose turn has not come; a group whose run is gone said *ready to
    launch*, its launch then refused (R-L28)."""
    import shutil

    from molbuilder.calculation import open as open_calculation
    from molbuilder.jobset.materialize import job_folder, latest_attempt
    from support.road import described, jobset, road_tree
    bundle = described(road_tree(tmp_path, monkeypatch))
    r = jobset("status", "--bundle", bundle)
    assert r.exit_code == 0, r.output
    first, *rest = steps = _steps(r.output)
    assert "not prepared -- next:" in first, steps
    assert "prep task --group 1 --bundle" in first, steps
    assert rest and all("next" not in g and "jobset" not in g
                        for g in rest), steps
    assert jobset("prep", "task", "--group", "1", "--bundle", bundle,
                  "--target", "this").exit_code == 0
    r = jobset("status", "--bundle", bundle)
    first, *rest = steps = _steps(r.output)
    assert "ready to launch -- next:" in first, steps
    assert "launch task --group 1" in first, steps
    assert rest and all("not prepared" in g and "jobset" not in g
                        for g in rest), steps

    calc = open_calculation(bundle)
    stage = calc.task.stages[0].name
    shutil.rmtree(latest_attempt(job_folder(calc, calc.job(stage))))
    r = jobset("status", "--bundle", bundle)
    assert _row(r.output, stage)[4] == "missing", r.output
    first = _steps(r.output)[0]
    assert f"missing: {stage}" in first, first
    assert "prep task --group 1 --bundle" in first, first
    assert "ready to launch" not in first and "launch task" not in first, \
        first
    answer = web_client.get("/api/task-setup/folder",
                            query_string={"dir": str(bundle)}).get_json()
    row = next(s for s in answer["ladder"]["stages"] if s["stage"] == stage)
    assert row["detail"].startswith("missing: "), row
    assert 1 in answer["ladder"]["offer"], answer["ladder"]["offer"]
    lines = web_client.post("/api/task-setup/commands", json=dict(
        dest=str(bundle), kind="task", groups=[1])).get_json()["lines"]
    assert "prep task --group 1" in lines[0], lines
    assert jobset("prep", "task", "--group", "1", "--bundle", bundle,
                  "--target", "this").exit_code == 0
    r = jobset("status", "--bundle", bundle)
    assert "ready to launch -- next:" in _steps(r.output)[0], r.output


def test_a_benchmark_trial_whose_run_is_gone_is_said_before_any_launch(
        tmp_path, monkeypatch):
    """`job-system.md` § 7 (*`jobset status` on the sweep*) and R-L28: a
    prepared benchmark with one trial not launched and another whose run's
    folder was removed by hand says the missing trial by name, with its way
    back -- the state saved before the benchmark's prep -- and offers no
    launch of the sweep, which R-L28 would refuse.  The silent failure: the
    sweep's next step said *launch the sweep* first (2.56.2's first
    revision)."""
    import shutil

    from molbuilder.calculation import open as open_calculation
    from molbuilder.jobset.materialize import latest_attempt
    from molbuilder.jobset.model import JobSet
    from molbuilder.jobset.naming import job_dir_names, stage_home
    from molbuilder.paths import bench_container
    from support.road import (described, jobset, read_request, road_tree,
                              save_both)
    bundle = described(road_tree(tmp_path, monkeypatch))
    calc = open_calculation(bundle)
    first = calc.task.stages[0].name
    request = read_request(bundle)
    request["bench"] = {"mpi_np": [1, 2], "omp_threads": [1]}
    task = json.loads((bundle / "task.json").read_text())
    assert save_both(bundle, task, request).status_code == 200
    r = jobset("prep", "bench", first, "--bundle", bundle, "--target",
               "this")
    assert r.exit_code == 0, r.output
    container = bundle / bench_container(stage_home(bundle, calc.task,
                                                    first).token)
    sweep = JobSet.of(container)
    gone = sweep.jobs[0]
    shutil.rmtree(latest_attempt(bundle / job_dir_names(sweep)[gone.name]))

    r = jobset("status", "--bundle", container)
    assert r.exit_code == 0, r.output
    said = r.output.split("\n\n")[-1]
    assert f"{gone.name}: missing" in said, r.output
    assert "the state saved before the benchmark's prep" in said, r.output
    assert "launch" not in said, r.output
