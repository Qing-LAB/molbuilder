"""`jobset status` lists the description's ladder -- through the road:
`jobset init`, the Task-setup save, `prep`, `status`, the Results tab.

PINS: ``docs/execution/job-system.md`` § 5.3 (the table is the description's
ladder: every stage with its number from the moment `init` writes it, the ones
not prepped yet as not-started, a disabled one never the stage to resume from;
`status <stage>` is a stage in full -- its deck, what it carries, its
resources -- since `plan` folded into it) and ``docs/web/results.md`` § 2.4
(the Results tab's ladder is `jobset_status`'s answer).

PREVENTS, each read in the code before 2026-10-01:

* `status` refusing a described calculation until its first prep, and listing
  only the stages prepped so far -- while the Results tab listed all of them,
  having composed the ladder itself;
* a stage disabled in the description named as the stage to resume from;
* the deck, carry set and resources behind a second verb, `plan`, that
  listed the same ladder from the same file.

Nothing here launches an engine.
"""
from __future__ import annotations

import json

import numpy as np


def _jobset(*args):
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    return CliRunner().invoke(jobset_group, [str(a) for a in args])


def _row(output, name):
    """The status table's row for ``name``: ``[seq, stage, attempt, state,
    ...]``."""
    return next(ln.split() for ln in output.splitlines()
                if ln.split()[1:2] == [name])


def test_status_lists_every_stage_from_the_description(tmp_path, monkeypatch):
    """A SIESTA relaxation, described with the shipped `publishable` ladder
    -- coarse and medium enabled, tight disabled.  `status` lists all three
    before the first prep and after it, numbered as `#N` names them; tight is
    listed as disabled, told how to run, and never the stage to resume from;
    once coarse has concluded, the command for medium is shown beside the run
    it continues from (`job-system.md` § 5.4); `status <stage>` shows what
    the stage is; the
    Results tab answers the same ladder.

    MUTATIONS THIS MUST FAIL AGAINST: rows from the job set alone; a disabled
    stage counted as the stage to resume from; `status <stage>` without the
    plan's columns; the next command without the run it continues from."""
    from conftest import write_pseudos
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.structure import Structure
    from molbuilder.web.app import create_app
    from molbuilder.workingcopy_structure import StructureCodec

    tree = tmp_path / "projects"
    (tree / "P" / "structure").mkdir(parents=True)
    (tree / "pseudopotential").mkdir()
    write_pseudos(tree / "pseudopotential", ["H"])
    StructureCodec().write(
        Structure(elements=["H", "H"],
                  positions=np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.74]]),
                  cell=np.diag([10.0, 10.0, 10.0]),
                  axis_kind=("isolated",) * 3),
        tree / "P" / "structure" / "h2.xyz")
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tree))
    monkeypatch.chdir(tree.parent)
    r = _jobset("init", "--structure", "P/structure/h2.xyz",
                "--bundle", "P/optimization/H2", "--engine", "siesta",
                "--shape", "hierarchical", "--name", "H2",
                "--stage-strategy", "publishable",
                "--psml-lib", "pseudopotential")
    assert r.exit_code == 0, r.output
    bundle = tree / "P" / "optimization" / "H2"
    (bundle / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": "true"}}))
    stages = json.loads((bundle / "task.json").read_text())["stages"]
    assert [s["enabled"] for s in stages] == [True, True, False], stages
    first, second, off = (s["name"] for s in stages)

    def concluded(n, name):
        """The stage's run has ended: its own conclusion marker."""
        (bundle / f"{n:02d}_{name}" / "run-0"
         / f"H2_{n:02d}_{name}-run0.concluded").write_text("rc=0\n")

    # BEFORE ANY PREP: every stage, numbered, none prepped
    r = _jobset("status", "--bundle", bundle)
    assert r.exit_code == 0, r.output
    for n, s in enumerate(stages, start=1):
        assert _row(r.output, s["name"])[:4] == [
            str(n), s["name"], "-", "not-started"], r.output
    assert "disabled" in " ".join(_row(r.output, off)), r.output
    assert f"molbuilder jobset prep run {first}\n" in r.output, r.output

    # THE FIRST STAGE PREPPED AND CONCLUDED: the next names its source
    r = _jobset("prep", "run", first, "--bundle", bundle, "--target", "this")
    assert r.exit_code == 0, r.output
    concluded(1, first)
    r = _jobset("status", "--bundle", bundle)
    assert r.exit_code == 0, r.output
    assert _row(r.output, first)[3] == "finished", r.output
    assert _row(r.output, second)[3] == "not-started", r.output
    assert (f"molbuilder jobset prep run {second}   # continues from "
            f"01_{first}/run-0" in r.output), r.output

    # ONE STAGE IN FULL: what it is, then what happened to it
    r = _jobset("status", first, "--bundle", bundle)
    assert r.exit_code == 0, r.output
    lines = {ln.split()[0]: ln.split(None, 1)[1] for ln in r.output.splitlines()
             if ln.startswith("  ") and len(ln.split()) > 1}
    assert lines["deck"] == f"H2_01_{first}.fdf", r.output
    assert "H2.XV" in lines["carries"], r.output
    assert lines["resources"] != "-", r.output
    r = _jobset("status", "#2", "--bundle", bundle)
    assert r.exit_code == 0, r.output
    assert (f"molbuilder jobset prep run {second}   # continues from "
            f"01_{first}/run-0" in r.output), r.output
    r = _jobset("status", off, "--bundle", bundle)
    assert r.exit_code == 0, r.output
    assert "Enable it in Task setup" in r.output, r.output
    assert "prep run" not in r.output, r.output

    # THE RESULTS TAB'S LADDER IS THE SAME ANSWER
    body = create_app(config={}).test_client().get(
        "/api/results/dir?path=" + str(bundle)).get_json()
    ladder = body["ladder"]
    assert [(s["name"], s["seq"]) for s in ladder["stages"]] == [
        (s["name"], n) for n, s in enumerate(stages, start=1)], ladder
    assert ladder["first_incomplete"] == second, ladder
    # ...the next prep's answer with it -- the one wire form (W52).
    assert ladder["resume_from"] == f"01_{first}/run-0", ladder

    # EVERY ENABLED STAGE CONCLUDED: the disabled one is not what is left
    # (the conclusion markers stand in for runs, which left no restart files
    # to hand over -- so this stage starts from the structure, `--cold`)
    r = _jobset("prep", "run", second, "--bundle", bundle, "--target", "this",
                "--cold")
    assert r.exit_code == 0, r.output
    concluded(2, second)
    r = _jobset("status", "--bundle", bundle)
    assert r.exit_code == 0, r.output
    assert "Every enabled stage finished. Nothing to resume." in r.output, (
        r.output)


def test_the_next_step_is_worded_by_the_stages_state(tmp_path, monkeypatch):
    """`status` says what the first incomplete stage's state calls for, as a
    command that works: prepped and not launched -- launch it; sent and not
    started -- let it finish; failed -- launch it again, which continues from
    its own latest run.  A later stage asked about by name is told why its
    prep would refuse, whole.

    MUTATIONS THIS MUST FAIL AGAINST: every state told to "re-submit that
    stage"; `status <stage>` answering a later stage without asking the
    continuation door (it was told "Prep it")."""
    from molbuilder.scheduler import Domain
    from support.road import (a_finished_run, a_queue_that_answers,
                              describe_h2, jobset)
    a_queue_that_answers(tmp_path, monkeypatch, [
        Domain(name="htc", partition="htc", qos="public",
               max_time="0-04:00:00")])
    bundle = describe_h2(tmp_path, monkeypatch)
    r = jobset("status", "medium", "--bundle", bundle)
    assert r.exit_code == 0, r.output
    assert "Its prep refuses for now" in r.output, r.output
    assert "molbuilder jobset prep run coarse" in r.output, r.output

    assert jobset("prep", "run", "coarse", "--bundle", bundle,
                  "--target", "this").exit_code == 0
    r = jobset("status", "--bundle", bundle)
    assert ("prepped and not launched:\n    molbuilder jobset launch run "
            "coarse") in r.output, r.output

    assert jobset("launch", "run", "coarse", "--bundle", bundle, "--mode",
                  "submit", "--domain", "htc", "--yes").exit_code == 0
    r = jobset("status", "--bundle", bundle)
    assert "coarse, queued -- let it finish" in r.output, r.output

    a_finished_run(bundle / "01_coarse" / "run-0", rc=1)
    r = jobset("status", "--bundle", bundle)
    assert "coarse, failed" in r.output, r.output
    assert ("continues from its own latest run:\n    molbuilder jobset "
            "launch run coarse") in r.output, r.output


def test_status_answers_what_it_cannot_read_and_where_it_was_asked(
        tmp_path, monkeypatch):
    """A template prep would refuse is said in the table's last line, never
    a traceback; a stage renamed in case only keeps its prepped job; asked
    from inside one of its stage folders, `status` names the calculation it
    belongs to.

    MUTATIONS THIS MUST FAIL AGAINST: the continuation door raising (a
    traceback); stages joined to jobs by exact name; a stage folder told to
    run `init`."""
    from molbuilder.web.app import create_app
    from support.road import describe_h2, jobset
    bundle = describe_h2(tmp_path, monkeypatch)

    template = next(bundle.glob("*.template.toml"))
    kept = template.read_text()
    template.write_text("this is not a template [\n")
    r = jobset("status", "--bundle", bundle)
    assert r.exit_code == 0 and r.exception is None, r.output
    assert "refuses for now" in r.output, r.output
    assert "continues from cannot be read" in r.output, r.output
    template.write_text(kept)

    assert jobset("prep", "run", "coarse", "--bundle", bundle,
                  "--target", "this").exit_code == 0

    task = json.loads((bundle / "task.json").read_text())
    task["stages"][0]["name"] = "COARSE"
    r = create_app(config={}).test_client().post(
        "/api/task-setup/save",
        json={"dest": str(bundle), "text": json.dumps(task)})
    assert r.status_code == 200, r.get_json()
    r = jobset("status", "--bundle", bundle)
    assert r.exit_code == 0, r.output
    assert _row(r.output, "COARSE")[3] == "pending", r.output

    r = jobset("status", "--bundle", bundle / "01_coarse")
    assert r.exit_code != 0, r.output
    assert "is a folder of the calculation at" in r.output, r.output
    assert "jobset init" not in r.output, r.output
