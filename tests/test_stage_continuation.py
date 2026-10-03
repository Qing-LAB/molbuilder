"""What an independent stage continues from -- through the road: `jobset init`,
`prep`, `status`, the decision ledger.

PINS: ``docs/execution/job-system.md`` § 5.4 (*What an independent stage
continues from*: by default the newest attempt of the enabled stage before
it, which must have concluded and not failed -- refused before anything is
written, naming the commands; one that did not converge taken with a warning;
`--from` taken as said, with what the run was; `--cold` none; the flat layout
by the same rule, nothing copied) and § 5.3 (`status` names the run the next
prep continues from); ``docs/plans/plan.md`` W37.

PREVENTS, each measured before 2026-10-01 (plan W37, found on
`PDT_FIX_moleculeonly`): a ladder's next stage prepped with nothing handed
over and nothing saying so -- `fine` started from the input geometry and
converged, `final` prepped the same way.

The finished run is the measured H2 relaxation (`tests/fixtures/siesta_relax`,
its README says what it pins), copied in as the coarse stage's.  Nothing here
launches an engine.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from support.road import RELAX as _RELAX
from support.road import a_finished_run, describe_h2
from support.road import jobset as _jobset


@pytest.fixture
def ladder(tmp_path, monkeypatch):
    """`jobset init` of the shipped `publishable` ladder -- coarse, medium
    (tight disabled) -- on a held H2 in a box (`support.road`)."""
    def make(shape="hierarchical"):
        return describe_h2(tmp_path, monkeypatch, shape=shape)
    return make


def _ran(where: Path, *, rc: int = 0, tolerance: str = "0.0100"):
    """The coarse stage's run, ended -- the measured relaxation, put where a
    run of it would have left it (`support.road.a_finished_run`)."""
    a_finished_run(where, rc=rc, tolerance=tolerance)


def _prep(bundle, stage, *more, input=None):
    return _jobset("prep", "run", stage, "--bundle", bundle,
                   "--target", "this", *more, input=input)


def _ladder_on_a_queue(tmp_path, monkeypatch):
    """The ladder on a machine whose queue takes it -- `support.road`'s
    stand-in `sbatch`, which queues nothing -- so a stage can be LAUNCHED
    again: a stage's next attempt is `launch`'s, never a re-prep
    (`job-system.md` § 5.0; it was a re-prep here until 2026-10-02)."""
    from molbuilder.scheduler import Domain
    from support.road import a_queue_that_answers
    a_queue_that_answers(tmp_path, monkeypatch, [
        Domain(name="htc", partition="htc", qos="public",
               max_time="0-04:00:00")])
    bundle = describe_h2(tmp_path, monkeypatch)
    tj = bundle / "task.json"
    d = json.loads(tj.read_text())
    d["allocation"] = {"domain": "htc", "time": "0-01:00:00", "mem": "8G"}
    tj.write_text(json.dumps(d, indent=2))
    return bundle


def _launch(bundle, stage):
    r = _jobset("launch", "run", stage, "--bundle", bundle, "--mode",
                "submit", "--domain", "htc", "--yes")
    assert r.exit_code == 0, r.output


def test_a_stage_continues_from_the_stage_before_it_by_default(ladder):
    """`prep run medium`, bare: refused while coarse has not run and while
    its newest attempt has not concluded -- before anything is written --
    then, once it has, it takes that run: says so, copies its files, records
    it; and `status` names the run beforehand.

    MUTATIONS THIS MUST FAIL AGAINST: no default (medium starts from the
    structure); the conclusion not checked; the verdict not read."""
    from molbuilder.jobset.ledger import LEDGER_FILE
    bundle = ladder()

    r = _prep(bundle, "medium")
    assert r.exit_code != 0, r.output
    assert "which has not run yet" in r.output, r.output
    assert ("molbuilder jobset prep run coarse --bundle P/optimization/H2\n"
            "    molbuilder jobset launch run coarse --mode direct --bundle "
            "P/optimization/H2   # here") in r.output, r.output
    assert "molbuilder jobset prep run medium --cold" in r.output, r.output
    assert not (bundle / "02_medium").exists(), "a refusal wrote the stage"

    r = _prep(bundle, "coarse")
    assert r.exit_code == 0, r.output
    r = _prep(bundle, "medium")
    assert r.exit_code != 0, r.output
    assert "01_coarse/run-0, which has not been launched" in r.output, r.output
    assert "molbuilder jobset launch run coarse" in r.output, r.output
    assert not (bundle / "02_medium").exists(), "a refusal wrote the stage"

    _ran(bundle / "01_coarse" / "run-0")
    r = _jobset("status", "--bundle", bundle)
    assert ("molbuilder jobset prep run medium --bundle P/optimization/H2"
            "   # continues from "
            "01_coarse/run-0") in r.output, r.output

    r = _prep(bundle, "medium")
    assert r.exit_code == 0, r.output
    assert ("continues from 01_coarse/run-0 (the stage before it; concluded "
            "rc=0 at Thu Sep 24 02:38:51 PM MST 2026; converged): copied "
            "H2.XV") in r.output, r.output
    attempt = bundle / "02_medium" / "run-0"
    assert (attempt / "H2.XV").read_bytes() == (_RELAX / "H2.XV").read_bytes()
    said = [json.loads(ln) for ln in
            (bundle / LEDGER_FILE).read_text().splitlines()
            if '"continues"' in ln]
    assert said and said[-1]["from_stage"] == "coarse", said
    assert said[-1]["source"] == "01_coarse/run-0", said
    assert said[-1]["by_default"] is True, said
    assert said[-1]["copied"] == ["H2.XV"], said


def test_a_choice_is_taken_as_said_and_a_failed_run_is_refused(
        tmp_path, monkeypatch):
    """The newest coarse attempt failed: refused, naming the earlier one that
    concluded.  Named by `--from`, a run is taken as said, with what it is;
    and redone by going back -- the state saved before medium's prep,
    restored -- `--cold` takes none.

    MUTATION THIS MUST FAIL AGAINST: a failed run taken as the default."""
    bundle = _ladder_on_a_queue(tmp_path, monkeypatch)
    assert _prep(bundle, "coarse").exit_code == 0
    _launch(bundle, "coarse")
    _ran(bundle / "01_coarse" / "run-0")
    _launch(bundle, "coarse")                 # again: run-1, from run-0
    _ran(bundle / "01_coarse" / "run-1", rc=1)

    r = _prep(bundle, "medium")
    assert r.exit_code != 0, r.output
    assert "01_coarse/run-1, which failed" in r.output, r.output
    assert ("molbuilder jobset prep run medium --from 01_coarse/run-0"
            in r.output), r.output
    assert not (bundle / "02_medium").exists(), "a refusal wrote the stage"

    # the save prep offers, taken: the state a redo goes back to
    r = _prep(bundle, "medium", "--from", "01_coarse/run-1", input="y\n\n")
    assert r.exit_code == 0, r.output
    assert "continues from 01_coarse/run-1 (named; concluded rc=1" in r.output
    assert "FAILED" in r.output, r.output

    # REDONE BY GOING BACK (`job-system.md` § 5.0): restored to the state
    # saved before medium's prep, medium is not prepped -- and preps anew.
    from click.testing import CliRunner
    from molbuilder.checkpoint import Repo
    from molbuilder.cli import cli
    before = Repo(str(bundle)).states()[0]
    assert before.note == "before prep run medium", before
    back = CliRunner().invoke(cli, ["checkpoint", "restore", before.id,
                                    "-p", str(bundle), "--force"])
    assert back.exit_code == 0, back.output
    r = _prep(bundle, "medium", "--cold")
    assert r.exit_code == 0, r.output
    assert "cold start -- nothing copied in" in r.output, r.output
    assert "continues from" not in r.output, r.output


def test_the_flat_layout_continues_by_the_same_rule(ladder):
    """Every stage shares one folder, so nothing is copied -- and the next
    stage still waits for the one before it to conclude, and says which run
    it continues from.

    MUTATION THIS MUST FAIL AGAINST: the flat layout exempt from the rule."""
    bundle = ladder("flat")
    assert _prep(bundle, "coarse").exit_code == 0
    r = _prep(bundle, "medium")
    assert r.exit_code != 0, r.output
    assert "`coarse`'s latest run, which has not been launched" in r.output, (
        r.output)
    # the ways on that work here: no attempt to name, no `--cold`
    assert "molbuilder jobset launch run coarse" in r.output, r.output
    assert "set its run card's `restart` to `clean`" in r.output, r.output
    assert "--cold" not in r.output and "--from" not in r.output, r.output
    _ran(bundle)
    r = _prep(bundle, "medium")
    assert r.exit_code == 0, r.output
    assert ("continues from coarse's latest run, whose files lie in this "
            "folder (the stage before it; concluded rc=0") in r.output, (
        r.output)


def test_a_flat_stage_records_the_run_it_continued_from(tmp_path,
                                                        monkeypatch):
    """The flat layout records what a stage continued from too (user,
    2026-10-01): coarse sent and finished in the one folder, medium prepped
    -- continuing from it -- and sent; medium's own launch record names
    coarse's run by what every file of it carries, `H2_01_coarse-run0`,
    which is what the Run panel reads.  Finished and launched again, medium
    continues from its own latest run, and its record names that one
    (`job-system.md` § 5.0, row 7).

    MUTATIONS THIS MUST FAIL AGAINST: prep leaving no record of the run on
    the flat layout (the launch record without `continued_from`); a flat
    stage launched again recording the stage before it (W55 D5)."""
    from molbuilder.jobset.materialize import read_run_launch
    from molbuilder.scheduler import Domain
    from support.road import a_queue_that_answers
    a_queue_that_answers(tmp_path, monkeypatch, [
        Domain(name="htc", partition="htc", qos="public",
               max_time="0-04:00:00")])
    bundle = describe_h2(tmp_path, monkeypatch, shape="flat")
    # A job sent to that queue states its queue, wall and memory
    # (`architecture.md` § 5.2) -- in the description, as a person does.
    import json
    tj = bundle / "task.json"
    d = json.loads(tj.read_text())
    d["allocation"] = {"domain": "htc", "time": "0-01:00:00", "mem": "8G"}
    tj.write_text(json.dumps(d, indent=2))
    assert _prep(bundle, "coarse").exit_code == 0
    assert _jobset("launch", "run", "coarse", "--bundle", bundle, "--mode",
                   "submit", "--domain", "htc", "--yes").exit_code == 0
    a_finished_run(bundle)
    r = _prep(bundle, "medium")
    assert r.exit_code == 0, r.output
    r = _jobset("launch", "run", "medium", "--bundle", bundle, "--mode",
                "submit", "--domain", "htc", "--yes")
    assert r.exit_code == 0, r.output
    record = read_run_launch(bundle, basename="H2_02_medium")
    assert record["continued_from"] == "H2_01_coarse-run0", record

    a_finished_run(bundle, stem="H2_02_medium")
    r = _jobset("launch", "run", "medium", "--bundle", bundle, "--mode",
                "submit", "--domain", "htc", "--yes")
    assert r.exit_code == 0, r.output
    record = read_run_launch(bundle, basename="H2_02_medium")
    assert record["continued_from"] == "H2_02_medium-run0", record


def test_a_linked_stage_says_its_input_is_preps_own(tmp_path, monkeypatch):
    """A SIESTA vibration's `freq` is written at the geometry `relax`
    relaxed to -- prep takes it, no run is continued -- so its prep says so,
    never that it is like a first stage (`job-system.md` § 5.4, the linked
    column).

    MUTATION THIS MUST FAIL AGAINST: the answer not saying the stage is
    linked (both doors fall back to "nothing carried in")."""
    bundle = describe_h2(tmp_path, monkeypatch, calculation="vibration")
    assert _prep(bundle, "relax").exit_code == 0
    a_finished_run(bundle / "01_relax" / "run-0", stem="H2_01_relax")
    r = _prep(bundle, "freq")
    assert r.exit_code == 0, r.output
    assert "its input is prep's own" in r.output, r.output
    assert "nothing carried in" not in r.output, r.output


def test_a_newer_run_is_never_passed_over_and_a_verdict_is_said(
        tmp_path, monkeypatch):
    """Coarse ran once and did not converge, then was launched again: the
    newer run has not concluded, so medium's prep refuses -- the older run
    never stands in by default -- and names it as a choice; named, it is
    taken, and the line says it did not converge.

    MUTATIONS THIS MUST FAIL AGAINST: an older concluded attempt taken by
    default; the verdict not said."""
    bundle = _ladder_on_a_queue(tmp_path, monkeypatch)
    assert _prep(bundle, "coarse").exit_code == 0
    _launch(bundle, "coarse")
    _ran(bundle / "01_coarse" / "run-0", tolerance="0.0001")
    _launch(bundle, "coarse")                 # again: run-1, queued

    r = _prep(bundle, "medium")
    assert r.exit_code != 0, r.output
    assert "01_coarse/run-1, which is queued" in r.output, r.output
    assert ("molbuilder jobset prep run medium --from 01_coarse/run-0"
            in r.output), r.output

    r = _prep(bundle, "medium", "--from", "01_coarse/run-0")
    assert r.exit_code == 0, r.output
    assert ("continues from 01_coarse/run-0 (named; concluded rc=0 at Thu "
            "Sep 24 02:38:51 PM MST 2026; NOT converged -- taken as it "
            "stands)") in r.output, r.output


def test_the_browser_prep_continues_as_the_terminal_does(ladder):
    """Task setup's Prep is the same prep: refused, by name, while coarse
    has not been launched; once it has concluded, the answer carries what
    it continues from and the line the terminal prints.

    MUTATION THIS MUST FAIL AGAINST: the answer without its continuation."""
    from molbuilder.web.app import create_app
    bundle = ladder()
    client = create_app(config={}).test_client()

    def prep(stage):
        # the save prep offers, answered: no
        return client.post("/api/task-setup/prep", json={
            "dest": str(bundle), "kind": "run", "stage": stage,
            "target": "this", "save": False})

    assert prep("coarse").status_code == 200
    r = prep("medium")
    assert r.status_code == 400, r.get_json()
    assert "which has not been launched" in r.get_json()["error"]
    _ran(bundle / "01_coarse" / "run-0")
    r = prep("medium")
    assert r.status_code == 200, r.get_json()
    got = r.get_json()["continuation"]
    assert (got["stage"], got["source"], got["by_default"]) == (
        "coarse", "01_coarse/run-0", True), got
    assert got["line"].startswith(
        "continues from 01_coarse/run-0 (the stage before it; concluded "
        "rc=0"), got


def test_a_stage_set_to_start_clean_is_never_told_it_continues(ladder):
    """Medium's run card says `restart: clean`: `status` names no run for
    it, and its prep carries nothing -- one answer, from one door.

    MUTATION THIS MUST FAIL AGAINST: status answering by a rule of its
    own."""
    bundle = ladder()
    task = json.loads((bundle / "task.json").read_text())
    task["stages"][1]["execution"] = {"restart": "clean"}
    from molbuilder.web.app import create_app
    r = create_app(config={}).test_client().post(
        "/api/task-setup/save",
        json={"dest": str(bundle), "text": json.dumps(task)})
    assert r.status_code == 200, r.get_json()
    assert _prep(bundle, "coarse").exit_code == 0
    _ran(bundle / "01_coarse" / "run-0")

    r = _jobset("status", "--bundle", bundle)
    assert "molbuilder jobset prep run medium" in r.output, r.output
    assert "continues from" not in r.output, r.output
    r = _prep(bundle, "medium")
    assert r.exit_code == 0, r.output
    assert "continues from" not in r.output, r.output
    assert "nothing carried in" in r.output, r.output


def test_the_browser_doors_offer_the_choice_and_take_it(ladder):
    """Task setup's doors: the folder answer offers medium's Continue from
    -- coarse's runs with what each was, `--cold`, and the default or why it
    is refused; the preview says what prep would take; the prep takes a run
    named by `from`, or none by `cold`, and refuses a path out of the
    calculation.

    MUTATIONS THIS MUST FAIL AGAINST: the route dropping `from` / `cold`;
    the folder answer without the choices."""
    from molbuilder.web.app import create_app
    bundle = ladder()
    client = create_app(config={}).test_client()

    def post(**body):
        # the save prep offers, answered: no
        return client.post("/api/task-setup/prep", json=dict(
            {"dest": str(bundle), "kind": "run", "target": "this",
             "save": False}, **body))

    assert post(stage="coarse").status_code == 200
    _ran(bundle / "01_coarse" / "run-0")
    cf = client.get("/api/task-setup/folder?dir=" + str(bundle)).get_json()[
        "continue_from"]
    assert set(cf) == {"medium"}, cf
    m = cf["medium"]
    assert (m["from_stage"], m["cold"], m["refused"]) == ("coarse", True,
                                                          None), m
    assert m["default"]["source"] == "01_coarse/run-0", m
    assert [r["source"] for r in m["runs"]] == ["01_coarse/run-0"], m
    assert m["runs"][0]["what"].startswith("concluded rc=0"), m

    r = post(stage="medium", plan=True).get_json()
    assert r["continuation"]["line"].startswith(
        "continues from 01_coarse/run-0 (the stage before it;"), r
    # COLD, as the preview takes it -- a prepped stage is not prepped again
    # (`job-system.md` § 5.0), so one medium prep is the run named below.
    r = post(stage="medium", plan=True, cold=True).get_json()
    assert r.get("continuation") is None, r
    # A PATH OUT OF THE CALCULATION, refused by its own rule -- `..` leads
    # to a folder that exists, so nothing else can be what refused it.
    (bundle.parent / "elsewhere").mkdir()
    r = post(stage="medium", **{"from": "../elsewhere"})
    assert r.status_code == 400, r.get_json()
    assert "--from names a run of this calculation" in r.get_json()["error"]
    r = post(stage="medium", **{"from": "01_coarse/run-0"})
    assert r.status_code == 200, r.get_json()
    assert r.get_json()["continuation"]["by_default"] is False, r.get_json()


def test_the_run_panel_says_what_a_run_continued_from(ladder):
    """A stage that continued from another, launched: its run record -- the
    Results tab's Run panel -- names the run, from its `run.json`.

    MUTATION THIS MUST FAIL AGAINST: the record without `continued_from`."""
    from molbuilder.jobset.materialize import write_run_launch
    from molbuilder.web.app import create_app
    bundle = ladder()
    assert _prep(bundle, "coarse").exit_code == 0
    _ran(bundle / "01_coarse" / "run-0")
    assert _prep(bundle, "medium").exit_code == 0
    attempt = bundle / "02_medium" / "run-0"
    # the launch's own record (submit writes it from `.continued-from`)
    write_run_launch(attempt, mode="direct", command=["bash", "x"],
                     continued_from=(attempt / ".continued-from")
                     .read_text().strip())
    shutil.copy2(_RELAX / "H2_01_relax-run0.out",
                 attempt / "H2_02_medium-run0.out")
    body = create_app(config={}).test_client().get(
        "/api/results/dir?path=" + str(attempt)).get_json()
    launch = body["record"]["computation"]["launch"]
    assert launch["continued_from"] == "01_coarse/run-0", launch
