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

import numpy as np
import pytest

_RELAX = Path(__file__).parent / "fixtures" / "siesta_relax" / "01_relax" / "run-0"


def _jobset(*args):
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    return CliRunner().invoke(jobset_group, [str(a) for a in args])


@pytest.fixture
def ladder(tmp_path, monkeypatch):
    """`jobset init` of the shipped `publishable` ladder -- coarse, medium
    (tight disabled) -- on a held H2 in a box; returns ``(bundle, shape)``."""
    def make(shape="hierarchical"):
        from conftest import write_pseudos
        from molbuilder.projects import PROJECTS_ROOT_ENV
        from molbuilder.structure import Structure
        from molbuilder.workingcopy_structure import StructureCodec
        tree = tmp_path / "projects"
        (tree / "P" / "structure").mkdir(parents=True)
        (tree / "pseudopotential").mkdir()
        write_pseudos(tree / "pseudopotential", ["H"])
        StructureCodec().write(
            Structure(elements=["H", "H"],
                      positions=np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.741]]),
                      regions={"frozen_atoms": [0]},
                      cell=np.diag([10.0, 10.0, 10.0]),
                      axis_kind=("isolated",) * 3),
            tree / "P" / "structure" / "h2.xyz")
        monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tree))
        monkeypatch.chdir(tree.parent)
        r = _jobset("init", "--structure", "P/structure/h2.xyz",
                    "--bundle", "P/optimization/H2", "--engine", "siesta",
                    "--shape", shape, "--name", "H2",
                    "--stage-strategy", "publishable",
                    "--psml-lib", "pseudopotential")
        assert r.exit_code == 0, r.output
        bundle = tree / "P" / "optimization" / "H2"
        (bundle / ".molbuilder.json").write_text(json.dumps(
            {"script_generation": {"activation": "conda activate",
                                   "preamble": "true"}}))
        return bundle
    return make


def _ran(where: Path, *, rc: int = 0, tolerance: str = "0.0100"):
    """The coarse stage's run, ended: the measured relaxation's output and
    geometry, and its conclusion marker.  A failed one (``rc`` nonzero) died
    partway, so its output stops before the engine's end -- the output's own
    ending is the strongest evidence of how a run ended (`running-a-job.md`
    § 4.2)."""
    text = (_RELAX / "H2_01_relax-run0.out").read_text().replace(
        "Force tolerance                             =     0.0100 eV/Ang",
        f"Force tolerance                             =     {tolerance} eV/Ang")
    (where / "H2_01_coarse-run0.out").write_text(
        text if rc == 0 else text[: len(text) // 3])
    shutil.copy2(_RELAX / "H2.XV", where / "H2.XV")
    (where / "H2_01_coarse-run0.concluded").write_text(
        f"rc={rc} at Thu Sep 24 02:38:51 PM MST 2026\n")


def _prep(bundle, stage, *more):
    return _jobset("prep", "run", stage, "--bundle", bundle,
                   "--target", "this", *more)


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
    assert "molbuilder jobset prep run coarse && molbuilder jobset launch " \
           "run coarse" in r.output, r.output
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
    assert ("molbuilder jobset prep run medium   # continues from "
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


def test_a_choice_is_taken_as_said_and_a_failed_run_is_refused(ladder):
    """The newest coarse attempt failed: refused, naming the earlier one that
    concluded.  Named by `--from`, a run is taken as said, with what it is;
    `--cold` takes none.

    MUTATION THIS MUST FAIL AGAINST: a failed run taken as the default."""
    from molbuilder.jobset.materialize import write_run_launch
    bundle = ladder()
    assert _prep(bundle, "coarse").exit_code == 0
    _ran(bundle / "01_coarse" / "run-0")
    write_run_launch(bundle / "01_coarse" / "run-0", mode="direct",
                     command=["bash", "x"])
    r = _prep(bundle, "coarse", "--from", "01_coarse/run-0")
    assert r.exit_code == 0, r.output
    _ran(bundle / "01_coarse" / "run-1", rc=1)

    r = _prep(bundle, "medium")
    assert r.exit_code != 0, r.output
    assert "01_coarse/run-1, which failed" in r.output, r.output
    assert ("molbuilder jobset prep run medium --from 01_coarse/run-0"
            in r.output), r.output
    assert not (bundle / "02_medium").exists(), "a refusal wrote the stage"

    r = _prep(bundle, "medium", "--from", "01_coarse/run-1")
    assert r.exit_code == 0, r.output
    assert "continues from 01_coarse/run-1 (named; concluded rc=1" in r.output
    assert "FAILED" in r.output, r.output

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


def test_a_newer_run_is_never_passed_over_and_a_verdict_is_said(ladder):
    """Coarse ran once and did not converge, then was launched again: the
    newer run has not concluded, so medium's prep refuses -- the older run
    never stands in by default -- and names it as a choice; named, it is
    taken, and the line says it did not converge.

    MUTATIONS THIS MUST FAIL AGAINST: an older concluded attempt taken by
    default; the verdict not said."""
    from molbuilder.jobset.materialize import write_run_launch
    bundle = ladder()
    assert _prep(bundle, "coarse").exit_code == 0
    _ran(bundle / "01_coarse" / "run-0", tolerance="0.0001")
    write_run_launch(bundle / "01_coarse" / "run-0", mode="direct",
                     command=["bash", "x"])
    assert _prep(bundle, "coarse", "--from", "01_coarse/run-0").exit_code == 0
    write_run_launch(bundle / "01_coarse" / "run-1", mode="direct",
                     command=["bash", "x"])

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
        return client.post("/api/task-setup/prep", json={
            "dest": str(bundle), "kind": "run", "stage": stage,
            "target": "this"})

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
        return client.post("/api/task-setup/prep", json=dict(
            {"dest": str(bundle), "kind": "run", "target": "this"}, **body))

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
    r = post(stage="medium", cold=True)
    assert r.status_code == 200, r.get_json()
    assert r.get_json()["continuation"] is None, r.get_json()
    assert r.get_json()["attempt"]["cold"] is True, r.get_json()
    r = post(stage="medium", **{"from": "01_coarse/run-0"})
    assert r.status_code == 200, r.get_json()
    assert r.get_json()["continuation"]["by_default"] is False, r.get_json()
    # A PATH OUT OF THE CALCULATION, refused by its own rule -- `..` leads
    # to a folder that exists, so nothing else can be what refused it.
    (bundle.parent / "elsewhere").mkdir()
    r = post(stage="medium", **{"from": "../elsewhere"})
    assert r.status_code == 400, r.get_json()
    assert "`from` names a run of this calculation" in r.get_json()["error"]


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
