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
    once coarse has concluded, the command for medium names the run it
    continues from -- an independent stage's bare prep takes nothing
    (`job-system.md` § 5.4); `status <stage>` shows what the stage is; the
    Results tab answers the same ladder.

    MUTATIONS THIS MUST FAIL AGAINST: rows from the job set alone; a disabled
    stage counted as the stage to resume from; `status <stage>` without the
    plan's columns; the next command without `--from`."""
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
    assert (f"molbuilder jobset prep run {second} --from 01_{first}/run-0"
            in r.output), r.output

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
    assert (f"molbuilder jobset prep run {second} --from 01_{first}/run-0"
            in r.output), r.output
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

    # EVERY ENABLED STAGE CONCLUDED: the disabled one is not what is left
    # (the conclusion markers stand in for runs, which left no restart files
    # for a `--from` to copy)
    r = _jobset("prep", "run", second, "--bundle", bundle, "--target", "this")
    assert r.exit_code == 0, r.output
    concluded(2, second)
    r = _jobset("status", "--bundle", bundle)
    assert r.exit_code == 0, r.output
    assert "Every enabled stage finished. Nothing to resume." in r.output, (
        r.output)
