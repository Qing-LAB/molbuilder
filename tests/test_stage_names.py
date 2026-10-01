"""A stage is named the way it is typed -- through the road: `jobset init`,
the Task-setup save, `prep`, the deck's own header, `launch --dry-run`.

PINS: ``docs/execution/job-system.md`` § 5.3 (a stage is its name, in any
case, or ``#N``, through one resolver; what molbuilder prints, you can type --
`identity.command_stage`) and ``docs/engines/stages.md`` § 2 (names compare
case-insensitively everywhere, through one key, `identity.stage_key`);
``docs/engines/template.md`` § 6.4 (a SIESTA vibration's stage named `relax`
is its relaxation); ``docs/plans/plan.md`` § 5w K12.

PREVENTS, each read in the code before 2026-10-01 (the M11 review):

* every staged deck's header printing ``jobset launch run 02_freq``, which
  `launch` refused -- the token is a legal name of another stage (SS-C11);
* a name typed in another case refused, and a SIESTA vibration's relax row
  renamed ``Relax`` rendered as a force-constant run while its ``freq`` was
  refused for a ladder with no relax stage (SS-C15).

Nothing here launches an engine.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pytest


def _jobset(*args):
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    return CliRunner().invoke(jobset_group, [str(a) for a in args])


def _described(tmp_path, monkeypatch, engine, calculation):
    """`jobset init` on a held H2 in a box, the way a person starts one."""
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
                "--bundle", f"P/{calculation}/H2", "--engine", engine,
                "--shape", "hierarchical", "--calculation", calculation,
                "--name", "H2",
                *(("--psml-lib", "pseudopotential") if engine == "siesta"
                  else ()))
    assert r.exit_code == 0, r.output
    bundle = tree / "P" / calculation / "H2"
    (bundle / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": "true"}}))
    return bundle


@pytest.mark.parametrize("engine,calculation", [
    ("siesta", "optimization"), ("pyscf", "optimization"),
    ("pyscf", "vibration")])
def test_the_launch_line_a_deck_prints_is_one_launch_takes(
        tmp_path, monkeypatch, engine, calculation):
    """Each deck's header tells a person how to run it.  The stage it names
    is the one the description holds -- by its name (`command_stage`) -- so
    the line, typed back, launches this deck's stage; typed in another case
    it is still that stage, and the decision ledger records it by the
    description's spelling.

    MUTATIONS THIS MUST FAIL AGAINST: a header printing the token again; the
    resolver comparing exact strings; `launch` recording what was typed."""
    from molbuilder.jobset.ledger import LEDGER_FILE
    bundle = _described(tmp_path, monkeypatch, engine, calculation)
    stage = json.loads((bundle / "task.json").read_text())["stages"][0]["name"]
    r = _jobset("prep", "run", stage, "--bundle", bundle, "--target", "this")
    assert r.exit_code == 0, r.output
    token = next(p.name for p in bundle.iterdir()
                 if p.is_dir() and p.name.endswith(f"_{stage}"))
    deck = next(p for p in (bundle / token).iterdir()
                if p.suffix in (".fdf", ".py"))
    typed = re.search(r"jobset launch run (\S+)", deck.read_text()).group(1)
    for spelling in (typed, typed.upper()):
        r = _jobset("launch", "run", spelling, "--bundle", bundle,
                    "--mode", "direct", "--dry-run")
        assert r.exit_code == 0, (spelling, r.output)
        assert re.search(rf"WOULD run\s+{stage}\s+bash H2_{token}\.run\.sh",
                         r.output), (spelling, r.output)
        # A dry run is ledgered as planned (W52) -- by the stage's name.
        planned = [json.loads(ln) for ln in
                   (bundle / LEDGER_FILE).read_text().splitlines()
                   if '"planned"' in ln][-1]
        assert planned["stage"] == stage, (spelling, planned)


def test_a_relax_stage_renamed_in_another_case_is_still_the_relaxation(
        tmp_path, monkeypatch):
    """A person renames a SIESTA vibration's ``relax`` row ``Relax`` in Task
    setup's stage table and saves.  In any case it is one name, so the rung
    renders the relaxation deck, and the force-constant stage takes its
    geometry from it -- refused here only because it has not run.

    MUTATIONS THIS MUST FAIL AGAINST: the role rule matching the exact name
    (`Relax` renders force constants); prep finding the relaxation rung by
    the exact name (`freq` is told the ladder has no relax stage)."""
    from molbuilder.web.app import create_app
    bundle = _described(tmp_path, monkeypatch, "siesta", "vibration")
    task = json.loads((bundle / "task.json").read_text())
    assert [s["name"] for s in task["stages"]] == ["relax", "freq"], task
    task["stages"][0]["name"] = "Relax"
    r = create_app(config={}).test_client().post(
        "/api/task-setup/save",
        json={"dest": str(bundle), "text": json.dumps(task)})
    assert r.status_code == 200, r.get_json()

    r = _jobset("prep", "run", "Relax", "--bundle", bundle,
                "--target", "this")
    assert r.exit_code == 0, r.output
    deck = (bundle / "01_Relax" / "H2_01_Relax.fdf").read_text()
    run_type = re.search(r"^\s*MD\.TypeOfRun\s+(\S+)", deck, re.M | re.I)
    assert run_type and run_type.group(1).upper() != "FC", (
        "the relaxation rung rendered the force-constant deck: "
        + (run_type.group(0) if run_type else "no MD.TypeOfRun"))

    r = _jobset("prep", "run", "freq", "--bundle", bundle,
                "--target", "this")
    assert r.exit_code != 0, r.output
    assert "takes its geometry from the `Relax` stage" in r.output, r.output
