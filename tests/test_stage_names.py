"""A stage is named the way it is typed -- through the road: `molbuilder describe`,
the Task-setup save, `prep`, the deck's own header, `launch --dry-run`.

PINS: ``docs/execution/job-system.md`` § 5.3 (a stage is its name, in any
case, or ``#N``, through one resolver; what molbuilder prints, you can type --
`identity.command_stage`) and ``docs/engines/stages.md`` § 2 (names compare
case-insensitively everywhere, through one key, `identity.stage_key`);
``docs/engines/template.md`` § 6.4 (a SIESTA vibration's stage named `relax`
is its relaxation); ``docs/plans/plan.md`` § 5w K12.

PREVENTS, each read in the code before 2026-10-01 (the M11 review):

* every staged deck's header printing ``jobset launch task --stage 02_freq``, which
  `launch` refused -- the token is a legal name of another stage (SS-C11);
* a name typed in another case refused, and a SIESTA vibration's relax row
  renamed ``Relax`` rendered as a force-constant run while its ``freq`` was
  refused for a ladder with no relax stage (SS-C15).

Nothing here launches an engine.
"""
from __future__ import annotations

from support.road import group_words

import json
import re

import numpy as np
import pytest
from support.road import molbuilder, read_request, write_request


def _jobset(*args):
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    return CliRunner().invoke(jobset_group, [str(a) for a in args])


def _described(tmp_path, monkeypatch, engine, calculation):
    """`molbuilder describe` on a held H2 in a box, the way a person starts one."""
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
    r = molbuilder("describe", "--structure", "P/structure/h2.xyz",
                "--bundle", f"P/{calculation}/H2", "--engine", engine,
                "--calculation", calculation,
                "--name", "H2",
                *(("--psml-lib", "pseudopotential") if engine == "siesta"
                  else ()))
    assert r.exit_code == 0, r.output
    bundle = tree / "P" / calculation / "H2"
    # THE RUN CARD STATES THE LAUNCH SHAPE, as a person's does: a run whose
    # shape is stated nowhere is refused at prep (`architecture.md` § 5.2).
    request = read_request(bundle)
    request["execution"] = ({"mpi_np": 2, "omp_threads": 1}
                            if engine == "siesta" else {"threads": 1})
    write_request(bundle, request)
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

    A dry run writes nothing (`job-system.md` § 6.0, step 3), so the line
    the ledger holds by the description's name is a launch's own -- here a
    refused one: on this workstation prep wrote no scheduler header.

    MUTATIONS THIS MUST FAIL AGAINST: a header printing the token again; the
    resolver comparing exact strings; `launch` recording what was typed."""
    from molbuilder.jobset.ledger import LEDGER_FILE
    bundle = _described(tmp_path, monkeypatch, engine, calculation)
    stage = json.loads((bundle / "task.json").read_text())["stages"][0]["name"]
    r = _jobset("prep", "task", *group_words(bundle, stage), "--bundle", bundle, "--target", "this")
    assert r.exit_code == 0, r.output
    token = next(p.name for p in bundle.iterdir()
                 if p.is_dir() and p.name.endswith(f"_{stage}"))
    deck = next(p for p in (bundle / token).iterdir()
                if p.suffix in (".fdf", ".py"))
    typed = re.search(r"jobset launch task --stage (\S+)", deck.read_text()).group(1)
    for spelling in (typed, typed.upper(), f"#{int(token.split('_')[0])}"):
        r = _jobset("launch", "task", "--stage", spelling, "--bundle", bundle,
                    "--mode", "direct", "--dry-run")
        assert r.exit_code == 0, (spelling, r.output)
        assert re.search(rf"WOULD start in the background\s+{stage}\s+bash H2_{token}\.run\.sh",
                         r.output), (spelling, r.output)
    ledger = bundle / LEDGER_FILE
    r = _jobset("launch", "task", "--stage", typed.upper(), "--bundle", bundle,
                "--mode", "submit")
    assert r.exit_code != 0 and "no scheduler header" in r.output, r.output
    last = json.loads(ledger.read_text().splitlines()[-1])
    assert (last["verb"], last["decision"], last["stage"]) == (
        "launch", "refused", stage), last
