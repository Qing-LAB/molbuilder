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

import re

import pytest

from support.road import described, group_words, jobset, road_tree


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
    from molbuilder.jobset import ledger
    from molbuilder.jobset.naming import stage_home
    from molbuilder.task import of as task_of
    bundle = described(road_tree(tmp_path, monkeypatch, engine=engine),
                       calculation=calculation, engine=engine,
                       stage_strategy="")
    task = task_of(bundle)
    stage = task.stages[0].name
    r = jobset("prep", "task", *group_words(bundle, stage), "--bundle", bundle,
               "--target", "this")
    assert r.exit_code == 0, r.output
    home = stage_home(bundle, task, stage)
    deck = home.dir / home.names.name(".fdf" if engine == "siesta" else ".py")
    typed = re.search(r"jobset launch task --stage (\S+)", deck.read_text()).group(1)
    for spelling in (typed, typed.upper(), f"#{home.seq}"):
        r = jobset("launch", "task", "--stage", spelling, "--bundle", bundle,
                   "--mode", "direct", "--dry-run")
        assert r.exit_code == 0, (spelling, r.output)
        assert re.search(rf"WOULD start in the background\s+{stage}\s+bash "
                         rf"{re.escape(home.names.name('.run.sh'))}",
                         r.output), (spelling, r.output)
    r = jobset("launch", "task", "--stage", typed.upper(), "--bundle", bundle,
               "--mode", "submit")
    assert r.exit_code != 0 and "no scheduler header" in r.output, r.output
    last = ledger.read(bundle)[-1]
    assert (last["verb"], last["decision"], last["stage"]) == (
        "launch", "refused", stage), last
