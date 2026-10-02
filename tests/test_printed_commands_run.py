"""What molbuilder prints, you can type -- through the road: `jobset init`,
`prep`, `launch`, `status`, and every command they print, typed back as
printed (`support.road.each_is_taken`).

PINS: ``docs/execution/job-system.md`` § 5.3 (*what molbuilder prints, you
can type*; every printed command composed in one place -- the calculation
named unless the reader stands in it, a launch's mode stated where the
config sets none, a refusal that asks for a stage offering the stages and
the command for the first, one command a line, any prose after `#`; a
benchmark's sweep answered with its own verbs) and § 7 (a calculation with
no benchmark says so); plan W52.

PREVENTS, each read in the code before 2026-10-01 (the W52 review): a next
step printing `--mode submit|direct`, which bash runs as a pipe; remedies
naming no calculation, so a line pasted from anywhere but inside it acted on
another or on none; a `<stage>` nobody can type; a stage-less launch offered
with no mode; a note on the command's own line; a bench verb offering a stage
with no benchmark, or a calculation that has none; a sweep's status worded as
a ladder's, its trials read from the wrong folder.

Each printed `molbuilder jobset ...` line is split as a shell would and run
from where the person stands -- the projects tree's parent, not inside the
calculation -- with `--dry-run` added to a launch, which plans and sends
nothing.  Nothing here launches an engine.
"""
from __future__ import annotations

import pytest

from support.road import (a_finished_run, describe_h2, each_is_taken, jobset,
                          printed_commands)


def test_every_command_a_refusal_or_a_next_step_prints_is_taken(
        tmp_path, monkeypatch):
    """`prep run medium` before coarse has run is refused, naming how to run
    coarse first and how to start medium cold; `prep run` and `launch run`
    naming no stage offer the stages and the command for the first, the
    launch with its mode; `prep run coarse`'s next step names its launch;
    `status` names the next one.  Each line, typed back from outside the
    calculation, is taken.

    MUTATIONS THIS MUST FAIL AGAINST: a launch line printing
    `--mode submit|direct`; a printed command naming no calculation; a
    stage-less refusal offering no command, a `<stage>`, or a launch with no
    mode."""
    bundle = describe_h2(tmp_path, monkeypatch)
    r = jobset("prep", "run", "medium", "--bundle", bundle,
               "--target", "this")
    assert r.exit_code != 0, r.output
    printed = list(printed_commands(r.output))
    assert printed[0][:3] == ["prep", "run", "coarse"], r.output
    assert each_is_taken(r.output) >= 3         # prep, launch(es), --cold

    for verb in (("prep", "run", "--target", "this"),
                 ("launch", "run", "--mode", "direct")):
        r = jobset(*verb, "--bundle", bundle)
        assert r.exit_code != 0, (verb, r.output)
        assert each_is_taken(r.output), r.output     # the first stage's

    r = jobset("prep", "run", "coarse", "--bundle", bundle,
               "--target", "this")
    assert r.exit_code == 0, r.output
    assert each_is_taken(r.output), r.output

    a_finished_run(bundle / "01_coarse" / "run-0")
    r = jobset("status", "--bundle", bundle)
    assert r.exit_code == 0, r.output
    assert each_is_taken(r.output), r.output


@pytest.mark.parametrize("shape, container", [
    ("hierarchical", "02_medium/bench"),
    ("flat", "bench_02_medium"),
])
def test_every_command_a_benchmark_prints_is_taken(tmp_path, monkeypatch,
                                                   shape, container):
    """Medium's benchmark prepped, nothing else: its next steps; `launch
    bench` naming no stage, which offers medium -- the stage with a
    benchmark -- and its launch with the mode; `launch run medium`, refused
    with the prep of the first stage and the benchmark's note apart from
    it; `status` in the bench folder, which reads its trials where they are
    and names the sweep's own verbs for the calculation.  On both layouts.
    Each line, typed back from outside the calculation, is taken.

    MUTATIONS THIS MUST FAIL AGAINST: the stage-less bench refusal offering
    every stage (coarse, never benched, first); the note on the command's
    line; a sweep's status read from its bench folder, or worded as a
    ladder's; a flat sweep's stage not found (its container is
    `bench_<NN>_<stage>`, not inside a stage folder)."""
    bundle = describe_h2(tmp_path, monkeypatch, shape=shape)
    r = jobset("prep", "bench", "medium", "--bundle", bundle,
               "--target", "this")
    assert r.exit_code == 0, r.output
    assert each_is_taken(r.output), r.output

    for verb in (("launch", "bench", "--mode", "direct"),
                 ("launch", "run", "medium", "--mode", "direct")):
        r = jobset(*verb, "--bundle", bundle)
        assert r.exit_code != 0, (verb, r.output)
        assert each_is_taken(r.output), r.output

    r = jobset("status", "--bundle", bundle / container)
    assert r.exit_code == 0, r.output
    printed = list(printed_commands(r.output))
    # its trials read where they are -- prepped, not launched -- so its
    # launch comes first, then the read-back
    assert printed and printed[0][:3] == ["launch", "bench", "medium"], (
        r.output)
    assert all(w[1] == "bench" for w in printed), r.output
    assert each_is_taken(r.output), r.output


def test_a_calculation_with_no_benchmark_says_so_first(tmp_path,
                                                        monkeypatch):
    """PySCF has no benchmark lane: `prep bench` and `launch bench`, naming
    no stage, say so -- the entry's own words (`bench_refusal`) -- rather
    than offer a stage whose `prep bench` is refused in turn.

    MUTATION THIS MUST FAIL AGAINST: the stage-less refusal asked before
    the calculation's own (it offered `prep bench coarse`)."""
    bundle = describe_h2(tmp_path, monkeypatch, engine="pyscf")
    for verb in (("prep", "bench", "--target", "this"),
                 ("launch", "bench", "--mode", "direct")):
        r = jobset(*verb, "--bundle", bundle)
        assert r.exit_code != 0, (verb, r.output)
        assert "only speaks SIESTA" in r.output, (verb, r.output)
        each_is_taken(r.output)
