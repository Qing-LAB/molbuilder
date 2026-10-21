"""`jobset status` lists the description's ladder -- through the road:
`molbuilder describe`, the Task-setup save, `prep`, `status`, the Results tab.

PINS: ``docs/execution/job-system.md`` § 5.3 (the table is the description's
ladder: every stage with its number from the moment `init` writes it, the ones
not prepared yet as `ready` or `waiting`, a disabled one never the stage to resume from;
`status <stage>` is a stage in full -- its deck, what it declares, its
resources) and ``docs/web/results.md`` § 2.4
(the Results tab's ladder is `jobset_status`'s answer).

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



def _jobset(*args):
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    return CliRunner().invoke(jobset_group, [str(a) for a in args])


def _row(output, name):
    """The status table's row for ``name``: ``[seq, stage, group, attempt, state,
    ...]``."""
    return next(ln.split() for ln in output.splitlines()
                if ln.split()[1:2] == [name])


def test_status_answers_what_it_cannot_read_and_where_it_was_asked(
        tmp_path, monkeypatch):
    """A template prep would refuse is said in the table's last line, never
    a traceback; a stage renamed in case only keeps its prepared job; asked
    from inside one of its stage folders, `status` names the calculation it
    belongs to.

    MUTATIONS THIS MUST FAIL AGAINST: the continuation door raising (a
    traceback); stages joined to jobs by exact name; a stage folder told to
    run `init`."""
    from support.road import (describe_calculation, jobset, read_request,
                              save_both)
    bundle = describe_calculation(tmp_path, monkeypatch)

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


def test_status_ends_with_each_groups_next_step(tmp_path, monkeypatch):
    """`job-system.md`, *The task*, *The next step*: `status` ends with every
    group, where it stands and the line that moves it, the first group not
    finished the next.  A described ladder: its first group not prepared,
    its line `prep task --group 1`; prepared, ready to launch, its line the
    launch of group 1 -- the groups after it not prepared, with no line.
    The silent failure: a person told no next line after a prep, or the
    line of a group whose turn has not come."""
    from support.road import describe_calculation, jobset
    bundle = describe_calculation(tmp_path, monkeypatch)
    r = jobset("status", "--bundle", bundle)
    assert r.exit_code == 0, r.output
    steps = r.output.split("each group, and what is next:\n", 1)[1]
    first, *rest = steps.split("\n  group ")
    assert "not prepared -- next:" in first, steps
    assert "prep task --group 1 --bundle" in first, steps
    assert rest and all("next" not in g and "jobset" not in g
                        for g in rest), steps
    assert jobset("prep", "task", "--group", "1", "--bundle", bundle,
                  "--target", "this").exit_code == 0
    r = jobset("status", "--bundle", bundle)
    steps = r.output.split("each group, and what is next:\n", 1)[1]
    first, *rest = steps.split("\n  group ")
    assert "ready to launch -- next:" in first, steps
    assert "launch task --group 1" in first, steps
    assert rest and all("not prepared" in g and "jobset" not in g
                        for g in rest), steps
