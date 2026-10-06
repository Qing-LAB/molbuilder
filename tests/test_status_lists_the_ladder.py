"""`jobset status` lists the description's ladder -- through the road:
`jobset init`, the Task-setup save, `prep`, `status`, the Results tab.

PINS: ``docs/execution/job-system.md`` § 5.3 (the table is the description's
ladder: every stage with its number from the moment `init` writes it, the ones
not prepped yet as not-started, a disabled one never the stage to resume from;
`status <stage>` is a stage in full -- its deck, what it declares, its
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



def _jobset(*args):
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    return CliRunner().invoke(jobset_group, [str(a) for a in args])


def _row(output, name):
    """The status table's row for ``name``: ``[seq, stage, attempt, state,
    ...]``."""
    return next(ln.split() for ln in output.splitlines()
                if ln.split()[1:2] == [name])


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
