"""The launch door -- through the road: `jobset init`, `prep`, `launch`.

PINS: ``docs/execution/job-system.md`` § 6 (*One request, three doors*: what
`prep` baked, under what was said at launch, admitted on the queue it is sent
to; the queue named once, for the work being launched; `-J
<calculation>/<job>`); ``docs/execution/architecture.md`` § 5.2 (every
launch value is stated -- the queue, the wall, the memory -- and a launch
flag overrides what prep baked, never fills what nobody stated);
``docs/execution/submission.md`` S4 (nothing is submitted unseen -- every
door; a judgement only the person can make takes "no" as Enter's answer);
``docs/execution/running-a-job.md`` § 5.5 (*`--mode ask` walks the identical
path `--mode submit` walks ... the line asked about is the line that would be
sent*; a flag the launch would not read is refused) and § 5.3.1 (`--mem 0` is
the whole node); ``docs/execution/project-layout.md`` § 1.6.4 (a stage
launched before is launched again by continuing it; one that never concluded
only on the person's judgement); plan W52.

PREVENTS, each read in the code before 2026-10-01 (the W52 review):

* a stage sent to the queue `--domain` named under a wall nobody stated --
  the one `prep`'s header had worked out for ANOTHER queue, a production
  stage killed at `debug`'s fifteen minutes -- and sent with no question,
  while the grouped bench and the bias chain asked;
* the queue one stage's prep baked routing every stage of the ladder;
* `--mode ask` asking about a line with no queue on it, while `submit` would
  have sent the baked one;
* `launch --mem 0` refused, though the help it shares with prep offers it;
* a GPU ask sent as typed (`--gres=a100:1`, a resource called `a100`), and
  a card in the ask sent to the queue at all (`scheduler.md` R2a:
  molbuilder names no card);
* a re-launch that could not continue leaving a fresh attempt behind, and a
  flat stage still in the queue launched again without a word.

The scheduler is a stub on PATH that queues nothing and writes every call
down (`support.road`); a finished run is the measured H2 relaxation.  Nothing
here launches an engine.
"""
from __future__ import annotations

import json

import pytest

from support.road import (a_finished_run, a_queue_that_answers, calls_made,
                          describe_h2, jobset)


def _queues(gpu: bool = False):
    from molbuilder.scheduler import Domain
    rows = [Domain(name="debug", partition="htc", qos="debug",
                   max_time="0-00:15:00"),
            Domain(name="htc", partition="htc", qos="public",
                   max_time="0-04:00:00")]
    if gpu:
        rows.append(Domain(name="gpu", partition="gpu", qos="public",
                           max_time="0-04:00:00",
                           gpu={"a100": 4}))
    return rows


def _states_its_wall_and_memory(bundle, **more):
    """The calculation states its wall and its memory -- task.json's
    `allocation` -- as a run on a scheduler does, or prep refuses
    (`architecture.md` § 5.2).  Its QUEUE each prep names for its own stage
    (`--domain`), which is what this file is about -- or, given here, the
    description names it for every stage."""
    task = json.loads((bundle / "task.json").read_text())
    task["allocation"] = {"time": "0-01:00:00", "mem": "8G", **more}
    (bundle / "task.json").write_text(json.dumps(task, indent=2))
    return bundle


@pytest.fixture
def cluster(tmp_path, monkeypatch):
    """A machine whose record names two queues -- `debug` (15 minutes) and
    `htc` (4 hours) -- a scheduler that answers, and the H2 ladder described
    on it, stating its wall and memory: ``(bundle, calls)``."""
    calls = a_queue_that_answers(tmp_path, monkeypatch, _queues())
    return _states_its_wall_and_memory(describe_h2(tmp_path, monkeypatch)), \
        calls


def _prep(bundle, stage="coarse", *more):
    r = jobset("prep", "run", stage, "--bundle", bundle, "--target", "this",
               *more)
    assert r.exit_code == 0, r.output
    return r


from support.road import sbatch_line as _line


def _flag(argv, flag):
    return argv[argv.index(flag) + 1]


def _ledger(bundle):
    from molbuilder.jobset.ledger import LEDGER_FILE
    return [json.loads(ln) for ln in
            (bundle / LEDGER_FILE).read_text().splitlines()]


def test_a_stage_is_shown_asked_and_sent_with_its_own_queues_wall(cluster):
    """`launch run coarse --mode submit --domain htc`: before prep it is
    refused, naming the prep; after it, the exact line is shown -- the
    calculation's name first in `-J`, the queue named at launch, the wall
    the description STATES, the launch-door claim -- and asked about; with
    no one to answer nothing is sent or recorded.  With `--yes` that very
    line is sent and recorded.  `prep`'s header named `debug`, the queue
    that prep named; the wall is the stated hour on either queue, never
    one queue's ceiling.

    The question and its answer are written down, a *no* included
    (`job-system.md` § 5.0, agreement 6).

    MUTATIONS THIS MUST FAIL AGAINST: the stage's door putting a queue's
    ceiling in place of the stated wall; sending without asking; the ledger
    calling a declined or a dry run a launch; a declined launch leaving no
    line (W55 D6)."""
    bundle, calls = cluster
    r = jobset("launch", "run", "coarse", "--bundle", bundle,
               "--mode", "submit", "--domain", "htc")
    assert r.exit_code != 0 and "prep run coarse" in r.output, r.output

    _prep(bundle, "coarse", "--domain", "debug")
    attempt = bundle / "01_coarse" / "run-0"
    header = next(attempt.glob("*.sbatch")).read_text()
    assert "#SBATCH -q debug" in header, header        # the queue prep named

    r = jobset("launch", "run", "coarse", "--bundle", bundle,
               "--mode", "submit", "--domain", "htc")
    assert r.exit_code == 0, r.output
    shown = _line(r.output)
    assert "about to submit" in r.output, r.output
    assert _flag(shown, "-J") == "H2/coarse", shown
    assert (_flag(shown, "-p"), _flag(shown, "-q")) == ("htc", "public")
    assert _flag(shown, "-t") == "0-01:00:00", (
        "sent under a wall nobody stated: " + " ".join(shown))
    assert "ALL,MB_LAUNCHED_BY=jobset-launch" in shown, shown
    assert "nothing submitted" in r.output, r.output
    assert calls_made(calls) == [], "sent without the person's yes"
    assert not (attempt / "run.json").exists()
    asked = _ledger(bundle)[-1]
    assert (asked["decision"], asked["answer"]) == (
        "question", "no answer (not a terminal): nothing sent"), asked

    r = jobset("launch", "run", "coarse", "--bundle", bundle,
               "--mode", "submit", "--domain", "htc", "--dry-run")
    assert r.exit_code == 0, r.output
    assert calls_made(calls) == [] and not (attempt / "run.json").exists()
    assert _ledger(bundle)[-1]["decision"] == "planned"

    r = jobset("launch", "run", "coarse", "--bundle", bundle,
               "--mode", "submit", "--domain", "htc", "--yes")
    assert r.exit_code == 0, r.output
    (where, argv), = calls_made(calls)
    assert where == attempt and argv == shown[1:], (where, argv, shown)
    record = json.loads((attempt / "run.json").read_text())
    assert record["job_id"] == "4242", record
    asked, sent = _ledger(bundle)[-2:]
    assert (asked["decision"], asked["answer"], sent["decision"]) == (
        "question", "yes (--yes)", "launched"), (asked, sent)


def test_the_queue_prep_baked_belongs_to_its_own_stage(cluster):
    """`prep run coarse --domain debug` names coarse's queue -- not medium's.
    Launched, coarse goes to `debug`; medium, prepped naming none, is
    refused at its prep with the record's queues listed (a run on a
    scheduler names its queue, `architecture.md` § 5.2); named, it goes to
    its own.

    MUTATION THIS MUST FAIL AGAINST: the baked queue read from every
    stage's row (medium sent to coarse's `debug`)."""
    bundle, calls = cluster
    _prep(bundle, "coarse", "--domain", "debug", "--time", "10m")
    r = jobset("prep", "run", "medium", "--bundle", bundle,
               "--target", "this", "--cold")
    assert r.exit_code != 0, r.output
    assert "(the target's record lists: debug, htc)" in r.output, r.output
    _prep(bundle, "medium", "--cold", "--domain", "htc")
    r = jobset("launch", "run", "coarse", "--bundle", bundle,
               "--mode", "submit", "--dry-run")
    assert r.exit_code == 0, r.output
    assert _flag(_line(r.output), "-q") == "debug", r.output
    r = jobset("launch", "run", "medium", "--bundle", bundle,
               "--mode", "submit", "--dry-run")
    assert r.exit_code == 0, r.output
    assert _flag(_line(r.output), "-q") == "public", r.output


def test_ask_asks_about_the_line_submit_would_send(cluster):
    """`--mode ask` on a stage whose prep named `htc`: the scheduler is
    asked once, with `--test-only` in front of exactly the line `submit`
    would send -- the baked queue on it -- and the line shown as the one to
    send carries no `--test-only`.  Nothing is recorded.

    MUTATION THIS MUST FAIL AGAINST: the queue resolved for `submit` alone
    (the question asks about a line naming no queue)."""
    bundle, calls = cluster
    _prep(bundle, "coarse", "--domain", "htc")
    r = jobset("launch", "run", "coarse", "--bundle", bundle,
               "--mode", "ask")
    assert r.exit_code == 0, r.output
    (where, argv), = calls_made(calls)
    assert argv[0] == "--test-only", argv
    assert (_flag(argv, "-p"), _flag(argv, "-q")) == ("htc", "public"), argv
    would = next(ln.split("would send:", 1)[1].split()
                 for ln in r.output.splitlines() if "would send:" in ln)
    assert would[0] == "sbatch" and "--test-only" not in would, would
    assert argv[1:] == would[1:], (argv, would)
    assert "2026-08-27T11:22:03" in r.output, r.output
    assert not (bundle / "01_coarse" / "run-0" / "run.json").exists()
    assert _ledger(bundle)[-1]["decision"] == "asked"


def test_memory_is_sent_as_said_and_zero_is_the_whole_node(cluster):
    """`--mem 64G` reaches the line as `--mem=64G`; `--mem 0` -- the help's
    own "all of the node's" -- as `--mem=0`; a spelling that is no amount of
    memory is refused in the verb's own words.

    MUTATION THIS MUST FAIL AGAINST: `launch` reading --mem as a number of
    gigabytes (`0` refused as "memory must be positive")."""
    bundle, _calls = cluster
    _prep(bundle, "coarse", "--domain", "htc")
    for said, sent in (("64G", "--mem=64G"), ("0", "--mem=0")):
        r = jobset("launch", "run", "coarse", "--bundle", bundle,
                   "--mode", "submit", "--domain", "htc", "--dry-run",
                   "--mem", said)
        assert r.exit_code == 0, (said, r.output)
        assert sent in _line(r.output), (said, r.output)
    r = jobset("launch", "run", "coarse", "--bundle", bundle,
               "--mode", "submit", "--domain", "htc", "--dry-run",
               "--mem", "12Q")
    assert r.exit_code != 0 and "--mem:" in r.output, r.output


def test_direct_runs_what_was_typed_and_refuses_what_it_would_not_read(
        cluster):
    """`--mode direct` runs the stage's own wrapper here, the ranks and
    threads its prep stated as arguments; a scheduler's flag beside it is
    refused by name.
    A bundle whose config says `launch.mode: submit` launches that way when
    nothing is typed -- to the queue its prep named -- and `--mode direct`
    typed over it runs here.

    MUTATION THIS MUST FAIL AGAINST: a direct run without -np/-omp."""
    bundle, _calls = cluster
    _prep(bundle, "coarse", "--np", "4", "--cpus-per-task", "2",
          "--domain", "htc")
    r = jobset("launch", "run", "coarse", "--bundle", bundle,
               "--mode", "direct", "--dry-run")
    assert r.exit_code == 0, r.output
    run = next(ln.split() for ln in r.output.splitlines()
               if "WOULD run" in ln)
    assert run[run.index("bash") + 1] == "H2_01_coarse.run.sh", run
    assert (_flag(run, "-np"), _flag(run, "-omp")) == ("4", "2"), run
    for flag, value in (("--domain", "htc"), ("--mem", "8G"),
                        ("--time", "1h")):
        r = jobset("launch", "run", "coarse", "--bundle", bundle,
                   "--mode", "direct", flag, value)
        assert r.exit_code != 0 and flag in r.output, (flag, r.output)

    from molbuilder.runtime_config import write_config_scope
    write_config_scope({"launch": {"mode": "submit"}})
    r = jobset("launch", "run", "coarse", "--bundle", bundle, "--dry-run")
    assert r.exit_code == 0, r.output
    assert _flag(_line(r.output), "-q") == "public", r.output
    r = jobset("launch", "run", "coarse", "--bundle", bundle,
               "--mode", "direct", "--dry-run")
    assert r.exit_code == 0, r.output


def test_a_stage_launched_before_continues_from_its_own_latest_run(cluster):
    """Launched and concluded, coarse launched again continues from its
    latest attempt: the plan says so and writes nothing; sent, `run-1` opens
    with the geometry carried -- and the optimizer's history, which a stage
    continuing from ITS OWN attempt is always handed: the one pair that
    cannot disagree with itself (A-3, final review 2026-08-13; pinned on a
    re-prep until 2026-10-02, when a prepped stage stopped being prepped
    again and this became the road to it).

    MUTATION THIS MUST FAIL AGAINST: a planned re-launch opening its
    attempt (a dry run that writes)."""
    bundle, calls = cluster
    _prep(bundle, "coarse", "--domain", "htc")
    run0 = bundle / "01_coarse" / "run-0"
    r = jobset("launch", "run", "coarse", "--bundle", bundle,
               "--mode", "submit", "--domain", "htc", "--yes")
    assert r.exit_code == 0, r.output
    a_finished_run(run0)
    (run0 / "H2.CG").write_text("cg history")

    r = jobset("launch", "run", "coarse", "--bundle", bundle,
               "--mode", "submit", "--domain", "htc", "--dry-run")
    assert r.exit_code == 0, r.output
    assert "WOULD continue 01_coarse/run-0 into run-1" in r.output, r.output
    assert not (bundle / "01_coarse" / "run-1").exists()

    r = jobset("launch", "run", "coarse", "--bundle", bundle,
               "--mode", "submit", "--domain", "htc", "--yes")
    assert r.exit_code == 0, r.output
    run1 = bundle / "01_coarse" / "run-1"
    assert (run1 / "H2.XV").read_bytes() == (run0 / "H2.XV").read_bytes()
    assert (run1 / "H2.CG").read_text() == "cg history", (
        "continuing from its own attempt withheld the optimizer history")
    assert [w for w, _a in calls_made(calls)] == [run0, run1]
    record = json.loads((run1 / "run.json").read_text())
    assert record["continued_from"] == "01_coarse/run-0", record


def test_a_run_that_never_concluded_is_followed_only_on_your_word(cluster):
    """Launched, its geometry saved, no conclusion: launching it again shows
    what only the person can judge -- it may still be running -- and with
    no one to answer sends nothing and opens nothing; `--yes` is the
    judgement recorded, and it continues.

    MUTATION THIS MUST FAIL AGAINST: the judgement taken without the
    person (an unconcluded run continued by default)."""
    bundle, calls = cluster
    _prep(bundle, "coarse", "--domain", "htc")
    run0 = bundle / "01_coarse" / "run-0"
    assert jobset("launch", "run", "coarse", "--bundle", bundle, "--mode",
                  "submit", "--domain", "htc", "--yes").exit_code == 0
    a_finished_run(run0, concluded=False)
    r = jobset("launch", "run", "coarse", "--bundle", bundle,
               "--mode", "submit", "--domain", "htc")
    assert r.exit_code == 0, r.output
    assert "never CONCLUDED" in r.output, r.output
    assert "nothing submitted" in r.output, r.output
    assert len(calls_made(calls)) == 1
    assert not (bundle / "01_coarse" / "run-1").exists()
    r = jobset("launch", "run", "coarse", "--bundle", bundle,
               "--mode", "submit", "--domain", "htc", "--yes")
    assert r.exit_code == 0, r.output
    assert (bundle / "01_coarse" / "run-1" / "H2.XV").is_file()


def test_a_relaunch_that_cannot_continue_opens_nothing(cluster):
    """Launched, and it left nothing to continue from (a run that died at
    startup): launching it again is refused with the story, and no attempt
    is opened behind the refusal -- and the way on it names is the rollback:
    a prepped stage is not prepped again (2026-10-02; it named a fresh `prep
    run` until then).

    MUTATION THIS MUST FAIL AGAINST: the new attempt opened before the
    continuation is checked."""
    bundle, calls = cluster
    # The QUEUE IN THE DESCRIPTION: the remedy is typed back as printed, a
    # bare `prep run`, so the stage's queue is the file's, not a flag's.
    _states_its_wall_and_memory(bundle, domain="htc")
    _prep(bundle, "coarse")
    assert jobset("launch", "run", "coarse", "--bundle", bundle, "--mode",
                  "submit", "--domain", "htc", "--yes").exit_code == 0
    r = jobset("launch", "run", "coarse", "--bundle", bundle,
               "--mode", "submit", "--domain", "htc", "--yes")
    assert r.exit_code != 0, r.output
    assert "impossible here" in r.output, r.output
    assert not (bundle / "01_coarse" / "run-1").exists()
    assert len(calls_made(calls)) == 1
    assert "molbuilder checkpoint list" in r.output, r.output


def test_a_flat_stage_still_unconcluded_is_asked_like_an_attempt(
        tmp_path, monkeypatch):
    """On the flat layout a stage's launch is its `<basename>.run.json`:
    launched and not concluded, launching it again asks first, as the
    hierarchy does -- nothing is sent over a run that may still be going.

    MUTATION THIS MUST FAIL AGAINST: "was it launched" reading an attempt's
    `run.json` only (a flat stage always reads never launched)."""
    calls = a_queue_that_answers(tmp_path, monkeypatch, _queues())
    bundle = _states_its_wall_and_memory(
        describe_h2(tmp_path, monkeypatch, shape="flat"))
    _prep(bundle, "coarse", "--domain", "htc")
    assert jobset("launch", "run", "coarse", "--bundle", bundle, "--mode",
                  "submit", "--domain", "htc", "--yes").exit_code == 0
    assert (bundle / "H2_01_coarse.run.json").is_file()
    r = jobset("launch", "run", "coarse", "--bundle", bundle,
               "--mode", "submit", "--domain", "htc")
    assert r.exit_code == 0, r.output
    assert "never CONCLUDED" in r.output, r.output
    assert len(calls_made(calls)) == 1, "sent again over a run in the queue"
