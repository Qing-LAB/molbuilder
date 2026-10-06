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

from support.road import a_queue_that_answers, calls_made, describe_h2, jobset


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
    prep's flag STATED, the launch-door claim -- and asked about; with no
    one to answer nothing is sent or recorded.  With `--yes` that very line
    is sent and recorded.  `prep`'s header named `debug`, the queue that
    prep named, under a wall it admits (prep admits the run on the queue
    it names, `job-system.md` § 5.0 checkpoint 4); the wall is the stated
    ten minutes on either queue, never one queue's ceiling.

    The question and its answer are written down, a *no* included
    (`job-system.md` § 5.0, agreement 6); a dry run writes nothing, the
    ledger included (§ 6.0, step 3).

    MUTATIONS THIS MUST FAIL AGAINST: the stage's door putting a queue's
    ceiling in place of the stated wall; sending without asking; the ledger
    calling a declined launch a launch; a declined launch leaving no line
    (W55 D6); a dry run writing one (D14)."""
    bundle, calls = cluster
    r = jobset("launch", "run", "coarse", "--bundle", bundle,
               "--mode", "submit", "--domain", "htc")
    assert r.exit_code != 0 and "prep run coarse" in r.output, r.output

    _prep(bundle, "coarse", "--domain", "debug", "--time", "10m")
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
    assert _flag(shown, "-t") == "0-00:10:00", (
        "sent under a wall nobody stated: " + " ".join(shown))
    assert "ALL,MB_LAUNCHED_BY=jobset-launch" in shown, shown
    assert "nothing submitted" in r.output, r.output
    assert calls_made(calls) == [], "sent without the person's yes"
    assert not (attempt / "run.json").exists()
    asked = _ledger(bundle)[-1]
    assert (asked["decision"], asked["answer"]) == (
        "question", "no answer (not a terminal): nothing sent"), asked

    was = _ledger(bundle)
    r = jobset("launch", "run", "coarse", "--bundle", bundle,
               "--mode", "submit", "--domain", "htc", "--dry-run")
    assert r.exit_code == 0, r.output
    assert calls_made(calls) == [] and not (attempt / "run.json").exists()
    assert _ledger(bundle) == was, "the dry run wrote to the ledger"

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


def test_a_send_over_a_folder_changed_since_its_plan_is_refused(cluster):
    """`job-system.md` § 6.0, step 4: the send checks the folder against the
    one its plan was made from -- a file the plan read, written since, or
    the stage launched since by another launch, is refused, saying to
    launch again; nothing is sent, nothing recorded.

    API-LEVEL because the road cannot reach it: one `launch` makes its plan
    and sends it in one command, and the folder changes in between only
    while the person reads the question.  The plan and the send are the
    entry's own two calls (`submit.plan_launch`, `submit.send_launch`); the
    change between them is a person's.

    MUTATION THIS MUST FAIL AGAINST: the send comparing nothing."""
    import os

    from molbuilder.jobset.model import JobSet
    from molbuilder.jobset.submit import SubmitError, plan_launch, send_launch
    bundle, calls = cluster
    _prep(bundle, "coarse", "--domain", "debug", "--time", "10m")
    js = JobSet.load(bundle / "job-set.json")
    attempt = bundle / "01_coarse" / "run-0"

    def plan():
        return plan_launch(js, bundle, mode="submit", only="coarse",
                           domain="debug")

    shown = plan()
    deck = next(attempt.glob("*.fdf"))
    st = deck.stat()
    os.utime(deck, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))
    with pytest.raises(SubmitError) as e:
        send_launch(shown)
    said = str(e.value)
    assert "the folder changed since this launch was planned" in said, said
    assert deck.name in said and "Launch again" in said, said
    assert calls_made(calls) == [] and not (attempt / "run.json").exists()

    shown = plan()
    r = jobset("launch", "run", "coarse", "--bundle", bundle,
               "--mode", "submit", "--yes")
    assert r.exit_code == 0, r.output
    with pytest.raises(SubmitError) as e:
        send_launch(shown)
    assert "the folder changed since this launch was planned" in str(e.value)
    assert len(calls_made(calls)) == 1, "the stage was sent twice"
