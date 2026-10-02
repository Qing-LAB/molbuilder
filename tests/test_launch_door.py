"""The launch door -- through the road: `jobset init`, `prep`, `launch`.

PINS: ``docs/execution/job-system.md`` § 6 (*One request, three doors*: what
`prep` baked, under what was said at launch, admitted on the queue it is sent
to, the wall that queue's own ceiling when nothing states one; the queue
named once, for the work being launched; `-J <calculation>/<job>`);
``docs/execution/submission.md`` S4 (nothing is submitted unseen -- every
door; a judgement only the person can make takes "no" as Enter's answer);
``docs/execution/running-a-job.md`` § 5.5 (*`--mode ask` walks the identical
path `--mode submit` walks ... the line asked about is the line that would be
sent*; a flag the launch would not read is refused) and § 5.3.1 (`--mem 0` is
the whole node); ``docs/execution/project-layout.md`` § 1.6.4 (a stage
launched before is launched again by continuing it; one that never concluded
only on the person's judgement); plan W52.

PREVENTS, each read in the code before 2026-10-01 (the W52 review):

* a stage sent to the queue `--domain` named under the wall `prep`'s header
  had worked out for ANOTHER queue -- a production stage killed at `debug`'s
  fifteen minutes -- and sent with no question, while the grouped bench and
  the bias chain asked;
* the queue one stage's prep baked routing every stage of the ladder;
* `--mode ask` asking about a line with no queue on it, while `submit` would
  have sent the baked one;
* `launch --mem 0` refused, though the help it shares with prep offers it;
* `--gpus a100:1` sent as `--gres=a100:1`, a resource called `a100`;
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
                          describe_h2, each_is_taken, jobset)


def _queues(gpu: bool = False):
    from molbuilder.scheduler import Domain
    rows = [Domain(name="debug", partition="htc", qos="debug",
                   max_time="0-00:15:00"),
            Domain(name="htc", partition="htc", qos="public",
                   max_time="0-04:00:00")]
    if gpu:
        rows.append(Domain(name="gpu", partition="gpu", qos="public",
                           max_time="0-04:00:00",
                           gpu={"type": "a100", "per_node": 4}))
    return rows


@pytest.fixture
def cluster(tmp_path, monkeypatch):
    """A machine whose record names two queues -- `debug` (15 minutes) and
    `htc` (4 hours) -- a scheduler that answers, and the H2 ladder described
    on it: ``(bundle, calls)``."""
    calls = a_queue_that_answers(tmp_path, monkeypatch, _queues())
    return describe_h2(tmp_path, monkeypatch), calls


def _prep(bundle, stage="coarse", *more):
    r = jobset("prep", "run", stage, "--bundle", bundle, "--target", "this",
               *more)
    assert r.exit_code == 0, r.output
    return r


def _line(output: str):
    """The `sbatch` line a launch showed -- in its question, or as a dry
    run's ``WOULD run`` -- as its arguments."""
    for ln in output.splitlines():
        words = ln.split()
        if "sbatch" in words:
            words = words[words.index("sbatch"):]
            return [w for w in words if not w.startswith("[")]
    raise AssertionError(f"no sbatch line shown:\n{output}")


def _flag(argv, flag):
    return argv[argv.index(flag) + 1]


def _ledger(bundle):
    from molbuilder.jobset.ledger import LEDGER_FILE
    return [json.loads(ln) for ln in
            (bundle / LEDGER_FILE).read_text().splitlines()]


def test_a_stage_is_shown_asked_and_sent_with_its_own_queues_wall(cluster):
    """`launch run coarse --mode submit --domain htc`: before prep it is
    refused, naming the prep; after it, the exact line is shown -- the
    calculation's name first in `-J`, the queue named, the wall of THAT
    queue, the launch-door claim -- and asked about; with no one to answer
    nothing is sent or recorded.  With `--yes` that very line is sent and
    recorded.  `prep`'s header named `debug`, the menu's own first choice,
    so a wall taken from the header would be fifteen minutes.

    MUTATIONS THIS MUST FAIL AGAINST: the stage's door without the wall
    default (the header's `debug` wall stands); sending without asking;
    the ledger calling a declined or a dry run a launch."""
    bundle, calls = cluster
    r = jobset("launch", "run", "coarse", "--bundle", bundle,
               "--mode", "submit", "--domain", "htc")
    assert r.exit_code != 0 and "prep run coarse" in r.output, r.output

    _prep(bundle)
    attempt = bundle / "01_coarse" / "run-0"
    header = next(attempt.glob("*.sbatch")).read_text()
    assert "#SBATCH -q debug" in header, header        # prep's own choice

    r = jobset("launch", "run", "coarse", "--bundle", bundle,
               "--mode", "submit", "--domain", "htc")
    assert r.exit_code == 0, r.output
    shown = _line(r.output)
    assert "about to submit" in r.output, r.output
    assert _flag(shown, "-J") == "H2/coarse", shown
    assert (_flag(shown, "-p"), _flag(shown, "-q")) == ("htc", "public")
    assert _flag(shown, "-t") == "0-04:00:00", (
        "sent to htc under another queue's wall: " + " ".join(shown))
    assert "ALL,MB_LAUNCHED_BY=jobset-launch" in shown, shown
    assert "nothing submitted" in r.output, r.output
    assert calls_made(calls) == [], "sent without the person's yes"
    assert not (attempt / "run.json").exists()

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
    assert _ledger(bundle)[-1]["decision"] == "launched"


def test_the_queue_prep_baked_belongs_to_its_own_stage(cluster):
    """`prep run coarse --domain debug` names coarse's queue -- not medium's.
    Launched, coarse goes to `debug`; medium, which named none, is asked to
    name one: the queues are listed and nothing is sent.

    MUTATION THIS MUST FAIL AGAINST: the baked queue read from every
    stage's row (medium sent to coarse's `debug`)."""
    bundle, calls = cluster
    _prep(bundle, "coarse", "--domain", "debug")
    _prep(bundle, "medium", "--cold")
    r = jobset("launch", "run", "coarse", "--bundle", bundle,
               "--mode", "submit", "--dry-run")
    assert r.exit_code == 0, r.output
    assert _flag(_line(r.output), "-q") == "debug", r.output
    r = jobset("launch", "run", "medium", "--bundle", bundle,
               "--mode", "submit", "--dry-run")
    assert r.exit_code != 0, r.output
    assert "no --domain, so no queue was chosen" in r.output, r.output
    assert "htc/public" in r.output, r.output           # the table, listed


def test_ask_asks_about_the_line_submit_would_send(cluster):
    """`--mode ask` on a stage whose prep named `htc`: the scheduler is
    asked once, with `--test-only` in front of exactly the line `submit`
    would send -- the baked queue on it -- and the line shown as the one to
    send carries no `--test-only`.  Nothing is recorded.

    MUTATION THIS MUST FAIL AGAINST: the queue resolved for `submit` alone
    (the question asks about the menu's own preference, `debug`)."""
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
    _prep(bundle)
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


def test_a_gpu_ask_reaches_sbatch_in_slurms_spelling(tmp_path, monkeypatch):
    """`prep run coarse --gpus a100:1` -- the help's own example -- is
    recorded and sent as `gpu:a100:1`, the spelling `--gres` reads.

    MUTATION THIS MUST FAIL AGAINST: the record keeping the ask as typed
    (`--gres=a100:1`, a resource called `a100`)."""
    calls = a_queue_that_answers(tmp_path, monkeypatch, _queues(gpu=True))
    bundle = describe_h2(tmp_path, monkeypatch)
    _prep(bundle, "coarse", "--gpus", "a100:1")
    js = json.loads((bundle / "job-set.json").read_text())
    coarse = next(j for j in js["jobs"] if j["name"] == "coarse")
    assert coarse["resources"]["gres"] == "gpu:a100:1", coarse
    r = jobset("launch", "run", "coarse", "--bundle", bundle,
               "--mode", "submit", "--domain", "gpu", "--dry-run")
    assert r.exit_code == 0, r.output
    assert "--gres=gpu:a100:1" in _line(r.output), r.output
    assert calls_made(calls) == []


def test_direct_runs_what_was_typed_and_refuses_what_it_would_not_read(
        cluster):
    """`--mode direct` runs the stage's own wrapper here, the ranks and
    threads its prep stated as arguments; a scheduler's flag beside it is
    refused by name.
    A bundle whose config says `submit` to `htc` launches that way when
    nothing is typed, and `--mode direct` typed over it is not handed the
    configured queue.

    MUTATIONS THIS MUST FAIL AGAINST: a direct run without -np/-omp; a
    configured `execution.domain` poured into a direct run."""
    bundle, _calls = cluster
    _prep(bundle, "coarse", "--np", "4", "--cpus-per-task", "2")
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

    cfg = json.loads((bundle / ".molbuilder.json").read_text())
    cfg["execution"] = {"mode": "submit", "domain": "htc"}
    (bundle / ".molbuilder.json").write_text(json.dumps(cfg))
    r = jobset("launch", "run", "coarse", "--bundle", bundle, "--dry-run")
    assert r.exit_code == 0, r.output
    assert _flag(_line(r.output), "-q") == "public", r.output
    r = jobset("launch", "run", "coarse", "--bundle", bundle,
               "--mode", "direct", "--dry-run")
    assert r.exit_code == 0, r.output


def test_a_stage_launched_before_continues_from_its_own_latest_run(cluster):
    """Launched and concluded, coarse launched again continues from its
    latest attempt: the plan says so and writes nothing; sent, `run-1` opens
    with the geometry carried and the job goes from there.

    MUTATION THIS MUST FAIL AGAINST: a planned re-launch opening its
    attempt (a dry run that writes)."""
    bundle, calls = cluster
    _prep(bundle)
    run0 = bundle / "01_coarse" / "run-0"
    r = jobset("launch", "run", "coarse", "--bundle", bundle,
               "--mode", "submit", "--domain", "htc", "--yes")
    assert r.exit_code == 0, r.output
    a_finished_run(run0)

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
    _prep(bundle)
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
    is opened behind the refusal -- and the new attempt it names is a
    command that is taken as printed.

    MUTATIONS THIS MUST FAIL AGAINST: the new attempt opened before the
    continuation is checked; the remedy naming no calculation, a note on
    its line (`job-system.md` § 5.3)."""
    bundle, calls = cluster
    _prep(bundle)
    assert jobset("launch", "run", "coarse", "--bundle", bundle, "--mode",
                  "submit", "--domain", "htc", "--yes").exit_code == 0
    r = jobset("launch", "run", "coarse", "--bundle", bundle,
               "--mode", "submit", "--domain", "htc", "--yes")
    assert r.exit_code != 0, r.output
    assert "impossible here" in r.output, r.output
    assert not (bundle / "01_coarse" / "run-1").exists()
    assert len(calls_made(calls)) == 1
    assert each_is_taken(r.output) == 1, r.output


def test_a_flat_stage_still_unconcluded_is_asked_like_an_attempt(
        tmp_path, monkeypatch):
    """On the flat layout a stage's launch is its `<basename>.run.json`:
    launched and not concluded, launching it again asks first, as the
    hierarchy does -- nothing is sent over a run that may still be going.

    MUTATION THIS MUST FAIL AGAINST: "was it launched" reading an attempt's
    `run.json` only (a flat stage always reads never launched)."""
    calls = a_queue_that_answers(tmp_path, monkeypatch, _queues())
    bundle = describe_h2(tmp_path, monkeypatch, shape="flat")
    _prep(bundle)
    assert jobset("launch", "run", "coarse", "--bundle", bundle, "--mode",
                  "submit", "--domain", "htc", "--yes").exit_code == 0
    assert (bundle / "H2_01_coarse.run.json").is_file()
    r = jobset("launch", "run", "coarse", "--bundle", bundle,
               "--mode", "submit", "--domain", "htc")
    assert r.exit_code == 0, r.output
    assert "never CONCLUDED" in r.output, r.output
    assert len(calls_made(calls)) == 1, "sent again over a run in the queue"
