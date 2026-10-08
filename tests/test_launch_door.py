"""The launch door -- through the road: `jobset init`, `prep`, `launch`.

PINS: ``docs/execution/job-system.md`` § 6 (*One request, three doors*: what
`prep` baked, under what was said at launch, admitted on the queue it is sent
to; the queue named once, for the work being launched; `-J
<calculation>/<job>`); ``docs/execution/architecture.md`` § 5.2 (every
launch value is stated -- the queue, the wall, the memory -- and a launch
flag overrides what prep baked, never fills what nobody stated);
``docs/execution/submission.md`` S4 (nothing is submitted unseen -- every
door);
``docs/execution/running-a-job.md`` § 5.5 (a flag the launch would not read
is refused) and § 5.3.1 (`--mem 0` is the whole node); plan W52.

PREVENTS, each read in the code before 2026-10-01 (the W52 review):

* a stage sent to the queue `--domain` named under a wall nobody stated --
  the one `prep`'s header had worked out for ANOTHER queue, a production
  stage killed at `debug`'s fifteen minutes -- and sent with no question,
  while the grouped bench and the bias chain asked;
* the queue one stage's prep baked routing every stage of the ladder;
* `launch --mem 0` refused, though the help it shares with prep offers it.

The scheduler is a stub on PATH that queues nothing and writes every call
down (`support.road`).  Nothing here launches an engine.
"""
from __future__ import annotations

import json

import pytest

from support.road import a_machine_with_queues, calls_made, describe_calculation, jobset


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
    `htc` (4 hours) -- an `sbatch` that writes down each call and refuses
    it (no scheduler text in the basic tests), and the H2 ladder described
    on it, stating its wall and memory: ``(bundle, calls)``."""
    calls = a_machine_with_queues(tmp_path, monkeypatch, _queues())
    return _states_its_wall_and_memory(describe_calculation(tmp_path, monkeypatch)), \
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


def test_a_send_over_a_folder_changed_since_its_plan_is_refused(cluster):
    """`job-system.md` § 6.0, step 4: the send checks the folder against the
    one its plan was made from -- a file the plan read, written since, is
    refused, saying to launch again; nothing is sent, and the refusal is
    written down.

    API-LEVEL because the road cannot reach it: one `launch` makes its plan
    and sends it in one command, and the folder changes in between only
    while the person reads the question.  The plan and the send are the
    entry's own two calls (`submit.plan_launch`, `submit.send_launch`); the
    change between them is a person's.

    MUTATION THIS MUST FAIL AGAINST: the send comparing nothing."""
    import os

    from molbuilder.jobset.model import JobSet
    from molbuilder.jobset.submit import SubmitError, plan_launch, send_launch
    from molbuilder.jobset.ask import Said
    bundle, calls = cluster
    _prep(bundle, "coarse", "--domain", "debug", "--time", "10m")
    js = JobSet.load(bundle / "job-set.json")
    attempt = bundle / "01_coarse" / "run-0"

    def plan():                  # the queue its prep admitted: `debug`
        return plan_launch(js, bundle, mode="submit", only="coarse",
                           told=dict(kind="run", stage="coarse", trial=None,
                                     mode="submit", flags=[]))

    shown = plan()
    deck = next(attempt.glob("*.fdf"))
    st = deck.stat()
    os.utime(deck, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))
    with pytest.raises(SubmitError) as e:
        send_launch(shown, said=Said(True, "yes (--yes)"))
    said = str(e.value)
    assert "the folder changed since this launch was planned" in said, said
    assert deck.name in said and "Launch again" in said, said
    assert calls_made(calls) == [] and not (attempt / "run.json").exists()
    assert _ledger(bundle)[-1]["decision"] == "refused"
