"""R9's second check asks "what does THIS box know", never "which machine
is this calculation FOR" -- and until 2026-08-23 it asked the second
question by accident.

**The crash, verbatim from a user's own terminal.**  A workstation with a
named target on file (``environments/sol.json``, probed on Sol and copied
back) but no probe of its own, running an already-prepped bundle:

    python -m molbuilder jobset launch bench coarse --bundle ... --domain htc

The top-level resolution succeeds -- the bundle carries the snapshot
`prep --target sol` wrote, so `place()` sees all nine Sol domains and picks
one.  Then `_reject_if_this_machine_says_no` (`jobset/submit.py`), the
re-admission check that runs AFTER placement, called
``get_routing(project_dir=None)`` to ask "does the box actually running
this process object" -- and fell into `machine_for`'s C1 guard, which
exists to stop a DIFFERENT question (which machine a calculation is FOR)
from silently answering itself.  The traceback: ``AmbiguousTarget: several
machines could be meant and none was named`` -- from a read-only check that
names no target at all.

**Why the fix is a new question, not a wider exception handler.**  Catching
`AmbiguousTarget` here would work by accident: it would also swallow a
genuine "which machine" ambiguity if this call is ever reused somewhere
that DOES mean to ask that.  `local_only=True` instead states the real
question -- read `machine_scope_path()`, full stop, and let named targets
existing elsewhere be irrelevant to it, because they ARE irrelevant to it.
"""
from __future__ import annotations

import json

import pytest

_SOL = {
    "schema": "molbuilder/environment@2",
    "detected_at": "2026-08-01T00:00:00+00:00",
    "scheduler": "slurm",
    "topology": {"sockets": 2, "cores_per_socket": 32, "threads_per_core": 1,
                 "numa_per_socket": None, "gpus_per_node": 4,
                 "gpu_type": "a100", "mem_total_gb": 503.5},
    "site": {"partition": "htc", "qos": None, "account": None},
    "domains": [{"name": "htc", "partition": "htc", "qos": "public",
                 "max_cores": 48, "max_mem_gb": 501.0,
                 "default_mem_per_core_gb": 2.0}],
}


@pytest.fixture
def workstation_with_a_named_target(tmp_path, monkeypatch):
    """Exactly the reported shape: a named target exists, this machine has
    no probe of its own.  (`_isolated_machine_scope`, autouse in this suite
    since 2026-08-23, gives this test its own XDG_CONFIG_HOME.)

    **A second isolation gap, found writing this test.**  That fixture never
    touches cwd, and ``molbuilder.json``'s server-wide lookup tries
    ``./molbuilder.json`` FIRST (`configuration.md` § 2.1) -- so a test
    calling `get_routing` from the repo root reads the real, gitignored dev
    server config (TLS + auth secrets + a real `scheduler.routing` block),
    not nothing.  First draft of this test asserted `get_routing(...) == []`
    and got back THREE real domain names instead -- proof, not suspicion.
    Chdir'd here, narrowly, rather than folded into the autouse fixture:
    `isolated_projects_root`'s own docstring records a prior blanket cwd
    override breaking 13 unrelated tests, so widening this needs its own
    look before it becomes automatic for every test in the suite.
    """
    monkeypatch.chdir(tmp_path)
    from molbuilder.scheduler import environments_dir
    d = environments_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / "sol.json").write_text(json.dumps(_SOL))
    return d


def test_the_named_target_alone_still_makes_machine_for_ambiguous(
        workstation_with_a_named_target):
    """The control.  C1 must still fire for the question it exists to
    protect -- a caller asking "which machine is this FOR" with no bundle,
    no --target, and a named record on file.  If this stops raising, the
    fix below is silencing something real instead of asking a new
    question."""
    from molbuilder.scheduler.record import AmbiguousTarget, machine_for
    with pytest.raises(AmbiguousTarget):
        machine_for(None)


def test_local_only_reads_this_machine_and_ignores_named_targets(
        workstation_with_a_named_target):
    """**The regression.**  Same setup, the actual question R9 asks:
    `local_only=True` must return quietly -- this machine truly has no
    record of its own -- rather than raise about a target nobody named."""
    from molbuilder.scheduler.record import machine_for
    assert machine_for(None, local_only=True) is None


def test_get_routing_local_only_does_not_crash_and_reflects_no_local_probe(
        workstation_with_a_named_target):
    """The actual call site, one layer up.  Empty (no declared fallback
    configured either) is correct -- not an exception."""
    from molbuilder.runtime_config import get_routing
    assert get_routing(project_dir=None, local_only=True) == []


def test_local_only_reads_a_real_local_probe_when_one_exists():
    """`local_only` is not "always return None" -- when this machine DOES
    carry its own environment.json (e.g. running directly on a login node,
    no named targets involved), R9's check must still see its domains."""
    from molbuilder.scheduler import machine_scope_path
    from molbuilder.scheduler.record import machine_for
    p = machine_scope_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_SOL))
    env = machine_for(None, local_only=True)
    assert env is not None
    assert [d.name for d in env.domains] == ["htc"]


def test_the_full_readmission_check_no_longer_raises(
        workstation_with_a_named_target):
    """End to end through the real function the user's traceback named:
    `_reject_if_this_machine_says_no`, given a placement whose queue this
    (target-less) local machine simply has no opinion about."""
    from molbuilder.jobset.model import Resources
    from molbuilder.jobset.submit import _reject_if_this_machine_says_no
    from molbuilder.scheduler import Request

    class _Placed:
        partition = "htc"
        qos = "public"

    want = Request(ranks=48, cpus_per_task=1, gpus=None,
                   mem_gb=None, walltime_s=3600)
    _reject_if_this_machine_says_no(_Placed(), want, False, "bench")
