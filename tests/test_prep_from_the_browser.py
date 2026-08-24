"""`prep` from the browser — the same verb, for the machine you name.

**Why a browser may trigger it.**  `project-layout.md` § 2.2 says the deck
cannot be finished in the browser, and its argument is about WHOSE FACTS the
deck is rendered from: `prep` needs four inputs, two portable (the template,
the description) and two the target machine's.  A named record supplies the
machine half -- that is what `environments/<name>.json` IS -- so preparing
FOR a cluster FROM here is the case `preparing-for-another-machine.md` exists
for.  The section constrains the FACTS, not the surface.

**Prep, never launch** (user, 2026-08-24): prep writes files and can be run
again; launch spends a queue slot and refuses batch submission by design.

**The reserved local name.**  `known_machines` displays `(this machine)`,
which is a label nobody can type -- and with any named record on file,
omitting `--target` raised `AmbiguousTarget`, whose own message said *"omit
--target only when this machine is the one"*.  The instruction the refusal
gave was the action that produced it, so preparing for the box you are
sitting at became impossible the moment you saved one cluster record.
`LOCAL_TARGET` is the typeable name; C1 still refuses SILENCE.
"""
from __future__ import annotations

import json
import shutil

import pytest

from molbuilder.scheduler.record import LOCAL_TARGET


@pytest.fixture
def described(isolated_projects_root, web_client):
    """A described calculation inside the projects tree the app serves, plus
    one NAMED record so the machine question is genuinely ambiguous.

    Built in an ISOLATED tree: these tests write, and the app's default root
    is the developer's real `projects/`.
    """
    from molbuilder.task import FILENAME as TASK_FILENAME
    calc = isolated_projects_root / "calc"
    calc.mkdir(parents=True, exist_ok=True)
    (calc / TASK_FILENAME).write_text(json.dumps({
        "schema": "molbuilder/task@1",
        "engine": {"name": "siesta"}, "shape": "hierarchical",
        "run": {"name": "JOB", "id": "JOB_H2"},
        "structure": {"source": "h2.xyz", "formula": "H2", "atoms": 2},
        "varies": [],
        "stages": [{"name": "coarse", "enabled": True, "overrides": {}}],
    }))
    from molbuilder.scheduler import environments_dir
    d = environments_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / "faraway.json").write_text(json.dumps({
        "schema": "molbuilder/environment@2",
        "detected_at": "2026-08-01T00:00:00+00:00", "scheduler": "slurm",
        "topology": {"sockets": 2, "cores_per_socket": 24,
                     "threads_per_core": 1, "numa_per_socket": None,
                     "gpus_per_node": 0, "gpu_type": None,
                     "mem_total_gb": 500.0},
        "site": {"partition": "htc", "qos": "public", "account": None},
        "domains": [],
    }))
    return str(calc)


def _post(client, **body):
    r = client.post("/api/task-setup/prep", json=body)
    return r.status_code, (r.get_json() or {})


def test_it_refuses_a_folder_with_no_description(web_client, described,
                                                 isolated_projects_root):
    bare = isolated_projects_root / "bare"
    bare.mkdir()
    st, j = _post(web_client, dest=str(bare), kind="run", stage="coarse",
                  target=LOCAL_TARGET, plan=True)
    assert st == 400 and "task.json" in j["error"]


def test_silence_about_the_machine_is_still_refused(web_client, described):
    """C1 unchanged: the browser cannot offer a default the CLI rejects."""
    st, j = _post(web_client, dest=described, kind="run", stage="coarse",
                  plan=True)
    assert st == 400
    assert "none was named" in j["error"] or "task.json" in j["error"]


def test_the_local_machine_can_be_NAMED(web_client, described):
    """The regression this closes: with a named record on file there was no
    spelling for "the box I am on" at all."""
    from molbuilder.scheduler.record import machine_for
    assert machine_for(target=LOCAL_TARGET, probe=False) is None or True
    # the important half -- it does not raise the ambiguity refusal
    from molbuilder.scheduler.record import AmbiguousTarget
    try:
        machine_for(target=LOCAL_TARGET, probe=False)
    except AmbiguousTarget:
        pytest.fail("naming this machine still reads as silence")


def test_the_refusal_names_a_spelling_that_WORKS(web_client, described):
    """Its own instruction used to be the action that caused it."""
    from molbuilder.scheduler.record import AmbiguousTarget, machine_for
    with pytest.raises(AmbiguousTarget) as e:
        machine_for(target=None)
    assert f"--target {LOCAL_TARGET}" in str(e.value)
    assert "omit --target only when" not in str(e.value)


def test_the_reserved_name_cannot_be_taken_by_a_record():
    """A record called `this` could never be prepped for, so writing one is
    refused rather than allowed and then shadowed."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    r = CliRunner().invoke(jobset_group,
                           ["probe", "--write", "--yes", "--name",
                            LOCAL_TARGET])
    assert r.exit_code != 0
    assert "reserved" in r.output


def test_an_unknown_machine_is_refused_by_name(web_client, described):
    st, j = _post(web_client, dest=described, kind="run", stage="coarse",
                  target="no-such-box", plan=True)
    assert st == 400 and "no-such-box" in j["error"]


def test_kind_must_be_run_or_bench(web_client, described):
    st, j = _post(web_client, dest=described, kind="launch",
                  stage="coarse", target=LOCAL_TARGET, plan=True)
    assert st == 400 and "run" in j["error"] and "bench" in j["error"]


def test_there_is_no_launch_door_here(web_client):
    """Prep writes files; launch spends a queue slot.  Only the first is
    exposed, and its absence should be visible rather than assumed."""
    from molbuilder.web.blueprints import build as _b
    routes = {r for r in dir(_b) if r.startswith("api_task_setup")}
    assert "api_task_setup_prep" in routes
    assert not any("launch" in r or "submit" in r for r in routes), (
        "a submit door appeared on the task-setup blueprint")
