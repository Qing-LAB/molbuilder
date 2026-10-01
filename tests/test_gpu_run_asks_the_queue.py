"""A run that uses the GPU asks the queue for one -- whatever its engine.

`execution/gpu.md` G1 and G7: ``use_gpu`` is one question for every engine,
and its answer travels to every reader -- the deck, the wrapper, the
scheduler's ask.  G5: an absent ``gpu_count`` is one device.  Until
2026-09-30 the ``.sbatch`` header asked a SIESTA deck alone ("only SIESTA
.fdf can be"), so a PySCF run whose run card said ``use_gpu`` was submitted
with no ``--gres``: no device, and the deck stops (G6, no CPU fallback).
PySCF carries no ``gpu_count`` at all, so the default was its only road.

Driven through ``jobset init`` -> the rung's run card (or the template) ->
``prep run``, on a machine whose record names a GPU queue with its own
partition.  Nothing is submitted.
"""
from __future__ import annotations

import json


def _jobset(*args):
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    return CliRunner().invoke(jobset_group, [str(a) for a in args])


def _prepped(tmp_path, monkeypatch, *, card=None, template_use_gpu=False,
             engine="pyscf", named_target=False):
    """A flat H2 calculation on ``engine``, its rung's card ``card``, prepped for a
    record whose GPU queue is ``public`` with ``gpu_partition = gpu`` and
    a100 cards: ``(the prep's result, the .sbatch text)``."""
    from pathlib import Path

    from conftest import write_machine_record
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.scheduler import (Domain, Environment, Topology,
                                      machine_scope_path, write_environment)

    queue = [Domain(name="public", partition="public", qos="public",
                    max_time="0-04:00:00", gpu_partition="gpu",
                    gpu={"type": "a100", "per_node": 4})]
    if named_target:
        # THIS machine is a workstation with no queue; the cluster is a
        # NAMED record whose probe saw no card (a login node) -- its queue
        # menu is where the card type lives.
        write_machine_record()
        named = Path(machine_scope_path()).parent / "environments"
        named.mkdir(parents=True, exist_ok=True)
        write_environment(Environment(
            scheduler="slurm", domains=queue,
            topology=Topology(sockets=2, cores_per_socket=8),
            script_generation={"activation": "conda activate",
                               "preamble": "true"}),
            named / "sol.json")
    else:
        write_machine_record(scheduler="slurm", domains=queue)
    tree = tmp_path / "projects"
    (tree / "P" / "structure").mkdir(parents=True)
    (tree / "P" / "structure" / "h2.xyz").write_text(
        "2\nh2\nH 0 0 0\nH 0 0 0.74\n")
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tree))
    monkeypatch.chdir(tree.parent)
    r = _jobset("init", "--structure", "P/structure/h2.xyz",
                "--bundle", "P/optimization/H2", "--engine", engine,
                "--shape", "flat", "--name", "H2")
    assert r.exit_code == 0, r.output
    bundle = tree / "P" / "optimization" / "H2"
    if engine == "siesta":
        from conftest import write_pseudos
        write_pseudos(bundle, ["H"])
    (bundle / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": "true"}}))
    if template_use_gpu:
        # THE TEMPLATE'S OWN VALUE, through its own writer -- the
        # description's base layer, with nothing on the card.
        import dataclasses

        from molbuilder.config.pyscf import PySCFConfig
        from molbuilder.template import (config_from_template,
                                         template_path, template_with_values)
        path = template_path(bundle, "H2")
        cfg = dataclasses.replace(
            config_from_template(path.read_text(), PySCFConfig), use_gpu=True)
        path.write_text(template_with_values(cfg, engine="pyscf",
                                             calculation="optimization"))
    desc = json.loads((bundle / "task.json").read_text())
    if card:
        desc["stages"][0]["execution"] = dict(card)
    (bundle / "task.json").write_text(json.dumps(desc))
    r = _jobset("prep", "run", "coarse", "--bundle", bundle,
                "--target", "sol" if named_target else "this")
    assert r.exit_code == 0, r.output
    [sbatch] = bundle.rglob("*.sbatch")
    return r, sbatch.read_text()


def test_a_pyscf_run_whose_card_says_gpu_asks_the_queue_for_one(
        tmp_path, monkeypatch):
    """GOAL: a PySCF GPU run is given a device by the queue.

    CONTRACT (`gpu.md` G1, G5, G7; `stages.md` § 6.8d): the rung's run card
    is the one place its ``use_gpu`` is said, and the run's ``.sbatch`` then
    asks for ``--gres=gpu:<the record's type>:1`` on the partition the
    record names for GPU work (`scheduler.md` § 4, ``gpu_partition``).
    """
    _r, header = _prepped(tmp_path, monkeypatch, card={"use_gpu": True})
    assert "#SBATCH --gres=gpu:a100:1" in header, header
    assert "#SBATCH -p gpu" in header, header


def test_a_template_that_says_gpu_asks_with_an_empty_card(tmp_path,
                                                          monkeypatch):
    """GOAL: the template's ``use_gpu`` is a device run too.

    CONTRACT (`generator.md` § 4.3a: every run that uses a device gets its
    ask at prep): the run's answer is the card over the template, and a
    template that says GPU with nothing on the card is asked one device.
    It returned before the ask until the K5 review (B1, 2026-09-30).
    """
    _r, header = _prepped(tmp_path, monkeypatch, template_use_gpu=True)
    assert "#SBATCH --gres=gpu:a100:1" in header, header


def test_a_count_without_the_gpu_asks_nothing_and_says_so(tmp_path,
                                                          monkeypatch):
    """GOAL: a ``gpu_count`` on a run that uses no device is not lost unsaid.

    CONTRACT (`gpu.md` G4, G5): a count is not an ask without ``use_gpu``,
    so no device is asked -- and ``prep`` says so, since the card offers both
    rows and a person half-way through deciding would otherwise lose the
    count silently (the K5 review's B4).  SIESTA's: PySCF's card carries no
    ``gpu_count``, and its description check refuses one.
    """
    r, header = _prepped(tmp_path, monkeypatch, card={"gpu_count": 2},
                         engine="siesta")
    assert "--gres" not in header, header
    assert "`gpu_count` is on the run card but `use_gpu` is off" in r.output, \
        r.output


def test_a_prep_for_another_machine_reads_that_machines_queues(
        tmp_path, monkeypatch):
    """GOAL: the card type comes from the machine the run is prepped FOR.

    CONTRACT (`generator.md` § 4.3a: the type is the target's, read from
    its own record): prepped here with ``--target sol``, a run whose card
    says ``use_gpu`` takes its type from Sol's queue menu -- Sol's probe saw
    no card, as a login node's does.  The menu was read from the folder,
    which on a fresh prep is this machine's (the K5 review's B2).
    """
    _r, header = _prepped(tmp_path, monkeypatch, card={"use_gpu": True},
                          named_target=True)
    assert "#SBATCH --gres=gpu:a100:1" in header, header
