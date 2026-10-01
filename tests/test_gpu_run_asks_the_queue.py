"""A run that uses the GPU asks the queue for one -- whatever its engine.

`execution/gpu.md` G1 and G7: ``use_gpu`` is one question for every engine,
and its answer travels to every reader -- the deck, the wrapper, the
scheduler's ask.  G5: an absent ``gpu_count`` is one device.  Until
2026-09-30 the ``.sbatch`` header asked a SIESTA deck alone ("only SIESTA
.fdf can be"), so a PySCF run whose run card said ``use_gpu`` was submitted
with no ``--gres``: no device, and the deck stops (G6, no CPU fallback).
PySCF carries no ``gpu_count`` at all, so the default was its only road.

Driven through ``jobset init`` -> the rung's run card -> ``prep run``, on a
machine whose record names a GPU queue.  Nothing is submitted.
"""
from __future__ import annotations

import json


def _jobset(*args):
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    return CliRunner().invoke(jobset_group, [str(a) for a in args])


def test_a_pyscf_run_whose_card_says_gpu_asks_the_queue_for_one(
        tmp_path, monkeypatch):
    """GOAL: a PySCF GPU run is given a device by the queue.

    CONTRACT (`gpu.md` G1, G5, G7; `stages.md` § 6.8d): the rung's run card
    is the one place its ``use_gpu`` is said, and the run's ``.sbatch`` then
    asks for ``--gres=gpu:<the record's type>:1`` on the partition the
    record names for GPU work (`scheduler.md` § 4, ``gpu_partition``).
    """
    from conftest import write_machine_record
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.scheduler import Domain

    write_machine_record(scheduler="slurm", domains=[
        Domain(name="public", partition="public", qos="public",
               max_time="0-04:00:00", gpu_partition="gpu",
               gpu={"type": "a100", "per_node": 4})])
    tree = tmp_path / "projects"
    (tree / "P" / "structure").mkdir(parents=True)
    (tree / "P" / "structure" / "h2.xyz").write_text(
        "2\nh2\nH 0 0 0\nH 0 0 0.74\n")
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tree))
    monkeypatch.chdir(tree.parent)
    r = _jobset("init", "--structure", "P/structure/h2.xyz",
                "--bundle", "P/optimization/H2", "--engine", "pyscf",
                "--shape", "flat", "--name", "H2")
    assert r.exit_code == 0, r.output
    bundle = tree / "P" / "optimization" / "H2"
    (bundle / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": "true"}}))
    desc = json.loads((bundle / "task.json").read_text())
    desc["stages"][0]["execution"] = {"use_gpu": True}
    (bundle / "task.json").write_text(json.dumps(desc))

    r = _jobset("prep", "run", "coarse", "--bundle", bundle,
                "--target", "this")
    assert r.exit_code == 0, r.output
    [sbatch] = bundle.rglob("*.sbatch")
    header = sbatch.read_text()
    assert "#SBATCH --gres=gpu:a100:1" in header, header
    assert "#SBATCH -p gpu" in header, header
