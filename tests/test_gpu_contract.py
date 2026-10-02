"""The GPU contract (`docs/execution/gpu.md` § 1), case by case.

THE CASES ARE DATA -- ``tests/data/gpu_contract.toml`` -- and each runs down
the road a person runs: `jobset init`, the description and the target's
record written as the case says, `prep`, `launch --dry-run`, and on a
machine with no queue the run script's own dry run.  Three layers (user,
2026-10-01: "(1) query, allow or deny, (2) correctly produce slurm
command/header, (3) correctly execute locally if that's the target"), and a
case checks the ones it names; the table's header says which key is which.

PREVENTS: a GPU rule held by tests of internal steps, each building its own
inputs and restating the same assertion, so that changing the rule meant
patching every restatement -- thirty tests in fifteen files for one rule on
2026-10-01.  A rule changes; its rows change.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

from support.road import (a_queue_that_answers, describe_h2, gpus_given,
                          jobset, sbatch_line, strip_preamble_activation)

TABLE = tomllib.loads(
    (Path(__file__).parent / "data" / "gpu_contract.toml").read_text())


def _target(case, tmp_path, monkeypatch) -> str:
    """The machine the case preps for, recorded as a probe records one --
    and the name `--target` takes for it."""
    from conftest import write_machine_record
    from molbuilder.scheduler import (Domain, Environment, Topology,
                                      machine_scope_path, write_environment)
    queues = [Domain.from_row(q) for q in case.get("queues", TABLE["queues"])]
    record = dict(case.get("record", {}))
    where = case.get("machine", "this")
    if where == "this":
        a_queue_that_answers(tmp_path, monkeypatch, queues, **record)
        return "this"
    write_machine_record(**record)          # this machine: no queue at all
    if where == "workstation":
        return "this"
    named = Path(machine_scope_path()).parent / "environments"
    named.mkdir(parents=True, exist_ok=True)
    write_environment(Environment(
        scheduler="slurm", domains=queues,
        topology=Topology(sockets=2, cores_per_socket=24),
        script_generation={"activation": "conda activate", "preamble": "true"}),
        named / "sol.json")
    return "sol"


def _describe(case, tmp_path, monkeypatch) -> Path:
    """`jobset init` of H2, then the case's template values, description
    blocks and `.molbuilder.json`."""
    engine = case.get("engine", "siesta")
    bundle = describe_h2(tmp_path, monkeypatch, engine=engine)
    values = dict(TABLE["siesta_template"]) if engine == "siesta" else {}
    values.update(case.get("template", {}))
    if values:
        from molbuilder.config.pyscf import PySCFConfig
        from molbuilder.config.siesta import SiestaConfig
        from molbuilder.template import (config_from_template, template_path,
                                         template_with_values)
        cls = SiestaConfig if engine == "siesta" else PySCFConfig
        path = template_path(bundle, "H2")
        cfg = dataclasses.replace(
            config_from_template(path.read_text(), cls), **values)
        path.write_text(template_with_values(cfg, engine=engine,
                                             calculation="optimization"))
    task = json.loads((bundle / "task.json").read_text())
    for block, key in (("execution", "run"), ("allocation", "allocation"),
                       ("bench", "bench")):
        if key in case:
            task[block] = case[key]
    (bundle / "task.json").write_text(json.dumps(task, indent=2))
    if "config" in case:
        mine = json.loads((bundle / ".molbuilder.json").read_text())
        mine.update(case["config"])
        (bundle / ".molbuilder.json").write_text(json.dumps(mine))
    return bundle


def _lines(case, key, text):
    """``key``'s lines are in ``text``; ``key_lacks``'s are not -- read with
    runs of blanks as one, since a deck aligns its values in columns."""
    flat = re.sub(r"[ \t]+", " ", text)
    for line in case.get(key, []):
        assert line in flat, f"{key}: {line!r} missing from\n{text}"
    for line in case.get(f"{key}_lacks", []):
        assert line not in flat, f"{key}: {line!r} present in\n{text}"


def _the_runs(bundle: Path, pattern: str) -> Path:
    """The file of this pattern the run's prep wrote -- in the stage's
    folder and its attempt, one text in both."""
    found = [p for p in bundle.rglob(pattern)
             if "bench" not in p.relative_to(bundle).parts]
    assert found and len({p.read_text() for p in found}) == 1, found
    return found[-1]


@pytest.mark.parametrize("case", TABLE["case"],
                         ids=[c["name"] for c in TABLE["case"]])
def test_the_gpu_contract(case, tmp_path, monkeypatch):
    target = _target(case, tmp_path, monkeypatch)
    if "given_gpus" in case:
        gpus_given(tmp_path, monkeypatch, case["given_gpus"])
    bundle = _describe(case, tmp_path, monkeypatch)
    kind = "bench" if "bench" in case else "run"
    r = jobset("prep", kind, "coarse", "--bundle", bundle,
               "--target", target, *case.get("prep", []))

    # 1 · ALLOWED OR REFUSED -- at prep, in its words
    if "refused" in case:
        assert r.exit_code != 0 and case["refused"] in r.output, r.output
        return
    assert r.exit_code == 0, r.output
    for words in case.get("said", []):
        assert words in r.output, r.output

    # 2 · WHAT IS PRODUCED -- the run's header, deck and run script
    deck = "*.py" if case.get("engine") == "pyscf" else "*.fdf"
    for key, pattern in (("header", "*.sbatch"), ("deck", deck),
                         ("run_sh", "*.run.sh")):
        if key in case or f"{key}_lacks" in case:
            _lines(case, key, _the_runs(bundle, pattern).read_text())
    # ...a benchmark's trials
    if kind == "bench":
        plan = next(bundle.rglob("bench/job-set.json"))
        asked = sorted({j["resources"]["gres"]
                        for j in json.loads(plan.read_text())["jobs"]})
        assert asked == sorted(case["bench_gres"]), asked
        for header in plan.parent.rglob("*.sbatch"):
            _lines({"h_lacks": case.get("bench_header_lacks", [])}, "h",
                   header.read_text())
    # ...and the `sbatch` line(s) launch shows -- one per shelf of a
    # benchmark -- or its refusal
    if "launch" in case:
        r = jobset("launch", kind, "coarse", "--bundle", bundle,
                   "--mode", "submit", "--dry-run", "--yes", *case["launch"])
        if "launch_refused" in case:
            assert r.exit_code != 0 and case["launch_refused"] in r.output, \
                r.output
        else:
            assert r.exit_code == 0, r.output
            sent = [ln for ln in r.output.splitlines() if "sbatch" in ln.split()]
            assert sent, r.output
            for ln in sent:
                _lines(case, "line", " ".join(sbatch_line(ln)))
        for words in case.get("listing", []):
            assert words in r.output, r.output

    # 3 · WHAT THE RUN SCRIPT DOES HERE -- its dry run, given the case's GPUs
    if "given_gpus" in case:
        script = _the_runs(bundle, "*.run.sh")
        script.write_text(strip_preamble_activation(script.read_text()))
        # The rank and thread counts this shell may carry are scrubbed, so
        # what resolves is the script's own chain.
        env = {k: v for k, v in os.environ.items()
               if k not in ("OMP_NUM_THREADS", "SLURM_CPUS_PER_TASK", "MB_NP",
                            "SLURM_NTASKS", "SLURM_JOB_ID", "PBS_NP",
                            "MOLBUILDER_MPI_NP", "MOLBUILDER_OMP_NUM_THREADS")}
        env.update(MB_LAUNCHED_BY="manual")
        done = subprocess.run(["bash", str(script), "--dry-run"],
                              cwd=script.parent, capture_output=True,
                              text=True, timeout=60, env=env)
        said = done.stdout + done.stderr
        assert done.returncode == 0, said[-2000:]
        for words in case["runs"]:
            assert words in said, said[-2000:]
