"""``prep-bench`` -- the on-target driver that detects the machine and
formats the benchmark scripts for it (benchmark-workflow.md § 7.2).

Step 1 of the target side: run `resolve_environment` (probes), persist
``environment.json`` (§ 5.2), then hand the Environment to the matching
scheduler adapter's ``format_bench`` and write what it returns (the
topology-sized ``job-gpu-sweep.sh``).  The user never hand-edits a queue
name or a core count -- this is what makes the bundle portable (§ 2).

Import-light (it pulls only the stdlib ``environment`` + ``adapters``
modules).  ``run_prep_bench`` is the testable core; the on-target
``prep-bench`` shim bootstraps the molbuilder host env and calls
``molbuilder bench prep`` (job-execution.md § 8.3, § 3.4 contract), which
calls this same core -- there is no shipped standalone copy (the old stdlib
``mbbench/`` was retired; benchmark-workflow.md § 30).
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from .adapters import resolve_launch_adapter
from ..environment import Environment, resolve_environment

# Topology override keys accepted from the CLI / caller (flow to
# detect_topology, where they win over detection).
_OVERRIDE_KEYS = ("cores_per_socket", "gpus_per_node", "gpu_type", "sockets",
                  "threads_per_core", "numa_per_socket", "mem_total_gb")


def utc_now_iso() -> str:
    """UTC timestamp ``YYYY-MM-DDThh:mm:ssZ`` for ``detected_at``."""
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def run_prep_bench(out_dir,
                   *,
                   overrides: Optional[dict] = None,
                   scheduler_override: Optional[str] = None,
                   ks: Optional[list] = None,
                   cs: Optional[list] = None,
                   mode: Optional[str] = None,
                   submit_via: str = "slurm",
                   now_iso: Optional[str] = None
                   ) -> Tuple[Environment, List[Path]]:
    """Detect the target, write ``environment.json``, and format the
    benchmark scripts into ``out_dir``.

    ``overrides`` is a flat dict of declared topology values (e.g.
    ``{"cores_per_socket": 24}``) that win over detection.  ``ks`` (e.g.
    ``[8, 16]``) overrides the swept ranks-per-GPU values; ``cs`` (e.g.
    ``[1, 8, 16]``) overrides the swept cores-per-rank values (§ 8.12).

    ``mode`` ("direct" | "submit") + ``submit_via`` select the LAUNCH adapter
    (job-execution.md § 8.13), so the baked sweep uses ``sbatch`` per point
    under ``submit`` and direct ``bash`` under ``direct`` -- independent of the
    detected scheduler.  ``mode=None`` derives it from detection.

    Returns ``(Environment, [written paths])``.  ``.sh`` outputs are made
    executable.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Step 1 of the five (`project-layout.md` § 2.3.1), and it is the SAME
    # step `jobset prep` now runs -- see `jobset/prep.py::resolve_target`.
    # A benchmark prep passes topology overrides and a pinned timestamp
    # because it is prep SPECIALISED (§ 2.3.1a), not because it is a
    # different act.
    env = resolve_environment(overrides=overrides or None,
                              scheduler_override=scheduler_override,
                              now_iso=now_iso)

    written: List[Path] = []
    env_path = out / "environment.json"
    env_path.write_text(env.to_json() + "\n", encoding="utf-8")
    written.append(env_path)

    adapter, _ = resolve_launch_adapter(env, mode=mode, submit_via=submit_via)
    for name, content in adapter.format_bench(env, ks=ks, cs=cs).items():
        p = out / name
        p.write_text(content, encoding="utf-8")
        if name.endswith(".sh"):
            p.chmod(0o755)
        written.append(p)

    # Also emit the sweep as a first-class JobSet (job-set.json), so the bench
    # bundle is describable/runnable by the engine-agnostic framework
    # (`molbuilder jobset plan/prep/submit`) -- the SAME (G,K,c) grid as the
    # bash sweep above (both iterate adapters.sweep_grid) and the SAME
    # point-G<g>K<k>C<c>/ dirs summarize reads.  Bench = a jobset producer
    # (job-system.md § 4.2).
    from .to_jobset import sweep_to_jobset
    pseudos = sorted(p.name for ext in ("*.psml", "*.psf", "*.vps")
                     for p in out.glob(ext))
    monitor = ["mb_monitor.py"] if (out / "mb_monitor.py").is_file() else []
    js = sweep_to_jobset(adapter, env, ks=ks, cs=cs,
                         shared=pseudos + monitor)
    written.append(js.write(out / "job-set.json"))

    return env, written


def _readiness_lines(env: Environment) -> List[str]:
    """Surface the EXISTING ``molbuilder envs`` readiness checks
    (job-execution.md § 3.4, "Job B"): prep *points at* them so the
    scientist runs them before spending a queue slot.  Point only -- never
    auto-run, never auto-install (assistant, not nanny -- design.md Stance).

    ``doctor`` always shows (env presence + each recipe's verify command);
    the GPU-env ``validate`` line shows only when GPUs were detected, since
    that is where the CUDA-stack / ELPA-GPU-codepath probe matters."""
    lines = [
        "  next: verify this target is ready before you submit "
        "(prep points; you run):",
        "    molbuilder envs doctor"
        "                          # envs present + each recipe's verify cmd",
    ]
    if env.topology.gpus_per_node:
        lines.append(
            "    molbuilder envs validate molbuilder-siesta-gpu"
            "  # CUDA stack + ELPA-GPU codepath")
    return lines


def _summary(env: Environment, written: List[Path]) -> str:
    t = env.topology
    lines = [
        "prep-bench: detected target",
        f"  scheduler : {env.scheduler}  (source: {env.source.get('scheduler')})",
        f"  topology  : sockets={t.sockets} cores/socket={t.cores_per_socket} "
        f"threads/core={t.threads_per_core} gpus={t.gpus_per_node} "
        f"type={t.gpu_type}  (source: {env.source.get('topology')})",
        f"  site      : partition={env.site.partition}  "
        f"(source: {env.source.get('site')})",
        "  wrote:",
    ]
    lines += [f"    {p}" for p in written]
    lines += _readiness_lines(env)
    return "\n".join(lines)


def _overrides_from(cores_per_socket=None, gpus_per_node=None, gpu_type=None):
    d = {"cores_per_socket": cores_per_socket, "gpus_per_node": gpus_per_node,
         "gpu_type": gpu_type}
    return {k: v for k, v in d.items() if v is not None} or None


def _parse_ks(spec):
    """``"8,16"`` -> ``[8, 16]``; ``None``/empty -> ``None`` (use default)."""
    if not spec:
        return None
    return [int(x) for x in str(spec).split(",") if x.strip()]


__all__ = ["run_prep_bench", "utc_now_iso"]
