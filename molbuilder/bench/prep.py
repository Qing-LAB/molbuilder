"""``prep-bench`` -- the on-target driver that detects the machine and
formats the benchmark scripts for it (benchmark-workflow.md § 7.2).

Step 1 of the target side: run `resolve_environment` (probes), persist
``environment.json`` (§ 5.2), then hand the Environment to the matching
scheduler adapter's ``format_bench`` and write what it returns (the
topology-sized ``job-gpu-sweep.sh``).  The user never hand-edits a queue
name or a core count -- this is what makes the bundle portable (§ 2).

**Stdlib-only** (it imports the stdlib-only ``environment`` + ``adapters``
modules): meant to run on the target backend env, which has no
molbuilder/numpy.  ``run_prep_bench`` is the testable core; ``main`` is
the standalone ``argparse`` entry (the bundle can ship this file like
``mb_monitor.py``); the ``molbuilder bench prep`` CLI calls the same core.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from .adapters import get_adapter
from .environment import Environment, resolve_environment

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
                   now_iso: Optional[str] = None
                   ) -> Tuple[Environment, List[Path]]:
    """Detect the target, write ``environment.json``, and format the
    benchmark scripts into ``out_dir``.

    ``overrides`` is a flat dict of declared topology values (e.g.
    ``{"cores_per_socket": 24}``) that win over detection.  ``ks`` (e.g.
    ``[8, 16]``) overrides the swept ranks-per-GPU values.  Returns
    ``(Environment, [written paths])``.  ``.sh`` outputs are made
    executable.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = resolve_environment(overrides=overrides or None,
                              scheduler_override=scheduler_override,
                              now_iso=now_iso)

    written: List[Path] = []
    env_path = out / "environment.json"
    env_path.write_text(env.to_json() + "\n", encoding="utf-8")
    written.append(env_path)

    adapter = get_adapter(env)
    for name, content in adapter.format_bench(env, ks=ks).items():
        p = out / name
        p.write_text(content, encoding="utf-8")
        if name.endswith(".sh"):
            p.chmod(0o755)
        written.append(p)

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


def main(argv=None) -> int:
    """Standalone ``argparse`` entry (zero third-party deps)."""
    import argparse
    p = argparse.ArgumentParser(
        prog="prep-bench",
        description="Detect this machine (scheduler + topology) and format "
                    "the benchmark scripts for it.  Run in the bundle dir on "
                    "the target.")
    p.add_argument("--out", default=".",
                   help="output directory (default: current dir)")
    p.add_argument("--scheduler", choices=["slurm", "workstation"],
                   default=None, help="force the scheduler (else detected)")
    p.add_argument("--cores-per-socket", type=int, default=None,
                   help="override detected cores/socket")
    p.add_argument("--gpus-per-node", type=int, default=None,
                   help="override detected GPUs/node")
    p.add_argument("--gpu-type", default=None,
                   help="override detected GPU type (e.g. a100)")
    p.add_argument("--gpu-ks", default=None,
                   help="comma-separated ranks-per-GPU (K) values to sweep "
                        "(e.g. 8,16); default = cores/socket divisors")
    a = p.parse_args(argv)

    env, written = run_prep_bench(
        a.out,
        overrides=_overrides_from(a.cores_per_socket, a.gpus_per_node,
                                  a.gpu_type),
        scheduler_override=a.scheduler,
        ks=_parse_ks(a.gpu_ks),
        now_iso=utc_now_iso())
    print(_summary(env, written))
    return 0


__all__ = ["run_prep_bench", "utc_now_iso", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
