"""Prep engine — render the per-job launchers and lay out the tree
(docs/execution/job-system.md, step 5).

This is the step BETWEEN the pure ``materialize`` (data symlinks) and the
``submit`` launch.  It mirrors what the benchmark already does in
``bench/generate.py`` — render the wrappers **once per distinct script, in
the bundle root, from the REAL file** (so ``write_run_wrapper``'s
``Path.resolve()`` is a no-op and the ``.run.sh`` / ``.sbatch`` land where
intended), then symlink those wrappers into each job's ``point-<name>/`` dir
alongside the data symlinks.  Per-job resource *variation* is NOT baked here
— it is applied by ``submit`` as scheduler CLI flags over the shared
wrapper, exactly as the bench launch line does.  That is what lets one
rendered ``.sbatch`` serve every point of a sweep.

Why render-in-root-then-symlink (not render-in-each-dir): the materialized
script is a SYMLINK back to the bundle root; ``write_run_wrapper`` resolves
symlinks and would write the wrapper next to the *resolved* target.
Rendering from the real bundle-root file is the only placement that is both
correct and consistent with the benchmark (shared wrapper, CLI-flag
variation), so the two job-set kinds stay one mechanism.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from .materialize import job_dir_names, shape_of, materialize, relink
from .model import JobSet


class PrepError(Exception):
    """A JobSet could not be prepped (invalid set, or a script missing from
    the bundle root)."""


def prep_jobset(jobset: JobSet, base_dir, *, env: str = None,
                emit_sbatch: bool = True) -> List[Path]:
    """Render launchers + lay out the per-job tree under ``base_dir``.

    Steps, in order:
      1. render each **distinct** ``job.script``'s ``.run.sh`` (and
         ``.sbatch`` when ``emit_sbatch`` and a scheduler is configured) in
         the bundle root, from the real file — reusing
         ``runwrap.write_run_wrapper`` (no reinvention).  The header carries
         the first-seen job's resources as defaults; ``submit`` overrides
         per job via CLI flags, so the defaults never decide the answer.
      2. ``materialize`` — data symlinks (shared package, script, carry).
      3. symlink each job's wrappers (+ shipped ``mb_monitor.py``) into its
         ``point-<name>/`` dir, so ``submit`` can ``sbatch``/``bash`` them
         there.

    Returns the per-job directories.  Raises :class:`PrepError` on an
    invalid JobSet or a script that isn't in the bundle root.
    """
    from ..runwrap import write_run_wrapper

    errs = jobset.validate()
    if errs:
        raise PrepError(
            "cannot prep an invalid JobSet:\n  - " + "\n  - ".join(errs))
    base = Path(base_dir).resolve()
    if not base.is_dir():
        raise PrepError(f"bundle root not found: {base}")

    # ---- 1. render wrappers once per distinct script (in the root) ------ #
    rendered: set = set()
    for job in jobset.jobs:
        if job.script in rendered:
            continue
        script_path = base / job.script
        if not script_path.is_file():
            raise PrepError(
                f"job {job.name!r}: script {job.script!r} not in bundle root "
                f"{base} (render the inputs before prep).")
        r = job.resources
        write_run_wrapper(
            script_path,
            env=env,
            mpi_np=r.mpi_np,
            cpus_per_task=r.cpus_per_task,
            time=r.time,
            gres=r.gres,
            mem=r.mem,
            exclusive=r.exclusive,
            # The warm-retry budget, which becomes no sbatch flag: the
            # wrapper bakes it into its own retry loop at install time
            # (running-a-job.md § 3.5).  This line is the second half of the
            # road job-contracts.md § 6.2 describes -- without it the field
            # was carried the whole way here and then dropped, which is why
            # `job-system.md § 4.1` recorded the SIESTA ladder as never
            # having implemented `continue` (2026-08-07, P2 unit 3).
            continue_retries=r.continue_retries,
            # localize carried restart files at run time so this job's writes
            # never clobber the producer's dir (job-system.md § 5.2).
            carry_in=[c.pattern for c in job.carry],
            emit_sbatch=emit_sbatch,
        )
        rendered.add(job.script)

    # ---- 2. data symlinks ---------------------------------------------- #
    dirs = materialize(jobset, base)

    # ---- 3. link the rendered wrappers (+ monitor) into each job dir ---- #
    has_monitor = (base / "mb_monitor.py").exists()
    # NOT named ``dirs``: step 2 above binds that to materialize's list of
    # created Paths, which is this function's return value.
    dir_of = job_dir_names(jobset, shape_of(jobset, base_dir))
    for job in jobset.jobs:
        d = base / dir_of[job.name]
        stem = Path(job.script).stem
        for wrapper in (f"{stem}.run.sh", f"{stem}.sbatch"):
            if (base / wrapper).exists():
                relink(d, f"../{wrapper}", wrapper)
        if has_monitor:
            relink(d, "../mb_monitor.py", "mb_monitor.py")

    # ---- 4. emit STAGE-PLAN.md (§ 5 D3; mirrors bench's BENCH-PLAN.md) --- #
    # The reviewable plan lands in the bundle at prep, not just on the
    # `jobset plan` command's stdout.
    from .plan import render_plan
    (base / "STAGE-PLAN.md").write_text(render_plan(jobset) + "\n",
                                        encoding="utf-8")
    return dirs


__all__ = ["prep_jobset", "PrepError"]
