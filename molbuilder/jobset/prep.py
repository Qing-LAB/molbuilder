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

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from .materialize import job_dir_names, shape_of, materialize, relink
from .model import JobSet


class PrepError(Exception):
    """A JobSet could not be prepped (invalid set, or a script missing from
    the bundle root)."""


# --------------------------------------------------------------------- #
#  Does the deck agree with the launch?  (P6 unit 2 · project-layout      #
#  § 2.3.1 — "step 3 cannot precede step 1")                             #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class LaunchAgreement:
    """Whether a deck was rendered for the launch it is about to get.

    ``verdict`` is one of:

    | ``"silent"`` | the deck makes no claim about its launch, so there is nothing to disagree with — and nothing to refuse |
    | ``"agrees"`` | the rank count it was rendered for is the one it will get |
    | ``"differs"`` | the two are different numbers, or one is `auto` and the other is not |

    ``rendered_for`` is what the deck's BENCH-MARKS block records (an int, or
    the string ``"auto"``); ``launching_at`` is the job's ``mpi_np`` (an int,
    or ``None`` meaning *let the wrapper decide*).
    """
    verdict:      str
    rendered_for: Optional[Union[int, str]] = None
    launching_at: Optional[int] = None

    @staticmethod
    def _fmt(v) -> str:
        return "auto" if v in ("auto", None) else str(v)

    @property
    def rendered_text(self) -> str:
        return self._fmt(self.rendered_for)

    @property
    def launch_text(self) -> str:
        return self._fmt(self.launching_at)


def launch_agreement(job_dir, job) -> LaunchAgreement:
    """Read the deck's launch claim and compare it with this job's.

    `project-layout.md § 2.3.1` states the five steps and says the order is
    **forced**: *"Step 3 cannot precede step 1, because a deck carries values
    that depend on how it will be launched — a block size derived from the rank
    count … A parameter that depends on the launch cannot be decided before the
    launch is known."*

    Today step 3 happens at `molbuilder fdf`, on whatever machine typed it, and
    the rank count is resolved hours later by the wrapper. **The two halves of
    one ordered sequence run in different places, and nothing carried the
    first's answer to the third.** On 2026-08-10 that produced a deck rendered
    with no rank count — so ``BlockSize`` from the size-only branch — launched
    at ``-np 14``, and SIESTA refused at startup with *"You have too many
    processors for the system size"*.

    P4 unit 5 put the launch quantity **into** the deck, which is why that
    failure was diagnosable at all. **Recording is not agreeing**: this is the
    comparison, and it lives here rather than in `submit` because `prep` is the
    step that owns the deck and the wrapper (§ 2.3), and because a person is
    owed the answer at the moment they are still deciding — not at the moment
    they are committing cluster time.
    """
    deck = Path(job_dir) / os.path.basename(job.script)
    if not deck.is_file():
        return LaunchAgreement("silent")
    from ..parse.scripts.bench_marks import _extract_bench_marks_dict
    marks = _extract_bench_marks_dict(deck.read_text(encoding="utf-8",
                                                     errors="replace"))
    if not marks or "mpi_np" not in marks:
        # A deck with no BENCH-MARKS block says nothing about its launch, so
        # there is nothing to disagree with.  The check is an agreement between
        # two statements, never a demand that every deck make one.
        return LaunchAgreement("silent")
    rendered_for = marks["mpi_np"]                 # an int, or the str "auto"
    launching_at = job.resources.mpi_np            # an int, or None == auto
    agree = ((rendered_for == "auto" and launching_at is None)
             or rendered_for == launching_at)
    return LaunchAgreement("agrees" if agree else "differs",
                           rendered_for, launching_at)


def check_launch_matches_deck(job_dir, job) -> None:
    """Refuse a launch the deck was not rendered for (P6 unit 2).

    Three outcomes, and the middle one is the live defect:

    * deck ``auto`` + launch ``auto`` — both defer to the wrapper. Fine.
    * deck ``auto`` + launch ``N`` — the deck's launch-derived values were
      computed with **no** rank count, and now one is being imposed. Refused.
    * deck ``N`` + launch ``M`` — refused, with both numbers named.

    This is the last honest moment, not the first: :func:`launch_agreement`
    answers the same question at `prep`, where it is still cheap to change
    your mind. Both call one comparison, so the warning and the refusal cannot
    come to different conclusions.
    """
    from .submit import SubmitError
    a = launch_agreement(job_dir, job)
    if a.verdict != "differs":
        return
    deck = os.path.basename(job.script)
    raise SubmitError(
        f"job {job.name!r}: this deck was rendered for mpi_np "
        f"{a.rendered_text}, and you are launching it at "
        f"{a.launch_text}.\n"
        f"  {deck} derives values from the rank count -- BlockSize above "
        f"all -- so a deck rendered for one launch is wrong for another "
        f"(project-layout.md § 2.3.1: a parameter that depends on the launch "
        f"cannot be decided before the launch is known).\n"
        f"  Re-render the deck for this launch, or launch it at "
        f"{a.rendered_text}.  The deck records what it assumed in its "
        f"BENCH-MARKS block, which is what made this checkable.")


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
        if d.resolve() == base.resolve():
            # FLAT: the wrappers step 1 just rendered, and the monitor, are
            # already in the directory this job runs in.  Linking here would
            # unlink the real files and point at the bundle's PARENT -- the
            # same destruction `materialize` guards against, one step later.
            continue
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


__all__ = ["prep_jobset", "PrepError",
           "launch_agreement", "check_launch_matches_deck",
           "LaunchAgreement"]
