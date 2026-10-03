"""What a stage continues from -- which run, and what that run was
(`execution/job-system.md` § 5.4, plan W37).

**Module:** L2 (jobset).  ONE answer, asked by `prep` before it writes a stage
-- the default, or what a run named by ``--from`` is -- and by `status` for the
stage it names next, so the two never say different things about the same run.
`status` answered by a rule of its own until the W37 review found it telling a
person a stage set to start clean would continue, and naming nothing on the
flat layout, where the next stage continues all the same.  Reads the folder; writes
nothing; raises nothing -- a refusal is an answer, which `prep` turns into its
own error and `status` prints.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional, Tuple


@dataclasses.dataclass(frozen=True)
class Continuation:
    """WHICH RUN A STAGE CONTINUES FROM, and what that run was -- printed by
    both prep doors and recorded in the decision ledger."""
    #: The stage the run belongs to.
    stage: str
    #: Its attempt, from the calculation folder (``01_coarse/run-0``) --
    #: ``None`` on the flat layout, where the run's files lie in the folder.
    source: Optional[str]
    #: The stage before it, newest (the default) -- or named by ``--from``.
    by_default: bool
    #: The conclusion marker's line, or ``None``: the run has not concluded.
    concluded: Optional[str]
    #: Its run state, `run_status`'s word (finished, failed, running, ...).
    state: Optional[str]
    #: The relaxation's verdict, ``None`` when the run relaxed nothing.
    converged: Optional[bool]

    def where(self) -> str:
        """The run, as a person reads it: its attempt, or -- on the flat
        layout -- the stage's latest run in the folder."""
        return (self.source if self.source is not None else
                f"{self.stage}'s latest run, whose files lie in this folder")

    def line(self, copied=()) -> str:
        """``continues from 01_coarse/run-0 (the stage before it; concluded
        rc=0 at ...; converged): copied H2.XV, ...`` -- what both doors
        print."""
        facts = ["the stage before it" if self.by_default else "named",
                 _what_it_is(self.concluded, self.state)]
        if self.converged is not None:
            facts.append("converged" if self.converged else
                         "NOT converged -- taken as it stands")
        return (f"continues from {self.where()} ({'; '.join(facts)})"
                + (f": copied {', '.join(copied)}" if copied else ""))

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    def ledger_facts(self) -> dict:
        """The decision-ledger line's facts -- the run's stage as
        ``from_stage``, beside the stage being prepped."""
        d = self.as_dict()
        d["from_stage"] = d.pop("stage")
        return d


def _what_it_is(concluded: Optional[str], state: Optional[str]) -> str:
    """A run's conclusion, in words: its marker's line when it concluded --
    FAILED beside it when its run failed -- else what its state says."""
    if concluded is not None:
        return ("concluded " + concluded).strip() + (
            ", FAILED" if state == "failed" else "")
    return {"pending": "NOT launched yet",
            "queued": "NOT concluded -- queued",
            "running": "NOT concluded -- still running"}.get(
        state, "NOT concluded -- stopped without its conclusion marker")


def read_run(base: Path, task, stage: str, attempt: Path,
             container: Path, *, verdict: bool = True
             ) -> Tuple[Optional[str], Optional[str], Optional[bool]]:
    """``(concluded, state, converged)`` of one run of ``stage`` -- its
    conclusion marker (`materialize.attempt_concluded`; an empty marker is a
    conclusion, so ``""``), its state through the one door
    (`parse.dirs.run_status`) and its relaxation's verdict (`parse.contract`):
    an attempt's own run on the hierarchy; on the flat layout THIS stage's
    files among the folder's -- its progress log, the one both engines write
    and the only one a PySCF run's stdout cannot stand in for, then its
    engine output.  ``verdict=False`` skips the relaxation's parse -- a list
    of runs needs each one's conclusion, not a parse of each output."""
    from ..parse import detect
    from ..parse.contract import relaxation_of, relaxation_of_output
    from ..parse.dirs import run_status
    from ..paths import Shape
    from ..runfiles import find as rf_find, stem as rf_stem
    from .materialize import conclusion_line, read_run_launch, stage_stdout
    from .prep import token_for
    token = token_for(task, stage)
    stem = rf_stem(task.label, token)
    sh = Shape.named(task.shape)
    concluded = conclusion_line(attempt, stem)
    try:
        launch = (read_run_launch(attempt) if sh.keeps_attempts_as_directories
                  else read_run_launch(container, basename=stem))
        state = run_status(attempt, sh.stage_glob(token, task.label),
                           launch=launch).state
    except Exception:                                    # noqa: BLE001
        state = None
    rec = None
    if not verdict:
        pass
    elif sh.keeps_attempts_as_directories:
        try:
            rec = relaxation_of(attempt)
        except Exception:                                # noqa: BLE001
            rec = None
    else:
        own = [p for p, _r in rf_find(attempt, task.label,
                                      role=".molwatch.log", stage=token)]
        out = stage_stdout(attempt, task.label, token, str(task.engine))
        for path in own + ([out] if out is not None else []):
            try:
                rec = relaxation_of_output(path, detect(path).parse(str(path)),
                                           engine=str(task.engine))
            except Exception:                            # noqa: BLE001
                rec = None
            if rec is not None:
                break
    return concluded, state, (None if rec is None else bool(rec["converged"]))


def continuation_answer(base, task, stage: str, *, from_attempt=None,
                        cold: bool = False, verdict: bool = True
                        ) -> Tuple[Optional[Continuation], Optional[str]]:
    """``(continuation, refusal)`` for ``stage`` (`job-system.md` § 5.4).

    First, what cannot be taken at all is refused, before prep writes
    anything (:func:`_cannot_be_named`): a run that is not one of this
    calculation's, ``--from`` with ``--cold``, either on the flat layout or
    a bias scan.  **Named** (``--from``): taken as said, for any kind -- what
    the run was is read and reported; `prepare_attempt` refuses only what
    cannot be done (no restart files in it).  **By default**, for a
    continuing stage of an independent ladder (`_stage_before`): the NEWEST
    attempt of the enabled stage before it, which must have concluded and not
    failed -- an older one never stands in, because a stage re-launched to
    tighten is the run the person means.  Otherwise the refusal, worded by
    what the run's state says and naming the commands that work on this
    layout.  ``(None, None)`` for ``--cold``, a linked kind's default, the
    first stage, a stage that starts clean, and one the description disables
    (prepped by name, it has no stage before it).  ``verdict=False`` leaves
    the relaxation unread -- for a reader that never prints it."""
    from ..identity import command_stage
    base = Path(base)
    refused = _cannot_be_named(base, task, stage, from_attempt, cold)
    if refused:
        return None, refused
    if cold:
        return None, None
    if from_attempt:
        # A RUN NAMED is taken as said, for any kind -- what it was is
        # reported and recorded, a linked stage's as much as any (W52: a
        # linked stage's `--from` was copied and neither said nor ledgered).
        attempt = base / from_attempt
        named = command_stage(Path(from_attempt).parts[0])
        concluded, state, converged = read_run(base, task, named, attempt,
                                               attempt.parent, verdict=verdict)
        return Continuation(stage=named, source=str(Path(from_attempt)),
                            by_default=False, concluded=concluded, state=state,
                            converged=converged), None
    try:
        prev = _stage_before(base, task, stage)
        if prev is None:
            return None, None
        return _by_default(base, task, stage, prev, verdict=verdict)
    except Exception as exc:                             # noqa: BLE001
        # RAISES NOTHING, as this module promises: a template prep would
        # refuse is a refusal here too, said -- `status` printed a traceback
        # and the Results tab lost its whole ladder over it (W52).
        return None, (f"what `{stage}` continues from cannot be read: "
                      f"{exc}")


def _cannot_be_named(base: Path, task, stage: str, from_attempt,
                     cold: bool) -> Optional[str]:
    """Why ``--from`` / ``--cold`` cannot be taken here -- said BEFORE prep
    writes anything (W52: each was refused only after the five steps had
    rendered, one of them after an earlier carry had been undone) -- or
    ``None``.  Both doors ask through this: the browser's prep route refused
    a path out of the calculation, and `--from` with `--cold`, on its own,
    while the terminal took both."""
    if not (from_attempt or cold):
        return None
    if from_attempt and cold:
        return ("--from and --cold are two answers to one question -- name "
                "the run it continues from, or start it from the "
                "calculation's structure.")
    from pathlib import PurePosixPath
    from ..identity import command_stage
    from ..paths import Shape
    from ..transport.stages import scan_points
    from .materialize import FLAT_HAS_NO_ATTEMPTS
    if not Shape.named(task.shape).keeps_attempts_as_directories:
        return FLAT_HAS_NO_ATTEMPTS
    if scan_points(task, stage):
        return ("--from / --cold name ONE attempt, and a bias scan keeps one "
                "per point -- per-point continuation is not named yet "
                "(engines/transport.md).")
    if from_attempt:
        p = PurePosixPath(str(from_attempt))
        if p.is_absolute() or ".." in p.parts:
            return (f"--from names a run of this calculation, by its folder "
                    f"(01_coarse/run-0): {from_attempt!r}")
        if not (base / from_attempt).is_dir():
            return (f"--from {from_attempt!r}: no such attempt in this "
                    f"calculation.  Name an attempt directory that has "
                    f"already run, e.g. '01_coarse/run-0'.")
        head = p.parts[0]
        try:
            named = command_stage(head)
        except ValueError:
            named = None
        if named not in {s.name for s in task.stages}:
            return (f"--from {from_attempt!r}: {head!r} is not a stage "
                    f"folder of this calculation -- name a run as "
                    f"<NN>_<stage>/run-<n>.")
    return None


def _by_default(base: Path, task, stage: str, prev: str, *, verdict: bool
                ) -> Tuple[Optional[Continuation], Optional[str]]:
    """The default for ``stage``, whose stage before it is ``prev`` -- the
    newest attempt, or the refusal (`continuation_answer`)."""
    from ..paths import Shape
    from ..paths import attempt_dir as _adir
    from ..runfiles import compose as rf_compose
    from .materialize import attempts as _attempts, latest_attempt
    from .prep import _engine_seam, token_for
    sh = Shape.named(task.shape)
    seam = _engine_seam(str(task.engine))
    token = token_for(task, prev)
    sd = sh.stage_dir(token)
    container = base if sd == "." else base / sd
    flat = not sh.keeps_attempts_as_directories

    # THE COMMANDS, from the one composer (`commands`): each names the
    # calculation and states the mode where its config sets none.
    from .commands import block, command, launch_lines, run_first
    run_prev = block(run_first(prev, base=base))
    launch_prev = block(launch_lines("run", prev, base=base))
    # THE WAY OUT THAT WORKS HERE (the W37 review's first finding): `--cold`
    # names an attempt-less start, which the flat layout -- one folder, no
    # attempts -- refuses; there a stage starts clean by its run card.
    clean = (f"or start `{stage}` clean: set its run card's `restart` to "
             f"`clean` (Task setup, or task.json)" if flat else
             f"or start `{stage}` from the calculation's structure --\n"
             + block([command("prep", "run", stage, base=base,
                              flags=("--cold",))]))
    rule = "(job-system.md § 5.4)"

    if flat:
        if not (base / rf_compose(task.label, seam.suffix, token)).is_file():
            return None, (f"`{stage}` continues from the stage before it, "
                          f"`{prev}`, which has not run yet.  Run it "
                          f"first --\n{run_prev}\n{clean}\n{rule}")
        attempt, source = base, None
    else:
        latest = latest_attempt(container) if container.is_dir() else None
        if latest is None:
            return None, (f"`{stage}` continues from the stage before it, "
                          f"`{prev}`, which has not run yet.  Run it "
                          f"first --\n{run_prev}\n{clean}\n{rule}")
        attempt, source = latest, str(latest.relative_to(base))
    concluded, state, converged = read_run(base, task, prev, attempt,
                                           container, verdict=verdict)
    if concluded is not None and state != "failed":
        return Continuation(stage=prev, source=source, by_default=True,
                            concluded=concluded, state=state,
                            converged=converged), None

    # REFUSED -- worded by what the run's state says, with a command for
    # each way on (§ 5.3: what molbuilder prints, you can type).
    what = (f"the newest attempt of `{prev}`, {source}," if source
            else f"`{prev}`'s latest run,")
    if concluded is not None:
        # A FAILED RUN IS LAUNCHED AGAIN -- a prepped stage is not prepped
        # again (`job-system.md` § 5.0); its re-prep was offered here until
        # 2026-10-02.
        why, first = (f"which failed ({concluded})",
                      f"Launch it again --\n{launch_prev}")
    elif state == "pending":
        why, first = "which has not been launched", f"Launch it --\n{launch_prev}"
    elif state in ("queued", "running"):
        why, first = f"which is {state}", "Let it finish"
    else:
        why = ("which stopped without its conclusion marker -- killed, or "
               "out of time (project-layout.md § 1.6)")
        first = f"Launch it again --\n{launch_prev}"
    other = None
    if not flat:
        # AN EARLIER RUN THAT CAN STAND IN WHEN ASKED FOR: the newest that
        # concluded and did not fail, typed out -- never a placeholder.  Its
        # conclusion is the question, not its relaxation.
        for n in reversed(_attempts(container)):
            a = _adir(container, n)
            if a == attempt:
                continue
            c, s, _v = read_run(base, task, prev, a, container, verdict=False)
            if c is not None and s != "failed":
                other = str(a.relative_to(base))
                break
    alt = (f"or continue from an earlier run of `{prev}` that concluded --\n"
           + block([command("prep", "run", stage, base=base,
                            flags=("--from", other))]) + "\n"
           if other else "")
    return None, (f"`{stage}` continues from {what} {why}.  {first}\n"
                  f"{alt}{clean}\n{rule}")


def _independent(task) -> bool:
    """An INDEPENDENT ladder, whose stages continue one from another: a kind
    without rung roles (`template.KIND_ROLES`).  A linked stage's input is
    prep's own (`prep._vibration_stage_geometry`,
    `prep.gather_transport_inputs`)."""
    from ..template import KIND_ROLES
    return (getattr(task, "calculation", None)
            or "optimization") not in KIND_ROLES


def _stage_before(base, task, stage: str) -> Optional[str]:
    """The enabled stage ``stage`` continues from by default -- the one
    before it in the ladder, resolved as `prep` resolves it (template ⊕
    stage overrides ⊕ the run card, so a card's ``restart: clean`` is read)
    -- or ``None``: a linked kind, the first stage, one that starts clean,
    one the description disables."""
    from ..identity import continues
    from ..resolve import resolved_ladder
    from ..template import template_path
    from .prep import _engine_seam
    if not _independent(task):
        return None
    tpl = template_path(Path(base), task.label)
    if not tpl.is_file():
        return None
    ladder = resolved_ladder(tpl.read_text(encoding="utf-8"), task,
                             _engine_seam(str(task.engine)).config_cls)
    names = [n for n, _c in ladder]
    if stage not in names:
        return None
    i = names.index(stage)
    if i == 0 or not continues(ladder[i][1]):
        return None
    return names[i - 1]


def continue_from_choices(base, task, stage: str) -> Optional[dict]:
    """What Task setup's **Continue from** offers ``stage`` -- the default
    and, when prep would refuse it, why; every run of the stage before it
    with what it was; whether the structure (``--cold``) is a choice here --
    or ``None`` when the stage continues from nothing by default
    (`job-system.md` § 5.4, `web/task-setup.md` § 11).  The answer prep acts
    on, served before it does -- without the relaxation's verdict, which the
    preview reads (`continuation_answer`): a folder is answered every time it
    is opened, and parsing each stage's output there is the cost of a
    preview nobody asked for."""
    from ..paths import Shape
    from ..paths import attempt_dir as _adir
    from .materialize import attempts as _attempts
    from .prep import token_for
    base = Path(base)
    prev = _stage_before(base, task, stage)
    if prev is None:
        return None
    got, refused = _by_default(base, task, stage, prev, verdict=False)
    sh = Shape.named(task.shape)
    runs = []
    if sh.keeps_attempts_as_directories:
        container = base / sh.stage_dir(token_for(task, prev))
        for n in (reversed(_attempts(container)) if container.is_dir()
                  else ()):
            a = _adir(container, n)
            c, s, _v = read_run(base, task, prev, a, container,
                                verdict=False)
            runs.append({"source": str(a.relative_to(base)),
                         "what": _what_it_is(c, s)})
    return {"from_stage": prev,
            "default": (dict(got.as_dict(), line=got.line())
                        if got is not None else None),
            "refused": refused, "runs": runs,
            "cold": sh.keeps_attempts_as_directories}
