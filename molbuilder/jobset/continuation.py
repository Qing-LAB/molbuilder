"""What a stage continues from -- which run, and what that run was
(`execution/job-system.md` § 5.4, plan W37): the stage before it in an
independent ladder, and the `relax` run a vibration's force-constant stage
builds on (`engines/vibration.md` § 5.2a, W38 F9).

**Module:** L2 (jobset).  ONE answer, asked by `prep` before it writes a stage
-- the default, or what a run named by ``--from`` is -- and by `status` for the
stage it names next, so the two never say different things about the same run.
`status` answered by a rule of its own until the W37 review found it telling a
person a stage set to start clean would continue, and naming nothing on the
flat layout, where the next stage continues all the same.  Reads the folder; writes
nothing; raises nothing the person can fix -- a refusal is an answer, which
`prep` turns into its own error and `status` prints; a bug raises.
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
    #: The line its conclusion said (`runrecord.ending`), or ``None``: the
    #: run has not ended on its own.
    concluded: Optional[str]
    #: Its run state, `run_status`'s word (finished, failed, running, ...).
    state: Optional[str]
    #: The relaxation's verdict, ``None`` when the run relaxed nothing.
    converged: Optional[bool]
    #: A force-constant stage building on its `relax` (`engines/vibration.md`
    #: § 5.2a), not a stage continuing from the one before it.
    linked: bool = False
    #: The files it carries into the attempt, by the pair's rule
    #: (`model.warm_carry`) of what the run holds -- counted where the plan's
    #: row is (`prep`), ``None`` until then.  On the flat layout nothing is
    #: copied: the files lie in the folder.
    carries: Optional[Tuple[str, ...]] = None

    def where(self) -> str:
        """The run, as a person reads it: its attempt, or -- on the flat
        layout -- the stage's latest run in the folder."""
        return (self.source if self.source is not None else
                f"{self.stage}'s latest run, whose files lie in this folder")

    def line(self, copied=()) -> str:
        """``continues from 01_coarse/run-0 (the stage before it; concluded
        rc=0 at ...; converged): copied H2.XV, ...`` -- what both doors
        print."""
        facts = [("named" if not self.by_default else
                  "the relaxation it builds on" if self.linked else
                  "the stage before it"),
                 _what_it_is(self.concluded, self.state)]
        if self.converged is not None:
            # NOT CONVERGED, said for what it costs: a force-constant stage
            # measured off a stationary point reports imaginary frequencies
            # (W38 F9).
            facts.append("converged" if self.converged else
                         "NOT converged -- expect imaginary frequencies"
                         if self.linked else
                         "NOT converged -- taken as it stands")
        return (f"continues from {self.where()} ({'; '.join(facts)})"
                + (f": copied {', '.join(copied)}" if copied else ""))

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    def ledger_facts(self) -> dict:
        """The decision-ledger line's facts -- the run's stage as
        ``from_stage``, beside the stage being prepped.  What came across is
        the line's ``copied``, read off the attempt opened."""
        d = self.as_dict()
        d["from_stage"] = d.pop("stage")
        d.pop("carries", None)
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


def usable(where, basename: str) -> bool:
    """THE RULE FOR A RUN TO BUILD ON (`execution/architecture.md` § 3.2,
    `job-system.md` § 5.4): it ended on its own with exit code 0
    (`runrecord.ending`).  The default of every hand-over -- a continuing
    stage, the frequency stage's geometry, a transport rung's inputs; a run
    named with ``--from`` is taken as said, and a structure stated relaxed
    needs no run, so neither asks it.  *(A frequency stage and a transport
    rung took a run that concluded with an error until 2026-10-03, while an
    independent stage refused it; user: "yes, for #1".)*"""
    from ..runrecord import ending
    return ending(where, basename).ok


def read_run(base: Path, task, stage: str, attempt: Path,
             container: Path, *, verdict: bool = True
             ) -> Tuple[Optional[str], Optional[str], Optional[bool]]:
    """``(concluded, state, converged)`` of one run of ``stage`` -- the
    line its conclusion said, through the one door (`runrecord.ending`;
    ``None`` when it has not ended on its own), its state through the one
    door (`parse.dirs.run_status`) and its relaxation's verdict
    (`parse.contract.relaxation_of`), read from the run's own files in
    either shape, the run door's (`runs.run_of`): its progress log --
    where a PySCF run writes its steps, which its stdout cannot stand in
    for -- then its engine output at the run's index.  ``verdict=False``
    skips the relaxation's parse -- a list of runs needs each one's
    conclusion, not a parse of each output."""
    from ..parse.contract import relaxation_of
    from ..parse.dirs import run_status
    from ..paths import Shape
    from ..runfiles import stem as rf_stem
    from ..runs import run_of
    from ..runrecord import ending, launch_record
    from .materialize import stage_home
    token = stage_home(base, task, stage).token
    stem = rf_stem(task.label, token)
    sh = Shape.named(task.shape)
    concluded = ending(attempt, stem).line
    try:
        launch = (launch_record(attempt) if sh.keeps_attempts_as_directories
                  else launch_record(container, basename=stem))
        state = run_status(attempt, stem,
                           launch=launch).state
    except Exception:                                    # noqa: BLE001
        state = None
    rec = None
    if verdict:
        # THIS STAGE'S RUN, the run door's, in either shape -- an attempt's
        # own, or this stage's among a flat folder's: its progress log,
        # then its engine output at the run's index.
        _run = run_of(attempt, stage=token)
        paths = ([_run.file(".molwatch.log"), _run.stdout]
                 if _run is not None else [])
        for path in (q for q in paths if q is not None):
            try:
                rec = relaxation_of(path)
            except Exception:                            # noqa: BLE001
                rec = None
            if rec is not None:
                break
    return concluded, state, (None if rec is None else bool(rec["converged"]))


def continuation_answer(base, task, stage: str, *, from_attempt=None,
                        cold: bool = False, verdict: bool = True,
                        template_text: Optional[str] = None,
                        bench: bool = False
                        ) -> Tuple[Optional[Continuation], Optional[str]]:
    """``(continuation, refusal)`` for ``stage`` (`job-system.md` § 5.4).

    First, what cannot be taken at all is refused, before prep writes
    anything (:func:`_cannot_be_named`): a run that is not one of this
    calculation's, ``--from`` with ``--cold``, either on the flat layout or
    a bias scan.  **Named** (``--from``): taken as said, for any kind -- what
    the run was is read and reported; `prepare_attempt` refuses only what
    cannot be done (no restart files in it).  **By default**, for a
    continuing stage of an independent ladder (`_stage_before`): the NEWEST
    attempt of the enabled stage before it, which must have ended on its own
    with exit code 0 (:func:`usable`) -- an older one never stands in,
    because a stage re-launched to tighten is the run the person means.  Otherwise the refusal, worded by
    what the run's state says and naming the commands that work on this
    layout.  ``(None, None)`` for ``--cold``, a linked kind's default, the
    first stage, a stage that starts clean, and one the description disables
    (prepped by name, it has no stage before it).  ``verdict=False`` leaves
    the relaxation unread -- for a reader that never prints it.

    A force-constant stage with no `relax` before it builds on nothing: the
    structure as given, when it is stated relaxed -- else refused, whatever
    was asked (`engines/vibration.md` § 5.2a's table; it was refused only at
    prep's step 4b until 2026-10-05, so `status` offered the prep that
    refused).

    ``template_text`` is the template as the caller read it -- prep's, read
    once (`script-preparation.md` § 3.0); with none it is read here.
    ``bench`` asks for a benchmark of the stage, which takes no ``--from``
    and is offered none."""
    from ..identity import command_stage
    base = Path(base)
    if force_constant_stage(task, stage) and relax_stage_of(task) is None:
        try:
            cfg = _stage_config(base, task, stage, template_text)
        except _unreadable() as exc:
            return None, (f"what `{stage}` builds on cannot be read: "
                          f"{exc}")
        why = unrelaxed_refusal(
            task, stage, stated=bool(getattr(cfg, "already_relaxed", False)))
        if why:
            return None, why
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
                            converged=converged,
                            linked=force_constant_stage(task, stage)), None
    try:
        prev, linked = _source_stage(base, task, stage, template_text)
        if prev is None:
            return None, None
        return _by_default(base, task, stage, prev, verdict=verdict,
                           linked=linked, bench=bench)
    except _unreadable() as exc:
        # RAISES NOTHING THE PERSON CAN FIX, as this module promises: a
        # template prep would refuse is a refusal here too, said -- `status`
        # printed a traceback and the Results tab lost its whole ladder over
        # it (W52).  A `TypeError` is a bug, and looks like one (it was
        # caught with the rest until 2026-10-05).
        return None, (f"what `{stage}` continues from cannot be read: "
                      f"{exc}")


def _unreadable() -> tuple:
    """What a description, its template or a run's records raise when they
    do not read -- the person's to fix, said as a refusal."""
    from ..resolve import ResolveError
    from .errors import PrepError
    return (ValueError, KeyError, OSError, ResolveError, PrepError)


def unrelaxed_refusal(task, stage: str, *, stated: bool) -> Optional[str]:
    """§ 5.2a's row *no enabled `relax`, the structure not stated relaxed*
    (`engines/vibration.md`): the box says *relax first* and the ladder
    holds nothing that would, so the description contradicts itself -- refused
    with the two ways out rather than measured at a geometry nobody chose
    (§ 2.2).  ``None`` when the ladder holds a `relax`, or the structure is
    stated relaxed."""
    from ..pyscf.stages import VIBRATION_RELAX_STAGE
    if relax_stage_of(task) is not None or stated:
        return None
    return (f"the structure is not stated to be relaxed (already_relaxed is "
            f"false in the template) and this ladder has no enabled "
            f"`{VIBRATION_RELAX_STAGE}` stage to relax it -- a harmonic "
            f"analysis off a stationary point reports the wrong frequencies "
            f"(engines/vibration.md 2.2).  Either add the "
            f"`{VIBRATION_RELAX_STAGE}` stage before `{stage}` (Task setup, "
            f"or task.json) and run it first, or state already_relaxed = true "
            f"in the template; the finish then measures the forces at this "
            f"geometry and says whether the statement held.")


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
    from .materialize import FLAT_HAS_NO_ATTEMPTS
    if not Shape.named(task.shape).keeps_attempts_as_directories:
        # A STAGE OF AN INDEPENDENT LADDER starts clean by its run card; a
        # linked one builds on what its kind says, and is told no way it
        # does not have.
        return FLAT_HAS_NO_ATTEMPTS + (
            "  A stage starts clean by its run card's `restart: clean`."
            if _independent(task) else "")
    if getattr(task, "calculation", None) == "transport":
        # A TRANSPORT RUNG'S INPUTS ARE ITS KIND'S: gathered from the rungs
        # upstream (`prep.gather_sources`), never named.  `--from` was taken
        # until 2026-10-05, naming another rung's run, and its carry and the
        # gather then wrote into one attempt; the same rung's earlier
        # attempt it was meant for is never there at prep -- a prepped rung
        # is not prepped again.
        return ("--from / --cold name what a stage continues from; a "
                "transport rung takes its inputs from the rungs upstream, "
                "gathered by its kind (engines/transport.md § 6.1), and is "
                "not prepped again once prepped (job-system.md § 5.0).")
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
        # A RUN OF A STAGE TURNED OFF is never continued from (user,
        # 2026-10-03, Q1: "never allow use").
        from ..task import stage_disabled
        why = stage_disabled(task, named)
        if why:
            return f"--from {from_attempt!r}: {why}"
    # A FORCE-CONSTANT STAGE BUILDS ON ITS LADDER'S `relax`, or on nothing
    # (`engines/vibration.md` § 5.2a's table): a run of `relax` may be
    # named; no other run, and no start from the structure while there is a
    # `relax` to measure at.
    if force_constant_stage(task, stage):
        relax = relax_stage_of(task)
        if relax is None:
            if from_attempt:
                return (f"--from {from_attempt!r}: nothing in this ladder "
                        f"relaxes -- `{stage}` measures the structure as "
                        f"given, stated relaxed (engines/vibration.md "
                        f"5.2a).  To build on a relaxation, add the "
                        f"`relax` stage before it and run it first.")
        elif cold:
            return (f"--cold: `{stage}` builds on `{relax}` -- it measures "
                    f"at the geometry `{relax}` reached.  To measure the "
                    f"structure as given, disable `{relax}` and state the "
                    f"structure relaxed (`already_relaxed` in the "
                    f"template) (engines/vibration.md 5.2a).")
        elif from_attempt and named != relax:
            return (f"--from {from_attempt!r}: `{stage}` builds on "
                    f"`{relax}` -- name one of its runs, "
                    f"<NN>_{relax}/run-<n> (engines/vibration.md 5.2a).")
    return None


def _by_default(base: Path, task, stage: str, prev: str, *, verdict: bool,
                linked: bool = False, bench: bool = False
                ) -> Tuple[Optional[Continuation], Optional[str]]:
    """The default for ``stage``, which builds on ``prev`` -- the stage
    before it, or (``linked``) the `relax` a force-constant stage builds on:
    the newest attempt, or the refusal (`continuation_answer`).  A linked
    stage is offered no start from the structure (`engines/vibration.md`
    § 5.2a), and a benchmark no earlier run (it takes no ``--from``)."""
    from ..paths import Shape
    from ..paths import attempt_dir as _adir
    from ..runfiles import compose as rf_compose
    from ..paths import attempts_in
    from ..runfiles import stem as rf_stem
    from .materialize import latest_attempt
    from .engines import engine_seam
    from .materialize import stage_home
    sh = Shape.named(task.shape)
    seam = engine_seam(str(task.engine))
    token = stage_home(base, task, prev).token
    stem = rf_stem(task.label, token)
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
    clean = ("" if linked else
             f"or start `{stage}` clean: set its run card's `restart` to "
             f"`clean` (Task setup, or task.json)\n" if flat else
             f"or start `{stage}` from the calculation's structure --\n"
             + block([command("prep", "run", stage, base=base,
                              flags=("--cold",))]) + "\n")
    rule = ("(engines/vibration.md § 5.2a)" if linked
            else "(job-system.md § 5.4)")
    # WHAT IT BUILDS ON, in the words of the ladder's kind.
    lead = (f"`{stage}` builds on `{prev}`" if linked else
            f"`{stage}` continues from the stage before it, `{prev}`")

    if flat:
        if not (base / rf_compose(task.label, seam.suffix, token)).is_file():
            return None, (f"{lead}, which has not run yet.  Run it "
                          f"first --\n{run_prev}\n{clean}{rule}")
        attempt, source = base, None
    else:
        latest = latest_attempt(container) if container.is_dir() else None
        if latest is None:
            return None, (f"{lead}, which has not run yet.  Run it "
                          f"first --\n{run_prev}\n{clean}{rule}")
        attempt, source = latest, str(latest.relative_to(base))
    concluded, state, converged = read_run(base, task, prev, attempt,
                                           container, verdict=verdict)
    if usable(attempt, stem):
        return Continuation(stage=prev, source=source, by_default=True,
                            concluded=concluded, state=state,
                            converged=converged, linked=linked), None

    # REFUSED -- worded by what the run's state says, with a command for
    # each way on (§ 5.3: what molbuilder prints, you can type).
    what = (f"the newest attempt of `{prev}`, {source}," if source
            else f"`{prev}`'s latest run,")
    if concluded is not None:
        # ENDED ON ITS OWN WITH AN ERROR -- never built on by default.  A
        # FAILED RUN IS LAUNCHED AGAIN -- a prepped stage is not prepped
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
    if not (flat or bench):
        # AN EARLIER RUN THAT CAN STAND IN WHEN ASKED FOR: the newest one
        # to build on, typed out -- never a placeholder.  Its ending is the
        # question, not its relaxation.
        for n in reversed(attempts_in(container)):
            a = _adir(container, n)
            if a != attempt and usable(a, stem):
                other = str(a.relative_to(base))
                break
    alt = (f"or continue from an earlier run of `{prev}` that concluded --\n"
           + block([command("prep", "run", stage, base=base,
                            flags=("--from", other))]) + "\n"
           if other else "")
    return None, (f"`{stage}` {'builds on' if linked else 'continues from'} "
                  f"{what} {why}.  {first}\n{alt}{clean}{rule}")


def force_constant_stage(task, stage: str) -> bool:
    """A force-constant stage of a SIESTA vibration -- `freq`, or any stage
    after `relax` in a displacement sweep, asked of its role and never of a
    name (`pyscf.stages.vibration_render_kind`, `engines/vibration.md`
    § 5.2a).  A PySCF vibration's one rung relaxes inside its deck."""
    if (str(getattr(task, "engine", "")) != "siesta"
            or getattr(task, "calculation", None) != "vibration"):
        return False
    from ..pyscf.stages import vibration_render_kind
    return vibration_render_kind(stage) == "vibration"


def relax_stage_of(task) -> Optional[str]:
    """The ladder's enabled relaxation rung, which every force-constant stage
    builds on, or ``None`` -- the structure is then measured as given, and
    must be stated relaxed (`engines/vibration.md` § 5.2a).  By the one role
    rule, the name in any case (plan § 5w K12)."""
    from ..pyscf.stages import vibration_render_kind
    return next((s.name for s in task.stages
                 if getattr(s, "enabled", True)
                 and vibration_render_kind(s.name) != "vibration"), None)


def _source_stage(base, task, stage: str, template_text: Optional[str] = None
                  ) -> Tuple[Optional[str], bool]:
    """``(the stage whose run ``stage`` builds on by default, linked)``: the
    stage before it in an independent ladder (:func:`_stage_before`), or the
    `relax` a force-constant stage builds on (linked) -- ``(None, ...)``
    when it builds on none."""
    if force_constant_stage(task, stage):
        return relax_stage_of(task), True
    return _stage_before(base, task, stage, template_text), False


def _independent(task) -> bool:
    """An INDEPENDENT ladder, whose stages continue one from another: a kind
    without rung roles (`template.KIND_ROLES`).  A linked stage's input is
    its kind's: a transport rung's gather (`prep.gather_sources`),
    and a force-constant stage builds on `relax` (:func:`_source_stage`)."""
    from ..template import KIND_ROLES
    return (getattr(task, "calculation", None)
            or "optimization") not in KIND_ROLES


def _ladder(base, task, template_text: Optional[str] = None) -> list:
    """``[(stage, resolved config)]`` of the enabled stages, as `prep`
    resolves them (`resolve.resolved_ladder`: template ⊕ stage overrides ⊕
    the run card) -- from ``template_text`` as the caller read it, else the
    template read here; ``[]`` with no template."""
    from ..resolve import resolved_ladder
    from .engines import engine_seam
    if template_text is None:
        from ..template import find_template
        tpl = find_template(Path(base), task.label)
        if tpl is None:
            return []
        template_text = tpl.read_text(encoding="utf-8")
    return resolved_ladder(template_text, task,
                           engine_seam(str(task.engine)).config_cls)


def _stage_config(base, task, stage: str, template_text: Optional[str] = None):
    """``stage``'s resolved config (:func:`_ladder`), or ``None``."""
    from ..identity import stage_key
    return next((c for n, c in _ladder(base, task, template_text)
                 if stage_key(n) == stage_key(stage)), None)


def _stage_before(base, task, stage: str,
                  template_text: Optional[str] = None) -> Optional[str]:
    """The enabled stage ``stage`` continues from by default -- the one
    before it in the ladder, resolved as `prep` resolves it (template ⊕
    stage overrides ⊕ the run card, so a card's ``restart: clean`` is read)
    -- or ``None``: a linked kind, the first stage, one that starts clean,
    one the description disables."""
    from ..identity import continues
    if not _independent(task):
        return None
    ladder = _ladder(base, task, template_text)
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
    from ..paths import attempts_in
    from .materialize import stage_home
    base = Path(base)
    prev, linked = _source_stage(base, task, stage)
    if prev is None:
        return None
    got, refused = _by_default(base, task, stage, prev, verdict=False,
                               linked=linked)
    sh = Shape.named(task.shape)
    runs = []
    if sh.keeps_attempts_as_directories:
        container = base / sh.stage_dir(stage_home(base, task, prev).token)
        for n in (reversed(attempts_in(container)) if container.is_dir()
                  else ()):
            a = _adir(container, n)
            c, s, _v = read_run(base, task, prev, a, container,
                                verdict=False)
            runs.append({"source": str(a.relative_to(base)),
                         "what": _what_it_is(c, s)})
    return {"from_stage": prev,
            # A FORCE-CONSTANT STAGE builds on its ladder's relaxation, not
            # always the stage before it -- the page's words follow.
            "linked": linked,
            "default": (dict(got.as_dict(), line=got.line())
                        if got is not None else None),
            "refused": refused, "runs": runs,
            # THE STRUCTURE, where the layout keeps attempts and the stage
            # may start from it -- a force-constant stage may not while
            # it builds on `relax` (`engines/vibration.md` § 5.2a).
            "cold": sh.keeps_attempts_as_directories and not linked}
