"""THE GATHER -- what a transport rung takes from the runs upstream, read
at its first launch or at a launch that gathers anew (`job-system.md` § 6.0;
`engines/transport.md` § 6.1): which file from which run
(:func:`gather_sources`, :func:`transport_inputs` -- three gates, each a
refusal naming what to do first), and the copies with their record
(:func:`gather_into`, `.gathered-from`).

A door: it answers and refuses, and writes only into the plan it is handed.
The ready door asks it what a first launch would take; the launch asks it,
then fills the run; prep asks it for the junction a transport calculation's
rungs describe, composed only while no stage is launched
(:func:`composed_for_prep`).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from .. import script_emit as _sc
from ..runfiles import RunNames
from ..deck_record import same_calculation
from ..runrecord import RunRef
from .errors import PrepError, user_error_as_prep
from .materialize import GATHERED, HandOver, hand_over
from .naming import stage_home
from .transport_rung import (at_frame, at_point, transport_rung,
                             transport_sources, transport_spec)


class ReadOnce:
    """What one gather reads for all its points: the junction, and each
    upstream rung's deck as it renders now, by the point it is rendered at
    -- one render for a rung that runs once, whichever point asks
    (`engines/transport.md` § 6.1); and what a calculation that reuses a
    finished one reads for all its groups (`jobset.reuse`, § 2a.16)."""

    def __init__(self, calc):
        self.calc = calc
        self._junction = None
        self._decks: dict = {}

    def junction(self):
        if self._junction is None:
            self._junction = composed_junction(self.calc)
        return self._junction

    def deck_now(self, stage: str, point) -> str:
        key = (stage, point)
        if key not in self._decks:
            self._decks[key] = rung_deck_now(self.calc, stage,
                                             self.junction(), point=point)
        return self._decks[key]


def transport_inputs(calc, stage: str, *, point=None,
                     once: Optional[ReadOnce] = None
                     ) -> List[Tuple[RunRef, str]]:
    """What one attempt of ``stage`` takes from the runs upstream -- the
    § 4.2 DAG's inputs: ``[(the run it comes from, filename)]``, decided at the
    stage's first launch, or a launch that gathers anew (`job-system.md`
    § 6.0), and copied into its run (:func:`gather_into`).

    A transport rung takes no ``--from`` (`continuation.cannot_be_named`):
    its sources are structural — fixed by the design's DAG, not named by a
    person — so what keeps it honest is not a name but three gates, per
    input, each a refusal naming what to do first (strict composition,
    transport never runs its pieces for you):

    * the upstream stage must have been PREPARED (its deck rendered);
    * its NEWEST run must have FINISHED -- the one rule every kind's default
      keeps (`job-system.md`, *The task*, D5): an older run never stands
      in, because a rung launched again is the run you mean;
    * that run must have run **the same engine input as the deck that rung
      renders NOW** -- from the current template, junction and run card,
      through the rung's own door (`transport_rung`), never the stage
      folder's last render, which a change since leaves as it was (plan § 5w
      K11) -- and hold the file (`engines/transport.md` § 6.1).

    A run that fails a gate is refused by name, with what to do.  What was
    taken from where lands in ``.gathered-from`` beside the copies, so a
    result can always say which electrode run fed it.  ``once`` is what the
    gather this belongs to reads for all its points (:class:`ReadOnce`).
    """
    from ..transport.stages import frames_of, inputs_of, rung_points
    from .continuation import usable

    base, task, jobset = Path(calc.base), calc.task, calc.jobset
    once = once if once is not None else ReadOnce(calc)
    inputs = inputs_of(task, stage)
    gathered: List[tuple] = []
    for upstream, filename in inputs:
        home = stage_home(base, task, upstream)
        token, up_dir = home.token, home.dir
        # AN UPSTREAM THAT RUNS PER POINT -- the device, for the
        # transmission -- is ONE RUN taken whole: its newest run, every point
        # done, the point this attempt is for read -- the same frame and the
        # same voltage (`engines/transport.md` § 2a.11).
        up_points = rung_points(task, upstream, frames=frames_of(base))
        # THE WAYS ON, by what the upstream rung's state says (§ 5.3: what
        # molbuilder prints, you can type): one not prepared is prepared and
        # launched; a prepared one is launched, or let finish, or -- to run
        # it as it is described now -- redone from the state saved before
        # its group's first launch (§ 5.0).
        from .commands import (block, launch_lines, launch_pending,
                               redo_launched, run_first as _run_first)
        from .continuation import read_run, state_remedy
        q2 = ("\n  (strict composition: transport never runs its "
              "pieces for you.)")
        redo = redo_launched(upstream, calc=calc)
        # PREPARED, by the plan's own answer, as the verb read it.
        if jobset is None or jobset.job(upstream) is None:
            raise PrepError(
                f"the {stage} stage consumes {filename} from {upstream}, "
                f"and {upstream} has not been prepared -- run it first --\n"
                + block(_run_first(upstream, calc=calc)) + q2)
        # THE NEWEST RUN, and only it (D5), by the layout's one answer
        # (`materialize.latest_attempt`): the one status door says it
        # finished (`continuation.usable`) -- exit code 0 and nothing in its
        # output saying the engine stopped -- or the refusal is worded by
        # its state, as a stage's own default is (`state_remedy`).
        from .materialize import latest_attempt
        newest = latest_attempt(up_dir) if up_dir.is_dir() else None
        if up_points and newest is not None:
            from ..runrecord import LaunchRecordError, launch_record
            from .continuation import points_state
            _names = home.names(task.shape)
            try:
                _launch = launch_record(newest, _names)
            except LaunchRecordError as exc:
                # A RECORD THAT DOES NOT READ is said in its reader's
                # words, as the refusal it is (`job-system.md` R-L21).
                raise PrepError(str(exc)) from None
            if _launch is None:
                # NEVER LAUNCHED, whatever its points' files: a run is
                # launched or it is not (`engines/transport.md` § 2a.11) --
                # a copy cut short reads so, its launch record written last.
                raise PrepError(
                    f"the {stage} stage takes {upstream}'s newest run, "
                    f"{newest.relative_to(base)}, whole -- and it has "
                    f"not been launched.  Launch it --\n"
                    + block(launch_pending(upstream,
                                           calc.request.group_of(upstream),
                                           newest.name, base=base))
                    + q2)
            else:
                # THE ONE RULE the status door asks (`continuation.
                # points_state`): a chain that stopped is a run that ended
                # with points not done, never one still queued.
                st, _d, rows = points_state(newest, _names, task, upstream,
                                            launch=_launch,
                                            base=base)
                not_done = [f"{p['point']} ({p['why']})" for p in rows
                            if not p["done"]]
                if st in ("queued", "running"):
                    raise PrepError(
                        f"the {stage} stage takes {upstream}'s newest run, "
                        f"{newest.relative_to(base)}, whole -- and it is "
                        f"still {st} ({len(rows) - len(not_done)} of "
                        f"{len(rows)} points done).  Let it finish." + q2)
                if not_done:
                    raise PrepError(
                        f"the {stage} stage takes {upstream}'s newest run, "
                        f"{newest.relative_to(base)}, whole -- and its "
                        f"points {', '.join(not_done)} are not done.  Launch "
                        f"it again: the next run takes the done points over "
                        f"and runs the rest --\n"
                        + block(launch_lines("task", upstream, base=base))
                        + q2)
        else:
            c, st, _v, d = (read_run(base, task, upstream, newest,
                                     verdict=False)
                            if newest is not None
                            else (None, "pending", None, None))
            if not usable(st):
                # A RUN NOT LAUNCHED -- prep's, with its group; one a refused
                # send opened, alone again (D4) -- and one launched, again
                # alone.
                why, first = state_remedy(
                    c, st, block(launch_pending(
                        upstream, calc.request.group_of(upstream),
                        newest.name if newest is not None else None,
                        base=base)
                        if st == "pending"
                        else launch_lines("task", upstream, base=base)),
                    detail=d)
                raise PrepError(
                    f"the {stage} stage consumes {filename} from "
                    f"{upstream}'s newest run"
                    + (f", {newest.relative_to(base)}," if newest else "")
                    + f" {why}.  {first}{q2}")
        # WHERE THE FILE IS: the run -- or, for an upstream that runs per
        # point, its point at this one.
        where = newest / point.rel if up_points else newest
        # THE SAME ENGINE INPUT, not the same bytes (`engines/transport.md`
        # § 6.1): what SIESTA reads of the deck that ran against the deck
        # the upstream rung renders NOW (plan § 5w K11, T-F30) -- never the
        # stage folder's last render, which a changed template value or a
        # re-pointed junction leaves as it was -- comments and molbuilder's
        # records aside, an explicit engine default the same as none; never
        # which molbuilder wrote it (`siesta.engine_input`).
        # THE UPSTREAM'S OWN POINTS (`rung_points`): a rung that runs once
        # ran at 0 V, on frame 0 -- one render for every point asking.
        differs = ran_otherwise(once, upstream, where, token=token,
                                point=point if up_points else None)
        if differs:
            raise PrepError(
                f"{upstream}'s newest run, {where.relative_to(base)}, did "
                f"not run what {upstream} renders now -- "
                f"{'; '.join(differs)} -- so its "
                f"{filename} answers a different calculation.  To run it as "
                f"it is described now, {redo}")
        # ...AND HOLDS THE FILE (`engines/transport.md` § 6.1).
        if not (where / filename).is_file():
            raise PrepError(
                f"{upstream}'s newest run, {where.relative_to(base)}, did "
                f"not write {filename}: it finished without producing what "
                f"the {stage} stage consumes.  To run it again, {redo}")
        gathered.append((RunRef.at(base, where), filename))
    return gathered


def ran_otherwise(once: ReadOnce, stage: str, where: Path, *, token: str,
                  point=None) -> List[str]:
    """§ 6.1'S QUESTION, OF A RUN OF EITHER CALCULATION (`engines/
    transport.md` § 6.1, § 2a.16): every difference between what the run in
    ``where`` ran -- a run, or a point of a run of points -- and the deck
    ``stage`` renders now in ``once``'s calculation at ``point``, through the
    one door of the same calculation (`deck_record.same_calculation`: the
    engine's reading, comments, molbuilder's records and the run settings
    aside); ``[]`` when it is the same calculation.  ``token`` is the
    stage's folder token, by which the run door reads the run.  Asked by the
    gather of the runs upstream, and by a calculation that reuses a finished
    one of that calculation's runs (`jobset.reuse`)."""
    from ..runs import run_of
    now = once.deck_now(stage, point)
    # THE DECK THAT RAN, asked of its run's folder through the run door
    # (`runs.run_of`): a reader of a run takes the folder.
    run = run_of(where, stage=token)
    ran = run.deck if run is not None else None
    if ran is None:
        return ["it holds no deck"]
    from ..parse.fdf import read_deck
    return same_calculation(once.calc.task.engine, read_deck(ran), now)


def gather_sources(calc, stage: str
                   ) -> List[Tuple[Optional[object], List[tuple]]]:
    """What this rung takes from the runs upstream -- ``[(point, [(source,
    filename), ...]), ...]``: one entry, ``point`` ``None``, for a rung that
    runs once; one per point for a rung that runs per point
    (`transport.stages.rung_points`), each gathered against **its own**
    point (the transmission at a frame and a voltage reads the device's
    point at that frame and voltage, never another's)
    (`engines/transport.md` § 2a.11).

    **Decided at the stage's first launch** (`job-system.md` § 6.0;
    recorded as `.gathered-from`), or at a launch that gathers anew
    (``--gather``): refused before anything is sent when an upstream is not
    done, or ran another engine input (:func:`transport_inputs`'s gates;
    strict composition).  Prep renders the decks without it: a
    deck is the reviewable artifact, whether or not the rungs before it have
    run."""
    from ..transport.stages import frames_of, rung_points
    base = Path(calc.base)
    once = ReadOnce(calc)
    return [(pt, transport_inputs(calc, stage, point=pt, once=once))
            for pt in (rung_points(calc.task, stage, frames=frames_of(base))
                       or (None,))]


def gather_into(base: Path, attempt_dir, inputs, plan, *, names: RunNames,
                run: int) -> Optional[HandOver]:
    """``inputs`` -- :func:`transport_inputs`' decision -- handed over into
    ``attempt_dir``, run ``run`` of the stage ``names`` names: the copies
    with the record of what came from where (`.gathered-from`), into
    ``plan`` (`materialize.hand_over`).  The hand-over, or ``None`` for a
    run that takes nothing."""
    if not inputs:
        return None
    h = HandOver(Path(attempt_dir), names, run, GATHERED, files=tuple(inputs))
    hand_over(base, h, plan)
    return h


def composed_junction(calc):
    """The junction this calculation is composed from -- its record beside
    ``task.json``, for the cited junction (`composed_for_prep` writes
    it before any rung's deck); read, never written: a record whose order a
    composition in memory finds is written by prep.  With no usable record,
    refused by its state, in the words prep says it with
    (:func:`no_usable_record`); a refusal of the composition's is
    said as the gather's (:class:`PrepError`), as every door that asks the
    gather catches it -- through the one read prep asks too
    (`transport_rung.junction_record`)."""
    from .transport_rung import junction_record
    composed, why, root = junction_record(calc)
    if composed is None:
        raise PrepError(no_usable_record(calc, why, root=root,
                                         afresh=False))
    return composed


def composed_for_prep(calc, plan, *, afresh: bool = True):
    """The junction a transport calculation's rungs describe: the record
    beside ``task.json`` when it is usable for THIS citation (the folder
    travels, `project-layout.md` § 2.1) -- written again when its order was
    found rather than read (`ComposedJunction.order_found`, a junction whose
    sidecar states none) -- else composed afresh from the projects tree,
    when ``afresh`` and no stage is launched.  With a stage launched, the
    recorded junction is never replaced: its runs ran on it; `prep task`
    with no group named writes only an order it found (``afresh`` false;
    `engines/transport.md` § 1.1, *The first prep*).  Refused otherwise,
    by state (:func:`no_usable_record`).

    Written into ``plan`` (`jobset.planned.Plan`), with everything else this
    prep writes, by one road (`transport_rung.written_and_read_back`)."""
    from ..transport.compose import compose_junction
    from .transport_rung import (compose_refusals, junction_record,
                                 written_and_read_back)
    base, task = calc.base, calc.task
    composed, why, root = junction_record(calc)
    try:
        if composed is not None:
            # ITS ORDER FOUND (`engines/transport.md` § 1.1, *The first
            # prep*): the record written again by its one writer -- the
            # recorded junction's atoms, places, cell and labels as they
            # were, its sidecar stating the order record.
            return (written_and_read_back(base, composed, plan, root=root,
                                           task=task)
                    if composed.order_found else composed)
        refusal = no_usable_record(calc, why, root=root, afresh=afresh)
        if refusal is not None:
            raise PrepError(refusal)
        return written_and_read_back(
            base, compose_junction(task.slots["junction"], tree_root=root,
                                   swap_electrodes=task.swap_electrodes),
            plan, root=root, task=task)
    except compose_refusals() as exc:
        raise PrepError(str(exc)) from exc


def no_usable_record(calc, why, *, root, afresh: bool) -> "str | None":
    """Why this calculation's junction is not composed afresh, by its state
    -- in words, with the way on -- or ``None`` when it may be: no stage
    launched, the projects tree here, and ``afresh``.  ``why`` is the
    record read's reasons (`compose.load_compose_record`).  THE ONE
    COMPOSER of this refusal, asked by prep (:func:`composed_for_prep`) and
    by the gather, which never composes (``afresh`` false;
    `gather.composed_junction`)."""
    from ..runrecord import LaunchRecordError
    from .commands import block, prep_group_lines, redo_launched
    from .ready import launched_stage
    base, task = calc.base, calc.task
    reason = why[-1] if why else "there is no usable record"
    try:
        launched = launched_stage(calc)
    except LaunchRecordError as exc:
        return (f"this calculation's junction is not composed again: "
                f"{reason}.  And {exc}")
    if launched is not None:
        # A LAUNCHED STAGE RAN ON THE JUNCTION RECORDED: never replaced,
        # whatever made the record unusable.
        return (f"this calculation's junction is not composed again: "
                f"{reason}.  `{launched}` ran on the junction recorded, and a "
                f"launched stage's junction is never replaced "
                f"(engines/transport.md § 1.1); to compose it again, "
                + redo_launched(launched, calc=calc))
    if root is None:
        return (f"this transport calculation cannot be composed here: "
                f"{reason}.  And {base} is not inside a projects tree, so the "
                f"citation {task.slots['junction']!r} cannot be resolved to "
                f"compose afresh.  Prep once inside the tree that holds the "
                f"cited junction -- the record then travels with the folder "
                f"(engines/transport.md § 3.1).")
    if not afresh:
        numbers = calc.request.numbers
        return (f"this calculation's junction is not composed again here: "
                f"{reason}.  It is composed by a prep of the calculation's "
                f"groups --\n" + block(prep_group_lines(numbers, base=base)))
    return None


def rung_deck_now(calc, stage: str, composed, *, point=None) -> str:
    """The deck ``stage`` of ``calc`` renders NOW -- its text exactly as
    `prep task` writes it, from the calculation's template, junction and
    the rung's run card (`transport_rung`, `transport_spec`,
    `script_emit.render_deck`), and the person's own USER-CUSTOM zone of the
    deck prep wrote for it, which prep keeps when it writes the deck again
    (`script_emit.write_script`) -- so a keyword typed there is the same
    engine input in the run and now.  A transport deck records no machine
    sizing, so no allocation is needed to render it."""
    from .model import Resources
    from .prep_inputs import run_inputs
    base, task = calc.base, calc.task
    # THE RUNG'S RUN SETTINGS, as `prep task` takes them from its card
    # (`run_inputs`).
    _shape, settings = run_inputs(calc.request, stage)
    struct, config, state, _element = transport_rung(
        task, stage, composed, Resources(), template_text=calc.template_text,
        run_settings=settings or None)
    struct = at_frame(struct, point)
    # A TRANSMISSION POINT WHOSE GRID IS A FINISHED POINT'S OWN renders that
    # point's contours (`reuse.that_point`; `engines/transport.md` § 2a.16).
    from .reuse import SHARING_RUNG, that_point
    contours = (that_point(calc, point, at_point(config, point))
                if stage == SHARING_RUNG else None)
    spec, cfg = transport_spec(task, stage, struct, config, state, point,
                                base=base, contours=contours)
    with user_error_as_prep():
        text = _sc.render_deck(spec, struct, cfg, verbose=True,
                               dest_dir=base,
                               sources=transport_sources(_element)).text
    # THE ZONE PREP KEEPS: from the deck prep wrote for this rung -- in its
    # point's folder for a rung that runs per point (`prep`'s
    # `prepare_deck`, `script_emit.merge_user_custom_from_target`).
    job = calc.jobset.job(stage) if calc.jobset is not None else None
    home = stage_home(base, task, stage)
    if job is not None and home.dir is not None:
        at = home.dir / point.rel if point is not None else home.dir
        text = _sc.merge_user_custom_from_target(text, at / job.script)
    return text


__all__ = ["ReadOnce", "ran_otherwise", "gather_sources",
           "transport_inputs", "gather_into",
           "composed_junction", "composed_for_prep", "no_usable_record",
           "rung_deck_now"]
