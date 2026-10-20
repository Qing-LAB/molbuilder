"""THE GATHER -- what a transport rung takes from the runs upstream, read
at its first launch or at a launch that gathers anew (`job-system.md` § 6.0;
`engines/transport.md` § 6.1): which file from which run
(:func:`gather_sources`, :func:`transport_inputs` -- three gates, each a
refusal naming what to do first), and the copies with their record
(:func:`gather_into`, `.gathered-from`).

A door: it answers and refuses, and writes only into the plan it is handed.
The ready door asks it what a first launch would take; the launch asks it,
then fills the run.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from .. import script_emit as _sc
from ..runfiles import RunNames
from ..siesta.engine_input import engine_input_differences
from ..runrecord import RunRef
from .errors import PrepError, user_error_as_prep
from .materialize import GATHERED, HandOver, hand_over, stage_home
from .transport_rung import (at_frame, transport_rung, transport_sources,
                             transport_spec)


def transport_inputs(calc, stage: str, *,
                     point=None) -> List[Tuple[RunRef, str]]:
    """What one attempt of ``stage`` takes from the runs upstream -- the
    § 4.2 DAG's inputs: ``[(the run it comes from, filename)]``, decided at the
    stage's first launch, or a launch that gathers anew (`job-system.md`
    § 6.0), and copied into its run (:func:`gather_into`).

    A transport rung takes no ``--from`` (`continuation.cannot_be_named`):
    its sources are structural — fixed by the design's DAG, not named by a
    person — so what keeps it honest is not a name but three gates, per
    input, each a refusal naming what to do first (strict composition,
    ruling Q2 — transport never runs its pieces for you):

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
    result can always say which electrode run fed it.
    """
    from ..transport.stages import (frames_of, inputs_of, points_in,
                                     products_of, rung_points)
    from .continuation import done, usable

    base, task = Path(calc.base), calc.task
    template_text, jobset = calc.template_text, calc.jobset
    inputs = inputs_of(task, stage)
    gathered: List[tuple] = []
    composed = None              # the junction, read once, when first needed
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
        from .group import number_of
        q2 = ("\n  (strict composition, ruling Q2: transport never runs its "
              "pieces for you.)")
        redo = redo_launched(upstream, calc=calc)
        # PREPARED, by the one door (`ready.job_of`), in the plan as the
        # verb read it.
        from .ready import job_of
        if job_of(jobset, upstream) is None:
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
            from ..parse.dirs import run_status
            from ..parse.dirs.job import LIVE_STATES
            from ..runrecord import LaunchRecordError, launch_record
            _names = home.names(task.shape)
            try:
                _launch = launch_record(newest, _names)
            except LaunchRecordError as exc:
                # A RECORD THAT DOES NOT READ is said in its reader's
                # words, as the refusal it is (`job-system.md` R-L21).
                raise PrepError(str(exc)) from None
            _prod = products_of(upstream, task.label, base_dir=base)
            _pts = points_in(newest, task, upstream, base=base)
            not_done = [f"{pt.words()} ({why})"
                        for p_, pt in _pts
                        for ok, why in [done(p_, _names, launch=_launch,
                                             products=_prod,
                                             launched_in=newest)]
                        if not ok]
            if not_done and _launch is not None and any(
                    run_status(p_, _names.stem, launch=_launch,
                               launched_in=newest).state
                    in LIVE_STATES for p_, _pt in _pts):
                # STILL QUEUED OR RUNNING: let it finish, as every run a
                # stage builds on (`job-system.md` § 5.4).
                raise PrepError(
                    f"the {stage} stage takes {upstream}'s newest run, "
                    f"{newest.relative_to(base)}, whole -- and it is still "
                    f"queued or running ({len(_pts) - len(not_done)} of "
                    f"{len(_pts)} points done).  Let it finish." + q2)
            if not_done and _launch is None:
                raise PrepError(
                    f"the {stage} stage takes {upstream}'s newest run, "
                    f"{newest.relative_to(base)}, whole -- and it has not "
                    f"been launched.  Launch it --\n"
                    + block(launch_pending(upstream, number_of(task, upstream),
                                           newest.name, base=base)) + q2)
            if not_done:
                raise PrepError(
                    f"the {stage} stage takes {upstream}'s newest run, "
                    f"{newest.relative_to(base)}, whole -- and its points "
                    f"{', '.join(not_done)} are not done.  Launch it again: "
                    f"the next run takes the done points over and runs the "
                    f"rest --\n"
                    + block(launch_lines("task", upstream, base=base)) + q2)
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
                        upstream, number_of(task, upstream),
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
        if composed is None:
            composed = composed_junction(calc)
        now = rung_deck_now(base, task, upstream, composed,
                             template_text=template_text,
                             # THE UPSTREAM'S OWN POINTS (`rung_points`): a
                             # rung that runs once ran at 0 V, on frame 0.
                             point=(point if up_points else None))
        # THE DECK THAT RAN, asked of its run's folder through the run door
        # (`runs.run_of`): a reader of a run takes the folder.
        from ..runs import run_of
        _run = run_of(where, stage=token)
        ran = _run.deck if _run is not None else None
        ran_text = ran.read_text() if ran is not None else None
        differs = (engine_input_differences(ran_text, now)
                   if ran_text is not None else ["it holds no deck"])
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
    strict composition, ruling Q2).  Prep renders the decks without it: a
    deck is the reviewable artifact, whether or not the rungs before it have
    run."""
    from ..transport.stages import frames_of, rung_points
    base = Path(calc.base)
    return [(pt, transport_inputs(calc, stage, point=pt))
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
    (`transport_rung.no_usable_record`); a refusal of the composition's is
    said as the gather's (:class:`PrepError`), as every door that asks the
    gather catches it."""
    from ..projects import find_projects_root
    from ..transport.compose import ComposeError, load_compose_record
    from ..transport.sort import SortError
    from .transport_rung import no_usable_record
    base, task = calc.base, calc.task
    root = find_projects_root(base)
    why: list = []
    try:
        composed = load_compose_record(base, citation=task.slots["junction"],
                                       tree_root=root, why=why,
                                       swap_electrodes=task.swap_electrodes)
    except (ComposeError, SortError) as exc:
        raise PrepError(str(exc)) from exc
    if composed is None:
        raise PrepError(no_usable_record(calc, why, root=root,
                                         afresh=False))
    return composed


def rung_deck_now(base, task, stage: str, composed, *, template_text,
                  point=None) -> str:
    """The deck ``stage`` renders NOW -- its text exactly as `prep task`
    writes it, from the current template, junction and the rung's run card
    (`transport_rung`, `transport_spec`, `script_emit.render_deck`).
    A transport deck records no machine sizing, so no allocation is
    needed to render it."""
    from .model import Resources
    from .prep_inputs import run_inputs
    # THE RUNG'S PINS, as `prep task` takes them from its card (`run_inputs`).
    _shape, pins = run_inputs(base, task, stage)
    struct, config, state, _element = transport_rung(
        task, stage, composed, Resources(), template_text=template_text,
        pins=pins or None)
    struct = at_frame(struct, point)
    spec, cfg = transport_spec(task, stage, struct, config, state, point,
                                base=base)
    with user_error_as_prep():
        return _sc.render_deck(spec, struct, cfg, verbose=True,
                               dest_dir=base,
                               sources=transport_sources(_element)).text


__all__ = ["gather_sources", "transport_inputs", "gather_into",
           "composed_junction", "rung_deck_now"]
