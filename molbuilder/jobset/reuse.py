"""A TRANSPORT CALCULATION THAT REUSES A FINISHED ONE (`engines/transport.md`
§ 2a.16): at its prep, for each group it is to prepare, in the ladder's
order and before it is planned, whether the group is **copied in** -- each
member's newest run in the finished calculation finished (a rung of points:
holding exactly the points this calculation describes, every one done), the
same calculation as the deck this calculation renders for it now, and every
stage it takes inputs from copied in too -- or **prepared here**, with what
decided it, and for a rung of points prepared here the points its first
launch will take over (:func:`decide`); every run of a group copied in,
copied into the prep's plan (:func:`copy_in`); the manifest that says it
before anything is written (:class:`Reuse`); at a rung of points' first
launch, the points it takes over from the finished calculation
(:func:`takes_over`, :func:`take_into`); and what each transmission point
it computes shares with the finished point -- a rung that runs once, its
one point -- decided at prep, given it at every launch that walks it
(:func:`shared_into`), and the grid a window is fitted to
(:func:`finished_window`).

A door on floor 4: it opens the finished calculation through the
calculation door, reads its runs through the run doors -- their state
(`continuation.points_state`, `continuation.read_run`, `continuation.done`),
and § 6.1's question through the gather's door (`gather.ran_otherwise`) --
and writes only into the plan it is handed.  The finished calculation is
never written.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..runrecord import LaunchRecordError


@dataclass(frozen=True)
class Copy:
    """One run of the finished calculation, copied in."""
    stage: str
    #: the finished calculation's run, and where it lands here
    source: Path
    run: Path
    #: the finished run, as the manifest names it: the finished
    #: calculation's folder and the run in it
    shown: str
    #: what its ``.copied-from`` names, tree-relative (:func:`origin`)
    copied_from: str
    #: how it ended, the status door's words
    ended: str
    #: what is copied, in bytes, and the engine's scratch left behind, by
    #: the folder it lies in
    size: int
    left: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Take:
    """One point of the finished calculation, taken over by a rung of points
    at its first launch."""
    #: this calculation's point
    point: object
    #: the finished point's folder, as the manifest names it, and what the
    #: point's ``.copied-from`` names (:func:`origin`)
    source: Path
    shown: str
    copied_from: str


@dataclass(frozen=True)
class Takes:
    """What a rung of points prepared here takes over at its first launch,
    from the finished calculation's newest run of it (``run``, as the
    manifest names it), and what it walks."""
    stage: str
    run: str
    taken: Tuple[Take, ...]
    walked: Tuple[object, ...]

    def line(self) -> str:
        took = (f"takes over {', '.join(t.point.words() for t in self.taken)}"
                f" from {self.run}" if self.taken else "takes over no point")
        walks = (f"walks {', '.join(p.words() for p in self.walked)}"
                 if self.walked else "walks none")
        return f"{self.stage}: {took}; {walks}"


@dataclass(frozen=True)
class Decision:
    """One group: copied in, its runs; or prepared here, why, what its rungs
    of points will take over, and what each transmission point it walks
    shares (``energies``, its words).  ``contours`` are the energies each
    transmission point's deck computes where it is not its window whole --
    ``(stage, point key, Contours)`` (`transport.energies`)."""
    number: int
    members: Tuple[str, ...]
    copies: Tuple[Copy, ...] = ()
    why: Tuple[str, ...] = ()
    takes: Tuple[Takes, ...] = ()
    energies: Tuple[str, ...] = ()
    contours: Tuple[tuple, ...] = ()
    #: what the first launch of its transmission copies into its points'
    #: ``shared/``, in bytes
    shares: int = 0

    @property
    def copied(self) -> bool:
        return bool(self.copies)


@dataclass(frozen=True)
class Reuse:
    """THE MANIFEST -- what this calculation copies, and what it runs: one
    answer, composed once by prep's question (:func:`decide`), shown
    wherever a prep is asked about and kept in `STAGE-PLAN.md` and the
    ledger (`engines/transport.md` § 2a.16, *The manifest*)."""
    #: the finished calculation, as the description's ``reuses`` names it
    reuses: str
    groups: Tuple[Decision, ...]

    @property
    def name(self) -> str:
        return Path(self.reuses).name

    def copied(self, number: int) -> bool:
        return any(d.number == number and d.copied for d in self.groups)

    def copies_of(self, stage: str) -> Tuple[Copy, ...]:
        return tuple(c for d in self.groups for c in d.copies
                     if c.stage == stage)

    def contours_for(self, stage: str) -> dict:
        """What each of ``stage``'s points computes, by its key, where it is
        not its window whole -- what its prep renders into its deck."""
        return {key: c for d in self.groups for st, key, c in d.contours
                if st == stage}

    def lines(self, base) -> List[str]:
        """The manifest in words, one group after another in the ladder's
        order, then the totals."""
        from ..units import size_words
        base = Path(base)
        out = [f"reuses {self.reuses} -- what this calculation copies, and "
               f"what it runs"]
        copies = [c for d in self.groups for c in d.copies]
        width = max((len(", ".join(d.members)) for d in self.groups),
                    default=0)
        w_run = max((len(str(c.run.relative_to(base))) for c in copies),
                    default=0)
        w_from = max((len(c.shown) for c in copies), default=0)
        for d in self.groups:
            head = f"  group {d.number}  {', '.join(d.members):<{width}}  "
            if d.copied:
                out.append(head + "copied in")
                for c in d.copies:
                    left = (f"  ({', '.join(c.left)} left behind)"
                            if c.left else "")
                    out.append(f"           "
                               f"{str(c.run.relative_to(base)):<{w_run}}  <-  "
                               f"{c.shown:<{w_from}}  {c.ended}  "
                               f"{size_words(c.size)}{left}")
            else:
                out.append(head + "prepared here -- " + d.why[0])
                out.extend(f"           {w}" for w in d.why[1:])
                out.extend(f"           at its first launch, {t.line()}"
                           for t in d.takes)
                if d.energies:
                    out.append("           its energies:")
                    out.extend(f"             {w}" for w in d.energies)
        size = sum(c.size for c in copies)
        shares = sum(d.shares for d in self.groups)
        if size or shares:
            out.append(f"  copied now: {size_words(size)}"
                       + (f" -- the transmission's first launch copies "
                          f"{size_words(shares)} into its points' shared/"
                          if shares else "")
                       + f" -- the next save stores it again: "
                         f"{size_words(size + shares)}")
        return out

    def to_wire(self, base) -> dict:
        return {"reuses": self.reuses, "lines": self.lines(base),
                "groups": [{"number": d.number, "members": list(d.members),
                            "copied": d.copied, "why": list(d.why),
                            "runs": [{"stage": c.stage,
                                      "run": str(c.run.relative_to(base)),
                                      "from": c.copied_from,
                                      "ended": c.ended, "size": c.size,
                                      "left": list(c.left)}
                                     for c in d.copies],
                            "energies": list(d.energies),
                            "takes": [{"stage": t.stage, "line": t.line(),
                                       "taken": [{**x.point.to_wire(),
                                                  "from": x.copied_from}
                                                 for x in t.taken],
                                       "walked": [p.to_wire()
                                                  for p in t.walked]}
                                      for t in d.takes]}
                           for d in self.groups]}


def finished_of(calc):
    """``(the finished calculation, why)`` -- the calculation ``calc``'s
    description reuses, opened through the calculation door; ``(None,
    why)`` when its folder is not there or holds no calculation that reads,
    every group then prepared here, saying so."""
    from .. import calculation
    from ..projects import find_projects_root
    root = find_projects_root(Path(calc.base))
    if root is None:
        return None, (f"{calc.base} is not inside a projects tree, so "
                      f"{calc.task.reuses} cannot be found")
    folder = Path(root) / calc.task.reuses
    if not folder.is_dir():
        return None, f"{calc.task.reuses} is not there"
    try:
        return calculation.open(folder), None
    except ValueError as exc:
        return None, str(exc)


def origin(base, folder) -> str:
    """THE COMPUTATION ``folder`` HOLDS -- a run, or a point of a run of
    points, of the calculation in ``base`` -- tree-relative: the run or
    point it is a copy of (`runrecord.copied_of`), or itself.  Two folders
    hold the same computation exactly when their origins are one -- what a
    point taken over is matched by, and what a copy's ``.copied-from``
    says, one hop."""
    from ..projects import find_projects_root
    from ..runrecord import copied_of
    said = copied_of(base, folder)
    if said:
        return said
    tree = Path(find_projects_root(Path(base))).resolve()
    return Path(folder).resolve().relative_to(tree).as_posix()


def point_key(pt) -> Tuple[int, float]:
    """A point as matched across two calculations: its frame and its
    voltage -- a rung that runs once (``None``, its one point the run
    itself, as the gather names it), frame 0 and 0 V."""
    return (0, 0.0) if pt is None else (pt.frame or 0, float(pt.volts or 0.0))


def _words(pt) -> str:
    """A point in words -- a rung that runs once, its 0 V point
    (`stages.Point.words`)."""
    from ..transport.stages import Point
    return (pt or Point()).words()


def _newest(calc, stage: str) -> Optional[Path]:
    """``stage``'s newest run in ``calc``."""
    from .materialize import latest_attempt
    from .naming import stage_home
    home = stage_home(Path(calc.base), calc.task, stage)
    return (latest_attempt(home.dir)
            if home.dir is not None and home.dir.is_dir() else None)


def _at(calc, stage: str, run: Path, pt) -> Optional[Path]:
    """``run`` of ``stage`` in ``calc`` at the point ``pt`` -- a frame and a
    voltage, matched as such (:func:`point_key`): its point's folder for a
    rung of points, the run itself for a rung that ran once, which is its
    0 V point on frame 0, as ``pt`` ``None`` is; ``None`` when it holds no
    such point."""
    from ..transport.stages import frames_of, rung_points
    theirs = rung_points(calc.task, stage, frames=frames_of(calc.base))
    if not theirs:
        return run if point_key(pt) == (0, 0.0) else None
    match = {point_key(q): q for q in theirs}.get(point_key(pt))
    return run / match.rel if match is not None else None


def _given_at(calc, upstream: str, run: Path, pt) -> Optional[Path]:
    """What ``run`` of ``upstream`` in ``calc`` gives a point ``pt`` of a
    rung downstream -- as the gather takes it: its point's folder at the
    same frame and voltage for a rung of points, the run itself for a rung
    that runs once, whichever point asks (`gather.transport_inputs`)."""
    from ..transport.stages import frames_of, rung_points
    theirs = rung_points(calc.task, upstream, frames=frames_of(calc.base))
    if not theirs:
        return run
    match = {point_key(q): q for q in theirs}.get(point_key(pt))
    return run / match.rel if match is not None else None


def decide(calc, numbers: Sequence[int]) -> Optional[Reuse]:
    """THE QUESTION, of each group ``numbers`` names, in the ladder's order
    (`engines/transport.md` § 2a.16, *At prep*) -- ``None`` for a
    calculation that reuses nothing, or a prep that asks it of no group (a
    launched group prepared again for its job request is not asked).  A
    group's upstream is answered first:
    copied in by this decision, or by an earlier prep (its newest run a
    copy, :func:`_member`) -- and a rung of points an earlier prep prepared
    and no launch has sent, by the points its first launch will take over.
    For a rung of points prepared here, the points its first launch will
    take over -- its inputs as they will be: copied in, or taken over by
    the rung upstream (:func:`takes_over`); and what each transmission
    point it computes shares -- a rung that runs once, its one point, the
    run itself.  Nothing is written."""
    from .gather import ReadOnce
    from .materialize import ever_launched
    from .transport_rung import junction_record
    from ..transport.stages import frames_of, rung_points
    if not calc.task.reuses or not numbers:
        return None
    finished, gone = finished_of(calc)
    if finished is not None:
        composed, said, _root = junction_record(calc)
        if composed is None:
            # NO JUNCTION RECORDED FOR THIS DESCRIPTION -- its citation or
            # its rename changed since it was described: composed anew at
            # this prep, every rung is another calculation (§ 1.2).
            finished, gone = None, ("this calculation's junction is "
                                    "composed anew at this prep -- "
                                    + "; ".join(said))
    once = ReadOnce(calc)
    copied: set = set()
    #: each rung of points prepared and not launched, the origin of each
    #: point it will take over, by the point's key -- what the rungs after
    #: it are given.
    forecast: Dict[str, Dict[Tuple[int, float], str]] = {}

    def given_to(st) -> Callable:
        return lambda pt: _forecast(calc, finished, st, pt, copied, forecast)

    def takes(st, points) -> Takes:
        t = takes_over(calc, finished, st, points, once=once,
                       inputs=given_to(st))
        forecast[st] = {point_key(x.point): x.copied_from for x in t.taken}
        return t
    out: List[Decision] = []
    asked = set(numbers)
    for n in calc.request.numbers:
        if n > max(asked):
            break
        members = tuple(calc.request.members(n))
        if n not in asked:
            # A GROUP AN EARLIER PREP PREPARED: its rungs of points no
            # launch has sent take their points over at their first launch.
            for st in (members if finished is not None else ()):
                points = rung_points(calc.task, st,
                                     frames=frames_of(calc.base))
                job = calc.job(st)
                if points and job is not None and not ever_launched(calc,
                                                                    job):
                    takes(st, points)
            continue
        if finished is None:
            out.append(Decision(n, members, why=(gone,)))
            continue
        copies: List[Copy] = []
        why: List[str] = []
        for st in members:
            try:
                got, said = _member(calc, finished, st, once, copied)
            except LaunchRecordError as exc:
                got, said = [], [f"{st}: {exc}"]
            copies.extend(got)
            why.extend(said)
        if not why:
            copied.update(members)
            out.append(Decision(n, members, copies=tuple(copies),
                                contours=tuple(_rendered(calc, members,
                                                         once))))
            continue
        rungs, energies, contours, shares = [], [], [], 0
        for st in members:
            points = rung_points(calc.task, st, frames=frames_of(calc.base))
            if points:
                t = takes(st, points)
                rungs.append(t)
                taken, walked = [x.point for x in t.taken], t.walked
            else:
                # A RUNG THAT RUNS ONCE takes nothing over -- copied in with
                # its group, or run here: its one point, the run itself.
                taken, walked = [], (None,)
            if st != SHARING_RUNG:
                continue
            # A POINT TAKEN OVER renders the finished point's own contours
            # (`engines/transport.md` § 2a.16, *is that point*); one walked
            # shares what it can.
            contours += _rendered(calc, (st,), once, only=taken)
            for pt in walked:
                c, said = _share(calc, finished, pt, once, given_to(st)(pt))
                energies.append(said)
                if c is not None:
                    contours.append((st, point_key(pt), c))
                    shares += _shared_size(calc, c)
        out.append(Decision(n, members, why=tuple(why), takes=tuple(rungs),
                            energies=tuple(energies),
                            contours=tuple(contours), shares=shares))
    return Reuse(calc.task.reuses, tuple(out))


#: THE RUNG WHOSE POINTS SHARE ENERGIES -- the transmission, TBtrans's
#: (`engines/transport.md` § 2a.16).
SHARING_RUNG = "transmission"


def _rendered(calc, members, once, *, only=None) -> List[tuple]:
    """The contours each point of ``members``' transmission renders now --
    a point whose grid is the finished point's own renders that point's
    (:func:`that_point`) -- read back from its deck rendered now
    (`gather.ReadOnce`); ``only`` those points.  A rung that runs once is
    its one point, the run itself."""
    from ..transport.energies import Contours
    from ..transport.stages import frames_of, rung_points
    out = []
    if SHARING_RUNG not in members:
        return out
    for pt in (only if only is not None else rung_points(
            calc.task, SHARING_RUNG, frames=frames_of(calc.base))
               or (None,)):
        c = Contours.of_deck(once.deck_now(SHARING_RUNG, pt))
        if c is not None:
            out.append((SHARING_RUNG, point_key(pt), c))
    return out


def finished_window(finished) -> Tuple[float, float, int]:
    """THE FINISHED GRID a window is fitted to (`engines/transport.md`
    § 2a.16): the grid the finished calculation's newest transmission run
    computed at its first point done, as its deck states it -- its window's
    ends and count, the numbers tbtrans computed its energies from, or, a
    point that shared energies, the grid its record describes
    (:func:`_window_of`).  :class:`ValueError` when it holds none done."""
    from ..runs import run_of
    from ..transport.stages import frames_of, rung_points
    for pt in (rung_points(finished.task, SHARING_RUNG,
                           frames=frames_of(finished.base)) or (None,)):
        src, _run = _finished_point(finished, SHARING_RUNG, pt)
        run = run_of(src) if src is not None else None
        window = (_window_of(run.deck.read_text(encoding="utf-8",
                                                errors="replace"))
                  if run is not None and run.deck is not None else None)
        if window is not None:
            return window
    raise ValueError(
        f"{Path(finished.base).name} holds no transmission point done -- a "
        f"window is fitted to the energies its transmission computed "
        f"(engines/transport.md 2a.16)")


def _shared_files(src: Path, label: str) -> List[Path]:
    """WHAT A SHARING POINT'S ``shared/`` HOLDS of the finished point in
    ``src`` (`engines/transport.md` § 2a.16) -- what the readers read of it
    (`model/parse.md` § 5e): its data files, each with its kept reading when
    there is one, and TBtrans's k-averaged text files; ``[]`` for a folder
    not there."""
    from ..parse.engines.tbtrans import averaged_files
    from ..parse.engines.tbtrans_nc import data_files, digest_of
    from ..transport.transiesta import LEAD_NAMES, LEAD_PAIR
    if not src.is_dir():
        return []
    out: List[Path] = []
    for nc in data_files(src, label).values():
        out.append(nc)
        if digest_of(nc).is_file():
            out.append(digest_of(nc))
    for got in averaged_files(src, label, pair=LEAD_PAIR,
                              leads=tuple(LEAD_NAMES)).values():
        for where in got.values():
            out += (list(where.values()) if isinstance(where, dict)
                    else [where])
    return out


def _shared_size(calc, contours) -> int:
    """What a point's ``shared/`` will hold, in bytes: the finished point's
    files (:func:`_shared_files`) and its own ``shared/``
    (:func:`shared_into`)."""
    from ..projects import find_projects_root
    from ..transport.energies import SHARED_DIR
    src = (Path(find_projects_root(Path(calc.base))).resolve()
           / contours.source)
    files = (_shared_files(src, calc.task.label)
             + (_files(src / SHARED_DIR) if (src / SHARED_DIR).is_dir()
                else []))
    return sum(f.stat().st_size for f in files)


def that_point(calc, pt, cfg):
    """A TRANSMISSION POINT WHOSE GRID IS THE FINISHED POINT'S OWN -- every
    energy that point holds, its own and those it shared, and no other --
    IS THAT POINT (`engines/transport.md` § 2a.16): its deck renders the
    finished point's contours, the ones that point's deck records, so it is
    the same calculation.  ``None`` for a point whose finished point
    computed its window whole, or another grid; ``cfg`` is the point's
    config, its window's; a rung that runs once, its one point (``None``),
    the run itself."""
    from ..transport.energies import grid, same_grid, sharing_of
    if not calc.task.reuses:
        return None
    finished, _gone = finished_of(calc)
    newest = (_newest(finished, SHARING_RUNG) if finished is not None
              else None)
    src = (_at(finished, SHARING_RUNG, newest, pt) if newest is not None
           else None)
    c = sharing_of(src) if src is not None and src.is_dir() else None
    if c is None:
        return None
    return c if same_grid(grid(*_window(cfg)), grid(*c.window)) else None


def _window(cfg) -> Tuple[float, float, int]:
    """A transmission point's window, from its config: its ends and its
    number of energies."""
    return (float(cfg.transmission_emin_ev), float(cfg.transmission_emax_ev),
            int(cfg.transmission_n_points))


#: A TBtrans energy block of a deck -- ``TBT.Contours`` and each
#: ``TBT.Contour.<name>`` -- whole, as fdf spells a label, in any case.
_ENERGY_BLOCK = re.compile(
    r"^[ \t]*%block[ \t]+(TBT\.Contours?(?:\.\S+)?)[ \t]*\n.*?"
    r"^[ \t]*%endblock[ \t]+\1[ \t]*$\n?", re.M | re.S | re.I)

#: The window line of a deck that computes its window whole.
_WINDOW_LINE = re.compile(
    r"%block[ \t]+TBT\.Contour\.window.*?from[ \t]+(\S+)[ \t]+eV[ \t]+to"
    r"[ \t]+(\S+)[ \t]+eV.*?points[ \t]+(\d+)", re.S | re.I)


def _with_energies(text: str, energies_of: str) -> str:
    """``text`` with its TBtrans energy blocks taken out and those of
    ``energies_of`` put in -- a deck rendered with the finished point's
    contours in place of its own (§ 2a.16)."""
    return (_ENERGY_BLOCK.sub("", text) + "\n"
            + "\n".join(m.group(0) for m in
                        _ENERGY_BLOCK.finditer(energies_of)))


def _window_of(text: str) -> Optional[Tuple[float, float, int]]:
    """The grid a transmission deck describes: its sharing record's, or its
    window's own line."""
    from ..transport.energies import Contours
    c = Contours.of_deck(text)
    if c is not None:
        return c.window
    m = _WINDOW_LINE.search(text)
    return ((float(m.group(1)), float(m.group(2)), int(m.group(3)))
            if m else None)


def _share(calc, finished, pt, once, mine) -> Tuple[object, str]:
    """``(its contours, its words)`` -- WHETHER A TRANSMISSION POINT
    WALKED HERE SHARES ENERGIES with the finished point (§ 2a.16): the
    finished calculation's newest transmission run holds it done; it is
    given what that point was given (``mine``, each file's origin); its deck
    rendered now, with the finished point's contours in place of its own,
    is the same engine input; and the two grids hold energies in common,
    the new one adding at least one.  The contours ``None`` when it shares
    none, its words saying why."""
    from ..deck_record import same_calculation
    from ..projects import find_projects_root
    from ..runs import run_of
    from ..transport.energies import grid, held, shared_contours
    words = f"{_words(pt):<6}"
    fbase = Path(finished.base)
    src, newest = _finished_point(finished, SHARING_RUNG, pt)
    if src is None:
        return None, (f"{words} shares none -- the finished calculation "
                      f"holds no {_words(pt)} point done")
    shown = f"{fbase.name}/{src.relative_to(fbase)}"
    if not _given_the_same(finished, SHARING_RUNG, newest, src, mine):
        return None, (f"{words} shares none -- it is not given what "
                      f"{shown} was given")
    now = once.deck_now(SHARING_RUNG, pt)
    ran = run_of(src).deck.read_text(encoding="utf-8", errors="replace")
    differs = same_calculation(calc.task.engine, ran,
                               _with_energies(now, ran))
    if differs:
        return None, (f"{words} shares none -- it is another calculation "
                      f"than {shown}: {'; '.join(differs)}")
    window = _window_of(now)
    theirs = held(src, calc.task.label)
    tree = Path(find_projects_root(fbase)).resolve()
    c = shared_contours(grid(*window), theirs, window=window,
                        source=src.resolve().relative_to(tree).as_posix())
    asked = (f"{window[0]:g}…{window[1]:g} eV at {window[2]} points")

    def offer() -> str:
        # THE WINDOW AT THE SAME SPACING THAT WOULD SHARE ALL, the person's
        # to take or leave (`transport.window.fitted`) -- beside a grid that
        # keeps fewer than every finished energy.
        from ..transport.window import fitted
        try:
            fit = fitted(_window_of(ran), window[0], window[1])
        except ValueError:
            return ""
        return (f"; fitted to the finished grid: {fit.emin:.5f}…"
                f"{fit.emax:.5f} eV at {fit.points} points, holding all "
                f"{fit.kept}")
    if c is not None:
        return c, (f"{words} shares {c.held} of {window[2]} energies with "
                   f"{shown} -- computes {c.computed} here"
                   + (offer() if c.held < len(theirs) else ""))
    from ..transport.energies import in_common
    if not in_common(grid(*window), theirs):
        return None, (f"{words} shares none -- {asked} holds none of "
                      f"{shown}'s {len(theirs)}{offer()}")
    return None, (f"{words} shares none -- every energy of {asked} is one "
                  f"{shown} holds: it asks a narrower question, and computes "
                  f"its {window[2]} again")


def _finished_point(finished, stage: str, pt):
    """``(the finished point's folder, its run)`` -- ``stage``'s newest run
    in the finished calculation at ``pt``, done -- or ``(None, run)``."""
    from ..runrecord import launch_record
    from ..transport.stages import products_of
    from .continuation import done
    from .naming import stage_home
    fbase, ftask = Path(finished.base), finished.task
    names = stage_home(fbase, ftask, stage).names(ftask.shape)
    newest = _newest(finished, stage)
    try:
        launch = (launch_record(newest, names) if newest is not None
                  else None)
    except LaunchRecordError:
        launch = None
    src = _at(finished, stage, newest, pt) if launch is not None else None
    if src is None or not done(
            src, names, launch=launch,
            products=products_of(stage, ftask.label, base_dir=fbase),
            launched_in=newest)[0]:
        return None, newest
    return src, newest


def _forecast(calc, finished, stage: str, pt, copied: set,
              forecast) -> Dict[str, Optional[str]]:
    """What ``stage``'s point ``pt`` will be given by the rungs upstream at
    its first launch, by each file's origin (:func:`origin`): from a rung
    copied in by this decision, the finished run it copies; from a rung
    launched here, its newest run; from a rung of points prepared here, the
    point it will take over -- ``None`` for one it computes here."""
    from ..transport.stages import inputs_of
    out: Dict[str, Optional[str]] = {}
    for upstream, filename in inputs_of(calc.task, stage):
        if upstream in copied:
            run = _newest(finished, upstream)
            at = (_given_at(finished, upstream, run, pt) if run is not None
                  else None)
            out[filename] = (origin(finished.base, at) if at is not None
                             else None)
        elif upstream in forecast:
            out[filename] = forecast[upstream].get(point_key(pt))
        else:
            run = _newest(calc, upstream)
            at = (_given_at(calc, upstream, run, pt) if run is not None
                  else None)
            out[filename] = (origin(calc.base, at) if at is not None
                             else None)
    return out


def _member(calc, finished, stage: str, once, copied: set
            ) -> Tuple[List[Copy], List[str]]:
    """``(its runs, copied; [])`` when ``stage`` is copied in, else ``([],
    what decided it)`` -- each stage it takes inputs from not copied in, and
    the finished run's state or each difference of its own deck."""
    from ..paths import attempt_dir, attempts_in
    from ..runrecord import launch_record, read_copied_from
    from ..transport.stages import (frames_of, inputs_of, points_in,
                                    products_of, rung_points)
    from .continuation import points_state, read_run, usable
    from .gather import ran_otherwise
    from .naming import stage_home
    # EVERY STAGE IT TAKES INPUTS FROM COPIED IN -- by this decision, or its
    # newest run here a copy, the run the gather takes (`job-system.md`,
    # *The task*, D5) -- said beside the rung's own differences.
    why: List[str] = []
    for upstream, filename in inputs_of(calc.task, stage):
        here = _newest(calc, upstream)
        if upstream not in copied and (here is None
                                       or read_copied_from(here) is None):
            why.append(f"{stage} takes {filename} from {upstream}, whose "
                       f"newest run here is not copied in")

    def no(said: str) -> Tuple[List[Copy], List[str]]:
        return [], why + [said]
    fbase, ftask = Path(finished.base), finished.task
    home = stage_home(fbase, ftask, stage)
    newest = _newest(finished, stage)
    if newest is None:
        return no(f"{stage}: the finished calculation holds no run of it")
    shown = f"{fbase.name}/{newest.relative_to(fbase)}"
    names = home.names(ftask.shape)
    points = rung_points(calc.task, stage, frames=frames_of(calc.base))
    theirs = rung_points(ftask, stage, frames=frames_of(fbase))
    if bool(points) != bool(theirs):
        # A RUN OF POINTS AND A RUN THAT RUNS ONCE are two shapes of run --
        # the treatment changed (§ 2a.10): never the same run.
        def words(pts):
            return ", ".join(p.words() for p in pts)
        return no(f"{stage}: {shown} "
                  + (f"holds the points {words(theirs)}" if theirs
                     else "ran once, at 0 V")
                  + "; this calculation runs it "
                  + (f"at the points {words(points)}" if points
                     else "once, at 0 V"))
    if points:
        # A RUNG OF POINTS: its newest run finished, holding exactly the
        # points this calculation describes, every one done.
        launch = launch_record(newest, names)
        if launch is None:
            return no(f"{stage}: {shown} was never launched")
        state, detail, rows = points_state(
            newest, names, ftask, stage, launch=launch,
            products=products_of(stage, ftask.label, base_dir=fbase),
            base=fbase)
        if state != "finished" or not all(r["done"] for r in rows):
            return no(f"{stage}: {shown} is {state} -- {detail}")
        held = {str(pt.rel): pt
                for _d, pt in points_in(newest, ftask, stage, base=fbase)}
        want = {str(pt.rel): pt for pt in points}
        if set(held) != set(want):
            return no(f"{stage}: {shown} holds the points "
                      f"{', '.join(p.words() for p in held.values())}; this "
                      f"calculation describes "
                      f"{', '.join(p.words() for p in want.values())}")
    else:
        _c, state, _v, detail = read_run(fbase, ftask, stage, newest,
                                         verdict=False)
        if not usable(state):
            return no(f"{stage}: {shown} is {state}"
                      + (f" -- {detail}" if detail else ""))
    # THE SAME CALCULATION as the deck this calculation renders for it now,
    # at each of its points (`engines/transport.md` § 6.1).
    for pt in (points or (None,)):
        where = newest / pt.rel if pt is not None else newest
        differs = ran_otherwise(once, stage, where, token=home.token,
                                point=pt)
        if differs:
            return no(f"{stage}: {fbase.name}/{where.relative_to(fbase)} "
                      f"ran another calculation -- {'; '.join(differs)}")
    if why:
        return [], why
    # EVERY RUN OF IT, under its own name.
    here = stage_home(Path(calc.base), calc.task, stage)
    return [_copy_of(finished, stage, attempt_dir(home.dir, n),
                     here.dir / attempt_dir(home.dir, n).name)
            for n in attempts_in(home.dir)], []


def _copy_of(finished, stage: str, source: Path, run: Path) -> Copy:
    """One run, as it will be copied: how it ended, what it holds -- its
    size, and the engine's scratch it leaves behind, named."""
    from ..warmfiles import is_scratch, never_carried
    from .continuation import stage_run_state
    state, _detail = stage_run_state(Path(finished.base), finished.task,
                                     stage, source)
    size, left = 0, []
    for f in _files(source, carried=False):
        if is_scratch(f.name):
            folder = f.parent.relative_to(source)
            left.append(f"{folder}/" if str(folder) != "." else f.name)
        elif not never_carried(f.name):
            size += f.stat().st_size
    return Copy(stage=stage, source=source, run=run,
                shown=(f"{Path(finished.base).name}/"
                       f"{source.relative_to(finished.base)}"),
                copied_from=origin(finished.base, source), ended=state,
                size=size, left=tuple(dict.fromkeys(left)))


def _files(run: Path, *, carried: bool = True) -> List[Path]:
    """Every file under ``run`` a copy carries, in a stable order -- never
    an engine's scratch or a file the catalogue calls transient
    (`warmfiles.never_carried`, the one answer every copier asks); with
    ``carried`` false, every file there."""
    from ..warmfiles import never_carried
    out: List[Path] = []
    for here, dirs, files in os.walk(run):
        dirs.sort()
        out.extend(Path(here) / f for f in sorted(files)
                   if not (carried and never_carried(f)))
    return out


def copy_in(calc, decision: Decision, plan) -> None:
    """A group copied in, into ``plan`` (`engines/transport.md` § 2a.16):
    for each member, a run of it prepared here and never launched removed
    first; then every run of it copied under its own name -- its files as
    they are, the engine's scratch left behind -- its ``.copied-from``, and
    its launch record last, so a copy cut short reads as a run never
    launched."""
    from ..paths import attempt_dir, attempts_in
    from ..runfiles import COPIED_FROM_FILE
    from ..runrecord import launch_record_path, write_copied_from
    from .materialize import empty_run
    from .naming import stage_home
    base, task = Path(calc.base), calc.task
    for stage in decision.members:
        home = stage_home(base, task, stage)
        for n in attempts_in(home.dir):
            opened = attempt_dir(home.dir, n)
            empty_run(opened, plan)
            plan.remove_folder(opened)
    for c in decision.copies:
        names = stage_home(base, task, c.stage).names(task.shape)
        record = launch_record_path(c.source, names, 0)
        for f in _files(c.source):
            # THE RUN'S OWN RECORDS are written for the copy, below; a
            # point's `shared/` keeps its own.
            if f in (record, c.source / COPIED_FROM_FILE):
                continue
            plan.copy(f, c.run / f.relative_to(c.source))
        write_copied_from(c.run, c.copied_from, plan=plan)
        plan.copy(record, c.run / record.name)


def takes_over(calc, finished, stage: str, points, *, once,
               inputs: Callable[[object], Dict[str, Optional[str]]]
               ) -> Takes:
    """THE POINTS A RUNG OF POINTS PREPARED HERE TAKES OVER at its first
    launch (`engines/transport.md` § 2a.16, *At prep*): each of ``points``
    the finished calculation's newest run of ``stage`` holds done -- a
    frame and a voltage, matched as such; a rung it ran once, its 0 V point
    on frame 0 -- that ran the same calculation as this rung's deck at that
    point (`gather.ran_otherwise`), and whose every input is the one that
    point was given: ``inputs(pt)``, each file's origin as this rung's
    point is given it (:func:`origin`; ``None`` for one computed here),
    against the origin of what the finished point gathered.  The rest are
    walked."""
    from .gather import ran_otherwise
    from .naming import stage_home
    fbase = Path(finished.base)
    token = stage_home(fbase, finished.task, stage).token
    taken: List[Take] = []
    walked: List[object] = []
    run = ""
    for pt in points:
        src, newest = _finished_point(finished, stage, pt)
        run = run or (f"{fbase.name}/{newest.relative_to(fbase)}"
                      if newest is not None else "")
        if (src is None
                or ran_otherwise(once, stage, src, token=token, point=pt)
                or not _given_the_same(finished, stage, newest, src,
                                       inputs(pt))):
            walked.append(pt)
            continue
        taken.append(Take(pt, src, f"{fbase.name}/{src.relative_to(fbase)}",
                          origin(fbase, src)))
    return Takes(stage, run, tuple(taken), tuple(walked))


def _given_the_same(finished, stage: str, run: Path, src: Path,
                    mine: Dict[str, Optional[str]]) -> bool:
    """Whether each file this rung's point is given has the origin of what
    the finished point ``src`` of ``stage``'s ``run`` was given -- what its
    run carries, its own record and, through `.continued-from`, what the
    runs it continues from gathered (`runrecord.carried_gather`: a device
    launched warm carries the seed's density its first launch gathered),
    and the point's own record (`runrecord.read_gathered_from`)."""
    from ..runrecord import carried_gather, read_gathered_from
    from .naming import stage_home
    fbase, ftask = Path(finished.base), finished.task
    names = stage_home(fbase, ftask, stage).names(ftask.shape)
    try:
        theirs = {fn: ref for ref, fn in carried_gather(fbase, run, names)}
        if src != run:
            theirs.update({fn: ref for ref, fn in read_gathered_from(src)})
    except LaunchRecordError:
        return False
    return (set(mine) == set(theirs)
            and all(o is not None and o == origin(fbase, theirs[fn].path(fbase))
                    for fn, o in mine.items()))


def take_into(calc, takes: Sequence[Take], points, plan, *,
              given: Sequence[str]) -> None:
    """The points taken over, into their folders of this run, ``points``
    ``[(folder, Point)]`` (`engines/transport.md` § 2a.16): each finished
    point's files -- the engine's and the run's own account -- but never
    its records of runs (``.continued-from``, ``.gathered-from``), its
    folder's stamp, a launch record, the engine's scratch, or the files
    the gather gave this point (``given``), which are this run's; then its
    ``.copied-from``."""
    from ..runfiles import (CALCDIR_FILE, CONTINUED_FROM_FILE,
                            COPIED_FROM_FILE, GATHERED_FROM_FILE,
                            LAUNCH_RECORD_FILE)
    from ..runrecord import write_copied_from
    from ..warmfiles import never_carried
    records = {CALCDIR_FILE, CONTINUED_FROM_FILE, COPIED_FROM_FILE,
               GATHERED_FROM_FILE, LAUNCH_RECORD_FILE, *given}
    from ..transport.energies import SHARED_DIR
    folder_of = {point_key(pt): pdir for pdir, pt in points}
    for t in takes:
        pdir = folder_of[point_key(t.point)]
        for f in sorted(t.source.iterdir()):
            if (f.is_file() and f.name not in records
                    and not never_carried(f.name)):
                plan.copy(f, pdir / f.name)
        # ITS SHARED ENERGIES, with it (§ 2a.16).
        carry_shared(t.source / SHARED_DIR, pdir / SHARED_DIR, plan,
                     root=calc.base)
        write_copied_from(pdir, t.copied_from, plan=plan)


def carry_shared(src: Path, dest: Path, plan, *, root) -> None:
    """A point's ``shared/`` carried whole into ``dest`` -- with the point
    taken over, warm or from the finished calculation, or into a run that
    walks the point again -- each folder of it stamped anew with its role
    where it now lies in the calculation at ``root``, its ``.copied-from``
    last, so a copy cut short reads as none (`engines/transport.md`
    § 2a.16); nothing where ``src`` holds no ``shared/``."""
    from .. import calcdirs
    from ..runfiles import CALCDIR_FILE, COPIED_FROM_FILE
    if not (src / COPIED_FROM_FILE).is_file():
        return
    for f in _files(src):
        if f.name == CALCDIR_FILE:
            plan.text(*calcdirs.record(dest / f.parent.relative_to(src),
                                       role=calcdirs.SHARED, root=root))
        elif f.name != COPIED_FROM_FILE or f.parent != src:
            plan.copy(f, dest / f.relative_to(src))
    plan.copy(src / COPIED_FROM_FILE, dest / COPIED_FROM_FILE)


def shared_into(calc, pdir: Path, pt, deck: Path, held: Optional[Path],
                plan) -> None:
    """WHAT A TRANSMISSION POINT SHARES, given it at a launch that walks it
    -- a rung that runs once, its one point (``None``), the run itself
    (`engines/transport.md` § 2a.16): nothing where ``deck``, the one its
    prep wrote, computes its window whole; else ``held`` -- the ``shared/``
    of the rung's newest run here at the point, when it holds them whole --
    or the finished point the deck names: its files (:func:`_shared_files`)
    and its own ``shared/``, the folder stamped with its role, its
    ``.copied-from`` last.  Refused, :class:`ValueError`, when neither is
    there."""
    from .. import calcdirs
    from ..projects import find_projects_root
    from ..runfiles import COPIED_FROM_FILE
    from ..runrecord import write_copied_from
    from ..transport.energies import SHARED_DIR, Contours
    contours = (Contours.of_deck(deck.read_text(encoding="utf-8",
                                                errors="replace"))
                if deck.is_file() else None)
    if contours is None:
        return
    dest = Path(pdir) / SHARED_DIR
    if held is not None and (held / COPIED_FROM_FILE).is_file():
        carry_shared(held, dest, plan, root=calc.base)
        return
    tree = Path(find_projects_root(Path(calc.base))).resolve()
    src = tree / contours.source
    files = _shared_files(src, calc.task.label)
    if not files:
        raise ValueError(
            f"{_words(pt)} shares energies with {contours.source}, which "
            f"holds none now, and no run of this calculation holds them -- "
            f"prepare the transmission again: it decides on what is there "
            f"(engines/transport.md § 2a.16)")
    plan.text(*calcdirs.record(dest, role=calcdirs.SHARED, root=calc.base))
    for f in files:
        plan.copy(f, dest / f.name)
    carry_shared(src / SHARED_DIR, dest / SHARED_DIR, plan, root=calc.base)
    # THE POINT IT IS A COPY OF, one hop -- read in the calculation that
    # point lies in, the finished one or one that one shared with.
    write_copied_from(dest, origin(calcdirs.where(src).root, src), plan=plan)


__all__ = ["Copy", "Take", "Takes", "Decision", "Reuse", "SHARING_RUNG",
           "finished_of", "finished_window", "origin",
           "point_key", "decide",
           "that_point", "copy_in", "takes_over", "take_into",
           "carry_shared", "shared_into"]
