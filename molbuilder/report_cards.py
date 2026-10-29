"""The chat card's layout, per receiver -- read from its data file
(`execution/run-reports.md` § 4.1c).

ONE file says how the one picture (`monitor._card`) is laid out for each
receiver -- Slack, Discord, the This machine tab: whether the fields sit side
by side, how many at most, whether the summary line shows, and which fields a
card shows for each event, in what order.  The title stays the code's.

**Stdlib only, and it travels beside every job** (`runwrap.MONITOR_COMPANIONS`,
with the data file as the bundle's one data member, `runwrap.MONITOR_DATA`):
the monitor reads it there, as `config_dir` and `report_fields` travel.  The
reader finds the file by the import's two ways: inside the package in
`molbuilder/data/`, beside the job beside itself in the bundle -- read through
``__loader__.get_data``, which serves a zip member and a plain file alike.

A copy at ``<config dir>/report-cards.json`` (:func:`layout_path`) replaces
the shipped file whole on that machine.  A file that does not read is refused
(:class:`LayoutError`) with its reason; nothing is guessed at.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:                                        # inside molbuilder
    from . import report_fields as _fields
    from .config_dir import config_dir as _config_dir
    _HERE = Path(__file__).parent / "data"
except ImportError:                         # beside the job
    import report_fields as _fields
    from config_dir import config_dir as _config_dir
    _HERE = Path(__file__).parent

#: The file's name -- spelled here, in the module that owns it
#: (`configuration.md` § 3.1), and nowhere else.
FILENAME = "report-cards.json"
#: The schema a file must declare.
SCHEMA = "molbuilder/report-cards@1"
#: The shipped file: the package's `data/` copy, or the bundle member.
SHIPPED = _HERE / FILENAME

#: The receivers a layout must lay out: the two chat wires of
#: `run-reports.md` § 4.1b and the This machine tab.
RECEIVERS: Tuple[str, ...] = ("slack", "discord", "web")
#: The events a card is sent for (`run-reports.md` § 2) and the setup page's
#: Test button (§ 4.1b).
EVENTS: Tuple[str, ...] = ("start", "scf_converged", "periodic", "wall_near",
                           "milestone", "part_failed", "finish", "test")
#: A wire's own ceiling on fields (`run-reports.md` § 4.1b): the most a
#: receiver's ``limit`` may say.  The web tab has none.
WIRE_LIMIT: Dict[str, int] = {"slack": 10, "discord": 25}
#: The colour of a state the table does not name.
OTHER = "other"

_COLOR_RE = re.compile(r"#[0-9A-Fa-f]{6}")


class LayoutError(ValueError):
    """The layout file does not read: its path and the reason."""


@dataclass(frozen=True)
class Receiver:
    """One receiver's layout."""
    #: The fields side by side (Slack's ``short``, Discord's ``inline``, the
    #: tab's grid three abreast) or each on its own line.
    inline: bool
    #: How many fields at most; ``0`` is no limit.
    limit: int
    #: Whether the summary line shows under the title.
    summary: bool
    #: The fields a card shows for each event, in order; an event with no
    #: row is laid out in the declaration's order, whole.
    order: Dict[str, Tuple[str, ...]]


@dataclass(frozen=True)
class Layout:
    """The whole layout, read and checked."""
    #: The card's bar by the report's state, as a number (``0xRRGGBB``).
    colors: Dict[str, int]
    receivers: Dict[str, Receiver]
    #: Where it was read from, for a message.
    source: str

    def color(self, state: str) -> int:
        """The bar's colour for ``state``; the table's ``other`` otherwise."""
        return self.colors.get(state, self.colors[OTHER])

    def receiver(self, kind: str) -> Receiver:
        """The receiver's layout; a name the file does not lay out is an
        error of the caller's, said by name."""
        try:
            return self.receivers[kind]
        except KeyError:
            raise LayoutError(f"{self.source}: no receiver {kind!r} -- the "
                              f"receivers are {', '.join(RECEIVERS)}") from None

    def fields(self, kind: str, report: Dict[str, Any],
               items: Optional[Sequence[str]] = None
               ) -> List[Tuple[str, str]]:
        """The card's fields for ``report`` at ``kind``: the receiver's order
        for the report's event, narrowed by ``items`` (the calculation's
        ``notify.report`` -- a ceiling, never a floor, `stages.md` § 6.9),
        then by what the run stated (an absent field stays absent) and by
        the part underway (a field whose part has ended is the summary
        line's, said there as ended -- `run-reports.md` § 2.8), then cut at
        the receiver's limit.  Each as ``(label, value said)``."""
        rc = self.receiver(kind)
        order = rc.order.get(str(report.get("event") or ""))
        names = _fields.NAMES if order is None else order
        out: List[Tuple[str, str]] = []
        for name in names:
            if rc.limit and len(out) >= rc.limit:
                break
            if items is not None and name not in items:
                continue
            val = report.get(name)
            if val is None or val == "":
                continue                 # absent stays absent -- unknown is not 0
            f = _fields.BY_NAME[name]
            if f.ended(report.get("part")):
                continue
            out.append((f.card, f.said(val)))
        return out


def layout_path() -> Path:
    """Where this machine's copy lives: ``<config dir>/report-cards.json``
    (`configuration.md` § 3.1)."""
    return _config_dir() / FILENAME


def shipped_text() -> str:
    """The shipped file's text, through the loader that brought this module
    -- a zip member in the bundle, a file in the package."""
    return __loader__.get_data(str(SHIPPED)).decode("utf-8")  # type: ignore[name-defined]


def load(path: Optional[Path] = None) -> Layout:
    """The layout in force here: the copy at :func:`layout_path` when there
    is one, else the shipped file.  ``path`` names a file to read instead.
    Refused (:class:`LayoutError`) with the path and the reason."""
    if path is None:
        copy = layout_path()
        if copy.is_file():
            path = copy
    if path is not None:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise LayoutError(f"{path}: cannot be read -- {exc}") from None
        return parse(text, str(path))
    return parse(shipped_text(), str(SHIPPED))


def parse(text: str, source: str) -> Layout:
    """``text`` as a :class:`Layout`, every rule of `run-reports.md` § 4.1c
    checked; ``source`` names the file in a refusal."""
    def refuse(why: str) -> LayoutError:
        return LayoutError(f"{source}: {why}")

    try:
        doc = json.loads(text)
    except ValueError as exc:
        raise refuse(f"not JSON -- {exc}") from None
    if not isinstance(doc, dict):
        raise refuse("the file is not a JSON object")
    if doc.get("schema") != SCHEMA:
        raise refuse(f"schema {doc.get('schema')!r}, not {SCHEMA!r}")

    colors_in = doc.get("colors")
    if not isinstance(colors_in, dict) or OTHER not in colors_in:
        raise refuse(f"'colors' must be an object naming {OTHER!r} among "
                     f"its states")
    colors: Dict[str, int] = {}
    for state, hexa in colors_in.items():
        if not isinstance(hexa, str) or not _COLOR_RE.fullmatch(hexa):
            raise refuse(f"colors.{state} is {hexa!r}, not '#RRGGBB'")
        colors[str(state)] = int(hexa[1:], 16)

    recv_in = doc.get("receivers")
    if not isinstance(recv_in, dict):
        raise refuse("'receivers' must be an object")
    for kind in RECEIVERS:
        if kind not in recv_in:
            raise refuse(f"no layout for the receiver {kind!r}")
    receivers: Dict[str, Receiver] = {}
    for kind, spec in recv_in.items():
        if kind not in RECEIVERS:
            raise refuse(f"{kind!r} is not a receiver; the receivers are "
                         f"{', '.join(RECEIVERS)}")
        receivers[kind] = _receiver(kind, spec, refuse)
    return Layout(colors=colors, receivers=receivers, source=source)


def _receiver(kind: str, spec: Any, refuse) -> Receiver:
    at = f"receivers.{kind}"
    if not isinstance(spec, dict):
        raise refuse(f"{at} must be an object")
    for key in ("inline", "summary"):
        if not isinstance(spec.get(key), bool):
            raise refuse(f"{at}.{key} must be true or false")
    limit = spec.get("limit")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise refuse(f"{at}.limit must be a count; 0 is no limit")
    cap = WIRE_LIMIT.get(kind)
    if cap is not None and (limit == 0 or limit > cap):
        raise refuse(f"{at}.limit is {limit}{' (no limit)' if limit == 0 else ''}; "
                     f"{kind} takes {cap} fields at most (run-reports.md 4.1b)")
    order_in = spec.get("order")
    if not isinstance(order_in, dict):
        raise refuse(f"{at}.order must be an object of event -> fields")
    order: Dict[str, Tuple[str, ...]] = {}
    for event, names in order_in.items():
        if event not in EVENTS:
            raise refuse(f"{at}.order.{event}: not an event; the events are "
                         f"{', '.join(EVENTS)}")
        if not isinstance(names, list) or not all(
                isinstance(n, str) for n in names):
            raise refuse(f"{at}.order.{event} must be a list of field names")
        for n in names:
            if n not in _fields.NAMES:
                raise refuse(f"{at}.order.{event} names {n!r}, which is not "
                             f"a report field; the fields are "
                             f"{', '.join(_fields.NAMES)}")
        if len(set(names)) != len(names):
            raise refuse(f"{at}.order.{event} names a field twice")
        order[event] = tuple(names)
    shown = {n for names in order.values() for n in names}
    unseen = [n for n in _fields.NAMES if n not in shown]
    if unseen:
        raise refuse(f"{at} shows {', '.join(unseen)} at no event -- a tick "
                     f"nobody can ever see")
    return Receiver(inline=spec["inline"], limit=limit,
                    summary=spec["summary"], order=order)
