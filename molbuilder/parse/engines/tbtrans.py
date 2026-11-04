"""TBtrans's loop, in words -- and the transmission files it wrote.

`model/parse.md` § 5d.5.  TBtrans's output is read by the SIESTA family's one
reading pass (`siesta_reader.SiestaReader`, whose ``now()["loop"]`` is its
account of its k-point × energy loop: its size, the spin passes it makes,
each pass's estimate while it runs and its time once it completed, its
currents); this module says that loop in the one sentence the monitor's
summary line, the run record and the transport report give
(:func:`loop_words`), and finds and reads TBtrans's k-averaged text files
per spin channel (:func:`averaged_files`, :func:`read_averaged`; § 5e).
Its data file is :mod:`tbtrans_nc`'s.

Not a registered viewer file: the transmission rung's result is the transport
record (`engines/transport.md` § 2a.12).

**Stdlib only, and it travels beside every job** (`runwrap.monitor_companions`),
for the monitor's summary line: imported from the package here, from its copy
there.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

try:                                        # inside molbuilder
    from . import siesta_grammar as _G
    from ... import runfiles as _rf
    from ...report_fields import pct_words, span_words
except ImportError:                         # beside a job, as the monitor's
    import siesta_grammar as _G
    import runfiles as _rf                  # type: ignore[no-redef]
    from report_fields import pct_words, span_words

#: WHAT THIS MODULE NEEDS BESIDE IT in a job's bundle, which it travels in by
#: the SIESTA-family output's catalogue row (`runfiles.WRITTEN`'s ``reads``,
#: ``loop_words``; `runwrap.monitor_companions`).
TRAVELS_WITH = ("molbuilder.parse.dirs.job",
                "molbuilder.report_fields",
                "molbuilder.runfiles",
                "molbuilder.parse.engines.siesta_grammar",
                "molbuilder.parse.engines.siesta_reader")


#: A pass's channel, in words -- one for each of `siesta_grammar.TBT_CHANNELS`'
#: names, which this module refuses to load without.
CHANNEL_WORDS: Dict[str, str] = {"up": "spin up", "down": "spin down",
                                 "unpolarized": "unpolarized"}
if set(CHANNEL_WORDS) != {ch for ch, _tag in _G.TBT_CHANNELS}:
    raise ValueError(
        f"tbtrans.CHANNEL_WORDS names {sorted(CHANNEL_WORDS)}, and "
        f"siesta_grammar.TBT_CHANNELS "
        f"{sorted(ch for ch, _tag in _G.TBT_CHANNELS)}: a channel each")
#: TBtrans's K-AVERAGED TEXT FILES, by the quantity each holds, as the
#: catalogue names them (`runfiles.ENGINE_FILES`): written beside the data
#: file by ``state_cdf2ascii`` from the data file's own variables, averaged
#: over the k-points with its weights (``m_tbt_save.F90`` ``name_save``;
#: `model/parse.md` § 5e) -- the transmission, its eigenchannels, the
#: device's DOS, the spectral DOS of a lead, the bulk DOS of a lead -- each
#: written when the deck asks for it.
AVERAGED = {"transmission": _rf.TBTRANS_AVTRANS,
            "eigenchannels": _rf.TBTRANS_AVTEIG,
            "dos": _rf.TBTRANS_AVDOS,
            "lead_spectral": _rf.TBTRANS_AVADOS,
            "lead_bulk": _rf.TBTRANS_AVBDOS}


def is_tbtrans_output(head_lines) -> bool:
    """Whether an output's head is TBtrans's -- the SIESTA family's program
    line (`siesta_grammar.program_of`, `model/parse.md` § 5d.2)."""
    return _G.program_of(head_lines) == "tbtrans"


def _count(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def loop_words(loop: Optional[Dict[str, Any]], *, over: bool = False) -> str:
    """TBtrans's loop -- the reading pass's ``now()["loop"]`` -- in the
    one sentence every surface says it in (`model/parse.md` § 5d.5); ``""``
    when the output states none of it.  ``over``: the run is over, by the
    run door's state -- a pass still underway then stopped, and an estimate
    is said only while it stands.

    Each pass is named by its channel and its place among the passes the
    set-up names -- *spin up (1 of 2)*; a time by the span rule
    (`report_fields.span_words`).  A pass still to come is named, never
    timed: TBtrans's estimate covers the pass underway, and only a pass
    already done is measured."""
    if not loop:
        return ""
    planned = list(loop.get("channels") or [])
    passes = list(loop.get("passes") or [])

    def name(i: int, channel: Optional[str]) -> str:
        word = CHANNEL_WORDS.get(channel or "", "pass")
        return (f"{word} ({i} of {len(planned)})" if planned
                else f"{word} (pass {i})")

    done = [(i, p) for i, p in enumerate(passes, start=1)
            if "loop_elapsed_s" in p]
    underway = (len(passes), passes[-1]) if (
        passes and "loop_elapsed_s" not in passes[-1]) else None
    to_come = planned[len(passes):]
    completed = "; ".join(f"{name(i, p.get('channel'))} completed in "
                          f"{span_words(p['loop_elapsed_s'])}" for i, p in done)

    if not passes:
        size = [f"{_count(loop['k_points'], 'k-point', 'k-points')} × "
                f"{_count(loop['energy_points'], 'energy', 'energies')}"
                if "k_points" in loop and "energy_points" in loop else "",
                (f"{_count(len(planned), 'pass', 'passes')} ("
                 + ", ".join(CHANNEL_WORDS.get(c, c) for c in planned) + ")")
                if planned else ""]
        size = ", ".join(s for s in size if s)
        if over:
            return "stopped while setting up its loop" + (
                f" of {size}" if size else "")
        return "setting up its loop" + (f": {size}" if size else "")

    if underway is not None:
        i, p = underway
        if over:
            said = (f"stopped during {name(i, p.get('channel'))} at "
                    f"{pct_words(p['done_pct'])} %" if "done_pct" in p else
                    f"stopped during {name(i, p.get('channel'))}, before "
                    f"its first estimate")
        else:
            said = (f"{name(i, p.get('channel'))}: "
                    + (f"{pct_words(p['done_pct'])} % done"
                       + (f", about {span_words(p['eta_s'])} left "
                          f"(TBtrans's estimate)" if "eta_s" in p else "")
                       if "done_pct" in p else "begun, no estimate yet")
                    + ("" if not to_come else ", then "
                       + ", then ".join(CHANNEL_WORDS.get(c, c)
                                        for c in to_come)))
        return said + (f"; {completed}" if completed else "")

    if to_come:
        nxt = name(len(passes) + 1, to_come[0])
        return completed + (f"; stopped before {nxt}" if over
                            else f"; {nxt} next")
    return "completed: " + ", ".join(
        f"{name(i, p.get('channel'))} in {span_words(p['loop_elapsed_s'])}"
        for i, p in done)


#: How far a k-averaged text file's energies, printed to five decimals of an
#: eV (``f10.5``, ``m_tbt_save.F90`` ``save_DAT``), may be from the data
#: file's own, at full precision: half that last decimal, and a
#: picoelectronvolt for the Ry/eV conversion's rounding -- an energy whose
#: sixth decimal is 5 lies half a decimal from its print exactly -- asked
#: wherever the two are matched (`model/parse.md` § 5e).
PRINTED_HALF_DECIMAL_EV = 0.5e-5 + 1e-12


def averaged_files(directory, label: str, *, pair: str,
                   leads: "tuple[str, ...]") -> Dict[str, Dict[str, Any]]:
    """TBtrans's k-averaged text files in ``directory`` (`model/parse.md`
    § 5e), per channel -- the unpolarized one, or the two of a polarized run
    (``siesta_grammar.TBT_CHANNELS``) -- by quantity (:data:`AVERAGED`):
    ``{channel: {"transmission": path, "eigenchannels": path, "dos": path,
    "lead_spectral": {lead: path}, "lead_bulk": {lead: path}}}``, each one
    that is there; ``pair`` the leads as their ``TS.Elecs`` blocks name them
    (``L-R``) and ``leads`` each of them, the caller's -- never the first of
    a glob -- found on the label through the catalogue
    (`runfiles.find_on_label`)."""
    out: Dict[str, Dict[str, Any]] = {}
    for channel, tag in _G.TBT_CHANNELS:
        got: Dict[str, Any] = {}
        for quantity, role in AVERAGED.items():
            if "{lead}" in role:
                per = {}
                for lead in leads:
                    hits = _rf.find_on_label(directory, label, _rf.tail(
                        role, tbt=tag, lead=lead))
                    if hits:
                        per[lead] = hits[-1]
                if per:
                    got[quantity] = per
            else:
                fields = {"pair": pair} if "{pair}" in role else {}
                hits = _rf.find_on_label(directory, label, _rf.tail(
                    role, tbt=tag, **fields))
                if hits:
                    got[quantity] = hits[-1]
        if got:
            out[channel] = got
    return out


def read_averaged(path) -> Dict[str, List]:
    """One of TBtrans's k-averaged text files (`model/parse.md` § 5e) --
    ``#`` lines, then a row per energy: the energy in eV, as printed (five
    decimals), then the file's values -- as ``{"energy_ev": [...],
    "columns": [[...], ...]}``, a list per value column, in the file's
    order.  :class:`ValueError` names a file with no row, or a row that is
    not the first one's width."""
    energies: List[float] = []
    columns: List[List[float]] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for n, line in enumerate(fh, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            try:
                values = [float(x) for x in parts]
            except ValueError:
                raise ValueError(f"{Path(path).name}, line {n}: {s!r} is "
                                 f"not a row of numbers") from None
            if not columns:
                columns = [[] for _ in values[1:]]
            if len(values) != len(columns) + 1:
                raise ValueError(f"{Path(path).name}, line {n}: "
                                 f"{len(values)} values, the first row "
                                 f"{len(columns) + 1}")
            energies.append(values[0])
            for col, v in zip(columns, values[1:]):
                col.append(v)
    if not energies:
        raise ValueError(f"{Path(path).name} holds no row")
    return {"energy_ev": energies, "columns": columns}


#: How much of an output's beginning says which program wrote it
#: (`siesta_grammar.program_of`) -- the build banner's first lines.
_HEAD_BYTES = 4 * 1024


def loop_record(out: Optional[Path], *, over: bool,
                head: Optional[str] = None,
                now: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """TBtrans's account of its loop, as its output states it --
    ``{progress, estimated_at}`` (`model/parse.md` § 5d.1b, § 5d.5) -- the
    run record's and the transport report's: the loop
    in its one sentence (:func:`loop_words`), said as over or not by the
    run's state, ``over``; and, while a pass's estimate stands, when the
    output last changed -- TBtrans writes a progress line, then nothing until
    the next.  ``{}`` for an output that is not TBtrans's
    (`siesta_grammar.program_of`, over ``head``, its head -- the caller's
    when it holds it) or that states none of it.

    Read whole, through the family's one read (`siesta_reader.read_output`):
    a TBtrans output is its loop's own account, a line every
    ``TBT.Progress`` per cent, and a tail could miss a pass's start or its
    end.  A caller that has read the output already hands its reading over
    as ``now`` (`SiestaReader.now`), and it is not read again."""
    try:                                    # inside molbuilder
        from ..dirs.job import _iso_z
        from .siesta_reader import read_output
    except ImportError:                     # beside a job, as the monitor's
        from job import _iso_z
        from siesta_reader import read_output
    if out is None:
        return {}
    if now is None:
        if head is None:
            try:
                with open(out, "rb") as fh:
                    head = fh.read(_HEAD_BYTES).decode("utf-8",
                                                       errors="replace")
            except OSError:
                return {}
        if not is_tbtrans_output(head.splitlines()):
            return {}
        try:
            now = read_output(out).now()
        except OSError:
            return {}
    said = loop_words(now.get("loop"), over=over)
    if not said:
        return {}
    got: Dict[str, Any] = {"progress": said}
    if not over and "done_pct" in now:
        try:
            got["estimated_at"] = _iso_z(Path(out).stat().st_mtime)
        except OSError:
            pass
    return got
