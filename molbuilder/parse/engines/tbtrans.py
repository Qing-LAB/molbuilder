"""TBtrans's loop, in words -- and the transmission files it wrote.

`model/parse.md` § 5d.5.  TBtrans's output is read by the SIESTA family's one
reading pass (`siesta_reader.SiestaReader`, whose ``now()["tbtrans"]`` is its
account of its k-point × energy loop: its size, the spin passes it makes,
each pass's estimate while it runs and its time once it completed, its
currents); this module says that loop in the one sentence the monitor's
summary line, the run record and the transport report give
(:func:`loop_words`), and finds the transmission files per spin channel
(:func:`transmission_files`).

Not a registered viewer file: the transmission rung's result is the transport
record (`engines/transport.md` § 2a.12).

**Stdlib only, and it travels beside every job** (`runwrap.MONITOR_COMPANIONS`),
for the monitor's summary line: imported from the package here, from its copy
there.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

try:                                        # inside molbuilder
    from . import siesta_grammar as _G
    from ...report_fields import pct_words, span_words
except ImportError:                         # beside a job, as the monitor's
    import siesta_grammar as _G
    from report_fields import pct_words, span_words


#: A pass's channel, in words -- one for each of `siesta_grammar.TBT_CHANNELS`'
#: names, which this module refuses to load without.
CHANNEL_WORDS: Dict[str, str] = {"up": "spin up", "down": "spin down",
                                 "unpolarized": "unpolarized"}
if set(CHANNEL_WORDS) != {ch for ch, _tag in _G.TBT_CHANNELS}:
    raise ValueError(
        f"tbtrans.CHANNEL_WORDS names {sorted(CHANNEL_WORDS)}, and "
        f"siesta_grammar.TBT_CHANNELS "
        f"{sorted(ch for ch, _tag in _G.TBT_CHANNELS)}: a channel each")


def _count(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def loop_words(loop: Optional[Dict[str, Any]], *, over: bool = False) -> str:
    """TBtrans's loop -- the reading pass's ``now()["tbtrans"]`` -- in the
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


def transmission_files(directory, label: str) -> Dict[str, List[Path]]:
    """``{channel: [AVTRANS file, ...]}`` -- the unpolarized file, or the two
    channels of a polarized run (``siesta_grammar.TBT_CHANNELS``)."""
    out: Dict[str, List[Path]] = {}
    for channel, tag in _G.TBT_CHANNELS:
        files = sorted(Path(directory).glob(f"{label}{tag}AVTRANS_*"))
        if files:
            out[channel] = files
    return out
