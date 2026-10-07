"""molbuilder.pipeline_log — what the preparation pipeline received, decided
and produced, as one file a person can read top to bottom.

**The question it answers is *"how did this value get into this deck?"***
Every other record in a bundle answers a different one: ``STAGE-PLAN.md`` says
what the plan IS, ``jobset-decisions.log`` says which decisions a verb took,
the deck's own PROVENANCE block says what the writer assumed.  None of them
says where a number came FROM -- through which step, out of which file, folded
with what -- and that is the question asked when one rung of a ladder converges
and the next does not.

**One writer.**  The engines never call this; the framework calls it at three
points inside `script_emit` and the conductor (`jobset/prep`) writes the rest
from the values the steps already return.  That is the same layering
``jobset/ledger`` states: *library layers RETURN decision data and the SURFACE
that acted on it appends the line*.  A ``print`` per engine would be two
writers and, within a month, two formats.

**Every prep writes one**, from either door (`script-preparation.md` § 4.5).
A route that is not a prep -- a deck
rendered on its own -- holds ``None``, and ``None`` means the framework skips
the calls.  Nothing about a generated artifact depends on it -- the log
observes the pipeline, it is not a step in it.

**Held, and written with the plan.**  It keeps what it is told, and `prep`
puts its text in the plan beside every other file, written once the whole
plan stands -- so a refused prep writes no log, and a hook that raised says
whose it was in the error it raises (`issues.calling`; `script-preparation.md`
§ 4.5).

Readable, and greppable, in that order
--------------------------------------

A phase opens with a banner and its events are indented under it, so a phase
boundary is visible at a glance and the file can be scanned without a tool.
Within a phase every line says what it is in its first column:

* ``in``  — received, and from where
* ``⊕``   — a value being decided, with its source
* ``out`` — produced, and what it was passed to

so ``grep '^  ⊕'`` still gives the whole derivation in one pass, and
``grep '^  in'`` gives every input the run read.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Mapping

from .runfiles import compose as _compose

#: How a log file is named.  The STEM is the deck's own -- ``<label>_<token>``
#: -- because that is what tells two rungs apart **in a flat calculation**,
#: where every stage preps into one directory and a per-calculation name would
#: have `tight` overwrite `coarse` (`job-contracts.md` § 6.3: in flat the name
#: does the selecting).  Engine and shape follow so the file still says what it
#: is once it has been copied off the machine.
LOG_SUFFIX = "pipeline.log"

_MAJOR = "═" * 71
_MINOR = "─" * 71

#: Column widths.  Fixed, so the ``in`` / ``⊕`` / ``out`` columns line up down
#: the whole file and a reader's eye can follow one of them.
_W_KIND, _W_NAME, _W_VALUE = 5, 22, 24


def log_name(label: str, token: str, engine: str, shape: str) -> str:
    """This prep's log file name.

    ``token`` is the stage's ``<NN>_<name>`` -- every description has a
    stage, so an empty one is refused.

    THE NAME IS THE CATALOGUE'S (`runfiles.WRITTEN`, job-contracts.md
    § 2.2): the role ``.{engine}.{shape}.pipeline.log`` on the
    ``<label>_<token>`` every run file is named from, its two facts about
    the prep declared fields -- so the file is a row like every other file
    molbuilder writes, and `--cold` and the cards know it.
    """
    if not token:
        raise ValueError("a prep's log is its stage's -- no stage token was "
                         "given")
    return _compose(label, "." + "{engine}.{shape}." + LOG_SUFFIX,
                    token, engine=engine, shape=shape)


class PipelineLog:
    """The one writer.  Seven verbs, and nothing that decides anything.

    Nothing is written as it is told -- the lines are kept, and
    :meth:`held_text` is what `prep` puts in its plan, written with every
    other file once the whole plan stands (`job-system.md` § 5.0, rule 3: a
    refusal writes nothing but its line in the ledger)."""

    def __init__(self, path):
        self.path = Path(path)
        self._held: List[str] = []

    # -- opening ------------------------------------------------------- #

    @classmethod
    def open(cls, record_dir, *, label: str, token: str, engine: str,
             shape: str) -> "PipelineLog":
        """A log for one prep, in ``record_dir``.

        ``record_dir`` is the bundle root for a run and the stage's ``bench/``
        container for a sweep -- the same home ``STAGE-PLAN.md`` uses, so a
        benchmark's log can never overwrite the run's (`prep.prep_jobset`'s
        ``plan_dir``).  **Inside the bundle**, because the bundle is what is
        still there when a job misbehaves on a cluster hours later
        (``jobset/ledger``'s stated reason for the same choice).
        """
        self = cls(Path(record_dir) / log_name(label, token, engine, shape))
        self._banner(label, token, engine, shape)
        return self

    def held_text(self) -> str:
        """What the log has been told, as the text its file gets."""
        return "".join(line + "\n" for line in self._held)

    def _banner(self, label, token, engine, shape) -> None:
        at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rung = f"stage {token}" if token else "no ladder"
        self._write("\n" + _MAJOR)
        self._write(f"  PREP  {label} · {rung} · {engine} · {shape} · {at}")
        self._write(_MAJOR)

    # -- the seven verbs ------------------------------------------------ #

    def phase(self, title: str) -> None:
        """One of the five steps — the heavy banner."""
        self._write("")
        self._write(_MAJOR)
        self._write(f"  {title}")
        self._write(_MAJOR)

    def step(self, title: str) -> None:
        """A sub-step within a step — the light banner."""
        self._write("")
        self._write(_MINOR)
        self._write(f"  {title}")
        self._write(_MINOR)

    def received(self, name: str, detail: Any = "") -> None:
        self._row("in", name, detail)

    def chose(self, name: str, value: Any, source: str) -> None:
        """One value, and WHERE it came from — the row this file exists for."""
        v = _flat(value)
        pad = max(_W_VALUE - len(v), 1)
        self._row("⊕", name, f"{v}{' ' * pad}<- {source}")

    def produced(self, name: str, detail: Any = "") -> None:
        self._row("out", name, detail)

    def text(self, body: str) -> None:
        """A block of someone else's rendering, indented under this phase.

        For records that already have ONE formatter -- the config provenance
        table above all.  Re-laying it out here would be a second spelling of
        a table whose whole point is that it has one
        (`runtime_config.format_provenance`).
        """
        for line in str(body).splitlines():
            self._write(f"       {line}" if line.strip() else "")

    def note(self, message: str) -> None:
        """A sentence about the phase itself — a finding, or an absence."""
        self._write(f"       {message}")

    def close(self) -> None:
        self._held.append("")

    # -- the only two places that write a line -------------------------- #

    def _row(self, kind: str, name: str, detail: Any) -> None:
        d = _flat(detail)
        # A GUARANTEED GAP, not a hoped-for one: a name wider than a
        # hardcoded column would print `environment/home/...` with nothing
        # between them, in a display whose whole job is making the source
        # legible.  A long field name here
        # pushes its value right; it never runs into it.
        self._write(
            f"  {kind:<{_W_KIND}}{str(name):<{_W_NAME}} {d}".rstrip())

    def _write(self, line: str) -> None:
        self._held.append(line)


def _flat(v: Any) -> str:
    """A value as one line.  Tuples read as tuples; nothing is truncated."""
    if v is None:
        return "None"
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)):
        return "(" + ", ".join(_flat(x) for x in v) + ")"
    return str(v)


def config_rows(values: Any, provenance: Mapping[str, str],
                folded: Any = None) -> "list[tuple[str, Any, str]]":
    """``[(field, value, source)]`` for one resolved element, decisions first.

    ``provenance`` is ``ResolvedConfig.provenance`` -- built by ``resolve`` on
    every element.  ``folded`` is ``render_config()``: the fields it
    changes are the ALLOCATION's, which no config file names and which are
    therefore the ones a reader is most likely to be surprised by.

    Sorted so the fields this rung DECIDED come first and the template's
    baseline follows: the first block is normally three to ten lines and is
    the answer to *"why is it 300 here and 350 there"*.
    """
    rank = {"pin": 0, "sweep": 1, "stage": 2, "role": 3, "allocation": 4,
            "template": 5}
    rows = []
    for f in dataclasses.fields(values):
        src = provenance.get(f.name, "template")
        value = getattr(values, f.name, None)
        if folded is not None:
            after = getattr(folded, f.name, value)
            if after != value:
                value, src = after, "allocation"
        rows.append((f.name, value, src))
    return sorted(rows, key=lambda r: (rank.get(r[2], 9), r[0]))


__all__ = ["PipelineLog", "log_name", "config_rows", "LOG_SUFFIX"]
