"""``task.json`` — a calculation's description on disk, and its one reader.

Contract: ``docs/engines/stages.md`` § 6.  This module is the *only* place
that turns those bytes into objects and back, which is what makes
*"the browser writes the same bytes as the CLI"* checkable rather than
coincidental (§ 6.4).

WHAT A DESCRIPTION IS.  One calculation: which engine, which layout, what it
is a calculation *of*, which settings the user chose to tune (``varies``), and
the per-stage cells that differ (``stages[].overrides``).

**It carries only what CHANGES.**  Everything that does not is in the template
(``<label>.template.toml``), written once — so there is no ``base`` key here, and a
stage that omits a varied field renders with the template's value (§ 4).  It
**names no machine** either: ranks, queues and walltimes are decided by ``prep``
on the target (``execution/project-layout.md`` § 2.1).

LAYER.  L1: it imports ``persist``, ``identity`` and the standard library, and
nothing else — both of those are themselves L1 on stdlib, so this stays a leaf.
That is deliberate rather than incidental — see *the split preflight* below.
``identity`` joined the list on 2026-08-09, when ``run.id`` stopped being a free
string and became something this module DERIVES and checks; it is the same
normaliser the CLI and the browser call, which is what § 3 rule 1's *"it happens
once"* actually requires.

THE SPLIT PREFLIGHT.  § 6.6 lists eight checks "in order, and all of it before
anything is written".  Four of them are answerable from the file alone and are
enforced here; the other four need the engine's field schema, and importing an
engine into an L1 codec is exactly what ``tests/test_layering.py`` prevents.
Those belong to resolution (P2 of ``execution/staged-runs-implementation-plan.md``),
which already has the schema in hand:

  here    the schema string's major · ``shape`` present and legal · stage names
          in ``[A-Za-z0-9_]+`` and unique case-insensitively · no ``overrides``
          key naming a stage field · ``overrides`` a SUBSET of ``varies`` ·
          unknown keys refused by name · ``run.id`` is what ``run.name`` and
          ``structure.formula`` derive (§ 6.1's first rule -- the check needs
          only ``identity``, which is L1 beside this one)
  P2      the engine has a generator · the schema fingerprint matches · every
          named field exists in the schema · every value is inside its bounds ·
          ``shape: "hierarchical"`` on an engine whose ladder runs in ONE
          process (§ 6.7 — PySCF; refused naming the engine) · and § 6.6a's
          warning for two stages that RESOLVE identically, since resolving is
          P2's verb

WHY THE MESSAGES NAME THINGS.  A description is JSON sitting beside the decks,
and as of 2026-08-07 editing it by hand is **supported** (the plan's decision 3).
So a refusal owes a person what it owes the browser: the offending key by name,
which stage it was in, and — where the key is one edit away from a real one —
what they probably meant.
"""
from __future__ import annotations

import contextvars
import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, NoReturn, Optional, Tuple

from .identity import normalise_id, run_id
from .persist import check_schema, read_json, write_json


SCHEMA = "molbuilder/task@1"
FILENAME = "task.json"

#: § 6.7 — required, no default, never inferred.
SHAPES = ("flat", "hierarchical")

#: § 2 — "three fields, and no others".  An ``overrides`` map naming one of
#: these would be a stage redefining what a stage is.
STAGE_FIELDS = ("name", "enabled", "overrides")

#: § 6.6 — a stage name becomes a filename, so the set is the narrow one.
STAGE_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")

_TOP_KEYS = ("schema", "engine", "shape", "run", "schema_fingerprint",
             "structure", "varies", "stages", "calculation")
_RUN_KEYS = ("name", "id", "created")
_STRUCTURE_KEYS = ("source", "formula", "atoms")


# --------------------------------------------------------------------- #
#  The shapes                                                           #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class Run:
    """What the user called it, and what it is called on disk.

    ``id`` is derived once and then quoted everywhere
    (``execution/run-identity.md`` § 2).  Since 2026-08-07 it does **not**
    name the directory — the level-③ folder is typed by the user, and this
    file is what says which calculation lives there (§ 3.0 there).

    Since 2026-08-09 (§ 2.0a, decision 26) it does not name the **files**
    either: the id is ``<label>_<formula>`` and lives here, while the label
    alone is the ``SystemLabel`` and the stem of everything on disk.  The
    label is not a field — it is :attr:`Task.label`, derived through the one
    normaliser — because storing it would be a second place for the same
    string to be wrong.  What makes deriving it safe is that ``id`` **is**
    stored and :meth:`Task.__post_init__` checks the two against each other.

    ``id`` is not defaulted.  A ``Run`` that filled it in from ``name`` would
    be a second deriver, and § 3 rule 1 is that there is one; use
    :func:`derive_run`."""
    name: str
    id: str
    created: str = ""


@dataclass(frozen=True)
class StructureRef:
    """§ 6.3 — a reference plus a witness, never a copy.

    ``source`` points into the project tree; ``formula`` and ``atoms`` record
    what was there when the description was written, so a description opened
    against a structure that has since changed can *say so* instead of
    silently building a different calculation under the same id."""
    source: str
    formula: str = ""
    atoms: int = 0


@dataclass(frozen=True)
class Stage:
    """§ 2 — a name, an enabled flag, and the cells that differ."""
    name: str
    enabled: bool = True
    overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """The name rule holds for the object too, not only for the file.

        ``_stage_from_obj`` checks this when parsing, but a producer handed
        hand-built stages would otherwise put an arbitrary string into a
        filename (``<label>_<name>.fdf``) and into an unquoted bash array
        literal in the runner.  The rule is the same one, read from the same
        pattern -- not a second copy of it."""
        if not isinstance(self.name, str) or not STAGE_NAME_RE.match(self.name):
            raise ValueError(
                f"stage name {self.name!r} must match [A-Za-z0-9_]+ -- it "
                "becomes a filename and a shell word "
                "(engines/stages.md 6.6)")


@dataclass(frozen=True)
class Task:
    """One calculation.

    ``stages`` and ``varies`` are ``None`` **together**: a description with
    no stages *is* a single-parameter-set calculation (§ 6.5), and an empty
    ``stages`` would be a second way to spell that.

    ``varies`` may be empty while ``stages`` is not, and that is a different
    state rather than a loophole — several stages differing in nothing but
    their name, which § 6.6a allows (they are the honest way to say *keep
    going* when each continues from the last)."""
    engine: str
    shape: str
    run: Run
    structure: StructureRef
    varies: Optional[Tuple[str, ...]] = None
    stages: Optional[Tuple[Stage, ...]] = None
    schema_fingerprint: str = ""
    #: WHICH KIND of calculation this describes -- the key into the
    #: engine's warm-file vocabulary (`job-contracts.md` § 4.2a: [base] +
    #: one section per type).  Absent-is-a-state, like ``stages``: the
    #: default IS "optimization", so an optimization description carries
    #: no key.  Membership (does the engine have this section?) is the
    #: rules file's question, answered where the file is read -- this
    #: codec checks only the SHAPE (U0, 2026-08-13).
    calculation: str = "optimization"

    def __post_init__(self) -> None:
        """§ 6.5 holds for the object too, not only for the file.

        Without this, ``Task(varies=(...), stages=None)`` is constructible
        and ``to_dict`` drops the ``varies`` silently -- a round-trip that
        loses intent, which is the one thing ``varies`` exists to carry
        (§ 6.2).  The two are absent together or present together, and
        ``stages`` is never an empty tuple: that would be the second
        spelling of "one stage" § 6.5 rules out."""
        if not isinstance(self.calculation, str) \
                or not STAGE_NAME_RE.match(self.calculation):
            raise ValueError(
                f"task: calculation {self.calculation!r} must match "
                "[A-Za-z0-9_]+ -- it names a section of the engine's "
                "warm-file vocabulary (job-contracts.md 4.2a)")
        if (self.varies is None) != (self.stages is None):
            raise ValueError(
                "task: 'varies' and 'stages' are absent together or present "
                f"together (varies={self.varies!r}, stages={self.stages!r}). "
                "A description with no stages is one parameter set, and there "
                "is nothing to vary across (engines/stages.md 6.5)")
        if self.stages is not None and not self.stages:
            raise ValueError(
                "task: 'stages' is present but empty. Omit it entirely for a "
                "single parameter set (engines/stages.md 6.5)")
        # The two LADDER-level refusals live here, in the codec, because
        # every route to a Task -- describe on a laptop, read_task on the
        # cluster, a hand-edited file -- passes through this constructor.
        # Three comments claimed these checks existed here (the u5
        # retirement in validation/siesta.py, describe_cmd's help,
        # describe._check's "codec's own checks") while NOTHING refused
        # either: the claims are made true at the claimed home (U15,
        # 2026-08-12).
        if self.stages is not None:
            # CASE-INSENSITIVELY, like both parsers (D6, redo 2026-08-12):
            # the names key filenames, and the filesystems these run on
            # include case-insensitive ones -- "Tight" and "tight" are one
            # deck there.  The constructor compared exact strings while
            # the parsers folded case, so the seam disagreed with its own
            # doors.
            names = [s.name.lower() for s in self.stages]
            dups = sorted({n for n in names if names.count(n) > 1})
            if dups:
                raise ValueError(
                    f"task: duplicate stage name(s) "
                    f"{', '.join(map(repr, dups))} (compared "
                    f"case-insensitively). A stage's name keys its "
                    f"directory, its deck and its status row -- two stages "
                    f"sharing one silently hand one the other's files "
                    f"(engines/stages.md 6.6)")
            if not any(s.enabled for s in self.stages):
                raise ValueError(
                    "task: every stage is disabled -- an all-disabled ladder "
                    "is an empty one spelled longer, and 6.5 already rules "
                    "the empty spelling out. Enable a stage, or omit "
                    "'stages' for a single parameter set")
        self._check_id()

    # ----- identity (run-identity.md 2, 2.0a, 3) ---------------------- #

    def _stage_names(self) -> Tuple[str, ...]:
        """The ladder, for the cap -- see :func:`~molbuilder.identity._cap_for`.

        The cap is derived from what ``<label>_<longest stage>.<longest
        extension>`` will occupy, so knowing the real ladder beats the fixed
        budget guessed when the stages do not exist yet.  Reading a
        description is exactly the moment they DO exist."""
        return tuple(s.name for s in self.stages) if self.stages else ()

    @property
    def label(self) -> str:
        """The ``SystemLabel`` / ``JOB`` literal, and the stem of every file.

        Derived rather than stored (see :class:`Run`), through the same
        normaliser that built half of ``run.id`` -- so the two cannot disagree
        about what the user's name normalises to, and ``__post_init__`` has
        already proved it."""
        return normalise_id(self.run.name, stage_names=self._stage_names(),
                            what="name")

    def _check_id(self) -> None:
        """``run.id`` is what ``run.name`` and ``structure.formula`` derive.

        § 3 rule 1 is *"it happens once, and the result is stored"* -- which
        only means something if somebody checks the stored value against its
        inputs.  Without this, ``id`` is a free string: a hand-edited ``name``
        (supported since 2026-08-07, decision 3) leaves the id behind, and the
        description then says two different things about which calculation it
        is.  § 1's second failure mode is exactly that edit, and its cost is a
        run that silently starts cold.

        It refuses rather than repairing.  Which of the three fields is the
        right one is not knowable here -- a corrected formula and a renamed
        calculation look identical from inside the file -- and quietly
        rewriting the id would be the *"append a digit and carry on"* that
        § 3 rule 3 rules out, one layer up."""
        expected = run_id(self.run.name, self.structure.formula,
                          stage_names=self._stage_names())
        if self.run.id != expected:
            raise ValueError(
                f"task: run.id {self.run.id!r} is not what this description "
                f"derives. run.name {self.run.name!r} + structure.formula "
                f"{self.structure.formula!r} give {expected!r}. The id is "
                f"normalised once and then quoted everywhere, so a mismatch "
                f"means one of the three was edited without the others -- and "
                f"which one is right is not something this reader may guess "
                f"(run-identity.md 2.0a and 3 rule 1)")

    # ----- persistence (task@1) -------------------------------------- #
    # ``to_dict`` / ``from_dict`` are the house names for a dataclass<->JSON
    # pair (26 of them under ``molbuilder/``); ``jobset/model.py::JobSet`` is
    # the closest analogue -- also a persisted plan for a multi-directory
    # unit of work, also major-checked through ``persist``.  The bodies live
    # at the foot of the module beside the refusals they use.

    def to_dict(self) -> dict:
        """This description as the JSON object § 6 specifies."""
        return _task_to_dict(self)

    @staticmethod
    def from_dict(obj: Mapping[str, Any]) -> "Task":
        """Parse a description, refusing rather than guessing."""
        return _task_from_dict(obj)


# --------------------------------------------------------------------- #
#  Refusals                                                             #
# --------------------------------------------------------------------- #

#: What a refusal calls the thing it is refusing -- ``task.json``, the file
#: this module owns.  (The surface-supplied-ladder parse that renamed it,
#: ``stages_from_dicts``, was deleted 2026-08-13 with zero production
#: callers; the default is now the only name in use.)
#:
#: A ContextVar rather than a plain global because the web layer serves
#: requests concurrently: two parses in flight would otherwise swap each
#: other's label and blame the wrong input.
_SOURCE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "task_refusal_source", default=FILENAME)


def _refuse(msg: str, *, where: str = "") -> NoReturn:
    raise ValueError(
        f"{_SOURCE.get()}{': ' + where if where else ''}: {msg}")


# (_refusals_name and the stages_from_dicts/_ladder_from_objs pair it
#  served were deleted 2026-08-13, V22: zero production callers -- the
#  docstring's claimed callers used config/pyscf's same-named, different
#  function.  _SOURCE stays: its default names every codec refusal.)


