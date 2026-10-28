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
**says nothing of how the calculation runs**: the queue, the run cards, the
groups, a benchmark's plan and the reports are its job request's,
``job-request.json`` (``execution/job-request.md``), and a key of that record
written here is refused, saying where it belongs.

LAYER.  L1: at load it imports only stdlib-only L1 modules (``persist``,
``identity``, ``paths``, ``runfiles``, ``config_dir``, ``report_fields``,
``scheduler.quantities``) and the standard library, so this stays a leaf.
That is deliberate rather than incidental — see *the split preflight* below.
``identity`` is here because ``run.id`` is something this module DERIVES and
checks; it is the same normaliser the CLI and the browser call, which is what
§ 3 rule 1's *"it happens once"* actually requires.

THE SPLIT PREFLIGHT.  § 6.6 lists eight checks "in order, and all of it before
anything is written".  Four of them are answerable from the file alone and are
enforced here; the other four need the engine's field schema, and importing an
engine into an L1 codec is the upward import the layer rule forbids
(`architecture.md` § 3, kept by review).
Those belong to resolution (P2 below), which already has the schema in hand:

  here    the schema string's major · ``shape`` present and legal · stage names
          in ``[A-Za-z0-9_]+`` and unique case-insensitively · no ``overrides``
          key naming a stage field · ``overrides`` a SUBSET of ``varies`` ·
          unknown keys refused by name · ``run.id`` is what ``run.name`` and
          ``structure.formula`` derive (§ 6.1's first rule -- the check needs
          only ``identity``, which is L1 beside this one) · a transport bias
          list of numbers, starting at 0, no two points in one folder
          (``bias_token``)
  P2      the engine has a generator · every
          named field exists in the schema · every value is one that field's
          DECLARED TYPE can hold · every value of a stage's may stand for
          its item on the kind (the
          one per-value door, ``template.why_not``: refused) and is inside
          its recommended range (warned), a bias point too ·
          and § 6.6a's
          warning for two stages that RESOLVE identically, since resolving is
          P2's verb

WHY THE MESSAGES NAME THINGS.  A description is JSON sitting beside the decks,
and editing it by hand is **supported** (the plan's decision 3).
So a refusal owes a person what it owes the browser: the offending key by name,
which stage it was in, and — where the key is one edit away from a real one —
what they probably meant.
"""
from __future__ import annotations

import contextvars
import difflib
import dataclasses as _dc
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, NoReturn, Optional, Tuple

from .identity import normalise_id, run_id, stage_key
from .persist import check_schema, read_json, write_json


SCHEMA = "molbuilder/task@2"
#: Its name is the catalogue's (`runfiles.WRITTEN`, `job-contracts.md` § 2.2).
from .runfiles import TASK_FILE as FILENAME  # noqa: E402

#: § 6.7 — required, no default, never inferred.
# The layout vocabulary lives with the layout (`paths`), which is floor 1
# and stdlib-only; this module validates a DECLARED shape against it.
from .paths import SHAPES

#: § 2 — "two fields, and no others".  An ``overrides`` map naming one of
#: these would be a stage redefining what a stage is.
#:
#: DERIVED FROM THE DATACLASS, not typed beside it: a hand-written tuple here
#: would be a second definition of what a `Stage` is.  Adding a field to
#: `Stage` updates this, `_check_keys`, and every caller at once.
#:
#: Defined after `Stage` (bottom of this module) because it reads its fields.

#: § 6.6 — a stage name becomes a filename, so the set is the narrow one.

STAGE_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


_TOP_KEYS = ("schema", "engine", "shape", "run",
             "structure", "varies", "stages", "calculation",
             "slots", "bias", "swap_electrodes", "reuses")
#: The job request's keys (`execution/job-request.md` § 3) -- refused here
#: by name, saying where each belongs: nothing in one record is read as the
#: other (A17).
_JOB_KEYS = ("allocation", "execution", "bench", "notify")
_JOB_STAGE_KEYS = ("group", "execution")
_BIAS_KEYS = ("voltages_v", "low_bias_approximation")

#: A slot citation names its junction explicitly, by its tree-relative
#: path (`engines/transport.md` § 3.1): a finished relaxation run's folder,
#: or a structure pair's .xyz -- what makes it citable is what it holds,
#: never its name or its position in molbuilder's own layout.  FORM only
#: here: a relative path with no traversal; the tree fence and the
#: condition live where the filesystem is (`molbuilder describe` resolves, `prep` composes
#: -- transport/compose.classify_citation).
_CITATION_RE = re.compile(r"^(?!/)(?!.*\.\.)[^\s]+$")
def bias_token(v: float) -> str:
    """The ONE spelling of a bias point's directory name — ``v0``,
    ``v0.2``, ``v-0.5`` (user ruling 2026-08-29: plain v-dirs, production
    names — never the bench spelling).  Deck naming, the per-point
    attempt layout, the gather, the chain walker -- and this codec's
    refusal of two points that would share one folder -- all read this.
    Here, beside the list it spells: the codec is the lowest reader."""
    return f"v{v:g}"

_RUN_KEYS = ("name", "id", "created")
_STRUCTURE_KEYS = ("source", "formula", "atoms", "fingerprint")


# --------------------------------------------------------------------- #
#  The shapes                                                           #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class Run:
    """What the user called it, and what it is called on disk.

    ``id`` is derived once and then quoted everywhere
    (``execution/run-identity.md`` § 2).  It does **not** name the
    directory — the level-③ folder is typed by the user, and this file is
    what says which calculation lives there (§ 3.0 there).

    Nor does it name the **files** (§ 2.0a, decision 26): the id is ``<label>_<formula>`` and lives here, while the label
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
    what was there when the description was written, and ``fingerprint`` the
    whole pair (`workingcopy_structure.StructureCodec.fingerprint`), so a
    description opened against a structure that has since changed -- an atom,
    a label, a held atom, the box -- can *say so* instead of silently
    building a different calculation under the same id.  A description
    written before the fingerprint was recorded states none, and is checked
    by ``formula`` and ``atoms`` alone."""
    source: str
    formula: str = ""
    atoms: int = 0
    fingerprint: str = ""


@dataclass(frozen=True)
class Stage:
    """§ 2 — a name and the cells that differ.  A stage is in the
    description or it is not: there is no on/off (`engines/stages.md` § 2).
    How it runs -- its run card, the job it runs in -- is the job request's
    (`execution/job-request.md` § 3.2, § 3.3)."""
    name: str
    overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """The name rule holds for the object too, not only for the file.

        ``_stage_from_obj`` checks this when parsing, but a producer handed
        hand-built stages would otherwise put an arbitrary string into a
        filename (``<label>_<name>.fdf``) and into an unquoted bash array
        literal in the runner.  The rule is the same one, read from the same
        pattern -- not a second copy of it."""
        if not isinstance(self.name, str) or not STAGE_NAME_RE.fullmatch(self.name):
            raise ValueError(
                f"stage name {self.name!r} must match [A-Za-z0-9_]+ -- it "
                "becomes a filename and a shell word "
                "(engines/stages.md 6.6)")


@dataclass(frozen=True)
class Task:
    """One calculation.

    **A Task read from disk always has both ``stages`` and ``varies``**, and
    ``stages`` always has at least one entry (§ 6.5, 2026-08-16): one stage is
    the ordinary starting point, not a special shape.

    They stay ``Optional`` on the dataclass all the same, and the asymmetry is
    deliberate rather than an oversight: the codec is the gate (:func:`of`
    refuses a description with no ``stages``, naming the fix), while the type
    still admits ``None`` so a producer can build a Task in pieces.  What the
    type must keep enforcing is that the two move TOGETHER — ``varies`` is what
    differs across the stages, so neither means anything without the other.

    ``varies`` may be empty while ``stages`` is not, and that is a different
    state rather than a loophole — several stages differing in nothing but
    their name, which § 6.6a allows (they are the honest way to say *keep
    going* when each continues from the last)."""
    engine: str
    shape: str
    run: Run
    #: ``None`` for exactly one calculation kind: transport, whose
    #: structure IS its junction citation (enforced in __post_init__).
    structure: Optional[StructureRef]
    #: WHICH KIND of calculation this describes -- the key into the
    #: engine's warm-file vocabulary (`job-contracts.md` § 4.2a: [base] +
    #: one section per type).  Stated by every description and every
    #: producer, never defaulted (W57 R10).
    #: Membership (does the engine have this section?) is the rules file's
    #: question, answered where the file is read -- this codec checks only
    #: the SHAPE (U0, 2026-08-13).
    calculation: str
    varies: Optional[Tuple[str, ...]] = None
    stages: Optional[Tuple[Stage, ...]] = None

    #: THE COMPOSITE'S INPUTS (engines/transport.md § 2a.14, § 3.1) -- slot name ->
    #: an EXPLICIT directory citation (a tree-relative path whose FILES
    #: satisfy the 4.1b condition; ruling Q1 + the 2026-08-29 amendment).
    #: Only a kind that cites (the kind table's ``description.cites`` -- a
    #: transport calculation's ``junction``) may carry slots, and it must
    #: carry exactly that one; its description carries NO ``structure``
    #: block, because its structure IS the citation, copied in at prep
    #: (strict composition, ruling Q2).
    #: Absent-is-a-state: empty writes no key.
    slots: Dict[str, str] = field(default_factory=dict)

    #: THE ELECTRODE RENAME IS THIS CALCULATION'S OWN (`engines/transport.md`
    #: § 4; user, 2026-10-04): ``True`` says the cited junction's
    #: ``L-electrode`` / ``R-electrode`` trade names in THIS calculation's
    #: copy of it -- compose applies it, the cited run is never written, and
    #: another calculation citing the same run makes its own choice.  Only
    #: a transport description carries it; false writes no key.
    swap_electrodes: bool = False

    #: THE BIAS SWEEP AXIS, volts (engines/transport.md § 2a.10) -- like
    #: a benchmark's plan, a list of points the description asks for, not a
    #: machine's answer.  Empty means zero-bias only (the default every
    #: transport description says by omitting the key).  When present
    #: the first entry must be 0.0: the treatment's one zero-bias device
    #: run, or the sweep's start from equilibrium.
    bias: Tuple[float, ...] = ()
    #: THE LOW-BIAS APPROXIMATION, stated for a list of more than one
    #: voltage and only for one (`engines/transport.md` § 2a.10): ``True`` --
    #: the device and the transmission run once, at 0 V, and the record
    #: computes each voltage's current from T(E, 0); ``False`` -- every
    #: voltage gets its own device SCF and transmission, a sweep.
    low_bias_approximation: Optional[bool] = None

    #: THE FINISHED CALCULATION THIS ONE REUSES (`engines/transport.md`
    #: § 2a.16) -- its folder, tree-relative as a slot's citation is: prep
    #: copies in the groups it already computed, and a rung of points takes
    #: its converged points over at its first launch.  Only a transport
    #: description carries it; none writes no key.
    reuses: str = ""

    @property
    def treatment(self) -> Optional[str]:
        """How this description treats its bias list, by the three names
        every surface uses (`engines/transport.md` § 2a.10): `single-bias`
        for one voltage; `low-bias-approximation` or `self-consistent` for
        several, by the stated switch.  None for a kind without a bias (the
        kind table's ``description.carries``).  Stated, never inferred from
        how many points ran."""
        from .kind_table import kind as _kind
        if "bias" not in _kind(self.calculation).description.carries:
            return None
        return ("single-bias" if len(self.bias) <= 1
                else "low-bias-approximation" if self.low_bias_approximation
                else "self-consistent")

    def __post_init__(self) -> None:
        """§ 6.5 holds for the object too, not only for the file.

        Without this, ``Task(varies=(...), stages=None)`` is constructible
        and ``to_dict`` drops the ``varies`` silently -- a round-trip that
        loses intent, which is the one thing ``varies`` exists to carry
        (§ 6.2).  The two are absent together or present together, and
        ``stages`` is never an empty tuple: that would be the second
        spelling of "one stage" § 6.5 rules out."""
        if not isinstance(self.calculation, str) \
                or not STAGE_NAME_RE.fullmatch(self.calculation):
            raise ValueError(
                f"task: calculation {self.calculation!r} must match "
                "[A-Za-z0-9_]+ -- it names a section of the engine's "
                "warm-file vocabulary (job-contracts.md 4.2a)")
        if (self.varies is None) != (self.stages is None):
            raise ValueError(
                "task: 'varies' and 'stages' travel together "
                f"(varies={self.varies!r}, stages={self.stages!r}) -- "
                "'varies' is what differs ACROSS the stages, so neither is "
                "meaningful without the other (engines/stages.md 6.5)")
        # A KIND THE KIND TABLE HOLDS, ON AN ENGINE IT RUNS ON
        # (`execution/architecture.md` § 3.3), each refused in the table's
        # sentence; and what its part on this engine fixes -- the shape,
        # and the stages, every one, in order -- held by no description of
        # it otherwise (`web/task-setup.md` § 4, § 5.3).  An engine the
        # engine table does not hold is the preflight's to refuse, naming
        # the engines there are (`validation.task.preflight`, row 1).
        from .engine_table import engines as _engines
        from .kind_table import kind as _kind
        fixed = (_kind(self.calculation).on(self.engine)
                 if str(self.engine) in _engines() else None)
        if fixed is not None and fixed.shape and self.shape != fixed.shape:
            raise ValueError(fixed.why)
        if fixed is not None and fixed.stages:
            names = tuple(st.name for st in (self.stages or ()))
            if names != fixed.stages:
                raise ValueError(
                    f"task: {_kind(self.calculation).with_article}'s stages "
                    f"are its rungs, every one, in order -- "
                    f"{', '.join(fixed.stages)} (got "
                    f"{', '.join(names) or 'none'}) -- the kind table's "
                    f"(execution/architecture.md § 3.3)")
        # WHAT THE KIND'S DESCRIPTION CARRIES BEYOND EVERY KIND'S FIELDS
        # (the kind table's ``description``): a kind that cites carries that
        # one slot and no structure block -- its structure IS the citation
        # (engines/transport.md § 3.1); a field only some kinds carry is
        # refused on any other, naming those that do.
        desc = _kind(self.calculation).description
        if desc.cites:
            if set(self.slots) != {desc.cites}:
                raise ValueError(
                    f"task: {_kind(self.calculation).with_article} carries "
                    f"exactly one slot, {desc.cites!r} (got "
                    f"{sorted(self.slots) or 'none'}) -- what it composes "
                    f"from")
            if self.structure is not None:
                raise ValueError(
                    f"task: {_kind(self.calculation).with_article} carries "
                    f"no 'structure' block -- its structure IS the "
                    f"{desc.cites} citation, copied in at prep")
        else:
            if self.slots:
                raise ValueError(
                    f"task: 'slots' belongs to a kind that cites -- "
                    f"{_carriers(None)} -- not {self.calculation!r}")
            if self.structure is None:
                raise ValueError(
                    f"task: 'structure' is required for "
                    f"{_kind(self.calculation).with_article}; only a kind "
                    f"that cites carries none ({_carriers(None)})")
        from .kind_table import DESCRIPTION_FIELDS
        for key in DESCRIPTION_FIELDS:
            if getattr(self, key) and key not in desc.carries:
                raise ValueError(
                    f"task: {key!r} belongs to {_carriers(key)} alone (this "
                    f"is {self.calculation!r})")
        for name, cite in self.slots.items():
            if not isinstance(cite, str) or not _CITATION_RE.fullmatch(cite):
                raise ValueError(
                    f"task: slot {name!r} must cite its junction by its "
                    f"tree-relative path (no leading '/', no '..', no "
                    f"whitespace); got {cite!r}: a finished relaxation "
                    f"run's folder, or a structure pair's .xyz "
                    f"(engines/transport.md § 3.1)")
        if self.reuses and not _CITATION_RE.fullmatch(self.reuses):
            raise DescriptionError(
                f"task: reuses names the finished transport calculation by "
                f"its folder's tree-relative path (no leading '/', no '..', "
                f"no whitespace); got {self.reuses!r} "
                f"(engines/transport.md § 2a.16)", where="reuses")
        if "bias" in desc.carries and not self.bias:
            raise DescriptionError(
                f"task: {_kind(self.calculation).with_article}'s description "
                f"states its voltages -- [0.0] for a single bias "
                f"(engines/transport.md 2a.10); one written before 2.16.3 "
                f"states none: `jobset migrate` writes it", where="bias")
        if self.bias:
            if any(not isinstance(v, (int, float)) or isinstance(v, bool)
                   for v in self.bias):
                raise DescriptionError(
                    f"task: bias voltages must be numbers (got {self.bias!r})",
                    where="bias")
            if float(self.bias[0]) != 0.0:
                raise DescriptionError(
                    f"task: the bias list must start at 0.0 (got "
                    f"{self.bias[0]!r}) -- a list starts from equilibrium: "
                    "the low-bias approximation converges its device there, "
                    "and a sweep starts each point from the one before "
                    "(engines/transport.md § 2a.10, § 2a.11)", where="bias")
            # TWO POINTS IN ONE FOLDER -- a repeated voltage, or two whose
            # folder names are one (`bias_token`: 0.1 and 0.1000001 are both
            # `v0.1/`) -- refused here, the one door every road reads a
            # description through (`engines/template.md` § 5.3): each point
            # is one device run in its own folder.  A point outside the
            # item's range is a recommendation's business: the description's
            # own check warns.
            folders: Dict[str, List[float]] = {}
            for v in self.bias:
                folders.setdefault(bias_token(float(v)), []).append(float(v))
            shared = [(tok, vs) for tok, vs in folders.items() if len(vs) > 1]
            if shared:
                tok, vs = shared[0]
                said = (f"repeats {vs[0]:g} V" if len(set(vs)) == 1 else
                        f"puts {' and '.join(repr(v) + ' V' for v in vs)} "
                        f"in one folder ({tok}/)")
                raise DescriptionError(
                    f"task: the bias list {said} -- each point is one device "
                    f"run in its own folder, so they would share it "
                    f"(engines/transport.md 2a.10)", where="bias")
        # THE TREATMENT, STATED where there is a choice -- a list of more
        # than one voltage -- and nowhere else (`engines/transport.md`
        # § 2a.10, TD7): never inferred from the count.
        if (len(self.bias) > 1
                and not isinstance(self.low_bias_approximation, bool)):
            raise DescriptionError(
                f"task: a bias list of {len(self.bias)} voltages states "
                f"bias.low_bias_approximation -- true (the device and the "
                f"transmission once, at 0 V; each voltage's current from "
                f"T(E, 0), the low-bias approximation) or false (every voltage "
                f"its own "
                f"device SCF and transmission); got "
                f"{self.low_bias_approximation!r} (engines/transport.md "
                f"2a.10)", where="bias.low_bias_approximation")
        if len(self.bias) <= 1 and self.low_bias_approximation is not None:
            raise DescriptionError(
                f"task: bias.low_bias_approximation answers a list of "
                f"several voltages; this one has "
                f"{len(self.bias) or 'no'} point (engines/transport.md "
                f"2a.10)", where="bias.low_bias_approximation")
        if self.stages is not None and not self.stages:
            raise ValueError(
                "task: 'stages' is present but empty. A job has at least one "
                "stage -- give it a single entry rather than emptying it, "
                "because an absent 'stages' is refused too "
                "(engines/stages.md 6.5)")
        # The two LADDER-level refusals live here, in the codec, because
        # every route to a Task -- describe on a laptop, `task.of` on the
        # cluster, a hand-edited file -- passes through this constructor.
        if self.stages is not None:
            # CASE-INSENSITIVELY, like the reader (D6): the names key
            # filenames, and the filesystems these run on include
            # case-insensitive ones -- "Tight" and "tight" are one deck there.
            names = [stage_key(s.name) for s in self.stages]
            dups = sorted({n for n in names if names.count(n) > 1})
            if dups:
                raise ValueError(
                    f"task: duplicate stage name(s) "
                    f"{', '.join(map(repr, dups))} (compared "
                    f"case-insensitively). A stage's name keys its "
                    f"directory, its deck and its status row -- two stages "
                    f"sharing one silently hand one the other's files "
                    f"(engines/stages.md 6.6)")
        self._check_id()

    # ----- identity (run-identity.md 2, 2.0a, 3) ---------------------- #

    def _label_tail(self) -> int:
        """What a rung adds to the label before its stage token -- its
        kind's longest lead stem (`kind_table.CalculationKind.label_tail`);
        nothing for a kind without leads."""
        from .kind_table import kind
        return kind(self.calculation).label_tail

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
                            what="name", label_tail=self._label_tail())

    def _check_id(self) -> None:
        """``run.id`` is what ``run.name`` and ``structure.formula`` derive.

        § 3 rule 1 is *"it happens once, and the result is stored"* -- which
        only means something if somebody checks the stored value against its
        inputs.  Without this, ``id`` is a free string: a hand-edited ``name``
        (decision 3) leaves the id behind, and the
        description then says two different things about which calculation it
        is.  § 1's second failure mode is exactly that edit, and its cost is a
        run that silently starts cold.

        It refuses rather than repairing.  Which of the three fields is the
        right one is not knowable here -- a corrected formula and a renamed
        calculation look identical from inside the file -- and quietly
        rewriting the id would be the *"append a digit and carry on"* that
        § 3 rule 3 rules out, one layer up."""
        # A KIND THAT CITES: the coordinates ARE the cited attempt, so the
        # citation is the identity's second half (run-identity.md § 2's
        # rule, extended by engines/transport.md § 3.1 -- an id that
        # named no input would collide across every junction).
        from .kind_table import kind as _kind
        cites = _kind(self.calculation).description.cites
        identity = (self.slots.get(cites, "") if cites
                    else self.structure.formula)
        expected = run_id(self.run.name, identity,
                          stage_names=self._stage_names())
        if self.run.id != expected:
            raise ValueError(
                f"task: run.id {self.run.id!r} is not what this description "
                f"derives. run.name {self.run.name!r} + identity "
                f"{identity!r} give {expected!r}. The id is "
                f"normalised once and then quoted everywhere, so a mismatch "
                f"means one of the three was edited without the others -- and "
                f"which one is right is not something this reader may guess "
                f"(run-identity.md 2.0a and 3 rule 1)")

    # ----- persistence (task@2) -------------------------------------- #
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
#: this module owns.
_SOURCE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "task_refusal_source", default=FILENAME)


class DescriptionError(ValueError):
    """A refusal of the description's own content that NAMES WHERE in the
    file (`bias`, `bias.low_bias_approximation`), so a surface puts it
    beside the control that wrote it (`science/validation.md` § 4.1 R2)
    rather than in a status line.  A `ValueError` still: every door that
    refuses a description keeps refusing it."""

    def __init__(self, msg: str, *, where: str = "") -> None:
        super().__init__(msg)
        self.where = where


def _refuse(msg: str, *, where: str = "") -> NoReturn:
    raise ValueError(
        f"{_SOURCE.get()}{': ' + where if where else ''}: {msg}")


def _as_object(value: Any, *, where: str) -> Mapping[str, Any]:
    """Refuse a non-object where § 6 specifies one, *saying* so.

    Without this the key check below iterates a string's characters and
    reports ``unknown key 's'`` for ``"engine": "siesta"`` — technically a
    refusal, but a person hand-editing the file (the plan's decision 3)
    learns nothing from it."""
    if not isinstance(value, Mapping):
        _refuse(f"expected an object, got {type(value).__name__}", where=where)
    return value


def _check_keys(obj: Mapping[str, Any], allowed: Tuple[str, ...], *,
                where: str, note: str = "") -> None:
    """§ 6.1 rule 1 — an unknown key is refused, not ignored, because an
    ignored key is a calculation quietly different from the one asked for.

    ``note`` is appended when there is no near-miss: *"unknown"* is true but
    useless when the name is a real setting that belongs in a DIFFERENT block,
    and sending a person to look for a typo they did not make is worse than
    saying nothing.
    """
    for key in obj:
        if key in allowed:
            continue
        near = difflib.get_close_matches(key, allowed, n=1, cutoff=0.7)
        hint = f" -- did you mean {near[0]!r}?" if near else \
               f" (known keys: {', '.join(allowed)}){note}"
        _refuse(f"unknown key {key!r}{hint}", where=where)


def _require(obj: Mapping[str, Any], key: str, *, where: str) -> Any:
    if key not in obj:
        _refuse(f"missing required key {key!r}", where=where)
    return obj[key]


# --------------------------------------------------------------------- #
#  Reading                                                              #
# --------------------------------------------------------------------- #

def _task_from_dict(obj: Mapping[str, Any]) -> Task:
    """Parse a description, refusing rather than guessing.

    The order is § 6.6's: the schema string first, so a file from a future
    major fails saying so instead of failing on whichever key moved."""
    if not isinstance(obj, Mapping):
        _refuse(f"expected a JSON object, got {type(obj).__name__}")

    check_schema(str(obj.get("schema") or ""), SCHEMA, label=FILENAME)
    # 2.5.0'S CENTRING CHOICE IS RETIRED: a transport junction enters its
    # calculation centred (`model/structure-periodicity.md` § 6.0).
    if "centre_junction" in obj:
        _refuse("centre_junction is retired (2.6.0): a transport junction "
                "enters its calculation centred in its cell.  "
                "`molbuilder jobset migrate` in the calculation's folder "
                "removes the key", where="centre_junction")
    for key in _JOB_KEYS:
        if key in obj:
            _refuse(f"{key!r} says how the calculation runs, which is its job "
                    f"request's, job-request.json -- not the description's "
                    f"(execution/job-request.md 1)", where=key)
    _check_keys(obj, _TOP_KEYS, where="")

    engine_obj = _as_object(_require(obj, "engine", where=""), where="engine")
    _check_keys(engine_obj, ("name",), where="engine")
    engine = str(_require(engine_obj, "name", where="engine"))

    shape = _require(obj, "shape", where="")
    if shape not in SHAPES:
        _refuse(f"shape {shape!r} is not one of {' / '.join(SHAPES)}. "
                "It is required and never inferred -- inferring it would "
                "hand you a directory tree you did not ask for "
                "(engines/stages.md 6.7)")

    run_obj = _as_object(_require(obj, "run", where=""), where="run")
    _check_keys(run_obj, _RUN_KEYS, where="run")
    run = Run(name=str(_require(run_obj, "name", where="run")),
              id=str(_require(run_obj, "id", where="run")),
              created=str(run_obj.get("created", "")))

    # the calculation TYPE first this once: whether `structure` is
    # required depends on it (transport's structure IS its citation).
    if "calculation" not in obj:
        _refuse("names no 'calculation' -- every description states its kind "
                "(job-contracts.md 6.1; an optimization left it out until "
                "2026-10-06)")
    calc = obj["calculation"]
    if not isinstance(calc, str) or not STAGE_NAME_RE.fullmatch(calc):
        _refuse(f"calculation {calc!r} must match [A-Za-z0-9_]+ -- it "
                "names a section of the engine's warm-file vocabulary "
                "(job-contracts.md 4.2a)")

    from .kind_table import kind as _kind
    cites = _kind(calc).description.cites
    if cites:
        if "structure" in obj:
            _refuse(f"{_kind(calc).with_article} carries no 'structure' "
                    f"block -- its structure IS the {cites} citation, "
                    f"copied in at prep (engines/transport.md § 3.1)",
                    where="structure")
        structure = None
    else:
        struct_obj = _as_object(_require(obj, "structure", where=""),
                                where="structure")
        _check_keys(struct_obj, _STRUCTURE_KEYS, where="structure")
        structure = StructureRef(
            source=str(_require(struct_obj, "source", where="structure")),
            formula=str(struct_obj.get("formula", "")),
            atoms=int(struct_obj.get("atoms", 0)),
            fingerprint=str(struct_obj.get("fingerprint", "")))

    # THE COMPOSITE'S KEYS (engines/transport.md § 2a.14).  Shape here;
    # the pairing rules (transport-only, junction required, bias starts
    # at 0.0) live on the dataclass so object builders meet them too.
    slots_obj = _as_object(obj.get("slots", {}), where="slots")
    slots = {}
    for name, cite in slots_obj.items():
        if not isinstance(cite, str):
            _refuse(f"slot {name!r} must be a citation string, got "
                    f"{type(cite).__name__}", where="slots")
        slots[str(name)] = cite
    bias_obj = _as_object(obj.get("bias", {}), where="bias")
    _check_keys(bias_obj, _BIAS_KEYS, where="bias")
    bias_raw = bias_obj.get("voltages_v", [])
    if not isinstance(bias_raw, list):
        _refuse(f"voltages_v must be a list, got "
                f"{type(bias_raw).__name__}", where="bias")
    bias = tuple(bias_raw)
    low_bias_approximation = bias_obj.get("low_bias_approximation")
    swap_electrodes = obj.get("swap_electrodes", False)
    if not isinstance(swap_electrodes, bool):
        _refuse(f"swap_electrodes is true or false, got "
                f"{type(swap_electrodes).__name__}", where="swap_electrodes")
    reuses = obj.get("reuses", "")
    if not isinstance(reuses, str):
        _refuse(f"reuses is a folder's tree-relative path, got "
                f"{type(reuses).__name__}", where="reuses")

    has_stages = "stages" in obj

    # THIS REFUSAL COMES FIRST, and that ordering is the whole point.  Until
    # 2026-08-16 a ``'varies' without 'stages'`` check sat above it and fired
    # instead on every real description -- `varies` travels with `stages`, so
    # deleting `stages` by hand always tripped the pairing check, which
    # answered with the RETIRED rule ("a description with no stages is one
    # parameter set") and never said to add a stage.  A refusal that states
    # the opposite of the contract is worse than no refusal.  The pairing
    # check is gone because this one subsumes it: `stages` is never absent,
    # so `varies` can never be the lone survivor.
    if not has_stages:
        # § 6.5 (2026-08-16): A JOB ALWAYS HAS AT LEAST ONE STAGE.  One stage
        # is the ordinary starting point, not a special shape -- it is named,
        # it gets the ordinal token, its artifacts are named like any other
        # stage's.  So there is no stage-less form to accept.
        #
        # The rule replaced one that let ``stages`` be absent, and the reason
        # is a transition nobody had specified: a stage-less run wrote
        # artifacts with NO token, and adding a second stage then left that
        # run belonging to no token at all -- with `.XV` / `.DM` unsuffixed
        # and shared in `flat`, so the new stage would warm-start from it
        # silently.
        _refuse("no 'stages'. A job has at least one stage -- one is the "
                "ordinary case, not a special shape (engines/stages.md 6.5). "
                "Give it a single entry: "
                '"stages": [{"name": "coarse", "overrides": {}}]')

    raw_stages = obj["stages"]
    if not isinstance(raw_stages, (list, tuple)) or not raw_stages:
        _refuse("'stages' is present but empty. A job has at least one stage "
                "(engines/stages.md 6.5) -- give it a single entry rather "
                "than removing the key, which is refused the same way: "
                '"stages": [{"name": "coarse", "overrides": {}}]')

    varies = tuple(str(v) for v in obj.get("varies", ()))
    stages = tuple(_stage_from_obj(s, varies, i)
                   for i, s in enumerate(raw_stages))

    _refuse_duplicate_stage_names(stages)

    # ``varies`` stays a tuple here even when empty: it travels WITH
    # ``stages``, and an empty one is a real state -- several stages that
    # differ in nothing but their name.  (``None`` is reserved for the
    # no-stages case above, which § 6.5 spells by omitting both keys.)
    return Task(engine=engine, shape=shape, run=run, structure=structure,
                varies=varies, stages=stages, calculation=calc,
                slots=slots, bias=bias,
                low_bias_approximation=low_bias_approximation,
                swap_electrodes=swap_electrodes, reuses=reuses)


def _stage_from_obj(obj: Mapping[str, Any], varies: Tuple[str, ...],
                    index: int) -> Stage:
    where = f"stage {index}"
    _as_object(obj, where=where)

    name = str(obj.get("name", ""))
    where = f"stage {name!r}" if name else where
    # NO ON/OFF (`engines/stages.md` § 2): a description written before
    # 2026-10-03 carries `"enabled"` on every stage -- refused by name,
    # naming `jobset migrate`, which removes it (one policy; a `true` was
    # read and dropped on Save until 2026-10-19).
    if "enabled" in obj:
        _refuse("\"enabled\" -- a stage has no on/off switch since 2026-10-03: "
                "it is in the description or it is not (engines/stages.md 2).  "
                "A description written before carries the key: `jobset "
                "migrate` removes it", where=where)
    for key in _JOB_STAGE_KEYS:
        if key in obj:
            _refuse(f"{key!r} says how the stage runs, which is the job "
                    f"request's, job-request.json -- its entry under "
                    f"'stages' (execution/job-request.md 3.2, 3.3)",
                    where=where)
    _check_keys(obj, STAGE_FIELDS, where=where)
    if "name" not in obj:
        _refuse("missing required key 'name'", where=where)
    if not STAGE_NAME_RE.fullmatch(name):
        _refuse(f"stage name {name!r} must match [A-Za-z0-9_]+ -- it becomes "
                "a filename (engines/stages.md 6.6)")
    overrides = dict(_as_object(obj.get("overrides") or {},
                                where=f"{where} overrides"))
    for key in overrides:
        if key in STAGE_FIELDS:
            _refuse(f"override {key!r} names a stage field. A stage has "
                    f"exactly {', '.join(STAGE_FIELDS)}; an override may not "
                    "redefine one (engines/stages.md 2)", where=where)

    # § 6.2 -- a SUBSET of `varies`, never a superset.
    #
    # No key outside `varies`: a demoted parameter must not leave a value
    # hiding in a stage nobody can see.  But a varied key may be ABSENT, and
    # absent means "this stage uses the TEMPLATE's value" -- a real state, and
    # the one the table draws as a quiet cell.  Requiring equality would make
    # `varies` redundant -- derivable as the key set of any stage -- and force
    # every cell to be filled with a copy.
    extra = sorted(set(overrides) - set(varies))
    if extra:
        _refuse(f"override(s) {', '.join(repr(k) for k in extra)} not listed "
                "in 'varies'. A stage may only override a field the user "
                "promoted, or a demoted parameter leaves a value hiding "
                "(engines/stages.md 6.2)", where=where)

    return Stage(name=name, overrides=overrides)


def fresh_varies(stages) -> Tuple[str, ...]:
    """What a fresh ladder VARIES -- :func:`varies_for` over its stages'
    overrides: nothing for the one-stage start, each tier's keys for a
    `--stage-strategy` ladder.  The one rule both writers of a description
    ask: `molbuilder describe` and the hand-over (`proposed_varies`, which the Task
    setup page copies; 2026-10-19)."""
    return varies_for(st.overrides for st in stages)


def varies_for(overrides_seq) -> Tuple[str, ...]:
    """The promoted columns of a ladder — every override key, first seen first.

    § 6.2: ``varies`` is *"the set of fields the user chose to tune"*, and
    ``overrides`` is a **subset** of it. A surface that supplies a ladder
    without stating ``varies`` separately has already said what varies by
    listing what each stage overrides, and asking for it twice would be a
    second place to disagree.

    **The union, and not one stage's keys**, because a stage may leave a
    promoted cell empty — that means *"use the template's value"*, which is a
    real state § 6.2 protects, not an absence of intent.
    """
    out: list = []
    for overrides in overrides_seq:
        for key in overrides or {}:
            if key not in out:
                out.append(key)
    return tuple(out)


def _refuse_duplicate_stage_names(stages) -> None:
    """The reader's case-insensitive duplicate check (D10).
    Case folds because names key filenames (engines/stages.md § 2)."""
    seen: dict[str, str] = {}
    for st in stages:
        low = stage_key(st.name)
        if low in seen:
            _refuse(f"two stages named {st.name!r}"
                    + (f" and {seen[low]!r}" if seen[low] != st.name else "")
                    + " -- names are compared case-insensitively because they "
                      "become filenames (engines/stages.md 6.6)")
        seen[low] = st.name


def derive_run(name: str, formula: str = "", *, created: str = "",
               stage_names: Tuple[str, ...] = ()) -> Run:
    """Build a :class:`Run` with its id derived, not retyped.

    The one place a description's id is *made*.  § 3 rule 1 -- *"it happens
    once, when the description is written, and the result is stored"* -- is
    only true if there is one place it can happen, and a producer spelling
    ``Run(name=n, id=run_id(n, f))`` inline is a second place waiting to drift
    (the browser and the CLI must write the same bytes, § 6.4).

    ``stage_names`` is worth passing when the ladder is known: it makes the
    cap the real one rather than the budget guessed for an unknown ladder.
    :meth:`Task.__post_init__` re-derives with the stages it can see, so a
    ``Run`` built without them still checks out."""
    return Run(name=name, created=created,
               id=run_id(name, formula, stage_names=stage_names))


def path_of(base) -> Path:
    """``<base>/task.json`` -- for its writer, and for `jobset migrate`,
    which reads it raw (`architecture.md` A16): the ground floor's join of
    the description's name to a folder (`calcdirs.description_path`)."""
    from .calcdirs import description_path
    return description_path(base)


def of(base) -> Optional[Task]:
    """THE DESCRIPTION'S DOOR (`architecture.md` § 3.2, A16): the folder's
    ``task.json`` read and parsed, or ``None`` when the folder holds none.
    One written before a rule its writer now keeps -- each that
    :func:`_written_before` names -- is refused naming the command that
    rewrites it, the calculation by its folder."""
    path = path_of(base)
    if not path.is_file():
        return None
    return _from_obj(read_json(path), said_as=str(path), base=Path(base))


def from_text(text: str, said_as: str = FILENAME) -> Task:
    """A description that is not yet a file -- the Task setup tab's save
    (`task-setup.md`) -- read by the one reader; ``said_as`` names it in a
    refusal."""
    import json
    return _from_obj(json.loads(text), said_as=said_as, base=None)


def _from_obj(obj, *, said_as: str, base: Optional[Path]) -> Task:
    before = _written_before(obj) if isinstance(obj, Mapping) else []
    if before:
        from .identity import migrate_as_typed
        raise ValueError(
            f"{said_as} was written before every description stated "
            f"{' and '.join(before)}: "
            f"`{migrate_as_typed(base)}` rewrites it, "
            f"every value kept and the old file beside the new.")
    return Task.from_dict(obj)


def never_as_zero(hours) -> bool:
    """Whether a notify period is the spelling of *never* a description
    used before 2026-10-06, ``0`` -- ``null`` since: a description written
    then is `molbuilder/task@1`, refused by the reader
    (:func:`_written_before`), and `jobset migrate` rewrites it."""
    return (isinstance(hours, (int, float)) and not isinstance(hours, bool)
            and hours == 0)


def _written_before(obj: Mapping[str, Any]) -> List[str]:
    """What a description written before a rule says otherwise, said as
    :func:`of` says it, with the command for its folder -- how it runs in
    its own record (`molbuilder/task@1`, 2.27.0; its notify block's and its
    groups' older rules with it, which the same rewrite settles), no
    `calculation` (2026-10-06), a stage's `enabled` (2026-10-03),
    `centre_junction` (retired 2.6.0), a transport description stating no
    voltages (2.16.3) -- or ``[]``."""
    out: List[str] = []
    if str(obj.get("schema") or "") == "molbuilder/task@1":
        out.append("how it runs -- its queue, its run cards, its groups, a "
                   "benchmark's plan, its reports -- in its job request, "
                   "job-request.json (2.27.0)")
    if "calculation" not in obj:
        out.append("its `calculation` (2026-10-06)")
    stages = obj.get("stages")
    if isinstance(stages, (list, tuple)) and any(
            isinstance(st, Mapping) and "enabled" in st for st in stages):
        out.append("its stages with no on/off switch (2026-10-03)")
    if "centre_junction" in obj:
        out.append("its junction centred, with no `centre_junction` (2.6.0)")
    bias = obj.get("bias")
    from .kind_table import carrying, kind as _kind
    calc = obj.get("calculation")
    if calc in carrying("bias") and (
            bias is None
            or (isinstance(bias, Mapping) and not bias.get("voltages_v"))):
        out.append(f"{_kind(calc).with_article}'s voltages, [0.0] for a "
                   f"single bias (2.16.3)")
    return out


def _carriers(field: Optional[str]) -> str:
    """The kinds whose description carries ``field`` -- with ``None``, the
    kinds that cite -- as a refusal names them (the kind table's
    ``description``): *calculation='transport'*."""
    from .kind_table import carrying, kind, kinds
    names = (carrying(field) if field else
             tuple(k for k in kinds() if kind(k).description.cites))
    return " or ".join(f"calculation={k!r}" for k in names) or "no kind"


# --------------------------------------------------------------------- #
#  Writing                                                              #
# --------------------------------------------------------------------- #

def _task_to_dict(task: Task) -> dict:
    """The JSON object for a description.

    Key order is § 6's, so a written file reads like the contract's own
    example and a diff between two descriptions lines up."""
    out: dict[str, Any] = {
        "schema": SCHEMA,
        "engine": {"name": task.engine},
        "shape": task.shape,
        "run": {"name": task.run.name, "id": task.run.id},
    }
    if task.run.created:
        out["run"]["created"] = task.run.created
    # EVERY DESCRIPTION STATES ITS KIND.
    out["calculation"] = task.calculation
    # the composite's inputs ride where the design's example puts them
    # (engines/transport.md § 2a.14): slots, then bias, then the stages.
    if task.slots:
        out["slots"] = dict(sorted(task.slots.items()))
    if task.swap_electrodes:
        out["swap_electrodes"] = True
    if task.bias:
        out["bias"] = {"voltages_v": [float(v) for v in task.bias],
                       **({"low_bias_approximation":
                           task.low_bias_approximation}
                          if task.low_bias_approximation is not None
                          else {})}
    if task.reuses:
        out["reuses"] = task.reuses
    if task.structure is not None:
        out["structure"] = {"source": task.structure.source,
                            "formula": task.structure.formula,
                            "atoms": task.structure.atoms,
                            **({"fingerprint": task.structure.fingerprint}
                               if task.structure.fingerprint else {})}
    if task.stages:
        # Absent together, never empty -- an empty list would be a second
        # way to spell "one stage" (§ 6.5).
        out["varies"] = list(task.varies or ())
        out["stages"] = [{"name": s.name, "overrides": dict(s.overrides)}
                         for s in task.stages]
    return out


def write_task(base, task: Task):
    """Write the folder's ``task.json`` atomically (``persist.write_json``)
    -- the path the owner forms (:func:`path_of`)."""
    return write_json(path_of(base), task.to_dict())


__all__ = ["SCHEMA", "FILENAME", "STAGE_FIELDS",
           "Run", "StructureRef", "Stage", "Task",
           "derive_run", "of", "from_text", "path_of",
           "varies_for", "write_task"]


#: See the note beside § 2 above.  Derived here, where `Stage` is in scope.
STAGE_FIELDS: "tuple[str, ...]" = tuple(f.name for f in _dc.fields(Stage))
