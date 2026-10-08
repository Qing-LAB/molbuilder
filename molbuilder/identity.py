"""The label and the run id — built from inputs, and normalised exactly once.

**Two names, and only one of them is ever a filename** (`run-identity.md`
§ 2.0a, decision 26 — 2026-08-09).  :func:`normalise_id` makes the **label**:
the ``SystemLabel`` / ``JOB`` literal, the stem of every file, and the thing
:data:`OUR_FILE_PATTERNS` and :data:`MAX_LABEL_BYTES` are about.  :func:`run_id`
attaches the structure's formula to it and makes the **id**, which is a record
in ``task.json`` and never a name on disk.

**Module:** L1. Imports the standard library and nothing else, which is what
lets the L1 codec (``task``), the L2 producers and the L3 surfaces all ask the
same function for a name instead of each spelling the rule again.

**Contract:** [`execution/run-identity.md`](?doc=execution/run-identity.md)
§ 2 (*the id is built from inputs, never from anything a run produced*) · § 2.1
(what is in the pin, and the longer list of what is not) · § 3 (the character
set, the cap, and the three rules) · § 3.1 (the worked table, which is this
module's test fixture — `tests/test_run_identity.py` parses it out of the
document rather than retyping it).

**Why this is a module and not two helpers on `task.py`.** ``task.py`` is the
codec for a description; deriving its identity is a different job with its own
contract, and putting it here keeps the document-to-module mapping one-to-one.
It also means the CLI, the web tab and the codec reach the same normaliser —
§ 3's rule 1 is *"it happens once, and the result is stored"*, which is only
true if there is one place it can happen.

``task.py`` calls in rather than trusting what it parsed: it derives
``run.id`` from ``run.name`` and ``structure.formula`` and refuses a
description that disagrees with itself (``Task._check_id``).  That is what makes
*stored* mean anything.

**On the character set.** § 3 is explicit that the set *"is not a new
decision"* — `job-contracts.md § 2.1` Rule 2 fixes it, and this module does not
get to widen it. Four validators in the tree spell it
(``projects._NAME_PATTERN``, ``checkpoint._CALC_NAME_RE``, ``bench/grid`` and
``config/pyscf``). **None of them normalises**; they all reject. This is the
one transform. What the tests do instead is assert that every id this
produces is accepted by the shipped validators, so the agreement is checked
rather than assumed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

try:                                   # inside molbuilder
    from .runfiles import patterns as _runfile_patterns
except ImportError:                    # beside a job -- it travels with the
    from runfiles import patterns as _runfile_patterns   # monitor and the vibration finish (runwrap.MONITOR_COMPANIONS, VIBRATION_COMPANIONS)


@dataclass(frozen=True)
class RestartGroup:
    """§ 4 — what an engine means by *"may I continue this?"*.

    § 4's claim is that this is **not a SIESTA quirk**: every engine has a
    notion of which prior state belongs to a run, and a set of parameters that
    decides whether that state is honoured when found. A design naming only the
    filename has described none of it.

    Two failures follow from getting the second half wrong, and they are
    silent in opposite directions — *honoured with nothing to load* (the deck
    says resume, the engine cold-starts) and *present but not honoured* (the
    files are right there and the stage starts from scratch). **One field
    cannot produce either**, which is why ``restart`` is one field and this is
    what it expands into.

    ``literal`` is the engine's job identity — the thing every warm file is
    keyed by. ``keys`` are declared input keys where the engine has them, and
    is empty where the mechanism is generated control flow instead;
    ``mechanism`` says which, in prose, because the difference is real and
    hiding it behind an empty tuple would be a lie of omission.

    § 4 rule 1: *"A new engine that cannot fill this in is a new engine whose
    restart behaviour nobody has thought about yet."* That sentence is the
    reason this is a declared object rather than an ``if engine ==`` somewhere
    in a renderer — a missing declaration is visible, and a missing branch is
    not.
    """
    literal: str
    keys: Tuple[str, ...]
    mechanism: str
    #: WHICH CONFIG FIELD carries the literal — ``system_label`` / ``job_name``.
    #:
    #: ``literal`` is what the engine calls it; this is what molbuilder calls
    #: it, and a surface that has a config in hand and needs the calculation's
    #: name needs this one: a calculation named from anything else carries a
    #: second name, and § 4's *"there is no second name"* is false on disk.
    #:
    #: Defaulted so an engine that has not been revisited still constructs;
    #: the two shipped engines both fill it in.
    field: str = ""

def continues(cfg) -> bool:
    """Does this run start from what is already in the folder? (§ 4 rule 2.)

    ``restart`` is ONE field with two answers, and the engine's renderer
    expands it into whatever that engine's restart group is -- three declared
    keywords for SIESTA, generated control flow for PySCF.  This is the
    reading of the field, and it lives here rather than in either engine
    because both ask the same question of the same field: a copy per engine
    would be two spellings of one rule, free to drift in the direction § 4
    calls silent.

    ``getattr`` with a default rather than ``cfg.restart``: this is also
    reached with a template-shaped object during stage resolution, and a
    missing field means *clean*, which is the safe reading of silence -- the
    dangerous direction is resuming when nobody asked.
    """
    return getattr(cfg, "restart", "clean") == "continue"


#: One character of the id's alphabet.  Deliberately a character class rather
#: than a whole-string pattern: this module *builds* names, and the shipped
#: patterns all *validate* them.
_ALLOWED_CHAR = re.compile(r"[A-Za-z0-9_-]")

#: A run of two or more separators.  § 3: *"runs of two or more separators
#: collapsed to a single `_`, and a lone separator does not"* — which is what
#: keeps `BDT-Au` from quietly becoming `BDT_Au`.
_SEPARATOR_RUN = re.compile(r"[-_]{2,}")

#: Leading and trailing separators, trimmed last.
_EDGE_SEPARATORS = re.compile(r"^[-_]+|[-_]+$")

#: The filesystem's per-name limit, in bytes.  255 is the practical floor
#: across ext4, xfs, APFS and NTFS.
_NAME_LIMIT = 255

#: The longest extension molbuilder writes beside a label, from
#: `job-contracts.md § 4.2`'s SIESTA inventory.  ``.STRUCT_NEXT_ITER`` is the
#: longest of the thirteen; PySCF's longest (``_geom_optim.xyz``) is shorter.
_LONGEST_EXTENSION = ".STRUCT_NEXT_ITER"

#: What ``_<stage name>`` may occupy when the ladder is not known yet.
#:
#: **This one is a budget, not a derivation, and the difference is worth
#: stating.** § 3 derives the cap from ``<label>_<longest stage name>.<longest
#: extension>``, and nothing in the system bounds a stage name — ``STAGE_NAME_RE``
#: is ``[A-Za-z0-9_]+`` with no length. A calculation is usually named before
#: its stages exist, so when they are unknown this reserves a generous fixed
#: budget; when they ARE known, pass ``stage_names`` and the real longest is
#: used instead of this.
_STAGE_BUDGET = 32

#: The most a label may occupy, derived: the name limit, less the longest
#: stage suffix and the longest extension that will be appended to it.
#:
#: It bounds what goes in a **filename**, and § 2.0a is that only the label does.
#: The id is longer and goes in ``task.json``, where nothing bounds it.
MAX_LABEL_BYTES = _NAME_LIMIT - _STAGE_BUDGET - len(_LONGEST_EXTENSION)


#: Glob patterns, keyed on the **label**, for **the files molbuilder itself
#: writes** into a run directory — from `job-contracts.md § 2.2`'s catalogue.
#: The label and not the id, because these are filenames and § 2.0a puts only
#: the label in one.
#:
#: **This is the list that can be complete, and that is the whole point.** An
#: engine's output set depends on its version and on which options are on, so
#: enumerating *that* is a snapshot pretending to be a rule. What we write is
#: knowable, because we write it. So every question of the form *"did the
#: engine leave something here"* is answered by subtraction: anything named
#: after the label that is **not** on this list came from the run.
#:
#: Ordered as § 2.2 lists them — inputs, wrapper, the canonical trajectory,
#: then the run-indexed logs, which are **history and not state**: they are
#: what a user goes back to read, and nothing here may treat them as leftovers.
#:
#: **DERIVED from `runfiles.WRITTEN`** — the catalogue that says what each of
#: these files IS, which is what the Task-setup card shows a person; `runfiles`
#: holds the notes on why each row is there.
#:
#: ⚠ **THIS LIST HAS A SECOND READER.** `runwrap._cold_restart_block`
#: derives `--cold`'s *"except what molbuilder wrote"* exception from these
#: same patterns, and it runs where an engine's output IS present.  That
#: exception is anchored on the run's id, never widened to a star: read as
#: `*`, a `{label}.xyz` row would make PySCF's `<JOB>_optimized.xyz` — warm
#: state — look like ours, and `--cold` would walk past the file it exists to
#: move (`test_the_exception_is_anchored_on_the_id_not_widened_to_a_star`).
OUR_FILE_PATTERNS: Sequence[str] = _runfile_patterns()


# --------------------------------------------------------------------- #
#  How a stage is NAMED, and how every surface REFERS to one.            #
#  Contract: decision 27 (the token) and decision 28 / § 8f (the         #
#  resolver) in `archive/2026-08-19-staged-runs-implementation-plan.md`  #
# --------------------------------------------------------------------- #

def stage_token(seq: int, name: str) -> str:
    """``<NN>_<name>`` — the one token every per-stage artifact is named from.

    Decision 27 (2026-08-10, user): *"we may have many stages connected so I'd
    rather use names with index number."*  The ordinal travels **with** the
    name, in both layouts, so a flat listing of eight decks sorts into the
    order the stages ran rather than alphabetically.

    This is the same token a stage *directory* uses (``01_coarse``), which is
    the point: one token, one meaning, whether it is a path segment or part of
    a filename.

    **It is not the stage's position in the list**, which
    ``engines/stages.md`` R5 forbids in a filename for a precise reason —
    inserting at the front would shift every later number and reassign outputs
    that already exist.  ``seq`` is assigned once and never reassigned
    (``project-layout.md`` § 4.2: *"insert something between 1 and 2 is not an
    insertion; it is a new stage that happens to be coarser, and numbering it
    03 is the truth"*), so it cannot shift and R5's failure cannot occur.

    Zero-padded to two digits because it **sorts** — the same reason the
    directory pads (``job-contracts.md`` § 6.3).
    """
    return f"{int(seq):02d}_{name}"


def stage_key(name: str) -> str:
    """THE ONE KEY a stage name is compared by -- its case folded
    (`engines/stages.md` § 2: names key filenames, and the filesystems these
    run on include case-insensitive ones, so ``Tight`` and ``tight`` are one
    stage).  A name the description holds is compared with one from outside
    it through this: the description's duplicate check, the verbs' resolver,
    the role rule (plan § 5w K12); a verb resolves once, at its entry, and
    uses the description's spelling after that.  A name is ``[A-Za-z0-9_]+``,
    so this is ASCII's fold -- the browser's ``toLowerCase``."""
    return str(name).lower()


def command_stage(token: str) -> str:
    """How a line that holds only a stage's TOKEN -- a deck's header -- names
    the stage in the command it prints: the spelling
    :func:`resolve_stage_ref` accepts, its NAME (``02_freq`` -> ``freq``;
    `job-system.md` § 5.3).  Every other printed line already holds a name.

    Never the token: ``02_freq`` is itself a legal stage name, of another
    stage (plan § 5w K12).  Never ``#N``: an unquoted ``#`` begins a
    comment in bash, so a pasted ``launch run #2`` names no stage."""
    m = re.fullmatch(r"\d{2,}_([A-Za-z0-9_]+)", str(token))
    if m is None:
        raise ValueError(f"not a stage token: {token!r}")
    return m.group(1)


#: What the mode means, beside a launch line a text read LATER carries -- a
#: deck's header, a result's remedy -- every engine's.
LAUNCH_MODE_NOTE = ("(--mode direct runs it here, --mode submit sends it to "
                    "the queue; unsaid, molbuilder.json's launch.mode "
                    "decides)")


def launch_as_typed(stage: str, trial: Optional[str] = None) -> str:
    """A stage's launch -- its run, or THIS trial alone -- as typed from the
    calculation's folder: what a text read LATER says, wherever it was
    copied (a deck's header, a result's remedy), so it names no folder and
    no mode; :data:`LAUNCH_MODE_NOTE` says what the mode means.  ONE wording,
    for every engine's deck and remedy (W52) -- never ``--mode
    direct|submit``, which bash reads as a pipe.  ``stage`` is the stage's NAME -- never its token, a
    legal name of another stage (`job-system.md` § 5.3)."""
    return ("molbuilder jobset launch "
            + (f"bench {stage} {trial}" if trial else f"run {stage}"))


def checkpoint_as_typed(folder=None) -> Tuple[str, str]:
    """``(list, restore)`` -- the two checkpoint commands a way back names,
    the calculation's folder in each (`checkpoint -p`): ONE spelling, which
    the command composer (`jobset.commands.rollback`) and every layer below
    it say.  ``folder`` ``None`` is a text read LATER, wherever it was copied
    -- then the person types them from the calculation's folder, and the
    sentence around them says so."""
    import shlex
    p = "" if folder is None else f" -p {shlex.quote(str(folder))}"
    return (f"molbuilder checkpoint list{p}",
            f"molbuilder checkpoint restore{p}")


def checkpoint_words(folder=None) -> str:
    """The way back, in a sentence's words: which command shows the
    folder's saved states and which restores the one picked
    (:func:`checkpoint_as_typed`)."""
    shown, restore = checkpoint_as_typed(folder)
    return (("from the calculation's folder: " if folder is None else "")
            + f"`{shown}` shows the folder's states, `{restore}` takes the "
              f"one you pick")


def deck_launch(stage_token: Optional[str],
                trial: Optional[str] = None) -> Optional[str]:
    """How a deck names its own managed launch -- :func:`launch_as_typed`,
    for the stage its token names; ``None`` for a deck written outside a
    ladder: it has no stage to name."""
    if not stage_token:
        return None
    return launch_as_typed(command_stage(stage_token), trial)


_TOKEN_RE = re.compile(r"(\d{2,})_([A-Za-z0-9_]+)")


def parse_token(token: str) -> Optional[Tuple[int, str]]:
    """The ``(seq, name)`` of a bare stage token -- ``01_coarse`` -- or
    ``None``: the inverse of :func:`stage_token`, for a stage DIRECTORY's
    name; a filename's is `runfiles.parse`'s."""
    m = _TOKEN_RE.fullmatch(str(token))
    return (int(m.group(1)), m.group(2)) if m else None


def seq_text(seq: Optional[int]) -> str:
    """What a ``seq`` column prints: the ordinal, or ``-`` where there is none.

    A sweep point has no order, and a ladder job whose deck carries no token has
    no assigned ordinal. Both must print *something*, and the one thing they
    must not print is the row they happen to occupy — that is the number
    ``engines/stages.md`` R5 forbids as an identifier, and putting it under a
    column headed ``seq`` is exactly how it gets read as one.

    ``-`` is the same *not applicable* these tables already print for an absent
    dependency or an absent carry, so it needs no explaining to a reader.
    """
    return "-" if seq is None else str(seq)


@dataclass(frozen=True)
class StageRef:
    """A stage, as every surface refers to one: an ordinal and a name.

    **Both halves, because each answers something the other cannot.** The name
    is the stage's *identity* (``engines/stages.md`` R5) and is what a filename
    can be read back to. The ``seq`` is what **sorts** — with eight stages, a
    listing ordered by name says nothing about the order the work happens in,
    which is the whole reason decision 27 put the ordinal in the token.

    **``seq`` is derived, never stored** (decision 28). Before a produce it
    comes from the ladder's full list; after one it is read back off the
    artifacts — `project-layout.md` § 4.1's *"`seq` is not a fourth field …
    the description does not carry it"*. Constructing this is therefore always
    someone reading, never someone deciding.

    **``seq`` is ``None`` where there is no assigned ordinal**, and that is a
    real state rather than a missing value: a sweep point's points are
    independent and have no order at all, and a hand-written ladder job may
    carry no token. It is never filled in from a position — § 4.2's number is
    assigned once and never guessed, so *unknown* stays unknown all the way out
    to the screen (:func:`seq_text`).
    """
    seq: Optional[int]
    name: str

    @classmethod
    def ladder(cls, names: Sequence[str]) -> Tuple["StageRef", ...]:
        """One ref per stage of the **full** ladder — the pre-produce builder.

        Decision 28's first arm, made callable: *"before a produce it comes
        from the ladder's full list"*. The ordinal is the stage's place in
        that list counted from 1 (a stage removed after its prep keeps its
        number on disk: `materialize.stage_home`'s rule, stated once). The after-produce arm — reading ``seq`` back off
        the decks — is `jobset/materialize.py::stage_refs`, and A4 allows
        exactly those two: the owner and the class's own method.
        """
        return tuple(cls(i, n) for i, n in enumerate(names, start=1))

    @property
    def token(self) -> Optional[str]:
        """``<NN>_<name>`` — or ``None``, when there is no ordinal to put in it.

        Optional rather than a bare name, because a job without an ordinal has
        no token: it is named by the *other* convention (``bench-<name>``), and
        handing back something token-shaped would invite a caller to write it
        into a path.
        """
        return None if self.seq is None else stage_token(self.seq, self.name)

    @property
    def label(self) -> str:
        """What a listing prints — the token where there is one, else the name."""
        return self.token or self.name

    @property
    def seq_text(self) -> str:
        """What a ``seq`` column prints, from the one rule."""
        return seq_text(self.seq)

    def __str__(self) -> str:                       # what a listing prints
        return self.label


def resolve_stage_ref(refs: Sequence["StageRef"], text: str) -> "StageRef":
    """The one job ``text`` names, or ``ValueError`` naming every candidate.

    TWO spellings, each unambiguous (user-settled 2026-08-21)::

        tight        the name           — in any case (:func:`stage_key`),
                                          nothing else
        #3           the stage's number — '#' then its assigned ``seq``

    A bare number and the whole token (``03_tight``) are LEGAL STAGE NAMES
    (`engines/stages.md` § 2 allows ``[A-Za-z0-9_]+``), so neither may mean
    an ordinal.  ``#`` cannot appear in a name, which is what makes
    the prefix collision-free; everything without it is a name, full stop.

    ``#N`` is matched against ``seq`` and **never** against a position in
    the list. That distinction is the one `engines/stages.md` R5 exists to
    protect: with stage 2 removed after its prep the ladder is ``01`` and
    ``03``, so
    ``#3`` must mean *tight* and can never mean *"the third row"* — the
    same number the directory (``03_tight``) shows.

    A set with no ordinals — a sweep — resolves by name through this same
    function, and the refusal stops offering numbers it does not have. One
    resolver for both kinds is the point: a second lookup for the kind
    without ordinals is a second refusal wording and a second listing
    format waiting to disagree with this one.
    """
    want = str(text).strip()
    if not want:
        raise ValueError(
            "no stage named; pass its name, or '#N' for its number")
    if want.startswith("#"):
        num = want[1:]
        if num.isdigit():
            for r in refs:
                if r.seq is not None and int(num) == r.seq:
                    return r
        if not refs:
            raise ValueError(
                f"no stage {text!r}: this job-set has no jobs")
        raise ValueError(
            f"no stage numbered {text!r} in this job-set; it has: "
            f"{render_stage_choices(refs)}.")
    for r in refs:
        if stage_key(want) == stage_key(r.name):
            return r
    if not refs:
        raise ValueError(f"no stage named {text!r}: this job-set has no jobs")
    raise ValueError(
        f"no stage named {text!r} in this job-set; it has: "
        f"{render_stage_choices(refs)}.")


def render_stage_choices(refs: Sequence["StageRef"]) -> str:
    """What a refusal offers — the TYPEABLE spellings: ``coarse ('#1'),
    tight ('#3')``; bare names where there are no ordinals.  ``#N`` is QUOTED,
    because bash reads an unquoted ``#`` as the start of a comment
    (`job-system.md` § 5.3).

    A sibling of :func:`render_stage_refs`, split on purpose: the table
    format shows what is ON DISK (the token, ``03_tight``), this shows
    what the resolver ACCEPTS (name or ``#N`` — user-settled 2026-08-21),
    and printing the on-disk form in a refusal would offer a spelling
    that no longer resolves.
    """
    return ", ".join(f"{r.name} ('#{r.seq}')" if r.seq is not None else r.name
                     for r in refs)


def render_stage_refs(refs: Sequence["StageRef"]) -> str:
    """The one listing format every surface prints — ``01_coarse, 03_tight``.

    One function so a refusal, a status table and a help string cannot show a
    user three different vocabularies for one set of jobs. Where there are no
    ordinals it degrades to the names, which is what a sweep has.
    """
    return ", ".join(r.label for r in refs)


def is_ours(name: str, label: str) -> bool:
    """Did **molbuilder** write this file, rather than an engine?

    The inversion `job-contracts.md § 4.2` rests on: we cannot enumerate what
    SIESTA produces, but we always know what we produced.

    ``label``, not the id: every pattern below is a **filename**, and since
    § 2.0a the stem of a filename is the label. Handing this the composite id
    would match nothing and report every warm file as the engine's — or, worse
    in :func:`~molbuilder.validation.identity.warm_files_present`, report that
    nothing has run at all.

    **A TRIAL'S LABEL IS A LABEL TOO**, and that is why the patterns are tried
    against two prefixes rather than being written out twice. A benchmark
    relabels each point ``<label>-<coordinate>`` so its warm files can never
    meet the run's (`project-layout.md` § 2.3.2), and everything molbuilder
    writes for that point is stemmed on the new label: ``bdt-G1K4C6_01_coarse.fdf``,
    its wrapper, its trajectory log -- matched against ``{label}`` alone, the
    whole trial set would read as engine restart state at the
    `run-identity.md` § 6 prompt, the one moment a person is asked to stop
    and read.

    Trying ``{label}-*`` as well as ``{label}`` is the rule
    `job-contracts.md` § 6.3 already states — ``-`` announces ONE qualifier,
    and ``resolve.point_token`` keeps the coordinate inside ``[A-Za-z0-9_]``
    so it cannot contain another. Deriving it costs one line; writing the
    qualified forms out would double a list whose failure mode is a pattern
    that silently stops matching.
    """
    import fnmatch
    return any(fnmatch.fnmatchcase(name, p.format(label=stem))
               for stem in (label, f"{label}-*")
               for p in OUR_FILE_PATTERNS)


def normalise_id(raw: str, *, stage_names: Sequence[str] = (),
                 what: str = "name") -> str:
    """§ 3's normalisation, and its three rules. Refuses rather than guessing.

    ``raw`` is what a person typed. The result is the ``SystemLabel`` / ``JOB``
    literal and the stem of every file in the calculation, so it is checked
    here and never rewritten downstream (§ 3 rule 1).

    ``stage_names``, when the ladder is already known, replaces
    :data:`_STAGE_BUDGET` with the real longest name — the cap is about what
    ``<label>_<stage>.<ext>`` will occupy, and guessing is only necessary while
    the stages do not exist yet.

    ``what`` names the thing being normalised, for the refusals only.
    :func:`run_id` calls this twice — once for the label and once for the
    formula (§ 2.0a) — and a person who mistyped a formula should not be told
    to rename their calculation.

    Raises ``ValueError`` for every case § 3 rule 3 refuses. A refusal names
    the offending character, because this is a name a person chose and *"say so
    and ask"* is useless without saying what.
    """
    text = raw or ""

    # -- rule 3, first half: a letter or a digit may not be replaced --------
    #  Checked BEFORE substituting, because afterwards every offender looks
    #  like an ordinary `_` and the thing that was lost is unrecoverable.
    for ch in text:
        if _ALLOWED_CHAR.fullmatch(ch) or not ch.isalnum():
            continue
        raise ValueError(
            f"{raw!r} is not a usable {what}: {ch!r} is a letter or a digit "
            f"outside [A-Za-z0-9_-], and dropping it would silently make a "
            f"different name. Rename it using unaccented ASCII "
            f"(run-identity.md § 3, rule 3)")

    # -- the transform, in § 3's order -------------------------------------
    out = "".join(c if _ALLOWED_CHAR.fullmatch(c) else "_" for c in text)
    out = _SEPARATOR_RUN.sub("_", out)
    out = _EDGE_SEPARATORS.sub("", out)

    # -- rule 3, second half: nothing left, or over the cap ----------------
    if not out:
        raise ValueError(
            f"{raw!r} is not a usable {what}: it is entirely separators, so "
            f"nothing is left to name the calculation "
            f"(run-identity.md § 3, rule 3)")

    cap = _cap_for(stage_names)
    if len(out.encode("utf-8")) > cap:
        raise ValueError(
            f"{raw!r} is not a usable {what}: it normalises to {len(out)} "
            f"characters and the limit is {cap}. It is refused rather than "
            f"truncated -- a shortened name is a different calculation wearing "
            f"the same one (run-identity.md § 3)")
    return out


def _cap_for(stage_names: Sequence[str]) -> int:
    """The cap, derived against a real ladder where one is known."""
    if not stage_names:
        return MAX_LABEL_BYTES
    longest = max(len(n) for n in stage_names)
    return _NAME_LIMIT - (len("_") + longest) - len(_LONGEST_EXTENSION)


def run_id(label: str, formula: str = "", *,
           stage_names: Sequence[str] = ()) -> str:
    """§ 2's id: what the user calls it, and what the coordinates are of.

    Those are the only two inputs, and that is the whole rule. **Nothing a run
    produced may reach here** — no positions, no energy, no convergence status,
    nothing read back off a ``.XV`` — because an id that depended on a result
    would change the moment a stage succeeded, orphaning the state it exists to
    continue from (§ 2).

    It is equally blind to everything a *stage* tunes: mesh, tolerances, force,
    steps, algorithm, basis, XC, ranks, threads, GPU (§ 2.1's table).  (The
    charge and spin are not among them: they belong to the calculation, and a
    stage cannot change them -- ES1, `science/chemistry-correctness.md`
    § 2a.)
    That blindness is not an oversight — it is what makes several stages one
    calculation rather than several.

    ``formula`` is optional because a label alone is a legitimate id; when
    absent the id and the label are the same string.

    The timestamp is not here on purpose. A description records when it was
    written (``task.Run.created``); putting that in the id would make every
    regeneration a new identity and therefore a cold start (§ 2).

    **The two halves are normalised apart, and that is not a refactor**
    (§ 2.0a, decision 26, 2026-08-09). The label becomes the ``SystemLabel``
    and the stem of every file, so it carries the **filename cap** — 255 bytes
    less what ``_<stage>.<longest extension>`` will occupy. The id goes into
    ``task.json`` and never becomes a filename, so nothing bounds the pair.
    Normalising the *joined* string would apply the filename cap to a string
    that is not a filename, and refuse a long name plus a long formula for a
    limit neither of them would ever meet on disk.

    The formula still goes through the same alphabet, because a witness a
    person may hand-edit is a witness that can be wrong; splitting the call is
    what lets each refusal name the half that is actually at fault.
    """
    stem = normalise_id(label, stage_names=stage_names, what="name")
    if not formula:
        return stem
    return f"{stem}_{normalise_id(formula, what='formula')}"


__all__ = ["LAUNCH_MODE_NOTE", "MAX_LABEL_BYTES", "OUR_FILE_PATTERNS",
           "RestartGroup", "StageRef", "command_stage", "continues",
           "checkpoint_as_typed", "checkpoint_words",
           "deck_launch", "is_ours", "launch_as_typed", "normalise_id",
           "render_stage_choices", "render_stage_refs",
           "resolve_stage_ref", "run_id", "seq_text", "stage_key",
           "stage_token"]
