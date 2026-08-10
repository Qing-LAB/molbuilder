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

Since 2026-08-09 ``task.py`` calls in rather than trusting what it parsed: it
derives ``run.id`` from ``run.name`` and ``structure.formula`` and refuses a
description that disagrees with itself (``Task._check_id``).  That is what makes
*stored* mean anything — before it, ``id`` was a free string, and the header
here still described a file that *"already has an id"*.

**On the character set.** § 3 is explicit that the set *"is not a new
decision"* — `job-contracts.md § 2.1` Rule 2 fixes it, and this module does not
get to widen it. Four validators in the tree already spell it
(``projects._NAME_PATTERN``, ``checkpoint._REF_SAFE_ID`` — which cites this very
contract — plus ``bench/adapters`` and ``config/pyscf``). **None of them
normalises**; they all reject. This is the first transform, so nothing is being
duplicated here. Consolidating those four spellings is a real cleanup and is
*not* this phase's — P3 subtracts *"any second normaliser"*, and there is not
even a first one. What the tests do instead is assert that every id this
produces is accepted by the shipped validators, so the agreement is checked
rather than assumed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


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
#: Called ``MAX_ID_BYTES`` until 2026-08-09.  It never bounded the id -- it
#: bounds what goes in a **filename**, and § 2.0a is that only the label does.
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
OUR_FILE_PATTERNS: Sequence[str] = (
    # inputs we generated
    "{label}.fdf", "{label}_*.fdf", "{label}.fdf.template",
    "{label}.py", "{label}_*.py",
    # the wrapper and its scheduler header
    "{label}.run.sh", "{label}_*.run.sh",
    "{label}.sbatch", "{label}_*.sbatch",
    # the canonical trajectory -- written before the engine even starts
    "{label}.molwatch.log", "{label}_*.molwatch.log",
    # engine stdout -- the run's history.  Three shapes, because three things
    # create one: the wrapper's run-indexed redirect, the flat ladder runner's
    # ``> ${BASENAME}_${stage}.out``, and a single unstaged run's ``.out``.
    #
    # The last two were MISSING until 2026-08-10 -- a pre-existing gap, not
    # fallout from the stage rename (``<label>-stage1.out`` did not match
    # either).  It matters because ``warm_files_present`` answers *has anything
    # run here* by SUBTRACTION: anything named after the label that is not on
    # this list is reported as the engine's restart state.  So a run's own
    # stdout was being offered back to the user as a warm file, and counted
    # among what a rename would orphan.
    "{label}.out", "{label}_*.out",
    "{label}-run*.out", "{label}_*-run*.out",
    "{label}-run*.pyscf.log", "{label}_*-run*.pyscf.log",
    # geomeTRIC's own optimizer log
    "{label}.log", "{label}_geom_*.log",
)


#: A stage's artifact token, ``<NN>_<name>`` — as it appears in a filename and
#: as a stage directory is named.  Two capture groups: the ordinal and the name.
#:
#: The name may itself contain ``_`` (``STAGE_NAME_RE`` is ``[A-Za-z0-9_]+``),
#: and so may the label — which is why this is only ever applied **anchored to
#: the end** of a filename, never searched loosely.
_STAGE_TOKEN = re.compile(r"(\d{2,})_([A-Za-z0-9_]+)")


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


def parse_stage_token(filename: str,
                      label: str = "") -> Optional[Tuple[int, str]]:
    """The ``(seq, name)`` a filename's stage token carries, or ``None``.

    Anchored at the tail: after the token comes an optional ``-run<N>``
    counter, then the extension, and nothing else.

    **Pass ``label`` whenever it is known.**  Both a label and a stage name may
    contain ``_``, so without it a calculation the user called
    ``run_01_setup`` has a plain deck ``run_01_setup.fdf`` that reads as stage
    1 *"setup"*.  With it, everything before the token must be exactly the
    label and the ambiguity is gone.  Callers that genuinely do not know the
    label get the best-effort read, which is what the old ``-stage<N>`` regex
    gave them too.
    """
    tail = r"_(\d{2,})_([A-Za-z0-9_]+?)(?:-run\d+)?\.[A-Za-z0-9.]+$"
    m = (re.fullmatch(re.escape(label) + tail, filename) if label
         else re.search(tail, filename))
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def is_ours(name: str, label: str) -> bool:
    """Did **molbuilder** write this file, rather than an engine?

    The inversion `job-contracts.md § 4.2` rests on: we cannot enumerate what
    SIESTA produces, but we always know what we produced.

    ``label``, not the id: every pattern below is a **filename**, and since
    § 2.0a the stem of a filename is the label. Handing this the composite id
    would match nothing and report every warm file as the engine's — or, worse
    in :func:`~molbuilder.validation.identity.warm_files_present`, report that
    nothing has run at all.
    """
    import fnmatch
    return any(fnmatch.fnmatchcase(name, p.format(label=label))
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
    steps, algorithm, basis, spin, XC, ranks, threads, GPU (§ 2.1's table).
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
    Normalising the *joined* string, as this did until today, applied the
    filename cap to a string that is not a filename: a long name plus a
    long formula was refused for exceeding a limit neither of them would ever
    meet on disk, and the message blamed the name.

    The formula still goes through the same alphabet, because a witness a
    person may hand-edit is a witness that can be wrong; splitting the call is
    what lets each refusal name the half that is actually at fault.
    """
    stem = normalise_id(label, stage_names=stage_names, what="name")
    if not formula:
        return stem
    return f"{stem}_{normalise_id(formula, what='formula')}"


__all__ = ["MAX_LABEL_BYTES", "OUR_FILE_PATTERNS", "RestartGroup", "is_ours",
           "normalise_id", "parse_stage_token", "run_id", "stage_token"]


# --------------------------------------------------------------------- #
#  StageRef — the one answer to "which stage is this"                    #
#  Contract: staged-runs-implementation-plan.md § 8f + § 9 (decision 28) #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class StageRef:
    """A stage, as every surface should refer to it: ``seq`` and ``name``.

    **Both halves, because each answers something the other cannot.** The name
    is the stage's *identity* (``engines/stages.md`` R5) and is what a filename
    can be read back to. The ``seq`` is what **sorts** — with eight stages, a
    listing ordered by name tells you nothing about the order the work happens
    in, which is the whole reason decision 27 put the ordinal in the token.

    **``seq`` is derived, never stored** (decision 28). It comes from the
    ladder's full list before a produce, and is read back off the artifacts
    after one — `project-layout.md` § 4.1's *"`seq` is not a fourth field …
    the description does not carry it"*. Constructing this object is therefore
    always someone reading, never someone deciding.
    """
    seq: int
    name: str

    @property
    def token(self) -> str:
        """``<NN>_<name>`` — the artifact token, from the one namer."""
        return stage_token(self.seq, self.name)

    def __str__(self) -> str:                       # what a listing prints
        return self.token


def resolve_stage_ref(refs: Sequence["StageRef"], text: str) -> "StageRef":
    """The one stage ``text`` names, or ``ValueError`` naming every candidate.

    Four spellings are accepted, because all four are things a person
    reasonably types for the same stage::

        tight        the name          — its identity
        3  /  03     the seq           — what sorts, and what you read off a listing
        03_tight     the whole token   — copied from a directory or a filename

    A number is matched against ``seq`` and **never** against a position in the
    list. That distinction is the one `engines/stages.md` R5 exists to protect:
    with stage 2 disabled the ladder is ``01`` and ``03``, so ``3`` must mean
    *tight* and never *"the third row"*.
    """
    want = str(text).strip()
    if not want:
        raise ValueError("no stage named; pass a name, a number, or a token")
    for r in refs:
        if want == r.name or want == r.token:
            return r
    if want.isdigit():
        for r in refs:
            if int(want) == r.seq:
                return r
    raise ValueError(
        f"no stage matches {text!r}. This calculation has: "
        f"{render_stage_refs(refs)}. Name it by its name (tight), its "
        f"number (3), or its token (03_tight).")


def render_stage_refs(refs: Sequence["StageRef"]) -> str:
    """The one listing format every surface prints — ``01_coarse, 03_tight``.

    One function so a refusal, a status table and a help string cannot show a
    user three different vocabularies for one set of stages.
    """
    return ", ".join(r.token for r in refs)


__all__ = __all__ + ["StageRef", "resolve_stage_ref", "render_stage_refs"]
