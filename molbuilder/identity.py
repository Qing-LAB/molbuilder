"""The run id — built from inputs, and normalised exactly once.

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
codec: it reads and writes a description that *already has* an id. Deriving one
is a different job with its own contract, and putting it here keeps the
document-to-module mapping one-to-one. It also means the CLI and the web tab
reach the same normaliser — § 3's rule 1 is *"it happens once, and the result is
stored"*, which is only true if there is one place it can happen.

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
from typing import Sequence, Tuple


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

#: The longest extension molbuilder writes beside an id, from
#: `job-contracts.md § 4.2`'s SIESTA inventory.  ``.STRUCT_NEXT_ITER`` is the
#: longest of the thirteen; PySCF's longest (``_geom_optim.xyz``) is shorter.
_LONGEST_EXTENSION = ".STRUCT_NEXT_ITER"

#: What ``_<stage name>`` may occupy when the ladder is not known yet.
#:
#: **This one is a budget, not a derivation, and the difference is worth
#: stating.** § 3 derives the cap from ``<id>_<longest stage name>.<longest
#: extension>``, and nothing in the system bounds a stage name — ``STAGE_NAME_RE``
#: is ``[A-Za-z0-9_]+`` with no length. An id is usually named before its
#: stages exist, so when they are unknown this reserves a generous fixed
#: budget; when they ARE known, pass ``stage_names`` and the real longest is
#: used instead of this.
_STAGE_BUDGET = 32

#: The most an id may occupy, derived: the name limit, less the longest stage
#: suffix and the longest extension that will be appended to it.
MAX_ID_BYTES = _NAME_LIMIT - _STAGE_BUDGET - len(_LONGEST_EXTENSION)


def normalise_id(raw: str, *, stage_names: Sequence[str] = ()) -> str:
    """§ 3's normalisation, and its three rules. Refuses rather than guessing.

    ``raw`` is what a person typed. The result is the ``SystemLabel`` / ``JOB``
    literal and the stem of every file in the calculation, so it is checked
    here and never rewritten downstream (§ 3 rule 1).

    ``stage_names``, when the ladder is already known, replaces
    :data:`_STAGE_BUDGET` with the real longest name — the cap is about what
    ``<id>_<stage>.<ext>`` will occupy, and guessing is only necessary while
    the stages do not exist yet.

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
            f"{raw!r} cannot become an id: {ch!r} is a letter or a digit "
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
            f"{raw!r} cannot become an id: it is entirely separators, so "
            f"nothing is left to name the calculation "
            f"(run-identity.md § 3, rule 3)")

    cap = _cap_for(stage_names)
    if len(out.encode("utf-8")) > cap:
        raise ValueError(
            f"{raw!r} cannot become an id: it normalises to {len(out)} "
            f"characters and the limit is {cap}. It is refused rather than "
            f"truncated -- a shortened id is a different calculation wearing "
            f"the same name (run-identity.md § 3)")
    return out


def _cap_for(stage_names: Sequence[str]) -> int:
    """The cap, derived against a real ladder where one is known."""
    if not stage_names:
        return MAX_ID_BYTES
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
    present it is joined with ``_`` and the pair is normalised **once**, so the
    join cannot introduce a separator run the normaliser never sees.

    The timestamp is not here on purpose. A description records when it was
    written (``task.Run.created``); putting that in the id would make every
    regeneration a new identity and therefore a cold start (§ 2).
    """
    joined = "_".join(p for p in (label or "", formula or "") if p)
    return normalise_id(joined, stage_names=stage_names)


__all__ = ["MAX_ID_BYTES", "RestartGroup", "normalise_id", "run_id"]
