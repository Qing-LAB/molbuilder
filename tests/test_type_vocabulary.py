"""One type vocabulary, five readers, and each says how much of it it reads.

**Why this file exists.** A parameter's `type` is read at six points — is this
stored value legal, snap it into shape, which form control, what Python type
does a swept value read back as, what type is this dataclass field, and what
do we tell an outside benchmark tool.  Five of those are internal readers of
ONE vocabulary (`template.TYPES`); that is healthy, not fragmentation.

What was missing is that **nothing checked them against the source.**  Add a
twelfth type today and nothing fails: the value checker silently stops
checking it, the form falls through to a text box, and the sweep reader drops
it.  Each satellite would be quietly wrong, and the first sign would be a user
typing a value nobody validated.

That is `max_mem_gb` one layer up — a name declared in one place and not
answered in another — so the guard is the same shape as the scheduler
contract's R2: *a new member arrives with its answer, or it does not arrive.*

Each satellite below declares **complete** or **narrower with its reason**, and
the omissions are named exactly.  A new type therefore forces a decision at all
five, and a satellite that quietly stops covering one fails here.
"""
from __future__ import annotations

import pytest

from molbuilder.jobset.summarize import _ITEM_TYPES
from molbuilder.script_emit import benchmark_declarable_types
from molbuilder.template import TYPES, _ANNOTATION_TYPES, _TYPE_CHECKS
from molbuilder.web.blueprints._shared import _CONTROL_FOR_TYPE

#: Every reader of the vocabulary: the keys it answers for, and what it omits.
#:
#: ``omits`` is EXACT, not a floor.  Naming the set rather than counting it is
#: what turns "we added a type" into a failure here rather than a silent hole
#: three modules away.
_SATELLITES = {
    "template._TYPE_CHECKS": (
        set(_TYPE_CHECKS), set(),
        "the value checker answers for every type or a value goes unchecked"),
    "_shared._CONTROL_FOR_TYPE": (
        set(_CONTROL_FOR_TYPE), {"enum"},
        "an enum is a select whatever its underlying type, so `_control_for` "
        "answers it from `choices` BEFORE consulting this map"),
    "summarize._ITEM_TYPES": (
        set(_ITEM_TYPES), {"float3", "int3", "intlist", "strlist", "text"},
        "nothing sweeps a list or free text, so no proposal writes one"),
    "template._ANNOTATION_TYPES": (
        set(_ANNOTATION_TYPES.values()),
        {"enum", "float3", "int3", "intlist", "pow2", "strlist", "text"},
        "a DERIVATION from a Python annotation, so it covers only the types a "
        "bare annotation can express -- `pow2` and `enum` are constraints a "
        "person declares, not shapes `int`/`str` reveal"),
    "script_emit.benchmark_declarable_types()": (
        set(benchmark_declarable_types()),
        {"bool", "float3", "int3", "intlist", "strlist", "text"},
        "the BENCH-MARKS override grammar: what a harness may be told it can "
        "turn.  DERIVED from the source by a stated rule rather than typed "
        "beside it: a benchmark varies a scalar it can order or enumerate, "
        "so a shape or a family is not declarable (`job-contracts.md` 3.3)"),
}


def test_the_source_vocabulary_is_the_one_everyone_narrows():
    """Sanity: `template.TYPES` is a set of distinct names, and it is the
    thing the rest of this file measures against."""
    assert len(set(TYPES)) == len(TYPES), "a duplicate name in TYPES"
    assert len(TYPES) >= 10, "TYPES shrank unexpectedly; re-read the table"


@pytest.mark.parametrize("name", sorted(_SATELLITES))
def test_no_reader_invents_a_type_of_its_own(name):
    """**One vocabulary.**  A satellite may read less than the source; it may
    never read something the source does not define.  A key here that `TYPES`
    lacks is a second vocabulary being born."""
    keys, _omits, _why = _SATELLITES[name]
    invented = sorted(keys - set(TYPES))
    assert not invented, (
        f"{name} answers for {invented}, which `template.TYPES` does not "
        f"define -- that is a second vocabulary, not a narrowing")


@pytest.mark.parametrize("name", sorted(_SATELLITES))
def test_each_reader_omits_exactly_what_it_declares(name):
    """**The guard.**  Each satellite states the types it does not answer for.
    Adding a type to `TYPES` fails every satellite that has not decided about
    it, which is the whole point: the decision happens once, at the source,
    for all five -- instead of being discovered later in whichever reader was
    asked first."""
    keys, omits, why = _SATELLITES[name]
    actual = set(TYPES) - keys
    assert actual == omits, (
        f"{name} omits {sorted(actual)} but declares {sorted(omits)}.\n"
        f"Its stated reason for narrowing: {why}\n"
        f"If a type was just added to `template.TYPES`, decide here what this "
        f"reader does with it -- answering for it, or naming it in `omits` "
        f"with why.")


def test_a_complete_reader_stays_complete():
    """The value checker is the one satellite that must answer for everything:
    a type it does not check is a value nothing validates, anywhere."""
    keys, omits, _ = _SATELLITES["template._TYPE_CHECKS"]
    assert not omits and keys == set(TYPES), (
        "`_TYPE_CHECKS` no longer covers the whole vocabulary -- some type's "
        "values are now accepted unchecked")
