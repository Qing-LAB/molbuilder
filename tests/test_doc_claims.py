"""Doc-claims contract — a closed vocabulary means the same thing in every
place that names it.

**The rule.** A vocabulary that is closed in code (``template.KINDS``,
``template.CATEGORIES``, the fields of ``jobset.Resources`` …) is also
*enumerated in prose* by the contract that owns it. Those two lists must agree,
in both directions: a member the document omits is a member no reader learns
about, and a member the document names that the code does not have is a reader
sent looking for something that is not there.

**Why this exists** (`docs/audit-2026-08-14-template-execution-review.md`
§ 22.1). A full-text review on 2026-08-14 found the same enumeration wrong in
two documents at once:

* ``jobset.Resources`` has **nine** fields. ``job-contracts.md`` § 6.2 says
  nine — it had said *"seven"*, was corrected, and was pinned with a test.
  ``job-system.md`` § 3 still showed **seven**, in a class diagram AND in an
  annotated example whose own caption two lines above said *"ALL NINE fields
  are always written"*.
* The fix to the contract never reached the guide, **because nothing connected
  them.** That is the defect this file closes: not a wrong number, but a
  correction that could not propagate.

**What this cannot do, stated so nobody assumes otherwise.** It checks
enumerations, not prose. The same review found four passages describing an
eigensolver-routing rule that had been deleted, a floor-3 module importing from
an engine package, and an invariant claiming a mechanism nobody wrote — none of
which any test of this shape can see. Roughly a third of that review was
mechanical; this is that third.

**Adding a vocabulary here is the point.** When a new closed set appears, add a
row: the test is the maintenance mechanism for the cross-index (§ 21.1), which
is otherwise a snapshot that rots the moment someone edits a document.
"""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from molbuilder import template
from molbuilder.jobset.model import Job, Resources, WarmFile
from molbuilder.task import Stage, Task

DOCS = Path(__file__).resolve().parents[1] / "docs"


def _members(obj) -> tuple:
    """The vocabulary: a tuple of strings, or a dataclass's field names."""
    if dataclasses.is_dataclass(obj):
        return tuple(f.name for f in dataclasses.fields(obj))
    return tuple(obj)


#: ``id: (the code's vocabulary, the document that OWNS it)``.
#:
#: The owner is the contract a reader is sent to for the definitive list --
#: `job-contracts.md` § 6.3's rule, applied to vocabularies: *if any other
#: document disagrees with the owner, the other is the bug.*
VOCABULARIES = {
    "item kind":        (template.KINDS,       "engines/template.md"),
    "category":         (template.CATEGORIES,  "engines/template.md"),
    "item type":        (template.TYPES,       "engines/template.md"),
    "resolver":         (template.RESOLVERS,   "engines/template.md"),
    "Resources field":  (Resources,            "execution/job-contracts.md"),
    "Job field":        (Job,                  "execution/job-contracts.md"),
    "WarmFile field":   (WarmFile,             "execution/job-contracts.md"),
    "Stage field":      (Stage,                "engines/stages.md"),
}


@pytest.mark.parametrize("vocab_id", sorted(VOCABULARIES))
def test_the_owning_document_names_every_member(vocab_id):
    """Every member of a closed vocabulary appears in its owning contract.

    A member the document never names is a member no reader learns about --
    which is how ``continue_retries`` and ``max_memory_mb`` were invisible in
    `job-system.md` § 3 while riding every `job-set.json` on disk.
    """
    members, doc_rel = VOCABULARIES[vocab_id]
    text = (DOCS / doc_rel).read_text(encoding="utf-8")
    missing = [m for m in _members(members) if m not in text]
    assert not missing, (
        f"{doc_rel} owns the {vocab_id} vocabulary but never names "
        f"{missing}.  Either the document is stale, or the member was added to "
        f"the code without the contract that defines it "
        f"(audit-2026-08-14-template-execution-review.md § 22.1)."
    )


def test_the_retired_type_is_named_nowhere_as_live():
    """``strmap`` was retired 2026-08-13 with PySCF's ``ecp`` rewrite.

    A retired member is the reverse failure of a missing one: the document
    offers a reader something the code will refuse.  This is narrow on purpose
    -- a general "no document names a non-member" check would fire on every
    prose mention of a retired thing, and retirements are worth recording.
    What must not survive is a retired member sitting in the LIVE type list.
    """
    text = (DOCS / "engines/template.md").read_text(encoding="utf-8")
    for line in text.splitlines():
        if "`strmap`" not in line:
            continue
        assert ("RETIRED" in line or "retired" in line), (
            f"engines/template.md still offers `strmap` as a live type:\n"
            f"  {line.strip()}\n"
            f"It left TYPES on 2026-08-13 -- {template.TYPES}."
        )


def test_resources_is_nine_fields_and_the_contract_says_so():
    """The instance that motivated this file, kept as its own test.

    `job-contracts.md` § 6.2 states the count in prose (*"exactly nine"*), so
    the number is checkable and not only the member list.  It was wrong once,
    in both directions -- the contract said seven while its own table carried
    eight, and the guide showed seven while its own caption said nine.
    """
    n = len(dataclasses.fields(Resources))
    assert n == 9, f"Resources has {n} fields; update this test AND § 6.2"
    text = (DOCS / "execution/job-contracts.md").read_text(encoding="utf-8")
    assert re.search(r"exactly \*\*nine\*\*|holds exactly \*\*nine\*\*", text), (
        "job-contracts.md § 6.2 no longer states the field COUNT in prose. "
        "The count is what drifted last time (seven vs nine), so it is worth "
        "stating and worth pinning."
    )


def test_the_template_required_keys_agree_across_their_three_homes():
    """§ 3's *"four required keys"* is stated in three places.

    The reader (`_REQUIRED_ITEM_KEYS`), the contract (§ 3's table), and
    ``Item``'s docstring -- which named a DIFFERENT four (`name`, `kind`,
    `type`, `help`) until this was found, omitting `category` and adding the
    key that is not a key at all.
    """
    required = template._REQUIRED_ITEM_KEYS
    assert set(required) == {"kind", "category", "type", "help"}, required
    text = (DOCS / "engines/template.md").read_text(encoding="utf-8")
    for key in required:
        assert f"`{key}`" in text, f"§ 3 never names the required key {key!r}"
