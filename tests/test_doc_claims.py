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


# --------------------------------------------------------------------- #
#  Gate A -- a worked example must be a VALID example                    #
#                                                                        #
#  audit-2026-08-14 3.1/3.2: the contract's own TOML carried a DUPLICATE  #
#  KEY (so it did not parse at all), one example omitted `category` and   #
#  two omitted `type` and `help` -- all four of them keys 3 calls         #
#  REQUIRED on every item.  A contract whose illustrations its own reader #
#  would refuse teaches the wrong shape to everyone who copies one.       #
# --------------------------------------------------------------------- #

def _fenced_toml(doc_rel: str):
    """Every ```toml block in a document, de-indented, with its index."""
    text = (DOCS / doc_rel).read_text(encoding="utf-8")
    for i, block in enumerate(re.findall(r"```toml\n(.*?)```", text, re.S)):
        # Some examples are indented inside a list item; TOML does not mind
        # leading spaces on a key line, but a uniform two-space indent is
        # stripped so the block reads as it would on disk.
        yield i, "\n".join(ln[2:] if ln.startswith("  ") else ln
                            for ln in block.splitlines())


def test_every_template_example_parses_as_toml():
    """§ 6.3's example carried ``category`` twice. TOML forbids a duplicate
    key, so the contract's illustration of the format could not be loaded by
    the format's own reader."""
    import tomllib
    for i, body in _fenced_toml("engines/template.md"):
        try:
            tomllib.loads(body)
        except Exception as exc:              # noqa: BLE001 - report any of them
            pytest.fail(f"engines/template.md fenced toml block {i} does not "
                        f"parse: {exc}\n{body[:400]}")


def test_every_template_example_item_carries_the_required_keys():
    """And each ``[item.*]`` satisfies § 3 -- the same four the reader wants,
    with a ``type`` from the closed vocabulary."""
    import tomllib
    bad = []
    for i, body in _fenced_toml("engines/template.md"):
        try:
            parsed = tomllib.loads(body)
        except Exception:
            continue                          # the test above owns parse errors
        for name, item in (parsed.get("item") or {}).items():
            missing = [k for k in template._REQUIRED_ITEM_KEYS if k not in item]
            if missing:
                bad.append(f"block {i}, item {name!r}: missing {missing}")
            t = item.get("type")
            if t is not None and t not in template.TYPES:
                bad.append(f"block {i}, item {name!r}: type {t!r} is not in "
                           f"TYPES {template.TYPES}")
    assert not bad, (
        "engines/template.md illustrates items its own reader would refuse:\n  "
        + "\n  ".join(bad)
        + "\n(§ 3 lists the required keys; § 5 the type vocabulary.)"
    )


def test_the_frozen_label_in_the_spec_matches_the_code_constant():
    """`job-contracts.md` § 3.4's ATOM-METADATA example must spell the frozen
    label the way the code writes it.

    It said ``"frozen"`` until 2026-08-14 while ``structure.FROZEN_LABEL`` is
    ``"frozen_atoms"``.  The SHAPE was right -- frozen is an ordinary label
    inside ``regions`` -- and the NAME was not, which matters more than a typo:
    that example is the specification a reader of these labels is written
    from, and **transport** is the named consumer (electrode / bridge / frozen
    membership).  A reader built from the old example looks up ``"frozen"``,
    finds nothing, and concludes the run froze no atoms.

    Guards the direction that actually drifts -- the doc spelling a constant
    instead of citing it.
    """
    from molbuilder.structure import FROZEN_LABEL
    text = (DOCS / "execution/job-contracts.md").read_text(encoding="utf-8")
    block = text[text.index("### 3.4"):text.index("### 3.5")]
    assert f'"{FROZEN_LABEL}"' in block, (
        f"job-contracts.md § 3.4 never spells the frozen label "
        f"{FROZEN_LABEL!r} as the code writes it.  Either the example drifted, "
        f"or structure.FROZEN_LABEL changed and § 3.4 did not follow."
    )
    assert '"frozen":' not in block, (
        'job-contracts.md § 3.4 still shows the retired label `"frozen":` -- '
        f'the code writes {FROZEN_LABEL!r} inside `regions`.'
    )



# --------------------------------------------------------------------- #
#  pow2 is DECLARED, so the checker that was written for it can run      #
#                                                                        #
#  audit-2026-08-14 SS 48/50: `pow2` was in TYPES, documented in SS 5 and  #
#  SS 12's example, had a correct entry in _TYPE_CHECKS -- and no code     #
#  path could produce it, so a user-supplied 96 reached the deck while   #
#  the AUTO path capped to a power of two.                               #
# --------------------------------------------------------------------- #

def test_block_size_declares_pow2_so_a_non_power_of_two_is_refused():
    """The three states of ``parallel_block_size`` all behave.

    ``None``/absent = auto (prep proposes) · ``0`` = omit the keyword, the
    engine's own default (`template.md` § 12, decision 35) · ``N`` = use N,
    **and N must be a power of two** -- which is what the declaration turns on.
    """
    from molbuilder.config.siesta import SiestaConfig

    item = template.one(template.read_template(
        template.render_template(SiestaConfig())), "parallel_block_size")
    assert item.type == "pow2", (
        "parallel_block_size must DECLARE pow2 -- otherwise the checker "
        "written for it is unreachable and a user's 96 is emitted verbatim")

    def _round_trip(v):
        return template.one(template.read_template(template.render_template(
            SiestaConfig(parallel_block_size=v))), "parallel_block_size").value

    # A legal value is untouched.
    for good in (1, 16, 64, 128):
        assert _round_trip(good) == good

    # An illegal one is COERCED, not refused (user, 2026-08-14).  DOWN, which
    # is the direction ``_auto_block_size`` already takes -- the largest power
    # of two that still leaves >= 2 blocks per rank -- so the snap can never
    # hand the engine a bigger block than was asked for.
    assert _round_trip(96) == 64
    assert _round_trip(100) == 64
    assert _round_trip(3) == 2

    # 0 is the THIRD STATE -- *omit the keyword entirely* -- not a value, so
    # the power-of-two constraint does not apply to it and it must survive.
    assert _round_trip(0) == 0


def test_a_declared_type_must_be_in_the_vocabulary():
    """``metadata['decl_type']`` is checked against ``TYPES`` -- a typo there
    would otherwise put an unknown type into every template."""
    import dataclasses
    fld = dataclasses.field(metadata={"decl_type": "not_a_type",
                                      "category": ("execution",),
                                      "help": "x"})
    fld.name, fld.type = "x", int
    with pytest.raises(ValueError, match="not in the type vocabulary"):
        template.declaration_for(fld, int)
