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

def test_block_size_is_a_plain_int_and_survives_the_round_trip():
    """The two states of ``block_size`` (tuning.md § 2.11, revised
    2026-08-15): absent = *auto*, the keyword is not emitted and SIESTA uses
    its own automatic; ``N`` = use N, verbatim.

    IT IS NOT ``pow2``, and that is the point of this test.  ``pow2`` does not
    check -- ``template._shape`` SNAPS the value down to the nearest power of
    two -- so a benchmarked 24 silently became 16 and nothing recorded why.
    The power-of-two rule is real but belongs to a different keyword under a
    narrower condition: the manual states it for ``Diag.BlockSize``, only with
    a GPU-enabled ELPA, where breaking it is not even an error (ELPA falls
    back to the CPU).  `prep` realigns it there, where the GPU flag and the
    rank count are both known; `pow2` survives in BENCH-MARKS, a constraint
    the benchmark puts on its own sweep.
    """
    from molbuilder.config.siesta import SiestaConfig

    item = template.one(template.read_template(
        template.template_with_values(SiestaConfig())), "block_size")
    assert item.type == "int", (
        "block_size must be a plain int -- `pow2` silently rewrites "
        "a measured value, which is the opposite of honouring a benchmark")

    def _round_trip(v):
        return template.one(template.read_template(template.template_with_values(
            SiestaConfig(block_size=v))), "block_size").value

    # Every value a user may set survives untouched, power of two or not.
    for v in (1, 16, 24, 64, 96, 100, 128):
        assert _round_trip(v) == v, f"{v} was rewritten"

    # Absent stays absent -- that is how *auto* is said, and it is what makes
    # the deck omit the keyword.
    assert _round_trip(None) is None


# --------------------------------------------------------------------- #
#  Gate B -- a worked example must illustrate a REAL item                 #
#                                                                        #
#  Gate A checks an example is a VALID template.  It cannot check the     #
#  example is a TRUE one, and on 2026-08-17 a review found § 4.2 -- the   #
#  section titled "A template, entire" -- illustrating three items that   #
#  do not exist: `frozen_indices` and `user_custom` were never added to   #
#  any schema, and `species_order` was shown as kind="deck" with an       #
#  `expands` it does not have (it is `produce`).  A worked example is the #
#  shape everyone copies, so a wrong one teaches a wrong classification.  #
# --------------------------------------------------------------------- #

#: Keys where the document and the catalogue must agree exactly when the
#: document chooses to show them.  ``value`` is deliberately absent: an
#: example may legitimately show a different answer than the shipped default,
#: because the answer is what a person chose.  ``help`` is absent for the same
#: reason as ``value`` plus one more -- the catalogue's own prose is long, and
#: § 12 says in as many words that its example abridges it.
_ILLUSTRATED = ("kind", "type", "anchor", "engine_key", "expands",
                "category", "engines", "resolver", "group", "unit",
                "range", "optional", "tier", "pattern", "default")


def _catalogue_items():
    import tomllib
    return tomllib.loads(template.load_catalogue())["item"]


def test_every_documented_item_matches_the_catalogue():
    """An ``[item.x]`` in a contract either IS the catalogue's ``x`` or is
    flagged as not built.

    Two failures are possible and both matter:

    * the document names an item the catalogue does not carry -- a reader
      copies it, and their template is refused by its own reader;
    * the document shows a real item with a key the catalogue spells
      differently -- which is how § 4.2 taught that ``species_order`` is a
      ``deck`` item for three days, contradicting the catalogue and § 6's own
      definition at once.

    **An unbuilt item is allowed, and must say so on the same line.** § 9.2
    illustrates ``user_custom`` deliberately, because that section is where
    the missing schema field is specified.
    """
    cat = _catalogue_items()
    docs_with_examples = ("engines/template.md", "engines/stages.md",
                          "execution/job-contracts.md")
    problems = []
    for doc_rel in docs_with_examples:
        text = (DOCS / doc_rel).read_text(encoding="utf-8")
        for i, body in _fenced_toml(doc_rel):
            import tomllib
            try:
                parsed = tomllib.loads(body)
            except Exception:
                continue                       # Gate A owns parse errors
            for name, shown in (parsed.get("item") or {}).items():
                real = cat.get(name)
                if real is None:
                    # Permitted only where the block itself says it is unbuilt.
                    if re.search(r"not built|NOT BUILT|does not exist|"
                                 r"needs a schema field|§ ?12\.1", body):
                        continue
                    problems.append(
                        f"{doc_rel} block {i}: [item.{name}] is in no "
                        f"catalogue. Either it was renamed, or the example is "
                        f"aspirational -- say so in the block and it passes.")
                    continue
                for key in _ILLUSTRATED:
                    if key not in shown:
                        continue               # showing a subset is fine
                    want, got = real.get(key), shown[key]
                    if isinstance(want, list) and isinstance(got, list):
                        same = list(want) == list(got)
                    else:
                        same = want == got
                    if not same:
                        problems.append(
                            f"{doc_rel} block {i}: [item.{name}] shows "
                            f"{key}={got!r}, the catalogue says {want!r}")
    assert not problems, (
        "a worked example disagrees with the catalogue it illustrates:\n  "
        + "\n  ".join(problems)
        + "\n(engines/template.md § 4.2 -- an example is the shape people "
          "copy, so a wrong one is a wrong classification taught.)"
    )


# --------------------------------------------------------------------- #
#  Gate C -- a number stated in prose is a claim about the code          #
#                                                                        #
#  The vocabulary gate above checks MEMBERSHIP and Gate A checks EXAMPLES.#
#  Neither can see a COUNT, and counts are what rotted: "84 items" when   #
#  there were 92, "307 facts" when there were 452, "PySCF declares 3      #
#  stage items" when it declares 11, "of 17 optional items only 12".      #
#  Each was true when written.  A count is worth stating when it is the   #
#  ARGUMENT -- "the duplication is measured rather than tolerated" means  #
#  nothing without the measurement -- and worth deleting when it is       #
#  decoration, which is why the diagram labels that carried item counts   #
#  no longer do.                                                          #
# --------------------------------------------------------------------- #

def _optional_items():
    cat = _catalogue_items()
    opt = [n for n, i in cat.items() if i.get("optional")]
    return len(opt), sum(1 for n in opt if cat[n].get("null_label"))


def _mirrored_fact_count():
    """The size of the debt `template.md` § 2.1a measures."""
    from tests.test_catalogue_agreement import MIRRORED, RENAMED
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.config.pyscf import PySCFConfig
    n = 0
    for cls in (SiestaConfig, PySCFConfig):
        for f in dataclasses.fields(cls):
            for key in MIRRORED + tuple(RENAMED):
                if f.metadata.get(RENAMED.get(key, key)) not in (None, "", (), []):
                    n += 1
    return n


def _stage_group(engine):
    return sum(1 for i in _catalogue_items().values()
               if i.get("group") == "stage" and engine in (i.get("engines") or ()))


def _exclusive_rows(engine):
    return sum(1 for i in _catalogue_items().values()
               if tuple(i.get("engines") or ()) == (engine,))


#: ``id: (document, the regex that must match, what it should say)``.
#:
#: The regex carries the number, so a stale document fails with the sentence
#: quoted rather than with a bare integer -- the failure has to be findable in
#: prose, which is the whole reason these live in prose.
MEASURED = {
    "mirrored facts (template.md § 2.1a)":
        ("engines/template.md", r"\*\*(\d+) facts live in two places\*\*",
         _mirrored_fact_count),
    "optional items (template.md § 5)":
        ("engines/template.md", r"of (\d+) optional items only \d+ carry one",
         lambda: _optional_items()[0]),
    "optional items carrying null_label (template.md § 5)":
        ("engines/template.md", r"of \d+ optional items only (\d+) carry one",
         lambda: _optional_items()[1]),
    "optional items (form-schema.md § 1.2)":
        ("web/form-schema.md", r"of \*\*(\d+)\*\* optional items",
         lambda: _optional_items()[0]),
    # Both engines' counts come out of ONE row, so neither pattern carries the
    # other's number.  The first draft spelled SIESTA's 44 inside the PySCF
    # pattern as an anchor, which made adding a SIESTA row break the PySCF
    # claim -- a test whose failure names the wrong document.
    "SIESTA exclusive catalogue rows (generator.md § 7.2)":
        ("execution/generator.md",
         r"catalogue rows \| (\d+) items \| \*\*\d+ items\*\*",
         lambda: _exclusive_rows("siesta")),
    "PySCF exclusive catalogue rows (generator.md § 7.2)":
        ("execution/generator.md",
         r"catalogue rows \| \d+ items \| \*\*(\d+) items\*\*",
         lambda: _exclusive_rows("pyscf")),
    "SIESTA stage-group items (stages.md § 1.1a)":
        ("engines/stages.md", r"\*\*It is (\d+) and \d+ now\*\*",
         lambda: _stage_group("siesta")),
    "PySCF stage-group items (stages.md § 1.1a)":
        ("engines/stages.md", r"\*\*It is \d+ and (\d+) now\*\*",
         lambda: _stage_group("pyscf")),
}


@pytest.mark.parametrize("claim_id", sorted(MEASURED))
def test_a_number_stated_in_prose_is_the_number_the_code_has(claim_id):
    """Every count a contract states about the code is measured on every run.

    A count is a claim, and it decays in one direction only -- the code grows
    and the sentence does not.  Four of these were wrong simultaneously on
    2026-08-17, one of them by 145, and each had been correct when written.

    **Adding a row here is the point**, exactly as it is for ``VOCABULARIES``:
    when a document states a new measurement, register it, or accept that it
    is true only on the day it is typed.
    """
    doc_rel, pattern, measure = MEASURED[claim_id]
    text = (DOCS / doc_rel).read_text(encoding="utf-8")
    m = re.search(pattern, text)
    assert m, (
        f"{doc_rel} no longer states {claim_id} in the shape this test reads "
        f"(/{pattern}/).  Either the sentence was reworded -- update the "
        f"pattern -- or the claim was dropped, and then this row should be "
        f"dropped with it.")
    stated, actual = int(m.group(1)), measure()
    assert stated == actual, (
        f"{doc_rel} says {stated} for {claim_id}; the code has {actual}.\n"
        f"  the sentence: {m.group(0)!r}\n"
        f"A count is an argument only while it is true (engines/template.md "
        f"§ 2.1a).")


# --------------------------------------------------------------------- #
#  A TIER TABLE in prose is the tier table the code ships                #
# --------------------------------------------------------------------- #

#: ``2.0×10⁻³`` is how a tier table writes a number, and ``2.0e-3`` is how the
#: code does.  Translating the superscripts is the whole difference.
_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


def _doc_number(cell: str):
    """The first number in a table cell, however the prose dresses it.

    Bold markers and a trailing gloss (``**1×10⁻⁹** (the `medium` rung)``) are
    editorial; the number is the claim.
    """
    m = re.search(r"[\d.]+(?:×10[⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)?", cell)
    if not m:
        return None
    return float(m.group(0).replace("×10", "e").translate(_SUPERSCRIPTS))


def _doc_section(doc_rel: str, heading: str) -> str:
    """One section's body: from its heading to the next heading of any level.

    Tier tables reuse the tier NAMES as row labels -- "loose preopt" heads a
    row in § 2.2 and another in § 2.5 -- so a document-wide search finds
    whichever comes first and silently checks the wrong table.  A claim names
    the section it is made in.
    """
    text = (DOCS / doc_rel).read_text(encoding="utf-8")
    m = re.search(r"^" + re.escape(heading) + r"\b.*?(?=^#|\Z)",
                  text, re.M | re.S)
    assert m, f"{doc_rel} has no section {heading!r}"
    return m.group(0)


def _doc_tier_row(doc_rel: str, heading: str, label: str):
    """The numbers of the markdown table row whose first cell is *label*.

    One reader for every tier table, so registering a new one is a row in
    ``TIER_TABLES`` rather than another hand-rolled parse -- the same shape
    ``MEASURED`` gives single numbers.
    """
    body = _doc_section(doc_rel, heading)
    m = re.search(r"^\|\s*`?" + re.escape(label) + r"`?\s*\|(.+)$",
                  body, re.M)
    assert m, (
        f"{doc_rel} {heading} has no tier-table row for {label!r}.  Either "
        f"the table was reshaped -- update this row -- or the claim was "
        f"dropped, and then the row should be dropped with it.")
    return [_doc_number(c) for c in m.group(1).split("|")]


def _pyscf_presets():
    from molbuilder.config.pyscf import PYSCF_STAGE_PRESETS
    return PYSCF_STAGE_PRESETS


#: ``id: (document, section, {item: (row label, {tier: column})}, code table)``.
#:
#: A tier table is a claim of the same kind ``MEASURED`` holds -- a number
#: stated in prose that the code also states -- only there are fifteen of them
#: and they are the ones that decide what a calculation converges to.  They are
#: registered rather than copied because copying is what put three wrong values
#: into the tight rung on 2026-08-18: they were read off the code they were
#: supposed to be checking.
TIER_TABLES = {
    "PySCF geomeTRIC criteria (tuning.md § 2.4)": (
        "engines/tuning.md", "### 2.4",
        # One row per item, the three tier columns read across it.
        {"geom_gmax": [("geom_gmax", {1: 0, 2: 1, 3: 2})],
         "geom_grms": [("geom_grms", {1: 0, 2: 1, 3: 2})],
         "geom_dmax": [("geom_dmax", {1: 0, 2: 1, 3: 2})],
         "geom_drms": [("geom_drms", {1: 0, 2: 1, 3: 2})],
         "geom_etol": [("geom_etol", {1: 0, 2: 1, 3: 2})]},
        _pyscf_presets),
    "PySCF SCF tolerance (tuning.md § 2.5)": (
        "engines/tuning.md", "### 2.5",
        # § 2.5 is transposed: one row per TIER, and the PySCF value is the
        # second cell of it.  Same reader, three rows for one item.
        {"scf_conv_tol": [("loose preopt", {1: 1}),
                          ("publishable", {2: 1}),
                          ("tight", {3: 1})]},
        _pyscf_presets),
}


@pytest.mark.parametrize("claim_id", sorted(TIER_TABLES))
def test_a_tier_table_stated_in_prose_is_the_table_the_code_ships(claim_id):
    """Every per-tier number a contract tabulates is measured on every run.

    `tuning.md` § 2.4 says outright that it is the authority for these values,
    which is only true if something checks.  Nothing did, and on 2026-08-18
    three of the tight rung's five criteria were wrong in the code -- copied
    from the implementation being replaced rather than read off the table.
    No test of the emitter could have caught it: every one of them asserts
    that the config's value reached the deck, and it did.
    """
    doc_rel, heading, rows, code_table = TIER_TABLES[claim_id]
    table = code_table()
    for item, places in rows.items():
        for label, columns in places:
            cells = _doc_tier_row(doc_rel, heading, label)
            for tier, col in columns.items():
                stated = cells[col]
                actual = table[tier][item]
                assert stated is not None, (
                    f"{doc_rel} {heading}: row {label!r} column {col} "
                    f"carries no number")
                assert stated == pytest.approx(actual, rel=1e-12), (
                    f"{doc_rel} {heading} says {item} = {stated!r} for tier "
                    f"{tier}; the shipped ladder uses {actual!r}.\n"
                    f"The DOCUMENT is the authority -- fix the code, unless "
                    f"the science changed, in which case fix the table "
                    f"first and the code from it.")


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


# --------------------------------------------------------------------- #
#  The template file has ONE door                                        #
# --------------------------------------------------------------------- #

def test_the_template_path_is_formed_in_exactly_one_place():
    """`engines/template.md` § 4.3 / `job-contracts.md` § 6.3 name the file
    ``<label>.template.toml``.  SEVEN call sites formed it independently until
    2026-08-17, in two INCOMPATIBLE ways -- from ``task.json``'s label, and by
    ``sorted(glob("*.template.toml"))[0]`` -- so a folder holding two templates
    had the web tab and ``prep`` reading different files.

    Walks the AST rather than the lines, so **prose is not code**: a docstring
    naming the file is how a contract is written, and the first version of
    this guard flagged four of them while missing the two sites that mattered
    (`build.py`'s ``template_name`` and `identity.py`'s pattern list) because
    it only knew the ``SUFFIX``-join spelling and not the literal one.

    What it bans, in executable code only: globbing for a template, joining
    ``SUFFIX`` by hand, and spelling ``.template.toml`` in a string.  Use
    :func:`template.template_filename`, :func:`template.template_path` or
    :func:`template.find_template` -- the last REFUSES an ambiguous folder
    rather than picking the alphabetical winner.
    """
    import ast
    import pathlib as _pl
    root = _pl.Path(__file__).resolve().parents[1] / "molbuilder"
    offenders = []
    for p in sorted(root.rglob("*.py")):
        if p.name == "template.py":
            continue                       # the door's own home
        if p.name == "identity.py":
            # THE ONE EXEMPTION, and it is a layering fact, not a lapse.
            # `identity.py` is L1 and `template` is L2, so it may not import
            # the suffix -- `tests/test_layering.py` fails if it does.  Its
            # pattern list therefore spells the name, and that single line is
            # the only place outside the door allowed to.
            continue
        tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
                body = getattr(node, "body", None)
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    docstrings.add(id(body[0].value))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if id(node) in docstrings:
                continue                   # prose, not code
            s = node.value
            if "catalogue.template.toml" in s:
                continue                   # the shipped master, a different file
            # PROSE IS NOT A PATH.  `task.1st.json`'s ``_what`` line explains
            # itself to a reader in a sentence that names the template beside
            # it -- and that is the file doing its job, not a path being
            # formed.  A path is short; a sentence is not.
            if len(s) > 40:
                continue
            where = f"{p.relative_to(root.parent)}:{node.lineno}"
            if "*.template.toml" in s:
                offenders.append(f"{where} globs for a template")
            elif ".template.toml" in s:
                offenders.append(f"{where} spells the suffix literally")
        # the f-string join, which is not one Constant
        for node in ast.walk(tree):
            if isinstance(node, ast.JoinedStr):
                names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
                if {"SUFFIX", "TEMPLATE_SUFFIX"} & names:
                    offenders.append(
                        f"{p.relative_to(root.parent)}:{node.lineno} "
                        f"joins the suffix by hand")
    assert not offenders, (
        "the template path is formed outside its one door:\n  "
        + "\n  ".join(offenders)
        + "\nUse template.template_filename / template_path / find_template."
    )
