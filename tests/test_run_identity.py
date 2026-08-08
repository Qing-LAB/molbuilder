"""P3 units 1-2 — the id is built from inputs, and normalised once.

Contract: ``docs/execution/run-identity.md`` § 2 (*built from inputs, never
from anything a run produced*), § 2.1 (what is in the pin and the longer list
of what is not), § 3 (the set, the cap, the three rules) and § 3.1 (the worked
table).

**§ 3.1 is parsed out of the document, not retyped**, which is the plan's rule
for a contract that carries a worked example: the table and the test cannot
drift, and adding a row to the document without implementing it fails here.
``tests/test_task_description.py`` does the same against ``stages.md § 6``;
this is that pattern, not a second one.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from molbuilder.checkpoint import _REF_SAFE_ID
from molbuilder.identity import MAX_ID_BYTES, normalise_id, run_id
from molbuilder.projects import _NAME_PATTERN


REPO = Path(__file__).resolve().parents[1]
CONTRACT = REPO / "docs" / "execution" / "run-identity.md"


# --------------------------------------------------------------------- #
#  The fixture IS the contract — § 3.1's table, parsed                  #
# --------------------------------------------------------------------- #

_ROW = re.compile(r"^\|\s*(?P<given>.+?)\s*\|\s*(?P<want>.+?)\s*\|.*\|\s*$")


def _worked_rows():
    """§ 3.1's rows as ``(inputs, expected-or-None-if-refused, raw)``."""
    text = CONTRACT.read_text(encoding="utf-8")
    start = text.index("### 3.1 Normalisation, worked")
    end = text.index("### 3.2", start)
    rows = []
    for line in text[start:end].splitlines():
        m = _ROW.match(line)
        if not m:
            continue
        given, want = m.group("given"), m.group("want")
        if given.startswith("What the user") or set(given) <= set("-: "):
            continue                                  # header / separator
        inputs = re.findall(r"`([^`]*)`", given)
        if not inputs:
            # The prose row ("a 300-character name") has nothing to quote;
            # it is asserted directly below instead.
            continue
        want_code = re.findall(r"`([^`]*)`", want)
        rows.append((tuple(inputs),
                     None if "refused" in want else want_code[0],
                     given))
    return rows


WORKED = _worked_rows()


def test_the_table_was_found_and_is_the_size_it_was():
    """A parsed fixture that finds nothing is a green test checking nothing.

    Seven of § 3.1's eight rows quote their input; the eighth is prose ("a
    300-character name") and is asserted on its own below."""
    assert len(WORKED) == 7, [r[2] for r in WORKED]


@pytest.mark.parametrize("inputs,expected",
                         [(i, e) for i, e, _ in WORKED],
                         ids=[r[2][:30] for r in WORKED])
def test_every_worked_row_of_section_3_1(inputs, expected):
    """Each row of the document's own table, run against the code."""
    if expected is None:
        with pytest.raises(ValueError):
            run_id(*inputs)
    else:
        assert run_id(*inputs) == expected


def test_the_prose_row_a_300_character_name_is_refused():
    """§ 3.1's eighth row. Refused, never truncated — a shortened id is a
    different calculation wearing the same name."""
    with pytest.raises(ValueError, match="refused rather than truncated"):
        run_id("a" * 300)


# --------------------------------------------------------------------- #
#  Unit 1 — built from inputs, never from anything a run produced       #
# --------------------------------------------------------------------- #

def test_the_id_is_knowable_before_the_calculation_exists():
    """§ 2. The whole rule, stated as a property: the id is a function of the
    two things a person knows before anything has run, so asking for it twice
    — a year apart, on another machine — gives the same answer, and the warm
    files are found."""
    assert run_id("BDT relax", "C6H4S2") == run_id("BDT relax", "C6H4S2")


def test_a_different_molecule_is_a_different_id():
    """§ 2.1's one positive row: a `.XV` is a list of positions for *these*
    atoms, so what the coordinates are OF has to be in the pin."""
    assert run_id("relax", "H2") != run_id("relax", "H2O")


def test_the_label_alone_is_a_legitimate_id():
    """`formula` is optional — § 3.1's rows 2-5 all pass one input."""
    assert run_id("BDT-Au") == "BDT-Au"


def test_the_id_is_blind_to_everything_a_stage_tunes():
    """§ 2.1's fifth row, and the reason it matters: *that* is what makes
    several stages one calculation rather than several.

    Asserted against the signature rather than by perturbing a config,
    because the guarantee is structural — there is no parameter through which
    a mesh cutoff or a force tolerance could reach the id at all."""
    import inspect
    params = set(inspect.signature(run_id).parameters) - {"stage_names"}
    assert params == {"label", "formula"}, (
        "run_id grew an input. § 2.1 lists what may be in the pin and it is "
        "two things; anything else makes tuning a parameter throw away a "
        "geometry the user should have kept.")


def test_the_timestamp_is_not_part_of_it():
    """§ 2. `task.Run.created` records when a description was written;
    putting it in the id would make every regeneration a cold start."""
    from molbuilder.task import Run
    r = Run(name="BDT relax", id=run_id("BDT relax", "C6H4S2"),
            created="20260808T031500Z")
    assert r.created not in r.id
    assert "2026" not in r.id


# --------------------------------------------------------------------- #
#  Unit 2 — normalisation                                               #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("raw", [
    "BDT/Au relax", "BDT  --  Au", "_relax_", "Relax.v2", "BDT-Au",
    "a b c", "x" * 200,
])
def test_every_id_this_makes_is_accepted_by_every_shipped_validator(raw):
    """§ 3: the set *"is not a new decision"*.

    This is the reuse obligation, asserted the only way that actually holds:
    not by sharing a constant — ``projects`` is L2 and this module is L1, so
    it cannot import it — but by checking the output against both shipped
    patterns. If any of the three spellings drifts apart, this fails."""
    out = normalise_id(raw)
    assert _NAME_PATTERN.match(out), f"{out!r} fails projects._NAME_PATTERN"
    assert _REF_SAFE_ID.match(out), f"{out!r} fails checkpoint._REF_SAFE_ID"


def test_case_is_preserved_because_a_formula_says_the_element():
    """Decided 2026-08-08 (§ 8 #14). Lowercasing `Co` yields the token `CO`
    also lowercases to, which erases the one thing the formula is in the id
    to say."""
    assert run_id("magnet", "Co4") == "magnet_Co4"
    assert run_id("gas", "CO4") == "gas_CO4"
    assert run_id("magnet", "Co4") != run_id("gas", "CO4")


def test_a_lone_separator_survives_but_a_run_collapses():
    """§ 3. Collapsing every `-` to `_` would be simpler and would quietly
    rename a project — `BDT-Au` is a name someone chose."""
    assert normalise_id("BDT-Au") == "BDT-Au"
    assert normalise_id("BDT_Au") == "BDT_Au"
    assert normalise_id("BDT  --  Au") == "BDT_Au"
    assert normalise_id("a---b") == "a_b"


def test_normalising_an_id_again_changes_nothing():
    """§ 3 rule 1 says it happens once and the result is stored. Nothing
    downstream should re-derive it — but if something does, it must not get a
    second, different answer. Idempotence is what makes rule 1 a policy
    rather than a landmine."""
    for raw in ("BDT/Au relax", "BDT  --  Au", "_relax_", "Relax.v2"):
        once = normalise_id(raw)
        assert normalise_id(once) == once


def test_a_refusal_names_the_character_it_refused():
    """§ 3 rule 3: *"say so and ask"*. Saying so without saying which
    character is what makes a user delete letters at random until it passes."""
    with pytest.raises(ValueError, match="'Ü'"):
        run_id("Über")


@pytest.mark.parametrize("raw", ["Über", "Ω-shape", "café", "naïve"])
def test_a_letter_or_digit_may_not_be_silently_replaced(raw):
    """Decided 2026-08-08 (§ 8 #15). Substituting a separator is expected;
    substituting a character typed INSIDE a word loses it."""
    with pytest.raises(ValueError, match="letter or a digit"):
        normalise_id(raw)


@pytest.mark.parametrize("raw", ["BDT/Au relax", "a b", "x.y", "p,q", "r;s"])
def test_a_separator_is_replaced_silently(raw):
    """The other side of the same line — this is all `BDT/Au relax` does, and
    refusing it would make the rule useless."""
    assert normalise_id(raw)


def test_nothing_left_is_refused_not_patched_with_a_digit():
    """§ 3 rule 3. `bdt_2` explains nothing about what it differs from."""
    with pytest.raises(ValueError, match="entirely separators"):
        run_id("///")


# --------------------------------------------------------------------- #
#  The cap — derived, not chosen                                        #
# --------------------------------------------------------------------- #

def test_the_cap_is_derived_from_what_gets_appended_to_the_id():
    """§ 3: ``<id>_<longest stage name>.<longest extension>`` must fit the
    filesystem's limit. Asserted as arithmetic rather than as a magic number,
    so a longer extension in `job-contracts.md § 4.2` moves the cap."""
    assert MAX_ID_BYTES == 255 - 32 - len(".STRUCT_NEXT_ITER")
    assert normalise_id("x" * MAX_ID_BYTES)
    with pytest.raises(ValueError, match="refused rather than truncated"):
        normalise_id("x" * (MAX_ID_BYTES + 1))


def test_a_known_ladder_tightens_the_cap_instead_of_guessing():
    """The 32-character stage budget is a guess for the case where the id is
    named before its stages exist. Where they DO exist, the real longest name
    is used — and a long stage name makes the id's room smaller, which is the
    honest direction for the error to go."""
    long_stage = "a_very_long_stage_name_indeed_beyond_the_budget"
    at_default = "x" * MAX_ID_BYTES
    assert normalise_id(at_default)
    with pytest.raises(ValueError, match="the limit is"):
        normalise_id(at_default, stage_names=["coarse", long_stage])
