"""`parse/fdf.py` — the one reader of SIESTA's input format.

**Why this file exists.** Until 2026-09-17 there were eight readers of deck
content across the tree and only one implemented fdf's actual keyword rule.
The others were hand-rolled regexes and awk that each got part of it, and two
of them disagreed about the same file.  The survivors all ask this module now,
so its rules are worth stating once, here.

The two cases below came FROM the wrapper's tests: they pinned an awk that read
`SystemLabel` out of a deck at launch.  That awk is deleted — the wrapper is
told its label (`gpu.md` G7) — but the behaviours it was pinned for are real
and still needed, because `web/blueprints/watch.py` reads a directory a person
points at, where there is no description to ask and the deck is all there is.
So the coverage moved rather than went.
"""
from __future__ import annotations

import pytest

from molbuilder.parse.fdf import _norm, _parse_fdf, system_label


class TestTheKeywordRuleIsTheFormatS:
    """fdf matches a keyword ignoring case AND `.`, `-`, `_`.

    All four spellings below are one keyword and SIESTA accepts each.  A regex
    anchored on the literal word matches the first and last only, which is what
    every hand-rolled reader did — `web/blueprints/watch.py` resolved a deck
    spelled `System.Label` to nothing, so the Results tab found no trajectory.
    """

    @pytest.mark.parametrize("spelling", [
        "SystemLabel", "systemlabel", "SYSTEMLABEL",
        "System.Label", "system_label", "System-Label",
    ])
    def test_every_legal_spelling_is_the_same_keyword(self, spelling):
        assert system_label(f"{spelling}  bdt\n") == "bdt"

    def test_norm_is_that_rule_and_nothing_else(self):
        assert _norm("System.Label") == _norm("system_label") == "systemlabel"


class TestAQuotedValueReadsAsSIESTAReadsIt:
    """`SystemLabel "foo"` is legal fdf and SIESTA then writes `foo.DM`.

    A reader that keeps the quotes looks for files that do not exist.  Measured
    when this landed: the wrapper's cold sweep missed the warm files entirely
    and would have overwritten them without a word.
    """

    @pytest.mark.parametrize("line,expect", [
        ('SystemLabel "foo"', "foo"),
        ("SystemLabel 'foo'", "foo"),
        ("SystemLabel foo", "foo"),
    ])
    def test_a_quote_pair_is_stripped(self, line, expect):
        assert system_label(line + "\n") == expect

    def test_an_unmatched_quote_is_not_stripped(self):
        """Only a PAIR is syntax.  A lone quote is a character in the value —
        and one the basename validator would have refused, so leaving it means
        the caller's own charset check sees it and falls back."""
        assert system_label('SystemLabel "foo\n') == '"foo'


class TestTheAbsentCases:
    def test_a_deck_that_states_none_answers_none(self):
        assert system_label("MeshCutoff 300 Ry\n") is None

    def test_an_empty_value_is_not_a_label(self):
        """`None`, not `""` — its sibling `pyscf.input.job_name` answers the
        same question the same way, so a caller cannot need two idioms."""
        assert system_label('SystemLabel ""\n') is None


class TestRepeatedKeywords:
    def test_the_first_wins_because_libfdf_takes_the_first(self):
        """`siesta/layout.py::check_rules` states the rule and cites the
        lookup: *"libfdf takes the FIRST match and ignores the rest
        (`fdf_locate` walks from the top and stops)"*.

        molbuilder never emits one — the deck gate refuses a keyword written
        twice with different values — so this only arises in a deck someone
        hand-edited or another tool wrote, which is exactly what this reader is
        pointed at.
        """
        assert _parse_fdf("MeshCutoff 300 Ry\nMeshCutoff 500 Ry\n")[0][
            "meshcutoff"] == ["300", "Ry"]

    def test_a_comment_is_not_a_value(self):
        assert system_label("SystemLabel bdt  # the junction\n") == "bdt"
