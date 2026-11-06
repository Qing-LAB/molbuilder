"""Which rule a line reaches -- the line-rule engine both reading passes run on
(`model/parse.md` § 4a, `parse/engines/_section_rules.py`).

Asked through `compile_rules(...).find_match`, the path a run takes: the regex
each matcher compiles to, and the plain predicates two of SIESTA's rules are.
The lines are neutral strings given to rules each case makes; what a reading
pass makes of an engine's real output is asked of runs made on the road.
"""
from __future__ import annotations

import pytest

from molbuilder.parse.engines._section_rules import (
    SectionRule, any_of, compile_rules, contains_ci, matches_regex_ci,
    starts_with_ci,
)


@pytest.mark.parametrize("start,line,reached", [
    (lambda: starts_with_ci("abc:"), "ABC: x", True),
    (lambda: starts_with_ci("abc:"), "  \tabc: x", True),
    (lambda: starts_with_ci("abc:"), "# abc: x", False),
    (lambda: starts_with_ci("a (d):"), "a (d): x", True),
    (lambda: contains_ci("abc"), "x ABC y", True),
    (lambda: contains_ci("a (d)"), "x a (d) y", True),
    (lambda: matches_regex_ci(r"^\s*abc\s+\S"), "  ABC x", True),
    (lambda: any_of(matches_regex_ci("aaa"), matches_regex_ci("bbb")), "aaa", True),
    (lambda: any_of(matches_regex_ci("aaa"), matches_regex_ci("bbb")), "bbb", True),
], ids=["prefix, another case", "prefix, indented", "prefix, not at the start",
        "prefix, its metacharacters are text", "substring, another case",
        "substring, its metacharacters are text", "regex, another case",
        "either of two, the first", "either of two, the second"])
def test_a_line_reaches_its_rule(start, line, reached):
    """THE SILENT FAILURE: a section the file holds never opened -- a build
    that capitalises a header, an indented one, a header holding a bracket, a
    rule's second spelling -- or opened by a line that only mentions it; the
    numbers it holds are then not read, and nothing says so."""
    rule = SectionRule(name="r", start=start())
    assert (compile_rules([rule]).find_match(line) is rule) is reached


def test_the_earlier_rule_wins_wherever_each_matches_on_the_line():
    """The order in the table decides, never the place on the line: a fatal
    form registered before a soft one must take a line holding both
    (`siesta_reader._rule_table`).  A line nothing owns reaches no rule."""
    first = SectionRule(name="first", start=contains_ci("bbb"))
    second = SectionRule(name="second", start=contains_ci("aaa"))
    assert compile_rules([first, second]).find_match("aaa bbb") is first
    assert compile_rules([second, first]).find_match("aaa bbb") is second
    assert compile_rules([first, second]).find_match("ccc") is None


def _is(want):
    return lambda line: line == want


@pytest.mark.parametrize("rules,line,winner", [
    ([("p", _is("x foo")), ("r", contains_ci("foo"))], "x foo", "p"),
    ([("r", contains_ci("foo")), ("p", _is("x foo"))], "x foo", "r"),
    ([("r", contains_ci("zzz")), ("p", _is("x foo"))], "x foo", "p"),
    ([("r", contains_ci("zzz")), ("p", _is("nope"))], "x foo", None),
], ids=["a-predicate-first", "a-regex-first",
        "no-regex-fires-the-predicate-still-runs", "nothing"])
def test_a_rule_that_is_a_predicate_takes_its_turn(rules, line, winner):
    """A plain predicate takes its turn in the table's order, and runs though
    no regex fires on the line.  THE SILENT FAILURE: the pre-filter skipping
    it -- SIESTA's `max_force` rules are predicates, so a step's largest force
    would not be read."""
    table = compile_rules([SectionRule(name=n, start=s) for n, s in rules])
    got = table.find_match(line)
    assert (got.name if got else None) == winner
