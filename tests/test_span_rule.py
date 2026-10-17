"""ONE LENGTH OF TIME, SAID ONE WAY -- `docs/web/ui-contract.md` § 4.

Each unit until one and a half of the next -- 90 s, 90 min, 36 h -- then
days, the nearest whole one, a half rounded up: written twice, for the page
(`lib/time-ago.js`'s ``timeSpan``) and for the monitor beside a job, where the
page's helper cannot reach (`report_fields.span_words`).  Each row is asked of
both.  The silent failure: a run's *90m* on the Run panel beside *2h* in the
message the monitor sent about it, one length said two ways.

API-LEVEL: both functions, asked the rows the contract's boundaries give.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from _node_esm import run_node
from molbuilder.report_fields import span_words

TIME_AGO = (Path(__file__).resolve().parents[1]
            / "molbuilder/web/static/lib/time-ago.js")

#: ``(seconds, words)`` -- each boundary of the rule and either side of it.
ROWS = [
    (0, "0s"), (12, "12s"), (89, "89s"),
    (90, "2m"), (300, "5m"), (5399, "90m"),
    (5400, "2h"), (21600, "6h"), (129599, "36h"),
    (129600, "2d"), (518400, "6d"),
]


@pytest.mark.parametrize("seconds, words", ROWS)
def test_the_monitor_says_a_length_of_time_by_the_rule(seconds, words):
    assert span_words(seconds) == words


def test_the_page_says_every_length_of_time_as_the_monitor_does():
    said = run_node([TIME_AGO], (
        "console.log(JSON.stringify(%s.map("
        "(s) => window.molbuilder.timeSpan(s))));")
        % [s for s, _w in ROWS])
    assert said == [w for _s, w in ROWS], list(zip(ROWS, said))
