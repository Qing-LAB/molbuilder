"""The fit -- how a request sits on a queue's machines (`docs/execution/
scheduler.md` § 3, § 4), case by case.

THE CASES ARE DATA -- ``tests/data/the_fit.toml`` -- each a request asked of
one queue of a record written out there, read back from the answer's
fields: the nodes that hold it, of how many, each limit it is over with its
numbers; the queue a job names, bound with its fit or refused for a
statement missing; what the queue table shows of a queue.

PREVENTS: a request refused, or called fitting, by one figure over a
mixture of machines -- `--mem 900G` refused on Sol's `htc` because the
queue's memory was its smallest node's, though three of its nodes hold
1007 GB and more (measured on the record copied from Sol, 2026-10-17); a
GPU job counted against cores no GPU node has; a limit the record does not
state counted as a small one.  Reading the code does not show these: each
is a number the record and the request meet in, and only the answer shows
which machines it was compared with.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from molbuilder.jobset.placement import queue_table
from molbuilder.scheduler import Domain, Request, fit, parse_mem_gb
from molbuilder.scheduler.place import Unplaceable, place

TABLE = tomllib.loads(
    (Path(__file__).parent / "data" / "the_fit.toml").read_text())
QUEUES = {r["name"]: Domain.from_row(r) for r in TABLE["queues"]}


def _request(said) -> Request:
    return Request(**(said or {}))


def _over(fitted):
    """Each limit as ``(limit, asked, allowed)``, ``""`` where it has no
    numbers -- the fields, never the sentence."""
    return [(o.limit, "" if o.asked is None else str(o.asked),
             "" if o.allowed is None else str(o.allowed))
            for o in fitted.over]


@pytest.mark.parametrize("case", TABLE["case"],
                         ids=[c["name"] for c in TABLE["case"]])
def test_the_fit(case):
    got = fit(QUEUES[case["queue"]], _request(case.get("request")))
    assert (got.nodes, got.of) == (case.get("nodes"), case.get("of")), got
    assert _over(got) == [tuple(o) for o in case.get("over", [])], got
    for words in case.get("note", []):
        assert any(words in o.note for o in got.over), (words, got.over)
    if "holding" in case:
        assert list(got.holding) == case["holding"], got.holding
    for words in case.get("line_has", []):
        assert words in got.line, (words, got.line)


@pytest.mark.parametrize("case", TABLE["placed"],
                         ids=[c["name"] for c in TABLE["placed"]])
def test_the_placement(case):
    rows = [QUEUES[n] for n in case.get("queues", list(QUEUES))]
    ask = dict(named=case.get("named"))
    if "refused" in case:
        with pytest.raises(Unplaceable) as exc:
            place(rows, _request(case.get("request")), **ask)
        for words in case["refused"]:
            assert words in str(exc.value), str(exc.value)
        return
    got = place(rows, _request(case.get("request")), **ask)
    if case.get("none"):
        assert got is None, got
        return
    assert [got.partition, got.qos] == case["bound"], got
    assert _over(got.fit) == [tuple(o) for o in case.get("over", [])], got


@pytest.mark.parametrize("case", TABLE["shown"],
                         ids=[c["name"] for c in TABLE["shown"]])
def test_what_a_queue_shows(case):
    d = QUEUES[case["queue"]]
    assert (d.core_range, d.mem_range) == (case["cores"], case["memory"])
    table = queue_table([d], Request()).splitlines()
    machines = [ln.strip() for ln in table if ln.lstrip().startswith("- ")]
    assert len(machines) == case["machines"], table
    for words in case.get("machine_line", []):
        assert any(words in ln for ln in machines), (words, machines)


@pytest.mark.parametrize("row", TABLE["memory_text"])
def test_memory_text_is_read_in_the_records_unit(row):
    got = [parse_mem_gb(t) for t in row["text"]]
    assert got == (row["gb"] or [None] * len(row["text"])), got
