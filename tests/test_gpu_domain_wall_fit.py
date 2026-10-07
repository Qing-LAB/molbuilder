"""Routing a GPU group to a domain whose ceiling can actually hold it.

The failure this closes (ASU Sol, 2026-08-23)
=============================================

    sbatch: error: QOSMaxWallDurationPerJobLimit
    sbatch: error: Batch job submission failed: Job violates accounting/QOS
    policy

`jobset launch bench coarse --mode submit` on Sol.  The chain:

  * the probe records domains **cheapest ceiling first**, so Sol's first
    gpu-capable row is ``htc/debug`` at 00:15:00;
  * ``gpu_domain_row`` returned that first row without regard to duration;
  * ``_preferred_domain`` then asked whether it fits, got "no", and
    returned ``None``;
  * ``None`` meant "the rendered header's directives stand" -- and the
    header said ``-p htc -q debug``, the very row just rejected as too
    small, because prep had chosen it by the same wall-blind rule;
  * the command line still carried ``-t`` for the whole group.

So a group needing tens of minutes was submitted into a fifteen-minute
ceiling, while ``htc/public`` (4h) and ``general`` (14d) sat further down
the same menu, both gpu-capable and both big enough.  The CPU branch had
always walked the menu for a row that fits; the GPU branch only ever
looked at row 0.

A job NAMES its queue (`architecture.md` § 5.2), and the named queue is
admitted on its row or refused with every reason
(`TestNamingADomainDoesNotSkipTheCheck` below).  What is here is that check -- what "fits" means, cores, memory, the GPU
column -- read against Sol's menu, verbatim from the ``environment.json``
copied back off the cluster, so a regression shows against the real rows
rather than only a simplified stand-in.
"""
from __future__ import annotations

import pytest

from molbuilder.scheduler import Domain, Request, admits
from molbuilder.scheduler.place import Unplaceable, place


def _row_holds(domain, needed_s):
    return not admits(domain, Request(walltime_s=needed_s))


#: ASU Sol, probed 2026-08-23 -- name, partition, qos, ceiling, gpu.
#:
#: Parsed through `Domain.from_row`, the same parser the real record goes
#: through, rather than hand-built objects: a fixture that skips the parser
#: cannot catch a column the parser drops.
_SOL_ROWS = [
    {"name": "debug",     "partition": "htc",       "qos": "debug",
     "max_time": "00:15:00",   "gpu": {"a100": 4}},
    {"name": "htc",       "partition": "htc",       "qos": "public",
     "max_time": "4:00:00",    "gpu": {"a100": 4}},
    {"name": "lightwork", "partition": "lightwork", "qos": "public",
     "max_time": "1-00:00:00", "gpu": {"a100.20gb": 16}},
    {"name": "public",    "partition": "public",    "qos": "public",
     "max_time": "7-00:00:00", "gpu": {"a100": 4}},
    {"name": "highmem",   "partition": "highmem",   "qos": "public",
     "max_time": "7-00:00:00", "gpu": None},
    {"name": "general",   "partition": "general",   "qos": "public",
     "max_time": "14-00:00:00", "gpu": {"a100": 4}},
]
SOL = [Domain.from_row(r) for r in _SOL_ROWS]
assert all(d is not None for d in SOL), "a Sol row failed to parse"


def _dom(**kw):
    """A one-off domain for the edge cases -- name/partition/qos are
    required by the parser, so state them once here."""
    base = {"name": kw.pop("name", "x"), "partition": "p", "qos": "q"}
    return Domain.from_row({**base, **kw})

_FIFTEEN_MIN = 15 * 60


class TestNoMenuPromisesNothing:

    def test_no_menu_and_no_name_is_not_a_refusal(self):
        """R6: a machine that promised nothing -- no queues, and none named
        -- gets no placement, rather than a refusal it cannot act on."""
        assert place([], Request(walltime_s=10 ** 6), prefer_gpu=True,
                     named=None) is None


class TestWhatFittingMeans:
    """One reader for both sides, so they cannot disagree."""

    def test_an_unstated_ceiling_never_bars(self):
        assert _row_holds(_dom(), 10 ** 9) is True
        assert _row_holds(_dom(max_time=None), 10 ** 9) is True

    def test_an_unreadable_ceiling_never_bars(self):
        assert _row_holds(_dom(max_time="whenever"), 10 ** 9) is True

    def test_exactly_equal_fits(self):
        """A 15-minute ceiling holds a 15-minute job; the boundary is not
        an off-by-one that silently drops the cheapest row."""
        assert _row_holds(_dom(max_time="00:15:00"), _FIFTEEN_MIN) is True
        assert _row_holds(_dom(max_time="00:15:00"), _FIFTEEN_MIN + 1) is False

    def test_the_day_form_is_understood(self):
        assert _row_holds(_dom(max_time="1-00:00:00"), 23 * 3600) is True
        assert _row_holds(_dom(max_time="1-00:00:00"), 25 * 3600) is False



def _why(domain, request):
    """`admits`, as `(limit, asked, allowed)` triples.

    A refusal carries its numbers as FIELDS, so nothing here reads a
    sentence: the message is rendered from these three and asserting it
    would be asserting the rendering.
    """
    from molbuilder.scheduler import admits
    return [(i.limit, i.asked, i.allowed) for i in admits(domain, request)]


class TestMemoryCanFinallyBeCompared:
    """`max_mem_gb` was declared, serialised, round-tripped -- and read by no
    code at all, for one boring reason: the record states gigabytes as a
    number and a job states memory as SLURM text, and nothing converted
    between them.  A limit that cannot be expressed in the same unit as the
    ask is a limit that will never be checked (contract R2).
    """

    @pytest.mark.parametrize("text,gb", [
        ("390G", 390.0),
        ("512M", 0.5),
        ("1T", 1024.0),
        ("2048", 2.0),        # bare number is megabytes, SLURM's default
        ("", None),
        (None, None),
        ("nonsense", None),   # unreadable is not small (R3)
    ])
    def test_slurm_memory_text_becomes_gigabytes(self, text, gb):
        from molbuilder.scheduler import parse_mem_gb
        assert parse_mem_gb(text) == gb

    def test_mem_zero_means_all_of_it_not_none_of_it(self):
        """`--mem=0` is SLURM for *all the memory on the node*.  Reading it as
        a request for zero would make every domain admit it."""
        from molbuilder.scheduler import parse_mem_gb
        assert parse_mem_gb("0") is None

    def test_a_request_too_big_for_the_node_is_now_refused(self):
        from molbuilder.scheduler import Request, admits
        d = _dom(name="small", max_mem_gb=256.0)
        assert _why(d, Request(mem_gb=390.0)) == [
            ("mem", "390 GB", "256 GB")]
        assert admits(d, Request(mem_gb=128.0)) == []


class TestTheRequestStatesCoresOnce:

    def test_cores_are_ranks_times_cpus_per_task(self):
        """The number `max_cores` is stated against.  prep's per-family cap
        computed `g * k * c` for itself; stating it on the request is what
        stops the two disagreeing about what "cores" means."""
        from molbuilder.scheduler import Request
        assert Request(ranks=64, cpus_per_task=1).cores == 64
        assert Request(ranks=16, cpus_per_task=4).cores == 64
        assert Request(ranks=8).cores == 8           # unstated cpus = 1
        assert Request().cores is None               # unasked stays unasked

    def test_every_limit_is_reported_at_once(self):
        """A refusal lists ALL the reasons, not the first -- a user who fixes
        the wall only to meet the core cap has been sent round twice."""
        from molbuilder.scheduler import Request, admits
        d = _dom(name="debug", max_time="00:15:00", max_cores=48,
                 max_mem_gb=256.0, gpu={"a100": 4})
        why = admits(d, Request(ranks=64, cpus_per_task=1, gpus=2,
                                mem_gb=390.0, walltime_s=2280))
        # WHICH limits refused, named: a refusal is an `Issue` carrying
        # `where`, so the claim is exact and says itself.
        assert {i.limit for i in why} == {"walltime", "cores", "mem"}


class TestNamingADomainDoesNotSkipTheCheck:
    """Contract § 5: `--domain` reaches the same admission test.  Your choice
    is honoured as a CHOICE, not as permission to skip verification -- until
    phase 4 it bypassed admission entirely."""

    def test_a_named_domain_too_small_is_refused(self):
        with pytest.raises(Unplaceable) as exc:
            place(SOL, Request(walltime_s=38 * 60), prefer_gpu=True,
                  named="debug")
        assert exc.value.reasons[0].allowed == "00:15:00"

    def test_a_named_domain_that_fits_is_used_even_if_not_cheapest(self):
        got = place(SOL, Request(walltime_s=600), prefer_gpu=True,
                    named="general")
        assert got.name == "general"      # not `debug`, the cheapest that fits

    def test_an_unknown_name_says_what_there_is(self):
        with pytest.raises(Unplaceable) as exc:
            place(SOL, Request(), prefer_gpu=True, named="nope")
        assert "debug" in exc.value.reasons[0].note
        assert "htc" in exc.value.reasons[0].note


class TestTheGpuColumn:
    """`Domain.gpu` maps a gres TYPE to its per-node COUNT, as the probe
    writes it (`scheduler.md` § 4, *Device*).
    """

    def test_an_unreadable_count_is_skipped_not_raised(self):
        """An unreadable value is not a small one (R3) -- and it must not
        take the whole submission down with a ValueError."""
        d = _dom(name="odd", gpu={"a100": "four"})
        assert admits(d, Request(gpus=2)) == []
