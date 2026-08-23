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

The menu below is Sol's, verbatim from the ``environment.json`` copied
back off the cluster, so this test fails if the real fix regresses rather
than only a simplified stand-in.
"""
from __future__ import annotations

import pytest

from molbuilder.jobset.submit import _row_holds, gpu_domain_row


#: ASU Sol, probed 2026-08-23 -- name, partition, qos, ceiling, gpu.
SOL = [
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

_FIFTEEN_MIN = 15 * 60


class TestTheSelectorSkipsACeilingItCannotUse:

    def test_a_short_group_still_takes_the_cheapest_row(self):
        """Cheapest-ceiling-first is unchanged when the job actually fits --
        a bounded single trial belongs in `debug`, which is the whole point
        of that ordering."""
        assert gpu_domain_row(SOL, needed_s=600)["name"] == "debug"

    def test_the_group_that_failed_on_sol_now_routes_to_a_row_that_holds_it(self):
        """Tens of minutes: `debug` is skipped, the next gpu-capable row
        that can hold it wins -- still the cheapest ceiling among those
        that fit, not the biggest."""
        row = gpu_domain_row(SOL, needed_s=38 * 60)
        assert row["name"] == "htc"
        assert (row["partition"], row["qos"]) == ("htc", "public")

    def test_a_very_long_group_walks_further_down(self):
        assert gpu_domain_row(SOL, needed_s=5 * 24 * 3600)["name"] == "public"

    def test_a_cpu_only_row_is_never_offered_to_the_gpu_side(self):
        """`highmem` has the ceiling but no devices."""
        for need in (600, 38 * 60, 5 * 24 * 3600):
            assert gpu_domain_row(SOL, needed_s=need)["gpu"] is not None

    def test_asking_nothing_about_duration_keeps_the_old_answer(self):
        """prep's device inventory and per-family core cap ask about
        CAPABILITY, not duration -- their answer must not move."""
        assert gpu_domain_row(SOL)["name"] == "debug"
        assert gpu_domain_row(SOL, needed_s=None)["name"] == "debug"

    def test_nothing_fits_is_reported_as_nothing(self):
        """Not "fall back to row 0" -- that is what submitted the doomed
        job.  The caller refuses on None."""
        tiny = [{"name": "debug", "partition": "htc", "qos": "debug",
                 "max_time": "00:15:00", "gpu": {"a100": 1}}]
        assert gpu_domain_row(tiny, needed_s=_FIFTEEN_MIN + 1) is None


class TestWhatFittingMeans:
    """One reader for both sides, so they cannot disagree."""

    def test_an_unstated_ceiling_never_bars(self):
        assert _row_holds({"name": "x"}, 10 ** 9) is True
        assert _row_holds({"name": "x", "max_time": None}, 10 ** 9) is True

    def test_an_unreadable_ceiling_never_bars(self):
        assert _row_holds({"max_time": "whenever"}, 10 ** 9) is True

    def test_exactly_equal_fits(self):
        """A 15-minute ceiling holds a 15-minute job; the boundary is not
        an off-by-one that silently drops the cheapest row."""
        assert _row_holds({"max_time": "00:15:00"}, _FIFTEEN_MIN) is True
        assert _row_holds({"max_time": "00:15:00"}, _FIFTEEN_MIN + 1) is False

    def test_the_day_form_is_understood(self):
        assert _row_holds({"max_time": "1-00:00:00"}, 23 * 3600) is True
        assert _row_holds({"max_time": "1-00:00:00"}, 25 * 3600) is False
