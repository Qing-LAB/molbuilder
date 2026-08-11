"""Pin the parse-cache + parser contract from
docs/web/results.md (PR 4).

Two surfaces covered:

1. **Parser surface** (parse/engines/siesta.py): in-progress + committed
   frames whose SCF didn't report a finite energy MUST NOT fall
   back to ``step_initial_etot``.  The preamble Etot is preserved
   in ``runtime_info["initial_etot"]`` for display.

2. **Cache surface** (web/blueprints/watch.py): ``_MERGE_PARSE_CACHE``
   is an LRU OrderedDict keyed on ``(mtime, size)``.  Inserts gated
   on a 200ms freshness window so a torn read can't poison the
   cache.  LRU bound = 64 entries; oldest evicts on insert.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict

import pytest

from molbuilder.web.blueprints import watch as watch_mod


# --------------------------------------------------------------------- #
#  Parser: step_initial_etot fallback retired                           #
# --------------------------------------------------------------------- #


class TestSiestaParserNoEtotFallback:
    """The fallback that used ``step_initial_etot`` as a frame energy
    is removed in PR 4.  Both call sites (committed frame, in-
    progress frame) must be gone."""

    @pytest.fixture(scope="class")
    def parser_body(self):
        # Module rehomed in parse-module H1.engines (2026-06-20):
        # legacy ``molbuilder/parsers/siesta.py`` → ``molbuilder/
        # parse/engines/siesta.py`` (absorbed body lives at the
        # ``SiestaParser`` class — public surface for the body
        # parser; see model/parse.md § 3).
        path = (Path(__file__).resolve().parent.parent
                / "molbuilder" / "parse" / "engines" / "siesta.py")
        return path.read_text()

    def test_committed_frame_no_etot_fallback(self, parser_body):
        """The committed-frame fallback at ~line 705 is removed.
        Pin: no ``frame_energy = step_initial_etot`` literal."""
        import re
        assert not re.search(
            r"\bframe_energy\s*=\s*step_initial_etot\b",
            parser_body,
        ), ("siesta.py still falls back to step_initial_etot for "
            "committed frames.  PR 4 contract § 6 requires the "
            "fallback to be removed; a divergent SCF's frame energy "
            "must be None, not the misleading preamble Etot.")

    def test_in_progress_frame_no_etot_fallback(self, parser_body):
        """The in-progress-frame fallback at ~line 1531 is removed.
        Pin: no ``if in_prog_energy is None and step_initial_etot``
        block.  The cycle-walking inner loop's own
        ``in_prog_energy = fv`` is the GOOD path (a real cycle
        energy got read) and stays."""
        import re
        assert not re.search(
            r"if\s+in_prog_energy\s+is\s+None\s+and\s+step_initial_etot",
            parser_body,
        ), ("siesta.py still falls back to step_initial_etot for "
            "in-progress frames.  PR 4 contract § 6 requires the "
            "fallback to be removed; placeholder frames must have "
            "energy=None.")

    def test_runtime_info_carries_initial_etot(self, parser_body):
        """``step_initial_etot`` must still be preserved in
        ``runtime_info["initial_etot"]`` for display purposes -- it
        just isn't a frame energy anymore."""
        import re
        assert re.search(
            r'runtime_info\["initial_etot"\]\s*=\s*float\(\s*step_initial_etot\s*\)',
            parser_body,
        ), ("siesta.py no longer surfaces step_initial_etot as "
            "runtime_info['initial_etot'].  The value is lost; the "
            "Results-tab can't show the user the preamble Etot.")


# --------------------------------------------------------------------- #
#  Cache: LRU + (mtime, size) key + 200ms freshness gate                #
# --------------------------------------------------------------------- #


class TestParseCacheLRU:
    """``_MERGE_PARSE_CACHE`` is an OrderedDict acting as LRU."""

    def setup_method(self, _):
        watch_mod._MERGE_PARSE_CACHE.clear()

    def test_cache_is_ordered_dict(self):
        """Pin the OrderedDict shape so LRU eviction (move_to_end +
        popitem(last=False)) works."""
        from collections import OrderedDict
        assert isinstance(watch_mod._MERGE_PARSE_CACHE, OrderedDict)

    def test_get_hit_moves_to_end(self):
        """On a cache hit, the entry moves to the most-recently-used
        position so it survives subsequent evictions."""
        watch_mod._merge_parse_cache_put("a", 1.0, 100, {"x": 1})
        watch_mod._merge_parse_cache_put("b", 2.0, 200, {"x": 2})
        # "a" is currently LRU; touch it.
        got = watch_mod._merge_parse_cache_get("a", 1.0, 100)
        assert got == {"x": 1}
        # After the hit, "a" should be at the END (most-recent).
        assert list(watch_mod._MERGE_PARSE_CACHE.keys()) == ["b", "a"]

    def test_get_miss_on_mtime_change(self):
        """Different mtime invalidates -- the cached value is NOT
        returned and the entry stays put (caller will overwrite via
        put)."""
        watch_mod._merge_parse_cache_put("a", 1.0, 100, {"x": 1})
        assert watch_mod._merge_parse_cache_get("a", 2.0, 100) is None

    def test_get_miss_on_size_change(self):
        """Size key is load-bearing: 1-second mtime granularity
        systems (NFS / FAT) get torn-read defense for free."""
        watch_mod._merge_parse_cache_put("a", 1.0, 100, {"x": 1})
        assert watch_mod._merge_parse_cache_get("a", 1.0, 200) is None

    def test_lru_bound_evicts_oldest(self):
        """When the cache exceeds _MERGE_PARSE_CACHE_LRU_BOUND
        entries, the oldest (front) evicts."""
        N = watch_mod._MERGE_PARSE_CACHE_LRU_BOUND
        for i in range(N):
            watch_mod._merge_parse_cache_put(
                f"file_{i}", float(i), i * 10, {"x": i})
        assert len(watch_mod._MERGE_PARSE_CACHE) == N
        # One more push evicts file_0.
        watch_mod._merge_parse_cache_put("new", 999.0, 0, {"x": 999})
        assert len(watch_mod._MERGE_PARSE_CACHE) == N
        assert "file_0" not in watch_mod._MERGE_PARSE_CACHE
        assert "new" in watch_mod._MERGE_PARSE_CACHE

    def test_freshness_gate_blocks_recent_writes(self, tmp_path):
        """``_merge_parse_cache_is_fresh`` returns False when the
        file's mtime is younger than 200 ms.  Caller MUST NOT cache
        in that case -- the file was still being written."""
        # mtime = now (file just modified) -> NOT fresh
        assert not watch_mod._merge_parse_cache_is_fresh(time.time())
        # mtime = 1 second ago -> fresh
        assert watch_mod._merge_parse_cache_is_fresh(time.time() - 1.0)

    def test_freshness_threshold_is_200ms(self):
        """Document the threshold value matches the contract § 6
        spec.  If the threshold drifts, callers may unexpectedly
        cache mid-write reads."""
        assert watch_mod._MERGE_PARSE_FRESHNESS_S == 0.200


class TestParseCacheIntegration:
    """End-to-end: torn-read defense + warm-cache reuse."""

    def setup_method(self, _):
        watch_mod._MERGE_PARSE_CACHE.clear()

    def test_torn_read_not_cached(self, tmp_path):
        """A file just written (mtime ~= now) MUST NOT be cached on
        first parse -- the read may have raced an in-flight write.
        Subsequent ticks re-parse cheaply."""
        path = tmp_path / "fresh.log"
        path.write_text("")
        mt = os.path.getmtime(path)
        sz = os.path.getsize(path)
        # Simulate the cache-miss branch: helper says "not fresh
        # enough to cache".
        assert not watch_mod._merge_parse_cache_is_fresh(mt)
        # Don't put.  Cache stays empty.
        assert len(watch_mod._MERGE_PARSE_CACHE) == 0

    def test_aged_file_caches(self, tmp_path):
        """A file with mtime older than the freshness window caches
        normally on first parse."""
        path = tmp_path / "aged.log"
        path.write_text("")
        # Backdate mtime to 1s ago.
        old = time.time() - 1.0
        os.utime(path, (old, old))
        mt = os.path.getmtime(path)
        sz = os.path.getsize(path)
        assert watch_mod._merge_parse_cache_is_fresh(mt)
        watch_mod._merge_parse_cache_put(str(path), mt, sz, {"y": 1})
        assert watch_mod._merge_parse_cache_get(str(path), mt, sz) == {"y": 1}
