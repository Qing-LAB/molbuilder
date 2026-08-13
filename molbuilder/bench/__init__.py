"""``molbuilder.bench`` — what remains after the benchmark unified into
`jobset` (the 2026-08-12 fold, completed 2026-08-13).

**The benchmark IS a jobset**: `molbuilder jobset prep bench <stage>`
renders the trial decks through the same five steps as a run, `submit
bench` launches them one per invocation, `summarize bench` reads the
outputs into a verdict (`docs/execution/generator.md`).  Two library
modules serve that path and live here:

- :mod:`.grid` — the machine-probed ``(G, K, C)`` sweep grid.
- :mod:`.result` — ``bench-result@1``: parse the trials' artifacts,
  choose, recommend.

**The legacy in-place ``siesta-gpu`` stack was DELETED here** (2026-08-13,
user decision — "we don't need to keep obsolete paths"): its own point
grammar, its own ``.out`` parser and CSV, the regex deck-splicing that
`engines/template.md` § 8.1 bans on the live path, and the hardcoded
``.DM/.CG/.XV`` trio that predated `job-contracts.md` § 4.2a's one rules
file.  It was a parallel benchmark framework; the framework it
paralleled is the one everything now goes through.
"""
from __future__ import annotations
