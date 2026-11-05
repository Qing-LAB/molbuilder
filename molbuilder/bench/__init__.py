"""``molbuilder.bench`` — the benchmark's library modules.

**The benchmark IS a jobset**: `molbuilder jobset prep bench <stage>`
renders the trial decks through the same five steps as a run, `launch
bench` launches them, `summarize bench` reads the
outputs into a verdict (`docs/execution/generator.md`).  The one library
module that serves that path lives here:

- :mod:`.result` — ``bench-result@1``: parse the trials' artifacts,
  choose, recommend.
"""
from __future__ import annotations
