"""The progress-log parser declines a file that is not a progress log.

What it reads from a log is tested on logs our own writer wrote
(`tests/test_molwatch_emitter.py`, `tests/test_molwatch_preview.py`, the
held-atom rows of `tests/data/held_atoms.toml`).
"""

from __future__ import annotations


from molbuilder.parse.engines.molwatch import MolwatchLogParser


def test_can_parse_rejects_non_molwatch(tmp_path):
    p = tmp_path / "garbage.txt"
    p.write_text("just some text\nnot a molwatch log\n")
    assert MolwatchLogParser.can_parse(str(p)) is False
