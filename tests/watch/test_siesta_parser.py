"""The SIESTA output parser declines a file that is not SIESTA's.

What it reads from an output is asked of runs made on the road with the
real SIESTA, in the e2e tier (`tests/test_siesta_flat_run_e2e.py`,
`tests/test_siesta_relax_run_e2e.py`, `tests/test_siesta_stopped_run_e2e.py`).
"""

from __future__ import annotations


from molbuilder.parse.engines.siesta import SiestaParser


def test_can_parse_rejects_non_siesta(tmp_path):
    p = tmp_path / "garbage.txt"
    p.write_text("just some random text\nhello world\n")
    assert SiestaParser.can_parse(str(p)) is False
