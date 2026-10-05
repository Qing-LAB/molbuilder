"""The SIESTA output parser declines a file that is not SIESTA's.

What it reads from an output is tested on outputs real runs wrote, read
where they lie: `tests/watch/fixtures/siesta_frozen`,
`tests/fixtures/siesta_relax` and `tests/fixtures/siesta_flat_h2`.  The tests
that parsed a synthetic output here were retired 2026-10-04
(`process/testing.md` § 6).
"""

from __future__ import annotations


from molbuilder.parse.engines.siesta import SiestaParser


# Retired 2026-10-04 (user: "any fucking faking tests should be retired"):
# 23 tests here parsed a SIESTA output invented as text (`process/testing.md` § 6).


def test_can_parse_rejects_non_siesta(tmp_path):
    p = tmp_path / "garbage.txt"
    p.write_text("just some random text\nhello world\n")
    assert SiestaParser.can_parse(str(p)) is False
