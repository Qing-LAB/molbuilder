"""What a deck says for each thing the description states (`docs/engines/
siesta.md`, `docs/engines/pyscf.md`), case by case.

THE CASES ARE DATA -- ``tests/data/the_deck.toml`` -- and each runs down
the road a person runs (`support.road.run_road_case`): `jobset init` of a
held H2 or the row's structure, `prep`, and the deck read where prep wrote
it.  A deck has one writer, `prep`, so a deck rendered any other way is not
the one a run reads (`process/testing.md` § 3a).

PREVENTS: a value the description states that never reaches the deck, or
reaches it in a form the engine ignores -- each row names the run it would
silently change.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from support.road import run_road_case

TABLE = tomllib.loads(
    (Path(__file__).parent / "data" / "the_deck.toml").read_text())


@pytest.mark.parametrize("case", TABLE["case"],
                         ids=[c["name"] for c in TABLE["case"]])
def test_the_deck_says_what_the_description_states(case, tmp_path,
                                                    monkeypatch):
    run_road_case(TABLE, case, tmp_path, monkeypatch)
