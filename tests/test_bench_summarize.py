"""The sweep reader's one deck reader, on the deck `prep` wrote.

The data-keyed path — ``discover_points_from_jobset`` /
``run_summarize_jobset`` — is covered end-to-end in
tests/test_prep_bench_fold.py.
"""
from __future__ import annotations




# --------------------------------------------------------------------- #
#  What the trial actually ran -- the WIRING, not the parsers            #
# --------------------------------------------------------------------- #


def test_one_deck_reader_and_it_takes_the_first_match(tmp_path, monkeypatch):
    """libfdf's `fdf_locate` walks from the top and STOPS at the first
    matching label, so a deck naming a keyword twice is read with its
    FIRST value.  There were two readers here -- this one and
    `_winner_mechanism`'s loop, which kept the LAST -- so a duplicated
    keyword made the verdict name an algorithm SIESTA never used, and
    that verdict is what `prep task` offers to apply to production.

    The deck is the one `prep` wrote for H2, its run card stating the
    solver and the block size; the second `Diag.Algorithm` is a line a
    person added at its end by hand, the one way a deck comes to name a
    keyword twice (`prep` writes each once)."""
    from molbuilder.jobset.summarize import deck_item
    from support.road import (RUN_CARD, described, group_words, jobset,
                              road_tree, stage_file)
    calc = described(road_tree(tmp_path, monkeypatch), execution={
        **RUN_CARD["siesta"], "diag_algorithm": "ELPA-2STAGE",
        "block_size": 256})
    r = jobset("prep", "task", *group_words(calc, "coarse"), "--bundle", calc,
               "--target", "this")
    assert r.exit_code == 0, r.output
    deck = stage_file(calc, "coarse", ".fdf")
    deck.write_text(deck.read_text() + "Diag.Algorithm D&C\n")
    # BY THE CATALOGUE ITEM, read as SIESTA reads its deck (its keyword
    # reader: fdf's rule folds separators and case).
    assert deck_item(deck, "diag_algorithm", "siesta") == "ELPA-2STAGE"
    assert deck_item(deck, "block_size", "siesta") == "256"
    # ...and an item an optimization's deck leaves at SIESTA's own default
    # is stated by none of its lines.
    assert deck_item(deck, "scf_must_converge", "siesta") is None
    assert deck_item(tmp_path / "absent.fdf", "block_size", "siesta") is None
