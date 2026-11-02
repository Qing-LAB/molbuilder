"""The sweep reader's pure units.

The data-keyed path — ``discover_points_from_jobset`` /
``run_summarize_jobset`` — is covered end-to-end in
tests/test_prep_bench_fold.py.
"""
from __future__ import annotations




# --------------------------------------------------------------------- #
#  What the trial actually ran -- the WIRING, not the parsers            #
# --------------------------------------------------------------------- #


def test_one_deck_reader_and_it_takes_the_first_match(tmp_path):
    """libfdf's `fdf_locate` walks from the top and STOPS at the first
    matching label, so a deck naming a keyword twice is read with its
    FIRST value.  There were two readers here -- this one and
    `_winner_mechanism`'s loop, which kept the LAST -- so a duplicated
    keyword made the verdict name an algorithm SIESTA never used, and
    that verdict is what `prep task` offers to apply to production."""
    from molbuilder.jobset.summarize import deck_item
    deck = tmp_path / "job.fdf"
    deck.write_text("Diag.Algorithm ELPA-1STAGE\n"
                    "Diag.Algorithm D&C\n"
                    "BlockSize 256\n")
    # BY THE CATALOGUE ITEM, read as SIESTA reads its deck (its keyword
    # reader: fdf's rule folds separators and case).
    assert deck_item(deck, "diag_algorithm", "siesta") == "ELPA-1STAGE"
    assert deck_item(deck, "block_size", "siesta") == "256"
    assert deck_item(deck, "mesh_cutoff", "siesta") is None
    assert deck_item(tmp_path / "absent.fdf", "block_size", "siesta") is None


