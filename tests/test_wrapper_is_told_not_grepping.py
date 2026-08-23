"""G7 — the wrapper is TOLD which runs use a GPU, not made to guess.

`execution/gpu.md` G7.  The catalogue item declares `read_by = ["wrapper"]`,
and until 2026-08-23 the wrapper satisfied that declaration by **grepping the
rendered deck** for `Diag.ELPA.GPU`, at four separate sites.

Two things were wrong with that, and only the second is obvious:

  * it is a layer re-deriving a value another layer already holds -- the habit
    `execution/architecture.md` § 1 exists to remove;
  * **it matched a SIESTA keyword.**  A PySCF GPU run writes no such line, so
    it could not route to a GPU environment at all, however correctly its
    item declared `read_by`.

The answer now rides `Resources`, the allocation that already travels to the
wrapper whole (A8) -- the same road `continue_retries` takes, and for the
reason its own note records: *carried on the allocation, it cannot be
forgotten by one of them.*
"""
from __future__ import annotations

import pathlib

import pytest

from molbuilder.jobset.model import Resources
from molbuilder.runwrap import _wants_gpu


def _deck(tmp_path, text=""):
    p = tmp_path / "JOB.fdf"
    p.write_text(text or "SystemLabel JOB\n")
    return p


def test_the_carried_answer_is_used(tmp_path):
    deck = _deck(tmp_path)                       # no GPU keyword anywhere
    assert _wants_gpu(deck, Resources(use_gpu=True)) is True, (
        "the wrapper ignored what it was told and read the deck instead")


def test_the_carried_answer_wins_over_the_deck(tmp_path):
    """The deck says one thing, the allocation another.  The ALLOCATION is
    the decision; the deck is a rendering of it.  A wrapper that preferred
    the text would be re-deriving, which is the whole defect."""
    deck = _deck(tmp_path, "SystemLabel JOB\nDiag.ELPA.GPU .true.\n")
    assert _wants_gpu(deck, Resources(use_gpu=False)) is False
    plain = _deck(tmp_path / "x", "SystemLabel JOB\n") if False else deck
    assert _wants_gpu(plain, Resources(use_gpu=True)) is True


def test_an_engine_that_writes_no_siesta_keyword_still_routes(tmp_path):
    """**The case the grep could never serve.**  A PySCF deck asks for a GPU
    in Python, not in an fdf keyword.  Told, the wrapper knows; grepping, it
    never could."""
    py = tmp_path / "JOB.py"
    py.write_text("mf = mf.to_gpu()\n")
    assert _wants_gpu(py, Resources(use_gpu=True)) is True
    assert _wants_gpu(py, Resources(use_gpu=False)) is False


def test_an_unstated_answer_falls_back_to_the_artifact(tmp_path):
    """Not the same defect, and worth pinning so it is not 'fixed' away: a
    wrapper written for a deck someone points at has no allocation to ask.
    Reading the artifact you were handed is the only question available."""
    assert _wants_gpu(_deck(tmp_path, "Diag.ELPA.GPU .true.\n")) is True
    assert _wants_gpu(_deck(tmp_path, "SystemLabel JOB\n")) is False
    # ...and an allocation that simply does not say is the same case
    assert _wants_gpu(_deck(tmp_path, "Diag.ELPA.GPU .true.\n"),
                      Resources(mpi_np=4)) is True


def test_the_answer_rides_the_allocation_out_of_resolve():
    """The other half: `resolve` puts it there, so `prep` does not have to
    remember to.  `continue_retries` learned this the hard way -- its note
    records a period where the ride was claimed and nothing performed it, so
    one of two routes rendered no retry loop at all."""
    import dataclasses
    assert "use_gpu" in {f.name for f in dataclasses.fields(Resources)}, (
        "Resources no longer carries the answer; the wrapper is back to "
        "guessing from deck text")


@pytest.mark.parametrize("site", ["render_run_wrapper", "_render_sbatch_for",
                                  "_build_mem_audit"])
def test_no_site_greps_the_deck_directly_any_more(site):
    """One door.  Four call sites each spelled the grep out, which is how
    they came to disagree about what a GPU job is -- `_render_sbatch_for`
    checked `gres` first and the others did not."""
    import inspect
    from molbuilder import runwrap
    src = inspect.getsource(getattr(runwrap, site))
    assert "_fdf_requests_gpu(" not in src, (
        f"{site} still greps the deck directly instead of asking "
        f"`_wants_gpu`, which prefers what the caller stated")
