"""Blocks of the run script (``molbuilder.runwrap``) that are their own
programs, run as bash.  (The cold sweep's: `test_runwrap_cold_restart.py`.)
"""

from __future__ import annotations


# --------------------------------------------------------------------- #
#  SIESTA (.fdf) wrapper text                                           #
# --------------------------------------------------------------------- #


def test_the_notice_that_replaced_it_is_about_ORBITALS_and_only_advises():
    """**The objective number, and it never refuses** *(user ruling)*.

    SIESTA distributes ORBITALS across ranks, so ``n_orbitals / mpi_np`` is
    the occupancy and it wants to be greater than one.  At or below it the
    user is told *"your CPUs are not going to be fully used"* -- and the run
    proceeds.

    The orbital count is the ``10 x n_atoms`` DZP estimate the BlockSize
    bound and the deck's BENCH-MARKS block already use, so the deck and the
    notice cannot disagree.
    """
    import subprocess
    from molbuilder.runwrap import _orbitals_per_rank_notice

    notice = _orbitals_per_rank_notice(10)          # ~100 orbitals

    def _say(ranks):
        r = subprocess.run(["bash", "-c", f"_mpi_np={ranks}\n" + notice],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return (r.stdout + r.stderr).strip()

    assert _say(20) == "", "20 ranks over ~100 orbitals is 5 each -- nothing to say"
    for ranks in (100, 200):
        out = _say(ranks)
        assert "not going to be fully used" in out, out
        assert str(ranks) in out and "100" in out, (
            "the notice must show both numbers so the claim is checkable")
        assert "not a limit" in out

    assert _orbitals_per_rank_notice(None) == "", (
        "a deck with no NumberOfAtoms gets no notice -- inventing the number "
        "is what the ruling removed")

