"""The wrapper is TOLD whether the run uses the GPU (audit § 25.4).

`enable_gpu` declares ``read_by = ("wrapper",)`` — the wrapper depends on it —
and until 2026-08-14 nothing handed the value over, so the wrapper re-derived a
**user decision** by grepping the artifact that decision produced. Eleven other
resolved values are passed to ``render_run_wrapper``; this was the one that was
not, and it was derived independently in **four** places.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from molbuilder.runwrap import _wants_gpu


def _deck(tmp_path: Path, gpu: bool) -> Path:
    p = tmp_path / "job.fdf"
    p.write_text("SystemLabel job\n"
                 + ("Diag.ELPA.GPU .true.\n" if gpu else ""))
    return p


def test_told_beats_reading_the_deck(tmp_path):
    """The whole point: prep resolved it, so the wrapper does not re-derive."""
    cpu_deck = _deck(tmp_path, gpu=False)
    assert _wants_gpu(cpu_deck, True, None) is True
    gpu_deck = _deck(tmp_path / "g", gpu=True) if (tmp_path / "g").mkdir() or True else None
    assert _wants_gpu(gpu_deck, False, None) is False


def test_not_told_falls_back_to_the_deck(tmp_path):
    """``write_run_wrapper`` is called directly on a bare deck outside prep --
    which is why every parameter is Optional -- and a person may edit the deck,
    which is what actually runs.  So the scan stays, as the fallback."""
    assert _wants_gpu(_deck(tmp_path, gpu=True), None, None) is True
    assert _wants_gpu(_deck(tmp_path / "c", gpu=False)
                      if (tmp_path / "c").mkdir() or True else None,
                      None, None) is False


def test_an_explicit_env_still_wins_over_both(tmp_path):
    """Unchanged: naming an env points away from GPU whatever the deck says,
    so a user choosing the source build for its external ELPA is not
    overridden."""
    assert _wants_gpu(_deck(tmp_path, gpu=True), True, "molbuilder-siesta") is False
    assert _wants_gpu(_deck(tmp_path / "d", gpu=True)
                      if (tmp_path / "d").mkdir() or True else None,
                      None, "molbuilder-siesta") is False


def test_one_function_decides_for_every_site():
    """It was derived in FOUR places -- the env choice, the rank/thread budget,
    the memory audit and the sbatch header -- each calling the scanner itself.
    Four derivations of one fact is four chances to disagree."""
    import inspect
    from molbuilder import runwrap
    lines = inspect.getsource(runwrap).splitlines()
    calls = [i for i, ln in enumerate(lines)
             if "_fdf_requests_gpu(" in ln
             and not ln.lstrip().startswith(("#", "f'#", '"'))
             and "def _fdf_requests_gpu" not in ln]
    assert len(calls) == 1, (
        f"the scanner is called {len(calls)} times; it must be called ONCE, "
        f"inside _wants_gpu, so every site shares one decision")
    # and that one call is the fallback inside the decision function
    before = "\n".join(lines[:calls[0]])
    assert before.rindex("def _wants_gpu") > before.rindex("def render_run_wrapper") \
        if "def render_run_wrapper" in before else True
