"""One warm-file inventory per engine, and every surface derives from it.

Contract: `staged-runs-architecture.md` item 12c, whose *Done when* is stated
as a mutation and is written that way here — **add a suffix to the list and
watch both behaviours change.**

**The failure this prevents is silent and asymmetric.** Two lists that agree
today, with nothing keeping them agreeing:

* add a warm hook to the **carry/banner** list alone and a ``--cold`` run
  warm-starts from it anyway — a contaminated calculation that reports
  success;
* add it to the **``--cold`` mover** alone and a run announces *"initial run
  (clean state)"* and then has that very file moved aside as warm state.

`run-identity.md § 5` says the banner is the half that must never be weakened,
because it is the one always present. SIESTA's pair was derived from a single
tuple by P3's Review 2 for exactly this reason. **PySCF's was missed until
2026-08-10**: its mover covered five suffixes while its banner tested ``.chk``
alone, so a run holding only ``<JOB>_optimized.xyz`` hit the second failure.

The third list — the one that fed attempt directories — is gone: P7 unit 1
retired ``_attempt_dir_block``, and the two helpers that built filenames for
it went too. **The plan predicted they had no other caller and was wrong**:
``validation/identity.py`` used them, then stripped the id back off to recover
the suffixes it actually wanted. It reads the tuples directly now, so the
subtraction landed and removed a re-derivation on the way out.
"""
from __future__ import annotations

import re

import pytest

from molbuilder import runwrap


ENGINES = (
    pytest.param("siesta", "_SIESTA_WARM_SUFFIXES", ".fdf",
                 "SystemLabel job\n", id="siesta"),
    pytest.param("pyscf", "_PYSCF_WARM_SUFFIXES", ".py",
                 'JOB = "job"\nimport pyscf\n', id="pyscf"),
)


def _wrapper(tmp_path, ext, body, **kw):
    p = tmp_path / f"job{ext}"
    p.write_text(body)
    return runwrap.render_run_wrapper(p, env="molbuilder-siesta", **kw)


@pytest.mark.parametrize("engine,const,ext,body", ENGINES)
def test_adding_a_suffix_changes_both_behaviours(tmp_path, monkeypatch,
                                                 engine, const, ext, body):
    """12c's *Done when*, executed rather than described.

    A suffix nobody would invent is added to the engine's one tuple, and the
    emitted wrapper must show it in **both** places: the ``--cold`` move-aside
    and the startup banner's warm-state test. One of the two is the silent
    contamination; the other is the silent false 'clean start'.
    """
    marker = "_mb_probe.state"
    before = _wrapper(tmp_path, ext, body)
    assert marker not in before, "the probe suffix must not already occur"

    monkeypatch.setattr(runwrap, const,
                        tuple(getattr(runwrap, const)) + (marker,))
    after = _wrapper(tmp_path, ext, body)

    banner, mover = _two_surfaces(after)

    assert re.search(rf'\[ -e "[^"]*{re.escape(marker)}" \]', banner), (
        "the startup banner does not read the engine's warm list: a run whose "
        "only warm file has this suffix would announce a clean start")
    assert marker in mover, (
        "the --cold move-aside does not read the engine's warm list: the file "
        "would survive --cold and warm-start a run told to start clean")


#: The generated comments that open the two blocks, in the order the wrapper
#: emits them: the ``--cold`` mover first, the startup banner after it.
#:
#: Anchoring on both is what makes the two assertions independent. An earlier
#: version of this file split on the first occurrence of the word *"cold"* --
#: which lands in an unrelated ``--help`` line -- so one block's match
#: satisfied the other's assertion, and two mutations retyping the mover's
#: list survived. **This module's own subject, reproduced inside its test.**
_MOVER_HEADING = "--- Cold-restart: move engine warm-start files aside ---"
_BANNER_HEADING = "--- Runtime status banner"


def _two_surfaces(text: str):
    """Split a wrapper into its ``(banner, --cold mover)`` halves.

    The two halves of the contract have to be checked apart, or a list read
    correctly by one of them covers for the other.
    """
    for anchor in (_MOVER_HEADING, _BANNER_HEADING):
        assert anchor in text, (
            f"{anchor!r} is not where this test expects it; update the anchor "
            "rather than widening the search, which is how the split stopped "
            "separating anything the first time")
    i, j = text.index(_MOVER_HEADING), text.index(_BANNER_HEADING)
    assert i < j, "the wrapper's block order moved; the split is now wrong"
    return text[j:], text[i:j]


@pytest.mark.parametrize("engine,const,ext,body", ENGINES)
def test_the_carry_inventory_that_fed_attempt_directories_is_gone(
        engine, const, ext, body):
    """Fix by subtraction (P7 unit 5): the third list went with the block it
    served, rather than becoming a third thing to keep in step."""
    assert not hasattr(runwrap, "_SIESTA_WARM_SUFFIX_FILES")
    assert not hasattr(runwrap, "_PYSCF_WARM_FILES")
    assert not hasattr(runwrap, "_attempt_dir_block")
    assert hasattr(runwrap, const), "the surviving inventory is named"


@pytest.mark.parametrize("engine,const,ext,body", ENGINES)
def test_no_generated_wrapper_changes_directory(tmp_path, engine, const,
                                                ext, body):
    """`job-contracts.md § 2.1`: **the caller's working directory is the
    contract** — both launchers establish it, and neither the wrapper nor the
    engine ever navigates.

    ``_attempt_dir_block`` was the one thing in the system that broke it: it
    resolved ``run-<n>/``, created it, and ``cd``'d in, so everything after
    that line ran somewhere the caller had not chosen. Retiring it restored an
    invariant rather than tidying one, and this is what holds it.
    """
    text = _wrapper(tmp_path, ext, body)
    cds = re.findall(r"(?m)^\s*cd\s+\S+", text)
    assert cds == [], f"{engine} wrapper navigates: {cds}"


def test_render_run_wrapper_no_longer_takes_attempt_dirs():
    """The parameter is gone from both entry points, so no caller can ask for
    the retired behaviour and quietly get the flat one instead."""
    import inspect
    for fn in (runwrap.render_run_wrapper, runwrap.write_run_wrapper):
        assert "attempt_dirs" not in inspect.signature(fn).parameters
