"""PySCF's SCF settings go through the one door — `script-preparation.md` § 4.2.

**What is new here, and why the existing PySCF tests could not say it.**  They
assert *properties of the artifact*: that ``mf.damp = 0.4`` appears, that
``mf.conv_tol_grad`` does not when it is unset.  Those stay true whoever writes
the file, which is exactly what makes them the gate a new writer must pass — and
they all still pass.

What none of them can say is **where the reason came from**.  A generator that
types its own sentence beside a value passes every one of those tests and still
drifts the moment a default moves, which is the defect this layer exists to end:
PySCF read the catalogue **zero** times while hand-typing 231 comment lines, 45
of them stating a number.

So these tests assert the rule rather than the text: the value and its reason are
one act, and the reason is the catalogue's.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from molbuilder.config.pyscf import PySCFConfig
from molbuilder.pyscf import layout
from molbuilder.pyscf.input import render_script
from molbuilder.script_emit import parameter
from molbuilder.structure import Structure


def _struct() -> Structure:
    return Structure(elements=["O", "H", "H"],
                     positions=np.array([[0., 0., 0.],
                                         [0.957, 0., 0.],
                                         [-0.24, 0.927, 0.]]))


def _dft_block(text: str) -> str:
    start = text.index("# --- Functional, density fitting, dispersion ---")
    rest = text[start:]
    end = rest.find("# ===")
    return rest[:end if end > 0 else 1200]


def _scf_block(text: str) -> str:
    """The SCF section as emitted, up to the next section or blank run."""
    start = text.index("# --- SCF settings ---")
    rest = text[start:]
    end = rest.find("mf.chkfile")
    return rest[:end if end > 0 else len(rest)]


def _render(**over) -> str:
    kw = dict(job_name="t", diis_space=16, damp=0.4, level_shift=0.2)
    kw.update(over)
    return render_script(_struct(), PySCFConfig(**kw))


# ------------------------------------------------------- the rule itself --

@pytest.mark.parametrize("name", [n for n in layout.DFT_SECTION.items
                                  if n not in ("density_fit",)])
def test_every_emitted_dft_value_carries_the_catalogue_s_own_note(name):
    """The functional, the grid and dispersion, each with its declaration."""
    block = _dft_block(_render())
    note = [ln for ln in parameter(name, "pyscf").note()
            if ln.startswith("# ") and len(ln) > 4]
    assert note, f"the catalogue declares no note for {name}"
    assert note[0] in block, f"{name} was emitted without its catalogue note"


def test_the_functional_deviation_and_its_citation_reach_the_script():
    """Why B3LYP rather than PySCF's fallback, in the file a scientist opens."""
    block = _dft_block(_render())
    assert "PySCF's own default is 'LDA,VWN'" in block
    assert 'mf.xc = "B3LYP"' in block


def test_a_non_dft_method_emits_no_functional_and_no_grid():
    """`is_dft` is the engine's context, and the door honours it."""
    block = _render(method="RHF")
    assert "mf.xc" not in block
    assert "mf.grids.level" not in block


@pytest.mark.parametrize("name", [n for n in layout.SCF_SECTION.items
                                  if n not in ("scf_conv_tol_grad",)])
def test_every_emitted_scf_value_carries_the_catalogue_s_own_note(name):
    """The value and its reason are ONE act, and the reason is not typed here."""
    block = _scf_block(_render())
    note = [ln for ln in parameter(name, "pyscf").note()
            if ln.startswith("# ") and len(ln) > 4]
    assert note, f"the catalogue declares no note for {name}"
    assert note[0] in block, (
        f"{name} was emitted without the catalogue's note above it; the "
        f"generator is stating its own reason again")


def test_the_deviation_note_reaches_the_script_so_a_reader_sees_why():
    """A number this project moved says so, in the file, with the engine's own.

    `max_cycle` is 100 here against PySCF's 50, and the catalogue explains that
    it is a runaway guard rather than a target.  A reader who never opens the
    catalogue still gets that.
    """
    block = _scf_block(_render())
    assert "PySCF's own default is 50" in block
    assert "mf.max_cycle = 100" in block


def test_the_section_carries_no_comment_the_catalogue_did_not_write():
    """No hand-typed sentence survives in this section — the whole point.

    A comment that is nobody's declaration is a claim with no source, and it is
    what goes stale.  Every comment line in the SCF block must be traceable to
    one of the section's own items.
    """
    block = _scf_block(_render())
    declared = set()
    for name in layout.SCF_SECTION.items:
        declared.update(ln for ln in parameter(name, "pyscf").note())
    stray = [ln for ln in block.splitlines()
             if ln.startswith("#") and ln not in declared
             and not ln.startswith("# ---")
             and "conv_tol_grad" not in ln]     # the derived-value advisory
    assert stray == [], f"comments with no declaration behind them: {stray}"


# --------------------------------------------------- emission conditions --

def test_a_parameter_that_declines_is_absent_entirely_not_commented_out():
    """`None` from the syntax door means the line is not there at all."""
    block = _scf_block(_render(diis_space=8, damp=0.0, level_shift=0.0))
    assert "mf.diis_space" not in block
    assert "mf.damp" not in block
    assert "mf.level_shift" not in block
    assert "mf.conv_tol" in block          # the unconditional ones remain


def test_quiet_scripts_keep_every_value_and_drop_every_note():
    quiet = _scf_block(_render(verbose_comments=False))
    assert "mf.max_cycle = 100" in quiet
    assert "mf.damp = 0.4" in quiet
    assert "PySCF's own default is 50" not in quiet


def test_the_spelling_the_engine_chose_is_the_spelling_emitted():
    """Syntax is the engine's business; the framework never reformats it."""
    block = _scf_block(_render())
    assert "mf.conv_tol  = 1e-09" in block          # two spaces, a column
    assert 'mf.init_guess = "minao"' in block       # a quoted string


def test_the_layout_names_catalogue_items_that_actually_exist():
    """A typo in the layout table would silently emit nothing at all."""
    missing = [n for n in layout.SCF_SECTION.items
               if not parameter(n, "pyscf").known]
    assert missing == [], f"layout names items PySCF does not declare: {missing}"


# ------------------------------------------- the effective-parameters record

def _record_block(text: str) -> str:
    start = text.index("_MB_PARAMS = {}")
    return text[start:text.index("for _k, (_d, _r, _e)", start)]


def _run_record(text: str, **engine_state):
    """Execute the record block against stub `mol` / `mf` objects.

    The point of the record is what the ENGINE holds, so a test of it has to
    supply an engine that holds something -- including something different from
    what was asked for, which is the case the record exists for.
    """
    import io
    import contextlib

    blk = text[text.index("def _mb_read("):
               text.index("_RUNTIME_INFO.update({_k:")]
    blk += "\n_OUT = dict(_MB_PARAMS)\n"

    class _Grids:
        level = engine_state.get("grid_level", 4)

    class _MF:
        conv_tol = engine_state.get("conv_tol", 1e-9)
        conv_tol_grad = None
        max_cycle = engine_state.get("max_cycle", 100)
        init_guess = "minao"
        level_shift = 0
        diis_space = 8
        damp = 0
        xc = engine_state.get("xc", "B3LYP")
        disp = "d3bj"
        chkfile = "t.chk"
        grids = _Grids()

    class _Mol:
        charge = 0
        spin = 0
        basis = "def2-SVP"
        symmetry = False
        max_memory = engine_state.get("max_memory", 4000)
        verbose = 4
        stdout = None

    ns = {"mol": _Mol(), "mf": _MF(), "_RUNTIME_INFO": {}}
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        exec(blk, ns)
    return ns["_OUT"], buf.getvalue()


def test_the_record_covers_every_parameter_this_engine_declares():
    """Full coverage is the point: defaults, customised values, and the ones
    that never reach the engine at all."""
    from molbuilder.pyscf.layout import recorded_items

    rec, _ = _run_record(_render())
    assert set(rec) == set(recorded_items())
    assert len(rec) >= 40, f"only {len(rec)} parameters recorded"


def test_the_record_shows_the_catalogue_default_beside_what_this_run_asked():
    """Two of the three columns, and the reason a reader can tell a deliberate
    choice from a value nobody touched."""
    rec, _ = _run_record(_render(scf_max_cycle=250))
    default, requested, _engine = rec["scf_max_cycle"]
    assert default == 100, "the catalogue's own recommendation"
    assert requested == 250, "what this description resolved to"


def test_a_setting_the_engine_silently_overrode_is_visible_as_a_disagreement():
    """**The reason the third column is read back and not echoed.**

    Here the engine holds 999 while the deck asked for 250.  A record that
    restated our own intent would print 250 and hide it.
    """
    rec, out = _run_record(_render(scf_max_cycle=250), max_cycle=999)
    default, requested, engine = rec["scf_max_cycle"]
    assert (requested, engine) == (250, 999)
    assert "999" in out, "the disagreement must reach the log, not just memory"


def test_a_parameter_the_engine_has_no_setting_for_is_marked_not_asked():
    rec, _ = _run_record(_render())
    assert rec["save_optimized_xyz"][2] == "-"
    assert rec["optimizer"][2] == "-"


def test_the_record_is_fenced_so_one_reader_serves_either_engine():
    from molbuilder.script_emit import BLOCK_PARAMETERS, begin_marker, end_marker

    _, out = _run_record(_render())
    assert begin_marker(BLOCK_PARAMETERS) in out
    assert end_marker(BLOCK_PARAMETERS) in out


def test_a_new_catalogue_item_joins_the_record_without_an_edit():
    """The list is generated from the catalogue, so coverage cannot go stale."""
    from molbuilder.pyscf.layout import recorded_items

    block = _record_block(_render())
    for name in recorded_items():
        assert f"_MB_PARAMS[{name!r}]" in block, f"{name} missing from the record"
