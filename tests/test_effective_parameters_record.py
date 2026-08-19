"""The effective-parameters record — one fence, both engines.

**What it is for.** When a result looks wrong, three questions get asked: what
does this project recommend, what did this run ask for, and what did the engine
actually do with it. Until now a log answered none of them: the deck stated the
request and nothing recorded what was heard.

**Why the engines answer differently, and why that is honest.** PySCF's script
can read its own ``mol`` / ``mf`` back after setup, so it records three columns
and a silent override shows up as a disagreement between two of them. SIESTA is
a separate process that has not started when its wrapper runs, so *what the
engine holds* is not knowable there; what the wrapper can say truthfully is what
it is handing over. The fence is shared so one reader serves both; the columns
differ where the truth differs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molbuilder.config.pyscf import PySCFConfig
from molbuilder.runwrap import _effective_parameters_block
from molbuilder.script_emit import (BLOCK_PARAMETERS, begin_marker,
                                    end_marker, parameter)
from molbuilder.structure import Structure


@pytest.fixture
def deck(tmp_path) -> Path:
    p = tmp_path / "t.fdf"
    p.write_text("# a comment\nSystemLabel  t\n\nMeshCutoff 200.0 Ry\n",
                 encoding="utf-8")
    return p


def test_both_engines_write_the_same_fence(deck):
    """**W8** (`script-preparation.md` § 4.2b) — the record covers every
    parameter and is generated, so one reader serves either engine.

    One reader, either engine — which is the point of sharing the name."""
    from molbuilder.pyscf.input import render_script

    siesta = _effective_parameters_block(deck)
    pyscf = render_script(
        Structure(elements=["H", "H"],
                  positions=np.array([[0., 0., 0.], [0.74, 0., 0.]])),
        PySCFConfig(job_name="t"))
    for text in (siesta, pyscf):
        assert begin_marker(BLOCK_PARAMETERS) in text
        assert end_marker(BLOCK_PARAMETERS) in text


def test_siesta_records_the_deck_as_the_engine_parses_it(deck):
    """Comments and blanks stripped — exactly the lines libfdf reads."""
    block = _effective_parameters_block(deck)
    assert 'grep -v "^[[:space:]]*#"' in block
    assert 'grep -v "^[[:space:]]*$"' in block
    assert deck.name in block


def test_siesta_reads_the_deck_at_launch_not_at_generation(deck):
    """A deck edited after `prep` must record what the engine will really see.

    So the wrapper greps the file at run time rather than baking its contents.
    """
    block = _effective_parameters_block(deck)
    assert "MeshCutoff 200.0" not in block, (
        "the deck's own values must not be baked into the wrapper; the run "
        "reads the file, so an edit after prep is still recorded")


def test_siesta_records_what_the_deck_leaves_out_with_its_default(deck):
    """A keyword absent from the deck takes the engine default, and a reader
    chasing a surprising number has to be able to see that it was never set."""
    block = _effective_parameters_block(deck)
    assert "the engine default applies" in block
    # this toy deck sets neither, so both must be named
    assert "xc_authors" in block
    assert "relax_force_tol" in block


def test_an_item_the_deck_does_carry_is_not_listed_as_absent(deck):
    block = _effective_parameters_block(deck)
    absent = block.split("the engine default applies")[1]
    assert "mesh_cutoff" not in absent, (
        "mesh_cutoff IS in this deck; listing it as absent would be a lie "
        "about what the engine reads")


def test_the_absent_list_is_generated_from_the_catalogue(deck):
    """Not a hand-kept list: a new SIESTA item joins it with no edit here."""
    import molbuilder.template as T
    from molbuilder.script_emit import _catalogue

    block = _effective_parameters_block(deck)
    declared = {i.name for i in T.select(_catalogue(), engine="siesta")
                if parameter(i.name, "siesta").writes}
    named = {n for n in declared if n in block}
    assert len(named) >= 10, (
        f"only {len(named)} of {len(declared)} keyword-writing items appear")


# ------------------------------------------- every route that writes, checks

def test_the_cli_convert_route_runs_the_check_gate_too(tmp_path, monkeypatch):
    """`prep` is not the only door into rendering, so it cannot be the only
    door that checks.

    A deck that does not parse is no less broken for having come from the CLI.
    Two routes writing one kind of artifact and only one of them verifying it
    is the exact shape of defect this layer exists to end -- it is how a
    charged deck once shipped an instruction to run a file that was never
    written.
    """
    import molbuilder.pyscf.layout as layout
    from molbuilder.issues import Issue, ValidationError
    from molbuilder.pyscf.input import convert

    src = tmp_path / "w.xyz"
    src.write_text("3\n\nO 0 0 0\nH 0.957 0 0\nH -0.24 0.927 0\n",
                   encoding="utf-8")

    # a clean conversion passes the gate
    assert convert(str(src), str(tmp_path / "ok.py"))["n_atoms"] == 3

    # and a refusal from the gate stops it
    monkeypatch.setattr(layout, "check_rules",
                        lambda text, struct, cfg: [
                            Issue("error", "deliberately wrong",
                                  where="deck.test")])
    with pytest.raises(ValidationError):
        convert(str(src), str(tmp_path / "bad.py"))


# ------------------------------------------------- SIESTA's own check rules

def _fdf(**over):
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.siesta.input import render_fdf

    struct = Structure(elements=["O", "H", "H"],
                       positions=np.array([[0., 0., 0.],
                                           [0.957, 0., 0.],
                                           [-0.24, 0.927, 0.]]))
    cfg = SiestaConfig(system_label="t", **over)
    return render_fdf(struct, cfg), struct, cfg


def test_a_clean_siesta_deck_passes_its_own_rules():
    from molbuilder.siesta.layout import check_rules

    text, struct, cfg = _fdf()
    assert check_rules(text, struct, cfg) == []


def test_a_keyword_written_twice_is_refused_because_libfdf_takes_the_first():
    """The worst kind of wrong: the deck reads as though it says what you meant.

    ``fdf_locate`` walks from the top and stops at the first match, so a
    duplicate does not conflict loudly -- it silently wins, and the line a
    person edited later is the one being ignored.
    """
    from molbuilder.siesta.layout import check_rules

    text, struct, cfg = _fdf()
    issues = check_rules(text + "\nMeshCutoff 999.0 Ry", struct, cfg)
    assert [i for i in issues if "twice" in i.message], issues


def test_a_duplicate_that_agrees_with_itself_is_not_an_error():
    """Harmless repetition is not the defect; a disagreement is."""
    from molbuilder.siesta.layout import check_rules

    text, struct, cfg = _fdf()
    same = [ln for ln in text.splitlines() if ln.startswith("MeshCutoff")][0]
    assert check_rules(text + "\n" + same, struct, cfg) == []


def test_a_deck_whose_identity_is_not_the_stamped_one_is_refused():
    from molbuilder.siesta.layout import check_rules

    text, struct, cfg = _fdf()
    broken = text.replace("SystemLabel       t", "SystemLabel       other")
    assert [i for i in check_rules(broken, struct, cfg)
            if "SystemLabel" in i.message]


def test_the_atom_count_must_match_the_coordinate_block():
    from molbuilder.siesta.layout import check_rules

    text, struct, cfg = _fdf()
    broken = text.replace("NumberOfAtoms     3", "NumberOfAtoms     5")
    assert [i for i in check_rules(broken, struct, cfg)
            if "NumberOfAtoms" in i.message]


def test_both_engines_now_answer_the_check_question():
    """The seam's check question, answered by both — **asked of the SPEC**.

    It was a member of ``EngineSeam`` until 2026-08-18, when the seam started
    carrying the engine's form instead of finished text: the rules are part of
    what an engine says about its deck, so they ride on the ``DeckSpec`` and
    the seam stopped holding a second copy of the answer
    (`script-preparation.md` § 4.3).
    """
    import numpy as np

    from molbuilder.jobset.prep import _engine_seam
    from molbuilder.structure import Structure

    struct = Structure(elements=["H", "H"],
                       positions=np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]),
                       vacuum=(8.0, 8.0, 8.0))
    for engine in ("siesta", "pyscf"):
        seam = _engine_seam(engine)
        assert not hasattr(seam, "check_rules"), (
            f"{engine}: the seam holds a second copy of the check rules")
        cfg = seam.config_cls()
        spec = seam.spec_for(struct, cfg)
        assert spec.check_rules is not None, engine


# ------------------------------------------------- one writer, every artifact

def test_the_wrapper_keeps_what_a_reader_put_in_their_own_section(tmp_path):
    """**W4** (`script-preparation.md` § 3.2) — one deck, one writer, and the
    writer keeps the reader's own section.

    The wrapper INVITES an edit, so it must not delete one.

    It emits a USER-CUSTOM block -- the one part of a generated file a person
    is meant to touch -- and wrote itself with a plain ``write_text``, so every
    re-prep silently removed whatever they had added. Decks were routed through
    the one writer on 2026-08-17; wrappers were not, and an invitation the next
    run revokes is worse than no invitation at all.
    """
    from molbuilder.jobset.model import Resources
    from molbuilder.runwrap import write_run_wrapper
    from molbuilder.script_emit import BLOCK_USER_CUSTOM, end_marker

    deck = tmp_path / "t.fdf"
    deck.write_text("SystemLabel t\nNumberOfAtoms 1\n", encoding="utf-8")

    wrapper = write_run_wrapper(deck, resources=Resources(mpi_np=1), env="e")
    marker = end_marker(BLOCK_USER_CUSTOM)
    wrapper.write_text(
        wrapper.read_text(encoding="utf-8").replace(
            marker, "export MY_OWN_FLAG=1\n" + marker), encoding="utf-8")

    write_run_wrapper(deck, resources=Resources(mpi_np=1), env="e")
    assert "MY_OWN_FLAG" in wrapper.read_text(encoding="utf-8")
    assert oct(wrapper.stat().st_mode)[-3:] == "755", "still runnable"


def test_the_shared_package_is_named_by_the_engine_that_put_it_there(tmp_path):
    """**W5** (`script-preparation.md` § 4.1) — "nothing" is an answer, and it
    is recorded: PySCF's empty package is the ANSWER, not a gap.

    Not guessed from a suffix in shared code.

    `_shared_for` globbed ``*.psml`` -- a SIESTA fact stated a floor below
    where SIESTA may speak -- so a second engine with data files of its own
    would have shipped none of them, silently.
    """
    from molbuilder.jobset.prep import _engine_seam, _shared_for

    (tmp_path / "C.psml").write_text("x", encoding="utf-8")
    (tmp_path / "H.psml").write_text("x", encoding="utf-8")

    assert _shared_for(tmp_path, _engine_seam("siesta")) == ["C.psml", "H.psml"]
    # PySCF's basis sets ship inside PySCF: an empty package is the ANSWER
    assert _engine_seam("pyscf").shared_package is None
    assert _shared_for(tmp_path, _engine_seam("pyscf")) == []
