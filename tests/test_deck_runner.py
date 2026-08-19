"""The step-3 runner and the CHECK gate — `execution/script-preparation.md`.

Phase 1 of `archive/2026-08-18-preparation-backend-plan.md`: the framework alone, with **no
engine in it**.  The engine here is a stub — three lambdas — so what these tests
exercise is the framework's own promises:

  * the sub-steps run in the contract's order;
  * a value cannot reach a deck without its reason (the engine is handed a
    ``Parameter`` and never a bare value);
  * a section whose parameters all decline contributes no heading;
  * the reader's own section survives a re-render;
  * and **check refuses a deck that does not say what it was meant to say** --
    the gate that no validator in this tree could run before, because every
    other one takes ``(struct, cfg)`` and never reads the artifact.

The stub names a REAL engine so the catalogue declarations, the notes and the
anchors are real; what is stubbed is the writing, which is the engine's job.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder import script_emit as se
from molbuilder.issues import ValidationError
from molbuilder.structure import Structure


def _struct() -> Structure:
    return Structure(elements=["O", "H", "H"],
                     positions=np.array([[0., 0., 0.],
                                         [0.957, 0., 0.],
                                         [-0.24, 0.927, 0.]]))


class _Cfg:
    """The two catalogue fields the stub layout asks for."""
    mesh_cutoff = 250.0
    max_scf_iter = 40
    relax_steps = None          # declines to emit -- see `line` below


def _line(p):
    """Door 2: one parameter -> one line, or None for *not emitted here*."""
    if p.value is None:
        return None
    return f"{p.declaration.anchor} {p.value}"


def _spec(**over) -> se.DeckSpec:
    kw = dict(
        engine="siesta",
        # The structure is a BLOCK IN THE LAYOUT, first, because that is where
        # this deck puts it -- not a separate door appended by the framework.
        layout=(se.Block("Atoms",
                         lambda s, c: "# --- Atoms ---\nNumberOfAtoms 3"),
                se.Section("Grid", ("mesh_cutoff",)),
                se.Section("SCF", ("max_scf_iter",)),
                se.Section("Relaxation", ("relax_steps",))),
        line=_line,
        created_by="stub",
    )
    kw.update(over)
    return se.DeckSpec(**kw)


# --------------------------------------------------------------- render --

def test_the_engine_never_sees_a_bare_value_so_the_reason_travels_with_it():
    """A value and its reason are ONE act, not a value plus a habit.

    The stub's ``line`` receives a ``Parameter`` and can only reach the number
    through it -- so the catalogue's note is written above the value by the
    framework, and an engine cannot emit one without the other.
    """
    out = se.render_deck(_spec(), _struct(), _Cfg())
    assert "MeshCutoff 250.0" in out.text
    # the note came from the catalogue, not from the stub
    assert "MeshCutoff" in out.text
    note = se.parameter("mesh_cutoff", "siesta").note()
    assert note, "the catalogue declares a note for mesh_cutoff"
    body = [ln for ln in note if ln.strip().startswith("#")][0]
    assert body in out.text


def test_a_section_whose_parameters_all_decline_gets_no_heading():
    """A heading over nothing is the block lying — the BENCH-MARKS rule."""
    out = se.render_deck(_spec(), _struct(), _Cfg())
    assert "--- Grid ---" in out.text
    assert "--- Relaxation ---" not in out.text
    assert "MD.Steps" not in out.emitted


def test_the_runner_reports_what_it_emitted_so_check_can_close_the_loop():
    out = se.render_deck(_spec(), _struct(), _Cfg())
    assert "MeshCutoff" in out.emitted
    assert "MaxSCFIterations" in out.emitted


def test_the_record_sits_below_the_banner_and_the_science_above_it():
    out = se.render_deck(_spec(), _struct(), _Cfg())
    banner = out.text.index("MOLBUILDER RECORD")
    assert out.text.index("MeshCutoff 250.0") < banner
    assert banner < out.text.index(se.begin_marker(se.BLOCK_PROVENANCE))


def test_verbose_false_drops_the_notes_and_keeps_the_values():
    out = se.render_deck(_spec(), _struct(), _Cfg(), verbose=False)
    assert "MeshCutoff 250.0" in out.text
    assert se.parameter("mesh_cutoff", "siesta").note()[1] not in out.text


# ---------------------------------------------------------------- check --

def test_check_passes_a_deck_the_runner_itself_wrote(tmp_path):
    spec, struct, cfg = _spec(), _struct(), _Cfg()
    out = se.render_deck(spec, struct, cfg)
    p = se.write_script(tmp_path / "ok.fdf", out.text)
    assert se.check_deck(p, spec, out, struct, cfg) == []


def test_check_catches_a_value_that_never_reached_the_file(tmp_path):
    """The writer-bug class, which a config gate structurally cannot see.

    `validate` runs on ``(struct, cfg)`` and would pass this deck happily: the
    configuration is sound.  What is wrong is the FILE.
    """
    spec, struct, cfg = _spec(), _struct(), _Cfg()
    out = se.render_deck(spec, struct, cfg)
    broken = out.text.replace("MeshCutoff 250.0", "")
    p = se.write_script(tmp_path / "lost.fdf", broken)
    issues = se.check_deck(p, spec, out, struct, cfg)
    assert [i for i in issues
            if i.severity == "error" and "MeshCutoff" in i.message], issues


def test_check_catches_a_missing_reader_section(tmp_path):
    spec, struct, cfg = _spec(), _struct(), _Cfg()
    out = se.render_deck(spec, struct, cfg)
    broken = out.text.replace(se.begin_marker(se.BLOCK_USER_CUSTOM), "")
    p = tmp_path / "nouser.fdf"
    p.write_text(broken, encoding="utf-8")
    issues = se.check_deck(p, spec, out, struct, cfg)
    assert [i for i in issues if i.severity == "error"], issues


def test_check_runs_the_engines_own_rules_too(tmp_path):
    """The engine's answer to *what must a finished deck of mine satisfy?*"""
    from molbuilder.issues import Issue

    def rules(text, struct, cfg):
        return ([] if "NumberOfAtoms" in text
                else [Issue("error", "no atom count", where="deck.atoms")])

    spec = _spec(check_rules=rules,
                 layout=(se.Block("Atoms", lambda s, c: "# nothing"),))
    struct, cfg = _struct(), _Cfg()
    out = se.render_deck(spec, struct, cfg)
    p = se.write_script(tmp_path / "noatoms.fdf", out.text)
    assert [i.message for i in se.check_deck(p, spec, out, struct, cfg)] == \
        ["no atom count"]


def test_check_reads_the_file_and_not_the_string_it_was_handed(tmp_path):
    """`write_script` merges the reader's section, so the file is the artifact.

    Here the file on disk is made to disagree with the rendered string.  A gate
    that trusted the string would pass it; this one opens the file.
    """
    spec, struct, cfg = _spec(), _struct(), _Cfg()
    out = se.render_deck(spec, struct, cfg)
    p = tmp_path / "edited.fdf"
    p.write_text(out.text.replace("MaxSCFIterations 40", "# gone"),
                 encoding="utf-8")
    assert [i for i in se.check_deck(p, spec, out, struct, cfg)
            if "MaxSCFIterations" in i.message]


# ----------------------------------------------------------- the spine ---

def test_prepare_deck_runs_the_sub_steps_in_order(tmp_path):
    p = se.prepare_deck(_spec(), _struct(), _Cfg(), tmp_path / "run.fdf")
    text = p.read_text(encoding="utf-8")
    assert text.index("NumberOfAtoms 3") < text.index("MeshCutoff 250.0")
    assert text.index("MeshCutoff 250.0") < text.index("MOLBUILDER RECORD")


def test_prepare_deck_refuses_a_broken_engine_rather_than_shipping_it(tmp_path):
    """The gate is only worth having if it stops the run."""
    from molbuilder.issues import Issue

    spec = _spec(check_rules=lambda t, s, c: [
        Issue("error", "this deck is wrong", where="deck.stub")])
    with pytest.raises(ValidationError):
        se.prepare_deck(spec, _struct(), _Cfg(), tmp_path / "bad.fdf")


def test_prepare_deck_keeps_what_a_reader_put_in_their_own_section(tmp_path):
    path = tmp_path / "keep.fdf"
    se.prepare_deck(_spec(), _struct(), _Cfg(), path)
    edited = path.read_text(encoding="utf-8").replace(
        se.end_marker(se.BLOCK_USER_CUSTOM),
        "MyOwnKeyword 7\n" + se.end_marker(se.BLOCK_USER_CUSTOM))
    path.write_text(edited, encoding="utf-8")
    se.prepare_deck(_spec(), _struct(), _Cfg(), path)
    assert "MyOwnKeyword 7" in path.read_text(encoding="utf-8")


# --------------------------------------------------------------------- #
#  The REAL engines' forms — the half a stub cannot prove               #
# --------------------------------------------------------------------- #
#
# Everything above tests the framework against a stub whose layout is already
# the table the contract describes.  That proves the framework and nothing
# about the engines, and the gap was real: SIESTA's whole 728-line deck was one
# `Block`, so `render_deck` collected zero keywords, the loop-closing rule ran
# on an empty list and passed, and `spec.layout` answered *what is in this
# deck?* with "the deck".  PySCF's geometry section was nested inside its
# optimise branch for the same reason.  These ask the real forms.

_ENGINES = ("siesta", "pyscf")


def _real(engine, **over):
    """A seam, a config and a structure for one engine — through `prep`'s own
    door, so this asks what the production route asks."""
    import dataclasses
    from molbuilder.jobset.prep import _engine_seam
    from molbuilder.structure import Structure

    seam = _engine_seam(engine)
    label = {"siesta": {"system_label": "t"}, "pyscf": {"job_name": "t"}}[engine]
    cfg = dataclasses.replace(seam.config_cls(**label), **over)
    struct = Structure(elements=["O", "H", "H"],
                       positions=np.array([[0., 0., 0.], [0.957, 0., 0.],
                                           [-0.24, 0.927, 0.]]),
                       vacuum=(8., 8., 8.))
    return seam, struct, cfg


@pytest.mark.parametrize("engine", _ENGINES)
def test_a_real_engines_layout_is_a_table_and_not_one_opaque_block(engine):
    """**§ 4.1: three rows say `spec.layout`, and that is the shape of a deck.**

    A layout of one `Block` satisfies the type and answers none of the
    question.  What makes the form READABLE — § 4.3's *"a function can only be
    called; a form can be read"* — is that the settings are `Section`s in it.
    """
    seam, struct, cfg = _real(engine)
    layout = seam.spec_for(struct, cfg).layout
    sections = [m for m in layout if isinstance(m, se.Section)]
    assert len(sections) >= 3, (
        f"{engine}: spec.layout has {len(sections)} Section(s) in "
        f"{len(layout)} members. A deck whose settings live inside a Block is "
        f"a deck the framework cannot read, and the check gate then has "
        f"nothing to compare the file against.")
    assert all(m.title for m in sections), (
        f"{engine}: a Section with no title -- the layout stops saying what "
        f"that part of the deck IS")


@pytest.mark.parametrize("engine", _ENGINES)
def test_a_real_engine_reports_the_keywords_its_deck_writes(engine):
    """**The loop-closing input, on a production deck.**

    `check_written` asks whether every keyword the parameters step says it
    wrote survived into the file.  With an empty list it asks nothing and
    passes -- which is exactly what it did for SIESTA on every route until
    2026-08-19.
    """
    seam, struct, cfg = _real(engine)
    deck = se.render_deck(seam.spec_for(struct, cfg), struct, cfg)
    assert len(deck.emitted) >= 10, (
        f"{engine}: the deck reports {len(deck.emitted)} written keywords for "
        f"{len(str(deck).splitlines())} lines. An empty or near-empty list "
        f"makes the check gate vacuous.")
    text = str(deck)
    for key in deck.emitted:
        assert se._mentions_keyword(text, key), (
            f"{engine}: reported writing {key!r} and the rendered deck does "
            f"not contain it -- the report is the gate's only input")


@pytest.mark.parametrize("engine", _ENGINES)
def test_the_gate_names_a_keyword_a_writer_bug_dropped(engine, tmp_path):
    """**The whole point, end to end**: mangle one keyword on its way to disk
    and the gate must refuse and NAME it.

    This is the defect class no other validator in the tree can see -- every
    other one takes ``(struct, cfg)`` and runs before the text exists.
    """
    from molbuilder.issues import ValidationError

    seam, struct, cfg = _real(engine)
    spec = seam.spec_for(struct, cfg)
    victim = se.render_deck(spec, struct, cfg).emitted[0]

    real = se.write_script
    try:
        se.write_script = lambda p, t: real(p, t.replace(victim, "MB_TYPO"))
        with pytest.raises(ValidationError) as caught:
            se.prepare_deck(spec, struct, cfg, tmp_path / f"t{seam.suffix}")
    finally:
        se.write_script = real
    assert victim in str(caught.value), (
        f"{engine}: the gate refused but did not name {victim!r}")


@pytest.mark.parametrize("engine", _ENGINES)
def test_a_conditional_section_is_omitted_from_the_layout_not_hidden_in_a_block(
        engine):
    """**A branch is a reason to OMIT a member, not to hide one.**

    Both engines nested a section inside a branch on the reasoning that a
    section which only sometimes appears could not be a top-level member.
    ``spec_for`` holds the config, so it can simply leave it out -- and then
    the layout still says what the deck contains, for both answers.
    """
    off = {"siesta": {"relax_type": "none"}, "pyscf": {"optimize": False}}[engine]
    titles = lambda **o: [m.title for m in
                          _real(engine, **o)[0].spec_for(*_real(engine, **o)[1:]).layout]
    with_it, without = titles(), titles(**off)
    dropped = set(with_it) - set(without)
    assert dropped, (
        f"{engine}: turning the geometry loop off changed no layout member, so "
        f"either the section is unconditional or it is hidden inside a Block")
    assert len(without) < len(with_it)
