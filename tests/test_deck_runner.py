"""The step-3 runner and the CHECK gate — `execution/script-preparation.md`.

The framework alone, with **no engine in it**.  The engine here is a stub — three lambdas — so what these tests
exercise is the framework's own promises:

  * the sub-steps run in the contract's order;
  * a value cannot reach a deck without its reason (the engine is handed a
    ``Parameter`` and never a bare value);
  * a section whose parameters all decline contributes no heading;
  * the reader's own section survives a re-render;
  * and **check refuses a deck that does not say what it was meant to say** --
    the gate no other validator can run, because every other one takes
    ``(struct, cfg)`` and never reads the artifact.

The stub names a REAL engine so the catalogue declarations, the notes and the
anchors are real; what is stubbed is the writing, which is the engine's job.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder import deck_record as dr
from molbuilder import script_emit as se
from molbuilder.cell import to_engine
from molbuilder.issues import ValidationError
from molbuilder.structure import Structure
from molbuilder.runfiles import RunNames


def _names(cfg):
    """The names prep gives this stage's deck (`runfiles.RunNames`),
    under the config's own label."""
    label = getattr(cfg, "system_label", None) or cfg.job_name
    return RunNames.of(label, '01_coarse', "hierarchical")


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
        # EVERY spec builder places its atoms and hands the frame over, the
        # stub included -- the renderer refuses a deck without one
        # (`model/structure-periodicity.md` § 6.0, check 3).
        engine_frame=to_engine(_struct()),
    )
    kw.update(over)
    return se.DeckSpec(**kw)


# --------------------------------------------------------------- render --

def test_the_engine_never_sees_a_bare_value_so_the_reason_travels_with_it():
    """**W2** (`script-preparation.md` § 3.2) — a value is written together
    with the reason it holds.

    A value and its reason are ONE act, not a value plus a habit.

    The stub's ``line`` receives a ``Parameter`` and can only reach the number
    through it -- so the catalogue's note is written above the value by the
    framework, and an engine cannot emit one without the other.
    """
    out = se.render_deck(_spec(), _struct(), _Cfg())
    assert "MeshCutoff 250.0" in out.text
    note = se.parameter("mesh_cutoff", "siesta").note()
    assert note, "the catalogue declares a note for mesh_cutoff"
    body = [ln for ln in note if ln.strip().startswith("#")][0]
    assert body in out.text


def test_a_section_whose_parameters_all_decline_gets_no_heading():
    """A heading over nothing is the block lying — the BENCH-MARKS rule."""
    out = se.render_deck(_spec(), _struct(), _Cfg())
    assert "--- Grid ---" in out.text
    assert "--- Relaxation ---" not in out.text
    assert not any("MD.Steps" in l for l in out.emitted)


def test_the_runner_reports_what_it_emitted_so_check_can_close_the_loop():
    """It reports the LINES it wrote, not the keywords.

    A keyword cannot tell a setting from a READ of that setting; a line is the
    assignment itself (`script-preparation.md` § 4.3).
    """
    out = se.render_deck(_spec(), _struct(), _Cfg())
    assert any(l.startswith("MeshCutoff") for l in out.emitted)
    assert any(l.startswith("MaxSCFIterations") for l in out.emitted)
    assert all(l in out.text for l in out.emitted), (
        "a reported line that is not in the deck makes the gate's input a lie")


def test_the_record_sits_below_the_banner_and_the_science_above_it():
    out = se.render_deck(_spec(), _struct(), _Cfg())
    banner = out.text.index("MOLBUILDER RECORD")
    assert out.text.index("MeshCutoff 250.0") < banner
    assert banner < out.text.index(dr.begin_marker(dr.BLOCK_PROVENANCE))


def test_verbose_false_drops_the_notes_and_keeps_the_values():
    out = se.render_deck(_spec(), _struct(), _Cfg(), verbose=False)
    assert "MeshCutoff 250.0" in out.text
    assert se.parameter("mesh_cutoff", "siesta").note()[1] not in out.text


# ---------------------------------------------------------------- check --

def test_rendering_touches_no_disk(tmp_path, monkeypatch):
    """**W7**: floor 3 returns TEXT.  It does not touch the disk.

    ``render_deck`` hands back a deck; the only thing that
    writes one is ``write_script`` (W4).  That is what keeps *one deck, one
    writer* true no matter which route rendered it -- a renderer that also
    wrote would be a second writer with no USER-CUSTOM merge and no gate.

    Run in an empty directory so a stray write has nowhere to hide.
    """
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.rglob("*"))
    out = se.render_deck(_spec(), _struct(), _Cfg())
    assert isinstance(out, str) and out.strip(), "no text came back"
    assert set(tmp_path.rglob("*")) == before, (
        f"rendering wrote to disk: "
        f"{sorted(p.name for p in set(tmp_path.rglob('*')) - before)}")


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


def test_a_parameter_that_writes_a_PAIR_still_closes_the_loop(tmp_path):
    """One `Parameter` may legitimately emit SEVERAL lines, and each of them
    is evidence.

    A fixed total spin needs ``Spin.Fix`` and ``Spin.Total`` together, and
    SIESTA's free-energy section is titled *"a PAIR: the value + its switch"*
    for the same reason -- ``line`` returns ``str | None`` and nothing says
    one line.  Comparing the emission WHOLE meant a two-line answer could
    never equal any member of a set of single lines, so the gate refused
    every spin-polarized SIESTA deck while the deck itself was correct
    (2026-08-19).

    Both halves are asserted: the pair passes when the file has it, and the
    gate still names the half a writer bug drops.
    """
    spec = _spec(line=lambda p: ("Spin.Fix          .true.\n"
                                 "Spin.Total        2.0")
                 if p.name == "relax_steps" else _line(p))
    struct, cfg = _struct(), _Cfg()
    out = se.render_deck(spec, struct, cfg)
    good = se.write_script(tmp_path / "pair.fdf", out.text)
    assert se.check_deck(good, spec, out, struct, cfg) == [], \
        "a correct deck whose parameter wrote a pair was refused"

    # and the loop is really closed over BOTH halves, not just the first
    half = out.text.replace("Spin.Total        2.0", "")
    p2 = se.write_script(tmp_path / "half.fdf", half)
    issues = se.check_deck(p2, spec, out, struct, cfg)
    assert [i for i in issues
            if i.severity == "error" and "Spin.Total" in i.message], issues


def test_check_catches_a_missing_reader_section(tmp_path):
    spec, struct, cfg = _spec(), _struct(), _Cfg()
    out = se.render_deck(spec, struct, cfg)
    broken = out.text.replace(dr.begin_marker(dr.BLOCK_USER_CUSTOM), "")
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
    """**W4** — one deck is written by one writer, and the writer keeps what
    the reader put in their own section (`script-preparation.md` § 3.2)."""
    path = tmp_path / "keep.fdf"
    se.prepare_deck(_spec(), _struct(), _Cfg(), path)
    edited = path.read_text(encoding="utf-8").replace(
        dr.end_marker(dr.BLOCK_USER_CUSTOM),
        "MyOwnKeyword 7\n" + dr.end_marker(dr.BLOCK_USER_CUSTOM))
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
    from molbuilder.jobset.engines import engine_seam
    from molbuilder.structure import Structure

    seam = engine_seam(engine)
    label = {"siesta": {"system_label": "t"}, "pyscf": {"job_name": "t"}}[engine]
    cfg = dataclasses.replace(seam.config_cls(**label), **over)
    struct = Structure(elements=["O", "H", "H"],
                       positions=np.array([[0., 0., 0.], [0.957, 0., 0.],
                                           [-0.24, 0.927, 0.]]),
                       vacuum=(8., 8., 8.))
    return seam, struct, cfg


@pytest.mark.parametrize("engine", _ENGINES)
def test_a_real_engine_reports_the_keywords_its_deck_writes(engine):
    """**The loop-closing input, on a production deck.**

    `check_deck` asks whether every keyword the parameters step says it
    wrote survived into the file.  With an empty list it asks nothing and
    passes -- which is exactly what it did for SIESTA on every route until
    2026-08-19.
    """
    seam, struct, cfg = _real(engine)
    deck = se.render_deck(seam.spec_for(struct, cfg, names=_names(cfg)),
                          struct, cfg)
    assert len(deck.emitted) >= 10, (
        f"{engine}: the deck reports {len(deck.emitted)} written lines for "
        f"{len(str(deck).splitlines())} lines. An empty or near-empty list "
        f"makes the check gate vacuous.")
    present = set(str(deck).splitlines())
    for line in deck.emitted:
        assert line in present, (
            f"{engine}: reported writing {line!r} and the rendered deck does "
            f"not contain that line -- the report is the gate's only input")
