"""The effective-parameters record — one fence, both engines.

**What it is for.** When a result looks wrong, three questions get asked: what
does this project recommend, what did this run ask for, and what did the engine
actually do with it.

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

from molbuilder.runwrap import _effective_parameters_block
from molbuilder.script_emit import parameter
from molbuilder.structure import Structure
from molbuilder.runfiles import RunNames

#: The names prep gives this stage's deck: its label, its rung, its layout
#: (`runfiles.RunNames`).
NAMES = RunNames.of('t', "01_coarse", "hierarchical")


@pytest.fixture
def deck(tmp_path) -> Path:
    p = tmp_path / "t.fdf"
    p.write_text("# a comment\nSystemLabel  t\n\nMeshCutoff 200.0 Ry\n",
                 encoding="utf-8")
    return p


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


def test_siesta_s_fence_states_every_item_s_default_and_the_deck_as_launched(
        deck, tmp_path):
    """The wrapper's block, RUN as the wrapper runs it, read back through the
    one reader both engines' blocks are read by (`model/parse.md` § 5d.3a):
    a row for every keyword-writing catalogue item with its catalogue default
    -- generated, so a new item joins with no edit -- and the deck the engine
    is given, echoed at launch.  SIESTA has not started, so ``asked`` is the
    deck's and ``used`` the fdf log's: this block leaves both empty.

    It listed only the items the deck left out until 2026-09-26, so a run
    recorded no default for any item it set -- the record's first column for
    most of a deck."""
    import subprocess
    import molbuilder.template as T
    from molbuilder.script_emit import _catalogue, read_parameters_fence

    out = subprocess.run(["bash", "-c", _effective_parameters_block(deck)],
                         cwd=deck.parent, capture_output=True, text=True,
                         check=True).stdout
    rows = {r["item"]: r for r in read_parameters_fence(out)}
    writing = {i.name for i in T.select(_catalogue(), engine="siesta")
               if parameter(i.name, "siesta").writes}
    assert set(rows) == writing
    assert rows["mesh_cutoff"]["default"] == parameter("mesh_cutoff",
                                                       "siesta").default
    assert {(r["asked"], r["used"]) for r in rows.values()} == {(None, None)}
    assert "#   MeshCutoff 200.0 Ry" in out


# ------------------------------------------------- SIESTA's own check rules

def _fdf(**over):
    """A SIESTA deck as the framework renders it for prep (`spec_for` ->
    `script_emit.render_deck`), with its structure and settings -- for the
    check's refusals, which only a deck prep never writes can show."""
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.script_emit import render_deck
    from molbuilder.siesta.input import spec_for

    struct = Structure(elements=["O", "H", "H"],
                       positions=np.array([[0., 0., 0.],
                                           [0.957, 0., 0.],
                                           [-0.24, 0.927, 0.]]))
    cfg = SiestaConfig(system_label="t", **over)
    return (render_deck(spec_for(struct, cfg, names=NAMES), struct, cfg),
            struct, cfg)


def test_a_keyword_written_twice_is_refused_because_libfdf_takes_the_first():
    """The worst kind of wrong: the deck reads as though it says what it
    meant.  ``fdf_locate`` walks from the top and stops at the first match,
    so a keyword written twice does not conflict loudly -- the first silently
    wins, and the later line is the one ignored.

    API-level: a refusal the road cannot reach -- prep writes no deck with a
    keyword twice, so the check is handed one here."""
    from molbuilder.siesta.layout import check_rules

    text, struct, cfg = _fdf()
    issues = check_rules(text + "\nMeshCutoff 999.0 Ry", struct, cfg)
    assert [i for i in issues if "twice" in i.message], issues


def test_the_atom_count_must_match_the_coordinate_block():
    """A count that disagrees with the coordinate block is refused.

    API-level: a refusal the road cannot reach -- prep writes the count from
    the atoms it places, so the check is handed a deck that disagrees."""
    from molbuilder.siesta.layout import check_rules

    text, struct, cfg = _fdf()
    broken = text.replace("NumberOfAtoms     3", "NumberOfAtoms     5")
    assert [i for i in check_rules(broken, struct, cfg)
            if "NumberOfAtoms" in i.message]

