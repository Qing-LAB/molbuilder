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

#: The names prep gives this stage's deck: its label and its rung
#: (`runfiles.RunNames`).
NAMES = RunNames("t", "01_coarse")


@pytest.fixture
def deck(tmp_path) -> Path:
    p = tmp_path / "t.fdf"
    p.write_text("# a comment\nSystemLabel  t\n\nMeshCutoff 200.0 Ry\n",
                 encoding="utf-8")
    return p


#: WHICH SETTINGS A STEP'S RECORD LISTS, by the calculation, the step and --
#: for a point of a run of points -- the role it plays (`running-a-job.md`
#: § 1 step 16, `template.md` § 6.4): an item listed and one never listed.
#: A TBtrans step reads TBtrans's settings, never an SCF's (`transport.md`
#: § 6.1b: tbtrans writes no forces); a relaxation reads no TBtrans setting,
#: which its calculation does not carry; a SIESTA vibration's phases are one
#: step's roles, the relaxation's step count in its own phase alone and the
#: displacement in the force constants' (until 2.65.0 every phase listed
#: every role's).
FENCE_ROWS = [
    pytest.param("optimization", "01_coarse", None, "mesh_cutoff",
                 "tbt_k_grid", id="relaxation-lists-no-tbtrans-setting"),
    pytest.param("transport", "05_transmission", None, "tbt_k_grid",
                 "write_forces", id="tbtrans-step-lists-no-scf-setting"),
    pytest.param("vibration", "01_freq", "relaxation", "relax_steps",
                 "fc_displacement",
                 id="a-relaxation-phase-lists-no-displacement"),
    pytest.param("vibration", "01_freq", "force_constants", "fc_displacement",
                 "relax_steps", id="a-force-constant-phase-lists-no-steps"),
]


@pytest.mark.parametrize("calculation,stage,role,listed,never", FENCE_ROWS)
def test_siesta_s_fence_states_the_step_s_items_and_the_deck_as_launched(
        deck, calculation, stage, role, listed, never):
    """The wrapper's block, RUN as the wrapper runs it, read back through the
    one reader both engines' blocks are read by (`model/parse.md` § 5d.3a):
    a row for each keyword-writing catalogue item THIS STEP reads, with its
    catalogue default, and none for an item it does not read -- until 2.64.0
    every rung listed every item of its engine, so a TBtrans step's record
    named SCF settings it never reads -- then the deck the engine is given,
    echoed at launch: the lines libfdf reads, no comment and no blank, and
    read when the block runs, so a deck edited after prep is echoed as
    edited.  SIESTA has not started, so ``asked`` is the deck's and ``used``
    the fdf log's: this block leaves both empty.

    It listed only the items the deck left out until 2026-09-26, so a run
    recorded no default for any item it set -- the record's first column for
    most of a deck."""
    import subprocess
    from molbuilder.deck_record import read_parameters_fence

    import sys
    from molbuilder.runfiles import MONITOR_BUNDLE
    from molbuilder.runwrap import monitor_bundle
    # AS A RUN HAS IT: the monitor bundle beside the deck, read with the
    # job's python (`mb_monitor.pyz deck`), and the run script's log helper.
    (deck.parent / MONITOR_BUNDLE).write_bytes(monitor_bundle())
    block = (f'_mb_py="{sys.executable}"\n_runwrap_log=/dev/null\n'
             '_log() { echo "$2" >&2; }\n'
             + _effective_parameters_block(deck.name, "siesta",
                                           calculation, stage, role=role))

    def run():
        return subprocess.run(["bash", "-c", block], cwd=deck.parent,
                              capture_output=True, text=True,
                              check=True).stdout

    out = run()
    rows = {r["item"]: r for r in read_parameters_fence(out)}
    assert listed in rows and never not in rows, sorted(rows)
    assert rows[listed]["default"] == parameter(listed, "siesta").default
    assert {(r["asked"], r["used"]) for r in rows.values()} == {(None, None)}
    assert "#   MeshCutoff 200.0 Ry" in out
    assert "a comment" not in out and "#   \n" not in out, out
    # AN EDIT AFTER PREP is what the engine reads, so it is what is echoed.
    deck.write_text(deck.read_text().replace("200.0", "350.0"))
    assert "#   MeshCutoff 350.0 Ry" in run()


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
    return (render_deck(spec_for(struct, cfg, names=NAMES, calculation="optimization"), struct, cfg),
            struct, cfg)


@pytest.mark.parametrize("typed", [
    "MeshCutoff 999.0 Ry",
    # A BLOCK typed in a person's USER-CUSTOM zone that the deck already
    # writes: ``fdf_block`` takes the first as ``fdf_locate`` does.
    "%block ChemicalSpeciesLabel\n 1 8 X\n%endblock ChemicalSpeciesLabel",
])
def test_a_keyword_written_twice_is_refused_because_libfdf_takes_the_first(
        typed):
    """The worst kind of wrong: the deck reads as though it says what it
    meant.  ``fdf_locate`` and ``fdf_block`` walk from the top and stop at
    the first match, so a keyword or a block written twice does not conflict
    loudly -- the first silently wins, and the later one, a person's, is the
    one ignored.

    API-level: a refusal the road cannot reach -- prep writes no deck with a
    keyword twice, so the check is handed one here."""
    from molbuilder.siesta.layout import check_rules

    text, struct, cfg = _fdf()
    issues = check_rules(text + "\n" + typed, struct, cfg)
    assert [i for i in issues if "twice" in i.message], issues


def _one_species_row_lost(text: str) -> str:
    lines = text.splitlines()
    opens = next(i for i, ln in enumerate(lines)
                 if ln.lower().startswith("%block chemicalspecieslabel"))
    del lines[opens + 1]
    return "\n".join(lines)


@pytest.mark.parametrize("disagree, where", [
    # SIESTA reads the count and the coordinate block separately.
    (lambda text: text.replace("NumberOfAtoms     3", "NumberOfAtoms     5"),
     "deck.atom_count"),
    # SIESTA stops on a species with no label, at startup, after the queue wait.
    (_one_species_row_lost, "deck.species_index"),
])
def test_two_blocks_of_a_deck_that_disagree_are_refused(disagree, where):
    """A deck whose blocks disagree is refused, and the deck it was made from
    is not.  API-level: a refusal the road cannot reach -- prep writes the two
    together, so the check is handed a deck that disagrees."""
    from molbuilder.siesta.layout import check_rules

    text, struct, cfg = _fdf()
    assert not [i for i in check_rules(text, struct, cfg) if i.where == where]
    assert [i for i in check_rules(disagree(text), struct, cfg)
            if i.where == where]

