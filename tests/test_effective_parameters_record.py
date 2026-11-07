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

import re
import subprocess
import sys

import pytest

from molbuilder.deck_record import (BLOCK_PARAMETERS, BLOCK_USER_CUSTOM,
                                    begin_marker, end_marker,
                                    read_parameters_fence)
from molbuilder.script_emit import parameter
from support.road import (H2, describe_calculation, jobset, prep_groups,
                          road_structure)

#: WHICH SETTINGS A STEP'S RECORD LISTS, by the calculation, the step and --
#: for a point of a run of points -- the folder of the role it plays
#: (`running-a-job.md` § 1 step 16, `template.md` § 6.4): an item listed and one
#: never listed.  A TBtrans step reads TBtrans's settings, never an SCF's
#: (`transport.md` § 6.1b: tbtrans writes no forces); a relaxation reads no
#: TBtrans setting, which its calculation does not carry; a SIESTA vibration's
#: phases are one step's roles, the relaxation's step count in its own phase
#: alone and the displacement in the force constants' (until 2.65.0 every phase
#: listed every role's).
FENCE_ROWS = [
    pytest.param("optimization", "01_coarse", "mesh_cutoff",
                 "tbt_k_grid", id="relaxation-lists-no-tbtrans-setting"),
    pytest.param("vibration", "01_freq/relax", "relax_steps",
                 "fc_displacement",
                 id="a-relaxation-phase-lists-no-displacement"),
    pytest.param("vibration", "01_freq/fc", "fc_displacement",
                 "relax_steps", id="a-force-constant-phase-lists-no-steps"),
]


def _prepared(tmp_path, monkeypatch, calculation="optimization"):
    """The road's H2 described -- one stage, a relaxation -- and prepared: the
    calculation."""
    bundle = describe_calculation(
        tmp_path, monkeypatch, calculation=calculation, stage_strategy="")
    prep_groups(bundle, "1")
    return bundle


@pytest.mark.parametrize("calculation,folder,listed,never", FENCE_ROWS)
def test_siesta_s_fence_states_the_step_s_items_and_the_deck_as_launched(
        tmp_path, monkeypatch, calculation, folder, listed, never):
    """The block of the run script prep wrote, RUN as the script runs it in the
    folder prep wrote, the monitor bundle beside the deck, and read back through
    the one reader both engines' blocks are read by (`model/parse.md` § 5d.3a):
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
    most of a deck.

    Only this block runs, so no engine starts."""
    the_fence_holds(_prepared(tmp_path, monkeypatch, calculation), folder,
                    listed, never)


def the_fence_holds(bundle, folder, listed, never) -> None:
    """The block of ``folder``'s run script, run and read back, as the test
    above says: its rows name ``listed`` and not ``never``."""
    (script,) = (bundle / folder).glob("*.run.sh")
    text = script.read_text()
    begin = f'echo "{begin_marker(BLOCK_PARAMETERS)}"'
    end = f'echo "{end_marker(BLOCK_PARAMETERS)}"'
    block = text[text.index(begin):text.index(end) + len(end)]

    def run():
        return subprocess.run(
            ["bash", "-c", f'_mb_py="{sys.executable}"\n'
                           '_runwrap_log=/dev/null\n'
                           '_log() { echo "$2" >&2; }\n' + block],
            cwd=script.parent, capture_output=True, text=True,
            check=True).stdout

    out = run()
    rows = {r["item"]: r for r in read_parameters_fence(out)}
    assert listed in rows and never not in rows, sorted(rows)
    assert rows[listed]["default"] == parameter(listed, "siesta").default
    assert {(r["asked"], r["used"]) for r in rows.values()} == {(None, None)}
    echoed = [ln for ln in out.splitlines() if ln.startswith("#   ")
              and not ln.startswith(("#   [", "#   each row"))]
    label = bundle.name
    assert re.search(rf"^#   SystemLabel\s+{label}$", out, re.M), out
    assert not [ln for ln in echoed if not ln[4:].strip()
                or ln[4:].lstrip().startswith("#")], echoed
    # AN EDIT AFTER PREP is what the engine reads, so it is what is echoed.
    deck = script.parent / script.name.replace(".run.sh", ".fdf")
    deck.write_text(re.sub(rf"^(SystemLabel\s+){label}$", r"\1edited",
                           deck.read_text(), flags=re.M))
    assert re.search(r"^#   SystemLabel\s+edited$", run(), re.M)


# ------------------------------------------------- SIESTA's own check rules

@pytest.mark.parametrize("typed", [
    "MeshCutoff 999.0 Ry",
    "%block ChemicalSpeciesLabel\n 1 8 X\n%endblock ChemicalSpeciesLabel",
], ids=["a keyword", "a block"])
def test_a_keyword_written_twice_is_refused_because_libfdf_takes_the_first(
        typed, tmp_path, monkeypatch):
    """The worst kind of wrong: the deck reads as though it says what it
    meant.  ``fdf_locate`` and ``fdf_block`` walk from the top and stop at
    the first match, so a keyword or a block written twice does not conflict
    loudly -- the first silently wins, and the later one, a person's, is the
    one ignored.

    What a person types in the deck's own section, which prep keeps
    (`engines/template.md` § 9.2), so the next prep's check meets it."""
    from molbuilder.jobset.naming import stage_home
    from molbuilder.task import of as task_of
    bundle = _prepared(tmp_path, monkeypatch)
    home = stage_home(bundle, task_of(bundle), "coarse")
    deck = home.dir / home.names.name(".fdf")
    closes = end_marker(BLOCK_USER_CUSTOM)
    deck.write_text(deck.read_text().replace(closes, typed + "\n" + closes))
    r = jobset("prep", "task", "--group", "1", "--bundle", bundle,
               "--target", "this")
    assert r.exit_code != 0, r.output
    assert "[deck.duplicate_keyword]" in r.output and "twice" in r.output, \
        r.output


def _one_species_row_lost(text: str) -> str:
    lines = text.splitlines()
    opens = next(i for i, ln in enumerate(lines)
                 if ln.lower().startswith("%block chemicalspecieslabel"))
    del lines[opens + 1]
    return "\n".join(lines)


@pytest.mark.parametrize("disagree, where", [
    # SIESTA reads the count and the coordinate block separately.
    (lambda text: re.sub(r"(?m)^(NumberOfAtoms\s+)2$", r"\g<1>5", text),
     "deck.atom_count"),
    # SIESTA stops on a species with no label, at startup, after the queue wait.
    (_one_species_row_lost, "deck.species_index"),
], ids=["the atom count", "a species row"])
def test_two_blocks_of_a_deck_that_disagree_are_refused(
        disagree, where, tmp_path, monkeypatch):
    """A deck whose blocks disagree is refused, and the deck prep wrote is not.
    API-level: a refusal the road cannot reach -- prep writes the two together,
    and what a person types is the keyword-written-twice refusal above -- so the
    check is handed a deck that disagrees."""
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.siesta.layout import check_rules
    from support.road import stage_file

    bundle = _prepared(tmp_path, monkeypatch)
    text = stage_file(bundle, "coarse", ".fdf").read_text()
    struct, cfg = road_structure(H2), SiestaConfig(system_label=bundle.name)
    assert not [i for i in check_rules(text, struct, cfg) if i.where == where]
    assert [i for i in check_rules(disagree(text), struct, cfg)
            if i.where == where]
