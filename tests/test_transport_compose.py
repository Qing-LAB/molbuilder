"""The transport citation, composed from a relaxation run of molbuilder's own
(`engines/transport.md` § 3.1, decision 7): what qualifies, what is refused
by name, and what this calculation's own copy carries.

Every cited run here is made on the road -- a labelled junction described
with `jobset init`, prepared and launched, the suite's stand-in engine
leaving the `.XV` SIESTA leaves -- so the folder holds what molbuilder
writes and never a file laid by a test (`process/testing.md` § 6).  Until
2026-10-08 these cases cited hand-written `.xyz + .molstruct.json` pairs,
the citation form that went with decision 7.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.transport.compose import (ComposeError, classify_citation,
                                          compose_junction)
from molbuilder.transport.sort import (REGION_BRIDGE, REGION_LEFT_ELECTRODE,
                                       REGION_RIGHT_ELECTRODE)

#: Six 2.5 Å gold layers a side (12.5 Å span -- over the wizard's 12 Å lead
#: floor), a four-atom bridge between, in a box whose c closes the leads'
#: boundary to one layer spacing (§ 6.1c).
_LAYERS_L = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5]
_BRIDGE = [("S", 15.0), ("C", 16.4), ("C", 17.8), ("S", 19.2)]
_LAYERS_R = [22.0, 24.5, 27.0, 29.5, 32.0, 34.5]


def _junction(*, frozen: bool = True) -> dict:
    """The junction as `describe_calculation` takes it: the structure a
    person built on the Molbuilder tab, its leads labelled and -- unless a
    case says otherwise -- held still for the relaxation."""
    elements, zs, labels = [], [], []
    for z in _LAYERS_L:
        elements.append("Au"); zs.append(z); labels.append(REGION_LEFT_ELECTRODE)
    for el, z in _BRIDGE:
        elements.append(el); zs.append(z); labels.append(REGION_BRIDGE)
    for z in _LAYERS_R:
        elements.append("Au"); zs.append(z); labels.append(REGION_RIGHT_ELECTRODE)
    regions: dict = {}
    for i, lab in enumerate(labels):
        regions.setdefault(lab, []).append(i)
    if frozen:
        regions["frozen_atoms"] = [i for i, lab in enumerate(labels)
                                   if lab != REGION_BRIDGE]
    return {"elements": elements,
            "positions": [[1.0, 1.0, z] for z in zs],
            "regions": regions,
            "cell": [8.0, 8.0, 37.0],
            "axis_kind": ["periodic", "periodic", "transport"]}


def _relaxed_on_the_road(tmp_path, monkeypatch, *, frozen: bool = True):
    """A finished relaxation run of the junction -- `jobset init`, `prep`,
    `launch` on the stand-in engine -- and its citation: ``(tree root,
    citation)``."""
    from conftest import write_machine_record
    from support.road import describe_calculation, jobset
    write_machine_record()
    bundle = describe_calculation(tmp_path, monkeypatch, name="J",
                                  structure=_junction(frozen=frozen),
                                  stage_strategy="")
    got = jobset("prep", "task", "--stage", "coarse", "--bundle", bundle,
                 "--target", "this")
    assert got.exit_code == 0, got.output
    monkeypatch.setenv("MB_STAND_IN_LEAVES_XV", "1")
    got = jobset("launch", "task", "--stage", "coarse", "--mode", "direct",
                 "--yes", "--bundle", bundle)
    assert got.exit_code == 0, got.output
    return tmp_path / "projects", "P/optimization/J/01_coarse/run-0"


def test_a_finished_relaxation_run_composes_and_says_how_it_ended(
        tmp_path, monkeypatch):
    """§ 3.1: the citation is a relaxation run of molbuilder's own that
    finished -- its deck and `.XV`, its run record -- and the composed copy
    carries how it ended and what it converged, so a geometry is cited
    knowingly.  The stand-in moved nothing, so the frozen leads pass the
    gate and the composed leads are the six-layer blocks."""
    root, cite = _relaxed_on_the_road(tmp_path, monkeypatch)
    cited = classify_citation(root / cite)
    assert cited.concluded and cited.exit_code == 0
    out = compose_junction(cite, tree_root=root)
    assert out.deck_text and "AtomicCoordinatesAndAtomicSpecies" in out.deck_text
    assert out.provenance["evidence"] == cited.concluded
    assert out.provenance["relaxation"]["exit_code"] == 0
    assert len(out.electrode_left.elements) == len(_LAYERS_L)
    assert out.provenance["swap_electrodes"] is False


def test_a_saved_structure_is_not_a_citation_and_the_refusal_names_the_road(
        tmp_path, monkeypatch):
    """Decision 7: a structure saved from the Molbuilder tab -- an
    `.xyz + .molstruct.json` pair -- is no citation: it brings no
    pseudopotentials and no record of a run.  The refusal states the whole
    condition and the road to a citable run."""
    root, _cite = _relaxed_on_the_road(tmp_path, monkeypatch)
    with pytest.raises(ComposeError) as e:
        compose_junction("P/structure", tree_root=root)
    msg = str(e.value)
    assert "no .fdf and no .XV" in msg and "relaxation run of molbuilder's own" in msg


def test_a_run_whose_leads_were_not_held_is_refused_by_name(
        tmp_path, monkeypatch):
    """§ 3's lead gate at compose: a lead must have come through the
    relaxation as frozen bulk.  A junction relaxed with its leads free is
    refused naming what to do, however its run ended."""
    root, cite = _relaxed_on_the_road(tmp_path, monkeypatch, frozen=False)
    with pytest.raises(ComposeError) as e:
        compose_junction(cite, tree_root=root)
    msg = str(e.value)
    assert "NOT FROZEN" in msg and "freeze them" in msg, msg


def test_the_rename_is_the_calculations_own_copy_and_the_cited_run_is_untouched(
        tmp_path, monkeypatch):
    """`transport.md` § 4 (user, 2026-10-04): the electrode rename is stated
    in the description (`swap_electrodes: true`) and applied to the
    calculation's own copy of the junction when it is composed; the cited
    run's files are read, never written, and a record composed with the
    other choice does not serve the description.

    Silent before this: the rename rewrote the label block inside the cited
    run's finished attempt, so every later citation of that run -- another
    calculation's included -- read the labels the other way round.
    """
    from molbuilder.transport.compose import (load_compose_record,
                                              write_compose_record)
    root, cite = _relaxed_on_the_road(tmp_path, monkeypatch)
    d = root / cite
    before = {p.name: p.read_bytes() for p in d.iterdir() if p.is_file()}

    out = compose_junction(cite, tree_root=root, swap_electrodes=True)
    regions = out.sorted.structure.regions
    zs = np.asarray(out.sorted.structure.positions)[:, 2]
    # The high-z block is now the LEFT electrode: traded names, nothing
    # else -- no coordinate moved.
    assert min(zs[regions[REGION_LEFT_ELECTRODE]]) > max(
        zs[regions[REGION_RIGHT_ELECTRODE]])
    assert out.provenance["swap_electrodes"] is True
    assert {p.name: p.read_bytes() for p in d.iterdir() if p.is_file()} == before, (
        "the cited run's files were written")

    # The record answers for the choice it was composed with.
    rec = tmp_path / "calc"
    rec.mkdir()
    write_compose_record(rec, out)
    why: list = []
    assert load_compose_record(rec, citation=cite, tree_root=root,
                               why=why, swap_electrodes=False) is None
    assert "swap_electrodes" in why[-1]
    assert load_compose_record(rec, citation=cite, tree_root=root,
                               swap_electrodes=True) is not None
