"""The transport citation (`engines/transport.md` § 3.1, decision 7): what
is refused by name before any run is read.

What a citation of a finished relaxation composes -- the leads it cuts, how
the run ended, the electrode rename on this calculation's own copy -- reads
the `.XV` SIESTA wrote, so it is asserted on the junction the end-to-end
pass relaxes with the engine (`tests/test_transport_on_a_real_junction_e2e.py`,
plan § 5y).  The layer positions below are the ones `support.junction`
builds its junction from.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.transport.compose import ComposeError, compose_junction
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


def test_a_saved_structure_is_not_a_citation_and_the_refusal_names_the_road(
        tmp_path):
    """Decision 7: a structure saved from the Molbuilder tab -- an
    `.xyz + .molstruct.json` pair -- is no citation: it brings no
    pseudopotentials and no record of a run.  The refusal states the whole
    condition and the road to a citable run."""
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec
    j = _junction()
    StructureCodec().write(
        Structure(elements=j["elements"],
                  positions=np.asarray(j["positions"], dtype=float),
                  regions=j["regions"], cell=np.diag(j["cell"]),
                  axis_kind=tuple(j["axis_kind"])),
        tmp_path / "P" / "structure" / "junction.xyz")
    with pytest.raises(ComposeError) as e:
        compose_junction("P/structure", tree_root=tmp_path)
    msg = str(e.value)
    assert "no .fdf and no .XV" in msg and "relaxation run of molbuilder's own" in msg


def test_a_lead_that_was_not_held_is_refused_by_name():
    """§ 3's lead gate, the one compose asks: a lead must have come through
    the relaxation as frozen bulk.  A junction whose leads were left free is
    refused naming what to do."""
    from molbuilder.structure import Structure
    from molbuilder.transport.wizard import extract_electrode_model
    j = _junction(frozen=False)
    device = Structure(elements=j["elements"],
                       positions=np.asarray(j["positions"], dtype=float),
                       regions=j["regions"], cell=np.diag(j["cell"]),
                       axis_kind=tuple(j["axis_kind"]))
    with pytest.raises(ValueError) as e:
        extract_electrode_model(device, REGION_LEFT_ELECTRODE)
    msg = str(e.value)
    assert "NOT FROZEN" in msg and "freeze them" in msg, msg


def _au_lead_junction(n_layers: int):
    """A junction whose leads are real fcc(111) Au, ``n_layers`` deep a side,
    labelled and held, one atom between them -- built with ASE."""
    from ase.build import fcc111
    from molbuilder.structure import Structure
    slab = fcc111("Au", size=(1, 1, n_layers), a=4.158,
                  orthogonal=False, vacuum=0.0)
    lead = np.asarray(slab.positions, dtype=float)
    lead[:, 2] -= lead[:, 2].min()
    span = lead[:, 2].max()
    right = lead.copy()
    right[:, 2] += span + 8.0
    bridge = np.array([[lead[0, 0], lead[0, 1], span + 4.0]])
    pos = np.vstack([lead, bridge, right])
    n = len(lead)
    cell = np.asarray(slab.get_cell(), dtype=float)
    cell[2] = [0.0, 0.0, pos[:, 2].max() + 10.0]
    leads = list(range(n)) + list(range(n + 1, 2 * n + 1))
    return Structure(
        elements=["Au"] * n + ["S"] + ["Au"] * n, positions=pos,
        regions={REGION_LEFT_ELECTRODE: list(range(n)),
                 REGION_BRIDGE: [n],
                 REGION_RIGHT_ELECTRODE: list(range(n + 1, 2 * n + 1)),
                 "frozen_atoms": leads},
        cell=cell, axis_kind=("periodic", "periodic", "transport"))


@pytest.mark.parametrize("n_layers,verdict", [(4, "ECLIPSED"), (6, "CONTINUES")])
def test_a_leads_periodic_seam_is_measured(n_layers, verdict):
    """The lead gate measures where a lead meets its own image one cell up
    into the lead's notes (`junction-cell.md` § 3.1): on fcc(111) a
    layer-count question -- four layers eclipse, six continue the crystal.
    Reported, never refused.  That each lead's note reaches the Transport
    card under the lead's own name is read on the junction the end-to-end
    pass relaxes (`tests/test_transport_on_a_real_junction_e2e.py`)."""
    from molbuilder.transport.wizard import extract_electrode_model
    device = _au_lead_junction(n_layers)
    for lead in (REGION_LEFT_ELECTRODE, REGION_RIGHT_ELECTRODE):
        seam = [n for n in extract_electrode_model(device, lead).notes
                if "periodic seam" in n]
        assert len(seam) == 1 and verdict in seam[0], (lead, seam)
