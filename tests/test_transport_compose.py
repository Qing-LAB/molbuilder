"""The lead checks the transport citation asks (`engines/transport.md` § 3.1,
§ 4): what is said or measured on the structure alone; and the composed
junction and its record read at their one version (§ 1.1).

What a citation of a finished relaxation composes -- the leads it cuts, how
the run ended, the electrode rename on this calculation's own copy -- reads
the `.XV` SIESTA wrote, so it is asserted on the junction the end-to-end
pass relaxes with the engine (`tests/test_transport_on_a_real_junction_e2e.py`,
plan § 5y).  The layer positions are `support.junction`'s.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from molbuilder.transport.sort import (REGION_BRIDGE, REGION_LEFT_ELECTRODE,
                                       REGION_RIGHT_ELECTRODE)


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


@pytest.mark.parametrize("n_layers,verdict", [(4, "eclipsed"), (6, "continues")])
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
        seam = [n.message for n in extract_electrode_model(device,
                                                           lead).notes
                if n.where == "lead.seam"]
        assert len(seam) == 1 and verdict in seam[0], (lead, seam)


@pytest.mark.parametrize("moved, said", [
    (None, False),
    (REGION_BRIDGE, False),
    (REGION_RIGHT_ELECTRODE, True),
], ids=["frames-alike", "the-bridge-moves", "a-lead-atom-moves"])
def test_a_frame_that_moves_a_lead_atom_is_said(moved, said):
    """`engines/transport.md` § 2a.9: a transport calculation citing a frame
    set computes the leads once, from the first frame, so every frame keeps
    the lead atoms where that frame has them -- a frame that moves one is
    said, naming the frame and the atom, never a stop (the user,
    2026-11-18); the bridge is the frames' to move.  The silent failure:
    the leads drifting frame by frame with nothing said, and TranSIESTA
    stopping each such frame's device stage."""
    from molbuilder.transport.compose import frames_moving_the_leads
    from pathlib import Path
    base = _au_lead_junction(4)
    frames = [np.asarray(base.positions, dtype=float)] * 2
    if moved is not None:
        second = frames[0].copy()
        second[base.regions[moved][0], 2] += 0.01
        frames = [frames[0], second]
    s = base.replace(frames=np.stack(frames))
    sentence = frames_moving_the_leads(s, Path("set.xyz"))
    assert (sentence is not None) is said, sentence
    if said:
        assert "frame 2 of 2" in sentence and "Au" in sentence, sentence


#: THE COMPOSED JUNCTION'S SIDECAR OF ANOTHER VERSION, a row each: the
#: version it states, by who wrote it, and the words it is refused in.
_JUNCTION_VERSIONS = {
    "composed before 2026-11-04": (
        lambda current: 16,
        ("was composed before 2026-11-04: its sidecar is version 16, saved "
         "before 2026-11-04, and this molbuilder reads version 17 alone.",
         "Describe a new transport calculation of the same junction and run "
         "it"),
        ("written by a later molbuilder",)),
    "a later molbuilder's": (
        lambda current: current + 1,
        ("does not read: its sidecar is version",
         "written by a later molbuilder than this one",
         "open it with the molbuilder that wrote it"),
        ("Describe a new transport calculation", "composed before")),
}


@pytest.mark.parametrize("version", list(_JUNCTION_VERSIONS))
def test_a_calculation_composed_before_the_one_pair_version_is_refused(
        version, tmp_path):
    """`engines/transport.md` § 1.1, `structure-molstruct.md` § 2 (the user,
    2026-11-26: "Everything before 2026-11-04"), `job-contracts.md` § 6.1
    (*The versions' words*): a transport calculation's composed junction
    whose sidecar states a version this build does not read is refused by
    name -- one composed before 2026-11-04 with the calculation's way on, a
    new calculation described and run, never a pair's; one a later
    molbuilder wrote with the later words alone.  The silent failures: such
    a junction read as a v17 one, or the person told to save a pair the
    calculation composed; a later molbuilder's junction told to be
    described again by this one."""
    import json
    from molbuilder.sidecars.molstruct import SCHEMA_VERSION, sidecar_path_for
    from molbuilder.structure import Structure
    from molbuilder.transport.compose import (JUNCTION_GEOMETRY, ComposeError,
                                              recorded_junction)
    from molbuilder.workingcopy_structure import StructureCodec
    stated, words, not_words = _JUNCTION_VERSIONS[version]
    path = StructureCodec().write(
        Structure.from_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n"),
        tmp_path / JUNCTION_GEOMETRY)
    side = sidecar_path_for(path)
    payload = json.loads(side.read_text())
    payload["schema_version"] = stated(SCHEMA_VERSION)
    side.write_text(json.dumps(payload))
    with pytest.raises(ComposeError) as refused:
        recorded_junction(tmp_path)
    said = str(refused.value)
    assert said.startswith(f"the composed junction, {JUNCTION_GEOMETRY}, "), (
        said)
    for w in words:
        assert w in said, (w, said)
    for w in not_words:
        assert w not in said, (w, said)


def test_a_composition_record_of_another_format_is_refused(tmp_path):
    """`execution/job-contracts.md` § 6.1, `engines/transport.md` § 1.1: the
    composition record reads its current format alone -- one of another, a
    calculation composed before 2.10.0, refused by name with the way on.
    The silent failure: an older record's absent fields read as defaults."""
    import json
    from molbuilder.runfiles import SLOT_PROVENANCE_FILE
    from molbuilder.transport.compose import read_provenance
    (tmp_path / SLOT_PROVENANCE_FILE).write_text(json.dumps(
        {"schema": "molbuilder/slot-provenance@1", "citation": "P/s/R"}))
    with pytest.raises(ValueError, match=re.escape(
            "'molbuilder/slot-provenance@1' -- a transport calculation "
            "composed before 2.10.0 -- and this molbuilder reads "
            "molbuilder/slot-provenance@2 alone: describe a new transport "
            "calculation of the same junction and run it")):
        read_provenance(tmp_path)
