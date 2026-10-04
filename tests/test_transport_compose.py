"""Prep composes from the citation — `engines/transport.md` § 3.1 (what makes
a directory citable) and § 1 (one citation, five derived stages).

*(How it was designed: `archive/2026-09-01-transport-design.md`
§ 4.1–4.2, `transport/compose.py`).

Properties under guard, each named for its failure:

* the happy path: citation → relaxed geometry FROM THE .XV (never a
  file copy), sorted canonical, electrode models extracted from the
  sorted blocks, provenance with content hashes;
* strict composition (ruling Q2): a missing attempt names the commands
  to run first; an unconcluded attempt refuses without deciding;
* frozen means unmoved (ruling Q3): a drifted electrode atom is
  refused naming the atom, its label, and the distance;
* the § 3 lead gates: a thin block (under the principal-layer floor)
  and a block that does not tile are refused naming the numbers;
* the .XV must describe the cited source (element mismatch refused);
* the composed record travels: written whole (sorted pair + the
  attempt's own deck + sidecars), loaded back without the cited tree,
  and a re-pointed or incomplete record reads as NO record.

(The prep arm itself — stage decks, wrappers, run dirs — is
`test_transport_prep.py`'s subject.)
"""
from __future__ import annotations

import numpy as np

import pytest

from molbuilder.transport.sort import (REGION_BRIDGE,
                                         REGION_LEFT_ELECTRODE,
                                         REGION_RIGHT_ELECTRODE)
from molbuilder.structure import Structure
from molbuilder.transport.compose import ComposeError, compose_junction

# THE one Bohr->Angstrom value (`molbuilder/constants.py`).  A literal
# here would be a second home: three of them had already drifted to
# THREE different values by 2026-09-09 (0.5291772108, 0.529177249 --
# CODATA 1986 -- and 0.529177).  Importing is not circular: every use
# below WRITES a fixture in Bohr, and the assertion is on the Angstrom
# value that comes back.
from molbuilder.constants import BOHR_ANGSTROM as _BOHR_ANGSTROM
_ANG_BOHR = 1.0 / _BOHR_ANGSTROM

#: six 2.5 Å layers a side (12.5 Å span — over the wizard's 12 Å lead
#: floor), the molecule between.
_LAYERS_L = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5]
_BRIDGE = [("S", 15.0), ("C", 16.4), ("C", 17.8), ("S", 19.2)]
_LAYERS_R = [22.0, 24.5, 27.0, 29.5, 32.0, 34.5]


def _junction_struct(layers_l=_LAYERS_L, layers_r=_LAYERS_R):
    elements, zs, labels = [], [], []
    for z in layers_l:
        elements.append("Au"); zs.append(z)
        labels.append(REGION_LEFT_ELECTRODE)
    for el, z in _BRIDGE:
        elements.append(el); zs.append(z); labels.append(REGION_BRIDGE)
    for z in layers_r:
        elements.append("Au"); zs.append(z)
        labels.append(REGION_RIGHT_ELECTRODE)
    positions = np.array([[1.0, 1.0, z] for z in zs])
    regions: dict = {}
    for i, lab in enumerate(labels):
        regions.setdefault(lab, []).append(i)
    frozen = [i for i, lab in enumerate(labels) if lab != REGION_BRIDGE]
    return Structure(elements=elements, positions=positions,
                     regions=regions, frozen_atoms=frozen,
                     cell=np.diag([8.0, 8.0, 40.0]))


class TestFormB:
    """4.1b form B: a labeled .xyz+.molstruct.json pair, anywhere."""

    def _pair_dir(self, tmp_path):
        from molbuilder.workingcopy_structure import StructureCodec
        root = tmp_path / "projects"
        d = root / "anything" / "at all"
        d.mkdir(parents=True)
        StructureCodec().write(_junction_struct(), d / "junction.xyz")
        return root, "anything/at all"

    def test_a_labeled_pair_composes_without_any_layout(self, tmp_path):
        root, cite = self._pair_dir(tmp_path)
        out = compose_junction(cite, tree_root=root)
        assert out.form == "structure"
        assert out.deck_text is None
        assert out.provenance["evidence"] == "given"
        assert len(out.electrode_left.elements) == 6


    def test_the_refusal_names_the_atom_the_PERSON_can_find(self, tmp_path):
        """`engine_atom_index`: the canonical atom identity is the index
        in the SOURCE FILE's order, which is what the Modify tab shows and
        what "go and freeze atom N" has to mean.  `categorical_sort` puts
        the device in TranSIESTA's deck order first, so a refusal built
        from the sorted device's own indices names an atom the person
        cannot find.

        THIS FIXTURE IS DELIBERATELY OUT OF ORDER -- the bridge is written
        first -- because the ordinary one sorts to the identity, where a
        missing translation and a correct one print the same number and
        the test would prove nothing.
        """
        from molbuilder.workingcopy_structure import StructureCodec
        from molbuilder.transport.sort import categorical_sort
        root = tmp_path / "projects"
        d = root / "shuffled"
        d.mkdir(parents=True)

        # bridge FIRST in the file, then the two leads
        elements = [el for el, _z in _BRIDGE] + ["Au"] * 12
        zs = ([z for _el, z in _BRIDGE] + list(_LAYERS_L) + list(_LAYERS_R))
        regions = {REGION_BRIDGE: [0, 1, 2, 3],
                   REGION_LEFT_ELECTRODE: list(range(4, 10)),
                   REGION_RIGHT_ELECTRODE: list(range(10, 16))}
        s2 = Structure(elements=elements,
                       positions=np.array([[1.0, 1.0, z] for z in zs]),
                       regions=regions,
                       frozen_atoms=list(range(4, 16)),
                       cell=np.diag([8.0, 8.0, 40.0]))
        # atom 4 is the L-electrode's first, and the sort moves it to 0
        s2.frozen_atoms = [i for i in s2.frozen_atoms if i != 4]
        srt = categorical_sort(s2)
        assert srt.sorted_to_original[0] == 4, (
            "the fixture must actually permute or this proves nothing")

        StructureCodec().write(s2, d / "junction.xyz")
        with pytest.raises(ComposeError) as e:
            compose_junction("shuffled", tree_root=root)
        msg = str(e.value)
        assert "1 atom(s)" in msg, (
            f"exactly one lead atom was left unfrozen; naming more means "
            f"the region or the frozen set was not remapped by the sort: "
            f"{msg}")
        assert "4 (Au)" in msg, (
            f"the atom must be named by its identity in the person's own "
            f"file (4), not by its place in the deck order (0): {msg}")

    def test_a_pair_whose_leads_are_not_frozen_is_refused(self, tmp_path):
        """Form B has no starting geometry, so the unmoved comparison
        cannot run here -- which is exactly why the frozen DECLARATION is
        asked separately, and of every route."""
        from molbuilder.workingcopy_structure import StructureCodec
        root = tmp_path / "projects"
        d = root / "loose"
        d.mkdir(parents=True)
        s2 = _junction_struct()
        s2.frozen_atoms = []
        StructureCodec().write(s2, d / "junction.xyz")
        with pytest.raises(ComposeError) as e:
            compose_junction("loose", tree_root=root)
        msg = str(e.value)
        assert "NOT FROZEN" in msg, msg
        assert "freeze them" in msg, f"and it must say what to do: {msg}"

    def test_a_pair_with_frozen_leads_and_a_free_bridge_composes(self, tmp_path):
        """The discriminating half: the gate asks about the LEADS, and a
        correct junction has a free bridge."""
        root, cite = self._pair_dir(tmp_path)
        out = compose_junction(cite, tree_root=root)
        free = set(range(len(out.sorted.structure.elements))) - set(
            out.sorted.structure.frozen_atoms or ())
        assert free, "the fixture must leave the bridge free or this proves nothing"

    def test_a_pair_without_a_cell_is_refused(self, tmp_path):
        from molbuilder.workingcopy_structure import StructureCodec
        root = tmp_path / "projects"
        d = root / "loose"
        d.mkdir(parents=True)
        s2 = _junction_struct()
        s2.cell = None
        StructureCodec().write(s2, d / "junction.xyz")
        with pytest.raises(ComposeError) as e:
            compose_junction("loose", tree_root=root)
        assert "no cell" in str(e.value)

    def test_a_bare_xyz_names_the_missing_sidecar(self, tmp_path):
        root = tmp_path / "projects"
        d = root / "loose"
        d.mkdir(parents=True)
        (d / "junction.xyz").write_text("1\n\nC 0 0 0\n")
        with pytest.raises(ComposeError) as e:
            compose_junction("loose", tree_root=root)
        msg = str(e.value)
        assert ".molstruct.json" in msg and ".fdf" in msg, (
            "the refusal states the whole condition")


    def test_a_form_B_pair_with_a_flat_box_is_refused_by_name(self, tmp_path):
        """Form B asked only *"is the cell None?"*, and that was enough for
        exactly as long as `Structure.__post_init__` refused a zero-volume
        lattice outright.

        That refusal was removed 2026-09-21 so a pair holding a bad box could
        be OPENED and fixed on the Cell page (`structure-periodicity.md`
        § 8.2, "reading does not judge") — and this guard had been relying on
        it without saying so.

        WHAT IT COSTS, stated correctly at the second attempt: not a crash.
        `axis_vacuum` inverts the cell unguarded, but nothing reaches it with
        a bad box — `render_deck` validates before the first block renders,
        so the deck path already answers "[cell.no_volume] This box is flat".
        The first write-up of this claimed the traceback, from probing that
        function in isolation. What the guard buys is the refusal landing at
        the CITATION door, naming the cited pair, the way form A has always
        done — instead of surfacing later as a complaint about a box, several
        steps from the file that holds it.
        """
        import json as _json
        from molbuilder.workingcopy_structure import StructureCodec
        root = tmp_path / "projects"
        d = root / "J"
        d.mkdir(parents=True)
        struct = _junction_struct()
        StructureCodec().write(struct, d / "j.xyz")
        side = _json.loads((d / "j.molstruct.json").read_text())
        side["cell"] = [[8.0, 0, 0], [0, 8.0, 0], [0, 0, 0.0]]   # flat
        (d / "j.molstruct.json").write_text(_json.dumps(side))

        # It still OPENS -- that is the § 8.2 half, and it must not regress.
        assert StructureCodec().load(d / "j.xyz").cell is not None

        with pytest.raises(ComposeError) as e:
            compose_junction("J", tree_root=root)
        assert "no volume" in str(e.value), str(e.value)


class TestTheRecordedContract:
    """`transport.md` § 3.1's form B: a pair that carries no recorded
    contract seals nothing."""

    def test_a_plain_pair_stays_open(self, tmp_path):
        from molbuilder.workingcopy_structure import StructureCodec
        root = tmp_path / "projects"
        d = root / "plain"
        d.mkdir(parents=True)
        StructureCodec().write(_junction_struct(), d / "junction.xyz")
        out = compose_junction("plain", tree_root=root)
        assert "recorded_contract" not in out.provenance


