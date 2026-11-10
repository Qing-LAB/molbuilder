"""End-to-end PDB workflow integration test.

The real Flask endpoints, hit in the real order a browser would.  No mocks,
no per-endpoint shortcuts.

What it pins (in order):

  1.  /api/build/load reads the PDB and returns its structure, the residue
      names among it.
  2.  The structure with ``frozen_atoms`` + a region is saved through
      /api/structure/save, as the Molbuilder tab saves it: a `.pdb` is read,
      never written (`model/structure.md` § 2.4), so the save writes
      ``<stem>.xyz`` and its sidecar, the PDB's residue identity riding the
      sidecar -- and /api/build/load re-read picks it up (the labels in the
      structure's ``regions``).

Run::

    python -m pytest tests/test_pdb_workflow_integration.py -v
"""
from __future__ import annotations

import pytest


# ONE PDB, BUILT HERE, so the test means the same thing on every machine.
_SYNTHETIC_PDB = (
    "HEADER    SYNTHETIC TRIPEPTIDE\n"
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
    "ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00  0.00           C\n"
    "ATOM      3  C   ALA A   1       2.100   1.300   0.000  1.00  0.00           C\n"
    "ATOM      4  O   ALA A   1       3.300   1.400   0.000  1.00  0.00           O\n"
    "ATOM      5  CB  ALA A   1       1.900  -1.000  -1.000  1.00  0.00           C\n"
    "ATOM      6  N   GLY A   2       1.350   2.350   0.000  1.00  0.00           N\n"
    "ATOM      7  CA  GLY A   2       1.900   3.700   0.000  1.00  0.00           C\n"
    "ATOM      8  C   GLY A   2       3.300   3.700   0.500  1.00  0.00           C\n"
    "ATOM      9  O   GLY A   2       4.000   2.700   0.500  1.00  0.00           O\n"
    "ATOM     10  N   SER A   3       3.700   4.900   1.000  1.00  0.00           N\n"
    "ATOM     11  CA  SER A   3       5.100   5.300   1.000  1.00  0.00           C\n"
    "ATOM     12  C   SER A   3       5.500   6.000   2.200  1.00  0.00           C\n"
    "ATOM     13  O   SER A   3       6.700   6.300   2.300  1.00  0.00           O\n"
    "ATOM     14  CB  SER A   3       5.500   6.100  -0.200  1.00  0.00           C\n"
    "ATOM     15  OG  SER A   3       6.900   6.300  -0.200  1.00  0.00           O\n"
    "END\n"
)


@pytest.fixture
def pdb_under_root(isolated_projects_root):
    """A PDB inside a projects tree of this test's own, the one door pointed
    at it (`isolated_projects_root`), which the selection endpoints accept.
    Returns ``(pdb_path, n_atoms, n_residues)``."""
    pdb_text = _SYNTHETIC_PDB
    dest = isolated_projects_root / "P" / "structure" / "test_workflow.pdb"
    dest.parent.mkdir(parents=True)
    dest.write_text(pdb_text)

    # Parse via the actual Python API so the test knows the truth
    # without re-parsing the file via the HTTP layer.  This is the
    # ORACLE the API assertions are checked against.
    from molbuilder.structure import Structure
    truth = Structure.from_pdb(dest.read_text())
    return dest, truth.n_atoms, truth.n_residues


@pytest.fixture
def web(pdb_under_root):
    pytest.importorskip("flask")
    from molbuilder.web.app import create_app
    app = create_app(config={})
    return app.test_client()


# --------------------------------------------------------------------- #
# THE workflow test                                                     #
# --------------------------------------------------------------------- #


class TestPdbWorkflowEndToEnd:
    """One test class, one structure picked, three steps walked in
    order."""

    @staticmethod
    def _envelope(pdb_path, regions=None):
        """The structure AS DATA, which is how every structure door takes it
        (web-api.md § 1).
        """
        from molbuilder.structure import Structure
        struct = Structure.from_pdb(pdb_path.read_text())
        if regions:
            # INSIDE the structure: labels ride with the atoms they describe.
            struct.regions = dict(regions)
            struct.__post_init__()
        return struct.to_dict()

    def _path(self, pdb_path):
        return str(pdb_path.resolve())

    def _saved(self, web, pdb_path, regions):
        """The structure with ``regions`` saved through the save route as
        the Molbuilder tab saves it -- ``<stem>.xyz`` beside the PDB, the
        pair written by the codec (`model/structure.md` § 3.1) -- and its
        path returned."""
        saved = pdb_path.with_suffix(".xyz")
        r = web.post("/api/structure/save", json={
            "path": self._path(saved),
            "structure": self._envelope(pdb_path, regions)})
        assert r.status_code == 200, r.get_json()
        return saved

    # ----- Step 1: load the PDB through the one load door ---------- #

    def test_step_1_the_load_door_reads_pdb(
        self, web, pdb_under_root,
    ):
        """`/api/build/load` is the door; its structure is
        `_shared.structure_answer`'s."""
        pdb_path, n_atoms, n_residues = pdb_under_root
        r = web.post("/api/build/load", json={"path": self._path(pdb_path)})
        assert r.status_code == 200, r.data
        structure = r.get_json()["structure"]
        assert len(structure["elements"]) == n_atoms, (
            f"the load door sees {len(structure['elements'])} atoms; "
            f"Structure.from_pdb sees {n_atoms}.  Wire format mismatch."
        )
        assert structure["elements"][0]
        # THE PDB's IDENTITY COLUMNS SURVIVE THE LOAD, in the structure --
        # `structure.py::IDENTITY_FIELDS` carries them "beside `metadata`".
        names = structure.get("residue_names")
        assert names, f"PDB residue names lost: {sorted(structure)}"
        assert names[0], names[:3]
        assert len(names) == n_atoms

    # ----- Step 2: assign frozen_atoms + a region, save, re-read ---- #

    def test_step_2_labels_saved_on_the_pair_come_back_on_the_atoms(
        self, web, pdb_under_root,
    ):
        """A PDB's save writes the `.xyz` pair, the PDB read and left as it
        was, and its labels and residue identity come back on the atoms
        (`structure-molstruct.md`).  THE SILENT FAILURE: the geometry saved and
        the frozen atoms lost, seen only when a run moves them."""
        from molbuilder.sidecars import molstruct as msj
        from molbuilder.structure import Structure
        from molbuilder.workingcopy_structure import StructureCodec
        pdb_path, n_atoms, _ = pdb_under_root
        pdb_was = pdb_path.read_text()
        # Pick a few atom indices valid for the synthetic PDB (15 atoms).
        frozen_indices = [0, 1, 2]
        region_indices = [3, 4]

        # The save writes the whole pair in one shot: the geometry and its
        # sidecar, the labels the structure carries.
        saved = self._saved(web, pdb_path, {"L-electrode": region_indices,
                                            "frozen_atoms": frozen_indices})

        # The sidecar lands on disk beside the saved .xyz, keyed by stem.
        sidecar = saved.with_name(saved.stem + ".molstruct.json")
        assert sidecar.exists(), (
            f"the save did not write the sidecar at {sidecar}"
        )
        on_disk = msj.load(sidecar)
        assert on_disk["n_atoms_total"] == n_atoms
        # ONE label store on disk: the reserved label is a member of `regions`,
        # not a key beside it.
        assert on_disk["regions"]["frozen_atoms"] == frozen_indices
        assert "frozen_atoms" not in on_disk, "a second key for one fact"
        assert on_disk["regions"]["L-electrode"] == region_indices

        # THE PDB'S RESIDUE IDENTITY rides the sidecar (`model/structure.md`
        # § 2.4, *the format choice*) -- and the PDB is read alone: the
        # sidecar beside it is the .xyz's, never its own.
        assert (StructureCodec().load(saved).residue_names
                == Structure.from_pdb(pdb_was).residue_names)
        assert not StructureCodec().load(pdb_path).regions, (
            "the .pdb took the .xyz's sidecar for its own")

        meta = web.post("/api/build/load", json={"path": self._path(saved)}
                        ).get_json()["structure"]["metadata"]
        # ONE representation: every label an atom carries is in `regions`,
        # the reserved `frozen_atoms` among them.  There is no second member.
        regions = meta["regions"]
        assert "frozen_atoms" not in meta and "is_frozen" not in meta, \
            sorted(meta)
        assert 0 in regions["frozen_atoms"], regions
        assert 3 in regions["L-electrode"], regions
        # An untagged atom should NOT carry these:
        assert 10 not in regions["frozen_atoms"], regions
        assert 10 not in regions["L-electrode"], regions
