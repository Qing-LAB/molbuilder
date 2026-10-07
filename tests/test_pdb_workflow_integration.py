"""End-to-end PDB workflow integration test.

The real Flask endpoints, hit in the real order a browser would.  No mocks,
no per-endpoint shortcuts.

What it pins (in order):

  1.  /api/build/load reads the PDB and returns the per-atom rows.
  2.  A sidecar with ``frozen_atoms`` + a region is written directly via
      the molstruct codec (keyed by the PDB's stem).
  3.  /api/build/load re-read picks up the sidecar (atoms carry the
      labels in ``regions``).

Run::

    python -m pytest tests/test_pdb_workflow_integration.py -v
"""
from __future__ import annotations

import json

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


def _seed_sidecar_for(struct_path, *, n_atoms, regions=None, frozen=None):
    """Write a ``.molstruct.json`` sidecar next to ``struct_path`` DIRECTLY
    via the codec."""
    from molbuilder.sidecars import molstruct as _msj
    from molbuilder.structure import FROZEN_LABEL
    labels = dict(regions or {})
    if frozen:
        labels[FROZEN_LABEL] = list(frozen)   # a reserved label is a label
    payload = _msj.to_dict(
        {"regions": labels},
        n_atoms_total=n_atoms,
        structure_hash=_msj.sha256_of_file(struct_path),
    )
    _msj.save(_msj.sidecar_path_for(struct_path), payload)


@pytest.fixture
def pdb_under_root(tmp_path, monkeypatch):
    """Place a PDB inside tmp_path AND make tmp_path the picker root
    so the selection endpoints accept it.  Returns ``(pdb_path, n_atoms, n_residues)``."""
    pdb_text = _SYNTHETIC_PDB
    dest = tmp_path / "test_workflow.pdb"
    dest.write_text(pdb_text)

    # Repoint Capabilities.file_picker_roots() at tmp_path so the
    # selection blueprint's allow-list accepts ``dest``.  Snapshot
    # + restore the module-level singleton too -- monkeypatch handles
    # the class-attribute patch, but ``set_capabilities`` mutates a
    # module-level slot that doesn't auto-undo, so a leftover from
    # one test was failing test_capabilities_returns_only_projects_root
    # in the rest of the suite (test-isolation regression).
    from molbuilder import diagnostics
    _orig_caps = diagnostics.get_capabilities()
    caps = diagnostics.Capabilities(
        runtime_config={}, conda_binary=None, conda_envs=frozenset(),
    )
    cls = type(caps)

    def _only_tmp_roots(self):
        return ((tmp_path.resolve(), "projects"),)
    monkeypatch.setattr(cls, "file_picker_roots", _only_tmp_roots)
    diagnostics.set_capabilities(caps)
    # Reset the singleton after the test even though monkeypatch
    # already undoes the class attribute patch.  The module-level
    # name is ``_snapshot`` (per molbuilder.diagnostics).
    monkeypatch.setattr(diagnostics, "_snapshot", _orig_caps)

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

    # ----- Step 1: load the PDB through the one load door ---------- #

    def test_step_1_the_load_door_reads_pdb(
        self, web, pdb_under_root,
    ):
        """`/api/build/load` is the door; its rows are `_shared.atoms_list`."""
        pdb_path, n_atoms, n_residues = pdb_under_root
        r = web.post("/api/build/load", json={"path": self._path(pdb_path)})
        assert r.status_code == 200, r.data
        body = r.get_json()
        assert len(body["atoms"]) == n_atoms, (
            f"the load door sees {len(body['atoms'])} atoms; "
            f"Structure.from_pdb sees {n_atoms}.  Wire format mismatch."
        )
        assert "element" in body["atoms"][0]
        # THE PDB's IDENTITY COLUMNS SURVIVE THE LOAD, at the top level --
        # `structure.py::IDENTITY_FIELDS` carries them "beside `metadata`",
        # not on the atom row.
        assert body.get("residue_names"), f"PDB residue names lost: {sorted(body)}"
        assert body["residue_names"][0], body["residue_names"][:3]
        assert len(body["residue_names"]) == n_atoms

    # ----- Step 2: assign frozen_atoms + a region via the codec ----- #

    def test_step_2_save_writes_current_schema_sidecar(
        self, web, pdb_under_root,
    ):
        pdb_path, n_atoms, _ = pdb_under_root
        # Pick a few atom indices valid for the synthetic PDB (15 atoms).
        frozen_indices = [0, 1, 2]
        region_indices = [3, 4]

        # The sidecar write is REPLACE-ALL: the whole sidecar (regions +
        # frozen) is written in one shot.
        _seed_sidecar_for(pdb_path, n_atoms=n_atoms,
                          regions={"L-electrode": region_indices},
                          frozen=frozen_indices)

        # The sidecar lands on disk next to the PDB, keyed by stem.
        sidecar = pdb_path.with_name(pdb_path.stem + ".molstruct.json")
        assert sidecar.exists(), (
            f"the codec sidecar seed didn't write the sidecar at {sidecar}"
        )
        on_disk = json.loads(sidecar.read_text())
        assert on_disk["n_atoms_total"] == n_atoms
        # ONE label store on disk: the reserved label is a member of `regions`,
        # not a key beside it.
        assert on_disk["regions"]["frozen_atoms"] == frozen_indices
        assert "frozen_atoms" not in on_disk, "a second key for one fact"
        assert on_disk["regions"]["L-electrode"] == region_indices

    # ----- Step 3: re-fetch atoms; the labels are on the atoms - #

    def test_step_3_a_reread_picks_up_the_sidecar(
        self, web, pdb_under_root,
    ):
        pdb_path, n_atoms, _ = pdb_under_root
        # Re-run step 2's save so this test is self-contained.
        _seed_sidecar_for(pdb_path, n_atoms=n_atoms,
                          regions={"L-electrode": [3, 4]}, frozen=[0, 1, 2])

        body = web.post("/api/build/load",
                        json={"path": self._path(pdb_path)}).get_json()
        # ONE representation: every label an atom carries is in `regions`,
        # the reserved `frozen_atoms` among them.  There is no second member.
        atom0 = body["atoms"][0]
        atom3 = body["atoms"][3]
        assert "frozen_atoms" in atom0["regions"], atom0
        assert "is_frozen" not in atom0, atom0
        assert "L-electrode" in atom3["regions"], atom3
        # An untagged atom should NOT carry these:
        atom10 = body["atoms"][min(10, len(body["atoms"]) - 1)]
        if atom10["index"] not in (0, 1, 2):
            assert "frozen_atoms" not in atom10["regions"], atom10
        if atom10["index"] not in (3, 4):
            assert "L-electrode" not in atom10["regions"], atom10

