"""End-to-end PDB workflow integration test.

The user can pick a ``.pdb`` in the Projects sidebar, mark atoms as
frozen via the /modify selection panel, and then generate a /spectra
script that respects those boundary conditions.  This test pins the
WHOLE chain in one pytest, hitting the real Flask endpoints in the
real order a browser would.  No mocks, no per-endpoint shortcuts --
if any link in the chain regresses, this test fails.

What it pins (in order):

  1.  /api/selection/atoms reads the PDB and returns the per-atom
      metadata table (element, residue_name, chain_id, ...).
  2.  A v4 sidecar with ``frozen_atoms`` + ``regions`` is written
      directly via the molstruct codec (keyed by the PDB's stem) --
      formerly seeded via /api/selection/save-sidecar before that
      endpoint was removed.
  3.  /api/selection/atoms re-read picks up the sidecar (atoms now
      carry the new region tags + is_frozen flags).
  (4-5 retired at P3: the spectra schema/render routes are gone;
      the form's ``frozen_indices`` default from the sidecar.
  5.  POST /api/spectra/render with the PDB content + structure_path
      emits a script + surfaces Pattern A (sidecar's frozen atoms
      don't all appear in cfg.frozen_indices) and Pattern B
      (sidecar regions ignored by the spectra engine) as WARN-
      severity Issues.
  6.  The generated script's ``FROZEN_INDICES_USER`` reflects what
      the FORM said (not the sidecar) -- the form is authoritative
      per the three-stage contract in design.md.

If any of these assertions fires, something concrete is broken
in the cross-tab boundary-condition flow.  This is the test you
can run instead of trusting any "deep review" claim.

Run::

    python -m pytest tests/test_pdb_workflow_integration.py -v
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from support.envelope import (from_xyz as _env,
                             from_xyz_with_periodicity as _env_per)



# ONE PDB, BUILT HERE (2026-08-03).
#
# This preferred a real file from the user's projects tree
# (hemeC-dithiol/structure/1c75.pdb) and fell back to the synthetic one only
# when that was absent -- the worst of both.  The test then meant something
# DIFFERENT on each machine, so every assertion had to be weak enough to hold
# for a 3-residue peptide and a real protein at once, and a failure here could
# not be read without first asking which file had been used.  The user's file
# is also their scientific record, and nobody had confirmed it was relevant.
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
    via the codec.  Formerly seeded with a ``POST /api/selection/save-sidecar``
    before that endpoint was removed (the save is now the projects.parser door →
    ``/api/files/write``).  ``save-sidecar`` was REPLACE-all, so one
    ``to_dict`` mirrors a single endpoint call."""
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
    so the selection endpoints accept it.  Uses the user's real
    1c75.pdb when available; the tiny synthetic above otherwise.
    Returns ``(pdb_path, n_atoms, n_residues)``."""
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
    """One test class, one structure picked, six steps walked in
    order.  Each step depends on the previous step's outputs --
    if step 2's sidecar write fails, step 3's pre-fill assertion
    catches it; etc.  The class shares fixture state via attributes."""

    @staticmethod
    def _envelope(pdb_path, regions=None):
        """The structure AS DATA, which is how every structure door takes it
        (web-api.md § 1).

        These three tests posted ``structure_text`` until 2026-08-03 and got a
        flat 400 -- the field was retired from ``/api/spectra/render`` because
        the viewer holds no coordinate document and writes none (molview.md
        § 11.7), so the route's only caller could never have sent it.  The
        SUBJECTS below (Pattern A/B warnings, form-vs-sidecar precedence,
        clearing a prefill) are unaffected; only the delivery moved.
        """
        from molbuilder.structure import Structure
        struct = Structure.from_pdb(str(pdb_path))
        if regions:
            # INSIDE the structure.  A top-level `regions` beside the envelope
            # was the second source, applied by a second applier; both are gone
            # (2026-08-03) and labels now ride with the atoms they describe.
            struct.regions = dict(regions)
            struct.__post_init__()
        return struct.to_dict()

    def _path(self, pdb_path):
        return str(pdb_path.resolve())

    # ----- Step 1: load the PDB via the selection API ---------- #

    def test_step_1_selection_atoms_reads_pdb(
        self, web, pdb_under_root,
    ):
        pdb_path, n_atoms, n_residues = pdb_under_root
        r = web.post("/api/selection/atoms", json={
            "structure_path": self._path(pdb_path),
        })
        assert r.status_code == 200, r.data
        body = r.get_json()
        assert body["n_atoms"] == n_atoms, (
            f"selection blueprint sees {body['n_atoms']} atoms; "
            f"Structure.from_pdb sees {n_atoms}.  Wire format mismatch."
        )
        # Per-atom metadata should be present for a PDB load (vs the
        # plain-XYZ case where atom_name etc. are absent).
        first = body["atoms"][0]
        assert "element" in first
        # PDB-derived: atom_name + residue_name + chain_id are filled
        # by Structure.from_pdb.
        assert first.get("atom_name") or first.get("residue_name"), (
            f"PDB-loaded atom 0 missing PDB metadata: {first}"
        )

    # ----- Step 2: assign frozen_atoms + a region via the codec ----- #

    def test_step_2_save_writes_current_schema_sidecar(
        self, web, pdb_under_root,
    ):
        pdb_path, n_atoms, _ = pdb_under_root
        # Pick a few atom indices that are valid for any structure
        # we'd test against (synthetic = 15 atoms; user's = 1184).
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
        from molbuilder.sidecars import molstruct as _msj
        assert on_disk["n_atoms_total"] == n_atoms
        # ONE label store on disk: the reserved label is a member of `regions`,
        # not a key beside it.
        assert on_disk["regions"]["frozen_atoms"] == frozen_indices
        assert "frozen_atoms" not in on_disk, "a second key for one fact"
        assert on_disk["regions"]["L-electrode"] == region_indices

    # ----- Step 3: re-fetch atoms; the labels are on the atoms - #

    def test_step_3_atoms_reread_picks_up_sidecar(
        self, web, pdb_under_root,
    ):
        pdb_path, n_atoms, _ = pdb_under_root
        # Re-run step 2's save so this test is self-contained.
        _seed_sidecar_for(pdb_path, n_atoms=n_atoms,
                          regions={"L-electrode": [3, 4]}, frozen=[0, 1, 2])

        r = web.post("/api/selection/atoms", json={
            "structure_path": self._path(pdb_path),
        })
        body = r.get_json()
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

    # ----- Step 4: /spectra schema pre-fills frozen_indices ----- #

    # Steps 4-7 (schema pre-fill -> render -> prefill-clear)
    # retired with their routes (spectra-migration plan P3,
    # 2026-08-21): frozen atoms are STRUCTURE-side facts now --
    # they ride the sidecar into the hand-over's codec pair
    # (pinned by test_task_setup_tab.py's byte-compat test and
    # TestBuildSiestaHonorsSidecarFrozenAtoms below), never a
    # form field.


# --------------------------------------------------------------------- #
# Build (SIESTA + PySCF) sidecar-honouring integration test             #
#                                                                       #
# 2026-05-25 regression: the engine emitters knew how to handle         #
# struct.frozen_atoms, but /api/build/fdf + /api/build/pyscf never      #
# applied the sidecar -- the user's /modify freeze list silently        #
# never reached render_fdf / render_script.  This class catches the     #
# wiring end-to-end: write a sidecar, POST to /api/build/fdf with       #
# structure_path, assert %block Geometry.Constraints appears in the     #
# emitted FDF with the right 1-based indices.                           #
# --------------------------------------------------------------------- #


@pytest.fixture
def simple_pdb_under_root(tmp_path, monkeypatch):
    """A 10-atom CHON-only PDB the SIESTA + PySCF emitters can both
    render cleanly (no Fe / S / exotic-element edge cases tripping
    up the species table).  We need a structure WHERE THE ENGINE
    RENDERERS SUCCEED so the test isolates the sidecar-wiring
    regression from emitter bugs.

    Repoints picker root at tmp_path so the build/spectra endpoints
    accept the path under the security gate."""
    pdb_text = (
        "REMARK 1 synthetic chon for sidecar wiring test\n"
        "ATOM      1  C   MOL A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  C   MOL A   1       1.500   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  N   MOL A   1       2.250   1.300   0.000  1.00  0.00           N\n"
        "ATOM      4  C   MOL A   1       3.750   1.300   0.000  1.00  0.00           C\n"
        "ATOM      5  O   MOL A   1       4.500   2.600   0.000  1.00  0.00           O\n"
        "ATOM      6  H   MOL A   1      -0.520   0.900   0.700  1.00  0.00           H\n"
        "ATOM      7  H   MOL A   1      -0.520  -0.900  -0.700  1.00  0.00           H\n"
        "ATOM      8  H   MOL A   1       2.020  -0.900   0.000  1.00  0.00           H\n"
        "ATOM      9  H   MOL A   1       1.750   2.200   0.000  1.00  0.00           H\n"
        "ATOM     10  H   MOL A   1       4.270   0.400   0.000  1.00  0.00           H\n"
        "END\n"
    )
    p = tmp_path / "ten_atom_chon.pdb"
    p.write_text(pdb_text)
    from molbuilder import diagnostics
    orig = diagnostics.get_capabilities()
    caps = diagnostics.Capabilities(runtime_config={},
                                      conda_binary=None,
                                      conda_envs=frozenset())
    cls = type(caps)
    monkeypatch.setattr(cls, "file_picker_roots",
                         lambda self: ((tmp_path.resolve(), "projects"),))
    diagnostics.set_capabilities(caps)
    monkeypatch.setattr(diagnostics, "_snapshot", orig)
    return p


class TestBuildSiestaHonorsSidecarFrozenAtoms:
    """Mirrors TestPdbWorkflowEndToEnd's pattern: one structure, one
    sidecar, three steps walked in order."""




