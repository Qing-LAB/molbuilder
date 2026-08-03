"""Results-tab metadata bridge: a run's embedded ATOM-METADATA block
(region labels / frozen tags / annotation channels the Build tab wrote
into the input .fdf / .py) must ride onto the structure MolView loads
from the run's OUTPUT logs.

The Results-tab trajectory inspector loads *coordinates* from
``.molwatch.log`` / ``.out`` / ``*_geom_optim.xyz`` -- geometry only.
The per-atom metadata lives in the input script.  This bridge recovers it
and re-applies it through the same ``apply_to_structure`` seam a
``.molstruct.json`` uses, so the loaded viewer shows the same regions /
frozen the user set in Build.

Three seams, each tested for its END RESULT (not just API presence):

  1. parse layer -- ``atom_metadata_json_for_run_dir`` (``parse/dirs``,
     the directory-scoped layer; the TextParser itself stays memory-only)
     recovers the block from a run dir, guards the atom count, returns
     None on mismatch.
  2. molview door -- ``/api/build/load`` applies a TRUSTED ``atom_metadata``
     block (distinct from an untrusted ``.molstruct.json`` ``sidecar``):
     the response's per-atom payload carries regions + is_frozen.
  3. results adapter -- ``/api/watch/load`` on a run directory surfaces the
     block as ``atom_metadata`` in the load response.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from molbuilder.parse.dirs.atom_metadata import atom_metadata_json_for_run_dir
from molbuilder.parse.scripts.atom_metadata import _extract_atom_metadata_dict
from molbuilder.script_emit import emit_atom_metadata


# --------------------------------------------------------------------- #
#  Fixtures                                                              #
# --------------------------------------------------------------------- #


def _block(regions, frozen, n, annotations=None) -> str:
    """The ATOM-METADATA block text emit_atom_metadata writes into a script.

    ``frozen`` is the reserved label, written into the ONE label store the
    emitter takes -- it has no parameter of its own for it."""
    from molbuilder.structure import FROZEN_LABEL
    labels = dict(regions or {})
    if frozen:
        labels[FROZEN_LABEL] = list(frozen)
    return emit_atom_metadata(
        regions=labels, n_atoms_total=n, annotations=annotations,
    )


def _fdf_with_block(regions, frozen, n) -> str:
    return "SystemLabel run\n\n" + _block(regions, frozen, n) + "\n"


_XYZ_4C_FRAME0 = "4\nframe0\nC 0 0 0\nC 1 0 0\nC 2 0 0\nC 3 0 0\n"


def _md_json_4c() -> str:
    """The JSON string the parse fn hands the client for a 4-carbon run."""
    return json.dumps(_extract_atom_metadata_dict(
        _block({"electrode_L": [0, 1], "device": [2, 3]}, [0, 1], 4)))


# --------------------------------------------------------------------- #
#  1. Parse layer: atom_metadata_json_for_run_dir                       #
# --------------------------------------------------------------------- #


class TestParseRecovery:
    def test_recovers_block_from_fdf(self, tmp_path):
        (tmp_path / "run.fdf").write_text(
            _fdf_with_block({"electrode_L": [0, 1], "device": [2, 3]}, [0, 1], 4))
        out = atom_metadata_json_for_run_dir(tmp_path, 4)
        assert out is not None
        md = json.loads(out)
        # ONE label store in the block: the reserved label with the rest.
        assert md["regions"] == {"electrode_L": [0, 1], "device": [2, 3],
                                 "frozen_atoms": [0, 1]}
        assert "frozen_atoms" not in md, "the same fact in the block twice"
        assert md["n_atoms_total"] == 4

    def test_recovers_block_from_py_pyscf(self, tmp_path):
        """PySCF runs embed the SAME block in the .py script."""
        (tmp_path / "job.py").write_text(
            "# job_name = job\n" + _block({"solvent": [0]}, [0], 3) + "\n")
        out = atom_metadata_json_for_run_dir(tmp_path, 3)
        assert out is not None and json.loads(out)["regions"] == {
            "solvent": [0], "frozen_atoms": [0]}

    def test_atom_count_mismatch_returns_none(self, tmp_path):
        """A block whose n_atoms_total disagrees with the trajectory would
        make apply_to_structure raise -> drop it so the load still shows
        coordinates."""
        (tmp_path / "run.fdf").write_text(
            _fdf_with_block({"a": [0, 1]}, [0], 4))
        assert atom_metadata_json_for_run_dir(tmp_path, 5) is None
        # No guard -> recovered.
        assert atom_metadata_json_for_run_dir(tmp_path, None) is not None

    def test_no_block_returns_none(self, tmp_path):
        (tmp_path / "run.fdf").write_text("SystemLabel run\n")
        assert atom_metadata_json_for_run_dir(tmp_path, 4) is None

    def test_empty_block_returns_none(self, tmp_path):
        """emit_atom_metadata returns None (no block emitted) when there's
        nothing to carry, so a dir with only-geometry scripts yields None."""
        assert _block({}, [], 4) is None
        (tmp_path / "run.fdf").write_text("SystemLabel run\n")
        assert atom_metadata_json_for_run_dir(tmp_path, 4) is None

    def test_non_dir_and_none_return_none(self, tmp_path):
        assert atom_metadata_json_for_run_dir(None) is None
        assert atom_metadata_json_for_run_dir(tmp_path / "nope") is None
        assert atom_metadata_json_for_run_dir(str(tmp_path / "x.fdf")) is None


# --------------------------------------------------------------------- #
#  2. MolView door: /api/build/load applies the trusted atom_metadata   #
# --------------------------------------------------------------------- #


@pytest.fixture()
def client():
    from molbuilder.web.app import create_app
    return create_app(config={}).test_client()


class TestBuildLoadDoor:


    def test_atom_count_mismatch_is_400_not_500(self, client):
        bad = json.dumps({**json.loads(_md_json_4c()), "n_atoms_total": 5})
        r = client.post("/api/build/load", json={
            "text": _XYZ_4C_FRAME0, "filename": "t.xyz", "atom_metadata": bad})
        assert r.status_code == 400
        assert "atom_metadata:" in r.get_json()["error"]

    def test_malformed_json_is_400(self, client):
        r = client.post("/api/build/load", json={
            "text": _XYZ_4C_FRAME0, "filename": "t.xyz",
            "atom_metadata": "{not valid"})
        assert r.status_code == 400
        assert "atom_metadata" in r.get_json()["error"]

    def test_trusted_block_bypasses_sidecar_envelope(self, client):
        """The block has NO structure_hash (it's a trusted fragment, not a
        .molstruct.json file).  Passed as ``sidecar`` it would 400 on the
        missing envelope; passed as ``atom_metadata`` it applies cleanly."""
        block_json = _md_json_4c()
        assert "structure_hash" not in json.loads(block_json)
        as_sidecar = client.post("/api/build/load", json={
            "text": _XYZ_4C_FRAME0, "filename": "t.xyz", "sidecar": block_json})
        assert as_sidecar.status_code == 400
        assert "structure_hash" in as_sidecar.get_json()["error"]
        as_meta = client.post("/api/build/load", json={
            "text": _XYZ_4C_FRAME0, "filename": "t.xyz",
            "atom_metadata": block_json})
        assert as_meta.status_code == 200

    def test_the_runs_cell_rides_in_the_same_block_as_its_labels(self, client):
        """Seam 4 (2026-08-03): a run's LATTICE travels with its labels.

        A trajectory has no `.molstruct.json`.  Its labels come from the input
        script and its lattice from the output logs, and both describe the same
        atoms -- so the Results tab sends one block and the server applies it
        through the one authority, rather than inventing a second door for the
        cell.

        AND NO ``pbc`` IS SENT, deliberately.  A run's 3x3 says nothing about
        which axes repeat, and guessing periodicity in the browser is the one
        thing the model refuses to do.  ``Structure``'s own rule resolves it --
        "a lattice implies periodicity" -- so a stated cell comes back fully
        periodic, decided in one place.
        """
        block = json.loads(_md_json_4c())
        block["cell"] = [[10.0, 0, 0], [0, 11.0, 0], [0, 0, 12.0]]
        assert "pbc" not in block, "the browser must not state periodicity"
        r = client.post("/api/build/load", json={
            "text": _XYZ_4C_FRAME0, "filename": "run.xyz",
            "atom_metadata": json.dumps(block)})
        assert r.status_code == 200, r.get_json()
        body = r.get_json()
        per = body.get("periodicity") or {}
        assert per.get("cell") == [[10.0, 0, 0], [0, 11.0, 0], [0, 0, 12.0]], (
            f"the run's cell did not reach the structure: {per!r} -- a "
            f"trajectory then draws no unit-cell box, at HTTP 200"
        )
        # The labels still landed: one block, both kinds of fact.
        atoms = body.get("atoms") or []
        assert any(a.get("regions") for a in atoms), (
            "the cell arrived but the labels did not; they travel together"
        )


# --------------------------------------------------------------------- #
#  3. Results adapter: /api/watch/load surfaces atom_metadata           #
# --------------------------------------------------------------------- #


def _register_tmp_as_picker_root(tmp_path, monkeypatch):
    """Watch's JSON-path mode constrains reads to the picker roots; point
    them at tmp so the test's run dir is loadable (mirrors
    test_results_folder_dispatch_e2e)."""
    from molbuilder import diagnostics
    caps = diagnostics.Capabilities(
        runtime_config={}, conda_binary=None, conda_envs=frozenset())
    monkeypatch.setattr(
        type(caps), "file_picker_roots",
        lambda self: ((tmp_path.resolve(), "projects"),))
    diagnostics.set_capabilities(caps)


_XYZ_MULTIFRAME_3 = "".join(
    "3\n"
    f"Iteration {i} Energy {-76.41 + i*0.001:.6f}\n"
    f"O 0 0 {i*0.01:.4f}\nH 0.957 0 0\nH -0.239 0.927 0\n"
    for i in range(3)
)


class TestWatchLoadSurfacesMetadata:
    def test_run_dir_load_returns_atom_metadata(self, client, tmp_path, monkeypatch):
        """END RESULT: loading a run DIRECTORY (coordinates from the geom
        trajectory) returns the input script's block as ``atom_metadata``,
        guarded to the trajectory's 3 atoms."""
        _register_tmp_as_picker_root(tmp_path, monkeypatch)
        run = tmp_path / "run"
        run.mkdir()
        # Input script carries the block; output the resolver finds carries
        # the coordinates (generic *_geom_optim.xyz fallback).
        (run / "run.fdf").write_text(
            _fdf_with_block({"solvent": [0], "hydrogens": [1, 2]}, [0], 3))
        (run / "run_geom_optim.xyz").write_text(_XYZ_MULTIFRAME_3)

        r = client.post("/api/watch/load", json={"path": str(run)})
        assert r.status_code == 200, r.get_json()
        d = r.get_json()
        assert d["ok"] is True
        assert d["atom_metadata"], "run-dir load must surface atom_metadata"
        md = json.loads(d["atom_metadata"])
        assert md["regions"] == {"solvent": [0], "hydrogens": [1, 2],
                                 "frozen_atoms": [0]}
        assert "frozen_atoms" not in md

    def test_run_dir_without_block_returns_none(self, client, tmp_path, monkeypatch):
        _register_tmp_as_picker_root(tmp_path, monkeypatch)
        run = tmp_path / "run2"
        run.mkdir()
        (run / "run.fdf").write_text("SystemLabel run\n")
        (run / "run_geom_optim.xyz").write_text(_XYZ_MULTIFRAME_3)
        d = client.post("/api/watch/load", json={"path": str(run)}).get_json()
        assert d["ok"] is True
        assert d.get("atom_metadata") is None
