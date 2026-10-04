"""Results-tab metadata bridge: a run's embedded ATOM-METADATA block
(region labels / frozen tags / annotation channels the Build tab wrote
into the input .fdf / .py) must ride onto the structure MolView loads
from the run's OUTPUT logs.

The Results-tab trajectory inspector loads *coordinates* from
``.molwatch.log`` / ``.out`` / ``*_geom_optim.xyz`` -- geometry only.
The per-atom metadata lives in the input script.  This bridge recovers it
and re-applies it through ``script_emit.apply_atom_metadata`` -- THE one
reader of this block, shared with the transport composite -- so the loaded
viewer shows the same regions / frozen the user set in Build.

Two seams, each tested for its END RESULT (not just API presence):

  1. parse layer -- ``atom_metadata_json_for_run_dir`` (``parse/dirs``,
     the directory-scoped layer; the TextParser itself stays memory-only)
     recovers the block from a run dir, guards the atom count, returns
     None on mismatch.
  2. results adapter -- ``/api/watch/load`` on a run directory surfaces the
     block as ``atom_metadata`` in the load response, and folds it into the
     envelope it answers.

(A third seam, ``/api/build/load`` applying an ``atom_metadata`` block posted
beside a text, went with the text branch's side blocks on 2026-09-25: the
Results tab installs the server's envelope, so nothing sent one -- plan § 5q
D15.)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from molbuilder.parse.dirs.atom_metadata import atom_metadata_json_for_run_dir
from molbuilder.script_emit import _extract_atom_metadata_dict
from molbuilder.script_emit import emit_atom_metadata

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from support.envelope import (from_xyz as _env,
                             from_xyz_with_periodicity as _env_per)



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


def _md_json_pre_v7() -> str:
    """The SAME block as a run older than v7 would carry it.

    DERIVED from the current one, not typed: it then keeps describing the
    same four atoms if the writer changes, and the only difference is the
    one the version history names -- before v7 the reserved `frozen_atoms`
    label sat in a top-level key instead of inside `regions`.  No current
    code can emit this, which is the whole point: it is what an ALREADY
    FINISHED run has on disk, and nothing can go back and rewrite it.
    """
    from molbuilder.structure import FROZEN_LABEL
    block = json.loads(_md_json_4c())
    block["frozen_atoms"] = block["regions"].pop(FROZEN_LABEL)
    block["schema_version"] = 4
    return json.dumps(block)


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


@pytest.fixture()
def client():
    from molbuilder.web.app import create_app
    return create_app(config={}).test_client()


# --------------------------------------------------------------------- #
#  2. Results adapter: /api/watch/load surfaces atom_metadata           #
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


# `TestWatchLoadSurfacesMetadata` and `TestRunPeriodicityBridge` stood here until
# 2026-10-04: a run folder a test laid by hand -- a deck and a trajectory
# written beside it -- loaded as a directory, and a `.source` pair beside it
# for the axis kinds.  A folder no calculation claims holds no run of ours
# (plan B11), and what a run of ours declared -- its labels, its held atoms,
# its frame -- reaches the viewer from its own deck through the run door
# (plan B12), on the road (`process/testing.md` § 6).


