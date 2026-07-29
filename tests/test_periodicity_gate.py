"""The frame-contract gate — structure-periodicity.md § 6.1 (heal table)
+ § 6.2 v3 (the unified door's regime model).

Python owns every periodicity-metadata change (the gate); the JS only
calls.  These tests pin each heal-table row, each op's v3 semantics, the
calibrate≡emit frame equivalence, and the unified endpoint envelope.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.structure import Structure
from molbuilder.periodicity_gate import (
    OPS, apply_edit, contains_atoms, expected_corner, validate_and_heal)


def _mol(off=(10.0, 10.0, 10.0), vacuum=(2.5, 2.5, 2.5)):
    """Two H atoms 2 Å apart along x, offset from the world origin."""
    o = np.asarray(off, dtype=float)
    return Structure(
        elements=["H", "H"],
        positions=np.array([o, o + [2.0, 0.0, 0.0]]),
        vacuum=tuple(vacuum),
    )


# ------------------------------------------------------------------ #
#  § 6.1 heal table (stored state)                                    #
# ------------------------------------------------------------------ #


class TestHealTable:

    def test_derived_state_is_untouched(self):
        s = _mol()
        healed, notes = validate_and_heal(s)
        assert healed.cell is None and not notes    # row 1

    def test_explicit_no_origin_containing_is_legal(self):
        s = _mol(off=(1.0, 1.0, 1.0))
        s.cell = np.eye(3) * 10.0
        s.__post_init__()
        healed, notes = validate_and_heal(s)
        assert healed.cell_origin is None and not notes   # row 2

    def test_hemec_state_heals_to_expected_corner(self):
        """Row 3 — the 2026-07 hemeC corruption: explicit cell, no origin,
        atoms far outside [0, cell)."""
        s = _mol()                                   # atoms near (10,10,10)
        s.cell = np.eye(3) * 7.0
        s.__post_init__()
        healed, notes = validate_and_heal(s)
        assert np.allclose(healed.cell_origin, [7.5, 7.5, 7.5])  # 10 − 2.5
        assert notes and notes[0]["level"] == "warn"
        assert contains_atoms(healed, healed.cell_origin)

    def test_user_owned_origin_is_never_healed(self):
        s = _mol(off=(1.0, 1.0, 1.0))
        s.cell = np.eye(3) * 10.0
        s.cell_origin = np.array([0.5, 0.5, 0.5])
        s.__post_init__()
        healed, notes = validate_and_heal(s)
        assert np.allclose(healed.cell_origin, [0.5, 0.5, 0.5])  # row 4
        assert not notes

    def test_live_origin_edit_is_accepted_with_warning(self):
        s = _mol()
        s.cell = np.eye(3) * 7.0
        s.cell_origin = np.array([100.0, 100.0, 100.0])  # nonsense corner
        s.__post_init__()
        healed, notes = validate_and_heal(s, live_edit=True)
        assert np.allclose(healed.cell_origin, [100.0, 100.0, 100.0])
        assert notes and notes[0]["level"] == "warn"     # row 5, live half

    def test_too_small_cell_is_a_hard_error(self):
        s = _mol()
        s.cell = np.eye(3) * 1.0                     # extent 2 Å can't fit
        s.__post_init__()
        with pytest.raises(ValueError, match="cannot contain"):
            validate_and_heal(s)

    def test_left_handed_cell_is_refused(self):
        s = _mol()
        s.cell = np.diag([7.0, 7.0, -7.0])
        s.__post_init__()
        with pytest.raises(ValueError, match="right-handed"):
            validate_and_heal(s)


# ------------------------------------------------------------------ #
#  § 6.2 v3 op semantics (the regime model)                           #
# ------------------------------------------------------------------ #


class TestApplyEditV3:

    def test_calibrate_is_not_a_periodicity_op(self):
        assert "calibrate" not in OPS
        with pytest.raises(ValueError, match="unknown periodicity op"):
            apply_edit(_mol(), "calibrate", None)

    def test_vacuum_edit_resets_manual_state_to_derived(self):
        s = _mol(off=(1.0, 1.0, 1.0))
        s.cell = np.eye(3) * 10.0
        s.cell_origin = np.array([0.5, 0.5, 0.5])
        s.__post_init__()
        out, notes = apply_edit(s, "vacuum", [3.0, 3.0, 3.0])
        assert out.cell is None and out.cell_origin is None
        assert out.vacuum == (3.0, 3.0, 3.0)
        assert any("DERIVED regime" in n["message"] for n in notes)

    def test_vacuum_edit_refused_on_a_periodic_axis(self):
        s = _mol()
        s.cell = np.diag([10.0, 10.0, 4.0])
        s.axis_kind = ("isolated", "isolated", "periodic")
        s.__post_init__()
        with pytest.raises(ValueError, match="periodic"):
            apply_edit(s, "vacuum", [3.0, 3.0, 3.0])

    def test_axis_edit_resets_to_derived_when_nonperiodic(self):
        s = _mol(off=(1.0, 1.0, 1.0))
        s.cell = np.eye(3) * 10.0
        s.__post_init__()
        out, notes = apply_edit(
            s, "axis_kind", ["isolated", "isolated", "isolated"])
        assert out.cell is None and out.cell_origin is None
        assert any("DERIVED regime" in n["message"] for n in notes)
        # A transport axis with zero structure extent would be a
        # degenerate derived box -> refused (its own pin below).

    def test_axis_to_periodic_keeps_the_explicit_cell(self):
        s = _mol(off=(1.0, 1.0, 1.0))
        s.cell = np.eye(3) * 10.0
        s.__post_init__()
        out, notes = apply_edit(
            s, "axis_kind", ["isolated", "isolated", "periodic"])
        assert out.cell is not None                      # respected, kept
        assert any("respected" in n["message"] for n in notes)

    def test_axis_to_periodic_without_a_cell_is_refused(self):
        with pytest.raises(ValueError, match="explicit commensurate cell"):
            apply_edit(_mol(), "axis_kind",
                       ["isolated", "isolated", "periodic"])

    def test_cell_edit_respects_existing_origin_first(self):
        s = _mol(off=(1.0, 1.0, 1.0))
        s.cell = np.eye(3) * 10.0
        s.cell_origin = np.array([0.5, 0.5, 0.5])
        s.__post_init__()
        out, _ = apply_edit(s, "cell", (np.eye(3) * 12.0).tolist())
        assert np.allclose(out.cell_origin, [0.5, 0.5, 0.5])

    def test_cell_edit_without_origin_respects_vacuum(self):
        s = _mol()                                   # atoms near (10,10,10)
        out, notes = apply_edit(s, "cell", (np.eye(3) * 8.0).tolist())
        assert np.allclose(out.cell_origin, expected_corner(out))
        assert np.allclose(out.cell_origin, [7.5, 7.5, 7.5])
        assert any("origin first, then vacuum" in n["message"]
                   for n in notes)

    def test_cell_null_returns_to_derived(self):
        s = _mol()
        s.cell = np.eye(3) * 10.0
        s.__post_init__()
        out, _ = apply_edit(s, "cell", None)
        assert out.cell is None and out.cell_origin is None

    def test_origin_edit_warns_vacuum_not_respected(self):
        s = _mol(off=(1.0, 1.0, 1.0))
        s.cell = np.eye(3) * 10.0
        s.__post_init__()
        out, notes = apply_edit(s, "cell_origin", [0.2, 0.2, 0.2])
        assert np.allclose(out.cell_origin, [0.2, 0.2, 0.2])
        assert any("NOT respected" in n["message"] for n in notes)

    def test_reset_origin_clears_to_none(self):
        s = _mol(off=(1.0, 1.0, 1.0))
        s.cell = np.eye(3) * 10.0
        s.cell_origin = np.array([0.5, 0.5, 0.5])
        s.__post_init__()
        out, notes = apply_edit(s, "cell_origin", None)
        assert out.cell_origin is None
        assert any("freedom back" in n["message"] for n in notes)

    def test_no_op_ever_moves_atoms(self):
        s = _mol()
        out, _ = apply_edit(s, "vacuum", [1.0, 1.0, 1.0])
        assert np.allclose(out.positions, s.positions)
        withcell, _ = apply_edit(s, "cell", (np.eye(3) * 9.0).tolist())
        assert np.allclose(withcell.positions, s.positions)
        for op, payload in [("cell_origin", [7.0, 7.0, 7.0]),
                            ("cell_origin", None),
                            ("cell", None)]:
            out, _ = apply_edit(withcell, op, payload)
            assert np.allclose(out.positions, s.positions), op


# ------------------------------------------------------------------ #
#  Calibrate ≡ emit — the implicit-translation equivalence            #
# ------------------------------------------------------------------ #


def _coords_block(fdf: str) -> str:
    lines = fdf.splitlines()
    a = next(i for i, l in enumerate(lines)
             if "%block AtomicCoordinatesAndAtomicSpecies" in l)
    b = next(i for i, l in enumerate(lines[a:], start=a)
             if "%endblock" in l)
    return "\n".join(lines[a:b + 1])


def test_calibrated_then_emit_equals_emit():
    """§ 6.2: emission translates implicitly — a user who never clicks
    calibrate gets the identical SIESTA frame."""
    from molbuilder.modify import calibrate_to_cell
    from molbuilder.siesta.input import render_fdf
    from molbuilder.config.siesta import SiestaConfig
    s = _mol()
    cfg = SiestaConfig(verbose_comments=False)
    direct = _coords_block(render_fdf(s, cfg))
    baked = _coords_block(render_fdf(calibrate_to_cell(s), cfg))
    assert direct == baked


# ------------------------------------------------------------------ #
#  The unified endpoint                                               #
# ------------------------------------------------------------------ #


class TestPeriodicityDoor:

    @pytest.fixture
    def client(self):
        pytest.importorskip("flask")
        from molbuilder.web.app import create_app
        return create_app(config={}).test_client()

    def _blob(self, struct):
        from molbuilder.workingcopy_structure import StructureCodec
        return StructureCodec().scratch_blob(struct)

    def test_vacuum_op_round_trips_the_truth_blob(self, client):
        s = _mol(off=(1.0, 1.0, 1.0))
        s.cell = np.eye(3) * 10.0                    # manual state...
        s.__post_init__()
        r = client.post("/api/structure/periodicity", json={
            "data": self._blob(s), "op": "vacuum",
            "payload": [3.0, 3.0, 3.0]})
        assert r.status_code == 200
        j = r.get_json()
        assert j["ok"] is True
        assert j["blob"]["sidecar"]["cell"] is None  # ...reset to derived
        assert j["blob"]["sidecar"]["vacuum"] == [3.0, 3.0, 3.0]
        assert j["resolved_cell"] is not None        # the view, recomputed
        assert any(n["level"] == "warn" for n in j["notices"])

    def test_unknown_op_is_a_400(self, client):
        r = client.post("/api/structure/periodicity", json={
            "data": self._blob(_mol()), "op": "calibrate"})
        assert r.status_code == 400


# ------------------------------------------------------------------ #
#  The LOADER gate (§ 6.1 clause 1-2) — the live hemeC symptom        #
# ------------------------------------------------------------------ #


class TestLoaderGate:
    """The gate must sit on BOTH pair seams.  from_scratch alone left
    /api/build/load serving corrupted pairs unhealed — MolView drew the
    box from the world origin while the Cell page showed healed values
    (observed live on projects/hemeC-dithiol, 2026-07-29)."""

    def _write_corrupted_pair(self, dirpath):
        import json as _json
        from molbuilder.workingcopy_structure import StructureCodec
        s = _mol()                          # atoms near (10,10,10), vac 2.5
        s.cell = np.eye(3) * 7.0            # explicit cell, no origin,
        s.__post_init__()                   # atoms far outside [0, cell)
        blob = StructureCodec().scratch_blob(s)
        (dirpath / "m.xyz").write_text(blob["xyz"], encoding="utf-8")
        (dirpath / "m.molstruct.json").write_text(
            _json.dumps(blob["sidecar"]), encoding="utf-8")
        return dirpath / "m.xyz"

    def test_codec_read_heals_a_corrupted_pair(self, tmp_path):
        from molbuilder.workingcopy_structure import StructureCodec
        xyz = self._write_corrupted_pair(tmp_path)
        healed = StructureCodec().read(xyz)
        assert np.allclose(healed.cell_origin, [7.5, 7.5, 7.5])
        assert contains_atoms(healed, healed.cell_origin)

    def test_load_endpoint_serves_the_healed_resolved_origin(
            self, tmp_path, monkeypatch):
        pytest.importorskip("flask")
        from molbuilder.diagnostics import Capabilities, set_capabilities
        monkeypatch.chdir(tmp_path)
        sdir = tmp_path / "projects" / "P" / "structure"
        sdir.mkdir(parents=True)
        xyz = self._write_corrupted_pair(sdir)
        set_capabilities(Capabilities(runtime_config={},
                                      conda_binary="/usr/bin/conda"))
        try:
            from molbuilder.web.app import create_app
            client = create_app(config={}).test_client()
            r = client.post("/api/build/load", json={"path": str(xyz)})
            assert r.status_code == 200, r.get_json()
            per = r.get_json()["periodicity"]
            assert per["resolved_cell_origin"] is not None
            assert np.allclose(per["resolved_cell_origin"],
                               [7.5, 7.5, 7.5])
        finally:
            set_capabilities(None)


# ------------------------------------------------------------------ #
#  Per-axis-kind containment (review findings, 2026-07-29)            #
# ------------------------------------------------------------------ #


class TestPeriodicAxesAreNeverContained:
    """Along a periodic axis, atoms outside [0, cell) are periodic
    images — legal.  Requiring containment there made real crystal and
    junction files unopenable."""

    def test_periodic_crystal_with_outside_atoms_is_legal(self):
        s = _mol(off=(-3.0, 25.0, 1.0), vacuum=(0.0, 0.0, 0.0))
        s.cell = np.eye(3) * 10.0
        s.axis_kind = ("periodic", "periodic", "periodic")
        s.__post_init__()
        healed, notes = validate_and_heal(s)
        assert healed.cell_origin is None and not notes

    def test_junction_periodic_periodic_transport_is_legal(self):
        """The BDT/Au junction shape: periodic x/y, transport z; atoms
        wrapped to negative fractionals along x/y must not trip the
        gate; the transport axis still heals via the corner."""
        s = _mol(off=(-3.0, -2.0, 10.0), vacuum=(0.0, 0.0, 0.0))
        s.cell = np.diag([10.0, 10.0, 2.0])
        s.axis_kind = ("periodic", "periodic", "transport")
        s.__post_init__()
        healed, notes = validate_and_heal(s)
        # x/y ignored (periodic); z extent 0 fits; origin healed to
        # bbox_min on the transport axis only where needed.
        assert contains_atoms(healed, healed.cell_origin)

    def test_stored_manual_origin_is_warned_never_healed(self):
        """Row 5 stored half: a manual origin round-trips verbatim (the
        silent flip on reload is the review finding)."""
        s = _mol()
        s.cell = np.eye(3) * 7.0
        s.cell_origin = np.array([100.0, 100.0, 100.0])
        s.__post_init__()
        healed, notes = validate_and_heal(s)          # NOT live_edit
        assert np.allclose(healed.cell_origin, [100.0, 100.0, 100.0])
        assert notes and notes[0]["level"] == "warn"

    def test_unfittable_cell_edit_is_refused_not_stored(self):
        """A cell the structure cannot fit is refused at the edit — a
        stored-but-invalid cell locked every later door."""
        s = _mol()                                    # extent 2 Å
        with pytest.raises(ValueError, match="cannot contain"):
            apply_edit(s, "cell", (np.eye(3) * 1.0).tolist())

    def test_reset_to_derived_refuses_a_degenerate_axis(self):
        s = _mol(vacuum=(2.5, 2.5, 0.0))              # extent 0 on y,z
        s.cell = np.eye(3) * 10.0
        s.__post_init__()
        with pytest.raises(ValueError, match="degenerate"):
            apply_edit(s, "vacuum", [3.0, 3.0, 0.0])
