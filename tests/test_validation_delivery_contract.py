"""The facts half of the delivery contract (docs/science/validation.md § 4.1).

F1/F2 inbound (the facts a validating request carries) and F4 (the gate derives
what a check needs).  Each test names the invisible failure it prevents.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.config.siesta import SiestaConfig
from molbuilder.structure import Structure
from molbuilder.validation import validate


def _thin_box_molecule() -> Structure:
    """A molecule in a box too small for its own basis orbitals -- the
    hemeC-dithiol shape that reached SIESTA unremarked (2.5 Å/side)."""
    rng = np.random.default_rng(0)
    pos = rng.uniform(-4.0, 4.0, size=(12, 3))
    return Structure(elements=["C"] * 12, positions=pos,
                     vacuum=(2.5, 2.5, 2.5))


class TestF4GateDerivesWhatChecksNeed:
    """PINS: docs/science/validation.md § 4.1 clause F4 — derived facts are
    derived server-side, from the facts.

    INVARIANT: a check can never be switched off by a caller omitting an
    argument.  ``validate()`` resolves the structure's own cell when none is
    passed, so the cell-dependent checks (volume, determinant, min
    atom-to-nearest-image) always run; an explicit cell still wins, for a
    generator validating a box that differs from the structure's.

    PREVENTS: the invisible failure that started this contract — every web
    caller omitted ``cell``, so the image-distance check had never once been
    shown in the browser and a 2.5 Å box reached SIESTA unremarked.
    """

    def test_cell_dependent_checks_run_without_a_cell_argument(self):
        """THE bug: every web caller omitted ``cell``, so the volume /
        determinant / image-distance checks silently never ran."""
        issues = validate(_thin_box_molecule(), SiestaConfig())
        wheres = {i.where for i in issues}
        assert "cell.image_distance" in wheres, (
            "the image-distance check did not run -- a caller must not be able "
            "to switch a check off by omitting an argument (F4)")

    def test_an_explicit_cell_still_wins(self):
        """A generator validating a box that differs from the structure's own
        must still be able to say so."""
        s = _thin_box_molecule()
        roomy = np.eye(3) * 60.0
        wheres = {i.where for i in validate(s, SiestaConfig(), cell=roomy)}
        assert "cell.image_distance" not in wheres

    def test_an_unresolvable_cell_is_reported_not_skipped_silently(self):
        """Silence is never the answer: a structure whose cell cannot be
        resolved says so as info."""
        s = Structure(elements=["H", "H"],
                      positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.8]]))
        s.axis_kind = ("periodic", "isolated", "isolated")   # no lattice
        s.__post_init__()
        issues = validate(s, SiestaConfig())
        assert any(i.where == "cell.unresolved" and i.severity == "info"
                   for i in issues), [i.where for i in issues]


class TestR5FindingsAreNeverWarnings:
    """PINS: docs/science/validation.md § 4.1 clause R5 — one channel means one
    channel.

    INVARIANT: a scientific finding travels as an ``Issue`` from a validator,
    never as a Python ``warnings.warn``.  A warning cannot reach a web user at
    all; as an Issue the same advice reaches BOTH surfaces — the browser panel
    through the endpoint's ``issues[]`` and the CLI through ``render_fdf``'s own
    ``report(validate(...))``.
    """

    def test_thin_vacuum_is_an_issue_not_a_python_warning(self):
        """It used to be warnings.warn inside render_fdf: server stderr only,
        invisible to every web user."""
        import warnings
        from molbuilder.siesta import render_fdf
        s = _thin_box_molecule()
        wheres = {i.where for i in validate(s, SiestaConfig())}
        assert "cell.vacuum_thin" in wheres
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            render_fdf(s, SiestaConfig())
        vacuum_warnings = [w for w in caught
                           if "vacuum" in str(w.message).lower()]
        assert not vacuum_warnings, (
            "the emitter still raises a Python warning for vacuum; a finding "
            "must travel as an Issue so the web surfaces it (R5)")

    def test_no_emitter_reaches_for_warnings_warn_about_findings(self):
        import pathlib
        src = pathlib.Path("molbuilder/siesta/input.py").read_text()
        assert "_warn_insufficient_vacuum" not in src


class TestF2NoSecondSource:
    """PINS: docs/science/validation.md § 4.1 clause F2 — no server-side second
    source.

    INVARIANT: an emitted / validated structure is built from the request body
    alone.  The ``.molstruct.json`` sidecar is never read for it, so the deck can
    never mix body geometry with disk labels the model has since changed, and a
    body with no label keys declares NO labels.

    NOTE: there is deliberately no server-side refusal keyed on
    ``structure_path`` — that field also anchors pseudopotential and dest-dir
    resolution, so refusing on it rejected legitimate callers.  Loudness lives on
    the side that can guarantee it (F1, below).
    """

    def test_a_body_without_label_keys_declares_no_labels(self):
        """No second source: absent keys mean "this structure has no labels",
        NOT "go and read the sidecar"."""
        from molbuilder.web.blueprints._shared import apply_labels_to_struct
        s = _thin_box_molecule()
        assert apply_labels_to_struct(
            s, {"xyz": "...", "structure_path": "/tmp/whatever.xyz"}) is None
        assert not s.frozen_atoms
        assert not s.regions

    def test_a_sidecar_on_disk_never_reaches_an_emitted_deck(self, tmp_path):
        """The property F2 actually guarantees, end to end: labels sitting in a
        .molstruct.json next to the .xyz do NOT appear in the emitted input.

        Before F2 the server read that sidecar whenever the body omitted the
        keys, so a deck could be emitted with frozen atoms the model had already
        cleared -- a structure nobody was looking at."""
        import json as _json
        pytest.importorskip("flask")
        from molbuilder.diagnostics import Capabilities, set_capabilities
        xyz_text = ("3\nprobe\nO 0.0 0.0 0.0\n"
                    "H 0.757 0.586 0.0\nH -0.757 0.586 0.0\n")
        xyz = tmp_path / "probe.xyz"
        xyz.write_text(xyz_text, encoding="utf-8")
        import hashlib
        (tmp_path / "probe.molstruct.json").write_text(_json.dumps({
            "schema_version": 3, "n_atoms_total": 3,
            "structure_hash": hashlib.sha256(xyz.read_bytes()).hexdigest(),
            "regions": {"L-electrode": [0]}, "frozen_atoms": [0],
            "selection_rules": {},
        }), encoding="utf-8")
        set_capabilities(Capabilities(runtime_config={},
                                      conda_binary="/usr/bin/conda"))
        try:
            from molbuilder.web.app import create_app
            client = create_app(config={}).test_client()
            r = client.post("/api/build/fdf", json={
                "xyz": xyz_text, "params": {},
                "structure_path": str(xyz),      # named, but NOT a label source
            })
            assert r.status_code == 200, r.data
            fdf = (r.get_json() or {}).get("fdf", "")
            assert "%block Geometry.Constraints" not in fdf, (
                "the disk sidecar's frozen atom leaked into the emitted deck")
        finally:
            set_capabilities(None)


class TestF1TheTabHoldsNoStructuralMirror:
    """PINS: docs/science/validation.md § 4.1 clause F1 — one fact holder.

    INVARIANT: coordinates, labels and periodicity are read from
    ``molview.data.factsForRequest()`` at request time.  No page keeps its own
    copy, and one accessor assembles the whole payload so a tab cannot send a
    PARTIAL set of facts.

    PREVENTS: the stale-geometry bug — the structure-optimization tab read labels
    and periodicity live but mirrored the geometry into ``state.xyz`` once at
    load, so a request could carry fresh labels, fresh periodicity and stale
    coordinates, and validation then judged a structure the viewer was not
    showing.
    """

    def test_structure_optimization_has_no_geometry_mirror(self):
        import pathlib
        src = pathlib.Path(
            "molbuilder/web/static/structure-optimization/viewer.js"
        ).read_text()
        # The mirror is gone from the state object...
        assert "xyz: null," not in src, (
            "the tab re-grew a page-local geometry mirror; read it live from "
            "molview.data.factsForRequest() instead (F1)")
        # ...and the live accessor is what request bodies use.
        assert "_liveXyz()" in src and "factsForRequest" in src

    def test_the_model_exposes_one_fact_payload(self):
        import pathlib
        src = pathlib.Path(
            "molbuilder/web/static/lib/molview/data-model.js").read_text()
        assert "factsForRequest:       factsForRequest" in src
        block = src.split("function factsForRequest()")[1][:900]
        for key in ("xyz:", "regions:", "frozen_atoms:", "periodicity:"):
            assert key in block, f"the fact payload omits {key}"


class TestF4DerivesOnlyABoxTheStructureAskedFor:
    """PINS: docs/science/validation.md § 4.1 clause F4, second half — the gate
    derives a cell only when the structure DECLARES a box (an explicit cell, a
    non-zero vacuum, or a non-isolated axis).

    The default must not INVENT a cell for a molecule in free space -- a
    planar gas-phase molecule's bounding box has zero thickness, and a
    determinant check on it reported a degenerate cell for a calculation that
    has no cell (caught by the spectra suite when F4 first landed)."""

    def test_planar_gas_phase_molecule_gets_no_cell_checks(self):
        from molbuilder.config.pyscf import PySCFConfig
        water = Structure(                     # planar: z extent is exactly 0
            elements=["O", "H", "H"],
            positions=np.array([[0.0, 0.0, 0.0],
                                [0.757, 0.586, 0.0],
                                [-0.757, 0.586, 0.0]]))
        wheres = {i.where for i in validate(water, PySCFConfig())}
        assert "cell.determinant" not in wheres, (
            "a gas-phase molecule that never asked for a box must not be "
            "judged against one")
        assert "cell.volume" not in wheres

    def test_a_declared_vacuum_does_bring_the_cell_checks_back(self):
        """The hemeC case: asking for vacuum IS asking for a box."""
        s = _thin_box_molecule()
        assert {"cell.image_distance", "cell.vacuum_thin"} <= {
            i.where for i in validate(s, SiestaConfig())}

    def test_an_explicit_cell_declares_a_box_even_with_no_vacuum(self):
        s = Structure(elements=["C"] * 2,
                      positions=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]))
        s.cell = np.eye(3) * 3.0
        s.__post_init__()
        wheres = {i.where for i in validate(s, SiestaConfig())}
        assert "cell.image_distance" in wheres


class TestSpatialAdequacyIsAdvisoryNotBlocking:
    """PINS: docs/science/validation.md § 4.1 clause R4, spatial-severity rule
    (owner's decision, 2026-07-29): "the validation of space is warning, not
    blocking".

    INVARIANT: a check about how MUCH space there is — vacuum thinness, image
    distance, cell tightness — is a `warn` and never stops a run.  The cell is
    well-formed; what is in question is the physics quality of the result, and
    that is the user's call (probing convergence, reproducing a tight-box paper,
    deliberately accepting image interaction).  A check about whether the cell
    can EXIST — zero volume, left-handed — stays an `error`, because a
    zero-volume lattice makes SIESTA fail when it builds reciprocal vectors, so
    a warning there would hand over a guaranteed-failed run dressed as a choice.

    PREVENTS: the two opposite failures — nagging (blocking a legitimate tight
    box) and dangerous (emitting an unrunnable cell with a shrug).
    """

    _ADEQUACY = ("cell.vacuum_thin", "cell.image_distance", "cell.volume")

    def test_every_adequacy_finding_is_a_warning(self):
        s = _thin_box_molecule()          # 2.5 Å/side: trips all three
        found = {i.where: i.severity for i in validate(s, SiestaConfig())
                 if i.where in self._ADEQUACY}
        assert found, "expected at least one adequacy finding for a 2.5 Å box"
        for where, severity in found.items():
            assert severity == "warn", (
                f"{where} is {severity!r}; space adequacy is advisory — it must "
                f"not block a run the user may want for good reasons")

    def test_a_thin_box_still_emits_and_does_not_raise(self):
        """The whole point: the deck is produced, with the warnings attached."""
        import warnings
        from molbuilder.siesta import render_fdf
        s = _thin_box_molecule()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fdf = render_fdf(s, SiestaConfig())      # must NOT raise
        assert "%block LatticeVectors" in fdf

    def test_representability_stays_blocking(self):
        """The boundary: 'this is not a box' is an error, not advice."""
        import numpy as np
        from molbuilder.validation import validate_geometry
        bad = [i for i in validate_geometry(_thin_box_molecule(),
                                            np.diag([8.0, 8.0, 0.0]))
               if i.where == "cell.determinant"]
        assert bad and bad[0].severity == "error", bad
