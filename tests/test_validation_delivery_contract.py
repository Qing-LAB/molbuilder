"""The facts half of the delivery contract (docs/science/validation.md § 4.1).

F2 inbound (the facts a validating request carries) and F4 (the gate derives
what a check needs).  Each test names the invisible failure it prevents.

What prep's gate judges of a deck -- the frame it writes, an ion's dipole, whom
the dipole advisory is for -- is prepared on the road, rows of
``data/prep_protocol.toml``.
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
        issues = validate(_thin_box_molecule(), SiestaConfig(),
                          calculation="optimization")
        wheres = {i.where for i in issues}
        assert "cell.image_distance" in wheres, (
            "the image-distance check did not run -- a caller must not be able "
            "to switch a check off by omitting an argument (F4)")

    def test_an_explicit_cell_still_wins(self):
        """A generator validating a box that differs from the structure's own
        must still be able to say so."""
        s = _thin_box_molecule()
        roomy = np.eye(3) * 60.0
        wheres = {i.where for i in validate(s, SiestaConfig(), cell=roomy,
                                                calculation="optimization")}
        assert "cell.image_distance" not in wheres

    def test_an_unresolvable_cell_is_reported_not_skipped_silently(self):
        """Silence is never the answer: a structure whose cell cannot be
        resolved says so as info."""
        s = Structure(elements=["H", "H"],
                      positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.8]]))
        s.axis_kind = ("periodic", "isolated", "isolated")   # no lattice
        s.__post_init__()
        issues = validate(s, SiestaConfig(), calculation="optimization")
        assert any(i.where == "cell.unresolved" and i.severity == "info"
                   for i in issues), [i.where for i in issues]

    @pytest.mark.parametrize("engine,kind", [
        ("siesta", "optimization"),
        ("siesta", "vibration"),
        ("pyscf",  "optimization"),
        ("pyscf",  "vibration"),
    ])
    def test_an_unresolvable_species_label_is_reported_not_a_crash(
            self, engine, kind):
        """The same rule one level down: a label naming no element is a
        finding, and the checks that cannot be answered without it stand
        down rather than taking the request with them.

        The electron count is the electronic state's alone,
        asked by `validate` for every kind; the kinds stay here because
        the vibration route reads the state through its own view too.
        """
        if engine == "pyscf":
            from molbuilder.config.pyscf import PySCFConfig
            cfg = PySCFConfig()
        else:
            cfg = SiestaConfig(system_label="probe",
                               spin_treatment="unrestricted")
        s = Structure(
            elements=["Au", "Au", "Au", "Au", "Xx"],
            positions=np.array([[0.0, 0.0, 0.0], [2.9, 0.0, 0.0],
                                [0.0, 2.9, 0.0], [2.9, 2.9, 0.0],
                                [0.0, 0.0, 3.5]]),
            vacuum=(12.0, 12.0, 12.0))
        issues = validate(s, cfg, calculation=kind)      # must not raise
        named = [i for i in issues
                 if i.where == "chemistry.species_label"
                 and i.severity == "error"]
        assert named, [f"{i.where}/{i.severity}" for i in issues]
        assert "Xx" in named[0].message


class TestF2NoSecondSource:
    """PINS: docs/science/validation.md § 4.1 clause F2 — no server-side second
    source.

    INVARIANT: an emitted / validated structure is built from the request body
    alone.  The ``.molstruct.json`` sidecar is never read for it, so the deck can
    never mix body geometry with disk labels the model has since changed, and a
    body with no label keys declares NO labels.
    """

    def test_a_structure_without_labels_declares_no_labels(self):
        """No second source: a structure carrying no labels HAS no labels.

        F2 is a property of the SHAPE: the labels are a field of the structure,
        so "no labels" is just an empty field, and there is no other key that
        could have carried them."""
        from molbuilder.web.blueprints._shared import struct_from_body
        s = struct_from_body({
            "structure": {"elements": ["O", "H", "H"],
                          "positions": [[0, 0, 0], [0.76, 0.59, 0],
                                        [-0.76, 0.59, 0]],
                          "metadata": {}},
        })
        assert not s.frozen_atoms
        assert not s.regions


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
        wheres = {i.where for i in validate(water, PySCFConfig(),
                                                calculation="optimization")}
        # ANY cell finding means a box was invented -- the least it would say
        # is `cell.vacuum_defaulted`.
        assert not any(w.startswith("cell.") for w in wheres), (
            "a gas-phase molecule that never asked for a box must not be "
            f"judged against one: {sorted(wheres)}")

    def test_a_declared_vacuum_does_bring_the_cell_checks_back(self):
        """The hemeC case: asking for vacuum IS asking for a box."""
        s = _thin_box_molecule()
        assert {"cell.image_distance", "cell.vacuum_thin"} <= {
            i.where for i in validate(s, SiestaConfig(), calculation="optimization")}

    def test_an_explicit_cell_declares_a_box_even_with_no_vacuum(self):
        s = Structure(elements=["C"] * 2,
                      positions=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]))
        s.cell = np.eye(3) * 3.0
        s.__post_init__()
        wheres = {i.where for i in validate(s, SiestaConfig(), calculation="optimization")}
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
        found = {i.where: i.severity for i in validate(s, SiestaConfig(), calculation="optimization")
                 if i.where in self._ADEQUACY}
        assert found, "expected at least one adequacy finding for a 2.5 Å box"
        for where, severity in found.items():
            assert severity == "warn", (
                f"{where} is {severity!r}; space adequacy is advisory — it must "
                f"not block a run the user may want for good reasons")


    def test_representability_stays_blocking(self):
        """The boundary: 'this is not a box' is an error, not advice."""
        import numpy as np
        from molbuilder.cell import check, resolve
        # The verdict is the ONE checker's, and "no volume" and "mirrored"
        # are different repairs, so each has its own id.
        rc = resolve(_thin_box_molecule(), box=np.diag([8.0, 8.0, 0.0]))
        bad = [i for i in check(rc) if i.where == "cell.no_volume"]
        assert bad and bad[0].severity == "error", bad
