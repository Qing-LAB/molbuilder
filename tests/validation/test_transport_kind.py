"""The transport KIND's science — three rules, each about an open boundary.

`_validate_transport_kind` is registered in ``_KIND_VALIDATORS`` and keyed on
``task.calculation``, so it fires for whatever config class the deck renders
from. That matters here: its sibling ``_validate_transport`` is keyed on
``TransportConfig`` in ``_ENGINE_VALIDATORS``, and every rung has resolved a
``SiestaConfig`` since the seam migration — so the engine-keyed one dispatches
for no rung, and this is the only transport science that runs on a prep.

**It carried three rules and none of them had a test** (found 2026-09-16). Each
one below is a case where being wrong costs a queue wait and produces a
plausible number rather than a crash, which is `engines/transport.md` § 0.4's
whole argument for guards over advice.

Every test asserts through ``validate()`` — the door every render goes through
— rather than calling the private function, and each carries the half that
makes it discriminating: the same value must be *accepted* where it is
legitimate, or the rule would be satisfied by a validator that refuses
everything.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.config.siesta import SiestaConfig
from molbuilder.structure import Structure
from molbuilder.validation import validate


@pytest.fixture
def junction():
    """A labelled two-terminal junction: [L-electrode][bridge][R-electrode]."""
    return Structure(
        elements=["Au", "Au", "S", "C", "Au", "Au"],
        positions=np.array([[0.0, 0.0, 2.36 * i] for i in range(6)]),
        cell=np.array([[8.65, 0, 0], [0, 8.65, 0], [0, 0, 14.16]]),
        regions={"L-electrode": [0, 1], "bridge": [2, 3],
                 "R-electrode": [4, 5]})


def _find(issues, where, severity=None):
    """Issues at *where*, optionally of one severity.

    The severity filter is load-bearing for the k-grid rule: the shared SIESTA
    validator also speaks at ``config.kgrid`` -- it WARNS that a 6x6 mesh over
    an 8.7 A supercell is probably denser than the images justify -- and that
    advisory is correct and unrelated. The kind's rule is an ERROR. Matching on
    the address alone made this file's own negative half fail, which is the
    check working: two rules, one address.
    """
    return [i for i in issues if i.where == where
            and (severity is None or i.severity == severity)]


class TestTheTransportAxisIsNotSampled:
    """kz must be 1: that axis is the open boundary, not a Brillouin zone."""

    def test_a_sampled_transport_axis_is_refused(self, junction):
        cfg = SiestaConfig(system_label="j", kgrid=(2, 2, 4))
        found = _find(validate(junction, cfg, calculation="transport"),
                      "config.kgrid", "error")
        assert found, (
            "kz > 1 imposes a fake Bloch periodicity along the wire and "
            "gives physically wrong transmission -- and the renderer writes "
            "1 regardless, so without this the control silently does nothing")

    def test_the_transverse_pair_is_still_the_persons(self, junction):
        """Without this half, a validator that refused every k-grid would pass.

        The transverse counts are the person's and are shared by the leads and
        the device; only the third component is fixed.
        """
        cfg = SiestaConfig(system_label="j", kgrid=(6, 6, 1))
        assert not _find(validate(junction, cfg, calculation="transport"),
                         "config.kgrid", "error")

    def test_an_optimization_may_sample_all_three(self, junction):
        """The rule belongs to the KIND, not to the structure or the engine."""
        cfg = SiestaConfig(system_label="j", kgrid=(2, 2, 4))
        assert not _find(validate(junction, cfg, calculation="optimization"),
                         "config.kgrid", "error")


class TestANetChargeIsRefusedRatherThanDropped:
    """§ 2a.7 defers net charge and gating; the deck wrote nothing either way.

    Measured on 2026-09-16: an optimization deck carried ``NetCharge -2`` and
    all five transport decks carried no such line, while the validation report
    written beside the transport deck *asserted* the charge was there. So a
    charged junction ran neutral -- the transmission curve is simply for a
    different molecule, and TranSIESTA does not complain.
    """

    def test_a_charged_junction_is_refused_by_name(self, junction):
        cfg = SiestaConfig(system_label="j", kgrid=(2, 2, 1), net_charge=-2)
        found = _find(validate(junction, cfg, calculation="transport"),
                      "config.net_charge")
        assert found and found[0].severity == "error"
        assert "chemical potentials" in found[0].message, (
            "the refusal must say WHY -- the boundaries are open, so the "
            "electron count is the leads' to set -- and where to go instead")

    def test_a_neutral_junction_is_fine(self, junction):
        """The half that stops this being satisfied by refusing always."""
        for charge in (None, 0):
            cfg = SiestaConfig(system_label="j", kgrid=(2, 2, 1),
                               net_charge=charge)
            assert not _find(validate(junction, cfg, calculation="transport"),
                             "config.net_charge"), f"net_charge={charge!r}"

    def test_an_optimization_still_honours_a_charge(self, junction):
        """The keyword is real; it is this KIND that cannot carry it."""
        cfg = SiestaConfig(system_label="j", net_charge=-2)
        assert not _find(validate(junction, cfg, calculation="optimization"),
                         "config.net_charge")


class TestTheTransmissionGridIsAGrid:
    """Every value in this field is a real k-grid -- there is no sentinel.

    It shipped `0 0 0` meaning "inherit the SCF's grid" until 2026-09-16: a
    triple of zeros in a field labelled k-grid, where every value is a
    scientific fact and that is not one of them. The range admitted a zero per
    AXIS and the emitter gated on `any(...)`, so `0 4 1` and `4 4 0` were
    written into the deck verbatim -- asking tbtrans for zero k-points along
    an axis.

    `jobset init` fills it from the cited run's transverse pair and the deck
    always states it, so what is left to refuse is a grid that is not one.
    """

    @pytest.mark.parametrize("grid", [(0, 4, 1), (4, 0, 1)])
    def test_fewer_than_one_k_point_on_a_transverse_axis_is_refused(
            self, junction, grid):
        cfg = SiestaConfig(system_label="j", kgrid=(2, 2, 1), tbt_k_grid=grid)
        found = _find(validate(junction, cfg, calculation="transport"),
                      "config.tbt_k_grid", "error")
        assert found and "zero" in found[0].message

    @pytest.mark.parametrize("grid", [(4, 4, 0), (4, 4, 2)])
    def test_the_transport_component_must_be_one(self, junction, grid):
        """Same physics as the SCF grid's kz: that axis is not sampled."""
        cfg = SiestaConfig(system_label="j", kgrid=(2, 2, 1), tbt_k_grid=grid)
        found = _find(validate(junction, cfg, calculation="transport"),
                      "config.tbt_k_grid", "error")
        assert found and "TRANSPORT" in found[0].message

    @pytest.mark.parametrize("grid", [(1, 1, 1), (4, 4, 1), (12, 12, 1)])
    def test_a_real_grid_passes(self, junction, grid):
        """Without this a rule refusing every grid would satisfy the two above.

        `1 1 1` is the shipped default -- Gamma-only transverse, right for a
        finite molecule between leads -- so refusing it would refuse every
        untouched description.
        """
        cfg = SiestaConfig(system_label="j", kgrid=(2, 2, 1), tbt_k_grid=grid)
        assert not _find(validate(junction, cfg, calculation="transport"),
                         "config.tbt_k_grid", "error")


class TestThePoleEnergyIsTiedToTheTemperature:
    """The equilibrium contour's pole COUNT is derived, and 20 is a floor.

    TranSIESTA computes N = E / (pi kT) from the pole energy and refuses fewer
    than twenty, stopping with *"The continued fraction method requires at
    least 20 poles"* -- after the queue wait, having read the electrodes.

    MEASURED against SIESTA 5.4.2 on a real device run, 2026-09-16: 1.5 eV
    gives 18 poles and aborts; 1.7 gives 20; 2.0 gives 24; 4.0 gives 49; with
    nothing written the engine picks 42. **1.5 was the shipped default**, so
    every device run stopped.

    This is the one rule in this file that is a RELATION rather than a
    constant, which is why the second case below exists: a check hard-coded to
    1.63 eV would pass at 300 K and miss every other temperature.
    """

    @pytest.mark.parametrize("pole,temp,poles", [(1.5, 300.0, 18),
                                                 (1.7, 1000.0, 6)])
    def test_too_few_poles_is_refused_with_the_arithmetic(
            self, junction, pole, temp, poles):
        cfg = SiestaConfig(system_label="j", kgrid=(2, 2, 1),
                           negf_eq_pole_ev=pole, electronic_temperature=temp)
        found = _find(validate(junction, cfg, calculation="transport"),
                      "config.negf_eq_pole_ev", "error")
        assert found, f"{pole} eV at {temp} K is {poles} poles, under 20"
        assert f"{poles} poles" in found[0].message, (
            "the refusal must show the arithmetic -- a person told only "
            "'too small' cannot tell what to raise it to")

    @pytest.mark.parametrize("pole,temp", [(1.7, 300.0), (2.0, 300.0),
                                           (6.0, 1000.0)])
    def test_an_adequate_energy_passes(self, junction, pole, temp):
        """The half without which refusing everything would pass.

        1.7 eV at 300 K is the measured boundary -- exactly 20 poles -- so it
        also pins that the check is not off by one against the engine.
        """
        cfg = SiestaConfig(system_label="j", kgrid=(2, 2, 1),
                           negf_eq_pole_ev=pole, electronic_temperature=temp)
        assert not _find(validate(junction, cfg, calculation="transport"),
                         "config.negf_eq_pole_ev", "error")

    def test_zero_means_the_engine_chooses_and_is_never_refused(self,
                                                                junction):
        """The default, and the escape from the relation.

        The engine's own choice scales with the temperature; a fixed number
        cannot. A check that refused 0 would refuse every shipped default.
        """
        for temp in (300.0, 1000.0):
            cfg = SiestaConfig(system_label="j", kgrid=(2, 2, 1),
                               negf_eq_pole_ev=0.0,
                               electronic_temperature=temp)
            assert not _find(validate(junction, cfg, calculation="transport"),
                             "config.negf_eq_pole_ev", "error"), temp


class TestTheLinearResponseAdvisory:
    """|V| > 2 V is outside the Landauer regime — a warn, never a refusal.

    Its old home (`TransiestaEngine.preflight`) is keyed on ``TransportConfig``
    and fires for no rung, so a person could describe a 3 V scan and be told
    nothing on the road that runs. Bias is the one axis a transport
    calculation exists to sweep.
    """

    def test_a_high_bias_point_is_advised(self, junction):
        cfg = SiestaConfig(system_label="j", kgrid=(2, 2, 1),
                           bias_voltage_v=3.0)
        found = _find(validate(junction, cfg, calculation="transport"),
                      "config.bias_voltage_v")
        assert found and found[0].severity == "warn", (
            "an advisory, not a refusal: a high-bias run is a legitimate "
            "experiment whose RESULT must be read as a point on a nonlinear "
            "I-V curve rather than a conductance")

    def test_an_ordinary_bias_says_nothing(self, junction):
        for v in (0.0, 0.2, 2.0):
            cfg = SiestaConfig(system_label="j", kgrid=(2, 2, 1),
                               bias_voltage_v=v)
            assert not _find(validate(junction, cfg, calculation="transport"),
                             "config.bias_voltage_v"), f"V={v}"
