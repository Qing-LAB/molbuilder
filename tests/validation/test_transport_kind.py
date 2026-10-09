"""The transport KIND's science — each rule about an open boundary.

`_validate_transport_kind` is registered in ``_KIND_VALIDATORS`` and keyed on
``task.calculation``, so it fires for whatever config class the deck renders
from, and it is the only transport science that runs on a prep.

Each rule below is a case where being wrong costs a queue wait and produces a
plausible number rather than a crash, which is `engines/transport.md` § 0.4's
whole argument for guards over advice.

Every test asserts through ``validate()`` — the door every render goes through
— rather than calling the private function, and each carries the half that
makes it discriminating: the same value must be *accepted* where it is
legitimate, or the rule would be satisfied by a validator that refuses
everything.

The k-point sampling -- the transport axis's one point, a lead's own count
and the transmission's grid -- is the k-point mesh's (`engines/siesta.md`
§ 6.1), refused on every door; it has no road case here --
a transport calculation cites a finished relaxation, which only a real run makes -- and `tests/test_k_point_mesh.py` holds a relaxation's mesh only.
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
    """Issues at *where*, optionally of one severity -- a rule's own verdict
    beside the advisories other checks may give at the same address."""
    return [i for i in issues if i.where == where
            and (severity is None or i.severity == severity)]


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


class TestTheCellWrapsAlongTransport:
    """I12.

    A junction's leads continue into the periodic image, so empty space along
    z is a SEVERED lead rather than padding. This is the reverse of what an
    isolated molecule is told, which is the reason it is keyed on the
    calculation kind: `cell.vacuum_thin` asks a molecule for *more* vacuum and
    is right to.
    """

    @staticmethod
    def _with_c(junction, c_ang):
        s = Structure(elements=list(junction.elements),
                      positions=junction.positions.copy(),
                      cell=np.array([[8.65, 0, 0], [0, 8.65, 0], [0, 0, c_ang]]),
                      regions=dict(junction.regions))
        return s


    def test_a_seamless_cell_says_nothing(self, junction):
        """The discriminating half -- and the number is derived from the
        structure, so the test cannot pass by agreeing with a constant."""
        span = float(junction.positions[:, 2].max()
                     - junction.positions[:, 2].min())
        s = self._with_c(junction, span + 2.36)      # one interlayer spacing
        assert not _find(validate(s, SiestaConfig(system_label="j"),
                                  calculation="transport"),
                         "cell.transport_vacuum")

    def test_a_fused_cell_is_refused_and_its_relaxation_warned(self, junction):
        """I12 the other way: c equal to the atoms' span puts the two leads'
        end layers in one plane through the image -- a collision the engines
        only note by removing couplings (the 2026-10-08 road walk ran its
        whole ladder on one).  A transport rung refuses it, naming the c
        that closes the boundary to one spacing; the junction's relaxation
        is warned, since the structure has not committed to transport."""
        span = float(junction.positions[:, 2].max()
                     - junction.positions[:, 2].min())
        s = self._with_c(junction, span)
        refused = _find(validate(s, SiestaConfig(system_label="j"),
                                 calculation="transport"),
                        "cell.transport_collision")
        assert refused and refused[0].severity == "error"
        assert f"{span + 2.36:.3f}"[:4] in refused[0].message
        warned = _find(validate(s, SiestaConfig(system_label="j"),
                                calculation="optimization"),
                       "cell.transport_collision")
        assert warned and warned[0].severity == "warn"

    def test_an_isolated_molecule_is_not_told_the_opposite(self, junction):
        """The same roomy cell, as an OPTIMIZATION, draws no transport
        complaint -- the two pieces of advice contradict each other and must
        never both fire."""
        span = float(junction.positions[:, 2].max()
                     - junction.positions[:, 2].min())
        s = self._with_c(junction, span + 9.0)
        assert not _find(validate(s, SiestaConfig(system_label="j"),
                                  calculation="optimization"),
                         "cell.transport_vacuum")


def test_the_transmission_window_must_reach_the_bias_window(junction):
    """`engines/transport.md` § 2a.10 (P2): the current is TBtrans's integral
    over the deck's window, weighted by both leads' Fermi functions at
    μ = ±V/2, and TBtrans says nothing when the window cuts it short -- so
    the transmission deck's gate refuses a window that does not reach
    ±(|V|/2 + 5 kT).  At 0.4 V and 300 K the reach is 0.2 + 5 × 0.02585 =
    0.329 eV: ±0.3 eV is refused, ±0.4 eV passes, and the rule is the
    transmission rung's alone -- the device writes no window.

    Silent before this: the window gate was a catalogue note (plan P2);
    above a few volts the current was cut without a word.
    """
    def issues(emax, rung):
        cfg = SiestaConfig(system_label="t", bias_voltage_v=0.4,
                           electronic_temperature=300.0,
                           transmission_emin_ev=-emax,
                           transmission_emax_ev=emax)
        return validate(junction, cfg, calculation="transport", rung=rung)

    short = _find(issues(0.3, "transmission"), "config.transmission_emax_ev",
                  "error")
    assert short and "0.329" in short[0].message and "widen" in short[0].message
    assert not _find(issues(0.4, "transmission"), "config.transmission_emax_ev")
    assert not _find(issues(0.3, "device"), "config.transmission_emax_ev")
