"""A harmonic mode's mechanics, `vibration`, against the science's closed forms
(`science/vibrational-averaging.md` § 2b.5; `engines/vibration.md` § 6.11
§ 7).  Every expected number comes from the theory -- the oscillator's
eigenfunctions, Mehler's sum, the swing's averages over its phase, Gauss
quadrature -- or from CODATA's constants (`scipy.constants`), never from
molbuilder's own constants; where a row holds two of molbuilder's answers
together, it is to the relation the science or the contract states between
them (Mehler's sum and the hot limit, science § 2b.3; the record's
temperature, `engines/vibration.md` § 6.11 § 9 item 6).  The end-to-end
row, a real PySCF H2, is `test_vibration_mechanics_e2e.py`.

WHY THE INPUT IS TYPED: a mode here is the science's own -- H2's stretch at
4400 cm-1, a spring between two masses -- whose every number is known on
paper; no run makes a mode whose answers a test knows in advance.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy import constants as codata
from scipy import integrate, special

from molbuilder import vibration as V
from molbuilder.spectra.normal_modes import frequencies_cm1, vibrational_modes

AMU_KG = codata.physical_constants["atomic mass constant"][0]
BOHR_M = codata.physical_constants["Bohr radius"][0]
HARTREE_J = codata.physical_constants["Hartree energy"][0]
M_H = 1.008                          # hydrogen's standard atomic weight, amu


def omega_si(nu_cm1: float) -> float:
    return 2.0 * math.pi * codata.c * nu_cm1 * 100.0


def q_zp_codata(nu_cm1: float) -> float:
    """``sqrt(hbar / 2 omega)`` in amu^1/2.Å, from CODATA."""
    return math.sqrt(codata.hbar / (2.0 * omega_si(nu_cm1)) / AMU_KG) * 1e10


def mode_of(*, index_1based, frequency_cm1, has_imag,
            eigenvector_canonical_per_amu12, free_atom_idxs, masses_amu,
            element_symbols, positions_ang, record_temperature_k=None,
            species_labels=(), regions=None, mode_temperatures_k=()):
    """A mode built as the result's reader builds it -- through its
    `vibration.Vibration`, which holds what every mode shares."""
    return V.Vibration(
        modes=(V.Mode(index_1based=index_1based, frequency_cm1=frequency_cm1,
                      has_imag=has_imag,
                      eigenvector_canonical_per_amu12=np.asarray(
                          eigenvector_canonical_per_amu12, dtype=float)),),
        free_atom_idxs=tuple(free_atom_idxs), masses_amu=masses_amu,
        element_symbols=tuple(element_symbols), positions_ang=positions_ang,
        record_temperature_k=record_temperature_k,
        species_labels=tuple(species_labels), regions=dict(regions or {}),
        mode_temperatures_k=tuple(mode_temperatures_k)).modes[0]


def h2_mode(nu_cm1: float = 4400.0, temperature_k=None) -> V.Mode:
    """H2's stretch along z: each atom ``±1/sqrt(2 m)``, the canonical form
    (``sum m |L|^2 = 1``)."""
    a = 1.0 / math.sqrt(2.0 * M_H)
    return mode_of(index_1based=1, frequency_cm1=nu_cm1, has_imag=False,
                   eigenvector_canonical_per_amu12=np.array([[0.0, 0.0, a],
                                                             [0.0, 0.0, -a]]),
                   free_atom_idxs=(0, 1),
                   masses_amu=np.array([M_H, M_H]), element_symbols=("H", "H"),
                   positions_ang=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.741]]),
                   record_temperature_k=temperature_k)


# --------------------------------------------------------------------- #
#  The zero-point amplitude, and each atom's move (§ 2b.1, § 2b.2)       #
# --------------------------------------------------------------------- #

def test_h2s_zero_point_amplitude_and_its_bond_move_are_the_textbooks():
    """§ 2b.2, H2 worked: at 4400 cm-1 ``Q_zp = 0.0619`` amu^1/2.Å; at it
    each atom moves 0.0436 Å and the bond changes by ``Q_zp/sqrt(mu)``, the
    textbook zero-point amplitude ``sqrt(hbar / 2 mu omega)``, each atom's
    move the largest."""
    mode = h2_mode()
    q = mode.zero_point_amplitude_amu12_ang
    assert q == pytest.approx(q_zp_codata(4400.0), rel=1e-7)
    assert round(q, 4) == 0.0619
    u = mode.displacement(zero_point_widths=1.0)
    mu_kg = M_H / 2.0 * AMU_KG
    bond = math.sqrt(codata.hbar / (2.0 * mu_kg * omega_si(4400.0))) * 1e10
    assert np.linalg.norm(u[0] - u[1]) == pytest.approx(bond, rel=1e-7)
    assert np.linalg.norm(u, axis=1) == pytest.approx([bond / 2] * 2, rel=1e-7)
    assert round(float(np.linalg.norm(u[0])), 4) == 0.0436
    assert mode.largest_zero_point_move_ang == pytest.approx(bond / 2,
                                                             rel=1e-7)


def test_the_largest_zero_point_move_is_an_atoms_whole_move_on_any_axis():
    """§ 6.11 § 3: the largest zero-point move is the length of the atom's
    move that is longest -- for H2 at 4400 cm-1 half the bond's
    ``sqrt(hbar / 2 mu omega)`` -- whichever way the bond points; the
    one door the record and the script door both ask
    (`vibration.largest_zero_point_move_ang`).  H2 along (1, 1, 1)/sqrt(3).
    THE SILENT FAILURE: the largest single coordinate taken for the move,
    which a bond along an axis cannot tell apart."""
    a = 1.0 / math.sqrt(2.0 * M_H)
    n = np.ones(3) / math.sqrt(3.0)
    mu_kg = M_H / 2.0 * AMU_KG
    bond = math.sqrt(codata.hbar / (2.0 * mu_kg * omega_si(4400.0))) * 1e10
    assert V.largest_zero_point_move_ang(
        4400.0, False, np.array([a * n, -a * n])) == pytest.approx(
            bond / 2, rel=1e-7)


def test_a_spring_between_two_masses_moves_as_the_two_body_problem_says():
    """§ 2b.5: a harmonic diatomic built from a spring constant ``k`` and two
    masses, through the road's own diagonalisation: its frequency
    ``sqrt(k/mu)``; at ``Q_zp`` the atoms move ``m2/M`` and ``m1/M`` of
    ``Q_zp/sqrt(mu)`` in opposite directions, the centre of mass still; the
    move projects back to ``Q_zp``; its harmonic energy is
    ``omega^2 Q^2 / 2``; each atom's share of the motion ``m2/M`` and
    ``m1/M`` (the kinetic energy's, § 6.11 § 3), and Gaussian's reduced mass
    ``1 / sum |L|^2 = m1 m2 M / (m1^2 + m2^2)``.  THE SILENT FAILURES: a
    share weighted by the displacement alone, handing the light atom the
    mode; a reduced mass by another convention, which a person comparing
    with Gaussian's output reads as a wrong number."""
    k_si, m1, m2 = 1900.0, 12.000, 15.995           # a C=O spring, N/m
    k_au = k_si * BOHR_M ** 2 / HARTREE_J
    z = np.diag([0.0, 0.0, 1.0])
    hessian = np.zeros((2, 2, 3, 3))
    hessian[0, 0] = hessian[1, 1] = k_au * z
    hessian[0, 1] = hessian[1, 0] = -k_au * z
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]])
    eig, modes, _ = vibrational_modes(hessian, [m1, m2], positions, held=[],
                                      axis_kind=("isolated",) * 3)
    assert len(eig) == 1
    nu = float(frequencies_cm1(eig)[0])
    mu_kg = m1 * m2 / (m1 + m2) * AMU_KG
    assert nu == pytest.approx(math.sqrt(k_si / mu_kg)
                               / (2 * math.pi * codata.c) / 100.0, rel=1e-6)
    mode = mode_of(index_1based=1, frequency_cm1=nu, has_imag=False,
                   eigenvector_canonical_per_amu12=np.asarray(modes[0]),
                   free_atom_idxs=(0, 1),
                   masses_amu=np.array([m1, m2]),
                   element_symbols=("C", "O"), positions_ang=positions)
    q_zp = mode.zero_point_amplitude_amu12_ang
    u = mode.displacement(zero_point_widths=1.0)
    reach = q_zp / math.sqrt(m1 * m2 / (m1 + m2))
    total = m1 + m2
    assert np.linalg.norm(u, axis=1) == pytest.approx(
        [m2 / total * reach, m1 / total * reach], rel=1e-9)
    assert float(np.dot(u[0], u[1])) < 0.0
    assert m1 * u[0] + m2 * u[1] == pytest.approx(np.zeros(3), abs=1e-15)
    assert mode.coordinate(u) == pytest.approx(q_zp, rel=1e-12)
    q = 0.137                                       # any coordinate
    u_bohr = mode.displacement(q) * 1e-10 / BOHR_M
    energy = 0.5 * np.einsum("ai,abij,bj->", u_bohr, hessian, u_bohr)
    q_bohr = q * 1e-10 / BOHR_M
    omega2_au = k_au / (m1 * m2 / (m1 + m2))       # k / mu, Eh/(Bohr^2 amu)
    assert energy == pytest.approx(0.5 * omega2_au * q_bohr ** 2, rel=1e-9)
    # THE TWO-BODY SPRING, by Gaussian's convention (§ 6.11 § 3): each
    # atom's share of the motion the OTHER's mass over the total -- the
    # lighter carbon carries more -- and mu = m1 m2 (m1 + m2) / (m1^2 + m2^2)
    assert mode.motion_share == pytest.approx([m2 / total, m1 / total],
                                              rel=1e-9)
    assert mode.reduced_mass_amu == pytest.approx(
        m1 * m2 * total / (m1 ** 2 + m2 ** 2), rel=1e-9)


def test_moves_laid_out_other_than_a_row_per_atom_are_refused():
    """§ 2b.2: ``Q = sum_A m_A L_A . u_A`` pairs each atom's move with its
    row; moves laid out otherwise -- an x, a y and a z row over the atoms,
    as numbers as many -- are refused, never paired with the wrong atoms
    into a wrong coordinate."""
    mode = h2_mode()
    with pytest.raises(ValueError, match="a row of x, y, z per atom"):
        mode.coordinate(mode.displacement(0.05).T)


def test_the_reduced_mass_and_force_constant_are_as_gaussian_prints_them():
    """§ 6.11 § 3, Gaussian's convention [Ochterski1999]: ``mu = 1 /
    sum_A |L_A|^2`` -- for H2's stretch the atom's own mass, 1.008 amu --
    and ``k = mu omega^2`` in mdyn/Å (100 N/m) from
    CODATA; an imaginary mode's ``k`` positive, ``mu |omega|^2``, as Gaussian
    prints it; a mode with no motion has neither, ``None`` -- the record's
    ``null``.  THE SILENT FAILURE: a unit off by the 100 between N/m and
    mdyn/Å, or the two-body reduced mass, either a plausible number beside
    Gaussian's."""
    mode = h2_mode()
    assert mode.reduced_mass_amu == pytest.approx(M_H, rel=1e-12)
    k_n_per_m = M_H * AMU_KG * omega_si(4400.0) ** 2
    assert mode.force_constant_mdyn_per_ang == pytest.approx(
        k_n_per_m / 100.0, rel=1e-8)
    imaginary = mode_of(index_1based=1, frequency_cm1=-4400.0, has_imag=True,
                        eigenvector_canonical_per_amu12=(
                            h2_mode().eigenvector_canonical_per_amu12),
                        free_atom_idxs=(0, 1), masses_amu=h2_mode().masses_amu,
                        element_symbols=("H", "H"),
                        positions_ang=h2_mode().positions_ang)
    assert imaginary.force_constant_mdyn_per_ang == pytest.approx(
        k_n_per_m / 100.0, rel=1e-8)
    still = mode_of(index_1based=1, frequency_cm1=4400.0, has_imag=False,
                    eigenvector_canonical_per_amu12=np.zeros((2, 3)),
                    free_atom_idxs=(0, 1), masses_amu=h2_mode().masses_amu,
                    element_symbols=("H", "H"),
                    positions_ang=h2_mode().positions_ang)
    assert (still.reduced_mass_amu, still.force_constant_mdyn_per_ang) == (
        None, None)


def test_the_shares_sum_by_element_label_and_region():
    """§ 6.11 § 3, § 6: each free atom's share summed by its element, its
    species label as written, or each of the structure's region labels as
    written (they may overlap) -- the three alike, largest first, a group
    holding no free atom left out (a held atom adds nothing to any), a mode
    with no motion ``{}``.  THE SILENT FAILURES: a label's two atoms merged
    into their element, or a region summed over atoms it does not hold."""
    a = 1.0 / math.sqrt(2.0 * M_H)
    mode = mode_of(index_1based=1, frequency_cm1=4400.0, has_imag=False,
                   eigenvector_canonical_per_amu12=np.array([[0.0, 0.0, a],
                                                             [0.0, 0.0, -a]]),
                   free_atom_idxs=(1, 2),
                   masses_amu=np.array([196.97, M_H, M_H]),
                   element_symbols=("Au", "H", "H"),
                   positions_ang=np.zeros((3, 3)),
                   species_labels=("Au1", "H1", "H2"),
                   regions={"L-electrode": (0,), "bridge": (1, 2)})
    assert mode.motion_share == pytest.approx([0.5, 0.5])
    assert mode.motion_shares(by="element") == pytest.approx({"H": 1.0})
    assert mode.motion_shares(by="label") == pytest.approx(
        {"H1": 0.5, "H2": 0.5})
    regions = {"L-electrode": (0,), "bridge": (1, 2), "H1 side": (1,)}
    mode = mode_of(index_1based=1, frequency_cm1=4400.0, has_imag=False,
                   eigenvector_canonical_per_amu12=(
                       mode.eigenvector_canonical_per_amu12),
                   free_atom_idxs=(1, 2), masses_amu=mode.masses_amu,
                   element_symbols=mode.element_symbols,
                   positions_ang=mode.positions_ang,
                   species_labels=mode.species_labels, regions=regions)
    by_region = mode.motion_shares(by="region")
    assert by_region == pytest.approx({"bridge": 1.0, "H1 side": 0.5})
    assert list(by_region) == ["bridge", "H1 side"]
    still = replace(mode, eigenvector_canonical_per_amu12=np.zeros((2, 3)))
    assert [still.motion_shares(by=b) for b in ("element", "label",
                                                "region")] == [{}, {}, {}]


def test_a_modes_thermal_row_is_its_spread_at_each_temperature():
    """§ 6.11 § 3: the record's spread and mean quanta at the modes' own
    temperatures -- 0, 4, 10, 20, 30 and 40 K below the thermochemistry
    grid, the grid's (§ 4.7), and the record's own, ascending, each once --
    the spread ``Q_zp sqrt(coth(h c nu / 2 k_B T))`` and the mean quanta
    ``1 / (exp(h c nu / k_B T) - 1)`` from CODATA, an imaginary mode's none
    and no largest zero-point move.  THE SILENT FAILURES: a temperature
    missing from the list or one too many; the quanta and the spread
    swapped; an imaginary mode handed a thermal state it does not have."""
    from molbuilder.spectra.normal_modes import mode_temperatures
    grid = np.linspace(50.0, 1500.0, 30)    # § 4.7: 50-1500 K in 30 points
    assert mode_temperatures(298.15) == pytest.approx(sorted(
        {0.0, 4.0, 10.0, 20.0, 30.0, 40.0, 298.15} | set(grid)), abs=1e-9)
    nu = 700.0
    mode = mode_of(index_1based=1, frequency_cm1=nu, has_imag=False,
                   eigenvector_canonical_per_amu12=(
                       h2_mode().eigenvector_canonical_per_amu12),
                   free_atom_idxs=(0, 1), masses_amu=h2_mode().masses_amu,
                   element_symbols=("H", "H"),
                   positions_ang=h2_mode().positions_ang,
                   mode_temperatures_k=(0.0, 298.15))
    x = codata.h * codata.c * nu * 100.0 / (codata.k * 298.15)
    row = mode.thermal
    assert row["temperature_k"] == [0.0, 298.15]
    assert row["thermal_spread_amu12_ang"] == pytest.approx(
        [q_zp_codata(nu), q_zp_codata(nu) * math.sqrt(1.0 / math.tanh(x / 2))],
        rel=1e-7)
    assert row["mean_quanta"] == pytest.approx([0.0, 1.0 / math.expm1(x)],
                                               rel=1e-8, abs=1e-15)
    with pytest.raises(ValueError, match="positive frequency"):
        V.mean_quanta(-700.0, 298.15)
    imaginary = replace(mode, frequency_cm1=-700.0, has_imag=True)
    assert imaginary.thermal == {"temperature_k": [0.0, 298.15],
                                 "thermal_spread_amu12_ang": None,
                                 "mean_quanta": None}
    assert imaginary.largest_zero_point_move_ang is None


def test_a_mode_at_another_frequency_keeps_its_pattern():
    """§ 2b.1: ``Q_zp`` from any frequency, the mode's pattern kept."""
    scaled = h2_mode().with_frequency(4400.0 * 0.96)
    assert scaled.zero_point_amplitude_amu12_ang == pytest.approx(
        q_zp_codata(4400.0 * 0.96), rel=1e-7)
    assert np.array_equal(scaled.eigenvector_canonical_per_amu12,
                          h2_mode().eigenvector_canonical_per_amu12)


# --------------------------------------------------------------------- #
#  The states (§ 2b.3)                                                   #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("n, beyond", [(0, 0.157), (1, 0.112), (2, 0.095),
                                       (5, None)])
def test_n_quanta_exactly_is_the_oscillators_own_density(n, beyond):
    """§ 2b.3: ``P_n`` integrates to 1, its ``<Q^2> = (2n+1) Q_zp^2`` and
    ``<Q^4> = 3 (2n^2+2n+1) Q_zp^4``, it has ``n`` zeros, and its weight
    beyond the classical turning points ``±Q_zp sqrt(2(2n+1))`` is the
    chapter's: 15.7 %, 11.2 %, 9.5 % for 0, 1, 2 quanta."""
    q = q_zp_codata(4400.0)
    s = h2_mode().state(quanta=n).state
    reach = q * (math.sqrt(2 * (2 * n + 1)) + 10)
    norm = integrate.quad(s.pdf, -reach, reach, limit=400)[0]
    q2 = integrate.quad(lambda x: x * x * s.pdf(x), -reach, reach,
                        limit=400)[0]
    q4 = integrate.quad(lambda x: x ** 4 * s.pdf(x), -reach, reach,
                        limit=400)[0]
    assert norm == pytest.approx(1.0, abs=1e-10)
    assert q2 == pytest.approx((2 * n + 1) * q * q, rel=1e-8)
    assert q4 == pytest.approx(3 * (2 * n * n + 2 * n + 1) * q ** 4, rel=1e-8)
    assert s.moment(2) == pytest.approx(q2, rel=1e-10)
    assert s.moment(4) == pytest.approx(q4, rel=1e-10)
    turning = q * math.sqrt(2 * (2 * n + 1))
    grid = np.linspace(-turning, turning, 20001)
    p = s.pdf(grid)
    minima = [i for i in range(1, len(p) - 1)
              if p[i] <= p[i - 1] and p[i] <= p[i + 1] and p[i] < 1e-6 * p.max()]
    assert len(minima) == n
    if beyond is not None:
        assert 1.0 - (s.cdf(turning) - s.cdf(-turning)) == pytest.approx(
            beyond, abs=5e-4)


def test_thermal_is_mehlers_sum_of_the_number_states():
    """§ 2b.3: the ``p_n``-weighted sum of the ``P_n`` is the Gaussian of
    ``sigma^2 = (2 nbar + 1) Q_zp^2``, ``p_n = nbar^n / (nbar+1)^(n+1)``."""
    mode, nbar = h2_mode(), 0.7
    thermal = mode.state(mean_quanta=nbar).state
    q = mode.zero_point_amplitude_amu12_ang
    xs = np.linspace(-4 * q, 4 * q, 17)
    mix = sum(nbar ** n / (nbar + 1) ** (n + 1)
              * mode.state(quanta=n).state.pdf(xs) for n in range(120))
    sigma = q * math.sqrt(2 * nbar + 1)
    gauss = np.exp(-0.5 * (xs / sigma) ** 2) / (math.sqrt(2 * math.pi) * sigma)
    assert mix == pytest.approx(gauss, rel=1e-10, abs=1e-12)
    assert thermal.pdf(xs) == pytest.approx(gauss, rel=1e-12)


@pytest.mark.parametrize("temperature_k", [0.0, 77.0, 298.15, 3000.0])
def test_a_temperatures_quanta_are_bose_and_einsteins(temperature_k):
    """§ 2a: ``nbar = 1 / (exp(h c nu / k_B T) - 1)``, 0 at 0 K, and the
    spread ``Q_zp sqrt(2 nbar + 1)`` -- ``Q_zp`` at 0 K."""
    nu = 700.0
    expected = (0.0 if temperature_k == 0 else 1.0 / math.expm1(
        codata.h * codata.c * nu * 100 / (codata.k * temperature_k)))
    s = h2_mode(nu).state(temperature_k).state
    assert s.mean_quanta == pytest.approx(expected, rel=1e-8, abs=1e-15)
    assert s.spread_amu12_ang == pytest.approx(
        q_zp_codata(nu) * math.sqrt(2 * expected + 1), rel=1e-7)


def test_hot_the_thermal_spread_is_the_classical_one():
    """§ 2b.3: ``sigma^2 -> k_B T / omega^2`` hot -- the classical state's."""
    nu, t = 10.0, 1.0e5                              # h c nu / k T ~ 1e-4
    mode = h2_mode(nu)
    classical = mode.state(t, classical=True).state
    assert classical.moment(2) == pytest.approx(
        codata.k * t / omega_si(nu) ** 2 / AMU_KG * 1e20, rel=1e-7)
    assert mode.state(t).state.moment(2) / classical.moment(2) == \
        pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("nbar, n_s", [(0.4, 3.0), (0.0, 1.0e4)],
                         ids=["three quanta", "ten thousand quanta"])
def test_a_swing_over_a_bath_is_the_bath_carried_along_the_swing(nbar, n_s):
    """§ 2b.3, § 2a.2: a swing of ``n_s`` quanta, ``Q_0 = 2 Q_zp sqrt(n_s)``,
    carrying a bell curve of spread ``sigma_T`` -- the warm driven mode's,
    ``nbar`` quanta of warmth, or the coherent state's ``Q_zp`` for none:
    ``<Q^2> = sigma_T^2 + Q_0^2/2``,
    ``<Q^4> = 3 sigma_T^4 + 3 sigma_T^2 Q_0^2 + 3/8 Q_0^4``, its density the
    bath's Gaussian averaged over the swing's phase -- a swing two hundred
    of its bath's spreads wide as much as one of three -- its mean quanta the
    bath's and the swing's added, its spread the one they give."""
    mode = h2_mode()
    q = mode.zero_point_amplitude_amu12_ang
    s = V.State(V.SWING, q, bath_quanta=nbar, swing_quanta=n_s)
    st2, q0 = (2 * nbar + 1) * q * q, 2 * q * math.sqrt(n_s)
    assert s.moment(2) == pytest.approx(st2 + q0 * q0 / 2, rel=1e-12)
    assert s.moment(4) == pytest.approx(
        3 * st2 ** 2 + 3 * st2 * q0 * q0 + 3 / 8 * q0 ** 4, rel=1e-12)
    assert s.mean_quanta == pytest.approx(nbar + n_s)
    assert s.spread_amu12_ang == pytest.approx(
        q * math.sqrt(2 * (nbar + n_s) + 1), rel=1e-14)
    for x in (0.0, 0.5 * q0, 0.9 * q0, 1.3 * q0):
        # over half the period, the turning phase its own break point
        turning = [math.acos(x / q0)] if x < q0 else None
        averaged = integrate.quad(
            lambda phi: math.exp(-0.5 * (x - q0 * math.cos(phi)) ** 2 / st2),
            0, math.pi, points=turning, epsabs=0.0, epsrel=1e-13,
            limit=400)[0] / (math.pi * math.sqrt(2 * math.pi * st2))
        assert s.pdf(x) == pytest.approx(averaged, rel=1e-10, abs=1e-300)


def test_a_classical_swing_alone_spends_its_time_at_its_turning_points():
    """§ 2a.2: ``rho(Q) = 1 / (pi sqrt(Q_0^2 - Q^2))`` inside ``±Q_0``, none
    beyond; ``<Q^2> = Q_0^2/2``, ``<Q^4> = 3/8 Q_0^4``."""
    mode = h2_mode()
    s = mode.state(swing_quanta=2.0, classical=True).state
    q0 = 2 * mode.zero_point_amplitude_amu12_ang * math.sqrt(2.0)
    assert s.pdf(0.5 * q0) == pytest.approx(
        1 / (math.pi * math.sqrt(q0 ** 2 - (0.5 * q0) ** 2)))
    assert s.pdf(1.01 * q0) == 0.0
    assert s.moment(2) == pytest.approx(q0 ** 2 / 2)
    assert s.moment(4) == pytest.approx(3 / 8 * q0 ** 4)
    assert s.cdf(0.0) == pytest.approx(0.5)


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


Q_ZP_4400 = q_zp_codata(4400.0)
SIGMA_THERMAL = Q_ZP_4400 * math.sqrt(2 * 0.7 + 1)          # 0.7 mean quanta
SIGMA_CLASSICAL = math.sqrt(codata.k * 3000.0 / omega_si(4400.0) ** 2
                            / AMU_KG) * 1e10                # 3000 K
Q0_TWO_QUANTA = 2 * Q_ZP_4400 * math.sqrt(2.0)
T_FOR_0_4 = (codata.h * codata.c * 440000.0
             / (codata.k * math.log1p(1.0 / 0.4)))     # 0.4 quanta at 4400 cm-1


def _swing_over_bath_cdf(x, sigma, q0):
    """``P(Q <= x)``: the bath's distribution averaged over the swing's
    phase."""
    return integrate.quad(lambda phi: _normal_cdf((x - q0 * math.cos(phi))
                                                  / sigma),
                          0, math.pi, epsabs=0.0, epsrel=1e-13)[0] / math.pi


@pytest.mark.parametrize("asked, x, expected", [
    (dict(mean_quanta=0.7), SIGMA_THERMAL, _normal_cdf(1.0)),
    (dict(mean_quanta=0.7), -2 * SIGMA_THERMAL, _normal_cdf(-2.0)),
    (dict(temperature_k=3000.0, classical=True), SIGMA_CLASSICAL,
     _normal_cdf(1.0)),
    (dict(swing_quanta=2.0, classical=True), Q0_TWO_QUANTA / 2, 2 / 3),
    (dict(swing_quanta=2.0, classical=True), -Q0_TWO_QUANTA / 2, 1 / 3),
    (dict(swing_quanta=0.0, classical=True), 0.0, 1.0),
    (dict(temperature_k=0.0, classical=True), 0.0, 1.0),
    (dict(temperature_k=0.0, classical=True), -1.0e-6, 0.0),
    (dict(swing_quanta=2.0, temperature_k=T_FOR_0_4), Q0_TWO_QUANTA / 2,
     _swing_over_bath_cdf(Q0_TWO_QUANTA / 2,
                          Q_ZP_4400 * math.sqrt(1.8), Q0_TWO_QUANTA)),
    (dict(swing_quanta=2.0, temperature_k=T_FOR_0_4), -1.2 * Q0_TWO_QUANTA,
     _swing_over_bath_cdf(-1.2 * Q0_TWO_QUANTA,
                          Q_ZP_4400 * math.sqrt(1.8), Q0_TWO_QUANTA)),
    (dict(swing_quanta=2.0), Q0_TWO_QUANTA / 2,
     _swing_over_bath_cdf(Q0_TWO_QUANTA / 2, Q_ZP_4400, Q0_TWO_QUANTA)),
    (dict(swing_quanta=2.0, temperature_k=3000.0, classical=True),
     Q0_TWO_QUANTA / 2,
     _swing_over_bath_cdf(Q0_TWO_QUANTA / 2, SIGMA_CLASSICAL, Q0_TWO_QUANTA)),
], ids=["thermal at one spread", "thermal two spreads below",
        "classical at one spread", "a classical swing halfway out",
        "a classical swing halfway in", "a classical swing of no quanta",
        "classical at 0 K, at rest", "classical at 0 K, below rest",
        "a warm swing halfway out", "a warm swing beyond",
        "the coherent state halfway out",
        "a classical swing carrying the classical bell curve"])
def test_each_states_distribution_is_its_closed_form(asked, x, expected):
    """§ 2b.3, § 2a.2: the chance that ``Q`` is ``x`` or less -- the
    Gaussian's ``(1 + erf(x / sqrt(2) sigma)) / 2``; the classical swing's
    ``1/2 + asin(x / Q_0) / pi``; all of it at rest for no quanta, and for
    a classical mode at 0 K; a swing, the bell curve it carries averaged
    over the swing's phase -- the coherent state's the zero-point width, a
    warm driven mode's its temperature's, a classical swing's the classical
    bell curve.  THE SILENT FAILURE: a driven mode handed the record's
    temperature it was never given, or a classical swing dropping the bell
    curve it carries."""
    s = V.state(Q_ZP_4400, 4400.0, **asked)
    assert s.cdf(x) == pytest.approx(expected, rel=1e-7, abs=1e-12)


@pytest.mark.parametrize("asked", [
    dict(temperature_k=0.0, classical=True),
    dict(swing_quanta=0.0, classical=True),
], ids=["classical at 0 K", "a classical swing of no quanta"])
def test_a_classical_mode_at_rest_sits_at_zero(asked):
    """§ 2b.3: classically, at 0 K or with no swing, the molecule is at rest
    at ``Q = 0`` -- all of it there: a point, its density infinite at 0 and
    nothing elsewhere, no spread, every frame at 0."""
    s = V.state(Q_ZP_4400, 4400.0, **asked)
    assert list(s.pdf(np.array([0.0, 1.0e-3, -1.0e-3]))) == [math.inf, 0.0,
                                                              0.0]
    assert s.spread_amu12_ang == 0.0
    assert set(s.frames(3).q_amu12_ang[:, 0]) == {0.0}


# --------------------------------------------------------------------- #
#  The frames (§ 2b.4)                                                   #
# --------------------------------------------------------------------- #

def _moment_of(frames, k):
    return sum(w * x ** k for x, w in zip(frames.q_amu12_ang[:, 0],
                                          frames.weights))


@pytest.mark.parametrize("nodes", [3, 5, 7])
def test_a_gaussian_states_frames_are_exact_to_their_degree(nodes):
    """§ 2b.4, § 5.1: the Gauss–Hermite rule of ``m`` frames reproduces the
    Gaussian's moments ``sigma^k (k-1)!!`` to degree ``2m - 1``; three
    frames sit at ``0`` and ``±sqrt(3) sigma``, weights 2/3 and 1/6."""
    s = h2_mode().state(298.15).state
    sigma = s.spread_amu12_ang
    f = s.frames(nodes)
    assert sum(f.weights) == pytest.approx(1.0)
    assert f.exact_to_degree == 2 * nodes - 1
    for k in range(0, 2 * nodes):
        exact = 0.0 if k % 2 else sigma ** k * math.prod(range(k - 1, 0, -2))
        assert _moment_of(f, k) == pytest.approx(exact, rel=1e-10,
                                                 abs=1e-14 * sigma ** k)
    if nodes == 3:
        assert sorted(f.q_amu12_ang[:, 0]) == pytest.approx(
            [-math.sqrt(3) * sigma, 0.0, math.sqrt(3) * sigma], abs=1e-15)
        assert sorted(f.weights) == pytest.approx([1 / 6, 1 / 6, 2 / 3])


@pytest.mark.parametrize("n, nodes", [(1, 3), (2, 3), (2, 5), (4, 4)])
def test_n_quantas_frames_are_the_gauss_rule_of_its_own_density(n, nodes):
    """§ 2b.4: the Gauss rule whose weight function is ``P_n`` -- exact for
    its moments to degree ``2m - 1``, each moment integrated here from the
    closed form, ``P_n = H_n(xi)^2 e^{-xi^2} / (2^n n! sqrt(2 pi) Q_zp)``."""
    s = h2_mode().state(quanta=n).state
    f = s.frames(nodes)
    assert f.exact_to_degree == 2 * nodes - 1
    q_zp = s.zero_point_amplitude_amu12_ang
    reach = q_zp * (math.sqrt(2 * (2 * n + 1)) + 10)

    def density(x):
        xi = x / (math.sqrt(2) * q_zp)
        return (special.eval_hermite(n, xi) ** 2 * math.exp(-xi * xi)
                / (2 ** n * math.factorial(n) * math.sqrt(2 * math.pi) * q_zp))
    for k in range(0, 2 * nodes):
        exact = integrate.quad(lambda x: x ** k * density(x), -reach, reach,
                               limit=400)[0]
        assert _moment_of(f, k) == pytest.approx(
            exact, rel=1e-8, abs=1e-12 * reach ** k)


def test_a_bare_swings_frames_are_chebyshevs_gauss_rule():
    """§ 2a.3: the Gauss rule of a bare swing's own distribution is
    Chebyshev's -- ``m`` frames at ``Q_0 cos((2j - 1) pi / 2m)``, each of
    weight ``1/m`` -- exact to degree ``2m - 1`` and not to ``2m``; three at
    0 and ``±0.866 Q_0``.  THE SILENT FAILURE: frames on the turning points,
    where the swing spends most time and a rule places none."""
    mode = h2_mode()
    s = mode.state(swing_quanta=2.0, classical=True).state
    q0 = 2 * mode.zero_point_amplitude_amu12_ang * math.sqrt(2.0)
    f = s.frames(5)
    assert sorted(f.q_amu12_ang[:, 0]) == pytest.approx(
        sorted(q0 * math.cos((2 * j - 1) * math.pi / 10) for j in range(1, 6)),
        abs=1e-15)
    assert list(f.weights) == pytest.approx([1 / 5] * 5)
    assert f.exact_to_degree == 9
    for k in range(0, 10):
        exact = q0 ** k * (math.comb(k, k // 2) / 2 ** k if k % 2 == 0 else 0)
        assert _moment_of(f, k) == pytest.approx(exact, abs=1e-15)
    assert _moment_of(f, 10) != pytest.approx(
        q0 ** 10 * math.comb(10, 5) / 2 ** 10, rel=1e-3)
    assert sorted(s.frames(3).q_amu12_ang[:, 0]) == pytest.approx(
        [-0.8660254 * q0, 0.0, 0.8660254 * q0], abs=1e-7 * q0)


# --------------------------------------------------------------------- #
#  Every temperature explicit (§ 6.11 § 9 item 6)                        #
# --------------------------------------------------------------------- #

def test_a_temperature_not_given_is_the_records_and_says_so():
    """§ 6.11 § 9 item 6: given, used; not given, the record's; the answer
    says which; a record stating none asks for one."""
    given = h2_mode(temperature_k=500.0).state(77.0)
    assert (given.temperature_k, given.temperature_from) == (77.0, "given")
    taken = h2_mode(temperature_k=500.0).state()
    assert (taken.temperature_k, taken.temperature_from) == (
        500.0, "the record's")
    assert taken.spread_amu12_ang == h2_mode().state(500.0).spread_amu12_ang
    with pytest.raises(ValueError, match="name a temperature"):
        h2_mode().state()


def test_an_imaginary_mode_has_moves_and_no_harmonic_state():
    """§ 2b.1: an imaginary mode has no zero-point width -- a coordinate
    given still moves its atoms; a state or a width of it is refused."""
    mode = mode_of(index_1based=1, frequency_cm1=-120.0, has_imag=True,
                   eigenvector_canonical_per_amu12=(
                       h2_mode().eigenvector_canonical_per_amu12),
                   free_atom_idxs=(0, 1), masses_amu=h2_mode().masses_amu,
                   element_symbols=("H", "H"),
                   positions_ang=h2_mode().positions_ang)
    assert mode.zero_point_amplitude_amu12_ang is None
    assert mode.coordinate(mode.displacement(0.05)) == pytest.approx(0.05)
    with pytest.raises(ValueError, match="imaginary"):
        mode.state(300.0)
    with pytest.raises(ValueError, match="imaginary"):
        mode.displacement(zero_point_widths=1.0)


def test_a_frames_geometries_are_each_atom_at_r_plus_q_l():
    """§ 2b.2, § 2b.4: each frame of a state is every atom at ``R + Q_j L``,
    its weight the rule's."""
    mode = h2_mode()
    got = mode.state(298.15).frames(3)
    for q, geometry in zip(got.q_amu12_ang[:, 0], got.positions_ang):
        assert geometry == pytest.approx(
            mode.positions_ang
            + q * mode.eigenvector_canonical_over_atoms_per_amu12, abs=1e-15)
    assert sum(got.weights) == pytest.approx(1.0)


@pytest.mark.parametrize("nodes", [3, 5])
def test_a_swing_over_a_bath_is_averaged_exactly_to_its_frames_degree(nodes):
    """§ 2a.3: a warm driven mode's frames -- the Gauss rule of its own
    distribution, the swing carrying a bell curve -- reproduce the state's
    moments, ``E[(X + Q_0 cos phi)^k]`` over the bell curve ``X`` and the
    phase, to degree ``2m - 1``, its weights summing to 1.  THE SILENT
    FAILURE: the folded equal phases of 2.79.0, exact only to
    ``2 (m - 1) - 1``."""
    mode = h2_mode()
    q = mode.zero_point_amplitude_amu12_ang
    s = V.State(V.SWING, q, bath_quanta=0.4, swing_quanta=3.0)
    st, q0 = q * math.sqrt(2 * 0.4 + 1), 2 * q * math.sqrt(3.0)
    f = s.frames(nodes)
    assert f.exact_to_degree == 2 * nodes - 1
    assert sum(f.weights) == pytest.approx(1.0)
    for k in range(0, f.exact_to_degree + 1):
        exact = sum(math.comb(k, j)
                    * (0.0 if j % 2 else st ** j * math.prod(range(j - 1, 0, -2)))
                    * q0 ** (k - j)
                    * (0.0 if (k - j) % 2 else
                       math.comb(k - j, (k - j) // 2) / 2 ** (k - j))
                    for j in range(k + 1))
        assert _moment_of(f, k) == pytest.approx(exact, rel=1e-10,
                                                 abs=1e-14 * q0 ** k)


def test_a_thousand_quanta_keep_the_oscillators_moments():
    """§ 2b.3 at a large ``n``: ``<Q^2> = (2n+1) Q_zp^2``, ``<Q^4> =
    3 (2n^2+2n+1) Q_zp^4``, half the weight below 0, a finite density, and
    frames that hold the spread -- no number overflows on the way."""
    n = 1000
    s = h2_mode().state(quanta=n).state
    q = s.zero_point_amplitude_amu12_ang
    assert s.moment(2) == pytest.approx((2 * n + 1) * q * q, rel=1e-12)
    assert s.moment(4) == pytest.approx(3 * (2 * n * n + 2 * n + 1) * q ** 4,
                                        rel=1e-12)
    assert s.cdf(0.0) == pytest.approx(0.5, abs=1e-9)
    assert math.isfinite(s.pdf(3.0 * q))
    f = s.frames(5)
    assert sum(f.weights) == pytest.approx(1.0)
    assert _moment_of(f, 2) == pytest.approx((2 * n + 1) * q * q, rel=1e-10)


def test_a_state_named_no_way_of_its_kinds_is_refused_naming_them():
    """§ 2b.3, § 6.11 § 6: a state is named by one of its kinds' sets of
    arguments; any other set is refused, naming the sets -- never one
    argument silently dropped (here the temperature beside n quanta)."""
    q = h2_mode().zero_point_amplitude_amu12_ang
    with pytest.raises(ValueError, match="named one of these ways"):
        V.state(q, 4400.0, quanta=2, temperature_k=300.0)


def test_a_driven_mode_alone_is_the_coherent_state_and_takes_no_temperature():
    """`engines/vibration-frames.md` § 2: a driven mode named by its swing
    quanta alone is the coherent state -- the zero-point width carried by
    the swing, its mean quanta the swing's, no temperature taken (the user,
    2026-11-23: "the 'driven mode' should only have one possible state");
    a classical state takes the record's temperature and says so; a
    classical swing alone takes none.  THE SILENT FAILURE: 2.79.0's swing
    over the record's bath, warm where the person asked for none."""
    cold = h2_mode(700.0, temperature_k=500.0).state(swing_quanta=2.0)
    assert (cold.temperature_k, cold.temperature_from) == (None, None)
    assert cold.mean_quanta == 2.0
    assert cold.carried_spread_amu12_ang == pytest.approx(q_zp_codata(700.0),
                                                          rel=1e-7)
    alone = h2_mode(700.0, temperature_k=500.0).state(swing_quanta=2.0,
                                                      classical=True)
    assert alone.temperature_k is None and alone.state.cdf(0.0) == 0.5
    hot = h2_mode(700.0, temperature_k=500.0).state(classical=True)
    assert (hot.temperature_k, hot.temperature_from) == (500.0, "the record's")
    assert hot.spread_amu12_ang == pytest.approx(math.sqrt(
        codata.k * 500.0 / omega_si(700.0) ** 2 / AMU_KG) * 1e10, rel=1e-7)


def test_a_held_atom_stays_still_and_adds_nothing_to_its_elements_share():
    """§ 2b.2, § 6.11 § 3, § 6.3: a mode over two free hydrogens beside a
    held gold -- the record's rows the free atoms', its zero-point field
    ``Q_zp`` times them; every answer over every atom keeps the gold where
    it is: its r.m.s. move 0 and each hydrogen's ``sigma |L_H|``, every frame
    and displaced geometry the gold at ``R_Au`` and each hydrogen at
    ``R_H + Q L_H``; the shares over the free atoms alone, the gold none."""
    a = 1.0 / math.sqrt(2.0 * M_H)
    rows = np.array([[0.0, 0.0, a], [0.0, 0.0, -a]])
    at = np.array([[0.0, 0.0, -2.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.741]])
    mode = mode_of(index_1based=1, frequency_cm1=4400.0, has_imag=False,
                   eigenvector_canonical_per_amu12=rows, free_atom_idxs=(1, 2),
                   masses_amu=np.array([196.97, M_H, M_H]),
                   element_symbols=("Au", "H", "H"), positions_ang=at)
    q_zp = q_zp_codata(4400.0)
    assert mode.motion_shares(by="element") == pytest.approx({"H": 1.0})
    assert mode.zero_point_displacement_ang == pytest.approx(q_zp * rows,
                                                             rel=1e-7)
    cold = mode.state(0.0)
    assert cold.atom_rms_ang == pytest.approx([0.0, q_zp * a, q_zp * a],
                                              rel=1e-7)
    frames = cold.frames(3)
    for q, geometry in zip(list(frames.q_amu12_ang[:, 0]) + [0.05],
                           list(frames.positions_ang) + [mode.displaced(0.05)]):
        assert geometry == pytest.approx(
            np.array([at[0], at[1] + q * rows[0], at[2] + q * rows[1]]),
            abs=1e-15)


# --------------------------------------------------------------------- #
#  The draws, frames for other states, the molecule's motion and its     #
#  snapshots (`engines/vibration-frames.md` § 2-§ 4, § 7, § 8)          #
# --------------------------------------------------------------------- #

def _model_vibration(record_temperature_k=300.0) -> V.Vibration:
    """A model molecule -- three atoms, three modes at 120, 700 and 1600 cm-1
    orthonormal with the masses, ``sum_A m_A L_k,A . L_l,A = delta_kl`` --
    built as a reader builds a result's: the closed forms of science § 2b.5
    and § 2c hold for any such set."""
    masses = np.array([12.0, 1.008, 15.999])
    u, _ = np.linalg.qr(np.random.default_rng(20261123).standard_normal((9, 3)))
    rows = u.T.reshape(3, 3, 3) / np.sqrt(masses)[None, :, None]
    modes = tuple(V.Mode(index_1based=k + 1, frequency_cm1=nu, has_imag=False,
                         eigenvector_canonical_per_amu12=rows[k])
                  for k, nu in enumerate((120.0, 700.0, 1600.0)))
    return V.Vibration(modes=modes, free_atom_idxs=(0, 1, 2),
                       masses_amu=masses, element_symbols=("C", "H", "O"),
                       positions_ang=np.array([[0.0, 0.0, 0.0],
                                               [1.09, 0.0, 0.0],
                                               [0.0, 1.2, 0.0]]),
                       record_temperature_k=record_temperature_k)


@pytest.mark.parametrize("asked", [
    dict(temperature_k=300.0), dict(swing_quanta=2.0),
    dict(swing_quanta=2.0, temperature_k=600.0), dict(quanta=3),
    dict(swing_quanta=2.0, classical=True),
    dict(swing_quanta=2.0, temperature_k=600.0, classical=True),
], ids=["thermal", "the coherent state", "a warm driven mode", "3 quanta",
        "a classical swing", "a classical swing carrying the bell curve"])
def test_each_states_draws_follow_its_own_distribution(asked):
    """§ 2c.3, `engines/vibration-frames.md` § 2: a state's draws follow its
    distribution (a two-sided Kolmogorov test against its ``cdf``), one
    standard normal number per draw from the mode's own stream -- for n
    quanta the ``Q`` where the distribution reaches ``Phi(z)``.  THE SILENT
    FAILURE: a state drawn from another's distribution, a bell curve for a
    swing, which no average of a few frames would show."""
    from scipy import stats
    s = _model_vibration().mode(2).state(**asked)
    x = s.draw(4000, seed=12)
    assert stats.kstest(x, s.cdf).pvalue > 1e-3
    if s.kind == V.NUMBER:
        z = np.random.default_rng([12, 2, 0]).standard_normal(4000)
        assert s.cdf(x) == pytest.approx(stats.norm.cdf(z), abs=1e-9)


def test_a_draw_continues_its_seed_from_the_modes_own_stream():
    """`engines/vibration-frames.md` § 2, § 3.3: draws ``start ...`` of a
    seed's sequence are the same however many were asked; each mode draws
    from its own streams, keyed by its number; the seed is required.  THE
    SILENT FAILURE: a set made larger later drawing other frames than its
    first part."""
    v = _model_vibration()
    s = v.mode(2).state(swing_quanta=1.0)
    whole = s.draw(16, seed=5)
    assert np.array_equal(s.draw(8, seed=5), whole[:8])
    assert np.array_equal(s.draw(8, seed=5, start=8), whole[8:])
    assert not np.array_equal(v.mode(3).state(swing_quanta=1.0).draw(
        16, seed=5), whole)
    with pytest.raises(ValueError, match="seed"):
        s.draw(4, seed=None)


@pytest.mark.parametrize("asked, nodes", [
    (dict(temperature_k=300.0), 3), (dict(temperature_k=300.0), 5),
    (dict(quanta=2), 5), (dict(swing_quanta=2.0), 3),
    (dict(swing_quanta=2.0, temperature_k=900.0), 7),
    (dict(temperature_k=300.0), 4), (dict(swing_quanta=2.0), 4),
], ids=["thermal 3", "thermal 5", "2 quanta 5", "driven 3", "warm driven 7",
        "thermal 4", "driven 4"])
def test_a_modes_frames_are_exactly_symmetric_in_the_sets_order(asked, nodes):
    """`engines/vibration-frames.md` § 4: every state's rule exactly
    symmetric -- an odd rule's middle node 0 exactly, its middle frame the
    relaxed geometry bit for bit; each pair ``q_- = -q_+`` exactly, its two
    weights equal -- in the order ``(0, +q1, -q1, +q2, -q2, ...)``; an even
    rule's frame 0 the relaxed geometry prepended at weight 0.  THE SILENT
    FAILURE: frame 0 a near-copy of the relaxed geometry, a second reference
    for a calculation, or an odd response that does not cancel."""
    v = _model_vibration()
    f = v.mode(2).state(**asked).frames(nodes)
    q, w = f.q_amu12_ang[:, 0], f.weights
    assert q[0] == 0.0 and np.array_equal(f.positions_ang[0], v.positions_ang)
    assert w[0] == 0.0 if nodes % 2 == 0 else w[0] > 0.0
    assert np.array_equal(q[2::2], -q[1::2])
    assert np.array_equal(w[2::2], w[1::2])
    assert np.all(np.diff(np.abs(q[1::2])) > 0.0) and np.all(q[1::2] > 0.0)
    assert len(q) == nodes + (nodes % 2 == 0)


def test_a_swings_frames_are_its_own_gauss_rule_as_the_science_works_them():
    """Science § 2a.3, worked: a swing ``Q_0 = 2`` carrying a bell curve of
    ``sigma = 1`` -- three frames at ``0, ±2.65`` with weights ``0.571,
    0.214``; five at ``0, ±2.0656, ±3.9105`` with ``0.4225, 0.2644, 0.0243``;
    on the contact bond ``e^-Q`` (true ``I_0(2) e^{1/2} = 3.7584``) 3.6068,
    3.7568 and, with nine, the true value.  THE SILENT FAILURE: the bell
    curve's frames, blind to the swing's shape (+19 % here)."""
    s = V.State(V.SWING, 1.0, swing_quanta=1.0)      # Q_0 = 2, carried 1
    three = s.frames(3)
    assert list(three.q_amu12_ang[:, 0]) == pytest.approx(
        [0.0, math.sqrt(7.0), -math.sqrt(7.0)], abs=1e-12)
    assert list(three.weights) == pytest.approx([24 / 42, 9 / 42, 9 / 42])
    five = s.frames(5)
    assert list(five.q_amu12_ang[:, 0]) == pytest.approx(
        [0.0, 2.0656, -2.0656, 3.9105, -3.9105], abs=5e-5)
    assert list(five.weights) == pytest.approx(
        [0.4225, 0.2644, 0.2644, 0.0243, 0.0243], abs=5e-5)
    true = special.i0(2.0) * math.exp(0.5)
    for nodes, expected in ((3, 3.6068), (5, 3.7568), (9, true)):
        f = s.frames(nodes)
        got = float((f.weights * np.exp(-f.q_amu12_ang[:, 0])).sum())
        assert got == pytest.approx(expected, abs=5e-5)


@pytest.mark.parametrize("ratio, expected", [
    (1, ((1.9573, 0.160), (2.0815, 0.017), (2.0873, 0.001))),
    (2, ((2.9146, 0.360), (3.6405, 0.087), (3.7494, 0.015))),
    (3, ((4.5101, 0.57), (7.1067, 0.26), (7.9011, 0.083))),
])
def test_frames_re_weighted_to_another_state_follow_the_sciences_table(
        ratio, expected):
    """Science § 5.4's table: three, five and seven frames written for a bell
    curve of spread ``sigma_T = 1`` re-weighted to a swing of ``Q_0 =
    ratio sigma_T`` carrying it, on the contact bond ``e^-Q`` -- each the
    table's average, and the share of the swing beyond the outermost frames
    the table's; the weights reproduce the state's averages of ``1 ...
    Q^(m-1)`` -- and of ``Q^m``, odd, the frames and the state both
    symmetric: exact to degree m; a weight below zero said.  THE SILENT
    FAILURE: a re-averaged number with the motion the frames never reached
    left unsaid."""
    bell = V.State(V.THERMAL, 1.0)
    swing = V.State(V.SWING, 1.0, swing_quanta=ratio ** 2 / 4.0)
    for nodes, (average, beyond) in zip((3, 5, 7), expected):
        q = bell.frames(nodes).q_amu12_ang[:, 0]
        w = swing.weights_at(q)
        assert float((w.weights * np.exp(-q)).sum()) == pytest.approx(
            average, abs=5e-5)
        assert w.beyond == pytest.approx(beyond, abs=0.006)
        assert w.exact_to_degree == len(q)
        for k in range(len(q)):
            assert float((w.weights * q ** k).sum()) == pytest.approx(
                swing.moment(k), rel=1e-9, abs=1e-12)
        assert any("beyond the outermost frames" in s for s in w.said)
    seven = swing.weights_at(bell.frames(7).q_amu12_ang[:, 0])
    assert (ratio == 3) == any("below zero" in s for s in seven.said)
    # EVERY MESSAGE COUNTS FRAMES FROM 1, in the one spelling
    # (`model/structure.md` § 2.2e): the middle frame, frame 0, weighted
    # below zero at Q0 = 3 sigma_T (science § 5.4's table).
    if ratio == 3:
        assert any(s.startswith("frame 1 of 7's weight is")
                   for s in seven.said), seven.said
    with pytest.raises(ValueError, match="frame 2 of 3 and frame 3 of 3 sit "
                                         "at one position"):
        swing.weights_at([0.0, 1.0, 1.0])


def test_a_mean_number_of_quanta_names_its_temperature():
    """Science § 2a.1, § 2a.3: the temperature whose thermal state holds
    ``n`` quanta, ``T = h c nu / (k_B ln(1 + 1/n))`` -- one quantum of a
    1000 cm-1 mode at 2075.7 K by CODATA -- 0 K for none; back through
    Bose and Einstein it is ``n``.  THE SILENT FAILURE: a pumped mode's set
    written at a temperature that holds other quanta than it was pumped to."""
    t = V.temperature_of_quanta(1000.0, 1.0)
    assert t == pytest.approx(codata.h * codata.c * 1.0e5
                              / (codata.k * math.log(2.0)), rel=1e-8)
    assert V.mean_quanta(1000.0, t) == pytest.approx(1.0, rel=1e-12)
    assert V.temperature_of_quanta(1000.0, 0.4) == pytest.approx(
        codata.h * codata.c * 1.0e5 / (codata.k * math.log(1.0 + 1.0 / 0.4)),
        rel=1e-8)
    assert V.temperature_of_quanta(1000.0, 0.0) == 0.0


def test_snapshots_of_a_thermal_motion_have_the_sciences_covariance():
    """Science § 2b.5, § 2c.1: every mode thermal, independent -- the
    snapshots' coordinates uncorrelated, each of variance ``sigma_k^2``;
    each atom's displacement covariance ``sum_k sigma_k^2 L_k,A L_k,A^T``;
    frame 0 the relaxed geometry at weight 0, then ``1/N`` each.  THE SILENT
    FAILURE: modes drawn with one shared number, correlated where the
    harmonic picture has them independent."""
    v = _model_vibration()
    motion = v.motion(300.0)
    snaps = motion.sample(40000, seed=3, antithetic=False)
    assert snaps.weights[0] == 0.0 and np.array_equal(
        snaps.positions_ang[0], v.positions_ang)
    assert snaps.weights[1:] == pytest.approx(np.full(40000, 1 / 40000))
    q = snaps.q_amu12_ang[1:]
    sigma = np.array([motion.states[k].spread_amu12_ang
                      for k in snaps.mode_index_1based])
    sigma2 = sigma ** 2
    cov = np.cov(q.T)
    # each mode's variance within five of its own standard errors, and every
    # pair uncorrelated within five of theirs
    assert np.all(np.abs(np.diag(cov) - sigma2)
                  < 5 * sigma2 * math.sqrt(2 / 40000))
    off = ~np.eye(3, dtype=bool)
    assert np.all(np.abs(cov[off])
                  < 5 * np.outer(sigma, sigma)[off] / math.sqrt(40000))
    rows = np.stack([v.mode(k).eigenvector_canonical_over_atoms_per_amu12
                     for k in (1, 2, 3)])
    expected = np.einsum("k,ki,kj->ij", sigma2, rows[:, 1], rows[:, 1])
    moves = snaps.atom_moves_ang[1:, 1]
    centred = moves - moves.mean(axis=0)
    products = centred[:, :, None] * centred[:, None, :]
    # each element within five of its own standard errors
    assert np.all(np.abs(products.mean(axis=0) - expected)
                  < 5 * products.std(axis=0) / math.sqrt(40000))


def test_a_drive_groups_modes_swing_in_step_by_the_persons_lags():
    """Science § 2c.2, § 2b.5: one laser's modes share one phase per
    snapshot, each at its lag -- ``Cov(Q_a, Q_b) = Q_0a Q_0b cos(l_a - l_b)
    / 2`` -- and the group's average of ``e^{sum a_d Q_d}`` is one factor,
    ``I_0(|sum a_d Q_0d e^{-i l_d}|) prod e^{a_d^2 sigma_0d^2 / 2}``,
    within the set's own error.  THE SILENT FAILURE: the group's modes drawn
    each with its own phase, which no single mode's distribution shows."""
    v = _model_vibration()
    lags = {2: 1.57, 3: 0.12}
    motion = v.motion(300.0).with_drive({2: 2.0, 3: 1.0}, lags_rad=lags)
    snaps = motion.sample(40000, seed=9, antithetic=False)
    cols = [list(snaps.mode_index_1based).index(k) for k in (2, 3)]
    q2, q3 = snaps.q_amu12_ang[1:, cols[0]], snaps.q_amu12_ang[1:, cols[1]]
    q0 = {k: motion.states[k].swing_amplitude_amu12_ang for k in (2, 3)}
    product = (q2 - q2.mean()) * (q3 - q3.mean())
    assert abs(product.mean()
               - 0.5 * q0[2] * q0[3] * math.cos(lags[2] - lags[3])) < \
        5 * product.std() / math.sqrt(40000)
    a = {2: 1.5, 3: -2.0}
    x = np.exp(a[2] * q2 + a[3] * q3)
    phasor = sum(a[k] * q0[k] * np.exp(-1j * lags[k]) for k in (2, 3))
    exact = special.i0(abs(phasor)) * math.exp(sum(
        a[k] ** 2 * motion.states[k].carried_spread_amu12_ang ** 2 / 2
        for k in (2, 3)))
    est = snaps.standard_error(np.concatenate([[np.nan], x]))
    assert abs(est.value - exact) < 4 * est.error


def test_mirrored_pairs_cancel_the_odd_part_exactly():
    """Science § 5.5: each snapshot followed by its mirror, every ``Q``
    negated -- each pair's coordinates sum to zero exactly, a response odd
    in ``Q`` averages to zero exactly; ``parts`` gives a purely odd
    response no even part and a purely even one no odd part; the error of a
    paired set from its pair means, ``X(Q) X(-Q) = 1`` making the pair
    means' variance ``(<X^2> + 1)/2 - <X>^2`` known before the draw.  THE
    SILENT FAILURE: a paired set's error from its frames as if independent,
    wrong both ways."""
    v = _model_vibration()
    motion = v.motion(300.0)
    snaps = motion.sample(80000, seed=4, antithetic=True)
    q = snaps.q_amu12_ang
    assert np.all(q[1::2] + q[2::2] == 0.0)
    odd = q @ np.array([1.0, -2.0, 0.5])
    assert snaps.standard_error(odd).value == 0.0
    assert snaps.parts(odd).even == 0.0
    assert snaps.parts(q[:, 0] ** 2).odd == 0.0
    a = np.array([0.8, -0.6, 1.1])
    x = np.exp(q @ a)
    sigma2 = np.array([motion.states[k].spread_amu12_ang ** 2
                       for k in snaps.mode_index_1based])
    mean = math.exp(float(a ** 2 @ sigma2) / 2)
    mean_sq = math.exp(2 * float(a ** 2 @ sigma2))
    variance = (mean_sq + 1) / 2 - mean ** 2
    pair_means = (x[1::2] + x[2::2]) / 2
    spread = (pair_means - pair_means.mean()) ** 2
    # the pair means' variance against the closed form, within five of the
    # variance estimate's own standard errors -- 40 000 pairs, so an error
    # off by sqrt(2) is far outside them
    assert abs(snaps.standard_error(x).error ** 2 * 40000 - variance) < \
        5 * spread.std() / math.sqrt(40000)
    with pytest.raises(ValueError, match="even"):
        motion.sample(7, seed=4, antithetic=True)


def test_a_seed_holds_a_set_and_its_continuation():
    """`engines/vibration-frames.md` § 3.3: the same seed, the same frames;
    ``sample(2N)`` beginning with ``sample(N)``, pairs or not; ``start=N``
    its next ``N``; a mode driven leaving every other mode's column as it
    was; a drive group's phase keyed by the modes it was made with whatever order the
    groups were added in; a mode in two groups refused; sets pooled the set
    drawn at once, overlapping ranges and pairs with unpaired refused.  THE
    SILENT FAILURE: a driven set and its baseline differing in every mode's
    draws, the difference drowned in their spread."""
    v = _model_vibration()
    motion = v.motion(300.0)
    for pairs in (False, True):
        whole = motion.sample(16, seed=7, antithetic=pairs)
        first = motion.sample(8, seed=7, antithetic=pairs)
        rest = motion.sample(8, seed=7, antithetic=pairs, start=8)
        assert np.array_equal(whole.q_amu12_ang[:9], first.q_amu12_ang)
        assert np.array_equal(whole.q_amu12_ang[9:], rest.q_amu12_ang[1:])
        assert np.array_equal(V.pooled(first, rest).q_amu12_ang,
                              whole.q_amu12_ang)
    driven = motion.with_mode(2, swing_quanta=3.0).sample(16, seed=7,
                                                          antithetic=False)
    base = motion.sample(16, seed=7, antithetic=False)
    assert np.array_equal(driven.q_amu12_ang[:, [0, 2]],
                          base.q_amu12_ang[:, [0, 2]])
    one = (motion.with_drive({1: 1.0}, lags_rad={1: 0.3})
           .with_drive({2: 2.0, 3: 1.0}, lags_rad={2: 1.0, 3: 0.2}))
    other = (motion.with_drive({2: 2.0, 3: 1.0}, lags_rad={2: 1.0, 3: 0.2})
             .with_drive({1: 1.0}, lags_rad={1: 0.3}))
    assert np.array_equal(one.sample(8, seed=2, antithetic=False).q_amu12_ang,
                          other.sample(8, seed=2, antithetic=False).q_amu12_ang)
    with pytest.raises(ValueError, match="one group at most"):
        one.with_drive({3: 1.0}, lags_rad={3: 0.0})
    with pytest.raises(ValueError, match="overlap"):
        V.pooled(first, first)
    with pytest.raises(ValueError, match="pairs pool only with pairs"):
        V.pooled(motion.sample(8, seed=1, antithetic=False),
                 motion.sample(8, seed=2, antithetic=True))


def test_every_analysis_takes_one_value_per_frame_frame_0_first():
    """`engines/vibration-frames.md` § 3.3, § 7: an analysis takes a
    calculation's values whole, one per frame, frame 0 first, and leaves the
    reference out; a value missing drops its pair, said; a pooled set's
    analyses take each set's values whole, in order.  THE SILENT FAILURE:
    frame 0 counted as a snapshot, or the values read one frame off."""
    motion = _model_vibration().motion(300.0)
    snaps = motion.sample(6, seed=1, antithetic=True)
    values = np.array([1e9, 1.0, 3.0, 2.0, 2.0, 5.0, 1.0])
    est = snaps.standard_error(values)
    assert (est.value, est.n) == (pytest.approx(14.0 / 6.0), 6)
    values[3] = np.nan
    gone = snaps.standard_error(values)
    assert (gone.value, gone.n) == (pytest.approx(10.0 / 4.0), 4)
    assert gone.said and "pair" in gone.said[0]
    with pytest.raises(ValueError, match="one value per frame"):
        snaps.standard_error(values[1:])
    pool = V.pooled(snaps, motion.sample(2, seed=1, antithetic=True, start=6))
    assert pool.standard_error(np.array([1e9, 1, 3, 2, 2, 5, 1.0]),
                               np.array([-1e9, 4.0, 4.0])).value == \
        pytest.approx(22.0 / 8.0)


def test_a_motion_says_what_it_takes_and_refuses_by_name():
    """`engines/vibration-frames.md` § 3.1: none left to move refused; the
    record's temperature taken and said; a cold driven mode in a warm motion
    said with its thermal quanta (Bose and Einstein's, from CODATA) and its
    two baselines, ``classical=False`` named in a classical motion;
    a state of another mode refused; a mode set apart twice taking the later
    call's state, said; a drive's lag stated for every mode, never
    defaulted.  THE SILENT FAILURE: a soft mode driven cold in a warm motion
    read as the warm driven mode."""
    v = _model_vibration()
    motion = v.motion()
    assert motion.temperature_from == "the record's"
    assert any("the record's" in s for s in motion.said)
    cold = motion.with_mode(1, swing_quanta=2.0)
    n_th = 1.0 / math.expm1(codata.h * codata.c * 12000.0 / (codata.k * 300.0))
    (said,) = [s for s in cold.said if s.startswith("mode 1 is driven")]
    assert f"{n_th:.3g}" in said
    assert "with_mode(1, temperature_k=0.0) beside the warm others" in said
    assert "with_mode(1, swing_quanta=2, temperature_k=300)" in said
    classical = v.motion(300.0, classical=True).with_mode(1, swing_quanta=2.0)
    assert any("with_mode(1, temperature_k=0.0, classical=False)" in s
               for s in classical.said)
    again = cold.with_mode(1, temperature_k=77.0)
    assert any("set apart again" in s for s in again.said)
    with pytest.raises(ValueError, match="takes a state of mode 2"):
        motion.with_mode(2, state=v.mode(3).state(300.0))
    with pytest.raises(ValueError, match="no lag for mode 3"):
        motion.with_drive({2: 1.0, 3: 1.0}, lags_rad={2: 0.0})
    with pytest.raises(ValueError, match="real mode to move"):
        v.motion(300.0, modes=[])


def test_a_vibrations_geometry_and_its_frequencies_scaled():
    """`engines/vibration-frames.md` § 1, § 3.2: a geometry for coordinates
    the person gives, ``R^0 + sum_k Q_k L_k``, modes by number; every
    frequency scaled at once, each pattern kept, the result's own frequency
    kept beside it, ``Q_zp`` from the new one; a mode the result does not
    have refused by name.  THE SILENT FAILURE: a scaled set's frames placed
    for the result's frequency while saying the scaled one."""
    v = _model_vibration()
    l2 = v.mode(2).eigenvector_canonical_over_atoms_per_amu12
    l3 = v.mode(3).eigenvector_canonical_over_atoms_per_amu12
    assert np.array_equal(v.displaced({}), v.positions_ang)
    assert v.displaced({2: 0.1, 3: -0.05}) == pytest.approx(
        v.positions_ang + 0.1 * l2 - 0.05 * l3, abs=1e-15)
    scaled = v.with_frequency_scale(0.96)
    m = scaled.mode(2)
    assert (m.frequency_cm1, m.result_frequency_cm1) == (
        pytest.approx(672.0), 700.0)
    assert m.zero_point_amplitude_amu12_ang == pytest.approx(
        q_zp_codata(672.0), rel=1e-7)
    assert scaled.frequency_scale == pytest.approx(0.96)
    with pytest.raises(ValueError, match="no mode 9"):
        v.mode(9)


def test_a_driven_modes_normal_numbers_are_its_baselines_own():
    """`engines/vibration-frames.md` § 3.3 (science § 2c.4, § 5.5): a mode
    driven draws its normal numbers from its own stream, the ones its thermal
    baseline drew -- its coordinate ``Q_0 cos(phi) + Q_zp z`` where the
    baseline's is ``sigma_T z``, the phases its own uniform stream's -- so a
    driven set and its baseline, one seed, differ by that mode's swing alone.
    THE SILENT FAILURE: the driven mode drawing other normal numbers, its
    difference from the baseline drowned in their spread."""
    v = _model_vibration()
    motion = v.motion(300.0)
    base = motion.sample(64, seed=17, antithetic=False)
    driven = motion.with_mode(2, swing_quanta=3.0).sample(64, seed=17,
                                                          antithetic=False)
    s = driven.motion.states[2]
    phi = 2 * math.pi * np.random.default_rng([17, 2, 1]).random(64)
    z_driven = ((driven.q_amu12_ang[1:, 1]
                 - s.swing_amplitude_amu12_ang * np.cos(phi))
                / s.carried_spread_amu12_ang)
    z_base = base.q_amu12_ang[1:, 1] / motion.states[2].spread_amu12_ang
    assert z_driven == pytest.approx(z_base, rel=1e-9, abs=1e-12)


def test_a_motions_set_apart_modes_take_its_temperature_and_its_classical_form():
    """`engines/vibration-frames.md` § 2's table, § 3.1: a thermal or
    classical mode set apart with no temperature takes the motion's, said;
    in a classical motion a thermal mode set apart is classical, a driven one
    quantum; a drive group's warmth its ``temperature_k``; a state and a
    state's arguments together refused by name.  THE SILENT FAILURE: a mode
    set apart in a 77 K motion drawn at the record's 300 K."""
    v = _model_vibration(record_temperature_k=300.0)
    cold = v.motion(77.0).with_mode(2, quanta=1).with_mode(2)
    assert (cold.states[2].kind, cold.states[2].temperature_k,
            cold.states[2].temperature_from) == (V.THERMAL, 77.0,
                                                 "the motion's")
    assert "77 K (the motion's)" in cold.set_apart_modes
    classical = v.motion(300.0, classical=True)
    assert classical.with_mode(2).states[2].kind == V.CLASSICAL
    assert classical.with_mode(2, swing_quanta=1.0).states[2].kind == V.SWING
    warm = v.motion(300.0).with_drive({2: 1.0}, lags_rad={2: 0.0},
                                      temperature_k=600.0)
    assert warm.states[2].state.bath_quanta == pytest.approx(
        1.0 / math.expm1(codata.h * codata.c * 70000.0 / (codata.k * 600.0)),
        rel=1e-8)
    with pytest.raises(ValueError, match="not both"):
        v.motion(300.0).with_mode(2, state=v.mode(2).state(300.0),
                                  swing_quanta=1.0)


def test_a_drive_group_keeps_its_phases_when_a_member_is_set_apart_again():
    """`engines/vibration-frames.md` § 3.1, § 3.3: a drive group's phase
    stream is keyed by the modes it was made with, ``[seed, 0, *modes]``, and
    kept when a member is set apart again -- the others draw exactly as
    before; the set-apart modes said in § 5.1's words, the group as it now
    stands, a group of one mode in the singular.  THE SILENT FAILURE: a
    mode set apart re-keying the others' shared phase, every other driven
    mode's draws changed by a call about one mode."""
    v = _model_vibration()
    group = v.motion(300.0).with_drive({2: 2.0, 3: 1.0},
                                       lags_rad={2: 1.57, 3: 0.12})
    both = group.sample(32, seed=4, antithetic=False)
    alone = group.with_mode(2, temperature_k=0.0).sample(32, seed=4,
                                                         antithetic=False)
    assert np.array_equal(alone.q_amu12_ang[:, 2], both.q_amu12_ang[:, 2])
    assert group.set_apart_modes == ("modes 2 and 3: one drive, 2 and 1 "
                                     "swing quanta, lags 1.57 and 0.12 rad, "
                                     "the zero-point width")
    words = group.with_mode(2, temperature_k=0.0).set_apart_modes
    assert words == ("mode 2: thermal at 0 K; mode 3: one drive, 1 swing "
                     "quantum, lag 0.12 rad, the zero-point width")
    # the group's phase is its key's stream, [seed, 0, 2, 3]
    s3 = group.states[3]
    phi = 2 * math.pi * np.random.default_rng([4, 0, 2, 3]).random(32)
    z3 = np.random.default_rng([4, 3, 0]).standard_normal(32)
    assert both.q_amu12_ang[1:, 2] == pytest.approx(
        s3.swing_amplitude_amu12_ang * np.cos(phi - 0.12)
        + s3.carried_spread_amu12_ang * z3, rel=1e-12, abs=1e-15)


def test_what_a_set_says_and_what_pools():
    """`engines/vibration-frames.md` § 3.3, § 4, § 7: a snapshot set says its
    largest single-atom move, its frame and the mode moving it most -- 0 at
    frame 0 -- beside each mode's spread; a mode's frames say their
    outermost move; an imaginary mode is left out of a motion and said; an
    odd start with pairs refused; two motions not pooled, the same result
    read twice pooled, a pooled set's values taken as written or set by set.
    THE SILENT FAILURE: an analysis taking a pooled set's values one frame
    off, or two reads of one result refused as two motions."""
    v = _model_vibration()
    snaps = v.motion(300.0).sample(16, seed=8, antithetic=False)
    largest = snaps.max_atom_displacement_ang
    i = int(np.argmax(largest))
    assert snaps.max_atom_mode_index_1based[0] == 0
    assert f"{largest[i]:.4f} Å, frame {i}" in snaps.said[0]
    frames = v.mode(3).state(300.0).frames(3)
    assert f"{frames.max_atom_displacement_ang.max():.4f} Å" in frames.said[0]
    imaginary = V.Vibration(
        modes=tuple(replace(m, frequency_cm1=-50.0, has_imag=True)
                    if m.index_1based == 1 else m for m in v.modes),
        free_atom_idxs=v.free_atom_idxs, masses_amu=v.masses_amu,
        element_symbols=v.element_symbols, positions_ang=v.positions_ang,
        record_temperature_k=300.0)
    motion = imaginary.motion(300.0)
    assert list(motion.states) == [2, 3]
    assert any("mode 1 left out: imaginary" in s for s in motion.said)
    measured = imaginary.mode(1).with_frequency(50.0).state(300.0)
    back = motion.with_mode(1, state=measured)
    assert not any("left out" in s for s in back.said)
    with pytest.raises(ValueError, match="even start"):
        v.motion(300.0).sample(4, seed=1, antithetic=True, start=3)
    with pytest.raises(ValueError, match="one motion"):
        V.pooled(v.motion(300.0).sample(4, seed=1, antithetic=False),
                 v.motion(77.0).sample(4, seed=2, antithetic=False))
    again = _model_vibration()
    first = v.motion(300.0).sample(4, seed=1, antithetic=False)
    second = again.motion(300.0).sample(4, seed=1, antithetic=False, start=4)
    pool = V.pooled(first, second)
    # frames 1 to 8 hold 1 ... 8: their mean 4.5, frame 0 left out either way
    assert pool.standard_error(np.arange(9.0)).value == pytest.approx(4.5)
    assert pool.standard_error(np.arange(5.0), np.array([-1e9, 5, 6, 7, 8])
                               ).value == pytest.approx(4.5)


def test_two_drives_made_apart_keep_apart_phases():
    """`engines/vibration-frames.md` § 3.3: two lasers' groups draw two
    phases -- a group's key the modes it was made with, unique among the
    groups a motion holds, even after a member left one group and started
    another.  THE SILENT FAILURE: the new group drawing the old one's
    phases, two independent lasers phase-locked without a word."""
    v = _model_vibration()
    motion = (v.motion(300.0)
              .with_drive({1: 2.0, 2: 2.0}, lags_rad={1: 0.0, 2: 0.0})
              .with_mode(1, temperature_k=0.0)
              .with_drive({1: 2.0, 3: 2.0}, lags_rad={1: 0.0, 3: 0.0}))
    assert sorted(d.key for d in motion.drives) == [(1, 2), (1, 3)]
    snaps = motion.sample(20000, seed=6, antithetic=False)
    q2, q3 = snaps.q_amu12_ang[1:, 1], snaps.q_amu12_ang[1:, 2]
    assert abs(np.corrcoef(q2, q3)[0, 1]) < 5 / math.sqrt(20000)


def test_a_motion_keeps_the_results_order_and_one_vibrations_states():
    """`engines/vibration-frames.md` § 3.1, § 3.3: a motion's modes in the
    result's order however it was built; a state from a second read of the
    same result accepted as this vibration's, one from another vibration
    refused by name.  THE SILENT FAILURE: two orders of one motion refused
    by `pooled`, or one result read twice taken for two."""
    v = _model_vibration()
    motion = v.motion(300.0, modes=[3, 1]).with_mode(2, quanta=1)
    assert list(motion.states) == [1, 2, 3]
    assert list(motion.sample(4, seed=1, antithetic=False)
                .mode_index_1based) == [1, 2, 3]
    again = _model_vibration()
    assert motion.with_mode(2, state=again.mode(2).state(77.0)).states[
        2].temperature_k == 77.0
    other = v.with_frequency_scale(0.96)
    with pytest.raises(ValueError, match="of another vibration"):
        motion.with_mode(2, state=other.mode(2).state(77.0))


@pytest.mark.parametrize("state", [
    V.State(V.THERMAL, 1.0, bath_quanta=0.5),
    V.State(V.SWING, 1.0, swing_quanta=2.0),
    V.State(V.NUMBER, 1.0, quanta=2),
], ids=["thermal", "driven", "n quanta"])
@pytest.mark.parametrize("nodes", [3, 4, 5, 7])
def test_a_states_own_frames_weighted_for_it_are_its_own_rule(state, nodes):
    """Science § 5.1, § 5.4; `engines/vibration-frames.md` § 7: frames placed
    for a state, weighted for that same state, are its own Gauss rule --
    their weights the rule's, exact to degree 2m - 1 on its m nodes -- and
    an odd rule's says nothing of frames placed for another motion: no
    share beyond, no blindness.  An even rule's prepended frame 0 at weight
    0 is never said below zero.  THE SILENT FAILURE: a set's own average
    re-weighted to its own state said to be extrapolated, or its zero
    weight said below zero."""
    f = state.frames(nodes)
    w = state.weights_at(f.q_amu12_ang[:, 0])
    assert w.weights == pytest.approx(f.weights, abs=1e-12)
    assert w.exact_to_degree == 2 * nodes - 1, w
    assert not any("below zero" in s for s in w.said), w.said
    if nodes % 2:
        assert w.said == (), w.said


def test_an_analysis_says_what_it_dropped_and_few_frames_say_their_degree():
    """`engines/vibration-frames.md` § 7; science § 5.4 item 2: a pair with
    a value missing left out of `parts` and said; none left said; three
    frames re-weighted said to be blind to the shape, two exact only to
    degree 1; one mirrored pair said to have no error.  THE SILENT FAILURE:
    a split or an error from fewer pairs than the person thinks."""
    v = _model_vibration()
    snaps = v.motion(300.0).sample(6, seed=1, antithetic=True)
    values = np.array([0.0, 1.0, 3.0, np.nan, 2.0, 5.0, 1.0])
    assert snaps.parts(values).said == ("1 pair left out: a value missing",)
    nothing = snaps.standard_error(np.full(7, np.nan))
    assert "no pair left" in nothing.said[-1] and math.isnan(nothing.error)
    bell = V.State(V.THERMAL, 1.0)
    swing = V.State(V.SWING, 1.0, swing_quanta=1.0)
    three = swing.weights_at(bell.frames(3).q_amu12_ang[:, 0])
    assert any("three frames carry only the curvature" in s
               for s in three.said)
    two = swing.weights_at([-1.0, 1.0])
    assert any("two frames are exact only to degree 1" in s for s in two.said)
    pair = v.motion(300.0).sample(2, seed=1, antithetic=True)
    assert any("one mirrored pair" in s for s in pair.said)


def test_the_mode_moving_a_frames_atom_most_is_judged_by_its_move():
    """`engines/vibration-frames.md` § 3.3: each frame's mode moving its
    most-moved atom most, by the move ``|Q_k L_k,A|`` at that atom -- not by
    the largest coordinate.  THE SILENT FAILURE: a soft mode's large
    coordinate named for a move another mode makes."""
    v = _model_vibration()
    snaps = v.motion(300.0).sample(64, seed=8, antithetic=False)
    moves = np.linalg.norm(snaps.atom_moves_ang, axis=-1)
    expected, by_q = [], []
    for i in range(1, 65):
        atom = int(moves[i].argmax())
        each = [abs(snaps.q_amu12_ang[i, c]) * np.linalg.norm(
            v.mode(int(k)).eigenvector_canonical_over_atoms_per_amu12[atom])
            for c, k in enumerate(snaps.mode_index_1based)]
        expected.append(int(snaps.mode_index_1based[int(np.argmax(each))]))
        by_q.append(int(snaps.mode_index_1based[
            int(np.abs(snaps.q_amu12_ang[i]).argmax())]))
    assert list(snaps.max_atom_mode_index_1based[1:]) == expected
    assert expected != by_q


@pytest.mark.parametrize("n", [0, 1, 3])
def test_n_quantas_average_of_an_exponential_is_laguerres(n):
    """Science § 2b.5: over n quanta exactly, ``<e^{aQ}> = e^{lambda^2/2}
    L_n(-lambda^2)``, ``lambda = a Q_zp``, ``L_n`` Laguerre's polynomial --
    reached by the state's own Gauss rule as its frames grow.  THE SILENT
    FAILURE: n quanta averaged as a bell curve of the same spread."""
    s = V.State(V.NUMBER, 1.0, quanta=n)
    lam = 0.8
    f = s.frames(14)
    got = float((f.weights * np.exp(lam * f.q_amu12_ang[:, 0])).sum())
    assert got == pytest.approx(
        math.exp(lam ** 2 / 2) * special.eval_laguerre(n, -lam ** 2),
        rel=1e-10)


def test_the_refusals_the_contract_states_and_one_set_pooled():
    """`engines/vibration-frames.md` § 3.1, § 3.3, § 7: `parts` of a set
    drawn without pairs refused by name; a bare `State` given as a mode's
    state refused, naming the call that makes one; one set pooled is that
    set, none refused -- no floor of ours; a `ModeState` handing on its
    state's own fields.  THE SILENT FAILURE: a split of an unpaired set read
    as an even and an odd part."""
    v = _model_vibration()
    motion = v.motion(300.0)
    alone = motion.sample(8, seed=1, antithetic=False)
    with pytest.raises(ValueError, match="drawn alone"):
        alone.parts(np.zeros(9))
    with pytest.raises(ValueError, match=r"v\.mode\(2\)\.state"):
        motion.with_mode(2, state=V.State(V.THERMAL, 0.1))
    assert V.pooled(alone) is alone
    with pytest.raises(ValueError, match="a set to pool"):
        V.pooled()
    driven = v.mode(2).state(swing_quanta=4.0)
    assert (driven.swing_quanta, driven.bath_quanta) == (4.0, 0.0)
