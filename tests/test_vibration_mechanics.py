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


def h2_mode(nu_cm1: float = 4400.0, temperature_k=None) -> V.Mode:
    """H2's stretch along z: each atom ``±1/sqrt(2 m)``, the canonical form
    (``sum m |L|^2 = 1``)."""
    a = 1.0 / math.sqrt(2.0 * M_H)
    return V.Mode(index_1based=1, frequency_cm1=nu_cm1, has_imag=False,
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
    mode = V.Mode(index_1based=1, frequency_cm1=nu, has_imag=False,
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
    imaginary = V.Mode(index_1based=1, frequency_cm1=-4400.0, has_imag=True,
                       eigenvector_canonical_per_amu12=(
                           h2_mode().eigenvector_canonical_per_amu12),
                       free_atom_idxs=(0, 1), masses_amu=h2_mode().masses_amu,
                       element_symbols=("H", "H"),
                       positions_ang=h2_mode().positions_ang)
    assert imaginary.force_constant_mdyn_per_ang == pytest.approx(
        k_n_per_m / 100.0, rel=1e-8)
    still = V.Mode(index_1based=1, frequency_cm1=4400.0, has_imag=False,
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
    mode = V.Mode(index_1based=1, frequency_cm1=4400.0, has_imag=False,
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
    mode = replace(mode, regions=regions)
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
    mode = V.Mode(index_1based=1, frequency_cm1=nu, has_imag=False,
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
    over a bath of spread ``sigma_T``: ``<Q^2> = sigma_T^2 + Q_0^2/2``,
    ``<Q^4> = 3 sigma_T^4 + 3 sigma_T^2 Q_0^2 + 3/8 Q_0^4``, its density the
    bath's Gaussian averaged over the swing's phase -- a swing two hundred
    of its bath's spreads wide as much as one of three -- its mean quanta the
    bath's and the swing's added, its spread the one they give."""
    mode = h2_mode()
    s = mode.state(mean_quanta=nbar, swing_quanta=n_s).state
    q = mode.zero_point_amplitude_amu12_ang
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
    (dict(mean_quanta=0.4, swing_quanta=2.0), Q0_TWO_QUANTA / 2,
     _swing_over_bath_cdf(Q0_TWO_QUANTA / 2,
                          Q_ZP_4400 * math.sqrt(1.8), Q0_TWO_QUANTA)),
    (dict(mean_quanta=0.4, swing_quanta=2.0), -1.2 * Q0_TWO_QUANTA,
     _swing_over_bath_cdf(-1.2 * Q0_TWO_QUANTA,
                          Q_ZP_4400 * math.sqrt(1.8), Q0_TWO_QUANTA)),
], ids=["thermal at one spread", "thermal two spreads below",
        "classical at one spread", "a classical swing halfway out",
        "a classical swing halfway in", "a classical swing of no quanta",
        "classical at 0 K, at rest", "classical at 0 K, below rest",
        "a swing over a bath halfway out", "a swing over a bath beyond"])
def test_each_states_distribution_is_its_closed_form(asked, x, expected):
    """§ 2b.3, § 2a.2: the chance that ``Q`` is ``x`` or less -- the
    Gaussian's ``(1 + erf(x / sqrt(2) sigma)) / 2``; the classical swing's
    ``1/2 + asin(x / Q_0) / pi``; all of it at rest for no quanta, and for
    a classical mode at 0 K; a swing over a bath, the bath's distribution
    averaged over the swing's phase."""
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
    assert set(s.frames(3).nodes_amu12_ang) == {0.0}


# --------------------------------------------------------------------- #
#  The frames (§ 2b.4)                                                   #
# --------------------------------------------------------------------- #

def _moment_of(frames, k):
    return sum(w * x ** k for x, w in zip(frames.nodes_amu12_ang,
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
        assert sorted(f.nodes_amu12_ang) == pytest.approx(
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


def test_a_swings_equal_phases_are_exact_to_degree_seven_on_five_frames():
    """§ 2a.3: eight equal phases fold onto five positions -- ``0``,
    ``±Q_0/sqrt(2)``, ``±Q_0`` at 1/4, 1/4, 1/8 -- exact to degree 7, and
    not to 8."""
    mode = h2_mode()
    s = mode.state(swing_quanta=2.0, classical=True).state
    q0 = 2 * mode.zero_point_amplitude_amu12_ang * math.sqrt(2.0)
    f = s.frames(5)
    pairs = sorted(zip(f.nodes_amu12_ang, f.weights))
    assert [x for x, _ in pairs] == pytest.approx(
        [-q0, -q0 / math.sqrt(2), 0.0, q0 / math.sqrt(2), q0], abs=1e-15)
    assert [w for _, w in pairs] == pytest.approx(
        [1 / 8, 1 / 4, 1 / 4, 1 / 4, 1 / 8])
    assert f.exact_to_degree == 7
    for k in range(0, 8):
        exact = q0 ** k * (math.comb(k, k // 2) / 2 ** k if k % 2 == 0 else 0)
        assert _moment_of(f, k) == pytest.approx(exact, abs=1e-15)
    assert _moment_of(f, 8) != pytest.approx(
        q0 ** 8 * math.comb(8, 4) / 2 ** 8, rel=1e-3)


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
    mode = V.Mode(index_1based=1, frequency_cm1=-120.0, has_imag=True,
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
    for q, geometry in zip(got.nodes_amu12_ang, got.positions_ang):
        assert geometry == pytest.approx(
            mode.positions_ang
            + q * mode.eigenvector_canonical_over_atoms_per_amu12, abs=1e-15)
    assert sum(got.weights) == pytest.approx(1.0)


@pytest.mark.parametrize("nodes", [3, 5])
def test_a_swing_over_a_bath_is_averaged_exactly_to_its_frames_degree(nodes):
    """§ 2a.3: a swing over a bath -- each phase position carrying the bath's
    own rule, each frame's weight the product -- reproduces the state's
    moments, ``E[(X + Q_0 cos phi)^k]`` over the bath ``X`` and the phase, to
    the lesser of the phases' degree ``2 (m - 1) - 1`` and the bath rule's
    ``2 m - 1`` (§ 5.1), its weights summing to 1."""
    mode = h2_mode()
    s = mode.state(mean_quanta=0.4, swing_quanta=3.0).state
    q = mode.zero_point_amplitude_amu12_ang
    st, q0 = q * math.sqrt(2 * 0.4 + 1), 2 * q * math.sqrt(3.0)
    f = s.frames(nodes)
    assert f.exact_to_degree == 2 * (nodes - 1) - 1
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


def test_a_swings_bath_and_a_classical_state_take_the_records_temperature():
    """§ 6.11 § 9 item 6: the person's door takes the record's temperature
    for a swing's bath and for a classical state, and says so; a classical
    swing alone takes none."""
    taken = h2_mode(700.0, temperature_k=500.0).state(swing_quanta=2.0)
    assert (taken.temperature_k, taken.temperature_from) == (
        500.0, "the record's")
    assert taken.mean_quanta == pytest.approx(
        1.0 / math.expm1(codata.h * codata.c * 70000.0 / (codata.k * 500.0))
        + 2.0, rel=1e-8)
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
    mode = V.Mode(index_1based=1, frequency_cm1=4400.0, has_imag=False,
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
    for q, geometry in zip(frames.nodes_amu12_ang + (0.05,),
                           frames.positions_ang + (mode.displaced(0.05),)):
        assert geometry == pytest.approx(
            np.array([at[0], at[1] + q * rows[0], at[2] + q * rows[1]]),
            abs=1e-15)
