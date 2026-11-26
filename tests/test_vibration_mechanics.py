"""A harmonic mode's mechanics, `vibration`, against the science's closed forms
(`science/vibrational-averaging.md` § 2b.5; `engines/vibration.md` § 6.11
§ 7).  Every expected number comes from the theory -- the oscillator's
eigenfunctions, Mehler's sum, Gauss quadrature -- or from CODATA's constants (`scipy.constants`), never from
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
        V.bose_einstein_mean_quanta(-700.0, 298.15)
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


def test_n_quanta_beyond_its_rules_reach_from_zero_keeps_its_own_rule():
    """§ 2b.3, § 2b.4: a dozen quanta -- more than any rule's nodes reach
    from the ground state, so its rule is taken on the states ``|n>``
    reaches alone -- answers its moments ``<Q^2> = (2n+1) Q_zp^2`` and
    ``<Q^4> = 3 (2n^2+2n+1) Q_zp^4``, and each rule's frames reproduce them
    to its degree.  The silent failure: a rule taken on the wrong states,
    its frames another state's."""
    n = 12
    q = q_zp_codata(4400.0)
    s = h2_mode().state(quanta=n).state
    # 1e-8: molbuilder's constants against CODATA's, as the density's row.
    assert s.moment(2) == pytest.approx((2 * n + 1) * q * q, rel=1e-8)
    assert s.moment(4) == pytest.approx(3 * (2 * n * n + 2 * n + 1) * q ** 4,
                                        rel=1e-8)
    for m in (3, 5, 7):
        f = s.frames(m)
        x, w = f.q_amu12_ang[:, 0], f.weights
        for k in range(0, 2 * m):
            assert float(np.sum(w * x ** k)) == pytest.approx(
                s.moment(k), rel=1e-9, abs=1e-12 * q ** k), (m, k)


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


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


Q_ZP_4400 = q_zp_codata(4400.0)
SIGMA_THERMAL = Q_ZP_4400 * math.sqrt(2 * 0.7 + 1)          # 0.7 mean quanta
SIGMA_CLASSICAL = math.sqrt(codata.k * 3000.0 / omega_si(4400.0) ** 2
                            / AMU_KG) * 1e10                # 3000 K


@pytest.mark.parametrize("asked, x, expected", [
    (dict(mean_quanta=0.7), SIGMA_THERMAL, _normal_cdf(1.0)),
    (dict(mean_quanta=0.7), -2 * SIGMA_THERMAL, _normal_cdf(-2.0)),
    (dict(temperature_k=3000.0, classical=True), SIGMA_CLASSICAL,
     _normal_cdf(1.0)),
    (dict(temperature_k=0.0, classical=True), 0.0, 1.0),
    (dict(temperature_k=0.0, classical=True), -1.0e-6, 0.0),
], ids=["thermal at one spread", "thermal two spreads below",
        "classical at one spread", "classical at 0 K, at rest",
        "classical at 0 K, below rest"])
def test_each_states_distribution_is_its_closed_form(asked, x, expected):
    """§ 2b.3: the chance that ``Q`` is ``x`` or less -- the Gaussian's
    ``(1 + erf(x / sqrt(2) sigma)) / 2``; all of it at rest for a classical
    mode at 0 K.  THE SILENT FAILURE: a classical mode given the zero-point
    width."""
    s = V.state(Q_ZP_4400, 4400.0, **asked)
    assert s.cdf(x) == pytest.approx(expected, rel=1e-7, abs=1e-12)


def test_a_classical_mode_at_rest_sits_at_zero():
    """§ 2b.3: classically, at 0 K, the molecule is at rest at ``Q = 0`` --
    all of it there: a point, its density infinite at 0 and nothing
    elsewhere, no spread, every frame at 0."""
    s = V.state(Q_ZP_4400, 4400.0, temperature_k=0.0, classical=True)
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


def test_a_classical_state_takes_the_records_temperature_and_says_so():
    """`engines/vibration-frames.md` § 2: a classical state given no
    temperature takes the record's and says so -- its spread the classical
    bell curve's at it, ``sqrt(k_B T) / omega``, by CODATA; n quanta takes
    none.  THE SILENT FAILURE: a classical state at a temperature no one
    named, unsaid."""
    counted = h2_mode(700.0, temperature_k=500.0).state(quanta=2)
    assert (counted.temperature_k, counted.temperature_from) == (None, None)
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
    """The model molecule (`support.model_vibration`, the one builder): the
    closed forms of science § 2b.5 and § 2c hold for any such set."""
    from support.model_vibration import model_vibration
    return model_vibration(record_temperature_k)


@pytest.mark.parametrize("asked", [
    dict(temperature_k=300.0), dict(quanta=3),
    dict(temperature_k=600.0, classical=True),
], ids=["thermal", "3 quanta", "classical"])
def test_each_states_draws_follow_its_own_distribution(asked):
    """§ 2c.3, `engines/vibration-frames.md` § 2: a state's draws follow its
    distribution (a two-sided Kolmogorov test against its ``cdf``), one
    standard normal number per draw from the mode's own stream -- for n
    quanta the ``Q`` where the distribution reaches ``Phi(z)``.  THE SILENT
    FAILURE: a state drawn from another's distribution, a bell curve for
    n quanta, which no average of a few frames would show."""
    from scipy import stats
    s = _model_vibration().mode(2).state(**asked)
    x = s.draw(4000, seed=12)
    assert stats.kstest(x, s.cdf).pvalue > 1e-3
    if s.kind == V.LEVEL:
        z = np.random.default_rng([12, 2, 0]).standard_normal(4000)
        assert s.cdf(x) == pytest.approx(stats.norm.cdf(z), abs=1e-9)


def test_a_draw_continues_its_seed_from_the_modes_own_stream():
    """`engines/vibration-frames.md` § 2, § 3.3: draws ``start ...`` of a
    seed's sequence are the same however many were asked; each mode draws
    from its own streams, keyed by its number; the seed is required.  THE
    SILENT FAILURE: a set made larger later drawing other frames than its
    first part."""
    v = _model_vibration()
    s = v.mode(2).state(quanta=1)
    whole = s.draw(16, seed=5)
    assert np.array_equal(s.draw(8, seed=5), whole[:8])
    assert np.array_equal(s.draw(8, seed=5, start=8), whole[8:])
    assert not np.array_equal(v.mode(3).state(quanta=1).draw(16, seed=5),
                              whole)
    with pytest.raises(ValueError, match="seed"):
        s.draw(4, seed=None)


@pytest.mark.parametrize("asked, nodes", [
    (dict(temperature_k=300.0), 3), (dict(temperature_k=300.0), 5),
    (dict(quanta=2), 5), (dict(quanta=3), 7),
    (dict(temperature_k=300.0), 4), (dict(quanta=1), 4),
], ids=["thermal 3", "thermal 5", "2 quanta 5", "3 quanta 7", "thermal 4",
        "1 quantum 4"])
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


@pytest.mark.parametrize("ratio", [1.5, 2.0, 3.0])
def test_frames_re_weighted_to_another_state_reproduce_its_averages(ratio):
    """Science § 5.4; `engines/vibration-frames.md` § 7: three, five and
    seven frames written for a bell curve of spread 1 re-weighted to a
    thermal state ``ratio`` times as wide -- the weights reproduce its
    averages of ``1 ... Q^(m-1)``, and of ``Q^m``, odd, the frames and the
    state both symmetric: exact to degree m; the share of it beyond the
    outermost frames ``erfc(q_max / ratio sqrt 2)``, said; a weight below
    zero said wherever one is.  THE SILENT FAILURE: a re-averaged number
    with the motion the frames never reached left unsaid."""
    bell = V.State(V.THERMAL, 1.0)
    wide = V.State(V.THERMAL, 1.0, thermal_quanta=(ratio ** 2 - 1.0) / 2.0)
    for nodes in (3, 5, 7):
        q = bell.frames(nodes).q_amu12_ang[:, 0]
        w = wide.weights_at(q)
        assert w.exact_to_degree == len(q)
        for k in range(len(q) + 1):
            terms = w.weights * q ** k       # odd k: terms cancelling to 0
            assert float(terms.sum()) == pytest.approx(
                wide.moment(k), rel=1e-9,
                abs=1e-12 * float(np.abs(terms).sum()))
        assert w.beyond == pytest.approx(
            math.erfc(q.max() / (ratio * math.sqrt(2.0))), rel=1e-9)
        assert any("beyond the outermost frames" in s for s in w.said)
        below = {f"frame {i + 1} of {len(q)}"
                 for i, x in enumerate(w.weights) if x < -1e-12}
        assert {s.split("'s weight")[0] for s in w.said
                if "below zero" in s} == below
    with pytest.raises(ValueError, match="frame 2 of 3 and frame 3 of 3 sit "
                                         "at one position"):
        wide.weights_at([0.0, 1.0, 1.0])


def test_a_mean_number_of_quanta_names_its_temperature():
    """Science § 2a.1, § 2a.3: the temperature whose thermal state holds
    ``n`` quanta, ``T = h c nu / (k_B ln(1 + 1/n))`` -- one quantum of a
    1000 cm-1 mode at 2075.7 K by CODATA -- 0 K for none; back through
    Bose and Einstein it is ``n``.  THE SILENT FAILURE: a pumped mode's set
    written at a temperature that holds other quanta than it was pumped to."""
    t = V.bose_einstein_temperature(1000.0, 1.0)
    assert t == pytest.approx(codata.h * codata.c * 1.0e5
                              / (codata.k * math.log(2.0)), rel=1e-8)
    assert V.bose_einstein_mean_quanta(1000.0, t) == pytest.approx(1.0, rel=1e-12)
    assert V.bose_einstein_temperature(1000.0, 0.4) == pytest.approx(
        codata.h * codata.c * 1.0e5 / (codata.k * math.log(1.0 + 1.0 / 0.4)),
        rel=1e-8)
    assert V.bose_einstein_temperature(1000.0, 0.0) == 0.0


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
    its next ``N``; a mode set apart leaving every other mode's column as it
    was.  THE SILENT FAILURE: a set and its variation differing in every
    mode's draws, the difference drowned in their spread."""
    v = _model_vibration()
    motion = v.motion(300.0)
    for pairs in (False, True):
        whole = motion.sample(16, seed=7, antithetic=pairs)
        first = motion.sample(8, seed=7, antithetic=pairs)
        rest = motion.sample(8, seed=7, antithetic=pairs, start=8)
        assert np.array_equal(whole.q_amu12_ang[:9], first.q_amu12_ang)
        assert np.array_equal(whole.q_amu12_ang[9:], rest.q_amu12_ang[1:])
    apart = motion.with_mode(2, quanta=3).sample(16, seed=7,
                                                 antithetic=False)
    base = motion.sample(16, seed=7, antithetic=False)
    assert np.array_equal(apart.q_amu12_ang[:, [0, 2]],
                          base.q_amu12_ang[:, [0, 2]])


def test_every_analysis_takes_one_value_per_frame_frame_0_first():
    """`engines/vibration-frames.md` § 3.3, § 7: an analysis takes a
    calculation's values whole, one per frame, frame 0 first, and leaves the
    reference out; a value missing drops its pair, said.  THE SILENT
    FAILURE:
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


def test_a_motion_says_what_it_takes_and_refuses_by_name():
    """`engines/vibration-frames.md` § 3.1: none left to move refused; the
    record's temperature taken and said; a state of another mode refused; a
    mode set apart twice taking the later call's state, said.  THE SILENT
    FAILURE: a motion at a temperature no one named, unsaid."""
    v = _model_vibration()
    motion = v.motion()
    assert motion.temperature_from == "the record's"
    assert any("the record's" in s for s in motion.said)
    again = motion.with_mode(1, quanta=2).with_mode(1, temperature_k=77.0)
    assert any("set apart again" in s for s in again.said)
    with pytest.raises(ValueError, match="takes a state of mode 2"):
        motion.with_mode(2, state=v.mode(3).state(300.0))
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


def test_a_motions_set_apart_modes_take_its_temperature_and_its_classical_form():
    """`engines/vibration-frames.md` § 2's table, § 3.1: a thermal or
    classical mode set apart with no temperature takes the motion's, said;
    in a classical motion a thermal mode set apart is classical; a state and
    a state's arguments together refused by name.  THE SILENT FAILURE: a
    mode set apart in a 77 K motion drawn at the record's 300 K."""
    v = _model_vibration(record_temperature_k=300.0)
    cold = v.motion(77.0).with_mode(2, quanta=1).with_mode(2)
    assert (cold.states[2].kind, cold.states[2].temperature_k,
            cold.states[2].temperature_from) == (V.THERMAL, 77.0,
                                                 "the motion's")
    assert "77 K (the motion's)" in cold.set_apart_modes
    classical = v.motion(300.0, classical=True)
    assert classical.with_mode(2).states[2].kind == V.CLASSICAL
    assert classical.with_mode(2, quanta=1).states[2].kind == V.LEVEL
    with pytest.raises(ValueError, match="not both"):
        v.motion(300.0).with_mode(2, state=v.mode(2).state(300.0), quanta=1)


def test_what_a_set_says():
    """`engines/vibration-frames.md` § 3.3, § 4: a snapshot set says its
    largest single-atom move, its frame and the mode moving it most -- 0 at
    frame 0 -- beside each mode's spread; a mode's frames say their
    outermost move; an imaginary mode is left out of a motion and said; an
    odd start with pairs refused.  THE SILENT FAILURE: a frame named one
    off, or an imaginary mode drawn as if it had a harmonic state."""
    v = _model_vibration()
    snaps = v.motion(300.0).sample(16, seed=8, antithetic=False)
    largest = snaps.max_atom_displacement_ang
    i = int(np.argmax(largest))
    assert snaps.max_atom_mode_index_1based[0] == 0
    # A FRAME NAMED FROM 1, as every message names one (`model/structure.md`
    # § 2.2e).
    assert f"{largest[i]:.4f} Å, frame {i + 1} of {len(largest)}" \
        in snaps.said[0]
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


def test_a_motion_keeps_the_results_order_and_one_vibrations_states():
    """`engines/vibration-frames.md` § 3.1, § 3.3: a motion's modes in the
    result's order however it was built; a state from a second read of the
    same result accepted as this vibration's, one from another vibration
    refused by name.  THE SILENT FAILURE: a set's columns in the order the
    calls were made, or one result read twice taken for two."""
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
    V.State(V.THERMAL, 1.0, thermal_quanta=0.5),
    V.State(V.LEVEL, 1.0, quanta=2),
], ids=["thermal", "n quanta"])
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
    wide = V.State(V.THERMAL, 1.0, thermal_quanta=1.5)
    three = wide.weights_at(bell.frames(3).q_amu12_ang[:, 0])
    assert any("three frames carry only the curvature" in s
               for s in three.said)
    two = wide.weights_at([-1.0, 1.0])
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
    s = V.State(V.LEVEL, 1.0, quanta=n)
    lam = 0.8
    f = s.frames(14)
    got = float((f.weights * np.exp(lam * f.q_amu12_ang[:, 0])).sum())
    assert got == pytest.approx(
        math.exp(lam ** 2 / 2) * special.eval_laguerre(n, -lam ** 2),
        rel=1e-10)


def test_the_refusals_the_contract_states():
    """`engines/vibration-frames.md` § 3.1, § 7: `parts` of a set drawn
    without pairs refused by name; a bare `State` given as a mode's state
    refused, naming the call that makes one; a `ModeState` handing on its
    state's own fields.  THE SILENT FAILURE: a split of an unpaired set read
    as an even and an odd part."""
    v = _model_vibration()
    motion = v.motion(300.0)
    alone = motion.sample(8, seed=1, antithetic=False)
    with pytest.raises(ValueError, match="drawn alone"):
        alone.parts(np.zeros(9))
    with pytest.raises(ValueError, match=r"v\.mode\(2\)\.state"):
        motion.with_mode(2, state=V.State(V.THERMAL, 0.1))
    counted = v.mode(2).state(quanta=4)
    assert (counted.quanta, counted.thermal_quanta) == (4, 0.0)


@pytest.mark.parametrize("steps, chosen", [
    (lambda s: s, [1, 2, 3]),
    (lambda s: s.window(500.0, None), [2, 3]),
    (lambda s: s.window(None, 1000.0), [1, 2]),
    (lambda s: s.window(500.0, None).add([1]).drop([3]), [1, 2]),
    (lambda s: s.drop([3]).window(500.0, None).add([1]), [1, 2]),
    (lambda s: s.only([3, 1]), [1, 3]),
    (lambda s: s.only([3, 1]).window(100.0, 1000.0), [1]),
    (lambda s: s.window(500.0, None).only([1, 2]), [2]),
    (lambda s: s.drop([2]).only([1, 2, 3]), [1, 3]),
], ids=["every real mode", "from 500 cm-1", "up to 1000 cm-1",
        "a window, one added, one dropped", "the same steps in another order",
        "only two, in the result's order", "only two, then a window",
        "a window, then only of what it left", "a drop, then only"])
def test_the_modes_that_move_are_chosen_in_steps(steps, chosen):
    """`engines/vibration-frames.md` § 3.1: a mode selection starts as every
    real mode and each step acts on what the step before left -- a window
    keeps the modes inside it (the model's modes at 120, 700 and 1600 cm-1),
    ``only`` keeps the listed, ``add`` and ``drop`` put in and take out --
    the final list in the result's order; the motion moves exactly those
    and says the steps.  THE SILENT FAILURE: a soft mode the person meant to
    leave out moving every snapshot."""
    v = _model_vibration()
    sel = steps(v.mode_selection())
    assert sel.final() == chosen
    motion = v.motion(300.0, modes=sel)
    assert list(motion.states) == chosen
    assert any(s == "the modes chosen: " + "; ".join(sel.steps)
               for s in motion.said)


def test_a_mode_selections_steps_are_said_in_the_contracts_words():
    """`engines/vibration-frames.md` § 3.1: the steps said in order --
    *every real mode; 500 cm-1 and up; mode 1 added; mode 3 dropped* -- the
    modes a step lists said as written, an empty list as no mode.  THE
    SILENT FAILURE: a set's record naming steps that are not the ones
    taken."""
    v = _model_vibration()
    sel = v.mode_selection().window(500.0, None).add([1]).drop([3])
    assert sel.steps == ("every real mode", "500 cm-1 and up",
                         "mode 1 added", "mode 3 dropped")
    assert v.mode_selection().add([]).steps[-1] == "no mode added"
    assert [v.mode_selection().only([3, 1]).steps[-1],
            v.mode_selection().add([3, 1]).steps[-1]] == [
        "only modes 3 and 1", "modes 3 and 1 added"]


@pytest.mark.parametrize("make, refused", [
    (lambda v: v.mode_selection().window(2000.0, None).final(), "no mode"),
    (lambda v: v.mode_selection().drop([1, 2, 3]).final(), "no mode"),
    (lambda v: v.mode_selection().add([9]), "no mode 9"),
    (lambda v: v.mode_selection().add(True), "named by its number"),
    (lambda v: v.mode_selection().window(900.0, 100.0), "high end"),
    (lambda v: v.mode_selection().window("50", None), "a number of cm-1"),
    (lambda v: v.with_frequency_scale(0.96).motion(
        300.0, modes=v.mode_selection()), "the vibration it was made from"),
], ids=["a window holding none", "every mode dropped", "a mode the result "
        "has none of", "a truth value for a mode", "a window backwards",
        "a window end in words",
        "a selection of another vibration"])
def test_a_mode_selection_that_moves_nothing_is_refused_by_name(make, refused):
    """`engines/vibration-frames.md` § 3.1: a selection that holds no mode
    at its end -- nothing would move -- a number the result holds no mode
    of, a window backwards or not in numbers, and a selection made of
    another vibration, each refused by name.  THE SILENT FAILURE: a motion
    of no mode, or of another molecule's."""
    with pytest.raises(ValueError, match=refused):
        make(_model_vibration())
