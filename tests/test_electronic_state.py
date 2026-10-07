"""The electronic state, through the road: describe -> ``jobset prep`` -> the deck.

PINS: ``docs/science/chemistry-correctness.md`` § 2a -- one class decides a
calculation's charge and spin, a blank item is the instruction "work it out",
and every reader (the deck writers, the settings gate, the form) reads the
one answer.  Each case here prepares a real deck and asserts what it carries,
or what prep refused and why.

PREVENTS, each measured before 2026-09-28:

* a formate ion prepared at -1 told to go open-shell (the count ignored its
  charge), and a bulk gold electrode told the same (its odd count per CELL was
  read as an unpaired electron);
* a blank charge written as 0 and a person's Hartree-Fock turned into DFT by
  a form fill, and ``dft.RKS`` with a nonzero spin re-ruled by PySCF into
  ROKS without a word;
* a stage able to change the spin between rungs, the ``.DM`` carried across.

Nothing here launches an engine: prep writes the deck and stops.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pytest

from molbuilder.config.pyscf import PySCFConfig
from molbuilder.config.siesta import SiestaConfig
from molbuilder.pyscf.stages import default_pyscf_stages, vibration_stages
from molbuilder.siesta.stages import default_siesta_stages
from molbuilder.structure import Structure

from test_engine_offset_reaches_every_deck import _prep


# --------------------------------------------------------------- structures

def _mol(elements, positions, vacuum=8.0, **kw):
    return Structure(elements=list(elements),
                     positions=np.asarray(positions, dtype=float),
                     vacuum=(vacuum,) * 3, **kw)


WATER = lambda: _mol("OHH", [[0, 0, 0], [0.757, 0.586, 0],
                             [-0.757, 0.586, 0]])
METHYL = lambda: _mol("CHHH", [[0, 0, 0], [1.08, 0, 0], [-0.54, 0.935, 0],
                               [-0.54, -0.935, 0]])                # 9 e-
FORMATE = lambda: _mol(["C", "O", "O", "H"],
                       [[0, 0, 0], [1.26, 0, 0], [-0.63, 1.09, 0],
                        [-0.55, -0.95, 0]])                        # 23 e- neutral
FE_AQUA = lambda: _mol(["Fe", "O", "H", "H"],
                       [[0, 0, 0], [2.0, 0, 0], [2.6, 0.76, 0],
                        [2.6, -0.76, 0]])                           # 36 e-
AU4 = lambda: _mol(["Au"] * 4, [[0, 0, 0], [2.8, 0, 0], [1.4, 2.42, 0],
                                [1.4, 0.81, 2.29]])
AU1 = lambda: _mol(["Au"], [[0, 0, 0]])                            # 79 e-
O2 = lambda: _mol(["O", "O"], [[0, 0, 0], [1.21, 0, 0]])           # a triplet


def _gold_lead():
    """Three gold atoms in a repeating cell: 237 electrons per cell, odd --
    the count that was read as an unpaired electron."""
    a = 2.884
    return Structure(elements=["Au"] * 3,
                     positions=np.array([[0, 0, 0], [a / 2, a * 0.866, 0],
                                         [0, 0, 2.355]]),
                     cell=np.diag([a, a * 1.732, 7.065]),
                     axis_kind=("periodic", "periodic", "periodic"))


def _hydrogen_chain():
    """One hydrogen per cell along a repeating axis: an ODD count per cell
    with no metal at all -- the detection table's last repeating row."""
    return Structure(elements=["H"], positions=np.array([[5.0, 5.0, 0.0]]),
                     cell=np.diag([10.0, 10.0, 0.9]),
                     axis_kind=("isolated", "isolated", "periodic"))


def _bcc_iron():
    a = 2.87
    return Structure(elements=["Fe", "Fe"],
                     positions=np.array([[0, 0, 0], [a / 2, a / 2, a / 2]]),
                     cell=np.diag([a, a, a]),
                     axis_kind=("periodic", "periodic", "periodic"))


def _siesta(root, struct, cfg=None, **kw):
    return _prep(root, struct, cfg or SiestaConfig(system_label="JOB"),
                 default_siesta_stages("publishable"), "siesta", **kw)


def _pyscf(root, struct, cfg=None, **kw):
    return _prep(root, struct, cfg or PySCFConfig(job_name="JOB"),
                 default_pyscf_stages("publishable"), "pyscf", **kw)


def _report(dest, stage) -> str:
    return next(next(dest.glob(f"*_{stage}")).glob("*.validation.txt")
                ).read_text()


def _spin_lines(fdf: str):
    """``(Spin word, Spin.Total or None)`` as the deck writes them."""
    spin = re.search(r"^Spin\s+(\S+)", fdf, re.M)
    total = re.search(r"^Spin\.Total\s+(\S+)", fdf, re.M)
    fixed = re.search(r"^Spin\.Fix\s+\.true\.", fdf, re.M)
    assert bool(total) == bool(fixed), "Spin.Fix and Spin.Total travel together"
    return (spin.group(1) if spin else None,
            float(total.group(1)) if total else None)


# ------------------------------------------- the detection table (§ 2a.1b)

@pytest.mark.parametrize("make, spin, total", [
    (WATER, "non-polarized", None),        # even count, no metal
    (METHYL, "polarized", 1.0),            # odd count in a molecule
    (AU4, "non-polarized", None),          # a metallic cluster, even count
    (AU1, "polarized", 1.0),               # a single atom's own doublet
    (FE_AQUA, "polarized", 2.0),           # an open-d metal's usual count
    (_gold_lead, "non-polarized", None),   # a repeating cell: not a spin
    (_bcc_iron, "polarized", None),        # a magnetic lattice: moment free
    (_hydrogen_chain, "non-polarized", None),  # repeating, no open-d: not a spin
], ids=["water", "methyl", "Au4", "Au1", "Fe-aqua", "gold-lead", "bcc-Fe",
        "H-chain"])
def test_a_blank_spin_is_decided_from_the_structure(
        isolated_projects_root, make, spin, total):
    """Every spin field blank: the SIESTA deck carries the class's answer,
    written explicitly -- ``Spin`` in every state, the pin only for a count
    beside ``polarized`` -- with where it came from."""
    _dest, _stage, fdf = _siesta(isolated_projects_root, make())
    assert _spin_lines(fdf) == (spin, total), fdf
    assert re.search(r"^# Spin: \S+ \(detected: ", fdf, re.M), fdf


def test_the_charge_is_judged_at_its_own_value(isolated_projects_root):
    """Formate at a stated -1 has 24 electrons: a closed shell.  At charge 0
    the count was 23 and the ion was told to go open-shell."""
    dest, stage, fdf = _siesta(isolated_projects_root, FORMATE(),
                               SiestaConfig(system_label="JOB",
                                            net_charge=-1))
    assert _spin_lines(fdf) == ("non-polarized", None), fdf
    assert re.search(r"^NetCharge\s+-1$", fdf, re.M)
    assert "# NetCharge: -1 (stated)." in fdf
    assert "open-shell" not in _report(dest, stage)


def test_a_blank_charge_is_written_with_its_reason(isolated_projects_root):
    """The deck states the charge that applies at 0 too (`template.md` § 6.6:
    nothing reaches the engine by omission), with why."""
    _d, _s, fdf = _siesta(isolated_projects_root, WATER())
    assert re.search(r"^NetCharge\s+\+0$", fdf, re.M), fdf
    assert ("# NetCharge: +0 (detected: no deprotonated phosphate groups)."
            in fdf)


@pytest.mark.parametrize("engine, kind", [("siesta", "optimization"),
                                          ("pyscf", "optimization"),
                                          ("pyscf", "vibration")])
def test_the_charge_step_on_every_deck(isolated_projects_root,
                                       deprotonated_diester, engine, kind):
    """§ 2a.1a's charge step on each deck a charge reaches: a blank takes the
    phosphate rule -- the diester's one deprotonated group, -1 -- and says
    so; a stated value wins, 0 included."""
    def deck(net_charge):
        if engine == "siesta":
            return _siesta(isolated_projects_root, deprotonated_diester,
                           SiestaConfig(system_label="JOB",
                                        net_charge=net_charge))[2]
        vib = kind == "vibration"
        return _prep(isolated_projects_root, deprotonated_diester,
                     PySCFConfig(job_name="JOB", net_charge=net_charge,
                                 already_relaxed=vib),
                     (vibration_stages("pyscf", already_relaxed=True) if vib
                      else default_pyscf_stages("publishable")),
                     "pyscf", calculation=kind)[2]
    blank = deck(None)
    why = "detected: 1 deprotonated phosphate group"
    if kind == "vibration":
        # A stated 0 still wins -- and makes the diester a radical, which a
        # PySCF vibration does not offer (`engines/template.md` § 6.3a):
        # refused, saying the treatment was detected from the structure.
        _d, _s, said = _prep(isolated_projects_root, deprotonated_diester,
                             PySCFConfig(job_name="JOB", net_charge=0,
                                         already_relaxed=True),
                             vibration_stages("pyscf", already_relaxed=True),
                             "pyscf", calculation="vibration", refused=True)
        assert "spin_treatment = unrestricted (detected" in said, said
        assert "charge     = -1," in blank and why in blank
        return
    zero = deck(0)
    if engine == "siesta":
        assert re.search(r"^NetCharge\s+-1$", blank, re.M), blank
        assert f"# NetCharge: -1 ({why})." in blank
        assert re.search(r"^NetCharge\s+\+0$", zero, re.M), zero
        assert "# NetCharge: +0 (stated)." in zero
    else:
        assert "charge     = -1," in blank and why in blank
        assert "charge     = 0," in zero


def test_a_charged_molecule_is_corrected_and_a_charged_cell_is_not(
        isolated_projects_root):
    """§ 2b, through prep.  A charged MOLECULE's deck says SIESTA applies the
    monopole correction itself in a cubic box (`siesta: Emadel`), and prep
    writes the script that reads it first.  A charged REPEATING cell gets no
    formula and no script -- the point-charge correction is the wrong one
    there -- and the report says its energy needs a defect-specific
    treatment.  A neutral deck gets neither.  (Both charged kinds got the
    molecule's note and script until the M6 review.)"""
    def script(dest, stage):
        return next(dest.glob(f"*_{stage}")) / "makov_payne_correction.py"

    def calc(name):
        # Each its own calculation: a stage folder another prep wrote into
        # would still hold that prep's files.
        (isolated_projects_root / name).mkdir()
        return isolated_projects_root / name

    dest, stage, fdf = _siesta(calc("molecule"), FORMATE(),
                               SiestaConfig(system_label="JOB",
                                            net_charge=-1))
    assert "siesta: Emadel" in fdf and script(dest, stage).is_file()

    dest, stage, fdf = _siesta(calc("cell"), _gold_lead(),
                               SiestaConfig(system_label="JOB",
                                            net_charge=-1))
    assert "CHARGED REPEATING CELL" in fdf and "Emadel" not in fdf
    assert not script(dest, stage).exists()
    assert "defect-specific treatment" in _report(dest, stage)

    dest, stage, fdf = _siesta(calc("neutral"), WATER())
    assert "Makov" not in fdf and not script(dest, stage).exists()


def _cluster(el, n):
    """``n`` atoms of ``el`` on a line, 2.6 Å apart -- composition is what
    the detection reads; the geometry only has to be a molecule."""
    return _mol([el] * n, [[2.6 * i, 0, 0] for i in range(n)])


@pytest.mark.parametrize("make, cls, spin", [
    (lambda: _cluster("Au", 2), "dft.RKS", 0),     # below four: parity, even
    (lambda: _cluster("Au", 3), "dft.UKS", 1),     # below four: parity, odd
    (lambda: _cluster("Au", 5), "dft.UKS", 1),     # a cluster, but odd
    (lambda: _cluster("Cu", 4), "dft.RKS", 0),     # the other noble metals too
    (lambda: _cluster("Pd", 2), "dft.RKS", 0),     # closed d10: parity
    (lambda: _mol(["Au", "Au", "Au", "Au", "Fe"],
                  [[2.6 * i, 0, 0] for i in range(5)]), "dft.UKS", 2),
    (lambda: _mol(["Mn", "O"], [[0, 0, 0], [1.9, 0, 0]]), "dft.UKS", 5),
    (lambda: _mol(["Ru", "O"], [[0, 0, 0], [1.9, 0, 0]]), "dft.UKS", 2),
    (lambda: _mol(["Fe", "H"], [[0, 0, 0], [1.6, 0, 0]]), "dft.UKS", 1),
    (_hydrogen_chain, "dft.UKS", 1),
], ids=["Au2", "Au3", "Au5", "Cu4", "Pd2", "Au4+Fe", "Mn", "Ru-usual-2",
        "Fe-parity-moved", "H-chain-computed-as-a-molecule"])
def test_the_detection_table_on_pyscf(isolated_projects_root, make, cls,
                                      spin):
    """The rest of § 2a.1b's rows, where PySCF writes the answer: a noble
    cluster below four atoms and a closed-d10 pair go by parity; an open-d
    metal decides even beside gold; Mn's usual count is 5, an unlisted
    open-d metal's 2; and a usual count of the wrong parity for this
    electron count (Fe-H has 27) moves by one.  And a structure that
    repeats is computed as a molecule (`cell.engine_axis_kinds`,
    `model/structure-periodicity.md` § 2.1): the hydrogen chain's odd count
    is a doublet here, where SIESTA's repeating cell is not a spin -- judged
    by the structure's axes, PySCF was handed a restricted single electron
    (the M6 review; the door since K8)."""
    _d, _s, py = _pyscf(isolated_projects_root, make())
    assert re.search(rf"^mf = {re.escape(cls)}\(mol\)$", py, re.M), py
    assert re.search(rf"^\s+spin\s+= {spin},$", py, re.M), py


# ----------------------------------------------------- PySCF's class (§ 2a.1)

@pytest.mark.parametrize("make, cfg, cls, spin", [
    (WATER, {}, "dft.RKS", 0),
    (METHYL, {}, "dft.UKS", 1),
    (METHYL, {"method": "HF"}, "scf.UHF", 1),
    (METHYL, {"method": "HF", "spin_treatment": "restricted-open"},
     "scf.ROHF", 1),
    (O2, {"unpaired_electrons": 2}, "dft.UKS", 2),
], ids=["water", "methyl", "methyl-HF", "methyl-ROHF", "triplet-O2"])
def test_pyscf_writes_the_class_it_composed(isolated_projects_root, make, cfg,
                                            cls, spin):
    """The class is composed from the method and the treatment and written
    out -- never left for PySCF to re-rule -- and ``gto.M`` pins the count."""
    _d, _s, py = _pyscf(isolated_projects_root, make(),
                        PySCFConfig(job_name="JOB", **cfg))
    assert re.search(rf"^mf = {re.escape(cls)}\(mol\)$", py, re.M), py
    assert re.search(rf"^\s+spin\s+= {spin},$", py, re.M), py


def test_a_stated_triplet_is_not_second_guessed(isolated_projects_root):
    """O2 with 2S = 2 stated: an even count the structure would call closed.
    Parity cannot see a triplet, and the person is the authority -- no
    finding (ES9)."""
    dest, stage, _py = _pyscf(isolated_projects_root, O2(),
                              PySCFConfig(job_name="JOB",
                                          unpaired_electrons=2))
    report = _report(dest, stage)
    assert "spin_treatment" not in report and "unpaired" not in report, report


# ------------------------------------------------ the findings (ES3 - ES9)

@pytest.mark.parametrize("cfg, words", [
    ({"unpaired_electrons": "free"}, "fixes the moment"),             # ES6
    ({"spin_treatment": "restricted", "unpaired_electrons": 2},
     "restricted-open"),                                              # ES5
    ({"spin_treatment": "non-collinear"}, "cannot run here"),         # ES4
], ids=["free-on-PySCF", "restricted-with-a-count", "non-collinear"])
def test_what_pyscf_cannot_run_is_refused_by_name(isolated_projects_root,
                                                  cfg, words):
    _d, _s, said = _pyscf(isolated_projects_root, METHYL(),
                          PySCFConfig(job_name="JOB", **cfg), refused=True)
    assert words in said, said


def test_a_restricted_open_vibration_is_refused(isolated_projects_root):
    """PySCF has no analytic ROHF/ROKS Hessian, and the vibration deck takes
    the analytic Hessian (§ 2a.3)."""
    _d, _s, said = _prep(
        isolated_projects_root, METHYL(),
        PySCFConfig(job_name="JOB", already_relaxed=True,
                    spin_treatment="restricted-open"),
        vibration_stages("pyscf", already_relaxed=True), "pyscf",
        calculation="vibration", refused=True)
    assert "no analytic ROHF/ROKS Hessian" in said, said


def test_a_closed_shell_stated_on_an_open_d_metal_is_warned(
        isolated_projects_root):
    """The hemeC guard, now at the resolved charge: restricted on an Fe
    complex is reported once, naming what the structure implies."""
    dest, stage, _fdf = _siesta(isolated_projects_root, FE_AQUA(),
                                SiestaConfig(system_label="JOB",
                                             spin_treatment="restricted"))
    report = _report(dest, stage)
    assert "implies unrestricted, 2S = 2" in report, report


def test_a_constrained_singlet_on_a_closed_shell_is_written_and_warned(
        isolated_projects_root):
    """ES9's second finding: unrestricted at 2S = 0 on a closed-shell
    structure runs -- it is the broken-symmetry singlet's setting -- but costs
    twice restricted for the same answer otherwise.  The deck pins it and
    says so beside it; the report warns once."""
    dest, stage, fdf = _siesta(isolated_projects_root, WATER(),
                               SiestaConfig(system_label="JOB",
                                            spin_treatment="unrestricted",
                                            unpaired_electrons=0))
    assert _spin_lines(fdf) == ("polarized", 0.0)
    assert "constrained singlet in open-shell DFT" in fdf
    report = _report(dest, stage)
    assert report.count("constrained singlet") == 1, report


def test_a_half_stated_spin_is_completed_in_the_one_order(
        isolated_projects_root):
    """§ 2a.1a with half the state stated: ``unrestricted`` on closed-shell
    water, the count blank.  The structure's own count, 0, is pinned -- a
    constrained singlet -- and said to be the structure's, not the person's."""
    _d, _s, fdf = _siesta(isolated_projects_root, WATER(),
                          SiestaConfig(system_label="JOB",
                                       spin_treatment="unrestricted"))
    assert _spin_lines(fdf) == ("polarized", 0.0), fdf
    assert "(2S): 0 (detected: the structure has no unpaired electron" in fdf


def test_what_siesta_cannot_run_is_refused_and_non_collinear_floats(
        isolated_projects_root):
    """ES4 / ES6 on SIESTA: non-collinear with the count blank floats -- no
    ``Spin.Fix``, which SIESTA stops on there -- and with a count it is
    refused by name; restricted-open is a formalism SIESTA does not have."""
    _d, _s, fdf = _siesta(isolated_projects_root, WATER(),
                          SiestaConfig(system_label="JOB",
                                       spin_treatment="non-collinear"))
    assert _spin_lines(fdf) == ("non-colinear", None), fdf
    for cfg, words in (
            (SiestaConfig(system_label="JOB", spin_treatment="non-collinear",
                          unpaired_electrons=2), "cannot be held here"),
            (SiestaConfig(system_label="JOB",
                          spin_treatment="restricted-open"),
             "no restricted open-shell formalism")):
        _d, _s, said = _siesta(isolated_projects_root, WATER(), cfg,
                               refused=True)
        assert words in said, said


def test_a_count_a_metal_decided_warns_until_it_is_stated(
        isolated_projects_root):
    """ES8: the usual count is a guess about the coordination.  Blank, it
    warns; stated, the warning goes."""
    dest, stage, _f = _siesta(isolated_projects_root, FE_AQUA())
    assert "was left blank, so it is 2" in _report(dest, stage)
    dest, stage, _f = _siesta(isolated_projects_root, FE_AQUA(),
                              SiestaConfig(system_label="JOB",
                                           unpaired_electrons=4))
    report = _report(dest, stage)
    assert "left blank" not in report, report
    assert "high-spin" in report, "the stated count is echoed as information"


def test_an_odd_count_under_restricted_runs_half_filled_on_siesta(
        isolated_projects_root):
    """SIESTA runs a restricted radical with its top level half filled -- a
    warning there; PySCF refuses the pair outright (ES3)."""
    dest, stage, _f = _siesta(isolated_projects_root, METHYL(),
                              SiestaConfig(system_label="JOB",
                                           spin_treatment="restricted"))
    report = _report(dest, stage)
    assert "half" in report
    # ONE finding per fact (ES9): the parity finding holds first, so the
    # stated-restricted-on-a-radical finding does not say it again.
    assert "implies unrestricted" not in report, report
    _d, _s, said = _pyscf(isolated_projects_root, METHYL(),
                          PySCFConfig(job_name="JOB",
                                      spin_treatment="restricted"),
                          refused=True)
    assert "the electron count and the spin disagree" in said.lower(), said


# --------------------------------------------- it belongs to the calculation

def test_a_stage_cannot_change_the_state(isolated_projects_root):
    """ES1: an override of a state item on one rung is refused by name --
    every rung's warm files are a density for one electronic state."""
    def override(dest):
        # Promoted (`varies`) as a stage override must be (`stages.md`
        # § 6.2), so the refusal met is the state's own.
        task = json.loads((dest / "task.json").read_text())
        task["varies"] = sorted(set(task.get("varies") or [])
                                | {"spin_treatment"})
        task["stages"][0].setdefault("overrides", {})["spin_treatment"] = \
            "unrestricted"
        (dest / "task.json").write_text(json.dumps(task))
    _d, _s, said = _siesta(isolated_projects_root, WATER(),
                           before_prep=override, refused=True)
    assert "shared by every stage of this calculation" in said, said
    assert "ES1" in said


# ----------------------------------------------- what the structure carries

def test_a_structure_carries_the_state_of_the_run_it_came_from(
        isolated_projects_root):
    """ES7: a structure exported from a finished run carries that run's
    record, and a blank item takes the recorded value -- a vibration of a
    charged relaxation starts charged."""
    s = FORMATE()
    s.info["calculation"] = {"engine": "siesta", "source": "relax.fdf",
                             "contract": {"net_charge": -1,
                                          "spin_treatment": "restricted",
                                          "unpaired_electrons": 0}}
    _d, _s, fdf = _siesta(isolated_projects_root, s)
    assert re.search(r"^NetCharge\s+-1$", fdf, re.M), fdf
    assert ("recorded: recorded by the run this structure came out of "
            "(relax.fdf)") in fdf
    # EDITED SINCE (a geometry or cell op): no longer the structure that run
    # came out of, so its record is not assumed (`web/molview.md` § 8.4).
    s.info["calculation"]["structure_modified"] = True
    _d, _s, fdf = _siesta(isolated_projects_root, s)
    assert re.search(r"^NetCharge\s+\+0$", fdf, re.M), fdf
    assert "recorded by the run" not in fdf
