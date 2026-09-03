"""Regression tests pinning down each P1 / P2 fix from the static reviews.

Each test maps to a specific finding:

    S1   peptide.py element field stripped of leading whitespace
    S2   SiestaConfig.net_charge / PySCFConfig.charge override the
         phosphate-only auto-detection heuristic
    S6   web app caps multipart upload size at 10 MB
    S7   Structure.to_pdb caps atom serial / residue id at PDB column
         widths
    T3   protonate_phosphate_oxygens is a no-op on a peptide
    T5   web layer's _xyz_to_structure delegates to Structure.from_xyz
    D1   SiestaConfig and Config alias coexist
    D3   charged-system cell padding auto-bumps to 25 A
"""

from __future__ import annotations

import numpy as np
import pytest

from molbuilder.structure import Structure
from molbuilder.chemistry import (
    formal_charge_from_phosphates,
    protonate_phosphate_oxygens,
)


# --------------------------------------------------------------------- #
#  S1 / T1 -- element field is stripped of leading/trailing whitespace  #
# --------------------------------------------------------------------- #


def test_s1_t1_element_strip_propagates_to_species():
    """If a Structure ever lands in the FDF generator with an element
    like ' C', species detection must NOT see two distinct species."""
    from molbuilder.siesta import _detect_species
    species = _detect_species(["C", "C", "H", "O"])
    import ase.data
    assert species == sorted(species, key=lambda s: ase.data.atomic_numbers[s])
    # If a downstream caller forgets to strip, the FDF generator should
    # crash visibly rather than silently produce a malformed input.
    try:
        sp = _detect_species(["C", " C", "H"])
    except KeyError:
        return   # acceptable: the unknown ' C' raises
    counts = {s: sp.count(s) for s in set(sp)}
    assert all(c == 1 for c in counts.values())


# --------------------------------------------------------------------- #
#  S2 -- net_charge override on SiestaConfig + PySCFConfig              #
# --------------------------------------------------------------------- #


def test_s2_siesta_net_charge_override():
    """A peptide with charged side-chains needs an explicit override."""
    from molbuilder.siesta import SiestaConfig, render_fdf
    s = Structure(
        elements=["C", "N", "O"],
        positions=np.array([[0,0,0],[1.5,0,0],[0,1.5,0]]), vacuum=(12.0, 12.0, 12.0))
    text_default = render_fdf(s, SiestaConfig(verbose_comments=False))
    assert "NetCharge" not in text_default
    text_charged = render_fdf(s, SiestaConfig(net_charge=-1, verbose_comments=False))
    assert "NetCharge       -1" in text_charged
    text_neutral = render_fdf(s, SiestaConfig(net_charge=0, verbose_comments=False))
    assert "NetCharge" not in text_neutral


def test_s2_siesta_net_charge_overrides_auto_detect(deprotonated_diester):
    """User-specified charge wins over the phosphate heuristic."""
    from molbuilder.siesta import SiestaConfig, render_fdf
    s = deprotonated_diester
    assert formal_charge_from_phosphates(s) == -1
    auto = render_fdf(s, SiestaConfig(verbose_comments=False))
    assert "NetCharge       -1" in auto
    forced0 = render_fdf(s, SiestaConfig(net_charge=0, verbose_comments=False))
    assert "NetCharge" not in forced0
    forced3 = render_fdf(s, SiestaConfig(net_charge=-3, verbose_comments=False))
    assert "NetCharge       -3" in forced3


def test_s2_pyscf_charge_override():
    from molbuilder.pyscf import PySCFConfig, render_script
    s = Structure(
        elements=["O", "H"],
        positions=np.array([[0,0,0],[0.957,0,0]]), vacuum=(12.0, 12.0, 12.0))
    text = render_script(s, PySCFConfig(net_charge=-1, verbose_comments=False))
    assert "charge     = -1," in text


# --------------------------------------------------------------------- #
#  D3 -- vacuum comes with the STRUCTURE (cell_padding removed 2026-07)  #
#                                                                       #
#  render_fdf derives the vacuum box from struct.resolve_cell() (bbox + #
#  2*vacuum, centred).  cfg.cell_padding is gone; a charged system with #
#  thin vacuum is WARNED (never auto-bumped) -- geometry is the user's.  #
# --------------------------------------------------------------------- #


def test_d3_charged_system_warns_on_thin_vacuum(deprotonated_diester):
    """A charged molecule with 10 A/side vacuum is below the >= 25 A charged
    recommendation (image-image Coulomb decays only as 1/L), so it must be
    reported -- and the geometry must be left alone.

    The DELIVERY changed on 2026-07-29: this used to be a Python
    ``warnings.warn`` inside ``render_fdf``, which reached the server's stderr
    and therefore no web user at all.  It is now an ``Issue`` from the SIESTA
    validator (``cell.vacuum_thin``), so the same advice reaches the browser
    panel and the CLI report alike -- clause R5 of the delivery contract,
    docs/science/validation.md 4.1.  The assertion follows the finding to its
    new channel; the invariant (report, never mutate) is unchanged.
    """
    import dataclasses
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.validation import validate
    s = dataclasses.replace(deprotonated_diester, vacuum=(10.0, 10.0, 10.0))
    findings = [i for i in validate(s, SiestaConfig(verbose_comments=False))
                if i.where == "cell.vacuum_thin"]
    assert findings, "no cell.vacuum_thin finding for a charged 10 A/side box"
    msg = findings[0].message
    assert "25" in msg and "charged" in msg, msg
    assert "periodic images" in msg, msg
    assert findings[0].severity == "warn"
    # Geometry untouched: the vacuum the user set is still the vacuum stored.
    assert s.vacuum == (10.0, 10.0, 10.0)


def test_d3_neutral_uses_the_structures_vacuum(water_structure):
    import dataclasses
    import warnings
    from molbuilder.siesta import SiestaConfig, render_fdf
    # The FDF cell note reports the STRUCTURE's vacuum (per side); a sufficient
    # neutral vacuum (10 >= 8) raises no thin-vacuum warning.
    s = dataclasses.replace(water_structure, vacuum=(10.0, 10.0, 10.0))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        text = render_fdf(s, SiestaConfig(verbose_comments=False))
    assert "vacuum = (10.0, 10.0, 10.0) A/side" in text, text
    assert not any("periodic images" in str(x.message) for x in w)


# --------------------------------------------------------------------- #
#  S7 -- PDB serial / residue id wrapped at column widths               #
# --------------------------------------------------------------------- #


def test_s7_pdb_serial_caps_at_99999():
    n = 100001
    s = Structure(elements=["C"] * n, positions=np.zeros((n, 3)), vacuum=(12.0, 12.0, 12.0))
    pdb = s.to_pdb()
    lines = [ln for ln in pdb.splitlines() if ln.startswith("ATOM")]
    assert len(lines) == n
    assert lines[0][6:11]      == "    1"
    assert lines[99998][6:11]  == "99999"
    assert lines[99999][6:11]  == "*****"
    assert lines[-1][6:11]     == "*****"


def test_s7_pdb_residue_id_caps_at_9999():
    n = 12000
    s = Structure(
        elements=["C"] * n,
        positions=np.zeros((n, 3)),
        residue_ids=list(range(1, n + 1)), vacuum=(12.0, 12.0, 12.0))
    pdb = s.to_pdb()
    lines = [ln for ln in pdb.splitlines() if ln.startswith("ATOM")]
    assert lines[0][22:26]    == "   1"
    assert lines[9998][22:26] == "9999"
    assert lines[9999][22:26] == "****"


# --------------------------------------------------------------------- #
#  T3 -- protonate is a no-op on phosphate-free input                   #
# --------------------------------------------------------------------- #


def test_t3_protonate_noop_on_peptide():
    pytest.importorskip("PeptideBuilder")
    import molbuilder
    s = molbuilder.build_peptide("AC", add_hydrogens=False)
    assert "P" not in s.elements
    assert formal_charge_from_phosphates(s) == 0
    s2, n_added = protonate_phosphate_oxygens(s)
    assert n_added == 0
    assert s2 is s   # same instance when no work to do


# --------------------------------------------------------------------- #
#  S6 -- web app rejects oversized uploads                              #
# --------------------------------------------------------------------- #


def test_s6_web_app_caps_upload_size(web_client):
    """The unified Flask app caps uploads at 50 MB.

    Pre-merge the build app capped at 10 MB and the watch app at 50 MB.
    Flask's MAX_CONTENT_LENGTH is a single global setting, so the merged
    app uses the larger of the two so /api/watch/load can accept
    realistic SIESTA / PySCF logs.  The /api/build/load endpoint still
    rejects oversize uploads -- just at the 50 MB threshold now.
    """
    app_cfg = web_client.application.config
    assert app_cfg.get("MAX_CONTENT_LENGTH") == 50 * 1024 * 1024
    big = "x" * (51 * 1024 * 1024)   # 51 MB > 50 MB cap
    r = web_client.post("/api/build/load",
                        json={"text": big, "filename": "big.xyz"})
    assert r.status_code == 413


# --------------------------------------------------------------------- #
#  T5 -- web app's _xyz_to_structure uses canonical parser              #
# --------------------------------------------------------------------- #


def test_t5_web_uses_canonical_xyz_parser():
    pytest.importorskip("flask")
    from molbuilder.web.blueprints.build import _xyz_to_structure
    text = (
        "2\n"
        "h2\n"
        "H 0 0 0\n"
        "H 0.74 0 0\n"
    )
    s = _xyz_to_structure(text)
    assert s.n_atoms == 2
    s_canonical = Structure.from_xyz(text)
    np.testing.assert_array_equal(s.positions, s_canonical.positions)
    assert s.elements == s_canonical.elements


# --------------------------------------------------------------------- #
#  D1 -- RETIRED 2026-09-02: `Config` was a backward-compat alias        #
#                                                                       #
#  `Config = SiestaConfig` existed so that code importing the old name   #
#  kept working.  The project's rule is that a rename DELETES the old    #
#  name everywhere rather than leaving a second spelling behind, and     #
#  this test was what held the second spelling in place: the alias had   #
#  no caller but this, and two docstring examples that taught it.        #
#  Alias, both `__all__` entries, the examples and this test are gone    #
#  together -- which is the only way a shim is actually removed.         #
# --------------------------------------------------------------------- #
