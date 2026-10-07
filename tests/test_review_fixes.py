"""Regression tests pinning down each P1 / P2 fix from the static reviews.

Each test maps to a specific finding:

    S6   web app caps multipart upload size
    S7   Structure.to_pdb caps atom serial / residue id at PDB column
         widths
    T3   protonate_phosphate_oxygens is a no-op on a peptide
    D3   a charged system's thin vacuum is warned, never bumped
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
#  D3 -- vacuum comes with the STRUCTURE                                #
#                                                                       #
#  The deck's box is the structure's (bbox + 2*vacuum, centred); a      #
#  charged system with thin vacuum is WARNED (never auto-bumped) --     #
#  geometry is the user's.                                              #
# --------------------------------------------------------------------- #


def test_d3_charged_system_warns_on_thin_vacuum(deprotonated_diester):
    """A charged molecule with 10 A/side vacuum is below the >= 25 A charged
    recommendation (image-image Coulomb decays only as 1/L), so it must be
    reported -- and the geometry must be left alone.

    It is an ``Issue`` from the SIESTA validator (``cell.vacuum_thin``), so the
    same advice reaches the browser panel and the CLI report alike -- clause
    R5 of the delivery contract, docs/science/validation.md 4.1: report,
    never mutate.
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
    from molbuilder.peptide import build_peptide
    s = build_peptide("AC", add_hydrogens=False)
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

    Flask's MAX_CONTENT_LENGTH is a single global setting, sized so
    realistic SIESTA / PySCF logs can be uploaded; /api/build/load rejects
    oversize uploads at that threshold.
    """
    app_cfg = web_client.application.config
    assert app_cfg.get("MAX_CONTENT_LENGTH") == 50 * 1024 * 1024
    big = "x" * (51 * 1024 * 1024)   # 51 MB > 50 MB cap
    r = web_client.post("/api/build/load",
                        json={"text": big, "filename": "big.xyz"})
    assert r.status_code == 413
