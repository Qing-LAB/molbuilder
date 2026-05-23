"""Tests for the PSML pseudopotential header parser + coverage check.

The 2026-05-23 SIESTA-help-text pass surfaced that the user is on
their own when picking pseudos from PseudoDojo (NC vs PAW, SR vs FR,
which XC family).  This module gives molbuilder enough metadata
awareness to validate a downloaded pseudo directory at preflight
and call out mismatches BEFORE the user discovers them in a wrong-
bond-length SIESTA run.

Tests:
  * Synthetic-PSML round-trip: build a minimal valid PSML, parse it,
    assert the canonical metadata.
  * scan_psml_directory: drop a few synthetic files, scan, get the
    right per-element mapping.
  * check_coverage: missing element, XC mismatch, relativistic
    mismatch.
  * Tolerance: malformed XML doesn't crash the parser.
"""
from __future__ import annotations

import pytest
from pathlib import Path


# Minimal PSML body covering the fields the parser reads.  Real
# PseudoDojo files have 100+ KB of grid / orbital data we don't need
# to fake here; the parser only touches the <header> + first
# <libxc-info> + <provenance>.
def _make_psml(element: str, *,
                z: int = None,
                libxc_id: int = 101,   # GGA_X_PBE
                rel: str = "scalar",
                creator: str = "ONCVPSP-test") -> str:
    if z is None:
        from ase.data import atomic_numbers as _Z
        z = _Z[element]
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<psml version="1.1" xmlns="http://launchpad.net/psml">
  <header atomic-label="{element}" atomic-number="{z}" z-pseudo="{z}"
          relativity="{rel}"/>
  <exchange-correlation>
    <libxc-info id="{libxc_id}"/>
  </exchange-correlation>
  <provenance creator="{creator}"/>
  <valence-configuration>
    <shell n="2" l="s" occupation="2.0"/>
    <shell n="2" l="p" occupation="4.0"/>
  </valence-configuration>
</psml>"""


class TestParsePsmlHeader:
    def test_basic_round_trip(self, tmp_path):
        p = tmp_path / "C.psml"
        p.write_text(_make_psml("C"))
        from molbuilder.pseudos import parse_psml_header
        info = parse_psml_header(p)
        assert info.element       == "C"
        assert info.atomic_number == 6
        assert info.xc_family     == "GGA"
        assert info.xc_authors    == "PBE"
        assert info.relativistic  == "scalar"
        assert info.generator     == "ONCVPSP-test"
        assert info.path          == p
        assert info.parse_warnings == []

    def test_pbesol_libxc(self, tmp_path):
        """libxc id 116 = XC_GGA_X_PBE_SOL."""
        p = tmp_path / "Fe.psml"
        p.write_text(_make_psml("Fe", libxc_id=116))
        from molbuilder.pseudos import parse_psml_header
        info = parse_psml_header(p)
        assert info.xc_family  == "GGA"
        assert info.xc_authors == "PBEsol"

    def test_fully_relativistic_normalised(self, tmp_path):
        """``relativity=dirac`` and similar variants should normalise
        to ``"spin-orbit"`` so downstream code has one canonical
        value."""
        p = tmp_path / "Pt.psml"
        p.write_text(_make_psml("Pt", rel="dirac"))
        from molbuilder.pseudos import parse_psml_header
        assert parse_psml_header(p).relativistic == "spin-orbit"

    def test_unknown_libxc_id_falls_back_to_unknown(self, tmp_path):
        """A libxc id we don't recognise (rare functionals) must NOT
        misclassify; mark as unknown so the user gets a clear warning
        rather than a wrong-family false-positive."""
        p = tmp_path / "X.psml"
        p.write_text(_make_psml("C", libxc_id=99999))
        from molbuilder.pseudos import parse_psml_header
        info = parse_psml_header(p)
        assert info.xc_family  == "unknown"
        assert info.xc_authors == "unknown"

    def test_malformed_xml_returns_parse_warning(self, tmp_path):
        """Garbage in the file -> PsmlInfo with empty element and a
        parse warning, NOT an exception.  scan_psml_directory relies
        on this to silently skip bad files."""
        p = tmp_path / "broken.psml"
        p.write_text("<psml not closed properly")
        from molbuilder.pseudos import parse_psml_header
        info = parse_psml_header(p)
        assert info.element == ""
        assert info.parse_warnings
        assert "parse" in info.parse_warnings[0].lower()


class TestScanPsmlDirectory:
    def test_one_file_per_element(self, tmp_path):
        for el in ("C", "H", "N", "Fe"):
            (tmp_path / f"{el}.psml").write_text(_make_psml(el))
        from molbuilder.pseudos import scan_psml_directory
        m = scan_psml_directory(tmp_path)
        assert sorted(m.keys()) == ["C", "Fe", "H", "N"]
        assert m["Fe"].atomic_number == 26

    def test_non_psml_files_ignored(self, tmp_path):
        (tmp_path / "C.psml").write_text(_make_psml("C"))
        (tmp_path / "README.txt").write_text("ignore me")
        (tmp_path / "Fe.psf").write_text("legacy format, not psml")
        from molbuilder.pseudos import scan_psml_directory
        assert sorted(scan_psml_directory(tmp_path).keys()) == ["C"]

    def test_missing_directory_returns_empty(self):
        from molbuilder.pseudos import scan_psml_directory
        assert scan_psml_directory(Path("/does/not/exist")) == {}

    def test_first_file_wins_on_duplicate_element(self, tmp_path):
        """Two files claiming the same element: the first one
        encountered (alphabetical order) wins.  Documented behaviour."""
        (tmp_path / "A_Fe.psml").write_text(_make_psml("Fe", libxc_id=101))   # PBE
        (tmp_path / "B_Fe.psml").write_text(_make_psml("Fe", libxc_id=116))   # PBEsol
        from molbuilder.pseudos import scan_psml_directory
        m = scan_psml_directory(tmp_path)
        assert m["Fe"].xc_authors == "PBE"   # A_Fe sorted first


class TestCheckCoverage:
    def test_all_present_with_matching_xc(self, tmp_path):
        for el in ("C", "H", "N", "Fe"):
            (tmp_path / f"{el}.psml").write_text(_make_psml(el))
        from molbuilder.pseudos import check_coverage
        entries = check_coverage(
            ("C", "H", "N", "Fe"), tmp_path,
            expected_xc_family="GGA", expected_xc_authors="PBE",
        )
        assert len(entries) == 4
        assert all(e.status == "ok" for e in entries)

    def test_missing_element_flagged(self, tmp_path):
        (tmp_path / "C.psml").write_text(_make_psml("C"))
        from molbuilder.pseudos import check_coverage
        entries = check_coverage(("C", "H", "Fe"), tmp_path)
        statuses = {e.element: e.status for e in entries}
        assert statuses["C"]  == "ok"
        assert statuses["H"]  == "missing"
        assert statuses["Fe"] == "missing"
        # Missing message must mention the source recommendation.
        h_msg = next(e.message for e in entries if e.element == "H")
        assert "pseudo-dojo" in h_msg.lower()

    def test_xc_family_mismatch(self, tmp_path):
        """LDA pseudo on a GGA calc -- silently-wrong bond lengths."""
        (tmp_path / "C.psml").write_text(_make_psml("C", libxc_id=1))  # LDA
        from molbuilder.pseudos import check_coverage
        entries = check_coverage(
            ("C",), tmp_path,
            expected_xc_family="GGA", expected_xc_authors="PBE",
        )
        assert entries[0].status == "xc_mismatch"
        assert "LDA" in entries[0].message
        assert "GGA" in entries[0].message

    def test_xc_authors_mismatch_within_family(self, tmp_path):
        """PBE pseudo + PBEsol calc -- same family, minor mismatch."""
        (tmp_path / "C.psml").write_text(_make_psml("C", libxc_id=101))  # PBE
        from molbuilder.pseudos import check_coverage
        entries = check_coverage(
            ("C",), tmp_path,
            expected_xc_family="GGA", expected_xc_authors="PBEsol",
        )
        assert entries[0].status == "xc_mismatch"
        assert "PBE" in entries[0].message and "PBEsol" in entries[0].message

    def test_relativistic_mismatch(self, tmp_path):
        """Scalar pseudo + spin-orbit calc -- WARN."""
        (tmp_path / "Pt.psml").write_text(_make_psml("Pt", rel="scalar"))
        from molbuilder.pseudos import check_coverage
        entries = check_coverage(
            ("Pt",), tmp_path, expected_relativistic="spin-orbit",
        )
        assert entries[0].status == "relativistic_mismatch"

    def test_duplicate_elements_in_structure_dedup(self, tmp_path):
        """A structure with many C atoms only counts as ONE coverage
        entry for C."""
        (tmp_path / "C.psml").write_text(_make_psml("C"))
        (tmp_path / "H.psml").write_text(_make_psml("H"))
        from molbuilder.pseudos import check_coverage
        # Pretend the structure has 6 C + 6 H.
        entries = check_coverage(["C"]*6 + ["H"]*6, tmp_path)
        assert len(entries) == 2
        assert {e.element for e in entries} == {"C", "H"}


class TestPseudosEndpoint:
    def test_check_pseudos_happy(self, tmp_path):
        """/api/siesta/check-pseudos returns per-element status."""
        # Write H + O pseudos for water.
        (tmp_path / "H.psml").write_text(_make_psml("H"))
        (tmp_path / "O.psml").write_text(_make_psml("O"))
        from molbuilder.web.app import create_app
        c = create_app(config={}).test_client()
        water = "3\nwater\nO 0 0 0\nH 1 0 0\nH -1 0 0\n"
        r = c.post("/api/siesta/check-pseudos", json={
            "psml_lib":       str(tmp_path),
            "structure_text": water,
            "xc_authors":     "PBE",
        })
        assert r.status_code == 200, r.data
        body = r.get_json()
        assert body["ok"] is True
        assert body["n_ok"]       == 2
        assert body["n_missing"]  == 0
        assert body["n_mismatch"] == 0

    def test_check_pseudos_flags_missing_element(self, tmp_path):
        """Water needs H + O; if only H is in the lib, O is missing."""
        (tmp_path / "H.psml").write_text(_make_psml("H"))
        from molbuilder.web.app import create_app
        c = create_app(config={}).test_client()
        water = "3\nwater\nO 0 0 0\nH 1 0 0\nH -1 0 0\n"
        r = c.post("/api/siesta/check-pseudos", json={
            "psml_lib":       str(tmp_path),
            "structure_text": water,
        })
        body = r.get_json()
        assert body["n_missing"] == 1
        missing = next(e for e in body["entries"] if e["status"] == "missing")
        assert missing["element"] == "O"

    def test_structure_analyze_organic_no_metals(self):
        """No metals -> closed-shell singlet (or doublet for odd-e)."""
        from molbuilder.web.app import create_app
        c = create_app(config={}).test_client()
        water = "3\nwater\nO 0 0 0\nH 1 0 0\nH -1 0 0\n"
        r = c.post("/api/structure/analyze",
                   json={"structure_text": water})
        body = r.get_json()
        assert body["ok"] is True
        assert body["metals"] == []
        sug = body["suggested"]["pyscf"]
        assert sug["charge"] == 0
        assert sug["spin"]   == 0
        assert sug["method"] == "RKS"

    def test_structure_analyze_fe_porphyrin_suggests_uks_spin_2(self):
        """A structure with Fe should suggest spin=2 (intermediate-
        spin Fe(II), FeTPP-style) + UKS, with rationale text mentioning
        the experimental-verification caveat."""
        from molbuilder.web.app import create_app
        c = create_app(config={}).test_client()
        # 5 atoms: Fe with 4 N around it.  Total electrons = 26 + 4*7 = 54 (even).
        # spin=2 (even) is parity-compatible.
        xyz = "5\nFeN4\nFe 0 0 0\nN 2 0 0\nN -2 0 0\nN 0 2 0\nN 0 -2 0\n"
        r = c.post("/api/structure/analyze",
                   json={"structure_text": xyz})
        body = r.get_json()
        assert body["metals"] == ["Fe"]
        sug = body["suggested"]["pyscf"]
        assert sug["spin"]   == 2
        assert sug["method"] == "UKS"
        assert "Fe" in sug["rationale"]
        assert "UKS" in sug["rationale"]
        # SIESTA equivalent: spin_polarized=True + spin_total=2.0
        ssug = body["suggested"]["siesta"]
        assert ssug["spin_polarized"]   is True
        assert ssug["spin_total"]       == 2.0
