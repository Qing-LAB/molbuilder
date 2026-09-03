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

import pathlib
import re

import pytest
from pathlib import Path
from tests.spectra._helpers import _spectra_cfg



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


def _make_pseudodojo_psml(element: str, *, z: int = None,
                            z_pseudo: int = None) -> str:
    """Produce a synthetic PSML that mirrors the REAL PseudoDojo
    format (used by users who download from www.pseudo-dojo.org).
    Differs from _make_psml in two important ways the parser bug
    of 2026-05-23 missed:
      * Element + Z + relativity live on <pseudo-atom-spec>, NOT
        <header>.  My original tests used <header>; real files use
        the longer name.
      * The libxc id is on <functional> CHILDREN of <libxc-info>,
        NOT directly on <libxc-info>.  Real files always nest.
    These tests pin the contract against real-world files so a
    future refactor can't regress.
    """
    if z is None:
        from ase.data import atomic_numbers as _Z
        z = _Z[element]
    if z_pseudo is None:
        z_pseudo = z      # for light elements z_pseudo == z; Fe has 16, etc.
    return f"""<?xml version="1.0" encoding="UTF-8" ?>
<psml version="1.1" energy_unit="hartree" length_unit="bohr"
 uuid="00000000-0000-0000-0000-000000000000"
 xmlns="http://esl.cecam.org/PSML/ns/1.1">
<provenance creator="ONCVPSP-3.3.0+psml-3.3.0-73 (scalar-relativistic)"/>
<pseudo-atom-spec atomic-label="{element}" atomic-number="{z}"
 z-pseudo="{z_pseudo}"
 flavor="Hamann oncvpsp" relativity="scalar" spin-dft="no">
<exchange-correlation>
<libxc-info number-of-functionals="2">
<functional name="Perdew, Burke &amp; Ernzerhof (GGA)" type="exchange" id="101"/>
<functional name="Perdew, Burke &amp; Ernzerhof (GGA)" type="correlation" id="130"/>
</libxc-info>
</exchange-correlation>
</pseudo-atom-spec>
</psml>"""


class TestParsePsmlHeader:
    def test_real_pseudodojo_format(self, tmp_path):
        """Pin parsing of REAL PseudoDojo PSML format.  The 2026-05-23
        regression was: my synthetic tests used <header> but real files
        use <pseudo-atom-spec>; my <libxc-info id="..."> structure but
        real files nest <functional id="..."> inside <libxc-info>.  Both
        bugs missed every real PseudoDojo file (returned element="",
        xc=unknown).  THIS test uses the real shape -- if it passes,
        the user's actual downloads work."""
        from molbuilder.pseudos import parse_psml_header
        # Fe specifically catches the z-pseudo (16 valence) vs Z (26)
        # bug: pre-fix returned atomic_number=16.
        (tmp_path / "Fe.psml").write_text(_make_pseudodojo_psml("Fe",
                                                                  z=26, z_pseudo=16))
        info = parse_psml_header(tmp_path / "Fe.psml")
        assert info.element       == "Fe"
        assert info.atomic_number == 26      # the TRUE element Z, not z_pseudo
        assert info.xc_family     == "GGA"
        assert info.xc_authors    == "PBE"
        assert info.relativistic  == "scalar"
        assert info.parse_warnings == []

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
        """LDA pseudo on a GGA calc -- silently-wrong bond lengths.  A
        FAMILY mismatch is a distinct status (xc_family_mismatch), which the
        SIESTA validator maps to ERROR (blocks); it is never physically
        correct.  (Same-family author diffs stay 'xc_mismatch' / WARN.)"""
        (tmp_path / "C.psml").write_text(_make_psml("C", libxc_id=1))  # LDA
        from molbuilder.pseudos import check_coverage
        entries = check_coverage(
            ("C",), tmp_path,
            expected_xc_family="GGA", expected_xc_authors="PBE",
        )
        assert entries[0].status == "xc_family_mismatch"
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


# (picker_root_at_tmp retired 2026-08-22 with the seven C-doors tests
#  it served -- install-pseudos/install-wrapper are gone, and the
#  surviving analyze tests need no picker root.)


class TestResolvePsmlLib:
    """The user-facing anchoring rule for cfg.psml_lib (introduced
    2026-05-24; three-stage since 2026-08-21): a relative path tries the
    calculation folder first, then the ``projects/`` tree the calculation
    lives in (walking up from the calculation folder), then
    ``<cwd>/projects/``; absolute paths pass through, ``~/...`` expands.
    Was motivated by a user typing ``../../../pseudopotential`` and
    being confused that the validator resolved it against the Flask
    server's CWD (repo root) rather than anything meaningful — and
    completed by the 2026-08-21 Sol bug, where the cwd fallback anchored
    a bare name at ``<calc>/projects/…`` (the module-level walk-up test
    at the end of this file)."""

    def test_absolute_inside_the_tree_passes_through(self, tmp_path):
        from molbuilder.pseudos import resolve_psml_lib
        out = resolve_psml_lib(str(tmp_path / "foo"), base=tmp_path)
        assert out == tmp_path / "foo"

    def test_absolute_outside_the_tree_is_refused(self, tmp_path):
        """2026-08-28: `psml_lib` always lives inside the tree, so an
        outside absolute path has no honest answer -- and the refusal
        names the tree it should move into."""
        from molbuilder.pseudos import PsmlLibError, resolve_psml_lib
        import pytest as _pytest
        with _pytest.raises(PsmlLibError) as e:
            resolve_psml_lib("/somewhere/else", base=tmp_path)
        assert "outside" in str(e.value) and str(tmp_path) in str(e.value)

    def test_relative_anchored_at_projects(self, tmp_path):
        from molbuilder.pseudos import resolve_psml_lib
        out = resolve_psml_lib("pseudopotential", base=tmp_path)
        assert out == tmp_path / "pseudopotential"

    def test_nested_relative_anchored_at_projects(self, tmp_path):
        from molbuilder.pseudos import resolve_psml_lib
        out = resolve_psml_lib("shared/pbe_sr", base=tmp_path)
        assert out == tmp_path / "shared" / "pbe_sr"

    def test_dotted_spellings_are_refused_with_the_reason(self, tmp_path):
        """2026-08-28: the dotted anchor retired with the cascade --
        pseudos beside the calculation are used without the field."""
        from molbuilder.pseudos import PsmlLibError, resolve_psml_lib
        import pytest as _pytest
        for raw in ("./foo", "../foo"):
            with _pytest.raises(PsmlLibError) as e:
                resolve_psml_lib(raw, base=tmp_path)
            assert "retired" in str(e.value)


# (_envelope + _from_file_text retired 2026-08-22: their last
#  caller left with the install-pseudos tests.)



class TestPseudosEndpoint:
    # (The seven install-pseudos / install-wrapper tests that opened this
    #  class retired 2026-08-21 with their doors (C-doors): both endpoints
    #  had zero browser callers -- `prep` writes the wrapper and resolves +
    #  copies the pseudopotentials on the described route.  The class name
    #  survives for the /api/structure/analyze tests below, which is the
    #  door that remains.)

    def test_structure_analyze_unknown_element_returns_400_not_500(self):
        """Unknown element symbol (typo, bad PDB column fallback) must
        return a clear 400 with the parser's message, NOT a 500
        Internal Server Error (which leaks the stack trace to the
        client).  Code-review fix 2026-05-23."""
        from molbuilder.web.app import create_app
        c = create_app(config={}).test_client()
        # BY HAND, not through `_envelope`: the subject is what the ROUTE
        # does with a symbol it cannot weigh, so the symbol has to reach it
        # rather than die in this process's own parser.
        r = c.post("/api/structure/analyze", json={"structure": {
            "elements": ["Xy"], "positions": [[0.0, 0.0, 0.0]],
            "metadata": {}}})
        assert r.status_code == 400, r.data
        body = r.get_json()
        assert body["ok"] is False
        # NAMES THE OFFENDING SYMBOL -- that is what makes the 400 actionable,
        # and it is what this test is for.  The phrase "unknown element" was
        # pinned here too until 2026-08-03; the parser now says "could not read
        # XYZ: 'Xy'", which is the same fact in its own words.  The docstring
        # above already delegates the wording ("with the parser's message"), so
        # pinning a phrase contradicted the test's own stated contract.
        assert "Xy" in body["error"]

    @staticmethod
    def _envelope(xyz: str) -> dict:
        """The XYZ as the route takes it (`web-api.md` § 1).

        These posted ``structure_text`` until 2026-09-02, when that field was
        removed from the route -- it had gone from `/api/spectra/render` on
        2026-08-03 and this door was missed by the sweep.  The file's own
        `test_analyze_accepts_the_shape_its_callers_actually_send` recorded
        the consequence a month earlier: *"No caller sends that … the covered
        shape and the used shape are different, which is exactly how
        install-pseudos answered 400 to every real save for weeks with its
        own tests green."*  The chemistry below is unchanged; only the
        delivery moved, and it moved onto the shape the tabs use.
        """
        from molbuilder.structure import Structure
        st = Structure.from_xyz(xyz)
        return {"structure": {"elements": list(st.elements),
                              "positions": [list(map(float, r))
                                            for r in st.positions],
                              "metadata": {}}}

    def test_structure_analyze_organic_no_metals(self):
        """No metals -> closed-shell singlet (or doublet for odd-e)."""
        from molbuilder.web.app import create_app
        c = create_app(config={}).test_client()
        water = "3\nwater\nO 0 0 0\nH 1 0 0\nH -1 0 0\n"
        r = c.post("/api/structure/analyze", json=self._envelope(water))
        body = r.get_json()
        assert body["ok"] is True
        assert body["metals"] == []
        sug = body["suggested"]["pyscf"]
        assert sug["net_charge"] == 0
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
        r = c.post("/api/structure/analyze", json=self._envelope(xyz))
        body = r.get_json()
        assert body["metals"] == ["Fe"]
        sug = body["suggested"]["pyscf"]
        assert sug["spin"]   == 2
        assert sug["method"] == "UKS"
        assert "Fe" in sug["rationale"]
        # The rationale is engine-AGNOSTIC after the Phase-1c refactor
        # (science/validation.md § 3) — it says "open-shell
        # treatment", not "UKS" / "RKS" (those are PySCF strings that
        # belong in the adapter output, not the analyzer's rationale).
        # The UKS / RKS choice is pinned by the assert on
        # ``sug["method"]`` two lines up.
        assert "open-shell" in sug["rationale"]
        # SIESTA equivalent: spin_treatment="polarized" + spin_total=2.0
        ssug = body["suggested"]["siesta"]
        assert ssug["spin_treatment"]   == "polarized"
        assert ssug["spin_total"]       == 2.0


# --------------------------------------------------------------------- #
#  Security: path-traversal protection (2026-05-23 review fix)         #
# --------------------------------------------------------------------- #


class TestEndpointPathSecurity:
    """Both new endpoints accept user-supplied paths.  Without picker-
    root validation, a malicious POST could pass ``/etc/passwd`` or
    ``/root/.ssh/id_rsa`` and the endpoint would happily read /
    process it.  These tests pin the validation contract: paths
    outside the picker roots must be 400-rejected."""

    def test_structure_analyze_rejects_path_outside_roots(self):
        """Same security check for /api/structure/analyze."""
        from molbuilder.web.app import create_app
        c = create_app(config={}).test_client()
        r = c.post("/api/structure/analyze", json={
            "structure_path": "/etc/passwd",
        })
        assert r.status_code == 400
        msg = r.get_json()["error"].lower()
        assert "outside" in msg or "root" in msg or "allowed" in msg


# --------------------------------------------------------------------- #
#  Open-shell-metal-conditional script templates                       #
#  (level_shift / spin-sweep added 2026-05-23 — discoverable hints in  #
#   the emitted script ONLY when an Fe / Mn / Co / Cu / Ni / etc. is   #
#   present.  Clean-organic scripts must NOT have them — noise.)       #
# --------------------------------------------------------------------- #


class TestMetalAwareScriptTemplates:
    def _fe(self):
        from molbuilder.structure import Structure
        import numpy as np
        return Structure(elements=["Fe", "N", "N", "N", "N"],
                         positions=np.array([[0, 0, 0], [2, 0, 0], [-2, 0, 0],
                                              [0, 2, 0], [0, -2, 0]]),
                         vacuum=(12.0, 12.0, 12.0))   # planar -> needs vacuum for a real box

    def _water(self):
        from molbuilder.structure import Structure
        import numpy as np
        return Structure(elements=["O", "H", "H"],
                         positions=np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]]),
                         vacuum=(12.0, 12.0, 12.0))   # linear -> needs vacuum for a real box

    def test_siesta_fe_emits_spin_sweep_template(self):
        from molbuilder.siesta import render_fdf
        from molbuilder.config.siesta import SiestaConfig
        fdf = render_fdf(self._fe(), SiestaConfig(
            net_charge=0, spin_treatment="polarized", spin_total=2.0,
        ))
        assert "Spin-state sweep template" in fdf
        assert "Fe(II) candidates" in fdf
        assert "Fe(III) candidates" in fdf
        # Mossbauer / EPR / UV-Vis caveat -- user must verify.
        assert "Mossbauer" in fdf or "ssbauer" in fdf

    def test_siesta_organic_skips_spin_sweep_template(self):
        from molbuilder.siesta import render_fdf
        from molbuilder.config.siesta import SiestaConfig
        fdf = render_fdf(self._water(), SiestaConfig(
            net_charge=0, spin_treatment="polarized", spin_total=0.0,
        ))
        assert "Spin-state sweep template" not in fdf

    def test_build_pyscf_fe_emits_level_shift_template(self):
        from molbuilder.pyscf.input import render_script
        from molbuilder.config.pyscf import PySCFConfig
        text = render_script(self._fe(), PySCFConfig(
            method="UKS", spin=2, optimize=False,
        ))
        assert "Hard SCF (typical for open-shell metals like Fe)" in text
        assert "# mf.level_shift = 0.2" in text   # commented template
        # Compile-check: commented template must not break syntax.
        compile(text, "<fe-build>", "exec")

    def test_build_pyscf_organic_skips_level_shift_template(self):
        from molbuilder.pyscf.input import render_script
        from molbuilder.config.pyscf import PySCFConfig
        text = render_script(self._water(), PySCFConfig(
            method="RKS", spin=0, optimize=False,
        ))
        assert "Hard SCF (typical for open-shell metals" not in text

    def test_spectra_pyscf_fe_emits_level_shift_template(self):
        # P3: the generator retired; the hint lives in the surviving
        # equilibrium-SCF emitter the vibration deck composes.
        from molbuilder.pyscf.vibration_emitters import _emit_equilibrium_scf
        text = "\n".join(_emit_equilibrium_scf(_spectra_cfg(
            method="UKS", spin=2,
        ), self._fe()))
        assert "Hard SCF (typical for open-shell metals like Fe)" in text
        assert "# mf.level_shift = 0.2" in text

    def test_spectra_pyscf_organic_skips_level_shift_template(self):
        from molbuilder.pyscf.vibration_emitters import _emit_equilibrium_scf
        text = "\n".join(_emit_equilibrium_scf(
            _spectra_cfg(method="RKS", spin=0), self._water()))
        assert "Hard SCF (typical for open-shell metals" not in text


# ===================================================================== #
#  Conformance tests for docs/science/pseudopotentials.md   #
#  C4 (generator/version) + C5 (dead KB projector).  Each test pins a   #
#  clause of the standard to the implementation.                        #
# ===================================================================== #


def _psml_with_projectors(element, projectors, *, z,
                          creator="ONCVPSP-3.3.0+psml-3.3.0-73 "
                                  "(scalar-relativistic)"):
    """Synthetic PSML carrying a <nonlocal-projectors> block.

    projectors: list of (l_letter, ekb_float).  Lets a test build a
    pseudo with a deliberately dead channel (all ekb=0 for an l).
    """
    proj = "\n".join(
        f'<proj l="{l}" seq="{i+1}" ekb="{ekb}" eref="0" type="oncv"/>'
        for i, (l, ekb) in enumerate(projectors)
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<psml version="1.1" xmlns="http://esl.cecam.org/PSML/ns/1.1">
<provenance creator="{creator}"/>
<pseudo-atom-spec atomic-label="{element}" atomic-number="{z}"
 z-pseudo="{z}" relativity="scalar"/>
<exchange-correlation><libxc-info>
<functional type="exchange" id="101"/>
<functional type="correlation" id="130"/>
</libxc-info></exchange-correlation>
<nonlocal-projectors>
{proj}
</nonlocal-projectors>
</psml>"""


# A sulfur with s + d projectors real but the ENTIRE p-channel dead --
# exactly the defective BDT S.psml (ONCVPSP-4.0.1) that motivated C5.
_S_DEAD_P = [("s", 6.774), ("s", 0.542),
             ("p", 0.0), ("p", 0.0),
             ("d", 0.0), ("d", 3.022)]
_S_GOOD = [("s", 6.764), ("s", 0.574),
           ("p", 3.227), ("p", 0.887),
           ("d", -3.548), ("d", -0.987)]


class TestDeadProjectorC5:
    def test_null_channel_detected(self, tmp_path):
        from molbuilder.pseudos import parse_psml_header
        p = tmp_path / "S.psml"
        p.write_text(_psml_with_projectors("S", _S_DEAD_P, z=16))
        info = parse_psml_header(p)
        # Only 'p' is fully null; 'd' has one real projector (3.022) so
        # it is NOT flagged -- the standard requires the WHOLE channel.
        assert info.null_channels == ["p"]

    def test_good_pseudo_has_no_null_channels(self, tmp_path):
        from molbuilder.pseudos import parse_psml_header
        p = tmp_path / "S.psml"
        p.write_text(_psml_with_projectors("S", _S_GOOD, z=16))
        assert parse_psml_header(p).null_channels == []

    def test_absent_channel_is_not_flagged(self, tmp_path):
        # A channel chosen as LOCAL has no <proj> entries at all; absent
        # != present-but-zero, so it must NOT be flagged (C5 clause).
        from molbuilder.pseudos import parse_psml_header
        p = tmp_path / "X.psml"
        p.write_text(_psml_with_projectors(
            "C", [("s", 5.0), ("s", 0.3)], z=6))  # no p/d entries at all
        assert parse_psml_header(p).null_channels == []

    def test_dead_projector_is_error_status(self, tmp_path):
        from molbuilder.pseudos import check_coverage
        (tmp_path / "S.psml").write_text(
            _psml_with_projectors("S", _S_DEAD_P, z=16))
        [entry] = check_coverage(["S"], tmp_path)
        assert entry.status == "dead_projector"
        assert "p" in entry.message and "ekb=0" in entry.message

    def test_dead_projector_maps_to_error_severity(self, tmp_path):
        # The validation layer must escalate dead_projector to ERROR.
        from molbuilder.validation.siesta import _check_siesta_pseudo_coverage

        lib = tmp_path / "projects" / "psml"
        lib.mkdir(parents=True)

        class _Cfg:
            psml_lib = str(lib)
            xc_authors = "PBE"
        (lib / "S.psml").write_text(
            _psml_with_projectors("S", _S_DEAD_P, z=16))

        class _Struct:
            elements = ["S"]
        issues = _check_siesta_pseudo_coverage(_Struct(), _Cfg(),
                                               dest_dir=lib)
        assert any(i.severity == "error" and "Kleinman" in i.message
                   for i in issues), [(_i.severity, _i.message) for _i in issues]


class TestGeneratorVersionC4:
    def test_mixed_major_version_warns(self, tmp_path):
        from molbuilder.pseudos import check_coverage
        (tmp_path / "C.psml").write_text(_psml_with_projectors(
            "C", _S_GOOD, z=6,
            creator="ONCVPSP-3.3.0+psml-3.3.0-73 (scalar-relativistic)"))
        (tmp_path / "S.psml").write_text(_psml_with_projectors(
            "S", _S_GOOD, z=16,
            creator="ONCVPSP-4.0.1+psml-4.0.1-76 (scalar-relativistic)"))
        entries = check_coverage(["C", "S"], tmp_path)
        gm = [e for e in entries if e.status == "generator_mismatch"]
        assert len(gm) == 1
        # The minority (S, the v4 stranger) is named.
        assert "S" in gm[0].element

    def test_patch_difference_does_not_warn(self, tmp_path):
        from molbuilder.pseudos import check_coverage
        (tmp_path / "C.psml").write_text(_psml_with_projectors(
            "C", _S_GOOD, z=6,
            creator="ONCVPSP-3.3.0+psml-3.3.0-73 (scalar-relativistic)"))
        (tmp_path / "S.psml").write_text(_psml_with_projectors(
            "S", _S_GOOD, z=16,
            creator="ONCVPSP-3.3.1+psml-3.3.1-99 (scalar-relativistic)"))
        entries = check_coverage(["C", "S"], tmp_path)
        assert not [e for e in entries if e.status == "generator_mismatch"]

    def test_generator_key_reduces_to_name_major(self):
        from molbuilder.pseudos import _generator_key
        assert _generator_key(
            "ONCVPSP-4.0.1+psml-4.0.1-76 (scalar-relativistic)") == "ONCVPSP-4"
        assert _generator_key(
            "ONCVPSP-3.3.0+psml-3.3.0-73 (scalar-relativistic)") == "ONCVPSP-3"


class TestErrorStatusesSharedBySurfaces:
    """The CLI (``molbuilder pseudo check``) and the SIESTA preflight must
    block on the SAME statuses.  They drifted until 2026-07-26: the CLI's
    exit set omitted ``xc_family_mismatch``, so an XC-family mismatch that
    the preflight blocked slipped past ``pseudo check`` with exit 0.  Both
    now consume ``pseudos.ERROR_STATUSES`` so the surfaces can't disagree."""

    def test_error_statuses_is_the_blocking_set(self):
        from molbuilder.pseudos import ERROR_STATUSES
        assert ERROR_STATUSES == frozenset(
            {"missing", "dead_projector", "xc_family_mismatch"})

    def test_cli_exits_nonzero_on_xc_family_mismatch(self, tmp_path):
        # LDA pseudo on a PBE (GGA) calc -> xc_family_mismatch -> ERROR.
        from click.testing import CliRunner
        from molbuilder.cli import pseudo_group
        (tmp_path / "C.psml").write_text(_make_psml("C", libxc_id=1))  # LDA
        res = CliRunner().invoke(
            pseudo_group, ["check", str(tmp_path), "--xc", "PBE"])
        assert res.exit_code == 1, res.output
        # A word-boundary match: the bare substring "C" was satisfied by
        # almost any output (a near-tautology, found 2026-08-12).
        assert "ERROR" in res.output
        assert re.search(r"\bC\b", res.output), res.output

    def test_cli_exits_zero_on_matching_set(self, tmp_path):
        # PBE (GGA) pseudo on a PBE calc -> ok -> exit 0.
        from click.testing import CliRunner
        from molbuilder.cli import pseudo_group
        (tmp_path / "C.psml").write_text(_make_psml("C", libxc_id=101))  # PBE
        res = CliRunner().invoke(
            pseudo_group, ["check", str(tmp_path), "--xc", "PBE"])
        assert res.exit_code == 0, res.output

    def test_preflight_maps_xc_family_mismatch_to_error(self, tmp_path):
        # The SAME fixture the CLI now blocks on must also be error-severity
        # in the preflight -- proves the two surfaces agree.
        from molbuilder.validation.siesta import _check_siesta_pseudo_coverage
        lib = tmp_path / "projects" / "psml"
        lib.mkdir(parents=True)
        (lib / "C.psml").write_text(_make_psml("C", libxc_id=1))  # LDA

        class _Cfg:
            psml_lib = str(lib)
            xc_authors = "PBE"

        class _Struct:
            elements = ["C"]
        issues = _check_siesta_pseudo_coverage(_Struct(), _Cfg(),
                                               dest_dir=lib)
        assert any(i.severity == "error" for i in issues), \
            [(i.severity, i.message) for i in issues]


def test_analyze_accepts_the_shape_its_callers_actually_send(tmp_path, monkeypatch):
    """THE GAP THAT LET THE install-pseudos BREAK GO UNNOTICED, closed one door
    over.

    Every test above drove /api/structure/analyze with ``structure_text``.  No
    caller sent that: structure-optimization, spectra and transport all send
    ``structure_path`` (four call sites, checked 2026-08-04).  So the covered
    shape and the used shape were different, which is exactly how
    install-pseudos answered 400 to every real save for weeks with its own
    tests green -- the suite exercised a request nobody makes.

    **Closed at the source on 2026-09-02**: the field is gone from the route,
    and the tests above now send the envelope.  This one stays, because the
    gap it names is not "that field" -- it is *the covered shape drifting
    from the used shape*, which the next field could reopen.

    This pins the request the tabs actually make.  It is deliberately thin: the
    chemistry is tested above and does not need repeating; what needs a test is
    that THE SHAPE THE CLIENT SENDS REACHES AN ANSWER.
    """
    from molbuilder import diagnostics
    from molbuilder.web.app import create_app

    xyz = tmp_path / "water.xyz"
    xyz.write_text("3\nwater\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n")
    caps = diagnostics.Capabilities(
        runtime_config={}, conda_binary=None, conda_envs=frozenset())
    monkeypatch.setattr(type(caps), "file_picker_roots",
                        lambda self: ((tmp_path.resolve(), "projects"),))
    diagnostics.set_capabilities(caps)

    c = create_app(config={}).test_client()
    r = c.post("/api/structure/analyze",
               json={"structure_path": str(xyz.resolve())})
    assert r.status_code == 200, r.data
    body = r.get_json()
    assert body["ok"] is True
    assert body["n_atoms"] == 3
    assert sorted(body["elements"]) == ["H", "O"]


def test_a_relative_lib_resolves_through_the_calculations_own_tree(
        tmp_path, monkeypatch):
    """The 2026-08-21 Sol bug: a verb run with the calculation
    folder as the working directory, and the old fallback anchored a bare
    ``pseudopotential`` at ``<cwd>/projects/...`` -- "stuck with the pwd".
    The calculation KNOWS its own tree: a bare spelling walks up from the
    calculation folder to the nearest ``projects`` ancestor and anchors
    there, wherever the process happens to be standing (`job-contracts.md`
    § 2.5a; the full matrix is `test_psml_anchor.py`)."""
    from molbuilder.pseudos import resolve_psml_lib
    lib = tmp_path / "projects" / "pseudopotential"
    lib.mkdir(parents=True)
    calc = tmp_path / "projects" / "Au-BDT-Au" / "optimization" / "Relax"
    calc.mkdir(parents=True)
    elsewhere = tmp_path / "somewhere-else"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    got = resolve_psml_lib("pseudopotential", dest_dir=calc)
    assert got == lib

    # A BARE NAME MEANS THE TREE EVEN WHEN A SAME-NAMED FOLDER SITS BESIDE
    # THE CALCULATION.  This asserted the opposite until 2026-08-21 -- back
    # then resolution tried the calculation folder first and took it if it
    # existed, so what a spelling meant depended on what happened to be on
    # disk.  That is the property A10 removes: the anchor is the spelling's,
    # not the filesystem's.  (Since 2026-08-28 there is no local spelling
    # at all: pseudos beside the calculation are used without the field.)
    local = calc / "mypseudos"
    local.mkdir()
    assert resolve_psml_lib("mypseudos", dest_dir=calc) == \
        tmp_path / "projects" / "mypseudos"

    # Outside any projects tree there is no tree to walk up to, so the
    # server's own declared root answers (2026-08-28) -- NOT the working
    # directory, and no longer the lone folder (the old cascade's last
    # anchor, retired: in-folder pseudos are used without the field).
    from molbuilder.projects import PROJECTS_ROOT_ENV
    lone = tmp_path / "lone-calc"
    lone.mkdir()
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tmp_path / "tree"))
    assert resolve_psml_lib("pseudopotential", dest_dir=lone) == \
        tmp_path / "tree" / "pseudopotential"
