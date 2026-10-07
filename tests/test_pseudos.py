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

import re

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
        <header>.
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


class TestResolvePsmlLib:
    """The user-facing anchoring rule for cfg.psml_lib: a relative path is
    read from the ``projects/`` tree (the walk-up from the calculation
    folder is the module-level test at the end of this file); an absolute
    path must be inside the tree, ``~/...`` expands; dotted spellings are
    refused."""

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


# --------------------------------------------------------------------- #
#  Open-shell-metal-conditional script templates                       #
#  (the level_shift hint appears in the emitted script ONLY when an    #
#   Fe / Mn / Co / Cu / Ni / etc. is present.  Clean-organic scripts   #
#   must NOT have it — noise.)                                         #
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


    def test_spectra_pyscf_fe_emits_level_shift_template(self):
        # The hint lives in the equilibrium-SCF emitter the vibration
        # deck composes.
        from molbuilder.pyscf.vibration_emitters import _emit_equilibrium_scf
        # The view resolves the state on the structure it is given -- the
        # metals the hint reads are that structure's.
        fe = self._fe()
        text = "\n".join(_emit_equilibrium_scf(_spectra_cfg(
            fe, spin_treatment="unrestricted", unpaired_electrons=2,
        ), fe))
        assert "Hard SCF (typical for open-shell metals like Fe)" in text
        assert "# mf.level_shift = 0.2" in text

    def test_spectra_pyscf_organic_skips_level_shift_template(self):
        from molbuilder.pyscf.vibration_emitters import _emit_equilibrium_scf
        water = self._water()
        text = "\n".join(_emit_equilibrium_scf(
            _spectra_cfg(water, spin_treatment="restricted"), water))
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
        """The five things that make a run wrong rather than suspect.

        `semilocal_only` joined 2026-09-03 (user ruling): a valence channel
        whose projectors STATE a strength of zero, which PseudoDojo v0.5 does
        for eleven elements.  Same class as `xc_family_mismatch` -- the run
        completes and the physics is wrong -- so it blocks for the same
        reason.

        `misnamed` joined 2026-09-19: the file for an element is present and
        healthy but is not called `<element>.psml`, so SIESTA -- which has no
        search path -- never opens it.  A certain start-up failure, and the
        only one of the five that is about the FOLDER rather than the file's
        physics.
        """
        from molbuilder.pseudos import ERROR_STATUSES
        assert ERROR_STATUSES == frozenset(
            {"missing", "misnamed", "dead_projector", "xc_family_mismatch",
             "semilocal_only"})

    def test_a_healthy_pseudo_under_the_wrong_name_blocks(self, tmp_path):
        """The check said `ok` for a folder SIESTA cannot start in.

        `scan_psml_directory` keys on what each file DECLARES, so a correct
        gold pseudopotential saved as `gold.psml` arrived as a healthy entry
        for `Au` and every value check passed it -- while SIESTA opens
        `Au.psml`, does not find it, and refuses.  Measured 2026-09-19: that
        folder answered `ok`.

        Verbatim, no case folding: the module's own rule is that SIESTA
        reads `<label>.psml`, so a species written `Au1` needs `Au1.psml`.
        """
        from molbuilder.pseudos import check_coverage, ERROR_STATUSES
        (tmp_path / "gold.psml").write_text(_make_psml("Au"))
        (tmp_path / "H.psml").write_text(_make_psml("H"))
        by = {e.element: e for e in check_coverage(["Au", "H"], tmp_path)}
        assert by["H"].status == "ok"
        assert by["Au"].status == "misnamed"
        assert by["Au"].status in ERROR_STATUSES
        assert "Au.psml" in by["Au"].message and "gold.psml" in by["Au"].message

    def test_cli_exits_nonzero_on_xc_family_mismatch(self, tmp_path):
        # LDA pseudo on a PBE (GGA) calc -> xc_family_mismatch -> ERROR.
        from click.testing import CliRunner
        from molbuilder.cli import pseudo_group
        (tmp_path / "C.psml").write_text(_make_psml("C", libxc_id=1))  # LDA
        res = CliRunner().invoke(
            pseudo_group, ["check", str(tmp_path), "--xc", "PBE"])
        assert res.exit_code == 1, res.output
        # A word-boundary match: the bare substring "C" is satisfied by
        # almost any output.
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
    # THE CALCULATION: the anchor is the spelling's, not the filesystem's
    # (A10).  There is no local spelling at all: pseudos beside the
    # calculation are used without the field.
    local = calc / "mypseudos"
    local.mkdir()
    assert resolve_psml_lib("mypseudos", dest_dir=calc) == \
        tmp_path / "projects" / "mypseudos"

    # Outside any projects tree there is no tree to walk up to, so the
    # server's own declared root answers -- NOT the working directory, and
    # not the lone folder: in-folder pseudos are used without the field.
    from molbuilder.projects import PROJECTS_ROOT_ENV
    lone = tmp_path / "lone-calc"
    lone.mkdir()
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tmp_path / "tree"))
    assert resolve_psml_lib("pseudopotential", dest_dir=lone) == \
        tmp_path / "tree" / "pseudopotential"
