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
  * scan_psml_directory: drop a few synthetic files, scan, get each
    file by its name.
  * check_coverage: missing element, XC mismatch, relativistic
    mismatch.
  * Tolerance: malformed XML doesn't crash the parser.
"""
from __future__ import annotations

import re

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
    def test_real_pseudodojo_format(self, tmp_path):
        """A real PseudoDojo file, as a person downloads it, read whole.  The
        2026-05-23 regression: synthetic tests used <header> where real files use
        <pseudo-atom-spec>, and an id on <libxc-info> where real files nest
        <functional id="...">; every real file read as element="" and xc=unknown.
        Carbon also tells Z (6) from the valence z-pseudo (4): pre-fix the
        atomic number read back was z-pseudo."""
        from molbuilder.pseudos import parse_psml_header
        info = parse_psml_header(
            Path(__file__).parent / "fixtures" / "psml" / "C.psml")
        assert info.element       == "C"
        assert info.atomic_number == 6       # the TRUE element Z, not z_pseudo
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


def _write_folder(directory, files):
    """A folder of synthetic pseudopotentials: file name -> the element it
    declares, or (element, `_make_psml` keywords), or ``None`` for a file
    whose XML does not parse."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, spec in files.items():
        if spec is None:
            (directory / name).write_text("<psml not closed")
            continue
        element, kw = spec if isinstance(spec, tuple) else (spec, {})
        (directory / name).write_text(_make_psml(element, **kw))


class TestScanPsmlDirectory:
    def test_missing_directory_returns_empty(self, tmp_path):
        from molbuilder.pseudos import scan_psml_directory
        assert scan_psml_directory(tmp_path / "absent") == {}


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

        `misnamed` joined 2026-09-19: the file for a species is present but
        not called `<label>.psml`, so SIESTA -- which has no search path --
        never opens it; or the file it opens declares another element.  A
        certain start-up failure or a wrong species, about the file's NAME
        rather than its physics.
        """
        from molbuilder.pseudos import ERROR_STATUSES
        assert ERROR_STATUSES == frozenset(
            {"missing", "misnamed", "dead_projector", "xc_family_mismatch",
             "semilocal_only"})

    #: C1 / C1a as rows (`science/pseudopotentials.md`): the folder's files
    #: (name -> the element it declares, or (element, `_make_psml`
    #: keywords)), the species labels the structure holds, and the entries
    #: in order -- (label, status, words its message must carry).
    NAMED_FILES = [
        pytest.param({"gold.psml": "Au", "H.psml": "H"}, ["Au", "H"],
                     [("Au", "misnamed", ["gold.psml", "Au.psml",
                                          "never reads"]),
                      ("H", "ok", [])],
                     id="a-file-SIESTA-never-opens"),
        pytest.param({"Au1.psml": "Au", "Au2.psml": "Au"}, ["Au1", "Au2"],
                     [("Au1", "ok", []), ("Au2", "ok", [])],
                     id="two-labels-of-one-element-two-files"),
        pytest.param({"Au.psml": "Au"}, ["Au1"],
                     [("Au1", "misnamed", ["Au.psml", "Au1.psml"])],
                     id="a-label-is-not-covered-by-its-element's-file"),
        pytest.param({"C.psml": "Au"}, ["C"],
                     [("C", "misnamed", ["C.psml", "declares element Au"])],
                     id="the-right-name-another-element"),
        pytest.param({"Au1.psml": "Au"}, ["Au1", "Au2"],
                     [("Au1", "ok", []), ("Au2", "missing", [])],
                     id="a-file-another-label-opens-is-that-label's"),
        pytest.param({"Au1.psml": ("Au", {"creator": "ONCVPSP-3.3.0"}),
                      "Au2.psml": ("Au", {"creator": "ONCVPSP-4.0.1"}),
                      "H.psml": ("H", {"creator": "ONCVPSP-3.3.0"})},
                     ["Au1", "Au2", "H"],
                     [("Au1", "ok", []), ("Au2", "ok", []), ("H", "ok", []),
                      ("Au2", "generator_mismatch", ["ONCVPSP-4"])],
                     id="the-set's-generators-are-the-files-the-labels-open"),
        pytest.param({"Au1.psml": ("Au", {"creator": "ONCVPSP-3.3.0"}),
                      "Au2.psml": ("Au", {"creator": "ONCVPSP-3.3.0"}),
                      "Au_old.psml": ("Au", {"creator": "ONCVPSP-4.0.1"})},
                     ["Au1", "Au2"],
                     [("Au1", "ok", []), ("Au2", "ok", [])],
                     id="a-file-no-label-opens-is-not-in-the-set"),
        pytest.param({"Au.psml": None}, ["Au"],
                     [("Au", "parse_warning", ["could not parse"])],
                     id="a-file-that-does-not-parse-is-C6's-warning"),
        pytest.param({"H.psml": "H"}, ["Au_1", "H"],
                     [("Au_1", "missing", ["not a species label"]),
                      ("H", "ok", [])],
                     id="a-label-no-file-can-be-named-for-is-said"),
    ]

    @pytest.mark.parametrize("files, labels, expected", NAMED_FILES)
    def test_the_file_siesta_opens_is_the_one_checked(self, tmp_path, files,
                                                      labels, expected):
        """SIESTA opens `<label>.psml` and has no search path, so the file a
        label opens is the one judged (C1, C1a).

        A correct gold pseudopotential saved as `gold.psml` answered `ok`
        (measured 2026-09-19), because the folder was keyed on what each
        file DECLARES; the same keying reported a species written `Au1`
        missing beside its own `Au1.psml`, told `Au2` to take `Au1.psml`
        away from `Au1`, and compared the set's generators by element, so
        labelled species were never compared.  Verbatim, no case folding.
        """
        from molbuilder.pseudos import check_coverage
        _write_folder(tmp_path, files)
        got = check_coverage(labels, tmp_path)
        assert [(e.element, e.status) for e in got] == \
            [(label, status) for label, status, _ in expected], \
            [(e.element, e.status, e.message) for e in got]
        for entry, (label, _, words) in zip(got, expected):
            for w in words:
                assert w in entry.message, (label, entry.message)

    #: `molbuilder pseudo check DIR` as rows: the folder (as
    #: `NAMED_FILES`), the options, the exit code, and lines the output must
    #: hold (regular expressions, so a bare "C" is a word, not any letter).
    CLI_ROWS = [
        pytest.param({"C.psml": ("C", {"libxc_id": 1})}, ["--xc", "PBE"],
                     1, [r"\[ERROR\] C\b"],
                     id="an-lda-pseudo-on-a-pbe-calc-blocks"),
        pytest.param({"C.psml": ("C", {"libxc_id": 101})}, ["--xc", "PBE"],
                     0, [r"\[OK   \] C\b"],
                     id="a-matching-set-passes"),
        pytest.param({"Au1.psml": "Au", "Au2.psml": "Au", "H.psml": "H"}, [],
                     0, [r"\[OK   \] Au1\b", r"\[OK   \] Au2\b"],
                     id="a-folder-of-labelled-species-is-screened-by-label"),
        pytest.param({"gold.psml": "Au", "H.psml": "H"}, [],
                     1, [r"\[ERROR\] Au\b.*gold\.psml.*Au\.psml"],
                     id="a-file-named-for-no-label-is-said-misnamed"),
        pytest.param({"Au.psml": None, "H.psml": "H"}, [],
                     0, [r"\[WARN \] Au\b.*could not parse"],
                     id="a-labelled-file-that-does-not-parse-is-screened"),
    ]

    @pytest.mark.parametrize("files, options, code, lines", CLI_ROWS)
    def test_cli_screens_a_folder_by_the_files_it_holds(
            self, tmp_path, files, options, code, lines):
        """With no `--elements`, the folder's own files say what to check:
        each file's label, or the element a file under no label's name
        declares (`pseudos.labels_in_directory`).  Keyed on the declared
        element, `Au1.psml` + `Au2.psml` exited 1 telling the person to
        rename them both to `Au.psml`."""
        from click.testing import CliRunner
        from molbuilder.cli import pseudo_group
        _write_folder(tmp_path, files)
        res = CliRunner().invoke(
            pseudo_group, ["check", str(tmp_path), *options])
        assert res.exit_code == code, res.output
        for line in lines:
            assert re.search(line, res.output), (line, res.output)

    #: The settings gate as rows: the library's files, the calculation
    #: folder's (`pseudos/`, or ``None`` for the library as the folder), the
    #: labels, and the findings by `where`, each (severity, words).
    GATE_ROWS = [
        # The SAME fixture the CLI blocks on is error-severity in the
        # preflight -- the two surfaces agree.
        pytest.param({"C.psml": ("C", {"libxc_id": 1})}, None, ["C"],
                     {"config.psml_lib.C": ("error", ["LDA"])},
                     id="an-lda-pseudo-on-a-pbe-calc-blocks"),
        # Labels read from two places -- Au1 and H from the folder, Au2
        # from the library -- are judged together: the library's Au1.psml
        # is Au1's, never Au2's to take (measured 2026-10-31: Au2 was told
        # to save Au's pseudopotential from it).
        pytest.param({"Au1.psml": "Au"}, {"Au1.psml": "Au", "H.psml": "H"},
                     ["Au1", "Au2", "H"],
                     {"config.psml_lib.Au2": ("error",
                                              ["no .psml file for Au2"])},
                     id="the-structure's-labels-are-judged-together"),
    ]

    @pytest.mark.parametrize("library, folder, labels, expected", GATE_ROWS)
    def test_the_settings_gate_says_each_label_once(self, tmp_path, library,
                                                    folder, labels,
                                                    expected):
        """The SIESTA preflight (`validation.siesta`) maps each label's
        coverage status to its severity through `ERROR_STATUSES`, reading
        each label's file where the run will open it -- the calculation's
        folder first, then the library (`pseudos.psml_sources`)."""
        from molbuilder.validation.siesta import _check_siesta_pseudo_coverage
        lib = tmp_path / "projects" / "psml"
        _write_folder(lib, library)
        calc = lib
        if folder is not None:
            calc = tmp_path / "projects" / "calc"
            _write_folder(calc / "pseudos", folder)

        class _Cfg:
            psml_lib = str(lib)
            xc_authors = "PBE"

        class _Struct:
            elements = labels
        issues = _check_siesta_pseudo_coverage(_Struct(), _Cfg(),
                                               dest_dir=calc)
        assert sorted(i.where for i in issues) == sorted(expected), \
            [(i.where, i.message) for i in issues]
        for i in issues:
            severity, words = expected[i.where]
            assert i.severity == severity, (i.where, i.message)
            for w in words:
                assert w in i.message, (i.where, i.message)


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
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tmp_path / "server" / "projects"))
    assert resolve_psml_lib("pseudopotential", dest_dir=lone) == \
        tmp_path / "server" / "projects" / "pseudopotential"
