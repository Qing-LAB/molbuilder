"""Tests for the PSML pseudopotential header parser + coverage check.

The 2026-05-23 SIESTA-help-text pass surfaced that the user is on
their own when picking pseudos from PseudoDojo (NC vs PAW, SR vs FR,
which XC family).  This module gives molbuilder enough metadata
awareness to validate a downloaded pseudo directory at preflight
and call out mismatches BEFORE the user discovers them in a wrong-
bond-length SIESTA run.

Correct input is a real file: PseudoDojo's own, `tests/fixtures/psml/`.  A real
file reads right, a set of them is judged right, and a file that is not one is
said so, with no element made up.
"""
from __future__ import annotations

import re

import pytest

import shutil

from _road import PSML_DIR


#: What a folder of a test holds: an element's real file, v0.4's set unless a
#: release is named.
_REAL = {"H": PSML_DIR / "H.psml", "C": PSML_DIR / "C.psml",
         "O": PSML_DIR / "O.psml",
         "S": PSML_DIR / "pseudodojo-v0.4" / "S.psml",
         "S v0.5": PSML_DIR / "pseudodojo-v0.5" / "S.psml"}


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
            _REAL["C"])
        assert info.element       == "C"
        assert info.atomic_number == 6       # the TRUE element Z, not z_pseudo
        assert info.xc_family     == "GGA"
        assert info.xc_authors    == "PBE"
        assert info.relativistic  == "scalar"
        assert info.generator.startswith("ONCVPSP-3.3.0")
        assert info.null_channels == []      # d has no <proj>: absent, not dead
        assert info.parse_warnings == []

    def test_a_fully_relativistic_file_reads_as_spin_orbit(self, tmp_path):
        """PSML's `relativity="dirac"`, so C3 can compare it with a spin-orbit
        calculation; read as unknown, C3 goes silent.  None of the scalar sets
        here is one, so this is a real file with that word changed."""
        from molbuilder.pseudos import parse_psml_header
        text = _REAL["H"].read_text(encoding="utf-8")
        assert 'relativity="scalar"' in text
        p = tmp_path / "H.psml"
        p.write_text(text.replace('relativity="scalar"', 'relativity="dirac"'),
                     encoding="utf-8")
        assert parse_psml_header(p).relativistic == "spin-orbit"

    @pytest.mark.parametrize("text", ["<psml not closed properly",
                                      "<psml version='1.1'/>"],
                             ids=["not-xml", "xml-holding-no-element"])
    def test_a_file_that_is_not_one_says_so(self, tmp_path, text):
        """Not a file this reader knows: no element made up, and a warning
        instead of an exception."""
        p = tmp_path / "broken.psml"
        p.write_text(text)
        from molbuilder.pseudos import parse_psml_header
        info = parse_psml_header(p)
        assert info.element == ""
        assert info.parse_warnings


def _write_folder(directory, files):
    """A folder of real pseudopotentials under these file names: file name ->
    a key of `_REAL`, or ``None`` for a file whose XML does not parse."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, element in files.items():
        if element is None:
            (directory / name).write_text("<psml not closed")
            continue
        shutil.copy(_REAL[element], directory / name)


class TestScanPsmlDirectory:
    def test_missing_directory_returns_empty(self, tmp_path):
        from molbuilder.pseudos import scan_psml_directory
        assert scan_psml_directory(tmp_path / "absent") == {}


class TestCheckCoverage:
    def test_all_present_with_matching_xc(self, tmp_path):
        _write_folder(tmp_path, {f"{el}.psml": el for el in ("C", "H", "O")})
        from molbuilder.pseudos import check_coverage
        entries = check_coverage(
            ("C", "H", "O"), tmp_path,
            expected_xc_family="GGA", expected_xc_authors="PBE",
        )
        assert len(entries) == 3
        assert all(e.status == "ok" for e in entries)

    def test_missing_element_flagged(self, tmp_path):
        _write_folder(tmp_path, {"C.psml": "C"})
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
        """A GGA pseudo on an LDA calc -- silently-wrong bond lengths.  A
        FAMILY mismatch is a distinct status (xc_family_mismatch), which the
        SIESTA validator maps to ERROR (blocks); it is never physically
        correct.  (Same-family author diffs stay 'xc_mismatch' / WARN.)"""
        _write_folder(tmp_path, {"C.psml": "C"})
        from molbuilder.pseudos import check_coverage
        entries = check_coverage(
            ("C",), tmp_path,
            expected_xc_family="LDA", expected_xc_authors="CA",
        )
        assert entries[0].status == "xc_family_mismatch"
        assert "LDA" in entries[0].message
        assert "GGA" in entries[0].message

    def test_xc_authors_mismatch_within_family(self, tmp_path):
        """PBE pseudo + PBEsol calc -- same family, minor mismatch."""
        _write_folder(tmp_path, {"C.psml": "C"})
        from molbuilder.pseudos import check_coverage
        entries = check_coverage(
            ("C",), tmp_path,
            expected_xc_family="GGA", expected_xc_authors="PBEsol",
        )
        assert entries[0].status == "xc_mismatch"
        assert "PBE" in entries[0].message and "PBEsol" in entries[0].message

    def test_relativistic_mismatch(self, tmp_path):
        """Scalar pseudo + spin-orbit calc -- WARN."""
        _write_folder(tmp_path, {"C.psml": "C"})
        from molbuilder.pseudos import check_coverage
        entries = check_coverage(
            ("C",), tmp_path, expected_relativistic="spin-orbit",
        )
        assert entries[0].status == "relativistic_mismatch"

    def test_duplicate_elements_in_structure_dedup(self, tmp_path):
        """A structure with many C atoms only counts as ONE coverage
        entry for C."""
        _write_folder(tmp_path, {"C.psml": "C", "H.psml": "H"})
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


def _a_channel_nothing_carries(directory):
    """Real v0.5 sulfur with its `<slps l="p">` block taken out: p's projectors
    zero and nothing carrying the channel, C5's case.  No file of either
    PseudoDojo table has it, so this is made from one that nearly does."""
    text = re.sub(r'<slps n="\d+" l="p".*?</slps>\s*', "",
                  _REAL["S v0.5"].read_text(encoding="utf-8"), flags=re.S)
    (directory / "S.psml").write_text(text, encoding="utf-8")


class TestDeadProjectorC5:
    def test_a_channel_nothing_carries_is_dead_and_blocks(self, tmp_path):
        """Only p is named: `d` has one real projector, and the standard asks
        of the WHOLE channel.  The screening and the preflight say so alike."""
        from molbuilder.pseudos import check_coverage, parse_psml_header
        from molbuilder.validation.siesta import _check_siesta_pseudo_coverage
        lib = tmp_path / "projects" / "psml"
        lib.mkdir(parents=True)
        _a_channel_nothing_carries(lib)
        assert parse_psml_header(lib / "S.psml").null_channels == ["p"]
        [entry] = check_coverage(["S"], lib)
        assert entry.status == "dead_projector"
        assert "p" in entry.message and "ekb=0" in entry.message

        class _Cfg:
            psml_lib = str(lib)
            xc_authors = "PBE"

        class _Struct:
            elements = ["S"]
        issues = _check_siesta_pseudo_coverage(_Struct(), _Cfg(), dest_dir=lib)
        assert any(i.severity == "error" and "Kleinman" in i.message
                   for i in issues), [(_i.severity, _i.message) for _i in issues]


class TestGeneratorVersionC4:
    def test_mixed_major_version_warns(self, tmp_path):
        """v0.4's carbon and hydrogen beside v0.5's sulfur: the two releases'
        generators, ONCVPSP 3 and 4, named by the stranger's element."""
        from molbuilder.pseudos import check_coverage
        _write_folder(tmp_path, {"C.psml": "C", "H.psml": "H",
                                 "S.psml": "S v0.5"})
        gm = [e for e in check_coverage(["C", "H", "S"], tmp_path)
              if e.status == "generator_mismatch"]
        assert [e.element for e in gm] == ["S"]

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
    #: (`_write_folder`), the species labels the structure holds, and the
    #: entries in order -- (label, status, words its message must carry).
    NAMED_FILES = [
        pytest.param({"carbon.psml": "C", "H.psml": "H"}, ["C", "H"],
                     [("C", "misnamed", ["carbon.psml", "C.psml",
                                         "never reads"]),
                      ("H", "ok", [])],
                     id="a-file-SIESTA-never-opens"),
        pytest.param({"C1.psml": "C", "C2.psml": "C"}, ["C1", "C2"],
                     [("C1", "ok", []), ("C2", "ok", [])],
                     id="two-labels-of-one-element-two-files"),
        pytest.param({"C.psml": "C"}, ["C1"],
                     [("C1", "misnamed", ["C.psml", "C1.psml"])],
                     id="a-label-is-not-covered-by-its-element's-file"),
        pytest.param({"C.psml": "O"}, ["C"],
                     [("C", "misnamed", ["C.psml", "declares element O"])],
                     id="the-right-name-another-element"),
        pytest.param({"C1.psml": "C"}, ["C1", "C2"],
                     [("C1", "ok", []), ("C2", "missing", [])],
                     id="a-file-another-label-opens-is-that-label's"),
        pytest.param({"S1.psml": "S", "S2.psml": "S v0.5", "H.psml": "H"},
                     ["S1", "S2", "H"],
                     [("S1", "ok", []), ("S2", "semilocal_only", []),
                      ("H", "ok", []),
                      ("S2", "generator_mismatch", ["ONCVPSP-4"])],
                     id="the-set's-generators-are-the-files-the-labels-open"),
        pytest.param({"S1.psml": "S", "S2.psml": "S", "S_old.psml": "S v0.5"},
                     ["S1", "S2"],
                     [("S1", "ok", []), ("S2", "ok", [])],
                     id="a-file-no-label-opens-is-not-in-the-set"),
        pytest.param({"C.psml": None}, ["C"],
                     [("C", "parse_warning", ["could not parse"])],
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
        pytest.param({"C.psml": "C"}, ["--xc", "CA"],
                     1, [r"\[ERROR\] C\b"],
                     id="a-gga-pseudo-on-an-lda-calc-blocks"),
        pytest.param({"C.psml": "C"}, ["--xc", "PBE"],
                     0, [r"\[OK   \] C\b"],
                     id="a-matching-set-passes"),
        pytest.param({"C1.psml": "C", "C2.psml": "C", "H.psml": "H"}, [],
                     0, [r"\[OK   \] C1\b", r"\[OK   \] C2\b"],
                     id="a-folder-of-labelled-species-is-screened-by-label"),
        pytest.param({"carbon.psml": "C", "H.psml": "H"}, [],
                     1, [r"\[ERROR\] C\b.*carbon\.psml.*C\.psml"],
                     id="a-file-named-for-no-label-is-said-misnamed"),
        pytest.param({"C.psml": None, "H.psml": "H"}, [],
                     0, [r"\[WARN \] C\b.*could not parse"],
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

    #: The settings gate as rows: the calculation's functional, the library's
    #: files, the calculation folder's (`pseudos/`, or ``None`` for the library
    #: as the folder), the labels, and the findings by `where`, each
    #: (severity, words).
    GATE_ROWS = [
        # The SAME mismatch the CLI blocks on is error-severity in the
        # preflight -- the two surfaces agree.
        pytest.param("CA", {"C.psml": "C"}, None, ["C"],
                     {"config.psml_lib.C": ("error", ["LDA"])},
                     id="a-gga-pseudo-on-an-lda-calc-blocks"),
        # Labels read from two places -- C1 and H from the folder, C2 from the
        # library -- are judged together: the library's C1.psml is C1's, never
        # C2's to take (measured 2026-10-31 on gold: the second label was told
        # to save the element's pseudopotential from it).
        pytest.param("PBE", {"C1.psml": "C"},
                     {"C1.psml": "C", "H.psml": "H"}, ["C1", "C2", "H"],
                     {"config.psml_lib.C2": ("error",
                                             ["no .psml file for C2"])},
                     id="the-structure's-labels-are-judged-together"),
    ]

    @pytest.mark.parametrize("xc, library, folder, labels, expected", GATE_ROWS)
    def test_the_settings_gate_says_each_label_once(self, tmp_path, xc, library,
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
            xc_authors = xc

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
