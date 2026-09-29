"""Tests for molbuilder.validation.siesta.

Per docs/process/testing.md (test layout mirrors source
layout).  Split from the pre-2026-06-13 flat tests/test_validation.py
on 2026-06-13; no test body was modified.  Shared fixtures
(``water_struct``, ``_vacuum_cell``) live in tests/validation/conftest.py.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from molbuilder.issues import Issue, ValidationError
from molbuilder.pyscf import PySCFConfig
from molbuilder.siesta import SiestaConfig
from molbuilder.structure import Structure
from molbuilder.validation import report, validate
from ._helpers import _peptide_struct, _vacuum_cell


# --------------------------------------------------------------------- #
#  Production config metadata: ranges read off SiestaConfig / PySCFConfig#
#                                                                        #
#  This is what makes Principle #1 load-bearing rather than aspirational #
#  -- the validator picks up out-of-range values from the production    #
#  configs without any per-field plumbing in the validator itself.      #
# --------------------------------------------------------------------- #


def test_siesta_mesh_cutoff_below_range_warns(water_struct):
    """mesh_cutoff has a metadata range in Ry.  A value of 5 Ry must emit a
    config.mesh_cutoff warn, and it must name BOTH the meaning and the
    keyword — see the sweep below for why that is a rule and not a taste."""
    cfg = SiestaConfig(mesh_cutoff=5.0)
    issues = validate(water_struct, cfg)
    out_of_range = [i for i in issues if i.where == "config.mesh_cutoff"]
    assert len(out_of_range) == 1
    assert "MeshCutoff" in out_of_range[0].message
    assert "Real-space grid cutoff" in out_of_range[0].message
    assert "Ry" in out_of_range[0].message


def test_every_range_warning_names_the_engine_keyword(water_struct):
    """The rule, swept over every field rather than pinned on one.

    A warning has two jobs. *"Real-space grid cutoff is too low"* says what is
    wrong and cannot be found in the input file; *"MeshCutoff"* can be searched
    for and says nothing. It must carry both (user, 2026-08-15).

    **This is a regression guard.** The labels used to BE the keywords, and
    were replaced with prose on 2026-08-14 when the catalogue became the
    master — silently taking the keyword out of seventeen warnings, because
    the range warning is built from the label. One test noticed, on one field.
    This asks the whole schema.

    A field whose ``engine_key`` is a molbuilder note rather than a keyword is
    exempt and asserted to be exempt: there is no word to offer, and inventing
    one would make a search fail rather than merely not help.
    """
    import dataclasses
    from molbuilder.template import _bare_anchor

    missing, invented = [], []
    for f in dataclasses.fields(SiestaConfig):
        rng = f.metadata.get("range")
        if not rng:
            continue
        lo, hi = rng
        # NON-SCALAR FIELDS ARE SKIPPED, and the skip has to be explicit.
        # This asked ``dataclasses.replace`` to raise for them -- but
        # ``replace`` does no type checking, so it happily stored the scalar
        # and the TypeError surfaced later, inside validate(), as a crash
        # rather than a skip.  Nothing noticed until `kgrid` and
        # `kgrid_displacement` gained a per-component ``range`` on
        # 2026-08-15 and became the first ranged tuples in the schema.
        #
        # They are genuinely out of scope for THIS rule: their warnings come
        # from their own ``validate`` callables and name a component
        # (``kgrid[0] = 641 ...``), which is a different sentence shape from
        # the ``label (KEYWORD) = value`` this test governs.
        if isinstance(getattr(SiestaConfig(), f.name), (tuple, list)):
            continue
        cfg = dataclasses.replace(
            SiestaConfig(), **{f.name: type(lo)(hi) * 10 + 1})
        hits = [i for i in validate(water_struct, cfg)
                if i.where == f"config.{f.name}" and "outside" in i.message]
        if not hits:
            continue
        kw = _bare_anchor(str(f.metadata.get("engine_key", "") or ""))
        label = f.metadata.get("label", f.name)
        msg = hits[0].message
        # Matched against the EXACT opening, not by searching for a bracket:
        # ``mpi_np``'s own label is "MPI ranks (np)", so a field's label may
        # legitimately carry parentheses of its own and a substring test
        # cannot tell those from a keyword this code added.
        want = f"{label} ({kw}) = " if kw else f"{label} = "
        if not msg.startswith(want):
            (missing if kw else invented).append(
                f"{f.name}: {msg!r} does not open with {want!r}")

    assert not missing, ("range warning(s) with no keyword to search for:\n  "
                         + "\n  ".join(missing))
    assert not invented, ("range warning(s) citing a keyword the engine does "
                          "not have:\n  " + "\n  ".join(invented))


def test_siesta_mesh_cutoff_in_range_no_warn(water_struct):
    cfg = SiestaConfig(mesh_cutoff=300.0)
    issues = validate(water_struct, cfg)
    assert [i for i in issues if i.where == "config.mesh_cutoff"] == []


def test_siesta_mesh_cutoff_below_production_floor_warns(water_struct):
    """2026-05-28: a value within the dataclass range but below the
    150 Ry production floor emits a SOFT WARN with a clear nudge.

    The slider lower bound is 100 Ry (the dataclass metadata-range
    check owns "below the slider floor").  100-149 Ry is the
    above-the-slider-floor-but-still-screening-grade window where
    the user benefits from a soft nudge.
    """
    cfg = SiestaConfig(mesh_cutoff=120.0)
    issues = validate(water_struct, cfg)
    mc_issues = [i for i in issues if i.where == "config.mesh_cutoff"]
    assert len(mc_issues) == 1, (
        f"expected exactly one mesh_cutoff issue; got "
        f"{[i.message for i in mc_issues]}"
    )
    assert mc_issues[0].severity == "warn"
    # Message must name the actual value AND the production floor
    # so the user knows what to set instead.
    assert "120" in mc_issues[0].message
    assert "150" in mc_issues[0].message
    assert ("200-300" in mc_issues[0].message
            or "production" in mc_issues[0].message), (
        f"message must mention production levels; got: "
        f"{mc_issues[0].message!r}"
    )


def test_siesta_mesh_cutoff_exactly_at_production_floor_no_warn(water_struct):
    """Boundary: 150 Ry is the threshold; >= 150 is silent."""
    cfg = SiestaConfig(mesh_cutoff=150.0)
    issues = validate(water_struct, cfg)
    assert [i for i in issues if i.where == "config.mesh_cutoff"] == []


def test_siesta_mesh_cutoff_at_slider_floor_warns_only_via_production_rule(
        water_struct):
    """Boundary: mc = 100 Ry is exactly the slider floor (lo of the
    dataclass metadata range, inclusive).  Below the production-
    defensible 150 Ry threshold → the production-floor rule warns.
    The metadata-range check is INCLUSIVE at lo, so it doesn't fire.
    Net: exactly one warn from the production rule.
    """
    cfg = SiestaConfig(mesh_cutoff=100.0)
    issues = validate(water_struct, cfg)
    mc_issues = [i for i in issues if i.where == "config.mesh_cutoff"]
    assert len(mc_issues) == 1
    # Must be the production-floor message (not the range-out message).
    assert "production" in mc_issues[0].message.lower(), (
        f"100 Ry is INSIDE the slider range; the warn here must be "
        f"the production-floor message, not the metadata-range one; "
        f"got: {mc_issues[0].message!r}"
    )


def test_siesta_charged_system_emits_makov_payne_notice(water_struct):
    """2026-05-28: a SIESTA calc with NetCharge != 0 emits a soft
    warn about the Makov-Payne image-charge bias.  The warn names
    the formula, the typical magnitude, AND that molbuilder does
    NOT auto-apply the correction.  Surfacing > implementing here.
    """
    cfg = SiestaConfig(net_charge=+1)
    issues = validate(water_struct, cfg)
    mp = [i for i in issues if i.where == "config.net_charge.makov_payne"]
    assert len(mp) == 1
    assert mp[0].severity == "warn"
    msg = mp[0].message
    # Quote the user's charge so the warn is self-explanatory.
    assert "+1" in msg
    # Phase-after-Phase-6: the warn now quotes the computed numeric
    # estimate at three representative cell sizes (post-#172) instead
    # of the qualitative "0.5-1.5 eV" range.  Check that at least one
    # of the bracket values appears.
    assert any(s in msg for s in ("1.36", "1.02", "0.82"))
    # Mention the companion script the wrapper now emits.
    assert "makov_payne_correction.py" in msg
    # Cite the paper.
    assert "Makov" in msg


def test_siesta_neutral_system_no_makov_payne_notice(water_struct):
    """net_charge == 0 (default, or user-set explicitly) -- the
    image-charge artefact doesn't apply, no notice."""
    cfg = SiestaConfig()   # default net_charge None -> auto-detect 0
    issues = validate(water_struct, cfg)
    assert [i for i in issues
            if i.where == "config.net_charge.makov_payne"] == []


def test_siesta_negative_charge_also_emits_notice(water_struct):
    """The Makov-Payne bias scales as q^2 -- both signs of charge
    trigger it equally."""
    cfg = SiestaConfig(net_charge=-2)
    issues = validate(water_struct, cfg)
    mp = [i for i in issues if i.where == "config.net_charge.makov_payne"]
    assert len(mp) == 1
    # The warn quotes the actual charge value for context.
    assert "-2" in mp[0].message


def test_siesta_mesh_cutoff_below_slider_floor_emits_only_one_warning(
        water_struct):
    """Regression: a value below the SLIDER floor (100 Ry) must
    produce exactly ONE warning -- the dataclass metadata-range one,
    not double-counted with the production-floor rule.

    Before the gate at >= 100 Ry in _check_siesta_mesh_cutoff, the
    same field would generate two warns and existing tests counting
    by ``where`` would fail.
    """
    cfg = SiestaConfig(mesh_cutoff=50.0)
    issues = validate(water_struct, cfg)
    mc_issues = [i for i in issues if i.where == "config.mesh_cutoff"]
    assert len(mc_issues) == 1, (
        f"expected exactly one mesh_cutoff issue; got "
        f"{len(mc_issues)} ({[i.message for i in mc_issues]})"
    )


def test_siesta_peptide_protonation_warn(water_struct):
    """Same hint fires under the SIESTA validator (it's an engine-
    independent property of the structure + charge config)."""
    s = _peptide_struct(["ALA","LYS","ASP","ASP","GLY"])  # +1 -2 = -1
    cfg = SiestaConfig(net_charge=None)   # auto -> 0 for non-nucleic
    issues = validate(s, cfg)
    msgs = [i for i in issues if i.where == "config.net_charge"]
    assert len(msgs) == 1
    assert "-1" in msgs[0].message


# --------------------------------------------------------------------- #
#  SIESTA pseudo-coverage check is wired into the preflight             #
#  (2026-05-23: previously the pseudos.py coverage check was not        #
#  exercised by the Build->Generate flow.  These tests pin that every   #
#  validate(struct, SiestaConfig) call now exercises the coverage       #
#  check, surfacing missing files / XC mismatches as preflight Issues.) #
# --------------------------------------------------------------------- #


def _make_pdojo_psml(element: str, *, z: int = None,
                     libxc_id_x: int = 101,  # PBE exchange
                     libxc_id_c: int = 130   # PBE correlation
                     ) -> str:
    """Real-PseudoDojo-shape PSML (pseudo-atom-spec + nested
    <functional> for libxc)."""
    if z is None:
        from ase.data import atomic_numbers as _Z
        z = _Z[element]
    return f"""<?xml version="1.0" encoding="UTF-8" ?>
<psml version="1.1" xmlns="http://esl.cecam.org/PSML/ns/1.1">
<provenance creator="test"/>
<pseudo-atom-spec atomic-label="{element}" atomic-number="{z}"
 z-pseudo="{z}" relativity="scalar">
<exchange-correlation>
<libxc-info number-of-functionals="2">
<functional name="x" type="exchange" id="{libxc_id_x}"/>
<functional name="c" type="correlation" id="{libxc_id_c}"/>
</libxc-info>
</exchange-correlation>
</pseudo-atom-spec>
</psml>"""


class TestSiestaPseudoCoverageInPreflight:
    """Pin the actual wiring: validate(struct, SiestaConfig) MUST
    surface pseudo-coverage findings as Issues."""

    def _water(self):
        from molbuilder.structure import Structure
        import numpy as np
        return Structure(elements=["O", "H", "H"],
                         positions=np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]]))

    def test_psml_lib_unset_emits_actionable_warn(self):
        """Default config (psml_lib=None), and NO calculation folder yet ->
        WARN telling the user how to get pseudos.  Without this warning the
        user discovers the missing-pseudo failure only after a 5-minute
        mpirun start-up.

        WARN and not ERROR is about what is knowable HERE: this runs in the
        Build tab before a folder exists, and pseudos already beside the
        calculation are used without this field (`job-contracts.md`
        § 2.5a) -- so whether the folder will supply them cannot be
        answered yet.  With a folder in hand it can, and the test below is
        that case.
        """
        from molbuilder.config.siesta import SiestaConfig
        from molbuilder.validation import validate
        issues = validate(self._water(), SiestaConfig())
        psml_issues = [i for i in issues if i.where == "config.psml_lib"]
        assert psml_issues
        assert psml_issues[0].severity == "warn"
        assert "psml_lib is not set" in psml_issues[0].message
        # Actionable: mentions PseudoDojo + the projects/pseudopotential/
        # convention.
        assert "pseudo-dojo.org" in psml_issues[0].message
        # The convention is the BARE name.  This asserted
        # "projects/pseudopotential" until 2026-08-21 -- the one spelling
        # that cannot work, because the walk-up finds the tree and joining
        # its name again nests it inside itself (job-contracts.md 2.5a).
        msg = psml_issues[0].message
        assert "`pseudopotential`" in msg
        assert "do NOT write the projects/ prefix" in msg

    def test_where_the_pseudos_come_from_is_stated_or_the_folder_has_them(
            self, tmp_path):
        """With a calculation folder in hand, silence is answerable — and
        the answer decides the severity.

        There are two sources and no third: `psml_lib`, or files already
        beside the calculation (`job-contracts.md` § 2.5a).  When neither
        exists the run cannot start, so saying so at configuration time
        beats finding out after MPI init — and leaving it at WARN is the
        one place this path still let the pseudopotential source go
        unstated (user, 2026-09-19: *"the pseudopotential file has to be
        explicit and it has to be strictly checked at the configuration
        time"*).

        The severity rule is the one the missing-directory branch beside it
        already uses: a folder makes it answerable, no folder does not.
        """
        import sys as _sys
        _sys.path.insert(0, "tests")
        from conftest import _PSML_FIXTURE
        from molbuilder.chemistry import atomic_number
        from molbuilder.config.siesta import SiestaConfig
        from molbuilder.validation.siesta import (
            _check_siesta_pseudo_coverage as _check)

        water, cfg = self._water(), SiestaConfig()      # psml_lib unset

        empty = tmp_path / "nothing"
        empty.mkdir()
        got = _check(water, cfg, dest_dir=empty)
        assert [i.severity for i in got] == ["error"], (
            "neither source exists, so the run cannot start")

        supplied = tmp_path / "beside"
        supplied.mkdir()
        for el in ("O", "H"):
            (supplied / f"{el}.psml").write_text(
                _PSML_FIXTURE.format(el=el, z=atomic_number(el)))
        assert _check(water, cfg, dest_dir=supplied) == [], (
            "the folder supplies them, so an unset psml_lib is correct")

        half = tmp_path / "partial"
        half.mkdir()
        (half / "H.psml").write_text(_PSML_FIXTURE.format(el="H", z=1))
        assert [i.severity for i in _check(water, cfg, dest_dir=half)] == [
            "error"], "covering SOME elements is not covering them"

    def test_psml_lib_bad_path_emits_error(self):
        from molbuilder.config.siesta import SiestaConfig
        from molbuilder.validation import validate
        issues = validate(self._water(),
                           SiestaConfig(psml_lib="/no/such/dir"))
        errs = [i for i in issues if i.where == "config.psml_lib"
                and i.severity == "error"]
        assert errs
        assert "/no/such/dir" in errs[0].message

    def test_complete_coverage_passes(self, tmp_path, monkeypatch):
        """All elements present + matching XC -> NO psml-related issues
        (other checks may fire; we filter to just the psml ones)."""
        for el in ("O", "H"):
            (tmp_path / f"{el}.psml").write_text(_make_pdojo_psml(el))
        from molbuilder.config.siesta import SiestaConfig
        from molbuilder.projects import PROJECTS_ROOT_ENV
        from molbuilder.validation import validate
        monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tmp_path))
        issues = validate(self._water(),
                           SiestaConfig(psml_lib=str(tmp_path), xc_authors="PBE"))
        psml_issues = [i for i in issues if "psml" in i.where.lower()]
        assert psml_issues == []

    def test_missing_element_emits_error(self, tmp_path, monkeypatch):
        """Only O.psml is present; H is missing -> ERROR Issue."""
        (tmp_path / "O.psml").write_text(_make_pdojo_psml("O"))
        from molbuilder.config.siesta import SiestaConfig
        from molbuilder.projects import PROJECTS_ROOT_ENV
        from molbuilder.validation import validate
        monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tmp_path))
        issues = validate(self._water(),
                           SiestaConfig(psml_lib=str(tmp_path)))
        h_issues = [i for i in issues
                    if i.where == "config.psml_lib.H"]
        assert h_issues
        assert h_issues[0].severity == "error"
        assert "no .psml file for H" in h_issues[0].message

    def test_xc_family_mismatch_emits_error(self, tmp_path, monkeypatch):
        """Pseudos are LDA but the calc requests PBE -> an XC-FAMILY
        mismatch, which is never physically correct (silently wrong bond
        lengths).  ERROR per element (upgraded from WARN in the 2026-07
        scientific-correctness audit) so it BLOCKS emission."""
        # libxc id 1 = XC_LDA_X
        for el in ("O", "H"):
            (tmp_path / f"{el}.psml").write_text(
                _make_pdojo_psml(el, libxc_id_x=1, libxc_id_c=9))
        from molbuilder.config.siesta import SiestaConfig
        from molbuilder.projects import PROJECTS_ROOT_ENV
        from molbuilder.validation import validate
        monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tmp_path))
        issues = validate(self._water(),
                           SiestaConfig(psml_lib=str(tmp_path),
                                          xc_authors="PBE"))
        mismatch_issues = [i for i in issues
                           if "psml" in i.where.lower()
                           and i.severity == "error"
                           and "LDA" in i.message]
        assert len(mismatch_issues) == 2   # O + H both flagged, blocking
        assert all("silently wrong" in i.message
                   for i in mismatch_issues)


# --------------------------------------------------------------------- #
#  Deck keyword CURRENCY -- we must not write options SIESTA retired.   #
# --------------------------------------------------------------------- #

#: Every keyword the SIESTA 5.4.2 manual formally retires, with what replaced
#: it.  Not prose-derived: the manual marks each one with ``\fdfdeprecates``,
#: and this table is that markup, read out of the manual source at tag 5.4.2
#: (``Docs/tex/sections/**.tex``; ``!`` in the markup means a dotted prefix).
#:
#: Carried as DATA rather than parsed at test time on purpose — the manual
#: lives in an optional source checkout, and a test that silently skips when
#: it is absent is a test that never runs. To refresh after a SIESTA bump:
#:
#:     grep -rhoP '\\fdfdeprecates\{[^}]+\}' <siesta>/Docs/tex | ...
SIESTA_542_DEPRECATED = {
    "MD.NumCGsteps":   "MD.Steps",
    "MD.MaxCGDispl":   "MD.MaxDispl",
    "DM.MixingWeight": "SCF.Mixer.Weight",
    "DM.NumberPulay":  "SCF.Mixer.History",
    "DM.NumberBroyden": "SCF.Mixer.History",
    "DM.MixSCF1":      "SCF.Mix.Spin",
    "MD.TargetPressure": "Target.Pressure",
    "MD.TargetStress": "Target.Stress.Voigt",
    "MD.FCDispl":      "FC.Displacement",
    "MD.FCFirst":      "FC.First",
    "MD.FCLast":       "FC.Last",
    "UseNewDiagk":     "Diag.WFS.Cache",
    "WriteMullikenPop": "Charge.Mulliken",
    "Write.HirshfeldPop": "Charge.Hirshfeld",
    "Write.VoronoiPop": "Charge.Voronoi",
    "Diag.DivideAndConquer": "Diag.Algorithm",
    "Diag.MRRR":       "Diag.Algorithm",
    "Diag.ELPA":       "Diag.Algorithm",
    "Diag.NoExpert":   "Diag.Algorithm",
    "SpinPolarized":   "Spin",
    "NonCollinearSpin": "Spin",
    "SpinOrbit":       "Spin",
}

#: Empty, and that is the point.  ``SpinPolarized`` sat here between the two
#: halves of this migration: ``Spin`` is not a rename of it but a consolidation
#: of THREE booleans into one four-valued enum, so it needed a type change in
#: the template rather than a sweep.  That landed 2026-08-15 and the exception
#: went with it.  An entry here is a debt, not a policy.
KNOWN_DEPRECATED_STILL_EMITTED: set = set()


def test_no_deprecated_siesta_keyword_reaches_the_deck(water_struct):
    """We must not write options the engine has retired.

    Five were being written until 2026-08-15 — ``MD.NumCGsteps``,
    ``MD.MaxCGDispl``, ``DM.MixingWeight``, ``DM.NumberPulay`` and
    ``SpinPolarized`` — and none of them was noticed by reading the code,
    because deprecation is a fact about the MANUAL, not about the source. The
    code accepts them happily; the manual is where they are marked retired.

    Swept across optimiser types and the switches that open conditional
    blocks, because a deprecated keyword can hide behind a branch.
    """
    import dataclasses

    # This import went missing in an earlier cleanup and the swallow
    # below ate the resulting NameError FIFTEEN times per run -- the
    # deprecated-keyword assert passed over an empty set until the
    # non-empty guard (U6 close) said so.
    from molbuilder.siesta import render_fdf
    seen = set()
    # The catalogue's own spellings: since 2026-09-28 a value outside an
    # enum's choices is refused by name, and the lower-case forms this swept
    # were refused on every combination -- which the guard below caught.
    for relax in ("CG", "Broyden", "FIRE", "Verlet", "Nose"):
        # The spin opens its branch with a PINNED count -- the pair of
        # keywords (`Spin.Fix` + `Spin.Total`) is where the retired
        # `SpinPolarized` flag would hide.
        for extra in ({}, {"spin_treatment": "unrestricted",
                           "unpaired_electrons": 2}, {"use_gpu": True}):
            try:
                cfg = dataclasses.replace(SiestaConfig(), relax_type=relax, **extra)
                deck = render_fdf(water_struct, cfg)
            except Exception:
                continue                     # a combination this build refuses
            for line in deck.splitlines():
                line = line.strip()
                if line and not line.startswith(("#", "%")):
                    seen.add(line.split()[0])

    # If EVERY combination refused, `seen` is empty and the vocabulary
    # assert below would pass over nothing -- a broken render_fdf would
    # read as a clean deck set.
    assert seen, "no combination rendered at all; the sweep saw nothing"
    bad = sorted((seen & set(SIESTA_542_DEPRECATED)) - KNOWN_DEPRECATED_STILL_EMITTED)
    assert not bad, (
        "deck writes keyword(s) SIESTA 5.4.2 deprecates:\n  "
        + "\n  ".join(f"{k} -> use {SIESTA_542_DEPRECATED[k]}" for k in bad))


def test_the_catalogue_declares_no_deprecated_keyword():
    """Same rule one level up — at the SOURCE rather than at the output.

    The deck is generated; the catalogue is authored. A deprecated keyword
    that reaches a deck got there because an item declares it, so this is the
    check that names the item to fix rather than the line to grep for.
    """
    from molbuilder import template as T
    cat = T.read_template(T.load_catalogue())
    bad = []
    for it in T.select(cat, engine="siesta", kind=("engine", "deck")):
        for kw in (list(it.expands) or ([it.anchor] if it.anchor else [])):
            if kw in SIESTA_542_DEPRECATED and kw not in KNOWN_DEPRECATED_STILL_EMITTED:
                bad.append(f"{it.name} declares {kw} -> use "
                           f"{SIESTA_542_DEPRECATED[kw]}")
    assert not bad, "catalogue declares deprecated keyword(s):\n  " + "\n  ".join(bad)
