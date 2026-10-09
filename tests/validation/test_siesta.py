"""Tests for molbuilder.validation.siesta.

Per docs/process/testing.md (test layout mirrors source
layout).  Shared fixtures live in tests/validation/conftest.py.
"""

from __future__ import annotations

from molbuilder.siesta import SiestaConfig
from molbuilder.validation import validate
from ._helpers import _peptide_struct


# --------------------------------------------------------------------- #
#  Production config metadata: ranges read off SiestaConfig / PySCFConfig#
#                                                                        #
#  This is what makes Principle #1 load-bearing rather than aspirational #
#  -- the validator picks up out-of-range values from the production    #
#  configs without any per-field plumbing in the validator itself.      #
# --------------------------------------------------------------------- #


def test_siesta_mesh_cutoff_below_range_warns(water_structure):
    """mesh_cutoff has a metadata range in Ry.  A value of 5 Ry must emit a
    config.mesh_cutoff warn, and it must name BOTH the meaning and the
    keyword — see the sweep below for why that is a rule and not a taste."""
    cfg = SiestaConfig(mesh_cutoff=5.0)
    issues = validate(water_structure, cfg)
    out_of_range = [i for i in issues if i.where == "config.mesh_cutoff"]
    assert len(out_of_range) == 1
    assert "MeshCutoff" in out_of_range[0].message
    assert "Real-space grid cutoff" in out_of_range[0].message
    assert "Ry" in out_of_range[0].message


def test_every_range_warning_names_the_engine_keyword(water_structure):
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
        # NON-SCALAR FIELDS ARE SKIPPED, and the skip has to be explicit:
        # ``dataclasses.replace`` does no type checking, so it would store
        # the scalar and the TypeError would surface later, inside
        # validate(), as a crash rather than a skip.
        #
        # They are genuinely out of scope for THIS rule: their range warning
        # names a component (``k-point mesh (%block kgrid_Monkhorst_Pack)[0]
        # = 641 ...``), which is a different
        # sentence shape from the ``label (KEYWORD) = value`` this test
        # governs.
        if isinstance(getattr(SiestaConfig(), f.name), (tuple, list)):
            continue
        cfg = dataclasses.replace(
            SiestaConfig(), **{f.name: type(lo)(hi) * 10 + 1})
        hits = [i for i in validate(water_structure, cfg)
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


def test_siesta_mesh_cutoff_in_range_no_warn(water_structure):
    cfg = SiestaConfig(mesh_cutoff=300.0)
    issues = validate(water_structure, cfg)
    assert [i for i in issues if i.where == "config.mesh_cutoff"] == []


def test_siesta_mesh_cutoff_below_production_floor_warns(water_structure):
    """2026-05-28: a value within the dataclass range but below the
    150 Ry production floor emits a SOFT WARN with a clear nudge.

    The slider lower bound is 100 Ry (the dataclass metadata-range
    check owns "below the slider floor").  100-149 Ry is the
    above-the-slider-floor-but-still-screening-grade window where
    the user benefits from a soft nudge.
    """
    cfg = SiestaConfig(mesh_cutoff=120.0)
    issues = validate(water_structure, cfg)
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


def test_siesta_mesh_cutoff_exactly_at_production_floor_no_warn(water_structure):
    """Boundary: 150 Ry is the threshold; >= 150 is silent."""
    cfg = SiestaConfig(mesh_cutoff=150.0)
    issues = validate(water_structure, cfg)
    assert [i for i in issues if i.where == "config.mesh_cutoff"] == []


def test_siesta_mesh_cutoff_at_slider_floor_warns_only_via_production_rule(
        water_structure):
    """Boundary: mc = 100 Ry is exactly the slider floor (lo of the
    dataclass metadata range, inclusive).  Below the production-
    defensible 150 Ry threshold → the production-floor rule warns.
    The metadata-range check is INCLUSIVE at lo, so it doesn't fire.
    Net: exactly one warn from the production rule.
    """
    cfg = SiestaConfig(mesh_cutoff=100.0)
    issues = validate(water_structure, cfg)
    mc_issues = [i for i in issues if i.where == "config.mesh_cutoff"]
    assert len(mc_issues) == 1
    # Must be the production-floor message (not the range-out message).
    assert "production" in mc_issues[0].message.lower(), (
        f"100 Ry is INSIDE the slider range; the warn here must be "
        f"the production-floor message, not the metadata-range one; "
        f"got: {mc_issues[0].message!r}"
    )


def test_siesta_charged_system_emits_makov_payne_notice(water_structure):
    """2026-05-28: a SIESTA calc with NetCharge != 0 emits a soft
    warn about the Makov-Payne image-charge bias.  The warn names
    the formula, the typical magnitude, AND that molbuilder does
    NOT auto-apply the correction.  Surfacing > implementing here.
    """
    cfg = SiestaConfig(net_charge=+1)
    issues = validate(water_structure, cfg)
    mp = [i for i in issues if i.where == "config.net_charge.makov_payne"]
    assert len(mp) == 1
    assert mp[0].severity == "warn"
    msg = mp[0].message
    # Quote the user's charge so the warn is self-explanatory.
    assert "+1" in msg
    # The warn quotes the computed numeric estimate at three
    # representative cell sizes.  Check that at least one of the bracket
    # values appears.
    assert any(s in msg for s in ("1.36", "1.02", "0.82"))
    # Mention the companion script the wrapper now emits.
    assert "makov_payne_correction.py" in msg
    # Cite the paper.
    assert "Makov" in msg


def test_siesta_neutral_system_no_makov_payne_notice(water_structure):
    """net_charge == 0 (default, or user-set explicitly) -- the
    image-charge artefact doesn't apply, no notice."""
    cfg = SiestaConfig()   # default net_charge None -> auto-detect 0
    issues = validate(water_structure, cfg)
    assert [i for i in issues
            if i.where == "config.net_charge.makov_payne"] == []


def test_siesta_negative_charge_also_emits_notice(water_structure):
    """The Makov-Payne bias scales as q^2 -- both signs of charge
    trigger it equally."""
    cfg = SiestaConfig(net_charge=-2)
    issues = validate(water_structure, cfg)
    mp = [i for i in issues if i.where == "config.net_charge.makov_payne"]
    assert len(mp) == 1
    # The warn quotes the actual charge value for context.
    assert "-2" in mp[0].message


def test_siesta_mesh_cutoff_below_slider_floor_emits_only_one_warning(
        water_structure):
    """Regression: a value below the SLIDER floor (100 Ry) must
    produce exactly ONE warning -- the dataclass metadata-range one,
    not double-counted with the production-floor rule, which
    _check_siesta_mesh_cutoff gates at >= 100 Ry.
    """
    cfg = SiestaConfig(mesh_cutoff=50.0)
    issues = validate(water_structure, cfg)
    mc_issues = [i for i in issues if i.where == "config.mesh_cutoff"]
    assert len(mc_issues) == 1, (
        f"expected exactly one mesh_cutoff issue; got "
        f"{len(mc_issues)} ({[i.message for i in mc_issues]})"
    )


def test_siesta_peptide_protonation_warn(water_structure):
    """Same hint fires under the SIESTA validator (it's an engine-
    independent property of the structure + charge config)."""
    s = _peptide_struct(["ALA","LYS","ASP","ASP","GLY"])  # +1 -2 = -1
    cfg = SiestaConfig(net_charge=None)   # auto -> 0 for non-nucleic
    issues = validate(s, cfg)
    msgs = [i for i in issues if i.where == "config.net_charge"]
    assert len(msgs) == 1
    assert "-1" in msgs[0].message


# --------------------------------------------------------------------- #
#  SIESTA pseudo-coverage check is wired into the preflight: every      #
#  validate(struct, SiestaConfig) call exercises the coverage check,    #
#  surfacing missing files / XC mismatches as preflight Issues.         #
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
        # The convention is the BARE name: "projects/pseudopotential"
        # cannot work, because the walk-up finds the tree and joining
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
        beats finding out after MPI init — and leaving it at WARN would
        let the pseudopotential source go unstated (user, 2026-09-19:
        *"the pseudopotential file has to be
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
        lengths).  ERROR per element so it BLOCKS emission."""
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
