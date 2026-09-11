"""Methods-section composer tests: citation-key extraction +
``render_methods_md`` across the pre-run / post-run / engine-fragment
forms.

`web/spectra.md` § 9a.2.  These tests verify the Methods prose
that ships embedded in the emitted script and in the post-run output:

  * citation keys parse cleanly from the prose's [Foo, Bar] markers;
  * the pre-run renderer emits the configured method/basis/dispersion
    text + the structure-conditional phrasings ("5 fixed Au atoms");
  * the post-run renderer substitutes actual numbers from a parsed
    SpectraResults;
  * the engine-fragment composition uses every engine's own
    ``methods_fragment()`` hook.

No PySCF / SCF anywhere -- prose composition only.
"""

from __future__ import annotations

import numpy as np
import pytest


from tests.spectra._helpers import _make_mode, _make_results, _spectra_cfg


# --------------------------------------------------------------------- #
#  L2 Methods composer (methods.py)                                     #
#                                                                       #
#  archived-spec § 11.2 + § 9.4.  Pure prose generation; no engine I/O.          #
# --------------------------------------------------------------------- #


class TestExtractCitationKeys:
    """The bibliography-extractor underlies both the trailing
    bibliography in render_methods_md and the
    SpectraResults.bibliography_keys field (archived-spec § 5).  Test it
    standalone so its semantics are pinned independently of the
    composer's prose choices."""

    def test_basic_extraction(self):
        from molbuilder.spectra import extract_citation_keys
        text = "We cite [Sun2020] and also [Becke1993]."
        assert extract_citation_keys(text) == ["Sun2020", "Becke1993"]

    def test_section_suffix_stripped(self):
        """`[Key §section]` patterns -- the §-clause is prose, the
        key alone is what resolves against references.bib."""
        from molbuilder.spectra import extract_citation_keys
        text = "anharmonic-cubic mixing < 1% [Mills1972 §2.4]"
        assert extract_citation_keys(text) == ["Mills1972"]

    def test_deduplication_preserves_first_occurrence(self):
        from molbuilder.spectra import extract_citation_keys
        text = "[Sun2020] then [Becke1993] then [Sun2020] again."
        assert extract_citation_keys(text) == ["Sun2020", "Becke1993"]

    def test_empty_or_no_citations(self):
        from molbuilder.spectra import extract_citation_keys
        assert extract_citation_keys("") == []
        assert extract_citation_keys("no citations here") == []
        # The regex requires the first char to be a letter -- digit-
        # leading "keys" don't match (BibTeX style: AuthorYYYY).
        assert extract_citation_keys("[123numeric] [9Sun2020]") == []
        # A purely alphabetic bracket-word like [array] DOES look like
        # a citation key structurally and will be extracted; that's
        # accepted as the cost of a permissive author key pattern --
        # the references.bib linter (archived-spec § 11.3) catches the false
        # positive at release-tag time.
        assert extract_citation_keys("an [array] of words") == ["array"]

    def test_underscores_allowed_in_keys(self):
        """BibTeX keys can contain underscores -- e.g. `Sun_2020`."""
        from molbuilder.spectra import extract_citation_keys
        assert extract_citation_keys("[Foo_Bar2020]") == ["Foo_Bar2020"]

    def test_separate_brackets(self):
        """Each `[Key]` bracket pair contributes its key."""
        from molbuilder.spectra import extract_citation_keys
        assert extract_citation_keys("[Galperin2007] [Frederiksen2007]") \
            == ["Galperin2007", "Frederiksen2007"]

    def test_comma_separated_keys_in_one_bracket_split(self):
        """`[Foo, Bar]` is common physics/chem prose style.  Each
        comma-separated key contributes its key, preserving
        first-appearance order."""
        from molbuilder.spectra import extract_citation_keys
        assert extract_citation_keys("PySCF [Sun2020, Sun2018] is widely used.") \
            == ["Sun2020", "Sun2018"]

    def test_comma_separated_keys_dedupe_against_earlier(self):
        from molbuilder.spectra import extract_citation_keys
        assert extract_citation_keys(
            "[Sun2020] then [Sun2020, Sun2018]"
        ) == ["Sun2020", "Sun2018"]

    def test_three_comma_separated_keys(self):
        from molbuilder.spectra import extract_citation_keys
        assert extract_citation_keys("[A2020, B2021, C2022]") == ["A2020", "B2021", "C2022"]


class TestRenderMethodsMdPreRun:
    """Pre-run path (`results=None`): the prose describes what
    *will* be done with the configured knobs.  Used by the
    Methods-preview modal (archived-spec § 9.4) before the user runs the
    script."""

    def test_minimal_config_produces_paragraph(self):
        """Default config (selector=none, no ES) -> single
        L2 paragraph with functional + basis + dispersion mentions."""
        from molbuilder.spectra import render_methods_md
        cfg = _spectra_cfg()
        md = render_methods_md(cfg)
        assert "## Methods" in md
        # Default level: B3LYP / def2-SVP / D3BJ.
        assert "B3LYP" in md
        assert "def2-SVP" in md
        assert "D3BJ" in md or "d3bj" in md.lower()
        # selector=none -> NO per-mode-ES paragraph.
        assert "per-mode electronic" not in md.lower()

    def test_dispersion_none_omits_dispersion_clause(self):
        from molbuilder.spectra import render_methods_md
        cfg = _spectra_cfg(dispersion="none")
        md = render_methods_md(cfg)
        assert "dispersion" not in md.lower()

    def test_compute_raman_false_omits_raman_prose(self):
        """diagnostic / Hessian-only run -> no dα/dR clause."""
        from molbuilder.spectra import render_methods_md
        cfg = _spectra_cfg(compute_raman=False)
        md = render_methods_md(cfg)
        assert "Raman activities" not in md
        assert "Komornicki1979" not in md

    def test_compute_raman_true_cites_komornicki_and_wilson(self):
        from molbuilder.spectra import render_methods_md
        cfg = _spectra_cfg(compute_raman=True)
        md = render_methods_md(cfg)
        assert "Komornicki1979" in md
        assert "Wilson1955" in md

    def test_selector_all_emits_es_paragraph(self):
        from molbuilder.spectra import render_methods_md
        cfg = _spectra_cfg(es_mode_selection="all")
        md = render_methods_md(cfg)
        assert "every vibrational mode" in md
        assert "Galperin2007" in md
        assert "Frederiksen2007" in md
        assert "Mills1972" in md
        # Default amplitude 0.02 Å should appear (lowered from 0.10
        # to 0.02 on 2026-05-19).
        assert "0.02" in md
        assert "A = 0.02" in md or "A=0.02" in md or "0.02 Å" in md

    def test_selector_top_n_named_in_prose(self):
        from molbuilder.spectra import render_methods_md
        cfg = _spectra_cfg(es_mode_selection="top_n", es_top_n=7)
        md = render_methods_md(cfg)
        assert "top 7" in md

    def test_selector_threshold_named_in_prose(self):
        from molbuilder.spectra import render_methods_md
        cfg = _spectra_cfg(es_mode_selection="threshold", es_threshold=2.5)
        md = render_methods_md(cfg)
        assert "Raman activity > 2.5" in md

    def test_selector_explicit_states_count(self):
        from molbuilder.spectra import render_methods_md
        cfg = _spectra_cfg(es_mode_selection="explicit",
                            es_explicit_indices=[3, 5, 8, 12])
        md = render_methods_md(cfg)
        assert "user-specified set of 4 modes" in md

    def test_frequency_window_clause_both_bounds(self):
        from molbuilder.spectra import render_methods_md
        cfg = _spectra_cfg(es_mode_selection="all",
                            freq_min_cm1=500.0, freq_max_cm1=2000.0)
        md = render_methods_md(cfg)
        assert "500" in md and "2000" in md
        assert "cm⁻¹" in md

    def test_frequency_window_one_sided(self):
        from molbuilder.spectra import render_methods_md
        cfg = _spectra_cfg(es_mode_selection="all", freq_min_cm1=1500.0)
        md = render_methods_md(cfg)
        assert "≥ 1500" in md or ">= 1500" in md

    def test_frequency_window_ignored_for_explicit(self):
        """selector=explicit ignores the freq window (archived-spec § 8.1);
        the prose shouldn't claim a window restriction that won't
        actually be enforced."""
        from molbuilder.spectra import render_methods_md
        cfg = _spectra_cfg(es_mode_selection="explicit",
                            es_explicit_indices=[1, 2],
                            freq_min_cm1=1000.0,
                            freq_max_cm1=2000.0)
        md = render_methods_md(cfg)
        # No "within 1000-2000 cm⁻¹" clause -- the window doesn't
        # apply to explicit selections.
        assert "1000-2000" not in md
        assert "within the 1000" not in md

    def test_non_b3_functional_omits_becke_citation(self):
        """Becke1993 is the B3-family paper; cite it only when the
        functional is in that family."""
        from molbuilder.spectra import render_methods_md
        cfg = _spectra_cfg(functional="PBE0")
        md = render_methods_md(cfg)
        assert "Becke1993" not in md

    def test_bibliography_listed_at_end(self):
        from molbuilder.spectra import render_methods_md
        cfg = _spectra_cfg(es_mode_selection="all")
        md = render_methods_md(cfg)
        assert "**Bibliography**" in md
        # The bibliography section appears AFTER the prose
        # (so a reader scrolling top-to-bottom hits the keys last).
        bib_pos = md.index("**Bibliography**")
        # All inline citations precede the bibliography.
        first_cite = md.index("[")
        assert first_cite < bib_pos


class TestRenderMethodsMdPostRun:
    """Post-run path (`results` provided): real numbers from the
    parsed SpectraResults replace pre-run placeholders.  Used to
    populate SpectraResults.methods_text (archived-spec § 5) -- the same
    prose lands in the JSON for downstream consumers."""

    def test_frequency_span_appended_when_modes_present(self):
        from molbuilder.spectra import render_methods_md
        cfg = _spectra_cfg()
        results = _make_results(complete=True)
        md = render_methods_md(cfg, results=results)
        # Real frequencies from _make_results: 412.3, 1023.4, 3656.0
        assert "3 modes" in md
        assert "412" in md
        assert "3656" in md

    def test_imaginary_modes_called_out(self):
        from molbuilder.spectra import render_methods_md
        cfg = _spectra_cfg()
        results = _make_results(complete=True)
        # Inject an imaginary mode.
        results.modes.append(_make_mode(index=4, freq=-150.0, with_es=False))
        md = render_methods_md(cfg, results=results)
        assert "imaginary" in md

    def test_selected_modes_line_post_run(self):
        """When ES data is present in results, the post-run prose
        ends with a "Selected modes: ..." line listing the indices
        + frequencies (archived-spec § 11.2)."""
        from molbuilder.spectra import render_methods_md
        cfg = _spectra_cfg(es_mode_selection="explicit",
                            es_explicit_indices=[2])
        results = _make_results(complete=True)
        md = render_methods_md(cfg, results=results)
        assert "Selected modes" in md
        # _make_results gives mode 2 ES at 1023.4 cm⁻¹.
        assert "mode 2" in md
        assert "1023" in md

    def test_es_count_appended_to_l4_paragraph(self):
        """The L4 paragraph gains "In the present run X modes
        received per-mode electronic-structure data." when results
        exist."""
        from molbuilder.spectra import render_methods_md
        cfg = _spectra_cfg(es_mode_selection="all")
        results = _make_results(complete=True)
        md = render_methods_md(cfg, results=results)
        assert "1 modes received" in md or "In the present run" in md


class TestRenderMethodsMdFragment:
    """The composer is engine-IGNORANT: the caller supplies the
    engine-specific paragraph as TEXT (``fragment_md``), and the
    composer interleaves it between the generic paragraphs.  P3
    retired the engine-class hook: the registry it looked up died
    with the old generator, and the one remaining producer (the
    vibration deck) knows its own engine -- it passes
    :func:`molbuilder.pyscf.vibration_emitters.pyscf_methods_fragment`.
    A raising callable cannot exist any more: text does not raise.
    Citation keys from the fragment flow into the trailing
    bibliography just like the generic prose's keys."""

    def test_fragment_appears_in_output(self):
        from molbuilder.spectra import render_methods_md
        md = render_methods_md(_spectra_cfg(), fragment_md=(
            "The analytic Hessian was obtained via "
            "`pyscf.hessian.rks` [Sun2020]."))
        assert "pyscf.hessian.rks" in md

    def test_fragment_citations_join_bibliography(self):
        """A citation key that appears only in the fragment must
        still land in the trailing **Bibliography** list."""
        from molbuilder.spectra import render_methods_md
        md = render_methods_md(_spectra_cfg(), fragment_md=(
            "Custom citation only here: [Sun2018]."))
        bib_section = md.split("**Bibliography**", 1)[1]
        assert "Sun2018" in bib_section

    def test_empty_fragment_is_omitted_whole(self):
        """No fragment means no engine paragraph and no placeholder --
        the default, and what a stripped test environment sees."""
        from molbuilder.spectra import render_methods_md
        md = render_methods_md(_spectra_cfg())
        assert "pyscf.hessian" not in md

    def test_the_deck_s_real_fragment_composes(self):
        """The one production caller's fragment: PySCF named with its
        package citations, the Hessian module matched to the method
        class, Raman prose riding only when compute_raman is on."""
        from molbuilder.spectra import render_methods_md
        from molbuilder.pyscf.vibration_emitters import pyscf_methods_fragment
        cfg = _spectra_cfg(method="UKS", compute_raman=True)
        md = render_methods_md(cfg, fragment_md=pyscf_methods_fragment(cfg))
        assert "pyscf.hessian.uks" in md
        assert "[Sun2020, Sun2018]" in md
        assert "Komornicki1979" in md
        bib = md.split("**Bibliography**", 1)[1]
        assert "Sun2018" in bib and "Komornicki1979" in bib

class TestRenderMethodsMdWithStruct:
    """Atom-count phrasing is gated on Structure availability."""

    def test_struct_none_omits_atom_clause(self):
        from molbuilder.spectra import render_methods_md
        md = render_methods_md(_spectra_cfg())
        assert "free, " not in md  # no "(N free, M held fixed)" clause
        assert "vibrational modes" not in md or "vibrational mode" in md
        # When struct is None, the L2 paragraph has no atom counts.

    def test_struct_provided_emits_atom_clause(self):
        """When a Structure is provided, the prose names total
        atoms, free atoms, and the 3N-6 mode count."""
        from molbuilder.spectra import render_methods_md

        class _Atom:
            def __init__(self, sym):
                self.symbol = sym
        # 5-atom water-cluster mock; no atoms fixed.
        atoms = [_Atom("O"), _Atom("H"), _Atom("H"), _Atom("O"), _Atom("H")]

        class _S:
            pass
        struct = _S()
        struct.atoms = atoms

        md = render_methods_md(_spectra_cfg(), struct=struct)
        assert "5 atoms" in md
        # 3*5 - 6 = 9 modes for all-free.
        assert "9 non-translational" in md or "9 " in md

    def test_struct_with_frozen_atoms_counts_correctly(self):
        """Frozen atoms subtract from n_free.  Named by INDEX -- the
        region store holds indices and the deck writes those into
        geomeTRIC's constraints file.

        Uses a real `Structure`.  This built a `_S` stub exposing
        `.atoms` of objects with `.symbol`, and `methods.py` carried a
        `_structure_element_symbols` adapter to read either shape --
        production code written for a test mock.  Both are gone
        (2026-08-22): there is one structure type.
        """
        import numpy as np

        from molbuilder.spectra import render_methods_md
        from molbuilder.structure import Structure

        # 4 Au + 3 organic = 7 atoms; freeze the four Au -> n_free = 3.
        struct = Structure(
            elements=["Au", "Au", "Au", "Au", "C", "H", "H"],
            positions=np.array([[float(i), 0.0, 0.0] for i in range(7)]))
        cfg = _spectra_cfg(struct=struct, frozen_indices=[0, 1, 2, 3])
        md = render_methods_md(cfg, struct=struct)
        assert "3 free" in md
        assert "4 frozen" in md

    def test_real_structure_dataclass_works(self):
        """A real molbuilder.Structure (elements as List[str], not
        list-of-atom-objects) should feed atom counts correctly --
        regression test against the mock-only earlier version."""
        from molbuilder.spectra import render_methods_md
        from molbuilder.structure import Structure
        struct = Structure(
            elements  = ["O", "H", "H"],
            positions = np.array([[0., 0., 0.],
                                  [0.96, 0., 0.],
                                  [-0.24, 0.93, 0.]]),
        )
        md = render_methods_md(_spectra_cfg(), struct=struct)
        assert "3 atoms" in md
        # 3*3 - 6 = 3 modes for water.
        assert "3 non-translational" in md

    def test_a_linear_molecule_gets_3N_MINUS_5(self):
        """CO2 has FOUR vibrations, not three.

        A linear molecule has two rotational degrees of freedom, not three --
        spinning it about its own axis moves nothing -- so 3N-5 vibrations
        survive the projection, not 3N-6. The paragraph said 3N-6 for
        everything until 2026-09-11: for the CO2 run that produced this
        finding it claimed 3 modes while the run's own `spectra.json` listed
        4, and for any diatomic it claimed 0 while the run listed 1.

        The COMPUTATION was never wrong -- PySCF's `harmonic_analysis`
        projects "the 6 (or 5 for linear molecules)" itself. Only the prose
        was, and a Methods paragraph is written to be pasted into a paper.
        """
        from molbuilder.spectra import render_methods_md
        from molbuilder.structure import Structure
        co2 = Structure(
            elements  = ["O", "C", "O"],
            positions = np.array([[-1.163, 0., 0.],
                                  [ 0.,    0., 0.],
                                  [ 1.163, 0., 0.]]),
        )
        md = render_methods_md(_spectra_cfg(), struct=co2)
        assert "4 non-translational" in md, (
            "a linear triatomic has 3N-5 = 4 vibrational modes; the Methods "
            f"paragraph says otherwise:\n{md}"
        )

        n2 = Structure(elements=["N", "N"],
                       positions=np.array([[0., 0., 0.], [1.10, 0., 0.]]))
        md2 = render_methods_md(_spectra_cfg(), struct=n2)
        assert "1 non-translational" in md2, (
            f"a diatomic has exactly one vibration:\n{md2}"
        )

    def test_the_engine_fragment_names_no_projection_COUNT(self):
        """The count belongs where the GEOMETRY is, and nowhere else.

        `pyscf_methods_fragment` is handed `cfg` alone -- it cannot tell a
        linear molecule from a bent one -- so it said "the six
        translational/rotational eigenvectors" for every system. Two places
        naming one number is how they came to disagree.
        """
        from molbuilder.pyscf.vibration_emitters import pyscf_methods_fragment
        frag = pyscf_methods_fragment(_spectra_cfg())
        assert "six" not in frag.lower(), (
            f"the fragment states a projection count it cannot know:\n{frag}"
        )

    def test_real_structure_with_frozen_atoms(self):
        """A real Structure with its two Au atoms frozen -> they leave
        the free count."""
        from molbuilder.spectra import render_methods_md
        from molbuilder.structure import Structure
        struct = Structure(
            elements  = ["Au", "Au", "C", "H"],
            positions = np.array([[0., 0., 0.],
                                  [2., 0., 0.],
                                  [4., 0., 0.],
                                  [5., 0., 0.]]),
        )
        # Frozen atoms are INDICES -- the TWO Au atoms are 0 and 1.
        cfg = _spectra_cfg(struct=struct, frozen_indices=[0, 1])
        md = render_methods_md(cfg, struct=struct)
        assert "2 free" in md
        assert "2 frozen" in md




# ── the HOMO rule -- ours, and it has a branch ──────────────────────────────

@pytest.mark.parametrize("mo_occ,expected,why", [
    ([2.0, 2.0, 0.0, 0.0],                    1, "restricted: 2 filled, 2 empty"),
    ([2.0],                                   0, "restricted: a single filled level"),
    ([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]],      1, "UNRESTRICTED: alpha+beta summed"),
    ([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]],      2, "unrestricted doublet: alpha goes higher"),
    ([2.0, 1.8, 0.3, 0.0],                    1, "fractional: 0.3 is not occupied"),
    ([2.0, 0.6, 0.4],                         1, "fractional: 0.6 is, 0.4 is not"),
    ([2.0, 0.0, 2.0],                         2, "unsorted occupancies: the HIGHEST index wins"),
])
def test_the_homo_is_the_highest_level_carrying_more_than_half_an_electron(
        mo_occ, expected, why):
    """SCIENCE. `homo_index` picks the highest MO with occupancy above 0.5 --
    summing spin channels first when the reference is unrestricted.

    THE FAILURE THIS CATCHES.  This index is molbuilder's, not PySCF's: PySCF
    reports occupation numbers and WE derive the HOMO from them. `web/spectra.md`
    § 3.1 has the level diagram, the HOMO-LUMO gap and the gap SHIFT all reading
    from it, with shifts of ~0.018 meV -- so an index off by one does not look
    like an error, it looks like a different answer.

    THE BRANCH IS THE POINT.  PySCF gives `mo_occ` as 1-D for RHF/RKS and as
    2-D `(alpha, beta)` for UHF/UKS. Miss the sum and every OPEN-SHELL
    calculation reports the beta channel's HOMO, or the alpha one, instead of
    the system's -- silently, and only for the open-shell half of the work.
    The two unrestricted rows are the ones that catch it.

    WHY THIS TEST CAN EXIST AT ALL (2026-09-09).  The rule lived only as text
    inside the emitted PySCF script, where nothing could call it, and the
    sidecar it writes carries `homo_idx` WITHOUT the occupations it came from --
    so the read side cannot check it either, and `SpectraResults` validates only
    that the index is in range. It is now a real function that the emitter
    splices into the script, so the tests exercise the implementation that runs.
    Recorded at `science/test-design-findings.md` § 7a.
    """
    from molbuilder.pyscf.vibration_emitters import homo_index
    assert homo_index(mo_occ) == expected, why


def test_a_reference_with_no_occupied_levels_is_refused_not_indexed():
    """SCIENCE. No occupied MO means there is no HOMO -- refuse, rather than
    hand back an index into an empty selection.

    `np.max` of an empty array raises a bare `ValueError` about zero-size
    reduction, which reaches a person as a traceback naming numpy rather than
    their calculation. This says what is actually wrong.
    """
    from molbuilder.pyscf.vibration_emitters import homo_index
    with pytest.raises(ValueError, match="no occupied levels"):
        homo_index([0.0, 0.0, 0.0])


def test_every_derived_quantity_is_named_in_the_provenance_table():
    """The provenance table in `web/spectra.md` § 9b names every quantitative
    key the emitter writes -- so a new number cannot ship undocumented.

    THE FAILURE THIS CATCHES.  A spectrum is a quantitative result, and the
    chain from PySCF's own objects to the number on screen is what makes it
    checkable by someone who did not write it. A key added to the sidecar
    without a row here is a number with no stated origin: a reader cannot tell
    whether molbuilder passed it through or derived it, and therefore cannot
    tell whose bug it would be.

    ARTIFACT LINT (`testing.md` § 3b): it quantifies over the class rather than
    naming today's keys, so the next one is caught the day it is added.

    Contract: `web/spectra.md` § 9b.
    """
    from pathlib import Path
    doc = (Path(__file__).resolve().parents[2] / "docs" / "web" / "spectra.md"
           ).read_text(encoding="utf-8")
    table = doc[doc.index("## 9b. Provenance"):doc.index("### 9b.1")]

    # The quantitative keys -- values a person reads off a chart or a paper.
    # Bookkeeping keys (schema_version, engine, elements, ...) are excluded by
    # name, so adding one does not force a row it does not need.
    quantitative = {
        "scf_energy_eh", "mo_energies_eh", "homo_idx", "frequency_cm1",
        "eigenvector_canonical", "ir_intensity_km_mol",
        "raman_activity_a4_amu", "amplitude_ang",
    }
    missing = sorted(k for k in quantitative if f"`{k}`" not in table
                     and k not in table)
    assert not missing, (
        f"these quantitative keys have no row in `web/spectra.md` § 9b, so "
        f"nothing states where their values come from: {missing}")
