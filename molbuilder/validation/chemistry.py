"""Chemistry-rule validators (engine-agnostic, callable from any engine).

The home for every validator that asks a question the chemistry
analyzer can answer.  Per docs/science/validation.md + Rule 1 of web-ui-coherence.md: every UI / preflight surface
that gates on "open-shell or closed?", "metal basis adequate?",
"protonation matches charge?" calls into THIS module — not its own
parallel logic.

Split from the pre-2026-06-13 flat ``molbuilder/validation.py`` per
docs/science/validation.md  Function bodies +
signatures are identical to the pre-split versions; only the home
moved.
"""

from __future__ import annotations

from typing import List

from ..issues import Issue
from ..structure import Structure


def _check_peptide_protonation(struct: Structure,
                               cfg_charge) -> List[Issue]:
    """Hint at the gap between gas-phase neutral build and
    physiological charge state for peptides with charged side chains.

    PeptideBuilder + AddHs produces a neutral molecule by default
    (Asp / Glu protonated, Lys / Arg uncharged amines).  At pH 7 the
    charged side chains carry a net charge.  Most users don't realise
    the script is silently using the gas-phase neutral form.

    Triggered only when:
      * the structure looks like a peptide (has standard amino-acid
        residue names);
      * the estimated pH-7 charge is non-zero;
      * the user hasn't explicitly set cfg.charge to a non-zero
        value (None or 0 means "auto / default neutral").

    Severity: warn (not error).  The neutral build may be exactly
    what the user wants -- the surface emits SIESTA / PySCF input
    that runs without modification.  This warning surfaces the
    INFORMATION gap, not a bug.
    """
    from ..chemistry import expected_pH7_peptide_charge
    expected = expected_pH7_peptide_charge(struct)
    if expected is None or expected == 0:
        return []
    # cfg_charge None -> auto-detection path; cfg_charge == 0 -> the
    # user explicitly forced neutral.  Both paths produce the same
    # gas-phase build, so both deserve the warning telling them about
    # the side-chain mismatch.  An explicit non-zero cfg_charge means
    # the user already accounted for this -- skip the warning.
    if cfg_charge not in (None, 0):
        return []
    return [Issue(
        "warn",
        f"peptide has charged side chains (estimated charge at "
        f"pH 7.4: {expected:+d}) but cfg.charge = 0; the script "
        f"will build the gas-phase neutral form (Asp/Glu protonated, "
        f"Lys/Arg neutral).  For physiological-state runs set "
        f"cfg.charge = {expected} (and adjust spin / basis: open "
        f"shells need diffuse functions like aug-cc-pVDZ for anions)",
        "config.charge",
    )]


def _check_metal_basis_adequacy(struct: Structure, *,
                                  basis: str, engine_label: str
                                  ) -> List[Issue]:
    """Shared chemistry rule: basis sets like STO-3G / 6-31G / 6-31G(d)
    have poor or no coverage of transition-metal d-orbitals.  Pair with
    Fe / Mn / Co / Ni / Cu / Mo etc. and the SCF converges to a
    distorted electronic structure with the wrong d-orbital ordering.

    Recommendations encoded here mirror the spec § Scientific
    correctness guidance: def2-SVP is the production minimum;
    def2-TZVP is publication-quality.  Anything smaller for a
    transition-metal-containing structure -> WARN.
    """
    # ALL transition metals, not just open-shell: d-orbital basis coverage is
    # equally needed for closed-shell d10 metals (Zn/Cd/Hg/Pd/Pt) -- the
    # concern is orbital coverage, orthogonal to spin state.
    from ..chemistry import detect_transition_metals
    metals = detect_transition_metals(struct)
    if not metals:
        return []
    b = (basis or "").lower().strip()
    # Bases known to be inadequate for transition metals (no d set
    # for first-row TMs, or no functions at all for second/third row).
    INADEQUATE = ("sto-3g", "sto-6g", "3-21g", "6-31g",
                  "6-31g(d)", "6-31g*", "6-31g**", "6-31gd", "6-311g")
    if any(b == bad or b.startswith(bad + "/") for bad in INADEQUATE):
        return [Issue(
            "warn",
            (f"Basis '{basis}' has inadequate coverage of transition-"
             f"metal d-orbitals.  {engine_label} requested for "
             f"structure containing {', '.join(metals)}.  Bases like "
             f"STO-3G / 6-31G(d) lack the polarisation/diffuse "
             f"functions needed to describe metal-ligand bonds + spin "
             f"states; the SCF often converges to a distorted "
             f"electronic structure (wrong d-orbital occupations, "
             f"wrong spin-gap energies).  Recommended minimum for "
             f"transition metals: def2-SVP.  Publication quality: "
             f"def2-TZVP or cc-pVTZ-DK.  PySCF auto-loads a Stuttgart "
             f"ECP for second/third-row TMs when ``basis='def2-SVP'``."),
            "config.basis",
        )]
    return []


def check_open_shell_metal(struct: Structure, *,
                              is_closed_shell: bool,
                              engine_label: str) -> List[Issue]:
    """Shared chemistry rule: structure whose ANALYZER recommends
    open-shell DFT requires an open-shell SCF (PySCF UKS/UHF + spin>0;
    SIESTA spin_polarized=True).  A closed-shell SCF on a true
    open-shell complex converges to a fictitious electronic state
    with garbage forces (hemeC-dithiol 2026-05-22 incident).

    Single source of truth: ``ChemistryAnalysis.suggested_treatment``.
    Pre-2026-06-13 this function checked ``analysis.metals`` (non-
    empty → warn) which incorrectly fired for Au-BDT-Au — Au IS a
    transition metal but in a metallic cluster context the analyzer
    correctly suggests closed-shell singlet (Stoner criterion fails
    for noble metals; published Au transport DFT is RKS by
    convention).  The validator was warning the user to "switch to
    open-shell" while the detection chip on the SAME form said
    "closed-shell singlet" — direct contradiction.  The fix:
    delegate the closed-vs-open decision to the analyzer, which
    already encodes the noble-metal cluster-context logic, and
    only fire when the analyzer's recommendation truly disagrees
    with the user's chosen treatment.

    By construction the validator and the auto-detect surface cannot
    disagree about the chemistry now; whatever the user sees on the
    Auto-detect chip at load time is the same conclusion that gates
    this warning at Generate time.  See
    ``docs/science/validation.md`` § 5.3 and § 3.4
    (noble-metal cluster-context rule).
    """
    from ..chemistry import analyze_structure
    analysis = analyze_structure(struct)
    # Only warn when the analyzer's recommendation is OPEN-SHELL but
    # the user picked a closed-shell SCF.  When the analyzer says
    # closed-shell (Au cluster, organic, closed-d10), no warning —
    # the chip's "closed-shell singlet" matches the validator's
    # silence.
    analyzer_says_open = (analysis.suggested_treatment == "open")
    if analyzer_says_open and is_closed_shell:
        return [Issue(
            "warn",
            (f"Analyzer recommends OPEN-SHELL DFT for this structure "
             f"({', '.join(analysis.metals)}) but {engine_label} "
             f"requests a closed-shell SCF.  Closed-shell SCF on a "
             f"true open-shell complex converges to a fictitious "
             f"state with unphysical forces.  Switch to open-shell "
             f"SCF and set a sensible spin (see config-field help "
             f"for the spin / spin_polarized field).  "
             f"{analysis.rationale}"),
            "config.spin",
        )]
    return []
