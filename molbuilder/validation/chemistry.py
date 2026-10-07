"""Chemistry-rule validators (engine-agnostic, callable from any engine).

The home for every check that asks a chemistry question of a structure and
a calculation: whether each species label names an element, THE ELECTRONIC
STATE'S ONE FAMILY (:func:`check_electronic_state` -- charge, parity, the
open-shell guard, what an engine cannot run; `science/chemistry-
correctness.md` § 2a), the peptide-protonation advisory, and whether a
metal's basis and pseudopotential are adequate.  The facts come from
``chemistry.analyze_structure`` and the state from
``electronic_state.electronic_state`` -- the same two the chemistry card and
the deck writers read -- so no surface keeps its own parallel logic.
"""

from __future__ import annotations

from typing import List

from ..issues import Issue
from ..structure import Structure


def check_species_labels(struct: Structure, *, engine_label: str = "the engine"
                          ) -> List[Issue]:
    """Every species label must NAME an element -- it need not BE one.

    We do not police names.  ``Au1`` / ``Au2`` is how a user asks for two
    gold species with different basis or pseudopotential, and SIESTA's own
    ``%block ChemicalSpeciesLabel`` is ``index Z label`` precisely so that
    works.  What a calculation cannot do is run a species whose element is
    unknown, so:

      * a label naming no element is an **error** -- it blocks emission with
        a message naming the label, instead of the ``KeyError`` the emitter
        would otherwise raise from inside a render;
      * a label that resolves but is not itself a symbol is **info**: worth
        saying once which element it was read as, not worth a warning that
        nags every run of a deliberate setup.

    Contract: ``chemistry.resolve_element``, which is the only place that
    turns a label into an element.
    """
    from ..chemistry import resolve_element

    issues: List[Issue] = []
    unresolved: List[str] = []
    labelled: List[str] = []
    for el in dict.fromkeys(str(e) for e in struct.elements):
        try:
            sym = resolve_element(el)
        except KeyError:
            unresolved.append(el)
            continue
        if el.strip() != sym:
            labelled.append(f"{el} -> {sym}")

    if unresolved:
        named = ", ".join(repr(u) for u in unresolved)
        issues.append(Issue(
            "error",
            (f"species {named} name no element, so {engine_label} has no "
             f"atomic number to run them with.  A label may carry more than "
             f"the symbol -- 'Au1' and 'Au2' are read as gold -- but it must "
             f"start with one.  Fix the element column of the structure file, "
             f"or rename the species"),
            "chemistry.species_label",
        ))
    if labelled:
        issues.append(Issue(
            "info",
            (f"species labels read as elements: {', '.join(labelled)}.  "
             f"The label is written through to the engine unchanged; the "
             f"atomic number beside it is the element's"),
            "chemistry.species_label",
        ))
    return issues


def _check_peptide_protonation(struct: Structure, charge) -> List[Issue]:
    """Hint at the gap between gas-phase neutral build and
    physiological charge state for peptides with charged side chains.

    PeptideBuilder + AddHs produces a neutral molecule by default
    (Asp / Glu protonated, Lys / Arg uncharged amines).  At pH 7 the
    charged side chains carry a net charge.  Most users don't realise
    the script is silently using the gas-phase neutral form.

    ``charge`` is the electronic state's resolved charge
    (``electronic_state.Resolved``).  Triggered only when the structure
    looks like a peptide, its estimated pH-7 charge is non-zero, and the
    charge that will run is 0 -- whether the phosphate rule found nothing
    (a peptide has no phosphates) or a person stated neutral: both build
    the same gas-phase form, so both deserve to hear about the side chains.

    Severity: warn (not error).  The neutral build may be exactly
    what the user wants -- the surface emits SIESTA / PySCF input
    that runs without modification.  This warning surfaces the
    INFORMATION gap, not a bug.
    """
    from ..chemistry import expected_pH7_peptide_charge
    expected = expected_pH7_peptide_charge(struct)
    if expected is None or expected == 0 or charge.value != 0:
        return []
    return [Issue(
        "warn",
        f"peptide has charged side chains (estimated charge at "
        f"pH 7.4: {expected:+d}) but the net charge is 0 ({charge.said}); "
        f"the script will build the gas-phase neutral form (Asp/Glu "
        f"protonated, Lys/Arg neutral).  For physiological-state runs "
        f"state net_charge = {expected} (and consider the basis: an anion "
        f"needs diffuse functions such as aug-cc-pVDZ)",
        "config.net_charge",
    )]


#: The row-5 boundary (Kr, Z=36).  It is NOT a definition of "heavy" and
#: decides nothing on its own -- it only bounds WHOM THE HINT BELOW NAMES,
#: and the hint prints the criterion so a reader can disagree with it.
#:
#: The user's ruling (2026-08-13): *"there is no point to limit matching
#: to heavy -- who defines heavy? there is no clear reasoning or standard
#: ... explicit is better than implicit."*  So it is the threshold of a
#: question, never of an action.
_ECP_HINT_Z = 36


def _check_ecp_declared_for_the_atoms_that_usually_want_one(
        struct: Structure, *, ecp: str, ecp_atoms, basis: str,
        engine_label: str) -> List[Issue]:
    """WARN when a structure carries post-Kr elements and no ECP covers them.

    **This hint asks; it never chooses.**  It speaks because of the other
    half of the user's ruling: *"you can still have the validation
    function to give hints -- that should be confirmed."*

    An element is covered by an ``ecp`` name whose ``ecp_atoms`` patterns
    select it -- and by nothing else.  **A def2 basis does NOT cover it by
    itself**: PySCF's def2 files
    carry the core potential (``def2-svp.dat``: *Au nelec 60*), but
    ``gto.M`` applies one only when ``ecp`` is given (PySCF 2.14
    ``gto/mole.py``, ``build``; ``check_sanity`` merely warns), and the deck
    gives one only when the person declares it.  So on a def2 basis the hint
    names that basis's own core potential -- the one it was built for.
    """
    from ..chemistry import atomic_number, resolve_pyscf_ecp

    covered = resolve_pyscf_ecp(struct, ecp, ecp_atoms) or {}
    uncovered: List[str] = []
    for el in struct.elements:
        sym = str(el).strip()
        try:
            z = atomic_number(sym)
        except KeyError:
            # A label naming no element -- `check_species_labels` already
            # reports it as an error.  Skipping keeps THIS check about ECPs.
            continue
        if z > _ECP_HINT_Z and sym not in covered and sym not in uncovered:
            uncovered.append(sym)
    if not uncovered:
        return []

    named = ", ".join(uncovered)
    them = "them" if len(uncovered) > 1 else "it"
    if (basis or "").lower().replace("_", "").replace("-", "").startswith("def2"):
        # THE BASIS'S OWN CORE POTENTIAL, named: a def2 basis for these
        # elements is a valence basis built for it, so all-electron on it is
        # not a cheaper answer but a broken one.
        message = (
            f"No effective core potential covers {named}, so "
            f"{engine_label} will put every electron of {them} into basis "
            f"'{basis}' -- a valence basis, built for its own core potential "
            f"({'these elements' if len(uncovered) > 1 else 'this element'} "
            f"past Kr).  {engine_label} applies that core potential only when "
            f"the script names it, and nothing names it for you.  To use it: "
            f"ecp = '{basis}' with ecp_atoms = {uncovered!r}.")
    else:
        message = (
            f"No effective core potential covers {named}, so {engine_label} "
            f"will treat {them} ALL-ELECTRON on basis '{basis}'.  For "
            f"elements past Kr that is usually wrong twice over: the cost is "
            f"large (Pt alone carries 78 electrons), and without a "
            f"scalar-relativistic ECP the bond lengths and orbital energies "
            f"are off -- Pt-Pt by ~0.1 A, Au gaps by ~1 eV.  To use one, name "
            f"it and say which atoms get it: ecp = 'lanl2dz' with ecp_atoms = "
            f"{uncovered!r} (or ['*'] for every element present); a def2 "
            f"basis's own is named the same way, ecp = the basis.  If you "
            f"meant all-electron, this is the confirmation: nothing is added "
            f"for you.  (Named here because Z > {_ECP_HINT_Z}; that bound "
            f"decides what gets mentioned, nothing else.)")
    return [Issue("warn", message, "config.ecp")]


def _check_metal_basis_adequacy(struct: Structure, *,
                                  basis: str, engine_label: str
                                  ) -> List[Issue]:
    """Shared chemistry rule: basis sets like STO-3G / 6-31G / 6-31G(d)
    have poor or no coverage of transition-metal d-orbitals.  Pair with
    Fe / Mn / Co / Ni / Cu / Mo etc. and the SCF converges to a
    distorted electronic structure with the wrong d-orbital ordering.

    Recommendations encoded here: def2-SVP is the production minimum;
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
             f"def2-TZVP or cc-pVTZ-DK."),
            "config.basis",
        )]
    return []


def check_electronic_state(struct: Structure, cfg, *,
                           calculation: str) -> List[Issue]:
    """The electronic state's findings -- ONE family, for every engine and
    every kind (`science/chemistry-correctness.md` § 2a, ES3-ES9).

    Asked once, by :func:`molbuilder.validation.validate`, of the state the
    deck will be written from (``electronic_state``).

    At most one finding per fact (ES9), and the first that holds wins:

    * **ES4 / ES5 / ES6 -- what cannot run** (errors): a treatment or a
      count the kind does not offer on this engine (the catalogue's
      `offered` -- a floating moment on PySCF, unrestricted on a PySCF
      vibration, non-collinear on transport), unpaired electrons beside
      restricted, a fixed count where SIESTA cannot hold one (non-collinear,
      spin-orbit, unrestricted on transport);
    * **ES3 -- parity**, for a finite system and a pinned count: an error
      where the engine refuses it or a fixed count contradicts it, a warning
      where SIESTA runs a restricted radical half-filled;
    * **ES9 -- a stated closed shell where the structure implies an open
      one** (the hemeC guard), and **a constrained singlet on a closed
      shell**; a stated open-shell count on an even structure is not a
      finding -- triplet O2 is exactly that, and parity cannot see it;
    * **ES8 -- a count a metal decided** warns until it is stated;
    * **ES7 -- a charge on a transport calculation** is refused: the rule
      makes the junction neutral, so a stated charge would otherwise be
      dropped without a word.

    The spin's findings stand down on a label naming no element: the state
    is an electron count, and ``check_species_labels`` owns that finding.
    """
    from ..chemistry import every_label_resolves
    from ..electronic_state import KINDS, electronic_state, engines_for
    from ..template import engine_name

    # Whose schema this config is -- the one answer (`template.engine_name`),
    # never an isinstance ladder here.  An engine that does not run the kind
    # has no electronic state to judge.
    engine = engine_name(type(cfg))
    if calculation not in KINDS or engine not in engines_for(calculation):
        return []
    out = _transport_charge(cfg) if calculation == "transport" else []
    if not every_label_resolves(struct):
        return out
    st = electronic_state(struct, cfg, kind=calculation)
    # The charge's own advisory is a separate fact from the spin's, so it
    # rides beside whichever spin finding holds -- except on a transport
    # calculation, whose charge is 0 by rule and refused when stated (ES7):
    # advice to state one there would be advice this family refuses.
    return (out + _spin_findings(struct, st, engine, calculation)
            + (_check_peptide_protonation(struct, st.net_charge)
               if calculation != "transport" else []))


def _transport_charge(cfg) -> List[Issue]:
    """ES7: a transport calculation carries no net charge.

    The one place that reads the STATED charge of a transport config: the
    state's rule makes the junction neutral (``electronic_state._charge``),
    so a value in the template would otherwise be dropped without a word.

    `engines/transport.md` 2a.7 defers net charge and gating, and the reason
    is the boundary condition: a transport calculation is OPEN, so the
    device's electron number is set by the electrodes' chemical potentials
    and found by the contour integration rather than fixed by the deck --
    and a lead is bulk metal that must stay neutral, because charging it
    moves the Fermi level every downstream stage is measured against.  A
    config can still carry a value -- a caller building it directly.
    Silently neutralising someone's
    charged junction is the defect; saying so is the fix.
    """
    charge = getattr(cfg, "net_charge", None)
    if not charge:
        return []
    return [Issue(
        "error",
        f"net_charge = {int(charge):+d}, and a transport calculation "
        f"cannot carry one.  Its boundaries are OPEN: the device's "
        f"electron count is set by the electrodes' chemical potentials "
        f"and found by the contour integration, not fixed by the deck, "
        f"and the leads are bulk metal that must stay neutral -- a "
        f"charged lead moves the Fermi level every stage is measured "
        f"against.  Net charge and gating are deferred by ruling "
        f"(engines/transport.md 2a.7); a gated or electrochemical "
        f"junction is separate work.  Remove net_charge from this "
        f"calculation's template, or relax the charged species as an "
        f"OPTIMIZATION, where the keyword is honoured.",
        "config.net_charge")]


def _spin_findings(struct: Structure, st, engine: str,
                   calculation: str) -> List[Issue]:
    """The spin's findings, first that holds wins (see
    :func:`check_electronic_state`)."""
    from ..chemistry import check_spin_charge_parity
    from ..electronic_state import FREE, count_must_float
    from ..template import (a_kind, catalogue, is_member, offered, one,
                            why_not_offered)

    t, c, q = st.spin_treatment, st.unpaired_electrons, st.net_charge
    _cat = catalogue()
    treatments = offered(one(_cat, "spin_treatment", engine=engine), engine,
                         calculation)

    def _not_offered(item, r, can):
        return [Issue(
            "error",
            f"{item} = {r.value} ({r.said}) cannot run here: "
            f"{a_kind(calculation)} on {engine} does not offer it -- "
            f"{why_not_offered(item, r.value, engine, calculation)}.  "
            f"It offers {', '.join(map(str, can))}.",
            f"config.{item}")]

    # ES4 -- declared, not discovered: what the engine runs for the kind is
    # the catalogue's (`offered`, `engines/template.md` § 6.3a), held to the
    # RESOLVED values -- a treatment detected from the structure can be one
    # the kind cannot run (a radical's `unrestricted` on a PySCF vibration),
    # and the state's own words say where it came from.
    if not is_member(t.value, treatments):
        return _not_offered("spin_treatment", t, treatments)
    # ES5 -- restricted means closed-shell, and says so before the count's
    # own set does: a stated `restricted` with a count is a contradiction
    # first, and its way out is what the kind offers.
    if t.value == "restricted" and c.value != 0:
        ways = []
        if "restricted-open" in treatments:
            ways.append("restricted-open (spin-pure, one set of spatial "
                        "orbitals)")
        if "unrestricted" in treatments:
            floats = count_must_float(engine, calculation, "unrestricted")
            ways.append("unrestricted (the two channels relax separately"
                        + (f"; its count floats here -- {floats}" if floats
                           else "") + ")")
        way = (f"State {' or '.join(ways)}." if ways else
               f"{a_kind(calculation)[:1].upper()}{a_kind(calculation)[1:]} "
               f"on {engine} offers restricted alone -- "
               f"{why_not_offered('spin_treatment', 'unrestricted', engine, calculation)}"
               f" -- so the count is 0.")
        return [Issue(
            "error",
            f"restricted means every electron paired, and "
            f"unpaired_electrons = {c.value} asks for "
            f"{'a floating moment' if c.value == FREE else f'{c.value} unpaired'}. "
            f"{way}",
            "config.spin_treatment")]
    counts = offered(one(_cat, "unpaired_electrons", engine=engine), engine,
                     calculation)
    if not is_member(c.value, counts):
        return _not_offered("unpaired_electrons", c, counts)
    # ES6 -- where the count can only float, a fixed one is refused: the
    # non-collinear and spin-orbit treatments on SIESTA, and unrestricted on a
    # transport rung (TranSIESTA).  A count nobody stated already floats.
    why = count_must_float(engine, calculation, t.value)
    if why and c.value != FREE:
        return [Issue(
            "error",
            f"unpaired_electrons = {c.value} cannot be held here: {why}.  "
            f"Leave unpaired_electrons blank, or state free.",
            "config.unpaired_electrons")]

    # ES3 -- parity binds a finite system with a pinned count.
    if st.finite and c.value != FREE:
        msg = check_spin_charge_parity(struct, q.value, int(c.value))
        if msg:
            half_filled = engine == "siesta" and t.value == "restricted"
            detail = (f"  The charge is {q.value:+d} ({q.said}).")
            if half_filled:
                return [Issue(
                    "warn",
                    f"An odd electron count ({st.n_electrons}) under "
                    f"restricted: SIESTA runs it with the top level half "
                    f"filled -- a restricted description of a radical, not "
                    f"its ground state.  State unrestricted (2S = 1) for a "
                    f"radical, or check the charge.{detail}",
                    "config.unpaired_electrons")]
            return [Issue("error", msg[0].upper() + msg[1:] + detail,
                          "config.unpaired_electrons")]

    out: List[Issue] = []
    rec = st.recommended
    # ES9 -- one finding for the stated state against the structure's.
    if t.source != "detected" and t.value == "restricted" \
            and rec.spin_treatment != "restricted":
        out.append(Issue(
            "warn",
            f"The spin treatment is restricted ({t.said}), and the structure "
            f"implies {rec.spin_treatment}, 2S = {rec.unpaired_electrons}: "
            f"{rec.why}.  A closed-shell SCF on an open-shell system "
            f"converges to a fictitious state with unphysical forces.  "
            f"Leave the spin fields blank to take the structure's answer, "
            f"or state the count.",
            "config.spin_treatment"))
    elif t.source != "detected" and t.value == "unrestricted" \
            and c.value == 0 and rec.spin_treatment == "restricted":
        out.append(Issue(
            "warn",
            f"unrestricted at 2S = 0 is a constrained singlet, and this "
            f"structure is closed-shell ({rec.why}): the same answer as "
            f"restricted at twice the cost.  Keep it only for a "
            f"broken-symmetry singlet.",
            "config.spin_treatment"))
    # What a count SOMEBODY SAID means for each open-d metal present -- the
    # person's, or the run's the structure came out of -- an echo, so the
    # oxidation state it implies can be checked against the chemistry (info:
    # it labels, it does not count).  Not an implied count: a stated
    # restricted's 0 on an iron complex is ES9's finding already.
    if c.source in ("stated", "recorded") and c.value != FREE:
        from ..chemistry import explain_metal_spin
        for m in st.facts.open_d_metals:
            label = explain_metal_spin(m, int(c.value))
            if label:
                out.append(Issue(
                    "info",
                    f"{m} with 2S = {c.value}: {label}.  Confirm it against "
                    f"your experimental data (Mössbauer / UV-Vis / EPR) or the "
                    f"chemistry of the rest of the molecule (axial ligands, "
                    f"protonation).",
                    "config.unpaired_electrons"))
    # ES8 -- a metal's usual count is a guess about the coordination.
    if st.metal_driven:
        hint = next((h for h in st.facts.metal_hints
                     if h.element == rec.metal), None)
        others = ([f"{sc.spin} ({sc.label.split(' -- ')[0]})"
                   for sc in hint.common_spins] if hint else [])
        out.append(Issue(
            "warn",
            f"unpaired_electrons was left blank, so it is {c.value}: "
            f"{c.why}."
            + (f"  {rec.metal}'s common counts: {'; '.join(others)}."
               if others else ""),
            "config.unpaired_electrons"))
    return out
