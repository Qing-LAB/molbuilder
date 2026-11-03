"""Chemistry-rule validators (engine-agnostic, callable from any engine).

The home for every check that asks a chemistry question of a structure and
a calculation, whichever engine runs it: whether each species label names an
element, THE ELECTRONIC STATE'S ONE FAMILY (:func:`check_electronic_state`
-- charge, parity, the open-shell guard, what an engine cannot run;
`science/chemistry-correctness.md` § 2a) and the peptide-protonation
advisory.  An engine's own chemistry -- a basis's coverage, a core
potential -- is its validator's (`validation/pyscf.py`).  The facts come from
``chemistry.analyze_structure`` and the state from
``electronic_state.electronic_state`` -- the same two the chemistry card and
the deck writers read -- so no surface keeps its own parallel logic.
"""

from __future__ import annotations

from typing import List

from ..issues import Issue
from ..structure import Structure


def check_species_labels(struct: Structure, *, engine: str = ""
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
    turns a label into an element.  ``engine`` names who would run them,
    said in its record's words; none, *the engine*.
    """
    from ..chemistry import resolve_element
    if engine:
        from ..engine_table import engine as _engine
        engine_label = _engine(engine).words
    else:
        engine_label = "the engine"

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


def check_electronic_state(struct: Structure, cfg, *, calculation: str,
                           engine: str, state) -> List[Issue]:
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
      spin-orbit, unrestricted on transport), and a setting of the engine's
      own that cannot run the spin (its config class's ``spin_refusal`` --
      SIESTA's OMM solver with a spin it does not hold);
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

    ``state`` is the one the settings gate worked out (``validate``); the
    spin's findings stand down where it is ``None`` -- a label naming no
    element: the state is an electron count, and ``check_species_labels``
    owns that finding.
    """
    from ..kind_table import kind as _kind

    fixed = _kind(calculation).charge
    out = _charge_fixed(cfg, _kind(calculation)) if fixed is not None else []
    if state is None:
        return out
    # The charge's own advisory is a separate fact from the spin's, so it
    # rides beside whichever spin finding holds -- except on a kind that
    # fixes its charge, refused when stated (ES7): advice to state one
    # there would be advice this family refuses.
    return (out + _spin_findings(struct, state, engine, calculation)
            + _spin_refusal(cfg, state)
            + (_check_peptide_protonation(struct, state.net_charge)
               if fixed is None else []))


def _spin_refusal(cfg, st) -> List[Issue]:
    """A setting of the engine's own that cannot run this spin -- its
    config's ``spin_refusal`` (SIESTA's OMM solver with a spin it does not
    hold, `config/siesta.py`), refused here, before the run, never at the
    engine's start."""
    refused = cfg.spin_refusal(st)
    if refused is None:
        return []
    item, why = refused
    return [Issue("error", why, f"config.{item}")]


def _charge_fixed(cfg, kind) -> List[Issue]:
    """ES7: a kind that fixes its charge carries no other -- a transport
    calculation is neutral (the kind table's ``charge``).

    The one place that reads the STATED charge of such a config: the
    state's rule fixes it (``electronic_state._charge``), so a value in the
    template would otherwise be dropped without a word -- silently
    neutralising someone's charged junction is the defect; saying so is the
    fix -- in the record's own reason (its ``charge.why``)."""
    fixed = kind.charge
    charge = getattr(cfg, "net_charge", None)
    if charge is None or int(charge) == fixed.value:
        return []
    return [Issue(
        "error",
        f"net_charge = {int(charge):+d}, and {kind.with_article} carries "
        f"{fixed.value:+d} -- {fixed.why}.  Remove net_charge from this "
        f"calculation's template.",
        "config.net_charge")]


def _spin_findings(struct: Structure, st, engine: str,
                   calculation: str) -> List[Issue]:
    """The spin's findings, first that holds wins (see
    :func:`check_electronic_state`)."""
    from ..chemistry import check_spin_charge_parity
    from ..electronic_state import FREE, count_must_float
    from ..engine_table import engine as _engine
    from ..kind_table import kind as _kind
    from ..template import (catalogue, is_member, offered, one,
                            why_not_offered)

    words = _engine(engine).words
    t, c, q = st.spin_treatment, st.unpaired_electrons, st.net_charge
    _cat = catalogue()
    treatments = offered(one(_cat, "spin_treatment", engine=engine), engine,
                         calculation)

    def _not_offered(item, r, can):
        return [Issue(
            "error",
            f"{item} = {r.value} ({r.said}) cannot run here: "
            f"{_kind(calculation).with_article} on {words} does not offer it -- "
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
               f"{_kind(calculation).with_article[:1].upper()}{_kind(calculation).with_article[1:]} "
               f"on {words} offers restricted alone -- "
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
            # WHAT THE ENGINE DOES with an odd count under restricted spin --
            # runs it anyway, said (SIESTA), or refuses it, the parity error
            # (PySCF): its config class's rule.
            from ..electronic_state import odd_count_restricted
            runs_anyway = (odd_count_restricted(engine, st.n_electrons)
                           if t.value == "restricted" else None)
            detail = (f"  The charge is {q.value:+d} ({q.said}).")
            if runs_anyway:
                return [Issue("warn", runs_anyway + detail,
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
