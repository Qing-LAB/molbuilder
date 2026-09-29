"""The electronic state of a calculation -- one class, one answer.

``docs/science/chemistry-correctness.md`` § 2a is the contract; this module is
its one implementation.

**Four template items say what a person decided** -- ``net_charge``,
``spin_treatment``, ``unpaired_electrons`` and ``method`` -- and a blank one
means *work it out*.  This module works it out, once, the way the charge rule
always worked out a blank charge, and every reader of charge or spin asks it:
the deck writers, the settings gate, the hand-over, the forms (through
``/api/structure/analyze``) and the read-back.  Nothing else decides.

Until 2026-09-28 the two halves of one state were not even decided the same
way.  The charge was resolved where it was used (``resolve_net_charge``); the
spin was never resolved at all -- the analyzer only *suggested* it, and the
suggestion reached a deck only if someone clicked Auto-detect, which copied it
into the form and overwrote whatever was there.  Each layer then read the raw
fields and interpreted them itself: three parity checks with three severity
rules, four R/U mappings, a formate ion told to go open-shell because the
count ignored its charge, a gold lead told the same because the count ignored
its cell (the contract's § 2a lists the five failures).

How a blank is answered (§ 2a.1a) -- a stated value always wins, and a blank
takes the first of:

1. **implied** by a stated item (``restricted`` ⇒ 0; ``non-collinear`` or
   ``spin-orbit`` ⇒ ``free``; a count above 0 or ``free`` ⇒ ``unrestricted``;
   a count of 0 ⇒ ``restricted``);
2. **recorded** by the run the structure came out of -- the record a
   structure exported from a finished run carries (ES7);
3. **detected** from the structure: the charge by the phosphate rule, then the
   spin at that charge, knowing whether the cell repeats (§ 2a.1b);

and a transport calculation's charge is 0 by rule.  ``method`` is never blank:
the catalogue's ``DFT`` is written into a template like any other value, and
SIESTA is DFT.

Every value carries where it came from and why, so each place that shows the
state can say *"-3 -- three deprotonated phosphates"* rather than a number.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Optional

from .chemistry import (
    ChemistryAnalysis,
    analyze_structure,
    count_element,
    explain_metal_spin,
    formal_charge_from_phosphates,
    total_electrons,
)
from .structure import Structure


#: How the two spin channels are solved (§ 2a.1) -- the engine-neutral words.
#: Each deck writer spells them: SIESTA ``Spin non-polarized`` / ``polarized``
#: / ``non-colinear`` / ``spin-orbit``, PySCF the R / RO / U of its SCF class.
TREATMENTS = ("restricted", "restricted-open", "unrestricted",
              "non-collinear", "spin-orbit")

#: Which theory.  SIESTA is DFT; PySCF writes ``dft.`` or ``scf.``.
METHODS = ("DFT", "HF")

#: A floating moment: the count is whatever the SCF finds (ES6).
FREE = "free"

#: ``unpaired_electrons``' choices -- whole numbers 0 to 10, and ``free``.  An
#: enum whose members are ints and one word (``engines/template.md`` § 5), so
#: the form offers a list and a template reads ``unpaired_electrons = 2``.
COUNTS = tuple(range(0, 11)) + (FREE,)

#: The calculation kinds a state is asked for.
KINDS = ("optimization", "vibration", "transport")

#: Treatments under which SIESTA stops on ``Spin.Fix``, so the count can only
#: float (``read_options.F90``: ``if (nspin .ne. 2) call die(...)``).
ALWAYS_FREE = frozenset({"non-collinear", "spin-orbit"})

#: § 2a.3 -- what each engine can run, per kind.  **Declared, not discovered**
#: (ES4): the settings gate refuses anything else by name, and the form does
#: not offer it.  Every entry was read from the engine's source.
CAPABILITY: Dict[str, Dict[str, FrozenSet[str]]] = {
    "siesta": {kind: frozenset({"restricted", "unrestricted",
                                "non-collinear", "spin-orbit"})
               for kind in KINDS},
    "pyscf": {
        # ROHF/ROKS gradients exist (`pyscf/grad/rohf.py`, `roks.py`).
        "optimization": frozenset({"restricted", "restricted-open",
                                   "unrestricted"}),
        # `pyscf/hessian/` holds rhf, rks, uhf, uks only, and the vibration
        # deck takes the analytic Hessian.
        "vibration": frozenset({"restricted", "unrestricted"}),
    },
}

#: The engines that can float a moment (ES6).  PySCF pins N-up and N-down
#: from ``mol.spin`` (``pyscf/scf/uhf.py``, ``get_occ``).
FLOATS = frozenset({"siesta"})

#: The engines that build the atoms as ONE MOLECULE whatever the structure's
#: cell: PySCF's ``gto.M`` is gas phase (``validation/pyscf.py`` says so of
#: a periodic structure).  Their calculation is finite, so its electron count
#: is a molecule's -- parity binds (ES3; ``gto.M`` refuses a mismatch) and no
#: moment floats (ES6).  Judged by the structure's axes alone, a periodic
#: structure handed to PySCF was told to float an iron moment PySCF cannot
#: float, and an odd count skipped the parity PySCF enforces (the M6 review).
MOLECULAR = frozenset({"pyscf"})

#: Why an engine cannot run a treatment -- the refusal's own words (ES4),
#: keyed ``(engine, treatment, kind)``; a ``None`` kind answers every kind.
_CANNOT = {
    ("siesta", "restricted-open", None):
        "SIESTA has no restricted open-shell formalism",
    ("pyscf", "non-collinear", None):
        "PySCF's molecular SCF here is collinear",
    ("pyscf", "spin-orbit", None):
        "PySCF's molecular SCF here has no spin-orbit coupling",
    ("pyscf", "restricted-open", "vibration"):
        "PySCF has no analytic ROHF/ROKS Hessian, and the vibration deck "
        "takes the analytic Hessian",
}


def engines_for(kind: str) -> tuple:
    """The engines that run ``kind`` -- ``CAPABILITY``'s own rows, so a form
    for an engine that does not run the kind has no state to ask for."""
    return tuple(e for e, kinds in CAPABILITY.items() if kind in kinds)


def cannot_run(engine: str, kind: str, treatment: str) -> Optional[str]:
    """Why ``engine`` cannot run ``treatment`` for ``kind``, or ``None``."""
    if treatment in CAPABILITY.get(engine, {}).get(kind, frozenset(TREATMENTS)):
        return None
    return (_CANNOT.get((engine, treatment, kind))
            or _CANNOT.get((engine, treatment, None))
            or f"{engine} does not run {treatment} for {kind}")


def offered(engine: str, kind: str) -> Dict[str, tuple]:
    """The choices a form offers for ``engine`` and ``kind`` -- exactly what
    the engine can run (ES4), in the catalogue's order."""
    can = CAPABILITY.get(engine, {}).get(kind, frozenset(TREATMENTS))
    return {
        "spin_treatment": tuple(t for t in TREATMENTS if t in can),
        "unpaired_electrons": tuple(c for c in COUNTS
                                    if c != FREE or engine in FLOATS),
    }


@dataclass(frozen=True)
class Resolved:
    """One item's answer: the value, where it came from, and why.

    ``source`` is ``stated`` (in the template -- typed, or written there from
    the run a transport calculation cites), ``implied`` (by a stated item),
    ``recorded`` (by the run the structure came out of), ``detected`` (from
    the structure) or ``rule`` (transport's charge; SIESTA's DFT).
    """
    value: Any
    source: str
    why: str

    @property
    def said(self) -> str:
        """Where the value came from, in words -- ``stated``, or ``detected:
        3 deprotonated phosphate groups`` -- the one phrasing a deck comment,
        a prep report and the form's card share."""
        return (self.source if self.why == self.source
                else f"{self.source}: {self.why}")

    def as_dict(self) -> Dict[str, Any]:
        # `said` travels too, so the card shows the deck comment's own words
        # rather than composing a second phrasing of them.
        return {"value": self.value, "source": self.source, "why": self.why,
                "said": self.said}


@dataclass(frozen=True)
class Recommended:
    """What the structure alone implies at the resolved charge (§ 2a.1b).

    Kept beside the resolved values so a check can say *"you stated
    restricted; Fe suggests unrestricted, 2S = 2"* (ES9).  ``metal`` names the
    open-d metal that decided it, or is ``None`` -- a decision a metal drove is
    a guess about coordination, and it warns until the count is stated (ES8).
    """
    spin_treatment: str
    unpaired_electrons: Any
    why: str
    metal: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {"spin_treatment": self.spin_treatment,
                "unpaired_electrons": self.unpaired_electrons,
                "why": self.why, "metal": self.metal}


@dataclass(frozen=True)
class ElectronicState:
    """The state of one calculation (§ 2a.1): four answers and what the
    structure adds.  Built only by :func:`electronic_state`."""
    net_charge: Resolved
    spin_treatment: Resolved
    unpaired_electrons: Resolved
    method: Resolved
    #: ΣZ − charge.  The valence count has the same parity: core shells hold
    #: an even number of electrons.
    n_electrons: int
    #: The calculation's system is finite: every axis isolated
    #: (``model/structure-periodicity.md`` § 2), or an engine that builds the
    #: atoms as one molecule (``MOLECULAR``).  Parity binds a finite system
    #: only (ES3).
    finite: bool
    recommended: Recommended
    #: The metals and their usual spins the decision was made from.
    facts: ChemistryAnalysis

    @property
    def pinned(self) -> Optional[int]:
        """The count a deck pins, or ``None`` when the moment floats or the
        treatment has no count to pin (restricted's 0)."""
        count = self.unpaired_electrons.value
        if count == FREE or self.spin_treatment.value == "restricted":
            return None
        return int(count)

    @property
    def metal_driven(self) -> bool:
        """Whether the count was decided from a metal's usual spin -- a guess
        about coordination that warns until the count is stated (ES8).  A
        magnetic metal in a repeating cell floats (``free``) and guesses
        nothing."""
        return (self.unpaired_electrons.source == "detected"
                and self.recommended.metal is not None
                and self.unpaired_electrons.value != FREE)

    def as_dict(self) -> Dict[str, Any]:
        """The state as JSON-able data -- the one serialisation, read by the
        forms (``/api/structure/analyze``) and the records."""
        return {
            "net_charge": self.net_charge.as_dict(),
            "spin_treatment": self.spin_treatment.as_dict(),
            "unpaired_electrons": self.unpaired_electrons.as_dict(),
            "method": self.method.as_dict(),
            "n_electrons": self.n_electrons,
            "finite": self.finite,
            "recommended": self.recommended.as_dict(),
        }


def electronic_state(struct: Structure, cfg: Any, *,
                     kind: str) -> ElectronicState:
    """THE resolver: ``cfg`` is either engine's config -- the four items are
    spelled alike in both, which is what merges them (``template.md`` § 6.3)
    -- and ``kind`` the calculation kind.

    Raises ``KeyError`` when a species label names no element: the answer is
    an electron count, and a count with an atom left out is not a smaller
    answer but a wrong one.  A caller that must not fail on it (the settings
    gate) asks ``every_label_resolves`` first and stands down -- the label
    check owns that finding.
    """
    if kind not in KINDS:
        raise ValueError(f"electronic_state: kind must be one of {KINDS}, "
                         f"not {kind!r}")
    from .template import engine_name
    engine = engine_name(type(cfg))
    facts = analyze_structure(struct)
    recorded = _recorded(struct)
    charge = _charge(struct, cfg, kind, recorded)
    n_electrons = total_electrons(struct, charge.value)
    finite = engine in MOLECULAR or all(
        k == "isolated" for k in (struct.axis_kind or ("isolated",) * 3))
    rec = recommend(struct, facts, n_electrons=n_electrons, finite=finite)
    treatment, count = _spin(cfg, rec, finite, recorded)
    return ElectronicState(
        net_charge=charge,
        spin_treatment=treatment,
        unpaired_electrons=count,
        method=_method(cfg, engine),
        n_electrons=n_electrons,
        finite=finite,
        recommended=rec,
        facts=facts,
    )


def _recorded(struct: Structure) -> Optional[Dict[str, Resolved]]:
    """The state of the run this structure came out of (ES7): the record a
    structure exported from a finished run carries, ``info.calculation``'s
    contract (``model/parse.md`` § 5b).  ``None`` for a structure no run
    left -- and for one EDITED SINCE (``structure_modified``: a geometry or
    cell op), which is no longer the structure that run came out of; an
    added or deleted atom changes the very count the record's spin was for,
    and ``web/molview.md`` § 8.4's flag says a later reader must not assume
    it.  (It was taken regardless until the M6 review.)  A value today's
    vocabulary cannot hold is not taken -- it is not a statement anybody can
    check."""
    info = getattr(struct, "info", None) or {}
    cal = info.get("calculation")
    contract = cal.get("contract") if isinstance(cal, dict) else None
    if not isinstance(contract, dict) or cal.get("structure_modified"):
        return None
    why = (f"recorded by the run this structure came out of"
           f"{' (' + str(cal['source']) + ')' if cal.get('source') else ''}")
    out: Dict[str, Resolved] = {}
    q = contract.get("net_charge")
    if isinstance(q, int) and not isinstance(q, bool):
        out["net_charge"] = Resolved(q, "recorded", why)
    t = contract.get("spin_treatment")
    if t in TREATMENTS:
        out["spin_treatment"] = Resolved(t, "recorded", why)
    c = contract.get("unpaired_electrons")
    if any(type(c) is type(m) and c == m for m in COUNTS):
        out["unpaired_electrons"] = Resolved(c, "recorded", why)
    return out or None


def _charge(struct: Structure, cfg: Any, kind: str,
            recorded: Optional[Dict[str, Resolved]]) -> Resolved:
    """The charge: transport's rule, a stated value (0 included), the run
    the structure came out of, or the phosphate rule
    (``model/chemistry.md`` § 1)."""
    if kind == "transport":
        return Resolved(0, "rule",
                        "a transport junction is neutral: its boundaries are "
                        "open and the leads set the electron number")
    stated = getattr(cfg, "net_charge", None)
    if stated is not None:
        return Resolved(int(stated), "stated", "stated")
    if recorded and "net_charge" in recorded:
        return recorded["net_charge"]
    q = formal_charge_from_phosphates(struct)
    if q:
        n = abs(q)
        return Resolved(q, "detected",
                        f"{n} deprotonated phosphate group{'s' if n != 1 else ''}")
    return Resolved(0, "detected", "no deprotonated phosphate groups")


def _method(cfg: Any, engine: str) -> Resolved:
    """``method`` is never blank: PySCF's is stated, SIESTA is DFT."""
    if engine == "siesta":
        return Resolved("DFT", "rule", "SIESTA is a density-functional code")
    return Resolved(cfg.method, "stated", "stated")


def _spin(cfg: Any, rec: Recommended, finite: bool,
          recorded: Optional[Dict[str, Resolved]]):
    """The treatment and the count, in the one order (§ 2a.1a)."""
    t_stated = getattr(cfg, "spin_treatment", None)
    c_stated = getattr(cfg, "unpaired_electrons", None)
    treatment = (Resolved(t_stated, "stated", "stated")
                 if t_stated is not None else None)
    count = (Resolved(c_stated, "stated", "stated")
             if c_stated is not None else None)

    # 1 · implied by a stated item.
    if count is None and treatment is not None:
        if t_stated == "restricted":
            count = Resolved(0, "implied", "restricted: every electron paired")
        elif t_stated in ALWAYS_FREE:
            count = Resolved(FREE, "implied",
                             f"{t_stated}: the moment floats -- SIESTA stops on "
                             f"a fixed total spin here")
    if treatment is None and count is not None:
        if c_stated == 0:
            treatment = Resolved("restricted", "implied",
                                 "no unpaired electrons")
        else:
            treatment = Resolved(
                "unrestricted", "implied",
                "a floating moment needs two spin channels" if c_stated == FREE
                else f"{c_stated} unpaired electron"
                     f"{'s' if c_stated != 1 else ''} need two spin channels")

    # 2 · recorded by the run the structure came out of (ES7) -- the whole
    # recorded pair when nothing is stated, or the count when the stated
    # treatment is the one that run used.  (A stated count always has its
    # treatment by now: step 1 implies one.)
    if recorded:
        if treatment is None and count is None:
            treatment = recorded.get("spin_treatment")
            count = recorded.get("unpaired_electrons")
        elif count is None \
                and recorded.get("spin_treatment") is not None \
                and recorded["spin_treatment"].value == treatment.value:
            count = recorded.get("unpaired_electrons")

    # 3 · detected from the structure.
    if treatment is None:
        treatment = Resolved(rec.spin_treatment, "detected", rec.why)
    if count is None:
        if rec.spin_treatment != "restricted" \
                or treatment.value == "restricted":
            count = Resolved(rec.unpaired_electrons, "detected", rec.why)
        elif finite:
            # A two-channel treatment the structure does not suggest: the
            # structure's own count is 0, pinned -- a constrained singlet.
            count = Resolved(0, "detected",
                             "the structure has no unpaired electron: 2S = 0, "
                             "a constrained singlet under "
                             f"{treatment.value}")
        else:
            count = Resolved(FREE, "detected",
                             "a repeating cell: the moment floats to its own "
                             "value")
    return treatment, count


#: The detected count for an open-d metal: a starting guess per element --
#: for most, its most common oxidation state's spin ("Mn(II) in
#: solution"); for Fe, the four-coordinate porphyrin case the row names,
#: where high-spin Fe(II) is the commoner state elsewhere.  A guess about
#: coordination, said as THE STARTING GUESS (user, 2026-09-29, keeping Fe's
#: 2), which is why a decision made from it warns until the count is
#: stated (ES8).  Second- and third-row
#: and f-block metals fall through to 2.  (Cu is a noble s1 metal here, not
#: an open-d one -- Cu(II)'s d9 doublet is the odd count's own answer.)
USUAL_COUNT: Dict[str, int] = {
    "Fe": 2,    # Fe(II), intermediate spin (S=1, 4-coordinate porphyrin --
                # the hemeC case).  High-spin Fe(II) is 4.
    "Mn": 5,    # Mn(II), high spin S=5/2 -- the overwhelming bio/aqueous case.
    "Co": 3,    # Co(II), high spin S=3/2 -- the common octahedral / aqueous /
                # weak-field case; low spin (S=1/2) needs strong-field ligands.
    "Ni": 0,    # Ni(II), square-planar low spin d8 (S=0) -- the common case
                # in metalloproteins; octahedral high spin is 2.
    "Cr": 3,    # Cr(III), d3 S=3/2.
    "V":  3,    # V(II), d3 S=3/2.
    "Ti": 2,    # Ti(II), d2 S=1.
    "Sc": 1,    # Sc(II), d1 S=1/2 -- rare; Sc(III) is d0 and closed-shell.
}

#: The size at which a noble-metal cluster's s-band delocalises and closes the
#: shell for an even count.  Four is the conservative cutoff -- what published
#: Au transport and surface DFT overwhelmingly does; Au2 / Au3 are left to the
#: electron count.
NOBLE_CLUSTER_THRESHOLD = 4


def recommend(struct: Structure, facts: ChemistryAnalysis, *,
              n_electrons: int, finite: bool) -> Recommended:
    """What the structure implies at this electron count (§ 2a.1b) -- the
    detection table.  The first row that matches wins, so an open-d metal
    decides even beside gold.
    """
    open_d, nobles = facts.open_d_metals, facts.noble_metals
    odd = n_electrons % 2 == 1

    if open_d:
        metal = open_d[0]
        if not finite:
            return Recommended(
                "unrestricted", FREE,
                f"{metal} is an open-d metal in a repeating cell: the moment "
                f"floats to its own value", metal=metal)
        count = USUAL_COUNT.get(metal, 2)
        if count % 2 != n_electrons % 2:
            count = count + 1 if count == 0 else count - 1
            matched = (f" (moved to match the {n_electrons}-electron count's "
                       f"parity)")
        else:
            matched = ""
        label = explain_metal_spin(metal, count)
        return Recommended(
            "unrestricted", count,
            f"{metal} is an open-d metal: 2S = {count} is the starting guess"
            f"{matched}{' -- ' + label if label else ''}.  The right count "
            f"depends on the coordination, not on the element: verify it "
            f"against experiment and state it to confirm",
            metal=metal)

    if not finite:
        if nobles:
            return Recommended(
                "restricted", 0,
                f"metallic {', '.join(nobles)} in a repeating cell: the s-band "
                f"delocalises and no moment forms")
        return Recommended(
            "restricted", 0,
            "a repeating cell with no open-d metal: the count per cell is not "
            "a spin")

    if nobles:
        n_noble = sum(count_element(struct, m) for m in nobles)
        if n_noble >= NOBLE_CLUSTER_THRESHOLD and not odd:
            return Recommended(
                "restricted", 0,
                f"a metallic {', '.join(nobles)} cluster ({n_noble} atoms, an "
                f"even count): the s-band delocalises and no moment forms")
        if n_noble == 1 and odd:
            # By the COUNT, not the atom's configuration: a bare Au atom's
            # doublet and a Cu(II) complex's d9 one are the same answer,
            # and "nd10 (n+1)s1" named Cu(0) as the reason for a Cu(II)
            # complex until the M6 review.
            return Recommended(
                "unrestricted", 1,
                f"one {nobles[0]} atom and an odd electron count "
                f"({n_electrons}): one unpaired electron")

    if odd:
        return Recommended("unrestricted", 1,
                           f"an odd electron count ({n_electrons}): one "
                           f"unpaired electron")
    return Recommended("restricted", 0,
                       f"an even electron count ({n_electrons}) and no open-d "
                       f"metal")
