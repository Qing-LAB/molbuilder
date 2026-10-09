"""What a cited run contributes to a transport description — the DEFAULTS.

Contract: [`engines/transport.md` § 2a.7](?doc=engines/transport.md), ruling 1
(*"the relaxation DEFAULTS the Class A values; it does not seal them"*), and
[`engines/template.md` § 6.4](?doc=engines/template.md), the `citation` marker.

**One function, called once, at `jobset init`.** It reads what the citation
answers with — a finished run's own deck (§ 3.1: one shape, decision 7) —
and returns the
:class:`~molbuilder.config.siesta.SiestaConfig` that `init` writes the
template from. After that the template is the answer and this module has no
further part: `prep` reads the file, not the citation.

**Why this is `init`'s job and not `prep`'s.** The physics (§ 2a.7) requires the leads and the device to agree
*with each other*, which needs the value to be **single**, not **inherited**.
One template shared by every stage satisfies that exactly, and permits the
ordinary practice of relaxing with a cheap description and transporting with an
accurate one.

**What is taken, and what is not.** Only the shared electronic description —
the items the catalogue tags `citation = ["transport"]`. The geometry,
the region partition and the pseudopotentials still travel with the citation
at prep: they are not parameters, they are the structure (§ 2a.3's
*"results that propagate"*, and § 2a.9's structure input).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:                                    # pragma: no cover
    from ..config.siesta import SiestaConfig

#: What the cited deck answers -> the catalogue's own spelling for it.
#:
#: The left side is :class:`~molbuilder.parse.fdf.FdfParams`, which
#: reads an `.fdf`; the right is :class:`SiestaConfig`, which is the
#: catalogue's vocabulary. The pairs here, and the k-point mesh below them
#: -- ``kgrid`` and its offset ``kgrid_displacement``, rows tagged
#: ``citation = ["transport"]`` too -- which is separate because it is the
#: one value not copied verbatim: the transport axis is laid on by the rule
#: (:func:`_apply_kgrid`).
#: Keeping the mapping beside the reader is what stops it drifting from the
#: declaration.
_FROM_DECK = {
    "basis_size":               "basis_size",
    "mesh_cutoff_ry":           "mesh_cutoff",
    "energy_shift_ry":          "pao_energy_shift",
    "electronic_temperature_k": "electronic_temperature",
    "xc_functional":            "xc_functional",
    "xc_authors":               "xc_authors",
    # THE SPIN THE CITED RUN CARRIED (`science/chemistry-correctness.md`
    # § 2a, ES7): shared by every rung, defaulted from the run the junction
    # was relaxed in.  The CHARGE is not a
    # transport item -- a cited charge is refused instead
    # (:func:`siesta_config_from_citation`).
    "spin_treatment":           "spin_treatment",
    "unpaired_electrons":       "unpaired_electrons",
    # HOW THE CITED RUN'S SCF CONVERGED -- the mixer and the criteria,
    # each SCF stage's starting value (TD6, the user 2026-10-09: "go with
    # way 1").  Not shared: every SCF stage carries its own and may change
    # it on its tab (`engines/transport.md` § 2a.13).
    "mixing_weight":            "mixing_weight",
    "pulay_history":            "pulay_history",
    "dm_tolerance":             "dm_tolerance",
    "dm_energy_tolerance_ev":   "dm_energy_tolerance",
    "scf_energy_converge":      "scf_energy_converge",
}

def _apply_kgrid(kw: dict, kgrid, shifts=None) -> None:
    """The cited run's k-point mesh, as the transport calculation takes it:
    the transverse pair and its offset carry over, and the transport axis is
    laid on by the rule -- one point, no offset -- through the k-point mesh's
    own door (``kmesh.with_fixed``; `engines/siesta.md` § 6.1).

    ONE rule for the cited deck.  The cited
    run was a closed periodic calculation and sampled all three axes; a
    transport calculation does not sample the transport axis at all.  Laid on
    here, where the value is born, so the template states what every rung
    writes rather than a component the form would show refused.

    The transmission grid starts at the same transverse pair: tbtrans would
    inherit the SCF's grid on its own if the deck said nothing
    (`m_tbt_kpoint.F90:800-812`).  The grid is KNOWN at this moment, so it is written
    down.  A starting point, not an answer: a grid converged for a total
    energy is routinely too coarse for a transmission.

    The OFFSET is carried: every rung writes it, and TranSIESTA stops on a
    lead whose offset differs from the device's.  A record that states none
    leaves the item unanswered -- the documented default fills it at prep.
    """
    from ..kmesh import with_fixed
    try:
        counts = tuple(int(v) for v in kgrid)
    except (TypeError, ValueError):
        return
    if len(counts) != 3:
        return
    kw["kgrid"] = with_fixed("kgrid", counts, "transport")
    kw["tbt_k_grid"] = with_fixed("tbt_k_grid", counts, "transport")
    try:
        offset = tuple(float(v) for v in shifts) if shifts else None
    except (TypeError, ValueError):
        offset = None
    if offset is not None and len(offset) == 3:
        kw["kgrid_displacement"] = with_fixed("kgrid_displacement", offset,
                                              "transport")


@dataclass(frozen=True)
class CitationAnswers:
    """What the cited run answers of the shared electronic description
    (`engines/transport.md` § 3.1, § 3.8.1): ``values`` in ``SiestaConfig``'s
    field names, ``source`` ``"deck"`` (the run's own deck) or ``"none"`` (a
    deck the unit door could not read), and ``source_name`` the file the
    answers were read from."""
    values: Dict[str, Any]
    source: str
    source_name: str = ""
    #: The charge the cited run carried -- 0 when it answers none.  Not a
    #: value of the transport template (the junction is neutral by rule);
    #: read so a charged citation can be refused by name (ES7).
    net_charge: int = 0


def citation_answers(cite_dir) -> CitationAnswers:
    """Read the cited directory once (`engines/transport.md` § 3.8.0: at
    `init`, into this calculation's own template): the cited run's deck
    answers the electronic description (§ 3.1 -- a citation is a finished
    relaxation run of molbuilder's own).
    """
    from .compose import classify_citation
    from ..parse.fdf import parse_fdf_params

    kw: Dict[str, Any] = {}
    cited = classify_citation(Path(cite_dir))
    p = None
    if cited.deck is not None:
        from ..units import UnknownUnit
        try:
            p = parse_fdf_params(cited.deck.read_text(encoding="utf-8",
                                                      errors="replace"))
        except UnknownUnit:
            # A deck the unit door refuses answers nothing here; the
            # refusal itself is prep's to raise, by name.
            p = None
    if p is None:
        # A deck the unit door refuses answers nothing here; the refusal
        # itself is prep's to raise, by name.
        source, source_name = "none", ""
        charge = 0
    else:
        for src, dst in _FROM_DECK.items():
            v = getattr(p, src, None)
            if v is not None:
                kw[dst] = v
        if getattr(p, "kgrid", None):
            _apply_kgrid(kw, p.kgrid,      # the one rule, both sources
                         getattr(p, "kgrid_displacement", None))
        source, source_name = "deck", cited.deck.name
        charge = int(p.net_charge)
    # A FIXED COUNT A TRANSPORT RUNG CANNOT HOLD is not defaulted
    # (`engines/transport.md` § 3.1's spin note, T-F20): TranSIESTA stops on
    # a fixed total spin, so under `unrestricted` the cited run's number is
    # not a value this calculation can take -- the count is left blank and
    # floats by the kind's own rule (`electronic_state.count_must_float`),
    # which each rung's deck says.
    from ..electronic_state import FREE, count_must_float
    from ..template import catalogue, is_member, offered, one
    # ...NOR A TREATMENT IT CANNOT RUN (non-collinear, spin-orbit: TranSIESTA
    # stops on more than two spin components) -- the treatment and its count
    # are left blank and worked out on the junction, which each rung's deck
    # says (`template.md` § 6.3a).
    _can = offered(one(catalogue(), "spin_treatment", engine="siesta"),
                   "siesta", "transport")
    if (kw.get("spin_treatment") is not None
            and not is_member(kw["spin_treatment"], _can)):
        kw.pop("spin_treatment")
        kw.pop("unpaired_electrons", None)
    if (count_must_float("siesta", "transport", kw.get("spin_treatment"))
            and kw.get("unpaired_electrons") not in (None, FREE)):
        kw.pop("unpaired_electrons")
    if "mesh_cutoff" in kw:
        kw["mesh_cutoff"] = float(kw["mesh_cutoff"])
    return CitationAnswers(values=kw, source=source, source_name=source_name,
                           net_charge=charge)


def siesta_config_from_citation(cite_dir, *, label: str, blank=(),
                                **chosen) -> "SiestaConfig":
    """The config `jobset init` and the describe door write a transport
    template from: the citation's answers (:func:`citation_answers`), then
    what the person chose on the shared panel (``chosen``, in the same
    field names) laid over them -- the person may change any of it, for
    all five rungs at once (`engines/transport.md` § 2a.7 ruling 1).

    ``blank`` names the items the person emptied on the panel: a blank is
    not chosen (`web/form-schema.md` § 1.1), so the citation's value is not
    applied to them -- for the electronic state that blank is the answer
    "work it out" (`science/chemistry-correctness.md` § 2a).
    """
    from ..config.siesta import SiestaConfig
    answers = citation_answers(cite_dir)
    if answers.net_charge:
        # ES7 (`science/chemistry-correctness.md` § 2a): a transport
        # calculation's boundaries are OPEN -- the leads set the electron
        # number -- so the junction must be neutral, and a geometry relaxed
        # at another charge is not this junction's.
        raise ValueError(
            f"the cited run carried a net charge of {answers.net_charge:+d} "
            f"({answers.source_name or 'its record'}), and a transport "
            f"calculation cannot: its boundaries are open and the leads set "
            f"the electron number, so the junction must be neutral "
            f"(engines/transport.md § 2a.7).  Relax the junction neutral and "
            f"cite that run.")
    kw = {k: v for k, v in answers.values.items() if k not in blank}
    kw.update(_the_persons(chosen))
    # UNDER BOTH, the kind's own recommendations (`engines/template.md`
    # § 6.3a): every door that writes a template lays them under what was
    # given -- the citation's answers here, then the person's.
    import dataclasses
    from ..template import apply_recommended
    base = apply_recommended(SiestaConfig(system_label=label), "transport",
                             engine="siesta")
    return dataclasses.replace(base, **kw)


def _the_persons(chosen) -> Dict[str, Any]:
    """What the person chose on the shared panel, with what follows from it
    by the one k-mesh rule (:func:`_apply_kgrid`): a k-point mesh they set
    starts the transmission grid at its transverse pair, as the cited one
    does.  Without it, a changed SCF mesh would leave the transmission on
    the cited one."""
    mine = {k: v for k, v in chosen.items() if v is not None}
    if mine.get("kgrid") is not None:
        _apply_kgrid(mine, mine["kgrid"])
    return mine


def transport_template_text(cite_dir, *, label: str, blank=(),
                            **chosen) -> str:
    """The text of a transport description's template -- **the one door
    both roads write it through** (`jobset init` and the browser's describe
    door), so the file cannot differ by the road that made it.

    Its values are :func:`siesta_config_from_citation`'s: the citation's
    answers with the person's choices laid over them.  **A `citation` row
    neither the citation nor the person answered is written VALUELESS**
    (`engines/transport.md` § 3.8.3; `template.md` § 6.6 obligation 2): a
    deck the unit door could not read answers no basis, and a value written
    there would claim a run said something no run said -- after which no
    surface can
    tell the person's 300 Ry from nobody's.  What `prep` then writes into
    the deck for such a row is the documented default, marked as nobody's
    choice (§ 6.6 obligation 4) -- that is `prep`'s to do, not this file's
    to pre-empt.

    ``blank`` (:func:`siesta_config_from_citation`) is written valueless
    too: a row the person emptied on the shared panel is not chosen -- a
    spin among them the junction's to work out at prep, as the chemistry
    card beside it says.
    """
    from .. import template as _T
    cfg = siesta_config_from_citation(cite_dir, label=label, blank=blank,
                                      **chosen)
    cited = citation_answers(cite_dir)
    mine = _the_persons(chosen)
    answered = (set(cited.values) - set(blank)) | set(mine)
    unanswered = sorted(
        it.name for it in _T.select(_T.catalogue(), engine="siesta",
                                    citation=True)
        if "transport" in it.citation and it.name not in answered)
    # WHERE EACH VALUE CAME FROM (`template.md` § 6.6 obligation 2): the
    # citation's own answers -- from the run's deck -- then what the person chose on
    # the shared panel over them, and the calculation's own name.  A value
    # the panel holds as the citation answered it is the citation's: the
    # panel is drawn holding those answers and sends what it holds.  Every
    # other item is nobody's choice (`default`).
    via = {"deck": "cited"}.get(cited.source)
    sources = {k: via for k in cited.values if via and k not in blank}
    sources.update({k: "person" for k, v in mine.items()
                    if not (k in sources and cited.values[k] == v)})
    sources["system_label"] = "person"
    return _T.template_with_values(cfg, engine="siesta",
                                   calculation="transport",
                                   valueless=unanswered, sources=sources)
