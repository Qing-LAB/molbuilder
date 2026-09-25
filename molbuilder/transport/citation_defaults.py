"""What a cited run contributes to a transport description — the DEFAULTS.

Contract: [`engines/transport.md` § 2a.7](?doc=engines/transport.md), ruling 1
(*"the relaxation DEFAULTS the Class A values; it does not seal them"*), and
[`engines/template.md` § 6.4](?doc=engines/template.md), the `citation` marker.

**One function, called once, at `jobset init`.** It reads what the citation
answers with — a finished run's own deck, or the settings a saved structure
recorded from the run it came out of (§ 3.1's three cases) — and returns the
:class:`~molbuilder.config.siesta.SiestaConfig` that `init` writes the
template from. After that the template is the answer and this module has no
further part: `prep` reads the file, not the citation.

**Why this is `init`'s job and not `prep`'s.** Until 2026-09-16 the citation
was read at *prep*, every time, and the items it answered were written
valueless into nothing (transport had no template at all). That is the SEALED
reading — the person could not change a value because there was nowhere to put
one. § 2a.7 reversed it: the physics requires the leads and the device to agree
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
#: catalogue's vocabulary. SIX pairs here and the k-grid below them, which
#: is the seventh row tagged ``citation = ["transport"]`` and is separate
#: because it is the one value not copied verbatim (:func:`_apply_kgrid`).
#: Keeping the mapping beside the reader is what stops it drifting from the
#: declaration.
_FROM_DECK = {
    "basis_size":               "basis_size",
    "mesh_cutoff_ry":           "mesh_cutoff",
    "energy_shift_ry":          "pao_energy_shift",
    "electronic_temperature_k": "electronic_temperature",
    "xc_functional":            "xc_functional",
    "xc_authors":               "xc_authors",
}

#: The same six, arriving from a RECORD instead of a deck -> the catalogue's
#: spelling.  The k-grid is the seventh and goes through the same
#: :func:`_apply_kgrid` as the deck's.
#:
#: The record's field names against ``SiestaConfig``'s -- the ONE table,
#: kept beside `_siesta_contract` in `parse/contract.py` (which writes the
#: record in those names) so the writer and this reader cannot drift.
from molbuilder.parse.contract import RECORD_TO_SIESTA_FIELD as _FROM_RECORD

#: The same six, arriving from a RECORD instead of a deck -> the catalogue's
#: spelling.  The k-grid is the seventh and goes through the same
#: :func:`_apply_kgrid` as the deck's.
#:
#: `parse/contract.py::_siesta_contract` reads the very same `FdfParams` this
#: module does and writes the block in ``TransportConfig``'s field names, so
#: this table is `_FROM_DECK` with its left column respelled -- of the seven
#: values only two are actually spelled differently (`mesh_cutoff_ry` ->
#: `siesta_mesh_cutoff_ry`, `kgrid` -> `k_mesh_transverse`).  It is written out rather than derived
#: because the two vocabularies are a fact about a class that is being
#: retired (§ 2a.14: `TransportConfig` goes when the NEGF block is tabled),
#: and a clever derivation would outlive the thing it derives from.
_FROM_RECORD = {
    "basis_size":               "basis_size",
    "siesta_mesh_cutoff_ry":    "mesh_cutoff",
    "energy_shift_ry":          "pao_energy_shift",
    "electronic_temperature_k": "electronic_temperature",
    "xc_functional":            "xc_functional",
    "xc_authors":               "xc_authors",
}


def _apply_kgrid(kw: dict, kgrid) -> None:
    """The transverse pair carries over; the transport axis is 1.

    ONE rule, both sources.  The cited run was a closed periodic calculation
    and sampled all three axes; a transport calculation does not sample the
    transport axis at all, because that axis is the open boundary.  Forced
    here, where the value is born, rather than corrected downstream by each
    consumer -- and shared between the deck and the record paths so they
    cannot come to force it differently.

    The transmission grid starts at the same transverse pair: tbtrans would
    inherit the SCF's grid on its own if the deck said nothing
    (`m_tbt_kpoint.F90:800-812`), and that fallback used to be spelled
    `0 0 0` in the form -- a sentinel in a field where every value is a
    scientific fact.  The grid is KNOWN at this moment, so it is written
    down.  A starting point, not an answer: a grid converged for a total
    energy is routinely too coarse for a transmission.
    """
    try:
        kx, ky = int(kgrid[0]), int(kgrid[1])
    except (TypeError, ValueError, IndexError):
        return
    kw["kgrid"] = (kx, ky, 1)
    kw["tbt_k_grid"] = (kx, ky, 1)


@dataclass(frozen=True)
class CitationAnswers:
    """What the cited directory answers of the shared electronic description
    (`engines/transport.md` § 3.1's three cases, § 3.8.1): ``values`` in
    ``SiestaConfig``'s field names, ``source`` one of ``"deck"`` (a finished
    run's own deck), ``"record"`` (a saved structure that remembers its
    run) or ``"none"`` (a saved structure that answers nothing), and
    ``source_name`` the file the answers were read from."""
    values: Dict[str, Any]
    source: str
    source_name: str = ""


def citation_answers(cite_dir) -> CitationAnswers:
    """Read the cited directory once (`engines/transport.md` § 3.8.0: at
    `init`, into this calculation's own template).

    * **a finished run** -- its deck answers the electronic description;
    * **a saved structure that remembers its run** -- its sidecar carries
      that run's own settings, recorded by the Results tab at export;
    * **a saved structure** -- answers none of it.

    *(This said the second and third were one case until 2026-09-23.)*
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
        from .compose import recorded_contract_of
        recorded = recorded_contract_of(cited)
        block = dict((recorded or {}).get("contract") or {})
        for src, dst in _FROM_RECORD.items():
            v = block.get(src)
            if v is not None:
                kw[dst] = v
        _apply_kgrid(kw, block.get("k_mesh_transverse"))
        source = "record" if recorded else "none"
        source_name = str((recorded or {}).get("source") or "")
    else:
        for src, dst in _FROM_DECK.items():
            v = getattr(p, src, None)
            if v is not None:
                kw[dst] = v
        if getattr(p, "kgrid", None):
            _apply_kgrid(kw, p.kgrid)      # the one rule, both sources
        source, source_name = "deck", cited.deck.name
    if "mesh_cutoff" in kw:
        kw["mesh_cutoff"] = float(kw["mesh_cutoff"])
    return CitationAnswers(values=kw, source=source, source_name=source_name)


def siesta_config_from_citation(cite_dir, *, label: str,
                                **chosen) -> "SiestaConfig":
    """The config `jobset init` and the describe door write a transport
    template from: the citation's answers (:func:`citation_answers`), then
    what the person chose on the shared panel (``chosen``, in the same
    field names) laid over them -- the person may change any of it, for
    all five rungs at once (`engines/transport.md` § 2a.7 ruling 1).
    """
    from ..config.siesta import SiestaConfig
    kw = dict(citation_answers(cite_dir).values)
    kw.update({k: v for k, v in chosen.items() if v is not None})
    return SiestaConfig(system_label=label, **kw)


def transport_template_text(cite_dir, *, label: str, **chosen) -> str:
    """The text of a transport description's template -- **the one door
    both roads write it through** (`jobset init` and the browser's describe
    door), so the file cannot differ by the road that made it.

    Its values are :func:`siesta_config_from_citation`'s: the citation's
    answers with the person's choices laid over them.  **A `citation` row
    neither the citation nor the person answered is written VALUELESS**
    (`engines/transport.md` § 3.8.3; `template.md` § 6.6 obligation 2): a
    hand-built structure answers no basis, and a value written there would
    claim a run said something no run said -- after which no surface can
    tell the person's 300 Ry from nobody's.  What `prep` then writes into
    the deck for such a row is the documented default, marked as nobody's
    choice (§ 6.6 obligation 4) -- that is `prep`'s to do, not this file's
    to pre-empt.

    Until 2026-09-24 the CLI's `init` wrote those rows WITH the class
    default while the describe door wrote them valueless: two roads, two
    files, for one description.
    """
    from .. import template as _T
    cfg = siesta_config_from_citation(cite_dir, label=label, **chosen)
    answered = set(citation_answers(cite_dir).values) | {
        k for k, v in chosen.items() if v is not None}
    unanswered = sorted(
        it.name for it in _T.select(_T.catalogue(), engine="siesta",
                                    citation=True)
        if "transport" in it.citation and it.name not in answered)
    return _T.template_with_values(cfg, engine="siesta",
                                   calculation="transport",
                                   valueless=unanswered)
