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

from pathlib import Path
from typing import TYPE_CHECKING

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


def siesta_config_from_citation(cite_dir, *, label: str) -> "SiestaConfig":
    """The config `jobset init` writes a transport template from.

    *cite_dir* is the directory being cited; *label* names the calculation.

    THE THREE CASES OF § 3.1, and each answers what it has:

    * **a finished run** — its deck answers the electronic description, and
      every answer becomes a template value the person may afterwards change;
    * **a saved structure that remembers its run** — its sidecar carries that
      run's own settings, recorded by the Results tab at export, and they
      answer exactly as a deck's do;
    * **a saved structure** — answers none of it, and the template is written
      from the catalogue's defaults. That is the honest result rather than a
      failure: nothing has been measured, so the person chooses.

    *(This said the second and third were one case — "a labeled structure
    answers none of it, there is no deck to read". § 3.1 of the live contract
    listed two forms where the 2026-08-29 ruling gave three, so this file was
    written to a contract missing the middle one. Restored 2026-09-23.)*
    """
    from ..config.siesta import SiestaConfig
    from .compose import classify_citation
    from ..parse.fdf import parse_fdf_params

    kw = {}
    cited = classify_citation(Path(cite_dir))
    p = None
    if cited.deck is not None:
        from ..units import UnknownUnit
        try:
            p = parse_fdf_params(cited.deck.read_text(encoding="utf-8",
                                                      errors="replace"))
        except UnknownUnit:
            # A default this build cannot convert is not offered; the
            # person fills the field themselves rather than starting from
            # a number wrong by a fixed ratio.
            p = None
    if p is None:
        # NO DECK TO READ -- either there is none, or the one there states a
        # unit this build cannot convert (above) -- BUT THE PAIR MAY REMEMBER
        # ONE.  § 3.1's middle case.  A form-A citation reaches here only by
        # the second road, and `recorded_contract_of` answers None for it, so
        # an unreadable deck still fills nothing rather than falling back to
        # some other run's numbers.
        # A structure exported from the Results tab after a run carries that
        # run's own settings in its sidecar (`info.calculation`), and they
        # answer here exactly as a deck's do: the reading transferred to the
        # recorded copy.
        #
        # THIS WAS RULED IN AND THEN LOST.  `archive/2026-09-01-transport-
        # design.md` § 4.1b (2026-08-29) gave the condition THREE shades and
        # § 3.1 of the live contract carried only two, so when the parameter
        # path moved onto the template on 2026-09-16 the code that acted on
        # the middle one -- `stages.config_for` -- had nothing to be
        # preserved against and was left without a caller.  Everything that
        # READS the record survived: the tab still said "contract RECORDED",
        # `compose` still warned the record might be stale.  Measured
        # 2026-09-23 before this was restored: a pair recording 400 Ry, TZP
        # and a 4x4 mesh produced a template of 300 Ry, DZP and Gamma-only.
        from .compose import recorded_contract_of
        recorded = recorded_contract_of(cited)
        block = dict((recorded or {}).get("contract") or {})
        for src, dst in _FROM_RECORD.items():
            v = block.get(src)
            if v is not None:
                kw[dst] = v
        _apply_kgrid(kw, block.get("k_mesh_transverse"))
    else:
        for src, dst in _FROM_DECK.items():
            v = getattr(p, src, None)
            if v is not None:
                kw[dst] = v
        if getattr(p, "kgrid", None):
            _apply_kgrid(kw, p.kgrid)      # the one rule, both sources

    # `mesh_cutoff` is declared as a float and an .fdf usually states an
    # integer; the type is the catalogue's to decide, not the deck's.
    if "mesh_cutoff" in kw:
        kw["mesh_cutoff"] = float(kw["mesh_cutoff"])
    return SiestaConfig(system_label=label, **kw)
