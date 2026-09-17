"""What a cited run contributes to a transport description — the DEFAULTS.

Contract: [`engines/transport.md` § 2a.7](?doc=engines/transport.md), ruling 1
(*"the relaxation DEFAULTS the Class A values; it does not seal them"*), and
[`engines/template.md` § 6.4](?doc=engines/template.md), the `citation` marker.

**One function, called once, at `jobset init`.** It reads the cited run's own
deck and returns the :class:`~molbuilder.config.siesta.SiestaConfig` that
`init` writes the template from. After that the template is the answer and
this module has no further part: `prep` reads the file, not the citation.

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
#: catalogue's vocabulary. The pairs are the seven rows tagged
#: ``citation = ["transport"]`` — *this* is the list that marker names, and
#: keeping the mapping beside the reader is what stops it drifting from the
#: declaration.
_FROM_DECK = {
    "basis_size":               "basis_size",
    "mesh_cutoff_ry":           "mesh_cutoff",
    "energy_shift_ry":          "pao_energy_shift",
    "electronic_temperature_k": "electronic_temperature",
    "xc_functional":            "xc_functional",
    "xc_authors":               "xc_authors",
}


def siesta_config_from_citation(cite_dir, *, label: str) -> "SiestaConfig":
    """The config `jobset init` writes a transport template from.

    *cite_dir* is the directory being cited; *label* names the calculation.

    A citation that carries a **deck** (form A) answers the electronic
    description, and every one of those answers becomes a template value the
    person can afterwards change. A citation that carries only a **labeled
    structure** (form B) answers none of it — there is no deck to read — and
    the template is written from the catalogue's own defaults instead, which
    is the honest result rather than a failure: the person then chooses the
    description themselves, which is what form B always meant.
    """
    from ..config.siesta import SiestaConfig
    from .compose import classify_citation
    from ..parse.fdf import parse_fdf_params

    kw = {}
    cited = classify_citation(Path(cite_dir))
    if cited.deck is not None:
        p = parse_fdf_params(cited.deck.read_text(encoding="utf-8",
                                                  errors="replace"))
        for src, dst in _FROM_DECK.items():
            v = getattr(p, src, None)
            if v is not None:
                kw[dst] = v
        if getattr(p, "kgrid", None):
            # THE ONE VALUE THAT IS NOT COPIED VERBATIM.  The cited run was a
            # closed periodic calculation and sampled all three axes; a
            # transport calculation does not sample the transport axis at
            # all, because that axis is the open boundary.  So the
            # transverse pair carries over and the third component is 1 —
            # forced here, where the value is born, rather than corrected
            # downstream by each consumer.
            kx, ky, _kz = p.kgrid
            kw["kgrid"] = (int(kx), int(ky), 1)
            # AND THE TRANSMISSION GRID, to the same transverse pair.
            #
            # tbtrans would fall back to the SCF's grid on its own if the deck
            # said nothing (`m_tbt_kpoint.F90:800-812`), and that fallback used
            # to be spelled `0 0 0` in the form -- a sentinel sitting in a
            # field labelled "k-grid", where every value is a scientific fact
            # and `0 0 0` is not one of them.  The grid is KNOWN at this
            # moment, so it is written down instead of implied, and the deck
            # states what it integrates over.
            #
            # It is a starting point, not an answer: a grid converged for a
            # total energy is routinely too coarse for a transmission, and the
            # row's help says how to converge it.
            kw["tbt_k_grid"] = (int(kx), int(ky), 1)

    # `mesh_cutoff` is declared as a float and an .fdf usually states an
    # integer; the type is the catalogue's to decide, not the deck's.
    if "mesh_cutoff" in kw:
        kw["mesh_cutoff"] = float(kw["mesh_cutoff"])
    return SiestaConfig(system_label=label, **kw)
