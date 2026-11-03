"""THE SETTINGS A SIESTA DECK STATES, in the catalogue's names -- SIESTA's
reading for the recorded contract (`parse.contract.contract_of`,
`model/parse.md` § 5b): the ``.fdf`` through the shipped parameter parser
(``parse_fdf_params``) -- basis, energy shift, XC spelling, mesh cutoff,
k-grid, electronic temperature, the electronic state, and how its SCF
converged.  The deck's catalogue row names it for ``contract``
(`runfiles.WRITTEN`), so the contract door picks it for a deck in SIESTA's
role.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def contract_fields(deck: Path) -> Optional[Dict[str, Any]]:
    """The settings ``deck`` states, ``{catalogue name: value}`` in the
    catalogue's units -- only the fields it states -- or ``None`` for a deck
    that does not read, or states a unit this build cannot convert: no
    contract rather than a wrong one."""
    from ..fdf import parse_fdf_params, read_deck
    from ...units import UnknownUnit
    try:
        # THE DECK AS THE ENGINE READS IT (`read_deck`): a block read from
        # a file beside it -- a vibration call's coordinates -- inlined.
        text = read_deck(deck)
    except OSError:
        return None
    try:
        p = parse_fdf_params(text)
    except UnknownUnit:
        # No contract rather than a wrong one: recording a number this
        # build could not convert would put a value in the record that
        # the deck does not state.
        return None
    contract = {k: v for k, v in {
        # The catalogue's names (`RECORDED_FIELDS`) in the catalogue's
        # units -- the parse converts each to it on the way in.
        "basis_size":               p.basis_size,
        "pao_energy_shift":         p.energy_shift_ry,
        "xc_functional":            p.xc_functional,
        "xc_authors":               p.xc_authors,
        "mesh_cutoff":              p.mesh_cutoff_ry,
        # THE K-POINT MESH (`K_MESH_RECORD_KEYS`): the run's counts, all
        # three, and its offset (`engines/siesta.md` § 6.1).
        "kgrid":                    (list(p.kgrid) if p.kgrid else None),
        "kgrid_displacement":       (list(p.kgrid_displacement)
                                     if p.kgrid_displacement else None),
        "electronic_temperature":   p.electronic_temperature_k,
        # What the run carried -- SIESTA's defaults answer what the deck
        # leaves out, so these are always known (ES7's hand-over reads them).
        "net_charge":               p.net_charge,
        "spin_treatment":           p.spin_treatment,
        "unpaired_electrons":       p.unpaired_electrons,
        # How its SCF converged (`SCF_RECORD_KEYS`).
        "mixing_weight":            p.mixing_weight,
        "pulay_history":            p.pulay_history,
        "dm_tolerance":             p.dm_tolerance,
        "h_tolerance":              p.h_tolerance_ev,
        "dm_energy_tolerance":      p.dm_energy_tolerance_ev,
        "scf_energy_converge":      p.scf_energy_converge,
    }.items() if v is not None}
    return contract


__all__ = ["contract_fields"]
