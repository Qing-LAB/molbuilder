"""``contract_of`` — the electronic contract a directory's deck records.

The ONE interface behind the Results tab's contract recording
(`plans/structure-info-plan.md` I5): given a directory, find the one
engine deck in it and answer the electronic contract it states, in the
exact field names ``TransportConfig`` speaks — so a recorded block
(`info.calculation`) fills a transport config 1:1 when a pair carrying
it is cited (`transport-design.md` § 4.1b, the recorded-contract
shade).

Per-engine, behind one door:

* **SIESTA** — the ``.fdf`` through the shipped parameter parser
  (``parse_fdf_params``): basis, energy shift, XC spelling, mesh
  cutoff, k-grid, electronic temperature.
* **PySCF** — no deck-parameter extractor exists yet; a ``.py`` deck
  answers ``None`` for now (recorded on the plan's board — the
  interface is the point, the second engine drops in behind it).

The same-directory rule as everywhere else (§ 4.1b): exactly one deck
defines the answer; zero or several answer ``None`` — recording a
guess would poison every consumer downstream.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional


def contract_of(directory) -> Optional[Dict[str, Any]]:
    """The recorded-contract block for *directory*, or ``None``.

    Shape (the ``info.calculation`` block):
    ``{"engine", "contract": {TransportConfig field -> value},
    "source": <deck name>, "source_sha256"}`` — only fields the deck
    actually states appear in ``contract``.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None
    decks = sorted(directory.glob("*.fdf"))
    if len(decks) == 1:
        return _siesta_contract(decks[0])
    return None


def _siesta_contract(deck: Path) -> Optional[Dict[str, Any]]:
    # Function-level import: parse_fdf_params lives with the transport
    # preflight for history; this module only needs the pure text
    # parser.
    from molbuilder.transport.preflight import parse_fdf_params
    try:
        text = deck.read_text(encoding="utf-8")
    except OSError:
        return None
    p = parse_fdf_params(text)
    contract = {k: v for k, v in {
        "basis_size":               p.basis_size,
        "energy_shift_ry":          p.energy_shift_ry,
        "xc_functional":            p.xc_functional,
        "xc_authors":               p.xc_authors,
        "siesta_mesh_cutoff_ry":    p.mesh_cutoff_ry,
        "k_mesh_transverse":        (list(p.kgrid) if p.kgrid else None),
        "electronic_temperature_k": p.electronic_temperature_k,
    }.items() if v is not None}
    if not contract:
        return None
    return {
        "engine": "siesta",
        "contract": contract,
        "source": deck.name,
        "source_sha256": hashlib.sha256(deck.read_bytes()).hexdigest(),
    }
