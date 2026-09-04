"""``contract_of`` — the electronic contract a directory's deck records.

The ONE interface behind the Results tab's contract recording
(`archive/2026-09-01-structure-info-plan.md` I5): given a directory, find the one
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


# --------------------------------------------------------------------- #
#  engine_of — WHICH ENGINE RAN HERE                                    #
# --------------------------------------------------------------------- #

#: The engines a run directory can answer.  ``"molwatch"`` is deliberately
#: NOT here: it is the ``source_format`` a ``.molwatch.log`` reports when
#: its header did not name an engine -- a FORMAT, and reading it as an
#: engine is exactly the substitution `running-a-job.md` § 4.2 forbids.
_ENGINES = ("siesta", "pyscf")

#: PySCF-only result files (`job-contracts.md` § 2.2).  A bare ``.py`` is
#: NOT on this list and must never be: ``mb_monitor.py`` and
#: ``config_dir.py`` ship beside every flat run, which is the same
#: foot-gun ``JobDirParser.can_parse`` documents for its own claim rule.
_PYSCF_CLUSTER = ("*.pyscf.log", "*_geom_optim.xyz", "*.chk")


def engine_of(directory) -> str:
    """Which engine ran in *directory* — ``"siesta"``, ``"pyscf"``, or
    ``"unknown"``.

    The resolution order is `running-a-job.md` § 4.2's, and that document
    owns it; this is the one implementation.  In short: the engine is
    **declared at script-generation time**, because that is the only
    moment it is known for certain, and a run directory gets copied away
    from everything that knew.

    1. the **PROVENANCE ``engine`` key** in any deck or wrapper
       (`job-contracts.md` § 3.2), read through the registered block
       extractor -- not a second regex;
    2. the **``.molwatch.log`` ``# engine:`` header**, through the same
       pattern the molwatch parser uses;
    3. the **file cluster**, for a directory molbuilder did not write;
    4. ``"unknown"``.

    **Disagreement answers ``"unknown"``, it does not vote.**  Same rule,
    same reason as ``contract_of`` above (§ 5b): a directory that says two
    things cannot be made to say one by picking, and an answer that might
    be the other engine's is worth less than no answer.  Engines do not
    share a run directory, so a disagreement is a real anomaly and the
    caller should see it as one.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return "unknown"

    for step in (_declared_in_provenance, _declared_in_molwatch, _from_cluster):
        found = step(directory)
        if len(found) == 1:
            return found.pop()
        if len(found) > 1:
            return "unknown"        # the directory contradicts itself
    return "unknown"


def _declared_in_provenance(directory: Path) -> set:
    """Step 1 — every PROVENANCE block in the directory, asked its engine.

    The wrapper is checked as well as the deck, and that is the point: a
    **TranSIESTA** run has no deck PROVENANCE (`job-contracts.md` § 3.1's
    per-engine table) but always has a ``.run.sh``, so the wrapper is the
    one artifact every prepared run carries whatever the task.
    """
    from molbuilder.parse.scripts.provenance import _extract_provenance_dict
    out = set()
    for pattern in ("*.run.sh", "*.fdf", "*.py"):
        for f in sorted(directory.glob(pattern)):
            try:
                block = _extract_provenance_dict(
                    f.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
            name = (block or {}).get("engine", "").strip().lower()
            if name in _ENGINES:
                out.add(name)
    return out


def _declared_in_molwatch(directory: Path) -> set:
    """Step 2 — the ``# engine:`` header of every molwatch log.

    Both generators write this at file-emission time, before the engine
    starts, so it answers for a run that has not produced a result yet.
    Only the header is read: the frames are irrelevant to the question
    and a growing log can be large.
    """
    from molbuilder.parse.engines.molwatch import _ENGINE_RE
    out = set()
    for f in sorted(directory.glob("*.molwatch.log")):
        try:
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                for _ in range(40):
                    line = fh.readline()
                    if not line:
                        break
                    m = _ENGINE_RE.match(line)
                    if m:
                        name = m.group(1).strip().lower()
                        if name in _ENGINES:
                            out.add(name)
                        break
        except OSError:
            continue
    return out


def _from_cluster(directory: Path) -> set:
    """Step 3 — what files are here, for a directory molbuilder did not
    write (a hand-made run, or one prepared before the declaration
    shipped on 2026-09-04)."""
    out = set()
    if any(directory.glob("*.fdf")):
        out.add("siesta")
    for pattern in _PYSCF_CLUSTER:
        if any(directory.glob(pattern)):
            out.add("pyscf")
            break
    return out
