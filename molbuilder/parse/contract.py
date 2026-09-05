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

#: The engine's stdout name, which only the wrapper knows
#: (`runwrap.py`: ``".pyscf.log" if suffix == ".py" else ".out"``).  It is
#: the one result-file fact NOT in an engine's warm-file vocabulary,
#: because a log is never warm-started from.
_STDOUT_SUFFIX = {"siesta": ".out", "pyscf": ".pyscf.log"}


def engine_of(directory) -> str:
    """Which engine ran in *directory* — ``"siesta"``, ``"pyscf"``, or
    ``"unknown"``.

    `running-a-job.md` § 4.2 owns the rule; this is the one
    implementation.  The engine is **declared at script-generation
    time**, because that is the only moment it is known for certain, and
    a run directory gets copied away from everything that knew.

    **TWO TIERS, not a precedence list.**

    *Declarations* -- the PROVENANCE ``engine`` key of any deck or
    wrapper (`job-contracts.md` § 3.2) and the ``.molwatch.log``
    ``# engine:`` header -- are weighed **together**.  One distinct
    answer among them is the answer.  Two is a run that contradicts
    itself, and that is ``"unknown"``: the same rule and the same reason
    as ``contract_of`` above (§ 5b) -- a directory that says two things
    cannot be made to say one by picking, and an answer that might be the
    other engine's is worth less than no answer.

    *The sniff* -- what files are present -- is consulted **only when
    nothing declared**, for a directory molbuilder did not write.  It
    never contradicts a declaration, because it is evidence of a
    different kind: files outlive the run that wrote them, so a stale
    ``.fdf`` beside a freshly re-prepped PySCF deck is not a second
    opinion, it is litter.

    **Why this is not a first-hit-wins list, which is what shipped on
    2026-09-04 and was wrong.**  Ordered rungs let ONE artifact decide
    while corroborating evidence goes unread: a PySCF run whose molwatch
    header AND whose whole file cluster said ``pyscf`` answered
    ``"siesta"`` because somebody had copied a foreign ``.run.sh`` into
    the directory.  That is worse than the constant it replaced *and*
    worse than the code before it -- the route had been answering from
    the loaded file's own ``source_format``, which was right.  A rung
    that returns before reading its peers is not a resolution order; it
    is a first-match search that happens to be spelled like one.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return "unknown"

    # EVERY DECLARATION IS WEIGHED TOGETHER; the sniff is consulted only
    # when there is none.  See the two tiers in the docstring above.
    declared = _declared_in_provenance(directory) | _declared_in_molwatch(directory)
    if len(declared) == 1:
        return declared.pop()
    if len(declared) > 1:
        return "unknown"            # the run contradicts itself -- say so
    sniffed = _from_cluster(directory)
    return sniffed.pop() if len(sniffed) == 1 else "unknown"


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
            # Case-insensitive on the KEY as well as the value.  The
            # emitter always writes `engine` lowercase, but the block
            # sits in a file whose USER-CUSTOM banner says "Edit
            # freely", and a hand-written `Engine` silently dropping
            # the declaration is the quiet failure this whole
            # mechanism exists to remove.
            name = ""
            for key, val in (block or {}).items():
                if key.strip().lower() == "engine":
                    name = str(val).strip().lower()
                    break
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
    """The sniff — what files are here, for a directory molbuilder did
    not write (a hand-made run, or one prepared before the declaration
    shipped on 2026-09-04).

    **The vocabulary is the engine's own, as data.**  `job-contracts.md`
    § 4.2a settled that each engine ships ONE ``<engine>/warm-files.toml``
    and *"every consumer derives from it"* -- a rule that exists because
    three hand-written copies of this vocabulary had already drifted
    apart, and whose history § 4.2a records.  This function held a
    FOURTH from 2026-09-04 until it was caught in review the same day:
    ``("*.pyscf.log", "*_geom_optim.xyz", "*.chk")``, two of whose three
    entries were verbatim rows of ``pyscf/warm-files.toml``.

    ``warmfiles.inventory`` is the door and says so in its own docstring
    -- *"a HINT about a directory, safe to over-include, required to
    under-include nothing"* -- which is exactly this question.  Measured
    when it went in: the two engines' inventories are disjoint (no
    shared suffix, no suffix-containment), and over the 113 real run
    directories in the tree the derived answer matches the hand-written
    one everywhere.

    Two facts are NOT warm files and stay here: the deck suffix (the
    seam's ``EngineSeam.suffix``, which `parse` may not import -- it
    lives a layer up in ``jobset``) and the wrapper's stdout name.  A
    bare ``.py`` is deliberately not a signal either: ``mb_monitor.py``
    and ``config_dir.py`` ship beside every flat run, so "there is a
    python file here" says nothing about the engine.
    """
    from molbuilder.warmfiles import WarmFilesError, inventory
    names = [f.name for f in directory.iterdir() if f.is_file()]
    out = set()
    if any(n.endswith(".fdf") for n in names):
        out.add("siesta")               # the deck: EngineSeam.suffix
    for engine in _ENGINES:
        if any(n.endswith(_STDOUT_SUFFIX[engine]) for n in names):
            out.add(engine)
        try:
            suffixes = inventory(engine)
        except (WarmFilesError, OSError):
            continue                    # a broken rules file is not an engine vote
        if any(n.endswith(s) for n in names for s in suffixes):
            out.add(engine)
    return out
