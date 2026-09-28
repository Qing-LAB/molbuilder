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
from typing import Any, Dict, Optional, Tuple


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
    # THE ROLE COMES FROM THE CATALOGUE.  `"*.fdf"` is `runfiles.WRITTEN`'s
    # `.fdf` spelled outside the module that declares it
    # (`project-layout.md` § 4.5).
    from ..runfiles import find_by_role
    decks = find_by_role(directory, ".fdf")
    if len(decks) == 1:
        return _siesta_contract(decks[0])
    return None


#: The recorded contract's field names against ``SiestaConfig``'s -- ONE
#: table, read by :func:`contract_fields_of` here and by the transport
#: citation fill (`transport/citation_defaults.py`), so the two cannot
#: disagree about which attribute a record's key is.  Only two spellings
#: differ; the table is written out because both vocabularies are facts
#: about classes, not derivable from each other.
RECORD_TO_SIESTA_FIELD: Dict[str, str] = {
    "basis_size":               "basis_size",
    "siesta_mesh_cutoff_ry":    "mesh_cutoff",
    "energy_shift_ry":          "pao_energy_shift",
    "electronic_temperature_k": "electronic_temperature",
    "xc_functional":            "xc_functional",
    "xc_authors":               "xc_authors",
}


def contract_fields_of(cfg) -> Dict[str, Any]:
    """The level of theory a config is ABOUT TO RUN, in the recorded
    contract's own field names -- the other half of :func:`_siesta_contract`,
    through :data:`RECORD_TO_SIESTA_FIELD`.  A consumer comparing a
    structure's ``info.calculation`` against the calculation it is being
    handed to compares this dict field by field.  Only fields the config
    class has appear; an engine with no recorded contract (PySCF today)
    answers ``{}``.
    """
    out: Dict[str, Any] = {}
    for key, attr in RECORD_TO_SIESTA_FIELD.items():
        if hasattr(cfg, attr) and getattr(cfg, attr) is not None:
            out[key] = getattr(cfg, attr)
    return out


def _force_tolerance_of(targets: Optional[Dict[str, Any]]) -> Optional[float]:
    """The run's own ``max_force_tol_eV_per_A`` out of a parsed
    ``convergence_targets`` block -- flat (SIESTA's echo, a single-stage
    molwatch header) or nested one level by stage (a staged molwatch
    header, `parse.engines.molwatch_grammar.parse_convergence_line`).  One stage
    nested is the run's; several is a header this run did not write alone,
    and the answer is ``None`` rather than a pick."""
    if not isinstance(targets, dict):
        return None
    flat = targets.get("max_force_tol_eV_per_A")
    if isinstance(flat, (int, float)):
        return float(flat)
    nested = [v for k, v in targets.items()
              if k != "source" and isinstance(v, dict)]
    if len(nested) == 1:
        v = nested[0].get("max_force_tol_eV_per_A")
        return float(v) if isinstance(v, (int, float)) else None
    return None


def relaxation_of(directory, *,
                  parsed: Optional[Tuple[Any, Any]] = None
                  ) -> Optional[Dict[str, Any]]:
    """The relaxation record -- what the run in *directory* did to the
    geometry it left -- or ``None`` (`model/parse.md` § 5b.1).

    Shape (the ``info.relaxation`` block)::

        {"engine", "source", "n_steps", "force_tolerance_ev_ang",
         "max_force_ev_ang", "max_force_free_ev_ang", "held_atom_idxs",
         "held_atom_keys", "converged", "run_state", "geometry_sha256"}

    Read through the doors the Results tab opens a run with
    (`dirs.openable_in`, the registry's `detect`), so the record describes
    the file a person would be looking at.  A run that relaxed nothing --
    a force-constant run, a single point, a seed nothing wrote into --
    echoes no force tolerance and has no record; a directory with no
    openable output, or one that does not parse, or one whose last
    reported forces are missing, is ``None``, never a guess.  The forces
    are the last step that reported any, judged as the engines judge them:
    the largest absolute Cartesian COMPONENT, over every atom and over the
    atoms the run moved (the held set excluded), from that step's per-atom
    forces -- or the engine's own ``Max`` lines when the step carries no
    per-atom block, which on SIESTA are that same component.  ``converged``
    is the moved atoms' figure within the run's tolerance; when a run held
    nothing the two figures are one.  ``geometry_sha256`` is the LAST
    frame's :meth:`Structure.geometry_fingerprint` and ``held_atom_keys``
    the held atoms' :meth:`Structure.geometry_lines`, so a consumer can tell
    whether the coordinates in front of it are the ones this record is
    about, whatever order it lists them in.  ``engine`` is ``None`` when the
    directory does not declare one.

    ``parsed`` is ``(path, parse)`` a caller already holds -- the viewer's
    load of this very file -- and is used when ``path`` is the file this
    record reads, so that file is parsed once, not twice.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None
    from .dirs import openable_in
    from .registry import detect
    try:
        path, _trail = openable_in(str(directory))
        if not path:
            return None
        if (parsed is not None
                and Path(parsed[0]).resolve() == Path(path).resolve()):
            traj = parsed[1]
        else:
            traj = detect(Path(path)).parse(path)
    except Exception:                                       # noqa: BLE001
        return None
    return relaxation_of_output(path, traj, engine=engine_of(directory))


def relaxation_of_output(path, traj, *,
                         engine: Optional[str] = None
                         ) -> Optional[Dict[str, Any]]:
    """The relaxation record of ONE output already parsed -- ``traj``, the
    parse of ``path`` -- in :func:`relaxation_of`'s shape, or ``None`` for a
    run that relaxed nothing.  :func:`relaxation_of` asks it of the file a
    directory's viewer opens; `prep` asks it of the `relax` stage's own
    output, the one it reads the relaxed geometry from
    (`engines/vibration.md` § 5.2a), so the record is of that run and no
    other, whatever else the directory holds."""
    import numpy as np
    frames = [fr for fr in getattr(traj, "frames", []) if fr.structure is not None]
    if not frames:
        return None
    info = getattr(traj, "runtime_info", None) or {}
    tol = _force_tolerance_of(info.get("convergence_targets"))
    if tol is None:
        return None
    held = sorted(int(i) for i in (info.get("frozen_atoms") or []))
    with_forces = [fr for fr in frames
                   if fr.forces is not None or fr.max_force is not None]
    max_all = max_free = None
    if with_forces:
        fr = with_forces[-1]
        if fr.forces is not None:
            # THE ENGINES' OWN CONVENTION: the largest absolute Cartesian
            # component per atom -- what SIESTA's `Max` line prints and
            # `MD.MaxForceTol` tests, and the rule both read-backs judge by
            # (`engines/vibration.md` § 2.2, V1.30).  A per-atom norm would
            # call a run the engine converged "not converged" by up to √3.
            comp = np.abs(np.asarray(fr.forces, dtype=float)
                          .reshape(-1, 3)).max(axis=1)
            max_all = float(comp.max()) if comp.size else None
            free = [i for i in range(len(comp)) if i not in set(held)]
            max_free = float(comp[free].max()) if free else None
        else:
            max_all = (float(fr.max_force) if fr.max_force is not None
                       else None)
            max_free = (float(fr.max_force_constrained)
                        if fr.max_force_constrained is not None else None)
    if max_all is None and max_free is None:
        return None
    judged = max_free if max_free is not None else max_all
    last = frames[-1]
    lines = last.structure.geometry_lines()
    return {
        "engine": (engine if engine and engine != "unknown" else None),
        "source": Path(path).name,
        "n_steps": int(last.step_index if last.step_index is not None
                       else len(frames) - 1),
        "force_tolerance_ev_ang": tol,
        "max_force_ev_ang": max_all,
        "max_force_free_ev_ang": max_free,
        "held_atom_idxs": held,
        "held_atom_keys": sorted(lines[i] for i in held if i < len(lines)),
        "converged": bool(judged <= tol),
        "run_state": str(getattr(traj, "run_state", "") or ""),
        "geometry_sha256": last.structure.geometry_fingerprint(),
    }


def _siesta_contract(deck: Path) -> Optional[Dict[str, Any]]:
    from .fdf import parse_fdf_params
    try:
        text = deck.read_text(encoding="utf-8")
    except OSError:
        return None
    from ..units import UnknownUnit
    try:
        p = parse_fdf_params(text)
    except UnknownUnit:
        # No contract rather than a wrong one: recording a number this
        # build could not convert would put a value in the record that
        # the deck does not state.
        return None
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

#: The engines a run directory can answer -- THE CATALOGUE'S OWN LIST, since
#: an engine becomes known to the run-file layer by having a `WRITTEN` row
#: (`model/parse.md` § 5.5: adding an engine is two edits).  It was the
#: literal pair ``("siesta", "pyscf")`` until 2026-09-18.
#:
#: ``"molwatch"`` is deliberately NOT here, and cannot be: it is the
#: ``source_format`` a ``.molwatch.log`` reports when its header did not name
#: an engine -- a FORMAT, and reading it as an engine is exactly the
#: substitution `running-a-job.md` § 4.2 forbids.  No `WRITTEN` row names it
#: as an engine either, so the derivation says the same thing the literal did.
from ..runfiles import engines as _engines, stdout_roles as _stdout_roles

_ENGINES = _engines()

# `_STDOUT_SUFFIX` STOOD HERE -- a hand-written
# ``{"siesta": ".out", "pyscf": ".pyscf.log"}`` whose own docstring quoted
# `runwrap.py`'s conditional back at it: two spellings of one map in two
# layers, and `runwrap` had a third.  It is `runfiles.stdout_roles(engine)`,
# asked where it is used: the stdout role is the one result-file fact NOT in
# an engine's warm-file vocabulary (a log is never warm-started from), so the
# CATALOGUE holds it, as the `output` column.
#
# A LIST and not one name, deliberately: nothing says an engine writes exactly
# one, and `[0]` on a tuple is how a second one would be dropped in silence.


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
    from molbuilder.script_emit import _extract_provenance_dict
    # THE THREE ROLES COME FROM THE CATALOGUE.  `.run.sh`, `.fdf` and `.py`
    # are all `runfiles.WRITTEN` rows, so the search is the catalogue's --
    # this spelled them as globs, which is the § 4.2a fault R1 closed
    # elsewhere in this same module, in its § 4.5 form.
    from ..runfiles import find_by_role
    out = set()
    for role in (".run.sh", ".fdf", ".py"):
        for f in find_by_role(directory, role):
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
    from molbuilder.parse.engines.molwatch_grammar import ENGINE as _ENGINE_RE
    out = set()
    from ..runfiles import find_by_role
    for f in find_by_role(directory, ".molwatch.log"):
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
    bare ``.py`` is deliberately not a signal either: "there is a python
    file here" says nothing about the engine -- a person's own script, or
    the monitor's shipped modules as every attempt held them until
    2026-09-26.
    """
    from molbuilder.warmfiles import WarmFilesError, inventory
    from ..runfiles import find_by_role
    names = [f.name for f in directory.iterdir() if f.is_file()]
    out = set()
    # Declared roles ask the catalogue.  The warm suffixes below cannot:
    # they are the engine's names, in no `runfiles.WRITTEN` row.
    # THE DECK STAYS A LITERAL, and that is a finding rather than a lapse
    # (`plans/plan.md` § 5c.2 step e listed it for derivation).  The catalogue
    # does carry `.fdf -> siesta`, but deriving from it would equally give
    # `.py -> pyscf` -- and the docstring above records why that is wrong:
    # a python file beside a run says nothing about its engine, so
    # `find_by_role(d, ".py")` can answer for a SIESTA directory.  The asymmetry
    # is that one extension is generic and the other is not, which is not a
    # fact the catalogue holds and does not earn a column for one case.
    if find_by_role(directory, ".fdf"):
        out.add("siesta")               # the deck: EngineSeam.suffix
    for engine in _ENGINES:
        if any(find_by_role(directory, r) for r in _stdout_roles(engine)):
            out.add(engine)
        try:
            suffixes = inventory(engine)
        except (WarmFilesError, OSError):
            continue                    # a broken rules file is not an engine vote
        if any(n.endswith(s) for n in names for s in suffixes):
            out.add(engine)
    return out
