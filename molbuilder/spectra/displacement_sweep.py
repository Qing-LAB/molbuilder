"""A SIESTA vibration's displacement sweep, summarized: how converged its force constants are.

MODULE  spectra.displacement_sweep (L2; on the host, molbuilder installed)
ROLE    `jobset summarize run` on a vibration calculation with two or more
        force-constant stages (`engines/vibration.md` § 5.9): read each stage's
        result where its run wrote it -- its spectrum and its raw force
        constants -- compare them as a function of what the stages varied
        (the displacement, the mesh), and write ``<label>.fc-sweep.json`` at the
        calculation root with the table printed
USED-BY jobset/_cli.py (``summarize run``); the Results tab's sweep presenter
        reads the record (`lib/inspectors/fc-sweep.js`)

WHY (`science/normal-modes.md` § 4b.6 C).  A force constant is a finite
difference of forces: too small a nudge drowns in the SCF's noise, too large
picks up anharmonic terms.  The check is to take the force constants at two or
more nudges and see them agree -- the frequencies, ``ω(δ) ≈ ω(δ/2)``, and the
matrix they come from, ``H(δ) ≈ H(δ/2)``.  § 9's H₂ ladder showed why a mesh
stage belongs beside the δ stages: most of the drift there was the grid.

A SUMMARY OF RESULTS THAT EXIST, and nothing else (`engines/vibration.md`
§ 5.5, the user's rule for `summarize`).  Every stage's files stay where its
run wrote them -- its ``<label>.FC`` (SIESTA's raw force constants, eV/Å²),
its output (the forces at every displacement), its ``<label>.spectra.json``
(every mode, both eigenvector forms, the removed motions, the thermochemistry,
the stationarity verdict, the displacement SIESTA used).  The record names
each by its path from the calculation root and adds only what a comparison
derives: which mode of each stage is which mode of the reference, how
similar their shapes are, how far each frequency moved, how far the force
constants moved.  Nothing is recomputed from a stage's data except those
differences.  A stage with no result says why in the words `run_status`
gives its attempt: not launched, queued or running is ``pending`` -- never a
failure -- and a run that failed, or ended without its result, is
``failed``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

#: The record's schema (`execution/job-contracts.md` § 6.1).
SWEEP_SCHEMA = "molbuilder/fc-displacement-sweep@1"


class SweepError(Exception):
    """A sweep that cannot be summarized -- the message names what to run or
    describe first, ready to surface verbatim."""


def sweep_path(base_dir, label: str) -> Path:
    """The record's one spelling, composed through the run-file grammar."""
    from ..runfiles import compose
    return Path(base_dir) / compose(label, ".fc-sweep.json")


def match_modes(reference, other, masses_amu) -> "tuple[list, list]":
    """Which mode of ``other`` is each mode of ``reference``, by SHAPE: the
    overlap of the mass-weighted eigenvectors over the free atoms both
    results share, assigned one-to-one so the total overlap is largest
    (`scipy.optimize.linear_sum_assignment`).  Returns ``(index, overlap)``
    per reference mode -- the matched 0-based index in ``other`` (``None``
    when ``other`` has fewer modes) and ``|⟨ref|other⟩|``, 1 for the same
    motion.

    Ranks are not identities: two near-degenerate modes swap order between
    two displacements, and comparing by rank would compare different motions.
    ``masses_amu`` are the free atoms' masses, in the rows' order.  Both
    results carry the same free atoms in the same order -- the caller's
    check (:func:`collect_sweep` refuses anything else), since a row of one
    is compared with the row of the other at the same position."""
    from scipy.optimize import linear_sum_assignment
    sqm = np.sqrt(np.asarray(masses_amu, dtype=float))[:, None]

    def _unit_rows(modes):
        rows = [(np.asarray(m.eigenvector_canonical, dtype=float) * sqm)
                .reshape(-1) for m in modes]
        out = []
        for v in rows:
            n = float(np.linalg.norm(v))
            out.append(v / n if n > 0 else v)
        return np.array(out)

    A, B = _unit_rows(reference.modes), _unit_rows(other.modes)
    if A.size == 0 or B.size == 0:
        return [None] * len(reference.modes), [None] * len(reference.modes)
    O = np.abs(A @ B.T)
    rows, cols = linear_sum_assignment(-O)
    index: List[Optional[int]] = [None] * len(reference.modes)
    overlap: List[Optional[float]] = [None] * len(reference.modes)
    for r, c in zip(rows, cols):
        index[int(r)] = int(c)
        overlap[int(r)] = float(O[r, c])
    return index, overlap


def stages_share_a_directory(task, *, include_disabled: bool = False
                             ) -> bool:
    """Whether two force-constant stages of ``task`` would run in one
    directory -- the flat layout, where every stage writes the same
    ``<label>.FC`` and ``<label>.spectra.json`` and the second overwrites the
    first's result.  Asked of the layout (`paths.Shape.stage_dir`), the rule
    `prep` refuses a sweep by (over every described stage,
    ``include_disabled``, since it preps any stage named) and `summarize`
    reads it by."""
    from ..jobset.materialize import stage_home
    from ..paths import Shape
    from ..pyscf.stages import force_constant_stages
    shape = Shape.named(task.shape)
    names = force_constant_stages(task, include_disabled=include_disabled)
    dirs = {shape.stage_dir(stage_home(None, task, n).token) for n in names}
    return len(dirs) < len(names)


def collect_sweep(base_dir, task, *,
                  tolerance_cm1: Optional[float] = None) -> Dict[str, Any]:
    """The sweep record of the calculation at ``base_dir`` (`task` its
    description): every force-constant stage's newest attempt, read where its
    run wrote it, and the comparison.  Raises :class:`SweepError` when there
    is nothing a sweep can be: a PySCF vibration (its Hessian is analytic),
    fewer than two force-constant stages, stages sharing a directory, no
    stage with a result yet, or stages that describe different atoms or were
    measured at different geometries."""
    from .. import __version__ as _mb_version
    from ..chemistry import atomic_mass
    from ..constants import HARTREE_BOHR_EV_ANGSTROM_ASE
    from ..engine_atom_index import from_engine_index
    from ..jobset.commands import rollback
    from ..jobset.materialize import latest_attempt
    from molbuilder.runrecord import launch_record
    from ..runfiles import RunNames
    from ..jobset.materialize import stage_home
    from ..parse.dirs import run_status
    from ..parse.engines.siesta_fc import (EV_PER_ANG2_TO_HARTREE_PER_BOHR2,
                                           hessian_from_fc, read_fc)
    from ..parse.errors import ParseError
    from ..paths import Shape
    from ..pyscf.stages import force_constant_stages
    from ..runfiles import compose
    from ..sidecars.spectra import SpectraJsonError, parse_spectra_json
    from ..template import catalogue, one

    base = Path(base_dir)
    if str(task.engine) != "siesta":
        raise SweepError(
            f"a {task.engine} vibration's second derivatives are analytic: "
            f"there is no displacement to sweep (engines/vibration.md 5.9)")
    names = force_constant_stages(task)
    if len(names) < 2:
        raise SweepError(
            f"this vibration has {len(names)} force-constant stage(s) "
            f"({', '.join(names) or 'none'}): a sweep compares two or more.  "
            f"Describe another stage with its own fc_displacement (or "
            f"mesh_cutoff), prep and launch it, then summarize again "
            f"(engines/vibration.md 5.9)")
    if stages_share_a_directory(task):
        raise SweepError(
            "the force-constant stages share one directory (the flat "
            "layout), so each overwrote the last one's <label>.FC and "
            "spectrum: a sweep needs the hierarchical layout "
            "(engines/vibration.md 5.9)")
    shape = Shape.named(task.shape)
    by_name = {s.name: s for s in task.stages}

    def rel(p: Path) -> str:
        return str(Path(p).resolve().relative_to(base.resolve()))

    # WHAT A STAGE VARIES, with the unit the catalogue gives it -- the
    # value alone (`fc_displacement = 0.02`) reads as the Å beside it.
    _items = catalogue()

    def _unit(item: str) -> Optional[str]:
        it = one(_items, item, engine="siesta")
        return getattr(it, "unit", None) or None

    spectrum_name = compose(task.label, ".spectra.json")
    stages: List[Dict[str, Any]] = []
    # A STAGE WITHOUT A RESULT, in the words its attempt's `run_status`
    # gives: still to come is pending, never a failure; a run that failed,
    # or ended without the result, is failed -- and says which.
    pending: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    loaded = []
    for name in names:
        token = stage_home(base, task, name).token
        attempt = latest_attempt(base / shape.stage_dir(token))
        if attempt is None:
            pending.append({"stage": name, "state": "not-started",
                            "detail": "no attempt opened yet -- prep and "
                                      "launch it"})
            continue
        spectrum = attempt / spectrum_name
        if not spectrum.is_file():
            names = RunNames.of(task.label, token, task.shape)
            st = run_status(attempt, names.stem,
                            launch=launch_record(attempt, names))
            entry = {"stage": name, "attempt": rel(attempt),
                     "state": st.state, "detail": st.detail}
            if st.state in ("pending", "queued", "running"):
                pending.append(entry)
            else:
                if st.state == "finished":
                    entry["detail"] = (
                        f"the run finished without writing {spectrum_name} "
                        f"({st.detail}) -- an attempt prepped before its job "
                        f"finished itself, which engines/vibration.md 5.5 "
                        f"finishes by hand")
                failed.append(entry)
            continue
        try:
            res = parse_spectra_json(spectrum)
        except SpectraJsonError as exc:
            failed.append({"stage": name, "attempt": rel(attempt),
                           "state": "unreadable",
                           "detail": f"{spectrum_name} could not be read: "
                                     f"{exc}"})
            continue
        meta = dict(res.engine_metadata or {})
        fc_file = attempt / str(meta.get("fc_file")
                                or compose(task.label, ".FC"))
        rx = dict(res.relaxation or {})
        f_eh = rx.get("max_force_eh_bohr")
        overrides = dict(getattr(by_name[name], "overrides", None) or {})
        stages.append({
            "name": name,
            "attempt": rel(attempt),
            "spectrum": rel(spectrum),
            "fc_file": rel(fc_file) if fc_file.is_file() else None,
            # WHAT THIS STAGE VARIES, as the description states it -- the
            # stage's own overrides of the calculation's template -- and the
            # unit each value is in, the catalogue's.
            "varies": overrides,
            "varies_units": {k: _unit(k) for k in overrides},
            "fc_displacement_ang": meta.get("fc_displacement_ang"),
            "fc_range_1based": meta.get("fc_range_1based"),
            "fc_asymmetry_max_ev_ang2": meta.get("fc_asymmetry_max_ev_ang2"),
            "n_modes": len(res.modes),
            "removed_motions": dict(res.removed_motions or {}).get("count"),
            "stationary": rx.get("converged"),
            "max_force_free_ev_ang": (None if f_eh is None else
                                      float(f_eh) * HARTREE_BOHR_EV_ANGSTROM_ASE),
            "force_criterion_ev_ang": meta.get(
                "reference_force_criterion_ev_ang"),
            "engine_version": res.engine_version,
        })
        loaded.append((name, res, fc_file, meta))
    if not loaded:
        raise SweepError(
            "no force-constant stage has a result yet: "
            + "; ".join(f"{p['stage']}: {p['state']} -- {p['detail']}"
                        for p in pending + failed))

    ref_name, ref, ref_fc_file, ref_meta = loaded[0]
    for name, res, _f, _m in loaded[1:]:
        if (list(res.free_atom_idxs) != list(ref.free_atom_idxs)
                or list(res.equilibrium_elements or [])
                != list(ref.equilibrium_elements or [])):
            raise SweepError(
                f"stages {ref_name!r} and {name!r} describe different atoms "
                f"(free atoms {list(ref.free_atom_idxs)} against "
                f"{list(res.free_atom_idxs)}): not one calculation's sweep")
        # ONE GEOMETRY, or the comparison is not a displacement's: each
        # result's structure hash pins the geometry its force constants were
        # taken at (`engines/vibration.md` § 6.2), and each stage took the
        # relax stage's NEWEST attempt when it was prepped (§ 5.2a) -- so a
        # relaxation re-run between two stages would be reported as the
        # displacement's effect.
        if res.structure_hash != ref.structure_hash:
            moved = ""
            if (res.equilibrium_positions_ang is not None
                    and ref.equilibrium_positions_ang is not None):
                d = np.linalg.norm(
                    np.asarray(res.equilibrium_positions_ang, float)
                    - np.asarray(ref.equilibrium_positions_ang, float),
                    axis=1)
                moved = f"; an atom moved {float(d.max()):.2e} Å between them"
            raise SweepError(
                f"stages {ref_name!r} and {name!r} were measured at different "
                f"geometries (their results' structure hashes differ{moved}): "
                f"a displacement sweep compares one geometry's force "
                f"constants, and this would report the geometry's change as "
                f"the displacement's.  Each force-constant stage takes the "
                f"relax stage's newest attempt when it is prepped "
                f"(engines/vibration.md 5.2a), and a prepped stage is not "
                f"prepped again (job-system.md 5.0): to measure them at the "
                f"relaxation they should share, "
                + rollback("their prep", base=base))

    # WHICH MODE IS WHICH, by shape, against the first stage with a result.
    masses = [atomic_mass(ref.equilibrium_elements[i])
              for i in ref.free_atom_idxs]
    matched = {name: match_modes(ref, res, masses)
               for name, res, _f, _m in loaded}
    modes: List[Dict[str, Any]] = []
    for k, m in enumerate(ref.modes):
        freq, idx, ovl, delta = {}, {}, {}, {}
        for name, res, _f, _m in loaded:
            j, o = matched[name][0][k], matched[name][1][k]
            idx[name] = None if j is None else j + 1
            ovl[name] = o
            freq[name] = None if j is None else float(res.modes[j].frequency_cm1)
            delta[name] = (None if freq[name] is None
                           else freq[name] - float(m.frequency_cm1))
        present = [f for f in freq.values() if f is not None]
        spread = (max(present) - min(present)) if len(present) > 1 else None
        modes.append({
            "index_1based": k + 1,
            "frequency_cm1": freq,
            "matched_index_1based": idx,
            "overlap": ovl,
            "change_from_reference_cm1": delta,
            "spread_cm1": spread,
            "flagged": (None if tolerance_cm1 is None or spread is None
                        else bool(spread > float(tolerance_cm1))),
        })

    # THE FORCE CONSTANTS THEMSELVES, H(δ) against the reference's, over the
    # block both runs nudged -- read from the raw files, in eV/Å².
    constants: List[Dict[str, Any]] = []
    rng = ref_meta.get("fc_range_1based") or []
    # SIESTA's atom numbers through the one door (`engine_atom_index`).
    displaced = (list(range(from_engine_index(int(rng[0]), "siesta"),
                            from_engine_index(int(rng[1]), "siesta") + 1))
                 if len(rng) == 2 else [])
    H_ref, ref_why = None, "a force-constant file is missing"
    if ref_fc_file.is_file() and displaced:
        try:
            H_ref = (hessian_from_fc(read_fc(ref_fc_file), displaced)
                     / EV_PER_ANG2_TO_HARTREE_PER_BOHR2)
        except (ParseError, OSError) as exc:
            ref_why = f"{rel(ref_fc_file)} could not be read: {exc}"
    for name, res, fc_file, meta in loaded[1:]:
        entry: Dict[str, Any] = {"stage": name, "against": ref_name,
                                 "max_abs_change_ev_ang2": None,
                                 "relative_change": None}
        if H_ref is None:
            entry["why"] = ref_why
        elif not fc_file.is_file():
            entry["why"] = "a force-constant file is missing"
        elif (meta.get("fc_range_1based") or []) != rng:
            entry["why"] = "the two runs nudged different atoms"
        else:
            try:
                H = (hessian_from_fc(read_fc(fc_file), displaced)
                     / EV_PER_ANG2_TO_HARTREE_PER_BOHR2)
            except (ParseError, OSError) as exc:
                entry["why"] = f"{rel(fc_file)} could not be read: {exc}"
                constants.append(entry)
                continue
            ix = np.ix_(displaced, displaced)
            d = np.abs(H[ix] - H_ref[ix])
            scale = float(np.max(np.abs(H_ref[ix]))) if d.size else 0.0
            entry["max_abs_change_ev_ang2"] = float(d.max()) if d.size else 0.0
            entry["relative_change"] = (entry["max_abs_change_ev_ang2"] / scale
                                        if scale > 0 else None)
        constants.append(entry)

    record: Dict[str, Any] = {
        "schema": SWEEP_SCHEMA,
        "label": task.label,
        "generated_at": datetime.now(timezone.utc).isoformat()
                        .replace("+00:00", "Z"),
        "molbuilder_version": str(_mb_version),
        "reference_stage": ref_name,
        "tolerance_cm1": (None if tolerance_cm1 is None
                          else float(tolerance_cm1)),
        "stages": stages,
        "modes": modes,
        "force_constants": constants,
        "pending": pending,
        "failed": failed,
    }
    return record


def write_sweep(base_dir, record: Dict[str, Any]) -> Path:
    """Write the record at the calculation root, atomically."""
    from ..persist import write_json
    out = sweep_path(base_dir, record["label"])
    write_json(out, record)
    return out


def _varies_text(stage: Dict[str, Any]) -> str:
    """What a stage varied, each value with its unit -- or the template's."""
    units = stage.get("varies_units") or {}
    return ", ".join(
        f"{k} = {v}" + (f" {units[k]}" if units.get(k) else "")
        for k, v in (stage.get("varies") or {}).items()
    ) or "(the template's own values)"


def _cell(m: Dict[str, Any], name: str, reference: str) -> str:
    """One mode at one stage: its frequency and -- beside the reference --
    the change, the shapes' overlap, and the mode it matched when that is
    not the same rank."""
    f = m["frequency_cm1"].get(name)
    if f is None:
        return "--"
    if name == reference:
        return f"{f:.1f}"
    j = m["matched_index_1based"].get(name)
    o = m["overlap"].get(name)
    d = m["change_from_reference_cm1"].get(name)
    parts = ([f"{d:+.2f}"] if d is not None else []) \
        + ([f"ovl {o:.4f}"] if o is not None else []) \
        + ([f"as #{j}"] if j is not None and j != m["index_1based"] else [])
    return f"{f:.1f} ({'; '.join(parts)})" if parts else f"{f:.1f}"


def sweep_table_text(record: Dict[str, Any]) -> str:
    """The record, printed: each stage -- what it varied and in what unit,
    the displacement SIESTA used, its files, its stationarity with the
    largest force and the criterion, its modes and removed motions, its
    SIESTA -- then every mode's frequency at every stage with the change, the
    overlap and the matched mode, the force-constant changes, and the stages
    without a result, each with its state (`engines/vibration.md` § 5.9)."""
    stages = record["stages"]
    names = [s["name"] for s in stages]
    ref = record["reference_stage"]
    others = record["pending"] + record.get("failed", [])
    w = max([len(n) for n in names] + [len(p["stage"]) for p in others]
            + [len("stage")])
    pad = " " * (w + 4)
    lines = [f"displacement sweep -- {record['label']}: "
             f"{len(names)} stage(s) with a result"
             + (f", {len(record['pending'])} pending"
                if record["pending"] else "")
             + (f", {len(record['failed'])} failed"
                if record.get("failed") else "")
             + f"; modes matched by shape to {ref!r}"]
    for s in stages:
        d = s.get("fc_displacement_ang")
        a = s.get("fc_asymmetry_max_ev_ang2")
        st = s.get("stationary")
        f, c = s.get("max_force_free_ev_ang"), s.get("force_criterion_ev_ang")
        lines.append(
            f"  {s['name']:<{w}}  delta "
            + (f"{d:.5f} A" if isinstance(d, (int, float)) else "--")
            + f"   varies: {_varies_text(s)}")
        lines.append(
            f"{pad}{s['attempt']}: {Path(s['spectrum']).name}"
            + (f", {Path(s['fc_file']).name}" if s.get("fc_file") else
               ", no force-constant file"))
        lines.append(
            f"{pad}stationary: "
            + ("yes" if st else "NO" if st is False else "--")
            + (f" (largest free-atom force {f:.4g} eV/A" if f is not None
               else " (largest free-atom force --")
            + (f", criterion {c:g})" if c is not None else ")")
            + f"; {s.get('n_modes', '--')} modes, "
            + f"{s.get('removed_motions', '--')} motions removed; "
            + "asymmetry "
            + (f"{a:.4g} eV/A2" if isinstance(a, (int, float)) else "--")
            + f"; SIESTA {s.get('engine_version') or '--'}")
    cells = [[str(m["index_1based"])]
             + [_cell(m, n, ref) for n in names]
             + [f"{m['spread_cm1']:.2f}" if m.get("spread_cm1") is not None
                else "--"]
             + ([{True: "over", False: "ok", None: "--"}[m["flagged"]]]
                if record.get("tolerance_cm1") is not None else [])
             for m in record["modes"]]
    head = ["mode"] + [f"{n} (cm-1)" for n in names] + ["spread"] \
        + (["flag"] if record.get("tolerance_cm1") is not None else [])
    widths = [max(len(r[i]) for r in cells + [head]) for i in range(len(head))]
    lines.append("  " + "  ".join(h.rjust(wd) for h, wd in zip(head, widths)))
    for r in cells:
        lines.append("  " + "  ".join(c.rjust(wd) for c, wd in zip(r, widths)))
    for c in record.get("force_constants", ()):
        if c.get("max_abs_change_ev_ang2") is None:
            lines.append(f"  force constants {c['stage']} vs {c['against']}: "
                         f"not compared ({c.get('why')})")
        else:
            rel = c.get("relative_change")
            lines.append(
                f"  force constants {c['stage']} vs {c['against']}: "
                f"max |dH| {c['max_abs_change_ev_ang2']:.4g} eV/A2"
                + (f" ({100.0 * rel:.3g}% of the largest)" if rel is not None
                   else ""))
    for p in record["pending"]:
        lines.append(f"  {p['stage']:<{w}}  pending -- {p['state']}: "
                     f"{p['detail']}")
    for p in record.get("failed", []):
        lines.append(f"  {p['stage']:<{w}}  FAILED -- {p['state']}: "
                     f"{p['detail']}")
    return "\n".join(lines)


__all__ = ["SWEEP_SCHEMA", "SweepError", "sweep_path", "match_modes",
           "stages_share_a_directory", "collect_sweep", "write_sweep",
           "sweep_table_text"]
