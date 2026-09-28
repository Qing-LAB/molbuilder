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
differences, and a stage with no result yet reads as pending, never as a
failure.
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
    ``masses_amu`` are the free atoms' masses, in the rows' order."""
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


def stages_share_a_directory(task) -> bool:
    """Whether two force-constant stages of ``task`` would run in one
    directory -- the flat layout, where every stage writes the same
    ``<label>.FC`` and ``<label>.spectra.json`` and the second overwrites the
    first's result.  Asked of the layout (`paths.Shape.stage_dir`), the rule
    `prep` refuses a sweep by and `summarize` reads it by."""
    from ..jobset.prep import token_for
    from ..paths import Shape
    from ..pyscf.stages import force_constant_stages
    shape = Shape.named(task.shape)
    names = force_constant_stages(task)
    dirs = {shape.stage_dir(token_for(task, n)) for n in names}
    return len(dirs) < len(names)


def collect_sweep(base_dir, task, *,
                  tolerance_cm1: Optional[float] = None) -> Dict[str, Any]:
    """The sweep record of the calculation at ``base_dir`` (`task` its
    description): every force-constant stage's newest attempt, read where its
    run wrote it, and the comparison.  Raises :class:`SweepError` when there
    is nothing a sweep can be: a PySCF vibration (its Hessian is analytic),
    fewer than two force-constant stages, stages sharing a directory, stages
    describing different atoms, or no stage with a result yet."""
    from .. import __version__ as _mb_version
    from ..chemistry import atomic_mass
    from ..constants import HARTREE_BOHR_EV_ANGSTROM_ASE
    from ..jobset.materialize import latest_attempt
    from ..jobset.prep import token_for
    from ..parse.engines.siesta_fc import (EV_PER_ANG2_TO_HARTREE_PER_BOHR2,
                                           hessian_from_fc, read_fc)
    from ..paths import Shape
    from ..pyscf.stages import force_constant_stages
    from ..runfiles import compose
    from ..sidecars.spectra import parse_spectra_json

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

    stages: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    loaded = []
    for name in names:
        token = token_for(task, name)
        attempt = latest_attempt(base / shape.stage_dir(token))
        spectrum = (attempt / compose(task.label, ".spectra.json")
                    if attempt is not None else None)
        if spectrum is None or not spectrum.is_file():
            pending.append({"stage": name, "why": (
                "no attempt opened" if attempt is None else
                f"no {compose(task.label, '.spectra.json')} in {rel(attempt)} "
                f"-- the run has not finished, or its finish failed (its "
                f"session log says which)")})
            continue
        res = parse_spectra_json(spectrum)
        meta = dict(res.engine_metadata or {})
        fc_file = attempt / str(meta.get("fc_file")
                                or compose(task.label, ".FC"))
        rx = dict(res.relaxation or {})
        f_eh = rx.get("max_force_eh_bohr")
        stages.append({
            "name": name,
            "attempt": rel(attempt),
            "spectrum": rel(spectrum),
            "fc_file": rel(fc_file) if fc_file.is_file() else None,
            # WHAT THIS STAGE VARIES, as the description states it -- the
            # stage's own overrides of the calculation's template.
            "varies": dict(getattr(by_name[name], "overrides", None) or {}),
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
            + "; ".join(f"{p['stage']}: {p['why']}" for p in pending))

    ref_name, ref, ref_fc_file, ref_meta = loaded[0]
    for name, res, _f, _m in loaded[1:]:
        if (list(res.free_atom_idxs) != list(ref.free_atom_idxs)
                or list(res.equilibrium_elements or [])
                != list(ref.equilibrium_elements or [])):
            raise SweepError(
                f"stages {ref_name!r} and {name!r} describe different atoms "
                f"(free atoms {list(ref.free_atom_idxs)} against "
                f"{list(res.free_atom_idxs)}): not one calculation's sweep")

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
    displaced = (list(range(int(rng[0]) - 1, int(rng[1])))
                 if len(rng) == 2 else [])
    H_ref = None
    if ref_fc_file.is_file() and displaced:
        fc0 = read_fc(ref_fc_file)
        H_ref = hessian_from_fc(fc0, displaced) / EV_PER_ANG2_TO_HARTREE_PER_BOHR2
    for name, res, fc_file, meta in loaded[1:]:
        entry: Dict[str, Any] = {"stage": name, "against": ref_name,
                                 "max_abs_change_ev_ang2": None,
                                 "relative_change": None}
        if H_ref is None or not fc_file.is_file():
            entry["why"] = "a force-constant file is missing"
        elif (meta.get("fc_range_1based") or []) != rng:
            entry["why"] = "the two runs nudged different atoms"
        else:
            H = hessian_from_fc(read_fc(fc_file), displaced) \
                / EV_PER_ANG2_TO_HARTREE_PER_BOHR2
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
    }
    return record


def write_sweep(base_dir, record: Dict[str, Any]) -> Path:
    """Write the record at the calculation root, atomically."""
    from ..persist import write_json
    out = sweep_path(base_dir, record["label"])
    write_json(out, record)
    return out


def sweep_table_text(record: Dict[str, Any]) -> str:
    """The printed summary: each stage and what it varied, each mode's
    frequency at every stage, and how far the force constants moved."""
    names = [s["name"] for s in record["stages"]]
    lines = [f"displacement sweep -- {record['label']}: "
             f"{len(names)} stage(s) with a result"
             + (f", {len(record['pending'])} pending"
                if record["pending"] else "")
             + f"; modes matched by shape to {record['reference_stage']!r}"]
    lines.append(f"  {'stage':<14} {'delta (A)':>10}  {'asym (eV/A2)':>12}  "
                 f"{'stationary':>10}  varies")
    for s in record["stages"]:
        d = s.get("fc_displacement_ang")
        a = s.get("fc_asymmetry_max_ev_ang2")
        st = s.get("stationary")
        varies = ", ".join(f"{k}={v}" for k, v in (s.get("varies") or {}).items())
        lines.append(
            f"  {s['name']:<14} "
            + (f"{d:>10.5f}" if isinstance(d, (int, float)) else f"{'--':>10}")
            + "  "
            + (f"{a:>12.4g}" if isinstance(a, (int, float)) else f"{'--':>12}")
            + f"  {('yes' if st else 'NO' if st is False else '--'):>10}  "
            + (varies or "(the template's own values)"))
    head = f"  {'mode':>4}  " + "  ".join(f"{n[:10]:>10}" for n in names) \
        + f"  {'spread':>8}  {'min overlap':>11}" \
        + ("  flag" if record.get("tolerance_cm1") is not None else "")
    lines.append(head)
    for m in record["modes"]:
        cells = []
        for n in names:
            f = m["frequency_cm1"].get(n)
            cells.append(f"{f:>10.1f}" if f is not None else f"{'--':>10}")
        ovl = [o for o in m["overlap"].values() if o is not None]
        sp = m.get("spread_cm1")
        lines.append(
            f"  {m['index_1based']:>4}  " + "  ".join(cells)
            + (f"  {sp:>8.2f}" if sp is not None else f"  {'--':>8}")
            + (f"  {min(ovl):>11.4f}" if ovl else f"  {'--':>11}")
            + ({True: "  over", False: "  ok", None: "  --"}[m["flagged"]]
               if record.get("tolerance_cm1") is not None else ""))
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
        lines.append(f"  {p['stage']}: pending ({p['why']})")
    return "\n".join(lines)


__all__ = ["SWEEP_SCHEMA", "SweepError", "sweep_path", "match_modes",
           "stages_share_a_directory", "collect_sweep", "write_sweep",
           "sweep_table_text"]
