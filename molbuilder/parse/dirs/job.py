"""Directory-level job DirParser.

Implements :class:`JobDirParser` + :func:`decode_run_dir` per
``docs/execution/running-a-job.md § 4`` + ``docs/model/parse.md``
§ 7 composer pattern.  Pure composer over the existing
infrastructure:

* ``molbuilder.parsers`` — file-level TrajectoryParser registry; the
  decoder calls ``detect_parser`` + ``.parse`` on each ``.out``.  It
  NEVER opens a SIESTA output file directly.
* ``molbuilder.script_contract`` — reads HEADER / PROVENANCE /
  BENCH-MARKS / ATOM-METADATA / USER-CUSTOM via the existing
  ``extract_*`` helpers.  The decoder does NOT re-grep for these
  blocks.

The only NEW parsing this module does is ``_parse_engine_body_summary``
which extracts a CURATED list of ~21 SIESTA engine-body directives.
That list is frozen by the doc; adding a key requires a doc update
+ a test.

Phase B of parse-module.md migration: ``decode_run_dir`` now returns
a :class:`JobResult` frozen dataclass.  The class API
(:class:`JobDirParser`) is the canonical entry point;
``decode_run_dir`` is a thin module-level convenience.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from molbuilder.identity import parse_stage_token
from molbuilder.parse.base import DirParser
from molbuilder.parse.types import JobResult, ParseWarning


# Doc-pinned constants ------------------------------------------------- #

SCHEMA_VERSION = 1

# Curated SIESTA engine-body keys (§ 2.1.1 of the doc).  Adding a key
# requires updating BOTH this list AND the doc.  Frozen ordering for
# stable JSON.  Missing keys in a given .fdf emit `null`, not absent.
ENGINE_BODY_KEYS: Tuple[str, ...] = (
    # System
    "SystemLabel", "SystemName", "NumberOfAtoms", "NumberOfSpecies",
    # SCF
    "MeshCutoff", "PAO.BasisSize", "PAO.EnergyShift",
    "DM.Tolerance", "DM.MixingWeight", "DM.NumberPulay",
    "MaxSCFIterations", "ElectronicTemperature",
    # XC
    "XC.functional", "XC.authors", "SpinPolarized",
    # Solver
    "SolutionMethod", "BlockSize", "Diag.Algorithm",
    "Diag.ELPA.GPU", "Diag.ParallelOverK",
    # MD / relax
    "MD.TypeOfRun", "MD.NumCGsteps", "MD.MaxForceTol", "MD.MaxCGDispl",
    # k-mesh (block-valued; reduced to "AxBxC" form)
    "kgrid_Monkhorst_Pack",
)


# Run-state detection is fully delegated to the engine trajectory
# parsers (detect().parse() -> traj.run_state); the end-of-run and
# failure markers live in engines/siesta.py + engines/pyscf.py, NOT
# here — the decoder never greps .out content itself (enforced by
# test_no_direct_out_grep_in_decoder).

# Default CG-step threshold for cg_step_milestone events.
CG_STEP_MILESTONE_DEFAULT = 50


# Exceptions ----------------------------------------------------------- #


class JobTypeAmbiguousError(ValueError):
    """Raised when sniff finds markers for more than one engine type
    in the same .fdf and no script-contract declaration breaks the
    tie.  Per § 4 step 2 of job-decoder.md."""


# Internal patterns ---------------------------------------------------- #

_KGRID_BLOCK_RE = re.compile(
    r"%block\s+kgrid_Monkhorst_Pack\s*\n(.*?)%endblock\s+kgrid_Monkhorst_Pack",
    re.S | re.IGNORECASE,
)


# ---- helpers --------------------------------------------------------- #


def _detect_stage(filename: str) -> Optional[int]:
    """A file's stage ORDINAL, or ``None`` when it carries no stage token.

    The token is ``<NN>_<name>`` (``bdt_au_01_coarse.fdf``) and this returns
    the ``NN``.  Read through :func:`molbuilder.identity.parse_stage_token`,
    which is the one place the shape is written down -- the decoder used to
    carry its own ``-stage(N)`` regex, a second spelling of the emitter's
    convention that could and did drift from it.

    **Still an int, and deliberately so.**  Decision 27 kept the ordinal in
    the filename, so everything downstream that ORDERS by stage --
    :func:`_anchor_sort_key`, the ``.out`` listing, the ``stage`` field of the
    engine-input envelope -- keeps working unchanged.  Had the token carried
    the name alone, the anchor rule would have lost its sort key and the
    Results tab its notion of "the active stage"; that is the trap
    ``staged-runs-implementation-plan.md`` § 8d walked into and § 8e closed.
    """
    hit = parse_stage_token(filename)
    return hit[0] if hit else None


def _detect_stage_name(filename: str) -> Optional[str]:
    """A file's stage NAME, or ``None``.  The half a person reads."""
    hit = parse_stage_token(filename)
    return hit[1] if hit else None


def _anchor_sort_key(p: Path) -> Tuple[bool, int, str]:
    """Sort key for picking the "highest-stage" .fdf as the anchor.

    Distinguishes "no stage" from "stage 0" so an unstaged template
    file can't beat a real staged file via lex tie-break.  Sort
    ascending → ``[-1]`` gives the highest-staged anchor; unstaged
    files sort BEFORE all staged ones (post-2026-06-19 round-2 fix
    for the B1 fix's residual sort-tiebreak bug).
    """
    stage = _detect_stage(p.name)
    return (stage is not None, stage if stage is not None else -1, p.name)


def _iso_z(ts: float) -> str:
    """Format a POSIX timestamp as an ISO-8601 UTC string."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(
        timespec="milliseconds").replace("+00:00", "Z")


def _parse_engine_body_summary(fdf_text: str) -> Dict[str, Optional[str]]:
    """Extract the curated ENGINE_BODY_KEYS from a .fdf body.

    Output: dict with every curated key, value as raw string OR None
    when the key is absent.  Values are emitted verbatim (units
    included) per § 9 forbidden pattern #3 — no interpretation.

    The function is intentionally simple-minded: line-oriented grep
    for ``^<key>\\b``, take the rest of the line minus any trailing
    comment.  Block-valued keys (kgrid_Monkhorst_Pack) get a special
    "NxNxN" reduction from their %block content.

    SIESTA key matching is case-insensitive (the engine itself
    treats fdf keys as case-insensitive); whitespace separators
    include tabs (post-2026-06-19 review fix I2).

    Defense-in-depth: strip a leading UTF-8 BOM from the text so
    a caller that bypasses the file-IO layer (or passes a string
    that happens to retain the BOM) still gets correct extraction
    (post-2026-06-19 round-4 fix D.2).
    """
    if fdf_text.startswith("﻿"):
        fdf_text = fdf_text[1:]
    out: Dict[str, Optional[str]] = {k: None for k in ENGINE_BODY_KEYS}
    # case-insensitive lookup: lowered key -> canonical key.
    _lc_lookup = {k.lower(): k for k in ENGINE_BODY_KEYS}
    # Line-by-line scan; only inspect lines that are outside any of
    # the reserved comment blocks (those are owned by the script-
    # contract extractors).  Coarse heuristic: skip lines starting
    # with `#` — they're either comments or script-contract block
    # content, neither of which is engine body.
    for raw in fdf_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Take "Keyword value [unit]" — strip trailing `# comment`.
        if "#" in line:
            line = line.split("#", 1)[0].rstrip()
        # split on any whitespace (tab OR space) per SIESTA convention.
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        key = parts[0]
        # Normalise internal whitespace (collapse tabs + repeated
        # spaces) so the stored value reads cleanly downstream
        # (post-2026-06-19 round-2 fix for I2 tab-in-value).
        value = " ".join(parts[1].split())
        canonical = _lc_lookup.get(key.lower())
        if canonical is not None:
            # SIESTA uses last-wins for duplicated fdf keys (manual
            # § 7.1).  A common pattern: a stub at the top of the
            # file with a default, then an override later that the
            # engine actually reads.  Match the engine's semantics
            # so engine_body_summary reflects what SIESTA used, not
            # the first occurrence (post-2026-06-19 round-3 fix I4).
            out[canonical] = value

    # kgrid_Monkhorst_Pack is a %block, not a directive.  Pull the
    # 3x3 diagonal and reduce to "AxBxC".
    #
    # Round-4 fix (E.1): the previous pattern indexed by
    # ``toks[len(rows)]`` which works only when every block-content
    # line is itself a valid row.  A blank line or comment between
    # rows would silently swallow the error + leave rows short.  We
    # now scan ONLY valid rows (>= 3 numeric tokens), and we read
    # the explicit diagonal element (row 0 col 0, row 1 col 1, row 2
    # col 2) from the row's OWN position in the validated sequence.
    m = _KGRID_BLOCK_RE.search(fdf_text)
    if m:
        valid_rows: List[List[str]] = []
        for raw in m.group(1).splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            if len(toks) >= 3:
                valid_rows.append(toks)
        if len(valid_rows) >= 3:
            try:
                diag = (
                    int(valid_rows[0][0]),
                    int(valid_rows[1][1]),
                    int(valid_rows[2][2]),
                )
                out["kgrid_Monkhorst_Pack"] = f"{diag[0]}x{diag[1]}x{diag[2]}"
            except (ValueError, IndexError):
                pass
    return out


def _classify_job_type(fdf_text: str) -> str:
    """Classify the job_type per § 4.

    Step 1: script-contract declaration > inference from BENCH-MARKS
    fields presence.
    Step 2: sniff fallback on engine-body markers.
    """
    from molbuilder.parse.scripts.bench_marks import (
        _extract_bench_marks_dict,
    )
    # Step 1a: explicit `job_type` field in BENCH-MARKS (forward-compat).
    bench = _extract_bench_marks_dict(fdf_text) or {}
    if isinstance(bench.get("job_type"), str):
        return bench["job_type"]
    bench_field_names = {f["name"] for f in bench.get("fields", []) if isinstance(f, dict)}

    # Step 1b: inference from BENCH-MARKS field list when block present.
    has_md = ("MD.NumCGsteps" in bench_field_names)

    # Step 2: sniff engine body.  All patterns are start-of-line
    # anchored (re.M) so commented-out post-processing templates
    # (`# %block ProjectedDensityOfStates`) don't false-match.
    has_md_engine = re.search(r"^\s*MD\.NumCGsteps\s+(\d+)", fdf_text, re.M)
    has_md_typeofrun = re.search(r"^\s*MD\.TypeOfRun\s+\S+", fdf_text, re.M)
    has_pdos = bool(re.search(
        r"^\s*%block\s+ProjectedDensityOfStates\b", fdf_text, re.M | re.I))
    has_transport = bool(re.search(
        r"^\s*%block\s+TS\.Elec\.\S+", fdf_text, re.M | re.I))

    matches: List[str] = []
    if has_md_engine and has_md_typeofrun:
        ncg = int(has_md_engine.group(1))
        if ncg > 0:
            matches.append("optimization")
    if has_pdos:
        matches.append("spectrum")
    if has_transport:
        matches.append("transport")

    if has_md:
        matches.append("optimization")

    matches = list(dict.fromkeys(matches))     # unique, ordered

    if len(matches) > 1:
        raise JobTypeAmbiguousError(
            f"Multiple job-type markers present: {matches}.  "
            "Add an explicit `job_type` field to the BENCH-MARKS block "
            "to disambiguate."
        )
    if not matches:
        return "unknown"
    return matches[0]


def _build_engine_input(fdf_path: Path) -> Dict[str, Any]:
    """Build the per-stage engine_input envelope per § 2.1."""
    from molbuilder.parse.scripts.atom_metadata import (
        _extract_atom_metadata_dict,
    )
    from molbuilder.parse.scripts.bench_marks import (
        _extract_bench_marks_dict,
    )
    from molbuilder.parse.scripts.header import _extract_header_text
    from molbuilder.parse.scripts.provenance import (
        _extract_provenance_dict,
    )
    from molbuilder.parse.scripts.user_custom import (
        _extract_user_custom_inner,
    )
    text = fdf_path.read_text(encoding="utf-8-sig", errors="replace")

    header = _extract_header_text(text)
    provenance = _extract_provenance_dict(text)
    bench = _extract_bench_marks_dict(text)
    atom_md = _extract_atom_metadata_dict(text)
    user_custom_lines = _extract_user_custom_inner(text)

    return {
        "schema_version": 1,
        "engine":         "siesta",
        "source_fdf":     fdf_path.name,
        "header":         header if header is not None else "",
        "provenance": {
            "present":            provenance is not None,
            **(provenance or {}),
        },
        "bench_marks": {
            "present": bench is not None,
            **(bench or {}),
        },
        "atom_metadata": {
            "present": atom_md is not None,
            **(atom_md or {}),
        },
        "user_custom_verbatim": "\n".join(user_custom_lines or []),
        "engine_body_summary":  _parse_engine_body_summary(text),
    }


# ---- file enumeration ------------------------------------------------ #


def _enumerate_files(run_dir: Path) -> Dict[str, List[Path]]:
    """Bucket relevant files in the dir by kind.

    Returns {"fdf": [...], "out": [...], "xv": [...], "struct_out": [...],
             "molstruct_json": [...], "ani": [...]}.  Paths sorted by
    name within each bucket.
    """
    by_kind: Dict[str, List[Path]] = {
        "fdf": [], "out": [], "xv": [], "struct_out": [],
        "molstruct_json": [], "ani": [],
    }
    for child in sorted(run_dir.iterdir()):
        if not child.is_file():
            continue
        name = child.name
        if name.endswith(".fdf"):
            by_kind["fdf"].append(child)
        elif name.endswith(".out"):
            by_kind["out"].append(child)
        elif name.endswith(".XV"):
            by_kind["xv"].append(child)
        elif name.endswith(".STRUCT_OUT"):
            by_kind["struct_out"].append(child)
        elif name.endswith(".molstruct.json"):
            by_kind["molstruct_json"].append(child)
        elif name.endswith(".ANI"):
            by_kind["ani"].append(child)
    return by_kind


def _build_source_files(files: Dict[str, List[Path]],
                        plot_data: Dict[str, Dict[str, List[List[float]]]],
                        out_run_states: Dict[str, str]
                        ) -> List[Dict[str, Any]]:
    """Build the source_files index per § 2.

    For each .fdf and .out we surface stage + mtime + size.  For .out
    we also surface the CG-step range observed in plot data.  Other
    kinds get the minimal envelope.
    """
    out: List[Dict[str, Any]] = []
    for kind_label, kind_key in (
        ("fdf", "fdf"), ("out", "out"), ("xv", "xv"),
        ("struct_out", "struct_out"), ("molstruct_json", "molstruct_json"),
        ("ani", "ani"),
    ):
        for path in files[kind_key]:
            entry: Dict[str, Any] = {
                "path":       path.name,
                "kind":       kind_label,
                "stage":      _detect_stage(path.name),
                "mtime":      _iso_z(path.stat().st_mtime),
                "size_bytes": path.stat().st_size,
            }
            if kind_label == "out":
                etot_bucket = plot_data.get("etot_per_cg", {}).get(path.name, [])
                if etot_bucket:
                    entry["cg_step_range"] = [
                        int(etot_bucket[0][0]),
                        int(etot_bucket[-1][0]),
                    ]
                else:
                    entry["cg_step_range"] = None
                entry["run_state"] = out_run_states.get(path.name, "unknown")
            out.append(entry)
    return out


# ---- plots from .out files ------------------------------------------- #


def _consolidate_plots(out_paths: List[Path]
                       ) -> Tuple[Dict[str, Dict[str, List[List[float]]]],
                                  Dict[str, str],
                                  List[Dict[str, Any]]]:
    """Parse each .out via the file-level TrajectoryParser registry,
    extract per-CG and per-SCF traces, bucket by .out filename.

    Returns:
        plots:           {plot_name: {out_filename: [[step, value], ...]}}
        out_run_states:  {out_filename: "finished"|"error"|"ongoing"|"unknown"}
                         (the engine parsers' run_state vocabulary,
                         passed through verbatim)
        parse_warnings:  flattened list of {source, line_no, snippet, error, category}
    """
    from molbuilder.parse import detect, UnknownFormatError
    plots: Dict[str, Dict[str, List[List[float]]]] = {
        "etot_per_cg":  {},
        "fmax_per_cg":  {},
        "scf_residual": {},
        "scf_etot":     {},
    }
    run_states: Dict[str, str] = {}
    warnings: List[Dict[str, Any]] = []

    # Order .out files by stage number (then by name).
    sorted_outs = sorted(
        out_paths,
        key=lambda p: ((_detect_stage(p.name) or 999), p.name),
    )

    scf_iter_global_offset = 0
    for out_path in sorted_outs:
        try:
            parser = detect(out_path)
            traj = parser.parse(out_path)
        except (UnknownFormatError, OSError, ValueError) as exc:
            warnings.append({
                "source":   out_path.name,
                "line_no":  None,
                "snippet":  None,
                "error":    f"parser failed: {exc}",
                "category": "parser-error",
            })
            run_states[out_path.name] = "unknown"
            continue

        run_states[out_path.name] = traj.run_state or "unknown"
        for w in (getattr(traj, "parse_warnings", None) or []):
            warnings.append({
                "source":   out_path.name,
                "line_no":  w.line_no,
                "snippet":  w.snippet,
                "error":    w.error,
                "category": w.category,
            })

        etot_bucket: List[List[float]] = []
        fmax_bucket: List[List[float]] = []
        scf_resid_bucket: List[List[float]] = []
        scf_etot_bucket:  List[List[float]] = []

        for frame in traj.frames:
            if frame.energy is not None:
                etot_bucket.append([int(frame.step_index), float(frame.energy)])
            if frame.max_force is not None:
                fmax_bucket.append([int(frame.step_index), float(frame.max_force)])
            scf_history = frame.scf_history or []
            for entry in scf_history:
                cycle = entry.get("cycle")
                if not isinstance(cycle, int):
                    continue
                global_iter = scf_iter_global_offset + cycle
                # Explicit None check — `... or ...` falsy-skips a
                # legitimate 0.0 (fully converged SCF cycle), then
                # falls through to dHmax (post-2026-06-19 fix I1).
                # bool is a subclass of int in Python; reject it
                # explicitly so a malformed True/False reading
                # doesn't get charted as a 1.0 / 0.0 residual.
                dDmax = entry.get("dDmax")
                if dDmax is None:
                    dDmax = entry.get("dHmax")
                if isinstance(dDmax, (int, float)) and not isinstance(dDmax, bool):
                    scf_resid_bucket.append([global_iter, float(dDmax)])
                e = entry.get("energy")
                if isinstance(e, (int, float)):
                    scf_etot_bucket.append([global_iter, float(e)])
            if scf_history:
                # advance offset by the per-frame SCF cycle count
                scf_iter_global_offset += len(scf_history)

        if etot_bucket:
            plots["etot_per_cg"][out_path.name] = etot_bucket
        if fmax_bucket:
            plots["fmax_per_cg"][out_path.name] = fmax_bucket
        if scf_resid_bucket:
            plots["scf_residual"][out_path.name] = scf_resid_bucket
        if scf_etot_bucket:
            plots["scf_etot"][out_path.name] = scf_etot_bucket

    return plots, run_states, warnings


# ---- geometry -------------------------------------------------------- #


def _build_geometry(run_dir: Path, files: Dict[str, List[Path]]
                    ) -> Dict[str, Any]:
    """Build the geometry field — reuses parsers/siesta_struct.read_xv
    when a .XV is present; falls back to .fdf initial coords otherwise."""
    from molbuilder.parse.coords.siesta_xv import (
        SiestaXVError,
        _read_xv as read_xv,
    )
    from ._assembler_helpers import (
        SiestaFdfStructureError,
        read_fdf_initial_coords,
    )

    structure = None
    coords_source: Optional[str] = None
    coords_state: str = "unknown"
    if files["xv"]:
        for xv in files["xv"]:
            try:
                structure = read_xv(xv)
                coords_source = xv.name
                coords_state = "converged"
                break
            except SiestaXVError:
                continue
    # Pick ONE anchor .fdf for everything — the highest-stage one,
    # matching the rest of decode_run_dir's "active stage"
    # semantics.  Mixing low-stage initial coords with high-stage
    # cell+regions silently produces a Frankenstein geometry
    # (post-2026-06-19 review fix B1).
    anchor_fdf: Optional[Path] = None
    if files["fdf"]:
        anchor_fdf = sorted(files["fdf"], key=_anchor_sort_key)[-1]
    if structure is None and anchor_fdf is not None:
        try:
            structure = read_fdf_initial_coords(
                anchor_fdf.read_text(encoding="utf-8-sig", errors="replace"))
            coords_source = anchor_fdf.name
            coords_state = "initial"
        except SiestaFdfStructureError:
            pass

    regions: Dict[str, List[int]] = {}
    frozen: List[int] = []
    if anchor_fdf is not None:
        from .bundle import _extract_script_source
        src = _extract_script_source(
            anchor_fdf.read_text(encoding="utf-8-sig", errors="replace"))
        regions = src["regions"] or {}
        frozen = src["frozen_atoms"] or []

    out: Dict[str, Any] = {
        "n_atoms":       len(structure.elements) if structure is not None else 0,
        "regions":       regions,
        "frozen_atoms":  frozen,
        "coords_source": coords_source,
        "coords_state":  coords_state,
    }
    if structure is not None:
        out["xyz"] = [
            [el, float(p[0]), float(p[1]), float(p[2])]
            for el, p in zip(structure.elements, structure.positions)
        ]
    else:
        out["xyz"] = []
    # Structure doesn't carry cell; read it from the same anchor
    # .fdf we used above (same single-source guarantee).
    out["cell"] = _read_cell_from_fdf(anchor_fdf) if anchor_fdf is not None else None
    return out


_LATTICE_BLOCK_RE = re.compile(
    r"%block\s+LatticeVectors\s*\n(.*?)%endblock\s+LatticeVectors",
    re.S | re.IGNORECASE,
)
_LATTICE_CONSTANT_RE = re.compile(
    r"^\s*LatticeConstant\b\s+([+\-0-9.eE]+)(?:\s+(\S+))?",
    re.M | re.IGNORECASE,
)
_BOHR_PER_ANG = 1.8897259886    # SIESTA's au -> Ang conversion factor


def _read_cell_from_fdf(fdf_path: Path) -> Optional[List[List[float]]]:
    """Return the LatticeVectors block contents as a 3x3 list in
    Angstroms, applying SIESTA's ``LatticeConstant <value> <unit>``
    scaling.  Returns None if the block is absent / malformed.

    SIESTA's convention (manual § 7.3.5): LatticeVectors are
    dimensionless multipliers of LatticeConstant.  Unit defaults
    to Bohr (``au`` or ``Bohr``) when not specified; ``Ang`` is
    the common explicit choice.  Without this scaling, a file with
    ``LatticeConstant 5.43 Ang`` + identity vectors silently
    produced a 1Å unit cube (post-2026-06-19 review fix B2).
    """
    text = fdf_path.read_text(encoding="utf-8-sig", errors="replace")
    m = _LATTICE_BLOCK_RE.search(text)
    if not m:
        return None

    # Resolve LatticeConstant (default: 1.0 Bohr per SIESTA convention).
    scale_to_ang = 1.0 / _BOHR_PER_ANG    # 1 Bohr in Å
    lc_m = _LATTICE_CONSTANT_RE.search(text)
    if lc_m:
        try:
            value = float(lc_m.group(1))
        except ValueError:
            return None
        unit = (lc_m.group(2) or "Bohr").strip().lower()
        if unit in ("ang", "angstrom"):
            scale_to_ang = value
        elif unit in ("bohr", "au"):
            scale_to_ang = value / _BOHR_PER_ANG
        else:
            # Unknown unit — assume Bohr per SIESTA's default.
            scale_to_ang = value / _BOHR_PER_ANG

    rows: List[List[float]] = []
    for raw in m.group(1).strip().splitlines():
        toks = raw.split()
        if len(toks) < 3:
            continue
        try:
            rows.append([
                float(toks[0]) * scale_to_ang,
                float(toks[1]) * scale_to_ang,
                float(toks[2]) * scale_to_ang,
            ])
        except ValueError:
            return None
        if len(rows) == 3:
            return rows
    return None


# ---- status + progress ---------------------------------------------- #


def _build_status(out_paths: List[Path],
                  out_run_states: Dict[str, str]
                  ) -> Dict[str, Any]:
    """Build the status envelope per § 5."""
    if not out_paths:
        return {
            "state":            "running",
            "detail":           "no .out file yet",
            "last_change_at":   None,
            "active_source":    None,
        }
    # Active source = highest stage, latest mtime.
    sorted_outs = sorted(
        out_paths,
        key=lambda p: (_detect_stage(p.name) or 0, p.stat().st_mtime),
    )
    active = sorted_outs[-1]
    active_state = out_run_states.get(active.name, "unknown")

    state = "running"
    detail = "running"
    age_s = max(0.0, _wall_now() - active.stat().st_mtime)
    # Two vocabularies meet here: the engine parsers emit run_state
    # "ongoing"|"finished"|"error"|"unknown"; the status envelope
    # (consumed by jobset/runstatus.py) speaks "running"|"stale"|
    # "failed"|"finished".  Map parser "error" -> envelope "failed".
    if active_state == "finished":
        state, detail = "finished", "job_completed"
    elif active_state == "error":
        state, detail = "failed", "engine error in active .out"
    elif age_s > 60.0:
        state, detail = "stale", f"no file growth in {int(age_s)}s"

    return {
        "state":          state,
        "detail":         detail,
        "last_change_at": _iso_z(active.stat().st_mtime),
        "active_source":  active.name,
    }


def _wall_now() -> float:
    """Wall-clock now() in POSIX seconds.  Indirected so tests can
    monkeypatch without touching time.time globally."""
    import time
    return time.time()


def _build_progress(plots: Dict[str, Dict[str, List[List[float]]]],
                    engine_inputs: Dict[str, Dict[str, Any]],
                    out_run_states: Dict[str, str],
                    stages_total_known: Optional[int],
                    ) -> Dict[str, Any]:
    """Build the progress envelope per § 6."""
    # current_cg_step = max CG step observed across all etot_per_cg buckets.
    current_cg_step = 0
    for bucket in plots.get("etot_per_cg", {}).values():
        if bucket:
            current_cg_step = max(current_cg_step, int(bucket[-1][0]))

    # target_cg_steps = MD.NumCGsteps of the highest-stage .fdf.
    target_cg_steps: Optional[int] = None
    if engine_inputs:
        last_fdf = sorted(
            engine_inputs.keys(),
            key=lambda n: (_detect_stage(n) or 0, n),
        )[-1]
        ebs = engine_inputs[last_fdf].get("engine_body_summary", {})
        raw = ebs.get("MD.NumCGsteps")
        if raw is not None:
            try:
                target_cg_steps = int(raw)
            except ValueError:
                pass

    # SCF iter global tally from scf_residual buckets.
    current_scf_iter_global = 0
    for bucket in plots.get("scf_residual", {}).values():
        if bucket:
            current_scf_iter_global = max(
                current_scf_iter_global, int(bucket[-1][0]))

    stages_completed = sum(
        1 for state in out_run_states.values() if state == "finished"
    )

    # ETA: only attempt when running + we have at least 2 SCF entries.
    eta_s: Optional[int] = None
    if target_cg_steps and current_cg_step < target_cg_steps:
        eta_s = None       # mean iter wall not yet tracked in Phase 1

    return {
        "current_cg_step":         current_cg_step,
        "target_cg_steps":         target_cg_steps,
        "current_scf_iter_global": current_scf_iter_global,
        "last_iter_wall_s":        None,
        "mean_iter_wall_s":        None,
        "estimated_remaining_s":   eta_s,
        "stages_completed":        stages_completed,
        "stages_total_known":      stages_total_known,
    }


# ---- top-level entry ------------------------------------------------- #


def decode_run_dir(run_dir: Path) -> JobResult:
    """Decode a project directory into a :class:`JobResult`.

    See ``docs/execution/running-a-job.md § 4`` § 2 for the field
    semantics.  This is the module-level convenience; equivalent
    to ``JobDirParser.parse(run_dir)``.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise ValueError(f"run_dir does not exist: {run_dir}")

    files = _enumerate_files(run_dir)

    # job_type classification on the highest-stage .fdf (the latest
    # config the user shipped).  Fail soft to "unknown" if no .fdf.
    job_type: str
    system_label: Optional[str] = None
    if files["fdf"]:
        active_fdf = sorted(files["fdf"], key=_anchor_sort_key)[-1]
        text = active_fdf.read_text(encoding="utf-8-sig", errors="replace")
        job_type = _classify_job_type(text)
        # SIESTA fdf keys are case-insensitive (post-2026-06-19 fix I4).
        m = re.search(r"^\s*SystemLabel\s+(\S+)", text, re.M | re.I)
        if m:
            system_label = m.group(1)
    else:
        job_type = "unknown"

    plots, out_run_states, parse_warnings = _consolidate_plots(files["out"])
    engine_input_by_stage: Dict[str, Dict[str, Any]] = {}
    for fdf in files["fdf"]:
        engine_input_by_stage[fdf.name] = _build_engine_input(fdf)
    geometry = _build_geometry(run_dir, files)
    source_files = _build_source_files(files, plots, out_run_states)
    stages_total_known: Optional[int] = None
    fdf_stages = {_detect_stage(p.name) for p in files["fdf"]}
    fdf_stages.discard(None)
    if fdf_stages:
        stages_total_known = max(fdf_stages)
    status = _build_status(files["out"], out_run_states)
    progress = _build_progress(plots, engine_input_by_stage,
                               out_run_states, stages_total_known)

    warnings_typed = [
        ParseWarning(
            source=w.get("source", ""),
            line_no=w.get("line_no"),
            snippet=w.get("snippet"),
            error=w.get("error", ""),
            category=w.get("category", ""),
        )
        for w in parse_warnings
    ]

    return JobResult(
        schema_version=SCHEMA_VERSION,
        parsed_at=_iso_z(_wall_now()),
        # Use the slug ("job-dir") to match the engines + sidecars
        # envelope convention (post-2026-06-19 round-2 fix —
        # was the literal classname "JobDirParser").
        parser_name=JobDirParser.name,
        source=str(run_dir.resolve()),
        job_type=job_type,
        engine="siesta",
        system_label=system_label,
        run_dir=str(run_dir.resolve()),
        status=status,
        progress=progress,
        geometry=geometry,
        plots=plots,
        source_files=source_files,
        engine_input_by_stage=engine_input_by_stage,
        parse_warnings=warnings_typed,
        diagnostics={
            "active_decoder_path": __file__,
            "tick_count":          1,
            "last_tick_wall_ms":   None,
        },
    )


# --------------------------------------------------------------------- #
#  Class wrapper — the canonical DirParser entry point.                 #
# --------------------------------------------------------------------- #


class JobDirParser(DirParser):
    """Parse a project directory (any SIESTA / PySCF run dir) into a
    :class:`JobResult`.

    Per parse-module.md § 7, composes registered FileParsers + the
    script-contract extractors — no inline file-level parsing.
    """
    name   = "job-dir"
    label  = "molbuilder job directory (SIESTA / PySCF)"
    output = JobResult

    @classmethod
    def can_parse(cls, run_dir: Path) -> bool:
        """Claim any directory that contains at least one ``.fdf``
        engine script (SIESTA / TranSIESTA).  A directory with only
        inputs but no .out yet is still a valid (running / pre-
        launch) job dir.

        ``.py`` (PySCF) dirs are NOT claimed yet — the decoder
        produces empty results for them.  Re-add when PySCF
        DirParser support ships (post-2026-06-19 fix B3).
        """
        try:
            for child in run_dir.iterdir():
                if child.is_file() and child.name.endswith(".fdf"):
                    return True
        except OSError:
            return False
        return False

    @classmethod
    def parse(cls, run_dir: Path) -> JobResult:
        return decode_run_dir(run_dir)
