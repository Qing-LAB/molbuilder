"""molbuilder bench -- in-tree benchmark harness.

Reads the BENCH-MARKS block (per docs/protocols/script-contract.md)
from a generated .fdf, sweeps over a small set of (np, omp,
BlockSize) test points, runs SIESTA on each variant, parses the
.out for the effective values and per-iter wall time, prints a
ranked comparison.

Single-engine for now: ``molbuilder bench siesta-gpu <project>``.
PySCF lands when its bench fields are wired in BENCH-MARKS.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .. import script_contract as _sc


# --------------------------------------------------------------------- #
#  Point grid                                                            #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class Point:
    """One (mpi_np, omp, BlockSize) test point."""
    np: int
    omp: int
    bs: int

    @property
    def slug(self) -> str:
        return f"np{self.np}_omp{self.omp}_bz{self.bs}"

    @classmethod
    def parse(cls, s: str) -> "Point":
        """Parse a "4,2,64" or "np=4,omp=2,bs=64" triplet."""
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 3:
            raise ValueError(f"need 3 comma-separated values, got: {s!r}")
        vals: List[int] = []
        for p in parts:
            tok = p.split("=", 1)[1] if "=" in p else p
            vals.append(int(tok))
        return cls(np=vals[0], omp=vals[1], bs=vals[2])


# Five-point default sweep from docs/protocols/script-contract.md
# discussion: holds variables one at a time to isolate which axis
# matters.  np * omp <= 10 for every row (fits the GPU-proximate
# socket on the standard workstation).
DEFAULT_POINTS: List[Point] = [
    Point(4, 2,  32),   # baseline-with-smaller-block
    Point(4, 2,  64),   # baseline anchor (ELPA published 4 ranks/GPU)
    Point(4, 2, 128),   # baseline-with-bigger-block
    Point(2, 5,  64),   # fewer ranks, more OMP (host-bound test)
    Point(10, 1, 64),   # max host parallelism (MPS Hyper-Q test)
]


# --------------------------------------------------------------------- #
#  Bench-marks parsing                                                  #
# --------------------------------------------------------------------- #


_FIELD_RE = re.compile(
    r"^#\s*field\s+(?P<name>\S+)\s+"
    r"anchor=(?P<anchor>\S+)\s+"
    r"type=(?P<type>\S+)"
    r"(?:\s+range=\[(?P<lo>[^,]+),(?P<hi>[^\]]+)\])?"
    r"(?:\s+unit=(?P<unit>\S+))?"
    r"(?:\s+default=(?P<default>\S+))?\s*$"
)


@dataclass(frozen=True)
class FieldDecl:
    name: str
    anchor: str
    type_: str
    range_: Optional[Tuple[float, float]] = None
    unit: Optional[str] = None
    default: Optional[str] = None


def parse_bench_marks(fdf_text: str) -> Optional[Dict[str, Any]]:
    """Read the BENCH-MARKS block from a rendered .fdf.

    Returns ``{"version": "v1", "metadata": {...}, "fields": [FieldDecl, ...]}``
    or ``None`` when the block is missing.
    """
    lines = fdf_text.splitlines()
    begin = end = None
    for i, line in enumerate(lines):
        m = _sc.MARKER_RE.match(line)
        if not m:
            continue
        if m.group(1) != _sc.BLOCK_BENCH_MARKS:
            continue
        if m.group(2) == "BEGIN":
            begin = i
        elif m.group(2) == "END" and begin is not None:
            end = i
            break
    if begin is None or end is None:
        return None
    body = lines[begin + 1: end]
    version = None
    metadata: Dict[str, Any] = {}
    fields: List[FieldDecl] = []
    for line in body:
        m = _FIELD_RE.match(line)
        if m:
            rng = None
            if m.group("lo") is not None and m.group("hi") is not None:
                try:
                    rng = (float(m.group("lo")), float(m.group("hi")))
                except ValueError:
                    pass
            fields.append(FieldDecl(
                name    = m.group("name"),
                anchor  = m.group("anchor"),
                type_   = m.group("type"),
                range_  = rng,
                unit    = m.group("unit"),
                default = m.group("default"),
            ))
            continue
        # Top-level key/value (e.g. "#   version v1", "#   n_atoms 212")
        stripped = line.lstrip("#").strip()
        if not stripped:
            continue
        bits = stripped.split(None, 1)
        if not bits:
            continue
        key = bits[0]
        val = bits[1] if len(bits) > 1 else ""
        if key == "version":
            version = val
        else:
            metadata[key] = val
    return {"version": version, "metadata": metadata, "fields": fields}


# --------------------------------------------------------------------- #
#  .fdf override + bench layout                                         #
# --------------------------------------------------------------------- #


def override_field_value(fdf_text: str, anchor: str, new_value: str) -> str:
    """Replace the first occurrence of ``^<anchor>\\b`` line in the
    engine body with ``<anchor>          <new_value>``.

    Returns the modified text.  Caller is responsible for confirming
    the anchor exists (e.g., via the field decl).
    """
    pattern = re.compile(rf"^(\s*){re.escape(anchor)}(\s+)(\S.*)$",
                         re.MULTILINE)
    replacement = rf"\g<1>{anchor}          {new_value}    # bench override"
    return pattern.sub(replacement, fdf_text, count=1)


def force_max_scf_iters(fdf_text: str, n: int) -> str:
    """Cap MaxSCFIterations.  If the line is missing, append it
    inside the ENGINE BODY (just before USER-CUSTOM if present)."""
    if re.search(r"^\s*MaxSCFIterations\s+", fdf_text, re.MULTILINE):
        return override_field_value(fdf_text, "MaxSCFIterations", str(n))
    # Inject after the last contract block, before user-custom.
    new_line = f"MaxSCFIterations  {n}    # bench override"
    if "user-custom BEGIN" in fdf_text:
        return fdf_text.replace(
            "# === molbuilder user-custom BEGIN ===",
            f"{new_line}\n\n# === molbuilder user-custom BEGIN ===",
            1,
        )
    return fdf_text + "\n" + new_line + "\n"


def disable_md(fdf_text: str) -> str:
    """Zero out MD.NumCGsteps / NumBroydenSteps so the bench runs SCF
    only (no relaxation).  Leaves the relaxer keyword intact."""
    for kw in ("MD.NumCGsteps", "MD.NumBroydenSteps", "MD.NumFIRESteps"):
        if re.search(rf"^\s*{re.escape(kw)}\s+", fdf_text, re.MULTILINE):
            fdf_text = override_field_value(fdf_text, kw, "0")
    return fdf_text


# --------------------------------------------------------------------- #
#  Result parsing                                                       #
# --------------------------------------------------------------------- #


@dataclass
class PointResult:
    point: Point
    walltime_s: int
    iters_done: int
    first_iter_s: Optional[float] = None
    effective_np: Optional[int] = None
    effective_omp: Optional[int] = None
    effective_bs: Optional[int] = None
    error: Optional[str] = None

    @property
    def avg_iter_s(self) -> Optional[float]:
        if self.iters_done >= 2 and self.first_iter_s is not None:
            return round(
                (self.walltime_s - self.first_iter_s) / (self.iters_done - 1),
                1,
            )
        if self.iters_done >= 1:
            return round(self.walltime_s / self.iters_done, 1)
        return None


_ITER_SCF_RE  = re.compile(r"^   scf:\s+\d+", re.MULTILINE)
_TIMER_RE     = re.compile(
    r"^timer:\s+Routine,Calls,Time,%\s*=\s*IterSCF\s+\d+\s+([\d.]+)",
    re.MULTILINE,
)
_BLOCKSIZE_RE = re.compile(
    r"ProcessorY,\s*Blocksize:\s*\d+\s+(\d+)", re.IGNORECASE
)
_BANNER_RE    = re.compile(
    r"chosen\s+(\d+)\s+ranks\s*[x××]\s*(\d+)\s+threads"
)


def parse_point_out(out_path: Path) -> Dict[str, Any]:
    """Pull iters done, first-iter time, and effective values from a
    SIESTA .out (or its run0.out variant).  Returns a partial dict;
    caller fills in walltime."""
    if not out_path.exists():
        return {"iters_done": 0}
    text = out_path.read_text(encoding="utf-8", errors="replace")
    iters_done = sum(1 for _ in _ITER_SCF_RE.finditer(text))
    first_iter_s: Optional[float] = None
    m = _TIMER_RE.search(text)
    if m:
        try:
            first_iter_s = float(m.group(1))
        except ValueError:
            pass
    effective_bs: Optional[int] = None
    m = _BLOCKSIZE_RE.search(text)
    if m:
        try:
            effective_bs = int(m.group(1))
        except ValueError:
            pass
    effective_np = effective_omp = None
    m = _BANNER_RE.search(text)
    if m:
        try:
            effective_np = int(m.group(1))
            effective_omp = int(m.group(2))
        except ValueError:
            pass
    return {
        "iters_done":    iters_done,
        "first_iter_s":  first_iter_s,
        "effective_np":  effective_np,
        "effective_omp": effective_omp,
        "effective_bs":  effective_bs,
    }


# --------------------------------------------------------------------- #
#  Bench runner                                                         #
# --------------------------------------------------------------------- #


def _resolve_project(project_dir: Path) -> Tuple[Path, Path, str]:
    """Find the .fdf and .run.sh inside a project dir.  Returns
    (fdf_path, runsh_path, basename) or raises ValueError."""
    fdfs = sorted(p for p in project_dir.glob("*.fdf")
                  if "bench" not in p.parts[-2:])  # skip bench subdirs
    if not fdfs:
        raise ValueError(f"no .fdf found in {project_dir}")
    fdf = fdfs[0]
    basename = fdf.stem
    runsh = project_dir / f"{basename}.run.sh"
    if not runsh.exists():
        raise ValueError(
            f"expected wrapper not found: {runsh}. "
            f"Run 'molbuilder run {fdf.name}' first."
        )
    return fdf, runsh, basename


def _prepare_point_dir(project_dir: Path, basename: str,
                       fdf_text: str, runsh_text: str,
                       point: Point, iters: int,
                       cold: bool) -> Path:
    """Create the subdirectory for one test point and seed it."""
    bench_root = project_dir / f"{basename}.bench"
    point_dir = bench_root / point.slug
    if point_dir.exists():
        shutil.rmtree(point_dir)
    point_dir.mkdir(parents=True)
    # Write modified .fdf
    modified = override_field_value(fdf_text, "BlockSize", str(point.bs))
    modified = force_max_scf_iters(modified, iters)
    modified = disable_md(modified)
    (point_dir / f"{basename}.fdf").write_text(modified, encoding="utf-8")
    # Copy the wrapper as-is.
    (point_dir / f"{basename}.run.sh").write_text(runsh_text, encoding="utf-8")
    os.chmod(point_dir / f"{basename}.run.sh", 0o755)
    # Symlink read-only siblings (psml, ion, ion.nc, ion.xml).
    for pattern in ("*.psml", "*.ion", "*.ion.nc", "*.ion.xml"):
        for sib in project_dir.glob(pattern):
            link = point_dir / sib.name
            try:
                link.symlink_to(sib)
            except OSError:
                # Cross-fs fallback: copy.
                shutil.copy2(sib, link)
    # Copy .DM/.CG/.XV warm-start unless --cold.
    if not cold:
        for suffix in (".DM", ".CG", ".XV"):
            for sib in project_dir.glob(f"*{suffix}"):
                shutil.copy2(sib, point_dir / sib.name)
    return point_dir


def run_point(point_dir: Path, basename: str, point: Point,
              quiet: bool = True) -> PointResult:
    """Execute one test point.  Times the wall, parses the .out
    afterwards, returns a PointResult."""
    env = os.environ.copy()
    env["MOLBUILDER_MPI_NP"]         = str(point.np)
    env["MOLBUILDER_OMP_NUM_THREADS"] = str(point.omp)
    start = int(time.time())
    try:
        cp = subprocess.run(
            ["bash", f"{basename}.run.sh", "--force"],
            cwd=str(point_dir),
            env=env,
            stdout=subprocess.DEVNULL if quiet else None,
            stderr=subprocess.STDOUT  if quiet else None,
            timeout=60 * 60,  # 1 hour ceiling per point
        )
        rc = cp.returncode
    except subprocess.TimeoutExpired:
        return PointResult(
            point=point, walltime_s=int(time.time() - start), iters_done=0,
            error="timed out (>1h)",
        )
    walltime = int(time.time() - start)
    out_path = point_dir / f"{basename}-run0.out"
    parsed = parse_point_out(out_path)
    return PointResult(
        point        = point,
        walltime_s   = walltime,
        iters_done   = int(parsed.get("iters_done", 0)),
        first_iter_s = parsed.get("first_iter_s"),
        effective_np = parsed.get("effective_np"),
        effective_omp= parsed.get("effective_omp"),
        effective_bs = parsed.get("effective_bs"),
        error        = None if rc == 0 else f"siesta exit rc={rc}",
    )


def write_results_csv(results: List[PointResult],
                      bench_root: Path) -> Path:
    """Write a per-bench CSV with requested + effective values."""
    csv = bench_root / "results.csv"
    with csv.open("w", encoding="utf-8") as fh:
        fh.write(
            "point,req_np,eff_np,req_omp,eff_omp,req_bs,eff_bs,"
            "iters,first_iter_s,avg_iter_s,wall_s,error\n"
        )
        for r in results:
            fh.write(
                f"{r.point.slug},"
                f"{r.point.np},{r.effective_np if r.effective_np else ''},"
                f"{r.point.omp},{r.effective_omp if r.effective_omp else ''},"
                f"{r.point.bs},{r.effective_bs if r.effective_bs else ''},"
                f"{r.iters_done},"
                f"{r.first_iter_s if r.first_iter_s else ''},"
                f"{r.avg_iter_s if r.avg_iter_s else ''},"
                f"{r.walltime_s},"
                f"{r.error or ''}\n"
            )
    return csv


__all__ = [
    "Point", "PointResult", "FieldDecl",
    "DEFAULT_POINTS", "parse_bench_marks",
    "override_field_value", "force_max_scf_iters", "disable_md",
    "parse_point_out",
    "_resolve_project", "_prepare_point_dir", "run_point",
    "write_results_csv",
]
