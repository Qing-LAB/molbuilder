"""SIESTA peak-memory estimator (for sbatch ``--mem`` sizing).

The sbatch emitter needs a defensible ``--mem`` value *before* the job
runs.  SIESTA's peak RSS is dominated by a handful of terms that we can
read straight out of the ``.fdf``:

  * the **dense Hamiltonian / overlap / density matrices** -- O(N_orb^2)
    per matrix, real (8 B) for Gamma-only runs, complex (16 B) when a
    k-point mesh forces complex arithmetic.  This is usually the single
    biggest term for the metal-junction systems molbuilder targets.
  * the **real-space integration mesh** -- the 3D grid SIESTA lays over
    the cell for the Hartree + XC potentials.  Grid density scales with
    ``sqrt(MeshCutoff)`` per cell edge.
  * a **per-rank replication** term -- ScaLAPACK / mesh bookkeeping that
    grows roughly linearly with the MPI rank count on a node.
  * a fixed **base** (binary, libraries, pseudo tables, I/O buffers).

None of this needs SIESTA itself or any heavy dependency: we parse the
``.fdf`` with a tiny tolerant reader (this module is pure-stdlib) and
reuse :mod:`molbuilder.pseudos` to read the per-element valence
configuration out of the PSML headers when a pseudo library is on disk
(so the orbital count reflects the *actual* basis, not a guess).

The model is intentionally a small set of tunable linear coefficients
(:class:`MemModel`) so a site can recalibrate it against observed RSS on
their hardware without touching the parsing.  Everything is best-effort
and never raises on a malformed ``.fdf``: missing keys collapse to
documented defaults plus a parse-warning string.

Public API:

    FdfMemInputs            -- the parsed-from-fdf inputs to the model
    parse_fdf_mem_inputs    -- read those inputs from a .fdf path
    estimate_norb           -- total basis-orbital count for the system
    MemModel                -- tunable coefficients (with from_config)
    MemEstimate             -- the result (request_gb + breakdown)
    estimate_siesta_memory  -- the top-level entry point

NOTE on the mesh term: :class:`FdfMemInputs` only carries the cell
*volume* (not the three lattice vectors), so the per-edge grid count
uses the isotropic cube-root edge ``V**(1/3)`` as a proxy for each of
the three lattice-vector norms.  For the near-orthorhombic cells these
calculations use this is within a few percent of the anisotropic count
and keeps the dataclass minimal.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..pseudos import parse_psml_header

# 1 Bohr in Angstrom (CODATA-ish; matches the value used elsewhere in
# the SIESTA layer).
_BOHR_ANG = 0.529177

# Subshell letter -> angular-momentum quantum number l.
_L_OF = {"s": 0, "p": 1, "d": 2, "f": 3}

# Hardcoded valence l-shells for the elements these calculations
# actually contain.  Reading the PSML header (when a library is given)
# overrides this; it is only the no-pseudo fallback.
_VALENCE_L_BY_ELEMENT: Dict[str, List[int]] = {
    "H":  [0],            # 1s
    "He": [0],            # 1s
    "C":  [0, 1],         # 2s 2p
    "N":  [0, 1],         # 2s 2p
    "O":  [0, 1],         # 2s 2p
    "S":  [0, 1],         # 3s 3p
    "Au": [0, 1, 2],      # 5d 6s (treat as s,p,d)
}

# Minimal symbol -> Z table, enough to drive the block-based fallback
# for elements not in _VALENCE_L_BY_ELEMENT.  Index = Z, so position 0
# is a placeholder.
_PERIODIC = [
    "",
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr",
]
_SYMBOL_Z = {sym: z for z, sym in enumerate(_PERIODIC) if sym}


# --------------------------------------------------------------------- #
#  fdf parsing                                                          #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class FdfMemInputs:
    """The slice of a SIESTA ``.fdf`` that drives the memory model."""
    n_atoms:          int
    mesh_cutoff_ry:   float
    cell_volume_ang3: float
    basis_size:       str
    n_kpoints:        int
    elements:         Dict[str, int]
    parse_warnings:   List[str] = field(default_factory=list)


def _norm_key(key: str) -> str:
    """SIESTA treats ``-``, ``_`` and ``.`` as interchangeable in keys
    and is case-insensitive.  Collapse all of that to a canonical form."""
    return (key.lower()
            .replace("-", "").replace("_", "").replace(".", ""))


def _strip_comment(line: str) -> str:
    """Drop an fdf ``#`` comment (whole-line or trailing)."""
    return line.split("#", 1)[0]


def _parse_fdf_raw(text: str) -> Tuple[Dict[str, List[str]],
                                       Dict[str, List[List[str]]]]:
    """Split fdf text into ``(scalars, blocks)``.

    * ``scalars[norm_key] = [tok, tok, ...]`` -- the tokens after the
      key on a one-line directive (first occurrence wins).
    * ``blocks[norm_name] = [[tok, ...], ...]`` -- one inner list of
      tokens per non-empty row of a ``%block``/``%endblock`` section.
    """
    scalars: Dict[str, List[str]] = {}
    blocks: Dict[str, List[List[str]]] = {}
    in_block: Optional[str] = None
    rows: List[List[str]] = []
    for raw in text.splitlines():
        line = _strip_comment(raw).strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("%block"):
            name = line[len("%block"):].strip()
            in_block = _norm_key(name)
            rows = []
            continue
        if low.startswith("%endblock"):
            if in_block is not None:
                blocks.setdefault(in_block, rows)
            in_block = None
            rows = []
            continue
        if in_block is not None:
            rows.append(line.split())
            continue
        toks = line.split()
        if not toks:
            continue
        scalars.setdefault(_norm_key(toks[0]), toks[1:])
    return scalars, blocks


def _det3(m: List[List[float]]) -> float:
    """Determinant of a 3x3 matrix given as row lists (pure stdlib)."""
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    return (a * (e * i - f * h)
            - b * (d * i - f * g)
            + c * (d * h - e * g))


def parse_fdf_mem_inputs(fdf_path) -> FdfMemInputs:
    """Read the memory-model inputs from a SIESTA ``.fdf`` file.

    Tolerant: any missing / malformed key falls back to a documented
    default and appends a human-readable string to ``parse_warnings``;
    this function never raises on a content problem.
    """
    warnings: List[str] = []
    path = Path(fdf_path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return FdfMemInputs(
            n_atoms=0, mesh_cutoff_ry=0.0, cell_volume_ang3=1.0,
            basis_size="DZP", n_kpoints=1, elements={},
            parse_warnings=[f"could not read {path}: {exc}"],
        )

    scalars, blocks = _parse_fdf_raw(text)

    # ----- NumberOfAtoms -------------------------------------------
    n_atoms = 0
    if "numberofatoms" in scalars and scalars["numberofatoms"]:
        try:
            n_atoms = int(float(scalars["numberofatoms"][0]))
        except ValueError:
            warnings.append(
                f"unparseable NumberOfAtoms {scalars['numberofatoms']!r}")
    else:
        warnings.append("NumberOfAtoms missing -> 0")

    # ----- MeshCutoff (value + optional Ry/Ha unit) ----------------
    mesh_cutoff_ry = 0.0
    if "meshcutoff" in scalars and scalars["meshcutoff"]:
        toks = scalars["meshcutoff"]
        try:
            val = float(toks[0])
            unit = toks[1].lower() if len(toks) > 1 else "ry"
            if unit in ("ha", "hartree"):
                val *= 2.0          # 1 Ha = 2 Ry
            mesh_cutoff_ry = val
        except ValueError:
            warnings.append(f"unparseable MeshCutoff {toks!r}")
    else:
        warnings.append("MeshCutoff missing -> 0 Ry")

    # ----- PAO.BasisSize -------------------------------------------
    if "paobasissize" in scalars and scalars["paobasissize"]:
        basis_size = scalars["paobasissize"][0]
    else:
        basis_size = "DZP"
        warnings.append("PAO.BasisSize missing -> DZP")

    # ----- LatticeConstant + LatticeVectors -> volume --------------
    lat_const_ang = 1.0
    if "latticeconstant" in scalars and scalars["latticeconstant"]:
        toks = scalars["latticeconstant"]
        try:
            val = float(toks[0])
            unit = toks[1].lower() if len(toks) > 1 else "ang"
            lat_const_ang = val * _BOHR_ANG if unit.startswith("bohr") else val
        except ValueError:
            warnings.append(f"unparseable LatticeConstant {toks!r}")
    else:
        warnings.append("LatticeConstant missing -> 1.0 Ang")

    cell_volume_ang3 = max(1.0, n_atoms * 20.0)  # ~20 A^3/atom fallback
    if "latticevectors" in blocks and len(blocks["latticevectors"]) >= 3:
        try:
            vecs = [[float(x) for x in row[:3]]
                    for row in blocks["latticevectors"][:3]]
            scaled = [[lat_const_ang * c for c in row] for row in vecs]
            cell_volume_ang3 = abs(_det3(scaled))
            if cell_volume_ang3 <= 0.0:
                cell_volume_ang3 = max(1.0, n_atoms * 20.0)
                warnings.append("LatticeVectors gave non-positive volume")
        except (ValueError, IndexError):
            warnings.append("unparseable LatticeVectors block")
    else:
        warnings.append("LatticeVectors block missing -> volume estimated")

    # ----- k-grid (Monkhorst-Pack diagonal) ------------------------
    n_kpoints = 1
    if "kgridmonkhorstpack" in blocks and len(blocks["kgridmonkhorstpack"]) >= 3:
        try:
            rows = blocks["kgridmonkhorstpack"][:3]
            diag = [int(float(rows[i][i])) for i in range(3)]
            prod = diag[0] * diag[1] * diag[2]
            n_kpoints = prod if prod > 0 else 1
        except (ValueError, IndexError):
            warnings.append("unparseable kgrid_Monkhorst_Pack block -> gamma")
    # No warning when absent: gamma-only is a legitimate default.

    # ----- elements (species labels x coords) ----------------------
    elements: Dict[str, int] = {}
    species_map: Dict[int, str] = {}
    if "chemicalspecieslabel" in blocks:
        for row in blocks["chemicalspecieslabel"]:
            if len(row) >= 3:
                try:
                    species_map[int(row[0])] = row[2]
                except ValueError:
                    continue
    coords = blocks.get("atomiccoordinatesandatomicspecies")
    if not coords:
        warnings.append(
            "AtomicCoordinatesAndAtomicSpecies block missing -> "
            "element counts unavailable")
    elif not species_map:
        warnings.append(
            "ChemicalSpeciesLabel block missing -> cannot map species "
            "indices to symbols")
    else:
        for row in coords:
            if not row:
                continue
            try:
                idx = int(float(row[-1]))
            except ValueError:
                continue
            sym = species_map.get(idx)
            if sym is None:
                warnings.append(f"species index {idx} not in "
                                f"ChemicalSpeciesLabel")
                continue
            elements[sym] = elements.get(sym, 0) + 1

    return FdfMemInputs(
        n_atoms=n_atoms,
        mesh_cutoff_ry=mesh_cutoff_ry,
        cell_volume_ang3=cell_volume_ang3,
        basis_size=basis_size,
        n_kpoints=n_kpoints,
        elements=elements,
        parse_warnings=warnings,
    )


# --------------------------------------------------------------------- #
#  orbital counting                                                     #
# --------------------------------------------------------------------- #


def _parse_basis(basis_size: str) -> Tuple[int, bool, int]:
    """Decode a PAO.BasisSize string into ``(nzeta, has_pol, n_pol)``.

    SZ/S -> 1, DZ/D -> 2, TZ/T -> 3 zeta functions; a trailing ``P``
    adds one polarization shell (``TZ2P`` -> two).  Unrecognized
    strings fall back to DZP ``(2, True, 1)``.
    """
    m = re.match(r"^([SDT])Z?(\d*)(P?)$", basis_size.strip().upper())
    if not m:
        return (2, True, 1)         # DZP default
    nzeta = {"S": 1, "D": 2, "T": 3}[m.group(1)]
    has_pol = bool(m.group(3))
    n_pol = int(m.group(2)) if (has_pol and m.group(2)) else 1
    return (nzeta, has_pol, n_pol)


def _parse_valence_config(valence_config: str) -> List[int]:
    """Extract the set of valence l-values from a config string like
    ``"[Ar] 3d6 4s2"`` -> ``[0, 2]``.  Returns sorted unique l-values
    (empty if nothing parseable)."""
    found = set()
    for _n, letter in re.findall(r"(\d)([spdf])", valence_config.lower()):
        found.add(_L_OF[letter])
    return sorted(found)


def _block_l_shells_for_z(z: int) -> List[int]:
    """Minimal Z -> valence-block rule for the fallback path."""
    if z <= 2:
        return [0]                          # H, He : s only
    if (21 <= z <= 30) or (39 <= z <= 48) or (72 <= z <= 80):
        return [0, 1, 2]                    # transition metals : s,p,d
    if (57 <= z <= 71) or (89 <= z <= 103):
        return [0, 1, 2, 3]                 # lanthanides/actinides : s,p,d,f
    return [0, 1]                           # main-group s/p


def _valence_l_shells(symbol: str,
                      psml_lib: Optional[Path],
                      warnings: List[str]) -> List[int]:
    """Resolve the valence l-shells for ``symbol``.

    Prefers the PSML header's ``valence_config`` (the basis the run will
    actually use) when ``psml_lib/<El>.psml`` exists; otherwise uses the
    hardcoded per-element table, then a Z-block fallback.
    """
    if psml_lib is not None:
        cand = Path(psml_lib) / f"{symbol}.psml"
        if cand.is_file():
            info = parse_psml_header(cand)
            ls = _parse_valence_config(info.valence_config)
            if ls:
                return ls
            warnings.append(
                f"{cand.name} has no parseable valence-config -> "
                f"using fallback l-shells for {symbol}")
    if symbol in _VALENCE_L_BY_ELEMENT:
        return list(_VALENCE_L_BY_ELEMENT[symbol])
    z = _SYMBOL_Z.get(symbol)
    if z is None:
        warnings.append(
            f"unknown element {symbol!r} -> assuming s,p valence")
        return [0, 1]
    return _block_l_shells_for_z(z)


def estimate_norb(inputs: FdfMemInputs,
                  psml_lib=None) -> Tuple[int, List[str]]:
    """Estimate the total number of basis orbitals in the system.

    For each element: ``Sum_over_valence_l (2l+1) * nzeta`` orbitals,
    plus -- when the basis is polarized -- one extra shell at
    ``l_max + 1`` contributing ``(2*(l_max+1)+1) * n_pol`` orbitals.
    Scaled by the per-element atom count.  Returns ``(N_orb,
    warnings)``.
    """
    warnings: List[str] = []
    psml = Path(psml_lib) if psml_lib is not None else None
    nzeta, has_pol, n_pol = _parse_basis(inputs.basis_size)

    n_orb = 0
    for symbol, count in inputs.elements.items():
        ls = _valence_l_shells(symbol, psml, warnings)
        l_max = max(ls)
        per_atom = sum((2 * l + 1) * nzeta for l in ls)
        if has_pol:
            per_atom += (2 * (l_max + 1) + 1) * n_pol
        n_orb += count * per_atom
    if not inputs.elements:
        warnings.append("no element counts available -> N_orb = 0")
    return n_orb, warnings


# --------------------------------------------------------------------- #
#  the model                                                            #
# --------------------------------------------------------------------- #


@dataclass
class MemModel:
    """Tunable linear coefficients for the peak-memory estimate.

    The defaults are deliberately conservative; a site can recalibrate
    them against observed peak RSS via :meth:`from_config`.
    """
    base_gb:     float = 2.0    # fixed binary + libraries + buffers
    c_dense:     float = 4.0    # ~4 complex N_orb^2 matrices per k-point
    c_mesh:      float = 0.05   # GB per million real-space mesh points
    c_rank:      float = 1.5    # GB of per-rank replication, per MPI task.
                                # Calibrated to bracket a real OOM: the BDT
                                # Au junction used ~162 GB at np=4 but
                                # >240 GB at np=64 -- ~1.4 GB/rank of
                                # replicated sparse structures + ScaLAPACK
                                # workspace + comm buffers.  Tune against a
                                # successful run's ``sacct MaxRSS``.
    safety:      float = 1.3    # global headroom multiplier
    extra_gb:    float = 0.0    # flat add-on (e.g. NEGF / Green funcs)
    floor_gb:    float = 4.0    # never request less than this
    node_mem_gb: Optional[float] = None  # hard cap (physical node RAM)

    @classmethod
    def from_config(cls, d: Optional[dict]) -> "MemModel":
        """Build a model from a dict, ignoring unknown keys and using
        defaults for anything missing.  Tolerant: ``None`` -> defaults."""
        if not d:
            return cls()
        fields = {f for f in cls.__dataclass_fields__}
        kept = {k: v for k, v in d.items() if k in fields}
        return cls(**kept)


@dataclass(frozen=True)
class MemEstimate:
    """The result of a memory estimate, with a compact breakdown."""
    request_gb: int             # final --mem value (GB), after cap+floor
    dense_gb:   float
    mesh_gb:    float
    repl_gb:    float
    base_gb:    float
    n_orb:      int
    mesh_mpts:  float
    capped:     bool            # True if node_mem_gb cap clamped it
    warnings:   List[str] = field(default_factory=list)
    # Carried so breakdown_str can render the safety x / extra + terms;
    # not part of the component breakdown itself.
    safety:     float = 1.0
    extra_gb:   float = 0.0

    def breakdown_str(self) -> str:
        """Compact one-line breakdown for logs, e.g.
        ``"dense=10 mesh=12 repl=26 base=2, x1.3 +0 -> 64G"``."""
        return (f"dense={int(self.dense_gb)} mesh={int(self.mesh_gb)} "
                f"repl={int(self.repl_gb)} base={int(self.base_gb)}, "
                f"x{self.safety:g} +{int(self.extra_gb)} "
                f"-> {self.request_gb}G")


def estimate_siesta_memory(fdf_path, ntasks: int, *,
                           model: Optional[MemModel] = None,
                           psml_lib=None) -> MemEstimate:
    """Estimate SIESTA peak memory from a ``.fdf`` and an MPI rank count.

    Returns a :class:`MemEstimate` whose ``request_gb`` is the value to
    pass to sbatch ``--mem`` (whole GB, after the safety multiplier,
    flat add-on, floor and optional node-RAM cap).
    """
    model = model or MemModel()
    inputs = parse_fdf_mem_inputs(fdf_path)
    n_orb, w1 = estimate_norb(inputs, psml_lib)

    # Dense matrices: real (8 B) for Gamma-only, complex (16 B) when a
    # k-mesh forces complex arithmetic.  The diagonaliser holds ~one set
    # of N_orb x N_orb matrices PER k-point, so the term scales with the
    # k-point count (c_dense ~= 4 such matrices: H, S, eigvecs, work).
    bpc = 16.0 if inputs.n_kpoints > 1 else 8.0
    dense_gb = model.c_dense * (n_orb ** 2) * bpc * inputs.n_kpoints / 1e9

    # Real-space mesh.  Grid points per cell edge ~
    # ceil(sqrt(MeshCutoff[Ry]) * L[Bohr] / pi).  Calibrated against real
    # SIESTA output: a 58.406 A (=110.4 bohr) edge at MeshCutoff 400 Ry
    # (SIESTA bumped to 420) produced 720 points -> sqrt(420)*110.4/pi =
    # 720.  Only the cell volume is carried here, so use the isotropic
    # cube-root edge for all three dimensions.
    edge_ang = max(1.0, inputs.cell_volume_ang3) ** (1.0 / 3.0)
    edge_bohr = edge_ang / _BOHR_ANG
    if inputs.mesh_cutoff_ry > 0:
        per_dim = max(1, math.ceil(
            math.sqrt(inputs.mesh_cutoff_ry) * edge_bohr / math.pi))
    else:
        per_dim = 1
    mesh_points = per_dim ** 3
    mesh_mpts = mesh_points / 1e6
    mesh_gb = model.c_mesh * mesh_mpts

    # Per-rank replication.
    repl_gb = model.c_rank * ntasks

    raw = model.safety * (model.base_gb + dense_gb + mesh_gb + repl_gb)

    request = math.ceil(raw) + math.ceil(model.extra_gb)
    request = max(request, math.ceil(model.floor_gb))
    capped = False
    if model.node_mem_gb and request > model.node_mem_gb:
        request = int(model.node_mem_gb)
        capped = True

    return MemEstimate(
        request_gb=int(request),
        dense_gb=round(dense_gb, 1),
        mesh_gb=round(mesh_gb, 1),
        repl_gb=round(repl_gb, 1),
        base_gb=round(model.base_gb, 1),
        n_orb=int(n_orb),
        mesh_mpts=round(mesh_mpts, 1),
        capped=capped,
        warnings=w1 + inputs.parse_warnings,
        safety=model.safety,
        extra_gb=model.extra_gb,
    )


__all__ = [
    "FdfMemInputs",
    "parse_fdf_mem_inputs",
    "estimate_norb",
    "MemModel",
    "MemEstimate",
    "estimate_siesta_memory",
]
