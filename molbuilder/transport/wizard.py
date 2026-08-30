"""Electrode wizard — derive a bulk-lead ``.fdf`` from a labeled device.

What this solves (transiesta-workflow.md § 6.2)
================================================

The #1 TranSIESTA footgun is that the device run and the separate
electrode run must share **one geometry + one numerical contract**, yet
the user assembles the electrode by hand and the couplings silently
drift (invariant set § 6.7).  The wizard removes the hand-assembly: it
extracts the ``*-electrode`` region's *exact* atoms from the device and
emits the matching bulk-lead ``.fdf``, so the cross-run invariants hold
**by construction**, not by the user remembering to copy values.

Guaranteed-by-construction invariants
-------------------------------------

* **I2 (pseudos) / I10 (geometry):** the electrode atoms are the device
  electrode-region coordinates + species, copied verbatim.
* **I6 (lateral cell):** the electrode ``a, b`` are the *device's* lateral
  cell (``_compute_cell_from_extents`` on the device), so the lead tiles
  the device cross-section exactly.
* **I7 (transverse k):** the electrode ``(kx, ky)`` are the device's
  ``k_mesh_transverse``.
* **I1/I3/I4/I5 (XC/MeshCutoff/EnergyShift/basis):** emitted from the
  *same* ``_emit_basis_and_xc(cfg)`` the device uses.
* **I9 (electrode kz dense) / I13 (writes ``.TSHS``):** set here.

What the USER must still verify (warned, not guaranteed)
-------------------------------------------------------

* **The bulk z-period.**  A finite slab does not tell us the lead's true
  periodic repeat unambiguously.  We *derive* it as
  ``z_period = z_span + d_interlayer`` (median interlayer spacing) so the
  slab tiles seamlessly under uniform spacing, and **warn** that the user
  must confirm it matches the real bulk lattice (e.g. the layer count is a
  whole stacking period — a multiple of 3 for FCC(111) ABC).  ``--z-period``
  overrides it.
* **Thickness vs the principal layer (I11):** reported; the consistency
  preflight (§ 6.3) gates on it.

This is the geometry+contract derivation only; it does NOT run SIESTA.
The emitted electrode ``.fdf`` is a regular single-point bulk SCF that
writes ``<label>.TSHS`` for the device's ``TS.Elec.<name>`` reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..cell import LAYER_TOL_ANG, bulk_z_period, detect_layers
from ..config.transport import TransportConfig
from ..structure import Structure
from .transiesta import (
    _compute_cell_from_extents,
    _emit_basis_and_xc,
    _find_electrode_regions,
)

# A bulk-lead kz default.  A thin lead cell has a large 1-D Brillouin
# zone along transport, so it needs dense sampling (§ 4.2); 40 is a
# safe starting point for a metal lead and is meant to be converged by
# the convergence sweep (§ 6.5), not trusted blindly.
DEFAULT_ELECTRODE_KZ = 40

#: A heuristic advisory floor (Å) for the wizard's note below --
#: nothing refuses on it.  The REAL principal-layer gate lives in
#: `transport/compose.py` and compares the orbital interaction range
#: (read from the citation's own ``.ion`` files, never guessed)
#: against the lead's period (transport-design.md § 3).
MIN_ELECTRODE_THICKNESS_ANG = 12.0


@dataclass
class ElectrodeModel:
    """The derived bulk-lead geometry + the invariants it pins.

    ``positions`` are shifted so the lowest layer sits at z = 0; the
    lateral cell is the *device's* (a, b); ``z_period`` is the proposed
    bulk repeat along the transport axis.
    """

    label: str                       # device region label, e.g. "L-electrode"
    block_name: str                  # sanitized TS.Elec name, e.g. "L"
    elements: List[str]
    positions: np.ndarray            # (M, 3) Å, shifted to min-z = 0
    lat_a: np.ndarray                # device lateral vector a (3,), Å
    lat_b: np.ndarray                # device lateral vector b (3,), Å
    cell_a: float                    # |lat_a| (Å), for display/checks
    cell_b: float                    # |lat_b| (Å)
    z_period: float                  # proposed bulk repeat (Å)
    z_span: float                    # top-layer − bottom-layer z (Å)
    n_layers: int
    d_interlayer: float              # median interlayer spacing (Å)
    n_atoms: int
    notes: List[str] = field(default_factory=list)


# --------------------------------------------------------------------- #
#  Geometry derivation                                                  #
# --------------------------------------------------------------------- #


def extract_electrode_model(
    device: Structure,
    label: str,
    *,
    z_period: Optional[float] = None,
    layer_tol_ang: float = LAYER_TOL_ANG,
    min_thickness_ang: float = MIN_ELECTRODE_THICKNESS_ANG,
) -> ElectrodeModel:
    """Build an :class:`ElectrodeModel` for one ``*-electrode`` region.

    The lateral cell is the **device's** (a, b) — so the lead tiles the
    device cross-section (I6).  The z-period is derived from the layer
    spacing unless ``z_period`` is given.
    """
    electrodes = {lab: (name, idxs)
                  for lab, name, idxs in _find_electrode_regions(device)}
    if label not in electrodes:
        raise ValueError(
            f"region {label!r} is not an electrode region; found "
            f"{sorted(electrodes) or 'none'} "
            f"(labels must end with the *-electrode convention)")
    block_name, idxs = electrodes[label]
    idxs = sorted(idxs)

    pos = np.asarray(device.positions, dtype=float)[idxs]
    elems = [device.elements[i] for i in idxs]

    # Lateral cell from the DEVICE (not the electrode's own extent) so
    # the lead tiles the device cross-section exactly (I6).  Prefer the
    # device's REAL lattice vectors (preserves a hexagonal Au(111)
    # surface); fall back to an orthorhombic extent box only when the
    # device carries no cell.
    if device.cell is not None:
        dc = np.asarray(device.cell, dtype=float)
        lat_a, lat_b = dc[0].copy(), dc[1].copy()
    else:
        a, b, _c = _compute_cell_from_extents(device)
        lat_a, lat_b = np.array([a, 0.0, 0.0]), np.array([0.0, b, 0.0])

    # Shift so the lowest layer is at z = 0 (clean periodic cell).
    z = pos[:, 2]
    layer_z = detect_layers(z, layer_tol_ang)
    pos = pos.copy()
    pos[:, 2] = pos[:, 2] - float(min(layer_z)) if layer_z else pos[:, 2]
    layer_z = [lz - float(min(layer_z)) for lz in layer_z] if layer_z else []

    notes: List[str] = []
    if z_period is not None:
        zper = float(z_period)
        # Still report the detected layer structure for the thickness check --
        # measured through the SAME derivation, with only the period overridden.
        # This used to recompute the median inline, which is a second copy of
        # the rule cell.bulk_z_period owns (science/junction-cell.md § 5).
        if len(layer_z) >= 2:
            _derived, d_inter, n_layers = bulk_z_period(layer_z)
        else:
            d_inter, n_layers = float("nan"), len(layer_z)
        z_span = float(layer_z[-1] - layer_z[0]) if layer_z else 0.0
        notes.append(
            f"z-period set explicitly to {zper:.3f} Å (overriding the "
            f"layer-spacing estimate).")
    else:
        zper, d_inter, n_layers = bulk_z_period(layer_z)
        z_span = float(layer_z[-1] - layer_z[0])
        notes.append(
            f"z-period DERIVED as z_span ({z_span:.3f}) + median interlayer "
            f"({d_inter:.3f}) = {zper:.3f} Å.  VERIFY this matches the lead's "
            f"true bulk repeat (e.g. {n_layers} layers is a whole stacking "
            f"period — a multiple of 3 for FCC(111) ABC); override with "
            f"--z-period if not.")

    if z_span < min_thickness_ang:
        notes.append(
            f"electrode z-span {z_span:.2f} Å < ~{min_thickness_ang:.0f} Å: "
            f"may be thinner than the electronic principal layer (§ 4.1).  "
            f"The compose gate verifies the real condition against the "
            f"basis's orbital ranges when .ion files sit beside the "
            f"citation; consider more lead layers if it refuses.")

    return ElectrodeModel(
        label=label, block_name=block_name, elements=elems, positions=pos,
        lat_a=lat_a, lat_b=lat_b,
        cell_a=float(np.linalg.norm(lat_a)), cell_b=float(np.linalg.norm(lat_b)),
        z_period=zper, z_span=z_span, n_layers=n_layers,
        d_interlayer=d_inter, n_atoms=len(elems), notes=notes)


# --------------------------------------------------------------------- #
#  .fdf emission                                                        #
# --------------------------------------------------------------------- #


def render_electrode_fdf(
    model: ElectrodeModel,
    cfg: TransportConfig,
    *,
    job_name: str,
    electrode_kz: int = DEFAULT_ELECTRODE_KZ,
) -> str:
    """Emit the bulk-lead ``.fdf`` that writes ``<job_name>.TSHS``.

    The numerical contract (basis/XC/MeshCutoff/EnergyShift) and the
    transverse k come from ``cfg`` — the *same* object the device fdf
    uses — so I1/I3/I4/I5/I7 hold by construction.  This is a regular
    single-point bulk SCF (no MD block); only ``kz`` and ``TS.HS.Save``
    distinguish it from an ordinary SIESTA point.
    """
    from ase.data import atomic_numbers as _Z

    species = sorted(set(model.elements), key=lambda e: e.capitalize())
    species_idx = {sp: i + 1 for i, sp in enumerate(species)}
    kx, ky, _kz = cfg.k_mesh_transverse

    L: List[str] = [
        "# ================================================================== #",
        f"#  TranSIESTA ELECTRODE (bulk lead) .fdf — {job_name}",
        "#  Generated by molbuilder's electrode wizard from the device's",
        f"#  '{model.label}' region.  Writes {job_name}.TSHS for the device's",
        "#  TS.Elec reference.  See docs/engines/transport.md",
        "#",
        "#  Guaranteed-by-construction vs the device: same atoms (clone),",
        "#  same lateral cell, same transverse k, same XC/MeshCutoff/",
        "#  EnergyShift/basis.  VERIFY the z-period below (bulk repeat).",
        "# ================================================================== #",
        "",
        f"SystemLabel            {job_name}",
        f"SystemName             Bulk electrode ({model.label}) for transport",
        "",
        "# --- Geometry (cloned from the device electrode region) ---",
        "",
        f"NumberOfAtoms          {model.n_atoms}",
        f"NumberOfSpecies        {len(species)}",
        "",
        "%block ChemicalSpeciesLabel",
    ]
    for sp in species:
        L.append(f"  {species_idx[sp]:>3}  {_Z.get(sp.capitalize(), 0):>3}  {sp}")
    L += [
        "%endblock ChemicalSpeciesLabel",
        "",
        "LatticeConstant        1.0 Ang",
        "# a, b = device lateral vectors VERBATIM (the lead tiles the device",
        "#        cross-section; hexagonal cells are preserved, not squared).",
        "# c    = bulk repeat along the transport axis.  VERIFY this matches",
        f"#        the lead's true periodicity ({model.n_layers} layers, "
        f"interlayer {model.d_interlayer:.3f} Å).",
        "%block LatticeVectors",
        f"  {model.lat_a[0]:14.8f} {model.lat_a[1]:14.8f} {model.lat_a[2]:14.8f}",
        f"  {model.lat_b[0]:14.8f} {model.lat_b[1]:14.8f} {model.lat_b[2]:14.8f}",
        f"  {0.0:14.8f} {0.0:14.8f} {model.z_period:14.8f}",
        "%endblock LatticeVectors",
        "",
        "AtomicCoordinatesFormat        Ang",
        "%block AtomicCoordinatesAndAtomicSpecies",
    ]
    for el, (x, y, z) in zip(model.elements, model.positions):
        L.append(f"  {x:14.8f} {y:14.8f} {z:14.8f}  {species_idx[el]}")
    L += [
        "%endblock AtomicCoordinatesAndAtomicSpecies",
        "",
    ]
    # Same numerical contract as the device (I1/I3/I4/I5).
    L += _emit_basis_and_xc(cfg)
    # Transverse k from the device (I7); kz dense (I9).
    L += [
        "# --- k-mesh: transverse matches the device; kz DENSE (bulk lead) ---",
        "# This is a periodic bulk run, so kz must be converged (§ 4.2) —",
        "# unlike the device, where kz = 1 (open boundary).",
        "%block kgrid_Monkhorst_Pack",
        f"  {kx:>3}    0    0      0.0",
        f"    0  {ky:>3}    0      0.0",
        f"    0    0  {int(electrode_kz):>3}      0.0",
        "%endblock kgrid_Monkhorst_Pack",
        "",
        "# --- Write the .TSHS for the device's TS.Elec reference ---",
        "# Single-point bulk SCF (no MD block = no relaxation; the lead is",
        "# the frozen bulk geometry).  TS.HS.Save writes <SystemLabel>.TSHS.",
        "# Confirm the keyword against your SIESTA version (manual § TranSIESTA).",
        "SolutionMethod         diagon",
        "TS.HS.Save             true",
        "SaveHS                 true",
        "",
    ]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------- #
#  Top-level driver                                                     #
# --------------------------------------------------------------------- #


def electrode_wizard(
    device: Structure,
    cfg: TransportConfig,
    *,
    which: str = "both",
    electrode_kz: int = DEFAULT_ELECTRODE_KZ,
    z_period: Optional[float] = None,
    layer_tol_ang: float = LAYER_TOL_ANG,
    min_thickness_ang: float = MIN_ELECTRODE_THICKNESS_ANG,
) -> List[Tuple[str, str, ElectrodeModel]]:
    """Derive electrode ``.fdf``(s) from a labeled device.

    ``which`` selects ``"both"`` (every ``*-electrode`` region), or a
    specific region label / block name (``"L-electrode"`` or ``"L"``).
    Returns a list of ``(job_name, fdf_text, model)`` — the caller
    writes the files (keeps this function pure / testable).
    """
    found = _find_electrode_regions(device)
    if not found:
        raise ValueError(
            "no *-electrode regions on the device; label the lead atoms "
            "(region picker / region-labels.md) before deriving electrodes")

    if which == "both":
        targets = [lab for lab, _name, _idxs in found]
    else:
        targets = [lab for lab, name, _idxs in found
                   if which in (lab, name)]
        if not targets:
            raise ValueError(
                f"--which {which!r} matched no electrode; available: "
                f"{[lab for lab, _n, _i in found]}")

    out: List[Tuple[str, str, ElectrodeModel]] = []
    for label in targets:
        model = extract_electrode_model(
            device, label, z_period=z_period, layer_tol_ang=layer_tol_ang,
            min_thickness_ang=min_thickness_ang)
        # File name matches the device emitter's TS.Elec HS reference:
        #   <device job>_<region label>.TSHS  (transiesta.py _emit_*).
        job_name = f"{cfg.job_name}_{label}"
        fdf = render_electrode_fdf(
            model, cfg, job_name=job_name, electrode_kz=electrode_kz)
        out.append((job_name, fdf, model))
    return out


def format_models(models: List[Tuple[str, str, ElectrodeModel]]) -> str:
    """Human-readable summary of what the wizard derived."""
    lines = ["electrode wizard — derived bulk leads", ""]
    for job_name, _fdf, m in models:
        lines.append(
            f"  {m.label:>14}  ->  {job_name}.fdf  "
            f"({m.n_atoms} atoms, {m.n_layers} layers, z-span "
            f"{m.z_span:.2f} Å, z-period {m.z_period:.3f} Å, "
            f"kz dense)")
        for note in m.notes:
            lines.append(f"        - {note}")
    return "\n".join(lines)


__all__ = [
    "ElectrodeModel",
    "extract_electrode_model",
    "render_electrode_fdf",
    "electrode_wizard",
    "format_models",
    "DEFAULT_ELECTRODE_KZ",
]
