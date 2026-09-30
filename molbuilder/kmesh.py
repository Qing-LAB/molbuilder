"""The k-point mesh a rung samples -- ONE home (`engines/siesta.md` § 6.1).

Every deck molbuilder writes for SIESTA or ``tbtrans`` samples the Brillouin
zone on a Monkhorst-Pack mesh: a count and an offset per axis of the cell.
What that mesh IS on a rung follows from two facts and the template's values:
each axis's kind, as the structure states it (``Structure.axis_kind``), and
what the rung does along the calculation's transport axis.  This module turns
them into a :class:`KMesh` once per deck; the deck's writer, the settings
gate, the derived settings (``Diag.ParallelOverK``) and the record read that
one object, and none of them works the mesh out again.

**Why a module and not a helper in each writer.**  Until 2026-09-30 nothing
derived a mesh from the axes: four writers each built their own block from
``cfg.kgrid``, the transport axis was forced to 1 in six places, one fact
carried two severities (a warning in the SIESTA validator, an error in the
transport kind's), and a lead's ``Diag.ParallelOverK`` was decided from the
template's ``kx ky 1`` while its deck wrote ``kx ky 40``.

It imports nothing from the catalogue -- :mod:`molbuilder.template` asks
:func:`fixed` for its one per-value door (``template.why_not``), so this module
must not import it back; :func:`check` reaches :mod:`molbuilder.issues`, and
numpy for the cell's lengths, when it runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: The transport axis of a transport calculation: the cell's third vector,
#: ``c`` (A3).  The composition states it (`transport/compose.py` builds every
#: junction's kinds as ``(across, across, "transport")``), and TranSIESTA's
#: electrodes are semi-infinite along it.  The composition and the mesh read
#: it here; `transport/sort.py` keeps a copy of its own and the transport
#: writers index ``c`` directly (owed, plan § 5w K3).
TRANSPORT_AXIS = 2

#: The components' names -- the form's triple labels, and how a refusal names
#: a component.
AXES = ("x", "y", "z")

#: The transport rungs whose transport axis is the LEAD's own periodic
#: direction, sampled by ``electrode_kz``.  Every other transport rung -- the
#: seed, the device, the transmission -- is open along it.  Keyed by the
#: rung's shape (`transport/deck.py`'s ``SHAPE_OF_RUNG``).
LEAD_RUNGS = frozenset({"electrode"})

#: Why the transport axis takes one point with no offset on a transport
#: calculation -- the reason every door gives for a fixed component.
WHY_OPEN = ("the transport axis is the open boundary on the seed, the device "
            "and the transmission, sampled at one point, and a lead samples "
            "it by electrode_kz -- no rung reads this component "
            "(engines/siesta.md 6.1)")
WHY_NO_OFFSET = ("both engines sample the transport axis with no offset -- "
                 "TranSIESTA in ts_kpoint_scf.F90, tbtrans in m_tbt_kpoint.F90 "
                 "-- and a lead is compared with the device on it "
                 "(engines/siesta.md 6.1)")

#: The items that answer a mesh's counts -- an axis's ``source`` is one of
#: them, or ``open`` when the axis itself answers.
ITEMS = ("kgrid", "tbt_k_grid", "electrode_kz")

#: What a transport calculation fixes in each item a mesh reads.
_FIXED_ON_TRANSPORT: Dict[str, Tuple[Any, str]] = {
    "kgrid": (1, WHY_OPEN),
    "tbt_k_grid": (1, WHY_OPEN),
    "kgrid_displacement": (0.0, WHY_NO_OFFSET),
}

#: A sampled axis whose periodic images sit at least this far apart earns a
#: hint (`science/validation.md` § 4.1, user 2026-08-20).
VACUUM_HINT_A = 5.0


@dataclass(frozen=True)
class KAxis:
    """One axis of a rung's mesh."""
    #: The axis's kind as the structure states it: ``periodic``,
    #: ``isolated`` or ``transport``.
    kind: str
    #: What this rung does along it: ``sampled`` (the template's count),
    #: ``gamma`` (an isolated axis -- the template's count, 1 the only one
    #: that samples anything but images of vacuum), ``open`` (a transport
    #: calculation's transport axis on the seed, the device and the
    #: transmission: one point, no offset) or ``lead`` (the same axis on an
    #: electrode rung: ``electrode_kz`` points, no offset).
    role: str
    count: int
    #: The grid's offset along the axis, in units of one mesh spacing.
    shift: float
    #: The item that answered the count: ``kgrid``, ``tbt_k_grid``,
    #: ``electrode_kz`` -- or ``open``, when the axis itself does.
    source: str


@dataclass(frozen=True)
class KMesh:
    """The mesh one deck writes: three axes and the program that reads it."""
    axes: Tuple[KAxis, KAxis, KAxis]
    #: ``siesta`` writes ``%block kgrid_Monkhorst_Pack``; ``tbtrans``
    #: ``%block TBT.k``.
    program: str = "siesta"

    @property
    def counts(self) -> Tuple[int, int, int]:
        return tuple(a.count for a in self.axes)

    @property
    def shifts(self) -> Tuple[float, float, float]:
        return tuple(a.shift for a in self.axes)

    @property
    def n_points(self) -> int:
        """The grid's points before time reversal folds k onto -k -- the
        determinant of the diagonal grid (SIESTA's ``kgrid.F``)."""
        return prod(self.counts)

    @property
    def single_point(self) -> bool:
        """Every count 1: one k-point, whatever its offset."""
        return all(c == 1 for c in self.counts)


def _triple(value, cast) -> Optional[Tuple[Any, Any, Any]]:
    """``value`` as three ``cast`` numbers, or ``None`` when it is not one --
    a malformed value is the declared type's refusal, never this module's."""
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(isinstance(v, bool) for v in value)):
        return None
    try:
        return tuple(cast(v) for v in value)
    except (TypeError, ValueError):
        return None


def mesh_for(cfg, axis_kind: Optional[Sequence[str]], *,
             kind: str = "optimization", rung: Optional[str] = None,
             program: str = "siesta") -> Optional[KMesh]:
    """The mesh this rung writes -- the ONE derivation.

    ``axis_kind`` is the structure's (``Structure.axis_kind``); ``kind`` the
    described calculation; ``rung`` a transport rung's shape (``seed`` ·
    ``electrode`` · ``device`` · ``transmission``), and ``None`` on any other
    kind; ``program`` whose keywords write it -- ``tbtrans`` reads
    ``tbt_k_grid`` across the transport axis, every SIESTA run ``kgrid``.

    ``None`` when a value it needs is not three numbers: that is the declared
    type's refusal, which the settings gate gives, and a crash here would
    pre-empt it.
    """
    kinds = tuple(axis_kind) if axis_kind else ("isolated",) * 3
    source = "tbt_k_grid" if program == "tbtrans" else "kgrid"
    counts = _triple(getattr(cfg, source, None), int)
    shifts = _triple(getattr(cfg, "kgrid_displacement", (0.0, 0.0, 0.0)),
                     float)
    if counts is None or shifts is None or len(kinds) != 3:
        return None
    axes = []
    for i in range(3):
        if kind == "transport" and i == TRANSPORT_AXIS:
            if rung in LEAD_RUNGS:
                try:
                    kz = int(getattr(cfg, "electrode_kz"))
                except (AttributeError, TypeError, ValueError):
                    return None
                axes.append(KAxis(kinds[i], "lead", kz, 0.0, "electrode_kz"))
            else:
                axes.append(KAxis(kinds[i], "open", 1, 0.0, "open"))
            continue
        # A `transport` axis in any other calculation -- relaxing a junction
        # -- is periodic in the deck, so it is sampled like one (user,
        # 2026-09-30); only an isolated axis is Gamma's.
        role = "gamma" if kinds[i] == "isolated" else "sampled"
        axes.append(KAxis(kinds[i], role, counts[i], shifts[i], source))
    return KMesh(tuple(axes), program)


def fixed(item: str, kind: str) -> Dict[int, Tuple[Any, str]]:
    """``{component: (value, why)}`` -- the components of ``item`` that no
    rung of a ``kind`` calculation reads as a choice, and what they hold.

    Structure-free, so the doors that meet a value before any structure is
    at hand ask it: the description's own check, ``resolve`` and the form
    (through ``template.why_not`` and the schema).  On a transport
    calculation the transport axis's count is 1 and its offset 0 on every
    open rung, and a lead's count is ``electrode_kz`` -- so the third
    component of ``kgrid``, ``tbt_k_grid`` and ``kgrid_displacement`` is
    nobody's."""
    if kind != "transport" or item not in _FIXED_ON_TRANSPORT:
        return {}
    return {TRANSPORT_AXIS: _FIXED_ON_TRANSPORT[item]}


def with_fixed(item: str, value, kind: str):
    """``value`` with the components ``kind`` fixes laid on -- how a value
    born outside the template (a cited run's grid) enters it already
    answering the rule.  A value that is not a triple is returned as given."""
    held = fixed(item, kind)
    if not held or not isinstance(value, (list, tuple)) or len(value) != 3:
        return value
    out = list(value)
    for i, (v, _why) in held.items():
        out[i] = v
    return tuple(out)


def write(mesh: KMesh) -> List[str]:
    """The mesh's text -- the ONLY one.  SIESTA's ``%block
    kgrid_Monkhorst_Pack`` and ``tbtrans``'s ``%block TBT.k``: three rows of
    counts, each with its offset column.  SIESTA reads a row of one to four
    values (``kpoint_t.F90``); ``tbtrans`` reads a block row only when it
    carries the offset (``m_tbt_kpoint.F90``, ``read_kgrid``), so the column
    is written always -- and ``TBT.k``'s list form carries no offset, so the
    block is written always too."""
    name = "TBT.k" if mesh.program == "tbtrans" else "kgrid_Monkhorst_Pack"
    rows = []
    for i, axis in enumerate(mesh.axes):
        cells = [axis.count if j == i else 0 for j in range(3)]
        rows.append("  " + "  ".join(f"{c:>3}" for c in cells)
                    + f"    {float(axis.shift)}")
    return [f"%block {name}", *rows, f"%endblock {name}"]


def _a_stripe(mesh: KMesh, i: int) -> bool:
    """Is ``mesh`` a transport calculation's SCF mesh whose isolated axis
    ``i`` sits beside a periodic transverse axis -- the junction TranSIESTA
    refuses to attach a lead to when ``i`` is sampled more than once?  Its
    check skips a lead isolated on both transverse axes (``is_Gamma``,
    ``ts_electrode.F90``), and ``tbtrans`` does not make it."""
    transport = mesh.axes[TRANSPORT_AXIS].role in ("open", "lead")
    return (transport and mesh.program == "siesta"
            and any(a.role == "sampled" for j, a in enumerate(mesh.axes)
                    if j not in (i, TRANSPORT_AXIS)))


def check(meshes: Sequence[Optional[KMesh]], struct, *, cell=None,
          refused=frozenset()) -> List[Any]:
    """The findings on the meshes ONE deck writes -- one rule each
    (`engines/siesta.md` § 6.1; `science/validation.md` § 4.1).

    Warnings: ``k > 1`` is the person's explicit statement (user,
    2026-08-20), and ``k = 1`` states nothing and is checked not at all --
    save one refusal, where the engine itself stops: a transport
    calculation's SCF mesh sampling an isolated axis more than once while the
    other transverse axis is periodic (a stripe), which TranSIESTA refuses at
    the device after the seed and both leads have run (``ts_electrode.F90``,
    ``check_in_cell``; refused for now, user 2026-09-30).  A
    count at or below its limit and a fixed component are refused by the
    one per-value door (``template.why_not``) on every surface; they are not
    judged here again -- and a mesh built from a value that cannot stand
    (``refused``: the items that door refused) is not judged at all, since a
    value refused draws that refusal alone (`engines/template.md` § 5.3).

    The count rules are per mesh -- each mesh's counts are its own item's.
    The OFFSET is one value every mesh of the deck shares, so its finding is
    said once, naming each mesh that samples the axis at one point: the
    transmission deck's two meshes warned twice about it until the K3
    review.
    """
    from .issues import Issue
    judged = [m for m in meshes if m is not None and not (
        ({a.source for a in m.axes} | {"kgrid_displacement"}) & set(refused))]
    out: List[Any] = []
    lengths = extent = None
    if cell is not None:
        import numpy as np
        box = np.asarray(cell, dtype=float)
        lengths = [float(np.linalg.norm(box[i])) for i in range(3)]
        pos = np.asarray(getattr(struct, "positions", ()), dtype=float)
        extent = ((pos.max(axis=0) - pos.min(axis=0)) if pos.size
                  else np.zeros(3))
    for mesh in judged:
        for i, axis in enumerate(mesh.axes):
            where = f"config.{axis.source}"
            name = f"{axis.source}[{i}]"
            if axis.role == "gamma" and axis.count > 1 and _a_stripe(mesh, i):
                out.append(Issue(
                    "error",
                    f"{name} = {axis.count} on an isolated axis of a junction "
                    f"that is periodic across the other: TranSIESTA stops the "
                    f"device on it -- a lead that repeats along one "
                    f"transverse axis must sample the other, where it has no "
                    f"neighbours, at one point (ts_electrode.F90, \"found "
                    f"incompatible k-grids\"), and the seed and both leads "
                    f"would have run first.  Set it to 1; such a junction is "
                    f"refused for now (engines/siesta.md 6.1)",
                    where))
            elif axis.role == "gamma" and axis.count > 1:
                out.append(Issue(
                    "warn",
                    f"{name} = {axis.count} on an isolated axis: the "
                    f"structure does not repeat there, so the extra points "
                    f"sample images of vacuum -- cost for nothing; 1 is the "
                    f"usual choice (engines/siesta.md 6.1)",
                    where))
            elif (axis.role == "sampled" and axis.count > 1
                  and lengths is not None):
                gap = max(0.0, lengths[i] - float(extent[i]))
                if gap >= VACUUM_HINT_A:
                    out.append(Issue(
                        "warn",
                        f"{name} = {axis.count} samples a supercell along "
                        f"an axis whose periodic images sit ~{gap:.1f} A "
                        f"apart; if the images are meant not to interact, "
                        f"k = 1 is the usual choice -- if a weak image "
                        f"interaction is deliberate, carry on",
                        where))
    for i in range(3):
        # ...read modulo 1, as both engines read it (`kpoint_t.F90`,
        # `m_tbt_kpoint.F90`): an offset of 1.0 is Gamma again.
        single = [m.axes[i] for m in judged
                  if m.axes[i].count == 1 and m.axes[i].shift % 1.0 != 0.0]
        if single:
            counts = ", ".join(dict.fromkeys(
                f"{a.source}[{i}] = 1" for a in single))
            out.append(Issue(
                "warn",
                f"kgrid_displacement[{i}] = {single[0].shift} shifts an "
                f"axis sampled at a SINGLE k-point ({counts}), which moves "
                f"that point off Gamma.  For an "
                f"isolated molecule only Gamma is meaningful -- set this "
                f"component to 0.",
                "config.kgrid_displacement"))
    return out


__all__ = ["AXES", "ITEMS", "KAxis", "KMesh", "LEAD_RUNGS", "TRANSPORT_AXIS",
           "VACUUM_HINT_A", "check", "fixed", "mesh_for", "with_fixed",
           "write"]
