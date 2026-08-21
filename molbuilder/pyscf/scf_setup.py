"""The SCF dresser's generator — one home for `_mb_configure_scf(mf)`.

THE CONTRACT IS `docs/engines/pyscf.md` § 7a, written before this module
(the user's process ruling, 2026-08-21): the framework never spells an
SCF knob twice.  A deck that constructs MANY ``mf`` objects — the
vibration deck builds an equilibrium one, one per displaced point, and a
relaxation one — emits the function this module generates ONCE and calls
it at every construction; a deck with one ``mf`` (optimization) applies
the same section inline through the Sections machinery.  Either way the
membership is ``layout.SCF_SECTION`` and the spelling is ``layout.line``
— this module adds NO per-knob code, it only walks the two tables.

``density_fit_line`` is the second export: § 7a's note that ``auxbasis``
rides ``density_fit``'s spelling as its argument — a caller that builds
an ``mf`` with density fitting asks HERE for the line, so the auxiliary
basis cannot be honored in one deck and dropped in another.
"""
from __future__ import annotations

from typing import List, Optional


def emit_scf_configure_fn(cfg, *, verbose: bool = True) -> List[str]:
    """Lines defining ``_mb_configure_scf(mf)`` for this ``cfg``.

    The body is generated from ``SCF_SECTION`` + ``line`` at render
    time, so it is exactly the set and spelling the optimization deck's
    inline section would produce — with each knob's catalogue note above
    its line (the note-with-the-value rule reaches emitted functions the
    same way it reaches sections).
    """
    from .. import script_emit as _sc
    from . import layout as _layout

    is_dft = str(getattr(cfg, "method", "")).upper() in ("RKS", "UKS")
    spell = _layout.line(cfg, is_dft=is_dft)

    out: List[str] = [
        "",
        "",
        "def _mb_configure_scf(mf):",
        '    """The one SCF dresser (molbuilder: engines/pyscf.md § 7a).',
        "",
        "    Every mf this deck constructs passes through here, so the",
        "    SCF machinery below applies identically to the equilibrium,",
        "    displaced-point and relaxation cycles.  Site-specific extras",
        "    (checkpointing, GPU promotion, the Newton wrap) stay at the",
        '    construction sites -- see the contract\'s role table."""',
    ]
    emitted = 0
    for name in _layout.SCF_SECTION.items:
        p = _sc.parameter(name, "pyscf", config=cfg)
        ln = spell(p)
        if ln is None:
            continue
        if verbose:
            out.extend("    " + n for n in p.note())
        out.append("    " + ln)
        emitted += 1
    if not emitted:
        out.append("    pass  # every SCF knob at its engine default")
    out.append("    return mf")
    return out


def density_fit_line(cfg) -> Optional[str]:
    """``density_fit``'s one spelling (auxbasis rides it as an argument),
    for callers building an ``mf`` outside the Sections machinery."""
    from .. import script_emit as _sc
    from . import layout as _layout
    is_dft = str(getattr(cfg, "method", "")).upper() in ("RKS", "UKS")
    return _layout.line(cfg, is_dft=is_dft)(
        _sc.parameter("density_fit", "pyscf", config=cfg))


def emit_density_fit_kw(cfg) -> List[str]:
    """Lines defining ``_MB_DF_KW`` — the one home for density_fit's
    keyword arguments in a multi-site deck (pyscf.md § 7a's auxbasis
    ride).  Both of the vibration deck's ``density_fit`` calls unpack
    it, so the auxiliary basis cannot be honored at one site and
    dropped at the other."""
    aux = getattr(cfg, "auxbasis", None)
    return [
        "# density_fit's keyword arguments, ONE home (pyscf.md § 7a):",
        "# auxbasis rides the density-fitting call as its argument.",
        f"_MB_DF_KW = {{'auxbasis': {aux!r}}} if {aux!r} else {{}}",
    ]
