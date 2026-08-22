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

``emit_density_fit_kw`` is the auxbasis door: § 7a's note that
``auxbasis`` rides ``density_fit``'s call as its argument — the deck's
density-fit sites all unpack the ONE ``_MB_DF_KW`` it defines, so the
auxiliary basis cannot be honored at one site and dropped at another.
(A ``density_fit_line`` helper was added beside it 2026-08-21 and died
the same day with zero callers.)
"""
from __future__ import annotations

from typing import List


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


def emit_dft_configure_fn(cfg, *, verbose: bool = True) -> List[str]:
    """Lines defining ``_mb_configure_dft(mf)`` for this ``cfg`` -- the
    SCF dresser's symmetric sibling (M1.2, 2026-08-21).

    The DFT trio (functional / grid level / dispersion) had two
    spellings: the optimization deck's through ``DFT_SECTION`` +
    ``layout.line``, and hand-written constants inside the vibration
    deck's constructions -- the same drift class § 7a closed for SCF,
    one section over.  This body is generated from the SAME section and
    the SAME ``line``, minus ``density_fit``: the fitting call REBINDS
    ``mf`` and is per-site conditional (the Raman polarizability path
    forces non-DF), so it stays at the construction sites.

    Not emitted on a wavefunction-method deck (RHF/UHF): both call
    sites branch on the method before calling, so a pass-through body
    would be text nothing runs.  Ordering note, MEASURED 2026-08-22
    (molbuilder-pySCF, pyscf 2.13): ``density_fit()`` carries ``disp``
    across the rebind in both directions, so dressing before the
    fitting call is safe.
    """
    from .. import script_emit as _sc
    from . import layout as _layout

    is_dft = str(getattr(cfg, "method", "")).upper() in ("RKS", "UKS")
    spell = _layout.line(cfg, is_dft=is_dft)

    out: List[str] = [
        "",
        "",
        "def _mb_configure_dft(mf):",
        '    """The one DFT dresser (molbuilder: engines/pyscf.md § 7a).',
        "",
        "    The functional / grid / dispersion trio, one spelling for",
        "    every mf this deck constructs.  density_fit is deliberately",
        '    NOT here -- it rebinds mf and is per-site conditional."""',
    ]
    if not is_dft:
        return []          # both call sites branch on the method; an
                           # emitted pass-through would be dead text
    emitted = 0
    for name in _layout.DFT_SECTION.items:
        if name == "density_fit":
            continue
        p = _sc.parameter(name, "pyscf", config=cfg)
        ln = spell(p)
        if ln is None:
            continue
        if verbose:
            out.extend("    " + n for n in p.note())
        out.append("    " + ln)
        emitted += 1
    if not emitted:
        out.append("    pass  # every DFT knob at its engine default")
    out.append("    return mf")
    return out


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


#: Dielectric constants for the PCM continuum -- MOVED here 2026-08-21
#: from input.py's module table so both decks read one home (input.py
#: aliases it).  Values are the standard PCM defaults.
SOLVENTS = {
    "water":      78.3553,
    "methanol":   32.613,
    "ethanol":    24.852,
    "acetone":    20.493,
    "dmso":       46.826,
    "thf":        7.4257,
    "chloroform": 4.7113,
    "toluene":    2.3741,
    "hexane":     1.8819,
}


def emit_solvent_lines(cfg) -> List[str]:
    """The PCM decoration, ONE spelling for both decks.

    Category 2 of the integration plan (2026-08-21, PROBED live against
    pyscf 2.13): PCM supports the full derivative chain a vibration
    needs -- analytic gradient, analytic Hessian (RKS and UKS, with and
    without density fitting), and the CPHF polarizability WITH the
    solvent in the response (measured: the tensor shifts, it is not a
    gas-phase number under a solvent label).  SMD is compiled out of
    this build (-DENABLE_SMD) and ddCOSMO has no analytic Hessian --
    both refuse upstream in the kind validator with the reason and the
    references; only PCM reaches this emitter for a vibration deck.
    Empty ``solvent`` -> no lines (gas phase).
    """
    solvent = str(getattr(cfg, "solvent", "") or "").lower()
    if not solvent:
        return []
    eps = SOLVENTS.get(solvent)
    if eps is None:
        raise ValueError(
            f"unknown solvent {solvent!r}; valid: {sorted(SOLVENTS)}")
    method = str(getattr(cfg, "solvent_method", "IEF-PCM") or "IEF-PCM")
    return [
        "# PCM solvation -- continuum model (cheaper than ddCOSMO).",
        f"mf = mf.PCM()",
        f'mf.with_solvent.method = "{method}"',
        f"mf.with_solvent.eps = {eps}    # {solvent} dielectric",
    ]


def emit_solvent_apply_fn(cfg) -> List[str]:
    """``_mb_apply_solvent(mf)`` for a multi-site deck -- the same
    lines :func:`emit_solvent_lines` writes inline for a one-``mf``
    deck, wrapped once so the vibration deck's equilibrium, displaced
    and relaxation constructions decorate identically.  CONSISTENCY IS
    THE SCIENCE here: mixing gas-phase derivatives with a solvated
    energy is not an approximation, it is an inconsistency -- every
    ``mf`` passes through this function or none does.  Gas phase ->
    an identity function, so call sites never branch."""
    lines = emit_solvent_lines(cfg)
    out = [
        "",
        "",
        "def _mb_apply_solvent(mf):",
        '    """Solvent decoration, identical at EVERY construction',
        '    (equilibrium / displaced / relaxation) -- pyscf.md § 7a."""',
    ]
    if lines:
        out += ["    " + ln for ln in lines]
    else:
        out.append("    pass  # gas phase")
    out.append("    return mf")
    return out
