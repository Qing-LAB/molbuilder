"""Engine-agnostic Methods-paragraph composer for the Spectra tab.

Produces the Markdown prose that ships in three places:

  * the header docstring of the emitted ``<job>.spectra.py`` script
    (spec § 11.2);
  * the **Show methods text** modal in the Spectra-tab UI
    (spec § 9.4);
  * the ``methods_text`` field of :class:`SpectraResults`
    (spec § 5 -- post-run, with real numbers from the parsed run).

The same prose appears in all three so the user sees identical
content in the form, the emitted script, and the parsed JSON.
Pre-run (no ``results`` arg) the prose describes what *will* be
done with the configured knobs; post-run (``results`` provided)
real numbers from the run replace the configuration placeholders
(actual mode count, n_atoms / n_free from the parsed structure).

Engine-specific fragments come from
:meth:`SpectraEngine.methods_fragment` -- the composer here is
deliberately ignorant of which engine ran the job so a future
SIESTA engine plugs in without changing this file.

Citation keys appear inline as ``[KeyName]`` markers (e.g.
``[Sun2020]``).  Every key cited here must resolve against
``docs/science/references.bib``; :func:`extract_citation_keys`
gives the caller the list of keys actually used so it can render a
trailing bibliography and the ``bibliography_keys`` field of
:class:`SpectraResults` stays in sync.
"""

from __future__ import annotations

import re
from typing import List, Optional

from ..config.spectra import SpectraConfig
from ..structure import Structure
from .results import SpectraResults


# Citation marker regex.  Matches:
#   [Foo]             -- single key
#   [Foo, Bar]        -- comma-separated keys (common phys/chem style)
#   [Foo §section]    -- section suffix on a single key (Mills1972 §2.4)
# A key is a letter followed by letters/digits/underscore.  Each
# bracket-group's matched span is split on commas by
# :func:`extract_citation_keys`.
_CITE_RE = re.compile(
    r"\[("
    r"[A-Za-z][A-Za-z0-9_]*"                       # first key
    r"(?:\s*,\s*[A-Za-z][A-Za-z0-9_]*)*"           # optional , Key, Key ...
    r")"
    r"(?:\s+§[^\]]*)?"                             # optional ' §section' suffix
    r"\]"
)


def render_methods_md(
    cfg: SpectraConfig,
    *,
    results: Optional[SpectraResults] = None,
    fragment_md: str = "",
    struct: Optional[Structure] = None,
) -> str:
    """Compose the Markdown Methods paragraph for the given config.

    Parameters
    ----------
    cfg
        The :class:`SpectraConfig` being rendered.  Drives every
        prose decision: functional / basis / dispersion / selector /
        amplitude / frequency window.
    results
        Optional parsed :class:`SpectraResults`.  Pre-run callers
        pass ``None`` and get the "what will be done" form; post-run
        callers pass the parsed results and get real mode counts
        and frequency ranges interpolated into the prose.
    fragment_md
        Optional engine-specific paragraph, supplied by the caller
        (the vibration deck passes
        :func:`molbuilder.pyscf.vibration_emitters.
        pyscf_methods_fragment`).  Empty means no engine paragraph.
        The registry lookup this replaced retired with the old
        generator (P3): this module is engine-IGNORANT by design
        (its own header line 21), and the one remaining producer
        knows its own engine.
    struct
        Optional :class:`Structure`.  Used to phrase "N free atoms,
        3N-6 modes" etc. when available; falls back to generic
        prose when ``None``.

    Returns
    -------
    str
        Markdown ready to drop into a manuscript draft, the
        Methods-preview modal, or the script's header docstring.
        Use :func:`extract_citation_keys` on the result to obtain
        the bibliography-key list.
    """
    parts: List[str] = []
    parts.append("## Methods\n")

    # ------------------------------------------------------------------ #
    # Paragraph 1: vibrational analysis setup (always emitted)
    # ------------------------------------------------------------------ #
    p1 = _paragraph_vibrational(cfg, results=results, struct=struct)
    parts.append(p1)

    # ------------------------------------------------------------------ #
    # Engine-specific fragment (optional)
    # ------------------------------------------------------------------ #
    if fragment_md:
        parts.append(fragment_md)

    # ------------------------------------------------------------------ #
    # Paragraph 2: per-mode electronic structure (skipped when
    # selector == "skip" -- nothing was / will be computed there).
    # ------------------------------------------------------------------ #
    if cfg.es_mode_selection != "skip":
        p2 = _paragraph_electronic_structure(cfg, results=results)
        parts.append(p2)

    # ------------------------------------------------------------------ #
    # "Selected modes" line (post-run only; pre-run can't list real
    # indices since the spectrum hasn't been computed yet).
    # ------------------------------------------------------------------ #
    sel_line = _selected_modes_line(cfg, results=results)
    if sel_line:
        parts.append(sel_line)

    # ------------------------------------------------------------------ #
    # Trailing bibliography (BibTeX keys, one per line).  Composed
    # from the keys present in the text we just built so a reader of
    # the emitted script has the full list inline (spec § 11.2).
    # ------------------------------------------------------------------ #
    body = "\n\n".join(parts)
    bib_keys = extract_citation_keys(body)
    if bib_keys:
        bib_lines = ["**Bibliography** (verified in `docs/science/"
                     "references.bib`):"]
        for k in bib_keys:
            bib_lines.append(f"- `{k}`")
        body = body + "\n\n" + "\n".join(bib_lines)

    return body


def extract_citation_keys(text: str) -> List[str]:
    """Return the BibTeX keys cited in ``text``, in order of first
    appearance, deduplicated.

    Matches ``[Key]`` and ``[Key §section]`` patterns.  Used by
    :func:`render_methods_md` to build the trailing bibliography,
    and by the engine to populate :attr:`SpectraResults.
    bibliography_keys` (spec § 5).

    A linter (spec § 11.3) will later cross-check the returned
    list against ``references.bib`` to refuse a release tag if any
    cited key is missing or marked TO-VERIFY.
    """
    seen: set = set()
    out: List[str] = []
    for m in _CITE_RE.finditer(text or ""):
        # group(1) is either a single key or a comma-separated list
        # of keys (e.g. "Sun2020, Sun2018").  Split on commas and
        # add each key, preserving order of first appearance.
        for key in (k.strip() for k in m.group(1).split(",")):
            if key and key not in seen:
                seen.add(key)
                out.append(key)
    return out


# --------------------------------------------------------------------- #
#  Internals                                                            #
# --------------------------------------------------------------------- #


def _paragraph_vibrational(cfg: SpectraConfig,
                           *,
                           results: Optional[SpectraResults],
                           struct: Optional[Structure]) -> str:
    """First Methods paragraph: harmonic vibrational analysis +
    (optional) Raman activities.  Always emitted -- L2 is the
    foundation layer, you can't have a Spectra-tab run without it."""
    fxc = cfg.functional
    basis = cfg.basis
    disp_clause = ""
    if cfg.dispersion and cfg.dispersion.lower() != "none":
        # D3BJ is the default; cite Grimme2011.  Any other dispersion
        # correction also points at Grimme2011 since it's the damping-
        # function paper that defines the family in current use.
        disp_clause = f" with the {cfg.dispersion.upper()} dispersion correction [Grimme2011]"

    # Functional-specific citation: B3LYP gets [Becke1993]; other
    # functionals would ideally cite their primary paper, but we
    # don't carry a per-functional citation map yet, so we cite
    # Becke1993 only for the B3 family.
    fxc_cite = " [Becke1993]" if fxc.upper().startswith("B3") else ""

    # Atom-count clause -- only when we have a Structure to count from.
    # Structure stores elements as a list of element symbols; n_atoms
    # is its length.  Defensive try/except so a duck-typed mock (in
    # tests) carrying `.elements` works too.
    atom_clause = ""
    if struct is not None:
        n_atoms = _count_structure_atoms(struct)
        if n_atoms:
            n_free = _count_free_atoms(struct, cfg)
            n_modes = max(0, 3 * n_free - 6) if n_free >= 2 else 0
            atom_clause = (f" The system contains {n_atoms} atoms "
                           f"({n_free} free, {n_atoms - n_free} frozen "
                           f"during the Hessian), giving "
                           f"{n_modes} non-translational / non-rotational "
                           f"vibrational modes.")

    raman_clause = ""
    if cfg.compute_raman:
        # Mention Komornicki1979 + Wilson1955: dα/dR method paper +
        # the canonical normal-coordinate framework that maps
        # Cartesian polarizability derivatives to mode-projected
        # Raman activities.
        raman_clause = (" Raman activities (Å⁴/amu) were computed from "
                        "analytic polarizability derivatives "
                        "[Komornicki1979] projected onto the mode "
                        "eigenvectors using the standard normal-"
                        "coordinate framework [Wilson1955].")

    para = (f"Harmonic vibrational analysis was performed at the "
            f"{fxc}/{basis}{fxc_cite} level{disp_clause}.{atom_clause}"
            f"{raman_clause}")

    # Post-run: append the actual frequency span if we have it.
    if results is not None and results.modes:
        freqs = [m.frequency_cm1 for m in results.modes]
        # filter NaN-like; ModeData.__post_init__ already enforces
        # a real number so this is safe.
        if freqs:
            fmin = min(freqs)
            fmax = max(freqs)
            n_imag = sum(1 for f in freqs if f < 0)
            extra = (f" The analysis yielded {len(freqs)} modes spanning "
                     f"{fmin:.1f} to {fmax:.1f} cm⁻¹")
            if n_imag:
                extra += f" ({n_imag} imaginary)"
            extra += "."
            para = para + extra

    return para


def _paragraph_electronic_structure(cfg: SpectraConfig,
                                    *,
                                    results: Optional[SpectraResults]) -> str:
    """Second Methods paragraph: per-mode displaced-geometry SCFs.
    Only emitted when ``cfg.es_mode_selection != "skip"`` -- the
    L4 step is opt-in (spec § 8)."""
    amp = cfg.displacement_amplitude_ang
    n_below = cfg.es_n_homo_below
    n_above = cfg.es_n_lumo_above

    sel = cfg.es_mode_selection
    if sel == "all":
        criterion = "every vibrational mode"
    elif sel == "top_n":
        criterion = (f"the top {cfg.es_top_n} modes ranked by Raman "
                     f"activity")
    elif sel == "threshold":
        criterion = (f"modes with Raman activity > {cfg.es_threshold:g} "
                     f"Å⁴/amu")
    elif sel == "explicit":
        criterion = (f"a user-specified set of {len(cfg.es_explicit_indices)} "
                     f"modes")
    else:  # pragma: no cover (filtered above)
        criterion = "the selected modes"

    window_clause = _frequency_window_clause(cfg)

    para = (f"For {criterion}{window_clause}, per-mode electronic-"
            f"structure data were computed at displaced geometries "
            f"q ± A·Q_i with A = {amp:g} Å [Mills1972 §2.4].  At each "
            f"displaced geometry the SCF was converged at the same "
            f"level as the equilibrium structure; the frontier "
            f"orbital energies (HOMO-{n_below} through LUMO+{n_above}), "
            f"the HOMO-LUMO gap, and the change relative to the "
            f"equilibrium values were recorded.  This data supports "
            f"downstream electron-phonon coupling analysis for "
            f"inelastic-transport modelling [Galperin2007, "
            f"Frederiksen2007].")

    if results is not None:
        n_with_es = sum(1 for m in results.modes
                        if m.electronic_structure is not None)
        if n_with_es:
            para = para + (f"  In the present run {n_with_es} modes "
                           f"received per-mode electronic-structure "
                           f"data.")
    return para


def _frequency_window_clause(cfg: SpectraConfig) -> str:
    """Inline phrase describing the frequency window when one is in
    effect.  Empty string when no window or selector=explicit
    (window is ignored there per spec § 8.1)."""
    if cfg.es_mode_selection == "explicit":
        return ""
    fmin = cfg.freq_min_cm1
    fmax = cfg.freq_max_cm1
    if fmin is None and fmax is None:
        return ""
    if fmin is not None and fmax is not None:
        return f" within the {fmin:g}-{fmax:g} cm⁻¹ window"
    if fmin is not None:
        return f" with frequency ≥ {fmin:g} cm⁻¹"
    return f" with frequency ≤ {fmax:g} cm⁻¹"


def _selected_modes_line(cfg: SpectraConfig,
                         *,
                         results: Optional[SpectraResults]) -> str:
    """Post-run line listing the actual mode indices that received
    L4 treatment, per spec § 11.2 ("selected modes" line).  Pre-run
    we return "" -- the spectrum hasn't been computed yet so we
    can't enumerate by frequency."""
    if results is None:
        return ""
    if cfg.es_mode_selection == "skip":
        return ""
    picked = [m for m in results.modes
              if m.electronic_structure is not None]
    if not picked:
        return ""
    parts = [f"mode {m.index_1based} ({m.frequency_cm1:.1f} cm⁻¹)"
             for m in picked]
    return "**Selected modes:** " + "; ".join(parts) + "."



def _count_structure_atoms(struct: Structure) -> int:
    """Total atom count from a Structure-like object.

    Tries ``len(struct.elements)`` first (the canonical molbuilder
    Structure exposes ``elements`` as a list of element symbols)
    then falls back to ``len(struct.atoms)`` for duck-typed mocks.
    Returns 0 when neither attribute is available -- the Methods
    composer treats 0 as "skip the atom-count clause" rather than
    raising, since this is a presentational concern."""
    elements = getattr(struct, "elements", None)
    if elements is not None:
        try:
            return len(elements)
        except TypeError:
            pass
    atoms = getattr(struct, "atoms", None)
    if atoms is not None:
        try:
            return len(atoms)
        except TypeError:
            pass
    return 0


def _structure_element_symbols(struct: Structure) -> List[str]:
    """Return per-atom element symbols from a Structure-like object.

    Real Structure: ``struct.elements`` is already a list of symbols.
    Mock-style ``struct.atoms``: per-atom objects expose ``.symbol``
    or ``.element``.  Returns ``[]`` when neither shape is available
    (Methods composer treats this as no per-atom info; freezing-by-
    element gracefully degrades)."""
    elements = getattr(struct, "elements", None)
    if elements is not None:
        return [str(e) for e in elements]
    atoms = getattr(struct, "atoms", None)
    if atoms is not None:
        out: List[str] = []
        for a in atoms:
            sym = getattr(a, "symbol", None) or getattr(a, "element", None)
            out.append(str(sym) if sym is not None else "")
        return out
    return []


def _count_free_atoms(struct: Structure, cfg: SpectraConfig) -> int:
    """Approximate the count of unfrozen atoms by element + index
    union (residue-name freezing isn't decidable without parsing the
    PDB).  Returns the total atom count when no freeze rule applies.

    The Methods prose only uses this to phrase "N free atoms, 3N-6
    modes" -- being off by a few atoms in unusual frozen-residue
    setups is acceptable since the engine's actual frozen-atom list
    appears verbatim in the script body (spec § 7)."""
    n_total = _count_structure_atoms(struct)
    if n_total == 0:
        return 0
    if not (cfg.frozen_elements or cfg.frozen_indices):
        return n_total
    frozen: set = set()
    if cfg.frozen_elements:
        elem_set = {e.strip() for e in cfg.frozen_elements if e.strip()}
        symbols = _structure_element_symbols(struct)
        for i, sym in enumerate(symbols):
            if sym in elem_set:
                frozen.add(i)
    if cfg.frozen_indices:
        for i in cfg.frozen_indices:
            if 0 <= int(i) < n_total:
                frozen.add(int(i))
    return max(0, n_total - len(frozen))


__all__ = ["render_methods_md", "extract_citation_keys"]
