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

Engine-specific fragments are PASSED IN as ``fragment_md`` by the
caller that knows its own engine (today the vibration deck, with
:func:`molbuilder.pyscf.vibration_emitters.pyscf_methods_fragment`).
The composer here stays ignorant of which engine ran the job, so a
future SIESTA engine adds a producer without changing this file.

*(This named ``SpectraEngine.methods_fragment`` and a registry
lookup until 2026-09-17.  That class and ``spectra/engine_base.py``
were deleted at the spectra migration's P3, 2026-08-21 -- there is
no engine registry anywhere in the tree; see ``engines/overview.md``
§ 5.  The ``fragment_md`` parameter below has described the real
mechanism the whole time.)*

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

from typing import TYPE_CHECKING
if TYPE_CHECKING:            # annotations only -- importing
    # vibration_deck at run time would cycle (it imports this), and the
    # other two are named in annotations and never called: this module
    # travels beside a SIESTA vibration job (`runwrap.VIBRATION_COMPANIONS`),
    # where neither is importable.
    from .vibration_deck import VibrationConfigView
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
    cfg: "VibrationConfigView",
    *,
    results: Optional[SpectraResults] = None,
    fragment_md: str = "",
    struct: Optional[Structure] = None,
) -> str:
    """Compose the Markdown Methods paragraph for the given config.

    Parameters
    ----------
    cfg
        The vibration deck's config view being rendered.  Drives every
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
        Markdown ready to drop into a manuscript draft.  Two surfaces
        carry it: the deck's own header docstring, and the Results
        panel's "Methods text" block (the preview MODAL this used to
        name left with the Generate lane at P3, and the paragraph had
        no surface at all between then and 2026-09-11).

        INCOMPLETE BY DESIGN on the pre-run path.  Which dmu/dR route a
        run took cannot be known here -- it is settled inside the job --
        so :func:`with_ir_route` adds that sentence later, from the load
        path, where the results are in hand.
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


def ir_route_sentence(ir_route: str,
                      fd_step_ang: Optional[float] = None) -> str:
    """The one Methods fact that cannot be known when Methods is written.

    `render_methods_md` runs in the deck composer, on the host, BEFORE
    the job exists -- and which dmu/dR route runs is a property of the
    env the deck lands in (the analytic one needs
    ``pyscf.prop.infrared``, which no PyPI release carries).  So the
    paragraph itself stays route-neutral, which is the only honest thing
    it can be, and this sentence is added later by whoever holds the
    RESULTS.

    One home for the wording, called from the load path, so the prose a
    reader copies into a paper and the provenance the viewer shows can
    never drift into two different claims.  Returns "" when there is
    nothing to say.
    """
    if ir_route == "analytic":
        return ("These were evaluated analytically, from the "
                "coupled-perturbed self-consistent-field response also "
                "used for the force constants.")
    if ir_route == "finite-difference":
        step = (f" (±{fd_step_ang:g} Å per Cartesian coordinate)"
                if fd_step_ang else "")
        return f"These were evaluated by central finite differences{step}."
    return ""


def with_ir_route(methods_md: str, ir_route: str,
                  fd_step_ang: Optional[float] = None) -> str:
    """``methods_md`` with the route sentence placed in the paragraph.

    Before the bibliography, never after it: a trailing sentence under a
    reference list reads as a footnote to the references rather than as
    part of the method.
    """
    sentence = ir_route_sentence(ir_route, fd_step_ang)
    if not sentence or not methods_md:
        return methods_md
    marker = "\n**Bibliography**"
    if marker in methods_md:
        head, _, tail = methods_md.partition(marker)
        return f"{head.rstrip()} {sentence}\n{marker}{tail}"
    return f"{methods_md.rstrip()} {sentence}\n"


def siesta_methods_text(*, displacement_bohr: float, n_free: int,
                        n_held: int, n_rigid: int, siesta_version: str) -> str:
    """The Methods paragraph for SIESTA's force-constant route
    (`engines/vibration.md` § 5.5), with the citation keys the science
    contract carries -- composed by the job's finish once the run has said
    its version and the analysis how many motions it removed."""
    held = (f" {n_held} atom(s) were held fixed and no force constant was "
            f"taken with respect to them -- partial Hessian vibrational "
            f"analysis [Head1997, LiJensen2002], the block taken from the "
            f"forces of the full system [Besley2008]." if n_held else "")
    removed = (f" The {n_rigid} whole-body motion(s) "
               f"{'the held geometry permits' if n_held else 'of the free system'}"
               f" were projected out before diagonalisation [Ghysels2008]."
               if n_rigid else "")
    ver = f" (SIESTA {siesta_version})" if siesta_version else ""
    return (
        "## Methods\n\n"
        f"Harmonic force constants were obtained by central finite "
        f"differences of the analytic forces{ver}: each of the {n_free} "
        f"free atoms was displaced by ±{displacement_bohr:g} Bohr along "
        f"x, y and z (`MD.TypeOfRun FC`).{held} The resulting "
        f"{'partial ' if n_held else ''}Hessian was mass-weighted with isotope-averaged masses and "
        f"diagonalised.{removed} Infrared and Raman intensities are not "
        f"computed on this route."
    )


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


def _paragraph_vibrational(cfg: "VibrationConfigView",
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
            n_modes, n_rigid = _mode_count(struct, cfg, results=results)
            removed = (f" after projecting out the {n_rigid} whole-body "
                       f"motion(s) the geometry permits" if n_rigid
                       else "")
            atom_clause = (f" The system contains {n_atoms} atoms "
                           f"({n_free} free, {n_atoms - n_free} frozen "
                           f"during the Hessian), giving "
                           f"{n_modes} non-translational / non-rotational "
                           f"vibrational modes{removed}.")

    raman_clause = ""
    if cfg.compute_raman:
        # Mention Komornicki1979 + Wilson1955: dα/dR method paper +
        # the canonical normal-coordinate framework that maps
        # Cartesian polarizability derivatives to mode-projected
        # Raman activities.
        # ONE statement of the method (engines/vibration.md § 4.6): the polarizability
        # is analytic at each displaced point, its DERIVATIVE is a central
        # finite difference -- the engine's own fragment quotes the step.
        raman_clause = (" Raman activities (Å⁴/amu) were computed from "
                        "polarizability derivatives -- the static "
                        "polarizability analytic at each displaced "
                        "geometry, its derivative by central finite "
                        "differences over the free Cartesian coordinates "
                        "[Komornicki1979] -- projected onto the mode "
                        "eigenvectors using the standard normal-"
                        "coordinate framework [Wilson1955].")

    # IR.  DELIBERATELY ROUTE-NEUTRAL.  Methods text is composed at
    # EMIT time (deck composer, results=None), and which dmu/dR route
    # runs is only settled inside the job -- the analytic one needs
    # `pyscf.prop.infrared`, a property of the env the deck lands in.
    # A sentence here claiming "analytic" would therefore be a claim
    # this code cannot keep, which is the one thing a Methods section
    # must never contain.  Both routes compute the same derivative and
    # project it the same way, so the sentence describes THAT, and the
    # route actually taken is reported beside the results
    # (`SpectraResults.ir_route`, shown in the viewer's run summary).
    # When this text is ever re-rendered WITH results in hand, this is
    # the clause that gains the route.
    ir_clause = ""
    if cfg.compute_ir:
        ir_clause = (" Infrared intensities (km mol⁻¹) were obtained "
                     "from dipole-moment derivatives with respect to "
                     "the nuclear Cartesian coordinates "
                     "[Komornicki1979], projected onto the "
                     "mass-weighted normal coordinates [Wilson1955].")

    para = (f"Harmonic vibrational analysis was performed at the "
            f"{fxc}/{basis}{fxc_cite} level{disp_clause}.{atom_clause}"
            f"{ir_clause}{raman_clause}")

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


def _paragraph_electronic_structure(cfg: "VibrationConfigView",
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


def _frequency_window_clause(cfg: "VibrationConfigView") -> str:
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


def _selected_modes_line(cfg: "VibrationConfigView",
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



def _mode_count(struct: Structure, cfg: "VibrationConfigView", *,
                results: "Optional[SpectraResults]" = None):
    """How many vibrations this system has, and how many motions were removed.

    Returns ``(n_modes, n_rigid)``.

    **THE CALCULATION IS THE AUTHORITY when there is one**: where results
    exist the count is the length of the mode list the run produced and the
    removed count is what the run recorded, never a formula re-derived
    beside them (science/normal-modes.md R6).

    Before the run, the count is R2 -- ``3 N_free - n_rigid`` -- with
    ``n_rigid`` from the one derivation (R1): the rank rule in
    ``spectra.normal_modes``, the same function the deck splices, so the
    paragraph written into the deck header and the list the run produces
    agree by construction.  Straight molecules, a lone atom, held atoms
    on a line: none is a case here, because none is a case there.
    """
    if results is not None:
        modes = getattr(results, "modes", None)
        if modes is not None:
            removed = getattr(results, "removed_motions", None) or {}
            return len(modes), int(removed.get("count", 0) or 0)
    from .normal_modes import rigid_motions
    n_free = _count_free_atoms(struct, cfg)
    try:
        positions = struct.positions
    except AttributeError:                  # a duck-typed mock: no geometry
        return max(0, 3 * n_free - 6), 6
    frozen = [int(i) for i in (cfg.frozen_indices or [])
              if 0 <= int(i) < len(positions)]
    n_rigid = len(rigid_motions(
        positions, frozen,
        getattr(struct, "axis_kind", None) or ("isolated",) * 3,
        cell=getattr(struct, "cell", None)))
    return 3 * n_free - n_rigid, n_rigid


def _count_free_atoms(struct: Structure, cfg: "VibrationConfigView") -> int:
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
    # FROZEN ATOMS ARE NAMED BY INDEX, and only by index.  A
    # freeze-by-ELEMENT arm stood here and was unreachable: the one caller
    # is the vibration deck, which passes its config view, and that view
    # supplies `frozen_elements = []` always.  Only the retired
    # `SpectraConfig` could carry a value, and nothing constructed it
    # outside tests -- so the arm was exercised by its own fixture and by
    # nothing else (2026-08-22).  The region store holds indices, and the
    # deck writes those indices into geomeTRIC's constraints file.
    if not cfg.frozen_indices:
        return n_total
    frozen: set = set()
    if cfg.frozen_indices:
        for i in cfg.frozen_indices:
            if 0 <= int(i) < n_total:
                frozen.add(int(i))
    return max(0, n_total - len(frozen))


__all__ = ["render_methods_md", "extract_citation_keys"]
