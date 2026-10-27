"""Engine-agnostic Methods-paragraph composer for the Spectra tab.

Produces the Markdown prose that ships in two places:

  * the header docstring of the emitted deck;
  * the ``methods_text`` field of :class:`SpectraResults`, which the
    run writes from the same paragraph and the Results panel's
    "Methods text" block shows.

The same prose appears in both, so the emitted script and the parsed
JSON say the same thing.  It is
composed ONCE, before the run, from the configured knobs: the count it
states is R2's, equal to the run's by construction (R6), and the one
thing only the run knows -- which dmu/dR route it took -- is added at
load by :func:`with_ir_route`.

Engine-specific fragments are PASSED IN as ``fragment_md`` by the
caller that knows its own engine (today the vibration deck, with
:func:`molbuilder.pyscf.vibration_emitters.pyscf_methods_fragment`).
The composer here stays ignorant of which engine ran the job.

Citation keys appear inline as ``[KeyName]`` markers (e.g.
``[Sun2020]``).  Every key cited here must resolve against
``docs/science/references.bib``; :func:`extract_citation_keys`
gives the caller the list of keys actually used so it can render a
trailing bibliography and the ``bibliography_keys`` field of
:class:`SpectraResults` stays in sync.
"""

from __future__ import annotations

import re
from typing import Any, List, Mapping, Optional, Sequence

from typing import TYPE_CHECKING
if TYPE_CHECKING:            # annotations only -- importing
    # vibration_deck at run time would cycle (it imports this), and the
    # other is named in annotations and never called: this module travels
    # beside a SIESTA vibration job (`runwrap.VIBRATION_COMPANIONS`), where
    # it is not importable.
    from ..pyscf.vibration_deck import VibrationConfigView
    from ..structure import Structure


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
    fragment_md
        Optional engine-specific paragraph, supplied by the caller
        (the vibration deck passes
        :func:`molbuilder.pyscf.vibration_emitters.
        pyscf_methods_fragment`).  Empty means no engine paragraph.
        This module is engine-IGNORANT by design; the caller knows its
        own engine.
    struct
        Optional :class:`Structure`.  Used to phrase "N free atoms,
        3N-6 modes" etc. when available; falls back to generic
        prose when ``None``.

    Returns
    -------
    str
        Markdown ready to drop into a manuscript draft.  Two surfaces
        carry it: the deck's own header docstring, and the Results
        panel's "Methods text" block.

        INCOMPLETE BY DESIGN, since it is composed before the run.  Which
        dmu/dR route a run took cannot be known here -- it is settled
        inside the job -- so :func:`with_ir_route` adds that sentence
        later, from the load path, where the results are in hand.
        Use :func:`extract_citation_keys` on the result to obtain
        the bibliography-key list.
    """
    parts: List[str] = []
    parts.append("## Methods\n")

    # ------------------------------------------------------------------ #
    # Paragraph 1: vibrational analysis setup (always emitted)
    # ------------------------------------------------------------------ #
    p1 = _paragraph_vibrational(cfg, struct=struct)
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
        p2 = _paragraph_electronic_structure(cfg)
        parts.append(p2)

    # ------------------------------------------------------------------ #
    # Trailing bibliography (BibTeX keys, one per line).  Composed
    # from the keys present in the text we just built so a reader of
    # the emitted script has the full list inline.
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
    return _with_sentence(methods_md, ir_route_sentence(ir_route,
                                                         fd_step_ang))


def _with_sentence(methods_md: str, sentence: str) -> str:
    """``methods_md`` with ``sentence`` placed at the end of the paragraph,
    before the bibliography."""
    if not sentence or not methods_md:
        return methods_md
    marker = "\n**Bibliography**"
    if marker in methods_md:
        head, _, tail = methods_md.partition(marker)
        return f"{head.rstrip()} {sentence}\n{marker}{tail}"
    return f"{methods_md.rstrip()} {sentence}\n"


def _direction_words(view: Mapping[str, Any]) -> str:
    """The light's direction a view names -- its axis name, or the vector
    itself where it is none."""
    return view.get("direction_name") or "(" + ", ".join(
        f"{float(x):g}" for x in view["direction"]) + ")"


def infrared_view_sentence(view: Mapping[str, Any]) -> str:
    """The sentence naming the infrared a page shows of a SIESTA result
    (`web/spectra.md` § 3.2): the atoms, the partition and how the molecule
    meets the light -- held, in one orientation, along the light's direction,
    three times the orientation average; or free, nothing held, the
    orientation average itself (`science/vibrational-optical-response.md`
    § 2.3)."""
    partition = str(view["partition"]).capitalize()
    head = ("The infrared shown is the whole system's, from the dipole "
            "change the engine computed along each mode"
            if view.get("source") == "dipole" else
            f"The infrared shown is that of {view['atoms_label']}, from "
            f"their {partition} charges")
    if view.get("orientation") == "free":
        return (f"{head}, averaged over orientations: no atom is held, so the "
                f"molecule tumbles, as a free molecule's infrared intensity "
                f"reports.")
    return (f"{head}, for light polarized along {_direction_words(view)}: "
            f"atoms are held, "
            f"so the molecule meets the light in one orientation -- three "
            f"times the orientation average a free molecule's infrared "
            f"intensity reports.")


def with_infrared_view(methods_md: str, view: Mapping[str, Any]) -> str:
    """``methods_md`` with :func:`infrared_view_sentence` placed in the
    paragraph -- composed at load, as the view is."""
    return _with_sentence(methods_md, infrared_view_sentence(view))


def raman_view_sentence(view: Mapping[str, Any]) -> str:
    """The sentence naming the Raman a page shows of a SIESTA result
    (`engines/vibration.md` § 5.11 § 10): the route, the light's direction,
    ``δ``, the laser, the width, and that the dielectric function is of
    independent states with no local field
    (`science/vibrational-optical-response.md` § 3.3)."""
    from .raman import width_words
    return (f"The Raman strengths shown are R = |∂α/∂Q|², the squared "
            f"modulus of the polarizability's change along each mode for "
            f"light polarized along {_direction_words(view)} [Gillet2013]: "
            f"α from SIESTA's "
            f"dielectric function of independent states, with no local "
            f"field, at ±δQ along each mode (the atom that moves most moved "
            f"by {float(view['delta_ang']):g} Å), complex at the laser's "
            f"{float(view['laser_nm']):g} nm "
            f"({float(view['photon_energy_ev']):.4f} eV) -- its real part by "
            f"Kramers–Kronig, its imaginary part the absorption there -- "
            f"the spectrum broadened as SIESTA's Optical.Broaden, a Gaussian "
            f"exp(-(E/s)²) of s = {width_words(view['broaden_ev'])} eV, the "
            f"run's own s being {width_words(view['run_broaden_ev'])} eV. "
            f"They are strengths, not intensities: no laser-frequency or "
            f"temperature factor is applied.")


def frequency_range_words(freq_min_cm1, freq_max_cm1) -> str:
    """A frequency range in words, either end blank for none -- *from 1000
    to 2000 cm⁻¹*, *from 1000 cm⁻¹ up*, *up to 2000 cm⁻¹* -- or ``""`` for
    no range.  One wording for every place that says Raman for's range:
    the Methods paragraph, the PySCF deck, migrate's account."""
    if freq_min_cm1 is None and freq_max_cm1 is None:
        return ""
    if freq_max_cm1 is None:
        return f"from {float(freq_min_cm1):g} cm⁻¹ up"
    if freq_min_cm1 is None:
        return f"up to {float(freq_max_cm1):g} cm⁻¹"
    return (f"from {float(freq_min_cm1):g} to {float(freq_max_cm1):g} "
            f"cm⁻¹")


def siesta_raman_sentence(raman_route: str, raman=None) -> str:
    """The Methods paragraph's Raman sentence on SIESTA's route, by the
    route the run took and what it asked (`engines/vibration.md` § 5.11
    § 8): its optical calls planned -- the strengths derived on request at
    a laser; Raman asked (``raman``, the deck's `vibration` block's) over a
    range that kept no mode; or none asked."""
    if raman_route == "optical":
        return ("Raman strengths are derived on request, at a laser the "
                "reader names, from SIESTA's optical calculation "
                "(independent states, no local field) at ±δQ along each "
                "picked mode.")
    if raman is not None:
        where = frequency_range_words(raman.get("freq_min_cm1"),
                                      raman.get("freq_max_cm1"))
        return (f"Raman strengths were asked for the modes {where}; none of "
                f"the run's modes lies in that range, so no optical "
                f"calculation ran." if where else
                "Raman strengths were asked; the run reports no mode, so no "
                "optical calculation ran.")
    return "Raman strengths were not requested in this run."


def with_siesta_raman_route(methods_md: str, raman_route: str,
                            raman=None) -> str:
    """``methods_md`` with :func:`siesta_raman_sentence` placed in the
    paragraph."""
    return _with_sentence(methods_md,
                          siesta_raman_sentence(raman_route, raman))


def with_raman_view(methods_md: str, view: Mapping[str, Any]) -> str:
    """``methods_md`` with :func:`raman_view_sentence` placed in the
    paragraph -- composed at load, as the view is."""
    return _with_sentence(methods_md, raman_view_sentence(view))


#: Each partition the force-constant run's charges may be taken by, as the
#: Methods paragraph names and cites it (`engines/vibration.md` § 5.10 ②).
CHARGE_PARTITION_WORDS = {"hirshfeld": "Hirshfeld [Hirshfeld1977]",
                          "voronoi": "Voronoi [FonsecaGuerra2004]"}


def siesta_methods_text(*, displacement_bohr: float, n_free: int,
                        n_held: int, n_rigid: int, siesta_version: str,
                        charge_partitions: Sequence[str] = ()) -> str:
    """The Methods paragraph for SIESTA's force-constant route
    (`engines/vibration.md` § 5.5), with the citation keys the science
    contract carries -- composed by the job's finish once the run has said
    its version, the analysis how many motions it removed, and its output
    which partitions' charges it printed (``charge_partitions``)."""
    held = (f" {n_held} atom(s) were held fixed and no force constant was "
            f"taken with respect to them -- partial Hessian vibrational "
            f"analysis [Head1997, LiJensen2002], the block taken from the "
            f"forces of the full system [Besley2008]." if n_held else "")
    removed = (f" The {n_rigid} whole-body motion(s) "
               f"{'the held geometry permits' if n_held else 'of the free system'}"
               f" were projected out before diagonalisation [Ghysels2008]."
               if n_rigid else "")
    ver = f" (SIESTA {siesta_version})" if siesta_version else ""
    named = [CHARGE_PARTITION_WORDS[p] for p in charge_partitions]
    # THE RAMAN SENTENCE IS THE RUN'S ROUTE'S (`with_siesta_raman_route`),
    # added once the run knows whether its optical calls were planned.
    intensities = (
        f" {' and '.join(named)} atomic charges were recorded at every "
        f"displacement, from which the infrared of a chosen set of atoms is "
        f"derived."
        if named else
        # A FINISH THAT KEEPS NO CHARGES IS A RUN THAT ASKED FOR NONE: one
        # that asked and printed none is refused (`siesta_vibration.
        # partial_charges_of`).
        " Infrared intensities were not requested in this run.")
    return (
        "## Methods\n\n"
        f"Harmonic force constants were obtained by central finite "
        f"differences of the analytic forces{ver}: each of the {n_free} "
        f"free atoms was displaced by ±{displacement_bohr:g} Bohr along "
        f"x, y and z (`MD.TypeOfRun FC`).{held} The resulting "
        f"{'partial ' if n_held else ''}Hessian was mass-weighted with isotope-averaged masses and "
        f"diagonalised.{removed}{intensities}"
    )


def extract_citation_keys(text: str) -> List[str]:
    """Return the BibTeX keys cited in ``text``, in order of first
    appearance, deduplicated.

    Matches ``[Key]`` and ``[Key §section]`` patterns.  Used by
    :func:`render_methods_md` to build the trailing bibliography,
    and by the engine to populate :attr:`SpectraResults.
    bibliography_keys`.
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


#: Each dispersion choice as the literature writes it, and the papers that
#: define it: D3 itself [Grimme2010]; Becke-Johnson damping on top of it
#: [Grimme2011]; D4 [Caldeweyher2019].
_DISPERSION_PROSE = {
    "d3bj":   ("D3(BJ)", "Grimme2010, Grimme2011"),
    "d3zero": ("D3(0)", "Grimme2010"),
    "d4":     ("D4", "Caldeweyher2019"),
}


def _paragraph_vibrational(cfg: "VibrationConfigView",
                           *,
                           struct: Optional[Structure]) -> str:
    """First Methods paragraph: harmonic vibrational analysis +
    (optional) Raman activities.  Always emitted -- L2 is the
    foundation layer, you can't have a Spectra-tab run without it."""
    basis = cfg.basis
    # THE LEVEL OF THEORY AS RUN (engines/vibration.md § 4.10): the one
    # answer the deck's SCF construction asks, `is_dft`.  Hartree-Fock is
    # named with its SCF class (composed from the method and the spin
    # treatment, `pyscf/layout.scf_class`) and basis -- it has no
    # functional, whatever that item still holds.  The dispersion correction is named on either
    # method: HF takes it too, with parameters fitted for it (HF-D3(BJ)).
    if cfg.is_dft:
        fxc = cfg.functional
        # Functional-specific citation: B3LYP gets [Becke1993]; other
        # functionals would ideally cite their primary paper, but we
        # don't carry a per-functional citation map yet, so we cite
        # Becke1993 only for the B3 family.
        fxc_cite = " [Becke1993]" if fxc.upper().startswith("B3") else ""
        level = f"{fxc}/{basis}{fxc_cite}"
    else:
        level = f"Hartree-Fock ({cfg.scf_class})/{basis}"
    disp_clause = ""
    if cfg.dispersion != "none":
        name, cite = _DISPERSION_PROSE[cfg.dispersion]
        disp_clause = f" with the {name} dispersion correction [{cite}]"

    # Atom-count clause -- only when we have a Structure to count from.
    # Structure stores elements as a list of element symbols; n_atoms
    # is its length.  Defensive try/except so a duck-typed mock (in
    # tests) carrying `.elements` works too.
    atom_clause = ""
    if struct is not None:
        n_atoms = _count_structure_atoms(struct)
        if n_atoms:
            # THE ONE COUNT prep's check states (`validation.spectra.
            # modes_of`), imported here because this paragraph is written on
            # this machine alone -- the module travels for SIESTA's.
            from ..validation.spectra import modes_of
            n_free, n_rigid, n_modes = modes_of(struct, "pyscf")
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
    # EMIT time (the deck composer, before the run), and which dmu/dR route
    # runs is only settled inside the job -- the analytic one needs
    # `pyscf.prop.infrared`, a property of the env the deck lands in.
    # A sentence here claiming "analytic" would therefore be a claim
    # this code cannot keep, which is the one thing a Methods section
    # must never contain.  Both routes compute the same derivative and
    # project it the same way, so the sentence describes THAT, and the
    # route actually taken is reported beside the results
    # (`SpectraResults.ir_route`, shown in the viewer's run summary) and
    # added to this text at load (`with_ir_route`).
    ir_clause = ""
    if cfg.compute_ir:
        ir_clause = (" Infrared intensities (km mol⁻¹) were obtained "
                     "from dipole-moment derivatives with respect to "
                     "the nuclear Cartesian coordinates "
                     "[Komornicki1979], projected onto the "
                     "mass-weighted normal coordinates [Wilson1955].")

    return (f"Harmonic vibrational analysis was performed at the "
            f"{level} level{disp_clause}.{atom_clause}"
            f"{ir_clause}{raman_clause}")


def _paragraph_electronic_structure(cfg: "VibrationConfigView") -> str:
    """Second Methods paragraph: per-mode displaced-geometry SCFs.
    Only emitted when ``cfg.es_mode_selection != "skip"`` -- the
    L4 step is opt-in."""
    amp = cfg.displacement_amplitude_ang
    n_below = cfg.es_n_homo_below
    n_above = cfg.es_n_lumo_above

    sel = cfg.es_mode_selection
    if sel == "all":
        criterion = "every vibrational mode"
    elif sel == "explicit":
        criterion = (f"a user-specified set of {len(cfg.explicit_modes)} "
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
    return para


def _frequency_window_clause(cfg: "VibrationConfigView") -> str:
    """Inline phrase describing the frequency window when one is in
    effect.  Empty string when no window or selector=explicit
    (window is ignored there)."""
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


def _count_structure_atoms(struct: Structure) -> int:
    """The structure's atom count -- ``len(struct.elements)``, the one
    shape a `Structure` has (a duck-typed fallback for mocks stood here
    until 2026-10-18)."""
    return len(struct.elements)



__all__ = ["render_methods_md", "extract_citation_keys"]
