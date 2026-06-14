"""TranSIESTA engine for the Transport tab (Phase B.3, 2026-06-10).

Scope of THIS module — what's in vs deferred
============================================

**Shipped today (B.3 zero-bias scope):**

* ``TransiestaEngine`` class registered as ``"transiesta"`` in the
  :mod:`molbuilder.transport` engine registry.
* :meth:`TransiestaEngine.render_script` — emits a runnable device
  ``.fdf`` at ZERO bias (``TS.Voltage = bias_voltages_v[0]``).
  The script assumes the electrode ``.TSHS`` files already exist
  in the run directory; see "Electrode .fdf workflow" below for
  how to produce them.
* :meth:`TransiestaEngine.preflight` — basic region-label
  validation + the cross-engine ``check_open_shell_metal`` check.

**Deferred to follow-up sessions:**

* :meth:`TransiestaEngine.parse_output` — needs the
  ``<job>.transport.json`` schema designed first.  Raises
  ``NotImplementedError`` with a clear message today.
* :meth:`TransiestaEngine.methods_fragment` — full manuscript-
  ready paragraph; today returns a one-line placeholder.
* Electrode ``.fdf`` generation (see below).
* Bias-scan looping (see below).
* PySCF-NEGF engine.

Electrode .fdf workflow
=======================

TranSIESTA needs a ``.TSHS`` Hamiltonian file per electrode (left
+ right).  Each is the output of a SEPARATE SIESTA run on a lead
geometry (typically a few unit cells of the lead material — e.g.
Au(111) slab for gold electrodes — with periodic boundary
conditions along the transport direction).  The lead ``.fdf``
must include ``TS.HSFileEnable = T`` so SIESTA writes the
``.TSHS`` file on completion.

**Current path (manual):**

1. Build the lead geometry on the ``/molbuilder`` tab (Au(111)
   slab, etc.).
2. On ``/structure-optimization``, generate a SIESTA ``.fdf``;
   manually add ``TS.HSFileEnable = T`` to the emitted file
   (today's SIESTA emitter doesn't set this — a future cleanup).
3. Run SIESTA on the lead.  It emits ``<jobname>.TSHS``.
4. Rename to ``<device_jobname>_L_electrode.TSHS`` (or
   ``_R_electrode.TSHS``) and copy into the device run directory.
5. Repeat for the right electrode.
6. Generate the device ``.fdf`` here.  ``TS.HSFileLeft`` and
   ``TS.HSFileRight`` reference the files copied in step 4.

**Planned (B.3 follow-up):** Add an "electrode wizard" step on
the Transport tab that extracts the ``L-electrode`` /
``R-electrode`` regions from the labeled device structure, plus a
few periodic-image unit cells, and emits the matching SIESTA
``.fdf`` automatically.  Tracked in memory
``project_transport_electrode_bias_workflow.md``.

Bias-scan workflow
==================

TranSIESTA's ``TS.Voltage`` is a single value per run.  T(E) at
multiple bias values = multiple TranSIESTA runs (re-converge NEGF
density at each bias point — expensive).

``TransportConfig.bias_voltages_v`` is already a ``List[float]``.
Today's engine emits ONLY ``bias_voltages_v[0]``; the preflight
WARNs when ``len > 1``.

**Current path (manual):** Generate a separate ``.fdf`` for each
bias value by changing ``bias_voltages_v`` and re-clicking
Generate.  Run each ``.fdf`` separately.  Post-process the per-
run ``.TBT.AVTRANS_*`` files into a single I-V curve.

**Planned (B.3 follow-up):** For ``len(bias_voltages_v) > 1``,
emit ONE ``.fdf`` per bias point named ``<jobname>_V0.fdf``,
``<jobname>_V1.fdf``, etc., plus a ``<jobname>_run.sh`` driver
script that loops over them and post-processes the per-bias
transmission files into a single ``<job>.transport.json``.
Tracked in the same memory above.

Runnable assumptions
====================

The device ``.fdf`` emitted by :meth:`render_script` is runnable
**if and only if**:

* ``<jobname>_L_electrode.TSHS`` and ``<jobname>_R_electrode.TSHS``
  exist in the run directory (see Electrode workflow above).
* SIESTA-MPI is built with TranSIESTA enabled.
* Atomic pseudopotentials (``.psml`` files) for every species
  present are reachable via the standard SIESTA pseudo-lookup
  (project's ``pseudopotential/`` dir or ``$SIESTA_PP_PATH``).

If any of these are missing, SIESTA fails at parse time with a
clear error.  ``render_script`` doesn't try to predict the
runtime; preflight catches structural issues that would block
emission regardless.

Four-env model + the runner shell script
========================================

molbuilder's :doc:`four-env model </README_install>` separates the
host env from the per-backend envs.  TranSIESTA is part of the
SIESTA suite, so the device ``.fdf`` runs in
``molbuilder-siesta`` — the same env the standalone SIESTA
optimization flow uses.

The web blueprint pairs ``render_script``'s ``.fdf`` output with
a ``<jobname>.run.sh`` shell wrapper produced by
:func:`molbuilder.runwrap.render_run_wrapper`.  The wrapper does
NOT need to be customised per engine; runwrap routes ``.fdf``
extensions to the ``siesta`` category automatically + picks up
``molbuilder-siesta`` via ``Capabilities.env_for_category('siesta')``.
The wrapper handles the three conda-activation paths (already in
env / conda on PATH / conda missing) and surfaces a clear error
if the target env isn't installed.

So a user's full lifecycle is:

1. Generate the device ``.fdf`` from the Transport tab (this
   engine's :meth:`render_script`).
2. Web blueprint also emits ``<jobname>.run.sh`` via
   :func:`molbuilder.runwrap.write_run_wrapper`.
3. User runs ``bash <jobname>.run.sh`` from the run directory.
4. Wrapper activates ``molbuilder-siesta``, launches
   ``mpirun -np <N> siesta <jobname>.fdf``, captures output.

The conda env model is invisible to the engine's emitter; the
runner ``.run.sh`` carries it.  Pinned in memory
``project_four_env_model.md``.
"""

from __future__ import annotations

from typing import List, Optional

from ..config.transport import (
    EXPECTED_REGIONS_2T,
    REGION_BRIDGE,
    REGION_LEFT_ELECTRODE,
    REGION_RIGHT_ELECTRODE,
    TransportConfig,
)
from ..issues import Issue
from ..structure import Structure
from .engine_base import TransportEngine, register_engine
from .results import TransportResults


# --------------------------------------------------------------------- #
#  .fdf emission helpers — kept as small private functions so each      #
#  block is independently testable.  Each returns a list of lines;      #
#  ``render_script`` concatenates them.                                 #
# --------------------------------------------------------------------- #


def _emit_header(cfg: TransportConfig) -> List[str]:
    """SystemLabel + a banner explaining what the file is.

    The banner is verbose-comments-style (matches the SIESTA
    emitter convention) so a user reading the ``.fdf`` cold sees
    the deferred-scope notes immediately.
    """
    return [
        "# ================================================================== #",
        f"#  TranSIESTA device .fdf — {cfg.job_name}",
        "#  Generated by molbuilder (Phase B.3 zero-bias scope, 2026-06-10).",
        "#",
        "#  Assumes electrode .TSHS files exist in this directory:",
        f"#    {cfg.job_name}_L_electrode.TSHS",
        f"#    {cfg.job_name}_R_electrode.TSHS",
        "#  See the SIESTA manual § 'TranSIESTA' for electrode-generation",
        "#  workflow; molbuilder's automated electrode wizard ships in a",
        "#  follow-up release.",
        "#",
        "#  Single bias point per .fdf (zero-bias today).  Bias-scan",
        "#  generation lands alongside the .transport.json parser.",
        "# ================================================================== #",
        "",
        f"SystemLabel            {cfg.job_name}",
        f"SystemName             Transport device for {cfg.job_name}",
        "",
    ]


def _emit_geometry(struct: Structure) -> List[str]:
    """Lattice + AtomicCoordinates blocks.

    For now we DON'T set a cell — the device geometry comes from a
    relaxed structure that may not carry one.  SIESTA accepts a
    LatticeConstant-only fallback; for proper transport the user
    should ensure the device geometry's transport direction is
    aligned with the z-axis and the cell is set correctly.  This
    is documented in the Transport tab UI prose.
    """
    from ase.data import atomic_numbers as _Z
    species = sorted(set(struct.elements), key=lambda e: e.capitalize())
    species_idx = {sp: i + 1 for i, sp in enumerate(species)}

    lines: List[str] = ["# --- Geometry ---", ""]
    lines.append(f"NumberOfAtoms          {struct.n_atoms}")
    lines.append(f"NumberOfSpecies        {len(species)}")
    lines.append("")
    lines.append("%block ChemicalSpeciesLabel")
    for sp in species:
        z = _Z.get(sp.capitalize(), 0)
        lines.append(f"  {species_idx[sp]:>3}  {z:>3}  {sp}")
    lines.append("%endblock ChemicalSpeciesLabel")
    lines.append("")
    lines.append("LatticeConstant        1.0 Ang")
    lines.append("# Lattice vectors — replace with your device's box "
                  "if known;")
    lines.append("# the placeholder below is a cubic 50 Å box that "
                  "leaves room for")
    lines.append("# the device + vacuum in the transverse directions.")
    lines.append("%block LatticeVectors")
    lines.append("   50.0   0.0   0.0")
    lines.append("    0.0  50.0   0.0")
    lines.append("    0.0   0.0  50.0")
    lines.append("%endblock LatticeVectors")
    lines.append("")
    lines.append("AtomicCoordinatesFormat        Ang")
    lines.append("%block AtomicCoordinatesAndAtomicSpecies")
    for el, (x, y, z) in zip(struct.elements, struct.positions):
        lines.append(
            f"  {x:14.8f} {y:14.8f} {z:14.8f}  {species_idx[el]}"
        )
    lines.append("%endblock AtomicCoordinatesAndAtomicSpecies")
    lines.append("")
    return lines


def _emit_basis_and_xc(cfg: TransportConfig) -> List[str]:
    """Basis set + XC + mesh cutoff + electronic temperature.

    Production-defensible defaults matching the SIESTA emitter's
    Method tab.  Mesh cutoff comes from the form; everything else
    is hard-coded for the zero-bias scope (B.3 follow-up: surface
    these as form fields).
    """
    return [
        "# --- Basis + XC ---",
        "",
        "PAO.BasisSize          DZP",
        "PAO.EnergyShift        0.01 Ry",
        "XC.Functional          GGA",
        "XC.Authors             PBE",
        f"MeshCutoff             {cfg.siesta_mesh_cutoff_ry} Ry",
        f"ElectronicTemperature  {cfg.electronic_temperature_k:.1f} K",
        "",
    ]


def _emit_k_mesh(cfg: TransportConfig) -> List[str]:
    """Monkhorst-Pack block for the TRANSVERSE k-mesh.

    The transport direction is treated by NEGF and is NOT part of
    the BZ sum.  For a finite molecule between leads, (1, 1, 1) is
    correct; for 1D-periodic electrodes set Nx, Ny appropriately.
    """
    kx, ky, kz = cfg.k_mesh_transverse
    return [
        "# --- Transverse k-mesh ---",
        "# Transport direction is NOT BZ-summed (NEGF handles it).",
        "# For a finite molecule between leads: (1, 1, 1).  For",
        "# 1D-periodic electrodes set Nx, Ny to the lead",
        "# periodicities; Nz stays at 1.",
        "",
        "%block kgrid_Monkhorst_Pack",
        f"  {kx:>3}    0    0      0.0",
        f"    0  {ky:>3}    0      0.0",
        f"    0    0  {kz:>3}      0.0",
        "%endblock kgrid_Monkhorst_Pack",
        "",
    ]


def _emit_transiesta_block(struct: Structure,
                            cfg: TransportConfig) -> List[str]:
    """The TS.* block — TranSIESTA-specific NEGF keywords.

    Region atom counts come from ``struct.regions``; bias comes
    from ``cfg.bias_voltages_v[0]`` (today's deferred-bias-scan
    scope).  TBtrans block for transmission post-processing
    included so a single ``siesta`` run produces both the NEGF
    self-consistent density and the T(E) table.
    """
    regions = struct.regions or {}
    n_left  = len(regions.get(REGION_LEFT_ELECTRODE, []))
    n_right = len(regions.get(REGION_RIGHT_ELECTRODE, []))
    bias    = cfg.bias_voltages_v[0] if cfg.bias_voltages_v else 0.0
    erange_relative = "T" if cfg.transmission_relative_to_ef else "F"

    return [
        "# --- TranSIESTA NEGF ---",
        "",
        "# Both SolutionMethod and TS.SolutionMethod are required by",
        "# the SIESTA parser: the generic SolutionMethod switches to",
        "# the TranSIESTA module; the TS.* form is the engine-local",
        "# confirmation.  Per TranSIESTA manual.",
        "SolutionMethod         transiesta",
        "TS.SolutionMethod      transiesta",
        "",
        "# Electrode .TSHS Hamiltonian files (see header banner).",
        f"TS.HSFileLeft          {cfg.job_name}_L_electrode.TSHS",
        f"TS.HSFileRight         {cfg.job_name}_R_electrode.TSHS",
        f"TS.NumUsedAtomsLeft    {n_left}",
        f"TS.NumUsedAtomsRight   {n_right}",
        "",
        "# Bias voltage — single value per .fdf today.  See module",
        "# docstring under 'Bias-scan workflow' for the multi-bias path.",
        f"TS.Voltage             {bias:.4f} eV",
        "",
        "# Complex contour (NEGF density integration).",
        "# Brandbyge et al., Phys. Rev. B 65, 165401 (2002) § IV.",
        f"TS.ComplexContour.Emin     {cfg.contour_e_bottom_ev:.2f} eV",
        f"TS.ComplexContour.NumCircle  {cfg.contour_n_circle}",
        f"TS.ComplexContour.NumLine    {cfg.contour_n_real}",
        "",
        "# TBtrans transmission post-processing.",
        f"TS.TBT.NumE            {cfg.transmission_n_points}",
        f"TS.TBT.Emin            {cfg.transmission_emin_ev:.2f} eV",
        f"TS.TBT.Emax            {cfg.transmission_emax_ev:.2f} eV",
        f"TS.TBT.Erange.RelToEF  {erange_relative}",
        "",
    ]


# --------------------------------------------------------------------- #
#  The engine                                                            #
# --------------------------------------------------------------------- #


@register_engine
class TransiestaEngine:
    """TranSIESTA NEGF engine (Phase B.3 zero-bias scope).

    See the module docstring for what's in vs deferred, the
    electrode-generation workflow, and the bias-scan plan.  The
    engine self-registers via the ``@register_engine`` decorator
    on import; the production import path is
    ``molbuilder.web.blueprints.__init__`` (mirrors the chemistry
    adapters).
    """

    name = "transiesta"
    label = "TranSIESTA (NEGF, pseudopotentials)"

    @classmethod
    def render_script(cls, struct: Structure,
                       cfg: TransportConfig) -> str:
        """Emit a runnable device ``.fdf`` for TranSIESTA at zero bias.

        Concatenates: header / geometry / basis+XC / k-mesh / TS.*.
        Output assumes electrode ``.TSHS`` files exist at the
        standard names (see module docstring).
        """
        lines: List[str] = []
        lines.extend(_emit_header(cfg))
        lines.extend(_emit_geometry(struct))
        lines.extend(_emit_basis_and_xc(cfg))
        lines.extend(_emit_k_mesh(cfg))
        lines.extend(_emit_transiesta_block(struct, cfg))
        return "\n".join(lines) + "\n"

    @classmethod
    def preflight(cls, struct: Structure,
                  cfg: TransportConfig,
                  prior: Optional[TransportResults] = None,
                  ) -> List[Issue]:
        """Basic sidecar-region validation + cross-engine chemistry.

        Errors block generation; warnings are informational and
        the user can override (the form-rendering layer surfaces
        them inline).
        """
        issues: List[Issue] = []
        regions = struct.regions or {}

        # Required region labels for a 2-terminal calculation.
        missing = [r for r in EXPECTED_REGIONS_2T if r not in regions]
        if missing:
            issues.append(Issue(
                severity="error",
                message=(
                    f"TranSIESTA preflight: missing required region "
                    f"labels {missing}.  Assign them on the Molbuilder "
                    f"tab (region picker) before generating; the "
                    f".molstruct.json sidecar carries them through "
                    f"to the engine."
                ),
                where="struct.regions",
            ))
            # Without region labels we cannot validate further.
            return issues

        # Non-empty electrode atom counts.
        for r in (REGION_LEFT_ELECTRODE, REGION_RIGHT_ELECTRODE):
            n = len(regions.get(r, []))
            if n == 0:
                issues.append(Issue(
                    severity="error",
                    message=(
                        f"TranSIESTA preflight: region {r!r} is empty.  "
                        f"Each electrode region must contain at least "
                        f"one atom."
                    ),
                    where=f"struct.regions.{r}",
                ))

        # Non-empty bridge.
        n_bridge = len(regions.get(REGION_BRIDGE, []))
        if n_bridge == 0:
            issues.append(Issue(
                severity="error",
                message=(
                    f"TranSIESTA preflight: region {REGION_BRIDGE!r} is "
                    f"empty.  The device region (bridge) must contain "
                    f"at least one atom — typically the molecule between "
                    f"the two electrodes."
                ),
                where=f"struct.regions.{REGION_BRIDGE}",
            ))

        # CRITICAL — atom ordering for TS.NumUsedAtomsLeft / Right.
        #
        # TranSIESTA reads ``TS.NumUsedAtomsLeft = N`` as "the FIRST
        # N atoms in the AtomicCoordinatesAndAtomicSpecies block are
        # the left electrode."  Same for Right (last M atoms).  If
        # the user's input XYZ has atoms in any order other than
        # [L-electrode][bridge][R-electrode], the .fdf SILENTLY
        # misidentifies which atoms go into which electrode self-
        # energy — producing chemically wrong transmission curves
        # with no run-time error.
        #
        # Reference: Brandbyge et al., Phys. Rev. B 65, 165401
        # (2002) § III; TranSIESTA manual ``TS.NumUsedAtomsLeft``
        # description.
        #
        # Block emission with an error so the user re-exports a
        # contiguous-ordered structure from the Molbuilder tab
        # (a "reorder for transport" affordance is a planned
        # follow-up).
        left_idx   = sorted(regions.get(REGION_LEFT_ELECTRODE, []))
        bridge_idx = sorted(regions.get(REGION_BRIDGE, []))
        right_idx  = sorted(regions.get(REGION_RIGHT_ELECTRODE, []))
        if left_idx and bridge_idx and right_idx:
            ordering_ok = (
                left_idx[-1]  < bridge_idx[0] and
                bridge_idx[-1] < right_idx[0] and
                # Each region must be contiguous (no gaps); a non-
                # contiguous L-electrode would also break the
                # "first N atoms" assumption.
                left_idx   == list(range(left_idx[0],
                                          left_idx[-1] + 1)) and
                bridge_idx == list(range(bridge_idx[0],
                                          bridge_idx[-1] + 1)) and
                right_idx  == list(range(right_idx[0],
                                          right_idx[-1] + 1))
            )
            if not ordering_ok:
                issues.append(Issue(
                    severity="error",
                    message=(
                        "TranSIESTA preflight: atoms must be ordered "
                        "as [L-electrode][bridge][R-electrode] in the "
                        "AtomicCoordinates block, with each region "
                        "contiguous (no gaps).  TranSIESTA reads "
                        "TS.NumUsedAtomsLeft as 'first N atoms = "
                        "left electrode'; out-of-order labels "
                        "produce SILENTLY WRONG transmission with "
                        "no run-time error.  Got: "
                        f"L-electrode={left_idx[0]}..{left_idx[-1]}, "
                        f"bridge={bridge_idx[0]}..{bridge_idx[-1]}, "
                        f"R-electrode={right_idx[0]}..{right_idx[-1]}.  "
                        "Re-export the structure from the Molbuilder "
                        "tab with atoms in contiguous L→bridge→R "
                        "order before re-running."
                    ),
                    where="struct.regions",
                ))

        # High-bias INFO: Landauer linear-response regime breaks
        # down above ~2 V for typical molecular junctions (di Ventra,
        # Electrical Transport in Nanoscale Systems, 2008).  Surface
        # so users interpret high-bias results as snapshots of a
        # nonlinear I-V, NOT linearized conductance.
        if cfg.bias_voltages_v:
            max_v = max(abs(v) for v in cfg.bias_voltages_v)
            if max_v > 2.0:
                issues.append(Issue(
                    severity="warn",
                    message=(
                        f"Bias voltage |V| = {max_v:.2f} V is above the "
                        f"~2 V linear-response limit for typical "
                        f"molecular junctions.  TranSIESTA will still "
                        f"converge but the result should be interpreted "
                        f"as a single point on a nonlinear I-V curve, "
                        f"NOT a linearized Landauer conductance.  "
                        f"Consult Reed et al. 2006 / di Ventra 2008 "
                        f"for nonlinear-regime interpretation guidance."
                    ),
                    where="config.bias_voltages_v",
                ))

        # Multi-bias warning (today's scope is zero-bias only).
        if len(cfg.bias_voltages_v) > 1:
            issues.append(Issue(
                severity="warn",
                message=(
                    f"Bias-scan generation is deferred to a follow-up "
                    f"release.  Today's engine emits a single .fdf at "
                    f"V = {cfg.bias_voltages_v[0]:.4f} V; subsequent "
                    f"values {cfg.bias_voltages_v[1:]} are ignored.  "
                    f"For now, regenerate the .fdf with each bias "
                    f"value separately (see module docstring under "
                    f"'Bias-scan workflow')."
                ),
                where="config.bias_voltages_v",
            ))

        # Cross-engine chemistry: shared open-shell-metal check.
        # TransportConfig doesn't carry spin_polarized today;
        # treat the run as closed-shell unless future config adds
        # spin handling.  The check returns [] when there's no
        # open-shell metal present, so it's harmless on organics.
        from ..validation import check_open_shell_metal
        issues.extend(check_open_shell_metal(
            struct,
            is_closed_shell=True,
            engine_label="TranSIESTA (this Transport calculation)",
        ))

        return issues

    @classmethod
    def parse_output(cls, path: str) -> TransportResults:
        """Read a ``<job>.transport.json`` produced by this engine.

        Deferred to a follow-up release alongside the schema design
        for ``.transport.json``.  Tracked in memory
        ``project_transport_results_tab_framework.md``.
        """
        raise NotImplementedError(
            "TranSIESTA parse_output is deferred to a follow-up release "
            "alongside the <job>.transport.json schema design + the "
            "/results inspector.  Today's engine ships render_script + "
            "preflight only.  See molbuilder/transport/transiesta.py "
            "module docstring for the deferred-scope plan."
        )

    @classmethod
    def methods_fragment(cls, cfg: TransportConfig,
                          results: TransportResults) -> str:
        """Engine-specific paragraph for the Methods section.

        Today returns a one-line placeholder.  Full version lands
        alongside parse_output (the manuscript prose interpolates
        actual run parameters from ``results``, which doesn't exist
        yet for TranSIESTA).
        """
        return (
            f"Transport calculations were performed with TranSIESTA "
            f"[CITE:transiesta_brandbyge_2002] from the SIESTA suite, "
            f"using a DZP basis, the PBE functional, and a "
            f"{cfg.siesta_mesh_cutoff_ry} Ry real-space mesh cutoff.  "
            f"NEGF density integration used a complex contour with "
            f"{cfg.contour_n_circle} imaginary-axis points "
            f"(Brandbyge et al. 2002 § IV).  "
            f"Electronic temperature was {cfg.electronic_temperature_k:.0f} K.  "
            f"(Full Methods paragraph deferred to the follow-up release "
            f"that lands parse_output + the .transport.json schema.)"
        )
