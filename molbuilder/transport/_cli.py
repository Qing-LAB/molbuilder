"""``molbuilder transport`` CLI group (TranSIESTA workflow helpers)."""

from __future__ import annotations

from pathlib import Path

import click

from .preflight import format_report, parse_fdf_params, preflight_files
from .wizard import DEFAULT_ELECTRODE_KZ, electrode_wizard, format_models


def _load_device(device_xyz, sidecar_path, cell_fdf):
    """Load a region-labeled device Structure from xyz + sidecar, and
    resolve its lattice (sidecar 'cell' if present, else a reference
    fdf's LatticeVectors via --cell-fdf).  Shared by electrode/bundle."""
    import numpy as np
    from pathlib import Path as _P
    from ..sidecars.molstruct import (apply_to_structure,
                                       load as load_sidecar, sidecar_path_for)
    from ..structure import Structure

    struct = Structure.from_xyz(device_xyz)
    sc = _P(sidecar_path) if sidecar_path else sidecar_path_for(device_xyz)
    if not _P(sc).exists():
        raise click.ClickException(
            f"no region sidecar found at {sc}; the device must carry "
            f"region labels (*-electrode). Pass --sidecar explicitly.")
    apply_to_structure(struct, load_sidecar(sc))

    # A --cell-fdf reference lattice overrides whatever the sidecar held
    # (lets the user point at an existing relaxed .fdf's hex cell).
    if cell_fdf:
        cell = parse_fdf_params(_P(cell_fdf).read_text()).cell_ang
        if not cell:
            raise click.ClickException(
                f"no LatticeVectors block found in {cell_fdf}")
        struct.cell = np.asarray(cell, dtype=float)
    if struct.cell is None:
        click.echo("WARNING: device has no lattice (no sidecar 'cell', no "
                   "--cell-fdf); the emitter will fabricate an orthorhombic "
                   "vacuum box (isolated-cluster model, NOT a periodic "
                   "surface). Supply --cell-fdf for a real Au(111) lead.",
                   err=True)
    return struct


_TRANSPORT_EPILOG = """\
\b
A conductance run is the transport COMPOSITE (plans/transport-design.md):
cite a finished junction relaxation, and prep derives everything else --
the sorted copy, both electrode cells, the seed, the device SCF and the
TBtrans transmission -- as one five-stage calculation in the tree:
\b
  molbuilder jobset init --calculation transport --shape hierarchical \\
      --bundle <project>/transport/<name> \\
      --slot junction=<project>/optimization/<calc>@<stage>/run-N \\
      --bias 0.0,0.2
  molbuilder jobset prep run seed          # then launch, stage by stage
  molbuilder jobset summarize run          # -> <label>.transport.json
\b
(The old `transport bundle` three-run driver retired with the composite,
2026-08-29 -- deriving and running the pieces IS the composite's job.)
\b
The helpers below stand alone -- an electrode cell from a labeled device,
and the device<->electrode consistency preflight:
\b
  molbuilder transport electrode --device dev.xyz --which L-electrode \\
      --cell-fdf relaxed.fdf --out-dir run/
  molbuilder transport preflight --device run/junc.fdf \\
      --electrode run/junc_L-electrode.fdf
"""


@click.group("transport",
             context_settings={"help_option_names": ["-h", "--help"]},
             epilog=_TRANSPORT_EPILOG)
def transport_group() -> None:
    """TranSIESTA transport-workflow helpers.

    A conductance run is three coupled calculations -- relax the junction,
    a separate bulk-electrode `.TSHS`, then the NEGF device run -- whose
    correctness hinges on the device and electrode sharing ONE numerical
    contract + a geometric clone + commensurate k.  These commands enforce
    that consistency (the actual failure mode).  Scientific basis:
    docs/engines/transport.md.

    Run `molbuilder transport COMMAND -h` for a worked example of each.
    """


@transport_group.command("preflight",
                         short_help="check device<->electrode .fdf "
                                    "consistency before a TranSIESTA run",
                         epilog="\b\nEXAMPLE:\n"
                                "  molbuilder transport preflight \\\n"
                                "      --device junc.fdf "
                                "--electrode junc_L-electrode.fdf\n"
                                "\nExits non-zero on any ERROR (mismatched "
                                "MeshCutoff/XC/basis, non-commensurate k,\n"
                                "device kz!=1, electrode kz=1, ...).")
@click.option("--device", "device", required=True,
              type=click.Path(exists=True, dir_okay=False, resolve_path=True),
              help="the NEGF device .fdf (SolutionMethod transiesta).")
@click.option("--electrode", "electrode", required=True,
              type=click.Path(exists=True, dir_okay=False, resolve_path=True),
              help="the bulk-lead .fdf that writes the .TSHS.")
@click.option("--min-electrode-thickness", type=float, default=12.0,
              show_default=True,
              help="warn if the electrode z-length (Ang) is below this "
                   "(principal-layer heuristic).")
@click.option("--electrode-kz-warn", type=int, default=20, show_default=True,
              help="warn if the electrode kz is below this (bulk lead needs "
                   "a dense, converged kz).")
def cmd_preflight(device: str, electrode: str, min_electrode_thickness: float,
                  electrode_kz_warn: int) -> None:
    """Validate the cross-run consistency contract between a device and an
    electrode `.fdf` (docs/engines/transport.md).

    Checks: commensurate transverse k, device ``kz=1`` + dense electrode
    ``kz``, identical XC / MeshCutoff / EnergyShift / basis, matching
    lateral cell, electrode thickness, device z-vacuum, and that the
    electrode writes its `.TSHS`.  Exits non-zero on any ERROR.
    """
    report = preflight_files(
        device, electrode,
        min_electrode_thickness_ang=min_electrode_thickness,
        electrode_kz_warn=electrode_kz_warn)
    click.echo(format_report(report))
    if not report.ok():
        raise SystemExit(1)


@transport_group.command("electrode",
                         short_help="derive bulk-lead .fdf(s) from a "
                                    "labeled device structure",
                         epilog="\b\nEXAMPLE:\n"
                                "  molbuilder transport electrode \\\n"
                                "      --device dev.xyz --which L-electrode \\\n"
                                "      --job-name junc --mesh-cutoff 400 "
                                "--kx 4 --ky 4 \\\n"
                                "      --cell-fdf relaxed.fdf --out-dir run/\n"
                                "\nClones the lead atoms + device lateral cell "
                                "+ contract, so the\nelectrode passes "
                                "`transport preflight` against the device by "
                                "construction.")
@click.option("--device", "device_xyz", required=True,
              type=click.Path(exists=True, dir_okay=False, resolve_path=True),
              help="device structure (.xyz); its .molstruct.json sidecar "
                   "(region labels) is auto-discovered alongside it.")
@click.option("--sidecar", "sidecar_path", default=None,
              type=click.Path(exists=True, dir_okay=False, resolve_path=True),
              help="explicit .molstruct.json (overrides auto-discovery).")
@click.option("--which", default="both", show_default=True,
              help="'both', or an electrode label/name (e.g. 'L-electrode' "
                   "or 'L').")
@click.option("--job-name", default="transport", show_default=True,
              help="device job name; electrode files are <job>_<label>.fdf "
                   "(matches the device's TS.Elec HS reference).")
@click.option("--mesh-cutoff", type=int, default=None,
              help="MeshCutoff (Ry) -- MUST match the device fdf "
                   "(default: TransportConfig default).")
@click.option("--kx", type=int, default=None,
              help="transverse kx -- MUST match the device.")
@click.option("--ky", type=int, default=None,
              help="transverse ky -- MUST match the device.")
@click.option("--electrode-kz", type=int, default=DEFAULT_ELECTRODE_KZ,
              show_default=True,
              help="dense kz for the periodic bulk lead (converge it, § 4.2).")
@click.option("--z-period", type=float, default=None,
              help="bulk repeat along transport (Å); override the "
                   "layer-spacing estimate.")
@click.option("--cell-fdf", "cell_fdf", default=None,
              type=click.Path(exists=True, dir_okay=False, resolve_path=True),
              help="reference .fdf to read the device lattice (LatticeVectors) "
                   "from; preserves a hexagonal Au(111) cell.")
@click.option("--out-dir", type=click.Path(file_okay=False), default=".",
              show_default=True, help="where to write the electrode .fdf(s).")
def cmd_electrode(device_xyz, sidecar_path, which, job_name, mesh_cutoff,
                  kx, ky, electrode_kz, z_period, cell_fdf, out_dir):
    """Derive the bulk-electrode `.fdf`(s) from a region-labeled device
    (docs/engines/transport.md).

    Clones the `*-electrode` region's exact atoms + the device lateral
    cell + the device numerical contract, so the device<->electrode
    invariants (§ 6.7) hold by construction.  Run `transport preflight`
    on the resulting pair to confirm.
    """
    from ..config.transport import TransportConfig

    struct = _load_device(device_xyz, sidecar_path, cell_fdf)

    cfg_kw = {"job_name": job_name}
    if mesh_cutoff is not None:
        cfg_kw["siesta_mesh_cutoff_ry"] = mesh_cutoff
    if kx is not None and ky is not None:
        cfg_kw["k_mesh_transverse"] = (kx, ky, 1)
    cfg = TransportConfig(**cfg_kw)

    try:
        models = electrode_wizard(
            struct, cfg, which=which, electrode_kz=electrode_kz,
            z_period=z_period)
    except ValueError as exc:
        raise click.ClickException(str(exc))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for job, fdf, _model in models:
        (out / f"{job}.fdf").write_text(fdf)
    click.echo(format_models(models))
    click.echo(f"\nwrote {len(models)} electrode .fdf(s) to {out}/")
    click.echo("Next: run 'molbuilder transport preflight --device <device.fdf> "
               "--electrode <this.fdf>' to confirm the contract.")


__all__ = ["transport_group"]
