"""``molbuilder transport`` CLI group (TranSIESTA workflow helpers)."""

from __future__ import annotations

from pathlib import Path

import click

from .preflight import format_report, parse_fdf_params, preflight_files


def _load_device(device_xyz, cell_fdf):
    """Load a region-labeled device Structure through the ONE reading door,
    and resolve its lattice (the pair's own cell, else a reference fdf's
    LatticeVectors via --cell-fdf).

    THE PAIR IS THE FILE (`model/structure.md` § 2.4).  This used to walk the
    sidecar by hand -- `sidecar_path_for` + `apply_to_structure` beside a bare
    `from_xyz` -- which is `StructureCodec.load` rewritten line for line, and
    it carried a `--sidecar` override so the two halves could be pointed at
    files that were never a pair.  Both went 2026-09-07.

    The refusal changed with them, and the new one is the question the command
    actually has.  "No sidecar file exists" is a fact about a directory; what
    this needs to know is whether the DEVICE carries a lead, which it can ask
    the structure directly -- and which is still the right answer when the
    labels arrived some other way.
    """
    import numpy as np
    from pathlib import Path as _P
    from ..config.transport import is_electrode_label
    from ..workingcopy_structure import StructureCodec

    struct = StructureCodec().load(device_xyz)
    if not any(is_electrode_label(label) for label in (struct.regions or {})):
        raise click.ClickException(
            f"{_P(device_xyz).name} carries no electrode region; the device "
            f"must have a lead labelled (e.g. 'L-electrode' / 'R-electrode'), "
            f"and those labels travel in the .molstruct.json beside it.")

    # A --cell-fdf reference lattice overrides whatever the pair held
    # (lets the user point at an existing relaxed .fdf's hex cell).
    if cell_fdf:
        cell = parse_fdf_params(_P(cell_fdf).read_text()).cell_ang
        if not cell:
            raise click.ClickException(
                f"no LatticeVectors block found in {cell_fdf}")
        struct.cell = np.asarray(cell, dtype=float)
    if struct.cell is None:
        click.echo("WARNING: device has no lattice (none in the pair, no "
                   "--cell-fdf); the emitter will fabricate an orthorhombic "
                   "vacuum box (isolated-cluster model, NOT a periodic "
                   "surface). Supply --cell-fdf for a real Au(111) lead.",
                   err=True)
    return struct


_TRANSPORT_EPILOG = """\
\b
A conductance run is the transport COMPOSITE (archive/2026-09-01-transport-design.md):
cite a finished junction relaxation, and prep derives everything else --
the sorted copy, both electrode cells, the seed, the device SCF and the
TBtrans transmission -- as one five-stage calculation in the tree:
\b
  molbuilder jobset init --calculation transport --shape hierarchical \\
      --bundle <project>/transport/<name> \\
      --slot junction=<project>/optimization/<calc>/<stage>/run-N \\
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


# `molbuilder transport electrode` DELETED 2026-09-17 -- a deck rendered from
# command-line flags, which is the shape this project retired.
#
# It took --mesh-cutoff / --kx / --ky / --electrode-kz / --z-period and wrote
# a finished electrode `.fdf` through `wizard.render_electrode_fdf`, a SECOND
# writer of a deck `transport/deck.py::_electrode_layout` already writes for
# the electrode_L / electrode_R rungs.  `molbuilder fdf` went on 2026-08-11
# for exactly this -- "exposing every engine field as a CLI flag is how a
# finished deck got rendered straight from the command line, skipping the
# description entirely" (`cli.py`) -- and this was the same command wearing
# the transport package's name.
#
# A deck is rendered by `jobset prep` from a description.  The electrodes are
# DERIVED from the cited junction's own labelled atoms at prep, by
# `compose.py` through `wizard.extract_electrode_model`, which survives
# because that is the live path.
