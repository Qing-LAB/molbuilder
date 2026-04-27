"""molbuilder command-line entry point.

Subcommands:
    molbuilder peptide ARNDC --out file.xyz
    molbuilder dna ATGCATGC --out file.xyz
    molbuilder rna AUGCAUGCAU --out file.xyz
    molbuilder smiles "c1ccccc1" --out benzene.xyz
    molbuilder name "1,4-benzenedithiol" --out bdt.xyz
    molbuilder fdf in.xyz out.fdf --psml-lib /opt/psml --kgrid 4x4x1
    molbuilder serve --port 8000
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from .structure import Structure


# --------------------------------------------------------------------- #
#  Build subcommands (peptide / dna / rna / smiles / name)              #
# --------------------------------------------------------------------- #


def _emit(struct: Structure, args: argparse.Namespace) -> None:
    wrote_anything = False
    if args.out:
        struct.to_xyz(args.out)
        print(f"wrote {struct.n_atoms} atoms to {args.out}", file=sys.stderr)
        wrote_anything = True
    if args.pdb:
        struct.to_pdb(args.pdb)
        print(f"wrote {struct.n_atoms} atoms to {args.pdb}", file=sys.stderr)
        wrote_anything = True
    if args.pyscf:
        print(struct.to_pyscf(as_string=True))
        wrote_anything = True
    if not wrote_anything:
        sys.stdout.write(struct.to_xyz())
    print(struct.summary(), file=sys.stderr)


def _add_build_parser(sub, name: str, help_: str) -> argparse.ArgumentParser:
    s = sub.add_parser(name, help=help_, description=help_)
    s.add_argument("input", help="sequence / SMILES / name to build")
    s.add_argument("--out", help="write .xyz file to this path")
    s.add_argument("--pdb", help="write .pdb file to this path")
    s.add_argument("--pyscf", action="store_true",
                   help="print PySCF-format atom block to stdout")
    s.add_argument("--title", help="optional title")
    if name in ("dna", "rna"):
        s.add_argument("--backend", default="auto",
                       choices=["auto", "rdkit", "amber"],
                       help="builder backend (default: auto-detect)")
        s.add_argument("--form", default=None,
                       choices=["B", "A", "Z"],
                       help="helix form (B for DNA, A for RNA by default)")
        s.add_argument("--terminal", default="OH",
                       choices=["OH", "5P", "3P", "PP"],
                       help="terminal phosphate state")
        s.add_argument("--no-protonate-phosphates", action="store_true",
                       help="keep phosphates deprotonated (charge -1 each); "
                            "default is to add Hs so molecule is neutral")
    return s


# --------------------------------------------------------------------- #
#  fdf subcommand (XYZ -> SIESTA fdf)                                   #
# --------------------------------------------------------------------- #


def _parse_kgrid(s: str):
    cleaned = s.replace("x", " ").replace(",", " ")
    parts = cleaned.split()
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"k-grid must be 3 ints (e.g. '4x4x1'); got {s!r}"
        )
    try:
        return tuple(int(p) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def _add_fdf_parser(sub) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "fdf",
        help="convert an XYZ to a SIESTA .fdf input + copy psml files",
        description=(
            "Convert an XYZ file into a SIESTA .fdf input, optionally "
            "copying matching <Element>.psml files from a flat library."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("xyz", help="input XYZ file")
    p.add_argument("fdf", help="output .fdf file")

    g = p.add_argument_group("system")
    g.add_argument("--system-name",  default="siesta_run")
    g.add_argument("--system-label", default="siesta")

    g = p.add_argument_group("basis & grid")
    g.add_argument("--basis",            default="DZP")
    g.add_argument("--mesh-cutoff",      type=float, default=300.0)
    g.add_argument("--pao-energy-shift", type=float, default=0.02,
                   help="Ry; smaller = more diffuse PAOs / more accurate")

    g = p.add_argument_group("exchange-correlation")
    g.add_argument("--xc-functional", default="GGA")
    g.add_argument("--xc-authors",    default="PBE")

    g = p.add_argument_group("SCF")
    g.add_argument("--mixing-weight",       type=float, default=0.02)
    g.add_argument("--pulay-history",       type=int,   default=3)
    g.add_argument("--dm-tolerance",        type=float, default=1e-5)
    g.add_argument("--dm-energy-tolerance", type=float, default=1e-4,
                   help="redundant SCF energy guard (eV)")
    g.add_argument("--max-scf-iter",        type=int,   default=500)
    g.add_argument("--temperature",         type=float, default=300.0)
    g.add_argument("--solution-method",     default="diagon",
                   choices=["diagon", "OMM", "transiesta"],
                   help="diagon for most cases; OMM for very large systems")

    g = p.add_argument_group("k-points")
    g.add_argument("--kgrid", type=_parse_kgrid, default=(1, 1, 1),
                   help="Monkhorst-Pack mesh, e.g. '4x4x1'")

    g = p.add_argument_group("relaxation")
    g.add_argument("--relax",       default="CG")
    g.add_argument("--relax-steps", type=int, default=200)
    g.add_argument("--force-tol",   type=float, default=0.02)
    g.add_argument("--max-displ",   type=float, default=0.05)

    g = p.add_argument_group("output")
    g.add_argument("--no-write-forces",     action="store_true")
    g.add_argument("--no-write-coor-step",  action="store_true")
    g.add_argument("--no-write-coor-xmol",  action="store_true",
                   help="don't write a per-step .xyz")
    g.add_argument("--no-write-md-history", action="store_true",
                   help="don't write the .ANI trajectory")
    g.add_argument("--write-hs",            action="store_true",
                   help="write H+S matrices (TranSIESTA / DOS / transport)")
    g.add_argument("--no-use-save",         action="store_true",
                   help="disable DM/CG/XV continuation flags")

    g = p.add_argument_group("pseudopotentials")
    g.add_argument("--psml-lib", help="path to flat psml library")
    g.add_argument("--no-copy-psml", action="store_true")

    g = p.add_argument_group("misc")
    g.add_argument("--species-order",
                   help="comma-separated species order, e.g. 'C,H,S,Au'")
    g.add_argument("--cell-padding", type=float, default=15.0,
                   help="vacuum padding in Ang (auto-cell case)")
    g.add_argument("--no-wrap-into-cell", action="store_true",
                   help="don't fold atoms into [0, 1) fractional coords "
                        "even if a periodic cell is given")
    g.add_argument("--no-center-in-vacuum", action="store_true",
                   help="don't centre the molecule in the auto-vacuum cell")
    return p


def _run_fdf(args: argparse.Namespace) -> int:
    from .siesta import Config, convert
    cfg = Config(
        system_name=args.system_name,
        system_label=args.system_label,
        cell_padding=args.cell_padding,
        basis_size=args.basis,
        pao_energy_shift=args.pao_energy_shift,
        mesh_cutoff=args.mesh_cutoff,
        xc_functional=args.xc_functional,
        xc_authors=args.xc_authors,
        mixing_weight=args.mixing_weight,
        pulay_history=args.pulay_history,
        dm_tolerance=args.dm_tolerance,
        dm_energy_tolerance=args.dm_energy_tolerance,
        max_scf_iter=args.max_scf_iter,
        electronic_temperature=args.temperature,
        solution_method=args.solution_method,
        kgrid=args.kgrid,
        relax_type=args.relax,
        relax_steps=args.relax_steps,
        relax_force_tol=args.force_tol,
        relax_max_displ=args.max_displ,
        use_save_dm=not args.no_use_save,
        use_save_cg=not args.no_use_save,
        use_save_xv=not args.no_use_save,
        write_forces=not args.no_write_forces,
        write_coor_step=not args.no_write_coor_step,
        write_coor_xmol=not args.no_write_coor_xmol,
        write_md_history=not args.no_write_md_history,
        write_hs=args.write_hs,
        psml_lib=args.psml_lib,
        copy_psml=not args.no_copy_psml,
        wrap_into_cell=not args.no_wrap_into_cell,
        center_in_vacuum=not args.no_center_in_vacuum,
        species_order=(args.species_order.split(",")
                       if args.species_order else None),
    )
    summary = convert(args.xyz, args.fdf, cfg)
    print(f"Wrote {summary['fdf']}: {summary['n_atoms']} atoms, "
          f"{len(summary['species'])} species "
          f"({', '.join(summary['species'])})", file=sys.stderr)
    if summary["missing_psml"]:
        print(f"  ! missing pseudopotentials: "
              f"{', '.join(summary['missing_psml'])}", file=sys.stderr)
        return 2
    return 0


# --------------------------------------------------------------------- #
#  serve subcommand (Flask web UI)                                      #
# --------------------------------------------------------------------- #


def _add_serve_parser(sub) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "serve",
        help="run the browser UI (Flask + 3Dmol.js)",
        description=(
            "Start a Flask server that serves a one-page UI: enter a "
            "sequence / SMILES / name, see the 3-D structure in 3Dmol, "
            "and optionally generate a SIESTA .fdf input."
        ),
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--debug", action="store_true")
    return p


def _run_serve(args: argparse.Namespace) -> int:
    from .web.app import create_app
    app = create_app()
    print(f"molbuilder web UI starting at http://{args.host}:{args.port}",
          file=sys.stderr)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


# --------------------------------------------------------------------- #
#  Top-level dispatch                                                   #
# --------------------------------------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="molbuilder",
        description="Build a 3-D molecule from a sequence / SMILES / name "
                    "and turn it into SIESTA / PySCF / ASE input.",
    )
    sub = p.add_subparsers(dest="kind", required=True)

    _add_build_parser(sub, "peptide",
                      "build a polypeptide from sequence (1-letter + [SEP] etc)")
    _add_build_parser(sub, "dna",     "build ssDNA from sequence (B-form)")
    _add_build_parser(sub, "rna",     "build ssRNA from sequence (A-form)")
    _add_build_parser(sub, "smiles",  "build a molecule from SMILES (RDKit)")
    _add_build_parser(sub, "name",    "build a molecule from common/IUPAC name (PubChem)")

    _add_fdf_parser(sub)
    _add_serve_parser(sub)

    args = p.parse_args(argv)

    # Route -----------------------------------------------------------
    if args.kind in ("peptide", "dna", "rna", "smiles", "name"):
        from . import (
            build_dna, build_from_name, build_from_smiles,
            build_peptide, build_rna,
        )
        builders = {
            "peptide": build_peptide,
            "dna":     build_dna,
            "rna":     build_rna,
            "smiles":  build_from_smiles,
            "name":    build_from_name,
        }
        kwargs = {"title": args.title}
        if args.kind in ("dna", "rna"):
            kwargs["backend"]  = args.backend
            kwargs["terminal"] = args.terminal
            kwargs["protonate_phosphates"] = not args.no_protonate_phosphates
            if args.form is not None:
                kwargs["form"] = args.form
        struct = builders[args.kind](args.input, **kwargs)
        _emit(struct, args)
        return 0

    if args.kind == "fdf":
        return _run_fdf(args)

    if args.kind == "serve":
        return _run_serve(args)

    p.error(f"unknown subcommand {args.kind!r}")  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
