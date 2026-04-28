"""molbuilder command-line entry point.

Subcommands:
    molbuilder peptide ARNDC --out file.xyz
    molbuilder dna ATGCATGC --out file.xyz
    molbuilder rna AUGCAUGCAU --out file.xyz
    molbuilder smiles "c1ccccc1" --out benzene.xyz
    molbuilder name "1,4-benzenedithiol" --out bdt.xyz
    molbuilder fdf   in.xyz out.fdf --psml-lib /opt/psml --kgrid 4x4x1
    molbuilder pyscf in.xyz out.py --functional B3LYP --preopt
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
    if args.pyscf_atom_block:
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
    # NB: --pyscf-atom-block emits just the atom list (the gto.M `atom=`
    # block).  The full runnable PySCF script is the `pyscf` subcommand:
    #   molbuilder pyscf in.xyz out.py
    s.add_argument("--pyscf-atom-block", "--pyscf", action="store_true",
                   dest="pyscf_atom_block",
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
        help="convert XYZ / PDB to a SIESTA .fdf input + copy psml files",
        description=(
            "Convert an XYZ or PDB structure file into a SIESTA .fdf "
            "input, optionally copying matching <Element>.psml files "
            "from a flat library.  Format is auto-detected from the "
            "input file's extension."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input",
                   metavar="input",
                   help="input structure file (.xyz or .pdb)")
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
    from .siesta import SiestaConfig, convert
    cfg = SiestaConfig(
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
    summary = convert(args.input, args.fdf, cfg)
    print(f"Wrote {summary['fdf']}: {summary['n_atoms']} atoms, "
          f"{len(summary['species'])} species "
          f"({', '.join(summary['species'])})", file=sys.stderr)
    if summary["missing_psml"]:
        print(f"  ! missing pseudopotentials: "
              f"{', '.join(summary['missing_psml'])}", file=sys.stderr)
        return 2
    return 0


# --------------------------------------------------------------------- #
#  pyscf subcommand (XYZ / PDB -> runnable PySCF script)                #
# --------------------------------------------------------------------- #


def _add_pyscf_parser(sub) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "pyscf",
        help="convert XYZ / PDB to a runnable PySCF script",
        description=(
            "Convert an XYZ or PDB structure file into a self-contained "
            "PySCF Python script that builds the molecule, runs SCF, "
            "and (by default) optimises the geometry.  Defaults reproduce "
            "a modern hybrid-DFT setup: B3LYP+D3BJ / def2-SVP with density "
            "fitting and the geomeTRIC optimizer."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="input structure file (.xyz or .pdb)")
    p.add_argument("py",    help="output .py script")

    g = p.add_argument_group("system")
    g.add_argument("--job-name",  default="pyscf_relax")
    g.add_argument("--charge",    type=int, default=None,
                   help="net charge (default: auto-detect from phosphates)")
    g.add_argument("--spin",      type=int, default=0,
                   help="2S, NOT 2S+1; 0 = closed shell, 1 = doublet, ...")
    g.add_argument("--symmetry",  action="store_true")

    g = p.add_argument_group("method (production stage)")
    g.add_argument("--method",     default="RKS",
                   choices=["RKS", "UKS", "RHF", "UHF"])
    g.add_argument("--functional", default="B3LYP")
    g.add_argument("--basis",      default="def2-SVP")
    g.add_argument("--auxbasis",   default=None)
    g.add_argument("--no-density-fit", action="store_true")
    g.add_argument("--dispersion", default="d3bj",
                   help="d3, d3bj, d4, or 'none' to disable")

    g = p.add_argument_group("solvent (optional PCM)")
    g.add_argument("--solvent",        default=None,
                   help="water / methanol / dmso / chloroform / ...")
    g.add_argument("--solvent-method", default="IEF-PCM")

    g = p.add_argument_group("SCF")
    g.add_argument("--scf-conv-tol",  type=float, default=1e-9)
    g.add_argument("--scf-max-cycle", type=int,   default=100)
    g.add_argument("--scf-init-guess", default="minao",
                   choices=["minao", "atom", "1e", "huckel"])
    g.add_argument("--grid-level",   type=int,   default=3,
                   help="DFT integration grid (0-9; 3 = default, 5 = tight)")
    g.add_argument("--level-shift",  type=float, default=0.0,
                   help="Hartree; 0.1-0.3 helps for hard SCF")

    g = p.add_argument_group("pre-optimization (optional cheap warm-up)")
    g.add_argument("--preopt", action="store_true",
                   help="run a cheap PBE/def2-SVP pre-opt before main run")
    g.add_argument("--preopt-functional", default="PBE")
    g.add_argument("--preopt-basis",      default="def2-SVP")
    g.add_argument("--preopt-max-steps",  type=int,   default=50)
    g.add_argument("--preopt-grms",       type=float, default=1e-3)

    g = p.add_argument_group("main optimization")
    g.add_argument("--no-optimize",  action="store_true",
                   help="single-point only, no geometry optimization")
    g.add_argument("--optimizer",    default="geometric",
                   choices=["geometric", "berny"])
    g.add_argument("--geom-max-steps",   type=int,   default=200)
    g.add_argument("--geom-conv-energy", type=float, default=1e-6)
    g.add_argument("--geom-conv-grms",   type=float, default=3e-4)
    g.add_argument("--geom-conv-gmax",   type=float, default=4.5e-4)

    g = p.add_argument_group("runtime / output")
    g.add_argument("--max-memory",  type=int, default=4000,
                   help="MB hint for PySCF's max_memory")
    g.add_argument("--threads",     type=int, default=None,
                   help="OMP_NUM_THREADS pin; default = inherit env")
    g.add_argument("--verbose",     type=int, default=4,
                   help="0 silent, 4 info, 5 debug")
    g.add_argument("--no-chkfile",  action="store_true")
    g.add_argument("--no-log-file", action="store_true")
    g.add_argument("--no-trajectory", action="store_true",
                   help="don't ask geomeTRIC to stream <job>_geom_optim.xyz "
                        "(disables the molwatch live-streaming source)")
    g.add_argument("--no-verbose-comments", action="store_true",
                   help="strip the inline tuning hints from the script")
    return p


def _run_pyscf(args: argparse.Namespace) -> int:
    from .pyscf_input import PySCFConfig, convert
    disp = None if (args.dispersion or "").lower() in ("none", "") else args.dispersion
    cfg = PySCFConfig(
        job_name      = args.job_name,
        charge        = args.charge,
        spin          = args.spin,
        symmetry      = args.symmetry,
        method        = args.method,
        functional    = args.functional,
        basis         = args.basis,
        auxbasis      = args.auxbasis,
        density_fit   = not args.no_density_fit,
        dispersion    = disp,
        solvent       = args.solvent,
        solvent_method = args.solvent_method,
        scf_conv_tol  = args.scf_conv_tol,
        scf_max_cycle = args.scf_max_cycle,
        scf_init_guess = args.scf_init_guess,
        grid_level    = args.grid_level,
        level_shift   = args.level_shift,
        preopt        = args.preopt,
        preopt_functional = args.preopt_functional,
        preopt_basis  = args.preopt_basis,
        preopt_max_steps = args.preopt_max_steps,
        preopt_grms   = args.preopt_grms,
        optimize      = not args.no_optimize,
        optimizer     = args.optimizer,
        geom_max_steps = args.geom_max_steps,
        geom_conv_energy = args.geom_conv_energy,
        geom_conv_grms = args.geom_conv_grms,
        geom_conv_gmax = args.geom_conv_gmax,
        max_memory_mb = args.max_memory,
        threads       = args.threads,
        verbose       = args.verbose,
        chkfile          = not args.no_chkfile,
        log_file         = not args.no_log_file,
        write_trajectory = not args.no_trajectory,
        verbose_comments = not args.no_verbose_comments,
    )
    summary = convert(args.input, args.py, cfg)
    print(f"Wrote {summary['py']}: "
          f"{summary['n_atoms']} atoms, "
          f"charge={summary['charge']:+d}, "
          f"label={summary['label']!r}",
          file=sys.stderr)
    print(f"Run with:  python {summary['py']}", file=sys.stderr)
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
    _add_pyscf_parser(sub)
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

    if args.kind == "pyscf":
        return _run_pyscf(args)

    if args.kind == "serve":
        return _run_serve(args)

    p.error(f"unknown subcommand {args.kind!r}")  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
