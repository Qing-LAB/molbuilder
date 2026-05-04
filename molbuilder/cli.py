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
    molbuilder watch serve --port 5000

The CLI is built on click (since Phase 5).  ``main(argv)`` is the
back-compat entry point used by ``project.scripts``; tests call it
directly with an explicit argv list.

Late imports inside each command body keep ``monkeypatch.setattr`` on
the public ``molbuilder.build_*`` symbols working in tests -- they
patch the package attribute, so we re-resolve at call time.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
from typing import Iterator, Optional, Sequence

import click

from .structure import Structure


# --------------------------------------------------------------------- #
#  stdin support                                                        #
# --------------------------------------------------------------------- #


@contextlib.contextmanager
def _resolve_input_path(path: str) -> Iterator[str]:
    """Yield a real file path the rest of the pipeline can ``.read()``.

    If ``path`` is the literal ``"-"`` (Unix stdin convention), drain
    stdin, sniff XYZ vs PDB from the first non-blank line, write to a
    temp file with the right extension, and yield the temp path.  The
    temp file is removed on context exit.

    Sniff rule:
      * first non-blank line is an integer (atom count) -> XYZ
      * anything else -> PDB
    Both sniff branches handle the realistic stdin sources -- a
    ``molbuilder dna ATGC`` upstream pipes XYZ; a hand-cat'd PDB
    starts with HEADER / TITLE / REMARK / ATOM / HETATM.
    """
    if path != "-":
        yield path
        return
    text = sys.stdin.read()
    first = ""
    for line in text.splitlines():
        if line.strip():
            first = line.strip()
            break
    ext = ".xyz" if first.isdigit() else ".pdb"
    with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False,
                                     prefix="molbuilder_stdin_") as f:
        f.write(text)
        tmp = f.name
    try:
        yield tmp
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# --------------------------------------------------------------------- #
#  Shared helpers                                                       #
# --------------------------------------------------------------------- #


def _emit(struct: Structure, *,
          out: Optional[str],
          pdb: Optional[str],
          pyscf_atom_block: bool) -> None:
    """Write the built Structure to whatever destinations the user asked
    for.  No destination at all -> dump XYZ to stdout (Unix-pipeable)."""
    wrote_anything = False
    if out:
        struct.to_xyz(out)
        click.echo(f"wrote {struct.n_atoms} atoms to {out}", err=True)
        wrote_anything = True
    if pdb:
        struct.to_pdb(pdb)
        click.echo(f"wrote {struct.n_atoms} atoms to {pdb}", err=True)
        wrote_anything = True
    if pyscf_atom_block:
        click.echo(struct.to_pyscf(as_string=True))
        wrote_anything = True
    if not wrote_anything:
        sys.stdout.write(struct.to_xyz())
    click.echo(struct.summary(), err=True)


class KGridParam(click.ParamType):
    """`--kgrid 4x4x1` / `4,4,1` / `4 4 1` -> tuple[int, int, int]."""
    name = "kgrid"

    def convert(self, value, param, ctx):
        if isinstance(value, tuple):
            return value
        cleaned = value.replace("x", " ").replace(",", " ")
        parts = cleaned.split()
        if len(parts) != 3:
            self.fail(
                f"k-grid must be 3 ints (e.g. '4x4x1'); got {value!r}",
                param, ctx,
            )
        try:
            return tuple(int(p) for p in parts)
        except ValueError as e:
            self.fail(str(e), param, ctx)


KGRID = KGridParam()


# --------------------------------------------------------------------- #
#  Top-level group                                                      #
# --------------------------------------------------------------------- #


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
)
def cli() -> None:
    """Build a 3-D molecule from a sequence / SMILES / name and turn
    it into SIESTA / PySCF / ASE input."""


# --------------------------------------------------------------------- #
#  Build subcommands (peptide / dna / rna / smiles / name)              #
# --------------------------------------------------------------------- #


def _build_options(*, nucleic: bool):
    """Decorator factory: shared --out / --pdb / --pyscf-atom-block / --title
    options across all builder subcommands; nucleic adds backend / form /
    terminal / no-protonate-phosphates."""
    def deco(f):
        # Order matters because click stacks decorators bottom-up; later
        # decorators land later in --help.  Apply common opts first
        # (so they appear at the top of --help).
        if nucleic:
            f = click.option("--no-protonate-phosphates", is_flag=True,
                             help="keep phosphates deprotonated (charge -1 each); "
                                  "default is to add Hs so molecule is neutral")(f)
            f = click.option("--terminal", default="OH", show_default=True,
                             type=click.Choice(["OH", "5P", "3P", "PP"]),
                             help="terminal phosphate state")(f)
            f = click.option("--form", default=None,
                             type=click.Choice(["B", "A", "Z"]),
                             help="helix form (B for DNA, A for RNA by default)")(f)
            f = click.option("--backend", default="auto", show_default=True,
                             type=click.Choice(["auto", "rdkit", "amber", "threedna"]),
                             help="builder backend (auto-order is "
                                  "threedna > amber > rdkit)")(f)
        f = click.option("--title", default=None, help="optional title")(f)
        f = click.option("--pyscf-atom-block", "--pyscf", "pyscf_atom_block",
                         is_flag=True,
                         help="print PySCF-format atom block to stdout")(f)
        f = click.option("--pdb", default=None, type=click.Path(),
                         help="write .pdb file to this path")(f)
        f = click.option("--out", default=None, type=click.Path(),
                         help="write .xyz file to this path")(f)
        f = click.argument("sequence")(f)
        return f
    return deco


def _emit_built(struct, out, pdb, pyscf_atom_block):
    _emit(struct, out=out, pdb=pdb, pyscf_atom_block=pyscf_atom_block)


@cli.command("peptide", short_help="build a polypeptide from sequence")
@_build_options(nucleic=False)
def cmd_peptide(sequence, out, pdb, pyscf_atom_block, title):
    """Build a polypeptide from a 1-letter sequence (with [SEP] etc)."""
    from molbuilder import build_peptide
    s = build_peptide(sequence, title=title)
    _emit_built(s, out, pdb, pyscf_atom_block)


@cli.command("dna", short_help="build ssDNA from sequence (B-form)")
@_build_options(nucleic=True)
def cmd_dna(sequence, out, pdb, pyscf_atom_block, title,
            backend, form, terminal, no_protonate_phosphates):
    """Build single-stranded DNA from a sequence."""
    from molbuilder import build_dna
    kwargs = dict(title=title, backend=backend, terminal=terminal,
                  protonate_phosphates=not no_protonate_phosphates)
    if form is not None:
        kwargs["form"] = form
    s = build_dna(sequence, **kwargs)
    _emit_built(s, out, pdb, pyscf_atom_block)


@cli.command("rna", short_help="build ssRNA from sequence (A-form)")
@_build_options(nucleic=True)
def cmd_rna(sequence, out, pdb, pyscf_atom_block, title,
            backend, form, terminal, no_protonate_phosphates):
    """Build single-stranded RNA from a sequence."""
    from molbuilder import build_rna
    kwargs = dict(title=title, backend=backend, terminal=terminal,
                  protonate_phosphates=not no_protonate_phosphates)
    if form is not None:
        kwargs["form"] = form
    s = build_rna(sequence, **kwargs)
    _emit_built(s, out, pdb, pyscf_atom_block)


@cli.command("smiles", short_help="build a molecule from SMILES (RDKit)")
@_build_options(nucleic=False)
def cmd_smiles(sequence, out, pdb, pyscf_atom_block, title):
    """Build a molecule from a SMILES string (needs rdkit)."""
    from molbuilder import build_from_smiles
    s = build_from_smiles(sequence, title=title)
    _emit_built(s, out, pdb, pyscf_atom_block)


@cli.command("name", short_help="build a molecule from common/IUPAC name (PubChem)")
@_build_options(nucleic=False)
def cmd_name(sequence, out, pdb, pyscf_atom_block, title):
    """Build a molecule from a common or IUPAC name (needs pubchempy)."""
    from molbuilder import build_from_name
    s = build_from_name(sequence, title=title)
    _emit_built(s, out, pdb, pyscf_atom_block)


# --------------------------------------------------------------------- #
#  fdf subcommand (XYZ -> SIESTA fdf)                                   #
# --------------------------------------------------------------------- #


@cli.command("fdf", short_help="convert XYZ / PDB to a SIESTA .fdf input")
@click.argument("input_path", metavar="input")
@click.argument("fdf_path",   metavar="fdf")
# system
@click.option("--system-name",  default="siesta_run", show_default=True)
@click.option("--system-label", default="siesta",     show_default=True)
# basis & grid
@click.option("--basis",            default="DZP", show_default=True)
@click.option("--mesh-cutoff",      type=float, default=300.0, show_default=True)
@click.option("--pao-energy-shift", type=float, default=0.01, show_default=True,
              help="Ry; smaller = more diffuse PAOs / more accurate")
# XC
@click.option("--xc-functional", default="GGA", show_default=True)
@click.option("--xc-authors",    default="PBE", show_default=True)
# SCF
@click.option("--spin-polarized", is_flag=True,
              help="open-shell DFT; required for radicals / "
                   "transition metals / triplet systems")
@click.option("--spin-total", type=float, default=None,
              help="target total spin moment (mu_B); only emitted "
                   "when --spin-polarized")
@click.option("--mixing-weight",       type=float, default=0.02, show_default=True)
@click.option("--pulay-history",       type=int,   default=3,    show_default=True)
@click.option("--dm-tolerance",        type=float, default=1e-5, show_default=True)
@click.option("--dm-energy-tolerance", type=float, default=1e-4, show_default=True,
              help="redundant SCF energy guard (eV)")
@click.option("--max-scf-iter",        type=int,   default=500,  show_default=True)
@click.option("--temperature",         type=float, default=300.0, show_default=True)
@click.option("--solution-method",     default="diagon", show_default=True,
              type=click.Choice(["diagon", "OMM", "transiesta"]))
# k-points
@click.option("--kgrid", type=KGRID, default=(1, 1, 1), show_default=True,
              help="Monkhorst-Pack mesh, e.g. '4x4x1'")
# relaxation
@click.option("--relax",       default="CG",  show_default=True)
@click.option("--relax-steps", type=int,   default=200, show_default=True)
@click.option("--force-tol",   type=float, default=0.02, show_default=True)
@click.option("--max-displ",   type=float, default=0.05, show_default=True)
# output
@click.option("--no-write-forces",     is_flag=True)
@click.option("--no-write-coor-step",  is_flag=True)
@click.option("--no-write-coor-xmol",  is_flag=True,
              help="don't write a per-step .xyz")
@click.option("--no-write-md-history", is_flag=True,
              help="don't write the .ANI trajectory")
@click.option("--write-hs",            is_flag=True,
              help="write H+S matrices (TranSIESTA / DOS / transport)")
@click.option("--no-use-save",         is_flag=True,
              help="disable DM/CG/XV continuation flags")
# pseudopotentials
@click.option("--psml-lib", default=None, type=click.Path(),
              help="path to flat psml library")
@click.option("--no-copy-psml", is_flag=True)
# misc
@click.option("--species-order", default=None,
              help="comma-separated species order, e.g. 'C,H,S,Au'")
@click.option("--cell-padding", type=float, default=15.0, show_default=True,
              help="vacuum padding in Ang (auto-cell case)")
@click.option("--no-wrap-into-cell", is_flag=True,
              help="don't fold atoms into [0, 1) fractional coords")
@click.option("--no-center-in-vacuum", is_flag=True,
              help="don't centre the molecule in the auto-vacuum cell")
def cmd_fdf(input_path, fdf_path,
            system_name, system_label,
            basis, mesh_cutoff, pao_energy_shift,
            xc_functional, xc_authors,
            spin_polarized, spin_total,
            mixing_weight, pulay_history, dm_tolerance, dm_energy_tolerance,
            max_scf_iter, temperature, solution_method,
            kgrid,
            relax, relax_steps, force_tol, max_displ,
            no_write_forces, no_write_coor_step, no_write_coor_xmol,
            no_write_md_history, write_hs, no_use_save,
            psml_lib, no_copy_psml,
            species_order, cell_padding,
            no_wrap_into_cell, no_center_in_vacuum):
    """Convert an XYZ or PDB structure into a SIESTA .fdf input."""
    from .siesta import SiestaConfig, convert
    cfg = SiestaConfig(
        system_name=system_name,
        system_label=system_label,
        cell_padding=cell_padding,
        basis_size=basis,
        pao_energy_shift=pao_energy_shift,
        mesh_cutoff=mesh_cutoff,
        xc_functional=xc_functional,
        xc_authors=xc_authors,
        mixing_weight=mixing_weight,
        pulay_history=pulay_history,
        dm_tolerance=dm_tolerance,
        dm_energy_tolerance=dm_energy_tolerance,
        max_scf_iter=max_scf_iter,
        electronic_temperature=temperature,
        solution_method=solution_method,
        kgrid=kgrid,
        relax_type=relax,
        relax_steps=relax_steps,
        relax_force_tol=force_tol,
        relax_max_displ=max_displ,
        use_save_dm=not no_use_save,
        use_save_cg=not no_use_save,
        use_save_xv=not no_use_save,
        write_forces=not no_write_forces,
        write_coor_step=not no_write_coor_step,
        write_coor_xmol=not no_write_coor_xmol,
        write_md_history=not no_write_md_history,
        write_hs=write_hs,
        psml_lib=psml_lib,
        copy_psml=not no_copy_psml,
        wrap_into_cell=not no_wrap_into_cell,
        center_in_vacuum=not no_center_in_vacuum,
        species_order=(species_order.split(",") if species_order else None),
        spin_polarized=spin_polarized,
        spin_total=spin_total,
    )
    with _resolve_input_path(input_path) as resolved_input:
        summary = convert(resolved_input, fdf_path, cfg)
    click.echo(
        f"Wrote {summary['fdf']}: {summary['n_atoms']} atoms, "
        f"{len(summary['species'])} species "
        f"({', '.join(summary['species'])})",
        err=True,
    )
    if summary["missing_psml"]:
        click.echo(
            f"  ! missing pseudopotentials: "
            f"{', '.join(summary['missing_psml'])}",
            err=True,
        )
        sys.exit(2)


# --------------------------------------------------------------------- #
#  pyscf subcommand (XYZ / PDB -> runnable PySCF script)                #
# --------------------------------------------------------------------- #


@cli.command("pyscf", short_help="convert XYZ / PDB to a runnable PySCF script")
@click.argument("input_path", metavar="input")
@click.argument("py_path",    metavar="py")
# system
@click.option("--job-name", default="pyscf_relax", show_default=True)
@click.option("--charge",   type=int, default=None,
              help="net charge (default: auto-detect from phosphates)")
@click.option("--spin",     type=int, default=0, show_default=True,
              help="2S, NOT 2S+1; 0 = closed shell, 1 = doublet")
@click.option("--symmetry", is_flag=True)
# method (production)
@click.option("--method",     default="RKS", show_default=True,
              type=click.Choice(["RKS", "UKS", "RHF", "UHF"]))
@click.option("--functional", default="B3LYP",   show_default=True)
@click.option("--basis",      default="def2-SVP", show_default=True)
@click.option("--auxbasis",   default=None)
@click.option("--no-density-fit", is_flag=True)
@click.option("--dispersion", default="d3bj", show_default=True,
              help="d3, d3bj, d4, or 'none' to disable")
# solvent
@click.option("--solvent",        default=None,
              help="water / methanol / dmso / chloroform / ...")
@click.option("--solvent-method", default="IEF-PCM", show_default=True)
# SCF
@click.option("--scf-conv-tol",   type=float, default=1e-9,  show_default=True)
@click.option("--scf-max-cycle",  type=int,   default=100,   show_default=True)
@click.option("--scf-init-guess", default="minao", show_default=True,
              type=click.Choice(["minao", "atom", "1e", "huckel"]))
@click.option("--grid-level",     type=int, default=3, show_default=True,
              help="DFT integration grid (0-9; 3 = default, 5 = tight)")
@click.option("--level-shift",    type=float, default=0.0, show_default=True,
              help="Hartree; 0.1-0.3 helps for hard SCF")
# pre-optimization
@click.option("--preopt", is_flag=True,
              help="run a cheap PBE/def2-SVP pre-opt before main run")
@click.option("--preopt-functional", default="PBE",      show_default=True)
@click.option("--preopt-basis",      default="def2-SVP", show_default=True)
@click.option("--preopt-max-steps",  type=int,   default=50,   show_default=True)
@click.option("--preopt-grms",       type=float, default=1e-3, show_default=True)
# main optimization
@click.option("--no-optimize", is_flag=True,
              help="single-point only, no geometry optimization")
@click.option("--optimizer", default="geometric", show_default=True,
              type=click.Choice(["geometric", "berny"]))
@click.option("--geom-max-steps",   type=int,   default=200,    show_default=True)
@click.option("--geom-conv-energy", type=float, default=1e-6,   show_default=True)
@click.option("--geom-conv-grms",   type=float, default=3e-4,   show_default=True)
@click.option("--geom-conv-gmax",   type=float, default=4.5e-4, show_default=True)
# runtime / output
@click.option("--max-memory", type=int, default=4000, show_default=True,
              help="MB hint for PySCF's max_memory")
@click.option("--threads",    type=int, default=None,
              help="OMP_NUM_THREADS pin; default = inherit env")
@click.option("--verbose",    type=int, default=4, show_default=True,
              help="0 silent, 4 info, 5 debug")
@click.option("--no-chkfile",         is_flag=True)
@click.option("--no-log-file",        is_flag=True)
@click.option("--no-trajectory",      is_flag=True,
              help="don't ask geomeTRIC to stream <job>_geom_optim.xyz")
@click.option("--no-verbose-comments", is_flag=True,
              help="strip the inline tuning hints from the script")
def cmd_pyscf(input_path, py_path,
              job_name, charge, spin, symmetry,
              method, functional, basis, auxbasis, no_density_fit, dispersion,
              solvent, solvent_method,
              scf_conv_tol, scf_max_cycle, scf_init_guess,
              grid_level, level_shift,
              preopt, preopt_functional, preopt_basis,
              preopt_max_steps, preopt_grms,
              no_optimize, optimizer,
              geom_max_steps, geom_conv_energy, geom_conv_grms, geom_conv_gmax,
              max_memory, threads, verbose,
              no_chkfile, no_log_file, no_trajectory, no_verbose_comments):
    """Convert an XYZ or PDB structure into a runnable PySCF script."""
    from .pyscf import PySCFConfig, convert
    disp = None if (dispersion or "").lower() in ("none", "") else dispersion
    cfg = PySCFConfig(
        job_name      = job_name,
        charge        = charge,
        spin          = spin,
        symmetry      = symmetry,
        method        = method,
        functional    = functional,
        basis         = basis,
        auxbasis      = auxbasis,
        density_fit   = not no_density_fit,
        dispersion    = disp,
        solvent       = solvent,
        solvent_method = solvent_method,
        scf_conv_tol  = scf_conv_tol,
        scf_max_cycle = scf_max_cycle,
        scf_init_guess = scf_init_guess,
        grid_level    = grid_level,
        level_shift   = level_shift,
        preopt        = preopt,
        preopt_functional = preopt_functional,
        preopt_basis  = preopt_basis,
        preopt_max_steps = preopt_max_steps,
        preopt_grms   = preopt_grms,
        optimize      = not no_optimize,
        optimizer     = optimizer,
        geom_max_steps = geom_max_steps,
        geom_conv_energy = geom_conv_energy,
        geom_conv_grms = geom_conv_grms,
        geom_conv_gmax = geom_conv_gmax,
        max_memory_mb = max_memory,
        threads       = threads,
        verbose       = verbose,
        chkfile          = not no_chkfile,
        log_file         = not no_log_file,
        write_trajectory = not no_trajectory,
        verbose_comments = not no_verbose_comments,
    )
    with _resolve_input_path(input_path) as resolved_input:
        summary = convert(resolved_input, py_path, cfg)
    click.echo(
        f"Wrote {summary['py']}: "
        f"{summary['n_atoms']} atoms, "
        f"charge={summary['charge']:+d}, "
        f"label={summary['label']!r}",
        err=True,
    )
    click.echo(f"Run with:  python {summary['py']}", err=True)


# --------------------------------------------------------------------- #
#  serve subcommand (Flask web UI)                                      #
# --------------------------------------------------------------------- #


@cli.command("serve", short_help="run the browser UI (Flask + 3Dmol.js)")
@click.option("--host",  default="127.0.0.1", show_default=True)
@click.option("--port",  type=int, default=8000, show_default=True)
@click.option("--debug", is_flag=True)
def cmd_serve(host, port, debug):
    """Start a Flask server with the molbuilder browser UI."""
    from .web.app import create_app
    app = create_app()
    click.echo(f"molbuilder web UI starting at http://{host}:{port}", err=True)
    app.run(host=host, port=port, debug=debug)


# --------------------------------------------------------------------- #
#  watch subcommand group (live trajectory viewer)                      #
# --------------------------------------------------------------------- #


@cli.group("watch", short_help="live trajectory viewer (Flask + 3Dmol.js)")
def cmd_watch():
    """Live trajectory viewer for SIESTA / PySCF / .molwatch.log."""


@cmd_watch.command("serve",
                   short_help="start the browser UI (build + watch tabs)")
@click.option("--host",  default="127.0.0.1", show_default=True)
@click.option("--port",  type=int, default=5000, show_default=True)
@click.option("--debug", is_flag=True)
def cmd_watch_serve(host, port, debug):
    """Start a Flask server hosting both the build page (/) and the
    watch page (/watch).  Reads any file the server can access -- a
    non-loopback --host binding emits a security warning."""
    from .web.app import create_app
    from .web.blueprints.watch import warn_if_remote
    warn_if_remote(host)
    app = create_app()
    click.echo(f"molbuilder web UI starting at http://{host}:{port}", err=True)
    click.echo(f"  build page:  http://{host}:{port}/",      err=True)
    click.echo(f"  watch page:  http://{host}:{port}/watch", err=True)
    app.run(host=host, port=port, debug=debug, threaded=True)


# --------------------------------------------------------------------- #
#  Entry points                                                         #
# --------------------------------------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Back-compat int-returning entry point.

    Kept for ``project.scripts`` and for tests that call
    ``cli.main([...])`` directly.

    The contract we need to preserve from the argparse era:
      * ``--help`` / ``-h``                 -> SystemExit(0)
      * missing / unknown args / commands   -> SystemExit(2)
      * normal command completion           -> return 0 (no SystemExit)

    Click in ``standalone_mode=True`` would sys.exit() on completion
    too (breaks the int-return contract); ``standalone_mode=False``
    swallows ``--help`` exits internally and returns 0 (breaks the
    SystemExit-on-help contract).  So we run in standalone_mode=False
    and post-condition the help case by hand: if argv contained a
    help flag, re-raise as SystemExit after click handled it.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    asked_for_help = "--help" in args or "-h" in args
    try:
        rc = cli.main(args=args, standalone_mode=False)
    except click.UsageError as e:
        # Missing required command, unknown subcommand, missing arg,
        # bad type conversion -- argparse would exit(2) on all of these.
        click.echo(f"Error: {e.format_message()}", err=True)
        sys.exit(2)
    except click.Abort:
        sys.exit(1)
    rc = rc or 0
    if asked_for_help:
        sys.exit(rc)
    return rc


def _run_watch_serve_entrypoint() -> int:
    """Console-script shim for the legacy ``molwatch`` entry point.

    Equivalent to ``molbuilder watch serve`` with the same default args.
    Kept for backwards compatibility with users / scripts that still
    invoke ``molwatch`` directly after the molbuilder + molwatch merge.
    """
    return main(["watch", "serve"] + sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
