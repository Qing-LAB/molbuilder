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
from typing import Iterable, Iterator, Optional, Sequence

import click

from .structure import Structure


# --------------------------------------------------------------------- #
#  stdin support                                                        #
# --------------------------------------------------------------------- #


# --------------------------------------------------------------------- #
#  add_dataclass_options: dataclass field metadata -> click.option      #
# --------------------------------------------------------------------- #


def add_dataclass_options(cls, *,
                          prefix: str = "",
                          tier: Optional[str] = None,
                          skip: Iterable[str] = ()):
    """Decorator factory: convert a dataclass's fields into click.option
    decorators on the wrapped command function.

    Field-to-option mapping
    -----------------------
    * field name ``foo_bar`` -> ``--<prefix>foo-bar``
    * field metadata ``"help"`` -> click help text
    * default = field.default (or None for fields with default_factory
      that doesn't trivially serialise)
    * type:
        - ``bool``           -> ``is_flag=True``  (with --foo / --no-foo
                                pair if default is False / True)
        - ``int``            -> ``type=int``
        - ``float``          -> ``type=float``
        - ``str`` / Optional[str] -> ``type=str``
        - everything else    -> ``type=str`` (user gets to pass strings)

    The ``tier`` filter accepts only fields whose metadata["tier"]
    matches (or any field if metadata["tier"] is unset).  Default
    None -> include all.  ``skip`` is an iterable of field names to
    exclude (useful when the command already has those options
    defined manually).

    Returns a decorator that, when applied to a function, stacks
    @click.option for each kept field on it.  The wrapped function
    receives the field values as kwargs (same names as the dataclass
    fields).

    Example
    -------
    >>> @cli.command()
    ... @add_dataclass_options(SiestaConfig, skip=("psml_lib", "copy_psml"))
    ... def cmd_demo(**fields):
    ...     cfg = SiestaConfig(**fields)
    ...     ...

    Why this exists
    ---------------
    The fdf / pyscf subcommands today maintain ~50 click.option
    lines each that mirror SiestaConfig / PySCFConfig fields.  Every
    time a new field lands (e.g. gap #10 added diis_space + damp),
    the field has to be added in three places: the dataclass, the
    generator, and the CLI option list.  This helper is the path
    out of that maintenance tax: a future subcommand or a refactored
    cmd_fdf / cmd_pyscf reads field metadata directly.
    """
    import dataclasses
    import typing

    skip_set = set(skip)
    # `fld.type` may be a string when the dataclass module uses
    # `from __future__ import annotations`.  Resolve once via
    # get_type_hints so the field-by-field logic below sees real
    # types (bool / int / float / Optional[str] / ...) instead of
    # strings.  Fall back gracefully for runtime-only annotations
    # the resolver can't evaluate.
    try:
        resolved_hints = typing.get_type_hints(cls)
    except Exception:
        resolved_hints = {}

    def deco(f):
        for fld in dataclasses.fields(cls):
            if fld.name in skip_set:
                continue
            if tier is not None and fld.metadata.get("tier") != tier:
                continue

            flag = "--" + prefix + fld.name.replace("_", "-")
            help_text = fld.metadata.get("help") or fld.metadata.get("label") or ""

            ann = resolved_hints.get(fld.name, fld.type)
            # Walk Optional[X] / Union[X, None]
            origin = typing.get_origin(ann)
            args   = typing.get_args(ann)
            if origin is typing.Union and type(None) in args:
                inner = next((a for a in args if a is not type(None)), str)
                py_t  = inner
            else:
                py_t = ann

            # Default: the dataclass field default; MISSING -> None.
            default = (fld.default
                       if fld.default is not dataclasses.MISSING
                       else None)

            if py_t is bool:
                # Generate --foo / --no-foo pair so the user can flip
                # either direction regardless of the default.
                neg_flag = "--no-" + prefix + fld.name.replace("_", "-")
                f = click.option(f"{flag}/{neg_flag}",
                                 fld.name,
                                 default=bool(default),
                                 help=help_text)(f)
            elif py_t is int:
                f = click.option(flag, fld.name, type=int,
                                 default=default, show_default=True,
                                 help=help_text)(f)
            elif py_t is float:
                f = click.option(flag, fld.name, type=float,
                                 default=default, show_default=True,
                                 help=help_text)(f)
            else:
                # str / Optional[str] / unknown -- accept as a string
                # and let the dataclass __post_init__ / call site coerce.
                f = click.option(flag, fld.name, type=str,
                                 default=default, show_default=(default is not None),
                                 help=help_text)(f)
        return f
    return deco


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


@cli.command("peptide", short_help="build a polypeptide from sequence")
@_build_options(nucleic=False)
def cmd_peptide(sequence, out, pdb, pyscf_atom_block, title):
    """Build a polypeptide from a 1-letter sequence (with [SEP] etc)."""
    from molbuilder import build_peptide
    s = build_peptide(sequence, title=title)
    _emit(s, out=out, pdb=pdb, pyscf_atom_block=pyscf_atom_block)


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
    _emit(s, out=out, pdb=pdb, pyscf_atom_block=pyscf_atom_block)


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
    _emit(s, out=out, pdb=pdb, pyscf_atom_block=pyscf_atom_block)


@cli.command("smiles", short_help="build a molecule from SMILES (RDKit)")
@_build_options(nucleic=False)
def cmd_smiles(sequence, out, pdb, pyscf_atom_block, title):
    """Build a molecule from a SMILES string (needs rdkit)."""
    from molbuilder import build_from_smiles
    s = build_from_smiles(sequence, title=title)
    _emit(s, out=out, pdb=pdb, pyscf_atom_block=pyscf_atom_block)


@cli.command("name", short_help="build a molecule from common/IUPAC name (PubChem)")
@_build_options(nucleic=False)
def cmd_name(sequence, out, pdb, pyscf_atom_block, title):
    """Build a molecule from a common or IUPAC name (needs pubchempy)."""
    from molbuilder import build_from_name
    s = build_from_name(sequence, title=title)
    _emit(s, out=out, pdb=pdb, pyscf_atom_block=pyscf_atom_block)


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
# net charge override (matches `pyscf --charge`); None -> auto-detect
# from phosphate protonation state via formal_charge_from_phosphates.
@click.option("--net-charge",  type=int, default=None,
              help="net charge (default: auto-detect from phosphates; "
                   "set explicitly for non-DNA charged systems "
                   "such as carboxylates / amines / sulfonates)")
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
            relax, relax_steps, force_tol, max_displ, net_charge,
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
        net_charge=net_charge,
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
@click.option("--grid-level",     type=int, default=4, show_default=True,
              help="DFT integration grid (0-9; 4 = default for hybrids, 5 = tight)")
@click.option("--level-shift",    type=float, default=0.0, show_default=True,
              help="Hartree; 0.1-0.3 helps for hard SCF")
# Hard-SCF troubleshooting knobs (gap #10 / TIER 2 #16).  Both default
# to PySCF's own defaults so behaviour is unchanged for the easy-
# converge case; bump them when SCF oscillates.
@click.option("--diis-space", type=int, default=8, show_default=True,
              help="DIIS subspace size; bump to 12-20 for oscillating SCFs")
@click.option("--damp",       type=float, default=0.0, show_default=True,
              help="Roothaan damping factor; 0.3-0.5 helps when DIIS alone isn't enough")
# ECP (effective core potential).  Default None means auto -- emit
# lanl2dz when heavy atoms (Z>36) are present AND basis isn't def2-*.
# Pass an explicit name ("lanl2dz" / "stuttgart" / "def2") to force,
# or "" / "none" to disable auto-emit.  Per-element dicts aren't
# accessible from the CLI; use the Python API for that.
@click.option("--ecp", default=None,
              help="effective core potential (e.g. 'lanl2dz'); "
                   "default = auto for heavy atoms on non-def2 bases; "
                   "pass 'none' to disable auto-emit")
# pre-optimization
@click.option("--preopt", is_flag=True,
              help="run a cheap PBE/def2-SVP pre-opt before main run")
@click.option("--preopt-functional",  default="PBE",      show_default=True)
@click.option("--preopt-basis",       default="def2-SVP", show_default=True)
@click.option("--no-preopt-density-fit", is_flag=True,
              help="disable density fitting on the pre-opt SCF")
@click.option("--preopt-dispersion",  default=None,
              help="dispersion correction on the pre-opt mf "
                   "(d3 / d3bj / d4); default off")
@click.option("--preopt-max-steps",   type=int,   default=50,   show_default=True)
@click.option("--preopt-grms",        type=float, default=1e-3, show_default=True)
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
@click.option("--no-molwatch-log",    is_flag=True,
              help="don't write the additive <job>.molwatch.log "
                   "(self-contained per-step coords/energy/forces "
                   "log consumed by molwatch)")
@click.option("--no-save-initial-xyz",  is_flag=True,
              help="don't snapshot the input geometry to <job>_initial.xyz")
@click.option("--no-save-optimized-xyz", is_flag=True,
              help="don't write the relaxed geometry to <job>_optimized.xyz")
@click.option("--no-verbose-comments", is_flag=True,
              help="strip the inline tuning hints from the script")
def cmd_pyscf(input_path, py_path,
              job_name, charge, spin, symmetry,
              method, functional, basis, auxbasis, no_density_fit, dispersion,
              solvent, solvent_method,
              scf_conv_tol, scf_max_cycle, scf_init_guess,
              grid_level, level_shift, diis_space, damp, ecp,
              preopt, preopt_functional, preopt_basis,
              no_preopt_density_fit, preopt_dispersion,
              preopt_max_steps, preopt_grms,
              no_optimize, optimizer,
              geom_max_steps, geom_conv_energy, geom_conv_grms, geom_conv_gmax,
              max_memory, threads, verbose,
              no_chkfile, no_log_file, no_trajectory, no_molwatch_log,
              no_save_initial_xyz, no_save_optimized_xyz,
              no_verbose_comments):
    """Convert an XYZ or PDB structure into a runnable PySCF script."""
    from .pyscf import PySCFConfig, convert
    disp = None if (dispersion or "").lower() in ("none", "") else dispersion
    # ECP "" / "none" -> disabled (PySCFConfig.ecp == "" means
    # explicitly disabled; None means auto-detect).  Anything else
    # passes through verbatim.
    if ecp is None:
        ecp_val = None
    elif ecp.lower() in ("none", ""):
        ecp_val = ""
    else:
        ecp_val = ecp
    preopt_disp = (None if (preopt_dispersion or "").lower() in ("none", "")
                   else preopt_dispersion)
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
        ecp           = ecp_val,
        solvent       = solvent,
        solvent_method = solvent_method,
        scf_conv_tol  = scf_conv_tol,
        scf_max_cycle = scf_max_cycle,
        scf_init_guess = scf_init_guess,
        grid_level    = grid_level,
        level_shift   = level_shift,
        diis_space    = diis_space,
        damp          = damp,
        preopt        = preopt,
        preopt_functional = preopt_functional,
        preopt_basis  = preopt_basis,
        preopt_density_fit = not no_preopt_density_fit,
        preopt_dispersion  = preopt_disp,
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
        save_initial_xyz   = not no_save_initial_xyz,
        save_optimized_xyz = not no_save_optimized_xyz,
        write_trajectory = not no_trajectory,
        molwatch_log     = not no_molwatch_log,
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
#  validate subcommand (geometry + optional config preflight, JSON out) #
# --------------------------------------------------------------------- #


@cli.command("validate",
             short_help="run validation checks on a structure; print Issue JSON")
@click.argument("input_path", metavar="input")
@click.option("--engine", default=None,
              type=click.Choice(["siesta", "pyscf"]),
              help="run engine-specific config checks too (default: "
                   "structure-only geometry checks)")
@click.option("--exit-on-error", is_flag=True,
              help="exit 2 when any error-severity Issue is found "
                   "(useful in CI / shell preflight loops)")
@click.option("--pretty", is_flag=True,
              help="indent the JSON output (default is one-issue-per-line "
                   "compact form, easier to grep in shell pipelines)")
def cmd_validate(input_path, engine, exit_on_error, pretty):
    """Run molbuilder's validation suite on a structure file.

    Reads an XYZ or PDB (or `-` for stdin), runs the geometry checks
    (min atom distance, h_ratio, polymer orientation, image distance,
    cell volume) plus optional engine-specific config checks, and
    emits the resulting Issue list as JSON to stdout.

    Exit code with --exit-on-error: 0 if no errors, 2 if any error.
    Without the flag: always 0 (warnings don't stop the run).

    Pipeline-friendly:

        molbuilder dna ATGC | molbuilder validate -

        molbuilder validate run.xyz --engine siesta --exit-on-error \\
            && molbuilder fdf run.xyz run.fdf
    """
    import json
    from .validation import validate, validate_geometry

    with _resolve_input_path(input_path) as resolved:
        struct, _cell = _struct_for_validate(resolved)

    if engine == "siesta":
        from .config.siesta import SiestaConfig
        issues = validate(struct, SiestaConfig(), cell=_cell)
    elif engine == "pyscf":
        from .config.pyscf import PySCFConfig
        issues = validate(struct, PySCFConfig())
    else:
        # Default: geometry-only.  No config to validate against.
        issues = validate_geometry(struct, cell=_cell)

    payload = {
        "input": input_path,
        "engine": engine,
        "n_issues": len(issues),
        "n_errors": sum(1 for i in issues if i.severity == "error"),
        "n_warnings": sum(1 for i in issues if i.severity == "warn"),
        "issues": [
            {"severity": i.severity, "message": i.message, "where": i.where}
            for i in issues
        ],
    }
    if pretty:
        click.echo(json.dumps(payload, indent=2))
    else:
        click.echo(json.dumps(payload))

    if exit_on_error and payload["n_errors"] > 0:
        sys.exit(2)


def _struct_for_validate(path):
    """Read either XYZ or PDB; return (Structure, optional cell array).

    A small wrapper around the SIESTA-side _struct_from_file so the
    validate command supports the same extended-XYZ + PDB inputs as
    the fdf / pyscf pipelines.
    """
    from .siesta.input import _struct_from_file
    return _struct_from_file(path)


# --------------------------------------------------------------------- #
#  modify subcommand (XYZ/PDB structure editor; nanojunction builder)   #
# --------------------------------------------------------------------- #


def _parse_index_csv(values, flag):
    """``--delete 12,13 --delete 7`` -> ``[12, 13, 7]``.  Flag is the
    name used in error messages."""
    out: list = []
    for v in values or ():
        for tok in v.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                out.append(int(tok))
            except ValueError:
                raise click.BadParameter(
                    f"{flag} expects comma-separated integers; "
                    f"got {tok!r}"
                )
    return out


def _parse_xy_csv(value, flag):
    """``"0.5,-0.3"`` -> ``(0.5, -0.3)``."""
    parts = value.split(",")
    if len(parts) != 2:
        raise click.BadParameter(
            f"{flag} expects 'dx,dy' (two floats, comma-separated); got {value!r}"
        )
    try:
        return float(parts[0].strip()), float(parts[1].strip())
    except ValueError:
        raise click.BadParameter(
            f"{flag} entries must be floats; got {value!r}"
        )


def _parse_two_ints_csv(value, flag):
    """``"3,5"`` -> ``(3, 5)``."""
    parts = value.split(",")
    if len(parts) != 2:
        raise click.BadParameter(
            f"{flag} expects two comma-separated integers; got {value!r}"
        )
    try:
        return int(parts[0].strip()), int(parts[1].strip())
    except ValueError:
        raise click.BadParameter(
            f"{flag} entries must be integers; got {value!r}"
        )


def _parse_size3(size_str, flag):
    """``"3x3x2"`` -> ``(3, 3, 2)``."""
    parts = size_str.split("x")
    if len(parts) != 3:
        raise click.BadParameter(
            f"{flag}: size {size_str!r} must be 'MxNxL' (three integers separated by 'x')"
        )
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        raise click.BadParameter(
            f"{flag}: size {size_str!r} components must be integers"
        )


def _parse_electrode_spec(spec):
    """Parse one ``--electrode`` value into a kwargs dict.

    Two modes, distinguished by the ``@key=`` substring:

      * **Pair (default):** ``ELEM:PLANE:MxNxL@gap=GAP:ATOP,ABOT``.
        Two anchors after the trailing colon, comma-separated.
        ``GAP`` is the total electrode-to-electrode distance.
      * **Single (rare):** ``ELEM:PLANE:MxNxL@contact=DIST:+z=N`` or
        ``ELEM:PLANE:MxNxL@contact=DIST:-z=N``.  ``DIST`` is the
        anchor-to-closest-layer distance for the chosen side.

    Returns a dict with key ``"mode"`` set to ``"pair"`` or
    ``"single"`` and the appropriate fields.
    """
    main, sep, after_at = spec.partition("@")
    if not sep or not after_at:
        raise click.BadParameter(
            f"--electrode {spec!r}: missing '@gap=...' or '@contact=...' "
            f"section.  See `molbuilder modify --help` for the format."
        )
    main_parts = main.split(":")
    if len(main_parts) != 3:
        raise click.BadParameter(
            f"--electrode {spec!r}: expected ELEM:PLANE:MxNxL before '@'; "
            f"got {main_parts!r}"
        )
    element, plane, size_str = main_parts
    size = _parse_size3(size_str, f"--electrode {spec!r}")

    # The @-section is "key=val:rest".  Split on the first colon.
    keyval, has_colon, rest = after_at.partition(":")
    if not has_colon or not rest:
        raise click.BadParameter(
            f"--electrode {spec!r}: missing trailing anchor section after "
            f"'@{keyval}:'."
        )
    key, has_eq, val = keyval.partition("=")
    if not has_eq:
        raise click.BadParameter(
            f"--electrode {spec!r}: '@{keyval}' must be '@gap=NUM' or "
            f"'@contact=NUM' (key=value form)"
        )
    try:
        distance = float(val)
    except ValueError:
        raise click.BadParameter(
            f"--electrode {spec!r}: distance {val!r} after '@{key}=' must be a float"
        )

    if key == "gap":
        # Pair mode: trailing field is "ATOP,ABOT"
        if "," not in rest:
            raise click.BadParameter(
                f"--electrode {spec!r}: '@gap=' (pair mode) requires the "
                f"trailing field to be 'ATOP,ABOT' (two anchor indices); "
                f"got {rest!r}"
            )
        anchor_top, anchor_bot = _parse_two_ints_csv(
            rest, f"--electrode {spec!r}"
        )
        return {
            "mode": "pair",
            "element": element, "plane": plane, "size": size,
            "gap": distance,
            "anchor_top": anchor_top, "anchor_bot": anchor_bot,
        }
    if key == "contact":
        # Single mode: trailing field is "+z=N" or "-z=N"
        side, has_eq2, anchor_str = rest.partition("=")
        if not has_eq2 or side not in ("+z", "-z"):
            raise click.BadParameter(
                f"--electrode {spec!r}: '@contact=' (single mode) requires "
                f"the trailing field to be '+z=N' or '-z=N'; got {rest!r}"
            )
        try:
            anchor = int(anchor_str.strip())
        except ValueError:
            raise click.BadParameter(
                f"--electrode {spec!r}: anchor {anchor_str!r} must be an integer"
            )
        return {
            "mode": "single",
            "element": element, "plane": plane, "size": size,
            "contact_distance": distance,
            "side": side, "anchor": anchor,
        }
    raise click.BadParameter(
        f"--electrode {spec!r}: unknown key {key!r}; expected 'gap' (pair) "
        f"or 'contact' (single)"
    )


def _struct_to_text(struct, fmt):
    """Serialise a Structure to a string in the requested format.
    Used for stdout output (output_path == '-')."""
    import os as _os
    import tempfile as _tempfile
    suffix = ".pdb" if fmt == "pdb" else ".xyz"
    fd, path = _tempfile.mkstemp(suffix=suffix)
    _os.close(fd)
    try:
        if fmt == "pdb":
            struct.to_pdb(path)
        else:
            struct.to_xyz(path)
        with open(path) as f:
            return f.read()
    finally:
        _os.unlink(path)


def _infer_output_format(path):
    """Default output format from the file extension; xyz fallback."""
    p = str(path).lower()
    if p.endswith(".pdb"):
        return "pdb"
    return "xyz"


@cli.command("modify",
             short_help="edit a structure: one operation per call "
                        "(delete / orient / rotate / electrode)")
@click.argument("input_path",  metavar="input")
@click.argument("output_path", metavar="output")
# Operation flags -- exactly one TYPE must be present per call (delete,
# orient, rotate, or electrode).  Multiple instances of the same TYPE
# are allowed where geometrically meaningful (delete: flatten;
# electrode: apply each in order).
@click.option("--delete", multiple=True, metavar="INDICES",
              help="comma-separated atom indices to delete (0-based); "
                   "may be repeated, all entries flattened into one pass")
@click.option("--orient-axis", default=None, metavar="A0,A1",
              help="rotate so the vector from atom A0 to atom A1 forms "
                   "--angle (degrees, default 0) with --axis")
@click.option("--rotate", default=None, metavar="AXIS:ANGLE",
              help="rotate every atom around AXIS (x/y/z) by ANGLE "
                   "degrees, e.g. 'z:90'")
@click.option("--electrode", multiple=True,
              metavar="ELEM:PLANE:MxNxL@KEY=VAL:ANCHOR",
              help="add an FCC electrode.  PAIR (default): "
                   "'Au:111:3x3x2@gap=8.5:3,0' -- gap is electrode-to-"
                   "electrode distance, anchors are (top, bot).  SINGLE "
                   "(rare): 'Au:111:3x3x2@contact=2.4:+z=3' -- contact is "
                   "anchor-to-closest-layer distance.  Repeat for stepped "
                   "contacts; mixing pair and single is allowed.")
# Sub-options for --orient-axis
@click.option("--axis", default="z", show_default=True,
              type=click.Choice(["x", "y", "z"]),
              help="target axis for --orient-axis")
@click.option("--angle", type=float, default=0.0, show_default=True,
              help="tilt angle (degrees) between anchor pair vector and "
                   "--axis after orient.  Default 0 = exactly aligned. "
                   "Tilt happens in xz-plane for --axis z, xy-plane for "
                   "--axis x, yz-plane for --axis y.")
@click.option("--center", default="midpoint", show_default=True,
              type=click.Choice(["first", "midpoint", "none"]),
              help="how to translate the structure after rotation")
# Sub-options for --electrode (apply uniformly to every --electrode in
# the call; for asymmetric cases, use multiple `molbuilder modify`
# invocations through a stdin/stdout pipe)
@click.option("--orthogonal", is_flag=True,
              help="use ASE's orthogonal supercell (only meaningful for "
                   "fcc(111))")
@click.option("--electrode-offset", default="0,0", metavar="DX,DY",
              show_default=True,
              help="lateral (Δx, Δy) shift in Å applied to every "
                   "--electrode slab in this call")
@click.option("--lattice-constant", type=float, default=None,
              help="override the lattice constant (Å) for every "
                   "--electrode in this call; default uses the value "
                   "from molbuilder/data/fcc_lattice.json")
# Universal
@click.option("--output-format",
              type=click.Choice(["xyz", "pdb"]), default=None,
              help="output file format (default: infer from extension; "
                   "stdout always xyz unless explicitly set)")
def cmd_modify(input_path, output_path,
               delete, orient_axis, rotate, electrode,
               axis, angle, center,
               orthogonal, electrode_offset, lattice_constant,
               output_format):
    """Edit a structure: one operation TYPE per CLI call.  The operation
    types are mutually exclusive; chain calls via stdin/stdout pipes
    (`-` for input or output) for multi-step workflows.

    Operation types:

      --delete INDICES         drop atoms (multi-instance: flatten)
      --orient-axis A0,A1      rotate anchor pair onto --axis
      --rotate AXIS:ANGLE      spin every atom around AXIS
      --electrode SPEC         add an FCC electrode (multi-instance allowed)

    Examples -- canonical Au-bdt-Au junction in a 3-step pipe:

        # input: relaxed BDT geometry with 4 atoms (S-C-C-S)
        molbuilder modify bdt.xyz - --orient-axis 0,3 --center midpoint |
          molbuilder modify - junction.xyz \\
              --electrode Au:111:3x3x2@gap=9.0:3,0

    Stepped 3×3 + 4×4 contact, both sides, in one electrode call:

        molbuilder modify oriented.xyz junction.xyz \\
            --electrode Au:111:3x3x1@gap=9.0:3,0 \\
            --electrode Au:111:4x4x1@gap=14.0:3,0

    Asymmetric junction (Au top, Cu bottom) -- two single-mode calls:

        molbuilder modify oriented.xyz step1.xyz \\
            --electrode Au:111:3x3x2@contact=2.4:+z=3
        molbuilder modify step1.xyz junction.xyz \\
            --electrode Cu:111:3x3x2@contact=2.0:-z=0

    See docs/spec/modify-tab.md for the full per-(plane, orthogonal)
    constraint table; ASE's own error message bubbles up if the
    requested (m, n) doesn't satisfy the chosen cell shape.
    """
    from .modify import (
        add_electrode_slab, add_symmetric_electrodes,
        delete_atoms, orient_along_axis, rotate_around_axis,
    )

    # Operation-type mutex: exactly one of the four types per call.
    op_types = {
        "--delete":      bool(delete),
        "--orient-axis": orient_axis is not None,
        "--rotate":      rotate is not None,
        "--electrode":   bool(electrode),
    }
    given = [k for k, v in op_types.items() if v]
    if not given:
        raise click.UsageError(
            "exactly one of --delete, --orient-axis, --rotate, --electrode "
            "is required.  Run separate `molbuilder modify` invocations "
            "for multiple operation types (use '-' for stdin/stdout to chain)."
        )
    if len(given) > 1:
        raise click.UsageError(
            f"only one operation TYPE per call; got {given!r}.  "
            f"Within --delete and --electrode multiple instances are fine; "
            f"mixing TYPES requires separate calls (use '-' to chain)."
        )

    with _resolve_input_path(input_path) as resolved:
        struct, _cell = _struct_for_validate(resolved)
    n_in = struct.n_atoms

    try:
        if op_types["--delete"]:
            indices = _parse_index_csv(delete, "--delete")
            struct = delete_atoms(struct, indices)

        elif op_types["--orient-axis"]:
            anchors = _parse_two_ints_csv(orient_axis, "--orient-axis")
            struct = orient_along_axis(struct, anchors, axis=axis,
                                        angle=angle, center=center)

        elif op_types["--rotate"]:
            ax, _sep, ang_str = rotate.partition(":")
            if not _sep or ax not in ("x", "y", "z"):
                raise click.BadParameter(
                    f"--rotate must be 'AXIS:ANGLE' with AXIS in x/y/z; "
                    f"got {rotate!r}"
                )
            try:
                ang = float(ang_str)
            except ValueError:
                raise click.BadParameter(
                    f"--rotate angle {ang_str!r} must be a float"
                )
            struct = rotate_around_axis(struct, axis=ax, angle=ang)

        else:  # electrode
            offset_xy = _parse_xy_csv(electrode_offset, "--electrode-offset")
            for spec_str in electrode:
                spec = _parse_electrode_spec(spec_str)
                if spec["mode"] == "pair":
                    struct = add_symmetric_electrodes(
                        struct,
                        element=spec["element"],
                        plane=spec["plane"],
                        size=spec["size"],
                        anchor_indices=(spec["anchor_top"], spec["anchor_bot"]),
                        gap=spec["gap"],
                        orthogonal=orthogonal,
                        offset=offset_xy,
                        lattice_constant=lattice_constant,
                    )
                else:  # single
                    struct = add_electrode_slab(
                        struct,
                        element=spec["element"],
                        plane=spec["plane"],
                        size=spec["size"],
                        anchor_index=spec["anchor"],
                        contact_distance=spec["contact_distance"],
                        side=spec["side"],
                        orthogonal=orthogonal,
                        offset=offset_xy,
                        lattice_constant=lattice_constant,
                    )
    except (ValueError, IndexError) as exc:
        raise click.ClickException(str(exc)) from exc

    fmt = output_format or (
        "xyz" if str(output_path) == "-" else _infer_output_format(output_path)
    )
    if str(output_path) == "-":
        click.echo(_struct_to_text(struct, fmt), nl=False)
    else:
        if fmt == "pdb":
            struct.to_pdb(output_path)
        else:
            struct.to_xyz(output_path)
        click.echo(
            f"Wrote {output_path}: {struct.n_atoms} atoms (input had {n_in})",
            err=True,
        )


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


@cmd_watch.command("parse",
                   short_help="parse a trajectory file; print frame JSON")
@click.argument("input_path", metavar="input")
@click.option("--frames-only", is_flag=True,
              help="emit only the per-frame energy / max_force / wall_time "
                   "table; skip per-atom coordinates (smaller payload)")
@click.option("--pretty", is_flag=True,
              help="indent the JSON output (default is one-payload-per-line)")
def cmd_watch_parse(input_path, frames_only, pretty):
    """Parse a SIESTA / PySCF / .molwatch.log file and emit the
    Trajectory as JSON to stdout.  One-shot, parses to EOF then exits.

    Same parser the watch web UI uses internally; this is the
    shell-friendly surface of it (issue #81).  Pipeable:

        molbuilder watch parse run.molwatch.log | jq '.frames[-1]'
        molbuilder watch parse - < run.out --frames-only | grep error
    """
    import json
    from .parsers import detect_parser, trajectory_to_legacy_dict, UnknownFormatError

    with _resolve_input_path(input_path) as resolved:
        try:
            parser = detect_parser(resolved)
        except UnknownFormatError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(2)
        traj = parser.parse(resolved)

    payload = trajectory_to_legacy_dict(traj)
    if frames_only:
        # Drop the heavy per-atom arrays; keep the per-frame summary
        # (iteration index, energy, max_force, wall_time).  Useful for
        # piping a long trajectory into jq / grep without slurping
        # megabytes of coordinates.
        payload = {
            "source_format": payload["source_format"],
            "run_state":     payload["run_state"],
            "error_message": payload["error_message"],
            "iterations":    payload["iterations"],
            "energies":      payload["energies"],
            "max_forces":    payload["max_forces"],
            "wall_times":    payload["wall_times"],
        }
    click.echo(json.dumps(payload, indent=2 if pretty else None))


@cmd_watch.command("tail",
                   short_help="poll a growing log; emit one JSON line per new frame")
@click.argument("input_path", metavar="input")
@click.option("--poll-ms", type=int, default=1000, show_default=True,
              help="poll interval in milliseconds")
@click.option("--max-frames", type=int, default=None,
              help="exit after emitting this many new frames (for tests)")
def cmd_watch_tail(input_path, poll_ms, max_frames):
    """Poll a still-growing trajectory; emit one JSON line per new
    frame as it lands.  The watch web UI does the same on a 15s
    timer; this is the shell-line surface of it (issue #81).

    The output is newline-delimited JSON (NDJSON): each line is a
    self-contained JSON object describing one frame.  Pipeable:

        molbuilder watch tail run.molwatch.log | jq '.energy'
        molbuilder watch tail run.out | head -5

    Loop ends when the run finishes (run_state becomes 'finished'
    or 'error') or after --max-frames frames, whichever comes first.
    Ctrl-C also exits cleanly.
    """
    import json
    import time
    from .parsers import detect_parser, trajectory_to_legacy_dict, UnknownFormatError

    if input_path == "-":
        click.echo("Error: stdin not supported for `watch tail` "
                   "(needs a real file to poll)", err=True)
        sys.exit(2)

    last_n = 0
    last_state = "ongoing"
    emitted = 0
    poll_s = poll_ms / 1000.0
    try:
        while True:
            try:
                parser = detect_parser(input_path)
            except UnknownFormatError:
                # Tolerate transient empty-file states at the very start
                # of a run -- the writer may not have flushed enough
                # bytes for the format to be detectable yet.
                time.sleep(poll_s)
                continue
            try:
                traj = parser.parse(input_path)
            except Exception:
                time.sleep(poll_s)
                continue

            payload = trajectory_to_legacy_dict(traj)
            n = len(payload["frames"])
            for i in range(last_n, n):
                line = {
                    "step":       payload["iterations"][i],
                    "energy":     payload["energies"][i],
                    "max_force":  payload["max_forces"][i],
                    "wall_time":  payload["wall_times"][i],
                    "n_atoms":    len(payload["frames"][i]),
                }
                click.echo(json.dumps(line))
                emitted += 1
                if max_frames is not None and emitted >= max_frames:
                    return
            last_n = n
            last_state = payload["run_state"]
            if last_state in ("finished", "error"):
                return
            time.sleep(poll_s)
    except KeyboardInterrupt:
        return


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
    except click.ClickException as e:
        # Domain-level error raised by a subcommand (e.g. ASE rejecting
        # an electrode slab; an unsupported element).  Print the
        # message and exit with the exception's exit_code (default 1).
        click.echo(f"Error: {e.format_message()}", err=True)
        sys.exit(e.exit_code)
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
