"""molbuilder command-line entry point.

Subcommands:
    molbuilder peptide ARNDC --out file.xyz
    molbuilder dna ATGCATGC --out file.xyz
    molbuilder rna AUGCAUGCAU --out file.xyz
    molbuilder smiles "c1ccccc1" --out benzene.xyz
    molbuilder name "1,4-benzenedithiol" --out bdt.xyz
    molbuilder fdf   in.xyz out.fdf --psml-lib /opt/psml --kgrid 4x4x1
    molbuilder pyscf in.xyz out.py --functional B3LYP
    molbuilder serve --port 8000
    molbuilder watch parse run.molwatch.log
    molbuilder watch tail run.molwatch.log

The CLI is built on click (since Phase 5).  ``main(argv)`` is the
back-compat entry point used by ``project.scripts``; tests call it
directly with an explicit argv list.

Late imports inside each command body keep ``monkeypatch.setattr`` on
the public ``molbuilder.build_*`` symbols working in tests -- they
patch the package attribute, so we re-resolve at call time.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import click

from .diagnostics import initialize as _initialize_diagnostics
from .envs._cli import envs_group
from .bench._cli import bench_group
from .transport._cli import transport_group
from .runtime_config import RuntimeConfigError, get_tls, read_config
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
            if fld.metadata.get("skip_cli"):
                # The dataclass marks this field as having a CLI handler
                # that's hand-rolled at the call site (custom parsing,
                # click.Path() type, etc.).  See SiestaConfig.kgrid /
                # psml_lib / species_order.
                continue
            if tier is not None and fld.metadata.get("tier") != tier:
                continue

            flag = "--" + prefix + fld.name.replace("_", "-")
            help_text = fld.metadata.get("help") or fld.metadata.get("label") or ""
            choices = fld.metadata.get("choices")

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

            # P1: enumerated values get a click.Choice so a typo fails at
            # CLI parse time instead of waiting for SIESTA / PySCF to
            # error out at execution time.  The choice list lives in the
            # dataclass field metadata so the dataclass stays the single
            # source of truth.  ``case_sensitive=False`` (R2) lets users
            # type ``--relax-type cg`` interchangeably with ``CG`` --
            # without it, the renderer's ``.upper()`` is dead code at
            # the CLI layer because click rejects mismatched case before
            # the renderer sees the value.
            if choices is not None:
                if py_t is bool:
                    raise TypeError(
                        f"{cls.__name__}.{fld.name}: 'choices' metadata is "
                        f"meaningless on a bool field"
                    )
                f = click.option(flag, fld.name,
                                 type=click.Choice(list(choices),
                                                   case_sensitive=False),
                                 default=default, show_default=True,
                                 help=help_text)(f)
                continue

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
            elif py_t is str:
                f = click.option(flag, fld.name, type=str,
                                 default=default,
                                 show_default=(default is not None),
                                 help=help_text)(f)
            else:
                # P3: bail loudly rather than silently coercing odd types
                # (Sequence[str], Tuple[int, int, int], dict-of-X, ...) to
                # str.  Author the field with skip_cli=True and hand-roll
                # the click.option at the call site, or add support for
                # the new type to this bridge.
                raise TypeError(
                    f"{cls.__name__}.{fld.name!r}: cannot auto-generate a "
                    f"CLI option for type {py_t!r}.  Mark the field with "
                    f"metadata={{'skip_cli': True}} and hand-roll a "
                    f"click.option at the call site, or extend "
                    f"add_dataclass_options to handle this type."
                )
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


# `molbuilder envs ...`  (doctor / install / list).  Recipe registry +
# doctor + install live under molbuilder/envs/; the CLI surface is a
# self-contained click group registered here as a single sub-group.
cli.add_command(envs_group)

# `molbuilder bench ...`  (siesta-gpu sweep over BENCH-MARKS).
# Reads the BENCH-MARKS block emitted by the generator (Step 2a) and
# runs short SIESTA jobs across a handful of (np, omp, BlockSize)
# combinations.  See molbuilder/bench/.
cli.add_command(bench_group)

# `molbuilder jobset ...`  (engine-agnostic staged execution: plan / prep /
# submit a bundle's job-set.json -- the SIESTA stage ladder, and later the
# bench sweep).  See molbuilder/jobset/ + docs/protocols/staged-execution.md.
from .jobset._cli import jobset_group
cli.add_command(jobset_group)

# `molbuilder transport ...`  (TranSIESTA workflow helpers; preflight =
# device<->electrode consistency gates).  See molbuilder/transport/.
cli.add_command(transport_group)


# `molbuilder pseudo ...`  (screen a pseudopotential set).
@click.group("pseudo", context_settings={"help_option_names": ["-h", "--help"]})
def pseudo_group() -> None:
    """Screen pseudopotential (.psml) sets before a run."""


@pseudo_group.command("check", short_help="screen a pseudopotential directory")
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--elements", default=None,
              help="comma-separated symbols to require (e.g. Au,C,H,S); "
                   "default: screen every .psml found in the directory.")
@click.option("--xc", "xc_authors", default=None,
              help="expected XC authors (e.g. PBE) -- flags pseudos that "
                   "don't match the calc's functional.")
@click.option("--relativistic", default="scalar", show_default=True,
              type=click.Choice(["scalar", "spin-orbit", "no"]),
              help="expected relativistic treatment.")
def cmd_pseudo_check(directory, elements, xc_authors, relativistic):
    """Screen a directory of .psml files: coverage, XC + relativistic
    match, dead Kleinman-Bylander projectors (ekb=0, a defective
    pseudo), and generator-version consistency across the set.

    Exits non-zero on any ERROR-severity issue -- missing pseudo,
    dead projector, or XC-family mismatch (the same ``ERROR_STATUSES``
    the SIESTA preflight blocks on) -- so it can gate a workflow.
    """
    from pathlib import Path as _P
    from molbuilder.pseudos import (scan_psml_directory, check_coverage,
                                    ERROR_STATUSES)

    d = _P(directory)
    if elements:
        els = [e.strip() for e in elements.split(",") if e.strip()]
    else:
        els = sorted(scan_psml_directory(d).keys())
        if not els:
            raise click.ClickException(f"no parseable .psml files in {d}")

    GGA = {"pbe", "pbesol", "blyp", "revpbe", "rpbe"}
    LDA = {"ca", "pz", "pw"}
    a = (xc_authors or "").lower()
    fam = "GGA" if a in GGA else "LDA" if a in LDA else None

    entries = check_coverage(els, d, expected_xc_family=fam,
                             expected_xc_authors=xc_authors or None,
                             expected_relativistic=relativistic)
    n_err = n_warn = 0
    for e in entries:
        if e.status == "ok":
            tag = "OK   "
        elif e.status in ERROR_STATUSES:
            tag = "ERROR"; n_err += 1
        else:
            tag = "WARN "; n_warn += 1
        click.echo(f"  [{tag}] {e.element:<8} {e.message}")
    click.echo(f"\n{len(entries)} checks: {n_err} error(s), {n_warn} warning(s).")
    if n_err:
        raise SystemExit(1)


cli.add_command(pseudo_group)


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


def _make_siesta_options_decorator():
    """Lazy-import SiestaConfig and apply ``add_dataclass_options``.

    Wrapped in a function so the decorator stack reads naturally and
    the import happens at command-build time (not at module import).
    """
    from .config.siesta import SiestaConfig
    return add_dataclass_options(SiestaConfig)


def _make_pyscf_options_decorator():
    """Same shape as :func:`_make_siesta_options_decorator` for PySCFConfig."""
    from .config.pyscf import PySCFConfig
    return add_dataclass_options(PySCFConfig)


@cli.command("fdf", short_help="convert XYZ / PDB to a SIESTA .fdf input")
@click.argument("input_path", metavar="input")
@click.argument("fdf_path",   metavar="fdf")
# Hand-rolled options for fields whose CLI handling needs custom parsing
# (SiestaConfig: kgrid / psml_lib / species_order have skip_cli=True so
# the bridge skips them; we wire them up explicitly here).  Everything
# else is auto-generated by add_dataclass_options(SiestaConfig).
@click.option("--kgrid", type=KGRID, default=(1, 1, 1), show_default=True,
              help="Monkhorst-Pack mesh, e.g. '4x4x1'")
@click.option("--psml-lib", default=None, type=click.Path(),
              help="path to flat psml library")
@click.option("--species-order", default=None,
              help="comma-separated species order, e.g. 'C,H,S,Au' "
                   "(default: auto from elements present)")
# --stage: minimum-viable per-stage overlay.  Picks tier-appropriate
# values for relax_type / relax_steps / relax_force_tol / relax_max_displ
# in one flag instead of remembering 4 separate --relax-* overrides.
# Applied AFTER the auto-generated --relax-* options so the user's
# explicit overrides ride through if they come after --stage in the
# command line -- Click parses left-to-right, all values land in
# ``fields`` first, then the stage overlay is applied last.
@click.option("--stage", type=click.Choice(["1", "2", "3"]), default=None,
              help="overlay per-stage tier defaults for relax_type / "
                   "relax_steps / relax_force_tol / relax_max_displ.  "
                   "Stage 1: CG warm-up (0.05 eV/A, 0.20 A); Stage 2: "
                   "Broyden publishable (0.04 eV/A, 0.05 A); Stage 3: "
                   "Broyden crystal-tight (0.01 eV/A, 0.02 A -- VASP "
                   "EDIFFG=-0.01 standard).  Anchored in docs/engines/"
                   "optimization-tuning.md sect. 2.3.1.  Mutually "
                   "exclusive with --stages-json / --stage-strategy "
                   "(those drive the multi-stage pipeline; --stage is "
                   "for a single-stage one-shot fdf).")
# --stages-json + --stage-strategy: power-user escape hatches for the
# multi-stage SIESTA pipeline (cfg.stages).  The everyday UI is the
# web form's stage-table widget; CLI users can paste a JSON payload
# or pick a named preset.  Applied in order: --stages-json replaces
# the entire ladder, then --stage-strategy overlays enable flags.
# When either flag is set, cmd_fdf switches into multi-stage mode and
# emits one ``<basename>_<stage>.fdf`` per enabled stage plus a
# ``<basename>.run.sh`` bash runner instead of the single one-shot fdf.
@click.option("--stages-json", "stages_json", default=None,
              metavar="JSON_OR_PATH",
              help="override the per-stage convergence ladder with a "
                   "JSON list-of-dicts (one entry per stage, keys = "
                   "SiestaStageSpec fields: name / enabled / relax_type / "
                   "relax_steps / relax_force_tol / relax_max_displ / "
                   "on_nonconvergence).  Accepts a literal JSON string "
                   "or a path to a .json file.  Unknown keys ignored.  "
                   "Applied BEFORE --stage-strategy so you can combine "
                   "them.  Sets multi-stage mode (one fdf per enabled "
                   "stage + a .run.sh runner).")
@click.option("--stage-strategy",
              type=click.Choice(["publishable", "loose-only", "vib-quality"]),
              default=None,
              help="override stage enable flags with a named preset: "
                   "'publishable' = stages 1+2 (default), 'loose-only' "
                   "= stage 1 only (cheap warm-up), 'vib-quality' = "
                   "1+2+3 (TIGHT tier for vib/IR/NEB Hessians).  "
                   "Mirrors the form's Stage strategy dropdown.  Sets "
                   "multi-stage mode (one fdf per enabled stage + a "
                   ".run.sh runner).")
# --jobset: also emit job-set.json so the bundle is runnable by the
# engine-agnostic framework (`molbuilder jobset prep/plan/submit`,
# staged-execution.md).  Opt-in (doesn't change the default file set);
# only meaningful in multi-stage mode.
@click.option("--jobset", "emit_jobset", is_flag=True, default=False,
              help="also write job-set.json (the ladder as a JobSet) so the "
                   "bundle runs via `molbuilder jobset prep/submit`.  "
                   "Requires --stage-strategy / --stages-json.")
@click.option("--stage-resources", "stage_resources", default=None,
              metavar="JSON_OR_PATH",
              help="per-stage scheduler resources for the job-set, as a JSON "
                   "object {stage_name: {domain?, time?, exclusive?, mem?, "
                   "gres?, mpi_np?, cpus_per_task?}} (literal or a .json "
                   "path).  This is HOW a ladder asks for a cheap warm-up + "
                   "an expensive final (staged-execution.md § 6).  Requires "
                   "--jobset; stages omitted here inherit the job-level config.")
# --vacuum: the structure's isolation padding (Å, per side).  Vacuum comes with
# the STRUCTURE, not the config (structure-periodicity.md) -- this is the CLI
# equivalent of the Modify -> Cell tab.  Needed for a flat/linear molecule loaded
# from a bare XYZ (no cell), which otherwise has vacuum 0 -> a degenerate cell.
@click.option("--vacuum", type=float, default=None, metavar="ANGSTROM",
              help="isolation vacuum (Å) per side on isolated axes; sets the "
                   "STRUCTURE's vacuum (CLI equivalent of Modify -> Cell).  "
                   "Required for a flat/linear molecule from a bare XYZ.")
@_make_siesta_options_decorator()
def cmd_fdf(input_path, fdf_path, kgrid, psml_lib, species_order, stage,
            stages_json, stage_strategy, emit_jobset, stage_resources,
            vacuum, **fields):
    """Convert an XYZ or PDB structure into a SIESTA .fdf input.

    Every SiestaConfig field is exposed as a CLI option (auto-generated
    by ``add_dataclass_options``).  Boolean fields generate a
    ``--foo / --no-foo`` pair; numeric and string fields take a value.
    See ``molbuilder/config/siesta.py`` for the authoritative parameter
    list and per-field help text.

    ``--stage {1,2,3}`` overlays tier-appropriate defaults for the
    relaxation algorithm + convergence thresholds.  Recommended workflow:
    ``--stage 1`` for an initial loose preopt, ``--stage 2`` for the
    publishable refine, ``--stage 3`` for a crystal-practical tight
    final stage.  Anchors the system-type-aware tier framework
    documented in ``docs/engines/optimization-tuning.md`` sect. 2.3.1.
    """
    from .siesta import SiestaConfig, convert
    from .config.siesta import apply_siesta_stage
    species_seq = species_order.split(",") if species_order else None
    cfg = SiestaConfig(
        kgrid=kgrid,
        psml_lib=psml_lib,
        species_order=species_seq,
        **fields,
    )

    # --stage (single-stage overlay) and --stages-json / --stage-strategy
    # (multi-stage pipeline) describe two different workflows; mixing
    # them is almost always user confusion (e.g. "I want stage 2 of a
    # 3-stage strategy" -- which means stage_strategy='vib-quality' +
    # picking the second entry, NOT --stage 2).  Reject the combination
    # loud rather than silently picking one.
    multi_stage = (stages_json is not None) or (stage_strategy is not None)
    if stage is not None and multi_stage:
        raise click.UsageError(
            "--stage is for a single-stage one-shot .fdf; "
            "--stages-json / --stage-strategy drive the multi-stage "
            "pipeline (one .fdf per enabled stage + a .run.sh runner).  "
            "Pick one path -- they're mutually exclusive."
        )
    if emit_jobset and not multi_stage:
        raise click.UsageError(
            "--jobset writes the job-set.json for a stage LADDER; it needs "
            "--stage-strategy or --stages-json (a single-stage one-shot .fdf "
            "is not a job-set)."
        )
    if stage_resources is not None and not emit_jobset:
        raise click.UsageError(
            "--stage-resources only applies to the job-set; pass --jobset "
            "(and --stage-strategy / --stages-json) too."
        )

    # Apply --stage overlay AFTER cfg is built so the user's per-knob
    # --relax-* overrides land in cfg first, then the stage values
    # overlay them.  Documented contract: stage wins on the 4
    # overlay knobs (relax_type / steps / force_tol / max_displ) AND
    # sets cfg.stage for the filename suffix + "Stage N" comment in
    # the emitted fdf; everything else (basis, mesh_cutoff, psml_lib,
    # ...) rides through.
    if stage is not None:
        import dataclasses as _dc
        cfg = apply_siesta_stage(cfg, int(stage))
        # cfg.stage drives the molwatch-log filename suffix and the
        # "# Stage N of a staged relaxation" header comment.  Setting
        # it here (in addition to the overlay) makes ``--stage N``
        # the one-flag way to produce a coherent stage-N fdf.
        cfg = _dc.replace(cfg, stage=int(stage))

    # Multi-stage branch: apply --stages-json + --stage-strategy to
    # cfg.stages, then emit one fdf per enabled stage + a bash runner.
    # cfg.system_label is force-aligned to ``Path(fdf_path).stem`` so
    # the per-stage filenames and the SystemLabel inside each fdf
    # never drift apart -- that's the contract the .XV auto-warmstart
    # relies on.
    if multi_stage:
        _emit_siesta_multi_stage(
            cfg=cfg,
            input_path=input_path,
            fdf_path=fdf_path,
            stages_json=stages_json,
            stage_strategy=stage_strategy,
            emit_jobset=emit_jobset,
            stage_resources=stage_resources,
        )
        return

    with _resolve_input_path(input_path) as resolved_input:
        summary = convert(resolved_input, fdf_path, cfg,
                          vacuum=((vacuum, vacuum, vacuum)
                                  if vacuum is not None else None))
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


def _parse_json_or_path(value: str, hint: str):
    """Parse a CLI value that is either literal JSON (starts with ``{``/``[``)
    or a path to a ``.json`` file.  Raises ``click.BadParameter`` with a clean
    message (never a stack trace) on bad JSON or a missing file."""
    import json as _json
    from pathlib import Path as _Path
    s = value.strip()
    if s[:1] in ("{", "["):
        try:
            return _json.loads(s)
        except _json.JSONDecodeError as e:
            raise click.BadParameter(
                f"{hint}: not valid JSON ({e.msg} at line {e.lineno}, "
                f"column {e.colno})", param_hint=hint)
    p = _Path(s)
    if not p.exists():
        raise click.BadParameter(f"{hint}: file not found: {p}",
                                 param_hint=hint)
    try:
        return _json.loads(p.read_text())
    except _json.JSONDecodeError as e:
        raise click.BadParameter(
            f"{hint}: not valid JSON in {p} ({e.msg} at line {e.lineno}, "
            f"column {e.colno})", param_hint=hint)


def _emit_siesta_multi_stage(*, cfg, input_path, fdf_path,
                              stages_json, stage_strategy,
                              emit_jobset=False, stage_resources=None):
    """Helper for cmd_fdf's multi-stage branch.

    Pulled out of cmd_fdf so the logic is unit-testable independent
    of Click's runner state, and so cmd_fdf's body stays readable.

    Side effects:
      * Writes ``{stem}_{stage}.fdf`` for each enabled stage.
      * Writes ``{stem}.run.sh`` (chmod +x) bash runner.
      * Optionally copies psml files + writes molwatch preview log
        next to the LAST enabled stage's fdf.
      * Prints a one-line summary per emitted file to stderr.
    """
    import dataclasses as _dc
    import json as _json
    import os as _os
    from pathlib import Path as _Path

    from .config.siesta import (
        apply_siesta_stage_strategy,
        siesta_stages_from_dicts,
    )
    from .siesta.input import _struct_from_file

    # --stages-json: replace cfg.stages wholesale.
    if stages_json is not None:
        s = stages_json.strip()
        if s.startswith("["):
            try:
                payload = _json.loads(s)
            except _json.JSONDecodeError as e:
                raise click.BadParameter(
                    f"--stages-json: not valid JSON ({e.msg} at "
                    f"line {e.lineno}, column {e.colno})",
                    param_hint="--stages-json",
                )
        else:
            p = _Path(s)
            if not p.exists():
                raise click.BadParameter(
                    f"--stages-json: file not found: {p}",
                    param_hint="--stages-json",
                )
            try:
                payload = _json.loads(p.read_text())
            except _json.JSONDecodeError as e:
                raise click.BadParameter(
                    f"--stages-json: not valid JSON in {p} "
                    f"({e.msg} at line {e.lineno}, column {e.colno})",
                    param_hint="--stages-json",
                )
        try:
            cfg.stages = siesta_stages_from_dicts(payload)
        except (TypeError, ValueError) as e:
            raise click.BadParameter(
                f"--stages-json: {e}", param_hint="--stages-json")

    # --stage-strategy: overlay enable flags onto cfg.stages.
    if stage_strategy is not None:
        cfg.stages = apply_siesta_stage_strategy(cfg.stages, stage_strategy)

    fdf_p = _Path(fdf_path)
    out_dir = fdf_p.parent if str(fdf_p.parent) else _Path(".")
    basename = fdf_p.stem
    # Force-align the SystemLabel to the on-disk filename stem.  This
    # is the single point where the multi-stage filename convention
    # (<stem>_<stage>.fdf) is wired to the SIESTA SystemLabel that
    # drives .XV / .DM auto-restart between stages.
    cfg = _dc.replace(cfg, system_label=basename)

    with _resolve_input_path(input_path) as resolved_input:
        struct, cell = _struct_from_file(resolved_input)

    out_dir.mkdir(parents=True, exist_ok=True)
    # Promotion A (staged-execution.md § 15.3): render the ladder's files via
    # the shared pure producer so the CLI and the web Build endpoint don't each
    # re-glue the sequence.  ``emit_jobset=False`` here -- the CLI builds its
    # own JobSet below from the pseudos actually present on disk (glob-fidelity
    # for legacy .psf/.vps), so the job-set.json stays byte-identical.
    from .siesta.stages import build_siesta_stage_bundle
    bundle = build_siesta_stage_bundle(struct, cfg, cell=cell,
                                       emit_jobset=False)
    fdfs = bundle.fdf_files
    runner = bundle.runner_text
    runner_path = out_dir / bundle.runner_name

    written: list = []
    for name, body in fdfs.items():
        p = out_dir / name
        p.write_text(body)
        written.append(p)
    runner_path.write_text(runner)
    _os.chmod(runner_path, 0o755)
    written.append(runner_path)

    # Optional psml copy, mirroring convert()'s behaviour.
    if cfg.psml_lib and cfg.copy_psml:
        from .siesta.input import copy_pseudopotentials, _detect_species
        from pathlib import Path as _P
        species = (list(cfg.species_order) if cfg.species_order
                   else _detect_species(struct.elements))
        lib = _P(cfg.psml_lib).expanduser()
        if lib.is_dir():
            copy_pseudopotentials(species, lib, out_dir)

    # --jobset: persist the ladder as a JobSet so `molbuilder jobset
    # prep/submit` can run this bundle (staged-execution.md § 5).  The Job
    # scripts are exactly the <label>_<stage>.fdf rendered above; ``shared``
    # is the pseudopotentials present in the bundle root (symlinked into each
    # stage dir at prep).  This is the host-side producer the framework was
    # missing -- it reuses stages_to_jobset + JobSet.write, no new logic.
    if emit_jobset:
        from .siesta.stages import stages_to_jobset
        from .jobset.model import Resources
        pseudos = sorted(p.name for ext in ("*.psml", "*.psf", "*.vps")
                         for p in out_dir.glob(ext))
        # --stage-resources: per-stage scheduler overrides (§ 6).  Validate
        # the stage names against the actual ladder so a typo is a loud error,
        # not a silently-ignored override.
        resources_for = None
        if stage_resources is not None:
            spec = _parse_json_or_path(stage_resources, "--stage-resources")
            if not isinstance(spec, dict):
                raise click.BadParameter(
                    "--stage-resources must be a JSON object "
                    "{stage_name: {resource fields}}",
                    param_hint="--stage-resources")
            valid = {s.name for s in cfg.stages if s.enabled}
            unknown = [k for k in spec if k not in valid]
            if unknown:
                raise click.BadParameter(
                    f"--stage-resources: unknown stage name(s) {unknown}; "
                    f"enabled stages are {sorted(valid)}",
                    param_hint="--stage-resources")
            # Validate each stage's body is an object with KNOWN resource
            # fields -- a typo'd field would otherwise be silently dropped by
            # Resources.from_dict (loud errors, not silent no-ops).
            import dataclasses as _dc
            res_fields = {f.name for f in _dc.fields(Resources)}
            for sname, body in spec.items():
                if not isinstance(body, dict):
                    raise click.BadParameter(
                        f"--stage-resources['{sname}'] must be an object of "
                        f"resource fields; got {type(body).__name__}",
                        param_hint="--stage-resources")
                bad = [k for k in body if k not in res_fields]
                if bad:
                    raise click.BadParameter(
                        f"--stage-resources['{sname}']: unknown field(s) "
                        f"{bad}; valid fields are {sorted(res_fields)}",
                        param_hint="--stage-resources")
            res_map = {k: Resources.from_dict(v) for k, v in spec.items()}
            resources_for = res_map.get
        try:
            js = stages_to_jobset(cfg, shared=pseudos,
                                  resources_for=resources_for)
        except ValueError as e:
            raise click.ClickException(f"--jobset: {e}")
        written.append(js.write(out_dir / "job-set.json"))

    # Per-stage molwatch preview log, one per enabled stage.  Each
    # log carries the stage's own convergence targets so the watch-
    # tab live plot draws the correct horizontal threshold for the
    # currently-running stage (per #542 / C1.4).
    if cfg.write_molwatch_log:
        from .trajectory_log import write_initial_preview
        from .siesta.input import _enabled_stages
        for stage in _enabled_stages(cfg):
            mw_path = out_dir / f"{basename}-{stage.name}.molwatch.log"
            write_initial_preview(
                struct, mw_path,
                job=basename,
                engine="siesta",
                stage_name=stage.name,
                convergence_targets={
                    "max_force_ev_per_ang": stage.relax_force_tol,
                    "max_steps":            stage.relax_steps,
                },
            )
            written.append(mw_path)

    click.echo(
        f"Wrote {len(fdfs)} stage fdf(s) + 1 runner to "
        f"{out_dir}: {', '.join(p.name for p in written)}",
        err=True,
    )
    click.echo(
        f"Run with: cd {out_dir} && ./{runner_path.name}",
        err=True,
    )


# --------------------------------------------------------------------- #
#  pyscf subcommand (XYZ / PDB -> runnable PySCF script)                #
# --------------------------------------------------------------------- #


@cli.command("pyscf", short_help="convert XYZ / PDB to a runnable PySCF script")
@click.argument("input_path", metavar="input")
@click.argument("py_path",    metavar="py")
# Hand-rolled --ecp because PySCFConfig.ecp is annotated str|dict|None
# and the dict variant is Python-API-only; the bridge has skip_cli=True
# on the field to avoid the union-type rejection in P3.
@click.option("--ecp", default=None,
              help="effective core potential (e.g. 'lanl2dz'); "
                   "default = auto for heavy atoms on non-def2 bases; "
                   "pass 'none' to disable auto-emit")
# --stages-json + --stage-strategy: power-user escape hatches for the
# staged-optimization ladder (cfg.stages).  The everyday UI is the
# web form's stage-table widget; CLI users can paste a JSON payload
# or pick a named preset.  Applied in order: --stages-json replaces
# the entire ladder, then --stage-strategy overlays enable flags.
@click.option("--stages-json", "stages_json", default=None,
              metavar="JSON_OR_PATH",
              help="override the per-stage convergence ladder with a "
                   "JSON list-of-dicts (one entry per stage, keys = "
                   "StageSpec fields).  Accepts either a literal JSON "
                   "string or a path to a .json file.  Unknown keys "
                   "are ignored.  Applied BEFORE --stage-strategy so "
                   "you can combine them (custom knobs + preset enable "
                   "flags).  Power-user escape hatch; the form's "
                   "stage-table is the everyday UI.")
@click.option("--stage-strategy",
              type=click.Choice(["publishable", "loose-only", "vib-quality"]),
              default=None,
              help="override stage enable flags with a named preset: "
                   "'publishable' = stages 1+2 (default), 'loose-only' "
                   "= stage 1 only (cheap warm-up), 'vib-quality' = "
                   "1+2+3 (TIGHT tier for vib/IR/NEB Hessians).  "
                   "Mirrors the form's Stage strategy dropdown.")
@_make_pyscf_options_decorator()
def cmd_pyscf(input_path, py_path, ecp, stages_json, stage_strategy, **fields):
    """Convert an XYZ or PDB structure into a runnable PySCF script.

    Every PySCFConfig field is exposed as a CLI option (auto-generated
    by ``add_dataclass_options``).  Boolean fields generate a
    ``--foo / --no-foo`` pair; numeric and string fields take a value.
    See ``molbuilder/config/pyscf.py`` for the authoritative parameter
    list and per-field help text.

    Two minor coercions on top of the bridge: ``--dispersion`` accepts
    the literal ``none`` (case-insensitive) or empty string as a way to
    spell ``None`` from the shell; ``--ecp`` does the same with an
    additional state where ``""`` means "explicitly disable auto-emit".
    """
    from .pyscf import PySCFConfig, convert

    def _none_if_empty(s):
        if s is None:
            return None
        return None if s.strip().lower() in ("", "none") else s
    fields["dispersion"] = _none_if_empty(fields.get("dispersion"))
    if ecp is not None:
        ecp_val = ecp.strip().lower()
        fields["ecp"] = "" if ecp_val in ("", "none") else ecp
    else:
        fields["ecp"] = None

    cfg = PySCFConfig(**fields)

    # Apply stage overrides (7c).  --stages-json wins on knob values;
    # --stage-strategy then overlays enable flags on top.  Either may
    # be omitted; both omitted leaves cfg.stages at its default.
    if stages_json is not None:
        import json as _json
        from pathlib import Path as _Path
        from .config.pyscf import stages_from_dicts
        s = stages_json.strip()
        # Heuristic: starts with '[' = literal JSON; otherwise treat
        # as a path.  Keeps the common "paste a JSON literal on the
        # shell" case clean without forcing a wrapper sentinel.
        if s.startswith("["):
            try:
                payload = _json.loads(s)
            except _json.JSONDecodeError as e:
                raise click.BadParameter(
                    f"--stages-json: not valid JSON ({e.msg} at "
                    f"line {e.lineno}, column {e.colno})",
                    param_hint="--stages-json",
                )
        else:
            p = _Path(s)
            if not p.exists():
                raise click.BadParameter(
                    f"--stages-json: file not found: {p}",
                    param_hint="--stages-json",
                )
            try:
                payload = _json.loads(p.read_text())
            except _json.JSONDecodeError as e:
                raise click.BadParameter(
                    f"--stages-json: not valid JSON in {p} "
                    f"({e.msg} at line {e.lineno}, column {e.colno})",
                    param_hint="--stages-json",
                )
        try:
            cfg.stages = stages_from_dicts(payload)
        except (TypeError, ValueError) as e:
            raise click.BadParameter(
                f"--stages-json: {e}", param_hint="--stages-json")

    if stage_strategy is not None:
        from .config.pyscf import apply_stage_strategy
        cfg.stages = apply_stage_strategy(cfg.stages, stage_strategy)

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


def _parse_ints_csv(value, flag):
    """``"3,0,7"`` -> ``[3, 0, 7]`` (one or more integers).  Used for the
    electrode centre-index list (the atoms whose centroid the slab centres
    on).  Rejects an empty list -- callers who want origin-centring omit the
    field entirely at the API level."""
    parts = [p.strip() for p in value.split(",") if p.strip() != ""]
    if not parts:
        raise click.BadParameter(
            f"{flag} expects one or more comma-separated atom indices; "
            f"got {value!r}"
        )
    try:
        return [int(p) for p in parts]
    except ValueError:
        raise click.BadParameter(
            f"{flag} entries must be integers; got {value!r}"
        )


def _parse_size3(size_str, flag):
    """``"3x3x2"`` -> ``(3, 3, 2)``.  Tolerates whitespace around the
    individual integer fields."""
    parts = size_str.split("x")
    if len(parts) != 3:
        raise click.BadParameter(
            f"{flag}: size {size_str!r} must be 'MxNxL' (three integers separated by 'x')"
        )
    try:
        return tuple(int(p.strip()) for p in parts)
    except ValueError:
        raise click.BadParameter(
            f"{flag}: size {size_str!r} components must be integers"
        )


def _parse_electrode_spec(spec):
    """Parse one ``--electrode`` value into a kwargs dict.

    Two modes, distinguished by the ``@key=`` substring:

      * **Pair (default):** ``ELEM:PLANE:MxNxL@gap=GAP:I,J,...``.
        One or more centre indices after the trailing colon,
        comma-separated -- the junction centres on their CENTROID (1
        index -> that atom, 2 -> their midpoint, N -> centroid).
        ``GAP`` is the total electrode-to-electrode distance.
      * **Single (rare):** ``ELEM:PLANE:MxNxL@contact=DIST:+z=I,J,...``
        or ``...:-z=I,J,...``.  ``DIST`` is the centre-to-closest-layer
        distance for the chosen side; the slab centres on the same
        centroid of the trailing index list.

    Returns a dict with key ``"mode"`` set to ``"pair"`` or
    ``"single"`` and the appropriate fields.
    """
    main, sep, after_at = spec.partition("@")
    if not sep or not after_at:
        raise click.BadParameter(
            f"--electrode {spec!r}: missing '@gap=...' or '@contact=...' "
            f"section.  See `molbuilder modify --help` for the format."
        )
    # Strip whitespace on every parsed field so e.g. ``+z = 3`` (with
    # spaces) and ``Au : 111 : 3x3x2`` are accepted gracefully.
    main_parts = [p.strip() for p in main.split(":")]
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
            f"--electrode {spec!r}: missing trailing centre-index section "
            f"after '@{keyval}:'."
        )
    key, has_eq, val = keyval.partition("=")
    key = key.strip().lower()                   # R3: case-insensitive
    val = val.strip()
    rest = rest.strip()
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
        # Pair mode: trailing field is a centre-index list "I,J,..."
        center_indices = _parse_ints_csv(rest, f"--electrode {spec!r}")
        return {
            "mode": "pair",
            "element": element, "plane": plane, "size": size,
            "gap": distance,
            "center_indices": center_indices,
        }
    if key == "contact":
        # Single mode: trailing field is "+z=I,J,..." or "-z=I,J,..."
        side, has_eq2, idx_str = rest.partition("=")
        side = side.strip()
        idx_str = idx_str.strip()
        if not has_eq2 or side not in ("+z", "-z"):
            raise click.BadParameter(
                f"--electrode {spec!r}: '@contact=' (single mode) requires "
                f"the trailing field to be '+z=I,J,...' or '-z=I,J,...'; "
                f"got {rest!r}"
            )
        center_indices = _parse_ints_csv(idx_str, f"--electrode {spec!r}")
        return {
            "mode": "single",
            "element": element, "plane": plane, "size": size,
            "contact_distance": distance,
            "side": side, "center_indices": center_indices,
        }
    raise click.BadParameter(
        f"--electrode {spec!r}: unknown key {key!r}; expected 'gap' (pair) "
        f"or 'contact' (single)"
    )


def _struct_to_text(struct, fmt):
    """Serialise a Structure to a string in the requested format.
    Used for stdout output (output_path == '-').  Both ``Structure.to_xyz``
    and ``Structure.to_pdb`` return the formatted text directly when
    called without a path; we just pick the right one."""
    return struct.to_pdb() if fmt == "pdb" else struct.to_xyz()


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
@click.option("--rotate", multiple=True, metavar="AXIS:ANGLE",
              help="rotate every atom around AXIS (x/y/z) by ANGLE "
                   "degrees, e.g. 'z:90'.  Single-instance per call: "
                   "passing --rotate twice is rejected.")
@click.option("--electrode", multiple=True,
              metavar="ELEM:PLANE:MxNxL@KEY=VAL:CENTER_INDICES",
              help="add an FCC electrode, centred on the CENTROID of the "
                   "trailing atom-index list (1 index -> that atom, 2 -> "
                   "their midpoint, N -> centroid).  PAIR (default): "
                   "'Au:111:3x3x2@gap=8.5:3,0' -- gap is electrode-to-"
                   "electrode distance.  SINGLE (rare): "
                   "'Au:111:3x3x2@contact=2.4:+z=3' -- contact is the "
                   "centre-to-closest-layer distance.  Repeat for stepped "
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

    See docs/tabs/molbuilder.md for the full per-(plane, orthogonal)
    constraint table; ASE's own error message bubbles up if the
    requested (m, n) doesn't satisfy the chosen cell shape.
    """
    from .modify import (
        add_electrode_slab, add_symmetric_electrodes,
        delete_atoms, orient_along_axis, rotate_around_axis,
    )

    # Reject duplicate --rotate (single-instance flag despite multiple=True
    # which is only used to detect repeats).
    if len(rotate) > 1:
        raise click.UsageError(
            f"--rotate is single-instance per call; got {len(rotate)} "
            f"values: {list(rotate)!r}.  Apply rotations one at a time "
            f"and chain via stdin/stdout pipes."
        )
    rotate_value = rotate[0] if rotate else None

    # Operation-type mutex: exactly one of the four types per call.
    op_types = {
        "--delete":      bool(delete),
        "--orient-axis": orient_axis is not None,
        "--rotate":      rotate_value is not None,
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

    # Sub-option warnings: catch "ignored sub-option" cases up front so the
    # user notices before they expect them to take effect.
    _ORIENT_DEFAULTS = {"axis": "z", "angle": 0.0, "center": "midpoint"}
    if not op_types["--orient-axis"]:
        for name, default in _ORIENT_DEFAULTS.items():
            if locals()[name] != default:
                click.echo(
                    f"warning: --{name} is a sub-option of --orient-axis; "
                    f"value {locals()[name]!r} is ignored without --orient-axis.",
                    err=True,
                )
    _ELECTRODE_NONDEFAULTS = (
        ("orthogonal",        orthogonal,        False),
        ("electrode-offset",  electrode_offset,  "0,0"),
        ("lattice-constant",  lattice_constant,  None),
    )
    if not op_types["--electrode"]:
        for name, value, default in _ELECTRODE_NONDEFAULTS:
            if value != default:
                click.echo(
                    f"warning: --{name} is a sub-option of --electrode; "
                    f"value {value!r} is ignored without --electrode.",
                    err=True,
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
            ax, _sep, ang_str = rotate_value.partition(":")
            ax = ax.strip()
            ang_str = ang_str.strip()
            if not _sep or ax not in ("x", "y", "z"):
                raise click.BadParameter(
                    f"--rotate must be 'AXIS:ANGLE' with AXIS in x/y/z; "
                    f"got {rotate_value!r}"
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
                        center_indices=spec["center_indices"],
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
                        center_indices=spec["center_indices"],
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
#  run subcommand (emit a shell wrapper for a generated script)         #
# --------------------------------------------------------------------- #


@cli.command("run", short_help="emit a shell wrapper that executes the script")
@click.argument("script", type=click.Path(exists=True, dir_okay=False,
                                            path_type=Path))
@click.option("--env", "env_override", default=None,
               help="override the routed conda env name "
                    "(default: env_for_category by file extension)")
@click.option("--np", "mpi_np", type=click.IntRange(min=1), default=None,
               help="MPI rank count for SIESTA jobs (.fdf); "
                    "ignored for .py scripts.  np=1 emits a "
                    "single-process wrapper (no mpirun); np>=2 emits "
                    "mpirun -np N")
@click.option("--omp-threads", "omp_threads", type=click.IntRange(min=1),
               default=None,
               help="OMP threads per MPI rank for SIESTA wrappers.  "
                    "Default (omitted): auto = physical_cores // mpi_np.")
@click.option("--max-memory-mb", "max_memory_mb", type=click.IntRange(min=1),
               default=None,
               help="MB cap per SIESTA rank (emitted as ulimit -v in "
                    "the wrapper).  PySCF scripts honor their own "
                    "in-script max_memory cfg instead.")
def cmd_run(script: Path,
            env_override: Optional[str],
            mpi_np: Optional[int],
            omp_threads: Optional[int],
            max_memory_mb: Optional[int]) -> int:
    """Generate ``<basename>.run.sh`` next to SCRIPT.

    Auto-routes by file extension:

      .fdf  ->  mpirun + siesta in the molbuilder-siesta env
      .py   ->  python in the molbuilder-pySCF env

    The wrapper is plain bash -- edit it to add custom flags, source
    it from a SLURM batch script, or run it directly:

      bash my-job.run.sh           # foreground
      nohup ./my-job.run.sh &      # background, detached

    molbuilder does NOT manage the resulting process; monitoring is
    via the existing Watch tab pointed at the run directory.
    """
    from .runwrap import write_run_wrapper, WrapperError
    try:
        wrapper = write_run_wrapper(
            script,
            env=env_override, mpi_np=mpi_np,
            omp_threads=omp_threads,
            max_memory_mb=max_memory_mb,
        )
    except WrapperError as exc:
        raise click.UsageError(str(exc)) from None
    click.echo(f"Wrote {wrapper}")
    click.echo(f"Run:   bash {wrapper.name}")
    return 0


@cli.command("xv2xyz",
             short_help="translate a SIESTA .XV to extended-XYZ (cell-preserving)")
@click.argument("xv_path", metavar="input.XV",
                type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("xyz_path", metavar="output.xyz", type=click.Path(path_type=Path))
def cmd_xv2xyz(xv_path: Path, xyz_path: Path) -> int:
    """Convert a SIESTA ``.XV`` final-coordinates file to extended-XYZ.

    The periodic cell is preserved on the comment line as an ASE
    ``Lattice="..."`` header (Å), so a downstream ``molbuilder fdf`` keeps
    the real cell instead of inventing a vacuum box.  This is the
    convenient ``.XV`` extraction entry; the underlying API is
    ``molbuilder.parse.coords.xv_to_xyz``.
    """
    from .parse.coords import xv_to_xyz
    text = xv_to_xyz(xv_path, xyz_path)
    n = text.splitlines()[0].strip() if text else "?"
    click.echo(f"Wrote {xyz_path}: {n} atoms (cell preserved as Lattice=…)")
    return 0


@cli.command("monitor",
             short_help="background job-monitor + notifier hooks (PoC)")
@click.option("--out", "out_path", required=True,
              type=click.Path(path_type=Path),
              help="the SIESTA .out (or -runN.out) to watch")
@click.option("--timing", "timing_path", required=True,
              type=click.Path(path_type=Path),
              help="the per-run .scf-timing.log (gives the iteration COUNT)")
@click.option("--log", "log_path", required=True,
              type=click.Path(path_type=Path),
              help="append status lines here (e.g. <basename>.monitor.log)")
@click.option("--interval", type=click.FloatRange(min=1.0), default=5.0,
              show_default=True, help="seconds between wakes = the util "
                                      "sample rate (status lines stay "
                                      "change-gated, so it won't spam)")
@click.option("--util", "util_path", default=None,
              type=click.Path(path_type=Path),
              help="append change-gated cpu%/mem/GPU-sm%/VRAM samples to "
                   "this CSV (e.g. <basename>.util.csv)")
@click.option("--stall-heartbeat", "stall_heartbeat_s",
              type=click.FloatRange(min=0.0), default=600.0, show_default=True,
              help="while the job makes no SCF/geometry progress, emit at "
                   "most one liveness ping this often (no per-iter timing "
                   "is printed while stalled); 0 = silence it entirely")
@click.option("--watch-pid", type=int, default=0,
              help="stop when this PID (the job wrapper) disappears; "
                   "0 = run until a .out completion marker")
@click.option("--nice", "nice_level", type=int, default=19, show_default=True,
              help="self-lower OS priority by this much so the monitor "
                   "never competes with compute ranks on the same node")
def cmd_monitor(out_path: Path, timing_path: Path, log_path: Path,
                interval: float, util_path: Optional[Path],
                stall_heartbeat_s: float,
                watch_pid: int, nice_level: int) -> int:
    """Periodically parse the running job's artifacts, append a status
    line, and fire notifier hooks -- the front end of the job-monitor /
    notifier surface (docs/protocols/slurm-integration.md § 11.0b).

    Lightweight by design: sleeps between wakes, does only tail-reads, and
    self-lowers its OS priority (``--nice``) so it yields to the compute
    task on a busy node.  Connect a real notifier via the ``MB_NOTIFY_URL``
    env (stdlib webhook POST) or ``molbuilder.monitor.register_notifier``.
    """
    from . import monitor as _mon
    # Belt-and-suspenders to the launcher's ``nice``: lower our own
    # priority so a busy node always favours the compute task.
    try:
        os.nice(max(0, nice_level))
    except (OSError, AttributeError):
        pass
    _mon.register_notifier(_mon.make_log_notifier(log_path))
    _mon.run_monitor(out_path, timing_path, log_path,
                     interval=interval, watch_pid=watch_pid,
                     stall_heartbeat_s=stall_heartbeat_s,
                     util_path=util_path)
    return 0


# --------------------------------------------------------------------- #
#  serve subcommand (Flask web UI)                                      #
# --------------------------------------------------------------------- #


_LOOPBACK_HOSTS = frozenset({
    "127.0.0.1", "localhost", "::1", "0.0.0.0:127.0.0.1",
})


def _is_loopback_host(host: str) -> bool:
    """True iff ``host`` is a loopback bind that no remote client
    can reach.  We treat ``0.0.0.0`` as NON-loopback even though
    Python can bind it -- it accepts connections from every NIC,
    including LAN + the public internet."""
    return host in _LOOPBACK_HOSTS or host.startswith("127.")


def _enforce_tls_for_remote_bind(host: str, ssl_ctx,
                                  allow_insecure: bool) -> None:
    """Refuse to bind a non-loopback host without TLS.  This is
    molbuilder's "you can't just publish your projects/ tree on the
    public internet by mistake" guard -- the file-ops endpoints
    have no auth, so cleartext over a real network is two attacks
    in one (passive sniffing + active tampering).

    Operators who genuinely want plain HTTP on a non-loopback host
    (e.g., behind a TLS-terminating reverse proxy on the same
    machine) can pass ``--allow-insecure-binding`` to bypass; we
    still print a loud warning so the choice is visible in logs.

    See ``docs/deployment.md`` for the recommended deployment
    shapes (reverse proxy + auth gateway).
    """
    if _is_loopback_host(host):
        return
    if ssl_ctx is not None:
        return
    if allow_insecure:
        click.echo(
            f"WARNING: --host={host} binds a non-loopback interface "
            f"WITHOUT TLS.  --allow-insecure-binding bypasses the "
            f"safety check.  Make sure your reverse proxy terminates "
            f"TLS and gates auth.  See docs/deployment.md.",
            err=True,
        )
        return
    raise click.UsageError(
        f"--host={host} is not a loopback address; binding it serves "
        f"the entire projects/ tree (read + write + delete) to every "
        f"client that can reach the interface.  molbuilder has no "
        f"built-in auth -- the file API is fully open.\n\n"
        f"For a real deployment you have three reasonable options:\n"
        f"  1. Pass --cert / --key to enable TLS (defense in depth; "
        f"still no auth!).\n"
        f"  2. Put molbuilder behind a reverse proxy that adds TLS + "
        f"auth (recommended -- see docs/deployment.md).\n"
        f"  3. Pass --allow-insecure-binding to override this check "
        f"(only sensible when something OUTSIDE molbuilder gates "
        f"access -- a same-host proxy, a VPN tunnel, etc.).\n"
    )


def _print_oauth_redirect_hint_if_auth_on(scheme, host, port):
    """When ``auth`` is configured in ``molbuilder.json``, print the
    callback URL for each configured OAuth provider so the operator
    can register them in the respective consoles without guessing.

    Each OAuth provider (google / github / microsoft / orcid) has its
    own console + its own Authorized-redirect-URIs list; molbuilder
    derives a per-provider URL of the form
    ``<scheme>://<host>:<port>/oauth-callback/<provider_id>``.  CAS
    providers use a separate callback path and are skipped here (CAS
    "service" URLs are auto-derived at request time and don't need
    pre-registration in the same way).

    We can't construct the full URL with certainty (``--host`` may be
    ``0.0.0.0`` for "bind every NIC"; the public hostname might be
    different from any local interface; a reverse proxy may rewrite
    the host).  We print best-guess URLs using the bind address and
    note that the operator must swap the host part for their public
    hostname if relevant.
    """
    try:
        from .runtime_config import read_config, get_providers
        cfg = read_config()
    except Exception:
        return  # bad / missing config -- handled elsewhere
    providers = get_providers(cfg)
    oauth_providers = [
        p for p in providers
        if p["kind"] in ("google", "github", "microsoft", "orcid")
    ]
    if not oauth_providers:
        return

    click.echo(
        "\nOAuth: each configured provider has its own console "
        "where you must register the redirect URI below as an "
        "'Authorized redirect URI' (Google), 'Authorization "
        "callback URL' (GitHub), 'Redirect URI' (Microsoft), or "
        "'Redirect URI' (ORCID):",
        err=True,
    )
    for p in oauth_providers:
        guess = f"{scheme}://{host}:{port}/oauth-callback/{p['id']}"
        click.echo(f"  {p['id']:>14s}  ({p['kind']:>9s})  ->  {guess}",
                    err=True)
    click.echo(
        "(If --host is 0.0.0.0 or you sit behind a reverse proxy, "
        "swap the host part for your public hostname -- the "
        "/oauth-callback/<id> path is the only fixed bit.)\n",
        err=True,
    )


def _resolve_tls(cert_cli, key_cli):
    """CLI flags > ./molbuilder.json > (None, None).

    Reads cert/key from ``./molbuilder.json`` (via
    :mod:`molbuilder.config`).  Both nested (``"tls": {"cert": ...,
    "key": ...}``) and flat (top-level ``"cert"`` / ``"key"``) shapes
    are accepted; see ``molbuilder.config`` for details.  A partial
    pair (cert without key or vice versa) is reported on stderr and
    falls back to HTTP.

    Readability of the resolved paths is NOT checked here -- this
    function only resolves the precedence chain, so it stays pure
    and the tests don't need to touch the filesystem.  The call site
    (``cmd_serve``, ``cmd_watch_serve``) invokes
    ``_check_tls_readable`` immediately after resolution so the
    failure surfaces as a clean ``click.UsageError`` instead of the
    bare ``PermissionError`` Werkzeug raises from
    ``load_cert_chain`` deep in the stack.
    """
    cert, key = cert_cli, key_cli
    if cert and key:
        return cert, key
    try:
        tls = get_tls(read_config())
    except RuntimeConfigError as exc:
        # Translate the L1 domain exception into the click surface
        # (preserves the SystemExit(2) contract the older inline code had).
        raise click.UsageError(str(exc)) from None
    cert = cert or tls.get("cert")
    key  = key  or tls.get("key")
    if (cert and not key) or (key and not cert):
        click.echo(
            "molbuilder: cert/key pair incomplete -- falling back to HTTP",
            err=True,
        )
        return None, None
    return cert, key


def _check_tls_readable(cert, key) -> None:
    """Verify the resolved TLS cert + key are readable by THIS process
    before handing them to Werkzeug.

    Raises ``click.UsageError`` with a concrete fix suggestion when
    either file is missing or unreadable.  No-op when ``cert`` or
    ``key`` is falsy (the caller has already decided no TLS is in
    play).

    The reason this is a *pre-flight* rather than letting Werkzeug
    discover the problem: ``load_cert_chain`` raises a bare
    ``PermissionError`` deep in the stack with no indication of
    which file failed (cert vs key), and the operator is left to
    diff two paths against ``ls -l`` output to figure out which one
    they need to chmod.  The typical cause is a Let's Encrypt
    install where ``/etc/letsencrypt/live/<domain>/privkey.pem`` is
    root-owned + mode 0600 while molbuilder runs as an unprivileged
    user; the error message points at the standard fix (reverse
    proxy from docs/deployment.md) so the operator doesn't reach
    for ``chmod 0644 privkey.pem`` instead.
    """
    if not cert or not key:
        return
    failures = []
    for label, path in (("cert", cert), ("key", key)):
        try:
            # Just open + close: matches what Werkzeug's
            # ``load_cert_chain`` does and surfaces the exact OS
            # error (Permission denied / No such file / Is a
            # directory) without us having to enumerate the cases.
            # ``os.access`` would be wrong here -- it can lie under
            # ACLs / sudo / Linux capabilities.
            with open(path, "rb"):
                pass
        except OSError as exc:
            failures.append(
                f"  {label}: {path}\n"
                f"    {type(exc).__name__}: {exc.strerror}"
            )
    if not failures:
        return
    raise click.UsageError(
        "TLS cert/key unreadable by this process:\n"
        + "\n".join(failures)
        + "\n\nTypical fix when the paths point at a system-managed "
          "cert store (e.g., Let's Encrypt's "
          "/etc/letsencrypt/live/<domain>/):\n"
          "  * Don't read those paths directly from molbuilder.  Put "
          "molbuilder behind a reverse proxy (nginx / Caddy) that "
          "owns TLS termination and forwards plain HTTP to molbuilder "
          "on 127.0.0.1.  See docs/deployment.md for the recommended "
          "shape.\n"
          "  * Or: copy cert + key into a directory the molbuilder "
          "user can read (mode 0600 on the key) and point "
          "molbuilder.json at the copy.  Add a renewal hook so the "
          "copy stays in sync.\n"
          "  * Or (less clean): add the molbuilder user to the group "
          "that owns the key + ``chmod g+r``.  Survives renewal "
          "iff the system installer preserves group + mode."
    )


@cli.command("auth-setup",
              short_help="generate molbuilder.json's auth block for "
                         "ASU CAS and/or Google OAuth (interactive)")
@click.option("--provider", type=click.Choice(["asu", "google", "both"]),
              default=None,
              help="which provider(s) to wire up.  Default: prompt.")
@click.option("--asurite", default=None,
              help="ASU username for the CAS allowlist.  Default: the "
                   "current system user (``getpass.getuser()``).")
@click.option("--google-email", default=None, multiple=True,
              metavar="EMAIL",
              help="Google-account email allowed to sign in via OAuth.  "
                   "May be passed multiple times.  Default: prompt.")
@click.option("--hosted-domain", default=None, multiple=True,
              metavar="DOMAIN",
              help="restrict Google sign-in to Workspace accounts in "
                   "DOMAIN (e.g. 'asu.edu').  May be passed multiple "
                   "times.  Default: no restriction.")
@click.option("--output", type=click.Path(dir_okay=False), default=None,
              help="where to write the molbuilder.json.  Default: "
                   "``./molbuilder.json`` in the current directory.")
@click.option("--force", is_flag=True,
              help="overwrite an existing molbuilder.json's auth block.  "
                   "Other top-level sections (envs, tls, ...) survive.")
def cmd_auth_setup(provider, asurite, google_email, hosted_domain,
                    output, force):
    """Interactive wizard to wire up sign-in for ``molbuilder serve``.

    Generates a ``molbuilder.json`` carrying one or both of:

    \b
      - ASU CAS    (https://weblogin.asu.edu/cas)
      - Google OAuth (your Google Cloud project's client_id + secret)

    Hard-coded into the wizard:

    \b
      * The CAS principal (== the ASU username) defaults to the
        SYSTEM USER ACCOUNT (``getpass.getuser()``).  No other
        identifier is assumed anywhere in molbuilder; the username
        you log in to the server with is the username CAS will
        authenticate against.
      * The Flask session signing key is generated locally with
        ``secrets.token_urlsafe(32)`` and written to a 0600 file under
        ``~/.config/molbuilder/secret_key``.  It is NEVER printed,
        NEVER logged, and NEVER placed into molbuilder.json -- the
        config file holds only the PATH.
      * The Google OAuth client secret is prompted via ``getpass``
        (hidden input, no echo, no shell history) and written to
        ``~/.config/molbuilder/google_client_secret`` with mode 0600.
        Same path-not-literal rule applies.
      * molbuilder.json itself is written mode 0600.

    Re-running with the same arguments is idempotent EXCEPT for the
    Flask session key + Google client secret, which are re-generated /
    re-prompted each run.  Use ``--force`` to acknowledge replacing an
    existing molbuilder.json's auth block.

    Where the file lives: ``molbuilder serve`` looks for
    ``./molbuilder.json`` in the directory it was launched from.  Drop
    this file there (the wizard's default), or pass --output to write
    somewhere else.
    """
    import getpass
    from pathlib import Path as _Path

    from . import auth_setup as _as
    from .runtime_config import _validate_provider as _validate

    # 1. Pick providers ------------------------------------------------
    if provider is None:
        click.echo("Pick provider(s):", err=True)
        click.echo("  1) ASU CAS only", err=True)
        click.echo("  2) Google OAuth only", err=True)
        click.echo("  3) Both", err=True)
        choice = click.prompt("Choice [1/2/3]",
                              type=click.Choice(["1", "2", "3"]),
                              show_choices=False)
        provider = {"1": "asu", "2": "google", "3": "both"}[choice]
    want_asu = provider in ("asu", "both")
    want_google = provider in ("google", "both")

    # 2. Resolve target paths -----------------------------------------
    output_path = _Path(output or "molbuilder.json").resolve()
    secret_dir = _as.default_secret_dir()
    secret_key_file = _as.secret_key_path()
    google_secret_file = _as.google_client_secret_path()

    # 3. Bail early on clobber unless --force --------------------------
    if output_path.exists() and not force:
        click.echo(
            f"Error: {output_path} already exists.  Re-run with "
            f"--force to overwrite, or pass --output PATH.",
            err=True,
        )
        sys.exit(2)

    # 4. ASU CAS entry -------------------------------------------------
    providers: list = []
    if want_asu:
        sys_user = getpass.getuser()
        if asurite is None:
            asurite = click.prompt(
                f"ASURITE (ASU username) for the CAS allowlist",
                default=sys_user,
            )
        entry = _as.build_asu_cas_entry(asurite)
        # Round-trip through the canonical validator so a future
        # schema change can't let the wizard emit something the
        # server then rejects at startup.
        _validate(entry, idx=len(providers))
        providers.append(entry)
        click.echo(
            f"  + ASU CAS configured for "
            f"{entry['allowed_users'][0]}",
            err=True,
        )

    # 5. Google OAuth entry --------------------------------------------
    if want_google:
        click.echo("", err=True)
        click.echo(
            "Google OAuth setup -- you'll need the OAuth client you "
            "created at https://console.cloud.google.com/apis/credentials",
            err=True,
        )
        client_id = click.prompt("  Google OAuth client_id")
        # getpass.getpass: no terminal echo, no shell history.
        client_secret = getpass.getpass(
            prompt="  Google OAuth client_secret (input hidden): ",
        )
        if not client_secret.strip():
            click.echo(
                "Error: client_secret is empty.  Aborting; nothing "
                "written.",
                err=True,
            )
            sys.exit(2)
        # Allowed emails: --google-email overrides; otherwise prompt.
        if google_email:
            emails = list(google_email)
        else:
            click.echo(
                "  Allowed Google-account email(s).  Press Enter on "
                "an empty line to finish.",
                err=True,
            )
            emails = []
            while True:
                e = click.prompt(
                    f"    email {len(emails)+1}",
                    default="", show_default=False,
                )
                if not e:
                    if not emails:
                        click.echo(
                            "    (need at least one)", err=True,
                        )
                        continue
                    break
                emails.append(e)
        # Save the secret out-of-band BEFORE building the entry so we
        # have a clean file path to reference in molbuilder.json.
        _as.write_secret_file(google_secret_file, client_secret.strip())
        entry = _as.build_google_entry(
            client_id=client_id,
            client_secret_file=google_secret_file,
            allowed_users=emails,
            hosted_domain=list(hosted_domain) if hosted_domain else None,
        )
        _validate(entry, idx=len(providers))
        providers.append(entry)
        click.echo(
            f"  + Google OAuth configured for {len(emails)} "
            f"allowed email(s); secret stored at {google_secret_file}",
            err=True,
        )

    # 6. Flask session signing key ------------------------------------
    session_secret = _as.generate_session_secret()
    _as.write_secret_file(secret_key_file, session_secret)
    # Wipe the in-memory copy promptly; the file is the source of truth.
    del session_secret
    click.echo(
        f"  + Flask session key generated; stored at {secret_key_file}",
        err=True,
    )

    # 7. Merge auth block into existing molbuilder.json (if any) -------
    existing = None
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            click.echo(
                f"Warning: could not parse existing {output_path} "
                f"({exc}).  --force is set, replacing it whole.",
                err=True,
            )
            existing = None
    auth_block = _as.build_auth_block(
        providers=providers,
        secret_key_file=secret_key_file,
    )
    _as.emit_molbuilder_json(
        output_path, auth_block,
        force=force, existing=existing,
    )

    click.echo("", err=True)
    click.echo(f"Wrote {output_path} (mode 0600)", err=True)
    click.echo("", err=True)
    click.echo("Next steps:", err=True)
    click.echo(
        f"  cd {output_path.parent}", err=True,
    )
    click.echo(
        f"  python -m molbuilder serve --port 8888 --host 127.0.0.1",
        err=True,
    )
    if want_google:
        click.echo("", err=True)
        click.echo(
            "Add this callback URL to your Google OAuth client's "
            "'Authorized redirect URIs':", err=True,
        )
        click.echo(
            "  http://localhost:8888/oauth-callback/google", err=True,
        )
        click.echo(
            "  (adjust host/port to match your tunnel + --port).",
            err=True,
        )


@cli.command("runtime-info",
             short_help="dump runtime_info as JSON sidecar from a SIESTA / PySCF output")
@click.argument("input_path", metavar="input")
@click.option("--out", "out_path", default=None,
              type=click.Path(dir_okay=False),
              help="Output JSON path.  Default: ``<input-stem>.runtime_info.json`` "
                   "next to the input.  Use ``-`` for stdout.")
@click.option("--pretty/--no-pretty", default=True, show_default=True,
              help="Indent the JSON output.")
def cmd_runtime_info(input_path, out_path, pretty):
    """Parse a SIESTA / PySCF / .molwatch.log file and write its
    ``runtime_info`` dict to a JSON sidecar for offline / CLI consumers.

    The same dict the watcher streams to the Results tab -- includes
    ``siesta_build`` (version, parallelisations, ELPA linkage, ...),
    ``siesta_diag`` (algorithm, GPU device, ...), ``convergence_targets``,
    ``frozen_atoms``, etc.  Useful for post-processing scripts that
    want to verify what build / diagonalizer a run actually used
    without running the full live watcher.

    Examples::

        molbuilder runtime-info job.out
        # -> writes job.runtime_info.json next to job.out

        molbuilder runtime-info job.out --out - | jq '.siesta_diag'
        # -> stream to jq for inspection

        molbuilder runtime-info job.out --out /tmp/x.json --no-pretty
        # -> compact single-line JSON to a specific path
    """
    import json
    from .parse import detect as detect_parser, UnknownFormatError

    with _resolve_input_path(input_path) as resolved:
        try:
            parser = detect_parser(resolved)
        except UnknownFormatError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(2)
        traj = parser.parse(resolved)

    # frozen_atoms is a Python set in-memory; convert to a sorted list
    # for JSON.  Any other non-JSON-native types should fail loudly so
    # we notice the schema drift -- don't paper over with default=str.
    runtime_info = dict(traj.runtime_info or {})
    if "frozen_atoms" in runtime_info and isinstance(
            runtime_info["frozen_atoms"], (set, frozenset)):
        runtime_info["frozen_atoms"] = sorted(runtime_info["frozen_atoms"])

    payload = json.dumps(runtime_info, indent=2 if pretty else None,
                         sort_keys=True)

    if out_path == "-":
        click.echo(payload)
        return

    if out_path is None:
        # Default: <stem>.runtime_info.json next to the input.  For
        # ``job.out`` this writes ``job.runtime_info.json``; for
        # ``job.molwatch.log`` it writes ``job.molwatch.runtime_info.json``
        # (stem strip is intentionally single-suffix -- matches
        # transport.json + molstruct.json conventions).
        in_path = Path(input_path)
        out_path = str(in_path.with_suffix("")) + ".runtime_info.json"

    out = Path(out_path)
    out.write_text(payload + "\n", encoding="utf-8")
    click.echo(f"wrote {out}", err=True)


@cli.command("serve", short_help="run the browser UI (Flask + 3Dmol.js)")
@click.option("--host",  default="127.0.0.1", show_default=True)
@click.option("--port",  type=int, default=8000, show_default=True)
@click.option("--debug", is_flag=True)
@click.option("--cert", type=click.Path(exists=True, dir_okay=False),
              help="TLS cert (PEM).  Overrides molbuilder.json.")
@click.option("--key",  type=click.Path(exists=True, dir_okay=False),
              help="TLS key (PEM).  Overrides molbuilder.json.")
@click.option("--allow-insecure-binding", is_flag=True,
              help="Bypass the loopback-or-TLS guard.  Only sensible "
                   "when something outside molbuilder (proxy / VPN) "
                   "gates access -- see docs/deployment.md.")
@click.option("--no-auth", is_flag=True,
              help="Run with NO authentication (ignores molbuilder.json's "
                   "auth/TLS).  Allowed ONLY on a loopback --host "
                   "(127.0.0.1 / localhost / ::1) so an unauthenticated "
                   "server can never be exposed; refuses otherwise.  For "
                   "local dev, screenshots, and tests.")
def cmd_serve(host, port, debug, cert, key, allow_insecure_binding, no_auth):
    """Start a Flask server with the molbuilder browser UI."""
    from .web.app import create_app

    if no_auth:
        # Auth-free is a LOCAL-ONLY convenience: refuse anything but a
        # loopback bind so an unauthenticated server is never reachable
        # off the machine.  create_app(config={}) is the supported
        # no-auth seam (see web/app.py:create_app); it ignores
        # molbuilder.json entirely (no providers -> no login), and the
        # projects root still resolves from the CWD.
        if not _is_loopback_host(host):
            raise click.ClickException(
                f"--no-auth requires a loopback --host (got {host!r}); "
                "refusing to start an unauthenticated server on a "
                "non-loopback interface.")
        app = create_app(config={})
        click.echo(
            f"molbuilder web UI (NO AUTH -- loopback only) starting at "
            f"http://{host}:{port}", err=True)
        app.run(host=host, port=port, debug=debug, ssl_context=None)
        return

    cert, key = _resolve_tls(cert, key)
    _check_tls_readable(cert, key)
    ssl_ctx = (cert, key) if cert and key else None
    _enforce_tls_for_remote_bind(host, ssl_ctx, allow_insecure_binding)
    scheme  = "https" if ssl_ctx else "http"
    app = create_app()
    click.echo(f"molbuilder web UI starting at {scheme}://{host}:{port}", err=True)
    _print_oauth_redirect_hint_if_auth_on(scheme, host, port)
    app.run(host=host, port=port, debug=debug, ssl_context=ssl_ctx)


# --------------------------------------------------------------------- #
#  watch subcommand group (live trajectory viewer)                      #
# --------------------------------------------------------------------- #


@cli.group("snapshot",
           short_help="git-based run-checkpoints (init, checkpoint, "
                      "list, tag, restore)")
def cmd_snapshot():
    """Manage the working dir's checkpoint history.

    Each working dir (the one containing your .fdf / .py / .run.sh)
    becomes a tiny self-contained git repo.  Big binaries (.DM, .HSX,
    .TSHS, .TBT.AVTRANS_*) are archived by SHA in .binsnapshots/<sha>/
    rather than committed.  Use this group to snapshot before risky
    changes and roll back if things go wrong.

    Examples:

        molbuilder snapshot init                      # one-time setup
        molbuilder snapshot checkpoint -m "stage 3 converged"
        molbuilder snapshot tag stage3-converged -m "ready for transport"
        molbuilder snapshot list
        molbuilder snapshot restore stage3-converged  # rewinds everything

    See docs/protocols/run-checkpoints.md for the full design.
    """


def _resolve_repo_path(path: Optional[str]) -> str:
    """Resolve the ``--path`` option / cwd default to an absolute dir."""
    if path is None:
        return str(Path.cwd().resolve())
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        click.echo(f"Error: {p} is not a directory", err=True)
        sys.exit(2)
    return str(p)


@cmd_snapshot.command("init",
                      short_help="initialise this dir as a checkpoint repo")
@click.option("--engine", type=click.Choice(["siesta", "pyscf"]), default=None,
              help="Engine whose big-binary files to archive (SIESTA: "
                   ".DM/.HSX/.TSHS/.TBT.*; PySCF: .chk/.cube).  The web UI "
                   "passes this from the task setup; on the CLI, set it so "
                   "the right files are archived.  Omit for the safe union.")
@click.option("-p", "--path", default=None, type=click.Path(),
              help="Working dir to initialise.  Default: cwd.")
def cmd_snapshot_init(engine, path):
    """Create a git repo + .gitignore + first commit + binary archive
    in the current working dir (or --path).

    Refuses if the directory contains nested working dirs (subdirs
    with .fdf / .py / .run.sh) -- each lowest-directory is its own
    checkpoint repo per the run-checkpoints.md § P5 rule.

    ``--engine`` selects which files count as big binaries (the persisted,
    editable classification -- see ``molbuilder snapshot config``).
    """
    from molbuilder.checkpoint import (
        Repo, NestedRepoRefusedError, CheckpointError,
    )
    repo = Repo(_resolve_repo_path(path))
    if repo.initialized:
        click.echo(f"{repo.path}: already initialised "
                   f"(HEAD = {repo._head_sha()[:7]})")
        return
    try:
        repo.init(engine=engine)
    except NestedRepoRefusedError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)
    except CheckpointError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    state = repo.state()
    click.echo(f"Initialised {repo.path}")
    click.echo(f"  HEAD = {state.head[:7] if state.head else '?'}")
    # state() is cheap and omits archive size (§ 6.2); ask for it
    # explicitly here -- this is a one-shot command, not a poll.
    archived = repo.archive_total_bytes()
    if archived:
        click.echo(f"  archived {archived / (1024 * 1024):.1f} "
                   f"MB of big binaries to .binsnapshots/")


@cmd_snapshot.command("config",
                      short_help="show / edit which files are archived as big binaries")
@click.option("--set", "set_globs", default=None, metavar="GLOBS",
              help="Comma-separated globs to REPLACE the archive set "
                   "(e.g. '*.DM,*.HSX,*.chk').  Regenerates .gitignore to "
                   "match.  Omit to just show the current set.")
@click.option("-p", "--path", default=None, type=click.Path(),
              help="Working dir.  Default: cwd.")
def cmd_snapshot_config(set_globs, path):
    """Show (or ``--set``) the big-binary patterns this repo archives -- the
    engine-specific, user-editable classification (run-checkpoints.md § 9).
    The web UI edits the SAME persisted table through the API; this is the
    CLI face of that one unified accessor."""
    from molbuilder.checkpoint import Repo, CheckpointError
    repo = Repo(_resolve_repo_path(path))
    if not repo.initialized:
        click.echo(f"Error: {repo.path} is not a checkpoint repo.", err=True)
        sys.exit(2)
    if set_globs is not None:
        globs = [g.strip() for g in set_globs.split(",") if g.strip()]
        try:
            repo.set_archive_globs(globs)
        except CheckpointError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        click.echo("updated archive globs (checkpoint to save the change):")
    for g in repo.archive_globs():
        click.echo(f"  {g}")


@cmd_snapshot.command("checkpoint",
                      short_help="commit current state + archive big binaries")
@click.option("-m", "--message", default=None,
              help="Commit message.  Default: 'checkpoint <ISO_TS>'.")
@click.option("-p", "--path", default=None, type=click.Path(),
              help="Working dir.  Default: cwd.")
def cmd_snapshot_checkpoint(message, path):
    """Stage everything (.fdf, .out, .molwatch.log, sidecars, ...),
    create a new commit, and archive the current big binaries
    (.DM, .HSX, .TSHS, .TBT.AVTRANS_*) under .binsnapshots/<new_sha>/.

    If nothing has changed since HEAD, prints a polite no-op and
    exits 0 -- not an error.
    """
    from molbuilder.checkpoint import Repo, CheckpointError
    repo = Repo(_resolve_repo_path(path))
    if not repo.initialized:
        click.echo(f"Error: {repo.path} is not a checkpoint repo.  "
                   f"Run `molbuilder snapshot init` first.", err=True)
        sys.exit(2)
    try:
        cp = repo.checkpoint(message=message)
    except CheckpointError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    if cp is None:
        click.echo("working tree clean; nothing to checkpoint")
        return
    click.echo(f"checkpointed {cp.short_sha}  {cp.summary}")
    if cp.has_archive:
        click.echo(f"  archived "
                   f"{cp.archive_bytes / (1024 * 1024):.1f} MB of big "
                   f"binaries")


@cmd_snapshot.command("tag",
                      short_help="annotate current commit (or another ref)")
@click.argument("label")
@click.option("-m", "--message", default="",
              help="Tag message.  Required by the design (no lightweight "
                   "tags); will prompt if not provided.")
@click.option("--at", default="HEAD",
              help="Ref to tag.  Default: HEAD.")
@click.option("-p", "--path", default=None, type=click.Path(),
              help="Working dir.  Default: cwd.")
def cmd_snapshot_tag(label, message, at, path):
    """Create an annotated tag.  Always annotated (carries a message)
    per the design § 11 decision 3."""
    from molbuilder.checkpoint import Repo, CheckpointError
    repo = Repo(_resolve_repo_path(path))
    if not repo.initialized:
        click.echo(f"Error: {repo.path} is not a checkpoint repo.",
                   err=True)
        sys.exit(2)
    if not message:
        message = click.prompt("tag message")
    try:
        repo.tag(label, message=message, at=at)
    except CheckpointError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    click.echo(f"tagged {at} as {label!r}")


@cmd_snapshot.command("branch",
                      short_help="fork a branch to explore an alternative")
@click.argument("name")
@click.option("-p", "--path", default=None, type=click.Path(),
              help="Working dir.  Default: cwd.")
def cmd_snapshot_branch(name, path):
    """Create a new branch and switch to it (run-checkpoints.md § 4.5) --
    for exploring an experimental parameter path without losing the current
    one.  Tags mark milestones; branches carry experiments (P6).  Your next
    ``snapshot checkpoint`` lands on this branch; ``snapshot restore`` /
    ``git checkout`` returns you to the original."""
    from molbuilder.checkpoint import Repo, CheckpointError
    repo = Repo(_resolve_repo_path(path))
    if not repo.initialized:
        click.echo(f"Error: {repo.path} is not a checkpoint repo.",
                   err=True)
        sys.exit(2)
    try:
        repo.branch(name)
    except CheckpointError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    click.echo(f"switched to a new branch {name!r} "
               "(subsequent checkpoints land here)")


@cmd_snapshot.command("list",
                      short_help="list checkpoints (most recent first)")
@click.option("-n", "--limit", type=int, default=20, show_default=True,
              help="Maximum number of checkpoints to show.")
@click.option("-p", "--path", default=None, type=click.Path(),
              help="Working dir.  Default: cwd.")
def cmd_snapshot_list(limit, path):
    """Show the checkpoint history, decorated with tags + branches +
    binary-archive presence."""
    from molbuilder.checkpoint import Repo, CheckpointError
    repo = Repo(_resolve_repo_path(path))
    if not repo.initialized:
        click.echo(f"Error: {repo.path} is not a checkpoint repo.",
                   err=True)
        sys.exit(2)
    try:
        cps = repo.list_checkpoints(limit=limit)
    except CheckpointError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    if not cps:
        click.echo("(no checkpoints)")
        return
    for cp in cps:
        refs = f"  [{', '.join(cp.refs)}]" if cp.refs else ""
        arch = (f"  ({cp.archive_bytes / (1024 * 1024):.1f} MB archived)"
                if cp.has_archive else "")
        click.echo(f"{cp.short_sha}  {cp.summary}{refs}{arch}")


@cmd_snapshot.command("restore",
                      short_help="rewind everything (text + binaries) to a ref")
@click.argument("ref")
@click.option("--no-binaries", is_flag=True,
              help="Skip the binary archive copy (text-only restore).")
@click.option("-p", "--path", default=None, type=click.Path(),
              help="Working dir.  Default: cwd.")
def cmd_snapshot_restore(ref, no_binaries, path):
    """Rewind the working tree to the state at REF.

    REF can be a tag (`stage3-converged`), a branch (`main`), or a
    short SHA (`63f143f`).  Refuses on a dirty working tree --
    checkpoint or discard your changes first.

    Copies the archived big binaries (.DM, .HSX, ...) from
    .binsnapshots/<sha>/ on top of the restored text files.  Pass
    --no-binaries for a text-only restore.
    """
    from molbuilder.checkpoint import (
        Repo, CheckpointError, DirtyWorkingTreeError, NoSuchRefError,
    )
    repo = Repo(_resolve_repo_path(path))
    if not repo.initialized:
        click.echo(f"Error: {repo.path} is not a checkpoint repo.",
                   err=True)
        sys.exit(2)
    try:
        restored = repo.restore(ref, include_binaries=not no_binaries)
    except DirtyWorkingTreeError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo(
            "  hint: `molbuilder snapshot checkpoint -m \"WIP\"` "
            "to save current state first, OR "
            "`git restore --staged --worktree .` to discard it.",
            err=True)
        sys.exit(2)
    except NoSuchRefError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)
    except CheckpointError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    click.echo(f"restored working tree to {ref}")
    if restored:
        click.echo(f"  copied {len(restored)} binary "
                   f"{'file' if len(restored) == 1 else 'files'} "
                   f"from archive: {', '.join(restored)}")
    elif not no_binaries:
        # #1: a missing archive in a binary-using project is a LOUD warning
        # (possible interrupted checkpoint), not a neutral note.
        warning = repo.missing_archive_warning(ref)
        if warning:
            click.echo(f"  WARNING: {warning}", err=True)
        else:
            click.echo("  (no archived binaries for this ref)")


@cmd_snapshot.command("migrate-manifest",
                      short_help="convert legacy 2-column MANIFEST to "
                                 "canonical 3-column form (§ 10.4)")
@click.argument("ref")
@click.option("-p", "--path", default=None, type=click.Path(),
              help="Working dir.  Default: cwd.")
def cmd_snapshot_migrate_manifest(ref, path):
    """One-shot conversion of a legacy ``sha256sum``-style 2-column
    MANIFEST in the archive for REF to canonical 3-column form.

    Behaviour (per run-checkpoints.md § 10.4):

      1. Resolves REF to a commit SHA.
      2. Reads .binsnapshots/<sha>/MANIFEST.
      3. If already canonical -> no-op, exits 0 with "already canonical".
      4. If 2-column legacy: parses sha + name, re-hashes each archived
         file's content, verifies against the recorded sha256, stat()s
         for the size column, writes canonical MANIFEST atomically.
      5. Any other shape -> error, no auto-fix; original MANIFEST is
         left untouched.

    REF can be a tag, branch, or short SHA.
    """
    from molbuilder.checkpoint import (
        Repo, CheckpointError, NoSuchRefError,
    )
    repo = Repo(_resolve_repo_path(path))
    if not repo.initialized:
        click.echo(f"Error: {repo.path} is not a checkpoint repo.",
                   err=True)
        sys.exit(2)
    try:
        entries = repo.migrate_manifest(ref)
    except NoSuchRefError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)
    except CheckpointError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    click.echo(f"migrated MANIFEST for {ref}: "
               f"{len(entries)} archived file"
               f"{'s' if len(entries) != 1 else ''}")
    for name, (sha256, size) in sorted(entries.items()):
        click.echo(f"  {sha256[:12]}…  {size:>12} bytes  {name}")


# --------------------------------------------------------------------- #
#  Watch group (live trajectory viewer)                                #
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
    from .parse import detect as detect_parser, UnknownFormatError
    from .parse.engines._helpers import trajectory_to_legacy_dict

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
    from .parse import detect as detect_parser, UnknownFormatError
    from .parse.engines._helpers import trajectory_to_legacy_dict

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


# ``molbuilder watch serve`` removed 2026-05-19 along with the /watch
# page route.  Use ``molbuilder serve`` instead -- it hosts the same
# blueprints (so /api/watch/* remain available for the /results
# trajectory inspector) and supports the same TLS / --allow-insecure-
# binding flags.  The ``molbuilder watch parse`` and ``molbuilder
# watch tail`` subcommands are pure CLI utilities (no web tab) and
# remain unchanged.


# --------------------------------------------------------------------- #
#  Entry points                                                         #
# --------------------------------------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Back-compat int-returning entry point.

    Kept for ``project.scripts`` and for tests that call
    ``cli.main([...])`` directly.

    The contract we need to preserve (inherited from the argparse
    predecessor; tests assert it):
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
    # Configure the root logger so warnings emitted by L1 modules
    # (e.g. ``molbuilder.projects.list_projects`` skipping invalid
    # directory names) actually reach the user.  Without this, Python's
    # root logger has no handler attached and warnings vanish silently.
    import logging
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s: %(message)s")
    # Bind the diagnostics snapshot once per CLI invocation, so every
    # backend's ``is_available`` and every ``run_tool`` dispatch read
    # from a consistent view of "what this machine has".  Cheap (~50 ms);
    # idempotent if called again.  Catch RuntimeConfigError so a
    # malformed molbuilder.json produces the same `Error: ...; exit 2`
    # surface as any other UsageError instead of a Python traceback.
    try:
        _initialize_diagnostics()
    except RuntimeConfigError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)
    try:
        rc = cli.main(args=args, standalone_mode=False)
    except click.UsageError as e:
        # Missing required command, unknown subcommand, missing arg,
        # bad type conversion -- all of these must exit(2) per the
        # contract above.
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

    Maps to ``molbuilder serve`` with whatever extra args the user
    passed.  Originally invoked ``molbuilder watch serve``, but that
    subcommand was removed 2026-05-19 along with the /watch page --
    ``molbuilder serve`` is the canonical entry point and serves the
    same blueprints (/api/watch/* remain available for the /results
    inspector).

    Kept for backwards compatibility with users / scripts that still
    invoke ``molwatch`` directly after the molbuilder + molwatch
    merge.  Operators are encouraged to migrate to ``molbuilder
    serve`` in their scripts; this shim makes the transition silent
    rather than breaking.
    """
    return main(["serve"] + sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
