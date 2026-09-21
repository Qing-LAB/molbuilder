"""molbuilder command-line entry point.

Subcommands:
    molbuilder peptide ARNDC --out file.xyz
    molbuilder dna ATGCATGC --out file.xyz
    molbuilder rna AUGCAUGCAU --out file.xyz
    molbuilder smiles "c1ccccc1" --out benzene.xyz
    molbuilder name "1,4-benzenedithiol" --out bdt.xyz
    molbuilder jobset init --structure P/structure/in.xyz \
        --bundle P/optimization/calc --shape flat --stage-strategy publishable
    molbuilder pyscf in.xyz out.py --functional B3LYP
    molbuilder serve start --port 8000
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
from typing import Iterator, Optional, Sequence

import click

from .diagnostics import initialize as _initialize_diagnostics
from .envs._cli import envs_group
from .runtime_config import RuntimeConfigError, get_tls, read_config
from .structure import Structure


# --------------------------------------------------------------------- #
#  add_dataclass_options: dataclass field metadata -> click.option      #
# --------------------------------------------------------------------- #


# `add_dataclass_options` DELETED 2026-09-17 with its only consumer.
#
# A dataclass -> click bridge: it read every PySCFConfig field's metadata and
# stacked a `click.option` per field, so `molbuilder pyscf` could take the
# whole engine surface as flags.  ~170 lines whose entire purpose was to make
# ONE command possible, and that command is the shape decision 34 deleted
# (`conventions.md` 3: "there is no `molbuilder fdf`" -- user, "obsolete
# residue from the flat-dir design").
#
# A parameter is said in a description, not on a command line.  Nothing else
# generates options from a dataclass, and the two config files that mention
# this bridge do so only to explain why a field opts out of it.


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

def _trajectory_parser_for(resolved):
    """The FileParser for a trajectory artifact, or a clean refusal.

    These verbs hand the result to trajectory code, so they want a file whose
    parser answers a TRAJECTORY.  Two conditions are PERMANENT: a directory
    (`detect()` answers for one since `JobDirParser` was registered) and a
    file whose parser answers something else -- both reach `.frames` and
    raise `AttributeError`.  The `is_dir()` test alone caught only the first,
    so `<job>_optimized.xyz` still crashed all three verbs.

    **The refusal EXITS rather than raising.**  `watch tail` calls this
    inside a poll loop that treats `ParseError` as transient -- the writer
    may not have flushed enough bytes to be detectable yet -- and sleeps.  A
    permanent condition raised as a `ParseError` there retried for ever: a
    silent hang where there used to be a traceback.  `SystemExit` cannot be
    swallowed by that `except`, and every verb prints the same
    "Error: ... / exit 2".
    """
    from pathlib import Path as _P
    from .parse import detect as _detect
    from .parse.types import answers_a_trajectory
    if _P(resolved).is_dir():
        click.echo(f"Error: {resolved} is a directory.  These verbs read one "
                   f"run artifact; name the file inside it (the Watch tab "
                   f"resolves a directory for you, via "
                   f"`parse.dirs.openable_in`).", err=True)
        sys.exit(2)
    parser_cls = _detect(resolved)
    if not answers_a_trajectory(parser_cls):
        kind = getattr(getattr(parser_cls, "output", None), "__name__",
                       "an unknown result")
        click.echo(f"Error: {resolved} is read by the {parser_cls.name!r} "
                   f"parser, which answers a {kind} -- not a trajectory.  "
                   f"These verbs read a run's frames.", err=True)
        sys.exit(2)
    return parser_cls


# --------------------------------------------------------------------- #
#  Shared helpers                                                       #
# --------------------------------------------------------------------- #


def _emit(struct: Structure, *,
          out: Optional[str],
          pdb: Optional[str],
          pyscf_atom_block: bool) -> None:
    """Write the built Structure to whatever destinations the user asked
    for.  No destination at all -> dump XYZ to stdout (Unix-pipeable)."""
    # THE PAIR IS THE FILE here too (`model/structure.md` § 2.4).  A freshly
    # built molecule usually has nothing to put in a sidecar, and then the
    # codec writes none -- "no .json == empty metadata" holds either way.  But
    # a builder that DOES produce metadata (the DNA/RNA duplex builders set
    # residue identity; `--electrode` work starts from these files) had it
    # dropped at the moment it was first written to disk.
    from .workingcopy_structure import StructureCodec
    codec = StructureCodec()
    wrote_anything = False
    if out:
        codec.write(struct, out, fmt="xyz")
        click.echo(f"wrote {struct.n_atoms} atoms to {out}", err=True)
        wrote_anything = True
    if pdb:
        codec.write(struct, pdb, fmt="pdb")
        click.echo(f"wrote {struct.n_atoms} atoms to {pdb}", err=True)
        wrote_anything = True
    if pyscf_atom_block:
        click.echo(struct.to_pyscf(as_string=True))
        wrote_anything = True
    if not wrote_anything:
        sys.stdout.write(struct.to_xyz())
    click.echo(struct.summary(), err=True)


# `KGridParam` / `KGRID` DELETED 2026-09-17.  A click ParamType accepting
# `4x4x1` / `4,4,1` / `4 4 1`, it existed for `molbuilder fdf --kgrid`, and
# went unused the moment that command was deleted (a038ad11, 2026-08-11) --
# which removed its only `type=KGRID` and left the type standing.  No command
# takes a k-grid on the command line: a k-grid is a PARAMETER, so it is said
# in the description and reaches the deck through the template.
#
# Two test docstrings cited it as the terminal's half of a parity claim --
# "what works in the terminal works in the table".  The browser's parser is
# the only one now, and those docstrings say so.


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

# `molbuilder bench ...` was DELETED 2026-08-17 (user: all verbs are unified
# under `jobset`).  Its four lifecycle verbs went in the 2026-08-12 fold, and
# its last inhabitant -- `probe-scheduler`, a scheduler-config helper that was
# never a benchmark verb -- moved to `jobset probe-scheduler`.  A group whose
# name no longer described anything it contained was the last thing keeping
# two spellings alive for one act (process/conventions.md 3).

# `molbuilder jobset ...`  (engine-agnostic staged execution: plan / prep /
# submit a bundle's job-set.json -- the SIESTA stage ladder, and later the
# bench sweep).  See molbuilder/jobset/ + docs/execution/job-system.md.
from .jobset._cli import jobset_group
cli.add_command(jobset_group)

# `molbuilder transport ...` DELETED 2026-09-17 -- the last calculation-KIND
# verb group, and the second half of a closure `bench` already received.
#
# `conventions.md` 3 named the problem on 2026-08-11: "`jobset`, `bench` and
# `transport` each ran calculations their own way".  `bench` was folded into
# `jobset` on 2026-08-17 with its four verbs deleted and its group removed;
# transport migrated to the composite on 2026-08-29 and its `bundle` verb went,
# but the group stayed with the two hand-assembly verbs still in it --
# `electrode`, which wrote a lead deck from flags, and `preflight`, which
# compared that deck against a device deck.  Both are gone, so the group is.
#
# NO CALCULATION KIND HAS A VERB: there is no `spectra`, no `optimization`, no
# `vibration`.  A kind is described in `task.json` and run through `jobset` --
# decision 7, "everything is a job set", and a second way in is a second way to
# lose your results.


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

    # The ONE table (`pseudos.expected_xc_family`).  This copy had no VDW arm,
    # so a van-der-Waals audit compared against no expected family at all and
    # passed a mismatch the preflight blocks.
    from .pseudos import expected_xc_family
    fam = expected_xc_family(xc_authors)

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


# `molbuilder pyscf` DELETED 2026-09-17 -- `molbuilder fdf`'s surviving twin.
#
# Both took a structure plus every engine field as a flag and wrote a finished
# deck, skipping the description.  `fdf` went on 2026-08-11 as decision 34;
# this one kept its flags on the stated grounds that "its ladder runs inside
# one emitted script" -- and the comment beside it refuted that in the same
# file: "THIS COMMAND WRITES ONE DECK, AND A LADDER IS N DECKS ... there is no
# `--stages-json` / `--stage-strategy` here ... `jobset init --engine pyscf
# --stage-strategy ...` is the one door that writes one."  The exemption
# described PySCF's script SHAPE, not a property of the verb, and
# `--stage-strategy` was taken off this command on 2026-08-18 for that reason.
#
# The framework already covered it: `prep.py` builds a full `EngineSeam` for
# PySCF, so `jobset prep` renders it through `spec_for` -> `prepare_deck` like
# any other engine.  It duplicated no mechanism -- it was a second WAY IN,
# which is what decision 7 names: "everything is a job set ... a second way in
# is a second way to lose your results."


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
            && molbuilder jobset init --structure P/structure/run.xyz --bundle P/optimization/calc --shape flat
    """
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
            i.to_json()
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


#: WHICH WALK CONTINUES THE CRYSTAL, per side.  `add_slab`'s `sequence` is
#: read along the growth direction, so "the crystal carries on outward from
#: the contact" is the forward walk going up and the backward walk going down.
#: Written once here because both electrodes of a junction want it and a
#: mapping retyped per call site is a mapping that eventually disagrees.
_CONTINUES_THE_CRYSTAL = {"+z": "ABC", "-z": "ACB"}


def _parse_electrode_spec(spec):
    """Parse one ``--electrode`` value into a kwargs dict.

**One slab per flag**: ``ELEM:PLANE:MxNxL@contact=DIST:+z=I,J,...``
    or ``...:-z=I,J,...``.  ``DIST`` is the centre-to-closest-layer distance
    for the chosen side, and the slab centres on the CENTROID of the trailing
    index list (1 index -> that atom, 2 -> their midpoint, N -> centroid).

    **``@gap=`` is gone** (redesign plan § 3.4).  It built a symmetric PAIR in
    one step, and pairs are not built as one step any more -- a junction is
    two flags, one per side, each saying where its slab goes.  ``gap`` was the
    pair's own parameter and had no meaning for a single slab, so it went with
    it rather than being kept as a second way to say ``contact``.

    Returns a dict whose ``"mode"`` is always ``"single"`` -- one slab per
    flag, with its own side and stand-off.
    """
    main, sep, after_at = spec.partition("@")
    if not sep or not after_at:
        raise click.BadParameter(
            f"--electrode {spec!r}: missing '@contact=...' "
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
            f"--electrode {spec!r}: '@{keyval}' must be "
            f"'@contact=NUM' (key=value form)"
        )
    if key == "gap":
        # PAIR MODE IS GONE, and `gap` went with it -- it was the PAIR's
        # parameter, the electrode-to-electrode distance, meaningless for one
        # slab.  Slabs are built one at a time now and where each goes is
        # stated (redesign plan § 3.4).
        #
        # Refused by name rather than falling into the generic "unknown key",
        # which would read as a typo: this key existed, did something, and was
        # removed, so the message has to say what to do instead.
        raise click.BadParameter(
            f"--electrode {spec!r}: '@gap=' is no longer supported.  It was "
            f"the electrode-to-electrode distance of a PAIR, and pairs are "
            f"not built as one step any more.  Pass --electrode twice, one "
            f"per side, each with '@contact=NUM:+z=I' or ':-z=I' -- which is "
            f"what the pair did internally, with the two positions now "
            f"stated rather than derived from a gap.")
    try:
        distance = float(val)
    except ValueError:
        raise click.BadParameter(
            f"--electrode {spec!r}: distance {val!r} after '@{key}=' must be a float"
        )

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
        f"--electrode {spec!r}: unknown key {key!r}; expected 'contact'"
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
              help="add ONE FCC slab, centred on the CENTROID of the "
                   "trailing atom-index list (1 index -> that atom, 2 -> "
                   "their midpoint, N -> centroid).  "
                   "'Au:111:3x3x2@contact=2.4:+z=3' -- contact is the "
                   "centre-to-closest-layer distance for that side.  Repeat "
                   "the flag for the other side, or for stepped contacts.")
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

    Examples -- canonical Au-bdt-Au junction in a 3-step pipe.  One slab
    per flag: each says which side it goes on and how far it stands off.

        # input: relaxed BDT geometry with 4 atoms (S-C-C-S)
        molbuilder modify bdt.xyz - --orient-axis 0,3 --center midpoint |
          molbuilder modify - junction.xyz \\
              --electrode Au:111:3x3x2@contact=2.4:+z=3 \\
              --electrode Au:111:3x3x2@contact=2.4:-z=0

    Stepped 3×3 + 4×4 contact on the same side:

        molbuilder modify oriented.xyz junction.xyz \\
            --electrode Au:111:3x3x1@contact=2.4:+z=3 \\
            --electrode Au:111:4x4x1@contact=2.4:+z=3

    Asymmetric junction (Au top, Cu bottom):

        molbuilder modify oriented.xyz step1.xyz \\
            --electrode Au:111:3x3x2@contact=2.4:+z=3
        molbuilder modify step1.xyz junction.xyz \\
            --electrode Cu:111:3x3x2@contact=2.0:-z=0

    See docs/web/tabs.md for the full per-(plane, orthogonal)
    constraint table; ASE's own error message bubbles up if the
    requested (m, n) doesn't satisfy the chosen cell shape.
    """
    import numpy as np

    from .modify import (
        add_slab,
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
    # THE VALUE, NOT ITS NAME.  This read `locals()[name]` against a dict of
    # parameter names, which couples a warning to the SPELLING of the
    # signature: rename the `center` parameter and `locals()["center"]` raises
    # `KeyError` (measured) -- from inside a cosmetic warning, so a rename
    # that changes nothing about the operation takes the whole command down,
    # and no test covers this path.  The electrode block below always passed
    # values directly; this is now the same shape, and neither can drift.
    _ORIENT_NONDEFAULTS = (
        ("axis",   axis,   "z"),
        ("angle",  angle,  0.0),
        ("center", center, "midpoint"),
    )
    if not op_types["--orient-axis"]:
        for name, value, default in _ORIENT_NONDEFAULTS:
            if value != default:
                click.echo(
                    f"warning: --{name} is a sub-option of --orient-axis; "
                    f"value {value!r} is ignored without --orient-axis.",
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
            # THE FLAG IS THE CONVENIENCE; `add_slab` IS THE BUILDER.
            #
            # `--electrode` says "put a slab this far from these atoms, on
            # this side" -- a genuinely useful way to ask, and the reason the
            # flag survives.  What went (2026-09-01) is the SECOND BUILDER it
            # used to call: `add_electrode_slab` placed relative to the
            # anchor itself, and mirrored the slab for `-z`, which is the
            # accidental layer-order flip `bench-and-junction-plan.md` § 2.3
            # records and the redesign set out to make unreachable.
            #
            # So the arithmetic happens HERE, where the convenience lives,
            # and the placement goes to the one builder: centroid -> an
            # absolute `start_z`, side -> `grow`, and the anchor's xy folded
            # into the absolute `offset`.  The crystal CARRIES ON outward
            # from the contact instead of reflecting, so `-z` no longer flips
            # the layer order.  In the walk vocabulary (`sequence`, 2026-09-07)
            # that is a different value per side -- read along the growth
            # direction, going up from a layer is the forward walk and going
            # down from one is the backward walk -- which is the mapping
            # `_CONTINUES_THE_CRYSTAL` below writes down once.
            offset_xy = _parse_xy_csv(electrode_offset, "--electrode-offset")
            for spec_str in electrode:
                spec = _parse_electrode_spec(spec_str)
                idx = spec["center_indices"]
                if idx:
                    for i in idx:
                        if not (0 <= i < struct.n_atoms):
                            raise click.BadParameter(
                                f"--electrode {spec_str!r}: centre index {i} "
                                f"is out of range for a {struct.n_atoms}-atom "
                                f"structure")
                    anchor = np.asarray(
                        struct.positions, dtype=float)[idx].mean(axis=0)
                else:
                    anchor = np.zeros(3, dtype=float)
                sign = 1.0 if spec["side"] == "+z" else -1.0
                struct = add_slab(
                    struct,
                    element=spec["element"],
                    plane=spec["plane"],
                    size=spec["size"],
                    start_z=float(anchor[2]
                                  + sign * spec["contact_distance"]),
                    grow=spec["side"],
                    sequence=_CONTINUES_THE_CRYSTAL[spec["side"]],
                    orthogonal=orthogonal,
                    offset=(float(anchor[0]) + offset_xy[0],
                            float(anchor[1]) + offset_xy[1]),
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
        # THE PAIR IS THE FILE (`model/structure.md` § 2.4).  This wrote the
        # geometry alone with `to_xyz`/`to_pdb`, so `modify` READ a pair
        # through the codec and wrote back half of it: a device carrying
        # `L-electrode`, `frozen_atoms` and an explicit cell came out of a
        # zero-degree rotation with none of them, at exit 0 and without a word.
        # The codec owns the pairing rule and the both-or-neither write; the
        # format is the one the user asked for, which may not be the one the
        # extension implies.
        from .workingcopy_structure import StructureCodec
        StructureCodec().write(struct, output_path, fmt=fmt)
        click.echo(
            f"Wrote {output_path}: {struct.n_atoms} atoms (input had {n_in})",
            err=True,
        )


# --------------------------------------------------------------------- #
#  run subcommand (emit a shell wrapper for a generated script)         #
# --------------------------------------------------------------------- #


@cli.command("xv2xyz",
             short_help="translate a SIESTA .XV to extended-XYZ (cell-preserving)")
@click.argument("xv_path", metavar="input.XV",
                type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("xyz_path", metavar="output.xyz", type=click.Path(path_type=Path))
def cmd_xv2xyz(xv_path: Path, xyz_path: Path) -> int:
    """Convert a SIESTA ``.XV`` final-coordinates file to extended-XYZ.

    The periodic cell is preserved on the comment line as an ASE
    ``Lattice="..."`` header (Å), so the cell travels with the structure into
    a description and reaches the deck ``jobset prep`` renders, instead of the
    geometry arriving as a molecule in a vacuum box.  This is the convenient
    ``.XV`` extraction entry; the underlying API is
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
@click.option("--interval", type=click.FloatRange(min=1.0), default=10.0,
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
    notifier surface (docs/execution/job-system.md).

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


#: Binds no remote client can reach.  `0.0.0.0` is deliberately ABSENT -- it
#: accepts from every NIC -- and so is the string `"0.0.0.0:127.0.0.1"`, which
#: sat here until 2026-09-21: it is not a host, no `--host` value can equal it,
#: and in a set that decides whether TLS is enforced it read as though
#: `0.0.0.0` were half-excused.  `127.` is matched by prefix below, not here.
_LOOPBACK_HOSTS = frozenset({
    "127.0.0.1", "localhost", "::1",
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
    public internet by mistake" guard: cleartext over a real network is
    two attacks in one (passive sniffing + active tampering), and what
    is sniffed includes the session cookie.

    **It checks host and TLS, and deliberately not auth**
    (`deployment.md` § 1).  Auth is opt-in, so the two are independent
    questions and TLS answers neither of them -- which is why the
    message below says so rather than implying a `--cert` makes this
    safe.  It said "the file-ops endpoints have no auth" until
    2026-09-21, which was true of a server with no `auth` section and
    false of one with providers configured, where every non-public
    endpoint needs a session and `/api/*` answers 401
    (`access-control.md` §§ 1.1, 2, 3.2).

    Operators who genuinely want plain HTTP on a non-loopback host
    (e.g., behind a TLS-terminating reverse proxy on the same
    machine) can pass ``--allow-insecure-binding`` to bypass; we
    still print a loud warning so the choice is visible in logs.

    See ``docs/ops/deployment.md`` for the recommended deployment
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
            f"TLS and gates auth.  See docs/ops/deployment.md.",
            err=True,
        )
        return
    raise click.UsageError(
        f"--host={host} is not a loopback address and there is no TLS, "
        f"so every request crosses the network in clear text -- the "
        f"session cookie included.\n\n"
        f"WHAT IS BEHIND IT depends on your `auth` section, which this "
        f"guard does not look at: with providers configured, every "
        f"non-public endpoint needs a session and /api/* answers 401; "
        f"with no `auth` section there is no sign-in at all, and the "
        f"projects/ tree is served read + write + delete to anyone who "
        f"can reach the interface (docs/ops/access-control.md 1.1).\n\n"
        f"For a real deployment you have three reasonable options:\n"
        f"  1. Pass --cert / --key to enable TLS.  TLS IS NOT "
        f"AUTHENTICATION -- it encrypts the wire and gates nothing; if "
        f"sign-in is not on, turn it on too (`molbuilder auth-setup`).\n"
        f"  2. Put molbuilder behind a reverse proxy that adds TLS + "
        f"auth (recommended -- see docs/ops/deployment.md).\n"
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
    """CLI flags > the machine config > (None, None).

    Reads cert/key from the machine config through
    `runtime_config.get_tls`.  ONE shape -- ``"tls": {"cert": ..., "key":
    ...}``; the flat top-level ``cert``/``key`` is refused by name
    (`runtime_config._FLAT_TLS_RETIRED`, 2026-09-02; this docstring said
    "accepted" until 2026-09-14).  A partial
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
    proxy from docs/ops/deployment.md) so the operator doesn't reach
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
          "on 127.0.0.1.  See docs/ops/deployment.md for the recommended "
          "shape.\n"
          "  * Or: copy cert + key into a directory the molbuilder "
          "user can read (mode 0600 on the key) and point "
          "molbuilder.json at the copy.  Add a renewal hook so the "
          "copy stays in sync.\n"
          "  * Or (less clean): add the molbuilder user to the group "
          "that owns the key + ``chmod g+r``.  Survives renewal "
          "iff the system installer preserves group + mode."
    )


def _refuse_an_unsafe_bind(host, cert, key, allow_insecure, no_auth):
    """Every reason this (host, TLS, auth) combination must not start.

    **One place, because `start` has to ask BEFORE it detaches.**  These three
    refusals lived only in `cmd_serve`, which `serve start` reaches as a
    CHILD -- after `daemonize()`, so the refusal went to the log while the
    terminal had already printed "starting in the background" and exit 0.
    Measured 2026-09-21: `serve start --host 0.0.0.0` with no TLS says it
    started, names a log and a pidfile, and ends with "then: molbuilder serve
    status" -- which is the failure `cmd_serve_start` records fixing for the
    PORT check, in the same words: *"the failure then landed in the log AFTER
    `daemonize()`, so nothing reached the terminal."*  That check was hoisted
    and this one was not.

    Returns the ssl context `cmd_serve` then runs with, so the resolution is
    not spelled twice (D4: two implementations of one rule drift).
    """
    if no_auth:
        # Auth-free is a LOCAL-ONLY convenience: refuse anything but a
        # loopback bind so an unauthenticated server is never reachable off
        # the machine.
        if not _is_loopback_host(host):
            raise click.ClickException(
                f"--no-auth requires a loopback --host (got {host!r}); "
                "refusing to start an unauthenticated server on a "
                "non-loopback interface.")
        return None                      # --no-auth never gets TLS
    cert, key = _resolve_tls(cert, key)
    _check_tls_readable(cert, key)
    ssl_ctx = (cert, key) if cert and key else None
    _enforce_tls_for_remote_bind(host, ssl_ctx, allow_insecure)
    return ssl_ctx


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
@click.option("--force", is_flag=True,
              help="replace the auth providers molbuilder.json already "
                   "carries.  Everything else -- in `auth` and outside it "
                   "-- survives: the write is a merge.")
def cmd_auth_setup(provider, asurite, google_email, hosted_domain, force):
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
      * The Google OAuth client secret is prompted via ``getpass``
        (hidden input, no echo, no shell history) and written to
        ``<config dir>/secrets/google_client_secret`` with mode 0600;
        molbuilder.json names that file by PATH, never the literal.
      * molbuilder.json itself is written mode 0600.

    The session key is NOT this wizard's.  The server creates
    ``<config dir>/secrets/secret_key`` on its first start and reads it from then
    on (§ 2.1e).  Until 2026-09-13 this wizard regenerated it on every run
    -- every signed-in person logged out by a command whose docstring said
    "idempotent" -- and with a different encoding from the server's own
    creator.

    Re-running is idempotent except for the Google client secret, which is
    re-prompted each run.  ``--force`` replaces the providers list and
    nothing else: the write is a merge through
    ``runtime_config.write_config_scope``, the one door for this file
    (§ 2.3), so ``auth.trust_proxy`` and every other section survive.

    Where the file lives: the machine config has ONE location, the config
    directory, and the door resolves it -- so the auth block cannot land in
    a file nothing consults.  A ``./molbuilder.json`` in the launch
    directory is not read, and is left alone.
    """
    import getpass

    from . import auth_setup as _as
    from .runtime_config import _validate_provider as _validate

    # 0. Say what is already wrong about this machine's config ---------
    #
    # BEFORE THE WIZARD ASKS ANYTHING, because all three warnings are about
    # the file it is about to write into and the directory it will write it
    # in.  This command is the one that puts provider credentials and
    # `client_secret_file` paths into `molbuilder.json`, and it printed
    # `(mode 0600)` about its own write while saying nothing about a file that
    # ARRIVED `0644` or a `./molbuilder.json` it is documented to leave alone
    # (D12).  `serve` and the jobset verbs already call this; the function's
    # own docstring named THIS command as the surface that did not.  To
    # stderr, so it reaches a person without entering piped output.
    from .placement import machine_config_warnings
    for _warning in machine_config_warnings():
        click.echo(_warning, err=True)

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
    #
    # THE WIZARD WRITES THE FILE THE READER WILL READ, through the reader's
    # own door: `write_config_scope` asks `machine_config_path` for the one
    # location, merges over what is there, validates the merge and writes
    # it 0600.  Until 2026-09-13 this command had a writer of its own
    # (`auth_setup.emit_molbuilder_json`), a third reader of the format, an
    # `--output` naming a file the server never reads, and a merge that
    # REPLACED `auth` wholesale -- so re-running it to add a provider
    # dropped `auth.trust_proxy`.  (And it defaulted to `./molbuilder.json`
    # until 2026-08-30: the git root, for anyone inside a checkout.)
    from .runtime_config import machine_config_path, write_config_scope
    output_path = machine_config_path()
    from .config_dir import google_client_secret
    google_secret_file = google_client_secret()

    # 3. Bail early on clobber unless --force --------------------------
    #
    # WHAT IS WORTH GUARDING IS A PROVIDERS LIST, NOT A FILE: the write is a
    # merge, so a seeded molbuilder.json (`envs init-config`, 2026-09-08 --
    # activation plus the comment keys) has nothing in it to clobber.  Read
    # through the server's reader: a file the server would refuse stops the
    # wizard HERE, before it asks for a secret, and there is no --force past
    # that -- overwriting a config a person could not read is how a
    # hand-edit's typo used to erase the auth providers and TLS paths (R10).
    try:
        prior = read_config()
    except RuntimeConfigError as exc:
        click.echo(f"Error: {exc}", err=True)
        click.echo("Nothing written; fix that (or move the file aside) and "
                   "run the wizard again.", err=True)
        sys.exit(2)
    prior_auth = prior.get("auth")
    if (isinstance(prior_auth, dict) and prior_auth.get("providers")
            and not force):
        click.echo(
            f"Error: {output_path} already carries auth providers.  Re-run "
            f"with --force to replace them; everything else in the file, "
            f"and in `auth`, survives.",
            err=True,
        )
        sys.exit(2)

    # 4. ASU CAS entry -------------------------------------------------
    providers: list = []
    if want_asu:
        sys_user = getpass.getuser()
        if asurite is None:
            asurite = click.prompt(
                "ASURITE (ASU username) for the CAS allowlist",
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

    # 6. Merge the auth block into the machine config ------------------
    auth_block = _as.build_auth_block(providers=providers)
    try:
        write_config_scope(None, {"auth": auth_block})
    except RuntimeConfigError as exc:
        click.echo(f"Error: not written -- {exc}", err=True)
        sys.exit(2)

    click.echo("", err=True)
    click.echo(f"Wrote {output_path} (mode 0600)", err=True)
    click.echo("", err=True)
    click.echo("Next steps:", err=True)
    click.echo(
        "  python -m molbuilder serve start --port 8888 --host 127.0.0.1",
        err=True,
    )
    click.echo(
        "  (the session key is the server's: created at "
        "<config dir>/secrets/secret_key on its first start and kept from then on)",
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
    from .parse import ParseError
    detect_parser = _trajectory_parser_for

    with _resolve_input_path(input_path) as resolved:
        try:
            parser = detect_parser(resolved)
        except ParseError as e:          # incl. AmbiguousFormatError
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
    # Through the one writer: a later parse reads this file, and a half-written
    # one is a parse error pointing at the wrong thing (I7).
    from .persist import write_bytes
    write_bytes(out, (payload + "\n").encode("utf-8"))
    click.echo(f"wrote {out}", err=True)


# The supervisor protocol lives in a leaf module that imports NOTHING, so this
# parent can read it without importing the application it restarts -- which is
# the property that lets a child failing to import leave the supervisor alive.
from .reload_protocol import RELOAD_EXIT_CODE, SUPERVISED_ENV


def _supervise_forever() -> int:
    """Run this same command as a child, and respawn it on RELOAD_EXIT_CODE.

    The shape is Werkzeug's reloader, minus the file watcher: a parent that
    NEVER IMPORTS APPLICATION CODE, whose only job is to start a child and
    start another when that one asks.  That is what makes it robust -- the
    parent cannot be broken by the code it restarts, so a child that fails to
    import leaves the supervisor alive and the next reload can fix it.

    The watcher is deliberately absent (docs/archive/2026-08-19-server-reload-plan.md § 3.1).
    Werkzeug's reloader stat-polls every imported module once a second and
    fires on ANY mtime change, so a chunked editor write or a `git checkout`
    touching fifty files reloads against a momentarily inconsistent tree and
    the child comes up importing half a module.  Here a person says when it is
    ready.

    The child is handed SUPERVISED_ENV, which does two things at once: it is
    how the child knows to run the server instead of becoming a supervisor
    itself (without it, `--supervise` would fork forever), and it is what tells
    `create_app` a restart is possible at all, so the reload route may exist.
    """
    import subprocess

    env = dict(os.environ)
    env[SUPERVISED_ENV] = "1"
    args = [sys.executable, "-m", "molbuilder", *sys.argv[1:]]
    _crashes: list = []
    while True:
        try:
            code = subprocess.call(args, env=env)
        except KeyboardInterrupt:
            # Ctrl-C reaches the whole foreground process group, so the child
            # is already stopping; the parent must not print a traceback over
            # its shutdown.  128+SIGINT is what a shell reports for this.
            # Mattered little while --supervise was opt-in; it is the default
            # path now, so every Ctrl-C goes through here.
            return 130
        if code == RELOAD_EXIT_CODE:
            click.echo("molbuilder: reload requested -- starting a fresh "
                       "server", err=True)
            continue
        if code < 0:
            # KILLED BY A SIGNAL -- the 2026-08-28 repair.  A hung child
            # that somebody killed by hand must come back; before this,
            # the supervisor read the kill as "not a reload" and quit,
            # taking the site down exactly when recovery was needed.
            # Flap-guarded through the same pure policy the daemon uses.
            from .serve_daemon import flapping
            import time as _time
            _crashes.append(_time.monotonic())
            if flapping(_crashes, _time.monotonic()):
                click.echo("molbuilder: two crashes within 30s -- giving "
                           "up rather than flapping", err=True)
                return 1
            click.echo(f"molbuilder: server died by signal {-code}; "
                       f"starting a fresh one (the hung-child repair)",
                       err=True)
            continue
        return code


@cli.command("notify-token",
              short_help="issue a run-report signing key for one user, and "
                         "say where to put it")
@click.argument("user")
@click.option("--host", default=None,
              help="the server as the JOB will reach it, e.g. "
                   "https://host:8888.  The route segment is appended for "
                   "you; used only to print the file to save on the cluster.")
@click.option("--route", default=None,
              help="reuse an existing route segment instead of generating "
                   "one.  Rarely needed: a second key READS the segment out "
                   "of the key file and joins it automatically.  Letters, "
                   "digits, '-' and '_' only -- it is one path component of "
                   "the URL the job posts to.  (Said 'pass the value already "
                   "in molbuilder.json' until 2026-09-12; `notify_route` in "
                   "config has been retired and refused since 2026-08-31.)")
@click.option("--channel", default="molbuilder",
              help="the name this listener is called on the cluster.  It is "
                   "what a description ticks, so re-issuing under the same "
                   "name replaces the credential and no description changes.")
@click.option("--replace", is_flag=True,
              help="reissue for a user who already has one.  The old key "
                   "stops working immediately.")
def cmd_notify_token(user, host, route, channel, replace):
    """Issue a signing key so one person's jobs can report progress.

    Two files, two machines, one secret:

    \b
      * THIS machine (the server) gets `notify_keys`, which carries the
        route segment and maps the user to their key.  That file IS the
        switch: the listener is registered because it exists and names a
        route, and `molbuilder.json` needs nothing at all.
      * THE CLUSTER gets `notify`, holding the named channels the monitor
        reads beside a running job.  This adds one to it.

    Both are mode 0600 and neither is ever placed in `molbuilder.json`:
    the config carries PATHS (`ops/deployment.md` 5.1).  It carried
    `notify_keys_file` and `notify_route` until 2026-08-31, which was one
    fact in two places -- a path to a file molbuilder had itself written,
    and a copy of a segment molbuilder had itself issued.  Both are retired
    and both are now refused.

    **The key signs the body and never travels.**  A bearer token -- what
    this issued until 2026-08-27 -- is on the wire on every report, so one
    capture yields a credential good forever and for any body.  A signature
    is valid for one exact body: it cannot be altered and cannot mint a new
    report (`access-control.md` § 8 rule 7).

    **The key is printed once, and that is a deliberate exception.**
    `auth-setup` never prints a secret, and is right not to -- a session
    key never leaves the server that made it.  This one is a SHARED secret
    by design: it has to reach a second machine, and the only thing that
    could carry it there without showing it to you is a channel molbuilder
    does not have.  Copy it now; it is not recoverable from the server's
    file in a form you can read back out of this command.

    **`serve` never generates or rotates these** (`run-reports.md` § 4.4).
    The counterpart lives on a cluster molbuilder cannot reach, so a
    startup rotation would leave every running job signing with the old key
    -- refused, and silently, because a notifier swallows failures by
    design.  Rotating is this command with --replace, and then copying the
    new file across.
    """
    import json as _json
    from . import auth_setup as _as
    from .monitor import (default_notify_path, is_channel_name,
                          notify_keys_path)

    if not is_channel_name(channel):
        raise click.UsageError(
            f"{channel!r} is not a channel name. Letters, digits, '-' and "
            f"'_' -- it is written into a description and rendered into the "
            f"monitor's command line.")
    # NO --keys-file.  The key file has ONE home and the server reads only
    # that one (`web/app.py` asks `read_notify_keys()` with no path), so a flag
    # naming another was a way to write a key nowhere that works, get "success",
    # and learn nothing until reports stopped arriving -- the same two-spellings
    # failure `configuration.md` 2.1e records for the session key.  Removed
    # 2026-09-12; it had no production caller, and `auth_setup.default_secret_dir`
    # had already dropped its `home=` for the same reason on 2026-08-31.  A test
    # that wants another root sets MOLBUILDER_CONFIG_DIR, like everything else.
    # ONE DOOR (`auth_setup.issue_notify_key`), shared with the This-machine
    # tab.  Issuing written twice would be free to generate a second route
    # segment from the same file and silence everyone already set up.
    try:
        token, seg, existing_route = _as.issue_notify_key(
            user, route=route, replace=replace)
    except _as.NotifyKeyError as exc:
        raise click.UsageError(str(exc))
    base = (host or "https://YOUR-SERVER:8888").rstrip("/")
    client = _json.dumps(
        {"channels": {channel: {"url": f"{base}/api/{seg}", "key": token}}},
        indent=2)

    click.echo(f"\nIssued a run-report signing key for {user}.\n")
    click.echo(f"  server side, written now : {notify_keys_path()}  (0600)")
    click.echo( "  molbuilder.json needs    : nothing.  The file above IS the "
                "switch --")
    click.echo( "                             it carries the route, and the "
                "server reads it.")
    # AND WHEN IT READS IT: once, at startup.  `web/app.py` registers the
    # listener blueprint only if the file already names a route and holds keys,
    # and nothing re-reads it afterwards -- so the FIRST key issued against a
    # running server does not work until that server restarts.  Every report
    # gets a 404 and the notifier swallows it by design, which means the job
    # succeeds and the reports are simply absent.  Said here because this is the
    # moment the person can act on it; `run-reports.md` 4.4 carries the rule.
    if not existing_route:
        click.echo("")
        click.echo( "  RESTART THE SERVER        : this is the first key, so "
                    "the route did not exist")
        click.echo( "                             when the running server "
                    "started.  Until it is")
        click.echo( "                             restarted every report gets "
                    "a 404 -- and a notifier")
        click.echo( "                             is silent on failure, so you "
                    "would see no error,")
        click.echo( "                             just no reports.")
        click.echo( "                                 molbuilder serve restart "
                    "--port <port>")
    if route and existing_route and route != existing_route:
        click.echo(f"\n  MOVED the route from {existing_route} to {seg} "
                   f"because you passed --route.")
        click.echo( "  Every key issued under the old segment stops working, "
                    "and silently.")
    elif existing_route:
        click.echo(f"\n  joined the route already in that file ({seg}), so "
                   f"everybody already set up keeps working.")
    elif route:
        click.echo(f"\n  adopted the segment you passed ({seg}); it is now "
                   f"kept in the file.")
    else:
        click.echo(f"\n  first key here, so the route segment was generated: "
                   f"{seg}")
        click.echo( "  every later key joins it automatically -- there is "
                    "nothing to pass.")
    # THE WHOLE RULE, NOT TWO THIRDS OF IT (configuration.md § 2.1c).  The
    # config directory is $MOLBUILDER_CONFIG_DIR first, exactly as given, and
    # only then the XDG pair.  These lines printed the XDG pair alone, so on
    # any machine with MOLBUILDER_CONFIG_DIR set -- which is the case this
    # session's cutover created -- following them wrote the key where the
    # monitor does not look, and silently: a notifier swallows failures by
    # design, so the job simply never reports.  The Task-setup card emits the
    # same three branches (task-setup/viewer.js); it stays shell text on both
    # surfaces because it resolves on the FAR machine, not this one.
    # DERIVED, never spelled.  This said "the config directory's `notify`"
    # and printed "$cfg/notify" below; when every credential moved into
    # `secrets/` the recipe went on telling people to write a webhook where
    # nothing reads it -- and a notifier swallows failures, so they would
    # never learn.  `relative_home` answers from the monitor's own resolver.
    from .config_dir import relative_home
    _notify_rel = relative_home(default_notify_path)
    click.echo(f"\nOn the CLUSTER this is the channel `{channel}`, in the "
               f"config directory's `{_notify_rel}`, mode 0600:\n")
    click.echo(client)
    click.echo("\n  cfg=\"${MOLBUILDER_CONFIG_DIR:-"
               "${XDG_CONFIG_HOME:-$HOME/.config}/molbuilder}\"")
    click.echo(f"  mkdir -p -m 700 \"$cfg/{Path(_notify_rel).parent.as_posix()}\"")
    click.echo(f"  # paste the JSON above into \"$cfg/{_notify_rel}\"")
    click.echo(f"  chmod 600 \"$cfg/{_notify_rel}\"")
    # MERGE, NOT OVERWRITE -- and it has to be said, because the file now
    # holds every channel rather than one destination.  Pasting over a file
    # that already has a Slack channel in it deletes that channel, and
    # silently: nothing is sent there and nothing says why.
    click.echo(f"\n  If `$cfg/{_notify_rel}` already exists, add `{channel}` "
               f"to its `channels` object")
    click.echo( "  instead of replacing the file -- pasting over it deletes "
                "the others, silently.")
    click.echo("\nOn an HPC login node $HOME is usually NFS-mounted and "
               "backed up.")
    click.echo("Setting XDG_CONFIG_HOME to somewhere local keeps the key "
               "off it.\n")
    click.echo("The key is shown ONCE. Copy it before this scrolls away.")
    click.echo(f"Then tick `{channel}` and what is worth a message on the "
               f"Task-setup tab;")
    click.echo("with nothing ticked a run still reports when it ends.\n")
    return 0


@cli.group("serve", short_help="run the browser UI (Flask + 3Dmol.js)")
def serve_group():
    """The web server, as verbs: ``start`` (background, logged, rotated),
    ``status`` / ``restart`` / ``stop`` (acting on your own instance,
    verified before signalled), and ``foreground`` (the terminal-bound
    development run).  `docs/ops/deployment.md` § 1.

    *(The bare ``molbuilder serve`` form was retired 2026-08-28 with the
    group -- a verb is named, never implied.)*
    """


#: The port every `serve` and `jupyter` verb defaults to.  One home, because
#: it is the ADDRESS those verbs act at: a default that differs between two
#: of them sends a stop to a server that is not the one just started.
_DEFAULT_SERVE_PORT = 8000


def _serve_port_flag(f):
    """``--port`` for a verb that ACTS ON a running molbuilder.

    `_serve_flags` below already existed for the same reason -- *"one
    decorator, so the two verbs cannot drift apart"* -- and seven other verbs
    hand-wrote this option anyway, three hundred lines down (`plan.md` § 5n,
    J9).  **They had already drifted**: two carried help text that disagreed
    (*"the SERVE port -- the notebook's own is that plus one"* against *"the
    SERVE port this notebook belongs to"*) and three carried none at all, so
    `molbuilder jupyter status --help` explained nothing while its sibling
    did.
    """
    return click.option(
        "--port", type=int, default=_DEFAULT_SERVE_PORT, show_default=True,
        help="the port molbuilder serves on -- the address this verb acts "
             "at.")(f)


def _serve_port_filter(f):
    """``--port`` for a verb that REPORTS -- optional, because *"which port?"*
    is the question the person is asking.

    Distinct from `_serve_port_flag` on purpose: an acting verb must be told
    exactly which server to stop, while a reporting one defaulting to 8000
    told somebody running on 8888 that nothing was running -- confidently
    wrong, with the port sitting on disk the whole time (`plan.md` § 5n, J14,
    the user's own).
    """
    return click.option(
        "--port", type=int, default=None,
        help="one server to report on.  Omit it and every server with a "
             "pidfile is surveyed.")(f)


def _serve_flags(f):
    """The flags ``foreground`` and ``start`` share -- one decorator, so
    the two verbs cannot drift apart about what a server accepts."""
    for opt in reversed([
        click.option("--host",  default="127.0.0.1", show_default=True),
        # THE SAME OPTION THE OTHER SEVEN VERBS TAKE.  J9 unified those and
        # left this one -- the eighth copy, inside the decorator whose own
        # docstring is "so the two verbs cannot drift apart", and the only
        # one with no `help=`, so `serve start --help` explained nothing
        # about `--port` while `serve stop --help` did (found in review
        # 2026-09-15; the commit that unified the seven claimed this was
        # covered, and it was not).
        _serve_port_flag,
        click.option("--cert", type=click.Path(exists=True, dir_okay=False),
                     help="TLS cert (PEM).  Overrides molbuilder.json."),
        click.option("--key",  type=click.Path(exists=True, dir_okay=False),
                     help="TLS key (PEM).  Overrides molbuilder.json."),
        click.option("--allow-insecure-binding", is_flag=True,
                     help="Bypass the loopback-or-TLS guard.  Only sensible "
                          "when something outside molbuilder (proxy / VPN) "
                          "gates access -- see docs/ops/deployment.md."),
        click.option("--no-auth", is_flag=True,
                     help="Run with NO authentication (ignores "
                          "molbuilder.json's auth/TLS).  Loopback hosts "
                          "only."),
    ]):
        f = opt(f)
    return f


#: Seconds a TLS handshake may take before the connection is dropped.
#: Generous for a real client on a slow link; fatal to one that never speaks.
HANDSHAKE_TIMEOUT_S = 10.0


def _harden_tls_accept_loop() -> None:
    """Take the TLS handshake OFF the accept loop, for every TLS server here.

    **The outage this exists for.** `socketserver.TCPServer.get_request` is
    ``return self.socket.accept()``, and under TLS that socket is an
    ``ssl.SSLSocket`` whose ``accept()`` also performs the HANDSHAKE -- inside
    the accept loop.  A client that completes the TCP connection and then says
    nothing holds that call open with no timeout, and nobody else is ever
    accepted.

    Measured on the 8888 dev server, 2026-09-05: up 8h50m on **7 seconds of
    CPU**, 52 connections queued and never accepted, and its own request log
    silent for seven hours.  ``SIGUSR1`` (`faulthandler`, registered in
    ``cmd_serve``) put the main thread in ``ssl.py:do_handshake``.  The
    connections holding it were internet scanners; the server binds ``0.0.0.0``.

    **Authentication cannot reach this.**  The handshake precedes every byte of
    HTTP, so no request exists, no route runs, and `web/rate_limit.py`'s IP
    blocklist -- a ``before_request`` hook -- never sees the connection.  One
    TCP connection, no credentials, whole server.

    So ``get_request`` accepts a RAW socket with a deadline on it, and
    ``finish_request`` -- which ``ThreadingMixIn`` runs in the WORKER thread --
    does the TLS wrap.  A silent client now costs one thread for
    ``HANDSHAKE_TIMEOUT_S`` and costs the accept loop nothing.

    Bounding the handshake *in place* was the first attempt and is not enough:
    it turns an unbounded hang into an N x timeout stall, because the loop still
    waits for each silent client in turn.  Three were enough to time out an
    ordinary request in the test.

    Patched onto Werkzeug's server CLASS rather than replacing ``app.run``,
    because ``app.run`` is the seam the TLS tests capture
    (``capture_flask_run``) -- bypassing it silently un-tested the ssl_context
    wiring and hung the suite.
    """
    import socket as _socket
    import ssl as _ssl

    from werkzeug.serving import ThreadedWSGIServer

    if getattr(ThreadedWSGIServer, "_molbuilder_tls_hardened", False):
        return

    def get_request(self):
        if not isinstance(self.socket, _ssl.SSLSocket):
            return self.socket.accept()          # plain HTTP: unchanged
        # The RAW accept.  SSLSocket.accept would wrap-and-handshake here.
        raw, addr = _socket.socket.accept(self.socket)
        raw.settimeout(HANDSHAKE_TIMEOUT_S)
        return raw, addr

    def finish_request(self, request, client_address):
        listener = self.socket
        if not isinstance(listener, _ssl.SSLSocket):
            self.RequestHandlerClass(request, client_address, self)
            return
        try:
            conn = listener.context.wrap_socket(
                request, server_side=True, do_handshake_on_connect=True,
                suppress_ragged_eofs=listener.suppress_ragged_eofs)
        except OSError:
            return          # timed out, or spoke something that is not TLS
        try:
            conn.settimeout(None)      # the REQUEST is blocking, as it was
            self.RequestHandlerClass(conn, client_address, self)
        finally:
            # `wrap_socket` DETACHES the raw socket, so socketserver's own
            # `shutdown_request(request)` no longer owns the fd.  Close ours.
            try:
                conn.shutdown(_socket.SHUT_WR)
            except OSError:
                pass
            conn.close()

    ThreadedWSGIServer.get_request = get_request
    ThreadedWSGIServer.finish_request = finish_request
    ThreadedWSGIServer._molbuilder_tls_hardened = True


@serve_group.command("foreground",
                     short_help="the terminal-bound run (dev): supervisor "
                                "+ child in this shell, Ctrl-C to stop")
@_serve_flags
@click.option("--debug", is_flag=True)
@click.option("--supervise/--no-supervise", default=True, show_default=True,
              help="Run under a parent that can restart the server in place.  "
                   "This is what makes the admin Reload button exist; without "
                   "it that route is absent, because nothing would bring the "
                   "server back.  Turn it OFF when something else already owns "
                   "restarts (systemd, Docker, gunicorn) -- a supervisor inside "
                   "one of those is a second answer to a question already "
                   "answered.  --debug turns it off on its own.  "
                   "See docs/archive/2026-08-19-server-reload-plan.md.")
def cmd_serve(host, port, debug, cert, key, allow_insecure_binding, no_auth,
              supervise):
    """Run the server in THIS terminal -- the development mode."""
    # WHERE THE TREE IS, before anything starts.  The server is the surface a
    # person puts calculations INTO, so it must not be the one surface that
    # leaves them guessing which tree they are filling (user, 2026-09-12).
    # Echoed in the parent, before the supervisor fork, so it appears once.
    from .projects import projects_root_with_source
    click.echo(f"molbuilder serve: {projects_root_with_source().describe()}")
    # AND WHAT IS WRONG WITH THE CONFIG IT JUST READ.  `serve` was not a caller
    # of either warning until 2026-09-12 -- only the jobset verbs and
    # `config_provenance` were -- so a `molbuilder.json` that ARRIVED loose
    # (copied from another machine, restored from a backup, unpacked without
    # modes) stayed world-readable with its `tls.key` path and provider
    # credentials in it, and the server that read it said nothing.  § 2.1b exists
    # for exactly the cases no writer can control.  To stderr, so it reaches a
    # person without entering piped output.  ONCE: this function runs in the
    # supervisor and again in the child it re-execs (SUPERVISED_ENV=1), and
    # printed the warnings both times until 2026-09-13 (K-L5).
    if os.environ.get(SUPERVISED_ENV) != "1":
        from .placement import machine_config_warnings
        for _warning in machine_config_warnings():
            click.echo(_warning, err=True)

    # NO APPLICATION IMPORT ABOVE THE PARENT BRANCH.  ``from .web.app import
    # create_app`` used to sit here, one line into the function and well before
    # the fork below -- so the supervisor imported the entire web app, Flask
    # included, before it ever spawned anything.  That is the one thing the
    # supervisor must not do: its whole value is that a child which fails to
    # import leaves the parent alive to be fixed and reloaded, and a parent
    # that imported the same broken module first dies with it.  The import now
    # happens in the child, below, after the parent has already returned.

    # --debug hands the process tree to Werkzeug's reloader, which forks its
    # own child and respawns it on ANY exit -- including the sentinel the
    # reload route uses to ask for a fresh server.  The request would never
    # reach our supervisor, so the two cannot both be in charge.  Debug is the
    # single-process, restart-it-yourself mode; say so rather than starting a
    # supervisor whose button would quietly not work.
    if supervise and debug:
        supervise = False
        click.echo("molbuilder: --debug owns the process tree, so this server "
                   "runs unsupervised and the Reload button is absent.",
                   err=True)

    # THE PARENT BRANCH, and it returns without importing the app: when
    # supervision is on and this process is not already the child, this process
    # becomes the supervisor and nothing else here runs.
    if supervise and os.environ.get(SUPERVISED_ENV) != "1":
        raise SystemExit(_supervise_forever())

    # From here down we ARE the server -- either the supervised child or an
    # unsupervised run -- so importing the app is what we are for.

    # THE STACK-DUMP HOOK (deployment.md 1.0c; the 2026-08-28 wedge left
    # nothing to read).  `kill -USR1 <this pid>` appends every thread's
    # stack to the stacks log, so the next hang is diagnosed from a file
    # instead of theorized from thread counts.  Registered before the app
    # import; the handle is kept for the life of the process.
    try:
        import faulthandler
        import signal as _signal
        from .config_dir import ensure_private_dir, serve_stacks_log
        from .serve_daemon import open_private
        _sp = serve_stacks_log(port)
        # 0700 around it and 0600 on it, through the same doors the supervisor
        # uses.  A bare `mkdir` + `open(_sp, "a")` here landed 0775/0664 on a
        # file that holds thread stacks of a process carrying a provider's
        # `client_secret` -- and `configuration.md` § 3.1 said 0600 while this
        # line made it otherwise (A2).  Under --no-supervise/--debug no LogRoll
        # runs, so nothing tightened the directory either.
        ensure_private_dir(_sp.parent, tighten=True)
        globals()["_STACKS_FH"] = open_private(_sp, "a")
        faulthandler.register(_signal.SIGUSR1, file=globals()["_STACKS_FH"],
                              all_threads=True)
    except (OSError, ValueError, AttributeError):
        pass          # a box without SIGUSR1 or a read-only home still serves

    from .web.app import create_app

    # THE SAME PREFLIGHT `serve start` RUNS BEFORE IT DETACHES.
    ssl_ctx = _refuse_an_unsafe_bind(host, cert, key,
                                     allow_insecure_binding, no_auth)

    if no_auth:
        # create_app(config={}) is the supported no-auth seam (see
        # web/app.py:create_app); it ignores molbuilder.json entirely (no
        # providers -> no login), and the projects root still resolves from
        # the CWD.
        app = create_app(config={})
        # THIS PROCESS'S PORT, on the Flask app rather than through
        # `create_app(config=)` -- that argument is the RUNTIME config
        # dict and `{}` there is load-bearing.  `web.app.serve_port`
        # reads this; parsing it back out of `Host:` gave the wrong
        # answer behind a proxy.
        app.config["MOLBUILDER_SERVE_PORT"] = port
        click.echo(
            f"molbuilder web UI (NO AUTH -- loopback only) starting at "
            f"http://{host}:{port}", err=True)
        app.run(host=host, port=port, debug=debug, ssl_context=None)
        return

    scheme  = "https" if ssl_ctx else "http"
    app = create_app()
    # This process's port -- see `web.app.serve_port`.
    app.config["MOLBUILDER_SERVE_PORT"] = port
    click.echo(f"molbuilder web UI starting at {scheme}://{host}:{port}", err=True)
    _print_oauth_redirect_hint_if_auth_on(scheme, host, port)
    if ssl_ctx is not None:
        _harden_tls_accept_loop()
    app.run(host=host, port=port, debug=debug, ssl_context=ssl_ctx)


@serve_group.command("start",
                     short_help="run the server in the BACKGROUND: "
                                "detached, logged with rotation, "
                                "addressable by stop/restart/status")
@_serve_flags
@click.option("--log-max-mb", type=int, default=20, show_default=True,
              help="rotate the log past this size; the full file is "
                   "gzipped and archives shift up")
@click.option("--log-keep", type=int, default=5, show_default=True,
              help="how many gzipped archives survive; the oldest is "
                   "deleted")
def cmd_serve_start(host, port, cert, key, allow_insecure_binding, no_auth,
                    log_max_mb, log_keep):
    """Detach, then run the same supervisor+child pair ``foreground``
    runs.  `deployment.md` 1.0a: pidfile and logs under YOUR OWN home,
    which is what makes every bit of this per-user."""
    from .config_dir import serve_log, serve_pidfile
    from .serve_daemon import daemonize, pid_state, read_pid, supervise
    # SAID HERE TOO, because this verb detaches.  `foreground` prints these
    # where the person is watching; `start` hands the terminal back, and its
    # child's stderr goes into the log -- so a warning emitted only by the child
    # is one nobody reads until they go looking for why sign-in broke.
    from .placement import machine_config_warnings
    for _warning in machine_config_warnings():
        click.echo(_warning, err=True)
    state = pid_state(read_pid(port))
    if state == "ours":
        raise click.ClickException(
            f"already running (pid {read_pid(port)}, port {port}).  "
            f"`molbuilder serve status` to look, `serve restart` to "
            f"recycle it.")
    # AND IS THE PORT ACTUALLY FREE?  The pidfile answers "is a supervisor of
    # mine there", which is not the same question (`plan.md` § 5n.8).
    #
    # `kill -9` of a supervisor leaves the server CHILD reparented and still
    # holding the port -- the child has no PDEATHSIG, and the supervisor's
    # cleanup never ran, so the pidfile reads "dead" and this verb happily
    # detached into a child that could not bind.  The failure then landed in
    # the log AFTER `daemonize()`, so nothing reached the terminal and the
    # person was left doing exactly what `serve status` had told them to.
    #
    # Checked here because this is the last moment anything reaches them --
    # the same reason the notebook's clash warning is printed just below.
    # A REFUSAL rather than a warning: a web server that cannot bind has
    # nothing left to do, unlike a notebook whose server is still useful.
    # AND WILL THE CHILD EVEN AGREE TO SERVE THIS BIND?  Same reason as the
    # port check below, and the same failure it records: every refusal in
    # `cmd_serve` is raised in the CHILD, which runs after `daemonize()`, so
    # without this the terminal reads "starting in the background ... then:
    # molbuilder serve status" at exit 0 while the server never came up
    # (measured 2026-09-21, `--host 0.0.0.0` with no TLS).
    _refuse_an_unsafe_bind(host, cert, key, allow_insecure_binding, no_auth)
    _held = _port_in_use(host, port)
    if _held:
        raise click.ClickException(
            f"port {port} is already in use on {host}, and no supervisor of "
            f"yours holds it ({_held}).\n"
            f"  Most likely an orphaned server child -- `kill -9` of a "
            f"supervisor leaves one running, because the child has no "
            f"PDEATHSIG and the pidfile is removed.\n"
            f"  Find it with `ss -ltnp | grep :{port}` and stop it, or "
            f"start on another port.")
    child = [sys.executable, "-m", "molbuilder", "serve", "foreground",
             "--host", host, "--port", str(port), "--no-supervise"]
    if cert:
        child += ["--cert", cert]
    if key:
        child += ["--key", key]
    if allow_insecure_binding:
        child += ["--allow-insecure-binding"]
    if no_auth:
        child += ["--no-auth"]
    click.echo(f"molbuilder serve: starting in the background on port "
               f"{port}")
    from .projects import projects_root_with_source
    click.echo(f"  {projects_root_with_source().describe()}")
    click.echo(f"  log:     {serve_log(port)}  (cap {log_max_mb} MB, "
               f"keep {log_keep} archives)")
    click.echo(f"  pidfile: {serve_pidfile(port)}")
    click.echo("  then:    molbuilder serve status")
    _logout = _runtime_dir_dies_at_logout()
    if _logout:
        click.echo(_logout, err=True)
    # THE NOTEBOOK'S COMMAND LINE, built here and handed over -- the
    # supervisor holds a list of strings and imports nothing of the
    # application (docs/web/jupyter.md § 3.4).  Nothing starts yet: the tab
    # or `molbuilder jupyter start` asks for it.
    #
    # RESOLVED TLS, not the raw flags.  The server CHILD resolves its own
    # (`cmd_serve`'s `_resolve_tls`, which also reads the `tls` block in
    # molbuilder.json); this path never did, so a machine that configures TLS
    # in the file rather than on the command line served the page over https
    # and started the notebook over http.  The tab builds the frame URL from
    # `location.protocol`, so it then asked https of a plain-http server and
    # the frame died as mixed content.  The two schemes must agree, and this
    # is where they are made to (found in review 2026-09-14).
    _nb_cert, _nb_key = _resolve_tls(cert, key)
    from .jupyter import port_clash, shepherd_argv
    notebook = shepherd_argv(port, host=host, cert=_nb_cert, key=_nb_key)
    # SAY IT BEFORE DETACHING.  This is the last moment anything reaches the
    # terminal, and a clash is knowable now (`jupyter.port_clash`).  A
    # WARNING, not a refusal: the web server on this port is perfectly
    # startable, and only its notebook is doomed -- refusing the whole verb
    # would be deciding for somebody who may not want a notebook at all.
    _clash = port_clash(port)
    if _clash:
        click.echo(f"  NOTE:    {_clash}", err=True)
    daemonize()
    # from here we are the detached supervisor; nothing prints to the
    # terminal again -- the roll owns every later byte
    raise SystemExit(supervise(
        port, child,
        log_max_bytes=log_max_mb * 1024 * 1024, log_keep=log_keep,
        jupyter_argv=notebook))


# --------------------------------------------------------------------- #
#  jupyter -- the notebook tab's process, from a terminal                 #
# --------------------------------------------------------------------- #

@cli.group("jupyter", short_help="the notebook server behind the JupyterNB tab")
def jupyter_group():
    """Start, stop and inspect the notebook server.

    **The tab's buttons do the same thing through the same door**, so a
    notebook that will not start is debuggable without a browser --
    `docs/web/jupyter.md` § 5, the same reason `serve` has these verbs.

    The server is parented to the molbuilder SUPERVISOR, not to the web
    server, so a code reload does not destroy notebook state and
    `serve stop` takes the notebook with it, and Jupyter's own cleanup
    takes the kernels with that.  These verbs
    signal that supervisor; without one (`serve foreground`,
    `--no-supervise`) there is nobody to hold the notebook and they say so.
    """


#: What a supervisor that PREDATES the notebook feature cannot do, and the
#: only thing that gives it the ability.  A supervisor survives a code reload
#: by design (`jupyter.md` § 3.4), so this is an ordinary state on a machine
#: that has been running molbuilder since before the feature landed.
_PREDATES_NOTE = (
    "  If the notebook does not come up, this supervisor may PREDATE the\n"
    "  notebook feature -- it survives a code reload, so `molbuilder serve\n"
    "  stop` then `serve start` is what gives it one.  `molbuilder serve\n"
    "  status` names the server log, which says `notebook: not configured\n"
    "  for this server` in that case.")


def _port_in_use(host: str, port: int) -> str:
    """``""`` when the port is free to bind, else why it is not.

    A bind test on the address the CHILD will use, so the answer is the
    child's: a wildcard bind and a loopback bind fail differently, and
    guessing from `127.0.0.1` would pass a port held on another interface.
    """
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
        return ""
    except OSError as exc:
        return str(exc)


def _runtime_dir_dies_at_logout() -> str:
    """``""``, or a sentence about a daemon that will become unaddressable.

    `runtime_dir()` is ``$XDG_RUNTIME_DIR/molbuilder`` when that variable is
    set, and its own docstring already names the tension: that directory is
    *"cleared when the session ends, which is right for a pidfile and wrong
    for anything meant to outlive a logout."*  `serve start` DETACHES, so it
    is by definition meant to outlive the terminal.

    What happens without this: log out of your last session and logind
    removes the directory.  The daemon and its notebook keep running -- both
    are `setsid`-detached -- but every verb loses its address.  `serve
    status` says nothing is running, `serve stop` says there is no pidfile,
    and the tab shows the no-supervisor state while a notebook with live
    kernels is up.  Reconciliation cannot find them either: it reads the
    pidfile that is gone.  The kernels can then only be killed by hand,
    which on a GPU box holds a device (found in review 2026-09-15,
    `plan.md` § 5n.8).

    Quiet when lingering is on, because then the directory survives and
    there is nothing to say.  Quiet too when `loginctl` is absent -- a
    machine without logind is not the case this warns about.
    """
    import os
    import subprocess
    rt = os.environ.get("XDG_RUNTIME_DIR", "")
    if not rt.startswith("/run/user/"):
        return ""                       # not session-scoped; nothing to say
    try:
        out = subprocess.run(
            ["loginctl", "show-user", str(os.getuid()), "-p", "Linger"],
            capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return ""                       # no logind: not this case
    if "Linger=yes" in out:
        return ""
    return (f"note: the pidfile lives under {rt}, which your session manager "
            f"removes at your LAST logout.\n"
            f"  The server and its notebook keep running, but every verb "
            f"loses its address -- `serve status` will say nothing is "
            f"running while kernels are still up.\n"
            f"  `loginctl enable-linger` keeps the directory, or set "
            f"XDG_RUNTIME_DIR to somewhere that outlives the session.")


def _jupyter_signal(port: int, sig: int, verb: str) -> None:
    """Ask the supervisor.  Verified before signalled, like `serve` does.

    **A DELIVERED SIGNAL IS NOT A DONE DEED.**  `signal_supervisor` answers
    for the delivery; the handler runs in another process and, if that
    supervisor holds no notebook argv, writes *"notebook: not configured for
    this server"* into the SERVE log and returns.  So this used to print
    "asked the supervisor to start the notebook" and exit 0 for a start that
    could never happen, and `jupyter status` then prescribed the same
    command again -- a loop with no diagnosis, while the tab explained the
    identical condition perfectly well (found in review 2026-09-15,
    `plan.md` § 5n.8).  Saying it on the way IN is the cheap half; the other
    half is `cmd_jupyter_status` not recommending this verb as the remedy.
    """
    from .serve_daemon import signal_supervisor
    ok, msg = signal_supervisor(port, sig)
    if not ok:
        raise click.ClickException(
            f"cannot {verb} the notebook: {msg}.\n"
            f"  The notebook server is held by `molbuilder serve`'s "
            f"supervisor; start one with `molbuilder serve start --port "
            f"{port}`.")
    click.echo(f"asked the supervisor to {verb} the notebook: {msg}")
    if sig_starts_a_notebook(sig):
        click.echo(_PREDATES_NOTE, err=True)


def sig_starts_a_notebook(sig) -> bool:
    """Is this the START signal?  Only a start can fail the way above."""
    import signal as _signal
    return sig == _signal.SIGUSR1


@jupyter_group.command("start", short_help="start the notebook server")
@_serve_port_flag
def cmd_jupyter_start(port):
    """Ask the supervisor to start the notebook server (idempotent)."""
    import signal as _signal
    _jupyter_signal(port, _signal.SIGUSR1, "start")
    click.echo("  then:  molbuilder jupyter status --port %d" % port)


@jupyter_group.command("stop", short_help="stop the notebook server; its kernels go with it")
@_serve_port_flag
def cmd_jupyter_stop(port):
    """Stop the notebook server AND its kernels.

    The stop takes the shepherd's process group, which is the shepherd, the
    manager's `run` and jupyter-server.  The kernels go because the SERVER
    goes -- Jupyter collects its own (`jupyter.md` § 3.2).
    """
    import signal as _signal
    _jupyter_signal(port, _signal.SIGUSR2, "stop")


@jupyter_group.command("restart", short_help="stop it, then start it again")
@_serve_port_flag
def cmd_jupyter_restart(port):
    """Stop and start.  **Every kernel dies** -- a notebook's variables are in
    the kernel, so this is not the harmless verb `serve restart` is.

    WAITS FOR THE STOP TO LAND before asking for the start.  It used to sleep
    a flat 1.0 s, which is exactly the shepherd's own minimum teardown, so the
    start signal routinely arrived while the supervisor was still inside
    `_stop_jupyter`; `_start_jupyter` then saw the old shepherd still alive,
    answered "already running", and the stop that followed left nothing at
    all.  The verb silently degraded to `stop` (measured 2026-09-14).
    """
    import signal as _signal
    import time as _time
    from .jupyter import read_pid, pid_state
    # WAIT ON THE PROCESS, NOT ON ITS PIDFILE.
    #
    # The pidfile stopped being a liveness signal in the same commit that
    # added this wait: `_on_stop` now unlinks it FIRST and only then begins
    # the `_STOP_GRACE_S` teardown, so the file is gone within milliseconds
    # while the shepherd lives on for five more seconds.  Waiting on the file
    # therefore returned at once and the start signal landed mid-teardown --
    # reproducing the exact degradation this verb was fixed for.  Two fixes
    # in one commit, each undoing the other (found in review 2026-09-14).
    #
    # The pid read BEFORE the stop is the honest handle: `pid_state` answers
    # "ours" only while that process is alive and really is a shepherd.
    doomed = read_pid(port)
    _jupyter_signal(port, _signal.SIGUSR2, "stop")
    if doomed is not None:
        # `while/else` would have fired its else-branch when nothing was
        # running at all (empty loop, no break) and announced a stop that
        # had not failed -- so the "was anything there" question is asked
        # once, here, rather than folded into the loop condition.
        deadline = _time.monotonic() + 20.0
        while _time.monotonic() < deadline:
            if pid_state(doomed) != "ours":
                break
            _time.sleep(0.25)
        else:
            click.echo("  the notebook did not stop within 20s; not starting "
                       "a second one.  `molbuilder jupyter status --port %d`"
                       % port)
            return
    _jupyter_signal(port, _signal.SIGUSR1, "start")


@jupyter_group.command("status",
                       short_help="is it up, is it ANSWERING, where")
@_serve_port_filter
def cmd_jupyter_status(port):
    """Two questions, answered separately -- `deployment.md` § 1.0b's rule:
    the failure worth catching is a process that is up and not answering.

    **With no `--port`, every notebook is surveyed** -- the same reason
    `serve status` does (`plan.md` § 5n, J14).  A notebook is keyed by the
    SERVE port it belongs to, and that number is on disk; defaulting to 8000
    reported *"not running"* to somebody whose notebook was up beside a
    server on another port.
    """
    from .config_dir import jupyter_log
    from .jupyter import status

    if port is None:
        raise SystemExit(_survey_notebooks())

    st = status(port)
    if not st["running"]:
        if st["pid_state"] == "foreign":
            click.echo("not running -- and the pidfile is stale: its pid is "
                       "not a notebook of yours")
        else:
            click.echo("not running")
        click.echo(f"  start it:  molbuilder jupyter start --port {port}")
        # BOTH LOGS, NAMED WHEN IT IS NOT RUNNING -- which is exactly when a
        # person needs them, and they hold DIFFERENT failures.  Everything
        # `_start_jupyter` cannot do (no argv, the log would not open, the
        # spawn raised) is in the SERVE log; everything jupyter-server itself
        # refuses -- a taken port above all -- is in the NOTEBOOK log.  This
        # named only the second, which is empty in the commonest case, and
        # then offered the start verb again as the remedy (`plan.md` § 5n.8).
        from .config_dir import serve_log
        nb_log = jupyter_log(port)
        if nb_log.exists():
            click.echo(f"  last log:  {nb_log}")
        click.echo(f"  and:       {serve_log(port)}  (why a start did not "
                   f"take)")
        click.echo(_PREDATES_NOTE)
        _say_which_notebooks_else(port)
        raise SystemExit(1)
    click.echo(f"running    pid {st['pid']}, port {st['port']}")
    click.echo("answering  " + ("yes" if st["answering"] else
                                "NO -- it is up but not serving; see the log"))
    click.echo(f"log        {jupyter_log(port)}")
    # WHAT IS OPEN -- paid for and then withheld until 2026-09-15.  `status`
    # asks Jupyter for its sessions whenever `include_private` is on, so this
    # verb was already making the HTTP round trip and printing none of it;
    # the tab shows the same fact (`plan.md` § 5n.8).
    for _nb in (st["open"] or []):
        click.echo(f"open       {_nb.get('path', '?')}"
                   f"   ({_nb.get('state') or 'unknown'})")
    # THE TOKEN IS NOT PRINTED.  It authenticates a browser to a live kernel,
    # which is code execution as this account; the tab reads it server-side.
    if st["url"]:
        click.echo(f"url        {st['url']}  (the tab adds the token)")
    raise SystemExit(0 if st["answering"] else 1)


def _live_notebook_ports():
    """The serve ports that have a notebook of ours running."""
    from .config_dir import ports_with_pidfile
    from .jupyter import pid_state, read_pid
    return [p for p in ports_with_pidfile("jupyter")
            if pid_state(read_pid(p)) == "ours"]


def _say_which_notebooks_else(missing_port: int) -> None:
    """After a miss on one port, name the notebooks that ARE there."""
    others = [p for p in _live_notebook_ports() if p != missing_port]
    if others:
        click.echo("  but a notebook IS running beside the server on port "
                   + ", ".join(str(p) for p in others)
                   + " -- `molbuilder jupyter status` with no --port surveys "
                     "them all.")


def _survey_notebooks() -> int:
    """Every notebook with a pidfile, one line each.  Returns the exit code.

    0 when at least one answers, 1 otherwise -- this verb's own two codes,
    not `serve status`'s three, because that is what its single-port path
    already returns and a survey must not invent a third vocabulary.
    """
    from .config_dir import jupyter_log
    from .jupyter import status
    ports = _live_notebook_ports()
    if not ports:
        click.echo("no notebook server is running.")
        click.echo("  start one:  molbuilder jupyter start --port <serve "
                   "port>   (or the JupyterNB tab's button)")
        return 1
    answered = False
    for p in ports:
        st = status(p, include_private=False)
        answered = answered or bool(st["answering"])
        click.echo(f"serve {p:<6} notebook pid {st['pid']:<8} "
                   f"port {st['port']:<6} "
                   f"answering: {'yes' if st['answering'] else 'NO'}   "
                   f"log {jupyter_log(p)}")
    click.echo("  detail on one:  molbuilder jupyter status --port <port>")
    return 0 if answered else 1


@jupyter_group.command("_shepherd", hidden=True)
@click.option("--serve-port", type=int, required=True)
@click.option("--host", default="127.0.0.1")
@click.option("--cert", default=None)
@click.option("--key", default=None)
def cmd_jupyter_shepherd(serve_port, host, cert, key):
    """NOT FOR PEOPLE -- the process the supervisor launches.

    Hidden because running it by hand gets the parentage wrong: started from a
    shell it is parented to that shell, so `PR_SET_PDEATHSIG` fires when the
    shell exits and the notebook dies with your terminal.  `jupyter start` is
    the verb.
    """
    from .jupyter import run_shepherd
    raise SystemExit(run_shepherd(serve_port, host=host, cert=cert, key=key))


def _probe_health(port: int, *, timeout: float = 5.0):
    """Does the molbuilder on ``port`` ANSWER?  -> ``(bool, one line)``.

    THE SECOND QUESTION (`deployment.md` § 1.0b): the 2026-08-28 wedge was a
    server that was UP and not ANSWERING, and a status conflating the two
    calls that healthy.

    Loopback, either scheme, verification off -- a cert made for the public
    name fails on `127.0.0.1`, and this asks about liveness, not identity
    (`serve_daemon.unverified_ctx` owns that knob).

    One home because the survey and the single-port report must not answer
    this two ways.  The timeout only ever bites on a server that is actually
    wedged: a refused connection returns at once and a live one answers at
    once, so surveying several costs nothing extra unless one of them is the
    case worth waiting for.
    """
    import urllib.error
    import urllib.request

    from .serve_daemon import unverified_ctx
    ctx = unverified_ctx()
    for scheme in ("https", "http"):
        try:
            with urllib.request.urlopen(
                    f"{scheme}://127.0.0.1:{port}/api/health",
                    timeout=timeout,
                    context=ctx if scheme == "https" else None):
                return True, f"yes ({scheme}, /api/health)"
        except urllib.error.HTTPError:
            return True, (f"yes ({scheme}; /api/health refused, which is "
                          f"still an answer)")
        except (urllib.error.URLError, OSError, TimeoutError):
            continue
    return False, (f"NO -- the process is up but /api/health gave nothing "
                   f"within {timeout:g}s")


def _note_wedge(port: int, pid: int) -> None:
    """Append the detection to the server's own log.

    The log is the record of concerns and detections (`deployment.md`
    § 1.0c, user ruling 2026-08-28) -- a wedge belongs there too, not only in
    whichever terminal happened to ask.  A one-line append beside the
    daemon's own writes; the rare rotation race can cost this line at worst,
    never a daemon byte.
    """
    import time as _time

    from .config_dir import ensure_private_dir, serve_log
    try:
        # The same doors the daemon uses.  When `status` is the FIRST writer --
        # a box where the server has never started -- a bare mkdir + append
        # created the log 0664 in a 0775 directory, and a later supervisor
        # start tightened both, so the window was "until one runs" (I5).
        from .serve_daemon import open_private
        ensure_private_dir(serve_log(port).parent, tighten=True)
        with open_private(serve_log(port), "ab") as fh:
            fh.write((f"[serve-status] "
                      f"{_time.strftime('%Y-%m-%dT%H:%M:%S%z')} "
                      f"DETECTED: process up (pid {pid}) but /api/health "
                      f"gave nothing within 5s\n").encode())
    except OSError:
        pass                     # the terminal report above still stands


@serve_group.command("status", short_help="is it up, is it ANSWERING, where")
@_serve_port_filter
def cmd_serve_status(port):
    """Two questions, answered separately (`deployment.md` 1.0b): the
    2026-08-28 wedge was a server that was UP and not ANSWERING, and a
    status that conflates the two calls that healthy.

    **With no `--port`, every server is surveyed.**  The port is on disk --
    `runtime_dir()` holds one `serve-<port>.pid` per server -- so a status
    that has to be told which port can only ever confirm a guess.
    """
    from .config_dir import serve_log, serve_pidfile
    from .serve_daemon import pid_state, read_pid

    if port is None:
        raise SystemExit(_survey())

    pid = read_pid(port)
    state = pid_state(pid)
    if state == "dead":
        if pid is not None:
            click.echo(f"not running -- stale pidfile at {serve_pidfile(port)} "
                       f"(pid {pid} is gone)")
        else:
            click.echo(f"not running (no pidfile at {serve_pidfile(port)})")
        _say_where_else(port)
        raise SystemExit(3)
    if state == "foreign":
        click.echo(f"pidfile names pid {pid}, which is NOT your molbuilder "
                   f"serve -- stale file, recycled pid.  Nothing to act on.")
        _say_where_else(port)
        raise SystemExit(3)
    click.echo(f"process:   up (supervisor pid {pid})")
    click.echo(f"log:       {serve_log(port)}")
    ok, said = _probe_health(port)
    click.echo(f"answering: {said}")
    # A SUPERSET OF THE SURVEY ROW THAT SENT YOU HERE.  The survey prints
    # `notebook: yes|no` and the port-clash note and then says "detail on
    # one: ... --port <port>", and the detail knew LESS than the line it came
    # from (`plan.md` § 5n.8).
    from .jupyter import pid_state as nb_state, port_clash
    from .jupyter import read_pid as nb_pid
    nb_up = nb_state(nb_pid(port)) == "ours"
    click.echo(f"notebook:  {'running' if nb_up else 'not running'}"
               f"   (molbuilder jupyter status --port {port})")
    clash = port_clash(port)
    if clash:
        click.echo(f"  NOTE: {clash}")
    if ok:
        return
    click.echo("  `kill -USR1` the child pid and read the stacks log, or "
               "`molbuilder serve restart`.")
    _note_wedge(port, pid)
    raise SystemExit(4)


def _say_where_else(missing_port: int) -> None:
    """After a miss on one port, name the servers that ARE there.

    The whole of J14 in one line: being told *"not running"* while another
    molbuilder serves happily two ports away is the answer that sends a
    person looking for a bug in the server.
    """
    from .config_dir import ports_with_pidfile
    from .serve_daemon import pid_state, read_pid
    others = [p for p in ports_with_pidfile()
              if p != missing_port and pid_state(read_pid(p)) == "ours"]
    if others:
        click.echo("  but a molbuilder IS running on port "
                   + ", ".join(str(p) for p in others)
                   + " -- `molbuilder serve status` with no --port surveys "
                     "them all.")


def _survey() -> int:
    """Every server with a pidfile, one line each.  Returns the exit code.

    0 when at least one answers, 4 when some are up and none answer, 3 when
    there is nothing to report -- the same three codes the single-port path
    uses, so a script does not have to learn a second vocabulary.
    """
    from .config_dir import ports_with_pidfile
    from .serve_daemon import pid_state, read_pid

    rows, stale = [], []
    for p in ports_with_pidfile():
        pid = read_pid(p)
        if pid_state(pid) == "ours":
            rows.append((p, pid))
        else:
            stale.append(p)

    if not rows:
        click.echo("no molbuilder server is running.")
        if stale:
            click.echo("  stale pidfile(s) for port "
                       + ", ".join(str(p) for p in stale)
                       + " -- the process is gone; nothing to act on.")
        # SAY WHAT THIS CANNOT SEE.  `serve foreground` writes no pidfile, so
        # an empty survey is "nothing left a record", which is not the same
        # sentence as "nothing is running".
        click.echo("  (a `serve foreground` writes no pidfile and cannot "
                   "appear here.)")
        return 3

    answered = False
    for p, pid in rows:
        ok, said = _probe_health(p)
        answered = answered or ok
        # THE LOG IS THE RECORD, whichever form was typed.  `_note_wedge`
        # states the rule (`deployment.md` § 1.0c, user ruling 2026-08-28):
        # a detection belongs in the log "not only in whichever terminal
        # happened to ask".  Before J14 the no-argument form defaulted to
        # 8000 and took the single-port path, so it logged; making it a
        # survey silently dropped that for the COMMON invocation (found in
        # review 2026-09-15).
        if not ok:
            _note_wedge(p, pid)
        # THE PID ONLY, no HTTP: the survey's question is "what is
        # running", and `jupyter status --port N` is the verb that asks a
        # notebook whether it ANSWERS.
        from .jupyter import pid_state as nb_state, port_clash
        from .jupyter import read_pid as nb_pid
        nb = "yes" if nb_state(nb_pid(p)) == "ours" else "no"
        click.echo(f"port {p:<6} pid {pid:<8} answering: {said}"
                   f"   notebook: {nb}")
        # The survey is where somebody looks when a notebook will not start.
        clash = port_clash(p)
        if clash:
            click.echo(f"  NOTE: {clash}")
    if stale:
        click.echo("stale pidfile(s) for port "
                   + ", ".join(str(p) for p in stale) + ".")
    click.echo("  detail on one:  molbuilder serve status --port <port>")
    return 0 if answered else 4


@serve_group.command("restart", short_help="recycle the server in place")
@_serve_port_flag
def cmd_serve_restart(port):
    """Signal the supervisor to recycle the child -- the Reload button's
    effect, workable from a script, and workable when the child is HUNG
    and the button's route cannot answer (`deployment.md` 1.0b)."""
    import signal as _signal
    from .serve_daemon import signal_supervisor
    ok, msg = signal_supervisor(port, _signal.SIGHUP)
    click.echo(("restarting: " if ok else "") + msg)
    raise SystemExit(0 if ok else 1)


@serve_group.command("stop", short_help="bring the background server down")
@_serve_port_flag
def cmd_serve_stop(port):
    import signal as _signal
    from .serve_daemon import signal_supervisor
    ok, msg = signal_supervisor(port, _signal.SIGTERM)
    click.echo(("stopping: " if ok else "") + msg)
    raise SystemExit(0 if ok else 1)


# --------------------------------------------------------------------- #
#  watch subcommand group (live trajectory viewer)                      #
# --------------------------------------------------------------------- #


@cli.group("checkpoint",
           short_help="save a calculation folder so you can come back to it")
def cmd_checkpoint():
    """Save the state of a calculation folder, and come back to it.

    A **state** is a saved snapshot of the whole folder: it has an id, a note
    you wrote, and the state it came from.  A **tag** is a name you give a
    state so you can find it again.  That is the whole vocabulary.

    \b
        molbuilder checkpoint init
        molbuilder checkpoint save -m "stage 1 converged, 41 steps"
        molbuilder checkpoint list
        molbuilder checkpoint tag stage1-good -m "geometry I trust"
        molbuilder checkpoint restore 4f9ca71

    Going back to a state and saving from it is how you branch: the new state's
    parent is the one you restored, both attempts stay listed, and neither can
    shadow the other.  There is no branch verb because there is nothing to
    declare.

    \b
    ONE RULE FOR YOU: use these verbs, not bare git.  The folder IS a git
    repository, but git alone sees only half of it -- the big files live in a
    side archive it is told to ignore -- so `git checkout` leaves the folder in
    a state no save ever produced.  See docs/execution/checkpointing.md § 2.0.
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


def _repo_or_exit(path):
    from molbuilder.checkpoint import Repo
    repo = Repo(_resolve_repo_path(path))
    if not repo.initialized:
        click.echo(f"Error: {repo.path} is not a checkpoint folder.  "
                   f"Run `molbuilder checkpoint init` first.", err=True)
        sys.exit(2)
    return repo


_PATH_OPT = click.option("-p", "--path", default=None, type=click.Path(),
                         help="The calculation folder.  Default: cwd.")


@cmd_checkpoint.command("init",
                      short_help="make this folder a checkpoint folder")
@click.option("--engine", default=None,
              help="Engine hint, so families that are always large skip the "
                   "size check (siesta, pyscf).  Omit and every file is "
                   "measured, which is always correct and merely slower.")
@click.option("-m", "--note", default="set up",
              help="The note for the first state.")
@click.option("--calculation", default=None,
              help="Which calculation this folder's history belongs to.  "
                   "Default: the folder's name.  Written verbatim into every "
                   "state, so a name needing repair is refused.")
@_PATH_OPT
def cmd_checkpoint_init(engine, note, calculation, path):
    """Make this folder a checkpoint folder and save its first state.

    One repository per calculation.  A folder whose subdirectories are working
    dirs is accepted only when this folder carries its description --
    task.json or job-set.json -- because without one,
    several independent calculations would share a history and rewind together.
    """
    from molbuilder.checkpoint import (
        Repo, CalculationNameError, CheckpointError, NestedRepoRefusedError)
    repo = Repo(_resolve_repo_path(path))
    already = repo.initialized
    # `init` IS THE REPAIR VERB, so this no longer returns before calling it.
    #
    # It used to print "already a checkpoint folder" and stop -- which was fine
    # while init only ever created things.  Once a folder somebody `git init`-ed
    # by hand needed a *name* before it could be saved (L3), `save` started
    # telling people to run exactly this command, and this command did nothing.
    # A remedy that no-ops is worse than no remedy: it reads as "I tried that,
    # it is still broken", and the verbs stop covering the work (§ 2.0).
    try:
        state = repo.init(engine=engine, note=note,
                          calculation=calculation)
    # Exit 2 for both of these: they are the two things the person running the
    # command fixes -- the folder holds several calculations, or the name needs
    # repair.  Exit 1 is kept for a fault, so a script can tell "my input was
    # wrong" from "the machine is broken".
    except (NestedRepoRefusedError, CalculationNameError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)
    except CheckpointError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    if already:
        where = f"standing at {state.short}" if state else "no states yet"
        click.echo(f"{repo.path}: already a checkpoint folder ({where})")
        click.echo(f"  calculation  {repo.calculation()}")
        return
    click.echo(f"initialised {repo.path}")
    click.echo(f"  calculation  {state.calculation}")
    click.echo(f"  {state.short}  {state.note}")


@cmd_checkpoint.command("save",
                      short_help="save the folder as a new state")
@click.option("-m", "--note", required=True,
              help="What happened, and what you were about to do.  Required: "
                   "it is the only thing that answers the question you bring "
                   "to a history a month later.")
@_PATH_OPT
def cmd_checkpoint_save(note, path):
    """Save the whole folder as a new state.

    The new state's parent is wherever the folder currently stands, so saving
    after a restore forks -- and both attempts stay listed.
    """
    from molbuilder.checkpoint import CalculationNameError, CheckpointError
    repo = _repo_or_exit(path)
    # Say what is happening before it happens.  A save checksums every big file
    # to name the archive, so a folder with a few gigabytes of density matrices
    # pauses here -- and a pause nobody explained reads as a hang.
    click.echo("checking the folder and checksumming its large files…", err=True)
    try:
        state = repo.save(note)
    # Exit 2, the same split `init` draws: this is the person's input, and the
    # message names the command that fixes it.  A save gained this refusal when
    # L3's name check moved here, and without this clause a one-command fix
    # exited 1 -- the code a script reads as "the machine is broken".
    except CalculationNameError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)
    except CheckpointError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    if state is None:
        click.echo("nothing changed since the state this folder stands at.")
        return
    click.echo(f"saved {state.short}  {state.note}")


@cmd_checkpoint.command("list", short_help="the states you have saved")
@click.option("-n", "--limit", default=None, type=int,
              help="Show only the newest N.")
@click.option("--check", is_flag=True,
              help="Compare file CONTENT rather than size and timestamp.  "
                   "Slower on large folders, and only worth it when you want "
                   "certainty right now -- a save or a restore always checks "
                   "content regardless.")
@_PATH_OPT
def cmd_checkpoint_list(limit, check, path):
    """Every state, newest first, with the state each came from.

    Two states sharing a parent are alternatives from the same point -- that is
    what a fork looks like here, and nothing had to be named for it.
    """
    from molbuilder.checkpoint import CheckpointError
    repo = _repo_or_exit(path)
    states = repo.states(limit=limit)
    if not states:
        click.echo("no states saved yet.")
        return
    if check:
        # Say what is happening before it happens.  --check reads every large
        # file end to end, and a pause nobody explained reads as a hang.
        click.echo("comparing file content…", err=True)
    # One call answers both questions: `status` has to read where the folder
    # stands in order to say what is unsaved, so asking git again for the same
    # state was a second pair of subprocesses for a value already in hand.
    #
    # AND IT IS THE ONE CALL HERE THAT CAN FAIL.  It reads the standing state's
    # MANIFEST, and a lost or tampered archive is *named* rather than absorbed
    # (I2b) -- which arrived here as an uncaught exception, so `checkpoint list`
    # answered a damaged folder with a Python traceback while the HTTP route
    # returned a structured error for the identical condition.
    #
    # The STATES are unaffected by that damage and are exactly what somebody
    # recovering needs to read, so they are printed either way and the damage
    # is reported after them.  `standing_at` is asked separately because it
    # reads only the commit, never the archive.
    status, damage = None, None
    try:
        status = repo.status(deep=check)
        here = status.standing_at
    except CheckpointError as e:
        damage, here = str(e), repo.standing_at()
    for state in states:
        mark = "->" if here and state.id == here.id else "  "
        parent = state.parent[:7] if state.parent else "-"
        tags = f"  [{', '.join(state.tags)}]" if state.tags else ""
        click.echo(f"{mark} {state.short}  {state.note}{tags}")
        click.echo(f"      {state.at}   from {parent}")
    if here:
        click.echo(f"\n-> is where this folder stands ({here.short}).")
    if damage:
        # Named where it was found (I2b), with the history above it intact and
        # readable.  Exit 1, not 2: this is the machine, not the person's input.
        click.echo(f"\nError: {damage}", err=True)
        sys.exit(1)
    if not status.clean:
        click.echo(f"   {len(status.unsaved())} unsaved change(s) here; "
                   f"`molbuilder checkpoint save` keeps them.")
        for name in status.unsaved():
            click.echo(f"     {name}")
    elif check:
        # Silence after an explicit request for certainty reads as "it did not
        # run".  The whole point of --check is to be told, now.
        click.echo("   nothing unsaved (content compared).")
    else:
        click.echo("   nothing unsaved (by size and timestamp; --check "
                   "compares content).")
    if status.ignore_edited:
        click.echo("   note: .gitignore's generated block was edited by hand. "
                   "The next save rewrites it from the classification.")


@cmd_checkpoint.command("tag", short_help="name a state so you can find it")
@click.argument("name")
@click.option("-m", "--note", required=True,
              help="Why this state is worth returning to.")
@click.option("--at", default=None,
              help="Which state to name.  Default: where the folder stands.")
@_PATH_OPT
def cmd_checkpoint_tag(name, note, at, path):
    """Give a state a name.

    Nothing tags a state on your behalf -- the namespace is yours alone, which
    is what makes your own tags easy to see.
    """
    from molbuilder.checkpoint import CheckpointError, NoSuchRefError
    repo = _repo_or_exit(path)
    try:
        tag = repo.tag(name, note, at=at)
    except NoSuchRefError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)
    except CheckpointError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    click.echo(f"tagged {tag.state[:7]} as {tag.name}")


@cmd_checkpoint.command("restore",
                      short_help="put the folder back to a state")
@click.argument("state")
@click.option("--force", is_flag=True,
              help="Accept the loss of unsaved work without being asked.  For "
                   "scripts; interactively you are asked instead.")
@_PATH_OPT
def cmd_checkpoint_restore(state, force, path):
    """Make the folder equal STATE exactly -- text and big files together.

    STATE is a state id or a tag.

    Anything here that is not saved is named first, and you are asked: at a
    terminal you answer the question, and a script passes --force.  It is lost
    if you go ahead -- checkpointing is not responsible for work you never
    saved, and nothing is stashed or set aside on your behalf.  Files that are
    simply absent from the target are removed without a warning -- they are
    still in the state that holds them.
    """
    from molbuilder.checkpoint import (
        CheckpointError, DirtyWorkingTreeError, NoSuchRefError)
    repo = _repo_or_exit(path)
    click.echo("verifying the archive, then checking what is unsaved here…",
               err=True)
    try:
        target = repo.restore(state, force=force)
    except DirtyWorkingTreeError as e:
        # A5 IN TWO STEPS, and the order is the rule (§ 7).  The refusals about
        # the target come first; only once they have passed is anything asked,
        # and by then the files are named and in front of the person deciding.
        #
        # Answering yes runs the restore again, which verifies the archive a
        # second time.  That is the honest cost of putting the question last:
        # the check that guards the folder happens immediately before the
        # folder changes, not before a prompt somebody sat reading.  What the
        # second pass no longer repeats is the EXPENSIVE half -- `force` is the
        # answer, so it does not re-hash the working tree to re-ask a question
        # this branch has already had answered.
        click.echo(str(e), err=True)
        from .envs.hints import stdin_can_answer
        if not stdin_can_answer():
            # No terminal, no --force: nothing may be assumed either way.
            sys.exit(2)
        if not click.confirm("\nGo ahead and lose it?", default=False):
            click.echo("nothing was changed.")
            sys.exit(2)
        try:
            target = repo.restore(state, force=True)
        except CheckpointError as e2:
            click.echo(f"Error: {e2}", err=True)
            sys.exit(1)
    except NoSuchRefError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)
    except CheckpointError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    click.echo(f"restored to {target.short}  {target.note}")


@cmd_checkpoint.command("config",
                      short_help="which files are stored where, and why")
@_PATH_OPT
def cmd_checkpoint_config(path):
    """Show the classification this folder is saved under.

    It lives in molbuilder.json, not in the folder: one home, so two folders
    cannot behave differently for no recorded reason.  Edit it there.
    """
    repo = _repo_or_exit(path)
    cls = repo.classification()
    limit = int(cls["size_limit_bytes"])
    click.echo(f"calculation   {repo.calculation()}")
    click.echo(f"size limit    {limit} bytes ({limit / (1024 * 1024):.1f} MB)")
    click.echo("              over it -> the archive; under it -> git")
    always = cls["always_large"]
    if always:
        click.echo("always large  " + ", ".join(always))
        click.echo("              these skip the size check; a hint can make "
                   "a save faster,")
        click.echo("              it can never make it store less")
    else:
        click.echo("always large  (none) -- every file is measured")
    click.echo("\nedit molbuilder.json to change it "
               "(docs/execution/checkpointing.md § 4)")


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
              help="emit only the per-frame energy / max_force / time "
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
    from .parse import ParseError
    detect_parser = _trajectory_parser_for
    from .parse.engines._helpers import trajectory_to_legacy_dict

    with _resolve_input_path(input_path) as resolved:
        try:
            parser = detect_parser(resolved)
        except ParseError as e:          # incl. AmbiguousFormatError
            click.echo(f"Error: {e}", err=True)
            sys.exit(2)
        traj = parser.parse(resolved)

    payload = trajectory_to_legacy_dict(traj)
    if frames_only:
        # Drop the heavy per-atom arrays; keep the per-frame summary
        # (iteration index, energy, max_force, and both clocks).  Useful for
        # piping a long trajectory into jq / grep without slurping
        # megabytes of coordinates.
        payload = {
            "source_format": payload["source_format"],
            "run_state":     payload["run_state"],
            "error_message": payload["error_message"],
            "iterations":    payload["iterations"],
            "energies":      payload["energies"],
            "max_forces":    payload["max_forces"],
            # Both clocks ship; either may be an all-null series when
            # the engine cannot report it (parse.md § 2a).
            "wall_clock_s":  payload["wall_clock_s"],
            "elapsed_s":     payload["elapsed_s"],
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

    Loop ends when the run is over -- run_state one of 'ended',
    'stopped' or 'out_of_memory' (`model/parse.md` § 2b, P-S1) -- or
    after --max-frames frames, whichever comes first.  'unknown' keeps
    polling: no evidence either way is not evidence of ending.
    Ctrl-C also exits cleanly.
    """
    import time
    from .parse import ParseError
    detect_parser = _trajectory_parser_for
    from .parse.engines._helpers import trajectory_to_legacy_dict
    from .parse.engines._run_ending import CONCLUDED

    if input_path == "-":
        click.echo("Error: stdin not supported for `watch tail` "
                   "(needs a real file to poll)", err=True)
        sys.exit(2)

    # A DIRECTORY IS A PERMANENT CONDITION, SO IT IS REFUSED BEFORE THE LOOP.
    # The loop below tolerates `ParseError` as a TRANSIENT state (the writer
    # has not flushed enough bytes to be detectable yet) and sleeps.  The
    # directory refusal is an `UnknownFormatError`, which IS a `ParseError`,
    # so raising it inside the loop retried for ever -- a silent hang where
    # there used to be a traceback.  Measured 2026-09-18.
    from pathlib import Path as _P
    if _P(input_path).is_dir():
        click.echo(f"Error: {input_path} is a directory.  `watch tail` "
                   f"follows one run artifact; name the file inside it.",
                   err=True)
        sys.exit(2)

    last_n = 0
    last_state = "running"
    emitted = 0
    poll_s = poll_ms / 1000.0
    try:
        while True:
            try:
                parser = detect_parser(input_path)
            except ParseError:           # incl. AmbiguousFormatError
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
                    "wall_clock_s": payload["wall_clock_s"][i],
                    "elapsed_s":    payload["elapsed_s"][i],
                    "n_atoms":    len(payload["frames"][i]),
                }
                click.echo(json.dumps(line))
                emitted += 1
                if max_frames is not None and emitted >= max_frames:
                    return
            last_n = n
            last_state = payload["run_state"]
            if last_state in CONCLUDED:
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


if __name__ == "__main__":
    sys.exit(main())
