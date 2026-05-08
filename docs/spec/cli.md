# Spec — command-line interface

**Module**: `molbuilder/cli.py` &nbsp;·&nbsp; **Entry point**:
`molbuilder = "molbuilder.cli:main"` (declared in `pyproject.toml`)

The CLI is a thin shell over the same Python API the web UI uses.
Subcommands are mutually exclusive; each maps to one builder /
emitter / server function.

## Subcommands

| subcommand | purpose | accepts | emits |
| --- | --- | --- | --- |
| `peptide` | build polypeptide | `1-letter sequence` | XYZ to stdout / `--out` / `--pdb` |
| `dna`     | build ssDNA       | `1-letter sequence` | XYZ to stdout / `--out` / `--pdb` |
| `rna`     | build ssRNA       | `1-letter sequence` | XYZ to stdout / `--out` / `--pdb` |
| `smiles`  | build from SMILES | SMILES string       | XYZ to stdout / `--out` / `--pdb` |
| `name`    | build from name   | common / IUPAC name | XYZ to stdout / `--out` / `--pdb` |
| `fdf`     | XYZ/PDB → SIESTA fdf | path + path        | writes `.fdf` |
| `pyscf`   | XYZ/PDB → PySCF script | path + path     | writes `.py` |
| `serve`   | run web UI        | `--host`, `--port`  | starts Flask on port 8000 |

## Build-subcommand contract

* Input is positional: `molbuilder peptide ARNDC ...`.
* Outputs: at least one of `--out <path>`, `--pdb <path>`,
  `--pyscf-atom-block` (prints PySCF atom-block to stdout).  If none
  is given, default is XYZ to stdout.
* `--title` overrides the auto-generated title.
* DNA / RNA accept extra knobs: `--backend`, `--form`, `--terminal`,
  `--no-protonate-phosphates`.
* On success, prints summary line to stderr (`<Structure '...': N
  atoms, ...>`).
* On failure, exits non-zero with a single human-readable error
  line on stderr.

The legacy flag `--pyscf` is preserved as an alias for
`--pyscf-atom-block` (deprecation note in the help).  Don't confuse
it with the `pyscf` subcommand which emits a full runnable script.

## `fdf` / `pyscf` subcommand contract

* First positional arg `input`: path to `.xyz` or `.pdb` (auto-
  detected from extension).
* Second positional arg `fdf` / `py`: output path.
* All `SiestaConfig` / `PySCFConfig` fields are exposed as
  `--<kebab-case-field>` flags (auto-generated from dataclass
  metadata via `add_dataclass_options` in `molbuilder/cli.py`).
* Boolean fields surface as a `--<field> / --no-<field>` pair
  regardless of the default; either form may be passed.
* Fields whose dataclass metadata carries a `choices` tuple are
  validated at CLI parse time via `click.Choice`, so a typo on
  e.g. `--method`, `--solution-method`, `--scf-init-guess`, or
  `--optimizer` fails with exit code 2 before any conversion runs.
* Five fields opt out of the bridge via `metadata['skip_cli']`
  and are hand-rolled at the call site because their CLI handling
  needs custom parsing (`--kgrid`, `--psml-lib`, `--species-order`)
  or because they have a tri-state `None`/`True`/`False`
  semantic the bridge can't represent (`parallel_block_size`,
  `parallel_over_k`).
* `pyscf` exposes `--no-write-trajectory` to disable the
  `<job>_geom_optim.xyz` streaming output.
* On success, prints "Wrote <path>: ..." to stderr.
* On `fdf` with missing pseudopotentials, exits with code 2 (other
  errors exit 1).

## `serve` subcommand contract

* Default `--host 127.0.0.1`, `--port 8000`.
* `--debug` opt-in only; warns about Flask debugger danger.
* Loud stderr warning when `--host` is anything other than
  loopback / `localhost` / `::1`.

## Forbidden patterns

The CLI must NOT:

1. Hide an exit code mismatch: `fdf` with missing pseudopotentials
   returns 2 specifically so a wrapping shell script can branch on
   it.
2. Mix subcommand outputs with logging.  Build subcommands write
   structure data to **stdout**; their summary goes to **stderr**.
   Pipe-friendly: `molbuilder smiles "c1ccccc1" | head` works.
3. Couple subcommand output to argv order.  `--out X.xyz --pdb X.pdb`
   and `--pdb X.pdb --out X.xyz` produce identical files.

## Test coverage gap

The CLI itself has no dedicated test file (it's exercised
end-to-end via the web UI and via direct `python -m molbuilder.cli`
smoke tests in CI).  Future improvement: `tests/test_cli.py`.
