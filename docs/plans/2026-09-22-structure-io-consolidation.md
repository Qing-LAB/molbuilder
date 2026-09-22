# Structure I/O — what is duplicated, and how it should be organised

**Role:** plan — consolidation, from a full reading
**Domain:** model · parse · engines · cli
**Opened:** 2026-09-22
**Why it exists:** three separate defects (a bare `.xyz` writer, a bare `n-1`,
a duplicate `.XV` parser) turned out to be **one shape**, and patching them
one at a time would have left the shape.

**Read in full before writing this:** `parse/engines/_sidecar.py` (342 lines),
`parse/coords/siesta_xv.py` (259), `workingcopy_structure.py` (379),
`transport/compose.py`'s `.XV`/sidecar paths, `docs/model/structure.md` § 2.4,
`docs/model/parse.md` § 5.3 / § 7.

---

## 0. The rule, which already exists

`docs/model/structure.md` § 2.4, quoted by the codec's own module docstring:

> **every structure-to-bytes translation goes through this codec, and every
> adapter has exactly one door.**

And `docs/model/parse.md` § 7, for the read side:

> a reader takes a path **or** it takes text, never both, and a caller holding
> a path reads the file itself.

Neither is new. The work below is not inventing a rule — it is finishing one
that the documents state and the code half-keeps. The doc even names its own
last violation (§ 2.4, "Open work", task #73, plan row **W15**).

---

## 1. What is actually duplicated

### 1.1 The `.XV` format is parsed **three** times

| | what it reads | Z → symbol | notes |
|---|---|---|---|
| `parse/coords/siesta_xv.py::_read_xv` | atoms only — **parses the cell and discards it** | `ase.data.chemical_symbols` | returns a geometry-only `Structure` |
| `parse/coords/siesta_xv.py::_read_xv_cell` | the cell only — **re-opens and re-parses the same file** | — | so `xv_to_xyz` parses one file twice |
| `transport/compose.py::read_xv` | both, in one pass | `chemistry.symbol_for_z` | a complete second parser, same name, different return shape |

`siesta_xv.py`'s own first paragraph says *"this is the only `.XV` reader."*

**This pair has already cost something.** From `constants.py`'s header:

> *"Two modules read the same SIESTA `.XV` file … so the same file gave
> coordinates **4e-7 apart** depending on which reader was asked."*

That was fixed by unifying the Bohr constant. **Both parsers were left
standing**, so the mechanism survived with one fewer way to diverge.

Note the functional difference, which is an argument for a particular
direction: `siesta_xv` needs **ase** for the element table; `compose` uses
molbuilder's own `chemistry.symbol_for_z`. Unifying on the latter removes a
third-party dependency from the parser rather than spreading it.

### 1.2 The companion-file lookup is implemented twice, in one file

Both in `parse/engines/_sidecar.py`:

| | finds | strategy |
|---|---|---|
| `read_frozen_atoms` (~80 lines of it) | `<stem>.molstruct.json` | candidate stems → `-run<N>` strip → label-aware rung strip → **lone sidecar in the directory**, via the framework search `sidecars_in` |
| `_siesta_fdf_path_for` | `<stem>.fdf` | suffix strip → `-run<N>` strip → **lone `*.fdf` in the directory**, via a hand-rolled `os.listdir` |

The duplication is **already acknowledged in a comment** in the first one:

> *"This is `_siesta_fdf_path_for`'s own pattern, in this same module."*

…and the two disagree about the very rule that comment cites: one asks through
the framework's search *("not a hand-rolled glob, `project-layout.md` 4.5")*,
the other hand-rolls `os.listdir`.

### 1.3 Two index conventions, in one module

`_sidecar.py:248` — correct, and says why:
```python
# SIESTA echoes constraints 1-based; translate back to the 0-based
# Structure identity through the engine index API (never a bare n - 1,
# which would be wrong for a 0-based engine).
return {from_engine_index(n, "siesta") for n in one_based}
```
`_sidecar.py:342` — 94 lines later, same file, same fact:
```python
return {i - 1 for i in frozen_one_based}
```
`engine_atom_index.py` forbids this in as many words: *"Nothing else in the
codebase may apply a bare `i + 1` OR `n - 1` to an atom index."*

Harmless **today** — `n-1` happens to be right for SIESTA — and that is exactly
why it survived. It stops being harmless the moment the helper is pointed at a
0-based engine, or the moment it becomes the path by which a run's frozen atoms
reach a saved structure (§ 4, step 5).

### 1.4 Two JSON-read conventions, four lines apart

`_sidecar.py:141` bypasses the door:
```python
with open(sidecar_path, "r", errors="replace") as fh:   # no encoding=
    data = _json.load(fh)                               # not molstruct.load
```
`:146` then imports the door it just bypassed:
```python
from molbuilder.sidecars import molstruct
return set(molstruct.frozen_atoms(data))
```
Skips envelope/schema validation, and passes **no `encoding=`**, so it decodes
under the platform locale — a non-ASCII region label mojibakes.

Two more of the same shape live in `transport/compose.py` (`:300` and `:505`),
one of them twenty lines from a correct `molstruct.load` in the same file,
under a docstring claiming *"ONE reader for compose and both web doors."*

### 1.5 The write-side path doors were never closed

The **read** side was closed deliberately — `Structure.from_xyz(path)` refuses
and points at the codec, citing `parse.md` § 7. The **write** side was not:
`to_xyz`, `to_extxyz` and `to_pdb` all still take a `path` and write a lone
file (`structure.py:1525 / :1616 / :1655`).

**And the blast radius of closing them is almost nothing.** Every production
caller already passes text, with exactly two exceptions:

- `parse/coords/siesta_xv.py:251` — the bare-`.xyz` writer we are fixing anyway;
- `molbuilder/__init__.py:17-18` — the package's **front-page docstring
  example**, which teaches `s.to_xyz("out.xyz")`.

So the door that made the violation possible can be closed for the price of one
docstring.

### 1.6 A private name used from outside, with a public alias available

`validation/identity.py:236` imports `_read_xv_cell`; `siesta_xv.py:223`
defines `read_xv_cell = _read_xv_cell` seventy lines below the function.

### 1.7 A policy the docstring claims and a caller owns

`_sidecar.py`'s module docstring says a caller *"consults them in order and
uses the first non-empty result"* and lists the precedence. Only
`parse/engines/siesta.py:1885-1889` implements it (`.out` → sidecar → `.fdf`);
`pyscf.py` and `molwatch.py` use the sidecar alone — correctly, since a PySCF
run has neither a constraints echo nor an `.fdf`.

**This is not a defect, it is a wrong docstring.** The smallest correct fix is
to say the precedence is SIESTA's and lives in the SIESTA parser — *not* to
invent a new entry point. (Rule: delete > one home > parameter > abstraction.)

---

## 2. What it should look like

Four statements, each removing one of the duplications above.

**S1 — one `.XV` parser, parsing once.** `siesta_xv` reads the file a single
time and returns everything it read: a `Structure` **carrying its cell**. The
"geometry-only" choice its docstring already calls *historical* ends;
`read_xv_cell` becomes a thin read of that structure, not a second pass;
`transport/compose.read_xv` keeps its tuple shape for its one caller and gets
it by delegating. Element symbols come from `chemistry.symbol_for_z`, and the
`ase` import goes.

**S2 — one companion lookup.** One helper — *given an artifact path, an
optional label and a suffix, find the sibling* — with the three-step strategy
written once: exact stem, rung-stripped stem, then the lone file of that suffix
in the directory, asked through the framework search. `read_frozen_atoms` and
`_siesta_fdf_path_for` both become calls to it with different suffixes.

**S3 — one door per read.** `molstruct.load` for every sidecar;
`from_engine_index` for every engine→canonical index. No exceptions, and both
already exist.

**S4 — the write side mirrors the read side.** `to_xyz` / `to_extxyz` /
`to_pdb` return text. A caller holding a path goes through
`StructureCodec().write`, which owns the pairing. The violation class becomes
unrepresentable rather than fixed.

---

## 3. What this is *not*

- **Not** a new frozen-atom entry point (§ 1.7 — fix the docstring).
- **Not** a change to how a user declares frozen atoms.
- **Not** the identity/hash redesign; that is its own item in the audit.
- **Not** touching the deck's spliced writer, which is exempt for a measured
  reason (molbuilder is not importable from the run environment) and already
  takes its sidecar from the codec.

---

## 4. The work, in order

Ordered so that each step is independently shippable and each one makes the
next smaller. Steps 1–3 were approved before this reading; **steps 2 and 4 are
what the reading added**.

| # | step | why here | risk |
|---|---|---|---|
| **1** | **S3 for indices** — `_sidecar.py:342` routes through `from_engine_index`, matching its twin at `:248` | one line, no behaviour change today, and it is a prerequisite for step 5 | none |
| **2** | **S3 for sidecar reads** — `_sidecar.py:141` and `compose.py:300/:505` use `molstruct.load` | same class, found by the same reading; fixes the encoding bug too | low — validation now runs where it did not, so a malformed sidecar starts failing loudly |
| **3** | **S1 — one `.XV` parser** | removes the duplicate that has already caused a measured bug | **the real one**: `read_xv` starts returning a cell. `web/blueprints/modify.py:1018` and the periodicity gate must be checked |
| **4** | **S2 — one companion lookup** | the two implementations are what make steps 1–3 feel like scattered patches | medium — this is the most-commented code in the module, and every comment records a measured incident. Its tests must be read first |
| **5** | **`xv2xyz` through the codec**, with an explicit `--from-run` flag | needs 1 (frozen atoms read correctly) and 3 (the cell survives) | low, once 1 and 3 are in |
| **6** | **S4 — close the write doors** + fix `__init__.py`'s example | last, because it is the guard that stops the class recurring; cheap only *after* step 5 removes the one real caller | none once 5 lands |

### On step 5's two modes

- **bare `.XV`** → the pair is written; the sidecar carries the **real lattice
  and periodicity from the file** (a `.XV` is not metadata-free), regions and
  annotations empty. Nothing is defaulted that the file actually knows.
- **`--from-run`** → additionally reads the siblings for frozen atoms, through
  the reader fixed in step 1.
- **Explicit flag, not auto-detect** (user, 2026-09-22): guessing "am I in a
  calculation directory" and silently pulling in a frozen-atom set is hard to
  notice when wrong.

A `.XV` is a SIESTA artifact, so its periodicity is `("periodic",) * 3` — which
is also what makes `science/normal-modes.md` § 3.1a give 3 rigid-body modes
rather than 6 for a slab.

---

## 5. What must not break — the checks each step owes

| step | the check |
|---|---|
| 1 | frozen atoms still read identically from a SIESTA `.fdf` — `n-1` and `from_engine_index(·, "siesta")` must agree, which is the point |
| 2 | a sidecar with a non-ASCII region label survives the read |
| 3 | `web/blueprints/modify.py:1018` and every periodicity gate, with a `.XV` that now carries a cell; and `transport/compose.py:468` gets byte-identical numbers |
| 4 | the measured cases the existing comments record: `bdt.out`, `bdt_01_coarse.out`, `-run0.out`, `.molwatch.log`, `.pyscf.log`; a lone unrelated sidecar refused; two sidecars declining |
| 5 | both modes write a readable pair; `--from-run` recovers the frozen set a real run declared |
| 6 | nothing in production calls the removed path arguments |

**And one guard none of them has today.** Both audit scans noted it
independently: the docstrings say *"nothing else in the codebase may…"* and
*"this is the only `.XV` reader"*, and **nothing enforces either**. Every
duplication in § 1 is drift those claims did not prevent. A grep-style
exclusivity test over `molbuilder/` — one reader per format, one door per
write, no bare index arithmetic — is the thing that keeps this consolidated
after it is consolidated. Worth writing with step 6.
