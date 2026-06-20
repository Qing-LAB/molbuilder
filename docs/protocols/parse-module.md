# Parse module protocol

**Status:** v1 draft (2026-06-19) — supersedes the parser-related
parts of `parsers.md` (file-level engine output), the per-block
extractors of `script-contract.md`, and the per-function pattern in
`bundle-contract.md`.  Those three docs become section pointers
into this one once migration completes (§ 8).

**Module**: [`molbuilder/parse/`](../../molbuilder/parse/)
&nbsp;·&nbsp; **Tests**:
[`tests/parse/`](../../tests/parse/)

## 1. Position — one module for "turn a file or directory into a Python data structure"

Before this protocol, molbuilder had **four parallel parsing
patterns**:

| Where | Pattern | Returns |
|---|---|---|
| `molbuilder/parsers/{siesta,pyscf,molwatch_log}.py` | `TrajectoryParser` ABC + registry + `detect_parser` (clean) | `Trajectory` dataclass |
| `molbuilder/parsers/{molstruct,spectra,transport}_json.py` | Ad-hoc `load(path)` functions | `dict` |
| `molbuilder/parsers/{siesta,pyscf}_struct.py` | Ad-hoc `read_xv(path)`, `read_fdf_initial_coords(text)`, etc. | `Structure` |
| `molbuilder/script_contract.py` | Ad-hoc `extract_*(text)` family | `Optional[Dict]` per block, or `ScriptSource` umbrella |
| `molbuilder/script_bundle.py`, `molbuilder/jobs/decoder.py` | Ad-hoc `assemble_from_run_dir(path)` / `decode_run_dir(path)` | `RunBundle` dataclass / `dict` |

Five styles, three return-type flavors, no shared base. Each new
engine or output type added a parallel pattern.  No way to ask
"what's in this path?" without knowing which module to import.

This protocol **collapses everything into one package** with three
ABCs (file / text / directory), one frozen-dataclass hierarchy
(`ParseResult` + subclasses), and one registry + dispatch
(`detect()` / `parse()`).  Every existing parser becomes a class
that conforms to one ABC and returns a typed `ParseResult`
subclass.  Every consumer (web blueprints, CLI, tests) imports
from the same package and queries the registry.

## 2. The three ABCs

```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Type

class FileParser(ABC):
    """One file path → one ParseResult."""
    name:   str          # short id ("siesta-out", "molstruct-json")
    label:  str          # UI-friendly ("SIESTA stdout (.out / .log)")
    hint:   str          # what file to point at when can_parse returns False
    output: Type["ParseResult"]   # the concrete ParseResult subclass returned

    @classmethod
    @abstractmethod
    def can_parse(cls, path: Path) -> bool: ...
    @classmethod
    @abstractmethod
    def parse(cls, path: Path) -> "ParseResult": ...


class TextParser(ABC):
    """One text body (string in memory) → one ParseResult.  Pure: no I/O."""
    name:   str
    label:  str
    output: Type["ParseResult"]

    @classmethod
    @abstractmethod
    def parse(cls, text: str) -> "ParseResult": ...


class DirParser(ABC):
    """One directory → one ParseResult, composed from per-file parsers
    plus directory-level invariants (cross-file consistency, status,
    progress, etc.)."""
    name:   str
    label:  str
    output: Type["ParseResult"]

    @classmethod
    @abstractmethod
    def can_parse(cls, run_dir: Path) -> bool: ...
    @classmethod
    @abstractmethod
    def parse(cls, run_dir: Path) -> "ParseResult": ...
```

**No detection for `TextParser`.**  Callers know which TextParser
they want — a text body has no path to inspect.  `TextParser`
implementations are passed explicitly:
`parse_text(text, parser=ScriptSourceParser)`.

## 3. The `ParseResult` hierarchy

All return types are frozen dataclasses sharing a common base.
Consumers may type-narrow by `isinstance` or by `result_kind`.

```python
@dataclass(frozen=True)
class ParseResult:
    """Common envelope for every parse output."""
    schema_version: int
    parsed_at:      str               # ISO-8601 UTC
    parser_name:    str
    source:         str               # path str OR "<text>" for TextParsers
    result_kind:    str               # discriminator for switch-on


# ---- engine output ---- #

@dataclass(frozen=True)
class TrajectoryResult(ParseResult):
    """Per-step physics from an engine .out / .log."""
    result_kind:   str = "trajectory"
    frames:        List[Frame] = field(default_factory=list)
    lattice:       Optional[np.ndarray] = None     # shared cell
    source_format: str = "unknown"                 # "siesta" | "pyscf" | "molwatch"
    run_state:     str = "unknown"                 # running/finished/failed
    error_message: Optional[str] = None
    runtime_info:  Dict[str, Any] = field(default_factory=dict)
    parse_warnings: List["ParseWarning"] = field(default_factory=list)


# ---- structure-only ---- #

@dataclass(frozen=True)
class StructureResult(ParseResult):
    """Geometry from .XV / .STRUCT_OUT / .xyz / .fdf-coords-block / PySCF geom."""
    result_kind:    str = "structure"
    structure:      Structure   # the canonical dataclass
    cell:           Optional[np.ndarray] = None    # 3x3 Å, separate because
                                                   # Structure doesn't carry it
    source_format:  str = "unknown"
    parse_warnings: List["ParseWarning"] = field(default_factory=list)


# ---- molbuilder sidecar JSONs ---- #

@dataclass(frozen=True)
class SidecarResult(ParseResult):
    """Generic payload + schema tag for molstruct/spectra/transport sidecars."""
    result_kind:  str = "sidecar"
    payload:      Dict[str, Any] = field(default_factory=dict)
    schema:       str = "unknown/v0"   # "molstruct/v3" | "spectra/v1" | etc.


# ---- script-contract block extracts ---- #

@dataclass(frozen=True)
class ScriptResult(ParseResult):
    """The 6 reserved blocks in a .fdf / .py text body.

    Each sub-block carries a ``present`` flag so callers distinguish
    "block absent" from "block present-but-empty"."""
    result_kind:   str = "script"
    header:        Optional[str] = None
    provenance:    Optional[Dict[str, str]] = None
    bench_marks:   Optional[Dict[str, Any]] = None
    atom_metadata: Optional[Dict[str, Any]] = None
    user_custom:   Optional[List[str]] = None
    block_schema_versions: Dict[str, int] = field(default_factory=dict)


# ---- directory-level (composer output) ---- #

@dataclass(frozen=True)
class JobResult(ParseResult):
    """Directory-level decoded job — what JobMonitor + Results tab
    consume.  Schema pinned by job-decoder.md."""
    result_kind:           str = "job"
    job_type:              str = "unknown"   # optimization / spectrum / transport
    engine:                str = "siesta"
    system_label:          Optional[str] = None
    status:                Dict[str, Any] = field(default_factory=dict)
    progress:              Dict[str, Any] = field(default_factory=dict)
    geometry:              Dict[str, Any] = field(default_factory=dict)
    plots:                 Dict[str, Dict[str, List[List[float]]]] = field(default_factory=dict)
    source_files:          List[Dict[str, Any]] = field(default_factory=list)
    engine_input_by_stage: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    parse_warnings:        List["ParseWarning"] = field(default_factory=list)


@dataclass(frozen=True)
class BundleResult(ParseResult):
    """Run-dir handoff bundle — what the next-stage materialiser
    consumes.  Schema pinned by bundle-contract.md."""
    result_kind:  str = "bundle"
    structure:    Structure
    cell:         Optional[np.ndarray] = None
    regions:      Dict[str, List[int]] = field(default_factory=dict)
    frozen_atoms: List[int] = field(default_factory=list)
    notes:        List[str] = field(default_factory=list)
    schema_versions: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ParseWarning:
    """Level-3 fail-soft warning emitted by any parser."""
    source:   str       # filename or "<text>"
    line_no:  Optional[int]
    snippet:  Optional[str]
    error:    str
    category: str
```

### 3.1 Discriminator field

Every concrete `ParseResult` subclass sets `result_kind` to a
fixed string. Consumers that hold a `ParseResult` (e.g. cached in
a dict, sent over the wire as JSON) use this discriminator to
narrow:

```python
match result.result_kind:
    case "trajectory": ...
    case "structure":  ...
    case "sidecar":    ...
    case "script":     ...
    case "job":        ...
    case "bundle":     ...
```

Adding a new kind requires a new subclass + extending this list.
JSON serialisation reads `result_kind` to pick a deserialiser.

### 3.2 Why frozen dataclasses

* **No defensive copies** at API boundaries.
* **Hashable** — usable as dict keys, set members.
* **Trivial pickle / JSON serialisation** via `asdict()` for cache
  and webhook delivery.
* **Test pinning** — `result == expected` works without an
  `__eq__` override.

## 4. The registry + public API

The module exposes a single public surface at
`molbuilder/parse/__init__.py`:

```python
from molbuilder.parse import (
    # Public API
    detect, parse, parse_text, parse_dir, register,

    # ABCs (for plugin authors)
    FileParser, TextParser, DirParser,

    # ParseResult hierarchy (for type-narrowing consumers)
    ParseResult, TrajectoryResult, StructureResult,
    SidecarResult, ScriptResult, JobResult, BundleResult,
    ParseWarning,

    # Exceptions
    UnknownFormatError, AmbiguousFormatError,
)
```

### 4.1 `detect(path)`

```python
def detect(path: Path) -> Type[FileParser] | Type[DirParser]:
    """Return the first parser whose can_parse(path) is True.

    Tries directory parsers first (if path is a directory), then
    file parsers.  Raises UnknownFormatError with a tailored
    error message that lists every registered parser + the
    standard foot-gun hints (from parsers.md § 5).
    """
```

### 4.2 `parse(path)`

```python
def parse(path: Path) -> ParseResult:
    """detect + parse in one call.  Convenience for code that
    doesn't need to know the parser class."""
```

### 4.3 `parse_text(text, parser)`

```python
def parse_text(text: str, parser: Type[TextParser]) -> ParseResult:
    """Parse a known text body.  No detection — caller specifies
    the parser explicitly."""
```

### 4.4 `parse_dir(path)`

```python
def parse_dir(path: Path) -> ParseResult:
    """Force-detect among DirParsers only.  Used by JobMonitor +
    Results tab + bundle handoff where the contract is 'this is
    a project dir'."""
```

### 4.5 `register(parser)`

```python
def register(parser: Type[FileParser | TextParser | DirParser]) -> None:
    """Add a parser to the registry.  Module-init time; not for
    runtime registration.  Idempotent."""
```

## 5. Package layout

```
molbuilder/parse/
├── __init__.py          # public API: re-exports from below
├── base.py              # FileParser, TextParser, DirParser, ParseWarning
├── types.py             # ParseResult base + 6 concrete subclasses
├── registry.py          # _REGISTRY, detect, parse, parse_text, parse_dir, register
├── errors.py            # UnknownFormatError, AmbiguousFormatError
│
├── engines/             # ✓ shipped (Phase C) — engine .out / .log FileParsers
│   ├── __init__.py      # imports + registers each
│   ├── _helpers.py      # wrap_trajectory() Trajectory -> TrajectoryResult
│   ├── siesta.py        # was parsers/siesta.py
│   ├── pyscf.py         # was parsers/pyscf.py
│   └── molwatch.py      # was parsers/molwatch_log.py
│
├── coords/              # ✓ shipped (Phase E) — geometry-file FileParsers
│   ├── __init__.py      # imports + registers each
│   ├── _helpers.py      # build_structure_result() envelope helper
│   ├── siesta_xv.py     # ✓ was parsers/siesta_struct.py::read_xv + cell
│   ├── pyscf_geom.py    # ✓ was parsers/pyscf_struct.py::read_optimized_xyz
│   ├── siesta_fdf.py    # PENDING — was read_fdf_initial_coords
│   └── xyz.py           # PENDING — plain .xyz reader
│
├── sidecars/            # ✓ shipped (Phase D) — molbuilder JSON sidecar FileParsers
│   ├── __init__.py
│   ├── _helpers.py      # build_sidecar_result() envelope helper
│   ├── molstruct.py     # was parsers/molstruct_json.py
│   ├── spectra.py       # was parsers/spectra_json.py
│   └── transport.py     # was parsers/transport_json.py
│
├── scripts/             # ✓ shipped (Phase F) — TextParsers for the 5 fdf reserved blocks
│   ├── __init__.py      # public surface (re-exports per-block + umbrella classes)
│   ├── _helpers.py      # empty_script_result() envelope helper
│   ├── markers.py       # MARKER_RE + block-name constants (re-exported from legacy)
│   ├── header.py        # HeaderTextParser
│   ├── provenance.py    # ProvenanceTextParser
│   ├── bench_marks.py   # BenchMarksTextParser
│   ├── atom_metadata.py # AtomMetadataTextParser (also surfaces schema_version)
│   ├── user_custom.py   # UserCustomTextParser
│   └── source.py        # ScriptSourceTextParser (umbrella, composes all 5)
│
└── dirs/                # ✓ shipped (Phases B + G) — DirParsers (composers)
    ├── __init__.py      # JobDirParser auto-registers; BundleDirParser explicit-dispatch
    ├── bundle.py        # ✓ BundleDirParser (Phase G; wraps script_bundle.assemble_from_run_dir)
    └── job.py           # ✓ JobDirParser (Phase B; wraps decode_run_dir)
```

## 6. Plugin contracts

### 6.1 Adding a new engine FileParser

1. Create `parse/engines/<engine>.py`.
2. Define `class <Engine>OutParser(FileParser)` setting `name`,
   `label`, `hint`, `output = TrajectoryResult`.
3. Implement `can_parse(path)` — checks for engine-specific
   content markers in the first ~1000 bytes (per parsers.md § 5).
4. Implement `parse(path)` returning a `TrajectoryResult`.
5. Import + register at module init in `engines/__init__.py`:
   `register(<Engine>OutParser)`.
6. Add L2 tests in `tests/parse/engines/test_<engine>.py`.

### 6.2 Adding a new sidecar FileParser

1. Create `parse/sidecars/<kind>.py`.
2. Define `class <Kind>SidecarParser(FileParser)` returning
   `SidecarResult`.
3. `can_parse` matches the filename pattern (`.<kind>.json`).
4. `parse` reads the JSON, validates `schema_version`, returns
   `SidecarResult(payload=..., schema=f"<kind>/v{N}")`.
5. Register + test.

### 6.3 Adding a new block TextParser

1. Create `parse/scripts/<block>.py`.
2. Define `class <Block>BlockParser(TextParser)` returning
   `ScriptResult` (or a more specific subclass when warranted).
3. Implement `parse(text)` using `MARKER_RE` from
   `scripts/markers.py`.
4. Caller imports the class explicitly:
   `parse_text(text, parser=<Block>BlockParser)`.

### 6.4 Adding a new DirParser composer

1. Create `parse/dirs/<purpose>.py`.
2. Define `class <Purpose>DirParser(DirParser)` returning the
   appropriate `*Result` subclass.
3. **Must compose existing FileParsers + TextParsers** — no
   new file-level parsing inline.  Per § 9 forbidden pattern #1.
4. `can_parse` checks what files the dir contains (presence of
   `.fdf` + `.out` for jobs, etc.).
5. Register + test.

## 7. Composer pattern — DirParsers

DirParsers are the directory-level composers.  They MUST:

1. **Walk the directory** to identify relevant files.
2. **Dispatch each file via the registry** — call `detect(file)`
   + `.parse(file)`, or pick a specific FileParser when the
   selection rule isn't path-driven (e.g. "largest .fdf by atom
   count" per bundle-contract.md § 4.3).
3. **Compose** the per-file `ParseResult`s into a directory-level
   `*Result`.
4. **Apply cross-file invariants** that no single FileParser can
   see — atom-count consistency, lattice-vector handedness, stage
   ordering, status state machine.
5. **NEVER re-parse what a registered FileParser can produce.**
   Add a missing FileParser instead of side-grepping.

The two DirParsers shipped today:

* `BundleDirParser` (`parse/dirs/bundle.py`) — picks source `.fdf`,
  reads `.XV` for final coords, validates atom count, returns
  `BundleResult` for the next-stage materialiser.
* `JobDirParser` (`parse/dirs/job.py`) — walks all `.out` files,
  consolidates plots per source, returns `JobResult` for
  JobMonitor + Results tab consumption.

Both compose the same per-file parsers; they differ in WHAT they
extract + how they shape the result.

## 8. Migration plan

The current code stays working throughout migration. The package
ships in this order:

| Phase | Lands | Status (2026-06-19) |
|---|---|---|
| **A** | This doc + `parse/__init__.py`, `parse/base.py`, `parse/types.py`, `parse/registry.py`, `parse/errors.py` (skeleton only; no parsers yet) | ✓ shipped |
| **B** | Move `jobs/decoder.py` → `parse/dirs/job.py` as the first concrete `DirParser` example | ✓ shipped |
| **C** | Wrap `parsers/{siesta,pyscf,molwatch_log}.py` as `FileParser`s in `parse/engines/*` | ✓ shipped |
| **D** | Wrap `parsers/{molstruct,spectra,transport}_json.py` as `FileParser`s in `parse/sidecars/*` | ✓ shipped |
| **E** | Wrap `parsers/{siesta,pyscf}_struct.py` as `FileParser`s in `parse/coords/*` (StructureResult with cell — closes the Phase 1 lattice-extraction gap) | ✓ shipped |
| **F** | Split `script_contract.py` per-block → `parse/scripts/*` (HEADER / PROVENANCE / BENCH-MARKS / ATOM-METADATA / USER-CUSTOM / source-umbrella as TextParsers) | ✓ shipped |
| **G** | Move `script_bundle.py` → `parse/dirs/bundle.py` as `BundleDirParser` (explicit-dispatch; not auto-registered, since it shares the .fdf claim with JobDirParser but expresses a different user intent) | ✓ shipped |
| **H** | The clean break — re-scoped into 4 sub-phases per the 2026-06-20 pre-Phase-H audit (see below) | broken into H1-H4, pending |

**Phase H re-scope (2026-06-20 audit).**  The original Phase H
"delete legacy + update 8 imports" undercounted the real cost.
The audit found:

* 60 legacy symbols exported (≈42 load-bearing after filtering
  stdlib/typing imports).
* **114 import lines across 18 files** consume the legacy
  modules (8 self-deps in `parse/` + 10 production files +
  ~25 test files).
* Critical gaps the new module has NO equivalent for:
  - `trajectory_to_legacy_dict` — 159-line adapter feeding the
    3Dmol.js frontend; deleting it = blank Results plots.
  - Write-side script_contract `emit_*` family (used by
    `siesta/input.py`, `pyscf/input.py`, `runwrap.py`,
    `bench/__init__.py`) — Phase F only migrated read side.
  - `write_bundle_as_handoff` — bundle materializer; Phase G
    only migrated read side.
  - `apply_inbody_atom_metadata` — mutates a Structure from
    .fdf text; different pattern from immutable TextParsers.
  - Sidecar write-side (`save`, `with_lock`, `sidecar_path_for`,
    `sha256_of_file`, `to_dict`, `apply_to_structure`,
    `MolstructJsonError`, `dump_spectra_json`,
    `dump_transport_json`) — **22 callsites in selection.py
    alone**.
* 14 doc cross-references to legacy paths (`design.md`,
  `roadmap.md`, `package-layout.md`, `job-decoder.md`,
  `results-state-contract.md`, `save-flow.md`, `atom-selection.md`,
  `web-api.md`, `test-strategy.md` and others).

Total revised scope: **~6,000 LOC moved + ~40 consumer rewrites
+ 25 test files + 14 docs**.  This calls for a 4-phase split:

| Sub-phase | Lands | Notes |
|---|---|---|
| **H1** | Absorb legacy READ-side into the new wrappers.  Each `parse/engines/*`, `parse/sidecars/*`, `parse/coords/*`, `parse/scripts/*` parser inlines its legacy body so the wrapper no longer imports from `molbuilder.parsers` / `molbuilder.script_contract`.  `parse/dirs/bundle.py` absorbs the read half of `assemble_from_run_dir`.  `parse/dirs/job.py` switches internal calls from `script_contract.extract_*` to `parse_text(text, parser=...TextParser)`. | ≈3,200 LOC moved |
| **H2** | Rehome the WRITE side.  These don't belong in `parse/` (parse-module.md scopes parsing only).  Proposal: `molbuilder/sidecars/` (save / with_lock / sidecar_path_for / sha256 / to_dict / apply_to_structure / dump_* / exception families); `molbuilder/script_emit.py` (`emit_*` + `MARKER_RE` + `BLOCK_*` constants + `BenchField` + `SIESTA_BENCH_FIELDS` + `begin_marker` + `end_marker` + `merge_user_custom_from_target` + `molbuilder_git_sha` + `generated_at_now` + `apply_inbody_atom_metadata`); `BundleResult.materialize(dest_dir)` method (or `molbuilder/bundle_writer.py`).  Plus `trajectory_result_to_legacy_dict()` in `parse/engines/_helpers.py` for the 3Dmol.js adapter. | ≈1,200 LOC + 4 new modules |
| **H3** | Update consumers.  10 production files (`web/blueprints/{watch,spectra,selection,_shared,files,results}.py`, `siesta/input.py`, `pyscf/input.py`, `runwrap.py`, `bench/__init__.py`) plus 25 test files plus the 8 `parse/` self-deps now resolved by H1.  Delete `tests/parse/test_round2_fixes.py::test_migration_legacy_parsers_detect_still_works` in the same commit family (the migration shim test only made sense pre-H). | ≈40 callsites |
| **H4** | Delete `molbuilder/parsers/` (12 files) + `molbuilder/script_contract.py` (806 LOC) + `molbuilder/script_bundle.py` (507 LOC).  Doc redirects: `docs/types/parsers.md` → 20-line stub pointing here; `docs/protocols/script-contract.md` + `bundle-contract.md` keep their contracts but update code-pointer lines to point at the new homes.  Update the 14 cross-referencing docs.  Final test sweep + `grep -rn "from molbuilder.{parsers,script_contract,script_bundle}"` must return empty. | ≈14 docs touched |

Per the no-back-compat-shims convention, each sub-phase still
ships in a single commit family — no transition window between
H3 and H4.

**Status snapshot (2026-06-20):** Phases A-G + the half-Phase-H
audit-gap closures shipped.  H1-H4 await a forcing function;
the legacy modules are stable, tested, and cost nothing today.
When H1-H4 ship, they ship in audit order, and each is a
focused review-ready commit.

## 9. Forbidden patterns

These rules prevent the next round of parallel parse paths:

1. **DirParsers must compose registered FileParsers + TextParsers
   — no inline file-level parsing.**  If a DirParser needs a new
   file format read, add a FileParser for it first.  Caught by the
   `test_dir_parser_uses_registry` lint test.
2. **TextParsers do NO I/O.**  Path-taking callers must read the
   file themselves and pass the body.  The doc's job-decoder.md §
   9 pattern "no direct .out grep outside detect_parser" is a
   special case of this.
3. **FileParsers do NOT spawn subprocesses, network calls, or
   threads.**  Parsing is pure I/O over local files.  Async +
   background work belongs to the JobMonitor (Phase 2 of #507).
4. **`ParseResult` subclasses are frozen.**  Mutating them after
   construction is forbidden; create a new instance with the
   intended values via `dataclasses.replace(result, ...)`.
5. **Adding a new ParseResult subclass requires a new
   `result_kind` discriminator value + a doc update.**  Catches
   accidental shape drift; one path = one discriminator.
6. **Adding a curated key list (e.g. `engine_body_summary`,
   `ENGINE_BODY_KEYS`) requires a doc update + a test.**  Lists
   like these are the load-bearing contract for downstream
   consumers; silent additions silently break Results.
7. **No engine-specific code outside `parse/engines/` and
   `parse/coords/`.**  The composer (DirParser) is engine-agnostic
   in API; engine-specific logic lives in the leaf parsers.

## 10. Test coverage

Total **106 tests** across `tests/parse/` (collected 2026-06-20).
Grouped by the file/area they exercise:

| File | Tests | What it pins |
|---|---|---|
| `test_registry.py` | 9 | Registry mechanics: engine + dir parsers registered, `detect(siesta.out)` routes, unknown extension raises `UnknownFormatError`, `parse_dir` dispatches to `JobDirParser`, non-directory raises, result-kind discriminators unique, every `ParseResult` subclass frozen. |
| `test_coords.py` | 9 | Phase E coords parsers: `SiestaXVFileParser` claims `.XV` (uppercase) but not `.xml`, `PyscfGeomXyzFileParser` claims `*_optimized.xyz` but not plain `.xyz`, .XV round-trip carries cell + atomic-number→element mapping; `StructureResult` frozen. |
| `test_sidecars.py` | 7 | Phase D sidecar parsers: registered, claim correct suffixes (`.molstruct.json`, `.spectra.json`), dispatch via `detect`, return `SidecarResult` with parsed payload; frozen. |
| `test_scripts.py` | 10 | Phase F per-block TextParsers: HEADER / PROVENANCE / BENCH-MARKS / ATOM-METADATA / USER-CUSTOM extract / return-None / surface schema_version; `ScriptSourceTextParser` umbrella composes all blocks + handles no-blocks; `ScriptResult` frozen + I/O lint. |
| `dirs/test_job.py` | 20 | `JobDirParser` end-to-end on `TJ-BDT-Au111` + `BDT-withAuJunction` fixtures: typed `JobResult`, parse_dir dispatch, job-type classification (script-contract vs sniff vs ambiguous-raises), engine_input envelopes, engine_body_summary curated keys exact + raw string values + kgrid block extraction, multistage plot buckets, source_files index, geometry XYZ+cell, status shape, CG-step progress; frozen; `_no_direct_out_grep` source-level lint. |
| `dirs/test_bundle.py` | 9 | Phase G `BundleDirParser`: NOT in dispatch registry (vs `JobDirParser` collision), `can_parse` mirrors legacy precondition, real-fixture parse returns `BundleResult` w/ structure + regions + frozen_atoms + notes, BundleError propagates on empty + ambiguous-engine dirs; frozen; parser_name is slug. |
| `dirs/test_job_review_fixes.py` | 8 | Round-1 BLOCKER regressions: B2 LatticeConstant Å vs Bohr scaling + default-Bohr-when-unspecified, B3 `can_parse` rejects `.py`-only dir (claims fdf-only); I2 `engine_body_summary` case-insensitive + tab-separated + canonical keys unchanged. |
| `test_round2_fixes.py` | 14 | Round-2 BLOCKERs: anchor stage tie-breaker, B2 LatticeConstant word boundary, I2 tab/multi-space normalisation, I1 bool-not-charted, slug parser_name, sidecar + engine source paths resolved absolute, ambiguous-raises, migration legacy `parsers.detect_parser` still works (removed at H4), spectra payload JSON-serialisable, frozen-dataclass invariant per result kind. |
| `test_round3_fixes.py` | 5 | Round-3 BLOCKER: TrajectoryResult `frames` + `lattice` are copies not shared refs (frozen-invariant on inner mutables); I4 last-wins on duplicated fdf keys (SIESTA manual § 7.1) — single + duplicate + triple-override. |
| `test_round4_fixes.py` | 6 | Round-4 IMPORTANTs: D2 BOM-prefixed fdf parses + round-trips through `decode_run_dir`; E1 kgrid extracts diagonal + tolerates blank lines + tolerates comments + falls to None on malformed. |
| `test_audit_gaps.py` | 9 | **Half-Phase-H audit-gap closures (this commit family).** Public-surface coverage (`__all__` resolves), top-level re-exports identity-match sub-packages, `ParseWarning` constructible + frozen + None-tolerant, `parse_text` smoke via `ScriptSourceTextParser`, forbidden-pattern lints P2 (TextParsers no I/O), P3 (FileParsers no subprocess/network/threads), P6 (no engine names in core `types.py`/`base.py`/`registry.py`/`errors.py`). |

Levels — Most are L2 (in-process, fixture-driven, ≤100 ms).
The fixture-fed dirs/test_job.py + dirs/test_bundle.py +
test_coords.py XV round-trip are L3 (real project directories
under `projects/BDT/optimization/`).

## 11. Pinned references

* `docs/types/parsers.md` (will redirect here in Phase H) —
  current file-level parser contract; supersedes section §§ 1-4
  via this doc's § 2-3.
* `docs/protocols/script-contract.md` — the 6 reserved blocks in
  `.fdf` / `.py`; this doc's `parse/scripts/*` modules implement
  the per-block TextParsers.
* `docs/protocols/bundle-contract.md` — run-dir → handoff bundle;
  this doc's `parse/dirs/bundle.py` implements `BundleDirParser`.
* `docs/protocols/job-decoder.md` — directory-level decoded.json
  contract; this doc's `parse/dirs/job.py` implements
  `JobDirParser`.
* `docs/protocols/sidecar-contract.md` — molbuilder sidecar JSON
  conventions; this doc's `parse/sidecars/*` modules implement
  the per-kind FileParsers.

## 12. Decisions log

| Date | Decision |
|---|---|
| 2026-06-19 | Initial draft. Three ABCs (File / Text / Dir).  Frozen-dataclass `ParseResult` hierarchy with 6 concrete subclasses (TrajectoryResult / StructureResult / SidecarResult / ScriptResult / JobResult / BundleResult) atop the ParseResult base.  Phase A + B land in the first commit family; C-H follow incrementally. |
| 2026-06-19 | Phases C + D shipped — engine wrappers + sidecar wrappers.  Round-3 review pass: `_helpers.py` modules added to package tree; Phase H description now explicitly requires absorbing legacy logic before the legacy-module deletion (8 import sites identified). |
