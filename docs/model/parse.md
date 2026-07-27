# The parse stack — turn a file or directory into typed data

**Role:** contract
**Domain:** model
**Module:** `molbuilder/parse/` · **Tests:** `tests/parse/` (~106 tests).
**Companions:** `structure.md` (a `StructureResult` carries a `Structure`);
`engines/siesta.md` + `engines/pyscf.md` (the `.out`/`.log`/geometry formats the
leaf parsers read, migrating); `execution/job-decoder.md` +
`execution/handoff-bundle.md` (the directory-level `JobResult`/`BundleResult`
contracts the two DirParsers implement, migrating). The **write** side (the
inverse — turning data back into files) is `sidecars/molstruct.py`,
`script_emit.py`, and `bundle_writer.py`, not this module.

One package answers a single question: **"what is in this path, as a Python
object?"** — for a file, a text body, or a whole run directory. It is the sole
read-side source of truth; every consumer (web blueprints, CLI, tests) imports
from here and queries the registry rather than knowing which parser to call.

> **Why it exists.** Before this, molbuilder had **four parallel parsing
> patterns** — a `TrajectoryParser` registry for engine output, ad-hoc
> `load(path)` functions for sidecar JSON, ad-hoc `read_*` functions for
> geometry, and ad-hoc `extract_*`/`decode_run_dir` for scripts and run dirs —
> with three different return-type flavours and no shared base. Each new engine
> or output type added another parallel path. This package collapses them into
> **three ABCs, one frozen-dataclass result hierarchy, and one registry**.

---

## 1. The three ABCs

`molbuilder/parse/base.py`:

| ABC | Input → output | Detection |
|---|---|---|
| **`FileParser`** (`:26`) | one file path → one `ParseResult` | `can_parse(path)` — the registry auto-detects |
| **`TextParser`** (`:79`) | one in-memory text body → one `ParseResult`; **pure, no I/O** | none — the caller passes the parser explicitly |
| **`DirParser`** (`:98`) | one directory → one `ParseResult`, **composed** from per-file parsers plus directory-level invariants | `can_parse(run_dir)` |

Each declares `name` / `label` / `output` (the concrete `ParseResult` subclass
it returns); file/dir parsers add a `hint` (what to point at when `can_parse`
is `False`). **`TextParser` has no detection** — a text body has no path to
inspect, so the caller names the parser: `parse_text(text, parser=…)`.

---

## 2. The `ParseResult` hierarchy

Every parser returns a **frozen dataclass** on a shared base
(`molbuilder/parse/types.py`). A consumer type-narrows by `isinstance` or by the
`result_kind` discriminator string.

```mermaid
classDiagram
    class ParseResult {
        schema_version : int
        parsed_at : str
        parser_name : str
        source : str
        result_kind : str
    }
    class TrajectoryResult {
        frames · lattice · run_state · runtime_info
        result_kind = "trajectory"
    }
    class StructureResult {
        structure : Structure · cell
        result_kind = "structure"
    }
    class SidecarResult {
        payload · schema
        result_kind = "sidecar"
    }
    class ScriptResult {
        header · provenance · bench_marks ·
        atom_metadata · user_custom
        result_kind = "script"
    }
    class JobResult {
        job_type · status · progress ·
        geometry · plots · source_files
        result_kind = "job"
    }
    class BundleResult {
        structure · regions · frozen_atoms · notes
        result_kind = "bundle"
    }
    ParseResult <|-- TrajectoryResult
    ParseResult <|-- StructureResult
    ParseResult <|-- SidecarResult
    ParseResult <|-- ScriptResult
    ParseResult <|-- JobResult
    ParseResult <|-- BundleResult
```

Plus **`ParseWarning`** (`types.py:36`) — a fail-soft warning (`source`,
`line_no`, `snippet`, `error`, `category`) any parser can attach to its result
instead of raising.

- **The discriminator.** Each concrete subclass sets `result_kind` to a fixed
  string (`"trajectory"`, `"structure"`, `"sidecar"`, `"script"`, `"job"`,
  `"bundle"`). Consumers holding a `ParseResult` (cached, or sent over the wire)
  `match` on it; JSON deserialisation reads it to pick a class. Adding a kind =
  a new subclass + a new discriminator value (rule below).
- **Why frozen.** No defensive copies at API boundaries; hashable (dict/set
  usable); trivial `asdict()` serialisation; `result == expected` test pinning
  with no `__eq__` override.

---

## 3. The registry + public API

`molbuilder/parse/__init__.py` re-exports the whole surface; the dispatch lives
in `registry.py`.

```mermaid
flowchart TD
    P["parse(path)"] --> D["detect(path)"]
    D -->|"path is a dir"| DP["first DirParser whose<br/>can_parse is True"]
    D -->|"path is a file"| FP["first FileParser whose<br/>can_parse is True"]
    D -->|"none match"| ERR["raise UnknownFormatError<br/>(lists every parser + hints)"]
    DP --> R["ParseResult"]
    FP --> R
    PT["parse_text(text, parser)"] -->|"no detection"| R
    PD["parse_dir(path)"] -->|"DirParsers only"| R
```

| Function (`registry.py`) | Does |
|---|---|
| `detect(path)` (`:60`) | return the first parser whose `can_parse(path)` is `True` (dir parsers first if `path` is a directory, then file parsers); `UnknownFormatError` if none, with every registered parser + the standard foot-gun hints |
| `parse(path)` (`:135`) | `detect` + `parse` in one call |
| `parse_text(text, parser)` (`:158`) | parse a known text body — **no detection**, caller names the `TextParser` |
| `parse_dir(path)` (`:144`) | detect among **DirParsers only** — for callers whose contract is "this is a run directory" (JobMonitor, Results, bundle handoff) |
| `register(parser)` (`:30`) | add a parser at module-init time (idempotent; not for runtime registration) |

**Errors** (`errors.py`): `UnknownFormatError` (`:13`) and
`AmbiguousFormatError` (`:23`), both on a `ParseError` base (`:9`).

---

## 4. Package layout

```
molbuilder/parse/
├── base.py        # the 3 ABCs                (FileParser / TextParser / DirParser)
├── types.py       # ParseResult + 6 subclasses + ParseWarning
├── registry.py    # _REGISTRY, detect/parse/parse_text/parse_dir/register
├── errors.py      # ParseError, UnknownFormatError, AmbiguousFormatError
├── _log.py        # parse-side logging helper
│
├── engines/       # engine .out / .log → TrajectoryResult (FileParsers)
│   ├── siesta.py · pyscf.py · molwatch.py
│   ├── _helpers.py            # Trajectory → TrajectoryResult adapters
│   └── _section_rules.py · _sidecar.py   # shared extraction helpers
│
├── coords/        # geometry files → StructureResult (FileParsers)
│   ├── siesta_xv.py           # .XV / .STRUCT_OUT (+ cell)
│   ├── pyscf_geom.py          # *_optimized.xyz
│   └── _helpers.py            # StructureResult envelope
│
├── sidecars/      # molbuilder JSON sidecars → SidecarResult (FileParsers)
│   ├── molstruct.py · spectra.py · transport.py
│   └── _helpers.py
│
├── scripts/       # the reserved .fdf / .py comment blocks → ScriptResult (TextParsers)
│   ├── header.py · provenance.py · bench_marks.py
│   ├── atom_metadata.py · user_custom.py
│   ├── source.py              # ScriptSourceTextParser — umbrella over the blocks
│   ├── markers.py             # MARKER_RE + block-name constants
│   └── _helpers.py
│
└── dirs/          # directory composers (DirParsers)
    ├── job.py                 # JobDirParser + decode_run_dir → JobResult
    ├── bundle.py              # BundleDirParser → BundleResult (explicit-dispatch)
    └── _assembler_helpers.py  # shared dir-walk + .fdf-coords helpers
```

> **Two geometry formats have no leaf FileParser (by design).** Plain `.xyz`
> reading uses `Structure.from_xyz` directly (`structure.md`), and `.fdf`
> initial-coordinates reading lives in `dirs/_assembler_helpers.py` (used by the
> DirParsers), rather than as registered `coords/` parsers. Adding them as
> FileParsers was scoped but not needed.

---

## 5. Composer pattern — DirParsers

A DirParser turns a whole run directory into one result. The two shipped:

- **`JobDirParser`** (`dirs/job.py:822`; public entry `decode_run_dir` `:736`) →
  `JobResult` — walks the `.out` files, consolidates per-source plots,
  classifies the job type. This is what the Results tab + JobMonitor consume.
- **`BundleDirParser`** (`dirs/bundle.py:460`) → `BundleResult` — picks the
  source `.fdf`, reads the `.XV` final coordinates, validates the atom count,
  for the next-stage handoff. It is **explicit-dispatch, not auto-registered**
  (it shares the `.fdf` claim with `JobDirParser` but expresses a different user
  intent), so `parse_dir` routes to `JobDirParser`; a caller names
  `BundleDirParser` directly.

Every DirParser must: **walk** the directory, **dispatch each file through the
registry** (`detect`+`parse`, or pick a specific FileParser when the choice
isn't path-driven), **compose** the per-file results, and **apply cross-file
invariants** no single FileParser can see (atom-count consistency, lattice
handedness, stage ordering, status state machine). It must **never re-parse**
what a registered FileParser can produce — add the missing FileParser instead.

---

## 6. Adding a parser (plugin contracts)

- **Engine FileParser** → `parse/engines/<engine>.py`: a `FileParser` with
  `output = TrajectoryResult`; `can_parse` checks content markers in the first
  ~1000 bytes; `register(...)` in `engines/__init__.py`; L2 test.
- **Sidecar FileParser** → `parse/sidecars/<kind>.py`: returns
  `SidecarResult(payload=…, schema="<kind>/vN")`; `can_parse` matches
  `.<kind>.json`.
- **Block TextParser** → `parse/scripts/<block>.py`: returns `ScriptResult`;
  uses `MARKER_RE` from `scripts/markers.py`; the caller invokes it via
  `parse_text(text, parser=<Block>Parser)`.
- **DirParser composer** → `parse/dirs/<purpose>.py`: **must compose existing
  FileParsers + TextParsers** (forbidden pattern #1 below), not parse files
  inline.

---

## 7. Forbidden patterns

These stop the next round of parallel parse paths:

1. **DirParsers compose registered parsers — no inline file-level parsing.**
   Need a new file read? Add a FileParser first. (Lint: `test_dir_parser_uses_registry`.)
2. **TextParsers do NO I/O.** A path-taking caller reads the file and passes the
   body.
3. **FileParsers do not spawn subprocesses, network calls, or threads** —
   parsing is pure local-file I/O; background work belongs to the JobMonitor.
4. **`ParseResult` subclasses are frozen.** Never mutate after construction; use
   `dataclasses.replace(result, …)`.
5. **A new `ParseResult` subclass requires a new `result_kind` value + a doc
   update** — one shape, one discriminator.
6. **Adding a curated key list** (e.g. `engine_body_summary`) requires a doc
   update + a test — these lists are the load-bearing contract for downstream
   consumers.
7. **No engine-specific code outside `parse/engines/` and `parse/coords/`.** The
   composer stays engine-agnostic; engine logic lives in the leaf parsers.

---

## 8. History

The unified stack replaced the four parallel patterns above (shipped
incrementally 2026-06-19 → 06-21). The legacy `molbuilder/parsers/`,
`script_contract.py`, and `script_bundle.py` are deleted; their **read** side is
here, and their **write** side rehomed to `molbuilder/sidecars/`,
`script_emit.py`, and `bundle_writer.py` (parsing and emitting are inverse
concerns, kept in separate modules).
