# molbuilder — documentation index

The canonical index lives in **[`design.md`](design.md) § 0**.
Read that first.

**Before building anything**, consult **[`architecture.md`](architecture.md)**
— the design-foundation index of the major infrastructure/modules/APIs
(what already exists + which tool to reuse), the antidote to
reinventing/patching blind.

## The doc rule

> Tests must be derivable from the spec without reading the
> implementation. Code reviews must verify code matches spec, not
> code matches reviewer's expectations.

A bug shipped early in the project because a code review checked
the implementation against itself — tests asserted "the string
`mol = gto.M(...)` appears in the generated script" rather than
"the generated script must not truncate `<job>.log` between stages".
When the implementation was wrong, the test was wrong in lock-step.
The specs in this directory decouple the two.

## Categories at a glance

| Folder | What lives here |
|---|---|
| `protocols/` | How parts of the system talk to each other (HTTP API, CLI surface, JS module contracts, on-disk file layout, test patterns) |
| `tabs/` | Per-UI-tab specs + cross-tab `architecture.md` (`molbuilder`, `structure-optimization`, `spectra/`, `results`; transport-calculation tracked under `engines/transport.md`) |
| `engines/` | Per-engine emitter specs (SIESTA / PySCF) + build-backend contract |
| `types/` | L1 data-type contracts (Structure, parsers, chemistry helpers) |
| `archive/` | Superseded docs — NOT a source of truth |

## Versioning

This is a 1.x project. Spec changes that remove or rename
promised output files require a minor version bump (1.x → 1.x+1)
AND a deprecation note in design.md's decisions log. Adding new
optional fields or files is a patch-level change.

## Bibliographies

Bibliographies live alongside the spec they cite, in the same
subfolder. Adding a future tab with citations means a new folder
with both files together — exactly one place to look when reading
either.
