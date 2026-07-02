# Working-copy persistence — the transient-data foundation

**Status: PROPOSED (2026-07-02).** The system-wide, **format-agnostic** module
for holding *browser-owned / user-edited* data that is **not durable until
explicitly committed**. The `.xyz`+`.molstruct.json` case
(`browser-data-contract.md`) is the **first application** of this core, not the
core itself.

**The principle (one sentence):**

> **A working copy is transient: loaded from a source, mirrored to scratch so it
> survives reloads/crashes, and written to a durable target ONLY on explicit
> commit — gated by the source hash so a changed source can never be silently
> overwritten.**

**Companions:** `browser-data-contract.md` (the `.xyz`+`.json` *application* of
this core), `data-vocabulary.md` §3.2 (the atom-identity hash one application
feeds this core).

---

## 1. Why a foundation (not a one-off)

Any tab that loads a project artifact, lets the user change it, and might save it
back has the *same* needs: keep the edit safe across a reload, never write the
project file behind the user's back, and refuse to overwrite a source that
changed underneath. Structures are one such artifact; config files, generated
scripts, and future project artifacts are others. This module solves that once,
generically, so each application supplies only a **codec** (§4) and inherits the
safety guarantees (§5).

---

## 2. The two tiers (where this sits)

| Tier | What | Granularity | Trigger | Lifetime |
|---|---|---|---|---|
| **Transient (this doc)** | working-copy core → scratch | one artifact | auto/debounced on edit | until commit or session-end |
| **Durable** | project files (`.xyz`, `.json`, …) | one artifact | explicit commit | permanent |

Data flows *up*: edit → transient scratch → **commit** → durable file.

---

## 3. Core concepts

- **Source** — where the artifact was loaded from (a path), or *none* (a
  freshly-built artifact).
- **Source hash** — `sha256` of the source at load time. The gate compares it to
  the source's *current* on-disk hash at commit; a change means "someone/
  something edited the source underneath — do not silently overwrite."
- **Working copy** — the in-memory current state + its source + source hash +
  dirty flag.
- **Scratch record** — the working copy persisted server-side under
  `<project>/.molbuilder_workspace/`, so it survives reload / server restart /
  crash. Keyed by `(source-stem, session)`.
- **Codec** — the *only* format-specific part (§4), supplied by the application.
- **Commit** — the single hash-gated, atomic write to a durable target.

---

## 4. The codec interface (what an application supplies)

The core treats artifact data as **opaque**; the application plugs in:

```
codec.load(source_path)      -> data            # read durable -> working data
codec.hashSource(source_path)-> str             # the source hash (e.g. sha256 of the .xyz)
codec.files(data, target)    -> [(path, bytes)] # durable file(s) this artifact writes
codec.scratchBlob(data)      -> bytes/json      # how the working copy is stored in scratch
codec.fromScratch(blob)      -> data            # inverse (crash recovery)
```

`.xyz`+`.json` supplies a codec whose `files()` returns *two* paths (the `.xyz`
and the `.molstruct.json`) and whose `hashSource()` is `sha256(.xyz)`. A
config-file application returns one path. **The core never learns what an atom
is.**

---

## 5. The core API (format-agnostic)

```
open(source_path, codec)      -> WorkingCopy     # load + record source hash (no scratch write yet)
openNew(codec)                -> WorkingCopy     # freshly-built artifact, no source (see §7.2)
recover(scratchRecord, codec) -> WorkingCopy     # crash recovery (§7 / §8)

WorkingCopy.update(data)                         # mirror to scratch, debounced; sets dirty
WorkingCopy.data() / .isDirty() / .sourceHash()
WorkingCopy.commit(target_path, {onMismatch})    -> CommitResult
WorkingCopy.discard()                            # drop working copy + scratch, no write

# Crash recovery / housekeeping (project-scoped)
listOrphans(project)          -> [ScratchRecord] # records whose session is no longer live
discardOrphan(record) / cleanAll(project)
```

`commit` runs the gate → writes → clears scratch (§7.3). `onMismatch` is
`refuse` (MVP) or `choose` (keep / discard-stale / reload — the application's UX).

---

## 6. Invariants (the guarantees every application inherits)

1. **No durable file is written without an explicit `commit`.**
2. **A working copy always carries the source hash it was opened against.**
3. **Commit never launders** — on source-hash mismatch it refuses or forces an
   explicit user choice; it never silently writes stale data under a fresh hash.
4. **Transient data survives reload/crash** (scratch is authoritative server-side;
   any client mirror like `sessionStorage` is a cache — scratch wins on conflict).
5. **Editing is non-destructive** to durable files until commit.

---

## 7. Risks the core resolves once (so applications don't)

### §7.1 Concurrent sessions / the durable target moved
The gate compares source hash to the **current** on-disk source at commit — which
catches BOTH an external edit AND another session having committed in the
meantime (the target's hash changed). Either way: mismatch → refuse/choose. No
last-writer-wins clobber.

### §7.2 No source (freshly-built artifact)
`openNew()` has no source hash. Its first `commit` is a plain **save-as** to a
chosen path (nothing to mismatch); the gate engages only once a source exists.

### §7.3 Multi-file artifacts + partial-failure (real, not "atomic")
Two files (`.xyz`+`.json`) cannot be atomic together. The core instead:
- writes each file via **temp-file + rename** (per-file atomic),
- in a **defined order**,
- and **keeps the scratch record until ALL files are written** — so a partial
  failure leaves the scratch intact for retry, and the durable state is never
  half-committed-then-forgotten. Scratch is cleared only on full success.

### §7.4 Authority
Server-side scratch is the source of truth for the working copy. A browser
`sessionStorage` mirror is a same-tab fast-restore cache; on divergence the
scratch record wins.

---

## 8. Crash recovery (explicit, never silent)

A crash leaves a scratch record no automatic path can clear (its session is
gone). `listOrphans` surfaces them (source, hash, age, still-matches-source?);
the user `recover`s (adopt back, subject to the same commit gate) or `discard`s;
`cleanAll` wipes a project's scratch. The core **never** auto-deletes unsaved
work and **never** auto-adopts stale work.

---

## 9. Applications

| Application | Codec `files()` | Source hash | Spec |
|---|---|---|---|
| **Structure + sidecar** | `<stem>.xyz` + `<stem>.molstruct.json` | `sha256(.xyz)` | `browser-data-contract.md` |
| *(future)* config / script artifacts | their file(s) | their source hash | reuse this core unchanged |

---

## 10. Relationship to other contracts

- **`browser-data-contract.md`** — the *first application*; it now reads as "the
  structure+sidecar codec + the browser-side (sessionStorage/writeLabel/commit
  UX) wiring on top of this core."
- **`workspace-contract.md`** — owns the in-browser dispatcher/store + `dirty`
  flag that drives this core's client side.

---

## 11. Open decisions (before implementation)

1. **Where the core lives** — a new backend module (e.g.
   `molbuilder/workingcopy.py`) exposing the §5 API, plus a thin
   `/api/workingcopy/*` surface; the browser store calls it.
2. **Scratch record format** — one JSON envelope `{source, source_hash, session,
   ts, blob}` per record vs the codec owning the on-disk shape.
3. **Commit ordering for `.xyz`+`.json`** — `.xyz` then `.json`, or `.json`
   (metadata) then `.xyz`; define the partial-failure recovery direction.
4. **`onMismatch` default** — ship `refuse`, add `choose` (per
   `browser-data-contract.md` §8 #3) as the enhancement.
