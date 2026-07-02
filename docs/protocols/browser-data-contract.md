# Browser-owned transient data & explicit commit — contract

> **SUPERSEDED (2026-07-02).** This doc was built around a "changed-underneath" hash-gate + a laundering analysis. That premise was wrong — a save writes the whole self-consistent `.xyz`+`.json` pair, so there is nothing to launder and no gate is needed. The real model is the simple **load → edit → save (overwrite/save-as)** in [`working-copy-persistence.md`](working-copy-persistence.md). Treat the sections below as historical; they are being rewritten to just the browser wiring (hold labels in the working copy, Save/Save-As calls `/api/workingcopy/save`).

**Status: ACCEPTED (2026-07-02).** Decisions in §8 are locked. Defines how data that lives *in the browser*
— loaded from a project file and possibly edited by the user — is held
transiently but persistently, and the single explicit point at which it may
overwrite a durable project file.

**The principle (one sentence):**

> **Browser-owned data is transient. The project files (`.xyz` +
> `.molstruct.json`) are the durable copy and are overwritten ONLY on explicit
> user instruction.**

**This is one application of `working-copy-persistence.md`** — the system-wide,
format-agnostic working-copy core. This doc = that core's structure+sidecar
codec + the browser-side wiring (sessionStorage / `writeLabel` / commit UX).
Generic guarantees (hash-gate, scratch, crash-recovery, atomic commit) live in
the core; only the `.xyz`+`.json` specifics live here.

**Companions:** `working-copy-persistence.md` (the core this applies),
`workspace-contract.md` (the dispatcher/store this builds on),
`data-vocabulary.md` §3.1–3.2 (atom index base + the atom-identity hash the
commit gate enforces), `atom-annotations.md` §6.1 (the data-persistence contract
this refines).

---

## 1. Motivation — two problems this solves

1. **Atom-identity laundering.** Today the selection panel's `writeLabel`
   auto-POSTs `/api/selection/save`, which writes the project
   `.molstruct.json` on *every* label assignment. That save re-stamps the
   sidecar with the current file hash — so a label made against a stale
   structure gets a *matching* hash and **sails through the generation-time
   hash gate**. Auto-save is the leak.
2. **Inconsistency.** Structure edits are already transient (held in
   `sessionStorage`, committed only on explicit save + a `dirty` flag), but
   label / frozen / region edits are bonded straight to the project file. This
   contract makes both obey the same rule.

---

## 2. The two tiers

| Tier | What | Where | Durability |
|---|---|---|---|
| **Working copy** (transient) | structure **+** labels/frozen/selection/annotations, together | browser `sessionStorage` snapshot **+** a server-side **scratch file** (§3) | session-scoped; survives reload / tab-switch / server restart; **not** the durable copy |
| **Project copy** (durable) | `<stem>.xyz` + `<stem>.molstruct.json` | the project folder | permanent; written **only** on explicit "Save to project" (§4) |

The working copy always carries the **source hash** — the `sha256` of the
project `.xyz` at the moment it was loaded — so a later commit can prove the
structure hasn't changed underneath.

---

## 3. Scratch convention (the temp dir/file)

Server-side transient data lives in a per-project scratch directory:

```
<project_dir>/.molbuilder_workspace/
    <stem>.<session>.molstruct.json     # working annotations + source-hash + provenance
    <stem>.<session>.xyz                # working structure, ONLY if the structure was edited
```

- **Location.** A dot-directory *inside the project folder* (project-scoped,
  survives server restart, easy to find/clean, `.gitignore`-able). Created
  lazily on first edit.
- **Naming.** `.molbuilder_workspace/` (locked, §8 #1). Gitignore-able so
  transient work is never committed to version control.
- **Contents.** Each scratch record carries: the working structure/annotations,
  the **source path** it was loaded from, the **source hash** at load, the
  session id, and a timestamp.
- **Lifecycle.** Written/updated on edit (debounced, like the sessionStorage
  snapshot); removed automatically **only** on successful commit or session-end;
  crash-orphans removed via the explicit cleanup (§8.1) — never on a timer.

---

## 4. Lifecycle & the hash gate

```
LOAD      read <stem>.xyz (+ .molstruct.json) → working copy.
          record source_hash = sha256(<stem>.xyz).            [no scratch write yet]

WORK      every edit (label / frozen / structure) updates the WORKING COPY only.
          mirror to scratch + sessionStorage (debounced).     [NO project write]

COMMIT    "Save to project" — the ONLY write to project files:
   (explicit)  1. gate: assert working.source_hash == sha256(on-disk <stem>.xyz).
                  MISMATCH → refuse (the atom-group gate): the file changed since
                  load; the labels were made for the old atom order.  User
                  decides: save-as a NEW file, or reload+redo.  Never launder.
               2. confirm the target explicitly (same file, or a new path).
               3. write <stem>.xyz + <stem>.molstruct.json ATOMICALLY.
               4. clear dirty; remove the scratch record.
```

The commit gate is the *same* hard-refuse the generation path uses — but because
project files are now written *only* here, a laundered sidecar can no longer be
produced, so the generation gate is never bypassed.

---

## 5. Invariants (what the contract guarantees)

1. **No project file is written without an explicit user commit.**
2. **The working copy always carries the source hash it was loaded against.**
3. **Commit hard-refuses on hash mismatch** — the atom-identity gate; never
   launders stale indices into a fresh-looking sidecar.
4. **Transient data is consistent across reload within a session** (scratch +
   sessionStorage), so "browser owns transient, persistent, consistent data."
5. **Editing is non-destructive** to the project until commit — the user can
   always walk away and the on-disk files are untouched.

---

## 6. What changes (integration; build on what exists)

- **`writeLabel`** updates the working copy (+ scratch), and **no longer**
  auto-POSTs `/api/selection/save`.
- **`/api/selection/save`** becomes (or is replaced by) the explicit
  **commit** endpoint — hash-gated, target-confirmed, atomic.
- **Server-side scratch** read/write API for `.molbuilder_workspace/`.
- Reuse the existing `sessionStorage` snapshot, `dirty` flag, and
  `workspace-contract.md` §4.5 (`mountRestoreTarget`) — the transient plumbing
  already exists for structure; extend it to carry labels/frozen/annotations.

---

## 7. Relationship to other contracts

- **`data-vocabulary.md` §3.2** (atom-identity): the commit gate enforces the
  source-hash equality; this contract is what makes that gate *sufficient* (by
  removing the auto-save launder path).
- **`atom-annotations.md` §6.1** (data-loss): this **refines** "durable only on
  Save" → "durable only on explicit *commit to project*; transient work is held
  in `.molbuilder_workspace/` scratch, safe across reloads."
- **`workspace-contract.md`**: the dispatcher/store that owns the transient
  state and the `dirty` flag.

---

## 8. Decisions (locked 2026-07-02)

1. **Scratch dir name** — `.molbuilder_workspace/`.
2. **Structure-in-scratch** — mirror the working `.xyz` to scratch **only when
   the structure was edited**; a label-only session keeps just the working
   `.molstruct.json` record (the source structure stays the on-disk `.xyz`).
3. **Commit UX on mismatch** — **warn + choose**: surface the mismatch and let
   the user pick *keep* (accept + save as a new file), *discard-stale* (drop the
   carried labels, keep only the current edit), or *reload + redo*. Hard-refuse
   is the safe MVP to ship first; the choose-flow is the enhancement.
4. **Scratch cleanup** — automatic **only** on successful commit or session-end
   (no time-based age-out sweep — never delete possibly-unsaved work on a timer).
   Crash-orphaned scratch (where neither fired) is handled solely by the
   **explicit user cleanup** (§8.1).

### § 8.1 Crash recovery — explicit scratch cleanup

A crash (browser, server, or OS) can leave a scratch record in
`.molbuilder_workspace/` that **no automatic path will remove** — the session
that owned it is gone, so neither commit nor session-end cleanup ever fires.
The contract therefore REQUIRES an explicit, user-visible cleanup:

- **List** — surface orphaned scratch for a project (records whose session is no
  longer live): source file, source-hash, age, and whether it still matches the
  on-disk `.xyz`.
- **Recover or discard** — per record: *recover* (adopt it back into a working
  session, subject to the same commit hash-gate) or *discard* (delete the
  scratch file). Never auto-delete a record holding unsaved work without the
  user's say-so.
- **Clean all** — a one-shot "clear this project's workspace scratch" for a user
  who just wants a clean slate.

This preserves "editing is non-destructive + recoverable" across a crash —
without ever silently deleting unsaved work or silently adopting stale work.
