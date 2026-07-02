# Browser-owned transient data & explicit commit — contract

**Status: PROPOSED (2026-07-02).** Defines how data that lives *in the browser*
— loaded from a project file and possibly edited by the user — is held
transiently but persistently, and the single explicit point at which it may
overwrite a durable project file.

**The principle (one sentence):**

> **Browser-owned data is transient. The project files (`.xyz` +
> `.molstruct.json`) are the durable copy and are overwritten ONLY on explicit
> user instruction.**

**Companions:** `workspace-contract.md` (the dispatcher/store this builds on),
`data-vocabulary.md` §3.1–3.2 (atom index base + the atom-identity hash that the
commit gate enforces), `atom-annotations.md` §6.1 (the data-loss contract this
refines), `run-checkpoints.md` (a *different* "checkpoint" — engine restart
files; see §7).

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
- **Naming.** `.molbuilder_workspace/` — deliberately **not**
  `.molbuilder_checkpoint/`, to avoid collision with `run-checkpoints.md`
  (engine restart files, a different concept). *(Naming decision — open;
  `.molbuilder_checkpoint/` was suggested. Either works; pick one and pin it.)*
- **Contents.** Each scratch record carries: the working structure/annotations,
  the **source path** it was loaded from, the **source hash** at load, the
  session id, and a timestamp.
- **Lifecycle.** Written/updated on edit (debounced, like the sessionStorage
  snapshot); removed on successful commit; aged out / cleaned on session end.

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
- **`run-checkpoints.md`**: unrelated — that is *engine* restart data (`.XV`,
  `.DM`, …) written by a running job. The name "checkpoint" is why this
  contract's scratch dir is `.molbuilder_workspace/`, not `.molbuilder_checkpoint/`.
- **`workspace-contract.md`**: the dispatcher/store that owns the transient
  state and the `dirty` flag.

---

## 8. Open decisions (for sign-off before implementation)

1. **Scratch dir name** — `.molbuilder_workspace/` (recommended, collision-free)
   vs `.molbuilder_checkpoint/` (suggested).
2. **Structure-in-scratch** — always mirror the working `.xyz` to scratch, or
   only when the structure (not just labels) was edited (saves I/O for
   label-only sessions).
3. **Commit UX on mismatch** — refuse-only (simplest) vs the warn+choose
   (keep / discard-stale / save-as-new) flow from the save-path discussion.
4. **Scratch cleanup** — on commit + session end only, or also an age-out sweep
   for abandoned sessions.
