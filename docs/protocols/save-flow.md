# Save-flow contract — the Save-panel UI

> **The persistence MODEL is defined in
> [`workspace-contract.md`](workspace-contract.md) (§4 — memory is the truth; the
> server draft, the `sessionStorage` cache, and the durable files; §4.6 — the
> working-copy mechanism).** This doc does **not** restate the model; it is the
> authoritative contract for the **Save-panel UI only** — the dialog, the
> overwrite-confirm, and the button states on the Modify tab. Where the two touch
> (what a Save writes, and when), workspace-contract.md governs and this doc defers.
>
> The Save button on the Modify tab MUST behave exactly as specified here. Code that
> diverges is incorrect by definition; the contract is right and the code is wrong.
>
> **Companion docs:**
>
> * [`workspace-contract.md`](workspace-contract.md) — the persistence model +
>   the `ws.*` API surface, including `ws.save()` and `ws.getStructure()`.
> * [`sidecar-contract.md`](sidecar-contract.md) — `.molstruct.json`
>   on-disk schema + region/frozen_atoms semantics.
> * [`projects-sidebar.md`](projects-sidebar.md) — sidebar's current
>   directory tracking + onCommit / onChange.

---

**New to this?** Start with the plain-language [`structure-guide.md`](../structure-guide.md) — the build+save lifecycle (the load gate, the six Source panels, saving + sidecar, gotchas). This doc is the precise Save contract.

## §1 Principles

Three rules govern the Save flow.  They are intentionally simple
so the user always knows where their work lands:

1. **The user names the file — there is no default.**  Every Save click opens a
   dialog with a **blank** filename box; Save stays disabled until the user types
   a name.  The Modify tab exists to *modify* the structure, so a save is a
   **save-as** to a file the user names — we never pre-fill the loaded (or
   last-saved) name, which would invite silently overwriting the source.

2. **The directory is always the sidebar's current project dir.**
   There is no "Save back to original location" semantic.  If the
   user has navigated the sidebar to a different project, the Save
   goes to the new project.  If the user wants to save back where
   they loaded from, they navigate the sidebar there first.

3. **Overwriting an existing file is ALWAYS confirmed.**  If the chosen name
   already exists in the project dir, the user must confirm before it is
   replaced — there is **no** save-back-to-source skip.

These rules make the Save semantics one-line learnable:

> "Save commits the current workspace into the project I'm looking
> at, under a filename I type; replacing an existing file is always confirmed."

---

## §2 The Save panel — button enablement

Located in `lib/structure/save.js::refreshState()`.  The button
disables when ANY of the following is true:

| Disabled when | Why |
|---|---|
| Workspace is empty | Nothing to save |
| No sidebar current dir is set | No destination |
| In-flight save is pending | Prevent double-click double-fire |

The readout text reflects the next user action:

| State | Readout |
|---|---|
| Backing file + clean | "Target: `<basename>`" |
| Backing file + dirty | "Unsaved — Target: `<basename>`" |
| Generator + sidebar dir | "Save as… into `<project-name>`/" |
| Generator + no dir | "Pick a project directory in the sidebar to Save as…" |
| Empty workspace | "" |

---

## §3 The Save flow — step by step

The implementation lives in `lib/structure/save.js::save()`.  On
Save-button click:

### §3.1 Resolve destination directory

```
dir = projects.getCurrentDir()
if not dir:
    refuse with "Pick a project directory in the sidebar before saving."
```

### §3.2 Open the name dialog — BLANK (§1)

There is **no default filename**.  The Modify tab makes a modified version, so the
user names a new file — never the loaded/last-saved name.

```
saveDialog.chooseSaveName("")     # empty box; Save disabled until a name is typed
```

The dialog enforces:

| Rule | Why |
|---|---|
| Non-empty (after trim) | The user must name the file |
| No `/` or `\` | Path-separator → navigate the sidebar instead |
| Not `.` or `..` | Reserved names |

If no dialog is mounted the save cannot proceed (there is no name to save to).

### §3.3 Compute final path

```
final_path = dir + "/" + chosen_name
```

### §3.4 Save the dataset — ONE atomic call, overwrite ALWAYS confirmed (§1)

`save.js::_saveDataset` asks the workspace to serialise ITSELF via
`ws.getScratchBlob()` → `{xyz, sidecar:{regions, frozen_atoms, cell, axis_kind,
vacuum, kgrid}}` (built from the §1.2.1 accessors — save.js no longer hand-rolls the
scan) and writes it with a SINGLE `/api/workingcopy/save` call, identifying the draft
to drop via `ws.draftIdentity()` — the server writes the `.xyz` + `.molstruct.json`
pair (§4).

```
env = POST /api/workingcopy/save {source, target: final_path, data: blob, overwrite: false}
if env.status == 409:                            # file already exists
    proceed = await saveDialog.confirmOverwrite(basename(final_path))
    if not proceed:
        return {ok: false, cancelled: true}
    env = POST /api/workingcopy/save {source, target: final_path, data: blob, overwrite: true}
```

Overwrite is **always** confirmed — there is no save-back-to-source skip.  Any
chosen name that already exists re-prompts.  A failure is **surfaced** (the save
returns `{ok:false, error}`), never swallowed.

### §3.5 Post-save — mark saved

On `env.ok`:

```
structurePage.markSavedTo(final_path)     # dirty clears + last_save_to = final_path
```

The `.xyz` + `.molstruct.json` pair was already written together by the single
`/api/workingcopy/save` call in §3.4 (§4 / workspace-contract.md §4.0: the
WHOLE store's sidecar, hash-tied — regions + frozen_atoms + periodicity
gathered from `ws.getAtoms()` / `ws.getStructure()`).  There is no separate
sidecar POST.

---

## §4 Sidecar handling — XYZ + `.molstruct.json` pair

The `.molstruct.json` sidecar holds region labels + frozen_atoms
indexed against the XYZ.  See
[`sidecar-contract.md`](sidecar-contract.md) for the file schema.

### §4.1 Live label edits while modifying

When the user clicks Assign / Add to target / Remove from target
in the selection panel, `selection.store::writeLabel` applies the
label change PURELY IN MEMORY — no HTTP, no disk write.  Like every
other modify op, the change stays in the store until an explicit
Save serialises the whole store (regions + frozen + periodicity)
into the sidecar.  The in-memory atoms are updated in place (no
`_fetchAtoms()` disk re-read — that would clobber unsaved modifier
ops).

### §4.2 Every save writes the WHOLE store's sidecar

There is **no** save-back-vs-save-as distinction (workspace-contract.md §4.0 — the
store is the truth; a save writes it whole).  `ws.getScratchBlob()` serialises the
store (the ONE serialiser, shared by save + the crash-draft):

1. Gather from the store:
   * `regions: {label: [indices]}` — from each atom's `labels[]` (`ws.getAtoms()`)
   * `frozen_atoms: [indices]` — atoms with `isFrozen=true`
   * `periodicity: {cell, axis_kind, vacuum, kgrid}` — from `ws.getStructure()`
2. The single `/api/workingcopy/save` call carries the whole blob (`{xyz, sidecar}`)
   — the server writes the `.xyz` then **REPLACES** the entire sidecar atomically,
   recomputing `structure_hash` from the just-written XYZ, so the `.xyz` + `.json`
   pair is always coherent (no merge with prior contents).  (Before the 2026-06
   unification this was a separate `/api/selection/save-sidecar` POST after a
   `writeFile`; both were removed.)

This fires on EVERY save, even with no labels — it wipes any stale sidecar at the
destination so the store's authoritative state (including "no labels") is what
lands.  Unlike the old fire-and-forget sidecar POST, a failure now **fails the whole
save** (`{ok:false, error}`) — `markSavedTo` runs only on success, so a lost `.json`
can never be silent.

### §4.3 The atomic-write contract

`/api/files/write` (and every JSON sidecar writer in `molbuilder/sidecars/`)
follows the atomic-write pattern:

1. `tempfile.mkstemp(prefix=name + ".", suffix=".tmp", dir=parent)`
2. `os.fdopen(fd, "w", encoding="utf-8")` + write + flush + fsync
3. `os.replace(tmp, dest)` — same-filesystem rename

This guarantees that a crash mid-write can never leave the
destination file half-written.  Symlink writes are explicitly
refused before the atomic write begins.

---

## §5 Default focus + safe-action discipline

Both modals follow the warning-modal pattern (per
`lib/structure/warning-modal.js`):

| Modal | Default focus | Destructive action |
|---|---|---|
| `chooseSaveName` | the filename input (auto-selected) | Save (Enter accepts) |
| `confirmOverwrite` | Cancel | Overwrite (requires explicit travel) |

ESC always resolves to Cancel/null (the safe action).

---

## §6 Single-instance dialogs

Both `chooseSaveName` and `confirmOverwrite` are single-instance:
calling either while a prior call's modal is still open returns the
SAME pending promise.  Two rapid Save clicks don't stack two
dialogs.

---

## §7 Test compliance map

Every clause is pinned by a test ID:

| Clause | Pinning test |
|---|---|
| §1.1 dialog opens on every Save | `test_save_writes_to_source_and_clears_dirty` |
| §1.2 directory = sidebar.currentDir | `test_uses_last_save_to_basename_when_set` |
| §1.3 overwrite confirm + cancel | `test_save_dialog_rename_to_existing_file_prompts_overwrite`, `test_save_dialog_overwrite_cancel_aborts` |
| §2 button enablement | `test_save_button_disabled_for_smiles_without_prior_save` |
| §3.3 filename input rules | `test_rejects_path_separators`, `test_empty_name_keeps_save_disabled` |
| §3.5 pre-confirm same-path | inlined in `test_writes_to_source_file_and_marks_saved` |
| §4.1 label writes preserve workspace | `test_panel_assign_works_on_dirty_workspace_after_electrode` |
| §4.3 Save-as sidecar propagation | `test_save_as_propagates_labels_to_new_sidecar` |
| §4.4 atomic write | tested implicitly via `tests/test_web_files.py` |
| §5 default focus + ESC | `test_cancel_is_default_focus`, `test_esc_resolves_null` |
| §6 single-instance | `test_single_instance_reuses_promise` |

---

## §8 Future work

1. **Save-as overwrite-confirm for sidecar.**  When Save-as
   destination has an EXISTING sidecar at the new path, the user
   should be warned that the existing labels would be replaced
   (not just the XYZ).  Today the XYZ overwrite confirm covers
   only the XYZ; the sidecar gets silently rewritten by the §4.3
   label propagation.

2. **Bulk sidecar-write endpoint.** *(Done — superseded.)* The
   whole-store sidecar is now written in one shot by
   `/api/workingcopy/save` (which carries `{regions, frozen_atoms,
   periodicity}` alongside the XYZ). The interim
   `/api/selection/save-sidecar` + `/api/selection/refresh-hash`
   endpoints have been removed.

3. **Sidebar locked during in-flight save.**  Per
   `projects-sidebar.md` § C2, file-list mutations are gated on
   the sidebar lock.  Save should acquire the lock for the
   duration of the `/api/workingcopy/save` call so concurrent
   file mutations can't race.

---

## §9 Change process

1. PR the contract change AND the code AND the test together.
2. Update §7 if the test ID changes.
3. NEVER ship a code change that diverges from this contract.  If
   the contract is wrong, change it explicitly.
