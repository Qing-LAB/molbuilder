# Save-flow contract — sole source of truth

> **This document is the authoritative contract for the Save panel
> + dialog + sidecar handling.**  The Save button on the Modify
> tab MUST behave exactly as specified here.  Code that diverges
> is incorrect by definition; the contract is right and the code is
> wrong.
>
> **Companion docs:**
>
> * [`workspace-contract.md`](workspace-contract.md) — `ws.*` API
>   surface, including `ws.save()` and `ws.getStructure()`.
> * [`sidecar-contract.md`](sidecar-contract.md) — `.molstruct.json`
>   on-disk schema + region/frozen_atoms semantics.
> * [`projects-sidebar.md`](projects-sidebar.md) — sidebar's current
>   directory tracking + onCommit / onChange.

---

## §1 Principles

Three rules govern the Save flow.  They are intentionally simple
so the user always knows where their work lands:

1. **The user confirms the filename.**  Every Save click opens a
   dialog with the destination filename pre-filled.  The user can
   accept the default by pressing Enter or clicking Save, or edit
   the name before confirming.

2. **The directory is always the sidebar's current project dir.**
   There is no "Save back to original location" semantic.  If the
   user has navigated the sidebar to a different project, the Save
   goes to the new project.  If the user wants to save back where
   they loaded from, they navigate the sidebar there first.

3. **Overwriting an existing file requires explicit confirmation.**
   The exception is "Save back to the workspace's current source
   file" (same dir + same name as `last_save_to`) — that's
   unambiguous and skips the second confirm.

These rules make the Save semantics one-line learnable:

> "Save commits the current workspace into the project I'm looking
> at, using the filename I confirm."

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

### §3.2 Resolve default filename

```
path = workspace.last_save_to or source.file
if path:
    default_name = basename(path)        # e.g. "water.xyz"
else:
    src = workspace.source
    default_name = src.kind + ".xyz"     # e.g. "smiles.xyz"
```

### §3.3 Show the confirm-name dialog

`saveDialog.chooseSaveName(default_name)` opens a modal with:

* a single text input pre-filled with `default_name` (auto-focused,
  auto-selected so Enter accepts or typing replaces)
* Cancel + Save buttons
* an inline error slot for validation messages

The dialog enforces these filename rules:

| Rule | Why |
|---|---|
| Non-empty (after trim) | Empty path clobbers nothing useful |
| No `/` or `\` | Path-separator → user should navigate sidebar instead |
| Not `.` or `..` | Reserved names |

The Save button + Enter key both validate before resolving.

### §3.4 Compute final path

```
final_path = dir + "/" + chosen_name
```

### §3.5 Write with the overwrite gate

```
pre_confirmed = (final_path == path)   # exactly the source file
write_envelope = projects.writeFile(final_path, struct.text, {
    overwrite: pre_confirmed,
})
if write_envelope.status == 409:
    # File exists, overwrite was false
    proceed = await saveDialog.confirmOverwrite(basename(final_path))
    if not proceed:
        return {ok: false, cancelled: true}
    write_envelope = projects.writeFile(final_path, struct.text, {
        overwrite: true,
    })
```

The `pre_confirmed` rule: clicking Save with the EXACT same path
the workspace was loaded from (or last saved to) is unambiguous —
no second confirm needed.  Any other path (renamed, different
sidebar dir, generator save-as) routes through the confirm gate.

### §3.6 Post-write sync

On `write_envelope.ok`:

```
structurePage.markSavedTo(final_path)
# Workspace dirty bit clears + last_save_to records final_path.

# Sidecar housekeeping — see §4.
fetch("/api/selection/refresh-hash", {structure_path: final_path})
```

---

## §4 Sidecar handling — XYZ + `.molstruct.json` pair

The `.molstruct.json` sidecar holds region labels + frozen_atoms
indexed against the XYZ.  See
[`sidecar-contract.md`](sidecar-contract.md) for the file schema.

### §4.1 Live label edits while modifying

When the user clicks Assign / Add to target / Remove from target
in the selection panel, `selection.store::writeLabel` POSTs
`/api/selection/save` with the workspace's `n_atoms` so the server
validates indices against IN-MEMORY state (not disk).  The server
writes the sidecar AT THE WORKSPACE'S CURRENT SOURCE PATH.  The
in-memory atoms are updated in place from the server's response
(no `_fetchAtoms()` disk re-read — that would clobber unsaved
modifier ops).

### §4.2 Save back to source (same path)

When `final_path == workspace.source.file`, the sidecar at that
path already exists (it's the one writeLabel wrote to).  After
the XYZ write, `/api/selection/refresh-hash` recomputes the
sidecar's `structure_hash` against the just-written XYZ bytes,
preserving regions + frozen_atoms verbatim.  The XYZ + sidecar
end up fully coherent.

### §4.3 Save-as (new path) — label propagation

When `final_path != workspace.source.file` (any rename or different
sidebar dir), `_postWriteSuccess` propagates the workspace's
in-memory labels to the destination's sidecar atomically via the
bulk-replace endpoint `/api/selection/save-sidecar`:

1. Traverse `ws.getAtoms()` to gather:
   * `regions: {label: [indices]}` — one entry per atom's `labels[]`
   * `frozen:  [indices]` — atoms with `isFrozen=true`
2. POST `/api/selection/save-sidecar` with `{structure_path,
   n_atoms, regions, frozen_atoms}` — server **REPLACES** the
   entire sidecar atomically (no merge with prior contents).
   The workspace is the authoritative source; any stale labels
   that pre-existed at the destination's sidecar are wiped.
3. THEN fire `/api/selection/refresh-hash` so the destination
   sidecar's `structure_hash` matches the just-written XYZ.  The
   refresh sequences AFTER the label write via Promise chaining.

The bulk-replace call ALWAYS fires on Save-as, even when the
workspace has no labels — this wipes any stale sidecar at the
destination so the user's authoritative state is reflected.
Without this, saving an unlabelled workspace to a previously-
labelled file would silently preserve the old labels.

**Why bulk-replace instead of N+1 per-target calls:**
`/api/selection/save` has REPLACE-per-target semantics (only the
named region/frozen_atoms is replaced; other regions are
preserved).  Per-target writes from a Save-as would MERGE the
workspace's labels with whatever was already at the destination
— silently wrong.  The bulk endpoint takes the full sidecar
payload and writes it atomically.

The label propagation is fire-and-forget; a partial failure
doesn't unwind the successful XYZ write — the status panel reports
"Saved" as soon as the XYZ lands, and the user can re-Save if a
label call fails.

### §4.4 The atomic-write contract

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
| §3.6 sidecar refresh-hash | `tests/test_selection_blueprint.py::TestRefreshHash` |
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

2. **Bulk sidecar-write endpoint.**  §4.3 issues N+1 HTTP calls
   (one per region label + one for frozen_atoms).  A new
   `/api/selection/save-sidecar` endpoint that accepts the full
   sidecar payload `{regions, frozen_atoms, n_atoms}` in one shot
   would be faster + atomic from the client's perspective.

3. **Sidebar locked during in-flight save.**  Per
   `projects-sidebar.md` § C2, file-list mutations are gated on
   the sidebar lock.  Save should acquire the lock for the
   duration of the writeFile + refresh-hash sequence so concurrent
   file mutations can't race.

---

## §9 Change process

1. PR the contract change AND the code AND the test together.
2. Update §7 if the test ID changes.
3. NEVER ship a code change that diverges from this contract.  If
   the contract is wrong, change it explicitly.
