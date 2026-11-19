*Verdict: closed in 2.77.7.* The correctness review of F12 fix 3, read before its release: every finding was fixed in 2.77.7 but one (a failed removal of a replaced hand-over's file answered as "could not write"), left as rare. Kept as the record of what the review found.

# F12 fix 3: correctness review (are the person's files safe?)

Reviewer: the correctness lens. Read only. The measurements ran in scratch folders under
`scratch-correctness/`: a Flask app holding only the Files blueprint on a scratch `projects/`
(`files_probe*.py`), scratch git repos driven through `checkpoint._run_git` and `Repo`
(`git_attr_probe.py`, `safedir_probe.py`), and `materialize.resolve_attempt` on scratch
calculations (`runnum_probe.py`).

**The tree moved while I was reviewing.** `files.py` changed under me. Rename, move and copy now
accept any name without a separator (`_is_basename`), and `_writable` now lets a file that is
already there keep its own name. So the concern I had first (the page's default copy name
`water copy.xyz` was refused) is gone in the current tree; I measured that it now passes. Every
finding below was re-measured against the tree as it stands now. Line numbers are the current
ones.

**9 findings.**

---

## 1. The documented redo still gives a new run a number that a saved state holds (medium)

- **Where:** `molbuilder/jobset/materialize.py:261-262`
  ```python
  if last is not None and newest_launched(stage_dir) != last:
      return last, False
  ```
- **What is wrong:** the newest run is reused whenever *the folder now* holds no launch record
  for it. The "never one used before" rule (`project-layout.md` § 4.3, as now written) is
  applied only to a fresh run. A group's first launch saves the folder before it sends
  (`submit.py:1687-1700`, `save_before`). At that moment the run prep opened, say `run-0`, is
  there and has no `run.json`. A restore to that state brings `run-0` back unlaunched, and the
  next prep reuses it.
- **Scenario:** this is the road `job-system.md` § 6.0 gives for "launched, changed in what it
  computes":
  1. Save the folder as it stands (this state holds `01_s/run-0`, launched, with its results).
  2. Restore the state saved before the group's first launch.
  3. Prepare the group again.

  Prep reuses `run-0`, and the launch runs it as `run-0`. The state from step 1 still holds a
  different, launched `run-0`. Two runs now carry one number, which is the C-U3 defect on the
  one road the docs prescribe.
- **How sure:** measured. Scratch calculation B: state S_b holds `run-0` with no `run.json`;
  S_c holds `run-0` with `run.json` and an output; restore S_b. `resolve_attempt` answers
  `(…/01_s/run-0, False)`, and `entries_ever_held` gives `{'run-0'}`. The new test
  (`test_a_run_number_a_saved_state_held_is_never_used_again`) covers only the restore to a
  state without the run.
- **Smallest fix:** reuse the newest run only when no saved state ever held its launch record:
  ask git whether any state held `<stage>/run-N/run.json`. Otherwise open a fresh number past
  every number held. `entries_ever_held` as written cannot answer this, because it keeps only
  entries that hold a `/` (folders).

## 2. Checkpoint's git no longer reads the person's `safe.directory`, so a folder they declared safe stops saving (medium, on shared trees)

- **Where:** `molbuilder/checkpoint.py:364-365`
  ```python
  env["GIT_CONFIG_NOSYSTEM"] = "1"
  env["GIT_CONFIG_GLOBAL"] = os.devnull
  ```
- **What is wrong:** git (2.35.2 and later) refuses to work in a repository owned by another
  user unless `safe.directory` allows it. It reads that setting only from system, global or
  command-line configuration, never from the repository's own. Clusters often set it
  system-wide (`/etc/gitconfig`), and people set it in `~/.gitconfig` for group-shared project
  trees. Both sources are now cut off.
- **Scenario:** a group project tree on a cluster where a labmate made the calculation folder,
  or a server running under an account other than the folder's owner. Before this change,
  every checkpoint call worked. Now every one fails with "detected dubious ownership". `save_before`
  raises, so prep, the first launch and Task setup's Save are all stopped (§ 9). Also,
  `entries_ever_held` (`check=False`) silently returns nothing.
- **How sure:** measured with git 2.55 from the molbuilder env. I could not make a folder owned
  by another user, so git's own test hook `GIT_TEST_ASSUME_DIFFERENT_OWNER=1` stood in for one.
  With a global `[safe] directory = *`, plain git exits 0. Through `_run_git` it exits 128 with
  `fatal: detected dubious ownership in repository`.
- **Smallest fix:** before isolating, read the person's `safe.directory` entries
  (`git config --system/--global --get-all safe.directory`) and carry each one through as
  another `GIT_CONFIG_KEY_n`/`VALUE_n` pair (command scope, which git honours for this key).
  Nothing else of theirs needs to come through.

## 3. Deleting or renaming a link that points out of the tree, dangles outward, or loops answers Flask's HTML 500; a link to a topic folder or the root is refused as that folder (medium-low)

- **Where:** `molbuilder/web/blueprints/files.py:1545-1546` (rename), `1814-1815` (delete),
  `140-142` (`_is_canonical_topic_dir`)
  ```python
  rel_parts = _rel_parts_inside_root(resolved)
  depth = len(rel_parts)
  ```
- **What is wrong:** the entry door keeps the link, but its depth is still read *through* it:
  `_rel_parts_inside_root` → `projects.segments` → `Path(folder).resolve()`. For a link out of
  the tree, `segments` returns `None` and `len(None)` raises `TypeError`, and no handler turns
  that into the one failure shape. For a link to `P/spectrum` or to the root, the parts are the
  target's, so the link is refused as the topic folder or as the root.
- **Scenario:** `projects/P/user/data`, the person's link to a folder on another disk, shows in
  the sidebar as a directory with Delete and Rename. Delete answers an HTML 500 page. Before
  this release it was a JSON 400 "outside every configured root". The fix meant to let this
  link be deleted alone, "whether or not what it points to still is" (`_existing_entry`). The
  same happens to a whole project that is a link to another disk (`projects/ext`, Delete
  project) and to a dangling or looping link (listed as `kind: other`). A link
  `P/user/spec_link → P/spectrum` is refused: "refusing to delete canonical-topic directory
  'spectrum' … Pass force=true", even though the listing marks it `topic: false`.
- **How sure:** measured: `outlink`, `dangle_out`, `loopa` and a project link out of the tree
  each give 500 `TypeError: object of type 'NoneType' has no len()`; `rootlink` gives 400
  "projects root itself"; `spec_link` gives 400 with the topic sentence. No data was lost in any
  case: links inside the tree were removed alone and their targets were kept (measured).
- **Smallest fix:** read the entry's place from its fenced folder plus its own name
  (`segments(resolved.parent) + (resolved.name,)`). The folder is already inside the tree, so
  the parts are never `None`, and the depth and the topic test are the link's own.

## 4. A copy of a link is now a second link, so editing "the copy" changes the original, and a relative link copied into another folder points elsewhere (medium-low)

- **Where:** `files.py:349-350` together with the write door at `files.py:1405`
  ```python
  step = ((lambda a, b: shutil.copy2(a, b, follow_symlinks=False))
          if copy else os.replace)
  ...
  resolved = _resolve_within_roots(raw_path)      # the write: the link followed
  ```
- **What is wrong:** I-D1's harm came from delete, rename and move acting on the target. A
  copy of a file link used to make an independent file, which is also `cp`'s default; it never
  touched the source. Now it makes a link. The write door and the structure save both resolve
  the leaf, and `_writable`'s "not a symlink" check only ever sees the resolved path, so they
  write through the copied link into the original. Also, the listing shows a working link as a
  plain `file`, so nothing tells the person it is a link.
- **Scenario 1:** `user/abs.txt` is the person's link to `orig.txt`. They choose Copy to… and
  name it `mine.txt`, then edit and save `mine.txt` in the preview. `orig.txt` now holds the
  edit, and the answer's `path` is `…/orig.txt`.
- **Scenario 2:** `rel.xyz → ../structure/w.xyz` is copied into a subfolder. The new link
  points at `user/structure/w.xyz`, which does not exist.
- **How sure:** measured, both scenarios. The structure-save half is traced: the load and the
  save both go through `_resolve_within_roots`.
- **Smallest fix:** let copy follow a link to a file inside the tree, as it did before: copy
  its content, with the source resolved through the fence. Keep "the entry, never its target"
  for delete, rename and move, the operations that harm the target. A link out of the tree can
  stay copied as a link, so no new refusal is needed.

## 5. The person's git *attributes* file still decides what a save stores (low-medium)

- **Where:** `checkpoint.py:366-368`
  ```python
  env["GIT_CONFIG_COUNT"] = "1"
  env["GIT_CONFIG_KEY_0"] = "core.excludesFile"
  env["GIT_CONFIG_VALUE_0"] = os.devnull
  ```
- **What is wrong:** the reason given for naming the excludes file is "git reads
  `~/.config/git/ignore` by default, with no setting naming it". The same is true of
  `~/.config/git/attributes` (`$XDG_CONFIG_HOME/git/attributes`), which is read whenever
  `core.attributesFile` is unset. A `text`/`eol` line there still converts line endings at
  save, so the stored bytes differ from the file, and a restore writes them back changed. The
  docstring says line-ending conversion no longer decides what a save keeps.
- **Scenario:** the person's `~/.config/git/attributes` holds `*.fdf text eol=lf`. A deck with
  CRLF line endings is saved, and the store holds LF. A restore writes the deck with LF.
- **How sure:** measured with git 2.55 and 2.34 and a scratch HOME. The global `.gitconfig`
  (`autocrlf=true`) and the XDG ignore (`*.log`) are no longer applied: `plain.txt` stored with
  CRLF, `small.log` tracked. But `deck.fdf` was stored as `b'SystemLabel h2\nNumberOfAtoms 2\n'`,
  and `check-attr` shows `text: set`, `eol: lf`.
- **Smallest fix:** name `core.attributesFile` empty the same way (`GIT_CONFIG_COUNT=2`,
  `KEY_1=core.attributesFile`, `VALUE_1=/dev/null`).
- **Related (plausible, not a regression):** hooks that a global `init.templateDir` copied into
  repositories made before this release still run from `.git/hooks`. The docstring's "hooks"
  holds only for repositories made from now on. Naming `core.hooksPath=/dev/null` the same way
  would close that.

## 6. Deleting a structure that is a link leaves its sidecar; rename, move and copy carry it (low)

- **Where:** `files.py:1864-1866`
  ```python
  link = resolved.is_symlink()
  its_sidecar = (None if link or resolved.is_dir()
                 else paired_sidecar(resolved))
  ```
- **What is wrong:** `web-api.md` § 4 says "rename, move, copy and delete take a structure's
  pair as one". For a link named `w.xyz`, `_pair_to` carries `w.molstruct.json` on a rename,
  move or copy, but delete skips it.
- **Scenario:** the person has `lk.xyz` (a link) with `lk.molstruct.json` beside it, and deletes
  `lk.xyz`. The labels file stays, orphaned. Later saves or moves of an `lk.xyz` there then meet
  "a file named lk.molstruct.json is already here" or a 409.
- **How sure:** measured. Delete returned `sidecar_removed: None` and the sidecar was left;
  renaming the same kind of link moved the sidecar.
- **Smallest fix:** pair the sidecar for a link as well (`paired_sidecar(resolved)` unless it
  is a real folder). `unlink` removes the sidecar entry without following it.

## 7. The hand-over replacement removes every `*.template.toml` in the folder, and a failed removal answers "could not write" when the new hand-over is already written (low)

- **Where:** `web/blueprints/build.py:1184`; `describe.py:470-474`; `build.py:1159-1165`
  ```python
  out = list(find_by_role(folder, ROLE_TEMPLATE))
  ...
  if (old.parent == out_dir and old.name not in kept
          and os.path.lexists(old)):
      old.unlink()
  ```
- **What is wrong, part (a):** the files removed include every template in the folder, not
  only the one the old hand-over wrote under its label. A template the person put there (a
  copy from another calculation, for comparison) is deleted on the next send, with no question.
  No state holds it, because a hand-over folder has no checkpoint yet. Before this release it
  stayed. Save then refused "holds 2 templates", naming it, so the person kept it.
- **What is wrong, part (b):** the removal runs after the new files are in place. If an
  `unlink` raises (for example, the named geometry is a folder: `EISDIR`), `publish` raises,
  and the route answers 500 "could not write into <folder>" even though the new hand-over is
  written and some old files remain.
- **How sure:** traced. Part (a) is plausible: it needs a template molbuilder did not write.
  Part (b) needs a folder or an unwritable entry under the old name.
- **Smallest fix:** remove the old hand-over's own template, `template_path(folder, <old
  label>)`, where the old label comes from its record (the stem of its `structure.source`, or
  its run label), not every template present. Treat a failed removal as a leftover the answer
  names, not as a failed publish.

## 8. Auto-rename on an existing name outside the strict rule now answers 500 "directory may be malfunctioning" (low, new with the `_writable` change)

- **Where:** `files.py:291` (`refused = (None if dest.is_file() else …)`), and then
  `_next_free_name` (`files.py:228-239`) from upload (`1304-1309`) and write (`1438`).
- **What is wrong:** an existing `my notes.txt` (a rename may now give that name) passes
  `_writable`. With `auto_rename`, every `my notes-<n>.txt` fails the rule, so the loop gives up
  after 998 names. Before this release the same request was a 400 naming the allowed
  characters.
- **Scenario:** a save-as or export with auto-rename onto a file the person renamed to a name
  with a space.
- **How sure:** measured. Write gave 500 "could not find an unused auto-rename candidate for
  'my notes.txt' after 998 tries". Upload gave the same, plus "; directory may be
  malfunctioning".
- **Smallest fix:** when no `-<n>` name can pass the rule because the base name itself does not,
  answer the rule's own 400 for that name instead of the 500.

## 9. `entries_ever_held` raises for a stage folder that is a link, so prep fails with a traceback where it used to work (low)

- **Where:** `checkpoint.py:1475-1476`
  ```python
  rel = Path(folder).resolve().relative_to(
      self.root.resolve()).as_posix()
  ```
- **Scenario:** `01_s` is a link to a scratch disk, and prep calls `resolve_attempt` on it.
  Result: `ValueError … is not in the subpath of …`. Before this release, `resolve_attempt` read
  only the folder and worked.
- **How sure:** measured (scratch calculation D). The layout keeps runs as real files, so this
  needs a person's own link.
- **Smallest fix:** take the stage's path relative to the calculation lexically
  (`Path(folder).relative_to(self.root)`, both as given). The run numbers are names, so nothing
  needs resolving.

---

## Checked, no finding

- **Entry door:** a trailing slash on a link acts on the link (the folder behind it was kept);
  the root, with or without a slash, is the root and refused; a path through a link out of the
  tree is refused (400); a raw path spelled through an in-tree link reaches the real entry;
  move/copy `dst == src` holds through links (both folders resolved); moving a link onto its own
  target gives 409. All measured.
- **Identity:** `GIT_AUTHOR_*` and `GIT_COMMITTER_*` were set from the environment before and
  still are, and the environment wins over any `user.*`, so no change. Checkpoint calls no git
  outside `_run_git`.
- **Run numbering:** a calculation with no `.git` makes no git call. A repository of the
  person's own with no state refs gives an empty set; `--glob` matching nothing does not fall
  back to HEAD (measured). 400 states take 35 ms (measured). A trial's run is always `run-0` by
  § 1.5a and does not go through `resolve_attempt`.
- **Older repositories:** a person with a global `autocrlf=true` will see their CRLF files as
  changed once, and a restore of an old state writes the LF bytes it stored. That is a
  consequence of the intended change, and only on that setting.
- **Scheduler stubs:** the kind is still detected by `sbatch` alone (`detect_scheduler`), which
  was already stubbed, so no test now sees a scheduler it did not before. The probe now records
  `squeue`/`scancel`/`scontrol` at the stubs, and `queue.listed`, `standing` and `cluster` answer
  `None` with or without them. Only `jobs stop`'s wording would differ. `sinfo`/`sacctmgr` are
  not stubbed, but they only read. The field tests use the full paths in a record probed
  outside the suite.
- **Hand-over replacement:** same label: the files are kept, not removed. Different engine: the
  old template goes. A record that does not read: only the templates go.
- **Workspace save:** an `OSError` now answers `err(…, 500)` with `strerror`, and the
  window-trim runs only after a successful write.
- **Pseudopotential README:** `<project>/pseudopotential` resolves to the project's folder
  through `pseudos.resolve_psml_lib`, and the bare name to the tree's library.
