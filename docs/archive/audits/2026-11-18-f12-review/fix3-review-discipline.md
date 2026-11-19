*Verdict: closed in 2.77.7.* The discipline review of F12 fix 3, read before its release: every finding was fixed in 2.77.7 — the user's 2026-11-09 rule on a renamed file's name restored, which the release had reversed — but the generic range sentence for one lead k-point, which stays general. Kept as the record of what the review found.

# F12 fix 3 — discipline review

Read whole: `web/blueprints/files.py`, `projects.py`, `describe.publish`, `checkpoint._run_git` and
`Repo`'s state readers, `jobset/materialize.py` (run opening), `build.py`'s hand-over route and
folder reader, `workspace_storage.py`, `paths.py`'s run helpers, the touched tests and conftest
part, and the five touched chapters plus `web/projects.md` § 2 and § 4.1 and `project-layout.md`
§ 1.6.2. Measured only in `scratch-discipline/`. pyflakes is clean on every touched file, and
`_is_basename` has no reference left.

17 findings: 3 high, 5 medium, 9 low.

---

## 1. HIGH — rename and delete find the link by its own name, then ask its depth and topic of what it points to

`molbuilder/web/blueprints/files.py:1519-1526` (rename) and `:1791-1800` (delete):

```python
resolved = _entry_within_roots(raw_path)      # the link itself
...
rel_parts = _rel_parts_inside_root(resolved)  # projects.segments -> Path(folder).resolve()
depth = len(rel_parts)
```

`projects.segments` resolves its argument (`here = Path(folder).resolve()`, `projects.py:373`).
So the new entry door hands over the link, and the next line works out the depth, the "projects
root" check and the topic check from the link's target. The two doors disagree about which thing
was picked, inside one route. The listing (`files.py:533-536`) works out `topic` from the entry's
own place (`rel + (e["name"],)`), so the page and the server now answer "is this a topic folder"
in two different ways for a link.

**Scenario (measured with the route's own helpers in scratch):** a person keeps data off the tree
and links it into their free-form folder: `ln -s /scratch/data projects/proj/user/data`.
- Delete or rename `user/data`: `segments` returns `None`, `len(None)` raises `TypeError`, and the
  page gets Flask's HTML 500. That is the failure shape I-D3 fixes elsewhere in this release.
  Before the release this was a clean 400.
- A link `proj/user/opt -> other/optimization`: delete is refused with *"refusing to delete
  canonical-topic directory 'optimization' … Pass force=true"*. The listing did not mark it as a
  topic, so the page offers a plain Delete that fails.
- A link `proj/user/top -> projects`: refused with *"refusing to delete the projects root itself;
  this would wipe every project at once"*, when only the link would go.

The new `_LINK_ROWS` (`tests/test_web_files.py:2020`) only have links whose target is a
calculation inside the tree, so none of these cases is tested.

**How sure:** measured (`_entry_within_roots` → `_existing_entry` → `_rel_parts_inside_root` on
three scratch links: `TypeError`; depth 2 with the topic refusal True; depth 0).

**Smallest fix:** one helper for "the entry's place" — its folder's segments plus its own name, as
the listing already does. The listing, rename and delete all ask it. Then add rows to `_LINK_ROWS`
for a link that leads out of the tree and a link to a topic folder.

---

## 2. HIGH — "a stage's first run is `run-0`" is still stated in `paths.opened_by_refused_send`; C-U3 changes that fact

`molbuilder/paths.py:110-115`:

```python
def opened_by_refused_send(run) -> bool:
    """… any run but a stage's first, which prep opens; …"""
    return bool(attempt_index(Path(run).name))
```

After this release, prep opens a stage's FIRST run past every number a saved state held
(`materialize.resolve_attempt`). The new test shows it: a stage with no runs left after a restore
gets `run-2`. Any first run numbered above 0 now reads as "opened by a launch whose send was
refused".

**Scenario:** describe → prep (its own save before prep, then `run-0`) → launch → save → restore
to the state before the prep → prep again. Prep opens `run-1` (or higher) as the stage's first
run, never launched. Then:
- `jobset status` and the Results tab say *"opened by a launch whose send was refused -- not
  launched; its next launch fills it anew"* (`continuation.pending_words`,
  `runstatus.py:413,657`), not *"prepared, not launched"*.
- The next-step line (`commands.launch_pending`, `runstatus.py:927`, `continuation.py:401`,
  `transport/gather.py:158,195`) names `launch task <stage>` alone. For a group of several,
  launch refuses that: *"was never launched, and its group N holds … too -- a group goes whole
  the first time"* (`submit.py:1572`).

**How sure:** traced (the callers read; `resolve_attempt`'s answer of `run-2` for a fresh stage is
the new test's own assertion).

**Smallest fix:** answer "opened by a refused send" from the stage's records, not from the run's
number: an unlaunched newest run is prep's when the stage was never launched
(`newest_launched(container) is None`), and a refused send's otherwise. Then sweep the
`project-layout.md` sentences in finding 4.

---

## 3. HIGH — `web/projects.md` § 4.1 still says rename, move and copy take any name, in the user's words, and that the fence resolves every path

`docs/web/projects.md:385-386`:

> A rename, a move or a copy keeps any name without a separator *(the user, 2026-11-09)*

The changelog has the same rule at 2.69.15: *"A rename, a move or a copy keeps any name without a
separator (the user's word)."* The release reverses it in the code (`files.py:1546`, `:1624`) and
in `web-api.md` § 4 ("one rule for a file's name … rename, move and copy"). The chapter that owns
"the rules the file operations follow" still states the old rule and attributes it to the user.
`docs/web/projects.md:381-382` also says *"The fence resolves every path it is given, a link
included"*, which is no longer true for the four mutations. The new link rule appears only in a
`web-api.md` row, not in § 4.1, its home.

**Scenario:** a reader of `projects.md` — or the next release's reviewer — is told a rename keeps
`my notes.txt`, while the server refuses it. The reversal of a dated user decision is recorded
nowhere: the triage lists R2-9 #5 under *Safety*, and that is a reviewer's line, not the user's
word.

**How sure:** traced.

**Smallest fix:** give the user the reversal of their 2026-11-09 rule (memory: *trace a rule to
the user's words*), and cite their newer word in § 4.1 when they confirm. Rewrite the § 4.1
sentence to the one name rule and add the link rule ("rename, move, copy and delete act on the
entry picked: a link as the link"). The `web-api.md` § 4 row should point to § 4.1 rather than
restate the character list a second time; § 4.1 already has it at line 380.

---

## 4. MEDIUM — `project-layout.md` restates the run number as `run-0` and `run-<n+1>`, which § 4.3 now contradicts after a restore

- `docs/execution/project-layout.md:533-534`: *"Prep opens a stage's first run. For a stage never
  launched it opens `run-0/`"*.
- `:551`: *"a launch again opens `run-<n+1>/` beside the stage's newest run"*.
- `:19` (the short version) and `:761` (the order table): `run-<n+1>/`.
- `job-system.md:2577`: *"`prep_calculation` opens `run-0`"*.

§ 4.3 now says the number is past every run a saved state held. After a restore, prep opens
`run-2` for a stage never launched, and a launch again after `run-0` opens `run-3`.

**Scenario:** the test's own case (restore, then prep) produces a first run that § 1.6.2 says
cannot exist. Finding 2 is the same restatement in code.

**How sure:** traced.

**Smallest fix:** each sentence says "opens its first run" or "opens the next run" and points to
§ 4.3 for the number.

---

## 5. MEDIUM — the topic README table has a special case for one topic, chosen by its name

`molbuilder/projects.py:475-480`:

```python
def _topic_readme(topic: str, project_name: str) -> str:
    …
    if topic == "pseudopotential":
        return _pseudopotential_readme(project_name)
    if topic in _TOPIC_READMES:
```

`_TOPIC_READMES` was the one data table for topic READMEs. Now one topic's text lives outside it
and is chosen by a branch on its name, in shared code (memory: *a branch on a name in shared code
is the defect*).

**Scenario:** a second topic whose README needs the project's name (or a rename of the
`pseudopotential` topic) has to find this branch; the table no longer tells the whole story.

**How sure:** traced.

**Smallest fix:** keep every topic in `_TOPIC_READMES`, its text a template filled with the
project's name (`{project}`, and `CONVENTIONAL_LIBRARY` / `PROJECTS_ROOT_NAME` as now), formatted
in `_topic_readme` for every topic alike. `_pseudopotential_readme` goes.

---

## 6. MEDIUM — the entry door is a word-for-word copy of the fence's door, and "is this entry a folder" is now spelled three ways

- `files.py:185-202`: `_entry_within_roots` repeats `_resolve_within_roots` (`:157-182`) line for
  line: the missing-path refusal, `_root()`, the `except OutsideRoot` with both refusal sentences.
  Only `contain` versus `contain_entry` differs.
- `files.py:205-208`: `_existing_entry` wraps `_existing` only to let a dangling link through.
  `_existing` with `os.path.lexists` would answer the same for every other caller, because every
  path they pass is already resolved and has no link left in it.
- "Is this entry a folder": `resolved.is_dir() and not resolved.is_symlink()` (rename `:1540`,
  delete `:1828`, then `link = resolved.is_symlink()` again at `:1848`); `src.is_dir()` (move
  `:1672`, copy `:1722`); and `resolved.is_dir()` inside `_is_canonical_topic_dir`.

**Scenario:** move or copy of a link to a folder (API) is refused with *"moving directories is not
supported in v1"*, against `web-api.md` § 4's new sentence *"a link is renamed, moved, copied or
deleted as the link"*. A later change to the refusal sentences lands in one copy of the door and
not the other.

**How sure:** traced.

**Smallest fix:** one `_resolve_within_roots(raw, *, entry=False)` (or the fence passed in); fold
`_existing_entry` into `_existing`; one `_is_folder(entry)` predicate (a folder and not a link)
asked by all four routes and the topic test.

---

## 7. LOW — `_validate_upload_filename` no longer says what it is

`files.py:1183-1191`:

```python
# Upload filename regex.  … no leading dot (so dotfiles can't be uploaded
# accidentally …
_UPLOAD_FILENAME_RE = …
def _validate_upload_filename(name: str) -> Optional[str]:
    """THE ONE RULE FOR A FILE'S NAME, asked by every door that names a file …
```

Rename, move, copy, `_writable` and `_next_free_name` all ask "the upload's" rule. `_writable`'s
docstring says *"its name an upload's"*.

**Scenario:** the next reader of rename sees an upload check and takes it for a slip — the
two-rules defect R2-9 #5 came from.

**How sure:** traced.

**Smallest fix:** rename to one name for the file-name rule (for example `_file_name_refusal`,
`_FILE_NAME_RE`) at its 6 product callers and in `tests/test_a_validator_validates_the_whole_value.py:51`,
and say "a file's name" in the comment.

---

## 8. LOW — docstrings and comments that describe the old behaviour

- `files.py:25-29`: *"Every endpoint that takes a `path` … runs it through `_resolve_within_roots`
  … Resolves to absolute (follows symlinks)"*. `:41-42`: *"every mutation runs through
  `_resolve_within_roots`"*. The four mutations no longer do.
- `files.py:1484`, `:1491-1494` (rename): *"`path` must resolve inside the projects tree"*;
  *"`new_name` must be a non-empty string with NO path separator, NO `..` segments, and NOT equal
  to `.` or `..`"*. That is the removed `_is_basename` rule.
- `files.py:1644-1646` (move): *"Same basename validation as rename."*
- `files.py:1757`, `:1766` (delete): *"must resolve inside the projects tree"*; *"Empty
  directories + plain files can be deleted without it"* (links too, now).
- `files.py:1738` (copy): *"source and destination resolve to the same path"*. They are compared
  as entries now (`dst == src`).
- `web/static/lib/projects/state.js:602-603`: *"`new_name` is the new basename (no path
  separators; no `..`)"*.

**Scenario:** a reader trusts the module docstring and adds a mutation through
`_resolve_within_roots`, acting on a link's target again.

**How sure:** traced.

**Smallest fix:** rewrite each to the entry door and the one name rule.

---

## 9. LOW — messages this release touches, in plain words

- `files.py:1548`, `:1626`: `f"invalid 'new_name': {refused}"`. The page shows it as is (`list.js`
  *Rename failed* toast): *"invalid 'new_name': filename 'my notes.txt' contains unsupported
  characters; …"*. `'new_name'` is a JSON key.
- `workspace_storage.py:150`: *"the workspace state was not saved: …"*. The page wraps it in its
  own sentence (`app-notifications.js:159`), so the person reads *"Couldn't save state to disk (the
  workspace state was not saved: No space left on device)."*: "not saved" twice, and "workspace
  state" is not a word the page uses.
- `tests/conftest.py:438`: *"the test suite must never run {name}: a job would be queued for
  real"*. It is now written for `squeue`, `scancel` and `scontrol` too, which queue nothing.

**How sure:** traced.

**Smallest fix:** the rule's own sentence alone (it already names the file); the reason alone
(`exc.strerror`); *"a real job would be submitted, queried or cancelled"*.

---

## 10. MEDIUM — the run-number test builds runs by hand and calls an internal step, though the road reaches the case

`tests/test_checkpoint_states.py:212-229`: in the `calc` fixture — a checkpoint folder that is not
a calculation (no description) — it makes `01_coarse/run-0/notes.txt` and `run-1/notes.txt` with
`mkdir` and `write_text`, then calls `materialize.resolve_attempt` directly.

The road reaches the same state in tier 1: describe → prep (prep's own save before it, then
`run-0`) → save → restore to the state before the prep → prep again.

**Scenario:** the hand-built state hides what the road shows. The road test would also have read
the stage's status and seen finding 2 ("opened by a launch whose send was refused"). It also keeps
a run folder no opener made (no `calcdir.json` stamp, no deck), which `testing.md` Tier 1 calls a
hand-built product.

**How sure:** traced.

**Smallest fix:** drive the road (`tests/support/road.py` / `_road.py`) and assert prep's run
number and the status words.

---

## 11. MEDIUM — the git-settings test pins only the excludes setting; its `.gitconfig` changes nothing it measures

`tests/test_checkpoint_states.py:232-252`. Measured in scratch with plain git:
- no fix: `run.log` is ignored;
- only `GIT_CONFIG_COUNT`/`KEY_0`/`VALUE_0` (the global and system settings mutated away): kept;
- only `GIT_CONFIG_GLOBAL=/dev/null`: still ignored, because git reads `~/.config/git/ignore` by
  default.

So the test passes with `GIT_CONFIG_GLOBAL` and `GIT_CONFIG_NOSYSTEM` removed. The `.gitconfig`
naming the same file is inert: the default path alone decides. The test's name and
`checkpointing.md` § 2.1 claim the person's global and system settings are not read at all
(*"a signing key the web server has no agent for would stop a save"*). Nothing pins that half.

**How sure:** measured (git 2.34.1, scratch repository, `HOME` pointed at scratch).

**Smallest fix:** drop the inert `.gitconfig`, and make the global-config half a row of its own —
for example a global `commit.gpgsign = true` with no key, where the save fails without the fix.

---

## 12. LOW–MEDIUM — the copy test's `shutil.copy2` stand-in names a need that does not hold

`tests/test_web_files.py:1931-1938` (touched here: `**kw` added). Its docstring says *"a disk that
fills between the two legs is a state no run here makes on demand"*. But the sidecar leg fails for
real when the source sidecar cannot be read.

**Scenario (measured, `_pair_to` in scratch):** `chmod 000 water.molstruct.json`, copy
`water.xyz` → `500 copy failed: sidecar copy failed (Permission denied); the structure's copy
removed, so no half pair is left.`, and the destination is empty. That is the same rollback, with
no fake. The new workspace test in this release causes its failure the same way (`chmod`).

**How sure:** measured.

**Smallest fix:** retire the stand-in. Make the sidecar unreadable, and restore its mode in a
`finally`, as the workspace test does.

---

## 13. LOW — the link and name rows: real-road input and contract words

`tests/test_web_files.py:2043-2065`:
- The links are made with `os.symlink` to absolute targets. That is how `ln -s /abs/path` makes
  one, so they are made the way a person would. No row has a relative link (`ln -s ../calcA`),
  and no row has a link out of the tree or to a topic (finding 1).
- The project is hand-built (`mkdir proj/spectrum`, `proj/optimization/calcA`) — a project with
  two topics and no README, which the road never makes. The name-rule test beside it makes its
  project through `/api/projects/create`.

`tests/test_web_files.py:1463-1497`: the refused rows expect `"unsupported characters"`, the
code's phrase. The contract (`web-api.md` § 4) says *"letters, digits, `.`, `_`, `-`, starting
with a letter or digit"*.

**How sure:** traced.

**Smallest fix:** make the project through New project; add the out-of-tree and topic link rows;
expect the contract's words ("letters, digits").

---

## 14. LOW — the scheduler command list is computed twice

`tests/conftest.py:432-434` and `tests/test_the_suite_stays_in_its_sandbox.py:66-72` both compute
`{c for k in names() for c in kind(k).commands.values()} | {"srun", "salloc"}`.

**Scenario:** a job-step command added to the stub list in conftest and not in the test (or the
reverse) is a refusal the test never checks, or a test of a command the suite never stubs.

**How sure:** traced.

**Smallest fix:** one helper in `tests/support` that both ask (memory: *the test harness has one
door*).

---

## 15. LOW — the hand-over sentence in two chapters, and a broken "them"

- `docs/web/handover-procedure.md:68-72`: the inserted sentence sits between *"answers what it
  wrote (`written`)."* and *"Two of them can only be made here"*. "them" now reads as "the new
  files" or "its template and the structure pair", not "the files written".
- *"leaves the folder one hand-over's"* is hard to read.
- `docs/web/web-api.md` (the `/api/task-setup/handover` row) restates the same sentence (*"whole:
  its template and the structure pair its record names are removed once the new files are in
  place"*) instead of pointing to Step 1, which the row already links.

**How sure:** traced.

**Smallest fix:** move the sentence after the "Two of them" paragraph, in words like "so the folder
holds the new hand-over's files alone"; the `web-api.md` row says "replaced whole (handover-procedure
§ 2, Step 1)".

---

## 16. LOW — `Repo.entries_ever_held` returns folders only

`molbuilder/checkpoint.py:1468-1483`. The docstring says *"The names of the entries directly in
`folder` … that any state ever held"*. The code keeps a name only when something lies beneath it
(`"/" in line[len(prefix):]`), so a file directly in the folder (`01_coarse/calcdir.json`) is never
returned.

**Scenario:** the next caller asks for the entries a state held in a folder of files, and gets an
empty set.

**How sure:** traced.

**Smallest fix:** name it for what it answers (`folders_ever_held`), or say so in the docstring.

---

## 17. LOW — `_the_handover_files` restates what `_fill` writes

`molbuilder/web/blueprints/build.py:1175-1196`. It rebuilds the hand-over's file set — every
template by role, plus `structure.source` and its sidecar — beside `_fill` (`:1151-1157`), which
writes the pair, the template and the record. It also adds its own "a bare name" check
(`pathlib.Path(source).name == source`, `:1190`), next to the separator check of the file-name rule
and the removed `_is_basename`.

**Scenario:** a hand-over that later writes one more file into the folder leaves it behind when it
is replaced, and the I-D2 refusal (*"holds 2 …"*) comes back for that file.

**How sure:** plausible (no such file is written today).

**Smallest fix:** one function naming a hand-over's files from its label and structure source,
which `_fill` writes and the replacement removes.

---

### Checked and left alone

- `describe.publish(..., replaces=)` extends the one transaction instead of adding a second door —
  the right place.
- `projects.contain_entry` is an entry point built on the one fence. Its own `..` check is needed: a
  raw path ending in `..` would otherwise leave the fence through the kept name.
- Checkpoint runs git in one place only (`_run_git`).
- The sandbox scheduler test is outcome-only.
- The hand-over test runs on the real road (`handed_over`, `as_the_page_saves`, `save_both`).
- The workspace test causes a real failure (`chmod`).
- The project README test reads the product's own output and resolves the spelling through
  `resolve_psml_lib`.
