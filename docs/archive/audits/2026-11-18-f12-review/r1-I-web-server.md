# F12 review, round 1 — slice I (the web server)

Checkout `the checkout` at f578482db (2.77.2). Read only; no test,
server or writing verb run. One read-only Python check (`parse.detect` on a
piped SIESTA output, written to a temp folder outside the checkout, `-B`).

**What was read.** Chapters, whole: `web/web-api.md`, `web/overview.md`,
`web/handover-procedure.md`, `web/form-schema.md`, `web/notifications.md`,
`web/runtime.md`, `web/jupyter.md`, `web/presenters.md`,
`web/bench-summary.md`. Code, whole: `web/app.py`, `web/admin.py`,
`web/auth.py`, `web/tabs.py`, `blueprints/_shared.py`, `build.py`,
`files.py`, `modify.py`, `transport.py`, `watch.py`, `spectra.py`,
`results.py`, `bench.py`, `checkpoint.py`, `notify.py`, `notify_setup.py`,
`docs.py`, `selection.py`, `workspace_storage.py`, `system_load.py` (route,
GPU snapshot, snapshot), `jupyter.py` (blueprint), `molbuilder/jupyter.py`.
`rate_limit.py`: the defaults, the check, the hooks and the admin routes.
`doc_indexes.py`: its index table and `compose` only. `cli.py`: the
entry point, builders, `validate`, the CLI helpers, `xv2xyz`, the
TLS/bind guards, every `serve` verb, every `jupyter` verb, and
`watch parse`/`tail`. **Not read:** `auth_providers/*`, `zip_stream.py`,
and these `cli.py` verbs: `modify`'s body, `auth-setup`, `atom-index`,
`frameset`, `runtime-info`, `notify-token`, `checkpoint`,
`_supervise_forever`.

`notifications.md`, `runtime.md` and most of `presenters.md`,
`bench-summary.md` and `overview.md` describe browser JavaScript, which is
outside this slice. For those chapters only the server-side claims were
checked; none of them was contradicted.

Totals: **3 defects, 7 untrue sentences.**

---

## Defects

### D1 — A delete, rename, move or copy through a symlink acts on the link's TARGET (data loss)

- **Where:** `molbuilder/web/blueprints/files.py:1762`, `1823-1829`
  (delete), `1499`, `1527`, `1547` (rename), `1642`, `1691` (move, copy). The
  fence, `files.py:174` → `projects.py:288`, has
  `resolved = Path(raw).expanduser().resolve()`.
- **Rule:** `web-api.md` § 4, Files row: "rename, move, copy and delete take a
  structure's pair as one". `web-api.md` § 2.1: every path is resolved through
  `projects.contain`. Neither chapter says that the entry the person
  picked may be swapped for another one.
- **Code:** every mutation route acts on `resolved = _resolve_within_roots(raw_path)`.
  That value is the path with every symlink followed. The delete then runs
  `shutil.rmtree(resolved)` or `resolved.unlink()`. The rename computes
  `dst = resolved.parent / new_name` and moves `resolved`. The listing shows a
  link to a folder as `kind: "directory"` (`_entry_dict`, `files.py:408`:
  `p.is_dir()` follows the link), so the sidebar offers it like any folder.
- **Scenario:** `projects/P/optimization/shared-pseudos` is a person's link to
  `projects/P/optimization/calcA`. The person deletes `shared-pseudos` from
  the sidebar with recursive on. `rmtree` runs on `calcA`, so the other
  calculation's whole folder is gone, and the link stays behind, dangling.
  Renaming the link renames `calcA` inside `calcA`'s own parent folder.
  Moving or copying a linked file moves or copies the target file.
- **How sure:** read and traced. molbuilder writes no symlinks itself, so this
  road needs a link the person made.

### D2 — Sending the hand-over again under another name leaves two templates, and Task setup then refuses to save

- **Where:** `molbuilder/web/blueprints/build.py:975-985` (the only folder
  checks), `1166-1168`, `1178-1185`. The refusal comes from
  `build.py:1419-1422` → `template_file.py:68-74`.
- **Rule:** `web-api.md` § 4, `/api/task-setup/handover`: "the answer's
  `handover_here` says a hand-over already there is being replaced".
  `handover-procedure.md` § 2: "writes them into the folder, every file or
  none".
- **Code:** the door asks only `refuse_place` and `refuse_described`.
  `refuse_described` looks for `task.json`, not for a hand-over. `publish`
  adds the new files and removes none. So only `task.1st.json` is replaced:
  `<old label>.template.toml` and `<old label>.source.xyz` with its sidecar stay
  beside the new label's files. The save then reads the template with
  `template_of(dest, task.label)`, which raises `"<folder> holds 2 templates
  (...) remove the other"`, and the route answers that as a 400.
- **Scenario:** a person sends from Structure optimization into `P/opt/w1`
  with the run-name field blank, so the label is `w1`. They then type `water`
  in the run-name field and send again to the same folder. The page says
  that the hand-over there was replaced. Task setup's template card shows an
  error, and Save is refused until the person deletes `w1.template.toml` by
  hand. The old `w1.source.xyz` pair also stays behind.
- **How sure:** read and traced.

### D3 — A workspace save that fails on disk answers an HTML 500, so the page cannot say why

- **Where:** `molbuilder/web/blueprints/workspace_storage.py:132-133`
  (`d.mkdir(...)` and `persist.write_json(...)`, no try/except). `app.py:474-512`
  registers no handler for a general exception.
- **Rule:** `web-api.md` § 1: "`err(msg, code=400, **fields)` … — **every**
  failure answer is built by it". `notifications.md` § 3: the persist-error
  row says *"Couldn't save state to disk (the reason)…"*.
- **Code:** an `OSError` (a full disk, or a projects root the server cannot
  write) leaves the view uncaught, and Flask answers with its own HTML 500
  page.
- **Scenario:** the projects disk fills up during an editing session. Every
  workspace write answers an HTML page, so the page's reader (`ask`) can say
  only that the answer could not be read, and the notification shows that
  instead of the OS reason, which never reaches it.
- **How sure:** the server side is read and traced. That the reader shows
  only its generic sentence is plausible: `ui-contract.md` § 12 was not read.

---

## Untrue sentences

### U1 — `web-api.md` § 1, *Strict where a file is read, lenient where the wire is*

- **Quote:** "An unrecognised key on a request is our own client disagreeing
  with our own server … not a condition to turn into a 400 for someone
  running a calculation. Readers of wire-only blocks take the names they set
  and leave the rest."
- **Code:** `blueprints/_shared.py:118-131`: `_struct_from_envelope` checks the
  keys against `known` and raises `ValueError(f"structure: {said}.")` for any
  other key ("refused rather than dropped"). Every structure door answers that
  error as a 400 (for example `modify.py:258-259`, `build.py:408-410`).
- **Effect:** a body whose `structure` holds an unknown top-level key gets a
  400. The chapter is also inconsistent with itself: its own table, *What the
  envelope must be able to carry*, says `apply_metadata_dict` REFUSES an
  unrecognised key. **How sure:** read and traced. The chapter may have meant
  "wire-only blocks" to exclude the envelope, but the general sentence says
  otherwise.

### U2 — `web-api.md` § 1: "every failure answer is built by it" (`err`)

- **Code:** `app.py:474-512` handles only `RequestEntityTooLarge`,
  `PeriodicityRefused`, `_PickerError` and `DeckRecordUnreadable`. Any other
  exception answers Flask's HTML 500:
  - D3 above;
  - `files.py:1382` calls `float(expected_mtime)` with no guard;
  - `transport.py:576`: `read_points` can raise an `OSError`;
  - `results.py:184`: `calculation.at` can raise an `OSError`, which is not
    caught.
- **How sure:** read. Each crash path is traced only as far as the missing
  handler.

### U3 — `web-api.md` § 1, *Status codes* table (the 403, 404, 409 and 413 rows)

- **403** — "Authorization only: the caller is known and may not do this".
  The files routes answer 403 for the server's own filesystem
  `PermissionError`: `files.py:502`, `587`, `985`, `1074`, `1138`, `1299`,
  `1429`, `1831`.
- **413** — "over the 50 MiB global upload cap". `/api/files/read` answers 413
  for a file larger than `max_bytes` (default 1 MB), a size limit on the
  answer, not an upload (`files.py:565-573`).
- **404** — "one answer for every route that needs a path to be there
  (`files._existing`)". `/api/docs/read` of a document that is not there
  answers 400, "not a docs .md file" (`docs.py:76-77`, `508-509`).
- **409 / describe doors** — the hand-over row says "The folder's rules are this
  door's (… as the transport describe door's are)". The same
  `refuse_described` refusal is a 400 at the hand-over
  (`build.py:984-985`) and a 409 at `/api/transport/describe`
  (`transport.py:738`, `409 if exc.folder`). The 409 row does not list the
  second case.
- **How sure:** read.

### U4 — `web-api.md` § 4, `/api/files/zip_prepare` row

- **Quote:** "a token unused for an hour is forgotten".
- **Code:** tokens are swept only inside `zip_prepare` (`files.py:794-801`,
  `823`). `download_zip` pops the token without checking its age
  (`files.py:868-869`).
- **Effect:** if no prepare runs in between, a three-hour-old token still
  downloads. **How sure:** read.

### U5 — `web-api.md` § 5, *Admin — server reload*

- **Quote:** "`GET /api/admin/reload/available` is **always registered** and
  always answers `200 {available: bool}`".
- **Code:** on a server with sign-in configured, the auth gate
  (`auth.py:298-332`) answers a request with no session `401` before the
  route runs. The endpoint is not in `_PUBLIC_ENDPOINTS` (`auth.py:66-90`).
  Only a signed-in non-admin gets `200 {available: false}` (`app.py:733-745`).
- **How sure:** read.

### U6 — `overview.md` § 3

- **Quote:** "A molecule always crosses the wire in the one shape
  `_shared.structure_answer` builds".
- **Code:** two routes send a molecule as a bare `Structure.to_dict()`, with no
  `resolved_box`, `atom_order_said` or `labels`:
  - `/api/watch/load` (`watch.py:216-217`, `493`);
  - `/api/transport/describe_attempt` (`transport.py:294`, `336`, `444`).
- **Effect:** the pages put these back through `/api/build/load`
  (`trajectory/core.js` → `installMolecule`), so nothing visible fails; only
  the sentence is untrue. `web-api.md` § 1 lists the doors that do use the
  shape and is accurate. **How sure:** read.

### U7 — `jupyter.md` § 2

- **Quote:** "The JupyterLab page's security policy is every page's."
- **Code:** `blueprints/jupyter.py:438-440` sets only `frame-ancestors 'self'`
  when Jupyter sends no policy, and otherwise keeps Jupyter's own. The app's
  `after_request` then leaves that header alone (`setdefault`, `app.py:528`),
  so molbuilder's full CSP never reaches a passed-through page.
  `web-api.md` § 2 says it correctly ("its answers carry Jupyter's own
  policy").
- **How sure:** read. The sentence may have meant something else; as written,
  the code contradicts it.

---

## Checked and found to agree (no finding)

- The four advisory-200 sites (`notify_setup.py:492`, `511`; `build.py:240`;
  `checkpoint.py:93`) match the chapter's count of four.
- The checkpoint routes' 400, 404, 409, 500 and advisory split.
- The hand-over:
  - `dest` is required;
  - the place and described-folder refusals;
  - the pseudopotential refusal comes before anything is written;
  - the cell gate refuses before `publish`;
  - every file or none;
  - `proposed_stages`, `proposed_groups` and `proposed_varies`.
- Task-setup save:
  - `read_as` is checked: 400 when it is missing, 409 when the file is stale;
  - the 400 and 409 refusals the chapter lists;
  - *nothing to save*;
  - the checkpoint is taken before the write.
- `/api/task-setup/{value, sweepable, columns, presets, prep, commands, bench-grid, prep-plan, folder}`
  agree with their rows.
- The form schema: each key in form-schema § 1.1, the eight controls (§ 4),
  the `staging` filter, and the `common`/`state`/`own` surfaces and their
  refusals.
- The Jupyter pieces:
  - the gate's rules 1–3, with the loopback name;
  - the Origin check;
  - the 404, 403, 202, 409 and 502 answers;
  - the 16-key status payload (8 from `status()`, 7 added, plus `ok`);
  - the token is in the environment only;
  - `molbuilder.json` is read before `start` and `restart` signal (exit 2 /
    409);
  - the per-server workspace wipe.
- The notify routes return the whole state; addresses are masked; the
  listener answers a plain 404.
- The security headers and the CSP; the rate-limiter defaults (total cap 0,
  loopback allowlisted, empty 429).
- The bench summary uses `bench_owner` → `calculation.open` → `sweep_view`
  and stays read-only.
