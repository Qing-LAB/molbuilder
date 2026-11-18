# F12 review, round 2 — R2-9: the web server and the command line

Checkout `the checkout` at f578482db (2.77.2). Read only. No test,
server or writing verb was run. One read-only Python check compared the two
citation recognizers over the live chapters (run from `/tmp` with `-B`; it
wrote nothing).

## What was read

**Code, whole (every file of the slice):**
- `web/`: `__init__.py`, `app.py`, `tabs.py`, `admin.py`, `zip_stream.py`,
  `auth.py`, `rate_limit.py`, `doc_indexes.py`, and
  `auth_providers/{__init__,oauth,cas}.py`.
- `web/blueprints/`: `__init__`, `_shared`, `build`, `files`, `modify`,
  `transport`, `watch`, `results`, `spectra`, `bench`, `selection`,
  `checkpoint`, `workspace_storage`, `notify`, `notify_setup`,
  `system_load`, `jupyter` and `docs` (all `.py`).
- The package root: `cli.py`, `jupyter.py`, `auth_setup.py`,
  `references.py`, `issues.py` and `reload_protocol.py`.
- To check the owner side of a finding, I also read parts of these:
  `modify.py`, `checkpoint.py`, `runtime_config.py`, `monitor.py`,
  `diagnostics.py`, `builders/backends/__init__.py`, `spectra/methods.py`,
  `spectra/export.py`, `scheduler/record.py`, `job_request.py`,
  `serve_daemon.py`, `config_dir.py`, and the JavaScript callers of
  `/api/files/write` and `/api/selection/eval`.

**Tests.** 139 test files import or drive the slice. Reading all of them whole
was not possible.
- **Read whole:** `tests/support/envelope.py`, `tests/support/live_server.py`,
  `test_shared.py`, `test_blueprint_label_adoption.py`,
  `test_catalogue_refs.py` and `test_selection_blueprint.py`.
- **Read in part:**
  - `conftest.py`: the `app_at` and `web_client` fixtures;
  - `test_structure_save_endpoint.py` 1–120;
  - `test_web.py`: its outline, 560–610 and 855–875;
  - `test_web_files.py` 1118–1145;
  - `test_docs_tab.py` 355–400;
  - `test_slab_route.py` 20–40;
  - `test_task_setup_tab.py` 248–262;
  - `test_structure_envelope_protocol.py` 20–40;
  - `test_structure_authority_roundtrip.py` (its outline);
  - `test_script_emit.py` 330–345.
- **Searched with grep across all 139 files:** `create_app(` call sites, reads
  of source or document text, and hand-typed `"elements": [` envelopes.
- **Not read beyond those searches:** the e2e files, `test_rate_limit`,
  `test_auth_config`, `test_auth_setup`, `test_notify_setup_api`,
  `test_notify_listener`, `test_jupyter_contract`, `test_cli`,
  `test_checkpoint_cli`, `test_checkpoint_routes`, `test_web_files` (apart
  from the lines above), `test_task_setup_tab` (apart from the lines above)
  and `test_serve_verbs`.

**Totals: 30 code findings and 4 test findings.** Each one gives both places,
the owner, how it drifts, the smallest fix and how sure I am. They are ordered
by weight.

---

## Code findings

### 1. The checkpoint routes decide "your input or a fault" themselves, and type the refusals and the default note again

- **Owner:** `molbuilder/checkpoint.py:234-241`. `class InputError(CheckpointError)`
  is "A refusal the PERSON resolves with their own input … a surface that
  cannot tell them apart reports 'please fix your input' for a broken disk".
  `NoSuchRefError`, `NestedRepoRefusedError` and `CalculationNameError` are
  subclasses of it.
- **CLI:** asks the owner. `cli.py:3326`, `3366` and `3472` say
  `except InputError … sys.exit(2)`; everything else exits 1.
- **Web:** names subclasses route by route, and each route keeps its own list:
  - `web/blueprints/checkpoint.py:245-250` (init): `NestedRepoRefusedError` →
    advisory, `CalculationNameError` → advisory, any other `CheckpointError` →
    500.
  - `:287-297` (save): only `CalculationNameError` is advisory; every other
    `CheckpointError` is a 500.
  - `:327-334` (tag): `NoSuchRefError` → 404, `GitNotInstalledError` → 500,
    any other `CheckpointError` → advisory.
- **Refusals typed twice.**
  - `web/blueprints/checkpoint.py:277-281` gives its own sentence for a blank
    note ("a state needs a note saying what happened and what you were about
    to do"). The owner's is at `checkpoint.py:1709-1714`.
  - `web/blueprints/checkpoint.py:316-320` does the same for a blank tag name
    and note. The owner's are at `checkpoint.py:1960-1966`.
- **Default note typed three times:**
  - `checkpoint.py:1207`: `note: str = "set up"`;
  - `cli.py:3293`: `default="set up"`;
  - `web/blueprints/checkpoint.py:239`: `body.get("note") or "set up"`.
- **Drift:** add a new `InputError` refusal to `Repo.save` (say a name rule).
  The CLI exits 2 ("your input"). The web route answers HTTP 500 ("server
  fault") for the same thing. Tag goes the other way: a tag that hits a damaged
  archive comes back as a 200 advisory.
- **Fix:** each route maps `InputError` to the advisory (or 404 for
  `NoSuchRefError`, which is already an `InputError`) and every other
  `CheckpointError` to 500, as the CLI does. Drop the route's own note and name
  checks and its `"set up"`, and let `Repo` refuse and default.
- **Sure:** read on both sides.

### 2. The Modify routes check the axis and centre vocabularies a second time; the CLI types them a third

- **Owner:** `molbuilder/modify.py:208`, `_AXES = ("x", "y", "z")`.
  - `:336-338`: `if axis not in _AXES … if center not in ("first", "midpoint", "none")`;
  - `:426-428`: the same for `rotate_around_axis` (`("origin", "centroid")`).
- **Web:** `web/blueprints/modify.py`
  - `:441-449`: `if axis not in ("x", "y", "z")` and
    `if center not in ("midpoint", "first", "none")`;
  - `:489-501`: the same for rotate.

  The route already catches the domain's `ValueError` and answers 400
  (`:458`, `:511`).
- **CLI:** `cli.py`
  - `:861-863`: `type=click.Choice(["x", "y", "z"])`;
  - `:869-870`: `type=click.Choice(["first", "midpoint", "none"])`;
  - `:1043`: `ax not in ("x", "y", "z")`.
- **Drift:** add an axis (lattice `a/b/c`) or a centre mode (`centroid` for
  orient) to `modify.py`. The web route and the CLI still refuse it, each in
  its own words.
- **Fix:** export the vocabularies from `modify.py`. The routes then pass the
  value through and say the domain's `ValueError`, and the CLI choices read the
  exported tuples.
- **Sure:** read all three.

### 3. The lattice table's reference columns are typed twice in one blueprint, and its notes are a second notice shape

- **Owner:** the lattice table, `modify.load_fcc_lattice_full` (`modify.py:496-554`:
  `a_experimental`, `a_pbe`).
- **Typed twice in `web/blueprints/modify.py`:**
  - `:204`: `_NAMED = {"experimental": "a_experimental", "pbe": "a_pbe"}`
    (`/api/modify/spacings`);
  - `:1028`: `for key, label in (("a_experimental", "experimental"), ("a_pbe", "PBE")):`
    (`_lattice_notes`).
- **A second notice shape.** `_lattice_notes` (`:991-1036`) builds bare
  `{"severity": …, "message": …}` dicts. These have no `where` and no `about`,
  and they go out under `notes`. Every other finding in the slice is an
  `issues.Issue` (`issues.py:77-92`, "THE key set, in one place") or a
  `periodicity_gate._notice` (used in this same file at `:394` and `:691`).
- **Drift:**
  - add a third reference column (say `a_pbesol`) to the table. `/spacings`
    refuses it as "neither 'experimental', 'pbe', nor a number", and the
    lattice notes never compare against it;
  - a key added to `Issue.to_json` (as `stage` was) never reaches these notes.
- **Fix:**
  - the table names its reference columns and their words once, and both
    sites read that list;
  - `_lattice_notes` returns `Issue`s, sent through the one serialiser.
- **Sure:** read on both sides.

### 4. `/api/selection/eval` reads a second structure shape, with two spellings per key and Structure's defaults typed again

- **Owner:** the one deserialiser, `Structure.from_dict` through
  `_shared.struct_from_body` (`_shared.py:105-151`), and Structure's own column
  defaults (`structure.py:914-915`: `["MOL"] * n`, `["A"] * n`; the atom name
  defaults to the element).
- **Second reader:** `web/blueprints/selection.py:136-174`.
  - `:154`: `a.get("labels") or a.get("regions")`;
  - `:158`: `a.get("residueName") or a.get("residue_name") or "MOL"`;
  - `:163`: `a.get("atomName") or a.get("atom_name") or element`;
  - `:165`: `a.get("chainId") or a.get("chain_id") or "A"`;
  - `:152`: `or "X"`.
- **The only caller** (`static/lib/molview/model-jobs.js:1010-1023`) sends
  `labels`, `residueName`, `atomName` and `chainId`, so the four snake_case
  spellings and `regions` are dead alternatives. `web-api.md` documents the
  cut-down list as a deliberate exception, so the separate shape itself is not
  the defect. The second spellings and the retyped defaults are.
- **Drift:** change Structure's default chain or residue name. A filter
  `by_chain_id "A"` then matches atoms the structure itself would not call
  "A".
- **Fix:** read the one documented spelling of each key. Pass a column as
  `None` when no atom names it, so `Structure` applies its own default.
- **Sure:** read on both sides and at the caller.

### 5. Two rules for what a file may be named: a rename can make a file the write door then refuses

- **The rule the write door applies:** `web/blueprints/files.py:1163-1186`.
  `_validate_upload_filename` checks the regex `^[A-Za-z0-9][A-Za-z0-9._-]*$`,
  so a name has no spaces, no shell characters and no leading dot. `_writable`
  asks it (`:258`) for every upload, every write and every structure save.
- **The rule rename, move and copy apply:** `:276-278`, `_is_basename`, which
  only checks for no separator and not `.` or `..`. It is used by rename
  (`:1493`) and by `_new_name` for move and copy (`:1599`). A folder's rename
  goes through `validate_folder_name` (`:1519-1523`); a file's goes through
  nothing more.
- **Drift:**
  - rename `notes.txt` to `my notes.txt` and the rename succeeds. The preview's
    Save then answers 400, "contains unsupported characters".
  - rename a file to `.draft`: `_list_entries` hides dotfiles (`:439`), so the
    file vanishes from the sidebar.

  Two answers to one question: may a file have this name?
- **Fix:** rename, move and copy put a file's new name through
  `_validate_upload_filename`. Keep `_is_basename` only if a folder needs it.
- **Sure:** read.

### 6. `/api/files/write` keeps a branch only tests reach: the "generate-and-save" path, with a deck-format merge in the content-blind layer

- **Where:** `web/blueprints/files.py`.
  - `:1335-1345`: "Generate-and-save (Spectra / Build Generate buttons): no
    `expected_mtime`".
  - `:1417-1421`: if `expected_mtime is None`, the route calls
    `script_emit.merge_user_custom_from_target`.
  - `:1391-1403`: the `overwrite` and `auto_rename` branches.
- **Callers.** The text writers in `static/` are `projects/preview.js:365`
  and `inspectors/markdown.js:120`, and both send `expected_mtime`. The one
  `autoRename` caller (`projects/molview-doors.js:135`) writes a Blob, which
  goes to `/api/files/upload` (`projects/state.js:465-475`). The Generate
  buttons no longer exist: the hand-over writes the template, and `prep`
  renders decks.
- **The owner of the merge** is the deck writer: `script_emit.prepare_deck`,
  and `transport/gather.py:454`.
- **Drift:** the files layer is documented as content-blind
  (`projects.md` § 1). The merge rewrites any text whose target holds
  USER-CUSTOM markers. Only `test_web_files.py:1122ff` and
  `test_script_emit.py:339` exercise this branch.
- **Fix:** remove the merge and the generate-and-save branches (`overwrite` and
  `auto_rename` for text). `expected_mtime` becomes the write's one conflict
  rule. Retire the tests that pin the old workflow. **This is a deletion, so
  list it for the user first.**
- **Sure:** plausible. I read every JavaScript caller, but a text write whose
  `mtime` was never set would still reach the branch.

### 7. The notebook env's name, its "is it installed" check, its start/stop signals and "read the config first" each live in two or three places

- **Recipe name.** The owner is `diagnostics.py:71`,
  `DEFAULT_ENV_NAMES["jupyter"] = "molbuilder-jupyternb"`, which
  `envs.recipes.effective_name` reads. It is also typed at:
  - `jupyter.py:791`: `recipe = recipe_by_name("molbuilder-jupyternb")`;
  - `web/blueprints/jupyter.py:192`: the same line.

  The probe that follows is written twice: `effective_name` →
  `caps.env_available` is at `jupyter.py:792-801` and at
  `blueprints/jupyter.py:193-213`. Only the web copy asks conda again when the
  answer is no.
- **Signals.** `serve_daemon.py:572-573` installs `SIGUSR1` as notebook start
  and `SIGUSR2` as stop. The same pairing is typed in:
  - `cli.py:2820`, `2828`, `2842`, `2873` and `2889`;
  - `web/blueprints/jupyter.py:315` and `:332`.

  In the server child, `cli.py:2519` uses `SIGUSR1` for the stack dump
  (another process, so this is correct, but it shows the number carries two
  meanings).
- **"A start reads molbuilder.json first"** is written twice:
  - `cli.py:2803-2804`: `if sig_starts_a_notebook(sig): read_config()`;
  - `web/blueprints/jupyter.py:310-314`.
- **Drift:**
  - rename the recipe, or route the notebook through `envs.jupyter` in
    `molbuilder.json`. The two literal lookups break together, and the
    category default no longer agrees with them;
  - swap the signal pairing in `serve_daemon`. The CLI and the tab then send
    the wrong one.
- **Fix:** `jupyter.py` gains one door for the notebook env (the recipe and
  whether it is installed) and asks the category through its owner.
  `serve_daemon` names `START_NOTEBOOK` and `STOP_NOTEBOOK`. One
  `jupyter.ask_start(port)` reads the config and sends the signal, for both
  surfaces.
- **Sure:** read all sides.

### 8. The CLI's build options hand-type the backends, helix forms and terminal states the build door holds

- **CLI:** `cli.py`
  - `:311-314`: `type=click.Choice(["auto", "rdkit", "amber", "threedna"])`,
    with help text "auto-order is threedna > amber > rdkit";
  - `:307-308`: `click.Choice(["B", "A", "Z"])`;
  - `:303-305`: `click.Choice(["OH", "5P", "3P", "PP"])`.
- **Owner:** `builders/backends/__init__.py`.
  - `:97-99`: `_modules()` returns `{"threedna": …, "amber": …, "rdkit": …}`,
    the registry;
  - `:120`: `_AUTO_ORDER`.

  The terminal states are spelled in each backend's tables
  (`_rdkit.py:35-41`, `_threedna.py:223`). The web door asks
  `GENERATORS[kind].asks` and adds no list (`build.py:201-217`).
- **Drift:** register a fourth backend, or change the auto order. The CLI
  refuses the new backend and its help states the old order.
- **Fix:** the builders package exports its backend names, auto order, forms
  and terminal states. The click choices and the help text are built from
  them.
- **Sure:** read on both sides.

### 9. The "not a trajectory" refusal is composed twice, in two sentences

- **CLI:** `cli.py:82-116`, `_trajectory_parser_for`:
  `kind = getattr(getattr(parser_cls, "output", None), "__name__", "an unknown result")`
  and "…which answers a {kind} -- not a trajectory". A directory refusal of
  its own is at `:102-107`. `cmd_watch_tail` has a third, differently worded,
  at `:3697-3701`.
- **Web:** `web/blueprints/watch.py:120-153`, `_refuse_if_not_a_trajectory`,
  uses the same `getattr` chain and its own sentence.
- **Owner:** `parse.types.answers_a_trajectory`, which both ask first.
- **Drift:** change what counts as a trajectory, or how a parser names its
  answer. The two surfaces then describe the refusal differently.
- **Fix:** `parse.types` returns the refusal sentence (or `None`). Both
  surfaces print it, and the CLI keeps only its exit.
- **Sure:** read both.

### 10. `/api/build/schema/<engine>` keeps its own list of the engines that have a form

- **Where:** `web/blueprints/build.py:658`,
  `prefixes = {"siesta": "p", "pyscf": "py", "common": "common"}`. Anything
  else answers 404, "unknown form".
- **Owner:** the engine table, `engine_table.engines()` and `engine(name)`. The
  route's other doors already ask it (`_not_in_tables`, `:714-728`).
- **Drift:** an engine added to the table has a catalogue and a kind that runs
  on it, and the form route answers 404. `test_web.py:864` pins
  `("siesta", "p"), ("pyscf", "py")` a second time.
- **Fix:** the engine's record carries its form id prefix, or the route
  derives one from the engine's name, and membership is the table's.
- **Sure:** read.

### 11. "This machine" vs `this`: translated by hand in three places in `build.py`

- **Where:** `web/blueprints/build.py`
  - `:694-699`, `_machine_named`:
    `LOCAL_TARGET if target in (LOCAL_LABEL, LOCAL_TARGET) else target`;
  - `:1318-1323`: `if d.get("machine") == LOCAL_TARGET: d["machine"] = LOCAL_LABEL`;
  - `:2629-2636`, `_folder_set_to`: `LOCAL_LABEL if name == LOCAL_TARGET else name`.
- **Owner:** `scheduler/record.py:1469-1471` defines both constants, and
  `known_machines()` (`:1169`) already emits the label.
- **Drift:** the pairing has to be applied at every door that says a machine.
  A new route (or the commands route, which passes `_machine_named` in but
  never maps back out) shows `this` where every other answer says
  `(this machine)`. `jobset/commands.py:212` spells it a third way,
  `"this machine"`.
- **Fix:** `scheduler.record` exposes `label_of(name)` and `name_of(label)`,
  and the routes ask them.
- **Sure:** read.

### 12. Two normalisations of an email: `lower()` and `casefold()`

- **The session's email** is stored casefolded: `web/auth.py:425`,
  `email_cf = (email or "").casefold()`, then `:437-438`
  `session["user"] = {"email": email_cf, …}`.
- **The providers** lowercase:
  - `auth_providers/oauth.py:258`, `295`, `306`, `310`, `322`, `331`;
  - `auth_providers/cas.py:244`, `249`.
- **The admin list** lowercases (`runtime_config.get_admin_emails`, said to
  match the auth layer), and so does `web/admin.py:113`,
  `email = (user.get("email") or "").strip().lower()`.
- **Drift:** for ASCII addresses the two agree. For a local part with `ß` they
  do not: `"Straße@x".lower()` gives `"straße@x"`, while the session holds
  `"strasse@x"`. An admin named that way is never an admin.
- **Fix:** one function, `runtime_config.email_key`, used by the allow-list
  check, the session record, the admin list and the providers.
- **Sure:** read all sides. The case is rare but concrete.

### 13. The rate limiter's keys are listed twice: types in one table, defaults in another

- **Types:** `runtime_config.py:869-875`, `_RATE_LIMIT_KEYS` (nine keys).
- **Defaults:** `web/rate_limit.py:124-138`, `DEFAULTS` (the same nine).
  `RateLimiter.__init__` (`:189-198`) indexes `c["…"]` for each.
- **Drift:**
  - a key added to `_RATE_LIMIT_KEYS` but not to `DEFAULTS` raises a
    `KeyError` at server start when the file leaves it out;
  - a key added to `DEFAULTS` only is refused in `molbuilder.json` as
    unknown, so it can never be set.
- **Fix:** one row per key (type and default) in `runtime_config`'s section
  row, and `RateLimiter` reads the merged section.
- **Sure:** read.

### 14. A run report's envelope keys are listed by the sender and again by the receiver

- **Sender:** `monitor.py:1982-2006`, `_hook` record: `**ident` (`run`, `job`,
  `host`; `:2043-2058`), `sent_at`, `event`, `text`, `state`, `part`,
  `call_unit`, then `**{f.name: … for f in _fields.FIELDS}`.
- **Receiver:** `web/blueprints/notify.py:102-103`,
  `_FIELDS = ("run", "job", "host", "sent_at", "event", "text", "state", "part", "call_unit") + NAMES`.
  `_clean` (`:235-250`) drops every other key without a word.
- **Drift:** the two agree today. A key the monitor starts sending (as `part`
  and `call_unit` once were) is silently dropped by the listener until
  someone edits the tuple.
- **Fix:** the monitor (the module both ends import) names its envelope keys
  once, and `notify._FIELDS` is that tuple plus `report_fields.NAMES`.
- **Sure:** read on both sides.

### 15. The listener's address is composed in two places

- **Where:**
  - `cli.py:2214`: `"url": f"{base}/api/{seg}"`;
  - `web/blueprints/notify_setup.py:641`: `"url": f"{host}/api/{seg}"`.

  The route is `web/blueprints/notify.py:281`, `@bp.route("/api/<route>")`.
- **Drift:** move the listener under a prefix. Both surfaces keep printing an
  address that 404s, and the notifier swallows 404s by design.
- **Fix:** `issue_notify_key` (already the one door both use) returns the
  path, or `notify` exports `listener_path(seg)`.
- **Sure:** read.

### 16. Two recognizers of how a text cites a bibliography key

- **The References index:** `web/doc_indexes.py:55`,
  `_CITED_SHAPE = re.compile(r"\[`?([A-Z][A-Za-z]*\d{3,4}[a-z]?)`?\](?!\()")`,
  and `:58-64`, `_cites(key)`, which takes a single key in brackets or
  `references.bib: Key`.
- **The methods text:** `spectra/methods.py:52-58`, `_CITE_RE`, which takes
  comma lists (`[Head1997, LiJensen2002]`) and `[Key §section]`.
- **Owner:** `references.py`, the bibliography door, which defines no
  citation syntax.
- **Measured:** the chapters do cite in lists. `engines/vibration.md`,
  `science/normal-modes.md` and others use `[Head1997, LiJensen2002, Besley2008]`.
  Today every listed key is also cited alone somewhere, so the index's
  "cited by" column is right by luck. In the other direction, the index finds
  backticked and `references.bib:` citations that `_CITE_RE` misses (14 keys
  differ).
- **Drift:** a key cited only inside a list is reported "neither a chapter nor
  the package cites it". An unknown key inside a list never reaches "Cited,
  not in the bibliography".
- **Fix:** `references.py` owns one `cited_keys(text)`, and both read it.
- **Sure:** both regexes read, and measured over the live chapters.

### 17. The watch answer is built three times

- **Where:** `web/blueprints/watch.py`
  - `:473-495`: the path load's dict;
  - `:567-589`: the upload load's dict;
  - `:644-665`: `_changed`, the poll's dict.

  Each repeats `path`, `mtime`, `format` (via `_engine_of`), `label`, `data`,
  `targets` and `uploaded`.
- **Drift:** `targets` was added to all three by hand. A field added to the
  poll alone means a load and a poll of the same file answer different shapes
  (the page reads both through `core.js`).
- **Fix:** one `_answer(state, *, search_dir)` that the three extend with
  their own fields (`resolved_from`, `run`, `structure`, `notices`).
- **Sure:** read.

### 18. A provider kind's facts live in three tables or branches

- **Where:**
  - `runtime_config.py:427-442`: `_PROVIDER_KINDS` (validator and keys) and
    `OAUTH_KINDS`;
  - `web/auth_providers/oauth.py:136-179`: `_client_kwargs_for_kind`, an
    `if kind == "google" … "microsoft" … "orcid" … "github"` chain;
  - `oauth.py:382-387`: `_IDENTITY_HANDLERS`;
  - `web/auth_providers/__init__.py:43`: `if kind == "cas"`.
- **Drift:** the package docstring (`auth_providers/__init__.py:25-28`) says a
  new kind is "a sibling module … and the kind's row to `_PROVIDER_KINDS`". For
  an OAuth kind, the if-chain and the handler dict must also be edited, or the
  first sign-in raises "unsupported kind".
- **Fix:** one record per OAuth kind in `oauth.py` (registration kwargs and
  identity handler), keyed as `_PROVIDER_KINDS` is, with each kind's backend
  module named in that row.
- **Sure:** read.

### 19. The auth wizard checks rules the validator owns, and one of them disagrees

- **Wizard:** `auth_setup.py`
  - `:232-235`: `client_id` required;
  - `:237-246`: an empty `allowed_users` is refused, "An empty list = nobody
    can sign in".
- **Owner:** `runtime_config.py`
  - `:471-474`: "An empty list is a degenerate-but-valid case";
  - `:315-317`: `_require_str(entry, "client_id", idx)`.

  The wizard writes through `write_config_scope`, which runs that validator.
  `build_auth_block`'s own comment (`auth_setup.py:273-276`) says "NO
  EMPTINESS CHECK HERE … the rule has one home".
- **Drift:** the two answers to "may `allowed_users` be empty?" already
  disagree, and a change to the validator's `client_id` rule is not seen by
  the wizard.
- **Fix:** drop the wizard's two checks. The validator's sentence is the
  refusal, as `build_auth_block` already intends.
- **Sure:** read.

### 20. Each allocation dict on the plan card is re-keyed by hand

- **Where:** `web/blueprints/build.py`
  - `:1988`: `return {"domain": res.domain, "time": res.time, "mem": res.mem}`;
  - `:2074-2077`: `"allocation": {"domain": job_req.allocation.domain, "time": …, "mem": …}`.
- **Owner:** `job_request.Allocation` and `JobRequest.to_dict` (`job_request.py:296-308`,
  which writes `allocation` with its own key set, `gpu_binding` included).
- **Drift:** a field of the allocation (`gpu_binding` today, or a new one)
  reaches the description's wire form but not the plan card.
- **Fix:** `Allocation` (and `Resources`) offers one wire form, and both rows
  use it.
- **Sure:** read. This is low weight.

### 21. `bibliography_keys` is recomputed in only one of three places that change the methods text

- **Where:** `web/blueprints/spectra.py`
  - `:196-200` (`with_ir_route`) and `:279-281` (`with_infrared_view`) rewrite
    `methods_text`;
  - only the Raman branch (`:298-302`) re-derives
    `payload["bibliography_keys"] = extract_citation_keys(...)`.
- **Owner:** `spectra.methods.extract_citation_keys`, which derives the keys
  from the text.
- **Drift:** the IR sentences cite nothing today. Add a citation to
  `infrared_view_sentence` and the page shows a bibliography missing it.
- **Fix:** recompute `bibliography_keys` once, after the last rewrite.
- **Sure:** read. This is low weight.

### 22. The server's nvidia-smi query is a second query, and its comment claims it is the monitor's

- **Server:** `web/blueprints/system_load.py:70-76`, `_SMI_QUERY`
  (`name,utilization.gpu,…,clocks.mem`), described as "the same invocation
  shape the compute-node monitor has always used". Its row split is at
  `:90-117`.
- **Monitor:** `monitor.py:1080`, `--query-gpu=index,utilization.gpu,utilization.memory,…`,
  with its own split at `:1171-1177`.
- **Owner:** `runtime_info` (§ 3.2's row, "what nvidia-smi says"), which both
  use only for the field reader.
- **Drift:** the claim in the comment is already untrue. Column positions
  (`f[3]`, `f[4]`) are written twice, against two different field lists.
- **Fix:** a `runtime_info` door that runs a named field list and returns rows
  keyed by field (stdlib only, so the monitor can still carry it), or at least
  a corrected comment.
- **Sure:** read. The monitor's stdlib-only constraint may justify a copy.

### 23. The loopback rule and the `Host:` parse are written several ways

- **Loopback:**
  - `cli.py:1345-1355`: `{"127.0.0.1", "localhost", "::1"}` or the prefix
    `127.`;
  - `web/blueprints/jupyter.py:125-151`: `ipaddress.ip_address(...).is_loopback`
    plus `localhost`;
  - `rate_limit.py:132`: the allow-list `["127.0.0.1", "::1"]`.
- **The `Host:` header:**
  - `app.py:212`: `int((request.host or "").rsplit(":", 1)[1])`;
  - `files.py:844-846`: `request.host.rsplit(":", 1)[0] if not request.host.endswith("]") else request.host`;
  - `blueprints/jupyter.py:145`: `urlsplit(f"//{request.host}").hostname`.
- **Drift:**
  - `--host 0:0:0:0:0:0:0:1` is non-loopback to the bind guard and loopback to
    the notebook gate;
  - an IPv6 `Host:` with a port loses its port in `serve_port` and keeps the
    brackets in the rsync line.
- **Fix:** one `is_loopback(name)` and one `host_and_port(request)`, asked by
  all of them.
- **Sure:** read. This is low weight.

### 24. The OAuth callback path is typed in the CLI

- **Where:**
  - `cli.py:1454`: `f"{scheme}://{host}:{port}/oauth-callback/{p['id']}"`;
  - `cli.py:1823`: `"http://localhost:8888/oauth-callback/google"`.
- **Owner:** the route, `web/auth.py:260`,
  `@app.route("/oauth-callback/<provider_id>")`. The server also logs it
  itself through `url_for` (`auth.py:357-392`).
- **Drift:** rename the route and the CLI's registration hint names a URL the
  provider will redirect to, which then 404s.
- **Fix:** `auth.py` exports `OAUTH_CALLBACK = "/oauth-callback/{id}"`, and the
  route and the CLI both use it.
- **Sure:** read. This is low weight.

### 25. The config-directory rule is restated as shell in `notify-token`

- **Where:** `cli.py:2270-2271`,
  `cfg="${MOLBUILDER_CONFIG_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/molbuilder}"`.
- **Owner:** `config_dir.config_dir()` (`config_dir.py:72-110`).
  `relative_home` is already asked for the file's own path two lines above.
- **Drift:** a change to the resolution order (a new override, or the
  override's "exactly as given" rule) leaves the printed recipe putting the
  key where the monitor does not read it.
- **Fix:** `config_dir` exports the shell spelling beside the rule.
- **Sure:** read. This is low weight.

### 26. A dead re-export

- **Where:** `web/blueprints/__init__.py:26-28`,
  `from . import watch as watch  # re-export for "from .blueprints import watch"`.
  No module or test imports `watch` that way: a repository-wide grep finds
  only submodule imports such as `from molbuilder.web.blueprints import notify`.
- **Fix:** delete it. This is a deletion, so list it.
- **Sure:** grep.

### 27. Two code blocks in `files.py` parse and bound `max_bytes` the same way

- **Where:** `web/blueprints/files.py:550-558` (`read`) and `:948-957`
  (`read_range`), the same parse, bound and two sentences. The upload's
  boolean words (`:1251-1254`) are parsed twice too, and differ from the
  transport route's (`transport.py:162-163`: no `"on"`).
- **Fix:** one `_bounded_int(arg, default)` and one `_said_yes(value)`.
- **Sure:** read. This is low weight.

### 28. The structure-folder root is composed four ways

- **Where:**
  - `web/blueprints/files.py:90-100`, `_root()` (resolved, with a 400 when it
    cannot be read);
  - `transport.py:75` and `:696`: `Path(projects_root()).resolve()`;
  - `transport.py:862`, `878`, `884`: `projects_root()` unresolved;
  - `spectra.py:412`, `workspace_storage.py:89`: `projects_root()`.
- **Drift:** a projects root reached through a symlink is compared resolved in
  one route and unresolved in the next. A route that skips `_root()` answers an
  unreadable tree with a 500 instead of the files layer's sentence.
- **Fix:** the web routes ask `files._root()`.
- **Sure:** read. This is low weight; `projects.at` resolves internally.

### 29. The hand-over sender's tab label is found by searching the tab list

- **Where:** `web/blueprints/build.py:943-944`,
  `_sender = next(t["label"] for t in TABS if t["key"] == _sender)`.
- **Owner:** `web/tabs.py`'s `TABS`. This is the right source, but the only
  lookup by key is this inline scan.
- **Fix:** `tabs.label_of(key)`. This is minor; I note it only because a second
  inline scan would make two.
- **Sure:** read.

### 30. Two refusal sentences for "a directory" in `cli.py`

- **Where:** `cli.py:102-107` (`_trajectory_parser_for`) and `:3697-3701`
  (`cmd_watch_tail`). The second exists because the loop must refuse before
  polling. It could call the first, with the same exit.
- **Fix:** `watch tail` calls `_trajectory_parser_for` once before the loop.
  This also covers finding 9's rule.
- **Sure:** read.

---

## Test findings

### T1. 56 test files build their own Flask client beside the one fixture

- **The fixture:** `tests/conftest.py:847-866`, `web_client`. Its docstring:
  "Tests that need an auth-enabled or otherwise non-default app build their
  own fixture".
- **Measured:** 31 files use `web_client`. 56 files call `create_app(` 90
  times, almost all as `create_app(config={}).test_client()` with no
  non-default section.
  - `test_periodicity_gate.py` defines the same class fixture five times
    (`:520`, `:938`, `:1087`, `:1555`, `:1658`) and calls it inline three
    more times.
  - `test_transport_on_a_real_junction_e2e.py` calls it 9 times.
- **Two spellings of one client:** `config={}` (limiter on, but the test
  client is loopback and allow-listed) and
  `config={"rate_limit": {"enabled": False}}` (`test_the_path_fence.py:63`,
  `test_selection_blueprint.py:22`, `test_notify_setup_api.py:34`). The limiter
  difference is moot, but a reader has to work that out.
- **Fix:** files whose app needs no section of its own take `web_client`. The
  rest keep a fixture that states the section they change.
- **Sure:** grep-counted. Each call site was not opened.

### T2. API tests hand-type the structure envelope the support door builds

- **The door:** `tests/support/envelope.py:1-61`, "THE way a test builds a
  request body carrying a structure … `structure(..., regions=, frozen=, cell=)`".
- **Hand-typed copies:**
  - `test_structure_save_endpoint.py:41-55`, `_browser_blob`, types
    `"metadata": {"regions": …, "frozen_atoms": […], "cell": …, "engine_offset": …}`;
  - `test_structure_envelope_protocol.py:23-40`, `_envelope`, re-implements
    the door's `frozen=` → reserved-label convenience with the literal
    `"frozen_atoms"`, where `support.envelope` uses `FROZEN_LABEL`;
  - `test_shared.py:46-50` and `:73-77`;
  - `test_blueprint_label_adoption.py:20-24`;
  - `test_structure_analyze_endpoint.py` (3 envelopes, by grep);
  - `test_selection_blueprint.py:26-32`. This one sends an `"index"` key the
    reader never reads, and none of the documented
    `residueName`/`atomName`/`chainId` keys. Finding 4's dead spellings are
    therefore untested too.
- **Drift:** a renamed metadata key or reserved label passes these tests while
  the real writer has moved on. The door exists so that "a field the envelope
  grows lands here".
- **Fix:** build each structure through `support.envelope.structure(...)`.
  Where a test means to pin the browser's literal shape, say so in its first
  line.
- **Sure:** read for the files named; grep for the counts.

### T3. Two tests pin `Structure.from_dict` applying labels, beside the structure layer's own test

- **Tests:**
  - `test_blueprint_label_adoption.py:15-28`: regions and the frozen label
    reach the structure through `struct_from_body`;
  - `test_shared.py:39-64`: annotations ride through `struct_from_body`.
- **Owner's test:** `test_structure_authority_roundtrip.py:40-48` and `:89`
  (`test_to_dict_from_dict_round_trip_preserves_all_metadata`), which covers
  the same regions, reserved label and annotations through `from_dict`.
  `struct_from_body` adds only the key check (`_shared.py:116-149`).
- **Drift:** none. Two tests fail for one cause and the count grows.
- **Fix:** retire `test_blueprint_label_adoption.py`, and keep in
  `test_shared.py` only what `struct_from_body` itself adds (its refusal of an
  unknown key). **This is a retirement, so list it for the user first.**
- **Sure:** plausible. I read the three; I did not check each assertion
  line for line.

### T4. A test pins the form-prefix table finding 10 names

- **Where:** `test_web.py:863-871`,
  `@pytest.mark.parametrize("engine,prefix", [("siesta", "p"), ("pyscf", "py")])`.
  It is a third copy of `build.py:658`'s `prefixes`, kept by hand.
- **Fix:** it goes with finding 10. The test asks the engine table for the
  engines and their prefixes.
- **Sure:** read.

---

## Not findings

These were checked and left out on purpose:

- `_shared.structure_answer` and `ok_structure_response` are the one structure
  answer.
- The fence: `files._resolve_within_roots`, `checkpoint._resolve_path`,
  `projects.at` in transport, and `docs`' own root. These are
  `web-api.md` § 2.1's named entry points.
- `issues_to_json` and `card_of_finding` go through `template.card_of`.
- `catalogue_to_form_schema` and `item_record` are each the one record per
  setting.
- `docs.py`'s three uses of `projects.contain`.
- `doc_indexes` reading the layer rule from `architecture.md`, which is
  documented ("read, never judged").
- The archive's exclusions (`checkpoint.NEVER_STORED`,
  `warmfiles.never_carried`, `workspace_storage.SCRATCH_DIR`) are each their
  owner's.
- `reload_protocol` is a leaf with one copy of each constant.
- `references.citation_for` and `known_keys`.
- `auth.sign_in_is_configured` is the one predicate.
- `admin.is_admin_request` is shared by the reload route and the rate
  limiter.
- `spectra.py` asks `infrared_view_of` and `raman_view_of`, the same doors as
  `postprocess` and the export.
- Round 1's I-D1…D3 and I-U1…U7 are not repeated here. None of them is caused
  by a duplicate listed above. U2 (HTML 500s) overlaps finding 1 only in that
  the checkpoint routes catch broadly.
