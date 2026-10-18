# Renaming molbuilder to Benchmate — the plan

**Role:** plan (the rename's working plan; one row in [`plans/plan.md`](?doc=plans/plan.md) § 2 points here — see § 0)
**Domain:** all
**Started:** 2026-10-18
**Status:** all decisions D1–D7 made; nothing renamed yet; next is Phase 1. Baseline: commit `83e72b2`, molbuilder 2.9.4.

> *(user, 2026-10-18: "I like H5, make a folder to provide a detailed suggested step/plan to make the change of name with this icon design")*

**In this folder**

| File | What it is |
|---|---|
| `README.md` | This plan |
| [`inventory.md`](?doc=plans/rename-benchmate/inventory.md) | Measured list of every place the current name appears, with the commands that counted them |
| `icon/benchmate.svg` | The chosen icon (H5), the master file |
| `icon/benchmate-preview.png` | The icon at 300, 96, 48, 32 and 16 px, on light and dark backgrounds |

---

## 0. How this folder relates to rule R3

[`docs/README.md`](?doc=README.md) rule R3 says there is exactly one plan, `plans/plan.md`. The user asked for this rename to have its own folder. To keep R3's purpose (one list of open work), the two are connected like this:

- `plans/plan.md` § 2 carries **one row, N1**, that names this folder and its current phase.
- This folder holds the detail: decisions, steps, checklists and the inventory.
- When the rename is finished, this folder moves to `docs/archive/` as a record, and row N1 is closed. Nothing in this folder is a contract.

---

## 1. The decision

### 1.1 Name

| Where it appears | Spelling |
|---|---|
| Title, README, papers, talks | **Benchmate** |
| Python import and package directory | `benchmate` |
| Command people type | `benchmate` (e.g. `benchmate serve start`, `benchmate jobset prep task`) |
| Distribution name in `pyproject.toml` | `benchmate` |
| Conda environments | `benchmate`, `benchmate-siesta`, `benchmate-siesta-gpu`, `benchmate-pySCF`, `benchmate-MDtools`, `benchmate-jupyternb` |
| Per-user directories | `~/.config/benchmate`, `$XDG_STATE_HOME/benchmate`, `$XDG_RUNTIME_DIR/benchmate` |
| Machine config file | `benchmate.json` |
| Environment variables | `BENCHMATE_*` |
| GitHub repository | `Qing-LAB/benchmate` |

**Tagline:** *Benchmate — your mate at the modeling bench.*
Longer form, for README and papers: *a lab mate for molecular and materials modeling: build structures, set up calculations, run them anywhere, and read the results.*

**Why this name:** "bench" is the modeling bench, where models are built and tested *in silico*. "mate" is the colleague beside you, who helps but leaves the decisions to the scientist. That matches the stance in [`design.md`](?doc=design.md): *an assistant to the scientist, not a nanny*. The name does not mention any engine, so adding CP2K, VASP, ORCA or Quantum ESPRESSO later never makes it wrong.

**Name checks, 2026-10-17:** `benchmate` is unused on PyPI. Not yet checked: GitHub organisation and repository names, conda-forge, trademarks (Phase 0, step 0.3).

### 1.2 Icon — H5

![Benchmate icon](icon/benchmate-preview.png)

**What it shows.** Each element stands for part of the work:

| Element | Colour | Meaning |
|---|---|---|
| Chip outline with pins | slate `#4a7bb7` | *In silico*: the work happens in the computer |
| Two close-packed atom piles, narrowing to single-atom tips | teal `#1fb5a8` | Metal electrodes, drawn as atoms (atomic-level materials) |
| 3-D molecule bridging the tips (atom sizes, a thicker bond toward the viewer, a faded atom behind) | coral `#f2665a` | The molecular structure being modeled |
| Wave packet crossing the top edge, a large packet then a smaller transmitted one, both ends fading | amber `#f0a830` | Quantum tunneling / transport |
| Three electrons crossing the bottom edge | amber `#f0a830` | The particle side: wave–particle duality, and current |

**Style rules**, kept for any future edits or variants:

- Line drawing only, no filled shapes. Stroke widths are 2–4 px at the 128 px design size, with round caps and joins.
- The same four colours on light and dark backgrounds. The icon has no background tile, so it sits on whatever page colour is behind it.
- The wave and the electrons may break the chip outline. Those breaks are drawn with SVG masks, never by painting over the outline with a background colour.
- The wave may touch the molecule's top atom once, briefly. The centre stays uncluttered.

**Sizes:**

- At 32 px and above, the full icon is used.
- At 16 px (the browser tab), the detail merges together. Phase 1 step 1.2 makes a simplified 16 px favicon: the chip, the wave across the top and a coral centre dot.

---

## 2. Principle: change identity, keep data

**Rename every name a person sees or types. Keep every name stored in a file that already exists.**

Project folders and run folders on workstations and on clusters already contain `.molwatch.log` files, `.molstruct.json` sidecars and JSON records stamped `molbuilder/task@1`. Readers must go on accepting those forever, so renaming them gains nothing and risks losing data. The full list is in [`inventory.md`](?doc=plans/rename-benchmate/inventory.md) § 3; § 3 below states the decision for each.

The rename also needs one more property: **a machine that already runs molbuilder must keep working at every step.** Old commands, old config directories, old environment variables and old conda environments keep working through a compatibility period (Phases 3–5), with a clear notice. Phase 6 removes the fallbacks.

---

## 2a. The word "molbuilder" means three different things — never one search-and-replace

*(user, 2026-10-18: "there is a tab that has 'molbuilder' in its design. don't mix that with the name of the whole project. this needs some intelligent decision/differentiation")*

The same word is used for three different things today. The rename treats each one differently, so **no step in this plan is a blanket search-and-replace of "molbuilder"**. Every change matches one meaning only.

| Meaning | Where it appears | What happens |
|---|---|---|
| **1. The whole project** (the product) | Package and import name, the command, page titles ("molbuilder — Results"), the header, docs prose, conda environments, config directory, `MOLBUILDER_*` variables | **Renamed to Benchmate** (Phases 1–5) |
| **2. The structure-building tab** (one feature inside the product) | The tab labelled **Molbuilder**, its key `molbuilder` and its address `/molbuilder` (`web/tabs.py:41-43`, `web/app.py:542`, `templates/modify.html:23`), links to it (`lib/inspectors/structure.js:69,85`, `templates/transport_calculation.html:88`), and 50 "Molbuilder tab" mentions in the docs | **Kept by default** (decision D7) |
| **3. Names stored in files or sent over the wire** | File formats, record identifiers, bundled tools (§ 3), plus the notification wire format: the channel kind `"molbuilder"` (`monitor.py:1326,1347`, `this-machine/page.js:140,334`, `this_machine.html:80`), the default channel name (`cli.py:1810`) and the HTTP headers `X-Molbuilder-Timestamp` / `X-Molbuilder-Signature` (`monitor.py:1588-1589`) | **Kept**: existing files, configured channels and listeners depend on them |

**How to tell them apart when editing:**

- If the word names *what the person is using* ("molbuilder serves…", "molbuilder — Results", `python -m molbuilder`), it is meaning 1.
- If it names *the tab where structures are built and edited* ("the Molbuilder tab", `/molbuilder`, `active_tab = "molbuilder"`), it is meaning 2.
- If changing it would make an existing file, channel or listener unreadable, it is meaning 3.

**Why keep the tab name "Molbuilder" (recommended for D7).** After the rename, the word goes back to what it literally says: the *molecule builder*. Benchmate is the whole bench; Molbuilder is the tool on it where you build and edit structures. The other tabs are named for what they do too ("Structure optimization", "Transport calculation"). Keeping it also means:

- the 50 "Molbuilder tab" references in the docs stay correct;
- the address `/molbuilder` and people's bookmarks keep working;
- users who learned the tab by that name find it where it was.

**The writing rule that keeps them apart:**

- The product is always **Benchmate**.
- The tab is always written **"the Molbuilder tab"**, capitalised and followed by "tab", never "molbuilder" alone.
- A page title reads `Benchmate — Molbuilder` (product — tab), the same pattern as `Benchmate — Results`.

**If you prefer a fully separate name (D7, option B):** rename the tab's *label* to **"Structure builder"**, which matches the other descriptive tab names. Keep its key `molbuilder` and address `/molbuilder` as they are, so links and bookmarks still work. Only the 50 doc mentions and one label change.

---

## 3. What is NOT renamed

| Kept as it is | Why |
|---|---|
| `.molwatch.log` and its `v1` format | Every existing run folder has these; the trajectory viewer reads them |
| `.molstruct.json` sidecars | Every saved structure has one |
| Schema identifiers `molbuilder/<kind>@<n>` (16 kinds) | Written into existing records. A **new** schema version may use `benchmate/<kind>@<n+1>`, while readers keep accepting the old prefix |
| `molbuilder_version` field in records | Existing records carry it. New records write **both** `molbuilder_version` and `benchmate_version` for one major version, then only the new one |
| `molbuilder-vibration/v3` record format name | Same rule as the schema identifiers |
| `mb_pyscf.pyz`, `mb_monitor.pyz`, `mb_vibration.pyz` | Existing run folders carry them. "mb" also reads as "Benchmate" |
| `MB_LAUNCHED_BY`, `MB_NP` and the other `MB_*` wrapper variables | Existing wrappers and site scripts set them |
| Browser storage keys `molbuilder.current_dir` and four others | Renaming them makes every browser forget its remembered folder and sidebar layout once. Optional later (§ 6) |
| The **Molbuilder tab**: its key `molbuilder`, its address `/molbuilder`, and (by default) its label | It names the structure builder inside Benchmate, not the project (§ 2a, D7) |
| Notification channel kind `"molbuilder"` and the headers `X-Molbuilder-Timestamp` / `X-Molbuilder-Signature` | Configured channels and existing listeners send and check these. A new kind `"benchmate"` may be *added* and accepted alongside, but the old one is never removed |

---

## 4. Phases

Each phase is one or more commits, ends at a point where everything works, and has a done-condition you can check. Phases 1 and 2 are low risk and can ship on their own; the name the public sees changes there. Phases 3–5 change machine-level names and need the compatibility layer.

The version moves to **3.0.0** with Phase 3. [`process/conventions.md`](?doc=process/conventions.md) § 2 says the first number moves on "a major framework change — the user declares it"; renaming the package and the command is one. The user confirms this at step 0.4.

### Phase 0 — Prepare (no code changes)

| Step | Action | Done when |
|---|---|---|
| 0.1 | Re-run the commands in [`inventory.md`](?doc=plans/rename-benchmate/inventory.md) at the commit where work starts, and update its counts | The inventory is dated at the starting commit |
| 0.2 | Add row **N1** to [`plans/plan.md`](?doc=plans/plan.md) § 2, pointing to this folder (done with this plan) | The row exists |
| 0.3 | Check that the name is free: GitHub (`Qing-LAB/benchmate`), PyPI (already free), conda-forge, a quick web and trademark search | Results noted in this file under § 8 |
| 0.4 | The user confirms the open decisions in § 7 (version 3.0.0, repository rename timing, storage keys) | Each answered in § 7 |
| 0.5 | Tell other users of the lab machines and cluster when Phases 3–5 will land, and what they need to do (normally nothing; see Phase 4) | Message sent |

### Phase 1 — Visual identity (risk: very low)

Only what a person sees in the browser changes. No import, path or command changes.

| Step | Action | Files |
|---|---|---|
| 1.1 | Copy `icon/benchmate.svg` into the app as `molbuilder/web/static/img/benchmate.svg` | new file |
| 1.2 | Make the 16 px favicon: a simplified SVG (chip, top wave, coral centre dot), plus `favicon.ico` (16 + 32 px) and `apple-touch-icon.png` (180 px), rendered from the SVGs | `molbuilder/web/static/img/` |
| 1.3 | Add `<link rel="icon" …>` once, in the shared head that every page includes (or in each of the 10 templates if there is no shared head) | `molbuilder/web/templates/*.html` |
| 1.4 | Change the 10 page titles from `molbuilder &mdash; <page>` to `Benchmate &mdash; <page>`. `modify.html`'s title becomes `Benchmate &mdash; Molbuilder` (product — tab, § 2a). The tab label itself is not touched | 10 templates |
| 1.5 | Change the app header from `<h1>molbuilder</h1>` to the icon plus "Benchmate", with the tagline as its `title` text | `_app_header.html:30` |
| 1.6 | Find tests that pin the old titles and header text, and update them to the new text | `grep -rn "molbuilder &mdash;\|<h1>molbuilder" tests/` |
| 1.7 | Re-capture the README screenshots, which show the old header | `scripts/capture-readme-screenshots.py`, `docs/img/` |

**Done when:** every page shows the new title, header and favicon on light and dark themes, and the tier-1 test run (`tools/testrun.py run none2e`) passes.

### Phase 2 — Words a person reads (risk: low)

Text changes only. Commands still say `molbuilder` here, because the command itself is renamed in Phase 3.

| Step | Action |
|---|---|
| 2.1 | Root `README.md`: new title, tagline and icon; a short "formerly molbuilder" note near the top; keep every command as it is until Phase 3 |
| 2.2 | `docs/README.md` title and the docs-tab labels in `docs/toc.json` (e.g. "How molbuilder works" → "How Benchmate works") |
| 2.3 | `pyproject.toml` `description`: replace with the tagline |
| 2.4 | Prose in the spine docs (`design.md`, `architecture.md`, `workflow.md`, `ops/getting-started.md`) where the word names the *product*, not a file, path or command |
| 2.5 | Text written into generated decks and scripts: "generated by molbuilder" → "generated by Benchmate", in `pyscf/input.py:256,1578`, `pyscf/vibration_emitters.py:125`, `runwrap.py:645,2394,3451,3456` and `script_emit.py:1515`. **First confirm no reader detects a file by this text** (§ 5, check C2) |
| 2.6 | Add a changelog line to `process/changelog.md` (a last-number move, e.g. 2.9.5, unless the user decides otherwise): the product name shown changes; no file format changes |

**Done when:** a new user reading the README and getting-started docs sees only "Benchmate". The tier-1 run passes, including tests that compare generated decks against expected text (those expected strings are updated in the same commit).

### Phase 3 — Package and command (risk: medium; version 3.0.0)

This is the one large mechanical change. Do it in a single commit with **no other change mixed in**, so it can be reviewed as a pure rename and reverted as one.

| Step | Action |
|---|---|
| 3.1 | `git mv molbuilder benchmate` |
| 3.2 | Rewrite imports with a script that matches only real imports, never prose: `^(\s*)from molbuilder(\.|\s)` → `\1from benchmate\2`, `^(\s*)import molbuilder\b` → `\1import benchmate`, and the same inside `importlib.import_module("molbuilder…")` and `mock.patch("molbuilder…")` strings. Run it over `benchmate/`, `tests/`, `tools/` and `scripts/`. Commit the script in this folder for review. It must not touch the string `"molbuilder"` used as the tab key, the `/molbuilder` address or the notification kind (§ 2a) |
| 3.3 | `pyproject.toml`: `name = "benchmate"`; `[project.scripts]` gets `benchmate = "benchmate.cli:main"` **and** `molbuilder = "molbuilder.__main__:main"` (the compatibility command); `packages.find include = ["benchmate*", "molbuilder*"]`; package-data paths |
| 3.4 | Add the compatibility package `molbuilder/` (small and permanent until Phase 6). `molbuilder/__init__.py` installs an import alias, so that `import molbuilder.x.y` returns `benchmate.x.y` (one `sys.meta_path` finder, about 30 lines), and warns once per process: *"molbuilder is now Benchmate; import benchmate instead."* `molbuilder/__main__.py` prints *"`python -m molbuilder` is now `python -m benchmate`"* to stderr and forwards the arguments to `benchmate.cli.main` |
| 3.5 | Commands printed to users: `python -m molbuilder …` → `python -m benchmate …` (`runwrap.py:2033,3289`, `envs/_cli.py:626`, `cli.py:1633`, `envs/recipes.py:1227`, and every docs page) |
| 3.6 | `scripts/install-env.sh`: invoke `python -m benchmate`; the shim's own text says Benchmate |
| 3.7 | The `.pyz` bundles are rebuilt at prep, so new run folders get `benchmate` inside them. Old run folders keep their old self-contained bundles. **Check** (§ 5, C3) that a run folder prepared before the rename still launches and its monitor still runs |
| 3.8 | `__version__ = "3.0.0"`, the changelog line, and the git tag `v3.0.0` |
| 3.9 | Update `.pre-commit-config.yaml`, `Makefile` and `tools/` references to the package path |

**Done when:**
- `python -m benchmate --help` works;
- `python -m molbuilder --help` works and prints the notice;
- the full tier-1 run passes;
- a run folder prepared under 2.x launches and is watched under 3.0.0 (C3);
- `grep -rn "molbuilder" benchmate/` shows only the allowed names in § 3, the compatibility notices and history in comments (§ 5, C1).

### Phase 4 — Names on the machine (risk: medium)

Directories, environment variables and conda environments. Each one **reads the new name first and falls back to the old one**, so an existing machine needs no action.

| Step | Action |
|---|---|
| 4.1 | **Config, state and runtime directories** (`config_dir.py`): new `DIRNAME = "benchmate"`. Resolution: (1) `$BENCHMATE_CONFIG_DIR`, then (2) `$MOLBUILDER_CONFIG_DIR` (with a notice), then (3) `~/.config/benchmate` if it exists, then (4) `~/.config/molbuilder` if it exists (with a notice saying how to move it), then (5) create `~/.config/benchmate`. The same rule for the state and runtime directories |
| 4.2 | **Config file name:** read `benchmate.json`, or `molbuilder.json` when only that exists; the single writer `write_config_scope()` writes `benchmate.json`. The file's contents are unchanged |
| 4.3 | **Move command:** `benchmate envs migrate-config` moves `~/.config/molbuilder` to `~/.config/benchmate` and renames the file. It refuses when both exist, and keeps file modes (secrets stay 0600, via `placement.py`). `envs doctor` reports when a machine is still on the old layout |
| 4.4 | **Environment variables:** one helper in `config_dir.py`, `env_var("CONFIG_DIR")`, reads `BENCHMATE_<NAME>` and falls back to `MOLBUILDER_<NAME>` with a once-per-process notice. Replace every direct `os.environ` read of the 35 `MOLBUILDER_*` names with it. `install-env.sh` gets the same rule in a small shell function |
| 4.5 | **Conda environments** (`diagnostics.py` `DEFAULT_ENV_NAMES`, the host env name, `recipes.py`): new installs create `benchmate-*`. Resolution order for a category: (1) `envs.<category>` in the config, then (2) `benchmate-<x>` if it exists, then (3) `molbuilder-<x>` if it exists (`envs doctor` notes it). So existing machines keep using their existing environments |
| 4.6 | **Run wrappers already on clusters** name `molbuilder-*` environments inside them. They are not edited. Keep the old environments until no live or restartable run uses them; a warm retry re-runs the same wrapper |
| 4.7 | **Optional rename of environments:** `benchmate envs rename-legacy` does `conda rename -n molbuilder-x benchmate-x` (conda ≥ 4.14), or create-new then remove-old where `conda rename` is not available (mamba, micromamba). It **never** runs on its own, and it warns if any run folder under the projects tree still names the old environment |

**Done when:**
- a fresh machine bootstraps with only `benchmate*` names;
- a machine that still has only the old names keeps working unchanged, with doctor notices;
- `envs migrate-config` and `envs rename-legacy` have been tested on a copy of a real config, including secrets permissions.

### Phase 5 — Repository and publishing (risk: low, but done by the owner)

| Step | Action |
|---|---|
| 5.1 | Rename the GitHub repository `Qing-LAB/molbuilder` → `Qing-LAB/benchmate` in its Settings. GitHub redirects the old URLs, so existing clones keep working; update them later with `git remote set-url origin …` |
| 5.2 | Update every link to the repository in the README and docs |
| 5.3 | Optional: reserve `benchmate` on PyPI with a placeholder, even though the project is installed from source (`pyproject.toml` says it is not published) |

### Phase 6 — Remove the compatibility layer (after two minor releases, or 3 months)

| Step | Action |
|---|---|
| 6.1 | Remove the `molbuilder/` compatibility package and the `molbuilder` console script |
| 6.2 | Remove the `MOLBUILDER_*` environment-variable fallback and the old directory fallback (keep `envs migrate-config` for one more release) |
| 6.3 | Stop writing `molbuilder_version` beside `benchmate_version` |
| 6.4 | **Never removed:** readers of `.molwatch.log`, `.molstruct.json`, `molbuilder/<kind>@<n>` schemas and `mb_*.pyz` |
| 6.5 | Move this folder to `docs/archive/2026-xx-xx-rename-benchmate/` and close row N1 |

---

## 5. Checks — how each phase is verified

| Check | What | How |
|---|---|---|
| **C1 — no stray old name** | After Phase 3 or 4, every remaining `molbuilder` in the code is one of the allowed kinds: the Molbuilder tab (§ 2a, meaning 2), stored or wire names (§ 3), compatibility notices, history comments | Keep an allow-list file in this folder; the command `grep -rnI molbuilder benchmate/ \| grep -vFf allowlist.txt` must print nothing. Run by review (the repository no longer pins its own text in tests; [`process/testing.md`](?doc=process/testing.md) § 3b) |
| **C2 — no reader depends on the old text** | Before step 2.5, confirm no parser or detector matches "generated by molbuilder" or `# molbuilder` | `grep -rnI "generated by molbuilder\|# molbuilder" benchmate/parse benchmate/runs.py benchmate/runfiles.py`; also run the parse tests on decks generated before and after |
| **C3 — old run folders still work** | A run folder prepared under 2.9.x launches, warm-retries, is watched and shows results under 3.0.0 | Prepare a small PySCF task and a SIESTA task under 2.9.4, then switch the checkout to 3.0.0 and `launch`, `watch`, open Results. Repeat on Sol for one SLURM job |
| **C4 — old machine still works** | A machine with `~/.config/molbuilder`, `MOLBUILDER_*` variables and `molbuilder-*` environments works after Phase 4 with no action | Run `envs doctor`, `serve start` and one prep and launch on such a machine (or a copy of its config in a test home directory) |
| **C5 — fresh machine** | `bash scripts/install-env.sh bootstrap --yes` on a clean account creates only `benchmate*` names | A clean container or test account |
| **C6 — the test suite** | Each phase ends with the tier-1 run green | `tools/testrun.py run none2e`; e2e on the person's word ([`process/testing.md`](?doc=process/testing.md)) |

---

## 6. Browser-side names — decided separately

The JavaScript namespace `window.molbuilder` (227 uses), the DOM events `molbuilder:*` (30 uses) and the 5 storage keys are invisible to a person.

**Recommendation: leave them in Phases 1–5.** Renaming them is a large diff with no visible benefit, and it can break the Python-driven JS tests in subtle ways.

If renamed later (a separate, optional Phase 7):
- rename the namespace to `window.benchmate`, keeping `window.molbuilder` as the same object for one release;
- rename the events in one constant table (`lib/constants.js`);
- migrate the storage keys by reading the old key once and writing the new one.

---

## 7. Decisions

*(user, 2026-10-18: "D1 yes, D2 after a week, D3 not now, D7 A"; then "D4 yes, D5 age out, D6 as suggested")*

| # | Question | Recommendation | Decision |
|---|---|---|---|
| D1 | Is the rename "a major framework change" that moves the version to 3.0.0 at Phase 3? | Yes: the command and the import name change | **Decided: yes, 3.0.0** at Phase 3 |
| D2 | When to rename the GitHub repository (Phase 5)? | After Phase 3 lands and has run for a week; GitHub keeps redirects either way | **Decided: one week after Phase 3** has run without problems |
| D3 | Rename the browser-side names (§ 6)? | Not now | **Decided: not now** (Phase 7 stays optional) |
| D4 | Keep the `mb_*.pyz` and `MB_*` names permanently? | Yes: they read as "Benchmate" too, and existing run folders carry them | **Decided: yes, kept permanently** |
| D5 | Rename existing conda environments on the lab machines and Sol, or let them age out? | Let them age out, and use `envs rename-legacy` only on machines with no live runs | **Decided: age out** — new installs create `benchmate-*`; old environments keep working as fallbacks and are renamed only on machines with no live runs |
| D6 | Length of the compatibility period (Phase 6)? | Two minor releases or three months, whichever is later | **Decided: two minor releases or three months, whichever is later** |
| D7 | The structure-building tab is called **Molbuilder**. After the rename, keep that label (A), or relabel it **Structure builder** (B)? Either way its address `/molbuilder` stays | A: keep "Molbuilder" — after the rename it means the molecule builder inside Benchmate (§ 2a) | **Decided: A, keep "Molbuilder"** |

All seven decisions are made. Phase 0 is complete apart from steps 0.3 (name checks on GitHub, conda-forge and trademarks) and 0.5 (telling the other users); Phase 1 can start.

---

## 8. Log

| Date | Phase / step | What happened |
|---|---|---|
| 2026-10-18 | — | Plan written; icon H5 chosen; inventory measured at `83e72b2` |
| 2026-10-18 | 0.4 | D4 yes (`mb_*` names kept), D5 age out, D6 two minor releases or three months. All decisions made |
| 2026-10-18 | 0.4 | D1 yes (3.0.0), D2 one week after Phase 3, D3 not now, D7 A (keep "Molbuilder"). D4–D6 open |
| 2026-10-18 | — | § 2a added: the Molbuilder tab and the notification wire names are separated from the project name (user's instruction); D7 added |
