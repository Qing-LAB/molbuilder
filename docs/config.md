# Configuration — unified data model (design decision)

> **Status**: design decision, drafted 2026-06-24.  This is the core
> design document for how molbuilder reads and writes configuration.
> Implementation lands in stages per § 8.  Subsequent changes to the
> shape, scope rules, or API of molbuilder's persistent configuration
> must update this document in the same commit.
>
> Cross-reference: [design.md](design.md) decision-log; this doc owns
> the configuration data model in full.

---

## 0. Decision summary

Five decisions, summarised first so any reader gets the shape in 30
seconds:

1. **One schema, two scopes.**  All persistent configuration lives in
   `molbuilder.json` (server-wide) and `.molbuilder.json` (per-project,
   in the project directory).  Both files use **the exact same
   schema** — every section + key is legal in either scope.  Predictable
   deep-merge rules; project wins.

2. **One unified read API.**  Subsystems call
   `read_effective_config(project_dir)` and get back the merged
   document.  Section-specific getters (`get_auth`, `get_envs`,
   `get_script_generation`, ...) read from that one merged dict.  No
   subsystem reads `molbuilder.json` directly anymore; no subsystem
   carries its own ad-hoc config file.  **No fragmentation.**

3. **One write API.**  `write_config_scope(project_dir, patch)`
   writes a partial document to whichever scope the caller targets,
   preserving keys outside the patch.  CLI tools (`auth-setup`,
   future wizards) and the web UI both use this.

4. **One UI surface per config family.**  Every web tab that emits a
   script carries a uniform "Environment preflight" card reading +
   writing `script_generation.preactivate`.  Every "Generate Script"
   button opens a preview modal where the user can edit the full
   rendered files before saving.  Two opportunities to finalize.

5. **Script generators read from one source.**  `runwrap.py` and any
   future generator pulls from `read_effective_config` — the same
   path the UI shows.  What the user sees in the preflight card is
   what lands in the wrapper, deterministically.

These five together produce the property the user asked for:
**no fragmented info, one data structure, one source, one API**.

---

## 1. The principle behind the decision

The molbuilder server, the CLI, and the web UI all consume persistent
configuration.  Without discipline, each grows its own file (auth in
`molbuilder.json`, env-routing in another, per-project settings in a
sidecar, server preferences in `~/.molbuilderrc`).  That's how
configuration becomes a maintenance burden — every new feature opens
the question of "which file does this go in?", and users have to
remember N lookup chains.

The principle: **persistent configuration is one logical document with
two physical scopes**.  Everything beyond per-job parameters
(those live in the L1 dataclasses + form schema) and ephemeral state
(those live in the project workspace) is in this one document.

### Alternatives considered + why rejected

| alternative | why not |
|---|---|
| `molbuilder.json` server-wide only; per-project context lives in dataclass sidecars (`.molstruct.json`) | dataclass sidecars are per-STRUCTURE (one per `.xyz`); some config (preactivate, default basis) is per-PROJECT.  Stuffing project config into the structure sidecar mis-uses that file's purpose. |
| `molbuilder.json` + a separate `project.molbuilder.json` with a different schema | violates the unification principle — users have to learn two schemas.  Catches no one if the same key means different things in the two scopes. |
| Per-subsystem config files (`auth.json`, `envs.json`, `script-generation.json`) | exactly the fragmentation we want to avoid.  Every new feature would add a file. |
| Inline config in the L1 dataclasses, no JSON at all | the L1 dataclasses are PER-JOB (one per `.fdf` / `.py` rendering).  Persistent config crosses jobs by definition; it doesn't belong there. |

The chosen design — one schema, two scopes, one merge — is the
minimum mechanism that satisfies "persistent + per-project optional"
without forking the schema.

---

## 2. One schema, two scopes

molbuilder reads configuration from up to two files:

| scope | file | who owns it | precedence |
|---|---|---|---|
| **server-wide** | `./molbuilder.json` in the cwd of `molbuilder serve` (and `python -m molbuilder ...` CLI runs).  Fallback: `~/.molbuilder/molbuilder.json`. | the operator who runs the server | base layer |
| **project** | `.molbuilder.json` in the project directory (the dir holding a `.fdf` / `.py` / `.xyz` group) | the user editing that project | overlay; wins on conflict |

**Both files share the exact same JSON schema.**  Every section + key
documented below is valid in either scope; what differs is precedence:
the project file's values win key-by-key when the same key appears in
both.  This is the same model as git config global / local, npm config
user / project, etc. — chosen because it's predictable and matches
operator expectations.

### Why both scopes?

* **Server-wide** is the right place for things the deployer arranged:
  TLS cert paths, auth providers + allowlists, conda env names if
  they're non-default, and (new) HPC environment-preparation steps
  that every job on this server needs (e.g. `module load mamba`).
* **Project** is the right place for things one specific calculation
  needs: a project-local preflight tweak, a non-default basis or
  cutoff that shouldn't pollute other projects, a project-specific
  output dir.

The cleanest analogy: server-wide is "what's true about this machine",
project is "what's true about this calculation".  If you're unsure
which, default to server-wide and only push into a project file when
you genuinely need it for that project alone.

### Lookup order (read by the runtime)

1. Read `./molbuilder.json` from `molbuilder serve`'s cwd OR the CLI's cwd.
2. If absent, fall back to `~/.molbuilder/molbuilder.json` (per-user).
3. Result of (1) or (2) is the **server-wide layer**.
4. For any operation scoped to a project directory (any
   `molbuilder fdf` / `molbuilder pyscf` / `molbuilder run` / web UI
   action against a project), read `<project_dir>/.molbuilder.json`
   if present.  Result is the **project layer**.
5. **Merge**: deep-merge server-wide ← project, where:
   * Scalars (string, number, bool, null): project value replaces server value.
   * Object values (`{...}`): keys merged recursively.
   * Array values (`[...]`): project value replaces server value entirely
     (no element-wise merge — predictable, even if occasionally verbose).

### Lookup order (write from the UI)

The web UI's "save" actions follow this convention:

* Edits made inside a project context (any tab open against a project)
  write to that project's `.molbuilder.json`, creating it if needed.
* Edits made in the (planned) Settings tab write to the server-wide
  `molbuilder.json`.
* The UI shows a "(server default)" / "(project override)" badge next
  to every value so the user can always see which scope a value came
  from.

---

## 3. Top-level schema

The full top-level layout, current + proposed:

```json
{
  "tls":               { ... },          // server-wide only in practice
  "auth":              { ... },          // server-wide only in practice
  "envs":              { ... },          // either scope
  "rate_limit":        { ... },          // server-wide only in practice
  "secret_key_file":   "...",            // server-wide only in practice
  "script_generation": { ... },          // either scope -- new in 2026-06-24
  "_comment_*":        "..."             // ignored; inline docs in templates
}
```

Sections marked "server-wide only in practice" are still legal in a
project file, but it's almost never the right place: a project-scope
`auth` block would only apply when that project is being edited,
which is meaningless for sign-in.

### 3.1 `tls`

| key | type | required | meaning |
|---|---|---|---|
| `cert` | string | yes if section present | path to PEM cert |
| `key`  | string | yes if section present | path to PEM private key |

See [deployment.md § Quick start: the config file](deployment.md).

### 3.2 `auth`

Setup walkthroughs per provider kind: [deployment.md § Setup walkthrough](deployment.md).
Generate via `python -m molbuilder auth-setup` (writes to server-wide).
Schema enforced by `molbuilder.runtime_config._validate_provider`.

### 3.3 `envs`

Conda env name overrides, mapping logical category → env name.

```json
"envs": {
  "siesta":  "molbuilder-siesta",
  "pyscf":   "molbuilder-pySCF",
  "mdtools": "molbuilder-MDtools",
  "tests":   "molbuilder-tests"
}
```

Defaults are these exact names; this section is only needed if you've
renamed your envs.

### 3.4 `rate_limit`

Server-wide only.  Per-IP throttle.  See `molbuilder/web/rate_limit.py`.

### 3.5 `secret_key_file`

Path to the Flask session signing key (mode 0600).  Auto-generated by
`auth-setup`; only set this manually if you're hand-editing.

### 3.6 `script_generation` (new — 2026-06-24)

The new home for everything the script generators read.  Today carries
the `preactivate` block; future fields land here too (e.g. default
SLURM preamble, default job-script header comments).

```json
"script_generation": {
  "preactivate": "module load mamba\nmodule load cuda/12.4",
  "preactivate_format": "shell"
}
```

| key | type | default | meaning |
|---|---|---|---|
| `preactivate` | string (multi-line bash) | `""` | shell commands run at the top of every generated `.run.sh`, BEFORE conda detection, BEFORE the env activation step.  Each line is copied verbatim. |
| `preactivate_format` | enum: `"shell"` | `"shell"` | reserved for future expansion (e.g. Lmod `module-load` lists); today only `shell` is supported. |

**Both scopes merge by string concatenation, in the order
server-wide → project**.  Rationale: the operator-set lines (cluster
modules) should run first; the project-set lines (project-specific
overrides) should run after.  Example:

* Server-wide `preactivate`: `module load mamba`
* Project `preactivate`: `export PROJECT_SCRATCH=/scratch/$USER/this-project`

Produces, at the top of every generated `.run.sh` in that project:

```bash
# === SERVER-WIDE PREACTIVATION (from ./molbuilder.json) =============
module load mamba

# === PROJECT PREACTIVATION (from .molbuilder.json in this project) ==
export PROJECT_SCRATCH=/scratch/$USER/this-project
```

Sentinel comments make the source of each block explicit so a user
inspecting the wrapper sees exactly where each line came from.

#### How the script generator reads it

The generator (`molbuilder.runwrap`) calls
`runtime_config.get_script_generation(cfg)` and gets back a dict with
`preactivate` already merged across scopes.  Embedded into the wrapper
text via the same sentinel-comment template regardless of source.

The runtime activation block (the 6-path detection added 2026-06-24)
remains as a safety net for anything `preactivate` didn't arrange.  The
order at runtime is:

1. **Baked-in `preactivate` block** (operator + project, from this section)
2. **`MOLBUILDER_PREACTIVATE_CMDS` env var** (runtime escape hatch,
   for rsync'd wrappers and one-off overrides)
3. **6-path conda detection** (auto-discovery fallback)
4. **`conda activate <target_env>`**

If `preactivate` already put conda on PATH (e.g. via `module load
mamba`), step 3 succeeds immediately on path 3 (`mamba info --base`).

---

## 4. Unified read / write API

A single Python entry point reads and merges both scopes:

```python
from molbuilder.runtime_config import read_effective_config

cfg = read_effective_config(project_dir=None)
# cfg is the server-wide layer

cfg = read_effective_config(project_dir=Path("/home/qqing/projects/BDT"))
# cfg is the merged result (server-wide ← project)
```

Subsystem getters (`get_auth`, `get_tls`, `get_envs`,
`get_script_generation`, ...) take that merged dict and return their
section.  None of them know which scope each value came from — the
merge is opaque at the consumer layer, which is what "unified API" means.

### 4.1 Provenance tracking (optional)

For UI surfaces that need to show "(server default)" / "(project
override)" badges, the same module exposes:

```python
from molbuilder.runtime_config import read_effective_config_with_provenance

cfg, provenance = read_effective_config_with_provenance(project_dir)
# cfg            = the merged dict (same as read_effective_config)
# provenance     = parallel dict {key_path: "server" | "project"}
#                  e.g. provenance["script_generation.preactivate"] == "project"
```

The web API exposes this so each rendered config field can carry its
badge.  CLI tools ignore it — they only care about effective values.

### 4.2 Writes

```python
from molbuilder.runtime_config import write_config_scope

write_config_scope(
    project_dir=None,                  # writes to server-wide
    patch={"script_generation": {"preactivate": "module load mamba"}},
)

write_config_scope(
    project_dir=Path("/home/qqing/projects/BDT"),
    patch={"script_generation": {"preactivate": "export ...\n"}},
)
```

`patch` is a partial document; existing keys outside the patch are
preserved.  Files are written mode 0600 (consistent with the
`auth-setup` precedent — same rationale: even if a section carries no
literal secrets, the operator-set values often carry deployment
context that's not meant to leak).

---

## 5. UI surface

### 5.1 Preflight card on each generating tab

Every web UI tab that emits a `.run.sh` (Structure-optimization,
Spectrum-calculation, Transport-calculation, and any future tab)
carries an **Environment preflight** card at the top.  The card:

* Shows the effective `script_generation.preactivate` text (merged
  result, with badges naming the contributing scopes).
* Provides a multiline textarea for editing.
* Save button writes the project-scope value (creates
  `<project_dir>/.molbuilder.json` if absent).
* A "Save to server defaults instead" link writes the server-wide
  value (the operator's path — requires write access to the file the
  server was launched against).
* Includes a collapsible help hint: "Lines run at the top of every
  generated .run.sh BEFORE conda activation.  Use for cluster setup
  like `module load mamba`.  Server defaults + project additions are
  concatenated in that order."

The card is the discovery surface for the feature — users don't have
to know about either JSON file to set it correctly.

### 5.2 Generate-script preview modal

Every "Generate Script" button (currently emits files to disk directly)
opens a preview modal instead:

* Shows the rendered content of every file about to be written (the
  `.fdf` / `.py` + the `.run.sh` wrapper + any per-stage fdfs for
  multi-stage SIESTA runs).
* Each file gets its own editable textarea.
* "Save" commits the (possibly-edited) content to disk.
* "Cancel" closes without writing.
* "Reset to generated" per-file discards manual edits and re-renders.

The modal gives the user a second opportunity to finalize: the
preflight card handles persistent per-project context; the modal
handles one-off edits ("for this run only, add an extra MPI env var").

### 5.3 Settings tab (later)

For sysadmins / power users who'd rather edit server-wide settings in
the browser than SSH in.  Phase 4 — defer until needed.

---

## 6. Migration / backward compat

* Existing `molbuilder.json` files continue to work unchanged — the
  new `script_generation` section is purely additive, and the new
  project-scope file is purely opt-in.
* The lookup chain falls back to existing behavior when the project
  file is absent (which is the default).
* No subsystem is moved into `script_generation` from elsewhere — it's
  a new section, not a reshuffle.
* Backward-compat for the `MOLBUILDER_PREACTIVATE_CMDS` env var is
  preserved indefinitely: the runtime activation block still honors
  it, AFTER the baked-in `preactivate` block.

---

## 7. Open questions (need user input before code)

| # | question | options | recommendation |
|---|---|---|---|
| 1 | Is `~/.molbuilder/molbuilder.json` the right per-user fallback path, or `~/.config/molbuilder/molbuilder.json` (XDG)? | (a) `~/.molbuilder/...` (matches the existing `secret_key_file` default in `molbuilder.json.example`) (b) `~/.config/molbuilder/...` (XDG-conformant; already used by `auth-setup`) | **(b) XDG** — auth-setup already establishes this; we should be consistent.  Migrate the `secret_key_file` default in a follow-up. |
| 2 | Project-scope filename: `.molbuilder.json` (hidden) vs `molbuilder.json` (visible) vs `<project>.molbuilder.json`? | (a) `.molbuilder.json` — hidden, parallels `.gitignore` / `.editorconfig` (b) `molbuilder.json` — visible, no naming collision possible if server-wide always lives in serve cwd (c) `<project>.molbuilder.json` — fully unambiguous, but ugly | **(a) `.molbuilder.json`** — hidden, conventional for per-project config sidecars.  Visible-file proponents can `ls -a`. |
| 3 | Array-value merge rule: project replaces vs project appends? | (a) replace (predictable) (b) deep-merge with element identity (complex) | **(a) replace** — simpler, predictable, the documented behavior in this draft. |
| 4 | Should the modal in § 4.2 ALSO allow editing the structure (`.fdf` content) and not just the wrapper?  Today's "Generate Script" emits both. | (a) yes — modal has tabs/textarea per file including the .fdf (b) no — modal edits only the wrapper; structure files always re-render from the dataclass | **(a) per-file**.  The whole point of the modal is "two opportunities to finalize"; restricting to the wrapper only is a half-feature. |
| 5 | Where do per-tab config overrides go in the schema?  e.g. a project might want different `preactivate` for SIESTA vs PySCF jobs (one needs `module load mamba`, the other needs `module load python/3.12`). | (a) keep `script_generation.preactivate` as a global string and let the user manage their own conditionals (b) make it `{ "default": "...", "siesta": "...", "pyscf": "..." }` with per-engine override | **(a)** for v1, **(b)** if real-world use shows it's needed.  Don't over-engineer the schema. |

---

## 8. Implementation plan

Once the questions above are settled, the rollout is:

1. **`runtime_config.py`** — add `read_effective_config(project_dir)`,
   `write_config_scope`, `get_script_generation`, provenance tracker.
   Schema validator for the new section.  Tests pin the merge rules.
   ~150 LOC, 12-15 test cases.

2. **`runwrap.py`** — read `script_generation.preactivate` from the
   effective config, embed it between sentinel comments at the top of
   the wrapper.  Unconditionally, even if empty (the comment block
   itself is the signal that the feature is active).  Tests pin the
   sentinel format + the concatenation order.  ~30 LOC, 6 test cases.

3. **`docs/molbuilder.json.example`** — add the new
   `script_generation` block as a commented-out template.  Add a
   parallel `.molbuilder.json.example` for project scope.  Both
   include `_comment_*` keys explaining each field.

4. **Web UI Phase A** — preflight card on the three generating tabs.
   Reads + writes via a new `/api/config/effective` + `/api/config/save`
   pair.  Shows scope badges next to each value.  ~250 LOC JS + CSS,
   ~80 LOC Python (API), ~8 e2e test cases.

5. **Web UI Phase B** — Generate-script preview modal.  Refactors
   each tab's Generate button + the corresponding API endpoint to
   return content instead of writing immediately.  Save endpoint
   accepts the (possibly-edited) content.  ~400 LOC JS + CSS,
   ~120 LOC Python (API), ~10 e2e test cases.

6. **`docs/deployment.md`** — cross-reference this doc, deprecate
   the auth-specific subsections that drift into general config.

Phases 1-3 are the **non-UI MVP**: solves the ASU sc002 case, no UI
work.  Phase 4 + 5 land the full UX vision.

---

## 9. What this doc supersedes

* Inline mentions of "molbuilder.json" scattered across `deployment.md`
  remain valid as-is, but new config-related text should land here and
  link out from `deployment.md`.
* The previous design conversation (the auth-setup wizard) created a
  config-writing CLI; `auth-setup` continues to be the canonical path
  for the `auth` section.  This doc covers everything else.

---

This file is the new source of truth for molbuilder configuration.
Update it in the same commit as any new config-reading code.
