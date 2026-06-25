# Configuration — wrapper contract + generator inputs

> **Status**: rewritten 2026-06-24 against the wrapper-independence
> contract.  Supersedes the previous version of this doc (which
> designed a config-driven wrapper — wrong shape, removed entirely).
>
> Cross-reference: [design.md](design.md) decision-log; any change to
> the wrapper's contract or the generator's inputs must update this
> file in the same commit.

---

## 1. The wrapper contract

The generated `<basename>.run.sh` is a **self-contained shell script**.
Given the `.fdf` (or `.py`) input sitting next to it and the env it was
generated for, it runs the job under SLURM (or directly).

**What the wrapper does at runtime, in this order:**

1. Open a log file in the **current working directory** named
   `<basename>.runwrap-<YYYYMMDD-HHMMSS>.log`.  Tees stdout + stderr
   to that file while keeping them on the original streams.
2. Run the **server preamble** verbatim (e.g. `module load mamba`).
   Baked at generate time from `molbuilder.json`.
3. Run **project additions** verbatim (env vars, extra setup).
   Baked at generate time from `.molbuilder.json` if present.
4. Run the **activation command** verbatim (e.g.
   `source activate molbuilder-siesta-gpu`).  Baked at generate
   time; both the form (`source activate` vs `conda activate` vs
   `mamba activate`) and the env name are decided by the generator.
5. Launch the engine: `mpirun -np $_mpi_np siesta < <basename>.fdf >
   <basename>-runN.out` (or the PySCF equivalent).

**What the wrapper does NOT do:**

* **Does not change cwd.**  SLURM lands the job in `SLURM_SUBMIT_DIR`
  by default; direct users `cd` to the project dir before
  invocation.  The caller's cwd is the contract.  Outputs (log,
  `-runN.out`, etc.) land where the wrapper was invoked.
* **Does not read any external file** other than the engine input
  (`<basename>.fdf` / `<basename>.py`).  No config files, no
  `~/.bashrc`-style sourcing, no module discovery.
* **Does not read any environment variable to alter behaviour.**
  No `MOLBUILDER_PREACTIVATE_CMDS`, no autodetect toggle.  The
  scheduler vars it reads (`SLURM_NTASKS`, `SLURM_GPUS`) only
  inform the launch command, never change which conda env or which
  preamble runs.
* **Does not probe.**  No `conda info --base`, no `mamba info --base`,
  no `module avail`, no filesystem walks, no PATH searching.
  Everything is baked.
* **Does not have a fallback path.**  If `module load mamba` fails,
  if the env doesn't exist, if `source activate` errors — the
  wrapper aborts (`set -euo pipefail`) with the real bash error in
  the log.  No silent rescue, no alternative path.

**What the wrapper assumes about its environment:**

* `$HOME` is set (SLURM inherits, doesn't strip).
* The target env is installed and reachable through the toolchain
  the server preamble loads.  molbuilder's `install-env.sh` /
  `molbuilder envs install` machinery guarantees this.
* The `.fdf` (or `.py`) is in the caller's cwd.

---

## 2. The generator contract

The generator is `molbuilder.runwrap.render_run_wrapper` plus the
config-reading helpers it calls.  At generate time it can read
anything it needs.  Its job is to produce a wrapper that satisfies §1.

**What the generator reads at generate time:**

* The target env name (from the recipe / capabilities snapshot).
* `script_generation.preamble` from `molbuilder.json` (server-wide).
  Lines copied verbatim to step 2 of the wrapper above.
* `script_generation.activation` from `molbuilder.json` (server-wide).
  Names the activation command form (`source activate` /
  `conda activate` / `mamba activate`).
* `script_generation.preamble` from `.molbuilder.json` (project-scope,
  if present).  Lines copied verbatim to step 3.
* Engine-specific knobs (mpi_np, omp, BlockSize) — same as today.

**What the generator emits:**

A `.run.sh` whose content is the result of substituting the read
values into a fixed template.  No conditionals at runtime, no
detection blocks, no fallback branches.

**Failure mode at generate time:**

If `script_generation.preamble` is empty AND no project-scope file
exists, the generator emits a wrapper with an empty preamble.  That
wrapper will fail at activation time on any cluster that needs
`module load`.  The fix is on the operator: populate
`molbuilder.json`.  The generator does NOT bake a guess.

---

## 3. Where config lives

| scope | file | lookup |
|---|---|---|
| **server-wide** | `molbuilder.json` | cwd of `molbuilder run` / `molbuilder serve`, OR `~/.config/molbuilder/molbuilder.json` (XDG fallback) |
| **project** (optional) | `.molbuilder.json` | in the project directory (the dir holding the `.fdf` / `.py`) |

Both files share the same JSON schema.  The generator reads both,
merges them (project wins on scalar conflicts; `preamble` strings
concatenate server-then-project), bakes the result into the wrapper.

**The wrapper never reads either file.**  Once generated, the wrapper
is independent of both.

---

## 4. Schema — the bits the generator needs

```json
{
  "script_generation": {
    "preamble":   "module load mamba",
    "activation": "source activate"
  }
}
```

| key | type | required | meaning |
|---|---|---|---|
| `preamble` | string (multi-line bash) | no (empty default) | Verbatim lines run at the top of the wrapper, before activation.  Typically `module load mamba` on Sol; can include `export FOO=bar` lines for env vars. |
| `activation` | one of `"source activate"`, `"conda activate"`, `"mamba activate"` | no (default: `"source activate"`) | How the wrapper activates the env.  `source activate` is the form Sol uses; other clusters may need a different form. |

**Both keys are valid in either scope.**  Merge rules:

* `preamble`: server-wide string + `"\n"` + project string (server first, project after).  Either may be empty.
* `activation`: project value wins if set; else server value; else default `source activate`.

There are no other keys under `script_generation`.  No `autodetect_conda`,
no `preactivate_format`, no `preactivate`.  Those existed in a prior
version of this doc and were removed.

---

## 5. Other top-level keys in `molbuilder.json`

This doc is only about `script_generation`.  Other top-level keys
(`tls`, `auth`, `envs`, `rate_limit`, `secret_key_file`) are
unchanged.  Their schemas + usage are documented in
[`docs/deployment.md`](deployment.md).

They are also read at generate time / serve time only; none of them
change the wrapper's runtime behaviour.

---

## 6. What was removed from the previous version

The previous draft of this doc designed a config-driven wrapper.
Everything below was deleted because the wrapper now does no runtime
discovery:

* `script_generation.preactivate` (renamed to `preamble`, but more
  importantly: the wrapper used to "respect" this at runtime via a
  baked-in block + a runtime env-var hook; the env-var hook is gone)
* `script_generation.preactivate_format` (only one format existed
  anyway; remove)
* `script_generation.autodetect_conda` (the 6-path detection is
  gone; this knob has no meaning anymore)
* `MOLBUILDER_PREACTIVATE_CMDS` runtime env var (gone)
* The 6-path conda detection block in the wrapper (gone)
* The "fail loud vs autodetect" decision (irrelevant under the new
  contract — the wrapper always does exactly what was baked)
* The per-run log "preactivate trace via `set -x`" idiom (gone; the
  baked preamble is plain lines, traced by the log file's stderr
  capture only if the operator explicitly opted in via xtrace in
  their preamble)

The wrapper itself shrinks substantially as a result.

---

## 7. Migration

Existing `molbuilder.json` files with `script_generation.preactivate`
should be renamed to `preamble`.  The config reader emits a warning
on the old key but treats it as `preamble` for one release, then
drops the alias.

Existing `molbuilder.json` files with `script_generation.autodetect_conda`
should drop the key — it has no meaning.  The reader silently ignores
unknown keys, so this isn't load-bearing.

The example file `docs/molbuilder.json.example` needs to be rewritten
against this schema in the same commit as the code change.

---

## 8. Implementation plan

1. **`runwrap.py`** — strip the 6-path block, the env-var hook, the
   `autodetect_conda` plumbing, the cd, the multi-line error
   heredoc.  Replace with a small render that emits: log setup +
   verbatim preamble (server then project) + verbatim activation +
   engine launch.  Tests pinning the new shape; old tests pinning
   the discarded behaviours get deleted, not patched.
2. **`runtime_config.py`** — replace `get_script_generation` with the
   new schema (preamble + activation, that's it).  Keep
   `read_effective_config` / `write_config_scope` — they're sound.
3. **`docs/molbuilder.json.example`** — rewritten `script_generation`
   block matching the new schema.
4. **`README.md`** — deployment section's HPC-preflight subsection
   updated for the new key names + the wrapper-self-contained
   property.

UI work for editing config from the browser (Preflight card,
Generate-script preview modal) was in the previous draft.  It is
**not part of this doc** — those are how the user EDITS config, not
how the wrapper consumes it.  Whatever UI we build later, it writes
to the same `molbuilder.json` / `.molbuilder.json` files and the
schema above.

---

This file is the source of truth for the wrapper contract + the
config keys the generator reads.  Update it in the same commit as
any change to either.
