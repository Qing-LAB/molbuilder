# Hand-over — the env-installer / config-and-secrets session of 2026-09-12

**Role:** Plan · **Domain:** ops / envs / configuration

## 0. THE TARGET — the design this work is converging on

**Read this first. Everything after it is distance from here.** The session this
document audits produced a list of defects, and a defect list with no target is
just patching: each item gets answered where it was noticed instead of where the
invariant lives, which is exactly how the § 2 list grew as long as it did.

Both frameworks have the same shape: **one home per decision · the data carries
the situation · one door executes · the verdict is derived.**

### 0.1 The env installer — two state machines, one runner

From [`env-framework.md`](?doc=ops/env-framework.md) § 2-§ 7, whose diagram is the
authority:

```
recipes.py ── Recipe ──┬── CondaPackage(spec, optional, reason)
  (the registry)       └── PipPackage(name, source, extras, optional,
   § 3                                force, fallback_to_index, reason)
          ┌─────────────────────────┴──────────────────┐
   create_step_for · pip_step_for · verify_step_for   audit_packages
     conda_argv       pip_argv                        (reads disk only)
        (record -> InstallStep; the ONLY translators)        │     § 6
                    │                                       │
              plan_install        ── steps, no side effects  │
                    ▼                                       ▼
                 run_step  ◄───────── repair § 7 ◄───── doctor verify § 6
              (the one door)
                    │
              Outcome.decide    ── the one transition rule
                    │
          OK · RECOVERED · DEGRADED · FAILED · SKIPPED        § 5.1
```

The target as checkable invariants:

| | invariant | owner |
|---|---|---|
| **T1** | **Two state machines, and nothing else decides success.** `EnvState` before we touch an env, `Outcome` after each step, `Outcome.decide` the only transition rule. No status string, no boolean, no second notion of success beside the five outcomes | § 2, § 5.1 |
| **T2** | **The registry is the only place that says what an env is.** Recipes are data; a fact about a package lives ON that package — no parallel `optional_*` list | § 3, § 3.2 |
| **T3** | **Every situation is a FIELD on a record, never a branch in the runner.** The runner has no per-phase special case | § 4.2, § 5.2 |
| **T4** | **One door runs every step — `run_step`.** Exactly two things sit outside it and the document names both: `validate.py`'s post-install probes, and `builds.run_build_spec`. A third is a finding, not a fix | § 5, § 5.4, § 5.5 |
| **T5** | **One translator per record kind, one speller per command line.** `create_step_for` · `pip_step_for` · `verify_step_for` · `conda_step_for`; `pip_argv` · `conda_argv`. Nothing else builds an argv | § 5 |
| **T6** | **The verdict is DERIVED from the steps**, never tracked beside them | § 5.3 |
| **T7** | **The audit reads disk, never the network**, and answers on identity *and* provenance; **repair re-reads the RECORD**, never a flattened copy of an instruction | § 6, § 7 |
| **T8** | **One speller for a remedy the program prints.** `_fix_cmd` is the launcher form; a printed fix names the *effective* env and the *detected* manager, and resolves through `recipe_by_name` | § 8 + `installation.md`'s remedy rule |
| **T14** | **The manager is asked, never imitated.** A step enters an env through the detected manager's own mechanism; a path inside an env is asked for, never derived from the manager binary or a layout guess; a remedy names the detected manager and may not destroy working state; a manager's bug is a declared fallback, not the default path. Stated as **M1-M5** *(user, 2026-09-12: "it might not be conda at all... use the detected activation method rather than directly hacking to the directory")* | `installation.md` § *"One door for RUNNING"* + `env-framework.md` § 5.6 |

### 0.2 Config and secrets — one resolver, one name, one writer

From [`configuration.md`](?doc=configuration.md) § 2.1c, § 2.3, § 3.1 and
[`architecture.md`](?doc=execution/architecture.md) § 7:

| | invariant | owner |
|---|---|---|
| **T9** | **Directories come from `config_dir`; a file's NAME belongs to its format owner.** `config_dir` spells only the files no format owns. Nothing climbs a parent chain to a root, and nothing re-spells a filename molbuilder writes | **A11** |
| **T10** | **One writer puts bytes at a path it means to keep — `persist.write_bytes` — and privacy is a PARAMETER of it.** A credential is not a reason to write your own. Two shapes may sit outside it (stage-then-validate, and append) and each must be named where the rule is stated | § 2.3 |
| **T11** | **Modes are set at creation and asserted on arrival.** The config root `0700`, a credential `0600`; a directory or file that ARRIVES loose is tightened or reported, because no writer controls how it arrived | § 2.1b |
| **T12** | **A path is asked for, never assembled**, and every surface that resolves one PRINTS where it resolved from | § 3.1, § 2.2 |
| **T13** | **A retired key is refused, never ignored**, and no live text advertises one | § 2.1a, § 2.1e |

### 0.3 What to do with a finding

**Restore the invariant, do not answer the instance.** For every item in § 2 the
question is *which T does this break, and where does that T live* — then fix it
there and sweep the restatements. § 2.G maps the items onto T1-T13 and collapses
them into the structural moves; working from that map is the difference between
closing this list and re-growing it.

Three items name a **missing** invariant rather than a broken one (no rule covers
them): **A0** (nothing forbids deleting the env the process runs from), **A4**
(nothing asserts the mode of a directory another verb created), **D13** (the host
env's name has no persistent home). Those need a rule written before code.

### 0.5 THE THREE SETTINGS — what is available where

*(User, 2026-09-13: "you need to make sure you understand the premise of the
environment / available things for each script to run on local machine as well
as on remote HPC.")* Verified against `running-a-job.md` § 2.0a and
`runwrap.MONITOR_COMPANIONS`, and **one item of § 2 below was wrong because of
it** — see **I7**.

| setting | what is available | what molbuilder code runs |
|---|---|---|
| **local, interactive** — `envs install` / `doctor` / `serve`, the web UI | the detected manager, the host env, the config directory | the whole package |
| **local, generated wrapper** — `--mode direct` | *nothing inherited.* The baked preamble + activation, and whatever they put on `PATH` | **`mb_monitor.py` + `config_dir.py` only** |
| **remote compute node** — `sbatch` | *nothing inherited*: a non-interactive shell that reads no rc files, no module loaded, no env active, `$HOME` often NFS | **`mb_monitor.py` + `config_dir.py` only** |

Three consequences, and each one constrains this migration:

1. **Exactly two files travel** (`runwrap.MONITOR_COMPANIONS`), run by the JOB's
   own python in a backend env with no molbuilder and no numpy. `monitor.py` has
   exactly one molbuilder import, inside a function, with a flat fallback
   (`from config_dir import config_dir`); `config_dir.py` imports nothing of
   ours. **The layering table cannot see this** — it would allow `config_dir` to
   import `persist`, both L1 and perfectly legal, and fatal here. The failure is
   silent: `runwrap` records the last time it happened, and it cost *every
   production run's monitor dying at import with stderr to `/dev/null`*. Now
   reproduced by a test that stages the two sources in a directory where
   `molbuilder` is genuinely unimportable and runs them
   (`test_layering.py::test_every_file_that_ships_beside_a_job_imports_without_molbuilder`).
2. **Activation on a node is the operator's declared line, never a probe.** The
   wrapper bakes `script_generation.activation` verbatim, because the machine it
   runs on cannot be asked at render time. M1 governs the local dispatch only,
   which `env-framework.md` § 5.6 now states.
3. **The machine record is read at PREP time**, on the submitting machine, and
   the calculation's snapshot travels inside the bundle. No node imports
   `scheduler/record.py` — which is why its resolvers may import `config_dir`.

---

## 0.4 What this document is, and why it exists

16 commits landed on 2026-09-12 (`4392964c^..8cb36f1f`, 48 files) across the
environment installer, config-and-secret handling, and their contracts. **Three
independent reviews then audited that work**, with instructions to falsify the
commit messages rather than confirm them. This file is the result: what is
genuinely done, and what is not.

**It exists because the session's own reports were not trustworthy.** Several
commit messages claim a sweep that stopped short, and four documentation
statements written that day are false — two of them in text that `envs
init-config` ships into a user's config directory. Treat a commit message from
that day as a hypothesis and this file as the corrected record.

**Every row below carries its evidence and how it was checked:**

| mark | means |
|---|---|
| **RAN** | reproduced by running something; the command or observation is given |
| **READ** | established by reading code at the cited `file:line`; not executed |

**Order of reading: § 0 the target, then § 1 what is already true, then § 2 the
distance.** § 1 exists to stop the next session re-deriving settled work, which is
this project's recorded failure mode; § 0 exists so § 2 is not worked through as a
list of patches.

## 1. DONE — verified, do not redo

Audited by replaying each commit's behavioural claim against the tree. **14 of 16
claims verified outright, 2 partial** (the partials are §§ 2.C and 2.D7).

| what | evidence | mark |
|---|---|---|
| `auth_setup.write_secret_file` is atomic **and** `0600`; a failed write leaves the previous secret byte-identical; a planted symlink is replaced, not followed | new file `0600`, parent forced `0700`, pre-existing `0644` → `0600`; lone-surrogate write raised and the original survived with no temp litter | **RAN** |
| `notify-token --route a/b` is refused and the existing keys survive | exit 2 "Nothing was written"; file byte-identical; writer and reader share `monitor.is_route_segment` | **RAN** |
| `bootstrap` exits non-zero when it cannot seed the config directory | writable+`--yes` → 0; unwritable root → 1; no-tty without `--yes` → 1 | **RAN** |
| `bootstrap` warns about an unwritable root **and** about having no terminal, **before** the first install | warning at output index 315 vs first `[1/N]` banner at 1309 — order asserted, not presence | **RAN** |
| `bootstrap --dry-run` writes nothing and runs no doctor pass when every env is present | config dir never created, no prompt, exit 0 | **RAN** |
| `bootstrap --help` does not create the host env when it is absent | stubbed shim: `--help` → no marker, exit 0; same harness with `--yes` → create reached | **RAN** |
| Read-only shim verbs (`doctor`/`list`/`validate`/`repair`) run with stdin closed and no flag | no "no TTY", no "No such option '--yes'"; gate is `_verb_changes_envs` (`scripts/install-env.sh:455`) | **RAN** |
| `install <conda-only> --clean` really removes the env, and the banner no longer promises artifacts that do not exist | captured argv `['…/conda','env','remove','-n','molbuilder-pySCF','-y']` | **RAN** |
| `get_scheduler()` returns `None` for the config `init-config` seeds | seeded file carries `"scheduler": {}` → `None`; falsy non-objects still raise (the hole did not widen) | **RAN** |
| `MOLBUILDER_HOST_ENV` is honoured by the Python layer; bootstrap plans no second host env | `envs list` shows the override; `install molbuilder --dry-run` emits `conda create -n mb-dev-zzz` | **RAN** |
| `machine_config_warnings()` is reached by `serve foreground` and `serve start`, before anything starts | marker-and-raise substitution hit on both paths | **RAN** |
| `service_validate_url` is neither required nor written, and `cas.py` never mentions it | wizard output has no such key; validator accepts its absence | **RAN** |
| `_run_build_phase` is called correctly — the TypeError that broke **every source build** for four commits is gone, and execution reaches a phase | signature matches the call at `builds.py:1833`; unmonkeypatched `run_build_spec` reached the phase | **RAN** |
| The serve log is `0600` in a `0700` directory **on the supervisor's path**, rotation included, and tightens a pre-existing loose dir/file without losing history. ⚠ **Narrowed 2026-09-12 after the residue sweep** — this row said it without the qualifier, and `serve status` creates the same log through its own bare `mkdir` + `open(..., "ab")` (`cli.py:2415`), landing `0664` in `0775` when it is the first writer. See **I5** | `.1.gz` / `.2.gz` both `0600`; `status`-first measured `0664`/`0775` | **RAN** |
| No recipe prints a fix command naming an unregistered recipe | 7 literal hits scanned; all resolve through `recipe_by_name` | **RAN** |
| `docs/ops/examples/` is gone and nothing points at it; no config file ships in the repo | — | **RAN** |
| `jobset probe --write` writes `environment.json`, `--name sol` writes `environments/sol.json`, `--name this` is refused | — | **RAN** |
| Every path and resolver drawn in `configuration.md` § 3.1 resolves as drawn (all 14) | — | **RAN** |
| `importorskip("pyscf.gto")` in all three files; host env no longer carries `pyscf-properties` and `envs doctor` reports 20/20 | — | **RAN** |

## 2. NOT DONE — the work list

Ordered by what it costs a person. **Nothing here has been started.**

### A. Defects introduced on 2026-09-12

| id | what | evidence | mark |
|---|---|---|---|
| **A0** | ✅ **CLOSED (Phase 0).** ⚠ **`envs install molbuilder --clean` DELETED THE ENV THE PROCESS IS RUNNING FROM, and `doctor` prints that command as its remedy.** Opening `--clean` to conda-only recipes (`_cli.py:1105`, `:1301`) put the **host** recipe in scope — it is conda-only (`recipes.py:890`, `category=None`) — and there is **no self-removal guard anywhere** in `_cli.py`. `_render_doctor` (`_cli.py:241-245`) prints `install <name> --clean --yes` for **any** env whose verify failed, the host included; with `--yes` there is no prompt (`_cli.py:1351`). The shim dispatches as `<prefix>/bin/python -m molbuilder` **without activating** (`install-env.sh:809-851`), so conda's own "cannot remove current environment" guard never fires: the removal proceeds, then a fresh `conda create` is planned from an interpreter whose prefix no longer exists. A failure between remove and create leaves the machine with **no host env and no molbuilder**. The new test for this change uses `molbuilder-pySCF`; the host — the one self-destructive case — is untested. **Fixed:** `install.runs_from_prefix()` is the one answer to *"is this the env we are running from"* — the registry's prefix against the interpreter's own, both GIVEN rather than derived, and `CONDA_PREFIX` deliberately not consulted because the shim dispatches without activating. `--clean` refuses with exit 2 before anything is removed and prints the three-step from outside the env; `doctor` stops offering `--clean` for that env and prints the same route, spelled with the detected manager (`remove_env_cmd`, one home). Tested on the **host** recipe — the case the 2026-09-12 test did not cover — and mutation-verified: the mutant dispatches `['/fake/conda','env','remove','-n','molbuilder','-y']` | `recipes.py` host is `build_spec=None`; `_env_prefix('molbuilder')` == `sys.prefix` == `/home/qqing/miniconda3/envs/molbuilder`; no guard in `_cli.py` | **RAN** |
| **A4** | ✅ **CLOSED (Phase 3a).** **`jobset probe --write` created the config directory world-readable** — `jobset/_cli.py:3665` `mkdir(parents=True, exist_ok=True)` with no mode. Measured `drwxrwxr-x`, and a later `envs init-config` reports it *"kept"* and **does not tighten it** (`initconfig._ensure_root` only sets the mode on a directory it creates). `runtime_config.CONFIG_DIR_MODE = 0o700` was added the same day and is **referenced by nothing**. `machine_config_mode_warning` checks the file, never the directory. So every secret later placed there — `secret_key`, `google_client_secret`, `notify_keys`, and the `tls.key` the new `secrets/README` invites — is `0600` inside a directory anyone on a shared login node can list and traverse. And `jobset probe --write` is the **first command** the `environments/README` seeded that day tells the user to run on the target. Sibling: `serve_daemon.py:305` created `run/` at the umask default three lines from `_mkdir_private`. **Fixed:** one creator, `config_dir.ensure_private_dir`, called by the probe writer, both seeding paths, `write_config_scope` (which also stops ignoring `CONFIG_DIR_MODE`) and the supervisor's `run/`. The audit reported this directory at `0775` on the developer's own machine the first time it ran | `drwxrwxr-x` after probe, unchanged after `init-config`; `CONFIG_DIR_MODE` grep: no consumer | **RAN** |
| **A1** | ✅ **FIXED, though not for the reason the row gave.** The claim — the rotated `.gz` stays unflushed until GC — is **not reproducible** on CPython 3.14: the archive is complete the moment rotation returns and no descriptor leaks. The CODE was still wrong: `gzip.GzipFile` borrows a `fileobj` and never closes one, so handing it ours left a handle with no owner, collected only by refcounting — an implementation detail, not a guarantee. **Whoever opens it closes it**; it now does | **RAN** |
| **A2** | ✅ **CLOSED (Phase 3a).** **`serve-<port>.stacks.log` was created world-readable** — `cli.py:2261-2263` does a bare `mkdir` + `open(_sp,"a")`, giving `0775`/`0664`, in the directory the same session tightened. `configuration.md` § 3.1 (written that day) states `0600`, and § 2.3's table names `serve_daemon._open_private` as the door for an appended log. Under `--no-supervise`/`--debug` no `LogRoll` runs, so nothing tightened the directory either. **Fixed:** `ensure_private_dir(..., tighten=True)` + `serve_daemon.open_private`, which is now PUBLIC — § 2.3's table named it as the door for an appended log while it was private, and a door the neighbours cannot reach is a door they route around. Three such logs were found at `0664` on this machine | measured dir `0o775`, file `0o664` with real stack output | **RAN** |
| **A5** | ✅ **CLOSED (Phase 0), found while fixing A0.** `cmd_install` had no `caps.conda_binary is None` guard, though `repair` (`_cli.py:384`) and `clean` (`:628`) both do — so on a machine with no manager detected, `install` reached `probe_env_state(effective, None)` and the person got `TypeError: expected str, bytes or os.PathLike object, not NoneType` instead of the one sentence naming the fix. **Fixed:** the same `UsageError` its two siblings raise | **READ** |
| **A3** | A new build test reaches its phase by cloning `https://example.com/a.git` — a network call in a unit test. `tests/test_envs_builds.py` (`test_run_build_actually_REACHES_a_phase`) | `repo_url="https://example.com/a.git"` | **RAN** |

### B. Documentation written that day which is false

These matter more than ordinary drift: **B1 and B5 ship into the user's config
directory**, and B2 is a rule stated as absolute in the contract that owns it.

| id | what | evidence | mark |
|---|---|---|---|
| **B1** | ✅ **CLOSED (Phase 3b).** **"The four files `molbuilder.json` cannot name"** — it is **three**. `google_client_secret` has a *default* home that `auth.providers[].client_secret_file` overrides, and `oauth.py:104-110` reads whatever the config says. Asserted in four places, one of them the seeded `secrets/README`: `configuration.md` § 3.1, § 3's table row, `installation.md`'s tree, `initconfig.py:391` **Fixed in all four places, the seeded README included**, and `google_client_secret` is now described as what it is: the DEFAULT home, which `auth.providers[].client_secret_file` overrides | a config naming `/elsewhere/not-the-fixed-home` is accepted and the path kept | **RAN** |
| **B2** | ✅ **CLOSED (Phase 3b).** § 2.3 said two shapes sit outside the one writer and **"nothing else may"**, and that **"every secret this package writes went through"** `write_secret_file`. Both false: `web/auth.py:524-530` creates the session key itself with `os.open(..., O_EXCL, 0o600)` — and that is the *normal* first-run path. Three further whole-file writers of files these documents map are unnamed: the stacks log (A2), the pidfile (`serve_daemon.py:306`), and the two seeded READMEs (`initconfig.py:384`) **Fixed by naming them, which is what makes an exception a rule**: the session key's `O_EXCL` first creation (the operation is *create if absent*, and replacing it signs everyone out), the pidfile (an address, rewritten every start), the two seeded READMEs (written once, never overwritten, no credential), and the monitor's writes (it ships to the compute node and may import nothing of ours). The stacks log is not on the list — it was a defect and now goes through `open_private` | patched `persist.write_bytes`; `_install_secret_key` never called it | **RAN** |
| **B3** | ✅ **CLOSED (Phase 3a).** § 3.1 states the config root is `0700`. `config_dir()` never creates it; `runtime_config.write_config_scope` creates it with no mode (`runtime_config.py:2185`) and `initconfig._ensure_root` never tightens one that already exists. **Fixed at creation and reported on arrival**, which is the split the two halves needed: `write_config_scope` creates it `0700` through the one creator, and a root that ARRIVED loose is a `doctor` finding rather than something a writer silently re-modes — on a cluster `XDG_CONFIG_HOME=/scratch/$USER` is the operator's decision, not ours to overrule | against a non-existent root: dir `0o775` | **RAN** |
| **B4** | ✅ **CLOSED (Phase 3b).** § 0's ownership row claimed this page owns "the mode and the durability of **every** file listed here". Several listed files have no stated mode and are created with none: `environment.json` (0644 via `write_json`), `reports/`, the runtime root, `serve-<port>.pid`; the two READMEs measure `0664` and are written non-atomically. **Fixed by making the claim true rather than softening it**: `placement.py` has a row per file, and one with no mode requirement says `mode=None` — a statement instead of an absence | — | **RAN** |
| **B5** | ✅ **CLOSED (Phase 3b).** The seeded `secrets/README` said the web UI **and** `molbuilder notify-token` "both write this file" of the `notify` channels. `notify-token` writes only `notify_keys` and *prints* the cluster-side JSON (`cli.py:1975`). `getting-started.md`'s table, written the same day, says it correctly — so the two disagreed and the wrong one was in the user's config directory. **Fixed:** the README now says the web UI writes it, and that `notify-token` is for the other side — it writes `notify_keys` and PRINTS the JSON to paste on the machine that runs the job | — | **RAN** |
| **B6** | ✅ **CLOSED (Phase 6), the half that matters.** The wrong DELETION DATES are bookkeeping and were left alone.  The two "*is* enforced" claims were not: `conventions.md` said the doc-structure rules "ride the same gate" and that the docs provenance header "*is* enforced, by `test_docs_structure.py`" -- in the very paragraph drawing the advisory-vs-enforced contrast, and three paragraphs above its own correct note that the file was retired.  A reader checking whether their header is guarded got "yes".  Both now say review, and name the commit.  `conventions.md`'s new note says three test files were deleted on 2026-09-10; two of them (`test_cli_run.py`, `test_cli_siesta_stages.py`) went on **2026-08-11** in `c773d5ac`, for a different reason. The same note claims §§ 1-2 held docs-structure rules; they do not — what they do hold, unswept, are two now-false "*is* enforced" claims (lines 39, 60) | `git log --diff-filter=D` | **RAN** |
| **B8** | Two further live `paths.logs` restatements beyond D5: `config_dir.py:113-118` tells the reader *"`molbuilder.json`'s `paths` block names this directory, and `molbuilder.runtime_config.logs_dir` is where that override is applied"* — advice that bricks a config, citing a function that **does not exist** (grep: no definition) — and `envs/_cli.py:41`. The first is in the module that owns the directory | `hasattr(runtime_config,'logs_dir')` → False | **RAN** |
| **B7** | ⚠ **MOSTLY NOT A DEFECT (Phase 3b).** The substance — *the mode is right before there is anything to read* — is still exactly what happens: `write_secret_file` goes through `mkstemp`, which creates the temp `0600` before it has a name. Only ONE site described an implementation that no longer exists (`configuration.md` § 2.1b, naming `os.open` + `fchmod`), and that is corrected because it sends a reader looking for code that is not there. The rest are true as written and were left alone. **Original claim:** § 2.3 retires the "mode on the descriptor before the first byte" mechanism, and three live places still teach it: `configuration.md` § 2.1b (same document), `run-reports.md:270`, `runtime_config.py:1292`, `notify_setup.py:126`. Related: § 2.3's own door for `molbuilder.json` — `write_config_scope` — is the "chmod afterwards" shape it disparages (`write_bytes` → 0644 → `os.chmod` 0600) | `write_bytes` on a new target lands `0644` | **RAN** |

### C. An instruction that was not implemented

> **The user, 2026-09-12:** *"it is hard to gauge because, a, the package might
> change depends on when you start to install, and, b, when you try to compile
> the downloaded size and everything to be bigger than the actual installation.
> So I wouldn't really bother that. We just remind user that you need to make
> sure you have enough free space... that's not really our job to get it."*

| id | what | evidence | mark |
|---|---|---|---|
| **C1** | ✅ **CLOSED (Phase 6).** `check_disk`'s docstring said **"Never an error, and there is no threshold"** and the code below it is `if free >= suggested: return free, None` — a threshold. The message lands in `warnings` (`builds.py:1079`), which `_build_callbacks` turns into **"Proceed despite warnings?"** — so the hard 30 GB gate was replaced by a derived gate. The derivation is inflated (reports 6.5 GB where `du` says 3.2, then ×2 → warns below ~13 GB for an env costing 3.2) and costs a full filesystem walk of every env. **Fixed:** it reports either way -- silence above a number IS a threshold, whatever the docstring says -- and the reminder goes to `info`, so it is a line to read rather than a "Proceed despite warnings?" prompt. The reference stays, because it is what you asked for: this machine's own largest env, doubled, labelled as a scale and not a requirement | docstring vs `builds.py:679-718`; walk measured 2.8 s warm for 6 envs | **RAN** |
| **C2** | ✅ **CLOSED (Phase 6).** The deleted constant **survived as recipe data**: `recipes.py:1966` declares `"~30 GB free disk space under $CONDA_PREFIX"` as a GPU system precondition, and `tests/test_envs_siesta_gpu_recipe.py:380` **asserts** a disk figure is present. Mitigated only by `system_preconditions` having no renderer. **Fixed:** the precondition says what the space is FOR -- the clone, the build tree, a parallel compile -- and points at what `envs install` reports, with no figure. The test asserts disk is mentioned, which is still true | grep: no consumer | **RAN** |

### D. Sweeps that stopped at the first instance

This is the session's dominant failure shape: **the instance was fixed and the
sibling left**, four times with the rule written down beside the code that does
not follow it.

| id | what | evidence | mark |
|---|---|---|---|
| **D1** | ✅ **CLOSED (Phase 4).** **`envs init-config` answered an unwritable config root with a raw `PermissionError` traceback** — while `seeding_blockers()`, added that day, holds the exact sentence for it, and `bootstrap` prints `init-config` as the remedy. Same *"a remedy the program prints that it then refuses to run"* class the session claims to have closed. **Fixed** exactly so, and the surface turns it into a `ClickException` at the one place `init_config` is called — `bootstrap` reaches it through there too, and its own handler then records the failure and carries on, which is what a forty-minute install needs | traceback at `initconfig.py:324` | **RAN** |
| **D2** | ✅ **CLOSED (Phase 1a) by USE, not deletion.** `run_build_spec`'s `conda_binary` was a **required** keyword argument with **zero** references in its body; `install.py:1257` still passes it and three tests pin it with `conda_binary="/bin/false"  # would fail if called`. Finishing `5ef047a0` is what A-list item A1 of the previous round was about — a parameter kept "for the signature" is how the TypeError survived three commits **Fixed:** `_run_build_phase` now needs it — a build phase enters the env through the manager like every other step — so it is threaded down and used. The three tests passing `/bin/false  # would fail if called` keep their point: it is a canary now rather than a pin on a dead parameter | AST scan of the function body: 0 uses | **RAN** |
| **D3** | ✅ **CLOSED (Phase 4).** `--clean`'s env removal was a bare `subprocess.run` (`_cli.py:1361-1382`), outside `run_step` and `conda_argv`, against `env-framework.md` § 5 (*"Every step goes through `run_step`"*, *"One place writes each command line"*). The session **promoted it from one recipe to five** and `installation.md:519` now calls it "the one door". **Fixed** exactly so. Two things fell out that say the change is real: `envs/_cli.py` no longer imports `subprocess` **at all** — that wipe was the last thing the surface dispatched itself — and the printed manual remedy is now the step's own argv, so it cannot differ from what was attempted. Three tests had been spying on `_cli.subprocess.run`; a spy on a door nothing uses is a test that passes for nothing, so they now watch `run_step` | — | **READ** |
| **D4** | The printed-remedy sweep stopped within four lines of itself: `_cli.py:913` bypasses `_fix_cmd`; **`_cli.py:1234` prints the *recipe* name where every neighbouring line uses `effective`** — so with `MOLBUILDER_HOST_ENV` (made live that day) or an `envs.<category>` override the hard stop names an env that does not exist; `install.py:851` prints a literal `conda`; `recipes.py:1939` hand-copies `_fix_cmd`'s output because the speller lives in the surface. **Fixed across Phase 0 and 1b, as prescribed:** `_cli.py:1234` prints `effective`; `EnvState` and `EnvReport` carry the detected manager so `remove_env_cmd` is one home for the removal line; and the speller moved to **`envs/hints.py`** (floor 1) so `recipes.py` imports `fix_cmd` instead of hand-copying its output — the copy being how that line came to name `siesta-gpu`, which `recipe_by_name` rejects. The four remaining launcher literals in `_cli.py`, two of them written in Phase 0, now read `hints.LAUNCHER` | `_cli.py:1234` vs `:1207` | **RAN** |
| **D5** | ✅ **CLOSED (Phase 6).** A **fourth** `paths.logs` restatement at `serve_daemon.py:57` — *"So `paths.logs` moves molbuilder's application logs"* — for a key that is refused. The commit said *"Three places said it, one was right."* There were four, and the fourth is in a file that commit edited | — | **RAN** |
| **D6** | ✅ **CLOSED (Phase 6).** `machine_config_warnings()` was created as the one home for the pair, and `jobset/_cli.py:182-193` — the loop its docstring calls *"the shape being named rather than invented"* — still spells the pair itself. So the function has two callers and the loop it was extracted from is not one of them; a third warning added later would reach `serve` and not the jobset verbs. See also **D12** | — | **READ** |
| **D7** | ✅ **CLOSED (Phase 4).** **`bootstrap` was a lossy copy of `install`'s orchestration.** The callbacks were extracted, but `run_build_spec` invokes `on_warnings` only `if report.warnings` (`builds.py:1706`), so on a clean preflight **nothing is asked** — while `--include-source-builds`' help still promises *"the user is asked to confirm before each source build starts unless `--yes` is also given"*. Bootstrap also never calls `format_install_summary` and skips the env-state probe + ORPHAN/GHOST/BROKEN hard stop `install` runs. **Fixed** exactly so: `_install_one(recipe, effective, caps, …)` is the one orchestration door and both verbs call it. It RAISES rather than exiting, because the two answer differently — a refusal ends `install` with exit 2 and is one line of a `bootstrap` report that still has four envs to build. Tested where it mattered most: handed a directory the registry does not know about, `bootstrap` now hard-stops instead of driving a `conda create` at it, asserted against a manager that writes down what it was asked | `builds.py:1706`; help at `_cli.py:1532` | **READ** |
| **D8** | ✅ **CLOSED (Phase 4).** `_cli.py:1411` passed an env **name** to `probe_toolchain(env_prefix)`, which documents and requires an absolute `$CONDA_PREFIX` (`builds.py:576-589`). So the install summary shown immediately before a source build's `Proceed?` reported gcc / OpenMPI / CUDA as undetected on a healthy env. **Fixed** with `caps.env_prefix(effective)`, which Phase 2 made free. Measured on this machine: `probe_toolchain("molbuilder-siesta")` reports OpenMPI **absent**, `probe_toolchain(<its prefix>)` reports **5.0.10** | — | **RAN** |
| **D11** | ✅ **CLOSED (Phase 3a).** `runtime_config.write_config_scope` was the one in-package caller that should pass the new `mode=` and did not: `write_bytes` widens the temp to `0644` **with the content in it**, then `os.chmod` fixes it up — the exact sequence `write_bytes`' own docstring says `mode=` exists to make impossible. Its docstring already claims the property ("at mode 0600"). Its `mkdir` at `:2188` also ignores `CONFIG_DIR_MODE`. **Latent, not live:** grep finds no production caller. **Fixed** exactly so, and its `mkdir` now honours `CONFIG_DIR_MODE`. The test drives it as the FIRST writer deliberately: when `init-config` has already made the file `0600`, a rewrite preserves that mode and the defect is invisible — which is how the first version of that test passed against the mutant | `os.chmod` spy: `requested=0o644` then `requested=0o600` | **RAN** |
| **D12** | ✅ **CLOSED (Phase 6).** `auth-setup` is the second surface `machine_config_warnings`' docstring names as a non-caller, and it still is one — so the command that writes provider credentials and `client_secret_file` paths into `molbuilder.json` says nothing when that file arrived `0644`, while printing `Wrote … (mode 0600)`. Same file, same rule: `notify-token --keys-file` was removed that day because *"a flag naming another was a way to write a key nowhere that works"*; **`auth-setup --output` is that flag** for the machine config, and the clobber guard at `cli.py:1605` advertises it as the way past | — | **READ** |
| **D13** | ✅ **CLOSED (Phase 6).** The `MOLBUILDER_HOST_ENV` fix **held only while the variable was exported.** The shim never records it, so: install with it set, later `conda activate mb-dev` and run `python -m molbuilder envs …` → the host recipe reports against `molbuilder` again, and `install molbuilder` from there creates the second host env the fix was for. The routed half has a persistent home (`envs.<category>`); the host half has none — and `"envs": {"host": "mb-dev"}` **validates and is silently ignored**, because `_effective_name` consults `env_for_category` only when `category is not None`. The seeded `_envs` comment advertises that block for env names. **Fixed:** the host is an ordinary category whose key is `envs.host`, so the config is its home and the variable overrides for one invocation — the same rule as every other env | `envs list` with and without the variable | **RAN** |
| **D14** | ✅ **CLOSED (Phase 3b).** A **second** private writer survived twenty lines below the one that was unified: `auth_setup._write_0600` (`:358-367`) is still `os.open(O_TRUNC, 0o600)` + `chmod`, and `emit_molbuilder_json` hand-rolls a `.new.<pid>` temp + `os.replace` instead of the shared writer — and that name lacks `mkstemp`'s `O_EXCL`. The next change to how a private file is written would have reached one of the two. **Fixed:** `_write_0600` is gone (it had exactly one caller) and the staging temp is `mkstemp`'s — unique by construction where `.new.<pid>` is only probably unique, and `0600` before it has a name. The `chmod` after `os.replace` went too: `os.replace` carries the inode, so it could never change anything | — | **READ** |
| **D9** | A third spelling of the tty predicate: `envs/_cli.py:1899` duplicates `cli._stdin_is_a_terminal` citing A7. A7 forbids depending on the surface, not a floor-1 home. `jobset/ask.py:346` does the bare unguarded `sys.stdin.isatty()` the guarded form exists to prevent | — | **READ** |
| **D10** | ✅ **PARTLY CLOSED (Phase 6)** — the contract entry exists now (`configuration.md` § 2.1c), and the asymmetry is gone with D13: the host has `envs.host` like every other env. The third spelling in `scripts/capture-readme-screenshots.py` and `_effective_name`'s home remain. **Original:** `MOLBUILDER_HOST_ENV` became a Python-level override with **no contract entry** (`configuration.md` § 2.1c owns the other `MOLBUILDER_*` variables) and a third spelling in `scripts/capture-readme-screenshots.py:121`. The host is now the one env whose name is overridable only by a variable and not by `envs.<category>`, an asymmetry nothing records. Also `_effective_name` lives in `doctor.py` while `env_for_category` lives on `diagnostics.Capabilities`, which is the home for "what env does this recipe mean" | — | **READ** |

### E. Pre-existing, found during the audit — not caused that day

| id | what | evidence | mark |
|---|---|---|---|
| **E1** | Two suite failures, **cause identified and not this session's**: both come from `3aaec645` (2026-09-11), which added `PySCFConfig.engine` at `config/pyscf.py:245`. Today's range touches none of `config/pyscf.py`, `template.py`, `pyscf/`, or either test. **`test_pyscf_has_no_vocabulary_gaps_left`** — the field's `engine_key` is parenthesised (`'(molbuilder: selects the deck composer + the backend env)'`) so it reads as a *note*, not a keyword, and the field declares no `metadata['item_kind']`; the error names the fix. **`test_every_shown_parameter_changes_the_deck_or_is_openly_pending`** — the field has `choices=("pyscf",)`, a one-option selector, so `_probe_value` cannot produce a value distinct from the default and the anti-silent-skip assertion fires; the fix belongs in the test's `_PROBES`/`STILL_OPEN`, not in the deck, since a one-choice field cannot change a deck by construction | `git log` over the range for those paths is empty | **RAN** |
| **E2** | `configuration.md` § 8's drift row says `jobset probe`'s `--set` / scheduler flag surface "is not built". **It is**: `probe --help` lists `--set KEY=VALUE` and `--scheduler`, `_cli.py:3484-3503` passes them to `resolve_environment(overrides=…)`, and the `environments/README` seeded that day teaches the flag. **The row should be closed** | `probe --help` | **RAN** |
| **E3** | `generator.md` § 6.1 and `script-preparation.md` say the wrapper reads the rendered deck for the **rank count**. `runwrap.py:2231` says a rank count comes from a record "and nowhere else"; the deck is read for the GPU keyword and an advisory notice. Contradicts `architecture.md` § 9.2 | — | **READ** |
| **E4** | `job-system.md`'s `resources` example lists **7** keys and a retained sentence says "seven"; `Resources` has **15** and `to_dict()` returns all 15. A comment added that day states 15 beside the wrong block instead of correcting it | 15 re-derived | **RAN** |
| **E5** | ✅ **CLOSED (Phase 1b).** `envs install --help` ended "**Source-build recipes only.**" for `--clean`, the opposite of both the behaviour and `installation.md`. **Fixed** in the same pass as M3: the text now says every recipe, names the **detected** manager rather than `conda`, and states the refusal for the running env | `envs install --help` | **RAN** |
| **E6** | ✅ **CLOSED (Phase 6).** Three of the four surfaces already said "three"; the survivor was `_seed_secrets_dir`'s own docstring, wrong in the count AND the membership (it named `google_client_secret`, which the README it writes explicitly excludes).  `envs init-config --help` says "the **two** secrets that cannot live here"; the README it describes says four (and the true number is three — B1) | — | **READ** |
| **E7** | ✅ **CLOSED (Phase 5).** `_render_validation` excluded advisories from the verdict but not from `n_pass`, so one real failure beside an absent MPS printed `4/6 checks passed` — two rules for one question. **Fixed:** the count is over the REQUIRED probes, which is the verdict's own rule, and the advisories are reported beside it | — | **READ** |
| **E8** | ✅ **CLOSED (Phase 6).** Both halves.  `docs/README.md:5`/`:128` and its floor row; and the five rows of `architecture.md` § 7 -- A1, A4, A7, A8, A11 -- under a heading that reads *"a rule nobody checks is a wish"*.  The PROPERTY each row states is kept verbatim (it is what a checker would assert, and what a reviewer reads the diff for); what is corrected is the claim that something asserts it today.  Also the § 8.2 citation and the § 2.1 floor table.  Whether those five get a checker back is F1, and yours.  `docs/README.md:5`/`:128` still say the doc-structure rules are "Enforced by `tests/test_docs_structure.py`", and ~14 rows in `architecture.md` (§ 2.1, A1/A4/A7/A8/A11, § 8.2) name `tests/test_architecture_rules.py`. Both files were deleted 2026-09-10 in a deliberate sweep, so **those rules are held by review alone** and the documents claim otherwise. A1/A4/A7/A8/A11 are "the rules that must never break" | `git log --diff-filter=D` | **RAN** |

### F. Needs the user, not more scanning

| id | question |
|---|---|
| **F1** | **E8 is a policy call.** Either the 2026-09-10 sweep was right and `architecture.md` § 7's "checked by" column must say *review* for A1/A4/A7/A8/A11 (and its opening *"a rule nobody checks is a wish"* revisited), or those five rules need their checker back. Not a defect — a decision. |
| **F2** | A PyPI-sourced `pyscf-properties` sat in the **host** env from 2026-09-11 21:39. It is the measurement `recipes.py` and `installation.md` § 3.1 cite for why `force=True` is needed (*"the tree stayed PyPI's and `infrared` stayed absent"*). It was removed on 2026-09-12, along with the two empty directories pip left, which were what made `import pyscf` succeed in that env. **Restore it if the artifact should stand in place.** The pySCF env itself was never touched and is correct: `direct_url.json` records git commit `4eee5a43` and `pyscf.prop.infrared` imports. |
| **F3** | ✅ **CLOSED (Phase 0) — and the diagnosis this row carried was wrong.** Measured: **the suite runs to completion.** A full `tests/ --ignore-glob=*_e2e.py` run executed **9082 tests — 4 failed, 9074 passed, 4 skipped, 45:37**; the four are F3a's, all pre-existing. `pytest-randomly` **is not installed**, so order was already deterministic and this row's `-p no:randomly` instruction was a no-op. The 4947 figure came from a **clobbered progress file**: `.test-progress/none2e.jsonl` had no `start` and no `collected` record, began at the 203rd of 392 test files in sorted order and ran sorted to the last — the TAIL of a complete run, written into a file truncated under it. `progress_plugin.pytest_configure` did `open(path,"w")` on every session, so a second pytest aimed at that file wiped a live run's records (`testrun.py`'s flock only guards `testrun.py run <same batch>`), and `_summarise` counted what survived and called it `done`. **Fixed:** the plugin truncates only a file whose previous run reached `done` and appends a new generation otherwise, every record carries a run id, and the reader has four states that are NOT results — `unusable`, `interleaved`, `partial`, `abandoned` — with `status` exiting non-zero on each. 15 tests in `tests/test_testrun_reports_only_what_it_measured.py`, every rule mutation-verified. It caught its own first discrepancy on the spot: 15 passed, 14 records, because one of those tests borrowed the plugin's module state and returned it in a teardown that runs after its own report | **RAN** |
| **F3a** | The four failures of that complete run, **all pre-existing** (see E1 for the 2026-09-11 cause): `test_pyscf_convergence_knobs.py::test_memory_is_one_item_across_both_engines`, `test_template_declarations.py::test_pyscf_has_no_vocabulary_gaps_left`, `test_vibration_form_honesty.py::test_every_shown_parameter_changes_the_deck_or_is_openly_pending` — one root cause, the `engine` field's missing `metadata['item_kind']`; and `test_results_blueprint.py::TestPartialSpectraInspectorEndpoint::test_partial_has_no_undocumented_ids` — six undocumented DOM ids (`display-floor`, `methods-block`, …), a separate and unexamined issue. |

### H. Residue of the pre-state-machine design — the installer

**This is the section § 2 was missing.** A–F came from auditing the 2026-09-12
diff; these came from sweeping `molbuilder/envs/`, the shim and `diagnostics.py`
against T1–T8 with no reference to any recent change. They are **not** today's
defects — they are what the migration left behind, which is why the T column
matters more than the date.

| id | breaks | what | mark |
|---|---|---|---|
| **H1** | **T1, T4** | ✅ **CLOSED (Phase 0).** ⚠ **A healthy env was labelled `GHOST`, and the program then tells the operator to delete it.** `probe_env_state` sets `dir_exists` from `info.envs_dirs/<name>` **only** (`install.py:900-909`) and never tests the prefix the registry just handed it — while the very same object carries that prefix (`prefix = prefix_from_registry or prefix_from_fs`, `:914`). So an env whose parent is not an `envs/` directory (`conda create -p /scratch/...`) reports `GHOST` with a correct prefix inside it, `install` hard-stops at `_cli.py:1209`, and `describe()` prints *"the directory is gone. Fix manually with: `conda env remove -n <name> -y`"*. `_env_prefix` resolves the same env correctly through four tiers (`:654-746`) — two rules for one question, which is the § 5 lesson still live inside the probe. § 2.1 defines GHOST as *"a registry entry with no directory"*, which is not what the code measured. **Fixed:** the directory and `conda-meta/` checks are taken on the prefix the registry named; the `envs_dirs` search answers only the opposite question (a directory no registry entry mentions — ORPHAN, BROKEN) and is consulted only then. One subprocess instead of two, because the prefix is no longer derived at all. § 2.1 now states which path *"the directory"* means. Both halves tested, both mutation-verified | **RAN** |
| **H2** | **T1** | ✅ **CLOSED (Phase 2).** **Three readers of the conda registry gave three answers**, and one of them is a second mechanism for the question `EnvState` owns. `diagnostics._list_conda_envs` (`:272-302`) filters on the parent directory being literally named `envs`; `install._env_prefix` resolves four ways; `probe_env_state` a fifth. `caps.env_available()` — not `EnvState` — is what gates `repair`, `clean`, `validate`, `bootstrap --skip-existing`, the `--clean` pre-check and `doctor`'s `present`. For the env in H1: `repair` said *"env does not exist. Install it first"*, `install` resolved its prefix, `doctor` said `MISSING`. **Fixed:** `diagnostics.conda_env_prefixes` is the one reader and answers `{name: prefix}`; the installation root is dropped by a relationship in the same document instead of the `envs/`-parent filter that also hid every `--prefix` env; `Capabilities.conda_envs` *is* that mapping, so the six gates and the state machine now answer from one reading. Measured on this machine: six envs with prefixes, root excluded | **RAN** |
| **H3** | **T4, T5** | ✅ **CLOSED (Phase 1a).** **Two hand-written copies of the `conda run` bypass, drifted four ways** — `install._bypass_conda_run` (`:53-157`) and `builds._run_build_phase`'s wrapper (`:1550-1648`). The consequential divergence: **`run_step` never sanitises the environment.** `run_streaming(run_argv, env=env, …)` takes `run_step`'s `env` parameter, which **no caller passes** (`install.py:383`), while `builds.py` passes `build_subprocess_env()` at two sites to strip `CPATH`/`CFLAGS`/`LIBRARY_PATH`/`CUDA_HOME`/`OMPI_*`. So every pip step and every `extra_steps` dispatch runs with exactly the host leakage `builds.py` exists to prevent. § 5.5 exempts builds' *executor* (sentinel resume), not a second copy of the rewrite. **Fixed:** one door, `builds.dispatch_into_env` — the manager's own `run` is the route, the wrapper is the measured fallback, and environment policy (host-leak stripping, TMPDIR, the pip cache) is a Python dict both paths share so they cannot drift. Three callers now enter an env identically, including the tool router that never had the workaround (**S17**). Measured: `envs doctor` verifies four envs with no generated shell at all | **RAN** |
| **H4** | **T3, T5** | ✅ **CLOSED (Phase 4).** **Two `InstallStep`s were hand-built in the planner**, bypassing one-translator-per-kind: the batched plain-pip step (`install.py:604-609`) and the `extra` step (`:613-617`). § 4.2's pseudocode named `pip_steps_for` and `extra_steps_for` — **neither existed**. **Fixed:** both are written, the planner calls them, and § 4.2's pseudocode is now true of the code rather than aspirational | **READ** |
| **H5** | **T1, T6** | ✅ **CLOSED (Phase 5).** **Two vocabularies for the five outcomes inside one run's output.** `_OUTCOME_WORD` (`install.py:1024`) prints `UNAVAILABLE -- optional, continuing` live; the CLI recap prints `(degraded)` for the same step (`_cli.py:1478`, `step.outcome.value`). `RECOVERED` is *"OK via the declared alternative"* live and `recovered` in the recap. `_OUTCOME_WORD`'s own comment claimed it was the ONE mapping. **Fixed:** `Outcome.word` is that mapping, on the state, and `Outcome.note` carries the *why* for the line with room for it. `_OUTCOME_WORD` is gone and nothing reads `.value` for display | **READ** |
| **H6** | **T3** | ✅ **CLOSED (Phase 2).** **The CLI re-derived the create decision the state machine owns, and paid for a second live probe.** `cmd_install` (`_cli.py:1200-1242`) re-implements the wreckage branch and the resume branch — the latter as a **string compare**, `state.state_label == "PRESENT"`, where `state.can_resume` is the accessor — then `run_install` probes again inside `_create_decision`. Instrumented: the same two JSON documents were read twice back to back per install, three times from the CLI. **Fixed:** the CLI hands `run_install` the state it probed (`env_state=`), reads `state.can_resume` instead of comparing the label, and **drops** the reading after `--clean` has removed the env — a stale `PRESENT` there would skip the `conda create` that must run, which is the 2026-06-15 regression from the other side and has its own test. `_env_prefix` consults the snapshot first, so `doctor` stops paying a 1.2 s registry read per recipe | **RAN** |
| **H7** | **T1** | ✅ **CLOSED (Phase 5).** **`EnvState`'s five states were a display string, branched on with `==` in four places** (`install.py:787-815`, `:820`, `:825`, `:829-857`, `_cli.py:1212`, `:1239`). This is the shape `StepRole` was introduced to remove — its own comment: *"renaming a label for clarity silently disabled the create-skip."* A rename of `"PRESENT"` silently turned `can_resume` False, which the code calls *"the worst answer for an env that is already wreckage"*. **Fixed:** `EnvPresence` is the enum, `state` computes it once, and `state_label` is derived from it — so a rename is now a loud error rather than a silent False, which the mutant confirms | **READ** |
| **H8** | ✅ **CLOSED (Phase 6).** Third and last: `_run_steps(skip_create_if_present=…)`.  Two call sites -- one passed `True`, the other runs only VERIFY steps, which can never be CREATE -- so the `False` branch in `_create_decision` was unreachable in production.  Both parameters deleted. | **Dead parameters, returns and branches**, each established against every call site in `molbuilder/` **and** `tests/`. ✅ **Two closed (Phase 1a):** `run_step(env=…)`, never passed and the hole in H3 — deleted, and the door computes the step's environment instead; and `_bypass_conda_run`'s always-`{}` second return value, gone with the function, its *"preserve the existing caller contract"* having had no party to it. **Still open:** `_run_steps(skip_create_if_present=False)` never passed, its docstring saying *"set False only in tests"* and no test doing so; `validate_recipe(quiet=…)` never passed; `_cli.py:431-440`'s `base` kind-stripping a no-op branch; `Outcome.decide(first_attempt=True)` ignored whenever `ok=False`; `envs/__init__.py:21-22` re-exporting `subprocess`/`shutil` as *"back-compat"* for six test monkeypatches and no product code — against the repo's no-back-compat rule | **READ** |
| **H13** | **T14** | ✅ **CLOSED (Phase 1a).** **`<mgr> run -n <name>` cannot address an env that lives outside `envs_dirs`** — conda's own `locate_prefix_by_name` (`conda/base/context.py:2239-2258`) searches `envs_dirs` and raises `EnvironmentNameNotFound` otherwise. So the M1 primary form has to address an env by the **prefix the registry gave us** (`run --prefix <prefix>`), which is what `builds.py` already does while `install.py` uses `-n`; **Fixed:** the door re-addresses a name-spelled argv at the prefix it was handed (`addressed_by_prefix`), and a caller that already has a prefix spells it directly (`conda_run_prefix_argv`) instead of deriving a name from the directory. One place still writes the command line | **READ** (conda source) |
| **H9** | **T5** | ✅ **CLOSED (Phase 1a, in passing).** **A fourth output-trim spelling**: `builds.py:1641` `combined[-4096:]`, a bare literal duplicating `install.OUTPUT_LIMIT` — in the module whose results are adapted into `InstallStep`s, so a build step's `output` obeyed a different constant from every other step's. **Fixed:** `OUTPUT_LIMIT` is defined in `builds.py` — the layer that produces the output — and `install.py` re-exports the one value | **READ** |
| **H10** | ✅ **CLOSED (Phase 6).** Both sites deleted -- `install._env_prefix`'s derived strategy and the shim's `_resolve_env_python` strategy 3.  The shim's two callers also each spelled the resolver's failure message, and had drifted: one still advertised the deleted strategy, the other led its remedy with `env remove -n <host env> -y`, which destroys a healthy env whenever the MANAGER is what failed (M5).  One reporter now, and it prints the LOOK first.  **T14** | ⚠ **ONE OF THREE CLOSED (Phase 1b).** **Three spellings of "derive the install root from the manager binary"**: `install.py:734-742`, `install-env.sh:721-725`, `initconfig.py:99`. The third was the loosest (`/usr/bin/conda` → `/usr`) and fed the activation preamble **written into the user's config** — so it is the one that was fixed: `initconfig.conda_hook` asks `<mgr> info --json` for `root_prefix`, and returns `None` rather than a guess when the manager will not say. The other two are last resorts behind a tier that asks the manager, so they mislead rather than decide; **Phase 6**. M1-M5 is the rule that covers all three now | **READ** |
| **H11** | **T6** | ✅ **CLOSED (Phase 5).** `validate`'s live marker and its table disagreed: `validate.py:610` streams `FAIL` for an advisory probe, `_cli.py:860` then prints `[NOTE]`. The CLI's own comment forbids exactly this — *"printing FAIL beside a verdict that ignores it is two statements about one fact"*. Distinct from **E7**, which is about the count. **Fixed:** `ProbeResult.tag` is the one word, read by both | **READ** |
| **H12** | ✅ **CLOSED (Phase 6).** The `skip_create_if_present` half was already fixed; the reversed `doctor` dependency was not.  — | **Obsolete statements of the old design sitting on the new one**: `install.py:19` still says verify *"re-uses `molbuilder.envs.doctor`"* — the dependency reversed at the migration (`doctor.py:489` imports from `install`); and `skip_create_if_present`'s docstring (`:1162-1168`) promises a skip *"reported as a no-op (returncode 0…)"*, which § 5.1 names as the defect that was fixed | **READ** |

**The shape of H1–H12, stated once:** the migration built the new doors and left
the old derivations standing beside them. Four items are one question answered in
two or more places (H1, H2, H3, H6), three are a string where the design says
state (H5, H7, H11), and the rest are what those leave behind. **H1 and H3 have
real consequences today** — a delete-this recommendation for a working env, and
every pip step running with host toolchain leakage.

### I. Residue of the pre-consolidation design — config and secrets

Swept against T9–T13, independently of any diff. **A11 is the rule most of these
break, and its own text is part of the problem** — see I4.

| id | breaks | what | mark |
|---|---|---|---|
| **I1** | **T9** | ✅ **CLOSED (Phase 1b).** `jobset probe --write` **climbed a parent chain to the config root and re-assembled the filename by hand**: `target = … machine_scope_path().parent`, `fname = f"{name}.json" if name else FILENAME` (`jobset/_cli.py:3645-3646`). `machine_scope_path()` already *is* `<config dir>/environment.json`; this takes it apart and puts it back, importing the bare `FILENAME` to do so. Two spellings of one path inside one expression — so moving the machine record moved the reader and left this writer behind. **Fixed:** the surface asks for the FILE — `machine_scope_path()` or the new `named_environment_path(name)` — and `--out DIR` takes the filename from the resolver rather than re-typing it. **A4's mode defect is on the same two lines and is deliberately NOT fixed here**: a `mode=` on this one `mkdir` is the instance, and Phase 3's one-creator-plus-checker is the mechanism | **READ** (resolution **RAN**) |
| **I2** | **T9, T12** | ✅ **CLOSED (Phase 1b).** `config_provenance` spelled the calculation-scope record path itself (`runtime_config.py:1416-1419`, `Path(project_dir) / ENV_FILENAME`) — in the one function whose whole job is to tell a reader which file answered. The owner's door is `scheduler/record.calculation_record()`, whose docstring says it exists because *"the join lived at three sites … two of them are places a reader is TOLD a path"*. This is the fourth. **Structural cause:** `scheduler/__init__.py` re-exported `FILENAME` but **not** `calculation_record`, so the façade offered the filename and hid the door. **Fixed at the cause:** the façade now exports `calculation_record` and `named_environment_path`, and the display asks for them | **READ** |
| **I3** | **T9, T12** | ✅ **CLOSED (Phase 1b).** ⚠ **A user-facing remedy hard-coded a path a resolver owns.** `jobset/prep.py:1189` says *"copy the record it writes into `~/.config/molbuilder/environments/` here"*. Measured with `MOLBUILDER_CONFIG_DIR` set: the resolver answers `<that dir>/environments`, the message says `~/.config/...`. So on exactly the machine the variable exists for, following the instruction puts the record where `prep` does not look — and `prep` refuses again with the same message. **This is the identical defect `notify-token --keys-file` was deleted for on the same day**; the sweep missed this site. **Fixed:** the message prints `environments_dir()`. Measured with the variable set: the remedy names `/tmp/mbcfg/environments`, and a test pins both halves (the resolved path present, `~/.config/molbuilder` absent) | **RAN** |
| **I4** | **T9** | ✅ **CLOSED (Phase 1b), rule text first.** `environments_dir()` reached the config root by `.parent` (`scheduler/record.py:803`), which A11 forbids in those words. `config_dir.config_dir() / "environments"` is the one-line form. **And A11's own elaboration licenses the climb**: `architecture.md:895-897` still says *"a per-user config path is `environment.machine_scope_path`'s"*, naming a pre-consolidation owner. **Fixed in that order:** A11's elaboration now names `config_dir.config_dir()` as the root's one owner and states that *a file's resolver is not a way to reach its root*, with the old sentence quoted as what sent two sites there; then `environments_dir()` became `config_dir() / "environments"`. The test that matters is the one the old "do all the doors follow the variable" test could not be: **move the machine record and the directory beside it must not move** | **READ** |
| **I5** | **T10, T11** | ✅ **CLOSED (Phase 3a).** `serve status` wrote the serve log through its own door — bare `mkdir` + `open(log_path(port), "ab")` (`cli.py:2415-2416`) — so when `status` is the first writer the log lands **`0664` in `0775`**, and that file is the measured `client_secret` sink. A later supervisor start tightens both, so the window is "until one runs". Distinct from **A2**, which is the *stacks* log. This is why the § 1 DONE row above is now qualified. **Fixed** with A2, through the same two doors — and `serve-6006.log`, found at `0664` on this machine, is one this defect made | **RAN** |
| **I6** | **T13** | ✅ **CLOSED (Phase 1b), and it was not alone.** `task.py:239-240` stated the **two-thirds** config-dir rule — `$XDG_CONFIG_HOME/molbuilder/notify`, else `~/.config/molbuilder/notify` — omitting `MOLBUILDER_CONFIG_DIR`. `cli.py:2030` documents that exact omission as having silently written a key where the monitor does not look. **Fixed, plus two more found by sweeping for the shape — both in `configuration.md` itself**: § 2.1a's *"the home is the per-user config directory"* paragraph and § 2.1's `environment.json` lookup row, i.e. the owning document stating two thirds of its own rule | **READ** |
| **I7** | **T10** | ✅ **CLOSED (Phase 3b)** — ⚠ **and one of these was NOT "fixed", deliberately.** Whole-file writes outside the one writer, none holding a credential but each able to leave a truncated file: `cli.py:1823` (`runtime-info` JSON that a later parse reads) and `projects.py:457,463` (project READMEs) are local and may be routed. **`monitor.py:1560` may not.** That module SHIPS BESIDE A JOB (§ 0.5): routing its `util.csv` header through `persist.write_bytes` adds a molbuilder import to a file executed by the job's own python in an env with no molbuilder, and the monitor then dies at import with stderr to `/dev/null` — no status, no util.csv, no reports, on every production run. It is an exception to § 2.3 with a hard reason, and the reason belongs in § 2.3 rather than in a later reader's head | **READ** |
| **I8** | ⚠ **MOSTLY CLOSED (Phase 6); one needs your call.** `serve_daemon.log_dir` deleted (zero callers in `molbuilder/` and `tests/`).  `auth_setup.default_secret_dir` / `secret_key_path` / `google_client_secret_path` deleted -- three one-line pass-throughs, i.e. a second public name for a door `config_dir` owns; the three production call sites ask `config_dir` now, and the two tests that kept `default_secret_dir` alive were retired (they asserted `config_dir`'s behaviour THROUGH the alias, and `test_config_dir_has_one_home.py` owns both facts against the real door).  ✅ **`read_effective_config` RESOLVED 2026-09-13, the other way round.** User: *"we try to unify the reading of config files into a set of API that all users should be using... see if you can detect any handcrafted code that should be redirected to use this facility."*  It was not a dead door, it was an UNUSED one with a hand-rolled copy: `get_execution` spelled the two-scope merge itself.  Redirected.  The other four two-scope readers each have a genuinely different rule and stay as they are -- see the note under § 2 I8. ~~OPEN: `read_effective_config`~~ -- no production caller, but exported in `__all__` with 19 references across two test files, so deleting it is a public-API call, not a cleanup.  (The row's claim that `architecture.md` advertises it is stale: the only doc mentions left are an archive file and this plan.) | **Dead and duplicate doors.** `serve_daemon.log_dir()` has **zero** callers anywhere (`:65-71`). `auth_setup.default_secret_dir()` is now `return config_dir()` with no production caller — a third public name for one directory, kept alive by the tests that assert it. `auth_setup.secret_key_path()` / `google_client_secret_path()` are aliases of the `config_dir` doors, and § 3.1 names one spelling while § 2.1e names the other. `runtime_config.read_effective_config` has no production caller while `architecture.md` advertises it as a door | **READ** |
| **I9** | ✅ **CLOSED (Phase 6).** Re-measured with pyflakes rather than trusting the list, and most of it was already fixed: the `auth_setup` chmod-after-`os.replace`, `base64`, `DIRNAME`, `serve_daemon` `sys`, and `runtime_config` `machine_for` were all gone or never unused.  Genuinely open and now fixed: `record` `re`; `web/auth` `request`/`url_for`; `notify.read_keys(path)` -- `app.py` called `read_notify_keys()` path-free and then stringified `notify_keys_path()` into Flask config for `read_keys` to `expanduser` and resolve AGAIN, so `MB_NOTIFY_KEYS_FILE` is gone and the blueprint asks path-free; `monitor._envelope`'s `text`, computed and dropped; `cli` `json`, imported at module scope and re-imported in four bodies.  pyflakes also found one NOT on the list: `runtime_config.get_routing` is annotated `List["Domain"]` with no binding for the name anywhere -- harmless at run time, undefined to every static reader.  Now a `TYPE_CHECKING` import. | **Dead operations and imports.** `auth_setup.py:459` chmods after `os.replace` of a temp already created `0600` — `os.replace` carries the inode, so it can never change anything, and it reads as the loose-window fix-up § 2.3 retires. `notify.py:204`'s `read_keys(path)` stringifies a resolved path into app config, reads it back and `expanduser`s it, three lines after `read_notify_keys()` answers path-free. Unused imports: `auth_setup` `base64` and **`DIRNAME`** (the one module outside `config_dir` importing the root's name component, and it does not use it), `serve_daemon` `sys`, `record` `re`, `runtime_config` `machine_for`, `web/auth` `request`/`url_for` | **RAN** (pyflakes) |
| **I10** | — | `oauth.py:87-91` and `:104-110` each independently `expanduser` and resolve `client_secret_file` — two interpreters of one value, where a disagreement shows as a hot-reload that never fires. No rule covers it (the file is user-named) | **READ** |

**What I1–I4 have in common, and why it is the first thing to fix here:** four
sites reach a path by taking another path apart, and **A11's own elaboration still
describes the pre-consolidation owner**, so a reader following the rule as written
is led to do it. `config_dir` is the root's one home and each file's owner exposes
its own door — **correct A11's text in `architecture.md`, then sweep I1, I2, I4**,
and export `calculation_record` from the `scheduler` façade so the door is as
reachable as the filename.

### G. Checked and clean — do not re-scan these

Recorded so the focused session does not spend itself here. All **RAN** unless noted.

- **`repair` against a `PipPackage`** (`env-framework.md` § 7): honours `source`, `force`, `optional` and `fallback_to_index` from the one record, via `pip_step_for` → `run_step`. `has_validator` is gone with no callers.
- **`get_scheduler`'s empty-block change**: only `{}` changed meaning. `[]`, `"slurm"`, `0`, `False` are still rejected earlier by `_normalise`'s type check, and no caller depended on the old raise (`runwrap.py:4442`, `jobset/submit.py:890`, `jobset/_cli.py:1134`, `web/blueprints/build.py:1535`).
- **The web surfaces**: `notify_setup.py` validates through `monitor.is_channel_name` / `_KINDS` and never builds a route, so the new write-door check covers its issuer too; `web/app.py:372-378` reads with no path, which is what justified removing `--keys-file`; `build.py` needs nothing.
- **`jobset/` and `scheduler/`**: nothing reads `paths.logs`/`run`/`reports`; `write_environment` goes through `persist.write_json`.
- **Other `persist.write_bytes` callers**: `checkpoint._atomic_write_bytes` and `write_json` correctly want the preserve-or-0644 default. The only caller that should pass `mode=` is **D11**.
- **`configuration.md` § 3.1's 14 paths and resolvers**: every one resolves as drawn.
- **From the config/secrets sweep:** no filename literal appears outside its owner as a code-level join (the one `$cfg/notify` construction is shell text for the far machine and carries the full three-branch rule); **no live reader of any retired key** (`secret_key_file`, `notify_keys_file`, `notify_route`, `paths.logs|run|reports`); `notify` has one format owner and one writer, `notify_keys` one reader and one writer shared by CLI and web; `record.write_environment` goes through `persist.write_json`; `write_bytes`'s `mode=` path is correct (mkstemp 0600 → chmod → replace); no secret reaches a log from the notifier path; one admin list with one meaning; no second per-user root anywhere.
- **From the installer residue sweep:** `pip_argv`/`conda_argv`/`conda_run_argv` are the only argv spellers and nothing in `envs/` concatenates a command line; `repair`'s two halves both map back to the record and dispatch through `run_step`; `audit_packages` is disk-only and iterates records for both kinds; there is **no** parallel `optional_*` list anywhere; `PackageAuditIssue` carries no install instruction; `succeeded` is derived from the steps and every recorded step carries an `Outcome`; `InstallStep.accepts` is the single accept rule; `plan_install` is pure; `create_step_for`'s degradation is one fallback, not a search; the shim consumes only `--gcc` and its host arrays match `_HOST`; `abi.py` holds no step-like dispatch.
- **`write_secret_file`'s atomicity and symlink behaviour**: a planted symlink is replaced, not followed; a failed write leaves the previous bytes and mode intact with no temp litter.

### J. Doors audited for hand-rolled parallels *(2026-09-13)*

*(User: "check the other APIs -- if it's not used but was designed as a uniform
door, or not used effectively, that a handcrafted code was used in parallel that
achieves the similar thing." Every door the three contracts name was counted and
every hand-rolled shape grepped: subprocess into an env, registry reads, atomic
writes, mode setting, per-user roots, env removal, warnings, envelopes, raw
return codes. Clean: `run -n` is spelled only in `builds.py`; `env list` has one
reader; no per-user root is computed outside `config_dir`; `read_notify_keys`,
`_envelope`, `emit_molbuilder_json`, `Outcome.decide` have exactly their
intended callers. What was not clean:)*

| | door | what stood beside it | status |
|---|---|---|---|
| **J1** | `builds.dispatch_into_env` -- env-framework § 5.6 / M1 say all three dispatches go through it, naming `_dispatch.run_in_env` as the third | `run_in_env` never calls it: re-spells the whole mamba-1.x fallback (the seen-gate, the stub check, three `subprocess.run`s) and flips the door's PRIVATE state by hand (`_builds._MANAGER_RUN_UNUSABLE["seen"] = True`). Cause: the door returns `(rc, output)`; the router's one caller (`_amber.py`, `tleap`) needs a `CompletedProcess` with `cwd`/`timeout`. A real need answered by a copy instead of by the door growing the shape. | **OPEN -- design.** Factor the DECISION (which argv, given the measured state; whether a result flips it) out of the RUN so both call the same two things. |
| **J2** | `runtime_config.write_config_scope` -- § 2.3's designated writer for `molbuilder.json` / `.molbuilder.json` | **Zero production callers.** `scheduler/probe.py:16` still says `probe --write` merges "via write_config_scope"; N4 moved that output to `environment.json` via `write_environment` and the docstring never followed. | **OPEN -- decision.** Delete (a door nobody sees is where the next copy comes from -- exactly `read_effective_config`'s history), or keep for a writer that does not exist yet. Fix probe.py's docstring either way. |
| **J2a** | `persist.write_json` -- "privacy is a PARAMETER of the one writer" | `write_bytes` had `mode=`; `write_json` never did, so `initconfig.seed_machine_config` did `touch(mode=0o600)` then a plain write, with nine lines of comment explaining the trick. | ✅ **CLOSED.** `write_json(mode=)` threads through; the seed passes `CREDENTIAL_FILE_MODE`. Mutant (drop the passthrough) fails `test_the_directory_and_the_config_are_not_world_readable`. |
| **J3** | one reader per manager document (Z3's pattern: `conda_env_prefixes` for `env list --json`) | `info --json` has THREE readers spelling the subprocess + `json.loads` + try/except each: `install._env_prefix`, `install.probe_env_state`, `initconfig._manager_root`; and the `envs_dirs` orphan walk is spelled twice inside `install.py`. | **OPEN -- small design.** `diagnostics.manager_info(binary) -> dict` beside `conda_env_prefixes`; one `_orphan_prefix` in install. |
| **J4** | `config_dir.ensure_private_dir` -- "the ONE creator" | Two hand-rolled creators in `auth_setup`: `emit_molbuilder_json`'s `mkdir(mode=0o700)` (identical semantics), and `write_secret_file`'s `mkdir` + `os.chmod(parent, 0o700)` -- which RE-MODES THE CONFIG ROOT on every secret write, against the decision `ensure_private_dir`'s docstring records and `write_config_scope` honours. | ✅ **CLOSED.** Both through `ensure_private_dir(parent)`. Measured: a pre-existing 0755 root stays 0755 after a secret write and `placement.findings` reports it; a fresh root is created 0700; the secret is 0600. **Behaviour change**: the wizard no longer chmods your config directory -- `envs doctor` tells you instead. |
| **J5** | `install.remove_env_cmd` -- "The ONE spelling" (M3) | `envs/_cli.py:1373` spelled the line by hand with its own copy of M3's reasoning (and would print `None env remove …` with no manager detected). `builds.py:1236` prints a literal `` `conda env remove` `` inside `preflight`, which has no manager in scope. | ✅ first site **CLOSED** (through the speller). **OPEN**: the `preflight` message -- name the program's own `--clean` route, or hand `preflight` the manager. |
| **J6** | `scheduler/record.FILENAME` -- the one home for `environment.json` | `placement.py:84` re-spelled it as a literal, in code written this session. Found by running the retired `test_architecture_rules.py` against today's tree (62 pass, this the one real failure). | ✅ **CLOSED.** `FILENAME as ENVIRONMENT_FILENAME`. |
| **J7** | the placement table -- the contract for expected modes | `web/blueprints/notify.py:172-173` declares its own `REPORT_MODE = 0o600` / `REPORT_DIR_MODE = 0o700` and `os.chmod`s after creating (the retired fix-up-after shape), while the table's `reports/` row says mode **None**, "no credential". | **OPEN -- which side is right?** If the reports are private, the table row is wrong and the code should say `ensure_private_dir(root, tighten=True)`; if not, the code is enforcing a mode no contract states. |

*Noted, not acted on (out of this migration's scope, but by § 2.3's letter "a
write not on the list is the finding"): ~10 `os.replace` / `mkstemp` sites in
sidecars, `web/blueprints/files.py`, `docs.py`, `watch.py`, `checkpoint.py`;
`envs/_cli.py:127` opens the install log `"w"` at the default mode;
`cli.py:2061` prints the config-dir rule as a bash expression for the far
machine (a second spelling, with a stated reason).*

### K. Static read of the whole set, end to end *(2026-09-13)*

*(User: "this is the time you fully read all code of this whole set and identify
logic, layer and redundancy ... tests are the last net, never the proof."  Every
file in scope was read in full -- `config_dir`, `persist`, `placement`,
`runtime_config`, `diagnostics`, `serve_daemon`, `auth_setup`, `scheduler/record`,
`envs/{builds,install,_dispatch,doctor,hints,recipes,_cli,initconfig,validate,abi}`,
`cli.py`'s config verbs, `web/auth`, the two notify blueprints, `monitor`, and the
shim.  Findings are from the code; where a test is mentioned it is to say whether
the net caught it.  Grouped by kind, ranked within each group.)*

#### K-L. Logic -- the behaviour is wrong today

| | finding | fix shape | decision? |
|---|---|---|---|
| **L1** | ✅ **CLOSED 2026-09-13.**  `StepRole.REMOVE`; the installer voids the snapshot and its prefix after the removal; the surface only asks; the test drives a PRESENT env with a fake manager whose state follows the commands it received, and asserts the steps after the create are addressed at the NEW directory.  Mutants: role back to CREATE fails "the removal was never dispatched"; reset removed fails "addressed at the OLD directory".  **`install --clean` on an env that exists does not remove it.** `remove_step_for` gives the removal `role=StepRole.CREATE` (`install.py:540`) to skip the prefix requirement; `_run_steps` sends every CREATE-role step through `_create_decision` (`:1151-1160`); a PRESENT env answers SKIPPED "already exists; skipping create" (`:1114-1119`).  The removal is skipped, the create is skipped for the same reason, and the run is a plain re-install that prints "remove env X: SKIPPED".  Two more halves: `run_install` resolves the prefix BEFORE the plan runs (`:1257`) and `_Dispatcher.ensure_prefix` never invalidates it, so even a successful removal would leave every later step addressed at the old directory; and the surface's `reset_capabilities()` (`_cli.py:1532`) runs before the removal it exists to account for.  The surface also `rmtree`s the artifact directory on the side (`:1515-1518`), so for the GPU env `--clean` deletes the built binaries and keeps the conda env -- while its help promises "every package is gone".  The one test on this path (`test_envs_install.py:933`) fakes the env FRESH, the state the defect cannot fire in. | a `StepRole.REMOVE` that bypasses the prefix requirement without the create decision; after it succeeds, `run_install` calls `reset_capabilities()` and clears `dispatcher.prefix`; delete the surface's early reset and side `rmtree`; retarget the test's fake to PRESENT and mutation-test it | a defect against env-framework § 5.4 and the `--clean` contract -- fix on your go, because it is the destructive path |
| **L2** | **Run-report files lose their mode on rotation.** `notify.py:279-290`: `RotatingFileHandler` opens the file at the umask mode in its constructor, one `os.chmod(0o600)` runs after, and on rollover the handler creates the next file at the umask mode with nothing re-tightening it.  The comment says "tighten it here AND after"; there is no after.  And the placement table's `reports/` row says mode None ("no credential") while this code enforces 0700/0600 -- contract and code disagree about what these files are. | decide the row (I would say private: they name the user and carry results); the handler's `_open` goes through `serve_daemon.open_private`, which is the door § 2.3 names for an appended log | yes -- the table row |
| **L3** | ✅ **CLOSED 2026-09-13.**  The dry run passes the snapshot's prefix or none; with none, `preflight` measures no disk, and the surface prints the home filesystem's free space through `check_disk` alone (no reference scale, so no walk).  **`install --dry-run` on a machine without the env walks every directory beside your home.** `_cli.py:1197` sets `disk_path = ~` when the env is absent and passes it as `env_prefix` to `preflight` (`:1222`), whose `env_size_reference_gb(Path(env_prefix).parent)` then `os.walk`s every sibling of `$HOME` -- all of `/home` on a shared login node. | pass no prefix; report free space at `~` through `check_disk` directly | no |
| **L4** | ✅ **CLOSED 2026-09-13.**  `component_install_valid(conda_binary=)` enters through `dispatch_into_env` under `_build_env` -- the same environment the phases run in -- quietly (`sink=None`).  Test: the verify command is `false`; bare it fails and the phases run, through the (faked) door the component is skipped whole.  Mutant (bare `_run_capture` again) fails it.  In the same commit `run_streaming(sink=None)` became CAPTURED ONLY (it defaulted to `sys.stderr`, so `doctor`'s verify probe -- whose comment said "captured rather than streamed" -- streamed into the report); the two streaming callers (`_run_build_phase`, `validate`) now say `sink=sys.stderr`; mutant (default restored) fails `test_no_sink_means_captured_only`.  **The source-build presence gate runs the built binary outside the env.** `component_install_valid` runs `{install}/bin/siesta --version` bare (`builds.py:1532`) while `_run_build_phase` runs the same command through the door with `LD_LIBRARY_PATH=<prefix>/lib` and the MPI tmpdir (`:1825-1850`).  The cmake `INSTALL_RPATH` (`recipes.py:1584`) probably makes the bare run work; the point is one command measured under two conditions, and M1. | route it through `dispatch_into_env` with `env_for_step` | yes -- it changes what "installed and working" is measured under |
| **L5** | **`serve foreground` prints the config warnings twice.** `cli.py:2221-2237` run in the parent and again in the re-exec'd child (`SUPERVISED_ENV=1`); the comment says "so it appears once". | guard on `SUPERVISED_ENV` | no |
| **L6** | **`auth-setup` overwrites the session key on every run, and gives `cd` advice for a file it says is unread.** `:1708-1709` regenerates `secret_key` (every session dies; the docstring calls this "idempotent EXCEPT") while `web/auth._install_secret_key` already creates one when absent -- with a different encoding (`token_bytes` vs `token_urlsafe`).  `:1742-1745` prints `cd <dir>` when the output landed in cwd, a location the same docstring says is NOT read.  `--output` is the `--keys-file` class this file retired at `:1984-1991`. | the wizard stops touching the session key; `--output` and the cwd branch go; the write goes through `write_config_scope` (D1) | with D1 |
| **L7** | **Every `molbuilder` invocation runs `nvidia-smi`.** `recipes.py:346-376` probes the driver at import; `cli.py:38` imports the envs group, which imports recipes, at module level -- so `--help` shells out with a 2 s timeout. | resolve `_CUDA_VERSION` when a recipe is asked for, not when the module loads | small design |
| **L9** | **`--clean` cannot remove an ORPHAN, and the ORPHAN hard-stop prints `--clean` as the remedy.**  `remove_step_for` addresses by `-n` (`install.py`), and an ORPHAN is by definition a directory conda's registry does not list -- `env remove -n` finds nothing.  Removing it needs `--prefix <dir>`, which only the probe knows (`prefix_from_fs`) and the pure plan cannot.  Found while fixing L1; not fixed there.  Shape: the removal step carries the probed directory as a declared alternative (`--prefix`), the way a pip step carries its index fallback. | a REMOVE step with a `--prefix` fallback when the probe found a directory the registry does not list | small design |
| **L8** | `/run/user/1000/molbuilder` is 0775 on this machine.  `supervise()` tightens it (`serve_daemon.py:321`); the running daemon predates that fix.  Machine state, not code: a `serve restart` clears the doctor line. | -- | no |

#### K-Y. Layering

| | finding | fix shape |
|---|---|---|
| **Y1** | `_effective_name` -- name resolution from config and an env var -- lives in `doctor.py` (the audit) and is imported by `install.py` at module level (`:56`), which is why `doctor` defers its own imports of `install` (`:515-518`, "the cycle is deliberate"). | move it beside `Capabilities.env_for_category`; the cycle disappears |
| **Y2** | `placement` and `runtime_config` import each other lazily (`placement.py:73`, `runtime_config.py:1323`).  `machine_config_warnings` is an audit sum. | let it live in `placement`; `runtime_config` then imports nothing upward |
| **Y3** | `scheduler/record.py` claims "stdlib-only ... ships to the target" (`:23-28`) and `diagnostics.local_facts` is placed where it is BECAUSE of that claim (`:482-486`) -- but the shipping list is `("mb_monitor.py", "config_dir.py")`, and `record.py` imports `..persist`, `..config_dir`, `.quantities`, `.admit`.  A constraint two modules reason from that nothing exercises. | drop the claim or make it true; decision |
| **Y4** | `known_machines()` composes user-facing summary strings (`?? not understood`) inside the record module (`record.py:833-962`). | note |
| **Y5** | J1 (`_dispatch.run_in_env`), sharpened: it also addresses the env BY NAME (`conda_run_argv`, `:73`) while the door re-addresses at the prefix (M2). | J1 |

#### K-D. Redundancy -- two homes for one thing

| | finding |
|---|---|
| **D1** | J2, sharpened.  `read_config(path)` is `json.loads` -> `_normalise` -> prefix the path (`runtime_config.py:109-151`), so `emit_molbuilder_json`'s "validate the bytes on disk" is the same validation `write_config_scope` runs in memory (`:2233`).  The wizard's merge REPLACES `auth` wholesale, so re-running it to add a provider drops `auth.trust_proxy`; the door's deep-merge keeps it.  The door's one gap -- it blames the patch when the existing file was already invalid (`:2229-2236`) -- is exactly what the wizard solved privately.  `cli._read_config_object` (`:1443`) is a third reader of the format, and `auth-setup` reads the same file three times in one run (`:1609`, `:1720`, `auth_setup.py:404`). |
| **D2** | Three pairs of mode constants for two values: `CONFIG_FILE_MODE`/`CONFIG_DIR_MODE` (`runtime_config.py:1266`), `CREDENTIAL_FILE_MODE`/`PRIVATE_DIR_MODE` (`config_dir.py:66`), `REPORT_MODE`/`REPORT_DIR_MODE` (`notify.py:172`).  The writer of `molbuilder.json` uses one pair; the audit checks the same file with another. |
| **D3** | `serve_daemon.run_dir` / `pid_path` / `log_path` / `stacks_path` are four one-line pass-throughs to `config_dir` doors (`:42-80`) -- I8's shape; `_mkdir_private` (`:87`) is a third name for `ensure_private_dir(tighten=True)`. |
| **D4** | Inside `runtime_config`: `_read_server_wide` is `read_config()` (`:1513`); `_read_scope` is `read_config(path)` with a pre-check the callee already makes (`:1189`); `_read_section` and `_require_object_section` both do "section as dict or raise" (`:154`, `:549`); `get_paths` re-validates an already-validated section inside a `try/except` that re-raises (`:1946`); `get_checkpoint*` and `get_script_generation` re-run validators on normalised output (`:1023`, `:1643`); `get_scheduler`'s two `elif ... _validate_scheduler(raw)` branches are unreachable (`:2010-2016`) -- the same dead guard removed from `get_execution` the day before. |
| **D5** | `machine_config_path()` returns `(path, "config-dir")` and the second element has been a constant since the cwd step was deleted; 14 callers index `[0]`. |
| **D6** | J3: `info --json` has three readers and the `envs_dirs` orphan walk is spelled twice in `install.py`.  Six private "run a command and capture" wrappers across `builds`, `validate`, `abi`, `record`, `diagnostics`, `initconfig`, each slightly different. |
| **D7** | The env prefix is resolved up to four times per install: `probe_env_state` puts it in `state.prefix`, then `_cli.py:1395`, `:1457` and `run_install:1257` each ask `_env_prefix` again. |
| **D8** | ✅ **CLOSED 2026-09-13.**  One `_build_env(env_prefix, paths)` from `build_subprocess_env`, used by the phases and the gate; `_run_build_phase` takes `paths` instead of climbing out of the log path.  Two temp-dir policies per build phase: `env_for_step` creates `<prefix>/var/tmp` and `var/cache/pip` (`builds.py:535-544`, mkdir on every phase) and `_run_build_phase` then overrides both (`:1826-1830`). |
| **D9** | ✅ **CLOSED 2026-09-13.**  `plan_build_spec(spec, paths, probe)` takes the layout and returns the steps.  `resolve_paths` is called five times inside one `run_build_spec`, one result named `_paths_unused` (`:1971`). |
| **D10** | `cmd_clean._du` vs `builds.env_size_reference_gb`; `_stdin_can_answer` copied from `cli._stdin_is_a_terminal` "because A7" (`_cli.py:2057`) -- the one function should move down, not be copied; `cmd_bootstrap` spells `initialize()` as `detect()` + `set_capabilities()` (`:1911`); `hints.fix_cmd` needs three call-site workarounds for a verb with no recipe (`_cli.py:250`, `:1339`, `:1896`). |
| **D11** | `read_notify_keys(path=)`, `issue_notify_key(path=)`, `load_channels(path=)`: `cli.py:1992-1998` resolves `notify_keys_path()` and passes it to a function that would resolve the same default -- the resolved-then-passed shape removed for `MB_NOTIFY_KEYS_FILE` the day before. |

#### K-X. Dead

| | finding |
|---|---|
| **X1** | ✅ **CLOSED 2026-09-13.**  Parameter, threading and comment deleted.  `preflight(conda_specs)` is never read; the "Forbidden packages" comment (`builds.py:1296`) has no code under it; the parameter is threaded through `run_build_spec` and `run_install` for nothing. |
| **X2** | ✅ **CLOSED 2026-09-13.**  `compute_fingerprint`, the rev-parse loop, the `.toolchain-fingerprint` write and the sentinel payload are gone; `write_sentinel(path)` touches; the name stays in `_ARTIFACT_ROOT_ENTRIES` so an old tree's file is not called stale; `cmd_clean`'s "kept" row no longer lists it; the five `compute_fingerprint` tests are retired and the sentinel test now drives a resume through `run_build_spec` (mutant: loop ignores sentinels -> fails).  The toolchain fingerprint is computed with a `git rev-parse` per component, written to `.toolchain-fingerprint` and into every sentinel, and read by nothing -- `run_build_spec` checks `sentinel.exists()` only (`:2012`).  `cmd_clean` tells the user it is "used by --clean / --force-resume" (`_cli.py:685`); it is not. |
| **X3** | ✅ **CLOSED 2026-09-13.**  All three deleted; `render_*_hook(spec)`.  `BuildStep.cwd` is always None and never passed; `render_activate_hook(spec, paths, probe)` ignores two of three arguments; `_inner_command`'s `--` branch (`builds.py:474`) handles a shape nothing produces -- the comment says `_run_build_phase` spells it, and `_run_argv` (`:415`) does not. |
| **X4** | ✅ **CLOSED 2026-09-13.**  Override deleted; the comment at the verdict says why it could never change the answer.  `run_install:1340-1342` overrides a verdict the adapted FAILED steps already decide. |
| **X5** | `cmd_bootstrap:1955` tests `'failures' in dir()` for a local defined in one branch. |

#### K-S. Statements that make someone do the wrong thing

| | where | says | truth |
|---|---|---|---|
| **S1** | `config_dir.py:130-135` | `paths.logs` is applied by `runtime_config.logs_dir` | the key is refused; that function does not exist |
| **S2** | `runtime_config.py:30-32` | flat `cert`/`key` are "honoured (folded into tls)" | refused by name (`:772-776`) |
| **S3** | `install-env.sh:93-94` | `./molbuilder.json` overrides "are read from CWD" | retired 2026-08-31 |
| **S4** | `install-env.sh:255-272` | entry points (a) and (b) are equivalent except `--gcc` | (b) has no disk probe for the manager (`_find_conda_binary` vs `detect_env_mgr` step 4): a shell where conda is only a function finds a manager through the shim and none through `python -m molbuilder` |
| **S5** ✅ closed 2026-09-13 | `builds.py:23-24` | "the only module that runs subprocesses" | seven others do; the docstring now says what is true: every command that ENTERS an env comes through here |
| **S6** | `runtime_config.py:1336-1338` | the config is written by an `fchmod`-before-first-byte writer | that writer is gone; `write_bytes(mode=)` is |
| **S7** | `install.py:981`, `:1252-1256` | resolving a prefix costs "3-5 registry calls" | usually zero since the snapshot strategy |

## 3. THE MIGRATION — finishing the design, in phases

**Scope of this plan: `install-env.sh` + the `envs` verbs (install / bootstrap /
doctor / repair / validate / clean / init-config), deployment, and how config and
secrets are PLACED and VALIDATED.** Items outside that — § 5 — are recorded and
deliberately not in the phases.

### 3.0 The end state, stated so it can be checked

When this migration is finished, all nine of these are true of the tree, not just
of the documents:

| | the end state (and the § 0 invariant it realises) | today |
|---|---|---|
| **Z1** (T3) | **One orchestration door.** `install` and `bootstrap` differ only in which recipes they are given. Probe, summary, confirm, log, run — one function. No verb has a branch the other lacks | ✅ **reached (Phase 4)** — `_install_one`, called by both |
| **Z2** (T4, T6) | **Everything the installer does is a step with an `Outcome`.** Including the `--clean` wipe. Nothing dispatches a subprocess outside `run_step` except the two things § 5.4/§ 5.5 name | ✅ **reached (Phase 4)** — `remove_step_for` goes through `run_step`, and `envs/_cli.py` no longer imports `subprocess` at all |
| **Z3** (T1) | **One answer to "does this env exist, and where".** `EnvState` owns it; `env_available` and the three registry readers collapse into it; no caller re-derives a prefix or a create decision | ✅ **reached (Phases 0 + 2)** — one reader answering `{name: prefix}`, the snapshot carrying it, the probe measuring the named prefix, and the create decision handed down rather than re-derived |
| **Z4** (T1, T6) | **The state and the outcome are enums, and one mapping prints each.** No `== "PRESENT"`, no second vocabulary between the live line and the recap | ✅ **reached (Phase 5)** — `EnvPresence` classifies and `state_label` is derived; `Outcome.word` is printed by both readers; `ProbeResult.tag` likewise |
| **Z5** (T9, T12, T8) | **Every path is asked for.** No module joins a directory to a filename it does not own; nothing climbs `.parent` to a root; **every remedy the program prints names a resolved path and the detected manager** | four climbs, three wrong remedies (**I1-I4**, **D4**) |
| **Z6** (T10) | **One writer puts bytes at a path, and privacy is its parameter.** The only exceptions are the ones § 2.3 names, and § 2.3 names all of them | ✅ **reached (Phase 3b)** — the second private writer is gone, the stacks log and `serve status` go through the private doors, and the five that legitimately sit outside are each named with a reason. The list is the rule now |
| **Z7** (T11) | **Modes are asserted on arrival, not only set at creation** — and the assertion is driven by the documented tree, so the document cannot drift from it. See § 3.4 | ✅ **reached (Phase 3a)** — `placement.places()` is the tree as 15 checkable rows, `config_dir.ensure_private_dir` the one creator, `placement.findings()` the audit, read by `envs doctor` and `machine_config_warnings`. It found **seven** real ones here on its first run |
| **Z8** (new rule) | **No remedy the program prints can destroy working state.** Refusing to delete the env you are running from, and never calling a healthy env `GHOST` | ✅ **reached (Phase 0)** — stated as M5, held by `runs_from_prefix` at both surfaces and by the probe measuring the named prefix |
| **Z9** (new rule) | **The suite runs to completion**, and `testrun.py` cannot report a pass count for a run that stopped early | ✅ **reached (Phase 0)** — it did run to completion all along (9082); what could not be trusted was the instrument, and four non-result states now say so |

### 3.1 Phase 0 — ✅ DONE *(2026-09-12)*

**Closed F3, A0, H1 — and A5, plus half of D4. → Z8, Z9.** What it actually
found, because two of the three were not what this document said they were:

1. **F3 was the instrument, not the suite.** The suite had been running to
   completion throughout (9082 tests, four pre-existing failures). The progress
   file had been truncated under a live run, and the reader counted the remains
   and called them `done`. The `-p no:randomly` advice was a no-op —
   `pytest-randomly` is not installed. Both halves fixed: the writer no longer
   destroys a generation it did not finish, and the reader has four states that
   refuse to print a result.
2. **A0** — one predicate, `runs_from_prefix`, asked by both surfaces that could
   propose the destruction: `--clean` refuses, `doctor` offers repair plus the
   three-step from outside instead. Tested on the host recipe.
3. **H1** — the probe measures the prefix the registry named. GHOST means what
   § 2.1 says, and the rule text now says which path it means.

**And the rule the user gave mid-phase is written down** *(M1-M5, T14)*: the
manager is asked, never imitated. That ruling is why A0's guard does not consult
`CONDA_PREFIX`, why the remedies are spelled with the detected binary, and why
**Phase 1 now begins with the dispatch** rather than leaving it to Phase 4 —
every later step would otherwise be written against the hand-rolled path.

### 3.2 Phase 1 — ask, do not derive: the dispatch, then the paths

Two halves of one rule — *ask, do not derive* — and the installer half comes
first because the rule text for it now exists and the tree contradicts it.

**Closes H3, H13, H4's sibling, I4, I1, I2, I3, I6, D4, and the façade gap.**
→ **Z5**, and T14 in the tree rather than only in the contract.

**1a. ✅ DONE — the dispatch stops imitating the manager** *(pulled forward from
Phase 4, 2026-09-12, on the user's ruling)*. `conda_run_argv` becomes the primary form and
the activation wrapper a **declared fallback** fired on the stub's signature, so
a modern manager uses its own activation and a mamba-1.x site records
`RECOVERED`; `builds.py`'s second copy goes; `TMPDIR` / `PIP_CACHE_DIR` and
`build_subprocess_env()` move onto `run_step(env=)`, the parameter nothing
passes. **H13 first inside this item**: address the env by the prefix the
registry gave us, because `-n` cannot reach an env outside `envs_dirs` at all.

**1b. ✅ DONE — the config-placement half.** The rule was corrected before the
sites: A11's elaboration named a pre-consolidation owner and so licensed the
climbs. Closed I4, I1, I2, I3, I6, D4, E5, the façade gap, and H10's
config-writing third. **A4 deliberately left**: the `mkdir` beside I1 creates the
config directory world-readable, and a `mode=` there is the instance — Phase 3
owns the mechanism.

1. **I4** — fix A11's elaboration in `architecture.md` to name `config_dir` as the
   root's one home and each format owner as its filename's.
2. Export `calculation_record` from `scheduler/__init__.py`. A façade that offers
   `FILENAME` and hides the door is why **I2** exists.
3. Sweep the climbs: **I1** (`jobset probe --write`), **I2**
   (`config_provenance`), **I4** (`environments_dir`).
4. **I3, I6, D4** — every printed path resolved, every printed command naming the
   `effective` env and `caps.conda_binary`. `_fix_cmd` moves below the surface so
   `recipes.py` can call it instead of hand-copying its output.

### 3.3 Phase 2 — ✅ DONE *(2026-09-12)* — one answer to "does this env exist, and where"

**Closes H2, H6, and the second probe.** → **Z3**

**The design, settled 2026-09-12 before any code** (measured costs, because they
decide the shape: `<mgr> env list --json` is **1.22 s** on this machine and
`info --json` **1.82 s** — these are not cheap reads, and `doctor` pays
`_env_prefix` once per recipe):

1. **One registry reader, and it answers with PREFIXES** —
   `diagnostics.conda_env_prefixes(conda) -> {name: prefix}`, replacing
   `_list_conda_envs`, which answered with names only and filtered on *"the
   parent directory is literally called `envs`"*. That filter is why
   `env_available` says **no** for an env `probe_env_state` reports **PRESENT**
   (H1/H2's disagreement): an env created with `--prefix` elsewhere is listed by
   the registry and invisible to the six gates.
   **Correction to this plan, 2026-09-12 while building it:** the design above
   proposed identifying the installation root by a relationship in the document
   (another prefix under `<p>/envs/`), and a test of `detect` showed why that is
   not enough — a registry listing *only* roots has no relationship to read, and
   those roots came back as envs named `miniconda3` and `anaconda3`. Measuring
   the real output answered it properly: `env list --json` carries
   **`envs_details`**, `{prefix: {"name", "base", …}}`, so **the manager names its
   own envs and flags its own base**. That is M2 rather than a heuristic, and it
   also stops the basename being invented as a name — the base env is called
   `base`, and `miniconda3` is not a name conda knows. A manager that reports no
   `envs_details` gets **every** listed prefix, nothing excluded — a second
   correction, and the tests made it: keeping the old filter on that path broke
   three tests of the state machine, because an env the registry lists but the
   map omits reads as **FRESH**, whereupon `conda create -n <name>` makes a
   SECOND env beside the real one. An installation root listed under its
   directory's basename is by contrast a name nothing asks about.
2. **`Capabilities.conda_envs` becomes that mapping**, plus an `env_prefix(name)`
   accessor. Every current consumer uses `in`, `sorted()` or iteration, all of
   which read a dict exactly as they read a frozenset, so the change is in what
   the snapshot KNOWS, not in what callers do.
3. **`_env_prefix` asks the snapshot first**, then a fresh registry read (an env
   may have been created mid-process), then the tiers it has. `doctor`'s
   per-recipe resolution becomes a dict lookup — five recipes × ~1.2 s of
   redundant registry reads, gone.
4. **`probe_env_state`'s registry half uses the same reader**, so the third
   spelling goes with the second.
5. **H6**: `cmd_install` hands the state it already probed to `run_install`
   instead of having `_create_decision` probe again, and reads `state.can_resume`
   rather than comparing `state.state_label == "PRESENT"`.

The six gates (`repair`, `clean`, `validate`, `bootstrap --skip-existing`, the
`--clean` pre-check, `doctor`'s `present`) are then answered from the same reading
the state machine uses, without touching the gates themselves.

### 3.4 Phase 3 — ✅ DONE *(2026-09-13)* — placement VALIDATED, from the document itself

This is the piece the design does not have yet, and it is what stops §§ A, B and I
from regrowing. **Make `configuration.md` § 3.1's tree executable.**

**Closes A2, A4, I5, I7, B2, B3, B4, B7, D11, D14.** → **Z6, Z7**

**The design, settled 2026-09-13 before any code.** Two creators already exist
and they disagree, which is the whole finding in miniature:
`serve_daemon._mkdir_private` makes a directory `0700` **and tightens one that
arrived loose**; `initconfig._ensure_root` makes it `0700` and deliberately
leaves an existing one alone, its docstring saying *"this seeds, it does not
police — `envs doctor` is where a permissions audit would belong."* That
sentence is the design. It just has no table and no audit yet.

1. **One table**, in `config_dir.py` — the root's owner, and below every surface.
   One row per entry of § 3.1's tree: the resolver, the expected mode, whether it
   is a directory, whether it holds a credential. **The table is the authority
   and § 3.1 cites it**, rather than a test comparing prose to code: a checker
   that silently omits a file is the defect to prevent, and a doc-parsing test
   would break on formatting while proving nothing about coverage.
2. **One creator and one checker over it.** `ensure_private_dir` generalises
   `_mkdir_private` and is what `initconfig`, `serve_daemon`, `jobset probe
   --write` (**A4**) and `write_config_scope` all call. The checker answers
   *"what arrived loose"* for the whole table and is read by `envs doctor`
   (reports it) and by `machine_config_warnings` (the serve path) — which
   subsumes `machine_config_mode_warning`, today a check of one file while
   **A4** measured the directory around it world-readable.
   Seeding stays seeding; policing stays `doctor`'s.
3. Route the unnamed writers through `persist.write_bytes` (**I7**, **D11**), or
   name them in § 2.3 with a reason (**B2**'s `O_EXCL` create, the pidfile, the
   seeded READMEs, the appends), and fix the modes (**A2**, **A4**, **I5**).
4. Correct the false statements that ship to users (**B1**, **B5**) and the count
   in the three other places.

### 3.5 Phase 4 — ✅ DONE *(2026-09-13)* — one orchestration door

**Closes D7, D3, D1, D2, H3, H4.** → **Z1, Z2**

1. `_install_one(recipe, caps, *, auto_yes, …)` — probe, summary, confirm, tee,
   run. Both verbs call it; `bootstrap` stops being a lossy copy.
2. `remove_step_for(env_name, conda)` through `run_step`, so the wipe carries an
   `Outcome` and the printed manual form is `_shell_join(step.argv)` (**D3**).
3. **H3** — decide whether install steps run sanitised, then make the door do it.
   The dead `run_step(env=)` is the unmade decision; `build_subprocess_env()` is
   the existing answer.
4. **H4** — write the two translators § 4.2 already names (`pip_steps_for`,
   `extra_steps_for`); **D2** — drop `run_build_spec`'s dead `conda_binary` and the
   three tests pinning it; **D1** — `init_config` raises through
   `seeding_blockers()` instead of a `PermissionError`.

### 3.6 Phase 5 — ✅ DONE *(2026-09-13)* — the state machines stop being strings

**Closes H7, H5, H11, and finding E7.** → **Z4**

`EnvState` and the outcome words become enums with one mapping each, so the live
line and the recap cannot disagree and a renamed label cannot silently disable the
create-skip.

### 3.7 Phase 6 — the user's instruction, and the cheap sweep

**Closes C1, C2, D5, D6, D12, D13, H8-H10, H12, I8, I9, B6, E5, E6.**

1. **C1, C2** — disk: report free space, print the reminder, no threshold, no
   `warnings` bucket, no filesystem walk; and sweep the `~30 GB` out of
   `recipes.py` with the test pinning it.
2. **D13** — give the host env's name a persistent home, or remove the override.
   `"envs": {"host": …}` validating and being ignored is the worst of the three.
3. Delete what nothing calls; collapse the duplicate literals; correct the
   docstrings that still describe the pre-migration direction.

### 3.8 Phase 7 — the two decisions

**F1** (A-rule "checked by" columns vs the retired guards) and **F2** (whether the
2026-09-11 measurement artifact is restored). Neither is a defect; both need the
user.

## 4. Method note for whoever picks this up

The three audits were told to **falsify** the commit messages, not confirm them,
and that is what made them useful — 14 claims held, and the ones that did not
were all overstatements of scope rather than outright wrong behaviour. Two habits
produced nearly every item in § 2:

- **Fixing the instance, not the rule.** D1-D10 are all one shape. The project's
  own rule is *fix the RULE in the document that owns the concept, then sweep the
  restatements* — §§ B and D are what happens when the restatement sweep is
  skipped and a commit message claims it anyway.
- **Writing the rule down while the code beside it disagrees.** B2, C1, D5 and
  B7 each state a property in a docstring that the adjacent code does not have.
  A docstring is not a mechanism.


## 5. Recorded, and OUT OF SCOPE for this migration

Not install, deployment or config. Listed so they are not lost and not worked on
here: **E1** (the `engine` field's missing `item_kind`, one cause behind three
suite failures, from 2026-09-11) · **E3** (`generator.md` / `script-preparation.md`
on the wrapper's rank count) · **E4** (`job-system.md`'s `resources` block listing
7 of 15) · **I10** (`oauth.py`'s two interpreters of `client_secret_file`) ·
`test_results_blueprint`'s six undocumented DOM ids (**F3a**).

**S17** has left this list: it was *"`run_tool` dispatches into an env without the
mamba-1.x workaround"*, and unifying the dispatch in Phase 1a closed it — the
trade that kept it open (resolving a prefix is too expensive for a per-structure
path) dissolves once the workaround is reached by measurement instead of applied
in advance.
