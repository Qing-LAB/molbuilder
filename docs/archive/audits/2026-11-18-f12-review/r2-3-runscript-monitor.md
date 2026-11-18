# F12 review, round 2 — slice R2-3: the run script, the live monitor, the machine and its configuration

Checkout `the checkout` at f578482db (2.77.2), read only. Nothing
was written in the checkout; the only Python run was two read-only `python
-B -c` checks (named where used).

## What was read

**Whole:** `runwrap.py`, `script_emit.py`, `monitor.py`, `monitor_log.py`,
`wrapper_log.py`, `pipeline_log.py`, `runtime_info.py`, `runtime_config.py`,
`placement.py`, `diagnostics.py`, `joblist.py`, `jobs_cli.py`,
`serve_daemon.py`, `config_dir.py`; `scheduler/` (`__init__`, `emit`, `fit`,
`kinds` + `kinds.toml`, `place`, `probe`, `quantities`, `queue`, `record`);
`config/` (`__init__`, `stages`, `siesta`, `pyscf`); `envs/__init__`,
`envs/hints`, `envs/_dispatch`.

**In part:** `envs/recipes.py` (lines 1–1700 and 2560–2720 read; the GPU
recipe's component records skimmed by symbol); `envs/builds.py` (1–145,
440–1300, 1636–1810 read; the rest by symbol list); `envs/initconfig.py`
(455–600 read, the rest by grep); `envs/validate.py` (141–232 + symbol
list); `envs/abi.py` (576–597 + symbol list); `envs/_cli.py` (2031–2150 +
grep + symbol list).
**Not read beyond a symbol list:** `envs/install.py`, `envs/doctor.py`.

**Tests:** whole — `test_runwrap.py`, `test_wrapper_preamble_preflight.py`,
`test_wrapper_notify_flags.py`, `test_machine_record.py`; in part —
`test_monitor_notify_policy.py` (1–80, 695–833 and its index), the
machine-snapshot helpers of `test_envs.py`, `test_envs_doctor.py`,
`test_envs_install.py`, `test_envs_install_plan_executes.py`,
`test_envs_enters_the_env_through_the_manager.py`, `test_diagnostics.py`.
**Scanned by grep, all 194 test files that import or drive this slice**: for
tests reading source or doc text (none found — the three hits read generated
decks), hand-typed machine records and log lines, and repeated preludes.

Round 1's findings (`r1-B-runscript.md`) are not repeated; two of them are
named below where a duplicate is their cause (D1, P1).

---

## Findings — 26

Ordered by weight. "Owner" is the place that should hold the fact; "Sure"
says whether both sides were read.

### F1. The two job bundles are hand-kept lists; the monitor's is derived — and the hand lists already miss members (cause of round-1 P1)

- **Places:**
  - `runwrap.py:3057–3103` — the monitor bundle is DERIVED: `_members()`
    walks each member's `TRAVELS_WITH`, so "no module is listed here by
    hand" (`runwrap.py:3042`).
  - `runwrap.py:3259–3322` — `VIBRATION_COMPANIONS: Dict[str, str] = {
    "siesta_vibration.py": ..., ..., "end_lines.py": ...}`, 44 hand-typed rows.
  - `runwrap.py:3391–3423` — `PYSCF_COMPANIONS`, 22 hand-typed rows.
- **Owner:** each module's own `TRAVELS_WITH` (the rule `_members()` already
  applies; architecture § 3.2 row "what travels beside a job").
- **What drifts — measured:** a read-only check of the transitive
  `TRAVELS_WITH` of each hand list's members found, today:
  `runrecord` names `molbuilder.scheduler.quantities`, absent from
  `mb_vibration.pyz`; `runfiles` names `molbuilder.runpoints`, absent from
  `mb_pyscf.pyz`. These are exactly round-1's P1. A member gaining an import
  is shipped in the monitor bundle automatically and silently missing from
  the other two. 29 of 44 vibration members and 21 of 22 PySCF members
  declare no `TRAVELS_WITH`, which is why the derivation cannot be applied yet.
- **Smallest fix:** give every member its `TRAVELS_WITH`, keep only the one
  entry module per bundle (as `_MONITOR_BASE` does), and build all three
  bundles with the one closure `_members()` already is.
- **Sure:** both sides read; the gap measured.

### F2. The run script works out "which GPUs this job has" three times in bash (cause of round-1 D1)

- **Places:**
  - `runwrap.py:771–777` (`_gpu_loadbalance_block`):
    `_ngpu=$(printf %s "$CUDA_VISIBLE_DEVICES" | tr , "\n" | grep -c . ...)`
    else `nvidia-smi -L | grep -c "^GPU "`, an absent GPU → `0`.
  - `runwrap.py:939–951` (the rank helper's heredoc): `IFS=, read -ra _gpus
    <<< "$CUDA_VISIBLE_DEVICES"` else a list `0..n-1` with a floor of one
    GPU; `_gpu=${_gpus[$_idx]}`. The block's own comment says the first
    "must agree with the per-rank launcher's own count".
  - `runwrap.py:1089–1094` (dry run): `_dr_g=$(( _dr_r * _dr_ng / _mpi_np ))`
    — an INDEX, printed and looked up as a device id; and `:1251`
    `_bash_numa_from_gpu("0", ...)` — device 0, for the banner's "GPU0 NUMA"
    and the whole-job `numactl` wrap.
- **Owner:** one bash step that sets the job's GPU list (`_gpus`) once, before
  the counts; every later block reads it.
- **What drifts:** this is round-1 D1's cause — with `CUDA_VISIBLE_DEVICES=1`
  the helper runs rank 0 on device 1 while the dry run, the banner and the
  wrap describe device 0. The two counts also differ when no GPU is listed
  (0 vs 1).
- **Smallest fix:** compute `_gpus`/`_ngpu` once in the load-balance block
  and export them; the helper, the dry run and the NUMA lookup read
  `${_gpus[i]}`, never an index or a literal 0.
- **Sure:** read; the rendered text traced.

### F3. The pipeline log's order re-lists, by hand, the vocabulary of who set a value

- **Places:** `pipeline_log.py:208–210`:
  `rank = {"pin": 0, "sweep": 1, "stage": 2, "stage_card": 3, "calc_card":
  4, "flag": 5, "run card": 5, "role": 6, "allocation": 7, "template": 8,
  "default": 9}` — against `template.py:2639–2651`, `SAID = {"template":
  ..., "stage": ..., "stage_card": ..., ..., "role": ...}`, "WHO SET A VALUE,
  in the words every reader says it … the pipeline log".
- **Owner:** `template.SAID` (architecture § 3.2; template.md § 5.4).
- **What drifts:** `config_rows` also orders every deck's PROVENANCE rows
  (`resolve.py:392–400`). A key added to `SAID` sorts at 9 silently; a key
  renamed there (`"run card"` → `"run_card"`) keeps the old key here and
  mis-sorts with no error.
- **Smallest fix:** state the order beside `SAID` (an ordered tuple, or `SAID`
  itself declared in decision order) and read the rank from it.
- **Sure:** both read.

### F4. The notebook shepherd's marker is spelled in four places

- **Places:** `serve_daemon.py:420` `stop_by_pidfile(nb_pidfile,
  marker=b"_shepherd")`; `jupyter.py:229` `_MARKER = b"_shepherd"`;
  `jupyter.py:203` the argv `"jupyter", "_shepherd"`; `cli.py:2996`
  `@jupyter_group.command("_shepherd", hidden=True)`.
- **Owner:** `serve_daemon` (L1, imported by `jupyter`; it cannot import
  `jupyter`, so the constant lives there or in `config_dir` beside
  `JUPYTER_PIDFILES`).
- **What drifts:** rename the hidden verb and the supervisor's start-up
  reconciliation (`serve_daemon.py:417–421`) reads the live shepherd as
  "foreign" and never stops it, while `jupyter.pid_state` still matches.
- **Smallest fix:** one constant in `serve_daemon`; `jupyter` builds its argv
  and `_MARKER` from it, `cli` names its command from it.
- **Sure:** all four read.

### F5. The configured-files table re-types the log and runtime file names `config_dir` owns

- **Places:** `placement.py:134` pattern `"serve-*.log"`, `:156`
  `"jupyter-*.log"`, `:174` `"jupyter-*.json"` — against `config_dir.py:357`
  `logs_dir() / f"serve-{port}.log"`, `:388` `f"jupyter-{serve_port}.log"`,
  `:399` `f"jupyter-{serve_port}.json"`. The pidfiles were given the one
  shape both ways (`pidfile_pattern`, `config_dir.py:314`); these were not.
- **Owner:** `config_dir` ("Nobody else may spell these", `config_dir.py:169`).
- **What drifts:** rename the serve log and `envs doctor` stops checking the
  mode of the file "measured 2026-09-12 carrying a client_secret" — silently,
  because a pattern that matches nothing reports nothing.
- **Smallest fix:** a `*_pattern` beside each name in `config_dir`, as
  `pidfile_pattern` is, and `placement` asks it.
- **Sure:** both read.

### F6. A calculation's copy of its machine record has two readers, and only one refuses a copy that names no machine

- **Places:** `scheduler/record.py:975–981` `calculation_environment`:
  `env = read_environment(path); if env is None and path.is_file(): raise
  UnknownTarget.unreadable_copy(bundle_dir)` — and `record.py:1607–1618`
  `machine_for` re-implements the same rule for the calculation scope
  (`if label == CALCULATION_SCOPE and Path(path).is_file(): raise
  UnknownTarget.unreadable_copy(...)`) and adds `if label ==
  CALCULATION_SCOPE and env.machine is None: raise
  UnknownTarget.unnamed_copy(...)`.
- **Owner:** `calculation_environment` ("that copy alone", architecture § 3.2).
- **What drifts:** a copy that names no machine is refused by `machine_for`
  and accepted by `calculation_environment`, which `joblist.scheduler_of`
  (`joblist.py:214–216`) asks — two answers for one file.
- **Smallest fix:** `machine_for` asks `calculation_environment` for the
  calculation scope, and the unnamed-copy refusal moves into it.
- **Sure:** both read; `joblist` traced.

### F7. `env_init`'s two keys are stated in two places

- **Places:** `runtime_config.py:680` `_ENV_INIT_KEYS = ("activation",
  "preamble")` (the section's validator and `get_env_init`); `record.py:499–502`
  `env_init={k: str(v) for k, v in ... if k in ("preamble", "activation")}`.
- **Owner:** `runtime_config` (the section's schema; the record carries "the
  same name there", `record.py:355`).
- **What drifts:** a key added to the section is copied by `jobset probe`
  (`diagnostics.local_facts` copies the dict whole) and then dropped by the
  record reader, with no word.
- **Smallest fix:** make the key tuple public in `runtime_config` and read it
  in `Environment.from_dict`.
- **Sure:** both read.

### F8. `SIESTA_RESTART_GROUP` is a second, unread copy of the restart keywords

- **Places:** `config/siesta.py:497–515` `SIESTA_RESTART_GROUP =
  RestartGroup(literal="SystemLabel", keys=("DM.UseSaveDM",
  "MD.UseSaveXV", "MD.UseSaveCG"), ...)`, whose own comment (`:494–496`)
  says "the one authority is `[item.restart].expands` … This tuple has no
  production reader and no test holds it to the catalogue";
  `config/pyscf.py:104` `PYSCF_RESTART_GROUP`; their class
  `identity.py:49` `RestartGroup`. A grep finds no reader of any of the three
  outside `config/` and `identity.py`'s own `__all__`.
- **Owner:** the catalogue's `[item.restart].expands`, read through
  `script_emit.parameter` / `restart_group`.
- **What drifts:** the next keyword added to `expands` leaves this tuple
  wrong, and a reader who finds it first believes it.
- **Smallest fix:** delete the two declarations and the class (the
  `mechanism` prose belongs in `run-identity.md`).
- **Sure:** grep over `molbuilder/` and `tests/`.

### F9. initconfig decides whether to write by matching a sentence it composed

- **Places:** `envs/initconfig.py:482`
  `if "declares no env_init.activation" not in note: return Step(path,
  "kept", note)`, where `note` comes from `_config_note()` at `:510–514`:
  `return ("left as it is -- but it declares no env_init.activation, so the
  probe has nothing to copy ...")`.
- **Owner:** `runtime_config.get_env_init()` — whether the file states an
  activation is a value, not prose.
- **What drifts:** reword the sentence and `init-config` silently stops
  adding the activation to an existing config, so every prep on that machine
  is refused.
- **Smallest fix:** `_config_note` returns the activation it read beside its
  words, and `seed_machine_config` branches on that.
- **Sure:** both read.

### F10. The activation forms are restated as literals outside their one home

- **Places:** `runtime_config.py:1275–1278` `ACTIVATION_FORMS: tuple =
  ("source activate", "conda activate")`, "in ONE home: … `envs
  init-config` offers them as the choices" — yet `envs/_cli.py:2049–2050`
  returns `"conda activate"` / `"source activate"`, `:2124–2133` prints the
  menu `"1) conda activate"`, `"2) source activate"` and maps
  `activation = "conda activate" if choice == "1" else "source activate"`;
  the rule "only `conda activate` needs the hook preamble" is written twice:
  `_cli.py:2133` `if activation not in (None, "conda activate"):
  preamble = None` and `initconfig.py:585` `if activation == "conda activate":`.
- **Owner:** `runtime_config.ACTIVATION_FORMS`.
- **What drifts:** a third form added there is validated but never offered;
  the hook rule changed in one file disagrees with the other's note.
- **Smallest fix:** make each form a row (name, needs-the-hook, menu words)
  in `runtime_config`; the menu and both checks read the rows.
- **Sure:** read.

### F11. The source-build layout is spelled three ways

- **Places:** `envs/builds.py:1661–1676` `resolve_paths` —
  `root = Path(env_prefix) / "opt" / spec.artifact_subdir`,
  `component_install(name)` = `root / name`; `envs/validate.py:173–176`
  `_stack_root` re-joins `Path(env_prefix) / "opt" /
  recipe.build_spec.artifact_subdir`, then `:190–191` `/ "siesta"` and
  `/ "bin"`; `envs/recipes.py:1609–1614` and `:1650–1652`, the
  activate/deactivate hooks, type `"$CONDA_PREFIX/opt/siesta-gpu-stack/siesta/bin"`,
  `.../elpa/lib`, `.../netcdf_fortran/lib`, repeating `artifact_subdir=
  "siesta-gpu-stack"` (`:2261`) and the component names. `BuildSpec`'s doc
  says the hook "Receives the same placeholders as component fields"
  (`recipes.py:739–741`), but `builds.render_activate_hook` (`:1794–1803`)
  is a pass-through, so it receives none.
- **Owner:** `builds.resolve_paths` / `BuildPaths`.
- **What drifts:** rename `artifact_subdir` or a component and the build
  installs to the new place while the hook puts the old one on PATH — SIESTA
  is not found in the env, and `validate` checks the wrong folder.
- **Smallest fix:** render the hooks with `{prefix}`/component placeholders
  through `_apply_template`, and give `validate` the `BuildPaths` door.
- **Sure:** all three read.

### F12. A production branch exists only for a test stand-in, and the stand-in is copied per test file

- **Places:** `diagnostics.py:159–170`, `Capabilities.env_prefix`: "It also
  answers ``None`` for a snapshot whose ``conda_envs`` is a plain set of
  names … a test binding a synthetic snapshot …" — `if not
  isinstance(envs, _abc.Mapping): return None`, while the field is declared
  `conda_envs: Mapping[str, str]` (`:119`). The stand-in is hand-built in six
  files with no `tests/support/` door: `test_envs.py:25` `_machine_with`,
  `test_envs_doctor.py:20` `_bind`, `test_envs_install.py:51` `_bind` (the
  same signature and body), `test_envs_install_plan_executes.py:41` `_caps`
  (`conda_envs=frozenset()`), `test_envs_enters_the_env_through_the_manager.py:429`
  `_machine_with`, `test_diagnostics.py:71` `_caps`; about 25 more inline
  `Capabilities(...)` across those files.
- **Owner:** one support door (beside `tests/support/machines.py`) that
  builds a snapshot in the one shape `detect()` builds.
- **What drifts:** the record is carried in two shapes (set vs mapping), and
  a reader of `env_prefix` written against the mapping passes with one
  helper and fails with another.
- **Smallest fix:** one support function with `{name: prefix}` only; drop the
  set branch in `env_prefix`.
- **Sure:** production read; four of the helpers read, two located by grep.

### F13. "Ship a bundle into an empty folder and import from it" is hand-repeated in seven tests, with two strengths of isolation

- **Places:** `test_monitor_notify_policy.py:733–746`, `:763–776`,
  `:815–829`; `test_pyscf_bundle_runs_alone.py:51–52`;
  `test_monitor_watches_a_live_run_e2e.py:433–442`;
  `test_the_rows_readers.py:48–61`; `parse/test_registry.py:72`. Each writes
  `(dir / BUNDLE).write_bytes(runwrap.<x>_bundle())`, strips `PYTHONPATH`,
  and runs `python -c "import sys; sys.path.insert(0, BUNDLE) ..."`; one
  (`:820`) adds `-I`, the others do not.
- **Owner:** a `tests/support/` door — "a job's python beside this bundle".
- **What drifts:** the tests that prove "molbuilder absent" do not agree on
  what absent means; one without `-I` passes while a user site, an `.pth` or
  the cwd still reaches the package.
- **Smallest fix:** one support function taking the bundle and the code,
  always `-I`; the seven call it.
- **Sure:** located by grep, three read.

### F14. `abi._quiet_run` and `builds._run_capture` are the same function

- **Places:** `envs/abi.py:582–589` and `envs/builds.py:877–883` — both
  `subprocess.run(list(argv), capture_output=True, text=True,
  timeout=timeout, env=dict(env) if env is not None else None)`.
  `builds` already imports `abi`. (`validate._run`, `validate.py:153–170`,
  and `scheduler/record._run`, `record.py:625–632`, are near cousins.)
- **Owner:** one helper in `abi` (the lower module).
- **What drifts:** a timeout or encoding fix lands in one.
- **Smallest fix:** delete one; import the other.
- **Sure:** both read.

### F15. The monitor spells "every channel" as a literal beside its owner

- **Places:** `monitor.py:1896` `if value.strip() == "*": return None` —
  `config_dir.py:555` `ALL_CHANNELS = "*"`, which `runwrap.py:2187` uses to
  write the flag. The monitor already imports `config_dir` both ways
  (`monitor.py:1356–1359`, `:1787–1791`).
- **Owner:** `config_dir.ALL_CHANNELS`.
- **What drifts:** change the token and the monitor reads it as one channel
  named `"*"` — reports go nowhere, said only as "not set up on this machine".
- **Smallest fix:** compare with `ALL_CHANNELS`, imported the two ways.
- **Sure:** both read.

### F16. `render_run_wrapper` keeps a second, lenient deck reader no caller reaches

- **Places:** `runwrap.py:2163–2168` — `if deck_text is None: try: deck_text =
  read_deck(engine, script_path) except (OSError, ValueError): deck_text =
  None`; its one caller, `render_wrappers` (`runwrap.py:3554–3571`), reads
  the deck itself (refusing on failure) and always passes `deck_text`.
- **Owner:** `render_wrappers`' strict read.
- **What drifts:** a future direct caller gets a run script whose held-atoms,
  restart and orbital lines silently say "not stated" for a deck that did not
  read, where the real route refuses.
- **Smallest fix:** make `deck_text` required and delete the branch.
- **Sure:** callers grepped (one).

### F17. `env_entry` carries scaffolding for preamble scopes that no longer exist

- **Places:** `runwrap.py:1597–1607` — `_preamble_chunks = [("target", ...)]`,
  `_scope_labels = {"target": "TARGET PREAMBLE (from the target machine's
  record)"}`, `_scope_labels.get(scope, scope.upper())`; and
  `_preamble_source_targets(chunks)` (`:1501–1526`) takes `(scope, text)`
  pairs and ignores the scope.
- **Owner:** the record's one `env_init.preamble`.
- **What drifts:** nothing yet; it invites a second scope back.
- **Smallest fix:** pass the one preamble string.
- **Sure:** read.

### F18. The run script types molbuilder's scheduler variable names it could take from their owner

- **Places:** `runwrap.py:931` `${MB_NTASKS:-1}`, `:974` `${MB_JOB_ID:-}`,
  `:1075–1077` `MB_JOB_ID`, `MB_NTASKS`, `MB_CPUS_PER_TASK`, `MB_JOB_GPUS`,
  `:2476–2481` the rank chain; `runwrap` already imports
  `scheduler.kinds.JOB_VARIABLES` (`:1030`). `monitor.py:942` and `:2051`
  spell `"MB_JOB_ID"` twice in one module.
- **Owner:** `scheduler/kinds.py:68–75` `JOB_VARIABLES` ("static review keeps
  them to this list" — allowed where `kinds` cannot be imported, the monitor;
  `runwrap` is not such a place).
- **What drifts:** a renamed variable leaves the rank chain skipping the
  scheduler's grant and the dry run printing `?`.
- **Smallest fix:** interpolate the names in `runwrap`; one module constant
  in `monitor`.
- **Sure:** read.

### F19. `PySCFConfig.molwatch_log` is a second name for a field, read by nothing

- **Places:** `config/pyscf.py:286–293` — a property and setter `molwatch_log`
  over `write_molwatch_log`, both `# pragma: no cover`. A grep of
  `molbuilder/`, `tests/`, the catalogue and the page scripts finds no
  `.molwatch_log` reader.
- **Owner:** `write_molwatch_log`.
- **What drifts:** a second spelling waiting to be read.
- **Smallest fix:** delete it.
- **Sure:** grepped.

### F20. `TOOL_TO_CATEGORY` answers "which env runs SIESTA" a second time, unread

- **Places:** `diagnostics.py:78–83` `"siesta": "siesta"`; the engine record
  `data/engine_table.toml:66, :74` — `environment = "siesta"`, `programs =
  { siesta = ..., tbtrans = ... }`, which is what runs use. The only
  `run_tool` / `tool_available` callers are AmberTools'
  (`builders/backends/_amber.py:55, :104`).
- **Owner:** `engine_table`.
- **What drifts:** the engine record moves SIESTA (or adds tbtrans) and this
  row disagrees for any future `run_tool("siesta")`.
- **Smallest fix:** drop the `siesta` row.
- **Sure:** read; callers grepped.

### F21. `script_emit` re-derives one item's keywords and looks the catalogue up three ways

- **Places:** `script_emit.py:663–667` `Parameter.writes` — "`expands` when it
  writes several, the single `anchor`"; `:800–801` `_deck_answers` repeats
  `keys = (tuple(decl.expands) if decl.expands else ((decl.anchor,) if
  decl.anchor else ()))`. The `try: _T.one(_T.catalogue(), name,
  engine=engine) except KeyError: None` lookup appears at `:579–582`,
  `:733–736`, `:834–837`.
- **Owner:** `Parameter.writes` / `parameter()`.
- **What drifts:** a third keyword form taught to `writes` is not read back.
- **Smallest fix:** `_deck_answers` and `deck_values` take the `Parameter`.
- **Sure:** read.

### F22. One deck states two generation times

- **Places:** `script_emit.py:1283–1288` `emit_provenance(...,
  generated_at=generated_at_now(), ...)` and `:1298–1299`
  `emit_atom_metadata(..., created_at=generated_at_now())`.
- **Owner:** one value read once per render.
- **What drifts:** the two can differ by a second; a reader comparing them
  sees two moments for one write.
- **Smallest fix:** read the clock once in `render_deck`.
- **Sure:** read.

### F23. The memory suffixes are written twice in one module

- **Places:** `scheduler/quantities.py:184–185` (`parse_memory`'s inline
  `("T", 1024.0), ("G", 1.0), ("M", 1 / 1024.0), ("K", 1 / 1048576.0)`) and
  `:200` `_MEM_UNITS = {"K": ..., "M": ..., "G": 1.0, "T": 1024.0}`.
- **Owner:** `_MEM_UNITS` (the two readers rightly differ on a bare number,
  not on a suffix).
- **Smallest fix:** `parse_memory` reads `_MEM_UNITS`.
- **Sure:** read.

### F24. `serve_daemon` verifies-and-cleans a pidfile twice

- **Places:** `serve_daemon.py:256–268` (`stop_by_pidfile`: dead → unlink,
  "stale pidfile …"; foreign → "refusing …") and `:656–669`
  (`signal_supervisor`: the same three branches, its own words). The first's
  doc calls itself "the general form of what `signal_supervisor` does".
- **Owner:** one `_verified(path, marker)` helper.
- **What drifts:** a fix to stale handling lands in one.
- **Smallest fix:** `signal_supervisor` calls the shared verify step.
- **Sure:** read.

### F25. `Directives.lines_of` only forwards to the kind

- **Places:** `scheduler/emit.py:120–126` `return
  kind.directive_lines_of(header_text)`; five production callers
  (`jobset/submit.py:371, :725`, `jobset/prep.py:1260, :1275, :2273`).
- **Owner:** `kinds.Kind.directive_lines_of`.
- **Smallest fix:** callers ask the kind.
- **Sure:** read; callers grepped.

### F26. The machines listing re-derives a queue's machine sizes beside the `Domain` property that owns them

- **Places:** `record.py:226–246` `Domain.core_range` / `mem_range` —
  `node_types` else `max_cores`/`max_mem_gb`, "ONE spelling, read by the
  queue table and the machines listing"; `record.py:1083–1088`
  (`known_machines._describe`) repeats the fallback over all domains:
  `cores_of += ([t.get("cores") for t in listed] if listed else
  [d.max_cores])`.
- **Owner:** `Domain` (a per-domain list of sizes it exposes).
- **What drifts:** a third source for a queue's sizes reaches one and not the
  other.
- **Smallest fix:** a `Domain.sizes` the property and the listing both read.
- **Sure:** read.

---

## Looked at and kept out

- `config/` dataclass defaults beside the catalogue's `default`: a measured
  check (read-only Python) found all 89 SIESTA and 62 PySCF items equal, and
  `template.md` § 2.1a records the user's choice to keep them held equal by a
  test. Not a finding.
- `runtime_info.physical_core_count` beside the run script's `lscpu` probe:
  it reads the script's export first (`NODE_CORES_VAR`); one source on the
  road.
- `monitor` / `runtime_info` / `kinds` literals inside modules that travel
  beside a job: allowed by `kinds.py:61–67`.
- `envs/recipes._detect_host_cuda_major`, `builds.detect_nvidia_driver`,
  `_detect_compute_cap`: separate questions, as architecture § 3.2 says.

## The five that matter most

1. **F1** — `VIBRATION_COMPANIONS` / `PYSCF_COMPANIONS` are hand lists beside a
   derived monitor bundle; two members are already missing (round-1 P1's cause).
2. **F2** — the job's GPU list is worked out three times in the run script;
   the dry run, banner and NUMA wrap describe device 0 / an index (round-1
   D1's cause).
3. **F3** — `pipeline_log.config_rows` re-lists `template.SAID`'s keys by hand
   and orders every deck's PROVENANCE by that copy.
4. **F9** — `envs init-config` decides whether to write the activation by
   matching a sentence it composed.
5. **F11** — the source-build layout (`opt/siesta-gpu-stack/<component>`) is
   spelled in `builds`, `validate` and the hook text; a rename leaves SIESTA
   off PATH.
