# F12 review, round 2 — R2-2: the run's files and records

Slice (under `molbuilder/`): runfiles, runs, runpoints, runrecord, calcdirs,
warmfiles, catalogue_tables, paths, projects, checkpoint, persist, task,
task_reader, job_request, describe, calculation, identity, deck_record, frame,
frameset, frame_average, chartdata, ncdata, postprocess, report_cards,
report_fields, stepsdata, structure_pair.

## What I read

**Source, read whole (all 28 modules).** Every module above, top to bottom.
In `runfiles.WRITTEN` and `ENGINE_FILES` a few long `writer=` strings were cut
at 220 characters on screen. No code was cut.

**Followed outside the slice, only as far as a claim needed:**
`atom_index.py` (the callers of `listed_by_of`, `counted_by_of` and `role_of`),
`kind_table.part`/`ROOT_OPENS`/`DESCRIPTION_FIELDS`, `pseudos.CONVENTIONAL_LIBRARY`
and `resolve_psml_lib`, `siesta/warm-files.toml`, `transport/stages.same_volts`,
`transport/record.py` (around line 995), `parse/dirs/record.py` (lines 285-295
and 445-460), `web/blueprints/build.py` (lines 2085-2155), `constants.py`,
`units.py` (its header), and `job-contracts.md` § 2.5a.

**Tests, read whole:** test_runfile_names.py, test_path_framework_doors.py,
test_warmfiles.py, test_warm_file_inventory.py, test_persist.py,
test_calculation_door.py, test_job_request.py, test_task_description.py,
test_chart_table.py, test_frame_set_units.py, test_frame_reaveraging.py.

**Tests, read in part:**
- test_describe.py: lines 1-120 and its test list.
- test_job_request_allocation.py: lines 1-60.
- test_projects.py: lines 1-26 and 140-200.
- test_checkpoint_config.py: head and test list.
- test_checkpoint_manifest.py: test list.
- test_checkpoint_states.py: helpers at lines 50-125 and 185-210, plus greps.
- test_results_blueprint.py: greps only.
- tests/support/road.py: its function list, lines 545-635 and 1570-1590.

**Tests not read:** about 250 other test files import a slice module, most of
them e2e. I checked them only with suite-wide greps for:
- reading source or doc text (`getsource`, `__file__` reads, `docs/...md`);
- hand-typed record schemas (`molbuilder/<record>@`);
- hand-written run records (`run.json`, `calcdir.json`, markers);
- private helpers repeated across files.

The greps turned up nothing in this slice beyond findings T1–T7 below.

---

## Findings — code

### C1 · Three readers say which catalogue row a file is, and they disagree *(medium, sure: both sides read and run)*

- `runfiles.py:620-647`, `_role_of` / `role_of`: works without a label, answers
  dotted roles only, picks the longest one. Its docstring says *"Underscore
  roles are not answered … without a label they cannot be told from a stage
  name."*
- `runfiles.py:2158-2168`, `_row_named`, which `listed_by_of` (2171-2176) and
  `counted_by_of` (2179-2184) both use: also works without a label, but takes
  any role by `name.endswith(a.role)`, underscore roles included, and matches
  fixed names only exactly:
  `found = [a for a in ON_THE_LABEL if (... else name.endswith(a.role))]`.
- `runfiles.py:2715-2744`, `row_for(path, label)`: reads the name back with
  its label. This is the owner door; the run door's `about` and `file_row`
  ask it. `runs.py:1392-1394` says why: *"a name read without its label is
  not taken for one of ours -- SIESTA's `fdf.<stamp>.log` is not PySCF's log"*.

What I measured with read-only Python:

| name | `_row_named` | `role_of` | `row_for(…, "H2")` |
|---|---|---|---|
| `plot.py` | `.py`, counted by pyscf and geometric | `.py` | None |
| `water_initial.xyz` | `_initial.xyz`, listed by engine index | None | None |
| `fdf.20260925T194936.695.log` | `.log` (PySCF's log) | `.log` | None |
| `Au1.psml` | None | None | `{species}.psml` |

**What goes wrong.** `atom_index._of_run` (`atom_index.py:215, 238`) holds the
run, and so its label, yet asks the label-less `listed_by_of(p.name)`.
`atom_index._counted` (`:161-162`) asks `counted_by_of(p.name)` of any file.
So a person's `plot.py` in a run folder gets the sentence *"is an engine's
input: its own text numbers the atoms as pyscf counts …"*. A person's
`<x>_initial.xyz` is said to be listed by engine index.

**Fix.** Keep one reader. `listed_by_of` and `counted_by_of` take the label
the caller holds and go through `row_for`. Drop `_row_named`. A caller with no
label gets `role_of`'s answer, dotted roles only, as `find_by_role` already
rules.

### C2 · One catalogue row goes onto the wire in three shapes *(medium-low, sure)*

- `runfiles.py:2579-2583`, `row_words`, documented as *"the one form of a row
  on the wire"*:
  `{"name", "what", "when", "level", "where", "only"}`.
- `runfiles.py:2695-2696`, `fixed()`, builds its own dict:
  `rows.append({"name": shown(a.name), "what": a.what, "when": a.when, "only": a.only})`.
  It has no `level` and no `where`.
- `runs.py:1206-1209`, `_row_about`:
  `{"ours": True, "what": row.what, "writer": row.writer, "when": row.when, "kind": row.kind}`.

`web/blueprints/build.py:2090-2104` puts the first two into one response: the
`bundle` rows come from `fixed()`, the `once` rows from
`manifest`/`row_words`. `_placed` (2113-2118) reads `r["level"]` on manifest
rows only.

**What goes wrong.** A field added to `row_words` never reaches the Task setup
card's "calculation" group. The page has to treat the two groups of one card
differently.

**Fix.** `fixed()` returns `row_words(a, shown(a.name))`. `_row_about` builds
on `row_words` too and adds `ours`, `writer` and `kind`.

### C3 · `task.py` and `job_request.py` each carry a copy of the same refusal kit *(medium, sure)*

- `task.py:549-616`: `_SOURCE`, `_refuse`, `_as_object`, `_check_keys`.
- `job_request.py:334-356`: the same four, with the same bodies.

```
task.py:549        _SOURCE: contextvars.ContextVar[str] = contextvars.ContextVar("task_refusal_source", default=FILENAME)
job_request.py:334 _SOURCE: contextvars.ContextVar[str] = contextvars.ContextVar("job_request_refusal_source", default=FILENAME)
```

`task.py` also keeps a hand list of the job request's keys:
`task.py:104-105` holds
`_JOB_KEYS = ("allocation", "execution", "bench", "notify")` and
`_JOB_STAGE_KEYS = ("group", "execution")`, beside
`job_request.py:48-49`'s `_TOP_KEYS` and `_STAGE_KEYS`.

**What goes wrong.** Say the job request gains a block (for example
`"reports"`). A description holding it is then refused as an *unknown key*,
not as *"says how it runs … job-request.json"*. A fix to the refusal wording
or the source prefix also lands in one copy only.

**Fix.** The owner of a record's unknown-key refusal is `persist`
(`unknown_keys` / `check_keys`, the § 3.2 row). One shared reader helper there
takes the source name. `task.py` imports `job_request`'s key tuples; there is
no import cycle, since job_request imports task only under `TYPE_CHECKING`.

### C4 · `task.py` checks each ladder rule twice, with words that already differ *(medium, sure)*

Every read runs both copies: the reader first, then `Task(...)` at
`task.py:758`, which runs `__post_init__`.

| rule | reader copy | object copy |
|---|---|---|
| duplicate stage names | `_refuse_duplicate_stage_names` (848-859, called at 752): *"two stages named …"* | 454-466: *"duplicate stage name(s) … compared case-insensitively"* |
| stage name | `_stage_from_obj` 790-792: *"… -- it becomes a filename"* | `Stage.__post_init__` 192-195: *"… -- it becomes a filename and a shell word"* |
| calculation name | 665-667 | 313-318 |
| empty `stages` | 742-746 | 446-450 |

The module docstring says the ladder-level refusals *"live here, in the
codec, because every route to a Task … passes through this constructor"* (451-453).
So the reader's copies are a second source.

**What goes wrong.** A file and an object built by a producer get different
sentences for the same mistake. The tests already pin both copies (see T4).

**Fix.** Keep the dataclass checks. The reader keeps only its shape checks
(`_as_object`, the unknown keys) and wraps the constructor's `ValueError` with
its source prefix.

### C5 · The engine's scratch is named in two places, outside the catalogue *(medium, sure)*

- `siesta/warm-files.toml:94`: `scratch = [".TSGFL", ".TSGFR"]`. These are
  typed suffix tails.
- `warmfiles.py:227-231` makes each glob by hand:
  `tuple(sorted({f"*{suffix}" for engine in engines() ...`
- The catalogue's row for these files is the template
  `SIESTA_TSGF = ".{bias}.TSGF{lead}"` (`runfiles.py:124`). Its lead values
  are `OUR_LEADS` (157-161). `runfiles.family()` (612-617) is *"EVERY NAME A
  ROLE MAKES, as a glob … never a star placed by hand at the call"*.
- `warmfiles.py:392-404` checks that every *row* of the shipped list is one
  the catalogue names. It never checks the `scratch` entries: `.TSGFL` is not
  a declared role.

**What goes wrong.** Rename a lead in `OUR_LEADS`, or add a scratch family as
a typed tail. The checkpoint (`never_carried`), `jobset scratch` and the
take-over then disagree with the deck about what is scratch. A save could then
keep tens of GiB of Green's functions.
`test_warmfiles.py:198-213` (`.{bias}.TSGF{lead}` → `"scratch"`) would catch a
lead rename. Nothing ties the two sources by construction.

**Fix.** The shipped `scratch` list names catalogue roles: the `SIESTA_TSGF`
template and `.contrib.EPSIMG`. The load check refuses any scratch entry that
is not `declared()`. `scratch_families` asks `runfiles.family(role)`.

### C6 · The word "pseudopotential" names two folders, and a README states the wrong one *(medium, sure)*

- `projects.py:54`: `("structure", "pseudopotential")`. This is a project's
  topic, `projects/<project>/pseudopotential/`.
- `pseudos.py:138`: `CONVENTIONAL_LIBRARY = "pseudopotential"`. This is the
  tree's own library, `projects/pseudopotential/`. `job-contracts.md` § 2.5a
  says a relative `psml_lib` is *"that folder under the tree's root"*.
- `projects.py:422-423`, the topic README written into every new project:
  *"A SIESTA calculation whose `psml_lib` is `pseudopotential`\ntakes the files
  it lacks from here at prep."*

**What goes wrong.** The README sits in `projects/<project>/pseudopotential/`
and says `psml_lib = pseudopotential` reads "here". That spelling actually
resolves to `projects/pseudopotential/`. A person who drops `.psml` files into
their project topic and sets `psml_lib = pseudopotential` gets "folder not
there", or another folder's files.

**Fix.** One home for each folder's name. Either the topic README states the
spelling that names it (`<project>/pseudopotential`), or the two concepts get
two names. Say which spelling reaches which folder, using
`pseudos.CONVENTIONAL_LIBRARY` for the library.

### C7 · Two resolvers of `module:NAME`, and a third syntax in the same catalogue row *(medium-low, sure)*

- `runpoints.py:529-546`, `part(ref)`: partition on `":"`, then
  `import_module`, then `getattr`. `runfiles.reader` (1841-1859) calls it
  *"the one resolver of a part a record names"*.
- `kind_table.py:1097-1108`, `part(ref, *, record, key)`: the same parse and
  import, with a different error type. `deck_record.same_calculation` (257-262)
  and `chartdata._variable` (128, 160) use it.
- In one `Artifact` row, `reads` is `"molbuilder.<module>:NAME"` (checked by
  `_check_reads`, `runfiles.py:1877-1905`). `door` is a dotted path *without*
  the `molbuilder.` prefix (`runfiles.py:1033-1036`), resolved by a third
  function, `catalogue_tables.door_resolves` (`catalogue_tables.py:30-44`),
  which tries cut points.

**What goes wrong.** The two `part`s already handle a missing name
differently: `ValueError` versus `KindTableError` naming the record. A change
to the reference grammar would be made in one place only.

**Fix.** One parser of `module:NAME` at floor 1. `kind_table.part` wraps it
for its error. `runpoints.part` keeps only its beside-a-job import rule.
Write `door` in the same `module:NAME` form, checked at import like `reads`.

### C8 · The stage-name character class is spelled four times, and its owner is named wrongly *(low-medium, sure)*

| where | spelling |
|---|---|
| `task.py:95` | `STAGE_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")` |
| `identity.py:409` | `_TOKEN_RE = re.compile(r"(\d{2,})_([A-Za-z0-9_]+)")` |
| `runfiles.py:54` | `_STAGE = re.compile(r"[0-9]{2,}_[A-Za-z0-9_]+")` |
| `runfiles.py:555` | `re.match(r"([0-9]{2,}_[A-Za-z0-9_]+)", after)`, beside `_STAGE` in the same module |

`runfiles.py:49-53` says *"`identity.py`'s ``STAGE_NAME_RE`` and
``stage_token``, which own the word itself"*, but identity has no
`STAGE_NAME_RE`. `tests/test_runfile_names.py:36-37` repeats the wrong owner.

**What goes wrong.** Widen the stage-name set in `task.py` and the grammar
still refuses the new names in filenames, and the folder reader
(`calcdirs.where` via `parse_token`) treats them as support folders.

**Fix.** One constant at the lowest floor, as `NAME_CHARS` is: runfiles, which
identity already imports. `task`, `identity` and `runfiles._STAGE` build from
it. Line 555 uses `_STAGE`.

### C9 · The writer's temp name is spelled apart from the catalogue row that recognises it *(low-medium, sure)*

- `persist.py:194-195`:
  `tempfile.mkstemp(dir=..., prefix=target.name + ".", suffix=".tmp")`.
- `runfiles.py:1767-1771` has the transient row `name="{target}.{random}.tmp"`,
  writer *"`persist` and the codec"*. `is_transient` and `never_carried` read
  it; the checkpoint and every copier exclude by it.

**What goes wrong.** Change the suffix in `persist` (for example to
`.partial`) and a crash's leftover temps are no longer transient. Checkpoints
store them, and downloads and reuse copy them. `test_persist.py:210-211` pins
the shape with literals, not with `runfiles.is_transient` (T5).

**Fix.** `persist` takes its prefix and suffix from the catalogue row, or the
test asserts `is_transient(temp)`. Both modules are stdlib and travel
together.

### C10 · Physical constants are written into unit meanings *(low-medium, sure)*

`ncdata.UNITS` (`ncdata.py:318-351`) writes numbers into the meanings every
data file stores as `unit_meanings` (`_check_stated`, 875):

- `"Eh": "hartree … 27.211386 eV"`;
- `"Eh/bohr": "… (1 bohr = 0.529177 angstrom)"`;
- `"G0": "… 7.748092e-5 S"`;
- `"amu": "… 1.66053907e-27 kg"`;
- `"e"` and `"eV"` as `1.602176634e-19`;
- the debye.

The owner is `constants.py` (`HARTREE_EV` at :40, `BOHR_ANGSTROM` at :37,
`AU_DIPOLE_DEBYE` at :95). The rule in `architecture.md` § 3 table: *"A value
lives in `constants`"*.

**What goes wrong.** A CODATA update in `constants` leaves every data file
stating the old value beside numbers computed with the new one.

**Fix.** Interpolate the meanings from `constants`, as the architecture's
"generated standalone script" rule does. Add any constant still missing to
`constants`.

### C11 · A run record's field path is spelled outside its owner *(low-medium, sure)*

- `runs.py:1118`: `WALL = "computation.time.engine_elapsed_s"`.
  `jobset/runstatus.py:540, 561` asks `record_of(run, fields=(WALL,))`.
- The owner is `parse/dirs/record.py:455`, whose contributor list declares
  `"computation.time.engine_elapsed_s"` and writes it (291-292).

**What goes wrong.** Move the field in the record and `runs.WALL` silently
asks for a path no contributor answers. The status's wall falls back with no
error.

**Fix.** `record.py` defines the constant. `runs` re-exports it or the caller
imports it.

### C12 · `runs._openable` rebuilds the catalogue's `staged` answer by hand *(low, sure)*

`runs.py:1224-1229` reads
`staged = {a.role: a.staged for a in _rf.ON_THE_LABEL}` and then
`... if staged.get(role, True) else _rf.find_on_label(...)`. The owner is
`runfiles._staged(role, door)` (2186-2194), which `RunNames.name` asks. That
one also knows `ENGINE_FILES` and refuses an unknown role, where this map
defaults to `True`.

**Fix.** Ask the runfiles door; make `_staged` public if needed.

### C13 · `longest_tail` fills fields its own way, and its argument is never anything else *(low, sure)*

`runfiles.py:809-823` fills only `{engine}`, through
`r.format(**fill)` with `except (KeyError, IndexError): t = r`. Any other
templated role is measured with its literal `{tbt}` placeholder, five
characters where the longest value is six. `names_made` (2113-2123) is the
module's own filler of every field.

The only callers, `identity.py:143-147` and one test, pass `runfiles.engines()`
back into runfiles. Today it is harmless: the pipeline log is longest.

**Fix.** Use the one filler: give `FIELDS["engine"]` its `values`, and drop
the parameter.

### C14 · `task_reader` spells the kind table's vocabulary word *(low, sure)*

`task_reader.py:61`: `if kind(task.calculation).root_opens != "description":`.
`runs.py:1352` asks the constant: `rec.root_opens == ROOT_OPENS[0]`, with
`ROOT_OPENS = ("description",)` at `kind_table.py:68`. If the word is renamed,
the registry stops claiming a transport `task.json` while the run door still
offers it. **Fix:** import `ROOT_OPENS`.

### C15 · `chartdata` composes the chart file's name twice *(low, sure)*

`chartdata.py:334`: `ds = in_memory(f"{label}.{chart}.nc")`.
`chartdata.py:387-388`: `stem = export_stem(label, chart)` then
`{f"{stem}.nc": data}`, through the owner `ncdata.export_stem` (1270-1275).
**Fix:** compose the name once, in `export`, and pass it in.

### C16 · `frame_average` keeps hand lists beside its own table, and a second voltage rule *(low, sure)*

- `AT_VOLTAGE` (851-853) and `SUMMARY_PART` (843-846) list by hand the
  variable names `data_file_variables()` (753-811) already declares over
  `bias_v`. A new average variable has to be added in three places.
- `reaveraged` (969) uses `at_voltage(..., lambda a, c: a == c)`. The owner is
  `transport/stages.same_volts` (`stages.py:89`), documented as *"THE rule a
  reader asks … (the record, its average, the data file)"*.
- `average_block` (415) computes `math.fsum(weights)` again, while
  `FrameRows.weight_sum` (`frameset.py:50-52`) gives the same number.

**Fix:**
- Derive the two lists from the variables' dimensions.
- Pass `same_volts`, or default the parameter to it.
- Use `weight_sum`.

### C17 · The voigt axis's fixed labels live in a writer, not in the axis table *(low, sure)*

`stepsdata.py:30` has `VOIGT = ("xx", …)`. Lines 115-116 hand-declare the
`voigt` coordinate, and line 236 fills it. `ncdata.AXES` (229) holds
`Axis("voigt", …, "")` with no definition and no labels. Its own pattern for
an axis with fixed points is a definition plus `labels`, used by `xyz` and
`cell_vector` through `coordinate()` and `axis_values()` (235-253, 304-310).
**Fix:** put the labels in the `voigt` row and use `coordinate("voigt")`.

### C18 · `checkpoint` repeats two small things *(low, sure)*

- `checkpoint.py:727-733`, `manifest_digest`, is
  `hashlib.sha256(raw).hexdigest()`. `persist.sha256_of_bytes` (391-395) is the
  door, and `sha256_of` from the same module is already imported (line 55).
- `git ls-files -z` is run and parsed in three places with the same
  comprehension: 1568-1570, 1743-1745 and 1889-1891.

**Fix:** use `sha256_of_bytes`, and add one `_tracked(path)` helper.

### C19 · `deck_record` has a second reader of the block fences *(low, sure)*

`deck_record.py:274-300`, `read_parameters_fence`, finds its block by exact
string: `if stripped == begin:`. The module's own fence grammar is `MARKER_RE`
(73-74), which allows variations in whitespace, and `_first_pair` (122-140).
A fence echoed with other spacing is found by one reader and missed by the
other. **Fix:** detect fences with `MARKER_RE`, keeping its every-block loop.

### C20 · Three one-line run records repeat one reading rule *(low, plausible)*

`runrecord.read_continued_from` (118-138), `read_gathered_from` (515-543) and
`read_copied_from` (602-620) each repeat the same steps:
- `FileNotFoundError` gives `None`/`[]`;
- `OSError` or `UnicodeDecodeError` gives `LaunchRecordError`;
- empty text gives `LaunchRecordError`.

Only the line grammar differs. **Fix:** one `_record_text(path, what)` helper
under the three readers.

## Findings — tests

**T1 · A hand-copied description where the road should write one (medium, sure).**
`test_task_description.py:42-49` reads `tests/data/task_example.json`, *"copied
… the document and the file are kept in line by review"*. Its siblings use
the road: `test_job_request.py:22-40` and `test_calculation_door.py:26` call
`support.road.described`. **Drift case:** the writer changes a key's spelling
or order. The example still reads, and the "byte-identical round trip"
(52-58) passes on bytes the writer no longer makes. **Fix:** take the
description from `described(road_tree(...))`, as the job request's tests do.

**T2 · Two test files pin the same runfiles doors (low-medium, sure).**
`test_path_framework_doors.py:63-95` covers the round trip, the field refusal
and label-less `find_by_role`. `test_runfile_names.py` covers the same:
- the round trip over the whole product, 50-65;
- the field refusal, 172-181;
- `TestFindingARoleWithNoLabel`, 446-466.

`test_runfile_names.py`'s own docstring says it is *"one file rather than a
check beside each writer"*. **Fix:** fold the doors file's cases into it as
rows.

**T3 · The same rule is pinned twice, once by a text grep (low, sure).**
Both files have a test called `test_the_description_names_no_machine`:
- `test_describe.py:60-77` checks the template holds no run setting;
- `test_task_description.py:294-305` greps the written `task.json` text for
  `"host"`, `"partition"`, `"queue"` and similar words.

The grep is an assertion on output text, and it also fails on a legal
structure name such as `node`. **Fix:** keep the outcome test and drop the
grep.

**T4 · One refusal pinned twice because the code has two copies (low, sure).**
`test_task_description.py:108-121` covers the reader path; 473-479 covers
`Task.__post_init__`. This follows C4. **Fix:** once C4 is fixed, keep one.

**T5 · The unique temp is pinned twice, by a literal shape (low, sure).**
`test_persist.py:175-187` (no `cfg.json.tmp` left) and 195-211
(`"cfg.json." in seen["src"] and seen["src"].endswith(".tmp")`) both pin it.
Neither asks the catalogue's transient row. This is the test side of C9.
**Fix:** one test asserting `runfiles.is_transient(temp)`.

**T6 · A hand stand-in for the voltage door (low, sure).**
`test_frame_set_units.py:33, 36, 54, 72` and `test_frame_set_meanings.py` pass
`lambda a, b: a == b` where production passes `transport.stages.same_volts`.
**Fix:** pass the door.

**T7 · A misnamed test that greps generated text (low, sure).**
`test_warm_file_inventory.py` has nothing to do with warm files. It checks
that the generated run script has no `cd`, using a regex over the script's
text (`re.findall(r"(?m)^\s*cd\s+\S+", ...)`). **Fix:** name the file for what
it checks, and keep it as the one lint of its kind.

---

**Count: 27 findings (20 code, 7 test).**
