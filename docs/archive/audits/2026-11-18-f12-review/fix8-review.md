# F12 fix 8 (the run script and the machine, 2.77.16, R2) — the two review rounds

**Verdict: closed with 2.77.16.** Round-1 defects B-D1..B-D4, C-D2, C-D3 and
O-D1..O-D9 (`r1-B-runscript.md`, `r1-C-reports-layout.md`,
`r1-H-builders-validation-ops.md`; `v-ABC.md`, `v-H.md`), the user's two
answers of 2026-11-18 -- a SIESTA vibration keeps the structure's notes ("Keep
them, as PySCF does"; F-U1) and a card layout copy that does not read stops only
a run that sends reports (C-OBS) -- and the user's "Two zones" for your own lines
in the run script (2026-11-19). Round 2's three safety items had shipped in
2.77.7. Two rounds of read-only readers in three parts: the run script and its
monitor (a correctness and a discipline reader), the machine -- installer, envs,
server verbs (the same two) -- and the notes (one reader, both questions); round
2's readers new.

## Round 1

| finding | verdict |
|---|---|
| SIESTA's wall counted before the zone before the engine; the zone after the point's start | fixed: the zone before the point's start and the wall's count, after the dry run's exit |
| the zone after the engine ran before the run's ending was read, before a point's end and the finish | fixed: after them, before the conclusion |
| the zones ran under `set -u`; a failing shell function skipped the ERR trap; a line that does not parse, or a copied marker line, found only when the job ran | fixed, then taken out at the user's word (2026-11-20: the script is the person's to edit; nothing is built around the lines they write) |
| the MPS test's skip marker moved onto the new GPU test | fixed |
| the monitor read the job's start and set the countdown twice; `main` let a refused layout trace back | fixed: `_job_started`, `_set_wall`; `main` returns 1 |
| a typed `--projects` skipped the folder-name check on an existing file; a typed `conda activate` said nothing of its hook | fixed: one check for both roads; the preamble rule wherever an activation is written |
| doctor's rebuild advice for the host env, `<recipe-name>` in bootstrap's next steps, `...` in the ABI fallback | fixed |
| the recipe's name optional where a fix is printed | fixed: a required keyword |
| four wordings of "no package manager" | fixed: one sentence, in the user's words |
| the pidfile read by two readers; status's address chosen by each caller; werkzeug's family rule copied | fixed: one reader; the probe reads the host itself; werkzeug asked |
| the install script's Python floor written five times | fixed: read from pyproject.toml |
| a test's TMPDIR assertion moved off its test; the named-fix and step-folder tests at the leaf | fixed: restored; through `preflight`, `install --dry-run`, `run_step` |
| a note holding U+2028/U+2029/U+0085 made the block unreadable after the force constants | fixed: the one JSON writer escapes them |
| dates, format versions, chapter rows for the notes | fixed |
| the monitor samples every GPU, not the job's | left for R3 (the monitor's one-home items): one GPU here, no test can tell them apart, and the system-load page shares the sampler |

## Round 2

| finding | verdict |
|---|---|
| a typed activation changed molbuilder.json while the record prep reads kept the old one, unsaid | fixed: the record's step names the difference and the probe that copies it |
| `clean`'s no-manager line named `validate`; `validate` and `install --check` had none; the sentence told a person to load a manager when the recorded one was the fault | fixed: one guard for every verb; the sentence names the recorded path when it is the cause; doctor, the notebook and the install script say the same |
| the branches for a config file that does not read were unreachable from the command line | removed: the reader's own refusal stands, before anything is asked |
| "what is asked" computed in three places; a typed tree refused only after the installs | fixed: `initconfig.unanswered`; the tree checked up front |
| an older supervisor that kept no host was told to restart | fixed: the note says stop then start |
| a point's own step failing after a clean engine concluded before the person's lines after it; the zone's failure line said "not started" for a failure inside `$( )` | fixed, then taken out with the zones' other machinery (the user, 2026-11-20) |
| a hand-edited wall in the launch record crashed the monitor | fixed: a `[WALL]` line |
| the marker count and `bash -n` each written twice | gone with prep's check of the zones (the user, 2026-11-20) |
| the chapter said a structure's notes feed no engine input | fixed: the run record's charge and spin (ES7) |
| tests: the conclusion's order, a test row that could not fail, the start card through a private entry | fixed |
| the thread count or the GPU list exported in the zone before the engine | left: a use imagined, not on molbuilder's road (the user, 2026-11-20) |
| `install_cmd` handed an env name in two run-script messages | left for R3 |
| a structure title pasted with U+2028 cannot be saved | left for R3 (the structure's title door) |
