# F12 fix 9 (the jobs, run files and records, 2.77.17, R3a) — the two review rounds

**Verdict: closed with 2.77.17.** Round 2's one-home items for the jobs
(`r2-1-jobset.md` F1, F4–F13, F15–F17, F19, T1) and for the run files and
records (`r2-2-runfiles-records.md` C1–C4, C7–C14, C16–C20, T4, T5), and
`r1-A-jobset.md` U6, each moved onto the door its chapter names. Two rounds of
read-only readers in three parts: the jobs (a correctness and a discipline
reader), the run files and records (the same two), and the notes and tests
(one reader); round 2's readers new, each comparing every changed answer with
2.77.16's on the names a real run writes.

**What round 1 caught that matters most.** C5 had made the engines' scratch
globs with the catalogue's glob builder (`*.*.TSGF*`): the walk's quoted shell
glob matched nothing, so a transport scan would have kept every voltage's
Green's functions; TranSIESTA's default-named ones stopped being scratch, and
a person's look-alike files became scratch. Reverted: the shipped list's
suffixes are the true statement (TranSIESTA's default name is in no catalogue
row), and a walk test now plants the names a device run writes. The cause, and
the guard kept since: a finding's proposed fix taken without its premise
checked against the chapter, and a changed answer not compared with the old one
on real names.

## Round 1

| finding | verdict |
|---|---|
| C5: the scratch globs made by the catalogue's glob builder — the walk's emptying matched nothing; TranSIESTA's default names no longer scratch; look-alike names scratch | reverted; a walk test plants a device run's Green's functions and fails on the C5 globs |
| C17: the voigt labels moved into `ncdata.AXES`, against `model/datafiles.md` § 3a (a one-file axis's coordinate is its writer's) | reverted |
| the temp file's `{random}` field matched any text, so a person's `notes.old.tmp` was saved by no state (since 2.31.1) | fixed: mkstemp's characters and count, stated once (`runfiles.MKSTEMP_CHARS`, `MKSTEMP_LENGTH`), the field's shape and its glob (`Field.glob`) made of them |
| the Projects page's .gitignore note said "edited by hand" where a release's patterns differ | fixed: the chapter's words (I2b), as the terminal says them |
| F1: the save step in `checkpoint`, writing the jobs' ledger (`checkpointing.md` § 9: the checkpoint knows a folder as files) | moved to `jobset/ledger`; the verb is handed the state saved before the writes, so a launch whose write stops prints the saved line again |
| F9: `probe_this_machine` made `scheduler.record` import `diagnostics` | moved into `diagnostics`, beside `local_facts` |
| F8: `summed_walls`, `largest_memory` in `quantities`, whose module states no workflow rule | moved into `jobset/group`, beside their two callers |
| F4: the trial name's letters in two tables | `template.GRID_AXIS_LETTERS`, in the name's order; `GRID_AXES` and `resolve.MACHINE_AXES` from it |
| F17: the launch table read by position | named fields, `LaunchField` |
| F19: the chapter still described the deleted reading; a non-object entry read as unrecorded | the chapter says the rule; a non-object entry is a record that does not read |
| A-U6 leftovers ("The read-only verb", "read-only" in the flowchart) | reworded |
| the Task setup group row's "; or" followed no first way out | the way out a clause each sentence joins |
| C7: a new `runfiles.resolve_part` beside the resolver `architecture.md` names | back to `runpoints.part`; `kind_table.part` asks it and says its cause |
| C1: the row reader answered fixed names only where no label is known | by name: a fixed name, else a dotted role |
| C11, C8, C10, C16, C18; persist's travel declaration; the duplicate refusal naming the repeat as written; names | fixed |
| left | prep's other unresolved `relative_to` (the same path whenever both come from the calculation's folder); the CLI's click options and `fit.py`'s words (`fit.py`'s floor cannot import `placement`); Task setup Save's own save-then-write (no ledger); `migrate._JOB_KEYS` (the layout before the split); fence readers in `pyscf/script_input` and `script_emit`, the generated scripts' constants, `template._refuse` (R3c); the catalogue's `door` column (a citation column people read) |

## Round 2

| finding | verdict |
|---|---|
| the temp glob `????????` admits more than the shape: `notes.Draft_v2.tmp` ignored by git while the rule keeps it; the zip and the rsync download disagreed | fixed: the glob is the shape's characters, eight times; the test asks the families the `.gitignore` and the exclude line are written from |
| status's warm column read the stage's own job list — empty for a stage set clean, a trial, a transport lead (round 1's fix, against `job-contracts.md` § 4.2a) | the chapter's: the kind's list in effect, composed on the run's label; the test plants another kind's file |
| the .gitignore note said "an earlier release" | "another release", the chapter's and the terminal's |
| five places still called status read-only | each says the one thing it writes |
| the duplicate tests did not see "named as written" | they assert each repeat as written |
| Task setup's "For the whole run" rows each said "in the calculation's folder" (C2's wire form) | placed as the card's other group is (`task-setup.md` § 7.2) |
| `catalogue_row`'s note: its section, "atom count", a description re-read | its question said; `runfiles.row_without_label` the by-name door; the place passed where held; both door tables |
| the save step's home unsaid in the ledger's note and three chapters; its `record` parameter shadowing the module's function | said; `say` |
| copies: two sentences typing the launch flags, one typing the trial letter "G"; the prepared-with keys spelled three ways | asked of their tables |
| a note naming `resolve_task`; `why_not_join`'s note backwards; a dead branch; `__all__`; blank lines | fixed |
| a PySCF run's pair in a calculation whose description does not read now opens as the person's pair | accepted, said in the changelog: PySCF sorts nothing |
| left | `{pid}` and `{tbt}` globs wider than their shapes (no one glob spells them; no person's name meets them); a third `module:NAME` reader in `builders` (R3e); `runwrap`'s split of a part's module (R3b) |
