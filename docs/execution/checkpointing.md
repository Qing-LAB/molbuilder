# Checkpointing — the invariants a run directory's history must hold

**Role:** contract
**Domain:** execution
**Companions:** [`execution/running-a-job.md`](?doc=execution/running-a-job.md)
— the guide (`running-a-job.md § 6`): what `molbuilder snapshot` is for and how
to drive it, which this document never repeats;
[`execution/job-contracts.md`](?doc=execution/job-contracts.md) — the registry
entries for `.mbcheckpoint.json` and the archive MANIFEST
(`job-contracts.md § 6.1`), and the run directory being checkpointed
(`job-contracts.md § 2`);
[`engines/stages.md`](?doc=engines/stages.md) — the folder this history is taken
of, and the two boundaries where it is taken (`stages.md § 7.3`);
[`execution/run-identity.md`](?doc=execution/run-identity.md) — the id every
commit and tag carries.

**Status: the shipped half is a contract; the added half is proposed.** § 2–§ 4
state invariants the shipped `Repo` core already holds and that a change must not
break. § 5 states what a staged folder adds and is not built yet. Every invariant
is written so a test can assert it — that is the point of the document.

**This contract owns:** what must never change in a checkpointed run directory,
and what information must be kept separate from what.

---

## 1. Why invariants rather than a description

The checkpoint history is what makes a folder of stages safe to rewrite, safe to
branch and safe to come back to (`engines/stages.md § 7.2`, `§ 7.3`). Everything
downstream of it — replacing a produce, redoing a stage, forking a what-if —
assumes the history is *complete and honest*. A history with a hole in it is
worse than none, because the hole is invisible until the moment somebody needs
what was in it.

So this document is a list of things that must always be true, each with the
failure it prevents and the check that catches it. It is the review sheet for the
code, not a tour of the feature.

---

## 2. The separations — what must be kept apart

### S1 — a file is git-tracked **or** content-archived, never both, never neither

`.mbcheckpoint.json`'s `archive_globs` classify a run directory's files: text
(including the small warm-restart `.XV` / `.CG`) is committed to git; the large
binaries are gitignored and archived by content under `.binsnapshots/<sha>/`.

| | |
|---|---|
| **How it fails** | A file matching an archive glob that is *also* tracked puts a multi-gigabyte blob in git history forever. A file matching neither is in no snapshot at all — a restore silently does not bring it back |
| **How to check** | For every file in a checkpointed directory: `matches_archive_glob(f) XOR git_tracked(f)` is true. Assert it over a fixture directory containing one of each engine's warm files plus a text log |

### S2 — shared state lives above; a stage writes only inside its own directory

A staged folder holds shared files once at the parent and one subdirectory per
stage (`engines/stages.md § 7.1`). Cross-contamination is exactly what that
separation exists to prevent.

| | |
|---|---|
| **How it fails** | A stage writing through an inherited symlink overwrites the *producing* stage's result, and the history then records one stage's outputs replacing another's with no diff that says so |
| **How to check** | After a stage runs, every file it created or modified is inside its own subdirectory. The shipped guard is localize-on-run — the wrapper replaces an inherited symlink with a real copy before the engine starts (`job-system.md § 5.2`) — so the direct assertion is: **no regular file outside a stage's subdirectory has an mtime inside that stage's run window** |

### S3 — inherited and owned are distinguishable on disk

A carried file is a **symlink** until its producer has run and the consumer
localizes it; after that it is a **regular file** the consumer owns.

| | |
|---|---|
| **How it fails** | If the distinction is lost, nothing can tell a stage that inherited a geometry from one that computed it, and a checkpoint records the same bytes with two different meanings |
| **How to check** | Before a stage runs, its carried entries are symlinks (possibly dangling — that is correct, `engines/stages.md § 7.4`). After it runs, they are regular files. The checkpoint diff across a run shows a type change, and that type change is the record |

### S4 — the description is input; everything else in the folder is derived

`stages.json` is written by the user's surface and read by the generator. Decks,
wrappers, links and outputs are derived from it.

| | |
|---|---|
| **How it fails** | A produce or a run that edits the description makes the folder self-modifying: the file that is supposed to explain the folder becomes a consequence of it, and reopening no longer restores intent |
| **How to check** | The description's bytes are unchanged by any produce that did not receive a new one, and by every run. Hash it before and after |

### S5 — identity is calculation-level; the run index is invocation-level

The id names the calculation; `-run0`, `-run1` name invocations of it
(`run-identity.md § 2`).

| | |
|---|---|
| **How it fails** | If anything about a run could change the id, the warm files it produced would be orphaned by the act of producing them |
| **How to check** | No code path derives an id from a run's output, a timestamp, or a run index. `stages.json`'s `run.id` is read, never recomputed (`run-identity.md § 3`, rule 1) |

---

## 3. The immutabilities — what must never change

### I1 — a written checkpoint is never modified

Git commits are immutable by construction; the binary archive must be too.

| | |
|---|---|
| **How it fails** | An archive directory rewritten in place means an old commit's binaries silently become a newer run's. Every restore before that point returns the wrong data, and nothing reports it |
| **How to check** | `.binsnapshots/<sha>/` is created once and never written again. A second checkpoint whose binaries are identical **dedupes by content** rather than rewriting; a second checkpoint whose binaries differ writes a *new* directory |

### I2 — a MANIFEST is authoritative for its archive

The 3-column `<sha256>  <bytes>  <name>` MANIFEST (`job-contracts.md § 6.1`)
describes exactly what is in that archive.

| | |
|---|---|
| **How it fails** | A MANIFEST that names a file the archive lacks turns a restore into a partial one; a sha that does not match turns it into a silent corruption |
| **How to check** | For every entry: the file exists, its size equals the recorded bytes, and its sha256 equals the recorded sha. Run it over every archive in a repository — this is the single most valuable test in the system |

### I3 — molbuilder never deletes a warm-restart file

`--cold` **moves warm files aside** into `<basename>-restart-aside-<UTC>/`; it
does not delete them (`job-contracts.md § 4.1`). No other operation removes them —
including a replacing produce, which may remove orphaned decks and wrappers but
never state (`engines/stages.md § 7.2`).

| | |
|---|---|
| **How it fails** | Hours of converged geometry destroyed by an operation whose job was to write an input file |
| **How to check** | Grep the produce, the cold-restart path and the checkpoint code for any unlink or rmtree whose target can match a warm-file suffix. There should be exactly one mover and no deleter |

### I4 — a generated wrapper contains no git

`running-a-job.md § 6.2` records that the wrapper-bootstraps-git path was
deliberately dropped: *"the wrapper is deliberately git-agnostic, so init is
CLI/UI-only."* A wrapper that committed would need git on the compute node, which
`running-a-job.md § 2`'s standalone contract forbids.

| | |
|---|---|
| **How it fails** | A run that dies on a node without git, for a reason having nothing to do with the calculation |
| **How to check** | No emitted `.run.sh` or `.sbatch` contains the string `git`. One grep over the rendered wrapper fixtures |

---

## 4. The atomicity rules — a mutation completes or does not happen

### A1 — archiving is build, verify, then swap

The shipped sequence is: build into a `.tmp`, hash, copy, **re-hash and verify the
copy**, write the MANIFEST, then `os.replace` (`running-a-job.md § 6.1`).

| | |
|---|---|
| **How it fails** | An interrupted archive leaves a directory that looks complete and is not, and I2's check would be the only thing that ever noticed |
| **How to check** | Kill the process between each step of a checkpoint; afterwards the archive set is either the old one or the new one, never a mixture |

### A2 — restore verifies before it mutates

Restore refuses on a dirty text tree, refuses on dirty binaries (sha-compared
against HEAD's archive), **verifies the target ref's archive before touching
anything**, and only then restores the worktree and copies binaries back.

| | |
|---|---|
| **How it fails** | A restore that half-completes leaves a worktree from one commit and binaries from another — a state no commit ever held, which nothing can diagnose afterwards |
| **How to check** | Corrupt one byte of the target ref's archive and attempt a restore: it refuses, and the worktree is byte-identical to what it was |

### A3 — the checkpoint precedes the mutation it protects

A pre-produce checkpoint is committed **before** the first file of the new produce
is written (`engines/stages.md § 7.3`).

| | |
|---|---|
| **How it fails** | Taken afterwards, it records the state it was supposed to preserve a way back to |
| **How to check** | The commit's timestamp precedes every written file's mtime, and a produce that fails partway leaves the checkpoint but no new files (which is also `engines/stages.md § 7.2`'s transactional rule) |

---

## 5. What a staged folder adds — proposed

Not built. Each of these follows from the layout in `engines/stages.md § 7.1`.

### P1 — one repository, at the parent

`job-system.md § 5.5` observes that each per-stage directory is a self-contained
run directory and so can be checkpointed alone. That stays true and is not enough:
a per-subdirectory repository cannot restore a shared file that lives **above** it
— a restored stage would have pseudopotential links pointing at nothing — and no
such repository contains the workflow, so *branch at stage 2* cannot be
expressed. **The repository is at the parent, and there is exactly one.**

*Check:* exactly one `.git` and one `.mbcheckpoint.json` in a produced folder, at
the top of it.

### P2 — the archive globs match at depth

Engine defaults are `*.DM`, `*.HSX`, … (`running-a-job.md § 6.1`) — written for a
flat directory. In this layout the binaries are at `<stage>/<id>.DM`, which those
patterns do not match, so **S1 breaks in the worst direction**: every large
binary lands in git as a blob.

*Check:* produce a two-stage folder, run both, checkpoint, and assert S1 over the
whole tree. It fails today; that failure is the acceptance test.

### P3 — every commit and tag names its calculation

The commit message carries the id and the stage; a finished stage is tagged
`<id>/<stage>/<UTC>` (`engines/stages.md § 7.3`). A folder can be moved, copied
to a cluster, or opened a year later, and a history whose commits say only
*"stage 2 converged"* cannot say which calculation that was.

*Check:* every commit message contains the folder's `run.id`, and every tag parses
into exactly three parts of which the first equals it.

### P4 — the tag namespace is stage completions only

Pre-produce checkpoints are commits, reachable through `snapshot list`. Tagging
them too would bury the points a user meant to reach among the ones they passed
through.

*Check:* the number of tags equals the number of observed stage completions.

---

## 6. The review sheet

One line each, for reading over a diff:

| | Invariant |
|---|---|
| **S1** | tracked XOR archived — never both, never neither |
| **S2** | a stage writes only inside its own directory |
| **S3** | inherited is a symlink; owned is a regular file |
| **S4** | the description is never modified by a produce or a run |
| **S5** | nothing a run produces can change the id |
| **I1** | a written archive is never rewritten; identical content dedupes |
| **I2** | every MANIFEST entry matches the file: name, size, sha256 |
| **I3** | warm files are moved aside, never deleted, by anything |
| **I4** | no generated wrapper contains git |
| **A1** | archive: build, verify the copy, then swap |
| **A2** | restore verifies the target archive before touching the worktree |
| **A3** | the checkpoint is committed before the mutation it protects |
| **P1** | one repository, at the parent |
| **P2** | archive globs match at depth |
| **P3** | every commit and tag names its calculation |
| **P4** | tags are stage completions only |

---

## 7. What this contract does not own

- **How to use the checkpoint system** — the CLI verbs, the HTTP routes, the
  sidebar panel, and what is still unbuilt (archive pruning, a `snapshot diff`
  CLI face) — [`running-a-job.md`](?doc=execution/running-a-job.md) `§ 6`.
- **The `.mbcheckpoint.json` schema and the MANIFEST column format** —
  [`job-contracts.md`](?doc=execution/job-contracts.md) `§ 6.1`.
- **The folder being checkpointed, and when a checkpoint is taken** —
  [`engines/stages.md`](?doc=engines/stages.md) `§ 7`.
- **Phasing and open questions** —
  [`web/staged-runs-architecture.md`](?doc=web/staged-runs-architecture.md) and
  [`roadmap.md`](?doc=roadmap.md) (R3).
