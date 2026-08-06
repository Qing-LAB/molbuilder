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

**Status: mixed, and each invariant says which it is.** Most hold of the shipped
`Repo` core today and a change must not break them; a few depend on the staged
layout (`engines/stages.md § 7.1`) and cannot be asserted until it exists. The
split is *not* by section — § 2 contains both — so every invariant carries a
**holds today** or **needs the layout** marker, and § 6's sheet collects them.
Every one is written so a test can assert it; that is the point of the
document.

**This contract owns:** what must never change in a checkpointed run directory,
and what information must be kept separate from what. Where the history sits in
the project tree, and why it sits there, is
[`execution/project-layout.md`](?doc=execution/project-layout.md) § 5.

---

## 1. Why invariants rather than a description

The checkpoint history is what makes a folder of stages safe to rewrite, safe to
branch and safe to come back to (`engines/stages.md § 7.2` and
`engines/stages.md § 7.3`). Everything
downstream of it — replacing a produce, redoing a stage, forking a what-if —
assumes the history is *complete and honest*. A history with a hole in it is
worse than none, because the hole is invisible until the moment somebody needs
what was in it.

So this document is a list of things that must always be true, each with the
failure it prevents and the check that catches it. It is the review sheet for the
code, not a tour of the feature.

### 1.1 What is shipped, and what this design changes

The `Repo` core, the `molbuilder snapshot` CLI, the HTTP routes and the sidebar
panel all ship (`running-a-job.md § 6`). Read this document as a statement of
what that code must keep doing, plus the six changes a staged folder needs:

| | Shipped today | With a staged folder |
|---|---|---|
| **What it covers** | one flat run directory | a parent and its per-stage subdirectories — **one repository, at the parent** (L1) |
| **Archive globs** | `*.DM`, `*.HSX`, … — matched beside the config | must match **at depth**; today's patterns miss `<stage>/<id>.DM` and every binary lands in git as a blob (L2) |
| **When a checkpoint is taken** | when the user runs `snapshot checkpoint` | **automatically**, before a replacing produce and when a stage's run finishes (`engines/stages.md § 7.3`) |
| **What it is called** | a free-text message; a tag name the user picks | the message carries the id and the stage; a stage completion is tagged `<id>/<stage>/<UTC>` (L3) |
| **Who initialises** | `snapshot init`, always explicit | a produce that *creates* the folder initialises it; one writing into a folder that already existed without a checkpoint does not |
| **`branch`** | CLI only — no HTTP route (`running-a-job.md § 6.2`) | the operation the staged design turns on, so the route is required rather than nice to have |

Everything else — the text/binary split, the archive format, build-verify-swap,
verify-before-restore — is unchanged, and §§ 2–4 exist to say so precisely enough
that a change can be checked against them.

---

## 2. The separations — what must be kept apart

### S1 — a **regular file** is git-tracked **or** content-archived, never both, never neither

*holds today.*

`.mbcheckpoint.json`'s `archive_globs` classify a run directory's files: text
(including the small warm-restart `.XV` / `.CG`) is committed to git; the large
binaries are gitignored and archived by content under `.binsnapshots/<sha>/`.

**Where a file sits decides whether it is ours.** `project-layout.md § 1.1`
makes every directory a *container* (setup — text, git's) or a *run* (what one
invocation produced). The archive covers the runs this calculation owns: a flat
root, or a stage's `run-N/`. A nested container's runs — a benchmark's
`point-*/`, two levels down — hold five-iteration throwaways and are not this
history's business. **Depth, not names, and no marker file.**

**No dotfiles either, and writer and reader must agree on that.** The MANIFEST
parser rejects a dot-prefixed component anywhere in a key, so the archive walk
must skip them at every level — basename included. It filtered only the parent
directories until 2026-08-06, and pathlib's `*` matches a leading dot, so a file
like `.hidden.DM` was archived under a key the parser refuses: **the archive wrote
a MANIFEST its own reader could never accept**, making that checkpoint
permanently unverifiable and unrestorable.

**Symlinks are outside this too, and the word "regular" is load-bearing.** A carried
restart file is a link to the stage that produced it, and a link has no content
of its own — the producer's file is archived once, under its own key. Git ignores
the link too (a gitignore pattern matches by name, not by file type), so a link
is in *neither* set, and that is correct rather than a hole: the layout is
reproducible from the description and the carry rules
(`engines/stages.md § 7.1`), while the bytes are already archived once. Archiving
the link as a second copy would duplicate content *and* restore a regular file
where a link belongs — S3, from the other side.

| | |
|---|---|
| **How it fails** | A file matching an archive glob that is *also* tracked puts a multi-gigabyte blob in git history forever. A file matching neither is in no snapshot at all — a restore silently does not bring it back |
| **How to check** | For every file in a checkpointed directory: `matches_archive_glob(f) XOR git_tracked(f)` is true. Assert it over a fixture directory containing one of each engine's warm files plus a text log |

**An archive is now always written, even with nothing to put in it.** A missing
archive directory used to mean two things — *this commit had no big binaries* and
*this commit's archive is lost* — which is why `missing_archive_warning` has to
guess from whether **other** commits have archives, its docstring conceding that
*"a lost archive cannot be proven"*. Every commit now gets a directory and a
MANIFEST, empty when there was nothing to archive, so absence is evidence rather
than a hint. (Repositories predating this have legitimately absent archives, so
the warning stays a warning; promoting it to an error is a later step.)

### S1a — the ignore set is *derived* from `archive_globs`, never kept beside it

*HOLDS as of 2026-08-06 — it did not, and my earlier reading of it was half
right.*

S1 rests on two lists agreeing: `archive_globs` in `.mbcheckpoint.json`, and what
git ignores in `.gitignore`. **One list, one writer**, or the agreement is a
coincidence somebody has to maintain.

`set_archive_globs` was already that writer and says so — *"derived from the same
resolved globs, so they never drift"*. **`init` was not.** It wrote `.gitignore`
only `if not gi.exists()`, so a directory that already had one — every benchmark
bundle, every directory anyone had worked in — kept an ignore file that said
nothing about the archive set. The result is S1's *other* branch: the file
archived **and** committed, a large blob in git history forever.

Both now go through one writer that keeps a **marked section** and leaves
everything outside it byte-for-byte. It also excises an unmarked block written
before the markers existed — appending beside one would leave the old patterns in
force, so narrowing the classification would leave a file ignored and no longer
archived, which is the losing branch again.

| | |
|---|---|
| **How it fails** | A second writer. Any future path that edits `archive_globs` without going through that API, or hand-edits `.gitignore`, makes a formerly-archived pattern start committing as a blob or a newly-archived one keep being tracked — **S1 broken by a configuration change rather than by a bug**, which is the kind nobody thinks to test |
| **How to check** | Grep for writers of `archive_globs` and of `.gitignore`: there is exactly one of each, and they are the same function. Then a regression test: `config --set` changes both files, and a fixture whose two disagree is *reported*, not obeyed |

### S2 — shared state lives above; a stage writes only inside its own directory

*needs the layout.*

A staged folder holds shared files once at the parent and one subdirectory per
stage (`engines/stages.md § 7.1`). Cross-contamination is exactly what that
separation exists to prevent.

| | |
|---|---|
| **How it fails** | A stage writing through an inherited symlink overwrites the *producing* stage's result, and the history then records one stage's outputs replacing another's with no diff that says so |
| **How to check** | **Two halves, and one alone is worse than useless.** (a) Checkpoint the folder, run one stage, read `git status` at the parent: every changed path is under that stage's subdirectory. (b) **`git status` cannot see the big binaries** — they are gitignored, which is S1 working as designed — so compare each archived file's sha against the head archive's MANIFEST for the same thing. The shipped restore already needs exactly this and has the helper (`_working_binaries_dirty`), whose own comment says *"big binaries are gitignored, so `git status` cannot see them"*. Half (a) alone would pass while a stage overwrote another stage's `.DM` — the single most valuable file it could destroy. The shipped guard is localize-on-run: the wrapper replaces an inherited symlink with a real copy before the engine starts (`job-system.md § 5.2`) |

### S3 — inherited and owned are distinguishable on disk

*needs the layout.*

A carried file is a **symlink** until its producer has run and the consumer
localizes it; after that it is a **regular file** the consumer owns.

| | |
|---|---|
| **How it fails** | If the distinction is lost, nothing can tell a stage that inherited a geometry from one that computed it, and a checkpoint records the same bytes with two different meanings |
| **How to check** | Before a stage runs, its carried entries are symlinks (possibly dangling — that is correct, `engines/stages.md § 7.4`). After it runs, they are regular files. The checkpoint diff across a run shows a type change, and that type change is the record |

### S4 — the description is input; everything else in the folder is derived

*needs the description (`engines/stages.md § 6`).*

`stages.json` is written by the user's surface and read by the generator. Decks,
wrappers, links and outputs are derived from it.

| | |
|---|---|
| **How it fails** | A produce or a run that edits the description makes the folder self-modifying: the file that is supposed to explain the folder becomes a consequence of it, and reopening no longer restores intent |
| **How to check** | The description's bytes are unchanged by any produce that did not receive a new one, and by every run. Hash it before and after |

### S5 — identity is calculation-level; the run index is invocation-level

*holds today, trivially: nothing derives an identity from a run at all — the
wrapper reads `SystemLabel` **out of the script** (`job-contracts.md § 4.3`),
which is an input. The invariant exists to keep it that way once a description
does.*

The id names the calculation; `-run0`, `-run1` name invocations of it
(`run-identity.md § 2`).

| | |
|---|---|
| **How it fails** | If anything about a run could change the id, the warm files it produced would be orphaned by the act of producing them |
| **How to check** | No code path derives an id from a run's output, a timestamp, or a run index. `stages.json`'s `run.id` is read, never recomputed (`run-identity.md § 3`, rule 1) |

### S6 — a restored folder is internally consistent

*needs the description.*

`stages.json` is tracked text (S1), so it travels with the commit. Restoring a
checkpoint therefore restores **the description together with the decks it
produced** — the folder explains itself at every point in its history, not only
at the tip.

| | |
|---|---|
| **How it fails** | If the description were untracked or archived, a restore would give you last week's decks beside this week's description, and nothing would say which the results came from |
| **How to check** | Restore any commit and re-run the produce with `dry_run`: it reports no change. **One exclusion, and only one:** PROVENANCE stamps `generated-at` at generation time (`job-contracts.md § 3.2`), so two produces of the same description are never byte-identical and the comparison ignores that key. Anything else differing is a real difference — which is exactly what makes this check worth running |

---

## 3. The immutabilities — what must never change

### I1 — a written archive's *content* is never modified

*holds today — read against the code.*

Git commits are immutable by construction; the archived bytes must be too.

**One shipped command edits an archive and is not a violation.**
`molbuilder snapshot migrate-manifest <ref>` rewrites a legacy 2-column MANIFEST
into the canonical 3-column form (`running-a-job.md § 6.2`). That changes how the
archive is *described*, never what is in it — so the invariant is about content,
and the migration carries its own: **every `(name, sha256)` pair survives it, and
the `bytes` column it adds agrees with the file on disk.**

| | |
|---|---|
| **How it fails** | An archive directory whose *files* are rewritten in place means an old commit's binaries silently become a newer run's. Every restore before that point returns the wrong data, and nothing reports it |
| **How to check** | For a given commit sha, the archived bytes never change. Re-archiving the *same* sha rebuilds the directory — legal only because one commit implies one tree — so the assertion is on **content for a sha**, not on the directory's mtime. The rebuild moves the published archive **aside** before publishing the new one and deletes it only after (A1), so no window exists in which neither is present. For the migration specifically: diff the parsed MANIFEST before and after — the name→sha mapping is unchanged, and I2 passes afterwards |

**One phrase in the shipped docs is misleading, and it misled me.**
`running-a-job.md § 6.1` and `job-contracts.md § 6.1` both say the archive is
*"deduped by content"*. Reading `checkpoint.py`, that describes deduping by
**basename within one MANIFEST** — so that overlapping globs like `*.DM` and
`*.D*` do not list a file twice and trap the strict parser. It is **not**
storage dedup across checkpoints: `_archive_binaries` copies every big binary
into `.binsnapshots/<commit-sha>/` on every checkpoint, with no hardlinking and
no content-addressed store. Ten checkpoints of a folder holding a 2 GB `.DM`
cost 20 GB. See L5.

### I2 — a MANIFEST is authoritative for its archive

*holds today.*

The 3-column `<sha256>  <bytes>  <name>` MANIFEST (`job-contracts.md § 6.1`)
describes exactly what is in that archive.

| | |
|---|---|
| **How it fails** | A MANIFEST that names a file the archive lacks turns a restore into a partial one; a sha that does not match turns it into a silent corruption |
| **How to check** | For every entry: the file exists, its size equals the recorded bytes, and its sha256 equals the recorded sha. Run it over every archive in a repository — this is the single most valuable test in the system |

### I3 — warm state is moved or restored, never incidentally lost

*holds today — read against the code: the cold path is a `mv` into
`<basename>-restart-aside-<UTC>/`, and the only `rm` in a generated wrapper
targets the rank helper and the MPS pipe directory. One mover, no deleter.*

`--cold` **moves warm files aside** into `<basename>-restart-aside-<UTC>/`; it
does not delete them (`job-contracts.md § 4.1`). No operation *whose purpose is
something else* removes or overwrites them — in particular a replacing produce,
which may remove orphaned decks and wrappers but never state
(`engines/stages.md § 7.2`).

**`restore` is the one exception, and it is not a leak.** Restoring an earlier
checkpoint replaces the worktree, warm files included — that is precisely what
the user asked for, it refuses on a dirty tree first (A2), and the state it
overwrites is itself in a commit. The invariant is therefore about *incidental*
loss: **exactly one operation may move warm state (`--cold`), exactly one may
replace it (`restore`), and nothing else may touch it.**

| | |
|---|---|
| **How it fails** | Hours of converged geometry destroyed by an operation whose job was to write an input file |
| **How to check** | Grep every path that writes into a run directory for an unlink, an rmtree or a truncating open whose target can match a warm-file suffix. There should be exactly two hits — the cold move-aside and the restore — and no third |

### I4 — a generated wrapper contains no git

*holds today.*

`running-a-job.md § 6.2` records that the wrapper-bootstraps-git path was
deliberately dropped: *"the wrapper is deliberately git-agnostic, so init is
CLI/UI-only."* A wrapper that committed would need git on the compute node, which
`running-a-job.md § 2`'s standalone contract forbids.

| | |
|---|---|
| **How it fails** | A run that dies on a node without git, for a reason having nothing to do with the calculation |
| **How to check** | No emitted `.run.sh` or `.sbatch` invokes git: grep the rendered wrapper fixtures for `git` **as a command word** (`(^\|[;&|(\s])git\s`), not as a substring — `digits` and `logging` are not violations, and a check that flags them is one somebody will disable |

---

## 4. The atomicity rules — a mutation completes or does not happen

### A1 — archiving is build, verify, swap, then delete

*holds today — read against the code, and it is stronger than this contract asked for.*

The sequence is: build into a `.tmp`, hash, copy, **re-hash and verify the copy**,
write the MANIFEST, move any published archive **aside**, `os.replace` the new one
into place, then delete the aside.

**The aside step was missing until 2026-08-06.** The code removed the published
archive and *then* renamed the new one in, leaving a window in which neither
existed — a crash there destroyed the archive that was already there and
published nothing, which is the one outcome "complete archive or nothing" must not
include. The worst case is now a stray `<sha>.old` beside a complete archive.

| | |
|---|---|
| **How it fails** | An interrupted archive leaves a directory that looks complete and is not, and I2's check would be the only thing that ever noticed |
| **How to check** | Kill the process between each step of a checkpoint; afterwards the archive set is either the old one or the new one, never a mixture |

The shipped implementation does three things this contract did not think to ask
for, and each is worth keeping: it **validates the sha directory name** before
writing anything; it hashes the **source**, copies, then re-hashes the **archived
copy** and requires them equal — because deriving the MANIFEST sha from the copy
alone would make a disk-corrupted copy *self-consistent*, later verifying against
its own bad sha and being restored as truth; and it cleans up the `.tmp` on
`BaseException`, so an interrupt leaves no stray directory. A change that
simplifies any of the three is a regression.

### A2 — restore verifies before it mutates

*holds today — read against the code, in exactly this order.*

Restore refuses on a dirty text tree, refuses on dirty binaries (sha-compared
against HEAD's archive), **verifies the target ref's archive before touching
anything**, and only then restores the worktree and copies binaries back.

| | |
|---|---|
| **How it fails** | A restore that half-completes leaves a worktree from one commit and binaries from another — a state no commit ever held, which nothing can diagnose afterwards |
| **How to check** | Corrupt one byte of the target ref's archive and attempt a restore: it refuses, and the worktree is byte-identical to what it was |

The shipped order is: refuse on dirty text (`git status --porcelain`), refuse on
dirty **binaries** (shas compared against the head archive, since git cannot see
them — S1 again), verify the *target* archive, only then `git restore`, only then
copy the binaries back. Four gates before the first mutation. The invariant is
that order, not merely the presence of the checks.

### A3 — the checkpoint precedes the mutation it protects

*needs automatic checkpoints (`engines/stages.md § 7.3`).*

A pre-produce checkpoint is committed **before** the first file of the new produce
is written (`engines/stages.md § 7.3`).

| | |
|---|---|
| **How it fails** | Taken afterwards, it records the state it was supposed to preserve a way back to |
| **How to check** | Interrupt a produce after the checkpoint and before the swap: the commit exists and **`git status` is clean** — no new file reached the folder. That is the same assertion as `engines/stages.md § 7.2`'s transactional rule seen from the history's side, and like S4's check it uses git rather than an mtime comparison, which clock skew and filesystem granularity both defeat |

---

## 5. What a staged folder adds

All four **need the layout**, and each follows from `engines/stages.md § 7.1`.

> **Why `L` and not `P`.** `molbuilder/checkpoint.py`'s own comments cite
> numbered principles — *"P3 (the user decides; the system never silently
> discards binary work)"* — from a design document that no longer exists in the
> tree. A second P-series in the same subsystem would make a code comment and a
> contract row read as the same reference and mean different things. **L** is for
> layout, which is what all four are about.

### L1 — one repository, at the parent

*BLOCKS THE STAGED DESIGN. Verified: `Repo.init` on the layout
`engines/stages.md § 7.1` specifies raises `NestedRepoRefusedError`. Until this
is settled, a staged folder cannot be checkpointed at all — so
`engines/stages.md § 7.3`'s automatic
history, the branch-from-a-stage story, and work items 10–11 are all
unreachable.*

`job-system.md § 5.5` observes that each per-stage directory is a self-contained
run directory and so can be checkpointed alone. That stays true and is not enough:
a per-subdirectory repository cannot restore a shared file that lives **above** it
— a restored stage would have pseudopotential links pointing at nothing — and no
such repository contains the workflow, so *branch at stage 2* cannot be
expressed. So this document wants one repository, at the parent.

**`Repo.init` refuses exactly that.** `_check_nested_working_dirs` walks for
subdirectories containing a working-dir marker (`.fdf`, `.py`, `.run.sh`) and
raises `NestedRepoRefusedError` — *"each lowest-directory must be its own
checkpoint repo"* — citing a **P5** rule from `run-checkpoints.md`, a document
the 2026-07 migration removed. So the guard is deliberate, tested, and its
rationale is no longer readable anywhere.

The likely reason is sound for the world it was written in: a parent holding
several *independent* run directories would checkpoint unrelated jobs together,
and restoring one would rewind the others. What a staged folder changes is the
premise — its subdirectories are not independent jobs, they are stages of one
calculation, which is exactly what makes the parent the right unit.

**This is a decision, not a defect** — but it is a *blocking* one, and the
options are narrow enough to state:

| Option | What it costs |
|---|---|
| **Teach the guard about descriptions** — a parent holding `stages.json` is one calculation, so its subdirectories are stages rather than rival jobs, and init proceeds | the guard keeps its meaning for every folder that has no description; one condition, one test. **Recommended** |
| **Repo per stage** — accept the guard, checkpoint each subdirectory | shared files above are outside every repo, so a restored stage's pseudopotential links dangle; and no repo contains the workflow, so *branch at stage 2* cannot be expressed. This is the option that made L1 exist |
| **Flatten the layout again** | reintroduces the defect `engines/stages.md § 7.1` exists to fix: every stage overwriting the last's `.ANI`, `.STRUCT_OUT`, `.EIG` |

Whichever is chosen, **it is chosen before the checkpoint work rather than
during it**, because the other two invariants in this section assume an answer.

*Check, once decided:* exactly one `.git` and one `.mbcheckpoint.json` in a
produced folder, at the top of it.

### L2 — the archive globs match at depth

*HOLDS as of 2026-08-06 — this was broken, and fixing it is what prompted the
rest of this section.*

**The most serious finding in this document, and the opposite of what an earlier
draft of it said.** I had written that a per-stage binary "lands in git as a
blob". It does not. It lands nowhere.

The two sides of S1's classification resolve depth **differently**:

| | Pattern | Reaches `coarse/<id>.DM`? |
|---|---|:--:|
| `.gitignore`, from `_render_gitignore` | the raw glob, `*.DM` | **yes** — a gitignore pattern with no slash matches at every level |
| the archive set, `_list_big_binaries` | `path.glob("*.DM")` | **no** — its own docstring says *"Top-level big binaries"* |

So a stage's density matrix is **gitignored and unarchived** — excluded from the
commit *and* absent from `.binsnapshots/`. It is in no snapshot at all, and a
restore silently does not bring it back. That is S1's "never neither" branch, the
one whose failure is losing data rather than wasting disk.

| | |
|---|---|
| **How it fails** | Quietly, at the moment of maximum trust. The user restores a checkpoint expecting a resumable state and gets the geometry (`.XV` is text, so it is tracked) with no density matrix beside it. The run starts over and nothing says why |
| **How to check** | Produce a two-stage folder, run both, checkpoint, assert S1 over the whole tree: every file is tracked or archived. It fails today, and that failure is the acceptance test |
| **Fixed by** | The archive walk is recursive and the MANIFEST key is a **repo-relative POSIX path** rather than a bare basename — so both sides resolve depth the same way. The key space *widened*: a basename is a valid relative path, so every archive written before it reads unchanged. Two consequences fell out and are part of the invariant: the walk **skips symlinks** (a carried restart file is inherited, not owned — S3 — and archiving the link would restore a regular file where a link belongs) and **skips dot-directories** (a recursive walk that forgot would archive the archive). Pinned by `tests/test_checkpoint_nested_layout.py` |

### L3 — every commit and tag names its calculation

The commit message carries the id and the stage; a finished stage is tagged
`<id>/<stage>/<UTC>` (`engines/stages.md § 7.3`). A folder can be moved, copied
to a cluster, or opened a year later, and a history whose commits say only
*"stage 2 converged"* cannot say which calculation that was.

*Check:* every commit message contains the folder's `run.id`, and every tag parses
into exactly three parts of which the first equals it.

### L4 — the tag namespace is stage completions only

Pre-produce checkpoints are commits, reachable through `snapshot list`. Tagging
them too would bury the points a user meant to reach among the ones they passed
through.

*Check:* **molbuilder** creates a tag only at a stage completion. A user tagging
by hand is their own business — `snapshot tag` exists for it — so the assertion is
on what the automatic path emits, not on the total.

### L5 — a checkpoint's cost is bounded by what changed, not by what exists

*needs work — but the layout now makes it structural rather than clever.*

**Immutable attempts change the shape of this.** `project-layout.md § 1.2` gives
every invocation its own directory, written once and never modified, so an
archived file never changes. A save then stores the attempts that appeared since
the last one and references the rest — *"archive what is new"*, correct by
construction.

That replaces the content-addressed store this invariant used to call for.
Hashing every file to discover most are unchanged is the right answer when
anything might have changed; when nothing inside an archived attempt can change,
the cheaper answer is also the more obvious one.

Automatic checkpoints (`engines/stages.md § 7.3`) fire twice per stage. If every
one copies every big binary, a five-stage mission pays ten full copies of its
`.DM` and `.HSX` set, and the folder this design exists to keep manageable
becomes the reason the disk fills.

| | |
|---|---|
| **How it fails** | Silently, and only at scale. Nothing errors; the archive grows linearly in checkpoints × binary size, and `prune` is listed as unbuilt (`running-a-job.md § 6.2`), so nothing reclaims it either |
| **Made worse by L7's fix** | Binary-only changes now produce commits that previously did not exist — correctly, since the alternative was losing them — and each one copies the full binary set again. Fixing the data-loss bug raised the disk cost, which is the right trade and a reason L5 should not wait |
| **How to check** | Checkpoint a folder twice with the binaries untouched between them. The second checkpoint's *incremental* disk cost is near zero |
| **How to fix** | Store an attempt's files once and let later saves reference them; immutability makes that safe without hashing anything. (Content-addressing — `.binsnapshots/by-content/<sha256>` with the MANIFEST referencing it — stays the fallback if the layout ever admits mutable runs.) |

### L8 — an archived attempt never differs afterwards

*needs the layout — and it is I2 pointed at a directory.*

An attempt is immutable by contract, not by permission bit
(`project-layout.md § 1.2`). Nothing stops an edit; what matters is that an edit
is **noticed**.

| | |
|---|---|
| **How it fails** | Silently. Someone edits a file inside an attempt already saved, and every later save carries a history whose earlier points no longer describe what is on disk |
| **How to check** | Re-hash an archived attempt's files against their MANIFEST entries — exactly I2, over a directory the layout says is frozen. A difference is reported, never merged |

### L7 — a change to a big binary alone leaves no checkpoint

*not held today — found by running L2's acceptance test.*

`checkpoint()` runs `git add .`, then `git status --porcelain`; if the tree is
clean it returns `None`. Big binaries are gitignored (S1), so **a change that
touches only them leaves the status clean and produces no commit and no new
archive.** The code half-anticipates this: if HEAD has *no* archive at all and
binaries exist, it writes one. It does not notice that HEAD's archive is stale.

| | |
|---|---|
| **How it fails** | A user re-runs a stage that rewrites its `.DM` and nothing else git can see, checkpoints, and is told the tree was clean. The new density matrix is in no snapshot. It surfaces later as a refused restore ("uncommitted binary changes"), which is the safe direction, but the checkpoint they asked for never happened |
| **How to check** | Change only a big binary, checkpoint, and assert a new archive exists whose MANIFEST records the new sha |
| **Note** | In a staged folder a run also writes text (`.out`, `.XV`), so the clean-status case is narrower than it looks — but "narrower" is not "absent", and `--force`-style re-runs hit it |

### L6 — a history can be verified without being restored

*needs work — the capability exists but has no door.*

`_verify_archived_binaries` already checks existence, size and sha256 against the
MANIFEST and touches nothing (I2). It is called from exactly two places, both on
the restore path. **So the only way to learn that an archive is intact is to
attempt a restore** — which is the worst possible moment to find out it is not,
and which a user will not do speculatively on a folder they are working in.

| | |
|---|---|
| **How it fails** | A checkpoint taken onto a failing disk verifies at write time (A1) and is never looked at again until the day someone needs it |
| **How to check** | A `snapshot verify [<ref>]` verb — CLI and HTTP — runs I2 over one archive or all of them and reports. It is a few lines over a helper that already exists, and it is the natural home for the most valuable test in this document |

---

## 6. The review sheet

One line each, for reading over a diff:

| | Invariant | Assertable |
|---|---|:--:|
| **S1** | every **regular file** is tracked XOR archived — never both, never neither (symlinks are layout, not content) | today |
| **S1a** | the git-ignore set is *derived* from `archive_globs`, never kept beside it | today |
| **S2** | a stage writes only inside its own directory | needs the layout |
| **S3** | inherited is a symlink; owned is a regular file | needs the layout |
| **S4** | the description is never modified by a produce or a run | needs the description |
| **S5** | nothing a run produces can change the id | today |
| **S6** | a restored folder explains itself: description and decks travel together | needs the description |
| **I1** | a written archive's *content* is never rewritten; identical content dedupes | today |
| **I2** | every MANIFEST entry matches its file: name, size, sha256 | today |
| **I3** | exactly one operation moves warm state, exactly one replaces it, nothing else touches it | today |
| **I4** | no generated wrapper invokes git | today |
| **A1** | archive: build, verify the copy, then swap | today |
| **A2** | restore verifies the target archive before touching the worktree | today |
| **A3** | the checkpoint is committed before the mutation it protects | needs automatic checkpoints |
| **L1** | one repository, at the parent | **blocked** — `init` refuses it by design |
| **L2** | archive globs match at depth | **holds** (fixed 2026-08-06) |
| **L3** | every commit and tag names its calculation | needs the layout |
| **L4** | molbuilder tags stage completions only | needs the layout |
| **L5** | a checkpoint costs what changed, not what exists | **not held today** |
| **L6** | a history can be verified without being restored | **not held today** |
| **L7** | a binary-only change still produces a checkpoint | **holds** (fixed 2026-08-06) |
| **L8** | an archived attempt never differs afterwards | needs the layout |

**Twelve of the twenty-one can be asserted against the code as it stands.** Two
(**L2**, **L7**) were broken and are now fixed, with tests. Two (**L5**, **L6**)
are not held and are work rather than checks; one (**L1**) is **blocked** by a
deliberate guard whose rationale outlived its document.

### What has actually been read

| Read against the code | Verdict |
|---|---|
| **S1** | **broken** — a dot-prefixed basename was archived under a key the parser refuses, so the archive wrote a MANIFEST it could never read back. Fixed 2026-08-06 |
| **S1a** | **broken** — `set_archive_globs` was a single writer, `init` was not; "the hazard is false" was half a reading. Fixed 2026-08-06 |
| **S2** | check was blind to the big binaries; both halves now |
| **S5** | holds trivially — nothing derives an identity from a run at all |
| **I1** | held, restated: content-for-a-sha — and the rebuild now moves the published archive aside instead of deleting it first |
| **I2** | implemented and correct — but reachable only through restore (→ L6) |
| **I3** | held — one mover (`mv` to `-restart-aside-`), no deleter |
| **I4** | held — no generated wrapper invokes git |
| **A1** | strong in its three parts (sha-dir validation, source-vs-copy fidelity, `.tmp` cleanup) — **but the swap itself was not atomic**. Fixed 2026-08-06 |
| **A2** | held, in exactly the stated order — four gates before the first mutation |
| **L2** | was **broken, losing data rather than wasting disk** — fixed 2026-08-06, pinned by tests |
| **L5** | **broken** — no cross-checkpoint dedup; my "cheap" claim was false |
| **L6** | **absent** — no way to verify a history without restoring it |
| **L1** | **blocked** — `Repo.init` refuses a parent repo by design (`NestedRepoRefusedError`) |
| **L7** | was **absent** — a binary-only change left `git status` clean and nothing was checkpointed; fixed 2026-08-06 |

**Six defects found, all fixed.** Two (L2, L7) turned up by *running* the
checks. **Four turned up by reading the module end to end** — `init` skipping an
existing `.gitignore`, a dot-prefixed basename writing an unreadable MANIFEST,
archive sizes counted only at the top level, and a swap that deleted the
published archive before publishing its replacement.

**The four found by reading were invisible to the tests, which passed
throughout.** That is the argument for reading a module in call order rather than
probing it, and it is the opposite lesson from the one the first two taught.

Three of my own invariants were wrong when written, each because a phrase in the
guide was trusted over the code: *"nothing keeps them in step"*, *"deduped by
content"*, *"lands in git as a blob"*. A fourth — the first fix for S1a — would
have shipped a **worse** bug than the one it closed, and was caught by statically
reviewing the patch rather than by its tests, which passed.

**Six still cannot be read**: S3, S4, S6, L3, L4 describe the per-stage layout
and the description, neither of which exists, and A3 describes automatic
checkpoints, which do not. They stay claims until there is something to assert
them against.

---

## 7. What this contract does not own

- **How to use the checkpoint system** — the CLI verbs, the HTTP routes, the
  sidebar panel, and what is still unbuilt (archive pruning, a `snapshot diff`
  CLI face) — [`running-a-job.md`](?doc=execution/running-a-job.md) —
  `running-a-job.md § 6`.
- **The `.mbcheckpoint.json` schema and the MANIFEST column format** —
  [`job-contracts.md`](?doc=execution/job-contracts.md) — `job-contracts.md § 6.1`.
- **The folder being checkpointed, and when a checkpoint is taken** —
  [`engines/stages.md`](?doc=engines/stages.md) — `engines/stages.md § 7`.
- **Phasing and open questions** —
  [`web/staged-runs-architecture.md`](?doc=web/staged-runs-architecture.md) and
  [`roadmap.md`](?doc=roadmap.md) (R3).
