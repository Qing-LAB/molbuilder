# Checkpointing — saving a calculation so you can always get it back

**Role:** contract
**Domain:** execution
**Companions:** [`running-a-job.md`](?doc=execution/running-a-job.md) § 6 — the
guide: what to type and what the buttons do, which this document never repeats;
[`job-contracts.md`](?doc=execution/job-contracts.md) § 6.1 — the file formats;
[`engines/stages.md`](?doc=engines/stages.md) § 7 — the folder being saved and the
two moments a save is offered;
[`project-layout.md`](?doc=execution/project-layout.md) § 6 — where the history
sits in the project tree; [`run-identity.md`](?doc=execution/run-identity.md) —
the id every commit and tag carries.

A calculation lives in a folder: inputs, a relaxed geometry, a density matrix,
logs. Checkpointing takes a **snapshot of that whole folder** so that whatever
you do next — rerun a stage, regenerate the decks, start over from scratch — you
can get back to exactly where you were.

It is git underneath, with one addition: files too large for git are copied into
a side store with a checksum each, so a snapshot holds the big binaries too. You
never interact with either directly.

**This document is written for two readers.** §§ 1–10 are for anyone using
molbuilder — what it does, when to reach for it, and what the commands do, with
worked examples. §§ 11–15 are for anyone changing the code: the rules that must
not break, how to test them, and where the code is.

---

## 1. The goal

> ### The promise
>
> **Whatever state a run is in, that state can be brought back.**
>
> Every rule in this document exists to keep that sentence true.

**Nothing else competes with it.** Not disk space, not speed, not tidiness. If a
rule here and a saving of disk ever disagree, the rule wins — a folder that is
cheap to store and wrong is worth nothing.

---

## 2. What checkpointing is not

- **It is not a backup system.** The snapshot lives inside the folder it is
  snapshotting. A deleted folder takes its history with it; copy the folder
  somewhere else for that.
- **It is not automatic.** Nothing is ever saved without you saying so (§ 9).
- **It does not decide what is worth keeping.** Point it at a folder and it
  saves the folder — all of it. Whether a benchmark's throwaway trials deserve
  keeping is decided by whoever runs the benchmark, by choosing what to point
  this at.
- **It does not run on the cluster.** Saving is a decision, and decisions happen
  where you are, not on a compute node.
- **It is not a way to move a calculation between machines.** It records
  history; `rsync` moves folders.
- **It is not a safety net for work you never saved.** A restore tells you what
  it is about to overwrite and then does it (A5). Nothing is stashed or set
  aside on your behalf.
- **It does not deal in files.** A state is the whole directory. Pulling one
  file out of the past is a different request with a different tool — `git show
  <ref>:<path>`.

### 2.1 What it decides, and what it does not

| | |
|---|---|
| **You choose** | *which folder* to save, and *when* |
| **Checkpoint chooses** | *how* each file is stored |
| **Checkpoint never chooses** | *whether* a file is stored |

That middle row is the only decision it owns, and it is mechanical: git is bad at
very large files, so those go somewhere else. A storage detail, not an opinion
about what matters.

> **This is not where the design started; it is what it learned.** A trajectory
> file (`*.MD`) sat outside every snapshot for months because it was *large* —
> the size was allowed to argue with the saving, and the size won. Nobody chose
> that; it followed from letting the two be weighed at all.

---

## 3. The overall shape

Two stores. Every file is in exactly one of them.

```mermaid
flowchart LR
    F["<b>a file in the folder</b>"] --> Q{"bigger than<br/>the size limit?"}
    Q -->|"no"| G[("<b>git</b><br/>.git/<br/>diffable, cheap")]
    Q -->|"yes"| A[("<b>the archive</b><br/>.binsnapshots/&lt;commit&gt;/<br/>whole copies + a checksum list")]
    G -.->|"the commit names the<br/>archive that goes with it"| A
```

**Which store, by measuring — not by file type.** A file over the limit goes to
the archive; everything else to git. Extensions are not consulted, because a name
is not the property being tested: a 4 GB `.EIG` nobody listed would be committed,
and an empty `.DM` somebody did list would be archived.

The engine entries in the config (§ 4) name families that are *always* large, so
those skip the measuring. That is an effort saving and nothing else — **a hint
can make a save faster; it can never make it store less.**

> ⚠ **This is the rule; it is not yet the code (S1b).** What ships today consults
> *only* the name: `_list_big_binaries` matches the glob list and measures
> nothing, so there is no size limit anywhere in the system. Every sentence above
> describes where this is going. The diagram is the contract, not a screenshot.

### 3.1 A real folder, after two saves

```text
BDT_Au_relax_Au38C6H4S2/
├── .git/                            the text history
├── .gitignore                       generated — lists exactly what the archive holds
├── .binsnapshots/                   the archive
│   ├── 4f9c…a71/                      one directory per commit
│   │   ├── 01_coarse/run-0/BDT_Au_relax_Au38C6H4S2.DM
│   │   └── MANIFEST                   what is in here, and its sha256
│   └── b2e0…33d/
│       ├── 01_coarse/run-0/BDT_Au_relax_Au38C6H4S2.DM   ← unchanged since the
│       │                                                   save above, so this is
│       │                                                   a hard link: no disk
│       ├── 02_tight/run-0/BDT_Au_relax_Au38C6H4S2.DM
│       └── MANIFEST
├── task.json                             ┐
├── BDT_Au_relax_Au38C6H4S2.fdf.template  │  small → git
├── 01_coarse/                            │
│   ├── BDT_Au_relax_Au38C6H4S2.fdf       │
│   └── run-0/                            │
│       ├── BDT_Au_relax_Au38C6H4S2.out   ┘
│       ├── BDT_Au_relax_Au38C6H4S2.XV      small → git
│       ├── run.json                        small → git
│       └── BDT_Au_relax_Au38C6H4S2.DM      large → the archive
└── 02_tight/…
```

And a MANIFEST is three columns — sha256, bytes, path — sorted by path:

```text
7ef4c645…344a  8388608  01_coarse/run-0/BDT_Au_relax_Au38C6H4S2.DM
36c54395…a572  8388608  02_tight/run-0/BDT_Au_relax_Au38C6H4S2.DM
```

**The path is relative to the folder root, not a bare filename.** That is what
lets one archive hold a `.DM` from every stage without them colliding. Every part
is checked so a restore cannot be steered out of the folder: no leading `/`, no
`..`, nothing starting with a dot.

**Identical content is stored once.** Two saves that both contain an unchanged
`.DM` record the same sha256, so the second links to the first's file rather than
copying it. The archive still holds a real file at that path, so nothing reading
it can tell.

---

## 4. The pieces

| Piece | What it is | Who writes it |
|---|---|---|
| **git** | the history of everything small | `checkpoint` |
| **the archive** — `.binsnapshots/<commit>/` | whole copies of everything large | `checkpoint` |
| **MANIFEST** | what is in one archive, with a sha256 each | `checkpoint` |
| **`.gitignore`** | what git skips — *generated*, so it matches the archive exactly | `checkpoint` only |
| **the config** | the size limit and the per-engine hints | you, in molbuilder's config |
| **`molbuilder snapshot …`** | the verbs you type (§ 5) | — |
| **`Repo`** (`molbuilder/checkpoint.py`) | the class every surface goes through | — |

**The config is molbuilder-wide, not per folder.** One `generic` entry — save
everything, choose the store by size — plus optional per-engine hints. A caller
may name its engine so the matching hint applies; with no name `generic` is used,
which is always correct and merely measures more.

> A per-folder config would let one folder behave differently from another for no
> recorded reason. That is a trap, not a feature.

> ⚠ **Also the rule and not yet the code (S1c).** Today every folder carries its
> own `.mbcheckpoint.json`, and two surfaces write it — `molbuilder snapshot
> config --set-globs` and the web route behind it. So the trap described above is
> currently open, and one of its consequences is I2c below.

---

## 5. Using it

```bash
# once, in the calculation folder
molbuilder snapshot init

# save the current state
molbuilder snapshot checkpoint -m "stage 1 converged, before retightening"

# what have I got?
molbuilder snapshot list

# mark one worth coming back to
molbuilder snapshot tag stage1-good -m "geometry I trust"

# go back to it — text AND the big binaries.  Anything here you have not
# saved is named and you are asked; --force answers yes for a script.
molbuilder snapshot restore stage1-good

# try something without losing the main line
molbuilder snapshot branch what-if-tighter

# which files count as "big" here
molbuilder snapshot config
```

**`restore` is a rewind, not a fetch.** It returns the *whole folder* to that
state — it does not pull one file out. To keep what you have as well, save it
first (§ 8). To read one old file without moving anything, git already answers
that: `git show <ref>:<path>`.

> ⚠ **`snapshot restore --no-binaries` exists today and should not** (A4). It
> rewinds the text and leaves every big file untouched, which is the mixed state
> S8 is about — reached on purpose instead of by accident.

---

## 6. Saving, step by step

```mermaid
flowchart TB
    S["checkpoint"] --> R{"is .gitignore<br/>what it should be?"}
    R -->|"edited by hand"| STOP1["refuse or repair<br/>— an edited ignore list<br/>silently drops files"]
    R -->|"yes"| M["measure every file"]
    M --> B["big ones → build the archive<br/>in a .tmp"]
    B --> V["hash the source, copy,<br/><b>re-hash the copy</b>, compare"]
    V -->|"differ"| STOP2["fail — a corrupt copy must<br/>never become self-consistent"]
    V -->|"match"| W["write MANIFEST"]
    W --> SW["move any published archive aside<br/>→ swap the new one in<br/>→ delete the aside"]
    SW --> C["commit the small files"]
```

**Why the copy is re-hashed rather than trusted.** If the MANIFEST's checksum came
from the copy alone, a copy corrupted on the way to disk would be
*self-consistent* — it would verify against its own bad checksum forever and be
restored as truth. Hashing the source and re-hashing the copy makes that
impossible.

**Why the old archive is moved aside instead of deleted first.** Deleting and then
renaming leaves a moment where neither exists; a crash there destroys the archive
that was already good. The worst case now is a leftover `.old` directory beside a
complete one.

---

## 7. Restoring, step by step

```mermaid
flowchart TB
    R["restore &lt;ref&gt;"] --> D0{"does &lt;ref&gt;<br/>resolve?"}
    D0 -->|"no"| X0["refuse — unknown tag,<br/>branch or sha"]
    D0 -->|"yes"| D3{"does the target's<br/>archive verify?"}
    D3 -->|"no"| X3["refuse — before any change"]
    D3 -->|"yes"| D1{"anything here<br/>not saved?"}
    D1 -->|"yes"| Q["say exactly what will be lost<br/>— text AND big files —<br/>and ask"]
    Q -->|"no / no answer"| X1["stop — nothing touched"]
    Q -->|"yes, or --force"| G
    D1 -->|"no"| G["restore the text from git"]
    G --> B["copy the big files back"]
```

**Two refusals, then one question, and the order is the rule.** The refusals
come first because they are about the *target* — an unknown ref or a corrupt
archive means the operation cannot happen at all, and nobody should be asked to
accept a loss for an operation that then fails for an unrelated reason. The
question comes last, immediately before the first byte moves.

**The question is answered by you, and the answer is honoured.** Unsaved work is
not rescued, stashed, renamed or moved aside. If you say yes, it is gone — you
called `restore` without calling `checkpoint`, and that is a choice the system
records the consequences of rather than second-guessing (A5).

**What must never happen is a half-restore**: text from one save and binaries
from another, a state no save ever held and nothing can diagnose afterwards.
That is why the archive is verified *before* the text is touched.

### 7.1 What the folder is afterwards

**A restore does not move HEAD.** It rewinds the *working tree* and leaves the
branch pointing where it was, so the moment it finishes the folder is dirty
relative to its own history — the restored content reads as a pile of
uncommitted changes. Three things follow, and none of them are obvious:

- **`git log` shows no record that a restore happened.** The history is
  unchanged; only the files moved.
- **A second restore asks before it proceeds**, because the folder now differs
  from HEAD — and the difference it names is the *first* restore's own work, not
  anything the user typed. Answering yes is correct and loses nothing that was
  not already in a checkpoint. ⚠ Today this is a flat refusal instead of a
  question (A5), and its message blames the user for changes they did not make.
- **The next checkpoint commits the rewound state forward** as a new commit.
  That is the intended way to keep it: history moves forward even when the
  folder moves back, so nothing is ever rewritten or lost.

**A restore reads only what the save wrote down.** It consults no configuration —
not the size limit, not the engine hints, not any file you can edit. So the config
can be changed, moved or deleted and every archive already written stays
restorable. That holds for what a restore *does*; ⚠ it does not yet hold for what
a restore *warns about*, which is I2c.

---

## 8. Why this matters far more in a flat folder

In the **nested** shape every stage and attempt is on disk at once. Going back to
stage 1's geometry means opening `01_coarse/run-0/`. A checkpoint there is
protection against loss and a way to branch — valuable, not load-bearing.

**In the flat shape it is load-bearing.** The restart files are unsuffixed and
shared *by design* — that is exactly what lets stage 2 continue from stage 1 —
which means stage 2 **overwrites** them.

> **Without a checkpoint, a flat folder can only move forward.** With one it can
> return to any state it was saved in and continue from there. That is not a
> convenience on top of the flat shape; it is what makes the flat shape usable for
> iterative work at all.

Two consequences worth saying out loud:

- **Saving before each stage is not housekeeping in a flat folder — it is the
  save point.** Miss one and that state is gone, because nothing else on disk
  holds it.
- **A restore is a rewind, not a fetch.** It returns the *whole* folder to a past
  state (S6). So going back means **save what you have, then restore the earlier
  one** — skip the first step and you lose the present. The nested shape never
  poses the question, because it never had to overwrite anything.

---

## 9. Who takes a checkpoint, and when you are asked

> **A checkpoint is always an explicit act. molbuilder never takes one on its
> own.** It *asks*, at the one moment where not having asked would cost something,
> and then does what it is told.

**The moment is `prep`, because prep is the change.** When prep is about to
rewrite a folder that already holds results, it says what will change and offers
to save first. You answer.

| | |
|---|---|
| **Where it asks** | interactive `prep`, when the target already holds results |
| **Who decides** | you, every time |
| **Non-interactive** (`--yes`, a script) | proceeds **without** saving and **says so** — it may not silently pick either way |
| **The last stage** | nothing asks, because nothing follows. A surface showing a finished, unsaved run should say it is unsaved |

**Never at run or submit time**, which may be a queued job — a prompt would block
a queue, and submitting starts something new rather than changing something that
exists. **Never on the compute node**, which would need git there (I4).

**Something already watches the run, and it is not this.** `mb_monitor.py` sits
beside the job, follows the launcher's PID so it knows when the run really ended,
reads the outputs, and can notify you — webhook, email, whatever you wire in. So
nobody has to be at the cluster at 3am. **What it must not do is act.** It
observes and tells; it never decides and never changes the calculation. Saving is
a decision, and decisions are yours.

---

## 10. Worked examples

### 10.1 Going back a stage, without losing today's work

The mistake this avoids: restoring straight away and discarding the present.
Nothing stops you — you would be asked once and it would be gone (A5).

```bash
# stage 2 made things worse.  Keep it anyway -- it costs nothing and you
# cannot get it back afterwards.
molbuilder snapshot checkpoint -m "stage 2, force tol worse than stage 1"

molbuilder snapshot list
#   b2e033d  2026-08-08 14:02  stage 2, force tol worse than stage 1
#   4f9ca71  2026-08-08 11:40  stage 1 converged

molbuilder snapshot restore 4f9ca71     # the whole folder rewinds
# now retune and rerun stage 2 from stage 1's geometry
```

### 10.2 Forking a what-if

```bash
molbuilder snapshot checkpoint -m "before trying a tighter mesh"
molbuilder snapshot branch tighter-mesh
# edit task.json, rerun.  The main line is untouched, and
# `snapshot restore <ref>` on the original branch returns to it.
```

### 10.3 From Python

```python
from molbuilder.checkpoint import Repo

repo = Repo("projects/BDT-Au/optimization/bdt-relax")
if not repo.initialized:
    repo.init(engine="siesta")          # engine picks the size-limit hints

cp = repo.checkpoint(message="stage 1 converged")
print(cp.sha if cp else "nothing changed")

for c in repo.list_checkpoints(limit=5):
    print(c.sha[:7], c.message)

repo.restore("stage1-good")             # text + binaries, or it refuses
```

`checkpoint()` returns `None` when there was nothing to save. `restore()` raises
rather than half-completing — see § 7.

## 11. The rules

Each one names what it prevents and how to test it. **Status is in § 8.**

### Everything is saved

**S1 — every regular file is in git or in the archive: never both, never
neither.** The only thing not stored is `.binsnapshots/` itself, because a store
cannot contain itself. No category of file is exempt.

*Symlinks are outside this, and "regular" is load-bearing.* A stage links its
deck and the shared pseudopotentials rather than copying them, so a saved tree is
full of links. A link has no content of its own — the real file is stored once,
wherever it lives — and recreating the layout is what a restore does anyway.

- **Fails as:** both → a multi-gigabyte blob in git history forever. Neither →
  the file is in no snapshot, and a restore silently does not bring it back.
- **Test:** walk every regular file and assert `archived` ≠ `tracked`, with
  `.binsnapshots/` the only excluded path. **No allow-list** — a file is stored
  or the test fails.

**S1a — `.gitignore` is generated from the classification, never hand-kept
beside it.** One *source*, not merely one writer.

- **Fails as:** a hand-kept list answers to no rule. That is exactly how `*.MD`
  came to be ignored with nowhere else to go.
- **Test:** assert the generated `.gitignore` contains **nothing that is not an
  archive pattern**. That check catches the whole class, and no fixture can be
  too short for it.

**S1b — which store a file goes to is decided by measuring the file.** Its size
is the property being tested, so its size is what is read. Names may only be
used to *skip* a measurement for a family that is always large (§ 3) — never to
decide the answer.

- **Fails as:** the two errors are not symmetric, and only one of them is
  survivable. A large file no pattern names is committed, and git carries that
  blob forever. A small file a pattern names is archived, which costs a little
  disk and nothing else.
- **Test:** an unlisted file above the limit and an unlisted file below it, each
  asserted into the right store — no pattern involved in either direction.

**S1c — the classification has one home, and it is molbuilder's config.** One
`generic` entry plus optional per-engine hints, outside every calculation
folder. A caller may name its engine; with no name `generic` is used, which is
always correct.

- **Fails as:** a per-folder copy makes two folders behave differently with
  nothing on disk explaining why, and makes the classification a thing a person
  can edit between a save and a restore — which is I2c.
- **Test:** two folders under different engines produce the same store decision
  for the same file, and no calculation folder contains a classification file at
  all.

### The record can be trusted

**I1 — an archive's content is never modified once written.** Re-archiving the
same commit rebuilds the directory, which is legal because one commit implies one
tree; the assertion is on *content for a commit*, not on the directory's mtime.
`snapshot migrate-manifest` rewrites a legacy 2-column MANIFEST into the 3-column
form and is not a violation: it changes how the archive is *described*, never
what is in it, and every `(name, sha256)` pair survives it.

**I2 — a MANIFEST is authoritative for its archive.** For every entry: the file
exists, its size matches, its sha256 matches.

- **Test:** run it over every archive in a folder. This is the single most
  valuable test in the system.

**I2a — a restore is decided by what the save recorded, never by
configuration.** Verified in code: `Repo.restore`, `_copy_archived_binaries` and
`_verify_archived_binaries` read no config, no glob list, no engine name.

- **Why it matters now:** the classification is moving out of the folder and into
  molbuilder's config. Because restore never re-derives, every archive written
  before that stays readable — an edit to one file instead of a migration of
  every archive in existence.
- **Test:** save, then change the engine, empty the hint list, delete the config
  outright — and assert the restored tree is byte-identical either way.

**I2b — every record the history leans on is tamper-evident.** Two records, and
they fail in opposite directions:

| record | decides | protected by |
|---|---|---|
| `MANIFEST` | what a **restore** returns | a sha256 **anchored outside itself** |
| `.gitignore` | what the **next save** even sees | **regeneration** — the correct content is computable |

*The MANIFEST records facts that cannot be recomputed, so it needs a digest.
`.gitignore` records a derivation, so the derivation is the check* — and
regeneration says which line is wrong, where a digest could only say that
something is.

- **Fails as:** delete a MANIFEST line and that file is simply not restored — it
  reads exactly like "this archive never held it". Change a sha *and* the file to
  match and a restore returns the wrong bytes **and reports success**. Add
  `*.XV` to `.gitignore` and the next save stores no `.XV` at all while looking
  perfectly healthy.
- **Test:** the three MANIFEST edits above must all be refused, naming the
  *record* as what failed. For `.gitignore`, edit inside the marked section as
  well as outside — anyone who knows about the markers will edit inside them.

> **Records are named so nobody mistakes one for a setting.** `MANIFEST` looks
> like a file a person may reasonably adjust; `MANIFEST.do_not_edit` does not.
> This buys nothing against deliberate tampering — that is the digest's job — and
> everything against somebody tidying a directory. `.gitignore` cannot take the
> suffix, since git requires that name. ⚠ A rename is an archive-format change:
> the reader must accept both names or archives written before it stop opening.

**I2c — what a restore *says* it will overwrite is read from the same record it
overwrites *from*.** I2a made the action config-free. This is its other half:
the warning A5 asks you to answer must be computed from the target's MANIFEST,
because the MANIFEST is exactly the set of files the restore is about to write
over. Anything else is a different question being reported as if it were this
one.

Today the warning is computed from the glob list instead
(`_working_binaries_dirty` → `_list_big_binaries` → the config), so the two
halves of one restore ask two different sources what the big files are.

- **Fails as: you are told the wrong thing and agree to it.** A `.DM` is
  archived while `*.DM` is classified big. The classification is later narrowed
  — one CLI call or one web request (S1c). You modify that `.DM` and restore an
  earlier checkpoint. The warning does not mention it, because the glob list no
  longer matches it; the copy overwrites it anyway, because the MANIFEST still
  lists it. **Losing it is your call to make** (A5) — being asked a question
  that omits it is not.
- **The trap in fixing it:** `_working_binaries_dirty` has a *second* caller —
  `checkpoint`, which uses it to notice a binary-only change (L7). There the
  MANIFEST alone is not enough, because a brand-new big file appears in no
  MANIFEST yet. **Restore's warning needs the MANIFEST's keys; the save's change
  detection needs both.** Collapsing them to one set breaks whichever rule was
  not being thought about.
- **Test:** archive a big file, narrow the classification so nothing matches it,
  modify it, restore an earlier ref — the warning must still name that file.
  Then, separately, create a big file that no MANIFEST mentions and assert
  `checkpoint` still sees it.

**L8 — a saved attempt never differs afterwards.** This is I2 pointed at a
directory the layout says is frozen. *Hierarchical only* — a flat folder's
`<id>.DM` is *expected* to change every stage, so there a difference is news
rather than a violation. Do not let a check written for one shape fail the other.

### A save or a restore completes, or does not happen

**A1 — archiving is build, verify, swap, then delete** (§ 4).

- **Test:** kill the process between each step; afterwards the archive set is the
  old one or the new one, never a mixture.

**A2 — a restore verifies before it changes anything** (§ 5), in that order.

- **Test:** corrupt one byte of the target archive and attempt a restore — it
  refuses, and the folder is byte-identical to before.

**A3 — the checkpoint precedes the change it protects.** A pre-produce save is
committed *before* the first new file is written.

- **Test:** interrupt a produce between the save and the swap: the commit exists
  and `git status` is clean — no new file reached the folder.

**A4 — a restore returns the whole folder, or it does not happen.** There is no
partial restore. Text and binaries are one state; returning half of one save and
keeping half of another produces a folder no save ever held, and § 1's promise
is about *states*, not about files.

**`snapshot restore --no-binaries` is that partial restore, and it ships.** It is
on three surfaces — the CLI flag, `include_binaries` in the
`/api/checkpoint/restore` body, and the Python keyword — and it does not merely
skip the copy: both remaining protections sit inside the same conditional, so it
also skips the dirty-binary gate and the archive verification. Its own test is
named for it.

*The flag should be removed rather than documented.* The one use it plausibly
serves — reading an old input without disturbing the present — is `git show
<ref>:<path>`, which touches nothing at all. A verb that produces a state the
contract calls a hazard cannot be kept because it is occasionally convenient.
⚠ Removing it is a code change and an API change to the route; it is not done by
this document.

- **Fails as:** exactly S8, minus the excuse. There the user typed a git command
  that means something else everywhere; here molbuilder offered it.
- **Test:** no surface accepts a request for a text-only restore. Until the flag
  is gone, the test is red and names it.

**A5 — a restore warns about unsaved work, then does what it is told.** It does
not refuse, and it does not rescue.

**Checkpoint is not responsible for work you did not save.** Calling `restore`
without calling `checkpoint` is a decision, and the answer to it is yours: the
warning names what will go, and `yes` — or `--force` for a script — proceeds and
loses it. There is no stash, no move-aside, no automatic save-before-restore, no
`.orig` copies. Every one of those is a mechanism that owns a decision it should
not, and each one leaves debris a later restore then has to reason about.

*A state is the whole directory.* Checkpoint deals in directory states, not in
files, so "keep this one file while rewinding the rest" is not a smaller request
than a restore — it is a **different** one, and the tool for it is `git show
<ref>:<path>` or a second folder.

- **Fails as, in both directions:** a refusal makes the user hand-delete files
  to get past it, which is worse than the loss they had already accepted. A
  silent rescue leaves the folder holding something no save produced (A4, S8).
- **Test:** with unsaved text *and* an unsaved big file, a non-interactive
  restore stops and changes nothing; `--force` completes; and afterwards the
  folder matches the ref exactly, with no rescue copies anywhere in it.
- **Status today:** it refuses on both, there is no `--force`, and the refusal
  for binaries reads as an error about something the user broke.

### Nothing else touches the state

**I3 — warm state is moved or restored, never incidentally lost.** Exactly one
operation may move it and exactly one may replace it (`restore`). Nothing else
may touch it. A replacing produce may remove orphaned decks and wrappers, never
state.

- **Test:** grep every path that writes into a run directory for a delete or a
  truncating open that could match run state. There should be no hit that is not
  one of those two.

**I4 — a generated wrapper contains no git.** A wrapper that committed would need
git on the compute node, which `running-a-job.md § 2` forbids.

- **Test:** no emitted `.run.sh` or `.sbatch` invokes `git` **as a command word** —
  `digits` and `logging` are not violations, and a check that flags them is one
  somebody will disable.

**S2 — shared state lives above; a stage writes only inside its own directory.**

- **Test, and one half alone is worse than useless:** (a) run one stage and read
  `git status` at the parent — every changed path is under that stage. (b) git
  **cannot see the big files**, so compare their shas against the last archive
  too. Half (a) alone passes while a stage overwrites another stage's `.DM`.

**S4 — the description is input; everything else is derived.** No produce and no
run may edit `task.json`.

- **Test:** hash it before and after.

**S5 — identity is calculation-level; the run index is invocation-level.** Nothing
about a run may change the id, or the files it produced would be orphaned by the
act of producing them.

### A history you can read back

**S3 — a run records what it started from.** `run.json`'s `continued_from` names
the run directory its files came from, or is absent when it started from the
structure.

*This survived a design change and its mechanism did not.* When stages chained, an
inherited file arrived as a symlink and became real when localised — the type
change *was* the record. Stages no longer chain, so the record is explicit. It is
also better: a symlink says *something was inherited*, not **which run**, and with
several attempts per stage that is the question you actually have.

- **Test:** for every run in a finished tree, `continued_from` names a directory
  that exists or is absent — never something deleted or never there.

**S6 — a restored folder is internally consistent.** `task.json` is tracked text,
so a restore brings back the description *together with* the decks it produced.

- **Test:** restore any commit and re-run the produce with `dry_run` — it reports
  no change. One exclusion only: PROVENANCE stamps `generated-at`, so that key is
  ignored.

**L3 — every commit and tag names its calculation.** The message carries the id
and the stage; a finished stage is tagged `<id>/<stage>/<UTC>`. Nothing is
normalised — a name needing repair is **refused**, because silently fixing an id
would decouple the history's name from the folder's.

**L4 — tags are stage completions only.** Pre-produce saves are commits, reachable
through `snapshot list`. Tagging them too would bury the points you meant to
reach among the ones you passed through. A hand-made tag is your own business.

### The folder is also a real git repository

*Found by a fresh workflow read, 2026-08-08. These are the rules that were
never written because the document assumed molbuilder's verbs are the only
door into the folder. They are not — the folder is a git repository, and the
people using it know git.*

**S7 — a file that changes category is removed from the store it left.**
Nothing untracks a file that becomes large, so it ends up **tracked *and*
archived**: S1's other losing branch, and a large blob committed on every save
from then on.

This was rare while the classification was a list of names — it took somebody
editing the list. **With the size gate it is ordinary**, because files grow: an
`.EIG` at 8 MB last save and 12 MB this one crosses the line by itself, with
nobody deciding anything.

- **Fails as:** git history grows a multi-gigabyte blob that no S1 walk of the
  *current* tree will notice, because the current tree is classified correctly.
- **Test:** save a file just under the limit, grow it past the limit, save
  again — assert it is archived **and no longer tracked**. Then the same in
  reverse: a file that shrinks below the limit is tracked and dropped from the
  archive.

**S8 — using git directly must not silently desynchronise the two stores.**
`.binsnapshots/` is gitignored, so `git checkout <older-commit>` in a
calculation folder rewinds the text and **leaves every big file exactly where it
was**. The result is a folder no save ever produced: inputs from one point,
density matrix from another.

Nothing reports it, and the person who did it has no reason to suspect: they used
a command that works correctly everywhere else.

- **Fails as:** the run that follows uses last week's inputs with this week's
  state, converges, and is believed.
- **Test:** `git checkout` an older commit by hand, then ask molbuilder for the
  folder's state — it says the binaries do not match the checked-out commit, and
  says which. It may not simply refuse the next operation with a message about a
  dirty tree, which is what happens today and explains nothing.

**S9 — two saves of one folder cannot interleave.** There is no lock. Git
serialises its own index, but the archive's build-verify-swap does not — and
two `prep` runs, or a CLI and the browser, can reach it at once.

- **Fails as:** two processes build `.binsnapshots/<same-sha>.tmp` and race the
  swap. A1's atomicity holds for one writer and says nothing about two.
- **Test:** two concurrent checkpoints of the same folder; afterwards exactly one
  archive exists for that commit and it verifies (I2).

### Depth, and both folder shapes

**L1 — one repository per calculation**, covering the root and every stage
beneath it.

**L2 — the archive matches at depth.** A gitignore pattern with no slash matches
at *every* level, so a nested `01_coarse/job.DM` is ignored by git — and if the
archive walk only looked at the top level it would be ignored **and** unarchived:
in no snapshot at all. Both sides must resolve depth the same way.

**L7 — a change to a big file alone still leaves a checkpoint.** Big files are
gitignored, so `git status` is clean when only they changed; the save must not
conclude there is nothing to do.

- **Fails as:** you re-run a stage that rewrites its `.DM` and nothing else, save,
  and are told the tree was clean. The new density matrix is in no snapshot.
- **Test:** change only a big file, save, and assert a new archive exists whose
  MANIFEST records the new sha.

---

## 12. Status

| Rule | | |
|---|---|:--:|
| **S1** | everything is stored; the only exclusion is the store itself | ⛔ `*.MD` / `*.MD_CAR` are ignored and unarchived |
| **S1a** | `.gitignore` generated, one source | ⛔ a hand-kept list sits beside the generated block |
| **S1b** | the store is chosen by measuring the file | ⛔ **not built** — there is no size limit anywhere; the gate is the glob list |
| **S1c** | the classification lives in molbuilder's config, one home | ⛔ **not built** — every folder has its own `.mbcheckpoint.json`, writable from CLI and web |
| **I1** | archived content is never modified | ✅ |
| **I2** | a MANIFEST is authoritative | ✅ |
| **I2a** | a restore replays the save, and consults nothing | ✅ |
| **I2b** | the records themselves are tamper-evident | ⛔ neither is |
| **I2c** | the restore's gate reads the record, not the config | ⛔ the gate derives from the glob list; the copy derives from the MANIFEST |
| **I3** | one mover, one replacer, no third | ✅ |
| **I4** | no git in a generated wrapper | ✅ |
| **A1** | build, verify, swap, delete | ✅ |
| **A2** | verify before mutating, in order | ✅ |
| **A3** | the save precedes the change | needs the prep prompt |
| **A4** | a restore is whole or does not happen | ⛔ `--no-binaries` ships on three surfaces and skips both checks |
| **A5** | warn about unsaved work, then obey | ⛔ refuses instead of asking; no `--force` |
| **S2** | a stage writes only inside itself | needs the layout |
| **S3** | a run records what it started from | needs the layout |
| **S4** | the description is never modified | needs the description |
| **S5** | nothing about a run changes the id | ✅ |
| **S6** | a restored folder explains itself | needs the description |
| **L1** | one repository per calculation | ✅ |
| **L2** | the archive matches at depth | ✅ |
| **L3** | every commit and tag names its calculation | ✅ |
| **L4** | tags are stage completions only | ✅ |
| **L7** | a big-file-only change still saves | ✅ fixed in `1e87e01e`; two tests in `test_checkpoint_nested_layout.py` |
| **S7** | a file that changes category leaves the store it came from | ⛔ nothing untracks; **routine once the gate is a size**, because files grow |
| **S8** | using git directly does not silently desynchronise the two stores | ⛔ `git checkout` rewinds text and leaves every big file |
| **S9** | two saves of one folder cannot interleave | ⛔ there is no lock |
| **L8** | a saved attempt never differs afterwards | needs the layout |

### What cannot be proved, and is flagged instead

**A lost archive cannot be detected, only suspected — and the code already says
so where this document did not.** Big files are gitignored, so **git records
nothing about what a commit *should* have contained.** If an archive directory
goes missing, there is no list anywhere saying it ever existed.

`Repo.missing_archive_warning` is the honest response: if you restore a commit
with no archive *and* the folder plainly uses big files — there are some on
disk, or other commits have archives — it says so loudly rather than returning
text-only and looking fine. It cannot distinguish "the archive was lost" from
"this commit legitimately had no binaries", and it does not pretend to.

**Archives are never reclaimed.** Delete a branch, or leave a commit
unreachable, and its `.binsnapshots/<sha>/` stays on disk forever. There is no
`prune` verb. Not a correctness problem — nothing is lost, which is the
direction this system errs in on purpose — but it means the archive only ever
grows.

### Two things that are not rules

> **These two were rules, and you will find them cited as `L5` and `L6`.** They
> were demoted here, not deleted — the behaviour is unchanged and still tested.
> **`L5` → *Disk cost*** below: storing identical content once is a property the
> config tunes, and § 1 already forbids the only thing that could make it an
> invariant (storing less). **`L6` → *Verifying without restoring*** below: a
> verb that does not exist yet is a gap, not a rule a change can break. **Neither
> id is valid; cite the paragraph.**

**Disk cost.** Identical content is stored once, so a second save of an unchanged
2 GB file costs nothing. That is a property, not an invariant: it is tuned through
the config and **never** by storing less. It sits here rather than in § 7 so
nobody reads the two as peers again — this document has already lost a file to
that mistake.

*Known and cosmetic:* the size the surfaces print sums file sizes and counts each
hard link in full, so ten saves of an unchanged 2 GB file display as 20 GB where
the disk holds 2. Five call sites, and no `prune` verb for the number to inform.

**Verifying without restoring.** `_verify_archived_binaries` already checks
everything I2 asks and touches nothing — but it is reachable only from the restore
path, so the only way to learn an archive is intact is to attempt a restore. That
is the worst moment to find out. A `snapshot verify [<ref>]` verb is a few lines
over a helper that exists.

---

## 13. How the tests are designed

**Every test is derived from this document, never from the source.** A test that
reads the implementation to build its assertion can only confirm the code still
says what it said; it passes for behaviour that has drifted away from this
contract and fails for a rename that changed nothing.

### 13.1 Fixtures come from the config, not from imagination

This is the rule this module learned the hard way, and it is worth the whole
section.

There **is** a test that walks a saved folder and asserts every file is in git or
in the archive. It is correctly written. It did not catch `*.MD` sitting in
neither store for months — because **a walk can only judge files the fixture
created, and no fixture ever created a `.MD`.**

> **So the fixture is generated from the configuration**, not hand-written:
>
> - for every engine entry, for every pattern it names → make a file, assert it
>   is stored;
> - make an *unlisted* file above the size limit and one below it → assert each
>   lands in the right store;
> - assert the "never stored" list is exactly `.binsnapshots/` and nothing else.
>
> Adding a pattern to the config extends the test. Adding an engine gets a full
> suite for free. Nobody has to remember.

### 13.2 The checks that cannot be defeated by a short fixture

A generated fixture is better than a hand-written one and still only covers what
the config knows about. Two assertions are made about the **rules themselves**,
so no fixture can be too small for them:

- **Every pattern in the generated `.gitignore` is an archive pattern.** If git
  is told to skip something the archive does not take, that file is in no store —
  and this catches it without any file existing at all.
- **The ignore file matches what regenerating it would produce.** `.gitignore`
  records a *derivation*, so the derivation is the check (I2b).

### 13.3 What that rules out

- **Asserting on emitted text where the end result is what matters.** A test
  that greps generated bash for an expected glob passes for a glob that is
  present and wrong. Run the thing and look at what moved.
- **Lists copied out of the code.** A pinned list of extensions is a second copy
  of the classification, and it will agree with the first until the day it
  matters.
- **Mocking the filesystem.** These rules are about what is on disk after an
  operation. A mocked `is_file` stays green while the real predicate reads the
  wrong names.

### 13.4 Where each rule is asserted

All eight `test_checkpoint_*.py` files are here. A file in no row is a file
nobody is maintaining against this document, which is how the two drift.

| Rule | Where |
|---|---|
| S1, S1a | `test_checkpoint_nested_layout.py` — the store-or-store walk |
| S1b, S1c | **not yet written** — the rules are new (§ 12) |
| I1, I2, A1, A2 | `test_checkpoint_invariants.py` |
| I2a, I2b, I2c | **not yet written** — see § 12 |
| I3, I4 | `test_checkpoint_invariants.py` |
| A4, A5 | **not yet written**, and A4 starts red: `test_checkpoint_lifecycle.py` currently *pins* the violation |
| L1, L2, L7 | `test_checkpoint_nested_layout.py`, `test_checkpoint_manifest_format.py` |
| L3, L4 | `test_checkpoint_invariants.py` |
| S5 | `test_checkpoint_invariants.py` |
| S7, S8, S9 | **not yet written** — see § 12 |
| the MANIFEST format | `test_checkpoint_manifest_format.py` |
| repo boundaries | `test_checkpoint_repo_scope.py` |
| the HTTP routes | `test_checkpoint_routes.py` |
| the wrapper carries no git | `test_checkpoint_wrapper_isolation.py` |
| the verbs end to end — init, checkpoint, tag, branch, restore, and their refusals | `test_checkpoint_lifecycle.py` |
| the sidebar's read is cheap and does not poll | `test_checkpoint_sensor_js.py` |

> **One of these tests contradicts this document, and that is worth knowing
> before you read it.** `test_restore_include_binaries_false_skips_integrity_and_binaries`
> asserts the behaviour A4 forbids — it pins the flag *and* its skipping of both
> gates. It is not a wrong test; it correctly describes what ships. It retires
> with the flag.

---

## 14. Every section, and what it is for

| | |
|---|---|
| **1** | the one promise everything else serves |
| **2** | the boundaries — what it will not do, and what it does not decide |
| **3** | the two stores, and why the choice between them is a measurement |
| **4** | the parts, and which of them you ever touch |
| **5** | the commands |
| **6–7** | what actually happens on a save and on a restore, in order |
| **8** | why a flat folder depends on this and a nested one merely benefits |
| **9** | who decides to save, and when you are asked |
| **10** | worked examples, including the mistake that loses today's work |
| **11** | the rules a change must not break |
| **12** | which of them hold right now |
| **13** | how to test them |
| **14–15** | this index, and where the code is |
| **16** | what belongs to another document |

---

## 15. The file map

| | |
|---|---|
| `molbuilder/checkpoint.py` | all of it — `Repo` is the class every surface goes through |
| `Repo.init` / `.checkpoint` / `.restore` | the three verbs everything else is built on |
| `Repo.branch` / `.tag` / `.list_checkpoints` / `.diff` | navigating a history |
| `Repo.archive_globs` / `.set_archive_globs` | the classification, moving to molbuilder's config |
| `checkpoint_message` / `stage_completion_tag` / `parse_stage_completion_tag` | the naming rules (L3, L4) — written and read through one parser so the two cannot drift |
| `CheckpointError` and its four subclasses | what a refusal raises |
| `molbuilder/web/blueprints/checkpoint.py` | the HTTP routes the sidebar calls |
| `tests/test_checkpoint_*.py` | eight files — § 13.4 maps them to rules |

---

## 16. What this does not own

- **How to use it** — the CLI verbs, the routes, the sidebar panel, and what is
  unbuilt (archive pruning, `snapshot diff`) —
  [`running-a-job.md`](?doc=execution/running-a-job.md) § 6.
- **The file formats** — [`job-contracts.md`](?doc=execution/job-contracts.md) § 6.1.
- **The folder being saved, and the two moments a save is offered** —
  [`engines/stages.md`](?doc=engines/stages.md) § 7.
- **Phasing and open questions** —
  [`web/staged-runs-architecture.md`](?doc=web/staged-runs-architecture.md) and
  [`roadmap.md`](?doc=roadmap.md) (R3).
