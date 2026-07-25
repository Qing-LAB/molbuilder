# Running a batch of jobs — the big picture

> **Who this is for:** anyone who wants to understand, end to end, how
> molbuilder takes a structure you set up in the browser and turns it into a
> *batch of related calculations* that run on your workstation or an HPC
> cluster — and how you monitor, checkpoint, and branch from there.
>
> This is the **plain-language overview**. The precise contracts live in
> [`protocols/staged-execution.md`](protocols/staged-execution.md) (the
> framework) and [`staged-relaxation-guide.md`](staged-relaxation-guide.md)
> (copy-paste CLI how-to). Read this first for the shape of the whole thing.
>
> **Status note:** the *framework* (produce a bundle, run it, checkpoint it,
> monitor it) is built and works from the command line today. The *browser
> buttons* that produce and track a bundle are being wired now (Phase 1–2 in
> `staged-execution.md` § 15). Where something is browser-planned vs
> already-working, this doc says so.

---

## 1. The one-sentence version

You set up **one structure and one recipe** in a tab; molbuilder turns it into
a **self-contained folder** (a "bundle") holding every job in the batch, wired
so they **share common inputs but keep their own results**; you **run** that
folder on a workstation or a cluster with one command; and you **watch,
checkpoint, and branch** it from the same places you already use.

```
   Build / set up            Produce                 Run                Watch + protect
   a structure  ───────►   a bundle folder  ───►  on workstation  ───►  results + checkpoints
   (a tab)                 (job-set.json +        or HPC             (Results tab +
                            per-stage files)      (one command)       sidebar snapshot panel)
```

The key idea: **a multi-stage relaxation and a benchmark sweep are the same
shape** — *N related jobs that share one package, each in its own directory,
each with its own scheduler resources.* So they use one machinery.

---

## 2. What you do in the browser, and what you decide

You work in a **setup tab** (Build a SIESTA relaxation, a PySCF spectrum, a
transport calculation). The viewer holds the structure; the form holds the
recipe. Two things are worth calling out:

**The cell travels with the structure.** The simulation box (cell / vacuum /
which axes are periodic) is part of the structure's data in the viewer. You
see and confirm it on the tab's **Cell page** before you generate. It is then
written *explicitly* into every generated input file — so what runs is exactly
what you saw. You never set the cell separately for the batch; it rides along.

**Staging is a small table.** For a relaxation, the tab has a **stage table**:
each row is one tier of the ladder (a loose warm-up → a tight final), and a
**"Stage strategy" preset** (e.g. *Publishable*) ticks the sensible set for
you. You can hand-tune any row.

### The "you decide / molbuilder does" split

molbuilder organizes and informs; **you** make every move that spends real
resources or changes state. Nothing irreversible happens from a button.

| molbuilder does (safe, automatic) | You decide (explicit) |
|---|---|
| Validate the recipe; show issues inline | Fix the issues, or accept warnings |
| Produce the bundle folder | Where it lands (a project folder) |
| Show the **plan** (what will run, with what resources, in what order) | Whether the plan is right |
| Show the exact **run commands** | Actually **submitting** the jobs (in a terminal) |
| Show per-stage **status** as it runs | When to **checkpoint / branch / restore** |

> Why isn't there a "Submit" button in the browser? Submitting consumes a real
> cluster allocation — it's the one irreversible, outward-facing step. Per the
> framework's "molbuilder informs, the user decides" rule, the browser shows
> you the precise command and you run it. (Producing, planning, and monitoring
> are all safe, so those *are* in the browser.)

---

## 3. What gets generated — the bundle

"Generate" produces a **bundle folder** in the project directory you pick. For
a 2-stage benzene-dithiol relaxation it looks like this:

```
bdt/                                  ← the bundle root: shared, written once
├── C.psml  S.psml  Au.psml  H.psml   ← pseudopotentials (shared by every stage)
├── mb_monitor.py                     ← the live progress monitor (shared)
├── bdt_stage1.fdf   bdt_stage2.fdf   ← one input file per stage
├── job-set.json                      ← the plan: the jobs, their order, resources
└── STAGE-PLAN.md                     ← the same plan, human-readable
```

`job-set.json` is the spine — it lists each job, its input file, its scheduler
resources (cores / memory / walltime / GPU), which job it **depends on**, and
which restart files to **carry forward**. It carries *opaque filenames and
resources only* — it never contains science, so the same machinery runs a
SIESTA ladder or a PySCF sweep unchanged.

`STAGE-PLAN.md` is that plan written out for you to read *before* you run:
each stage's resolved partition/walltime/hardware, its carry set, the
dependency graph, and what happens on non-convergence.

---

## 4. How the stages share inputs but stay separate

This is the part you asked about — and yes, it's **symbolic links**. When you
"prepare" the bundle (next section), molbuilder lays out **one subdirectory per
stage** and links the shared pieces in, so there's exactly one physical copy of
each big file:

```
bdt/
├── (shared root, as above)
├── point-stage1/                     ← stage 1's own world
│   ├── C.psml → ../C.psml            ← symlink to the shared pseudo
│   ├── bdt_stage1.fdf → ../bdt_stage1.fdf
│   ├── mb_monitor.py → ../mb_monitor.py
│   ├── bdt.out  bdt.XV  bdt.DM       ← its OWN results (real files, isolated)
│   └── .git/                         ← its OWN checkpoints (see § 6)
└── point-stage2/                     ← stage 2's own world
    ├── C.psml → ../C.psml
    ├── bdt_stage2.fdf → ../bdt_stage2.fdf
    ├── bdt.XV → ../point-stage1/bdt.XV   ← "carry-forward": start from stage 1's result
    └── bdt.out  bdt.XV  bdt.DM           ← its OWN results
```

Three tiers, three owners:

- **Shared, written once → the bundle root.** Pseudopotentials, the monitor,
  the input files, the plan. Every stage **symlinks** them in — one copy on
  disk, no duplication.
- **Private per stage → `point-<name>/`.** Each stage's own `.out` / `.XV` /
  `.DM` are *real files* in its own folder, so stage 2 can never clobber stage
  1's results. This folder is also the unit for checkpoints and resume.
- **Shared at runtime → carry-forward symlinks.** Stage 2's restart file
  (`bdt.XV`) is a *link* to stage 1's output. It's empty until stage 1 finishes;
  the dependency ordering guarantees stage 1 runs first. That's how "start the
  tight relaxation from where the loose one converged" happens with no copying.

Only the things that genuinely *differ* between stages (the `.fdf` with its
tighter force tolerance) are separate files; everything common is a link.

---

## 5. How you run it — workstation or HPC

Running is **three commands**, and the browser shows you the exact ones to
paste. They're the same whether you're on a laptop or a login node — only the
final `--mode` differs.

```bash
molbuilder jobset prep   ./bdt                     # lay out the point-*/ dirs + links + wrappers
molbuilder jobset plan   ./bdt                     # review the chain + per-job resources
molbuilder jobset submit ./bdt --mode direct       # WORKSTATION: run the stages in order, locally
#  or:
molbuilder jobset submit ./bdt --mode submit --domain public   # HPC: submit an sbatch chain
```

- **`prep`** builds the layout in § 4 and renders each stage's run script.
  Those run scripts **activate the right software environment** for you — this
  is what "assuming molbuilder is deployed correctly" means: the machine you
  run on must have its environment configured once, and then every job uses it.
- **`plan`** just prints the plan (read-only) so you can sanity-check before
  spending anything.
- **`submit --mode direct`** runs the stages one after another on *this*
  machine — right for a workstation or a short ladder.
- **`submit --mode submit`** hands the batch to the cluster's scheduler as a
  dependency chain (stage 2's job waits for stage 1 to converge), with each
  stage getting its own cores / memory / walltime / partition. Right when
  stages differ in cost or the whole thing exceeds one walltime.

Because molbuilder runs *on* the machine with your files (your workstation, or
the cluster's login node), there's normally **no copying between hosts**. If
you do keep the browser on a laptop and run on a remote cluster, you copy the
bundle folder over once (`scp`/`rsync`) and run the same three commands there —
the bundle is self-contained.

---

## 6. Checkpoints — protect a good state before you experiment

Each `point-<name>/` stage folder is its **own checkpoint repository** (that
`.git/` in § 4). This is the existing checkpoint framework — you already have
it in the **projects sidebar's checkpoint panel**, and it works on any run
folder you navigate into. A stage folder *is* a run folder, so it just works.

The move it's built for: **you got a converged result and now want to try a
variation without risking the good one.**

```
1. Tag the good state:        snapshot tag  stage2-converged
2. Branch an experiment:      snapshot branch stage2-tzp        (try a bigger basis)
3. …run the experiment…
4. Worse?  Rewind:            snapshot restore stage2-converged
```

- **tag** = a milestone ("this is the publishable one").
- **branch** = an experiment that leaves the tagged state fully recoverable.
- **restore** = rewind everything in the folder to the tag, byte-for-byte,
  including the big binary result files.

This is a *different* kind of history from carry-forward: carry-forward is the
*scientific* lineage down the ladder (stage 1 → 2 → 3); checkpoints are the
*exploratory* lineage *within one stage* (converged → tzp experiment). They
never conflict. And like submitting, molbuilder never tags/branches/restores on
its own — you decide.

---

## 7. How you monitor it — and yes, results parse the same as before

Nothing new to learn here: a running batch is just run folders, and molbuilder
reads them the **same way the Results tab already does**.

- **Live, per stage:** `molbuilder jobset status ./bdt` (or the browser status
  view, Phase 2) tells you, for each stage: *not-started / running / finished /
  failed*, which restart files exist, and **which stage is the first one that
  still needs to run** (your resume pointer). It reads each stage folder with
  the very same decoder the Results tab uses — no separate, drift-prone parser.
- **The trajectory + final result:** open a `point-<name>/` folder in the
  **Results tab** exactly as you open any finished run today. The optimization
  trajectory, energies, forces, and the final geometry come through the same
  path — including the region labels / frozen tags you set in Build, which now
  ride along onto the loaded structure (the "metadata bridge").

So the answer to *"is parsing the result the same as before?"* is **yes** —
the batch just produces ordinary run folders, and every existing viewer and
decoder works on them unchanged.

---

## 8. The whole loop, at a glance

```
 SET UP            PRODUCE               RUN                     WATCH                 EXPERIMENT
 ┌────────┐        ┌──────────┐          ┌───────────────┐       ┌─────────────┐       ┌──────────────┐
 │ tab:   │ Gen →  │ bundle/  │  prep →  │ point-stage1/ │ run → │ Results tab │  →    │ tag converged│
 │ struct │        │ job-set  │  plan    │ point-stage2/ │       │ + status    │       │ branch expt  │
 │ + cell │        │ .json    │  submit  │ (symlinked    │       │ (same       │       │ restore if   │
 │ + stages        │ + .fdfs  │  (direct │  shared,      │       │  parser)    │       │ worse        │
 └────────┘        │ + pseudos│  or HPC) │  own results) │       └─────────────┘       └──────────────┘
   you decide        molbuilder            you submit             molbuilder             you decide
   the recipe        assembles             (explicit)             informs                the experiment
```

---

## 9. Does this hang together? Open questions & possible gaps

This section exists so we can sense-check the design and find missing needs.
Honest status and the things worth a decision:

**Solid and already working (CLI):** the bundle format, the symlink layout,
carry-forward, the dependency chain, both run modes, per-stage checkpoints,
and status/monitoring. These are built and tested.

**Being wired now (browser):**
- *Produce a bundle from a tab.* The stage table exists in the Build tab but
  currently doesn't emit a bundle yet — that's Phase 1, in progress.
- *Plan + status in the browser.* Read-only views over the same functions the
  CLI uses — Phase 2.

**Open questions to decide:**
1. **Producer coverage.** SIESTA relaxation ladders are the first producer.
   Transport **bias scans** (one run per bias voltage) and spectra's layered
   chain are natural batches too, but aren't wired yet. Do we want them in the
   first release, or ladder-only to start? (`staged-execution.md` § 15, D6 says
   each generator gains the same `--jobset` option; order is open.)
2. **Remote deployment.** Today the assumption is molbuilder runs *on* the
   machine that runs the jobs (workstation or login node), so there's no
   host-to-host copy. If a real "browser on laptop, compute on a separate
   cluster" workflow matters, we'd need an automated ship step — deliberately
   out of scope for now. Is that assumption right for how you actually work?
3. **Branch from the browser.** The sidebar checkpoint panel exposes
   init/checkpoint/tag/restore; **branch** is CLI-only today. If "explore an
   alternative from a converged state" should be a one-click browser move,
   we'd add a branch endpoint. Worth it, or is branching fine at the terminal?
4. **Environment prerequisite.** The whole thing assumes each run machine has
   its molbuilder environment configured once (the `activation` setting). If
   it's not, `prep` fails with a clear message. Should the browser check this
   up front and warn *before* you produce a bundle you can't yet run here?
5. **Non-SIESTA checkpoints.** Checkpoints archive big binary result files by
   content hash. That's tuned for SIESTA outputs; PySCF's outputs differ. Does
   the same protection story hold for a PySCF batch, or does it need its own
   pass?

If the shape above matches how you'd actually want to work, the remaining work
is mostly *wiring* (browser buttons over built functions) rather than new
machinery — which is the sign the design is in the right place.
