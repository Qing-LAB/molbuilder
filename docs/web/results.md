# Results tab — opening a finished calculation

**Role:** contract
**Domain:** web
**Companions:** [`presenters.md`](?doc=web/presenters.md) — the registry that
picks the viewer (this tab drives it); `trajectory.md` and `spectra.md` — the two
heavy viewers this tab hosts (their own docs); [`projects.md`](?doc=web/projects.md)
— the file layer the picker lists through; [`web-api.md`](?doc=web/web-api.md) —
the `/api/watch/*` and `/api/system/load` routes.

You ran a calculation; you open it on the **Results** tab. The tab is a
**dispatch shell**: a file picker across the top, and one panel below that
becomes *whatever viewer fits the file you picked* — a 3D structure, a trajectory
movie, or a spectrum. The tab itself draws nothing; it delegates every file type
to a viewer.

## 1. What the page is

`/results` renders a template and does the rest in the browser. Its controller
(`results/viewer.js`) is deliberately tiny — it owns one mount point
(`#inspector-host`), holds exactly one live viewer handle, and does only three
things on each selection: **pick** the viewer for the file, **dispose** the
previous one, and **mount** the new one. All the file-type knowledge lives in the
viewers (the "presenters", [`presenters.md`](?doc=web/presenters.md)), not here —
so adding a result type is a new presenter module, never an edit to this
controller.

```mermaid
flowchart TD
  U["you pick a file (the dropdown auto-picks the newest)"] --> EV["a file-selected event"]
  EV --> CTRL["results/viewer.js — dispose the old viewer, mount the new one"]
  CTRL -->|"who shows a file named like this?"| REG["the presenter registry"]
  REG --> ENG["the matching viewer renders into the one panel"]
  ENG --> S["a 3D structure · a trajectory movie + plots · a spectrum + modes"]
  ENG -. "if the run is still going" .-> POLL["it polls for new data — every 15s (trajectory) / 2s (spectra)"]
  POLL -. "new data" .-> ENG
```

## 2. Picking a file

The picker (`lib/results/file-picker.js`) lists the **result-class** files in the
current project folder — the files some presenter has marked as a result
(`isResult`, see presenters.md) — newest first, grouped by kind (the group with
the newest file floats to the top). It **auto-picks the newest** so a viewer
appears without a click, and mirrors your pick to the sidebar so the highlight
matches.

### 2.1 The sidebar sets the scope; the dropdown decides what you see

These are the only two controls that touch what this tab displays, and they do
**different jobs**. Getting them confused is what produced the worst defect this
tab has had, so the division is written out in full:

| You do this | What happens |
| --- | --- |
| **Navigate the sidebar to another folder** | the dropdown **re-scopes**: it re-scans that folder for result-class files and **auto-picks the newest**, so the panel shows the run you just navigated to |
| **Single-click a file** in the sidebar | nothing here. That is a preview/browse gesture |
| **Double-click a file** in the sidebar | still nothing here. The sidebar never chooses a result |
| **Pick from the dropdown** | that file is mounted, and everything below follows it |
| **Refresh** | re-syncs the menu with the sidebar's current folder, re-scans for files written since, and tells a live viewer to re-fetch now |

**The dropdown dictates the display.** The four plots, the run-state badge, the
convergence-target card, the 3-D structure and the trajectory all render the
dropdown's current selection and nothing else. There is no second route to a
mounted viewer.

**The sidebar's only job on this tab is to say which folder you are in.** A file
click there cannot hijack a viewer mid-read — that is deliberate, and it is why
the dropdown exists as a separate control at all.

> **The defect this prevents** (2026-08-04). The dropdown used to scope itself
> once, when the tab mounted, and never again. Moving the sidebar to a different
> run folder therefore left it enumerating the *previous* folder's files — so the
> plots, the badge, the convergence targets and the structure all kept rendering
> a different run, with nothing on screen saying so. A **live** job was displayed
> as a finished one from another directory, and the numbers looked entirely
> plausible. Reloading the page or re-picking in the dropdown fixed it, because
> both make the picker speak; nothing else did. A menu that dictates the display
> has to be listing the folder you are actually in.

### 2.2 One scan, one choice, one announcement

Everything the picker does is one pass, and the shape matters because two of
this tab's three worst defects came from the same mistake — **deriving "which
file is current" twice, by two different routes.**

```mermaid
flowchart TD
  T["the folder changed · Refresh · tab re-entry"] --> S["scan it — list the folder,<br/>keep the result-class files, newest first"]
  S --> E{"any results?"}
  E -->|"none"| N["say so in the menu AND announce<br/>'nothing selected' — the panel clears"]
  E -->|"some"| K{"is the file we were<br/>already showing one of them?"}
  K -->|"yes"| KEEP["keep it"]
  K -->|"no"| NEW["take the newest"]
  KEEP --> ONE["<b>one</b> chosen file"]
  NEW --> ONE
  ONE --> A["label the menu with it<br/><b>and</b> announce it — same value, one step"]
  A --> M["the viewer mounts what was announced"]
```

**The chosen file is computed once and used for both.** The menu's label and the
mounted viewer are two uses of one value, never two derivations of "the right
one". Announcing without labelling — or labelling without announcing — is the
bug, not an optimisation.

> **Why this is spelled out.** The picker used to build two orderings of the same
> files: a flat list, newest-first, ties broken **by file name**; and a grouped
> list, ties broken **by category label**. It labelled the menu from the grouped
> one and mounted from the flat one. Those agree right up until several results
> share a timestamp — which is exactly what a job does when it finishes and
> flushes its outputs in the same second. On 2026-08-04 a pySCF run with four
> files stamped `10:31:08` showed `…molwatch.log` in the menu while displaying
> `…_optimized.xyz`. Both orderings were individually correct; having two was the
> defect.
>
> The empty case failed the same way from the other side: a folder with no
> results updated the menu and told nobody, so the previous folder's run stayed
> on screen — plots, badge and all.

### 2.3 A run is one result, not a pile of files

A calculation writes several files, and they are **not peers**. One of them is
the result; the rest are its working parts — the input echoed back, the seed the
next run warm-starts from, the optimizer's own per-stage streams, the engine's
verbose log. The menu lists **the result**, once.

**The master absorbs its satellites.** A presenter that recognises a master file
also says which files in the same folder that master subsumes; those do not get
their own entry. So a PySCF relaxation is one line in the menu, not five.

Two rules keep this from hiding anything:

- **Absorption needs the master present.** A satellite is only dropped when the
  file that subsumes it is in the same listing. Delete the master, or run a job
  that never wrote one, and the satellites list normally — you can always reach
  what is on disk.
- **The sidebar still shows every file.** Absorption narrows *the result menu*,
  which is a question about what you'd want to inspect. It is not a permissions
  or visibility rule; open any file from the sidebar as always.

> **What this fixes.** PySCF's generator already nominates a single result: its
> own manifest calls `<label>.molwatch.log` the *"unified per-step log… coords,
> energy (eV), forces (eV/Å), and SCF cycle history — single-file input for
> molwatch"*. The picker ignored that and listed the run's internals beside it,
> because the structure presenter claimed **every** `.xyz` and the trajectory
> presenter claimed `*_optim.xyz` as well. One relaxation therefore appeared as
> five results: the master log, the *input* coordinates, the warm-restart seed,
> and geomeTRIC's two per-stage trajectories. The generator was right; the menu
> was reading it at the wrong granularity.
>
> Note the naming: a staged run's master is `<label>-stage<N>.molwatch.log`
> while its satellites are plain `<label>_*`, so matching a master to its
> satellites means stripping the `-stage<N>` overlay suffix first
> (`execution/job-contracts.md § 2.3`).

> **⚠ This module is the consumer of that name, and it is changing.** The
> trajectory log will be named for **the deck that produced it** —
> `<label>_<stage>.molwatch.log`, the same name whether stages share a directory
> or each has its own — instead of carrying a `-stage<N>` infix
> ([`execution/job-contracts.md`](?doc=execution/job-contracts.md) § 6.3).
> **What that costs here:** the run decoder's stage regex keys on the hyphen
> form, so it changes with the rename, and anything that groups a staged run's
> logs by parsing `N` out of the filename must instead read the stage from the
> deck's name or its directory. Worth knowing before building on the current
> shape.

- **A file-selected event is the single source of truth.** When you choose from
  the dropdown, it fires a `fileSelected` event that the controller listens for.
- **Refresh re-scans the folder, and tells a live viewer to re-fetch its data
  *now*** rather than waiting for the next poll. The panel isn't torn down and
  rebuilt — the mounted viewer reloads in place — but that reload is a *clean*
  one (§ 4), so a trajectory jumps back to its first frame. The picker stays
  visible even when a folder has zero results, so Refresh is always reachable,
  and it re-scans automatically when you return to the tab (so a file written
  while you were away shows up).

## 3. Showing the file

The controller asks the registry "who shows a file named like this?", disposes
whatever was mounted (dropping its timers and 3D contexts so nothing leaks), and
mounts the chosen viewer into the one panel. For the slow, 3D viewers it first
drops an opaque **"parsing…" cover** over the panel so the *previous* scene can't
be mistaken for the new result while it loads; the cover lifts when the viewer
signals it has painted (or after a 15-second safety timeout).

The three viewers you can land in:

- a **read-only 3D structure** for a `.xyz`/`.pdb` ([`presenters.md`](?doc=web/presenters.md)),
- a **trajectory movie + plots** for an optimization log (`trajectory.md`),
- a **spectrum chart + modes** for a `.spectra.json` (`spectra.md`).

## 4. What a mounted viewer remembers

Each viewer keeps a small amount of state, and it's worth knowing the shape
because it explains how Refresh behaves. A viewer holds: the **parsed file**
(replaced whole on a file switch, never patched), your **per-file view** (which
frame or which mode you're looking at — reset when you switch files), and your
**per-session preferences** (which survive a file switch). If the run is still
going, a **poll timer** is running.

The one rule to remember: **"Refresh = open the same file again."** Refresh is
not a special path — it runs the exact same clean reload a file-switch does
(cancel anything in flight, clear derived data, reset the view, keep your
preferences). That single rule is what eliminated a whole class of
half-refreshed-state bugs. Two guards back it up: a **late response from a
previous file can't write into the current view**, and **partial frames** the
parser flags as in-progress are shown in the list but kept out of the plots.

## 5. Sending a finished run to the next stage — RETIRED (2026-08-29)

The always-visible **Bundle** card that stood below the viewer is gone,
with the whole calculation-to-calculation passing model it served (user
ruling): **one kind of job never bundles itself up for another.**  A
calculation that builds on a finished result CITES it — the transport
tab picks the junction attempt actively and prep fuses the final
geometry with the labels itself, with the sort and the gates the bundle
never had (`archive/2026-09-01-transport-design.md` § 4.1,
`transport/compose.py`).  Structure → execution hand-overs
(builder/modify → parameter tab → Task setup) are a different thing and
remain.  History: `docs/archive/2026-08-29-handoff-bundle.md`.

## 6. Watching the machine — the server-load strip

Below the viewer sits the always-visible **Server load** card — what
the machine itself is doing, as opposed to what your calculation produced. It
mounts on this tab and no other, because this is the tab you sit on while a run
proceeds; putting it on every page meant every page paid for a 1 Hz hardware
probe nobody was reading.

**It is collapsed on your first visit.** You click the `≡` pill to open it, and
that choice is remembered for the rest of the browser session (it resets when you
close the browser — this is a transient view preference, not a setting). The
default is deliberate: an expanded strip used to overlay the bottom of the plots
on every fresh visit, so you opt in to it rather than it opting you in.

While it is open, it asks the server for a fresh reading **once a second** and
keeps the last **600 readings — ten minutes** — as a sparkline behind each
number. It stops asking entirely when you collapse it, and pauses while the
browser tab is in the background: a hidden widget doing 1 Hz server work is pure
waste. Each cell colours its line by the current value — green below 50%, amber
below 80%, red at 80% and above.

### 6.1 The five cells, and the question each answers

| Cell | Number on the strip | The question it answers |
| --- | --- | --- |
| **CPU** | busy %, and `~N/M cores` | Is the run actually using the cores you gave it? |
| **RAM** | busy %, and used/total GB | Is this box about to run out of memory? |
| **GPU** | SM compute % | Are GPU kernels actually running? |
| **GPU BW** | memory-controller % | Is the GPU waiting on memory rather than computing? |
| **VRAM** | busy %, and used/total GB | Will a bigger system fit on this card? |

The GPU cells only appear when the server reports a usable GPU (§ 6.3).

### 6.2 The detail block is the part that diagnoses

Under each number is a text block that is always on screen — it used to be a
hover tooltip, which re-positioned itself on every 1 Hz redraw and was therefore
unreadable. The blocks carry the numbers a percentage alone hides:

- **CPU** — how many physical cores this box has, how many logical ones if SMT
  is on, and how many core-equivalents are busy right now. *Why it matters:*
  "50%" on a 20-physical / 40-logical box could mean ten cores pinned or twenty
  threads half-idle; `~10.0/20 cores` says which. It also prints the Unix load
  average over 1, 5 and 15 minutes and flags **`[over-subscribed: load > physical
  cores]`** — a run queue that is queueing looks identical to a healthy one at
  100% CPU, and only the load average tells them apart.
- **CPU, per socket** — on a multi-socket box, one row per socket. When one
  socket is above 70% and another more than 50 points below it, the block says
  **`[asymmetric: likely NUMA-pinned to one socket]`**. *Why it matters:* a
  SIESTA-GPU run pins its ranks to the socket nearest the GPU, so the other
  socket sitting idle is the sign the pin is **working** — aggregate CPU% reads
  that same healthy state as "the machine is half-used". Both sockets half-busy
  is the bad case: ranks spread across sockets, paying the interconnect penalty.
  (Absent on single-socket hosts and wherever `lscpu` can't be read.)
- **RAM** — used and total in GB.
- **GPU / GPU BW / VRAM** — one breakdown per device: name, SM compute %, memory
  bandwidth %, VRAM used/total, power draw, temperature, SM clock and memory
  clock. *Why those four extras matter:* power dropping mid-run means a thermal
  or power cap has engaged, and the SM clock falling while utilisation stays at
  100% is usually the underlying cause. The GPU BW cell adds the reading that is
  hardest to guess: **high bandwidth with low SM compute means the kernel is
  waiting on memory, not computing — more ranks won't help, a smaller block size
  might.** Fields the chip doesn't expose show as `—` rather than a zero.

### 6.3 When the GPU cells are missing

An empty GPU list has two causes, and they are opposite news:

```mermaid
flowchart TD
  Q["server reports no GPUs"] --> A{"was the NVIDIA library<br/>even installed?"}
  A -->|"no — this is a CPU-only install"| CPU["cells hidden, nothing said<br/>(nothing is wrong)"]
  A -->|"yes, and it refused to start"| ERR["cells hidden AND a warning line<br/>under the strip saying why"]
```

The server tells them apart with a `gpu_error` field on every reading: `null`
when this host simply has no GPU support installed, and the reason as text when
the NVIDIA library **was** installed — meaning this box is meant to have a GPU —
and could not reach the driver. Only the second case prints anything, because
only the second case is something being wrong.

That distinction was added on 2026-08-04 after the silent version cost real time:
a driver upgrade on the development host left the userspace library ahead of the
loaded kernel module, `nvidia-smi` was dead for five weeks, and all the monitor
did was quietly show a tidy two-cell strip that read as "this machine has no
GPU". The same broken driver would have failed any GPU calculation submitted in
that window, since CUDA reaches the driver the same way.

**You do not have to open the card to find out.** A fault that is only visible
inside a folded-away card is a fault nobody sees, and the card is folded by
default — so when the reading comes back faulted, the `≡` pill itself turns
amber and grows a `!`, and hovering it gives the reason. That is the one part of
the card that is always on screen. To get it, a collapsed card makes **exactly
one** request when the page loads and reads nothing from it but `gpu_error` —
one request per page load, not a stream, so the "collapsed means no polling"
rule still holds.

One more thing to know when you see that warning: **the driver is checked once,
when the server starts.** Fixing the host is not enough on its own — a server
that started while the driver was broken stays GPU-blind for its whole life.
Restart it. The server log carries the same message at warning level, so it is
also in the terminal you started the server from.

## 7. A worked example

You just ran a SIESTA geometry optimization; its `*_optim.molwatch.log` sits in
`projects/BDT/opt/`. Open the **Results** tab. The picker scans that folder, sees
the log is a trajectory-class file, and auto-selects it. The controller mounts the
**trajectory** viewer — a 3D movie of the relaxation plus energy and max-force
plots. The structure the viewer holds carries the run's own periodicity,
composed on the server (`/api/watch/load`: the cell from the output logs, the
axis kinds from the run's `.source` pair when one exists) — so what the Cell
page shows, and what an Export → Data writes, is the run's stated intent
rather than a browser guess *(2026-08-20; before this, every trajectory
export claimed all-isolated beside its own lattice)*. Because the run is
still going, the viewer polls `/api/watch/data` every
15 seconds and appends new frames live. When it converges you click **Bundle**,
and Results writes `handoff.xyz` + `handoff.molstruct.json` into
`projects/BDT/opt/handoff/`; the sidebar jumps there so you can load the
converged, labeled geometry into your next calculation.

## 8. When there's nothing to show

If no presenter is registered at all, the tab shows a clear configuration warning
rather than a blank panel. If the folder simply has no results yet, the picker
shows a placeholder and stays put — Refresh remains one click away.

## 9. Where the module stands (current → target ESM)

The Results shell is still **classic**: `results/viewer.js` plus
`lib/results/file-picker.js` are global-registered scripts
(`window.molbuilder.*`), not ES modules — they lean on the runtime registry to
load in order. Converting them is task #103 (the "remaining classic modules" pass,
alongside the runtime registry and the shared primitives —
[`archive/2026-09-01-roadmap.md § 3`](?doc=archive/2026-09-01-roadmap.md)). The heavy viewers this shell *mounts* are on
a different track — the trajectory and spectra engines convert in the #102
file-viewer pass (see [`presenters.md`](?doc=web/presenters.md)).

## 10. Test map

- `test_results_blueprint.py` — the page + the registered presenter set + script
  order.
- `test_results_folder_dispatch_e2e.py` — the pick → mount dispatch end to end.
- `test_inspector_pageshow_refresh_e2e.py` — the re-scan on tab return.
