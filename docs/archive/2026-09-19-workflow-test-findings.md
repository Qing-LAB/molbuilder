# Workflow-test findings, 2026-09-19 — archived 2026-09-20 (historical record)

> **Not a source of truth. All three findings are CLOSED**, each verified
> against the code on 2026-09-20 rather than from the commit log:
>
> * **W1** — resolved by decision, not by a fix. The tab-to-tab jump was
>   removed (user: *"we just skip the fancy tab to tab jump connection to
>   avoid implicit coupling"*), so there is no navigation left in
>   `_sendToTaskSetup` and the hint now reads *"Nothing runs, and nothing
>   opens: go to Task setup and pick that folder."*  The failure it describes
>   cannot be spelled.
> * **W2** — fixed. Every card the description branch paints is painted by the
>   hand-over and empty branches too, through the same renderers, and all
>   per-folder state is one object replaced wholesale. Pinned by
>   `tests/test_task_setup_one_folder_e2e.py::test_opening_another_folder_leaves_nothing_of_the_first`.
> * **W3** — fixed, and option **(b)** is what shipped: `/api/results/dir`
>   calls `openable_in` before the container branch and keeps `st = None`.
>   Measured on the directory this note names —
>   `transport/junction-ct` answers `place: container`,
>   `openable: junction-ct.transport.json`.
>
> The measured end-to-end run at the foot is kept: it is the record of the
> workflow this session exercised, and nothing else holds those numbers.


## W1 — "Send to Task setup" opens the WRONG calculation
The button's own text: *"...into the folder selected in the sidebar, then opens
Task setup **there**."*  It wrote the handover correctly into
`optimization/junction6-hier` (source xyz + molstruct + template.toml +
task.1st.json all landed) and then opened Task setup on
`spectrum/bridge-hier` — the page's own last-viewed folder, restored over the
hand-over's destination.  The user then edits the wrong calculation unless
they notice the breadcrumb.
Repro: hand over from Structure optimization into a folder that is not the one
Task setup last showed.

## W2 — Task setup keeps the previous calculation's cards on a directory change
With the sidebar moved to `junction6-hier` (breadcrumb, "What came over" and
the `task.1st.json` banner all correct and SIESTA), two cards still rendered
the PREVIOUS calculation:
  * "And when you prep, per stage" showed `freq` / `01_freq/` — bridge-hier's
    stage, while junction6 has no stages chosen yet;
  * "What this calculation writes" listed `bridgespec_initial.xyz`,
    `bridgespec.spectra.json`, `bridgespec_01_freq.py`, ... — the other
    calculation's filenames, under a heading claiming to describe this one.
A reload clears it: measured after F5, `bridgespec` appears nowhere in the page
text and `junction6` does.  So it is a stale render on the directory-change
path, not bad data — the same shape as the Results picker defect fixed in
e808ce68 (a scan that changes what is current must re-render everything it
owns, and these two cards were not re-rendered).

## W3 — a composite's PRODUCT lives at the root, which the door refuses to open
`jobset summarize run` writes the transport record — the whole point of the
calculation — to the calculation ROOT:
    projects/claude-junction/transport/junction-ct/junction-ct.transport.json
That root is `shape: hierarchical`, so § 1.4 makes it a CONTAINER, and
`/api/results/dir` skips `openable_in` for a container (6ddc551a).  Result:
`openable: null` for the one directory holding the calculation's answer.  The
file is listed and a person can click it — `parser: transport-json`, and
`lib/inspectors/transport.js` claims it — but the tab will never open it for
them, and the 5-rung ladder's runs each offer their own `.out` instead.

This contradicts the box added to `project-layout.md` § 1.0 this session:
*"A product has ONE home, and it is the run."*  A COMPOSITE's product is not
any one rung's — it aggregates across rungs (and across bias points, once a
sweep exists), so the root is arguably where it belongs and the box is too
strong.

TWO WAYS, and it is a contract decision, not a code one:
  (a) `summarize` writes the record into the transmission run, keeping § 1.0's
      box true — but then a bias sweep's aggregate has no home, because it
      spans several device/transmission runs;
  (b) § 1.0's box gains a second clause: a CALCULATION-level product lives at
      the root, and the container branch asks the door for it — the container
      still has no run STATE, which is the thing 6ddc551a actually fixed.
(b) looks right, and is a smaller change than it sounds: the container branch
would call `openable_in` and keep `st = None`.

## Measured, end to end (2026-09-19)
  optimization/bridge-flat        pyscf    flat          finished
  spectrum/bridge-hier            pyscf    hierarchical  finished (24 modes)
  optimization/junction6-hier     siesta   hierarchical  finished — 58 atoms
                                  (48 Au frozen), 45.3 s, 88 SCF iters, rc=0
  transport/junction-ct           siesta   5-rung ladder, all finished:
                                  seed 22 s, electrode_L, electrode_R,
                                  device (transiesta), transmission (tbtrans)
  summarize run  ->  junction-ct.transport.json, G(E_F) = 0.1475 G0 at V=0
Results matrix: 37 directories — 8 runs (all finished, each with the right
openable), 12 containers, 17 unmarked; none inventing a run state.
