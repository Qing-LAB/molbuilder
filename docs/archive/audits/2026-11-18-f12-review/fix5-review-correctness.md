*Verdict: closed in 2.77.9.* The correctness reviews of F12 fix 5, two rounds, read before its release. Every finding was fixed in 2.77.9 except these:
- **Export bar timing.** Between a new selection and its offer landing, the bar still exports the selection it was decided for. It never exports another calculation's selection.
- **A Load not started.** A Load that is not started, because another edit is running, says nothing (`tabs.md` § 2).
- **No test** for J2-D4 (the badge of a run with no frames yet), TS-D2 (template values per folder), TS-D5 (Continue-from and memory-need errors) and TM-D3 (the reports line under a filter).

Kept as the record of what the reviews found.

# F12 fix 5: correctness reviews (the pages show the folder and run they say)

## Round 1

1. **The run card's "+ Add setting" stayed silent** when the sweepable list failed, because it kept the last lane asks. *Fixed: the run's own asks are read off the held answer; the failure is said.*
2. **A folder that could not be read claimed its files absent** ("0 of the 5 listed"). *Fixed: its files card is hidden.*
3. **A failed column list kept the previous kind's roles** (a regression). *Fixed: the roles are read off the held answer for the key asked last.*
4. **The preset menu said "did not load" after a retry had loaded it.** *Fixed: views drawn from the lists redraw when a list changes (`keyedList.onChange`).*
5. **`failed` was not tied to the key asked last**, and a hand-over with no engine kept the last list. *Fixed.*
6. **The old bar exported the new path with the old offer's choice.** *Fixed: the selection is set with the bar.*
7. **"Viewer failed" was overwritten once the load landed.** *Fixed: it is said among the line's notices, through the one composer.*
8. **MB-D3 "Target" problems.** It named a `.pdb` that a save never writes. It also named a frame set's file over one frame of it. *Fixed: the readout shows `<stem>` plus the pairing rule's suffix, and one frame of several is no file's.*
9. **MB-D2 leftovers.** The no-WebGL case said "reload". *Fixed: `NO_VIEWER`. A Load that is not started stays silent by design.*
10. **TM-D1:** with every channel skipped, the line no longer said nothing is sent. *Fixed.*
11. **TM-D2:** the reports line had a second writer on failure. *Fixed: `told.error`, one painter.*
12. **Stale comments.** *Fixed.*

## Round 2

1. **An existing test broke.** The save module's hand-made page lacked the snapshot's `loadedFrom`. *Judged: the stand-in now answers the real snapshot's fields.*
2. **The raw-names note stayed after a retry loaded the labels.** *Fixed: it is derived from the lists' own failures.*
3. **The preset redraw could still be missed.** *Fixed: `onChange`, not a before-and-after snapshot.*
4. **"Viewer failed" was still overwritten** on the mount-throw path and the no-frames path. *Fixed.*
5. **Re-reading the same folder blanked the rail.** *Fixed: a re-read keeps it, and its template values.*
6. **An error was less visible than a warning.** *Fixed: `data-attention`.*
7. **One frame of several lost MolView's export name.** *Fixed: `source.frame`; the viewer keeps the file's name, and the page's note names none.*
8. **The export bar's brief window.** Known and accepted (see the verdict).
9. **A Load not started says nothing.** By design (see the verdict).
10. **Comments on the wrong code.** *Fixed.*
11. **Untested items.** Listed in the verdict.
