*Verdict: closed in 2.77.9.* The discipline reviews of F12 fix 5, two rounds, read before its release. Their brief was the user's directive of 2026-11-19: respect the existing frameworks, unified API and data-driven design, no reinvented wheels, a needed mechanism extended at the framework level first. Every finding was fixed in 2.77.9 except these:
- **Page words checked in tests.** Some tests check the page's words (*was not loaded*, *the column list did not load*). These are the outcome a person reads.
- **The `-ui-draft` id spelled again.** One test spells the workspace's `-ui-draft` storage id beside another test.
- **`_row`'s default.** `_row`'s default `usable=True` still answers `skipped: None` without asking the monitor; no page reads it.
- **The export bar window** (see the correctness verdict).

Kept as the record of what the reviews found.

# F12 fix 5: discipline reviews

## Before round 1: my own re-audit, on the user's directive

Two patches were rebuilt at the framework level before the readers saw them:
- **The frame on screen.** A frame draft and timer beside MolView's view context (§ 11.2b) was removed; the card keeps every frame and MolView restores the frame.
- **The export bar's path.** The viewer had been writing the path into the bar's element; it is now handed over as a getter, like the bar's `asks`.

## Round 1

1. **Template values outside the per-folder object**, with two painters of the rail's names. *Fixed: `_fs.tmpl`; `nameFile`.*
2. **`keyedList` was not the one rule** for the lists keyed by engine and kind: the report fields were a fourth list. *Fixed: `REPORT_FIELDS`.*
3. **What a list's answer hands beside its list escaped the key.** *Fixed: `answer()`, and `_roleRule()` / `_laneAsks()` derived from it.*
4. **One phrase was composed twice**, and one option did two jobs. *Fixed: `keyedList(name, …)`.*
5. **Server words were drawn as raw text.** *Fixed: `codeWords` / `wordsText` / `wordsEl`.*
6. **The reports line had two writers.** *Fixed.*
7. **The structure card no longer named the failed file**, and the loaders worded the event two ways. *Fixed: `projects.parser.notLoaded`, used by every loader of a picked structure file.*
8. **`save.js` rebuilt the page's rule.** *Fixed: one read of the page's snapshot (`_snapshot`); `_stem`.*
9. **Comments made untrue.** *Fixed.*
10. **A phrase copied** ("the Molbuilder page did not load"). *Fixed: `NO_VIEWER`.*
11. **The export bar had two sources for where it exports.** *Fixed: `at()` alone when handed.*
12. **Small items on This machine.** *Fixed.*
13. **A fixed sleep in a test.** *Fixed: the test waits for MolView's own view-context write.*

## Round 2

1. **Existing tests not revised: 15 failed.** *Judged, assertions untouched.*
   - The notify harness now lifts the keyed-list block and the engine reader by their anchors.
   - The save harness's stand-in page answers the real snapshot's fields, and its projects door answers the server's own `PAIRED_SUFFIX`.
2. **`".xyz"` was typed by hand.** *Fixed: `projects.pairedSuffix()`.*
3. **`_vocabFailed` was a second record**, never cleared. *Fixed: derived from `COLUMNS.failed || SWEEP.failed`.*
4. **Half the key moved up into the loaders.** *Fixed: `_openKey(task)`, the one key every list is asked by, the report fields' included.*
5. **The preset redraw patched a symptom.** *Fixed: the lists announce a change and the views drawn from them redraw (`_listsChanged`, subscribed in `boot`).*
6. **`notLoaded`'s chapter line was untrue.** *Fixed: the Results structure view asks it too, and the line names the loaders.*
7. **The parser's JSDoc was displaced.** *Fixed: the header names both doors.*
8. **"The 3-D viewer did not start" was typed three ways.** *Fixed: `NO_VIEWER`; every trajectory status goes through its composer.*
9. **`wordsText` was used on a `<div>`.** *Fixed: `wordsEl`.*
10. **Export bar timing.** See the verdict.
11. **Tests pin text and build input.**
    - The page's words are the outcome, so those checks are kept.
    - The suggested support door builds the transport kind's frame set, so the codec is used instead.
12. **Untested items.** See the correctness verdict.
13. **Stale text.** *Fixed: the need record's doc; `REPORT_FIELDS.name`; one snapshot read.*
