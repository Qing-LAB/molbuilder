# F12 fix 7a (VibrationView, 2.77.13) — the two review rounds

**Verdict: closed with 2.77.13.** Round-1 defects L-D2 and L-U1's code side
(`r1-L-spectra-viewers.md`, `v-L.md`) fixed at the handle and the exporter.
Two rounds, each a correctness reader (measuring in node from scratch) and a
discipline reader over the whole package and `web/vibrationview.md`,
read-only.

## Round 1

| finding | verdict |
|---|---|
| a mode asked during an export was put, when it ended, on a new structure installed meanwhile | fixed: a new structure waits for the export too, in the order asked, and ends the mode asked before it |
| a mode asked during an export answered *shown* before it was checked | fixed: judged at the call against the structure it will meet; only the showing waits |
| a dispose during the GIF encoder's first load said "nothing drawn"; a cancel on the last frame still handed back the file | fixed: an export whose signal is aborted when it ends rejects in the cancel's words, however it ended |
| a signal that is not an AbortSignal left the viewer locked | fixed: the forwarding is set up before the export is marked running |
| the hold test read only the stamp and the captions: an amplitude not held still passed | fixed: every frame's atoms and caption against an export left alone, a row per door |
| the refusal test read no caption, and kept a dead line | fixed |
| the ending of a mode split between `endMode` and its callers; the cancel said three times; `held` beside "held still" | fixed: `endMode` takes the marks and the caption; one statement at the export's own stop; renamed |
| the chapter: the refusal "draws nothing"; "the rate" without the cycle length; § 9.2 silent about the hold | fixed |

## Round 2

| finding | verdict |
|---|---|
| the reason for cancelling on a new structure ("nothing to draw") was false: the export could have finished | fixed: a new structure ends the mode the export is of (§ 5.1) |
| a pause during an export, and a mode's answer against the structure ahead, guarded by nothing | fixed: a pause row; a test of the answer |
| the held-still marks going on a refusal guarded by nothing | fixed: the earlier mode holds an atom still, and no mark follows the refusal |
| what a caller handed over read when the export ended, not at the call | fixed: copied and read at the call |
| a refusal's warning printed when the export ended; a signal that is not one threw instead of rejecting | fixed |
| § 12's cancel rule said twice; the stand-in's comment false; the cancel tests not a table | fixed |
| older text in the touched files: an unread default, a mount amplitude letting NaN through, the "no hatch" note twice, § 12.2 cited for § 12's facts, test docstrings quoting words the chapter no longer has, a failed-export test failing before its capture, the seal's notes numbered 1 of 2 / 3 of 3 | fixed |
| the spectra page ignores a refused mode under a swing line and a Pause button; a live run's poll rebuilds the animation, cancelling an export | the page's: recorded under fix 7 in the plan |
