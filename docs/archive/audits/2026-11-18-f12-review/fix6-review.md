# F12 fix 6 (MolView, 2.77.11) — the two review rounds

**Verdict: closed with 2.77.11.** Round-1 defects K-D1..D11 (`r1-K-molview.md`,
`v-K.md`) fixed at their doors. Two rounds, each a correctness reader and a
discipline reader over the whole touched modules and `web/molview.md`,
read-only.

## Round 1

| finding | verdict |
|---|---|
| the in-flight gate dropped a Save state pressed during an edit | fixed: the gate has two reasons; saving keeps the read-only one only |
| MolView's Metadata page read the gate's *nothing to do* as read-only, or lost the typing | fixed: the editors keep what was typed and say nothing |
| a load during an edit was overwritten by the edit's answer | fixed: the answer is not adopted; the edit's door answers null |
| a Retract racing an edit | fixed: a Retract holds the gate through its read |
| the in-flight rule written as lists in four places | fixed: once in § 9.4; § 6.9, § 11.1, § 13.3 point there |
| an append held through a rebuild could never add a frame | fixed: nothing is held; the rebuild reads the frames when it runs; the start-frame plumbing as shipped |
| the triad rule decided by redraw names inside `settle` | fixed: each REDRAW row declares `putsIn` |
| a dead branch and history-narrating comments; a second `reflect`; the shift rule and the colour rule each written twice | fixed |
| the Cell dot keyed on words where § 6.8 said "the next answer" | fixed: § 6.8 says the words' rule |
| the in-flight test checked answers, not outcome, and missed Retract | fixed: rows for an edit, a cell commit and a Retract out, the outcome read |

## Round 2

| finding | verdict |
|---|---|
| a Retract (or a reopened draft) whose read lands after a load restored the old structure | fixed at `history.js`: a read that lands after a structure was opened puts nothing back |
| the Molbuilder tab said "failed" / "read-only" for MolView's *nothing to do* | fixed: the status goes quiet |
| § 9.4 and § 11.1 overstated which doors wait | fixed: the install, `reloadFrames` and `save` named as landing |
| Add offered on an empty editable viewer | fixed: hidden while nothing is loaded (§ 9.4) |
| a test emptied by the append change; a test fed a pick no road makes | fixed: held forces; a pick isolate does not draw |
| a comment and a docstring citing the isolate rule the chapter retired; the colour reason restated where it no longer held | fixed |
| `reloadFrames` meets no in-flight reason | left: no editable viewer calls it; § 9.4 names it |
| Retract's message at the host when a cell commit is out | left: one round trip wide |
| the whole store applied as JSON | untested: its editor loads only in a browser; the bracket is pinned |
