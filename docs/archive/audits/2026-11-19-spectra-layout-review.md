# 2.77.12 review — the mode animation in a right column on the spectra panel

**Verdict: closed with 2.77.12.** The user's request (2026-11-19) and choice of
three layouts; two rounds, each a correctness reader (measuring in headless
Chromium on files from disk) and a discipline reader, read-only.

## Round 1

| finding | verdict |
|---|---|
| the test's wheel turned over the projects sidebar: the page never scrolled | fixed: the pane scrolled to the table, the viewer bounded by the pane; fails with the column not held |
| the electronic structure's 40rem query measured the whole panel: its figure squeezed below its floor beside the column | fixed: it asks `.spectra-mode-views` |
| Thermochemistry described as a view of the selected mode; "a fourth tab" | fixed: the whole run's; "a third, run-level tab" |
| the page code kept which tab is shown beside the strip that owns it | fixed: `MODE_TABS`, `state.modeTab`, `_modeTabShown` gone; the strip's `active()`/`select()` |
| an empty column before a mode can be shown | fixed: one column while the viewer's box is hidden |
| keyboard order unlike the stacked order | fixed: the column sits between the spectrum and the views in the markup |
| the 52rem threshold's arithmetic; the sheet's header naming the /spectra page | fixed |
| stale comments in the template, the sheet and core.js | fixed |

## Round 2

| finding | verdict |
|---|---|
| a height cap read the window while the column is held in the shorter pane: with the help open it hid the 3-D box | fixed: no cap -- the top held, the end reached with the page's end |
| the same cap's mismatch on the app's other held rail (Task setup, Transport) | left as it was shipped; recorded in the plan |
| the test's click checked nothing it caused | fixed: the picked row and the animation both in the pane |
| comments: the container's user, the column's top, the shell's tab note, a listener count | fixed |
| `.viewer-controls` written twice with two slider tokens | left: older than this release |
