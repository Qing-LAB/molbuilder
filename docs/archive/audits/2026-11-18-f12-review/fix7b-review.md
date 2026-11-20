# F12 fix 7b (SpectrumChart, 2.77.14) — the two review rounds

**Verdict: closed with 2.77.14.** Round-1 defect L-D1 (`r1-L-spectra-viewers.md`,
`v-L.md`; its cause R2-13 F10) and the user's request (2026-11-19: the pointer
readout clear of Plotly's toolbar, which covered it on a tall window). Two rounds,
each a correctness reader (measuring in headless Chromium with plotly.js 4) and a
discipline reader over the whole package and `web/spectrumchart.md`, read-only.

## Round 1

| finding | verdict |
|---|---|
| the first fix let a mode with no strength outrank the selection: its × lit up nothing, chosen or pointed at, against § 1 and § 6.3.1 | fixed: `pending` per mode in the picture beside `state`; the seal colours an × by its state and `recolour` restyles the bars and the ×s of the panel showing the choice |
| the choice showed only in the first panel, empty on a SIESTA result before a laser | fixed: shown once, in the first panel with something drawn |
| the toolbar's height followed the page's font size | fixed: the module states its own type on the frame |
| the note and the readout shared one line and wrote over each other on a narrow chart | fixed: a row each above the plot |
| "pending" decided twice (the state, the note) | fixed: one predicate, `pendingIn` |
| dead code: the curve colour and its token, `sign`, an axis loop, a fallback key, `known` and `rug.index` in the picture | fixed |
| § 6.3.1 said three rules and listed two; § 8.4's picture unlike what is sent | fixed |
| the tests: the pointer over an ×, the restyled trace, the e2e measuring the frame not the surface, nothing under the plot's top | fixed |

## Round 2

| finding | verdict |
|---|---|
| a lit × was half clipped at the axis and covered by the rug's tick | fixed: the ×s drawn last, unclipped |
| the taller strip took height from the plot: at the 220-px floor the axis titles collided sooner | fixed: the floor raised to 252 px, the panels' height before the strip grew |
| the everyday chart, with no note, unmeasured | fixed: rows with no mode waiting; they fail with the readout's row dropped |
| the note owned by no section; "in the plot" meaning two things; comments saying a direction places a panel; a reason for trace order no longer true; the plain colour said twice; the strip's numbers unexplained, the words' size written three times | fixed |
| the test stand-ins missing the lane and rug colours, still declaring the removed one; the seal test's picture not § 8.4's shape | fixed: every token the sheet declares; the picture's shape, and the seal's fallbacks for it gone |
| § 8.4's picture row has no test | named review's: the picture passes between two files of the module |
| an exported image keeps the strip above the plot, 46 or 62 px | left: the image is the drawing as it stands |
| an imaginary mode with no strength in a lane that has strengths is a zero-height bar there; the imaginary colour against the infrared lane's at 1.12:1 | left: older than this release |
