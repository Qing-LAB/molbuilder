# 2.77.10 review — Plotly's toolbar back, each chart's Export above the drawing

**Verdict: closed with 2.77.10.** Two rounds, each a correctness reader and a
discipline reader over the whole touched files, read-only; the correctness
readers measured in headless Chromium against the served Plotly (plotly.js
4.0.0, python plotly 7.0.0) on files from disk.

## Round 1

| finding | verdict |
|---|---|
| Plotly 4.0.0 shows a *Share chart* button by default (`showSendToCloud`) that sends a chart's numbers to Plotly's cloud | fixed: off in the plot box's rule and the seal's; the e2e helper allows the view's buttons only |
| a plot-box chart's Picture was 28 px taller than on screen (`clientHeight` counts the strip) | fixed: `Plotly.toImage` asked for its own layout's size; the trajectory e2e checks the picture's size |
| `displayModeBar` left to Plotly's default | fixed: stated, on hover, in both doors |
| spectrumchart.md § 12 had no row for the toolbar rule; § 5.3's row said "no button of any kind" | fixed |
| a box keeps its empty strip when a sentence replaces its chart; the strip arrives with the first exported draw | left, cosmetic: results.md § 2.5 says so |
| Plotly's toolbar, on hover, sits over the top right of what a chart draws | put to the user (below) |
| an empty `config` name, duplicate waits, a stand-in `<a>` where Plotly 4 draws a `<button>`, wording | fixed |

## Round 2

| finding | verdict |
|---|---|
| the picture check rounded the drawing's size; a canvas drops the fraction | fixed: under one pixel per axis |
| the picture's size read by byte offset; a name shadowing the test's own reader | fixed: PIL, `on_screen` |
| a dead `.modebar` position check; the DOM stand-in's `closest` parsed selectors itself; the Plotly stand-in's `querySelector` denied its own toolbar | fixed |
| the strip's height had no stated reason; docs phrased against the removal they undo; stale "six doors", an empty section header | fixed |
| the spectrum's toolbar covers its pointer readout on a tall window while it is read | put to the user (below) |

**Box and lasso select** are left out by this release's own choice — no chart
acts on a selection; the spectrum chart left them out before 2.76.0 too.
