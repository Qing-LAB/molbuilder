# F12 fix 7c (the spectra page, 2.77.15, R1) — the two review rounds

**Verdict: closed with 2.77.15.** Round-1 defects L-D3, L-D4, L-D5 and L-U1's
page side (`r1-L-spectra-viewers.md`, `v-L.md`), the first of the five releases
to the finish line *(the user, 2026-11-19)*. Two rounds, each a correctness
reader and a discipline reader over `lib/spectra/core.js`, `web/spectra.md` and
the tests, read-only.

## Round 1

| finding | verdict |
|---|---|
| a temperature change, or the Size slider, wrote the box's size beside the line under it: the line went stale, or came back over a refused mode | fixed: one door -- `_showMode` writes the line after the viewer's verdict; the knobs go through `renderModeViewer` |
| coming back to a file not followed said "Stopped following" and a retry count | fixed: a tick reads whether it follows; the refusal's words from one helper |
| a result with no geometry emptied the host of a kept box | fixed: one release path, `_releaseViewer` |
| the knob table not the one place a knob lives: init wired and primed the same pairs; the alias list typed again | fixed: init wires and primes from the table; aliases from the bucket's keys |
| the handle's `load(path)` door had no caller | removed |
| comments naming the old dispose-on-render, the hot swap, "Refresh"; `_ensureViewer` | fixed |
| the tests answered the page's requests (`route.fulfill`) and wrote its saved state by hand (`testing.md` § 5) | fixed, the user choosing a real run with held requests: the live update on a launched copy of the probed water run; the settings moved by keyboard and reloaded; the refused mode, which no run writes, listed unseen |
| the poll test could pass with a drag that turned nothing | fixed: the turn proved before it is compared |

## Round 2

| finding | verdict |
|---|---|
| the follow test's wait slept outside the browser driver, so a held request could go unseen for 900 s | fixed: `support.road.until` takes the page's own wait between looks |
| the follow test did not show the run's news was drawn | fixed: mode 3's probe awaited in the table |
| the Pause press could land before the box was built | fixed: the turn waits for the line under the box, and for *Play* |
| the coming-back test's waits were on the network only | fixed: the answer read whole; the finished run's limit said |
| the spectra-only test doors in the sidebar's module; PIL undeclared | fixed: `support/spectra_page.py`; PNGs read with the standard library |
| two inline opens left in the file | fixed: `spectrum_opened` |
| the line under the box ran ahead of the picture during an export | fixed: held with the box, drawn again at the export's end |
| a failed box build retried on every slider move | fixed: remembered until a load |
| a temperature answer written into results a poll had replaced; the temperature not in the fingerprint | fixed |
| the bar's knob pairs written beside the table; Update's and the frame set's refusals in raw words; the line's tooltip left standing by other writes | fixed |
| chapter: "Refresh", a doubled phrase, the admissible values, a failed coming back, what is not pinned | fixed |
| the file-menu pick copied in ten other test files | left for R4, its door now `support.sidebar.result_picked` |
| a release while the box is being built keeps the build's stale inputs | left: unreachable -- a file keeps its geometry across polls, and Reload is a new panel |
| the button reads *Pause* over a refused mode | left: the contract says nothing of it |
