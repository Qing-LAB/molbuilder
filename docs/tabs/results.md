# `/results` tab — registry dispatch + file picker

The `/results` tab is molbuilder's unified post-run inspector.
The user picks a result file from the project sidebar (or from
the tab-level dropdown), the inspector registry dispatches to the
right inspector module, and the inspector renders the file.

**Status (2026-06-02): this doc is a stub.** The substantive
contracts are already canonicalised across three sibling docs;
this file currently exists only so the doc index has a stable
landing for `/results`-tab-specific content as it grows.

## Where the live contracts already live

- **Dispatch architecture** — how files route to inspectors,
  the URL → inspector mapping, the page-level mount lifecycle:
  [`../protocols/results-tab.md`](../protocols/results-tab.md)
- **Inspector contract** — `mount(host, file, ctx) → {dispose}`,
  `isResult`, `resultCategory`, registration-order rules, the
  `pageshow` refresh contract, trajectory inspector internals:
  [`../protocols/inspector-registry.md`](../protocols/inspector-registry.md)
- **File-picker behaviour** — the tab-level result-file dropdown,
  scan/parse status messages, auto-pick policy: this content
  lives in `lib/results/file-picker.js` source + the e2e tests
  at `tests/test_results_file_picker_e2e.py` until it grows
  enough to merit a § here.
- **Sidebar integration** — `projects.onChange` subscription,
  `setShared` driven file selection:
  [`../protocols/projects-sidebar.md`](../protocols/projects-sidebar.md)

## What goes here when this doc grows

When the `/results` tab acquires UX surface that isn't already
documented in one of the above, that content lands here. Likely
candidates (none yet):

- Multi-pane layouts (e.g. inspector + side-by-side comparison
  panel — task #135 territory).
- Per-tab settings (default inspector for an extension, sticky
  filters, etc.).
- The "no results yet" empty state's UX.

Until then, this stub keeps the doc index honest without
duplicating content that belongs in the protocol docs.
