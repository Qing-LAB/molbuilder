# Browser-owned transient data — the structure app (working-copy)

> **SUPERSEDED / REDUCED (2026-07-02).** This doc originally specified a
> "changed-underneath" hash-gate + a laundering analysis for the structure
> editor. That premise was wrong — a save writes the whole self-consistent
> `.xyz`+`.json` pair, so there is nothing to launder and no gate exists. The old
> body has been removed. **Nothing here is authoritative; follow the links.**

The structure editor's load / edit / save is simply the **structure application
of the working-copy core** — no separate contract of its own:

- **The core** (load → edit → save, draft, crash-recovery) —
  [`working-copy-persistence.md`](working-copy-persistence.md), authoritative.
- **The codec** (`.xyz` + `.molstruct.json`) — `molbuilder/workingcopy_structure.py`.
- **The browser wiring** (hold labels in the working copy; Save / Save As →
  `/api/workingcopy/save`; this *replaces* the `writeLabel` auto-save) — designed
  in [`molview-module.md`](molview-module.md) (the MolView module) +
  `working-copy-persistence.md`. Not yet built.

There is **no** hash-gate, **no** "commit vs save" distinction, and **no**
source-hash carried by the browser. Save writes the pair — overwrite, or save-as.
