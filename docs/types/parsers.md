# Spec — parser plug-in interface (RETIRED)

This spec described the legacy `TrajectoryParser` ABC and the
trio of `molbuilder/parsers/{siesta,pyscf,molwatch_log}.py`
trajectory parsers.  All three modules — plus the
`molbuilder/parsers/` package as a whole — were deleted in
parse-module Phase H4b (2026-06-21).

The unified parser stack now lives under `molbuilder/parse/`.
See:

* [`docs/protocols/parse-module.md`](../protocols/parse-module.md)
  — the architectural contract for the new `FileParser` /
  `TextParser` / `DirParser` ABCs + the typed `ParseResult`
  hierarchy that replaced the legacy `Trajectory` dataclass.
* `molbuilder/parse/engines/{siesta,pyscf,molwatch}.py` —
  trajectory engines (the absorbed bodies of the legacy
  `SiestaParser` / `PySCFParser` / `MolwatchLogParser`).
* `molbuilder/parse/registry.py` — `detect(path)` + `parse(path)`
  (the new public dispatch entry points; the legacy
  `detect_parser` is gone).
* `molbuilder/parse/engines/_helpers.py:trajectory_to_legacy_dict`
  / `trajectory_result_to_legacy_dict` — JSON adapter used by
  the watch web layer to keep the 3Dmol.js wire shape stable.

For the broader migration plan see parse-module.md § 8 (Phases
A–H4 shipped 2026-06-19 → 2026-06-21).
