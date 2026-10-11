# Changelog — what each version changed in the files molbuilder writes

**Role:** record
**Domain:** process

One line per version, newest first: what changed in which file molbuilder
writes, and the date. The rule for when the version moves is
[`conventions.md`](?doc=process/conventions.md) § 2, *Versioning*; every deck
and run script records the version that wrote it
([`execution/job-contracts.md`](?doc=execution/job-contracts.md) § 3.2).

| version | date | what changed |
|---|---|---|
| **2.1.0** | 2026-10-11 | **A template records the version that wrote it** (`written_by`, `engines/template.md` § 3). **Blank and 0 are two answers on the two device broadenings** (`negf_neq_eta_ev` → `TS.Contours.nEq.Eta`, `tbt_contours_eta_ev` → `TBT.Contours.Eta`): blank is not set — no line, and the program takes a tenth of the leads' broadening, said in the not-set comment — and 0 is written as 0, no broadening. Before, a 0 there meant not set; a calculation written before 2.1.0 that holds one — in its template or as a rung's override in `task.json` — is refused, and `molbuilder jobset migrate` rewrites it blank (`engines/transport.md` § 2a.13). **New lines in the transport decks**: the device's `TS.Contours.nEq.Fermi.Cutoff` (5 kT, refused below 2.5), `TS.SolutionMethod` (`BTD`, or `full`) and `TS.Hartree.Fix C`; the transmission's `TBT.SolutionMethod BTD`; comments on `bloch 1 1 1`, the reservoirs' shared temperature, and tbtrans's own input log. `electrodes_bulk` is carried by the device and transmission only (`stages`) |
| 2.0.0 | 2026-10-11 | **The first release since 1.1.0** (tag `v2.0.0`). Every deck and run script records the version that wrote it (`generator-version 2.0.0 (git …)`); a file written before records only its git commit and reads as *before 2.0.0*. Each deck's provenance block lists who set each value it is rendered from (`parameters:`), and a setting left to SIESTA or PySCF says so where its line would be (`# <keyword> not written: …`) — `engines/template.md` § 6.6, `job-contracts.md` § 3.2. Everything between 1.1.0 and 2.0.0 is in `git log v1.1.0..v2.0.0` and the plan's archive (`docs/archive/`) |
| 1.1.0 | 2026-05-11 | tag `v1.1.0` |
| 1.0.0 | — | tag `v1.0.0` |
