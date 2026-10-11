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
| **2.4.0** | 2026-10-11 | **A spin-polarised transport record states each spin and their total** (`web/results.md` § 2.5): each point's `transmission_total` (T↑ + T↓) beside its `channels`, `current_by_spin_a`, and a DOS read from both `<label>.TBT_UP.nc` and `.TBT_DN.nc` — every quantity summed over spin, each spin's own total and eigenchannels in `dos.by_spin`; the I–V block's `current_up_a` and `current_down_a`. `<label>.transport.nc` holds a polarised point's DOS, summed, where it held the fill value. A non-polarised record is written as before |
| 2.3.0 | 2026-10-11 | **The device's deck states TranSIESTA's charge check**: `TS.SCF.dQ.Converge` (on, TranSIESTA's default) and `TS.SCF.dQ.Tolerance` (blank is not set — TranSIESTA's thousandth of the electrons on the atoms that are not buffer atoms, said in its not-set comment), two device settings (`engines/transport.md` § 2a.13). Asked for by a run that could not finish: its density had stopped changing 2.3 electrons off, and SIESTA writes the Hamiltonian the transmission reads only on convergence. **Every JSON file is valid JSON**: a number an engine printed unreadably (a Fortran overflow, read as NaN) is written `null` — not stated — where it was written `NaN` (`persist.json_safe`; `job-contracts.md` § 6.1) |
| 2.2.0 | 2026-10-11 | **A deck's `.validation.txt` states the H/heavy ratio as a reference note on every structure** — `info [geometry.h_ratio]: H/heavy atoms: 4/14 = 0.29 (for reference: organic molecules usually sit at 0.6-1.5)`, counted by element — where it carried a warning below 0.3 that asked whether hydrogens were missing (`science/overview.md` § 4; `f86627b6`). The decks are written as in 2.1.0. *(Version moved in its own commit, after the change: the rule was missed at `f86627b6`.)* |
| 2.1.0 | 2026-10-11 | **A template records the version that wrote it** (`written_by`, `engines/template.md` § 3). **Blank and 0 are two answers on the two device broadenings** (`negf_neq_eta_ev` → `TS.Contours.nEq.Eta`, `tbt_contours_eta_ev` → `TBT.Contours.Eta`): blank is not set — no line, and the program takes a tenth of the leads' broadening, said in the not-set comment — and 0 is written as 0, no broadening. Before, a 0 there meant not set; a calculation written before 2.1.0 that holds one — in its template or as a rung's override in `task.json` — is refused, and `molbuilder jobset migrate` rewrites it blank (`engines/transport.md` § 2a.13). **New lines in the transport decks**: the device's `TS.Contours.nEq.Fermi.Cutoff` (5 kT, refused below 2.5), `TS.SolutionMethod` (`BTD`, or `full`) and `TS.Hartree.Fix C`; the transmission's `TBT.SolutionMethod BTD`; comments on `bloch 1 1 1`, the reservoirs' shared temperature, and tbtrans's own input log. `electrodes_bulk` is carried by the device and transmission only (`stages`) |
| 2.0.0 | 2026-10-11 | **The first release since 1.1.0** (tag `v2.0.0`). Every deck and run script records the version that wrote it (`generator-version 2.0.0 (git …)`); a file written before records only its git commit and reads as *before 2.0.0*. Each deck's provenance block lists who set each value it is rendered from (`parameters:`), and a setting left to SIESTA or PySCF says so where its line would be (`# <keyword> not written: …`) — `engines/template.md` § 6.6, `job-contracts.md` § 3.2. Everything between 1.1.0 and 2.0.0 is in `git log v1.1.0..v2.0.0` and the plan's archive (`docs/archive/`) |
| 1.1.0 | 2026-05-11 | tag `v1.1.0` |
| 1.0.0 | — | tag `v1.0.0` |
