# F12 fix 11 (the job bundles, 2.77.19, R3b2) — the review

**Verdict: closed with 2.77.19.** R2-3 F1 (`r2-3-runscript-monitor.md`): the vibration's and
the PySCF script's bundles were hand-kept tables beside the monitor's derived one, and missed
modules their members import. Now one closure (`runwrap.bundle_members`) gathers all three
from their entries, each member stating its own `TRAVELS_WITH` (derived from its bare imports,
checked against them for all 62 modules) and, where its last name is another's or the PySCF script imports it by another, `TRAVELS_AS`.
F1 was moved out of R3b into its own release *(the user, 2026-11-20)*. Round 1: three readers
(correctness, one-home and discipline, notes and tests); round 2: two new readers; then two
new, independent readers of every touched file; then a reader of the fixes after them.

**What the review taught.** Judged against the chapters (the user: *"check for correctness not
for exact match"*), round 1 found two consequences the derivation alone could not show: a
declared need (`deck_record` → `atom_permutation`) carried numpy into the monitor's bundle,
whose rule is the standard library only — fixed at the owner, by moving the order record's
reader into its own module; and renaming two travel names would have changed the PySCF script's
text and refused a launched stage's prep as another calculation — fixed by keeping the names,
stated once each. Round 2 found the new import test blind to an import by a module's old name;
the test now counts both names, and the load tests import every member.

## Round 1 — three readers (correctness, one-home + discipline, notes and tests), judged against the chapters

| # | finding (readers) | verdict |
|---|---|---|
| 1 | `deck_record`'s declared need carried `atom_permutation` (numpy at load) into `mb_monitor.pyz`, whose rule is the standard library only (all three) | fixed at the owner — the order record's one reader, `order_record_of`, moved into `atom_permutation` (`docs/architecture.md`: "the order record — its class and its one reader (`atom_permutation`)"); `deck_record` imports nothing that needs numpy; the monitor's bundle is 2.77.18's 36 again; its load test now imports every member without numpy |
| 2 | renaming the emitter and the mode selection changes the PySCF script's import lines, so a stage launched before the upgrade is refused at prep as another calculation (one-home) | fixed — both keep their names, stated once each (`TRAVELS_AS = "molwatch_emitter"`, `"mode_selection"`); the vibration finish imports `mode_selection`; the PySCF decks render identical to 2.77.18's (2142 lines) |
| 3 | `monitor_companions`' cache dropped with the old body (one-home) | fixed — restored |
| 4 | a broken `TRAVELS_WITH` raised a bare ImportError (correctness) | fixed — said as `WrapperError`, naming the module |
| 5 | the name rule beside a job spelled twice (`TRAVELS_AS` vs `runpoints.part`'s last name) (correctness, one-home) | fixed — `bundle_members` states that a module a record names travels under its last name |
| 6 | configuration § 2.3's "its own table", run-reports § 2.3's emitter in the vibration finish, `keys.py`'s note, pyscf.md § 3 and the bundle row's "every function the script runs" (notes-tests, one-home) | fixed |
| 7 | the profile names the gone tables; `deck_record`'s note `runwrap._members`; `TRAVELS_AS` note for the sidecar says no why; runwrap's PySCF note runs into `_PYSCF_IMPORTED`'s (notes-tests, one-home) | fixed |
| 8 | the new test not listed where I23 names what pins it (notes-tests) | fixed — I23's column |
| 9 | the refusal of two modules under one name untested (notes-tests) | fixed — a test of `bundle_members` on two `spectra` modules |
| 10 | the new test parses the bundle's source (notes-tests, one-home) | left as is — it reads the built product a job gets; no outcome test reaches an import inside a function without calling every path |
| 11 | the new test's set of names is the three bundles' (notes-tests) | left as is — widening it finds nothing but the PySCF library's own `pyscf` |
| 12 | `emit_bundle_imports` checks against the whole closure, not the entry list (correctness) | left as is — what it accepts is shipped; a module leaving the closure is refused at render |
| 13 | the collision refusal no longer names the bundle (correctness) | left as is — a code defect's message, never a person's input |

## Round 2 — two new readers, judged against the chapters

| # | finding (reader) | verdict |
|---|---|---|
| 1 | the new import test could not see a module imported under its last name when it travels under another (`TRAVELS_AS`); the vibration load test imported four members only (one-home-notes-2) | fixed — the test counts a travelling module by either name (it now flags the finish importing `selection`); the vibration load test imports every member alone, as the monitor's and PySCF's do; I23's column says exactly that |
| 2 | vibration.md lists `order_record_of` under `deck_record` (both) | fixed — § 5.5's table and the § 8.1 module table |
| 3 | `deck_record`'s header: an import it no longer makes; one bundle named of two (one-home-notes-2) | fixed |
| 4 | configuration § 2.3: "joins only by a member stating it" — the entries and the monitor's record-named parts join otherwise (both) | fixed — entry, a part a record names, or a member's statement |
| 5 | the profile says only the monitor's bundle is built from `TRAVELS_WITH`; `pyscf_companions`' "imports them by these names"; a `#:` block that notes nothing (one-home-notes-2) | fixed |

## The two independent full-text readers

No defect in what the bundles hold; both read every member's import against its declaration and
imported every member alone on the real job environments. Their notes, all fixed:

| # | finding | verdict |
|---|---|---|
| 1 | `sidecars/spectra.py`, `molstruct.py`, `pyscf_vibration.py`, `emitter.py` say they load the standard library only (or only `constants`); they load numpy and their members | fixed — each says what it loads |
| 2 | I23's column named one load test as two | fixed |
| 3 | the § 3.2 row and § 3.3 left out the monitor's record-named parts; § 3.3's "refuses … outside it" read as the entry list | fixed |
| 4 | `runpoints.py`, `calcdirs.py`, `atom_permutation.py`, `fdf.py`'s notes name too few bundles or the old door | fixed |
| 5 | `runs.py` called `order_record_of` "the one reader" (it is `Permutation.from_dict`) | fixed |
| 6 | `monitor_companions`' cache restored as `maxsize=None`, 2.77.18 had `maxsize=1` | fixed — `maxsize=1` |
| 7 | § 3.3's new sentence attributed to the user (they chose the release, not the rule) | fixed — cites the finding |

## Topic → test

| change | test | on 2.77.18 |
|---|---|---|
| every bundle holds what its members import | test_monitor_notify_policy::test_every_module_a_member_imports_beside_the_job_travels_with_it (built bundles, both names) | cannot run as written (no closure there); adapted to its tables, fails at its assertion: vibration `quantities`, PySCF `runpoints`, monitor `atom_permutation` missing |
| two modules under one name refused | ::test_two_modules_under_one_name_beside_the_job_are_refused | cannot run there (a guard) |
| the monitor's bundle stays stdlib-only | ::test_the_shipped_monitor_loads_on_a_python_with_nothing_installed (every member, numpy blocked) | passes there; fails with round 1's `deck_record` need put back (seen) |
| the vibration finish finds every member | ::test_the_shipped_vibration_finish_loads_its_readers_from_the_bundle_alone (every member) | passes there |
| decks unchanged | compare_head: PySCF and SIESTA decks identical to 2.77.18's | — |
| real jobs | test_vibration_e2e, test_siesta_vibration_e2e, test_monitor_watches_a_live_run_e2e, test_pyscf_relaxation_outcome_e2e | pass (63, real runs) |
