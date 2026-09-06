# Checked-in pseudopotentials — for tests that RUN an engine

`H.psml` is a standard ONCVPSP-3.3.0 scalar-relativistic hydrogen
pseudopotential (generated 2017-10-31, PSML 1.1). It is here because
`tests/test_siesta_keyword_smoke.py` starts a real SIESTA, and SIESTA refuses
to start without a per-species pseudopotential it accepts.

**Why not `conftest.write_pseudos`.** That helper writes real, parseable PSML
and is the one home for every test that *preps* — prep's screening
(`science/pseudopotentials.md` § 1) reads it happily. **SIESTA does not
start on it**: swapping this file for `write_pseudos` output on 2026-09-06
failed five of this suite's tests with *"SIESTA printed no k-grid read-back;
it may have died"*. Measured, not assumed — so the two are not
interchangeable and neither replaces the other.

**Why not the projects tree.** It was read from
`projects/BDT/optimization/TJ-BDT-Au111/H.psml` until 2026-09-06 — a specific
real calculation's file, whose relevance nobody confirmed and which would
make the test skip silently on any other machine.
`test_no_tests_read_the_projects_tree.py` forbids exactly that; it missed
this one because it scanned a line at a time and the path was split across
two. Same reasoning as `tests/watch/fixtures/siesta_frozen/`: what cannot be
constructed honestly is checked in, versioned with the tests, and reviewed
when it changes.
