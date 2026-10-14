# Checked-in pseudopotentials — for tests that RUN an engine

`H.psml`, `C.psml` and `O.psml` are PseudoDojo's standard set, scalar
relativistic, PBE (`nc-sr-04_pbe_standard`): ONCVPSP-3.3.0, PSML 1.1,
generated 2017-10-31. They are here because the tests that start a real
SIESTA need one pseudopotential per species SIESTA accepts, and it refuses to
start without one: H for H2 and the hydrogen chain, C and O for the
relaxation-to-vibration road on CO2 (`docs/plans/plan.md`, row T-CO2). The
road (`tests/_road.live_siesta`) puts all three in every live tree's library.

| file | sha256 |
|---|---|
| `H.psml` | `178953a9d73ec5741569a67996385980b53a1e8411660816db757436efb07627` |
| `C.psml` | `1eef0a7c0f0808ac968b43f3c744206f7ccda4af33313999b6c914552a9af7c6` |
| `O.psml` | `5ca5d753a995bd054279007ba76be71d622b60460c6f9a0d9bb37d81bdf185fe` |

**Where they come from.** The set's archive,
`https://www.pseudo-dojo.org/pseudos/nc-sr-04_pbe_standard_psml.tgz`,
downloaded 2026-10-14 with the user's approval; C and O taken from it as they
are. `H.psml`, checked in on 2026-09-06, is byte for byte that archive's own:
the three are one generator release, as `science/pseudopotentials.md` § 1
asks of a set.

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
