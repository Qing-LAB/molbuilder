# Checked-in pseudopotentials

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
| `pseudodojo-v0.4/S.psml` | `f59163a4500c143514f1480eca12c3ca6122331dd7ca5adeb1e47a170429a9a4` |
| `pseudodojo-v0.4/Au.psml` | `b1d2a2e3626c6656de57470c205cb2428fe66275575357efe1287d6d04fe16a5` |
| `pseudodojo-v0.4/F.psml` | `3e74844f561066361a50a773268b340048163ef869a080ed334f5ab80eb6a8d2` |
| `pseudodojo-v0.4/Fe.psml` | `55c55eac2e5f7b872c210775f22d4fa672c51c39289b9dd175375984eaf95724` |
| `pseudodojo-v0.4/N.psml` | `691c380a778fb6234028c994766364d998a14371860b7e9c855be773158ef43f` |
| `pseudodojo-v0.4/P.psml` | `057149a6c6d72362347fabcfa84fc2d2afd91fd6debe6fbd59d55ca7faae443d` |
| `pseudodojo-v0.5/S.psml` | `18956da60521e261696ec75ecdc33e45f45e976f113b0797dea6e72ecf764832` |
| `pseudodojo-v0.5/Bi.psml` | `e481356f8288038a6c6850809d4af454e6cf2610e77a56aef63a71af7fb8fec6` |

**Where they come from.** The set's archive,
`https://www.pseudo-dojo.org/pseudos/nc-sr-04_pbe_standard_psml.tgz`,
downloaded 2026-10-14 with the user's approval; C and O taken from it as they
are. `H.psml`, checked in on 2026-09-06, is byte for byte that archive's own:
the three are one generator release, as `science/pseudopotentials.md` § 1
asks of a set.

**The sulfur and the bismuth are what the screening is about**
(`science/pseudopotentials.md` § 2a), so what it finds is read from the files
that made the rule: `pseudodojo-v0.4/S.psml`, every channel carrying a real projector;
`pseudodojo-v0.5/S.psml`, the file that broke a real run, its p projectors
zero; `pseudodojo-v0.5/Bi.psml`, its second p projector zero. Copied 2026-11-06
from the person's library and from `nc-sr-05_pbe_standard_psml.tgz` beside it,
bytes unchanged. Prep's and an engine's tests do not read them.

**`conftest.write_pseudos` puts these in a test's calculation.** The other
elements the suite's calculations hold are in `pseudodojo-v0.4/` beside the
sulfur, copied 2026-11-06 from the person's library, the same release, bytes
unchanged: gold, fluorine, iron, nitrogen and phosphorus. They sit in a folder so
an engine's tests, which take the three files above, do not take them too.

**Why not the projects tree.** It was read from
`projects/BDT/optimization/TJ-BDT-Au111/H.psml` until 2026-09-06 — a specific
real calculation's file, whose relevance nobody confirmed and which would
make the test skip silently on any other machine. The suite now fails any
test that reaches the real projects tree, by any spelling
(`process/testing.md` § 2a, the guard in `conftest.py`). Same reasoning as `tests/watch/fixtures/siesta_frozen/`: what cannot be
constructed honestly is checked in, versioned with the tests, and reviewed
when it changes.
