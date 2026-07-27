# Pseudopotential validation — standard & process

**Status:** source of truth for what molbuilder checks in a `.psml`
pseudopotential set, why, and how. The implementation
(`molbuilder/pseudos.py` + `molbuilder/validation/siesta.py` +
`molbuilder pseudo check`) MUST conform to this document; the tests
(`tests/test_pseudos.py`) pin each check to it. If a check here and the
code disagree, that is a bug in one of them — fix it, do not let them
drift.

This document exists because a **defective `S.psml` shipped silently**
into a real run (2026-06-26): an `ONCVPSP-4.0.1` sulfur pseudo with a
**dead p-channel** (`ekb=0`) sat among otherwise-`ONCVPSP-3.3.0`
H/C/Au pseudos. It produced wrong sulfur bonding AND crashed SIESTA at
`propor: ERROR: IMAX=0` — but only at high MPI rank counts, so a
low-np GPU run would have *silently* used it and reported plausible,
wrong numbers. None of the pre-existing checks caught it. The point of
the checks below is to catch that class of defect **at preparation
time**, before any compute is spent and before any wrong number is
trusted.

Reviewers: the value of this guard depends on the checks being both
**useful** (they catch real defects) and **correct** (they don't flag
good pseudos). Each check below states its scientific basis, its
false-positive risk, and its severity so you can judge both. Please
challenge anything that looks heuristic — several thresholds are.

---

## 1. Where it runs & the severity model

Two entry points, one engine:

| Surface | When | Module |
|---|---|---|
| SIESTA preflight | every `render_fdf` / Build / `molbuilder run` | `validation/siesta.py::_check_siesta_pseudo_coverage` → `pseudos.check_coverage` |
| `molbuilder pseudo check <dir>` | on demand (CI gate, manual audit) | `cli.py::cmd_pseudo_check` → `pseudos.check_coverage` |

Both call the **same** `pseudos.check_coverage(elements, dir, …)`, which
returns a list of `CoverageEntry(element, status, message, path)`. The
caller maps `status` → severity:

- **ERROR** (blocks the run / non-zero exit): `missing`, `dead_projector`.
- **WARN** (surfaced, does not block): `xc_mismatch`,
  `relativistic_mismatch`, `generator_mismatch`, `parse_warning`.
- **ok**: silent pass.

Rationale for the ERROR/WARN split: a check is ERROR only when the run
**cannot be correct** — the file is absent (SIESTA won't start) or a
valence channel is physically missing (wrong Hamiltonian). Everything
else is a strong smell that *might* be intentional (you really did mean
to mix functionals, or pair an SR pseudo with an FR calc), so we surface
it loudly but let the user proceed.

---

## 2. The checks

Each check parses the PSML header once (`parse_psml_header` →
`PsmlInfo`) and inspects one field.

### C1 — Coverage  ·  ERROR  ·  `status="missing"`
**What:** every element in the structure has exactly one `.psml` in the
pseudo directory.
**Why:** SIESTA refuses to start without a pseudo for every species.
**How:** `scan_psml_directory` maps `{element: PsmlInfo}`; a structure
element with no entry → `missing`.
**False positives:** none (a missing file is unambiguous).

### C2 — XC functional consistency  ·  WARN  ·  `status="xc_mismatch"`
**What:** the pseudo's exchange-correlation functional matches the
calculation's (`cfg.xc_authors`).
**Why:** a pseudo is generated *for* a specific XC functional. Using a
PBE pseudo in an LDA run (or vice versa) gives **silently wrong** bond
lengths and energies — the run completes, the numbers are off by an
amount that looks physical. Family mismatch (GGA vs LDA) is serious;
same-family author mismatch (PBE vs PBEsol) is minor (~1–2 kcal/mol).
**How:** `PsmlInfo.xc_family` / `xc_authors` decoded from the
`<libxc-info>` functional ids (`_LIBXC_MAP`), compared to the expected
family derived from `cfg.xc_authors`. `"unknown"` (unrecognized libxc
id) does **not** warn — we don't flag what we can't classify.
**False positives:** low; bounded by the libxc id table. An unmapped
functional is treated as `unknown` (no warn) rather than mis-classified.
**Severity note:** this is WARN today; arguably XC-family mismatch
should be ERROR (it is never correct). Flagged for reviewer decision.

### C3 — Relativistic treatment  ·  WARN  ·  `status="relativistic_mismatch"`
**What:** scalar-relativistic (SR) vs fully-relativistic/spin-orbit
(FR) matches the calculation's intent.
**Why:** FR pseudos are needed only when spin-orbit coupling matters
(heavy elements + SOC-sensitive properties); using FR when you meant SR
(or vice versa) changes the physics. SR is correct for the large
majority of work.
**How:** `PsmlInfo.relativistic` from the header `relativity` attribute,
compared to `expected_relativistic` (default `"scalar"`).
**False positives:** low; `unknown` does not warn.

### C4 — Generator/version consistency  ·  WARN  ·  `status="generator_mismatch"`  *(version control)*
**What:** the whole set comes from ONE generator release (e.g. all
PseudoDojo `ONCVPSP-3.3.0`), not a mix.
**Why:** a single stranger version is the *smell* that a pseudo was
hand-swapped from a different source — exactly how the bad S entered.
Mixed releases also mean the set was never validated together for
transferability/consistency.
**How:** `_generator_key(generator)` reduces each `creator` string to
`name-MAJOR` (e.g. `"ONCVPSP-4"`, `"ONCVPSP-3"`); a patch difference
(3.3.0 vs 3.3.1) does **not** warn, a major mix (3.x vs 4.x) does. The
minority version(s) are named as the likely stranger(s). Only pseudos
actually present are compared.
**False positives:** moderate — mixing versions *can* be intentional
(a curated set drawing the best pseudo per element from different
releases). Hence WARN, not ERROR. The message says "confirm the
odd-one-out is intended."

### C5 — Dead Kleinman-Bylander projector  ·  ERROR  ·  `status="dead_projector"`  *(value validation)*
**What:** no valence angular-momentum channel has its **entire** set of
KB projectors at `ekb ≈ 0`.
**Why (scientific basis):** SIESTA represents the nonlocal
pseudopotential in separable Kleinman-Bylander form
[Kleinman & Bylander, *PRL* **48**, 1425 (1982)]:
`V_NL = Σ_l |χ_l⟩ E_KB,l ⟨χ_l|`. The KB energy `E_KB` (the PSML `ekb`
attribute) **is** the projector's strength — `ekb=0` means that
projector contributes nothing to the potential. If *every* projector of
a given `l` is null, that angular momentum has **no nonlocal
projection** at all; its scattering is left to the local potential
only. For an element where that `l` is a valence channel (e.g. sulfur's
3p), this is physically wrong — the bonding through that channel is not
described — and it also leaves SIESTA's projector table degenerate,
which can trigger `propor: ERROR: IMAX=0`.
**How:** `parse_psml_header` reads every `<proj l=… ekb=…>`, groups by
`l`, and flags an `l` where the channel is **present but all** `|ekb| <
1e-6` → `PsmlInfo.null_channels`.
**Why "whole channel", not "any zero":** a pseudo legitimately may have
one weak projector among several for an `l`; we only flag an `l` whose
*entire* projector set is null, which is unambiguously broken. A channel
chosen as the **local potential** has **no** `<proj>` entries at all
(absent, not zero) and is correctly **not** flagged.
**Threshold:** `1e-6` (hartree·-ish KB units) is heuristic — chosen well
below any physical `ekb` (real ones are O(0.1–20)) and just above
exact-zero/round-off. Reviewer: if you know a valid pseudo with a
genuinely tiny non-zero `ekb`, raise it; the test fixtures use exact 0.
**False positives:** believed none for standard ONCVPSP/PseudoDojo
sets; the rule requires an *entire valence l-channel* to be null.

### C6 — Parse integrity  ·  WARN  ·  `status="parse_warning"`
**What:** the file parsed and carried the expected header fields.
**Why:** a malformed/truncated PSML is suspect; surface it but let
SIESTA make the final call at startup.
**How:** non-empty `PsmlInfo.parse_warnings` (bad XML, missing element,
unparseable Z, …).

---

## 3. What is NOT checked (limitations / future)

Be explicit so the guard isn't trusted beyond its reach. molbuilder
does **not** currently verify:

- **Valence-charge correctness** — e.g. whether Au uses an 11- or
  19-electron valence appropriate to the chemistry. (`z-pseudo` is
  parsed but only sanity, not correctness, could be checked.)
- **Ghost states** — spurious bound states below the valence; requires
  generation-level analysis, not header inspection.
- **Transferability / hardness** — whether the pseudo reproduces
  all-electron results across environments. This is the job of the
  *source* (PseudoDojo's δ-factor benchmarks); molbuilder trusts a
  vetted source and only checks consistency + integrity.
- **Mesh-cutoff adequacy** — `suggested_mesh_ry` is parsed (PseudoDojo
  extension) but not yet compared to `cfg.mesh_cutoff`. Advisory only.
- **Basis ↔ pseudo consistency** — that the PAO basis l-channels match
  the pseudo's. Deferred.

The guard's claim is bounded: it catches **missing, mis-functionaled,
mis-relativistic, version-strange, structurally-broken, or unparseable**
pseudos. It does **not** certify that a pseudo is *good* — that comes
from using a vetted source (PseudoDojo) consistently.

---

## 4. Process — how to use it

```bash
# screen a whole directory (CI gate; non-zero exit on any ERROR):
python -m molbuilder pseudo check projects/pseudopotential --xc PBE

# require specific elements + a relativistic level:
python -m molbuilder pseudo check <dir> --elements Au,C,H,S --xc PBE --relativistic scalar
```
The SIESTA preflight runs the same checks automatically before every
`render_fdf` — ERRORs block, WARNs print. To fix a flagged pseudo:
download a replacement from <http://www.pseudo-dojo.org> (PSML format,
matching the rest of your set's **generator version + XC**), drop it in
the pseudo directory, and re-run the check until clean.

---

## 5. Code/test conformance map

| Check | Severity | `pseudos.py` | Test |
|---|---|---|---|
| C1 coverage | ERROR | `check_coverage` `missing` | `test_pseudos.py` |
| C2 XC | WARN | `check_coverage` `xc_mismatch` | `test_pseudos.py` |
| C3 relativistic | WARN | `check_coverage` `relativistic_mismatch` | `test_pseudos.py` |
| C4 version | WARN | `check_coverage` + `_generator_key` | `test_pseudos.py` |
| C5 dead projector | ERROR | `parse_psml_header.null_channels` + `check_coverage` | `test_pseudos.py` |
| C6 parse | WARN | `parse_psml_header.parse_warnings` | `test_pseudos.py` |
| severity map | — | `validation/siesta.py` | `tests/test_validation*.py` |

Any change to a check here REQUIRES the matching code + test change in
the same commit (the "design doc is the source of truth, update it with
the code" rule).

---

## 6. References

- Kleinman, L. & Bylander, D. M. *Efficacious Form for Model
  Pseudopotentials.* Phys. Rev. Lett. **48**, 1425 (1982). — KB
  separable nonlocal form; `ekb` is the projector strength.
- Hamann, D. R. *Optimized norm-conserving Vanderbilt pseudopotentials.*
  Phys. Rev. B **88**, 085117 (2013). — ONCVPSP, the generator behind
  these PSML files.
- van Setten, M. J. et al. *The PseudoDojo.* Comput. Phys. Commun.
  **226**, 39 (2018). — the vetted source + δ-factor transferability
  benchmarks molbuilder trusts.
- PSML 1.1 format spec:
  <https://siesta-project.org/SIESTA_MATERIAL/Pseudos/Code/psml-1.1.pdf>.
