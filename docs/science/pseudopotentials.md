# Pseudopotential validation — the `.psml` coverage checks

**Role:** contract
**Domain:** science
**Companions:** [`validation.md`](?doc=science/validation.md) (the SIESTA
preflight that *runs* these checks); `engines/siesta.md` (the FDF emitter + the
`psml_lib` config field — engines wave, named not linked yet); `overview.md`
(the science contract — composed last).

This is the **sole source of truth** for what molbuilder checks in a `.psml`
pseudopotential set, **why**, and **how**. A **pseudopotential** is a stand-in for
an atom's chemically-inert core electrons, so only the outer (valence) electrons
are computed; `.psml` is the file format that carries one. The implementation
(`molbuilder/pseudos.py` + `molbuilder/validation/siesta.py` + `molbuilder pseudo
check`) must conform to this document; the tests (`tests/test_pseudos.py`) pin
each check to it. If a check here and the code disagree, that's a bug in one of
them — fix it, don't let them drift. *(Cross-cutting terms — pseudopotential, KB
projector, XC functional, valence, Ry — are in the
[`overview.md` glossary](?doc=science/overview.md); this doc glosses its own
specialised ones inline.)*

## Why this guard exists

A **defective `S.psml` shipped silently** into a real run (2026-06-26): an
`ONCVPSP-4.0.1` sulfur pseudo — **ONCVPSP** is a widely-used pseudopotential
*generator program*, and `-4.0.1` its release — with a **dead p-channel**
(`ekb = 0`: the projector strength for the p valence orbitals was zero, so those
orbitals contribute nothing) sat among otherwise-`ONCVPSP-3.3.0` H/C/Au pseudos.
It produced wrong sulfur bonding **and** crashed SIESTA at `propor: ERROR: IMAX=0`
(a crash from a degenerate projector table) — but only at high MPI rank counts (=
many parallel processes), so a low-process GPU run would have *silently* used it
and reported plausible, wrong numbers. None of the pre-existing checks caught it.
The checks below catch that class of defect **at preparation time**, before any
compute is spent and before any wrong number is trusted.

Each check states its scientific basis, its false-positive risk, and its
severity — several thresholds are heuristic and are called out as such.

---

## 1. Where it runs, and the severity model

Two entry points, one engine:

```mermaid
flowchart TB
    F[".psml files in the calculation"]
    RD["parse_psml_header → PsmlInfo<br/><b>identity</b>: element · XC · relativity · generator<br/><b>requirements</b>: cutoff_hint low/normal/high"]

    subgraph L1["LAYER 1 — file integrity (§ 2) · per FILE"]
      CC["pseudos.check_coverage(elements, dir, …)<br/>→ List[CoverageEntry(element, status, …)]"]
      ST["9 statuses: ok · missing · dead_projector · xc_family_mismatch ·<br/>xc_mismatch · relativistic_mismatch · generator_mismatch ·<br/>parse_warning · <b>semilocal_only</b> (§ 2a.2)"]
      SEV["severity — the ONE shared set pseudos.ERROR_STATUSES<br/>ERROR: missing · dead_projector · xc_family_mismatch ·<br/><b>semilocal_only</b><br/>WARN: the rest · ok: silent"]
      CC --> ST --> SEV
    end

    subgraph L2["LAYER 2 — calculation fitness (§ 2a) · the SET + the CONFIG"]
      REQ["the strictest requirement in the set<br/><b>max</b> over the elements that state one;<br/>silence abstains (§ 2a.1)"]
      CMP["cfg.mesh_cutoff vs that number<br/>→ Issue(warn, where=<b>config.mesh_cutoff</b>)"]
      FLOOR["no file states one?<br/>→ the 150 Ry literature floor answers"]
      REQ --> CMP
      REQ -.-> FLOOR
    end

    F --> RD
    RD --> CC
    RD --> REQ

    SEV --> DOOR
    CMP --> DOOR
    FLOOR --> DOOR
    DOOR["<b>the ONE door</b> — validate() → report()<br/>(validation.md § 1.1)"]
    DOOR --> S1["script generation<br/>render_deck"]
    DOOR --> S2["the web preflight<br/>the issues panel"]
    DOOR --> S3["jobset prep"]
    DOOR --> S4["CLI — molbuilder pseudo check<br/>(layer 1 only; it audits a directory,<br/>and has no config to be fit for)"]
```

**Read the two subgraphs by what their verdict is keyed to.** Layer 1 answers
per **element** — *this file is missing / defective / the wrong XC*. Layer 2
answers per **config field** — *your `mesh_cutoff` is below what these files
ask for*. A finding in the wrong one sends a person to fix the wrong thing:
told that something is wrong with sulfur, they re-download a perfectly good
file; told that `mesh_cutoff` is low, they change the number that is actually
low.

Both surfaces call the **same** `pseudos.check_coverage`, which parses each PSML
header once (`parse_psml_header` → `PsmlInfo`, `pseudos.py:168`) and returns
`CoverageEntry(element, status, message, path)` (`pseudos.py:418`) — the same
eight statuses. Both then map status → severity through the **one shared
constant** `pseudos.ERROR_STATUSES` (`pseudos.py:440`), so the preflight (which
blocks script generation, `validation/siesta.py:125`) and the CLI (which exits
non-zero, `cli.py:386`) **cannot disagree about what blocks**:

| Severity | Statuses | Meaning |
|---|---|---|
| **ERROR** (blocks generation / non-zero CLI exit) | `missing`, `dead_projector`, **`xc_family_mismatch`**, **`semilocal_only`** | the run **cannot be correct** — a file is absent (SIESTA won't start), a valence channel is physically missing (wrong Hamiltonian), the XC *family* is wrong, or a valence channel is present and **claims a strength of zero** (§ 2a.2). The last three are all *silently* wrong: the run completes and the answer is not right |
| **WARN** (advisory) | `xc_mismatch`, `relativistic_mismatch`, `generator_mismatch`, `parse_warning` | a strong smell that *might* be intentional (curated mix, deliberate SR/FR choice) — surfaced loudly, but the user may proceed |
| `ok` | — | silent pass |

> **History (2026-07).** Two changes produced the unified set above: (a) XC-*family*
> mismatch was split off as `xc_family_mismatch` → **ERROR** (an earlier version
> treated all XC mismatch as one WARN and left family-vs-author blocking "for
> reviewer decision"); (b) the CLI's exit set — once narrower
> (`{"missing", "dead_projector"}`, so an `xc_family_mismatch` slipped past with
> exit 0 while the preflight blocked it) — was unified onto the shared
> `ERROR_STATUSES`. Both pinned by
> `tests/test_pseudos.py::TestErrorStatusesSharedBySurfaces`.

---

## 2. The checks

### C1 — Coverage · ERROR · `missing`
**What:** every element in the structure has exactly one `.psml` in the pseudo
directory. **Why:** SIESTA refuses to start without a pseudo for every species.
**How:** `scan_psml_directory` (`pseudos.py:393`) maps `{element: PsmlInfo}`; a
structure element with no entry → `missing`. **False positives:** none — a
missing file is unambiguous.

### C2 — XC functional consistency · ERROR *(family)* / WARN *(author)*
**What:** the pseudo's exchange-correlation functional matches the calculation's
(`cfg.xc_authors`). **Why:** a pseudo is generated *for* a specific XC functional.
Using a PBE pseudo in an LDA run (or vice versa) gives **silently wrong** bond
lengths and energies — the run completes, the numbers are off by an amount that
looks physical.

- **Family mismatch** (GGA vs LDA) → `xc_family_mismatch` → **ERROR**. Never
  correct.
- **Author mismatch, same family** (PBE vs PBEsol) → `xc_mismatch` → **WARN**.
  Minor (~1–2 kcal/mol).

**How:** `PsmlInfo.xc_family` / `xc_authors` are decoded from the `<libxc-info>`
functional ids (**libxc** = the standard open-source library that assigns each XC
functional a numeric id) via `_LIBXC_MAP` (`pseudos.py:146`) and compared to the
family derived from `cfg.xc_authors`. An unrecognised libxc id stays `"unknown"` and
does **not** warn — we don't flag what we can't classify. **False positives:**
low, bounded by the libxc id table.

### C3 — Relativistic treatment · WARN · `relativistic_mismatch`
**What:** scalar-relativistic (SR) vs fully-relativistic / spin-orbit (FR)
matches the calculation's intent (SR ignores the coupling between an electron's
spin and its orbital motion; FR includes it). **Why:** FR pseudos are needed only
when spin-orbit coupling matters (heavy elements + SOC-sensitive properties);
using FR when you meant SR (or vice versa) changes the physics. SR is correct for the
large majority of work. **How:** `PsmlInfo.relativistic` (from the header
`relativity` attribute) compared to `expected_relativistic` (default `"scalar"`).
**False positives:** low; `unknown` does not warn.

### C4 — Generator / version consistency · WARN · `generator_mismatch`
**What:** the whole set comes from ONE generator release (e.g. all PseudoDojo
`ONCVPSP-3.3.0`), not a mix. (**PseudoDojo** = a curated, benchmark-validated open
pseudopotential library — molbuilder's recommended source.) **Why:** a single stranger version is the *smell*
that a pseudo was hand-swapped from a different source — exactly how the bad S
entered. Mixed releases also mean the set was never validated together for
**transferability** (whether a pseudo stays accurate across different chemical
environments). **How:** `_generator_key(generator)` (`pseudos.py:612`) reduces
each `creator` string to `name-MAJOR` (e.g. `"ONCVPSP-3"`); a patch difference
(3.3.0 vs 3.3.1) does **not** warn, a major mix (3.x vs 4.x) does. The minority
version(s) are named as the likely stranger(s); only pseudos actually present are
compared. **False positives:** moderate —
mixing *can* be intentional (a curated set drawing the best pseudo per element
from different releases). Hence WARN, not ERROR; the message says "confirm the
odd-one-out is intended."

### C5 — Dead Kleinman-Bylander projector · ERROR · `dead_projector`
**What:** no valence angular-momentum channel (the s / p / d / … orbital-momentum
component of the valence) has its **entire** set of KB projectors at `ekb ≈ 0`.
**Why (scientific basis):** SIESTA represents the
nonlocal pseudopotential in separable Kleinman-Bylander form [Kleinman & Bylander,
*PRL* **48**, 1425 (1982)]: `V_NL = Σ_l |χ_l⟩ E_KB,l ⟨χ_l|`. The KB energy `E_KB`
(the PSML `ekb` attribute) **is** the projector's strength — `ekb=0` means that
projector contributes nothing. If *every* projector of a given `l` is null, that
angular momentum has **no nonlocal projection** at all; its scattering is left to
the local potential only. For an element where that `l` is a valence channel
(e.g. sulfur's 3p) this is physically wrong — the bonding through that channel is
undescribed — and it leaves SIESTA's projector table degenerate, which can
trigger `propor: ERROR: IMAX=0`. **How:** `parse_psml_header` reads every
`<proj l=… ekb=…>`, groups by `l`, and flags an `l` (→ `PsmlInfo.null_channels`,
`pseudos.py:376`) only when **all three** hold:

1. the channel is **present** (has `<proj>` entries) — a channel chosen as the
   **local potential** has *no* `<proj>` at all (absent, not zero) and is not flagged;
2. **every** projector for that `l` has `|ekb| < 1e-6` (`_EKB_NULL`) — one weak
   projector among several is fine; we flag only an *entirely* null channel; and
3. there is **no** `<slps l=…>` semilocal potential for that `l`.

**Condition 3 is essential and easy to miss.** ONCVPSP-4.0.1 / psml-4.0.1 pseudos
write the nonlocal `<proj>` block as **zero placeholders** for channels that are
instead carried by the `<semilocal-potentials>` block, from which SIESTA rebuilds
the KB projector at read time. Standard, PseudoDojo-validated pseudos for **I, Xe,
Rb, Ba** do exactly this for their p channel — so an `ekb=0` there is *not* a dead
projector. Flagging them (the pre-2026-07 behaviour) was a **false positive that
ERROR-blocked those common elements**; the `l not in semilocal_ls` guard fixes it.
**Threshold:** `1e-6` (in the hartree-ish KB units of `ekb`) is heuristic — well
below any physical `ekb` (real ones are O(0.1–20)) and just above
exact-zero/round-off. **False positives:** believed none
for standard ONCVPSP / PseudoDojo sets after the semilocal exemption.

The three cases side by side, in the actual PSML header (`<proj l=… ekb=…>` — one
`<proj>` per projector, grouped by angular momentum `l`; `s`, `p`, `d`, … are the
channel names):

```xml
<!-- (a) HEALTHY p-channel — real ekb, contributes to bonding → ok -->
<proj l="p" seq="1" ekb="4.213" eref="0" type="oncv"/>

<!-- (b) DEAD sulfur p-channel — every projector zero AND no <slps l="p"> → dead_projector (ERROR) -->
<proj l="p" seq="1" ekb="0.0" eref="0" type="oncv"/>
<proj l="p" seq="2" ekb="0.0" eref="0" type="oncv"/>
<!-- (no <slps l="p"> in <semilocal-potentials>) -->

<!-- (c) EXEMPT iodine p-channel — projectors zero BUT a semilocal potential carries it → ok -->
<proj l="p" seq="1" ekb="0.0" eref="0" type="oncv"/>
<semilocal-potentials>
  <slps l="p" seq="1"> … </slps>   <!-- SIESTA rebuilds the KB projector from this -->
</semilocal-potentials>
```

The predicate that decides this is exactly the three-way AND above (simplified
from `pseudos.py:376`; `l` values are the channel letters `s`/`p`/`d`):

```python
_EKB_NULL = 1e-6
semilocal_ls = { slps["l"] for slps in header.iter("slps") }   # l's carried by <slps>

null_channels = [
    l for l, projectors in proj_by_l.items()               # channels that HAVE <proj>
    if all(abs(ekb) < _EKB_NULL for ekb in projectors)     # ...all ~zero
    and l not in semilocal_ls                              # ...and NOT rebuilt from <slps>
]
```

### C6 — Parse integrity · WARN · `parse_warning`
**What:** the file parsed and carried the expected header fields. **Why:** a
malformed / truncated PSML is suspect; surface it but let SIESTA make the final
call at startup. **How:** non-empty `PsmlInfo.parse_warnings` (bad XML, missing
element, unparseable Z, …).

---

## 2a. Two layers, and why the second one is new *(design, 2026-09-03)*

**Every check in § 2 answers one question: is this FILE sound?** One file in,
one status out, keyed by element — present, parseable, right XC family, right
relativity, no dead channel. Eight statuses, all of them properties of a file.

A pseudopotential is also something else, and nothing was asking about it:

> **A pseudopotential DECLARES REQUIREMENTS, and the calculation must satisfy
> them.**

That is a different question with a different shape. It takes the whole SET of
files *and* the configuration, and its verdict is about the **configuration**,
not about any file — so it lands on `config.mesh_cutoff`, which is the thing a
person must change, rather than on an element.

| | layer 1 — **file integrity** (§ 2) | layer 2 — **calculation fitness** |
|---|---|---|
| input | one file | every file in the set **+ the config** |
| asks | *is this file sound, and does it match my XC?* | *does my configuration honour what these files require?* |
| verdict about | the element | the **config field** |
| output | `CoverageEntry(element, status, …)` | `Issue(severity, message, where="config.…")` |
| the strictest what wins | — | the strictest **element** in the set |

**Why a layer and not two more checks.** A file's declared requirement is not a
special case of soundness — it is a fact the file states about the calculation
it belongs to, and *any* such fact enters the same way. Today that is the
recommended cutoff; a file that declared a required relativity, a valence
assumption or a minimum basis would be read by the same reader and compared by
the same rule. Bolting each one onto the coverage scan would make the scan
answer two unrelated questions and key the answer to the wrong thing.

**It costs no new wiring, and that is the test of the seam.** Layer 1 already
flows through `pseudos.check_coverage`; layer 2 is an ordinary
`validation/siesta.py` check, registered like every other. Both already reach
all three surfaces through the one door — `report(validate(...))` is called by
**script generation** (`render_deck`), the **web preflight**, and **`jobset
prep`** — so the backend, the UI panel and the emitted deck get the new findings
without a line of plumbing. A finding that needed new wiring to reach a surface
would be a sign it was put in the wrong place.

### 2a.1 A declared number outranks a literature floor

`_check_siesta_mesh_cutoff` warns below **150 Ry** — a defensible production
floor, and a *guess about this calculation*, because it knows nothing about the
pseudos actually in it. A PseudoDojo v0.5 file states its own numbers
(`cutoff_hint_low` / `_normal` / `_high`; sulfur: **72 / 147 / 162 Ry**).

> **When the files say, the files win. The floor is what answers when they say
> nothing.**

The same shape as the rank count, settled the same day: *read from a record, and
never guessed* (`running-a-job.md` § 3.1). A generic floor that overrides a
file's own statement is a guess outranking a measurement.

**The threshold is `normal`; `high` is named in the message.** `high` is for
tight and vibrational work, which is what the `tight` rung exists to ask for
(`tuning.md` § 1) — so it is information a person acts on, not a bar everyone
must clear.

#### With more than one element, the HIGHEST wins — and silence is not a vote

The mesh cutoff is **one global real-space grid** for the whole calculation.
Every species' density is represented on that same grid, so it must satisfy the
**most demanding** element in the system. Take less and that element is
under-resolved — and the error does not average away across the cell, it sits
on that atom, which in a junction is usually exactly the atom the study is
about (the metal, or the anchor).

So: **the maximum over the elements that state a number.**

**And an element that states nothing must not lower the bar.** In the v0.5
table only the eleven re-generated elements carry hints at all — so a real
system states fewer numbers than it has species:

```
BDT on gold — Au, C, H, S
  Au   (no hint)
  C    (no hint)
  H    (no hint)
  S    147 Ry          <- the only element that says anything
  ------------------------------------------------------------
  required: 147 Ry     (max of what was stated; silence abstains)
```

A rule that averaged, or that let a silent element pull the number down, would
answer *lower* the more elements a system has — which is backwards: adding a
species can only make a grid's job harder. Where nothing states a number, the
literature floor (§ 2a.1) is what answers, unchanged.

### 2a.2 Sound, and still unusable — the semilocal-only channel

A file can pass every check in § 2 and still fail to run, and this is not
hypothetical: it is the **S.psml incident of 2026-06**.

PseudoDojo **v0.5** re-generated eleven elements — Ba, Bi, I, Pb, Po, Rb, Rn,
**S**, Te, Tl, Xe — and in those files some valence channels carry **zero
Kleinman-Bylander projectors** (`ekb=0`, and the radial data is all zeros),
with the `<slps>` **semilocal** block carrying the channel instead. Measured
over both whole tables: **11 of 72 elements in v0.5, 0 of 72 in v0.4.** The
eleven are exactly the eleven the site's own release note names as updated.

For sulfur the affected channel is **p**, and S's valence is 3s² 3p⁴ — so the
channel with nothing in it is one of the two that make sulfur bond. The run
failed; replacing that one file with v0.4 fixed it.

**The file is schema-valid and its VALUES ARE WRONG** *(corrected 2026-09-03,
user: "why do you call this a valid shipment? with values obviously wrong?")*.
An earlier draft of this section called the zeros a legitimate representation
choice. They are not, and the difference is one this codebase already draws
elsewhere:

> **absent ≠ present-but-zero.** A channel chosen as the local potential has
> **no** `<proj>` at all — that is how PSML says *nothing to see here*.

The v0.5 files do not do that. For sulfur they emit
`<proj l='p' seq='1' ekb='0'>` followed by **462 explicit zeros**, twice — each
one fully formed, with a `type`, an `eref`, a `seq` and a `<radfunc>`, exactly
like a projector that means it. That is a positive claim: *there is a p
projector and it is zero everywhere*. Sulfur's p channel is not zero.

**And there is no markup that says otherwise** *(checked against the file,
2026-09-03)*. PSML has no protocol here: nothing in the file points at another
file, nothing marks the `<nonlocal-projectors>` block as a placeholder, and
nothing declares `<semilocal-potentials>` authoritative — both carry the same
`set="scalar_relativistic"`, and the one `action=` annotation is *provenance*
(it records that generation ran semilocal-first, then projectors) rather than
an instruction to the reader.

So a consumer reading the nonlocal block is not failing to notice a hint —
**there is no hint**. It reads the file correctly and gets sulfur with no p.
The engine did the right thing with wrong data, and that is what makes this a
defect in the file rather than a difference of opinion between two readers.

**C5 is still right not to fire**, but for a narrower reason than "the file is
fine": C5 asks *is a channel missing*, and this channel is present and lying.
That is a different question, which is why it is a different check.

**ERROR — it blocks** *(user ruling, 2026-09-03)*. The argument is the
`xc_family_mismatch` argument: the run **completes** and the answer is wrong.
There is no crash to investigate and no line in the output that looks
suspicious, which is exactly the failure a preflight exists to catch — and it
is what happened here, until a comparison against v0.4 explained it after the
fact.

The case for leaving it a warning was that these files' **semilocal** block is
complete and correct, so a code consuming *that* representation gets right
physics from the same file. That is true and it does not survive the ranking:
this preflight runs for SIESTA, whose reader took the zeros, and a check that
declines to block the reader it is protecting is not doing the job. A user
whose reader consumes the semilocal form points `psml_lib` at a set that does
not carry the claim — v0.4.1 of those eleven elements.

---

## 3. What is NOT checked (the bounded claim)

So the guard isn't trusted beyond its reach. molbuilder does **not** currently
verify:

- **Valence-charge correctness** — e.g. whether Au uses an 11- or 19-electron
  valence appropriate to the chemistry (`z-pseudo`, the count of valence electrons
  the pseudo treats explicitly, is parsed for sanity, not correctness).
- **Ghost states** — spurious bound states below the valence; needs
  generation-level analysis, not header inspection.
- **Transferability / hardness** — whether the pseudo reproduces all-electron
  results across environments. That's the job of the *source* (PseudoDojo's
  **δ-factor** benchmarks — a scalar scoring how closely the pseudo's
  equation-of-state matches all-electron); molbuilder trusts a vetted source and
  checks only consistency + integrity.
- **Mesh-cutoff adequacy from the pseudo** — `PsmlInfo.suggested_mesh_ry` is
  parsed (a PseudoDojo extension) but not yet compared to `cfg.mesh_cutoff`.
  *(SIESTA's mesh-cutoff floor is checked separately in `validation/siesta.py`,
  independent of the pseudo's suggestion.)*
- **Basis ↔ pseudo consistency** — that the **PAO** (pseudo-atomic-orbital, SIESTA's
  numerical basis set) l-channels match the pseudo's. Deferred.

The guard catches **missing, mis-functionaled, mis-relativistic, version-strange,
structurally-broken, or unparseable** pseudos. It does **not** certify that a
pseudo is *good* — that comes from using a vetted source (PseudoDojo)
consistently.

---

## 4. Process — how to use it

The SIESTA preflight runs these checks automatically before every `render_fdf` —
ERRORs block, WARNs print in the issues panel. To screen a directory yourself
(the CLI entry point is the `pseudo check` subcommand — the `pseudo` group is at
`cli.py:338`, the `check` command at `:342`):

```bash
# screen a whole directory (CI gate; exits non-zero on any ERROR-status pseudo:
# missing / dead-projector / xc_family_mismatch -- the same set the preflight blocks):
python -m molbuilder pseudo check projects/pseudopotential --xc PBE

# require specific elements + a relativistic level:
python -m molbuilder pseudo check <dir> --elements Au,C,H,S --xc PBE --relativistic scalar
```

It prints one line per element (`[ERROR]` / `[WARN ]` / `[OK   ]`) then a summary,
and exits non-zero when any line is an ERROR — the shape a CI job greps / gates on:

```text
  [ERROR] S        p-channel projectors all ekb=0 — dead Kleinman-Bylander projector
  [WARN ] Au       generator ONCVPSP-4 differs from the set's ONCVPSP-3 — confirm intended
  [OK   ] C

3 checks: 1 error(s), 1 warning(s).          # process exits 1 (an ERROR was found)
```

The same check is one Python call:

```python
from pathlib import Path
from molbuilder.pseudos import check_coverage

for e in check_coverage(["Au", "C", "H", "S"], Path("projects/pseudopotential"),
                        expected_xc_family="GGA", expected_xc_authors="PBE"):
    if e.status != "ok":
        print(e.status, e.element, "—", e.message)
```

**To fix a flagged pseudo:** download a replacement from
<http://www.pseudo-dojo.org> (PSML format, matching the rest of your set's
**generator version + XC**), drop it in the pseudo directory, and re-run the
check until clean.

---

## 5. Code / test conformance map

| Check | Severity | `pseudos.py` | Test |
|---|---|---|---|
| C1 coverage | ERROR | `check_coverage` → `missing` | `test_pseudos.py` |
| C2 XC family | **ERROR** | `check_coverage` → `xc_family_mismatch` | `test_pseudos.py` |
| C2 XC author | WARN | `check_coverage` → `xc_mismatch` | `test_pseudos.py` |
| C3 relativistic | WARN | `check_coverage` → `relativistic_mismatch` | `test_pseudos.py` |
| C4 version | WARN | `check_coverage` + `_generator_key` | `test_pseudos.py` |
| C5 dead projector | ERROR | `parse_psml_header` → `null_channels`, emitted by `check_coverage` | `test_pseudos.py` |
| C6 parse | WARN | `parse_psml_header` → `parse_warnings` | `test_pseudos.py` |
| severity set (shared) | — | `pseudos.ERROR_STATUSES` → `validation/siesta.py:125` (preflight) + `cli.py:386` (CLI exit) | `tests/test_pseudos.py::TestErrorStatusesSharedBySurfaces` |

Any change to a check here REQUIRES the matching code + test change in the same
commit (the "design doc is the source of truth, update it with the code" rule).

---

## 6. References

- Kleinman, L. & Bylander, D. M. *Efficacious Form for Model Pseudopotentials.*
  Phys. Rev. Lett. **48**, 1425 (1982). — the KB separable nonlocal form; `ekb`
  is the projector strength (basis for C5).
- Hamann, D. R. *Optimized norm-conserving Vanderbilt pseudopotentials.* Phys.
  Rev. B **88**, 085117 (2013). — ONCVPSP, the generator behind these PSML files.
- van Setten, M. J. et al. *The PseudoDojo.* Comput. Phys. Commun. **226**, 39
  (2018). — the vetted source + δ-factor transferability benchmarks molbuilder
  trusts.
- PSML 1.1 format spec:
  <https://siesta-project.org/SIESTA_MATERIAL/Pseudos/Code/psml-1.1.pdf>.
