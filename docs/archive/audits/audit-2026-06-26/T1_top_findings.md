# T1 — Top-30 findings sweep

**Date**: 2026-06-26
**Auditor**: Claude (general-purpose subagent)
**Method**: read design contracts (`docs/design.md`, `docs/protocols/`,
`README.md`) first, then verify the code in today's commits
(`checkpoint.py`, `web/blueprints/checkpoint.py`,
`web/static/lib/projects/checkpoint.js`,
`web/static/lib/inspectors/markdown.js`, `runwrap.py`,
`transport/results.py`, `parse/engines/siesta.py`, `parse/_log.py`)
against the design.  Spot-checked the web envelope contract and the
projects-sidebar persistence rules.

## Summary
- BLOCKER count: 3
- IMPORTANT count: 10
- NIT count: 7
- **Total**: 20 substantive findings

Key themes — same patterns recur across today's commits:

1. **Envelope drift on the new checkpoint blueprint.** The advisory
   shape (`errors_only: true` + `errors: [...]`) does not match the
   canonical web-api.md § 1.1 envelope (`errors_only: list[Issue]`,
   `issues`, not `errors`).  A future consumer that goes through
   the shared envelope reader will misread these.
2. **Design ↔ code drift on freshly-shipped contracts.** `run-checkpoints.md`
   § 8 still advertises endpoints (`/branch`, `/prune`, `list` returning
   `branches/tags`) that the new blueprint does not implement.  The doc
   was used as a punch-list, then code partially shipped without
   trimming the doc.
3. **Silent defaults that swallow contract violations.** Two of the
   three BLOCKERS are `dict.get(key, DEFAULT_OK_VALUE)` patterns where
   the DEFAULT is the success path — a missing field becomes a valid
   record instead of being rejected.
4. **Persistence rules half-followed in the new sidebar panel.**
   `ws.ui.checkpoint.collapsed` is READ from the workspace snapshot
   but never written, and `ws.ui.checkpoint.view` is written to raw
   `sessionStorage` instead of going through `ws.*`.
5. **Polling that is cheap per tick is expensive in aggregate.** The
   5 s `state()` poll walks `.binsnapshots/` recursively to compute
   `archive_total_bytes` on every tick.

---

## Findings (ranked)

### BLOCKER 1: `TransportResults.from_dict` silently accepts v1 sidecars when `schema_version` is missing

**File**: `molbuilder/transport/results.py:200`

**Design says** (docstring at line 187 of the same file):
> `from_dict`: v2 ONLY (2026-06-25 user directive).  v1 sidecars are
> rejected: they omit ``regions`` and ``frozen_atoms``, which are
> load-bearing for any transport calculation — the device geometry
> has no meaning without L/M/R region assignment …

**Code does**:
```python
version = d.get("schema_version", SCHEMA_VERSION)
if version != SCHEMA_VERSION:
    raise ValueError(...)
```
Because `SCHEMA_VERSION == "2"`, a v1 sidecar (which has NO
`schema_version` key) defaults to `"2"`, the equality check passes,
and the trailing `dict.get("regions") or {}` / `dict.get("frozen_atoms") or []`
calls (line 230-232) silently produce an empty-boundary record — exactly
the case the docstring says is refused.

**Why it's a blocker**: a v1 sidecar slips through as a v2 with no
regions and no frozen atoms.  The "empty-boundary record that downstream
code mistakes for a valid run" is the failure mode the user explicitly
called out.  Detected by reading the function side-by-side with its
docstring; not flagged by any test in `tests/test_transport_results.py`
(would need an explicit "v1 = absence of `schema_version`" fixture).

**Suggested action**: default to a sentinel (e.g. `d.get("schema_version", "1")`
or `d.get("schema_version")` with a `None`-rejection branch) so missing
== v1 == rejected.

---

### BLOCKER 2: Checkpoint advisory envelope does not match `web-api.md` § 1.1

**File**: `molbuilder/web/blueprints/checkpoint.py:95-104`

**Design says** (`docs/protocols/web-api.md` § 1.1, line 42):
> `errors_only` | `list[Issue]` | when the endpoint runs a validator
> or preflight | Pre-filtered subset of `issues` containing only the
> `severity == "error"` items.  Always emitted as a list, including
> `[]` on success.

**Code does**:
```python
def _advisory(message, where=""):
    body = {
        "ok":          False,
        "errors_only": True,                     # boolean, not list
        "errors":      [{"severity": "advisory", # key is "errors", not "issues"
                         "message":  message,
                         "where":    where}],
    }
```
And the JS consumer reads `res.body.errors_only` as a truthiness flag
(`projects/checkpoint.js:526`, `:643`) and `res.body.errors[0].message`.

**Why it's a blocker**: the checkpoint blueprint is the freshly-shipped
PR-B Phase 1.  Any future code that goes through a shared envelope
reader — workspace contract, results dispatcher, or even just a
shared `ws.envelope` helper a teammate writes next month — will
misread these responses (`errors_only=true` is not iterable; the key
is `errors` not `issues`).  The blueprint docstring (line 8-17) cites
"§ 1.6 four-bucket rule" yet the rendered shape doesn't conform to
it.  Reviewer-tripper.

**Suggested action**: replace `errors_only: True` with
`errors_only: [{...}]` (a list), rename `errors` → `issues`, and
add an `errors_only: []` field on the success branch too — match the
canonical envelope.

---

### BLOCKER 3: SIESTA `_warn` writes to ParseLogger inside the predicate dispatch but the build/diag probe path skips logging on its own malformed lines

**File**: `molbuilder/parse/engines/siesta.py:1456-1585`
(`_scan_runtime_info`)

**Design says** (`parse/_log.py:18-19`):
> 2026-06-25T16:30:00Z WARN  line 408: SCF column parse: float() failed -- "******"

**Code does**: `_scan_runtime_info(line)` returns `True` as soon as
any of the 7 probe regexes matches, regardless of whether the captured
value is parseable.  E.g. `_SIESTA_PARALLEL_RE` matches and writes
`siesta_build.parallelisations = tokens` with no validation; if SIESTA
prints a future shape like `Parallelisations: hybrid(MPI:8)` the tokens
get split on commas/whitespace into `["HYBRID(MPI:8)"]` and silently
become "what SIESTA was compiled with."  No `_warn` call, no
`_scan_log.warn`.  The downstream Results-tab badge displays a token
that came from a future SIESTA we don't understand.

**Why it's a blocker (borderline; promoting because runtime_info is
load-bearing for the user's "is my .fdf running on the env I think it
is?" check)**: the comment at line 134-137 explicitly calls this out
as ground-truth for which solver path took effect.  Drift between what
SIESTA actually compiled and what we report = silent science wrongness.

**Suggested action**: each probe should validate its capture (e.g.
`parallelisations` ∈ a known set ∪ "unknown(...)"; `elpa_gpu` token
strictly in `_TRUTHY_TOKENS`) and emit a `_warn(...)` on parse failures
instead of accepting whatever matched.

---

### IMPORTANT 1: `run-checkpoints.md` § 8 advertises endpoints + response fields the blueprint does not implement

**File**: `docs/protocols/run-checkpoints.md:485-498` vs
`molbuilder/web/blueprints/checkpoint.py` (no matching routes)

**Design says** (lines 486, 496-498):
- `GET /api/checkpoint/list` returns `{"checkpoints", "branches", "tags"}`
- `POST /api/checkpoint/branch` (Phase 4) — present in spec
- `POST /api/checkpoint/prune` (Phase 5)  — present in spec

**Code does**: The blueprint only implements
`state / list / diff / init / commit / tag / restore / migrate-manifest`.
`list` returns `{checkpoints: [...]}` only — `branches` and `tags`
fields are absent.  No `/branch` or `/prune` route exists.

**Why it's important**: the doc was used as a punch-list; code shipped
without trimming the doc.  Next reviewer (or next AI subagent) will
treat the doc as authoritative and either reintroduce the call sites
or write a phantom test against the missing endpoints.

**Suggested action**: either trim the doc to what's shipped or land
the missing Phase 4 + 5 routes; do not leave the design doc as a
phantom contract.

---

### IMPORTANT 2: `ws.ui.checkpoint.collapsed` is read on init but never written

**File**: `molbuilder/web/static/lib/projects/checkpoint.js:82-89, 504-515`

**Design says** (`docs/protocols/run-checkpoints.md:309`):
> Hidden state … Collapse state persists per session, also under
> `ws.ui.checkpoint.collapsed`.

**Code does**: `_attach()` reads `snap.state.ui.checkpoint.collapsed`
from `ws.readPersistedSnapshot()`, but `_onCollapseClick`
(line 504-515) only flips the in-memory `_state.userCollapsed` flag.
No call to any `ws.ui.set(...)` / `ws.persist(...)`.  Refresh, and the
collapse is gone.

**Why it's important**: a contract advertised in the design + half
implemented = user reports "I collapsed it and it came back."
Identical class to the `ws.ui.checkpoint.height_px` resize handle
that's specced (run-checkpoints.md:307) and not implemented at all
(grep returns nothing in the JS).

**Suggested action**: write the persistence side on collapse toggle
(canonical `ws.*` path per workspace-contract.md § 4.1, not raw
`sessionStorage`).

---

### IMPORTANT 3: `ws.ui.checkpoint.view` written via raw `sessionStorage.setItem` violates workspace-contract.md § "no other persistence key"

**File**: `molbuilder/web/static/lib/projects/checkpoint.js:74-75, 262`

**Design says** (`docs/protocols/workspace-contract.md:294`):
> **There is no other persistence key.** The legacy keys [molbuilder.structure_canvas, modify-state, molbuilder.panelMode] …

**Code does**: bare `sessionStorage.getItem("ws.ui.checkpoint.view")`
and `sessionStorage.setItem(...)`.  Bypasses the `ws.*` API.
`tests/test_no_legacy_store_consumers.py` (cited at workspace-contract.md:456)
guards against bare `sessionStorage`-by-the-legacy-keys but isn't a
generic "no raw sessionStorage outside lib/workspace/" gate, so this
slipped through.

**Why it's important**: today's commit `160900b workspace: retire legacy
sessionStorage mirrors` explicitly tightened this rule.  Same-day
checkpoint commit re-opens the loophole on a new key.  Pattern will
spread.

**Suggested action**: route through `ws.ui.set("checkpoint.view", mode)`
(or whatever the workspace API for transient-UI fields is called) and
either widen the regression test to ban raw `sessionStorage` outside
`lib/workspace/` or accept this is now the third key.

---

### IMPORTANT 4: `Repo.state()` walks `.binsnapshots/` recursively at every 5 s poll

**File**: `molbuilder/checkpoint.py:773-778`

**Code**:
```python
snaps = p / ".binsnapshots"
total = 0
if snaps.is_dir():
    for sub in snaps.rglob("*"):
        if sub.is_file() and sub.name not in ("MANIFEST", ".gitkeep"):
            total += sub.stat().st_size
```

**Why it's important**: `state()` is hit every 5 s by the sidebar
sensor (`run-checkpoints.md § 11 decision 7`).  For a project with
N checkpoints each archiving M binaries, every tick stats N × M
files.  A 100-checkpoint Au-BDT bias scan with .TSHS + .TSDE per
checkpoint = 200+ stats per tick per open browser tab.  Polls run
even when no checkpoint state has changed.

**Suggested action**: cache `archive_total_bytes` keyed on `(head_sha,
mtime of .binsnapshots)` so unchanged archives don't get re-walked.
Alternatively, drop `archive_total_bytes` from the sensor payload —
it's not displayed in the badge anyway (see `_renderState` in
`checkpoint.js:153-174`).

---

### IMPORTANT 5: `_check_nested_working_dirs` does `rglob("*")` on every entry then filters; on a working dir with 50 k files this is the entire init cost

**File**: `molbuilder/checkpoint.py:186-209`

**Code**: `for sub in path.rglob("*")` then `if sub.is_dir(): continue`
on every entry, then `sub.iterdir()` on each surviving directory.

**Why it's important**: `init` is a one-time call, but on a working
dir that already contains run-output files (e.g. a project the user
forgot to init from the start), the walk inspects every file and
every directory's listing.  Cost is O(files + dirs²).  Manifests as
"why is `molbuilder snapshot init` hanging?"

**Suggested action**: walk with `os.scandir` and skip the
`.binsnapshots` / `.git` subtrees AT WALK TIME (the current `parents`
check still descends INTO them and filters per-entry).

---

### IMPORTANT 6: Markdown inspector's `_loadLibs` doesn't reject on CSS load failure

**File**: `molbuilder/web/static/lib/inspectors/markdown.js:38-44`

**Code**:
```js
const loadCSS = (href) => new Promise((ok) => {
    const l = document.createElement("link");
    l.rel  = "stylesheet";
    l.href = href;
    l.onload = () => ok();
    document.head.appendChild(l);
});
```
No `onerror`.  A 404 on `/static/vendor/codemirror/codemirror.min.css`
leaves the promise hanging forever, so the user sees `"Loading…"`
indefinitely with no error surface (line 111 sets the status; nothing
clears it).

**Why it's important**: this is task #32 freshly shipped; broken CSS
asset = silent UI lockup.  The user can't even tell whether it's a
slow network or a missing file.

**Suggested action**: add `l.onerror = () => ok()` (proceed without
the stylesheet) or `l.onerror = () => ko(new Error(...))` (surface
the failure).

---

### IMPORTANT 7: `_loadLibs` short-circuits on cached promise but doesn't re-check the markdown mode

**File**: `molbuilder/web/static/lib/inspectors/markdown.js:28-60`

**Code**: `if (_libsPromise) return _libsPromise;` early-returns the
cached promise.  But the cached promise resolves once, and on a
second mount the `if (!window.CodeMirror.modes.markdown)` branch
inside the cached promise body has already been skipped.  If, between
the two mounts, something else loaded CodeMirror core but not the
markdown mode (an unrelated inspector preloaded the core), the
markdown mode silently doesn't get loaded but `_libsPromise` says we're done.

**Why it's important**: low-probability today (only this inspector
uses CodeMirror) but the failure mode is "save works, syntax
highlighting doesn't" which is the worst class of UI bug — looks like
a CSS theme issue, not a missing-module issue.

**Suggested action**: cache the result of "all libs loaded" rather
than the in-flight Promise; on every mount verify each library
individually with the existing `if (!window.X)` gates.

---

### IMPORTANT 8: `markdown.js` registers "BEFORE source.js" relying on script load order

**File**: `molbuilder/web/static/lib/inspectors/markdown.js:282-290`,
`source.js:55`

**Code**: comment "Register BEFORE source.js by relying on the
registry's first-match-wins-for-ties policy".  But both inspectors
match `.md`; whichever script the template loads first wins.  The
ordering is implicit in the template's script tag order — fragile
to template refactors.

**Why it's important**: this is the kind of contract that breaks
silently when a teammate alphabetizes the script tags in
`_head_inspectors.html`.

**Suggested action**: registry should accept a numeric `priority`
(or `specificity` derived from match string length) so the ordering
is in the inspector, not the template.

---

### IMPORTANT 9: `_renderError(stRes.body?.error || ...)` uses optional chaining but `_renderError` itself doesn't tolerate `null` message

**File**: `molbuilder/web/static/lib/projects/checkpoint.js:459-466,
176-180`

**Code**: when the fetch returns a status ≥ 500 with no body,
`stRes.body?.error` is `undefined`, and the OR-fallback gives
`"HTTP " + stRes.http`.  But the same defensive guard isn't present
on `_fetchJSON` (line 444-451): the `try {} catch (_) {}` on JSON
parse silently swallows a network error, leaving `payload = null`.
On a network-level fetch failure (offline, server killed), the OUTER
`catch (e)` (line 480) does fire; on a server returning HTML 500
(payload = null), the inner branch correctly handles it.  Mixed
handling between the two error paths.

**Why it's important**: the `_renderError` path is correct, but it
relies on every caller using the same pattern.  Consolidate.

**Suggested action**: centralise the "give me an error string" logic in
`_fetchJSON` so it always returns `{ok, error, body}` shape.

---

### IMPORTANT 10: Checkpoint blueprint `_get_body` returns `request.get_json(silent=True) or {}` — silently coerces invalid JSON to "no body"

**File**: `molbuilder/web/blueprints/checkpoint.py:117-121`

**Code**:
```python
def _get_body() -> Dict[str, Any]:
    return request.get_json(silent=True) or {}
```

**Why it's important**: malformed JSON in a POST body (curl typo,
double-encoded payload) becomes `{}`.  Downstream, missing required
fields surface as "missing parameter: path" — a misleading 400 error
when the real failure was "body wasn't JSON at all."  Same anti-
pattern that `feedback_three_stage_contract.md` flagged for the
modify endpoints.

**Suggested action**: distinguish "no body" (legal for some routes)
from "invalid body" — `silent=False` and catch the BadRequest,
returning a specific "request body is not valid JSON" 400.

---

### NIT 1: Dead branch in `_SIESTA_FEATURE_RE` handler

**File**: `molbuilder/parse/engines/siesta.py:1557-1560`

**Code**:
```python
name = m.group(1).lower().replace("-", "")
if name == "netcdf4":
    pass  # canonical key
runtime_info.setdefault("siesta_build", {})[name] = True
```
The `if name == "netcdf4": pass` branch is a no-op left from a half-
written canonicalisation step.  Either delete the branch or finish
the canonicalisation (e.g. map "elpa" → "elpa_support") — leaving
documentation-as-pass is misleading.

---

### NIT 2: `from_dict` docstring vs code contradict each other on v1 handling

**File**: `molbuilder/transport/results.py:184-233`

The docstring at line 187 says "v2 ONLY"; line 229's comment says
"v2 fields — back-compat: missing => empty (v1 sidecars)."  Pick
one and rewrite the other.  This is the surface of BLOCKER 1.

---

### NIT 3: Checkpoint advisory `where` field documented as 4-bucket-rule key but no caller uses it

**File**: `molbuilder/web/blueprints/checkpoint.py:103, 251, 350, 358`

`_advisory(..., where="path")` / `where=".binsnapshots"` / `where="working-tree"`
sets a field that the JS consumer never reads
(`projects/checkpoint.js:528, 644` just shows `errors[0].message`).
Either surface the `where` in the UI (e.g. as a tag in the advisory
chip) or drop the param — dead data.

---

### NIT 4: Inspector `mount` returns synchronously per registry contract but the async IIFE could update state after dispose

**File**: `molbuilder/web/static/lib/inspectors/markdown.js:190-261`

The async IIFE captures `cm`, `mtime`, `dirty` via closure and writes
to them after multiple `await` points.  Each await is followed by
`if (aborted) return;` — except step 3 mounts CodeMirror without
re-checking `aborted` AFTER the CodeMirror constructor returns
(line 225-235).  In practice this is < 1 ms; in a degenerate testing
scenario (very large markdown file blocking the main thread) the
constructor could race the dispose.  Move the `aborted` check to AFTER
the CodeMirror constructor.

---

### NIT 5: `_continue_force_args_parser` references `name_for_usage` parameter never used

**File**: `molbuilder/runwrap.py:133, 160`

The parameter is named but never appears in the returned bash text.
The doc comment ("Caller is responsible for the eventual ``--help``
text") suggests this was once threaded through.  Drop the param or
use it.

---

### NIT 6: Migration manifest CLI hint is hard-coded but never tested against the actual CLI name

**File**: `molbuilder/checkpoint.py:325`

> `molbuilder snapshot migrate-manifest <ref>`

The error string is verbatim user-facing; if the CLI subcommand gets
renamed (the run-checkpoints CLI is brand-new and may still drift)
this message becomes wrong.  Consider importing the CLI command name
from the CLI module, or have a single test that grep-asserts the
error string matches a real `cli` subcommand.

---

### NIT 7: `_archive_binaries` writes to `MANIFEST` even when binaries list is empty after the guard

**File**: `molbuilder/checkpoint.py:400-424`

Line 405 returns early if `not binaries`.  After that, the sorted-by-name
write at line 422-423 is safe.  Fine.  But on the `_restore_archived_binaries`
side (line 442-445), an empty `expected` dict raises — i.e. a manifest
with zero entries is illegal.  Symmetric guards: writer refuses to
write empty archives, reader refuses to read them, but the writer
guard is "no binaries to archive → no archive dir created" while the
reader guard is "archive dir present but empty MANIFEST → raise."
A user who pre-creates the archive dir then runs restore gets a
500 instead of a no-op.  Minor.

---

## What you did NOT review (be honest about gaps)

* **CSS / UI styling.**  T3 handles this; only noted markdown
  inspector's split-pane CSS is loaded from a separate module not
  inspected here.
* **Test depth.**  T4 handles this; I noted candidate gaps
  ("`tests/test_transport_results.py` doesn't have a no-`schema_version`
  v1 fixture") but did not exhaustively grade test→contract gating.
* **Parse engines beyond `siesta.py` build/diag probes.**  PySCF
  parser, molwatch parser, transport `parse_output` — all spot-read,
  none audited in depth.
* **Spectra subsystem.**  Skipped entirely; no design doc was named
  in the brief.
* **Auth + rate-limit.**  Skipped; not on the change surface today.
* **Embedded viewer.**  2 000-line doc; out of scope for a top-N sweep.
* **3DNA + nucleic / peptide builders.**  Skipped; not in today's commits.
* **The new `@gitgraph/js` vendored bundle.**  I read the JS adapter
  but did not inspect the vendored library source for integrity /
  CSP / license issues — worth a 10-min pass before users hit it.

## Cross-cutting themes (3-5 sentences)

The freshly-shipped checkpoint stack (T1 BLOCKER 2, IMPORTANTS 1-5,
NIT 3) repeatedly half-implements a contract: blueprint envelope
shape diverges from `web-api.md`, sidebar persistence is read-but-not-
write, design doc still advertises Phase-4/5 endpoints the blueprint
skips.  These are all "build the happy path, defer the contract
conformance" failures.  The pattern recurs in `from_dict` (BLOCKER 1)
— write the docstring, then code a default that contradicts it.
Recommended cross-cut: an "envelope round-trip" test that POSTs to
each `/api/checkpoint/*` route and asserts the response satisfies
`web-api.md` § 1.1, plus a `from_dict(missing_field)` test fixture
for every sidecar schema.  Both are cheap and would have caught
2/3 BLOCKERS here.
