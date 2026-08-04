# Closed tasks — snapshot 2026-08-04

**Role:** archive
**Domain:** process

Seven tasks that finished between 2026-08-02 and 2026-08-04, moved out of the
working list so it shows only live work. Kept in full because each records a
**decision and its reason**, and several record a *correction* — a thing that was
believed, measured, and found to be false. That is the part a commit message
tends to lose.

Live follow-ups spun out of these are named where they occur; they stay on the
working list under their own numbers.

---

## #41 — the one-label-store migration silently dropped frozen atoms

Closed by `e660831`. Labels could reach the server from two places — inside the
structure, and in a separate key beside it. A reader that knew only one dropped
the other without a word: **50 frozen atoms became 0 in a real run.**

Three surfaces, closed in order:

| | surface | how it closed |
|---|---|---|
| 1 | `.molstruct.json` sidecar | `SCHEMA_VERSION` 7 + a strict gate; `apply_metadata_dict` **refuses** an unknown key by name — correct there, because a sidecar is a **file** and outlives the program |
| 2 | `/api/spectra/render` body | route migrated; the tests still posting the retired shape fixed 2026-08-03/04 |
| 3 | in-script `ATOM-METADATA` block | `9d45ea9` — reads the version line, warns, translates |

**The root** was `apply_labels_to_struct`, a second reader whose last live caller
was the **transport tab** — 12 of its 13 call sites returned on their first line.
Transport sent the geometry as a file *path* with the labels beside it, the shape
everything else retired on 2026-07-31. Migrated it to `molview.exportFile()`, then
deleted the function and all 13 call sites.

> **Why a guard was not the fix — learned by shipping one and watching it fail.**
> A precedence rule ("the envelope wins") was written first, and it silently
> **discarded a label set**: "the envelope carried none" and "the envelope
> disagreed" are one input to any rule that ranks two sources. The absence case
> is what makes ranking unfixable. A structure crosses once, or the question has
> no correct answer.

**The cell had the identical split**, found while closing this and fixed by
`2f7b935`: an envelope stating an 8 Å box plus a top-level `periodicity` stating
20 Å emitted **20** — measured, not inferred.

**Untouched, deliberately:** 14 pre-v7 sidecars under `projects/` are refused by
the strict gate; 8 carry frozen sets. They are the user's scientific record. A
one-shot re-stamp command was offered so the user re-checks rather than re-does.

---

## #44 — MolView's persistency: a door that did not exist, and an edit never saved

Closed by `7447d7d` (write) + `e33d4f8` (read).

**Write half.** `history.js` called a two-call door of its own invention —
`read(step)` / `write(step, bytes)` — that no workspace has ever had. Nothing was
saved, and **every stub satisfied it perfectly**. It now calls the workspace's
real front door by name.

**Read half.** `load(0)` refused on a fresh viewer, so "the sequence outlives the
page" was false exactly where it mattered: the bytes were on disk and nothing
could reach them. A **generated** structure — SMILES, DNA, RNA, peptide, no file
behind it — was simply gone on leaving the tab.

Design decisions, doc first (§ 11.2a):

- a fresh viewer **adopts** the sequence already in storage; only step 0 adopts
- what comes back is the **draft**, not the point — the point is where the user
  chose to be able to return to; the draft is what was on screen. Returning to
  the point throws away every edit after it, silently
- three fields travel with the draft because a reopened page cannot infer them:
  position, highest, dirty
- the draft is written after a **save** as well as an edit: a save moves where
  you stand
- adopting is not anchoring — `anchor()` lays down a fresh point 0 and prunes;
  `adopt()` writes nothing
- two ways to find nothing, kept distinct: no draft = a first visit; an unknown
  version stamp = bytes from a layout this build has never seen

**Browser-verified on the case that had no path**: generate ethanol from SMILES
(9 atoms, no file), delete an atom, reload → 8 atoms back, badge up, selection
intact; Retract → 9 atoms, "saved #0".

**Spun out:** the workspace keeps state under `Path.cwd()/projects`, so an e2e run
writes into the developer's real project tree → **#46**.

---

## #48 — server reload without a manual restart (A–E)

Closed by `d01e2d7`. Plan kept at `docs/ops/server-reload-plan.md` as the record
of what was decided **and rejected**.

- **A** static revalidation. *The URL-version scheme was rejected:* it reaches
  only the 119 `url_for` references, leaving the 51 ESM imports written inside
  the JavaScript on cached copies.
- **B** `serve --supervise` — a parent that never imports application code.
  Constants live in `reload_protocol.py`, a leaf module importing nothing; under
  `web/` they pulled in `app.py` and Flask, destroying that property.
- **C/D** `POST /api/admin/reload`, registered only when a supervisor runs **and**
  `admin_emails` names somebody.
- **E** the button, hidden until availability says otherwise.

**Never exercised in a browser** — the button, the confirm, the poll-and-reload
and a real `--supervise` respawn are all untried against a running server.

**Spun out:** `admin_emails` now gates two unrelated subsystems → **#49**.

---

## #50 — the app locked its own user out

Closed by `4d193a2`. Two instances of *"the app manufactures a 4xx against its own
user."*

**Expired session.** The limiter counts 4xx, and the auth gate's "I do not know
who you are yet" **is** a 4xx. A session expiring with a tab open turned the
page's own 1 Hz poll into one 4xx per second: twenty in thirty seconds, and the
visitor was blocked for an hour **on every path, `/login` included** — silently.
From their side, a dead site.

The gate marks its own answer and the limiter skips that one. Chosen over "ignore
401 everywhere": a 401 from elsewhere has a different author.

**"Nothing saved" answered 404** (`57a194e`) — a 404 carrying `{"ok": true}`.

> **The rule both produced** (`access-control.md` § 7): a 4xx means the *request*
> was wrong, not that the *answer* was empty.

**Left open:** with auth on, the attack-signature check never runs on a path that
maps to a real page — hooks run in registration order and auth is installed
first. Nothing leaks, but the **block** is lost. Pinned as a known shape in
`test_rate_limit.py`; fixing it means reordering hooks, which changes which gate
speaks first for every request.

---

## #51 — read-only tabs do keep their structure

**The user corrected the framing.** Read-only means the *timeline* is a no-op and
`installMolecule` is allowed only into a blank viewer. It says nothing about
whether the **tab** may save its own things.

The tab saves the **structure**, not the path — a path would be re-read on the way
back, so a file changed on disk while you were away would quietly replace what you
had.

**`openMolecule` forces.** It is the user pressing Load, and without `enforce` a
read-only viewer holding a structure answers null — so on structure-optimization,
spectra, transport and the results inspector **you could load one file per page
and picking a second did nothing at all, silently.**

`exportFile` and `installMolecule` are now genuinely inverses.

---

## #52 — label chips were all one colour

Closed by `d26a80f`. **The cause was wiring, not colour.** The tones existed and
the code that picks one existed; *which names are special* was a mount option, and
only `demo.js` ever passed it. Five pages would each have had to repeat the same
four names.

**Decision (user):** the list is MolView's own. `frozen_atoms` is the one that
means something — the calculation holds those atoms still — so it looks *unlike*
the four conveniences rather than like a fifth one.

---

## #53 — a file with an unusable box now opens so it can be fixed

Closed by `dc48dc0`. A sidecar holding a left-handed cell could not be opened at
all: the reader raised and nothing appeared. **A trap with no way out inside the
app** — the Cell page is the one place a box can be corrected, and it needs the
structure on screen.

> **The "second half" was a false alarm, and the correction is the point.** I
> claimed the CLI would then generate from a bad box in silence. Measured:
> `validate()` already reports a left-handed cell as an **error**, and both
> emitters run `report(validate(...))` before writing a byte. I had grepped for
> callers of the periodicity gate, found none in `cli.py`, and concluded
> "unguarded" without checking whether another check already covered it.

| door | result |
|---|---|
| opening a file | opens, and says what is wrong |
| `render_fdf` / PySCF renderer | refuses |
| the six web emitting doors | refuses, 400, at the request seam |

Both halves pinned together in `TestReadingDoesNotJudge` — either alone is the
wrong behaviour.
