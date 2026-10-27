# Plan consolidation, 2026-10-27 — what moved out of the plan, verbatim

**Role:** record
**Domain:** all

*(the user, 2026-10-27: "consolidate the plan and give me an update")* — every section and row below stood in [`plans/plan.md`](?doc=plans/plan.md) until 2026-10-27 (2.46.2) and is kept here **verbatim**, under its verdict. Each verdict was read against the code at the lines it names, the changelog ([`process/changelog.md`](?doc=process/changelog.md)) and the git log, by four read-only reviewers, one quarter of the plan each; the plan keeps only what is open, with where it is in the code and which contract owns it.

---

## 1. Done — moved whole

**§ 1s — done 2.27.0**: `task.py` `task@2`, `job_request.py` `job-request@1`, the migrate split `jobset/migrate.py` `_split`.

## 1s. What is computed vs how it runs — **shipped 2.27.0** *(N5)*

Built as two records, `task.json` and `job-request.json` ([`execution/job-request.md`](?doc=execution/job-request.md)); the section as it stood is [`archive/2026-10-21-1s-job-request.md`](?doc=archive/2026-10-21-1s-job-request.md). A calculation described before 2.27.0 is read after `jobset migrate` (§ 8 there).

---

**§ 1v — done 2.25.0**: `web/zip_stream.py`, `web/blueprints/files.py` `zip_prepare`, `download_zip`.

## 1v. The folder download — sized, asked, streamed *(2026-10-20; the user's yes: "yes, go with your recommendation, add it to the plan" — built as 2.25.0, the contract in `web/projects.md` § 4 and `web-api.md`)*

*(the user, 2026-10-20: "i accidentally started a download of the huge
calculation AuBDTAu folder. i don't know how to stop it. is there a way to add
a UI warning for this operation"; "i worry about timeout … or other issues on
stream zip")*

**What happened, measured.** The Files page's folder download is two
requests: `POST /api/files/zip_prepare` compresses the whole folder into a
temporary file (`files._build_archive`, `tempfile` in `/tmp`, `ZIP_DEFLATED`),
then `GET /api/files/download_zip` streams it. The prepare request sends
nothing while it builds — no progress, and a cancel in the browser does not
reach it. On `AuBDTAu` (200 GB, 98 GB of it the device's two Green's-function
files) it had written 8.6 GB into `/tmp`, on a root disk with 42 GB free,
when the server was restarted to end it and the file removed.

**Decided** (the user's yes, 2026-10-20):

1. **The size, said and asked before anything is built.** The prepare step
   adds up the folder's sizes (names and sizes, no content read) and the
   page asks: *"`AuBDTAu` holds 200 GB — download it as a zip?"*, with what
   is left out (2) and, for a large folder, the line that copies it instead:
   `rsync -a <user>@<host>:<path> .`, the real one. It refuses only a zip
   that cannot fit where it would be written — none, once (3) holds.
2. **What is not the calculation, left out, and said**: the checkpoint's two
   stores (`.git/`, `.binsnapshots/` — the saved history, this machine's) and
   the engines' scratch (`warmfiles.is_scratch`, the one answer the
   checkpoint asks).
3. **Streamed, stored, with its exact length.** The zip is written straight
   into the response (`zipfile` into the stream, ZIP64 for files over 4 GB),
   each file stored, not compressed — the folder's large files are binary
   and barely compress, and deflate costs about an hour of one core for
   200 GB — so the archive's size is exact from the sizes recorded at the
   start, sent as `Content-Length`: the browser shows progress and an end
   time, and a cut connection reads as a failed download, never a short zip
   taken for whole. A file shorter than its recorded size ends the response.
   No file on the server's disk; a cancel in the browser ends the server's
   work at its next write.

**Timeouts, measured** (the user's worry): the server is Werkzeug's own
(`cli.py` `app.run`), which bounds the TLS handshake alone
(`HANDSHAKE_TIMEOUT_S`) and sets no timeout on a request after it
(`conn.settimeout(None)`); a reverse proxy in front (`deployment.md` § 1.1
allows one) cuts a response silent for its read timeout (nginx's default 60
s) — the case of today's prepare step, never of a stream that sends from its
first second.

**Order**: the contract first — `web/files.md` or the files section of
`web-api.md`, whichever owns the route, and the Files page's doc — then the
user's yes on the text, then the code: the route, the page's dialog, a test
that streams a folder holding a scratch file and a store and reads the zip
back through `zipfile` (the scratch and the stores absent, every other file
whole, the length exact).

---

**§ 1x — done 2.30.0** (its heading's *e2e rows and two review rounds next* were done before the release): `transport/reuse.py`, `jobset/reuse.py`, `transport/window.py` `fitted`, the e2e fixture `reused`.

## 1x. A transport calculation that reuses a finished one *(2026-10-22; the user's yes; built for 2.30.0 — the e2e rows and two review rounds next)*

*(The user, 2026-10-22: "a mechanism where we can cite a finished task … and
then say, hey, I want to expand this a bit, or I want to tune certain
parameters that would be able to use the existing, some of the heavy cost
calculation result"; "this is more likely another run or with a different
tag that I would like to create a separate directory that cites the
original one or copy things over if I need to, and then continue that
transport calculation"; "don't overcomplicate it"; "Yes".)*

**The science it rests on** is `engines/transport.md` § 1.2 (what each result
depends on, what a change re-runs) and § 1.3 (the junction by hand): the
leads' Hamiltonians do not depend on the bias; a converged device point does
not depend on the energies the transmission samples; every energy of a
transmission is independent; a new bias point starts from the closest
converged one; the leads' Green's-function files hold one bias and are never
reused.

**The contract draft** *(2026-10-22, uncommitted, for the user's yes)*:
`engines/transport.md` § 2a.16 — with its lines in `job-system.md` (*What a
stage takes*, the five acts, the three diagrams, checkpoints 4a and 6, the
answer's `takes` and `copied`, the hand-over, the way back, the save promise),
`job-contracts.md` § 6.1 and its gather note, `project-layout.md` § 1.5b,
§ 1.6.2, § 1.6.5, `checkpointing.md` § 9's note, `web/results.md` § 2.4,
`web/tabs.md` § 4 and `web/task-setup.md` § 3. In short: `molbuilder
describe --reuses <finished>` (or the Transport tab's picker, given a transport
calculation's folder) starts the new calculation as the finished one's
description, composed junction and `pseudos/`; its prep **copies in** each
group whose members' newest runs there ran the same calculation, with every
stage they take inputs from copied in too — **every run of each member, under
its own name**, so every record inside names a run that is here, `.copied-from`
beside; and a rung of points it prepares **takes over** the finished
calculation's converged points at its first launch — their engine's files and
account, never their records of runs. Every record either act leaves names a
run of this calculation; `.copied-from` alone names the finished one.

**Reviewed before it was brought — two rounds, five fresh reviewers**
(2026-10-22; every load-bearing claim checked against the code). Round 1, on a
first draft that read the finished runs in place: no one door answers a
stage's newest run (seven finders); a run reference outside the calculation
raises (`RunRef.at`); a reused rung needed new stage and group states, and a
`job-set.json` key every row must hold, so every calculation — the finished
one too — would need `jobset migrate`; the new calculation would not have been
whole; and comparing decks alone reused a device computed on leads that run
again (`electrode_kz` reaches only the leads' deck) — the rule now follows the
ladder. Rewritten to copy. Round 2, on the copy: a copied run's records name
runs by the finished calculation's numbers, and a launch again follows them
(`_carried_gather`), so a point taken over carried a record naming another run
of this calculation — now every run of a reused stage is copied under its own
name, and a point taken over brings no record of runs; the decision is made
before a group is planned (a plan takes no write back); a copied group's way
back is the prep's save, its note naming the copy; copied groups leave a
sequence of groups; `pseudos/` is copied at describe, so no second sameness
rule; the next save stores the copies again, so a reused GB costs two.

**REUSE-COPY decided — copied, with a manifest** *(the user, 2026-10-22:
"it's best that when gettings this REUSE-COPY done the user get a manifest of
what will be copied over to continue as a new task with the new parameter
request")*: § 2a.16, *The manifest* — each group copied in (its runs, where
each comes from, how it ended, its size) or prepared here (each difference of
the new request that decided it), a rung of points' take-over, the totals;
shown with the groups the terminal's `prep task` offers and in Task setup's
Preview, before anything is written; kept in `STAGE-PLAN.md` and the ledger.

**A transmission point's energies — each computed once, shared only when
exactly the same** *(the user, 2026-10-22: first "we should not do tricks
about saving points, rather, it is an honest parameter reset … the only thing
that is shared is the energy/density of states/device matrics (or anything
else that could be shared)"; then, shown `lowbias_local`'s measured times —
the transmission 527,441 s ≈ 6.1 days, some 22 min an energy at 16 ranks on
qlabsrv, against 13.2 h for its device — "can we offer the option of
detecting which points can be exactly shared and which ones must be
recalculated?", and "i agree with your recommendation": automatic, strict)*.
The window stays the person's, whole; an energy the finished point computed
to floating-point precision (10⁻⁹ eV, from its `.TBT.nc`) is shared, the rest
computed here as line contours written in the point's deck, the shared ones
copied into the point's `shared/` (a fourth folder role, `shared`) at every
launch that walks it; a point whose grid is the finished point's own is that
point, taken over or copied in; one door joins a point's energies for every
reader; a joined point's current is the record's, by tbtrans's rule; every
transmission deck writes its window's ends at full precision; the manifest
says each point's sharing and — sharing none — the window at the same spacing
that would share all. `engines/transport.md` § 2a.16, *A transmission point's
energies* — drafted 2026-10-22, two review rounds folded in (the second
replaced a contour file named by a hash of its energies with contours in the
deck, which § 6.1's door reads as it is).

**The window fitted to the finished grid, and the default** *(the user,
2026-10-22: "can't we use "old point gap" and target window to automatically
generate that list? i would agree to your recommendation of default numbers
for starting, but we are working on the "expanding" design")*: in a
calculation that reuses a finished one, the person states the range (and,
finer, an odd factor) and one door fits the grid to the finished point's
spacing and alignment, outward by whole spacings until the range is covered —
`transport.window.fitted`, asked by `describe --reuses … --window a,b
[--finer n]` and by the page's *use these* beside the window; written as
fitted, never applied to a typed window (§ 2a.16, *A wider window, fitted to
the finished grid*). **The catalogue's default window becomes −2.005…2.005 eV
at 401 points** — the energies on the 0.01 eV lattice, E_F among them, a
fitted −3…3 then −3.005…3.005 at 601 — for new calculations; a template keeps
what it records.

**The decisions, as they were put** (REUSE-COPY taken; WINDOW-SPLIT superseded by the energies' rule above):

| | the question | the draft | the other way |
|---|---|---|---|
| **REUSE-COPY** | reused results copied into the new calculation, or read where they lie | **copied** (recommended): every reader unchanged, the new calculation whole, the finished one removable once each rung of points has had its first launch; the price is disk — an estimate, 0.2–0.3 GB a device point and 0.6–0.9 GB a transmission point on a 3000-orbital junction, so some 10 GB for a ten-voltage sweep, twice that once the next save stores the copies, and prep copies it (minutes, on the Task setup tab's Prep too) | read in place: no copies, but the door, the cross-calculation run reference, the new states, the `job-set.json` key and its migration, and the finished calculation kept for as long as the new one is read |
| **WINDOW-SPLIT** | a widened window computing only its new energies (the user's 2026-10-22 *"I don't mind to occupy a bit more space if it saves time"*) | **left out** (recommended for now): a widened window runs the transmission whole on the copied device, which is the expensive part reused | kept: a deck that is not the one the description renders (so no later calculation can reuse that point), the curve joined from two runs in every reader of a point (the record, a selection's PDOS, the data file — whose one energy grid then needs the old energies held exactly, the window widened by whole spacings), the current summed from two runs' printed figures (exact: tbtrans's is a sum over its own energies, `m_tbt_save.F90`) and a new record; it saves two thirds of each widened transmission point — worth measuring on the Au–BDT–Au transmissions first |

**Next**: the user's yes on the draft and on the two decisions; then the
build — about 450 lines by the round-2 estimate: describe and the Transport
tab's door (`--reuses`, the junction record and `pseudos/` copied), the copy-in
decided before a group is planned, the gather's three gates made one door the
copy-in and the take-over ask, the take-over's source widened, `.copied-from`
and its catalogue row (`runfiles.WRITTEN`, which also names describe among the
junction files' writers), the way back's copied branch, the page's lines; and
the window fitted to the finished grid (`transport.window.fitted`, describe's
`--window` and `--finer`, the page's *use these*); the catalogue's default
window −2.005…2.005 eV at 401 points (its second home in `config/siesta.py`,
and the tests that hold 401); the energies — the window's ends at full
precision, the new energies as
line contours in the deck with its comment naming the finished point, the
per-point decision read back from the deck for a launched point, the
`shared/` copy at every launch that walks a sharing point and its role
(`calcdirs`), the take-over carrying `shared/`, the point equal to the
finished one's grid rendering its contours, the done rule, the one door that joins a point's energies for every reader (some four
sites today), the record's joined current and `current_from`, the data file's
schema moved — each step bounded by the transport e2e on a real H₂-scale
junction. The energies add some 300–400 lines by their review's estimate.

**Built (2026-10-22, for 2.30.0)** — all six steps, each run on real SIESTA,
TranSIESTA and TBtrans on the H chain in a scratch tree: `describe --reuses`
(`transport/reuse.py`); the copy-in, the take-over, the energies and the
manifest (`jobset/reuse.py`, floor 4); the one join of a point's energies
(`transport/energies.py`); the fitted window (`transport/window.py`) and the
catalogue's new default window; the record at `@9` and the data file at `@3`;
the page — the Transport tab's picker and describe door, the window's fit on
it and on Task setup (`lib/transport/fit.js`), Task setup's echo and Preview,
the Results tab's *copied from*. **About 2,500 lines against the estimate's
750–800 for the whole** — the energies and the page took most. As built, two
details differ from the draft and the contract now says them so: a fitted
window is the transmission's own values on both roads (the range asked said
where it is fitted, not stored), and a point's loop stays TBtrans's count,
the record saying what it shares beside it (`shared`). The e2e rows
(`test_transport_on_a_real_junction_e2e.py`, the `reused` fixture) and their
mutation checks are in, and two review rounds revised it — among them a
transmission that runs once now shares as its one point, and every point of
a record lies on one grid within the 10⁻⁹ eV rule. **Next**: the commit, at
the user's word.

---

**§ 1y — done 2.32.0**: `transport/export.py`, `ncdata.as_tables`, `jobset export`.

## 1y. Export the transport results — CSV and NetCDF *(2026-10-22; the user's yes; built for 2.32.0 — N8 in § 0a's queue)*

*(The user, 2026-10-22: "we also need to add "export data" for all the
results in the transport data including convergence data, in format of csv
and netcdf (whcih allows meta data and calclation specifics)".)*

**What is there today** (read 2026-10-22): `summarize task` writes
`<label>.transport.nc` beside `<label>.transport.json` — NetCDF-4, every
variable with its units, definition and source, the calculation's specifics
as attributes (the citation, the treatment, the I–V's computation, the SCF
criteria, the Fermi levels each in its run's frame), and the convergence: each
device point's SCF history (`device_scf_*` over `scf_cycle`) and each
once-run rung's (`rung_scf_*`) (`engines/transport.md` § 2a.12, *the data
file*). **Nothing exports it from the page** — no route serves the data file;
the folder download (§ 1v) takes the folder whole — **and nothing writes
CSV.**

**What is asked**: an export of every result of a transport calculation —
the transmission, the DOS, the eigenchannels, the I–V, the rungs' facts and
every SCF history — as NetCDF, which keeps the metadata and the calculation's
specifics, and as CSV, for a spreadsheet or a script.

**To settle in the contract** (`engines/transport.md` § 2a.12 and
`web/results.md` § 2.5): CSV holds no metadata, so a CSV export is a set of
tables — long form, one row an observation, each column's unit in its name —
with the specifics in a file beside them, all composed from the one
composition the record and the data file read (`record.collect_record`), never
a second reader; whether the export is the Results report's action, the
command line's (`summarize task` writes the data file already), or both; and
whether it is composed on request or written into the calculation's folder.

**The user's answers (2026-10-22)**: both, one composition; fresh on request,
nothing written into the calculation's folder; CSV as long tables, one per set
of axes, with a `README.txt`. **The contract**: `engines/transport.md` § 2a.12,
*Export*; `web/results.md` § 2.5 (the report's Export); `web/web-api.md`
(`GET /api/transport/export`); `execution/job-system.md` (`jobset export`).
**To build**: `datafile` handing the file's bytes as well as writing it;
`ncdata.as_tables` (a data file's variables as long tables and their README,
generic); `transport.export`; the route, the verb, the report's button.

---

**§ 1ab — done 2.32.0**: `atom_permutation.py`, `engine_atom_index.py`, `transiesta.py` `siesta_numbering`.

## 1ab. Every file says which index its atoms are listed by — the atom index map *(2026-10-23; the user's yes, "Agree. Go ahead."; built for 2.32.0 — AIM, with N8)*

*(The user, 2026-10-23, after reading how the transport data file and a
point's `.TBT.nc` list the atoms in different orders: "We don't want any
accidental error that could lead to misinterpretation of the data … we need
a good framework, agreement, or a contract such that we have a consistent
layer that the user use, rather than have to handcraft all these
translations".)*

**Measured** (two read-only surveys of every file molbuilder writes and every
engine file it reads, `file:line` for each claim, checked against the code,
2026-10-23): every crossing between orders goes through § 2.2a's API — but
only structure sidecars and decks state their order in a field a program can
read; the transport and vibration data files say it in words, the digest in
a prose attribute with no mapping, and the trajectory log prep seeds,
`runtime-info`'s and `watch parse`'s answers say nothing. Four errors against
§ 2's written rules: `transiesta.py:573` counts `elec-pos` to SIESTA's number
by hand (`:578` likewise from the end); the Transport tab's reuse answer sends
masses in the engine order beside a structure in the user order
(`web/blueprints/transport.py:347`, read by no page yet); `positions_ang`'s
words say "as composed" of values in the user order; a page comment calls the
engine order "ORIGINAL" (`trajectory/core.js:808`).

**The user's six answers (2026-10-23)**: the user's saved structures carry
the map too; a deck's block is measured against a finished calculation's
reuse before every deck writes one; one release with N8 (2.32.0); `jobset
migrate` writes the map into older digests and vibration results; the layer
lives in `molbuilder.parse` — the review found that package is floor 1, below the
run door, so the layer sits above both as `molbuilder.postprocess` (the
user's yes, 2026-10-23); a notebook's kernel imports the molbuilder its
server runs. With them: the words *user index* and *engine index*, the
translations named by what they change — **map** for which list the atoms
are in, `map_to_user_index` / `map_to_engine_index`; **convert** for where
counting starts (the user: *"use `convert_to_xxx`, to differentiate from other
designs"*; *"or we should use 'map' when we really mean 'order' of index, and
use 'convert' when we mean starting index"*) — the engine's counting — `convert_to_engine_numbering(i, engine)`,
`siesta_numbering` … with no default engine (the user: *"if
`to_engine_index` is siesta specific, maybe that should be explicit"*),
`molbuilder atom-index` to
see the correspondence, counting stated (`atom_index_base` 0), and the order
carried by the data into the one writer; the fields named as they read
(`atoms_listed_by`, no code word for the sort — the user: *"very unclear what
they represent"*); a structure given to no engine stating no engine index; a
structure file stating nothing read as listed by user index, warned where it
may not be (the user: *"assume user index, and give warning that no explicit
info is available"*), and `not_stated` written where molbuilder does not
know.

**The contract**: `model/overview.md` § 2.2b (and § 2.2a's words renamed to
the user and engine order); `model/structure.md` § 2.2,
`structure-molstruct.md` (v15); `model/parse.md` § 3 (the layer), § 5e (the
digest `@2`); `engines/transport.md` § 2a.12 (the data file's map, *Export*
reading nothing, the notebook examples moved);
`engines/vibration.md` § 6.10.5 (`spectra-data@2`, schema 10);
`web/results.md` § 2.5 (the PDOS steps, the Device view's line, the export
row); `web/jupyter.md` § 7; `execution/job-contracts.md` (the rows).

**To build, in order**: (1) `AtomIndexMap` in `atom_permutation` (it
travels); `ncdata.write_group` taking an atom dimension as its map,
`Var(atom_index=True)`; (2) every writer of § 2.2b's table given its map from
where its atoms came from — the transport data file, the vibration result's
two files, the digest (from its run's deck), the seeded trajectory log,
`runtime-info`, `watch parse`, the codec (sidecar v15), `junction.xyz`'s
comment line, each from the door that knows its columns (F2); (3)
`atom_index_map(path)` on the run door's floor and the post-processing layer
a notebook imports (`model/parse.md` § 3), `molbuilder atom-index`; the DOS by
region and a selection's PDOS checking the digest's map against the
junction's record (F7); (4) the rename — `in_user_order`,
`map_to_user_index`, `map_from_user_index`, the order record's
`convert_…` methods, `engine_atom_index`'s `convert_to_engine_numbering` /
`convert_from_engine_numbering` and `siesta_numbering`, `geometric_numbering`,
`pyscf_numbering` (no default engine), the words "the user order" in code and
messages, nothing
kept under the old names; (5) the four errors above; (6) `jobset migrate`'s
rule for an older digest (`job-request.md` § 8) — an older vibration result
is read with its map made, never rewritten (its hash is named by its data
file and the frame sets made from it); (7) the notebook kernel's path
(`jupyter.run_shepherd`), and xarray listed in the notebook env's recipe (the
user's yes, 2026-10-23); (8) the deck
measurement — a finished calculation's reuse against a deck that always
writes its block — and its answer brought to the user; (9) with N8: *Export*
reading nothing, the PDOS steps, the post-processing document
(`engines/transport-postprocessing.md`) written with every example run on a
real calculation.

**Built for 2.32.0 (2026-10-23)**, all nine steps — step 8 answered by the
user: every deck writes its block, and F4's `not_stated` is built (a view's
mark, never a file's field). **Reviewed three times** (the user: *"really
thorough review on this item index handling … Do it 3 times"*): the contract
against the code, the engine side, then adversarially on the real reordered
calculations the e2e tests build — each round's findings verified and
triaged, then revised once. Dropped in the third round's triage: refusal
messages of an out-of-range index not spelled through `atom_words`, and
`modify --electrode` asking agreement before refusing a negative centre
(polish of an odd input); the chemistry round trips' index-keyed metadata
and identity columns (only a builder's fresh output reaches them, and the
DNA builder's clash relief was measured keeping its identity columns).
Reported to the user, unchanged: RDKit's fallback adds fewer hydrogens than
OpenBabel on some inputs (ethanol: 4 against 6). **With it, the data-file
framework's page** (`model/datafiles.md`; the user: *"make it a general data
file framework doc … and add the specific case for transport"*): the door
rule moved there from `model/parse.md` § 5e, each data file's chapter
citing it.

---

**§ 1z — done 2.31.0**: `parse/engines/tbtrans_nc.py`, `web/blueprints/transport.py`.

## 1z. TBtrans's files read by the SIESTA family's parser, a data file's reading kept *(2026-10-22; the user's yes; shipped 2.31.0 — N9 in § 0a's queue)*

*(The user, 2026-10-22, after Sol's report of `AuBDTAu_singlebias_equilibrium`
took four minutes and showed nothing meanwhile: "can this be done ahead of
time … save as a quick access result?"; "we can let the result to generate
the digest - when it is in the transport directory it can show this
progress/function so user can allow it and/or refresh it"; "no need to
modify further the script side"; "we should use a unified api support rather
than handcrafted code to do this"; "make this part of the parser of siesta
engine would be my feeling".)*

**Measured 2026-10-22** on Sol's point (copied to `~/tmp`, read-only): the
report reads the 1.6 GB `.TBT.nc` six times over, 230,908 reads; Sol's
`collect_record` 3 min 15 s wall, 32 s CPU; the page's request answered after
237 s. One pass over the device DOS is 46,406 reads. TBtrans's k-averaged
text files hold every average the report shows but the DOS per orbital,
equal to the data file's to 5×10⁻⁸ relative.

**The contract**: `model/parse.md` § 5e (one reader per TBtrans file in
`parse/engines/`; `tbtrans-nc`, a registered FileParser, its reading kept
beside the data file as `<label>.TBT.digest.nc`, per version of the file;
`kept_state`, `fresh`, `progress`), `engines/transport.md` § 2a.12 (*A
point's digest*: the report opens on the text files; the DOS by region and a
selection's PDOS from the digests, read on the person's word with progress,
*Read again*; `summarize task` reads what is missing), `web/results.md`
§ 2.5 (the DOS tab's digest line), `web/web-api.md` (`/api/transport/digest`,
GET and POST), the file catalogue's three rows (`project-layout.md` § 5).

---

**§ 1aj — done 2.46.0, its review's follow-ups 2.46.1**: `frame_average.py` (`Row.means`, `warnings_of`, `citation_block`, `with_meanings`), `mode_frames.py`, `transport-result@14`.

## 1aj. A frame set says what its numbers mean, wherever it is cited *(2026-10-27; built 2.46.0)*

*(The user, 2026-10-27: "the details of what the weights and mass correction
is used should be in meta data, and should be in contract so that when these
data are cited and used in other places, the meaning is clear and tracked".)*

The contract: [`model/structure.md`](?doc=model/structure.md) § 2.2f (every
row written with its meaning as its `note`; the rows `sampling_rule`,
`sampling_nodes`, `mass_source`), [`engines/vibration.md`](?doc=engines/vibration.md)
§ 5.10 ③ (the generator writes them), [`engines/transport.md`](?doc=engines/transport.md)
§ 2a.9, § 2a.12, § 3.8.1 (the record's `frame_set` block — rows with
meanings, `info.vibration`, `mass_amu`, the pair and its sha256; the data
file's `definition` from each row's meaning and the average's naming its
rule; the page's hover), [`model/datafiles.md`](?doc=model/datafiles.md) § 2
(a cited value keeps what it means). **Built 2.46.0**: `frame_average.Row.means`;
`mode_frames.frameset_from_mode` writing every row with its unit and meaning
from the vocabulary, and `sampling_rule`, `sampling_nodes`, `mass_source`;
`frame_average.warnings_of`, the one line naming a vocabulary row written
without its meaning, at every reader; `frame_average.citation_block`, the
record's `frame_set` (`transport-result@13`), its masses in the user order;
`frame_average.with_meanings`, each frame variable's `definition` the set's
note, else the vocabulary's; the frame card's hover. Checked through the
road: a set written on the spectrum viewer, cited on the Transport tab,
prepared and launched with the printed commands, summarized — the record and
the data file read back through their doors. The first slice of the data
files' self-description (`model/datafiles.md`).

---

## 2. Partly done — the section as it stood; what is left is in the plan

**§ 1t — the layers: done 2.26.2–2.26.4**; left: the note for a calculation root whose description does not read, its wording the user's (`web/static/results/viewer.js` `PLACE_NOTE_UNKNOWN`).

## 1t. The layers, measured — every upward import *(2026-10-20; the three decisions the user's, given 2026-10-20 — N4 in § 0a's queue)*

*(the user, 2026-10-20: "my challenge to you is to make sure your design of
this is correctly arranged through framework and well separated layers such
that we are not dealing with random handcrafted bugs")* — after `jobset
migrate` met numpy through `Task.label` → `transport.stages` → `structure`.

A scan of every `molbuilder` import, at a module's top or inside a function,
against `architecture.md` § 3 (import depth L1/L2/L3) and
`execution/architecture.md` § 2.1 (floors 1–7): **35 upward imports**, all but
three inside functions — which is how review missed them. They are a dozen
facts on the wrong floor:

| the fact | lives on | asked from | belongs |
|---|---|---|---|
| a lead's own stem and its budget (`rung_label`, `lead_tail_bytes`) | `transport.stages` | `Task.label` | names (floor 1) |
| a stage's number, folder and file names (`stage_home`, `run_names`, `job_dir_names`, `open_container`) | `materialize` (4) | `group`, `plan`, `transport_rung`, `engines` (3) | one naming module below the plan |
| what a stage builds on (`source_stage`, `stage_before`) | `continuation` (4) | `group` (3) | the ladder's rule (3) |
| launched or not (`ever_launched`, `launched_stage`) | `materialize`, `ready` (4) | `group`, `transport_rung` (3) | asked from floor 4 and up only |
| a plan's job by stage (`ready.job_of`) | `ready` (4) | `commands.way_back` (3) | `JobSet.job` |
| the way back / migrate sentence | `commands` (3) | `task`, `jobset.machine` (2, 1) | floor 1 (`identity`, as the checkpoint words) |
| a description's default ladder (`fresh_stages`) | `task` | → `pyscf.stages` | the ladder producers |
| a whole calculation (`Calculation`, holding the plan) | listed floor 2; imports `jobset.model` (3) at its top | opened by `prep`, `submit`, `runstatus`, `migrate`, `_cli` and the web (3 and up) | floor 3 |
| the template catalogue's item facts | `template` (L2) | `identity`, `electronic_state`, `task`, `parse.contract` | the catalogue as floor-1 data; the template file's door on 2 |
| the queues and the fit, as a table (`queue_table`) | `ask` (5) | `placement` (3) | placement (3) |
| a run converged (`converged_of`) | `runstatus` (6) | `continuation.done` (4) | the run's records (1) |
| `WrapperError` | `runwrap` (3) | `jobset.errors` (1) | `jobset.errors` |

**Two rule conflicts**, each with the recommendation: the Results tab's
readers of `task.json` and `job-set.json` (`parse/sidecars`, floor 1) read
through their owners' doors (A16) on floors 2–3 — **each such reader sits on
its record's floor, registered into the parse registry, which imports none**;
`calcdirs.root_of` (1) asks `task.described` (2) — **"is this folder
described" is a name fact, moved to floor 1, `task` asking it**.

**Decided** *(the user, 2026-10-20: "go with your recommendations … but i
suggest your decision 2 could allow a more flexible decision making process.
a folder could be a run dir, but its calculation information is in parent …
shall we give this with a context?")*:

1. **The Results tab's readers of `task.json` and `job-set.json` sit on their
   record's floor**, each reading through its owner's door, registered into
   the parse registry, which imports none of them.
2. **"Where am I" is one ground-floor answer, with its context** — not a
   yes/no *is this folder described*: the calculation's folder, what this
   folder is (the calculation itself, a stage's folder, a run, a point of a
   run, a scratch folder, a benchmark's), and the stage and run it belongs
   to — read from the folder markers every folder molbuilder makes carries
   (`calcdir.json`, `calcdirs.read`) and the root's description file's name.
   *"Is this a calculation folder"* is one field of it.  The sidebar, the
   Results tab, the checkpoint panel and `task` ask it.  The `jobset` verbs
   keep pointing to the calculation's folder when run inside one of its
   folders; letting them act on it from there is a choice for later.
3. **`Calculation` sits on floor 3**, with the plan it holds; nothing below
   opens a whole calculation.

**When**: N4 — each fact moved to its floor, both layer documents
corrected, done when the scan reports none (§ 0a's queue, *NOW*).

**Built in 2.26.2** (the changelog's row): every fact in the table on its
floor — the names below the plan in `jobset/naming.py`, the ladder's rule in
`jobset/ladder.py` (with the fresh ladder), launched-or-not asked from floor 4
up (`ready.launched_without`, `regrouped`; `gather.composed_for_prep`), the
way back and migrate sentences in `identity`, `Calculation.job` for
`ready.job_of`, the catalogue as floor 1 (`template`) and the template file's
door on 2 (`template_file`), `queue_table` in placement, the converged words
`RunStatus.converged_words`, `WrapperError` in `jobset.errors`, `Calculation`
on floor 3 (decision 3); and the L1 imports the scan found beside them —
`warm_declaration` in `jobset.model` (so `warmfiles` is floor 1), the log's
footer lines in `trajectory_log.keys`, the pair's refusals in
`structure_pair`, `runs.saved_pair_row`, the lead stems in `identity`.
**Decision 2, its first step**: `calcdirs.described` reads the description's
file by its name, so the ground floor no longer asks `task`.  **L1T-D1, built in 2.26.3** *(the user, 2026-10-21: "go with your
recommendations on L1T-D1 and L1T-D2")*: the two readers sit on their
records' floors (`task_reader`, 2; `jobset/sweep_reader`, 3), and each
surface registers them as it starts (`calculation.register_readers` -- the
web app, the command line); the scan reports none.  The transport record's
and the displacement sweep's readers ask their writers' schema names in
modules no floor names (`transport.record`, `spectra.displacement_sweep`):
no floor rule judges them.  The Results folder door's join of
`job-set.json` to a folder (`runs._container_product`, A16) stays, its
floor-3 door out of a floor-2 module's reach.  **L1T-D2, built in 2.26.4**
*(the user's yes the same day)*: decision 2's whole answer, *where am I* with
its context, is `calcdirs.where` (floor 1) — the calculation's folder, what
the folder is, its record's role, and the stage, run, trial and point it sits
in, read off its place in the layout's names (`project-layout.md` § 1.4a);
`runs.place_of` is built on it (the Results tab's `place` carries it whole),
the checkpoint panel in the sidebar asks it, and so do the transport record's
route and a benchmark's owner and stage (`materialize.bench_owner`,
`bench_stage_of`).  A point's folder below its stage, the one prep renders
the point's deck into, now carries its record (invariant 6b, which it broke).
**Left, the user's to say**: the Results page's note for a calculation's ROOT
whose description does not read still says *not marked as part of a
calculation* (`results/viewer.js`, `PLACE_NOTE_UNKNOWN` -- a page's words; the
trail beside it says the reader's).  `Where.root` is the folder as given at a
root and resolved below it, as `root_of` always answered; every reader compares
resolved paths, and one spelling would change the paths printed under a
symlinked projects folder -- no visible change today.

---

**§ 1u — the memory a run needs: (a) done 2.29.0**; left: (b) the engine's own analysis for a stage's first run, (c) a formula where (a) and (b) leave a gap.

## 1u. The memory a run needs — a suggestion beside the request *(2026-10-20; the user's yes: "yes, put it in the plan, but prioritize among all remaining issues" — N6 in § 0a's queue; (a) shipped 2.29.0)*

*(the user, 2026-10-20: "can we have some UI feature to give estimate based
on the system/task? … estimate the memory need … as a suggested value, and
task setup … would put it next to the request")* — after asking what memory
the AuBDTAu transmission needs on Sol.

**Decided** (the user's yes to both, 2026-10-20):

1. **Computed when asked, never stored.** One door — the memory a stage needs
   at a rank count, for a calculation — answers Task setup, which shows it
   beside the memory request (*suggested: 49 GB — 0.77 GB a rank × 64,
   measured on 05_transmission/run-0*), and prep, which prints it on its fit
   line. It is a hint: never filled in, never a refusal. Not a key in
   `task.json`: the description is the science, pinned by its fingerprint,
   while the need is derived from it and from how it runs (ranks a node, the
   stage) — a stored copy is a second source, stale when either changes; and
   one key cannot hold it, since each rung of a ladder needs its own.
2. **Where the number comes from, the most trusted first:**
   - **(a) measured** — a finished run of the same stage: what its own
     processes held, bounded at the ranks asked — as `job-request.md` § 3.1a
     now states it (the first form of this line — the job's memory, its trials
     too, a peak ÷ ranks × ranks asked — did not survive its review, below).
   - **(b) the engine's own analysis** — TBtrans and TranSIESTA stop after
     setting up their matrices when asked (`TBT.Analyze`, `TS.Analyze`, both
     in 5.4.2's binaries); TBtrans prints its `[memory]` lines in a full run
     (electrode, H and S, the inversions). To check on one run, in the
     source at hand (`Util/TS/TBtrans/m_tbt_options.F90` and its callers),
     before it is designed: that those lines come before the analysis stops.
   - **(c) a formula** from the system — orbitals, k-points, grid, basis —
     for a SIESTA SCF or a PySCF Hessian with nothing measured: each formula
     brought with the developers' guidance and checked against measured
     runs before any is shown (`scientific-review`).

**Measured for the design** (2026-10-20, `lowbias_local`'s transmission, 16
ranks): TBtrans's `[memory]` lines — electrode 94 MB, H and S 103 MB, LHS
inversion 403 MB, RHS 97 MB — 0.70 GB a rank; each rank's resident memory
0.75–0.80 GB; `util.csv`'s peak 12.31 GB, 0.77 GB a rank. Every rank holds
its own copy (TBtrans divides k-points and energies among ranks), so the need
grows with the ranks a node, not with the k-points or the energies.

**Why after § 1s**: the suggestion sits beside the memory request, which
§ 1s moves out of `task.json` into the job request and its card; built
first, it would be built against a record about to move.

**Order**: the contract first (where the door lives, the line Task setup and
prep show, what a run with no measurement says), the user's yes, then (a);
then (b); (c) only where (a) and (b) leave a gap the user meets.

**The contract draft** *(2026-10-22, uncommitted, for the user's yes — the
user: "go ahead with N6, contract first")*: `job-request.md` § 3.1a, with its
lines in `running-a-job.md` § 5.3.1, `run-reports.md` § 2.1 and § 2.1a,
`model/parse.md` § 5c.1, `task-setup.md` § 6.1, § 7.1 and § 11.1, `web-api.md`
(the prep-plan route) and `job-system.md` § 5.3 (the answer's `need`). In
short: one door, `jobset.memory.need(calc, stage, ranks)`, the ranks the
asker resolved (prep's `--np` over the card; Task setup's card as edited); a
finished run held X — its run record's memory peak, as measured today — at
the N ranks its wrapper launched (`ranks_asked`), so it needs
**at most X · max(1, M/N)** at M: a bound for an engine that gives each rank
its own copy (TBtrans, TranSIESTA's solver) and for one that divides its
matrices, scaled up and never down; a point's need the smallest bound of its
finished runs, a stage's the largest of its points', naming the run, where it
ran and how it ended — *at most* from the kernel's counter, *about* from the
largest sample; a run that did not finish said as what it held, never a bound;
a job's the largest of its members', naming its stage; shown under Task
setup's memory field, in its per-stage list and Preview, and in prep's answer
(`need`) after the fit — never filled in, never compared in words, never a refusal.

**Reviewed — two rounds, three fresh reviewers** (2026-10-22; each
load-bearing claim checked in the code). Round 2, on the rewrite: the launch
record states no ranks (the engine's own count is the run record's); a run
that did not finish gives a floor, not a ceiling, so the bound is taken from
finished runs, a point's the smallest of them; the page asks when the folder
or a rank count changes, reading four facts a run. (A change of what the
monitor measures, which both rounds argued for, was declined — MEM-BASIS,
below: the figure stays the job's, on the high side, its limits stated.)
Round 1: The first draft read the run record's one peak,
and that peak is **the job's** — the cgroup's high-water mark since the job
began, read at the monitor's end and never reset (`monitor._read_mem_peak_gb`)
— so in a walk (a group, a run of points, a sequence) a later member's figure
was an earlier one's; and a cgroup is charged the file cache a run's writes
leave, which for a device writing its leads' Green's-function files would make
the need read as the ask. Scaling by ranks also erred low for an engine that
divides its matrices. Cut as more than the case needs: a benchmark trial as a
source, a note of each differing run setting, the deck-is-current filter (every
run of a launched stage is its calculation now, § 5).

**MEM-BASIS decided — the figure as measured today** *(the user, 2026-10-22:
"no need to refine memory calculation . this is neither scientifically
important or constraining us from resource request. we don't want to
misalign and allocate less memory than actually needed and one job at a
node/task is good enough to know, document the limitation, and that's it. we
don't waste cpu time for this trivial problem unless it will lead to failures
we don't want to see")*: the need reads the run record's one peak as the
monitor measures it now — under a scheduler the job's, file cache in it, so on
the high side; a run here its processes' largest sample, which can be under a
spike or up to a tenth — the limitations stated in § 3.1a, the monitor and its
contracts unchanged.

**What (a) leaves to (b)**: a calculation's first run of a stage has no figure
— the AuBDTAu transmission's first run on Sol is such a run, the run measured
on qlabsrv being another calculation's. The engine's own analysis (b,
`TBT.Analyze`, `TS.Analyze`) is what answers before a run; the order stands:
(a), then (b).

---

**§ 1w — one door per setting: A–D done 2.28.0, its review 2.28.1**; left: four items it parked, each still true in the code.

## 1w. One door per setting, one answer per launched group *(2026-10-21; the user's yes, with the recommendations below; the contract written and taken; A–D built, released as 2.28.0; its review after release, 2.28.1)*

*(the user, 2026-10-21, on the 2.27.0 review: "this sounds to me a problem of
unification and template inconsistency or bad design. identify the nature of
this problem, rather than hurrying to patch it up"; "you always break the rule
of coding of unified api and framework level and keep creating different sets
of handcrafted code")*

**What the review found, by its nature.** Three fresh reviewers read 2.27.0
whole (contract against code; the road describe → prep → launch → status →
migrate; the Task setup page). The road itself holds. What does not hold
falls into two errors of the same kind, both in the framework, not in a
module:

1. **A setting's value is handled place by place.** A setting has ONE
   definition, its catalogue item (`template.Item`, `engines/template.md`
   § 2.1a: every fact about a parameter has one home). But what is DONE to a
   value — judged, spelled, defaulted, labelled, allowed a home — is written
   separately at every place a value can be written. 2.27.0 added one such
   place (the run card) and removed another (the template's run-setting
   values) without giving them one door, so the edges leaked:
   - migrate drops a run-setting value from a `task@2` template (the template
     door refuses it BY CONTENT; migrate's move runs BY SCHEMA VERSION);
   - a run card's value is checked for no type and no choice (`template.why_not`
     leaves the type to a check that runs for overrides only), and is then
     refused at prep in the BENCH's words (cards go through the bench's
     checker, `prep_inputs._declared_execution_pins`);
   - a card's `null` reaches `int()` (`declared_run_shape`) — a traceback;
     a card's `"time": null` erases the allocation's wall (`fold_allocation`);
     a card's wall is stored as typed while the allocation's is spelled at read.
2. **"What may be done to a launched group" is decided in several places.**
   2.27.0 changed the rule (a launched group is prepared again for its job
   request, hierarchical) and moved SOME of the places: prep offers, ticked, a
   launched group whose members changed and then refuses it; `status` tells a
   failed launched run to restore where § 5 says prepare again; the page lets a
   new row join a launched group.

**The inventory, measured** (two read-only inventories, 2026-10-21; each cited
site read before it was written here):

| one fact | decided in | can disagree today |
|---|---|---|
| a setting's TYPE | the catalogue `Item.type` (template reader, `as_declared`); the config ANNOTATION (the override check, the form's `coerce_to_field_type`); config `validate` callables; the page's `CELL_READERS`; the form's number reader | yes — `4.0` for a count, a bool from text, a list's separators |
| its CHOICES | `Item.choices`/`offered`; hand copies in `electronic_state`, `SOLVENTS`, JS locks; `/sweepable` serves raw `choices` where `/columns` serves `offered` | yes |
| its DEFAULT | catalogue `value` and `default`; the config class default (what `resolve` actually applies, labelled *the catalogue's default*); `recommended`; run-time literals — `runwrap` `wall_slack` 5 s and `continue_retries` 0 where the catalogue says 600 and 1, the deck writer's `"ScaLAPACK"` where it says ELPA-1STAGE | equal today by hand and one test; the literals differ |
| its SPELLING (wall, memory, GPUs) | one codec (`scheduler.quantities`), three doors: the allocation spelled at read, a flag at the CLI, a card's wall only inside `Resources` at prep | yes |
| its HOME (which record may hold it) | three catalogue declarations (`category` execution, `run_card`, group `staging`) and five override-name rules (preflight, `resolve`, `/columns`, the transport door, `transport.stages`) | yes — `restart` is in `staging` but is a column; a shared or role override passes Save and is refused at prep |
| where a value CAME FROM | ten vocabularies (template `SOURCE_WORDS`, `resolve.LAYER_WORDS`, the pipeline log's, the fold's, placement's, the page's, the bench's, the transport panel's, the electronic state's) | yes — the pipeline log labels a card's rank count *allocation* |
| may group N be prepared / launched now | `GroupAnswer.preparable`/`prep_offer`, prep's checks, the page's `pickable`, `status`'s steps, the commands door, the CLI's question | yes — the cases above |
| the way on for a launched stage | `commands.way_back` (layout-aware), `redo_launched`, `identity.way_back_words`, `runstatus`'s own sentence, `runwrap`'s retry text, the page's | yes |

And homes with no check at all: a card's launch shape (no type, no range, no
`why_not`); a card's wall; a bench point's number type and range; a CLI
flag's positivity; the web's prep-plan door; the page's `0x10` read
as numbers.

**The design — what changes, framework first:**

- **A. The setting door** (floor 1, beside the catalogue — `template.py`'s
  item doors, gathered). One API every home asks, the catalogue item the only
  source of truth it reads:
  - `accepts(item, value, home, *, engine, kind, stage)` → the one refusal
    sentence or none: the item's TYPE (the catalogue's alone — the config
    annotation stops judging), its choices as the kind offers them, its limits,
    a rung-fixed value, `null` by the home's rule, and whether this HOME may
    hold it at all;
  - `spell(item, value)` → the one canonical form (`as_declared`, and the
    wall/memory/GPU codec for the asks), applied where a record is READ, so a
    card is spelled as the allocation is;
  - `default(item, kind)` → the catalogue's value (a kind's `recommended`
    first) — the one default; `resolve` applies it for an unstated setting,
    and the run-time literals go (a job always carries its value);
  - `home_of(item)` → which records and surfaces may hold a value — derived
    from the catalogue's markers, one table, one sentence for a value in the
    wrong home;
  - `said(layer)` → the one provenance vocabulary, used by the deck, the
    pipeline log, prep's lines and the page.
  The homes then ASK and decide nothing: the template reader, the
  description's preflight (overrides), the job request's checks (cards,
  bench), the CLI flags, `resolve`, prep's input assembly, the hand-over and
  Save, migrate, and the page (through the folder answer and one server door,
  never its own readers).
- **B. Run settings leave the template.** The template is the description's
  (what is computed); a run setting is the job request's. The template writer
  writes no run-setting item, the template reader refuses one as not a
  template item with the run card's sentence, and migrate moves a value it
  finds BY CONTENT, whatever the schema. `template.md` G2 (complete for a
  surface) is restated: the description's items in the template, the run
  settings offered by the run card from the catalogue.
- **C. One launched-group answer.** `ready.group_answer(calc, n)` → the
  group's state (described / prepared / prepared, members changed / launched /
  launched, members changed / held by a sequence), whether it may be prepared
  and launched now and why not, and its way on — one composer, layout-aware.
  Prep's refusals, both offers, `status`'s steps, the CLI's question and the
  page's boxes and group cells all READ it; their own sentences and partial
  rules go.
- **D. The page** (the Task setup review's state findings) is rebuilt on A and
  C: its cells and cards judged by the door, its per-folder state one object
  rebuilt from the folder answer.

**What it retires** (counted at the design's sign-off, not guessed here): the
override type check's config-annotation reader, the bench checker's use for
cards, `card_settings`' detour, the run-time default literals, the five
override-name rules, nine of ten provenance vocabularies, `GroupAnswer`'s
partial `preparable`, `runstatus`'s and `build.py`'s own launched sentences.
Tests follow the API: one table of `(item, home, value) → accepted / refused /
spelled`, the per-home value tests retired into it.

**Contract first** — the sections the change rewrites: `engines/template.md`
§§ 2, 2.1a, 5.3, 6.4, 10; `execution/job-request.md` §§ 2, 4, 7, 8;
`engines/stages.md` § 6.6; `execution/job-system.md` *The task* (the states,
the next-step table, R-P5, R-L7); `web/task-setup.md` §§ 5, 6.2b, 11;
`execution/architecture.md` (the door on floor 1). Then the code, A → B → C →
D, each milestone closed by two fresh full-text rounds and committed, and the
four pushed together as **one release, 2.28.0** (the user, 2026-10-21: *"fine
with your recommendation"* -- the contract's text is 2.28.0 throughout).

**Decided by the user, 2026-10-21** (*"go with your recommendation on 1w,
write the contract first"*):
1. A–D, as above;
2. B — run settings leave the template: no slot, so nothing to police;
3. the config classes keep their defaults, held equal by
   `test_the_DEFAULT_agrees`; every reader asks the catalogue's
   (`template.default_of`), and `resolve` lays it for a run setting no card
   states, so the class copy decides nothing;
4. the page asks a server door to read and judge what was typed
   (`POST /api/task-setup/value`) — no second implementation in the browser;
5. Save's stale-file check is built (`read_as`, `task-setup.md` § 8);
6. N5b before N6, so the memory suggestion is built on the door.

**The contract, as written** (uncommitted until the user's yes on the text):
`engines/template.md` § 2 (no run setting declared in a template; *a home that
must stay empty is removed, never policed*), § 2.1a (one default), § 4.2,
§ 4.3, § 5.3 (counts and the memory cap above 0), **§ 5.4 (the setting door:
`accepts`, `spell`/`read_text`, `default_of`, `home_of`, `SAID`, and the homes
that ask it)**, § 6.4, § 7, § 10, § 10a.2, § 12; `execution/job-request.md`
§ 2, § 3.2 (values judged by the door; `null` is *unset*, for an `optional`
item only), § 3.2a (a card's wall spelled at read), § 4, § 5, § 7, § 8
(migrate by content; a 2.27.x calculation migrated once more);
`execution/job-system.md` *The task* (`ready.group_answer`, read by every verb
and surface; Save refuses a launched group's members changed; the next-step
rows, R-P5, R-L7); `web/task-setup.md` § 2.1 (one per-folder state), § 5,
§ 5.2, § 6.1 (the stated queue kept), § 6.2b, § 8 (`read_as`), § 9a, § 11,
§ 11.1; `engines/stages.md` § 6.6; `execution/job-contracts.md`'s `task.json`
row; `execution/project-layout.md`'s ladder; `architecture.md`'s floor 1;
`web/web-api.md` (Save, the value door, prep's launched words);
`execution/architecture.md`, `generator.md`, `engines/transport.md` and the
README where they still said *declared, valueless*.

**A, as built (2026-10-21).** The door is `template.py`'s § 5.4 section —
`accepts`, `why_not_in`, `home_of`, `spell`, `shown`, `read_text`,
`default_of`, `SAID`. Asked by: the description's preflight (an override's
value, a `varies` column, a run card's value, a benchmark row —
`validation/task.py`, whose annotation-type check, bench-shape checker and
per-home value checks are deleted), prep's card read
(`prep_inputs._card_values`: a card's `null` no longer reaches `int()`),
`resolve` (an override's home, a card's value, and the one default laid for
every run setting no card states), the job request's reader (a card's wall
spelled, a `null` wall or queue refused), `jobset prep`'s flags, the
parameter forms (`_shared.config_from_params`; `coerce_to_field_type`
deleted) and the page's `POST /api/task-setup/value`. The run script's own
slack and retry fallbacks and the deck writer's `"ScaLAPACK"` are gone — a
job without them is refused. One vocabulary for who set a value: the deck,
the pipeline log (a flag's rank count no longer said *allocation*), the
placement line, the transport rung's log. The catalogue: no `value` on a run
setting (checked when it is read); `above` 0 on the counts and the memory
cap; `diag_algorithm`'s `engine_default` corrected to Divide-and-Conquer,
read in `diag_option.F90:213, 224, 242`. Tests: `test_the_setting_door.py`
(the door's tables), rows in `test_hard_limits.py`'s save table and
`data/launch_values.toml`; twelve mutations each failed their rows. The
fast batch: seven failures, each judged — four rendered a run script from a
job with no slack (the input now states it), one pinned our sentence naming
a vibration's rungs on a transport calculation, one the log's *allocation*,
one the route table missing the new door.

**A's first review round** (two fresh readers, 2026-10-21), triaged by the
road: fixed — a `null` benchmark point crashed prep (`int(None)`), and a
benchmark row naming what every trial fixes (`continue_retries`) or a launch
item the grid does not sweep (`max_memory_mb`) was saved and then refused or
silently overruled at prep (the bench's two facts now the door's,
`template.TRIAL_PINS` and `GRID_AXES`, `home_of` derived from `why_not_in`'s
one rule); the deck and the pipeline log composed "who set it" apart and
disagreed for a setting with no default (`ResolvedConfig.said`, composed
once); a stated 0 read as unstated (`placement`); the stage table's columns,
the Transport tab's override refusals and a template's type check each
decided for themselves (now the door's). Dropped as polish of inputs
molbuilder's road does not produce (the user, 2026-10-21: *"overengineering
need to be justified"*): `1e400`, `4x4x1x`, an endpoint body naming an
unknown engine, Python type names in the job request reader's older
messages. **For D**: the page's own readers (`cell-readers.js`,
`collectForm`) and its *Recommended* words still to go; `/api/task-setup/
sweepable` still offers every run setting to the bench card (it should
offer `home_of`'s); the page reads no `SAID`. **Outside § 1w**: GPU on a
non-ELPA solver is refused only when the deck is written, never at Save.

**A's second round** (one fresh reader, told the triage up front), fixed:
the pipeline log said the launch shape came from *the template* -- `resolve`
now says it by where it was stated (`launch_from`, the fold's sources), and
prep's own patch-up of the log is deleted; and deletions --
`effective_config`'s own type map and annotation fallback (it spells through
the door), the template reader's own type wording (one `_type_refusal`), the
"write [...]" hints for hand-typed files and the three tests pinning them,
the Transport tab's pre-check that the preflight repeats, `_cli`'s second
flag table, `resolve`'s second ask of card values prep's read already
judged, the run script's literal list of what it needs (the catalogue's
non-optional wrapper items), a test built on a job no prep makes. Dropped:
`--gpus 0`'s wording (refused by its reader already), the two `time` readers
(one spelling already), the bench's GPU default (the door's already).
Committed locally (5ea09a27).

**B, as built (2026-10-21).** The template writer writes no run setting
(`template_with_values`, every writer's one road); the template file's door
refuses a template holding one, valueless or not, naming `jobset migrate`
(`template_file._refuse_run_settings`); the two checks that policed a value
in the slot (`read_template`'s *"the SCHEDULER answers it"*,
`config_from_template`'s machine-fact story) are deleted with it. Migrate, by
what the files hold: one helper moves a template's run-setting values onto
the calculation's card (`_onto_the_card`) -- of the job request the split
writes, or of the one beside a 2.27.x description, read raw for every later
step -- and the template step takes every run-setting item out (rule
`2.28.0`). **Found on the way, an error of 2.27.0**: migrate's group step
wrote each stage's `group` into a `task@2` description, whose groups are the
job request's, so `jobset migrate` refused every 2.27.x calculation
(*"'group' says how the stage runs"*) -- now the step is a `task@1`
description's alone. Checked once by hand on a throwaway calculation in
scratch shaped as 2.27.x wrote it: prep refused it naming migrate; migrate
took the ten items out, moved a chosen `use_gpu = true` onto the card and
left a not-chosen default out; prep then wrote the catalogue's solver and
slack and said each value's source. Tests: eleven pinned the retired slot --
six read a catalogue fact off a template (now the catalogue's), two say
"none in the template", three retired with the slot; no migrate test (the
user, 2026-10-16: an older release's files are no test's subject).

**B's first round** (one fresh reader, real 2.27.1 and 2.26.4 calculations
in scratch -- refused, migrated, prepared, status, a dry-run launch: the
road holds), fixed: a hand-over a 2.27.x tab wrote and nobody saved was told
to migrate, which refuses a folder with no description -- it is told to send
the hand-over again; the door and migrate asked "is it a run setting" two
ways (the file's category, the catalogue's names) -- the catalogue's alone;
the contract's sentences that still had `read_template` refusing `mpi_np` or
a run setting as a template item's example; a dead clause, a docstring and a
test the slot left. **For D**: the bench card's *value in force* reads the
template, never the run card (`viewer.js`, since 2.27.0). **B's second
round** (a fresh reader, a real 2.27.x PySCF calculation in scratch: the road
holds) -- deleted: a guard for a 2.27.x description with no job request,
which no road leaves, and the fallback it left unreachable (which gave every
stage of a 2.27.x calculation a group of its own during the migration);
four contract sentences and two comments that still described the slot.
Committed locally (419dc072).

**C, as built (2026-10-21) -- no new API.** `ready.GroupAnswer`, which every
verb and surface already reads, is the one answer: `task_state` asks the one
rule (`ready.launched_without`) of each launched group, and one whose
members changed carries the rule's own words (`members_changed`,
`ready.joined_launched`) -- not preparable, that its why -- so prep's offer,
the command line's question, the folder answer, the page's boxes and
`status` (*launched, its members changed*) all decline it; the review's
error, prep offering such a group ticked and then refusing it, is gone.
Save refuses a stage put into a launched group or a launched member given
another group, in the same words; a member taken out of the description is
not Save's to refuse -- the group is then *launched, its members changed*.
`status`'s next step for a failed run names its way on through
`commands.way_back` (layout-aware) where it named a fixed restore. The
contract's `ready.group_answer` / `commands.way_on` are the existing
`ready.GroupAnswer` / `commands.way_back`. Rows: the prep table's regroup row
now meets Save's refusal (`regroup_refused`), and a row for a launched
member taken out of the description (status and prep); three mutations
each failed their row; the stopped-run e2e passes.

**C's review** (a fresh reader, the road driven in scratch), fixed: `prep
task` naming no group listed every group and named one its prep refuses --
it names the groups the answer offers and why each other is not offered;
`status` cut the way on to its first line, losing the restore's commands --
a composed way on is a step's lines, whole; a regrouped group's next step
named one group where prep forms several (`group.formed_again`, older than
C); and three more deciders of the rule went -- launch's group check
(`submit._the_group`), prep's `--one-job` branch and the group cells' own
sentences now ask `ready.launched_without` and say `ready.joined_launched`
(which names the launch of a member alone, as the contract lists) or
`ready.why_group_fixed`. **For D**: the page shows no why beside a box the
answer disables (`task-setup.md` § 11). **C's second round** fixed an error
of the first's: the group cell said a member launched WITH its group "was
launched without it" -- the note is a row below a launched group's alone,
why that group is fixed (`ready.why_group_fixed`); and `status` now says a
stage put into a launched group by hand as *launched, its members changed*,
its way on whole, before *not prepared* (the next-step table's order with it);
the folder answer's `matches` is false for such a group.

**D, as built (2026-10-21).** The page asks the doors. Save takes
`read_as` -- each record's SHA-256 as the folder answer read it -- and
refuses, `409`, nothing saved or written, a record changed on disk since;
the page sends it. Every value typed -- a stage cell, a run-card row, a
benchmark point -- is read and judged by `POST /api/task-setup/value` in its
home, under the edit's fence; refused text is kept as typed and the door's
sentence said; `cell-readers.js` is deleted. The Measure card offers what a
benchmark row may hold (`/sweepable`'s `homes`, `template.home_of`). From
the 2.27.0 page review: the machine and queue cards are drawn for every
folder, a hand-over too (no other folder's *Set to*); the queue the job
request names stays through a machine choice and is shown, with that said,
where the machine does not list it; the report period box returns to its
default; the Measure card waits for a job request, and a hand-over's bench
answer is its engine's and kind's; a blank run row's wall or queue shows the
queue card's; the first point offered is the calculation's run card; a
refused card edit puts each model back to its last text that parsed, the
shape's buttons too; a hand-over's shape builds from its text as it stands,
never over text that does not parse, a job request typed before it kept; an
edit and its undo leave the bar clean (the loaded texts end in the newline
the cards write); the Prep card says why a box is not offered, and
*nothing to prepare yet* before a Save; the words name what Save writes. The
lock on a row below a launched group has one home, the page, from the
answer's `launched` and `why_fixed` -- a row added or renamed since included
-- and `why_same` is `why_not_join`'s alone. Rows: Save's stale-file
refusal; the queue kept through a machine and an ask; nothing of a folder's
queue card on a hand-over opened next; the door's refusal said -- each
failed its mutation. Dropped as polish of what the road does not produce:
the group lock from the edited `builds_on` (Save refuses it), a plan whose
reader refuses (C19). Parked, outside the tab: the parameter forms' own
field reader (`form-schema.js` `collectField`, read again at the server by
`config_from_params`) and the stage table's *Recommended* hover word.

**D's first round** (a fresh reader, the page whole), fixed: a hand-over
read again with the same text left the page no model, so its shape pick
failed (an error of D: the proposal now builds from the model, set from the
hand-over); a folder set to a machine handed that machine to the next folder
as if picked (the person's pick is kept apart from `set_to`); the door's
refusal stayed after the value was corrected; a door that gave no answer
left the typed value on the card; the shape card did not follow a refused
edit on a proposed hand-over or a hand edit, nor the queue and notify cards
a refused block write; a flat layout's launched group showed a disabled box
with no why -- the rule was decided twice (`launched_already` and the group
answer's own layout flag), now `launched_already`'s alone, its words the
answer's `not_again`; the 409's way on did nothing (it says reload); a
hand-over's group tabs asked for command lines of a description not yet
saved; the run rows lost their unit; § 6.2b said the card offers an unset,
which is the editor's. Dropped: the commands route's bench branch on a
folder with no description (the page no longer asks it).

**D's second round** (a fresh reader), fixed: an error of the first
revision -- a hand-over opened from another folder refused its shape pick,
the empty job-request text it loads taken for a hand edit that does not
parse (only a hand edit now says whether its text parses; a row drives the
hand-over road from another folder, and fails without the fix); where a group
stands was worded in four places, against *The task*'s "none … words its way
on itself" -- now `GroupAnswer.said`, the state diagram's name, read by
`groups_text`, `status` and the page; the Continue-from rule decided twice on
the page; a dead `shared` field, a dead guard; a machine with no queues told
the person to pick another queue. Dropped: a server-down title left on the
state box, the *Loaded* line on a proposed hand-over (`removeColumn`'s
already), and `_fs.queue` -- the queue control's own value, as § 9a has the
queue and notify cards hold theirs.

**2.28.0's review, after release -- 2.28.1** (three fresh readers, the release whole:
the setting door and the template, the groups and the task's verbs, the
Task setup page; triaged by the road), fixed:
- *a calculation its readers refuse* -- a template holding a run setting, a
  plan written before a rule, each naming `jobset migrate` -- opened on the
  page as one never prepared, and Save wrote over it: the folder answer
  carries the readers' words (`refused`), the page refuses the folder with
  them, and Save refuses in the same words (no fallback);
- *the group answer, read whole*: `GroupAnswer` carries the groups its prep
  must name (`prep_with`, `group.formed_again`) and the sequence still going
  that holds it (`held`, `held_by_sequence`, its why `ready.held_why` --
  prep's refusal, one sentence); `status`, the bare prep and launch, the
  folder answer and the commands route read them -- no line naming a prep
  that prep refuses, no offer of a group a sequence holds, and a group no
  verb takes answers with its way on; `status` names the stage taken out of
  a launched group; the way on for a removed stage says Task setup; two
  sentences end ` --` so their reason trims; `--one-job`'s way on keeps
  `--one-job`;
- *a group cell refused what prep takes*: a stage stating the default
  (`use_gpu: false`) beside one stating nothing -- `group.card_shares`
  compares the cards as the run takes them (`default_of`, the
  allocation's queue);
- *one fact, one source*: the decks' own `resolved-defaults` rows (a second
  answer beside `parameters:`) and their `DeckSpec` slot are gone -- the
  four keys are the run script's; who set a value is said only for the
  items the kind carries; the preflight lists what a home may hold by
  `home_of`; the door's `template` home, asked by nobody and contradicting
  § 6.4, is gone (the template's two refusals named in § 5.4); the
  annotation-type readers no caller used; `/columns` and `/sweepable` ask
  the door for a default and for what a kind offers; the folder answer's
  unread `builds_on`/`linked`, the page's unread token map, `tsgf` and an
  unreachable refusal;
- *the page*: the column-removal warning names every stage's value it
  drops; a machine whose record lists no queues is said as the record says
  it (no refusal prep does not make); the queue and notify cards are drawn
  on a hand-over once its proposal makes a job request (§ 6.2b); a grid that
  keeps no cell still classes each Measure row; the Measure card's note
  sends the winner to the run card;
- contract sentences the code did not do, and stale comments.
Rows: a stage stating a default shares the group above (fast); a prepared
job's member regrouped beside a stage not prepared -- status names the one
prep that takes (the prep table); each fails its mutation.  Dropped: the
fold's *the run card* for the launch shape (the contract now lists the
word), the `run_card` marker equal to the run settings (parked -- migrate
reads 2.27.x templates holding the key), and every odd input the readers
dropped.

**Also found, outside A–D** (each to be checked before it is fixed): the
catalogue's `engine_default` for `diag_algorithm` was wrong — SIESTA 5.4.2 with
nothing written takes Divide-and-Conquer (`diag_option.F90:213, 242`), ELPA
only when asked, whatever the build carries (fixed in A); documents that contradict one
another (the gather and versions, the leads and the seed's density, the
run-setting ladder, `web-api.md`'s launched prep). The two gaps the same
read found in the notification secrets — a failed Test quoting the address,
`http://` accepted for a channel whose address is its credential — are
closed in 2.27.1 (`run-reports.md` § 3, `monitor.address_refusal` and
`monitor.delivery_failure`).

---

**§ 1aa — extending a finished transport calculation: release 1 done 2.33.0**; left: release 2, the page split, and a door for each control's run time.

## 1aa. Extending a finished transport calculation — what may change, and what is fixed *(2026-10-22; the user's four answers; release 1 shipped 2.33.0 — N11 in § 0a's queue)*

*(The user, 2026-10-22, on citing a finished calculation on the Transport
tab: "I see that all of the options are still open … Why can't we just have
the citation be limited to a few parameters that we actually allow them to
change or suggest they would change? … what are the things that we could
touch and what are the things that we should not touch? These are completely
unclear to me.")*

**Read 2026-10-22**: the command line's `describe --reuses` takes only
`--bias`, the treatment and `--window`/`--finer`; the Transport tab, picking
a finished transport calculation, fills all five cards and leaves every field
editable, the fitted window its one reuse-specific help; Task setup edits
anything after; prep's manifest (`reuse.decide`) then says what is copied
and what runs — after the fact.

**The user's answers (2026-10-22)**: the values that would re-run everything
**read-only, with a way out** (a new calculation from the junction with these
settings); offered — **the voltages and the treatment, the energy window and
its finer grid, the transmission's analysis, the device's convergence**; **the
same limits on Task setup**; the command line **saying** what will be reused,
never refusing. **The rule found in the ladder, not a list**: what runs once
(the seed, the leads — `stages.per_point_rungs`' complement) and the shared
values are fixed; what runs per point (the device, the transmission), the
voltages, the treatment and the window are the person's.

**The contract**: `engines/transport.md` § 2a.16, *Extending it*;
`web/tabs.md` § 4 (the Transport tab's extension panel);
`web/task-setup.md` § 3 (the same limits).

**The user, 2026-10-23, after using it** (*"maybe we should make 'extension'
citation … and 'complete calculation' citation … to have two different panel
design so we can avoid repeating parameter setup"*; *"it is unclear to me for
the extension of transport calculation, what is the meaning/limit on the
computational parameter setups"*; and, from Task setup, a Save with nothing
changed clearing the Prep panel and a Prep again rewriting every prepared
stage): measured — Task setup on an extension is the fresh calculation's
table, every value and every run card editable, the limits said only by the
Preview's manifest; Save writes unchanged files and retires the Prep's
answer; a Prep of a prepared, unchanged group writes every deck and script
again (new stamps), appends each pipeline log and the ledger. **Three
answers**: the Transport tab is the one editor of a transport calculation's
physics, before and after Describe, Task setup showing it read-only; a group
already prepared as prep would prepare it now has nothing to write, said with
its next step; **two releases** —

- **Release 1** (a middle number, shipped 2.33.0): the extension on the
  Transport tab (its fixed surfaces locked by the schema, Describe keeping
  them and refusing a change or a dropped voltage, its manifest, the way
  out with the finished settings) and its limits on Task setup, a copied-in
  group shown as its copy; Save with nothing unsaved offered nowhere and
  writing nothing; prep's *nothing to write* (`job-system.md`, *The task*,
  2; checkpoint 2a, 5 — the launch's own comparison, R-L15, which now
  compares a rung of points' decks too: a window changed after its prep
  had launched the old decks); each group's next step always on the Prep
  card; `describe --reuses` printing the manifest. Two review rounds
  revised it: each change measured against the description it is made
  from (an extension the command line wrote saves), the junction's rename
  and label kept, the way out carrying only the values a person chose,
  every sentence served whole by `Extension.says`, Save's *nothing to save*
  read by the records' readers, a `--one-job` with nothing to write sent
  as one job. **Open**: each control's
  run time — the time the finished runs of those rungs took — wants a door
  for a run's wall time, which nothing reads yet.
- **Release 2**: the page split — the Transport tab the physics editor of a
  described folder (Task setup's safeguards: the state saved first, `read_as`),
  Task setup the machine, groups, run cards and Prep, the physics read-only;
  labels in place of item names on the run cards and the stage table; card
  titles that say what they hold; a transport rung's help in its own words;
  an item shown only where a rung reads it; one line per page saying what
  is checked where (the k-point mesh check at Save too).

---

**§ 1ad.1 — the fifteen decisions: built 2.41.0** (`transport/record.py` `fermi_frames`, `jobset/prepared_with.py`, `jobset/commands.py`, `scheduler/record.py` `unnamed_copy`, `jobset/reuse.py`, the catalogue's `tbt_spin`).

### 1ad.1 The decisions — the user's, 2026-10-25, one by one

| # | the question | decided |
|---|---|---|
| D1 | the device's E_F is placed beside the seed's with the last NEGF cycle's `ts-Vha`; the periodic start's is the right one (SCI1), and a run continued from a density file has no periodic start | **B** — the periodic start's `ts-Vha` of the run where the chain began, that run named in the note (TranSIESTA carries E_F over in the density file it continues from: measured, the 0.2 V point read the 0 V point's `.TSDE` and kept E_F −6.151179 eV); the value alone only when that run is gone; no 0.01 eV *equal* cut. Display only: T(E), G, I–V never read it |
| D2 + D3 | what a row at a nonzero voltage is called; what the low-bias approximation is called | *"there is no reason to extrapolate low-bias assumption to 'linear response' — these are simply wrong. low bias assumption talks about T(E) dependence on bias, nothing about linearity. this is not an ohmic device by default"*: **linear response deleted** wherever it names this (the record's notes, the I–V's `computed` → `low-bias-approximation`, the data file, the command-line help, `task.py`, `transport.md`, the Results page); the low-bias approximation defined as T(E, V) taken as T(E, 0). **The flag stated** — `low_bias_approximation` `true`/`false` in the record and the data file's attributes (so the CSV README), and a *yes*/*no* line in `summarize` and the Results header; one voltage is *no* (its SCF and transmission run at that voltage). The conductance column keeps its name — *"leave user for interpretation"* |
| D4 | `TBT.Spin` 1 or 2: a polarized device's one channel is never done, and the record refuses it (BE1) | **A** — fixed at 0 for transport, written in the deck |
| D5 | a reuse copies a finished seed or lead whose SCF did not converge (only possible with `scf_must_converge = false`), while its plan's warning says *did not finish* (BE4) | **A** — a reuse takes what the calculation's own gather would; the warning says *finished, its SCF not converged* |
| D6 | describing a calculation that cites a failed relaxation succeeds; prep refuses; the folder cannot be re-pointed (WF2) | **A** — describe refuses, before writing anything, naming the way on (launch the relaxation again and cite the run that finishes, or cite another); a run still going stays legal |
| D7 | the closing lines of a prep for another machine say `# here` (WF4) | *"just claim what it is doing"* — `# runs directly on sol`, `# submits to sol's queue`; nothing added |
| D8–D14 | follow standing rules, not questions | D8 the finished treatment kept, as the tab does; D9 narrowed: the old-release clauses printed on current calculations removed (gf_files, the scratch docstring, the displacement sweep, the gather, `redo_launched`); the readers of result files written by an older release stay, since removing them would break existing results; D10, D11 the extension's words; D12 done 2.40.2; D13 not done — its premise was false: `err()` is `modify.py`'s `_err`, so it stays, and `web-api.md` with it; D14 an explicit error (`UnknownTarget.unnamed_copy`) when a calculation's copy of the target's record names no machine; the way on is `checkpoint restore` and `prep --target`, since `migrate` cannot know the machine |
| D15 | launch re-plans prep and compares byte for byte (R-L15, R-L16, R-L45's header) against the 2026-10-17 rule; nothing else says a prepared group no longer matches its description | **C** — prep records the fingerprint of the description, template and request it prepared from; `status` and Task setup say *changed since prepared*; launch sends what prep wrote. And *"make sure user understand: when prep with existing run, should save checkpoint, and start from beginning if possible unless there is no launch at all yet"*: never launched — prepare it again; launched — its runs computed the description as it was: save the folder as it stands, go back to the state saved before the group's first launch, prepare again, the commands printed (WF1's missing command with it) |

---

## 3. The handover of 2026-10-20, as it stood

Its *what runs next* (§ 1u, then steps 2–6) and its decisions were overtaken: § 1u (a) shipped 2.29.0 and N7–N14 after it; RL15 was settled by D15 (2.41.0). The rest is carried into the new handover.

## ▶ HANDOVER — where we are, and where to start *(2026-10-20)*

> **NOW (2026-10-26).** § 1ae M1–M4 built (2.42.0–2.44.0): SIESTA's
> vibration as one run in phases and its Raman at a laser, exported and read
> by a script. Next, each the user's: the NetCDF self-description (the data
> files' definitions in `model/datafiles.md` first, then the writers); M5,
> the contract closed (§ 1ae); the verified findings of § 1ad.2, last-number
> moves; the viewer walked on the user's server with Claude in Chrome; the
> corrections to a junction's electronic structure, a framework first
> (§ 1ag, planned); which modes get Raman, after the run and by symmetry
> (§ 1ah, planned). 2.45.0 laid the vibration's form out by result, drew the
> settings both engines share once, and made *Compute IR*, *Compute Raman*
> and *Raman for* one setting each on both engines; the Structure-optimization
> tab's two locks into the catalogue are next on that road (§ 1ai, planned).
> 2.46.0: a frame set says what its numbers mean wherever it is cited (§ 1aj,
> built), and the Spectrum tab draws the charge and spin in its chemistry
> card, step 2 (§ 1ai's question, decided and built); 2.46.1: `summarize`
> says a set's rows as the card does, the record's `frame_set` null for one
> structure, the shared parts' titles name no engine (the user's rule).

**Where we are.** Shipped since the last handover (2026-10-18; it is
[`archive/2026-10-20-plan-handover.md`](?doc=archive/2026-10-20-plan-handover.md),
verbatim), each row in the changelog: 2.16.3–2.18.0 (the seven-lane review's
fixes, the twelve design calls, the wall reaching every engine), 2.19.0 and
2.22.0 (the leads' surface Green's functions, once per voltage, in a device
run's own `tsgf/`), 2.20.0–2.21.0 (a lost sent run reads `failed`; a running
run's row says where it is), 2.23.0 (the chat card's layout as data),
2.23.1–2.23.2 (records through their owner's door; `jobset migrate` reviewed
whole), and **2.24.0** (`d77bb564`, pushed): the engine's scratch is never
saved by the checkpoint, and the copies older saves made are pruned; `jobset
scratch` lists and empties it; the launch sends the request prep wrote, tells
a run here its wall, and refuses at its boundary instead of a traceback.
Checked by the fast batch (6713 passed, none failed) and the real-junction
e2e (153 passed). **Then 2.24.1**: the 2.24.0 review's errors — one answer to
*is this an engine's scratch* (`warmfiles.is_scratch`), a restore that
removes it, the save before an act decided by content, `jobset scratch`
judged per run, the launch's refusals in words — and every document brought
to the code by nine readers and two rounds of fresh verifiers; fast batch
6,714 passed, real-junction e2e 153 passed. **Then** 2.25.0 (the folder
download, § 1v), 2.25.1–2.25.2 (the transport review's two rounds), 2.26.0
(a sequence of groups launched as one job, `job-system.md` § 6.0) — each a
changelog row — and N3c, the wall and the *written as the job ended* warning
written into the documents word for word (no move), and 2.26.1 (N3d: the
e2e batch's older failures owned, three errors of ours among them), 2.26.2
(N4: every fact on its floor, but the two record readers L1T-D1 holds),
2.26.3 (L1T-D1: those readers on their records' floors) and 2.26.4 (L1T-D2:
*where am I*, one ground-floor answer with its context), and **2.27.0** (N5, § 1s: what is computed and how it runs, two records -- `task.json` and `job-request.json`; `molbuilder describe`; `jobset migrate` splits a calculation described before), and 2.27.1 (a channel's address held to one rule, a failed send said by what failed, never the address).

**After 2.27.0, the user's:** every calculation under `AuBDTAu` is read once `jobset migrate` has run on it -- `lowbias_local` after its local transmission has ended (the `jobset launch` waiting on it holds older code and may end in an error once the run is over; the run's files are the run's own) -- and the dev server restarted on 2.27.0 (the Results tab's *scan failed* there was the server running two releases' code at once).

**On Sol, the user's:** `git pull` to 2.24.1, then `jobset prep task --group
3` and `jobset launch task --group 3 --mode submit` on
`AuBDTAu/transport/AuBDTAu_singlebias_equilibrium` — the prep that sat
copying the 98 GB `.TSGFL`/`.TSGFR` pair is what 2.24.0 fixed; `jobset
scratch --empty` frees the pair. **The transmission's k-grid** is molbuilder's
default, three times the SCF's in the electrode plane: 15×15×1 on the 5×5
SCF grid, 113 k-points after time reversal (TBtrans's own output) against a
9×9×1's 41 — the 2.8× that explains the user's two days on 64 ranks against
the report's 23 h (`projects/AuBDTAu/Time.docx`); the broadening (1 meV, the
device's a tenth) is the report's. The lateral cell is 5×5 Au(111), 14.7 Å:
9×9 is already 0.055 Å⁻¹. Once the local 15×15 run ends, its own file gives
3×3 → 5×5 → 15×15 at no compute (15 = 3 × 5) — the convergence measured,
offered to the user. Memory for its transmission, measured here
on `lowbias_local` (the same junction, the user's word; 2974 device orbitals): 0.77 GB per
TBtrans rank, from TBtrans's own `[memory]` lines and the monitor's peak
(12.31 GB on 16 ranks) — 64 ranks ≈ 50 GB; its wall, from the local run's
progress lines (≈ 150 h on 16 ranks of a 2016 Xeon), ≈ 37 h at most on 64.

**What runs next** is § 0a's queue, *NOW*: the memory suggestion (§ 1u)
— then § 0a's steps 2–6.

**Decisions that wait on the user** — SEQ-ALLOC, WAE, GKN, RL15, V1.9's
remainder, F11's go, SCHED-R's advance signal (§ 0a). The question this
handover carried — whether the run-setting keywords count in the same-input
comparison — is answered by § 1s item 3 (they do not), built in 2.27.0.

**Only Sol can show**, when the user is there: the junction on Sol (E13), the
AuBDTAu workflow, a group's header under a real scheduler (Q13), why no
instrument recorded SIGTERM on Sol job 65999920 (SCHED-R); before a vibration
runs there, Sol's PySCF environment repaired for `netcdf4` (`envs doctor`,
then `envs repair`).

**Off the plan** (the user, 2026-10-18): every feature the three roads do not
need; any check, warning or format the contracts do not state. The AuBDTAu
workflow is the user's, on Sol *(the user, 2026-10-10: "no, you should not
run it on this workstation")*.

---

## 4. § 0a's NOW queue of 2026-10-20, as it stood

Every row shipped but four, whose remainders are the new queue's: N6 (b) and (c) (§ 1u), N11 release 2 (§ 1aa), N12 group 4, N14's verified findings (§ 1ad.2).

### NOW — the Sol road's releases, before step 2 *(2026-10-20)*

*(the user, 2026-10-20: "prioritize among all remaining issues, don't switch
if not justified")* — the order agreed on 2026-10-20 (§§ 1s, 1t), with three
items given their place and the reason for it. Then steps 2–6, as the table
below says.

| # | release | what | why here |
|---|---|---|---|
| **N1** | with N2 — **shipped 2.24.1** | **The 2.24.0 documents sweep** — every contract, command line, printed line and diagram brought to what 2.24.0 does; five readers, one set of files each, every diff read; then five fresh verifiers against a frozen copy of 2.24.0's code (22 false claims, no rule deleted), corrected with N2's own changes, as one commit | the user, 2026-10-20: *"update document … updated systematically"*; *"go with your two choices, continue, complete the current review and fixes first"* — in progress |
| **N2** | **shipped 2.24.1** | **The 2.24.0 review's errors** (two fresh reviewers, 2026-10-20, and N1's readers). *Checkpoint*: one answer to *is this the engine's scratch* — the shipped vocabularies' `scratch` rows, needing no engine recorded — asked by the checkpoint, `jobset scratch`, the walk's take-over, the walk's emptying (its `rm` rendered from it, not `*.TSGF*`) and `status`'s `tsgf/` line (a history begun with no engine archives the pair today); a restore removes the scratch as it removes every file the target does not hold, so `tsgf/` comes back empty and no stray `run-N/` is left (as `transport.md` § 2a.11 says); scratch an older save tracked leaves git before `git add` and `status` never names it; the pruning after the state is recorded, from each archive's MANIFEST, and never failing a save; the save before an act decides *already saved* by content (§ 7.2: a save checksums); a restore asks no configuration (I2a); `checkpoint config` names the scratch; the big files untracked a few hundred paths a call; the save's fingerprints and a restore's verify side by side. *Launch*: the *written as the job ended* warning against the run whose files lie there (flat layout); a calculation's broken `warm-files.toml` a refusal, not a traceback; a header with no wall refused in R-L16's words; every ledger line carrying what the verb was told; `--gather` refused on a rung that gathers nothing; the refusals with no R-L row given one; a stage matched by its one key; `jobset scratch`'s *is this run going* through the jobs door, a run it cannot read kept. *Text*: the docstrings and help lines N1's readers found stale. Two choices in it are Claude's, said so they can be overruled: the scratch list needs no engine (no other engine writes `*.TSGFL`); a restore removes scratch (no later run reads a pair — `transport.md` § 2a.11, *Launched again*) | errors in what just shipped, on the road Sol runs; the walk's part is checked by the real-junction e2e, which alone reaches it |
| **N2b** | **shipped 2.25.0** | **The folder download** (§ 1v): the size said and asked before anything is built, `rsync` named for a large folder; the saved history and the engines' scratch left out; the zip streamed, stored, with its exact length — no file in `/tmp` | the user's yes, 2026-10-20 (*"yes, go with your recommendation, add it to the plan"*), after a download of `AuBDTAu` (200 GB) built 8.6 GB of zip in `/tmp` on a root disk with 42 GB free before the server was restarted; placed before N3 because it can fill the system disk |
| **N3** | **shipped** — round 1 in 2.25.1, round 2 in 2.25.2 | **The transport review** — `transport.md` and the frame-set contract read whole against the code, two fresh rounds; each error fixed as a last-number move | the user's yes, 2026-10-19 (the first two reviewers were cut off); placed here because it looks for wrong results on the road the user runs on Sol, which § 0a puts first |
| **N3b** | **shipped 2.26.0** | **A sequence of groups — one job launching each group in turn** (`job-system.md` § 6.0, *A sequence of groups*): `prep task --group … --one-job` writes its one header (one queue, the largest request, the walls summed); `launch task --group A --group B …` sends one job whose walk runs each group's own launch on the job's node, each run recording the job it rides (`run.json`'s `sequence`); the job's address lists its groups; `status`, the Results tab and `jobs status` say a group *goes in job N*; the file catalogue's `{group}.launch.json` | the user, 2026-10-20: *"how about we allow submission of multiple groups in sequence for one job? … so the lower level logic stays the same"*; *"yes, write the contract, make sure parser, and jobset design are both correctly updated with framework and api level design/implmentation, result should understand this new structure"* |
| **N3c** | **done 2026-10-21** — docs, no move | **The engine's wall and a file written as the job ended — is it documented?** Check that the work giving the engine its wall (SIESTA's `MaxWalltime` from the job's wall less its slack, `MB_WALL_S`, `running-a-job.md` § 3.6) and the check of a dependent file's write time against the engine's stop time — the warning that a file may have been cut off mid-write, a task or job stopped during its last write (the *written as the job ended* warning, `submit`'s `warned`) — are stated EXPLICITLY in the contract, the documents and the README: that they exist, their mechanism, and the exact message a person sees; a gap found is written in, a mismatch with the code reported. **Found and written in:** § 3.6 quoted none of the lines a person sees of the wall, gave the warning's scope as a stage launched again only (it is every launch that continues from a run's files, and the flat layout's files read in place), said it showed on "both doors" (it is the launch's two lines and the ledger; `status` and the browser do not compare), and quoted it with a dash the code does not print; now § 3.6 has every line word for word, the scope, the end it compares with (the closing record, to the whole second) and a run with none, and `siesta.md` § 8.9, `pyscf.md` § 3.9, `run-reports.md` § 2.1b, `job-system.md` § 5.4, the documents index and the README say it. No mismatch in the code; one question, WAE below | the user, 2026-10-20: *"there was a work on adding -wall time to engine, and correctly check the writing/access time of dependent file and the stop time of the engine to warn about potential corrupted writing (task/job stopped during last writing), see if this is correctly documented explicitly about its existence, mechanism, and message in the contract/document/README"* |
| **N3d** | **shipped 2.26.1** | **The e2e batch's older failures, owned one by one** — found by 2.26.0's full `e2e` batch (500 tests, 2026-10-20) and failing the same on the pushed 2.25.2, so older than the sequence work: `test_bench_summary_inspector_e2e` (4: the page's polling), `test_inspector_registry_e2e` (7: the dropdown's headings, the error card, a viewer's teardown), `test_results_file_picker_e2e` (a rescan's parsing state), `test_siesta_run_in_the_viewer_e2e` (opened again and again), `test_vibration_e2e` (the monitor's status line no longer says *SCF iteration*, it says *SCF rows*), and a notebook test that differs from run to run. Each judged against its contract -- the code wrong, or the test asserting a rule no longer stated -- never called flaky; a timing one read wait by wait against the page code. **What they were** (the changelog's 2.26.1 row): three errors of ours -- a PySCF run's status line without its SCF iteration (2.21.0), a plain `.xyz` opened as a trajectory on the Results tab (2.16.3), a benchmark's folder opening nothing -- and tests mounting viewers without the server's answer they match on since 2.13.3; the notebook's one miss did not come back on its own, and its wait now says why when it fails | 2.26.0's batch; the user, 2026-10-20: *"push when all green"* -- these are not green, said so at the push |
| **N4** | **2.26.2**, L1T-D1 **2.26.3**, L1T-D2 **2.26.4** | **The layers** (§ 1t) | agreed 2026-10-20; its three decisions are the user's, and their two choices of how (the user, 2026-10-21) |
| **N5** | **shipped 2.27.0** | **What is computed vs how it runs** (§ 1s), contract first | agreed 2026-10-20 |
| **N5b** | **Built — 2.28.0** (2026-10-22) — the contract taken as written (the user: *"continue"*); A–D, one release | **One door per setting, one answer per launched group** (§ 1w) — the nature of the 2.27.0 review's findings, contract first, then A → B → C → D | the user, 2026-10-21: *"identify the nature of this problem, rather than hurrying to patch it up"* |
| **N6** | **(a) shipped 2.29.0** | **The memory a run needs** (§ 1u) — (a), measured on a stage's runs, built 2026-10-22 (`job-request.md` § 3.1a); (b), the engine's own analysis for a stage's first run, next; (c) only where (a) and (b) leave a gap | the user's feature, 2026-10-20; after § 1s, which moves the request it sits beside; ahead of steps 2–3, *features before framework polish* (the user, 2026-10-18) |
| **N7** | **shipped 2.30.0** | **A transport calculation that reuses a finished one** (§ 1x, `engines/transport.md` § 2a.16) — built, two review rounds, released 2026-10-22 | the user, 2026-10-22: *"use existing information and previous result to continue a calculation by a certain design"* |
| **N8** | **built for 2.32.0** — the user's yes, 2026-10-22; revised 2026-10-23: the export reads no data file (the user: *"the export should only contain information that is already processed"*), the PDOS block as three numbered steps, the post-processing document | **Export the transport results — CSV and NetCDF** (§ 1y), contract first; its place in the order the user's | the user, 2026-10-22: *"export data for all the results in the transport data including convergence data, in format of csv and netcdf"* |
| **AIM** | **built for 2.32.0**, with N8 — the user's yes, 2026-10-23; reviewed three times, each revised once (§ 1ab) | **Every file says which index its atoms are listed by — the atom index map** (§ 1ab, `model/overview.md` § 2.2b) | the user, 2026-10-23: *"we need to be able to differentiate or actually detect it explicitly"* |
| **N9** | **shipped 2.31.0** | **TBtrans's files read by the SIESTA family's parser, a data file's reading kept** (§ 1z) — contract 2026-10-22 (`model/parse.md` § 5e, `engines/transport.md` § 2a.12), the user's yes (*"go ahead"*); built for 2.31.0, two review rounds, each revised once; with it, the scheduler asked after a sent run by the jobs list's rule (`running-a-job.md` § 4.2 — the user, 2026-10-22: *"you should fix that bug"*) | the user, 2026-10-22: Sol's report took four minutes; *"make this part of the parser of siesta engine"* |
| **N11** | a middle number — the user's four answers 2026-10-22; 2026-10-23: two releases, the first shipped 2.33.0 — the extension on the Transport tab, its limits on Task setup and prep's *nothing to write*; the second open (§ 1aa) | **Extending a finished transport calculation — what may change, and what is fixed** (§ 1aa) | the user, 2026-10-22: *"what are the things that we could touch and what are the things that we should not touch?"* |
| **N12** | group 1 **shipped 2.33.1**; groups 2 and 3 **2.33.2**; the four decisions **2.34.0**; group 4 open | **The six-agent review** (2026-10-23/24) — every calculation kind and Task setup, the whole contract against the production code, each finding verified before a fix: group 1, the four errors 2.33.0 shipped (2.33.1); groups 2 and 3, the older road errors and the contract's contradictions with the code (2.33.2), with BENCH-SHELF (§ 2) left for a real shelf to confirm; the four decisions, the user's answers (2.34.0): the device's charges explained region by region and `D(V) − D(0 V)` shown, a re-probe copying `env_init`, every data-file axis a row of `model/datafiles.md` § 2a, no verdict on the leads' Fermi levels. **Group 4, to bring for approval**: messages in the wrong words, unit spellings, a node's two core counts, dead whole-node code, the sweep presenter's columns, `read_data_file`'s atom map, a copied device's provenance, PySCF's two convergence doors, `--one-job`'s nothing-to-write | the user, 2026-10-23: a review of every calculation kind; *"make sure issues detected are first validated and investigated with fact and real code before fixing"* |
| **N13** | **shipped 2.35.0** — the user, 2026-10-24: *"we need 6 to be built first"* | **A mode's frame set — the generator** (step 6's first piece; V1.25, `engines/vibration.md` § 5.10 ③): one vibration mode written as one multi-frame pair a transport calculation cites — frame 0 the Hessian's geometry, the frames at the Gauss–Hermite nodes, each frame's weight, the rows of `model/structure.md` § 2.2f; a command-line verb and the spectrum viewer's button, one generator; the transport side (each frame a point, the weighted average) was built before it. **The frames' extent is set by a maximum displacement** (the user's answer, 2026-10-24: the largest atom's move at the outermost frame, in Å). Not yet clicked on the running server: *Write frame set…* is proven through its route in the real-junction e2e | the user, 2026-10-24: *"integrate vibration mode to a multiframe structure for transport calculation with weight integrated"* |
| **N14** | molecular dynamics removed **2.40.2**; the rest open | **The six-dimension review of 2.40** (§ 1ad) — the fifteen decisions taken (§ 1ad.1, the user, 2026-10-25), then the verified findings with them | the user, 2026-10-25: *"yes, remove MD from the optimization kind"*; the review's findings look for wrong results on the transport road |

---

## 5. § 2 rows done or settled

*TPL-L1 — done 2.26.2 — `template.py:58-60`:*

| # | area | item | from | state |
|---|---|---|---|---|
| **TPL-L1** | engines / template | **`template.md` § 2.2's import rule** (stdlib and `persist` only) is broken by `template.py`, which imports `identity`, `paths`, `runfiles`, `pyscf.stages`, `kmesh` — part of § 1t's layering, or the rule restated | 2.24.0 docs sweep | **done 2.26.2** — `template.py` imports the ground floor alone, the rule restated in `template.md` |

*DOC-TOC — done 2.24.1 — the test reads a copy of the tree (`tests/test_docs_tab.py:46-56`):*

| # | area | item | from | state |
|---|---|---|---|---|
| **DOC-TOC** | tests / docs | **A test writes into the checkout when a document is new.** `tests/test_docs_tab.py::test_toc_returns_each_document_once` asks `/api/docs/toc` over the real `docs/`, and the toc builder persists an unlisted document into `docs/toc.json` (`web/blueprints/docs.py:329-345`) — the suite's checkout guard (`conftest.the_suite_leaves_your_checkout_alone`) fired on it 2026-10-09, when this consolidation added its archive. The same rule is held on an isolated tree by `:200` and `:249` | found 2026-10-09 | open — judged redundant: retire it |

*LEAD-TOL — closed by the user's ruling, as the row itself records:*

| # | area | item | from | state |
|---|---|---|---|---|
| **LEAD-TOL** | transport / frames | **A frame set's lead check is looser than TranSIESTA's** (N3 round 1, 2026-10-20): `compose._frames_keep_the_leads` allows each electrode atom 1e-3 Å (its distance) from frame 0; TranSIESTA allows 1e-3 Å per component of each atom's place relative to the first electrode atom (`TS.Elecs.Coord.Eps`, `m_ts_options.F90`). Two lead atoms nudged opposite ways by just under the tolerance pass prep and stop the device at its start -- loud, after the queue wait; a frame set from a mode with the leads held never does it | N3 | **closed** -- the engine's own check, not ours (user, 2026-10-20: *"don't do the job of the engine"*) |

*RL15 — settled by decision D15 (§ 1ad.1), built 2.41.0 — launch compares nothing; prep records what it prepared with (`jobset/prepared_with.py`):*

| # | area | item | from | state |
|---|---|---|---|---|
| **RL15** | launch | **Launch's one check** — at a first launch, prep's planning asked again on the calculation's own files, refusing a prep no longer what the description gives (R-L15, R-L16); the alternative: Task setup's Save says *prep it again*, and launch checks nothing | the user, 2026-10-17 | **decision owed** — § 0a |

*V1.25 — done 2.35.0 — the frame generator (`mode_frames.py`), N13:*

| # | area | item | from | state |
|---|---|---|---|---|
| ↳ | **V1.25** | **The mode-displaced frame set** *(decided 2026-09-24, user)*: a generator that writes ONE multi-frame pair — the base as frame 0, then `R_A(Q) = R_A⁰ + Q·L_canonical` with `R_F` unmoved at the zero-point amplitude and its thermal growth — from a spectra file, for a named mode; the rule (run, mode, amplitudes) recorded in the pair's `info`. It is the interface to transport's frame axis (W32); a person's own script writes the same pair. Level one of `vibration.md` § 5.6. *W42 (draft 2026-09-28) proposes one set for many modes, at the Gauss–Hermite nodes of the thermal distribution (D6, D7)* | `engines/transport.md` § 2a.9 · `model/structure-molstruct.md` § 6.1 · `science/normal-modes.md` § 4b.6 G | **contract written 2026-09-24; re-ruled 2026-09-29 (TD10)**: the generator is a separate backend procedure, designed on its own — the equilibrium relaxed structure and a normal mode in, one multi-frame structure out — and **each frame's parameter set goes in the structure's `customized` section (W39), not `info`**; transport is its consumer, never its builder. Open, after W39 · *Re-checked 2026-10-09:* unblocked — W39 built in `c93da223`. The contract is `vibration.md` § 5.10 ③: the generator, its own module, writes one frame set for **one** mode (user, 2026-10-09) — `with_frames`, the mode on the structure's rows, each frame's node and `weight` through `set_customized(…, frame=i)`, frame 0's weight included, the set's summing to 1. **Its amplitude, to design** (the user, 2026-10-09: *"how much should the max displacement of a selected vibrational mode use? ... temperature involved? how about local EF enhanced (such as SERS in Raman etc) mode? what's the good starting point"*) — proposed 2026-10-09: the spread is the mode's own, `σ² = (ħ/2ω)(2n+1)` — thermal `n` from the experiment's temperature (zero-point above ~300 cm⁻¹ at 300 K, temperature-driven for contact modes), a stated effective `n` / `T_eff` for a pumped mode (SERS: incoherent, still Gaussian, wider), the max displacement then an output; a coherently driven mode is the one case where the max displacement `A` is the input — the arcsine distribution, Gauss–Chebyshev frames at `0, ±0.866 A`, weights `⅓` each; always a warning past a stated fraction of the nearest-neighbour distance. Its consumer, Q17, is built (a person's script writes the same set today, as the real-junction e2e's does). |

*A1.9 — done 2.32.0 — the read keeps the version the file states (`parse/sidecars/molstruct.py:55-57`):*

| # | area | item | from | state |
|---|---|---|---|---|
| ↳ | **A1.9** | **`load()` restamps `schema_version`.** A sidecar written at 7 reads back as 9 through `molstruct.load` — so no reader can say what version a file was. W33 moved the schema to v10 and inherited it: `parse/sidecars/molstruct.py` stamps v10 on every read | audit § 1.13 | open · *Re-checked 2026-10-09:* `_normalised_dict` stamps the current version (`parse/sidecars/molstruct.py:110,236-249`) — v11 now. · *2026-10-18:* v14 now (`sidecars/molstruct.py:89`; the stamp at `parse/sidecars/molstruct.py:112`) |
