# The template as an `.fdf` with item blocks — the retired design

**Archived 2026-08-11.** History, not policy. What replaced it is
[`engines/template.md`](?doc=engines/template.md).

This was `execution/job-contracts.md` § 3.7. It specified the template as a file
in the **engine's own format** — a real `.fdf` — with each parameter wrapped in
`# === molbuilder item <field> BEGIN/END ===` markers, its declaration in a
comment, and the deck line it produces as the block's **payload**.

**Two errors retired it, and the second one only became visible once the first
was fixed.**

**1. It said `prep` substitutes a stage's overrides at their anchors.** It does
not, and two contracts already said so — `engines/stages.md` § 4 (*"prep
rebuilds a config by scanning them, and nothing has to parse an `.fdf`"*) and
this section's own property 1, which had recorded the three forcing constraints
in 2026-08-07: one parameter can decide *where* another lands (`relax_type`
moves the step budget between `MD.NumCGsteps` and `MD.FinalTimeStep`), one
parameter can write two keywords (`spin_total`), and a parameter can write no
line at all at its default. The *"substituted at their anchors"* paragraph was
added on 2026-08-11 and contradicted all of it.

**2. With substitution gone, the reason for the engine's format went too.** The
section argued the file *"cannot be JSON or YAML, and it is not a preference"*
because the payload lines had to be the engine's own input. Once `prep` renders
rather than splices, the payload is not the source of anything — and it carried
a real cost: **the value was stored twice**, in the declaration's `value=` and
in the payload line beside it, so the file could disagree with itself and a
hand-edit of one would silently be ignored. A comparison table in the section
weighed four candidates and never compared a single data file — it compared a
**sidecar** (two files) against in-comment formats, which is a different thing.

**Kept and carried forward** into `engines/template.md`: the `kind=` closed
vocabulary that says which layer owns an item, `read_by=` for who else derives
from a value, per-kind losslessness, the two meanings of *complete*, and the
schema fingerprint.

**Vocabulary a reader may find in old code or old folders:**
`<label>.fdf.template`, `render_template(deck_text, config)`, `read_template`,
`emit_item_block`, `BLOCK_ITEM`, `MARKER_RE`, `_anchor_token`, `_payload_for`,
and the `(molbuilder: …)` prose-anchor convention that `kind=` replaced.

---

### 3.7 The template's item blocks — one file that is both the reference and the source

*Specified by the user, 2026-08-07.* The **template** (`engines/stages.md § 4`)
is the science backbone a generating tab writes: everything a script owns, with
the value the user set or the default they did not touch. This is its format.

**The rule in one line: every item appears exactly as it will be copied, wrapped
in markers, with its explanation beside it.**

```fdf
# === molbuilder item mesh_cutoff BEGIN ===
#   field mesh_cutoff  anchor=MeshCutoff  type=float  unit=Ry
#                      range=[50,2000]  default=300.0  group=stage
#   Mesh cutoff — the real-space integration grid, in Ry.  Higher is finer
#   and slower; convergence is checked, not assumed.
#   Tier ladder: 150 screening · 300 publishable · 500 tight.
MeshCutoff   300.0 Ry
# === molbuilder item mesh_cutoff END ===
```

#### What the template is for, who uses it, and what complete means

*Written down 2026-08-11, because "is the template lossless?" turned out to be
unanswerable until these three were stated.*

**Its function: the calculation's own catalogue.** Every item a script owns,
each with the value in force, the declaration that describes it, and the prose
that explains it. **One file that is both the reference and the source** — the
reference because you can read what the calculation is, the source because the
payload lines *are* the deck text.

**Its two consumers, and they ask different things of it:**

| consumer | what it needs from the template |
|---|---|
| **a generating surface** — building `task.json` without asking a server | the **declarations**: `type` and `choices` to pick a control, `range`/`unit` to bound it, `default` to show what untouched means, `group` to decide whether *vary per stage* starts ticked |
| **`prep`** — turning the description into each stage's deck, on the target | the **payloads and anchors**: a stage's overrides are substituted at their anchors, and what comes out is that stage's deck |

**The second consumer is what makes anchors load-bearing.** `anchor=` is not
documentation — it is how an override finds its site, **by anchor and not by
line number**, so a template a person has edited still substitutes correctly.

#### The marking format: one key, a closed vocabulary, inside the comment

**Why the file cannot be JSON or YAML, and it is not a preference.** The rule
above — *every item appears exactly as it will be copied* — makes the payload
lines the engine's own input. The template **is** the deck's source, so the file
must stay in the engine's format and every marking must live where that format
puts comments. A `.fdf` in JSON would be a description of a deck rather than a
deck, and `prep` would be transcribing instead of substituting.

**Four formats were weighed. The scores are not close, and one axis turns out
not to discriminate at all.**

| | **correctness** | **performance** | **readability** |
|---|---|---|---|
| **TOML inline table in engine comments** *(chosen)* | one file cannot drift from itself, **and** a specified grammar with a stdlib parser | irrelevant | highest — the declaration sits directly above the line it produces, and reads as prose does |
| **a bespoke `key=value` grammar** | one file, but a regex of ours: no spec, and it cannot express a string with spaces — which is how the kind ended up as prose | irrelevant | high |
| **JSON inside the comment** | as strong as TOML on parsing | irrelevant | poor — quote-dense, and a person editing it breaks it silently |
| **a sidecar `template.json`** | **worst: two files, and they can disagree** | irrelevant | fine, but you must read both to know one thing |
| **YAML in the comment** | weakest: `no` parses as false, versions differ | irrelevant | good |

**Correctness decides it, and not the way "easier to parse" suggests.** The
failure mode this format must prevent is not a parse error — it is a template
that *parses cleanly and describes a different deck than the one beside it*. A
sidecar makes that failure structural: the description and the payload live in
different files and nothing keeps them in step. Everything else here is a
recoverable syntax problem; that one is silent and produces a wrong
calculation. **One file cannot drift from itself** is worth more than any
parser's strictness.

**Between the two spec'd formats, readability decides**, and TOML wins it:
`kind="engine", anchor="MeshCutoff"` reads the way the rest of the block reads,
where the JSON equivalent is a wall of braces and quotes in a file
`project-layout.md` expects **a person to edit**. Rolling our own grammar was
considered and rejected on the same axis it appears to win: it is only simpler
until an item needs a string with a space in it, and then the prose goes
somewhere it cannot be parsed — which is the defect this section exists to
fix.

**Performance is not a discriminator and it would be dishonest to claim it
is.** A template is a few hundred lines, read once per `prep`. Every candidate
here costs microseconds. The only version of this question with a real answer
is a surface listing many calculations at once — and even there the cost is
reading files, not parsing them.

**Readability is the tiebreak, and it is load-bearing rather than cosmetic**:
§ 3.7's *"one file that is both the reference and the source"* only holds if a
person can read the reference. The declaration sitting immediately above the
line it produces is what lets you check a template by eye — the same property
that makes the payload copyable makes the file reviewable.

**Inside the comment: a TOML inline table, parsed by `tomllib`.** The
declaration is a mature, specified format with a standard-library parser — not
a grammar of ours:

```fdf
#   item mesh_cutoff = { kind="engine", anchor="MeshCutoff", type="float",
#                        unit="Ry", range=[50,2000], default=300.0, group="stage" }
```

**§ 3.3's existing grammar is already most of the way there**, which is why this
is a small move rather than a new notation: `range=[50,2000]` is a TOML array
and `default=300.0` a TOML float today. What TOML adds is a **specification**
instead of a regex, and the two cases the bespoke form cannot express —
**strings containing spaces** (a `help` line, an enum label) and **lists**
(`expands=["ChemicalSpeciesLabel"]`). Those are exactly the cases that pushed
the kind into prose in the first place.

> **The one real cost, stated plainly.** `tomllib` is standard library from
> **Python 3.11**, and `pyproject.toml` declares `requires-python = ">=3.9"`.
> So this needs either the floor raised to 3.11 — already a tracked item — or
> the `tomli` backport, which is the same parser under its pre-stdlib name.
> **That is a dependency decision, and it is the only thing standing between
> this format and being written down as settled.**

**What was genuinely unparseable was the KIND**, because it was written as
English inside the anchor: `anchor=(molbuilder: ChemicalSpeciesLabel block
ordering)`. A layer had to read prose to learn whose parameter it was. So the
marking is **one more key with a closed vocabulary**:

```fdf
# === molbuilder item mesh_cutoff BEGIN ===
#   field mesh_cutoff  kind=engine  anchor=MeshCutoff  type=float  unit=Ry
#                      range=[50,2000]  default=300.0  group=stage
#   Mesh cutoff — the real-space integration grid, in Ry.
MeshCutoff   300.0 Ry
# === molbuilder item mesh_cutoff END ===

# === molbuilder item species_order BEGIN ===
#   field species_order  kind=deck  expands=ChemicalSpeciesLabel  type=str
#   The order species are declared in. A .XV read against a different order
#   lands every coordinate on the wrong atom (run-identity.md § 4).
%block ChemicalSpeciesLabel
  1  6  C
%endblock ChemicalSpeciesLabel
# === molbuilder item species_order END ===

# === molbuilder item continue_retries BEGIN ===
#   field continue_retries  kind=wrapper  type=int  range=[1,5]  default=1
#   How many times the run wrapper retries a stage that did not converge.
#   No payload: this item shapes the wrapper, not the deck.
# === molbuilder item continue_retries END ===
```

| `kind=` | the item is | payload | who reads it |
|---|---|---|---|
| `engine` | one of the engine's own keywords | the keyword line, copied verbatim | the deck |
| `deck` | molbuilder's, but it shapes the deck — by expanding to keywords or ordering a block | the text it produces; `expands=` names the keywords | the deck |
| `wrapper` | shapes the run script | none | `runwrap` |
| `produce` | shapes what the produce step does | none | the producer |
| `monitor` | shapes what the monitor writes | none | `mb_monitor.py` |

**Text and choices raise the case the kinds alone cannot carry.** A number is
usually consumed where it lands. A **choice** often is not:

```fdf
# === molbuilder item diag_algorithm BEGIN ===
#   item diag_algorithm = { kind="engine", anchor="Diag.Algorithm", type="enum",
#                           choices=["ScaLAPACK","ELPA-1STAGE","ELPA-2STAGE"],
#                           default="ScaLAPACK", read_by=["wrapper"] }
#   Which eigensolver SIESTA uses.  ELPA also decides WHICH ENVIRONMENT the
#   wrapper activates: an ELPA or GPU deck routes to molbuilder-siesta-gpu
#   (§ 2.6), so this value leaves the deck and reaches the launch.
Diag.Algorithm   ScaLAPACK
# === molbuilder item diag_algorithm END ===
```

**`kind=` says who owns the payload. `read_by=` says who else derives from the
value.** They are different questions and one key cannot answer both — this item
is unambiguously the engine's, *and* the wrapper acts on it.

**Why that key earns its place rather than documenting a curiosity.** Today the
wrapper finds this out by **reading the deck text and looking for ELPA**
(`runwrap._fdf_requests_gpu`). That is a layer re-deriving an answer another
layer already holds — the one habit this architecture exists to remove
(`execution/architecture.md` § 1). With `read_by`, the wrapper is *told* which
items it depends on instead of pattern-matching someone else's artifact, and a
new engine declares its own without anyone editing `runwrap`.

**It also explains an ordering the contract already forces.** `project-layout.md`
§ 2.3.1 renders the deck (step 3) before the wrapper (step 4) and calls the
order forced rather than chosen. `read_by=["wrapper"]` is that dependency,
written on the item that creates it: the wrapper cannot be written until every
value it reads is fixed.

**Free text** (`type="str"`) needs nothing further — TOML quotes it, including
spaces and commas, which is the case the bespoke grammar could not hold.

**Three properties make it universal:**

- **A layer filters on one token.** *"Emit every `kind=engine` and `kind=deck`
  item; ignore the rest"* is the SIESTA producer's whole rule, and PySCF's, and
  any later engine's. No field lists, no prose.
- **The vocabulary is closed**, so an unknown `kind=` is an error a reader can
  report rather than something it silently drops.
- **`anchor` goes back to being an anchor.** It is a keyword or it is absent —
  never a sentence. Values become tokens, which is what keeps the `key=value`
  grammar sufficient and is why a richer format is not needed.

**And it states losslessness mechanically:** the items that must round-trip
exactly are `kind ∈ {engine, deck}`. The rest must be carried and legible.

**The template says what KIND each item is, and the anchor is where it says
it.** Not every item is the engine's: some are molbuilder's own, and a layer
must be able to tell which without carrying a list of field names.

| the anchor reads | the item is | who acts on it |
|---|---|---|
| a bare engine keyword — `MeshCutoff`, `WriteForces`, `PAO.BasisSize` | **the engine's** | the deck: the payload is copied in verbatim |
| `(molbuilder: …)` and the prose says it shapes the deck — *"expands to DM.UseSaveDM / MD.UseSaveXV / MD.UseSaveCG"*, *"ChemicalSpeciesLabel block ordering"*, *"pre-emission positioning"* | **molbuilder's, deck-shaping** | the deck, but through molbuilder's own rule rather than one keyword |
| `(molbuilder: …)` and the prose says it shapes something else — *"triggers .psml staging"*, *"baked into the run wrapper at install time"* | **molbuilder's, not the deck's** | whichever layer the prose names — the wrapper, the produce step, the monitor |

**This is what lets a producer refuse cleanly.** A SIESTA producer emits engine
anchors and the keywords molbuilder-owned items expand to, **and must not try to
emit a `(molbuilder: …)` item as a keyword** — SIESTA would not understand it.
An item it cannot place is not an error in the template; it is an item that
belongs to a different layer, and the block says so on its own face.

**So a layer filters by reading the block, never by knowing a field list.**
That is what *self-explanatory* buys: a new engine, a new surface or a new
environment reader can be written against the template alone.

**So "complete" means two different things, and both must hold:**

- **Complete for the surface** — every item a script owns has a block. A field
  the config carries and the template omits is a control the surface cannot
  offer and a value the user cannot see.
- **Complete for `prep`** — substituting a stage's overrides into the template
  yields **exactly** the deck that stage would otherwise have been rendered
  with. This is the testable form: *render a stage's deck both ways and compare
  the text.*

**Losslessness is per kind, not per file.** The **deck-affecting** items — the
engine's, plus molbuilder's deck-shaping ones — must round-trip exactly, because
the deck is the calculation. The rest must be **carried and legible**: a
`(molbuilder: …)` item that shapes the wrapper is not lost if the deck does not
contain it, so long as the layer that owns it can still read it. Demanding one
standard of both is how the question became unanswerable.

**The second implies the first but is not implied by it**, which is why the
weaker one is not enough on its own: a template can list every field and still
substitute to a different deck if an item's payload is not the text that lands.

> **Measured 2026-08-11:** `SiestaConfig` has 45 fields and 39 have blocks.
> `species_order`, `write_coor_step`, `write_forces` and `write_molwatch_log`
> are read by the deck renderer and have none — so the template is complete for
> neither consumer today. `copy_psml` is correctly absent (it is a produce-time
> file copy, not deck content), and `stage` should not be a config field at all
> (`engines/stages.md` § 1.1).

**Each block carries a `field` declaration line and then prose**, and the
declaration is **the grammar § 3.3 already defines** — `field <name>
anchor=… type=… range=[a,b] unit=… default=…`, `type ∈ {int, float, str, pow2,
enum}` — extended with `group=` (the `workflow_group`) and `choices=` for enums.
Not a parallel notation: the same shape, in the same file, parsed the same way.

**That declaration is what makes the template enough on its own.** A surface
holding this file needs nothing else to work:

| From the declaration | What it lets a surface do |
|---|---|
| `type`, `choices` | pick the control — number box, dropdown, checkbox |
| `range`, `unit` | bound it and label it |
| `default` | show what "untouched" means, and **tell whether the payload is the user's or the default** — they are the same value or they are not, so no extra marker is needed |
| `group` | decide whether the *vary per stage* box starts ticked |
| `anchor` | **find the substitution site** when a stage overrides the item — anchor-based, not line-number-based, so it survives layout drift above it (§ 3.3's own rationale) |

**So constructing `task.json` from the UI needs only the template.** When a user
ticks an item, the tab already knows its type, its bounds and its default from
the block itself — enough to render the per-stage cells and to validate what is
typed into them, without asking the server what the field is. That is what makes
the whole package portable in the sense `project-layout.md § 2.1` means: the
calculation travels with its own catalogue.

Four properties, and each buys something specific:

1. **The payload is what lands in the final deck, and that is a *checked*
   property.** *(Decided 2026-08-07 — this rule was written as "producing that
   deck is a scan and a copy, never a re-render", and three fields of the
   shipped schema cannot be served that way.)*

   `prep` rebuilds an ordinary config from the blocks, resolves the stage
   (`stages.md § 4`), and renders through the **same emitter every other deck
   goes through**. A test then asserts that for every item no stage overrode,
   the rendered line is byte-identical to the template's payload. So *a value
   cannot change shape between what a person read and what the engine got*
   survives as the guarantee — it is enforced by a guard rather than by the
   copy being literal.

   > **Why the literal copy could not hold.** A stage that overrides
   > `relax_type` from `CG` to `Verlet` moves the step budget's site from
   > `MD.NumCGsteps` to `MD.FinalTimeStep` — the *anchor itself* is chosen by
   > another field's value, so there is no fixed site to substitute at.
   > `spin_total` writes **two** lines (`Spin.Fix` + `Spin.Total`) from one
   > field. And ten fields write **no** line at their defaults. Re-rendering
   > handles all three for free; substitution handles none of them.
   >
   > The alternative considered and rejected was to allow only
   > single-anchor, always-emitted fields to be varied — which would make
   > *which settings may vary* a fixed list again, the exact arrow § 1.2 of
   > [`engines/stages.md`](?doc=engines/stages.md) exists to reverse.

   **`anchor=` therefore stops being load-bearing.** It still says where the
   value lands, which is worth knowing and is what BENCH-MARKS uses it for
   (§ 3.3); nothing reads it to produce a deck.
2. **The markers carry the field's name, and the declaration carries its
   value.** That is what lets `prep` walk the file and rebuild an ordinary
   `SiestaConfig` — *without an fdf parser*. **This is the whole reason the
   design works**: nothing in molbuilder can read an `.fdf` back into a config,
   and with named blocks nothing needs to.

   The declaration gains **`value=`** beside `default=`, and the reader takes
   the value from there rather than from the payload. Same grammar, one more
   key — and it is what makes the read total: a payload can be absent (the
   field emits no line), several lines (`spin_total`), or a `%block`, and none
   of those change how the value is read. `default=` stays, because the pair
   is what tells a surface whether the user set this or left it alone.
3. **The block holds what we know about the item** — what it is, what it is
   validated against, how the engine uses it, any hint worth having. It is
   generated from the field's own metadata (`web/form-schema.md § 1a`:
   `help`, `range`, `unit`, `choices`, `engine_key`, `workflow_group`), so the
   documentation and the form are the same source and cannot drift.
4. **Every allowed, validated item has a place in the file** — not only the ones
   a user touched. The template is the engine's whole surface, instantiated.

   > **This is the premise, not an aspiration, and it settles a class of
   > question before it is asked.** The template is built from what molbuilder
   > *knows*: every field the engine's config declares, every one validated, each
   > with what we have learned about it. There is no "what about a keyword we do
   > not model" case to design for — a keyword molbuilder does not model is
   > **work not done yet**, and the answer is to model it, not to invent a slot
   > for it. § 3.5's USER-CUSTOM is not that slot either: it is a zone copied
   > **byte-for-byte and never validated**, for a user's own text.



**So one artifact serves four readers.** A person opens it and learns the
calculation *and* the reasoning. The UI renders it. `prep` extracts the deck.
The validator gets a real config out of it. That is why it is worth being a
real, readable `.fdf` rather than a serialised blob — which was the alternative,
and it would have been none of those four things but the last.

> **The marker convention is the one that already ships**, not a new one:
> `# === molbuilder <name> BEGIN ===` / `# === molbuilder <name> END ===`
> (`script_emit.py`), the same shape as HEADER, PROVENANCE, BENCH-MARKS and
> USER-CUSTOM above. The comment character is the engine's — `#` for `.fdf` and
> `.py`; an engine whose comment is `!` uses `!`.
>
> **Multi-line items come free.** A `%block ChemicalSpeciesLabel` or a coordinate
> block is several lines, and the markers delimit it exactly as they delimit a
> one-liner. A format that keyed on "one line per setting" could not have carried
> those at all.

**Both of the questions this format was leaving open are answered above**
(2026-08-07): `engine_key` is not an anchor for every field, and ten fields emit
no line at their defaults — and re-rendering makes both harmless, which is why
the rule changed rather than the schema.

**How a stage's override lands:** `prep` reads every block's `value=` into an
ordinary config, applies the stage's `overrides` on top (`stages.md § 4`), and
renders. A field the stage did not name keeps the template's value — that is
`overrides ⊆ varies` seen on disk, the quiet cell in the table.

> **Two things this format leaves to be decided when it is built**, recorded
> rather than guessed:
>
> - **Items whose default is *derived*, not literal.** Some defaults come from
>   the compute resources — `BlockSize` from the rank count is the example — and
>   the rank count is not known when the template is written. **These still get a
>   block**, and the rule is (user, 2026-08-07): **an explicit user setting is
>   honoured; otherwise the value is derived at generation, and at generation
>   time both are available.** So the block's declaration says the default is
>   derived rather than naming a number, and the payload is either what the user
>   set or supplied by `prep`. *How* that is spelled — a `default=derived` marker
>   with the payload line absent until `prep` writes it, or a placeholder payload
>   — is the remaining detail, and it is small.
>
>   ⚠ **`Diag.Algorithm` is not one of these**, and it was wrong to file it here.
>   It is **an ordinary explicit option**: a user chooses it, it gets a plain
>   block with a plain payload, and nothing derives it. **Whether the engine can
>   deliver the choice is the engine's business, not the generator's** — an
>   `.fdf` asking for an ELPA solver that the build does not have fails when
>   SIESTA runs, and that is the correct place for it to fail. The generator does
>   not check, and does not need to.
> - **A hand-edited payload that no longer matches its block name.** The file is
>   meant to be edited; someone will change `MeshCutoff 300.0` to `400.0` by
>   hand. Reading the value back out is what makes that work — but it also means
>   the payload, not the metadata, is the authority. Worth stating explicitly.
> - ~~**A user-set value that equals the default** is indistinguishable from an
>   untouched one~~ — **closed, no consequence** (user, 2026-08-07). **Every item
>   is explicitly instantiated**: § 7 of [`engines/stages.md`](?doc=engines/stages.md)
>   already requires *every value the description determined, written rather than
>   left to an engine default*, and § 3.7 extends that to the whole surface — every
>   allowed item has a block, with a payload. So a deliberate `300.0` and an
>   untouched `300.0` produce **the same bytes** in the deck. There is nothing for
>   the distinction to change.
>
>   **And the payload being the authority makes that hold over time, too.** If a
>   molbuilder release changes a field's default, an existing calculation
>   regenerates from *its template's payload*, not from the new schema — so the
>   number a user saw is the number they keep. A design where the deck was rebuilt
>   from defaults would have made this distinction load-bearing and dangerous;
>   this one makes it moot.
>
> **And one relationship to keep straight:** these item blocks and BENCH-MARKS
> (§ 3.3) share a grammar and answer different questions. The item blocks live in
> the **template** and declare **every** field. BENCH-MARKS lives in a
> **generated deck** and declares the subset a *tool* may override. **Both must be
> emitted from the same field metadata** (`web/form-schema.md § 1a`) — two hand-
> maintained copies of `default=` would drift, and the drift would be silent.

