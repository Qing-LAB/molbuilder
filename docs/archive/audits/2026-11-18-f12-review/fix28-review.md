# F12 fix 28 (the motion toolkit's baseline, clean, 2.85.1) — the review

**Verdict: closed with 2.85.1.** A clean-up release after the key milestone of 2.85.0 (one layer: every mode's state
a chain or a distribution bound to the mode). The user asked for a baseline review before moving on: *"your next
review should include detection of obsolete design, compatibility that is not necessary anymore, redundant code,
dupilcated code, and any inconsistency by checking against contract. thiese should be holistic full text review so we
have a clean baeeline to move forward from a key milestone"*. Six readers read the whole area — the generic tools, the
molecule layer, frame sets and their doors, transport's side of a frame set, the documents as one body, and across the
area — each by those five things. Their findings were sorted by the user into three releases: this one, the
clean-up; 2.86.0, the weights (required, checked for normalization and refused otherwise, a function to assign them,
the snapshot analyses on the saved weights, a person's per-frame API); 2.87.0, the older files dropped.

The lessons: **transport does the transport calculation, and that's it** — the five assumptions it says beside an
average hold for any frames; what holds for frames of a vibration is the vibration set's own (science § 7, split by
owner). **One home for each fact, served**: the oscillator's mechanics in `oscillator` only; a state's and its
frames' words made once by the toolkit and printed by the command line and the page; the frames' average conductance
in `transport.conductance` only. **A hand-built value is checked as the maker checks it**: a chain built from steps
by hand, a chain's record and a frame set's saved state are refused or said by name, never computed as something
else or stopped by a Python error. **The science chapter names its calls**: every step ends with the call a script
reaches it by (the user: *"provide reference to api that each process can use to perform the calculation explain in
each step"*).

## The user's decisions (2026-11-26)

- **Frame sets written before 2.81.0**: *"Drop it"* — no handling of their own (their run not reopened from
  `info.vibration`, no harmonic sentence added on re-averaging); read as any set that records no state.
- **Transport's assumptions**: *"Split by owner"* — transport's five for any frames; the vibration's three sentences
  the end of every vibration set's `assumptions` row.
- **Older files** (results before schema 15, spectra data before @11, sidecars before v17, slot-provenance @1):
  *"Drop all as asked"*, in 2.87.0.
- **Weights**: *"you should not assume equal weight for all frames, weight is explicitly calculated and saved with
  multiframe data, and is required, and will be checked for normalization"*; a set off 1 *"Refuse"*; *"api should
  allow user to take results from individual frames and assign its own weight as a post-processing and analysis"* —
  2.86.0.
- **The rest of the baseline**: *"Yes, apply them"*.

## What was done

**The science chapter**: a *Computed by* block in each step's section (49) — the call a script reaches it by, a run
example, the manual's entry; § 8 completed (what a set records and how it is read back); § 7 split by owner.
**Transport's assumptions**: five for any frames (`frame_average.ASSUMPTIONS`); the vibration's three sentences end every
vibration set's `assumptions` row (`vibration_frames._FRAMES_OF_A_VIBRATION`). **The oscillator's mechanics in one home**,
`oscillator` — the seven names `vibration` handed on removed, every reader importing `oscillator`. **A chain built by
hand checked as its steps check** (`oscillator_state._checked_values`, `_start_step`, `_checked_steps`), a chain's record
the same way; `Chain.with_temperature`, `words_with_source` — the one wording of a temperature taken. **A frame set read
back**: one structure refused by name; a set that records no state read whole, its moves from its geometries, its run
not reopened; a recorded state or motion that does not bind again said, read without it. **Frame sets written before
2.81.0** get no handling of their own (`INFO_KEY`, `_OLDER_NAMES`, `assumptions_of_set` removed). **The doors' words
served once** — what a state holds and each frame's line (`ModeFrames.holds`, `frame_lines`) — printed by the command
line and the spectrum tab; *Write frame set…* announces the folder by its full path. **Task setup** says a cited pair
changed since its first prep. **Duplicates folded**: the frames' average conductance (`transport.conductance.
frames_average`), one normal distribution function, one list of the distribution families, one count's words.

## Round 1 — three readers (documents 12, code 7, tests 9 + 1 for the person): 29

**The user's, two choices.** Task setup drew the cited pair as it is now while saying *the frames shown are the ones
the calculation ran* — *"Show the frames it ran"*: once the calculation's first prep composed the junction, Task setup
draws the composed junction's frames, as the Results tab does (`compose.cited_frames(citation, tree_root, folder)`).
A thermal start built by hand naming both a temperature and a mean was accepted and its mean silently replaced when
placed — asked again with an example after *"i don't understand your jargon. without example this means nothing for
me to comment"*: *"Refuse it, like the normal way"*; a placed thermal start now holds only its mean, the temperature
it was placed at kept on the state (`OscillatorState.temperature_k`), and "no quantum to take" judged on the start as
typed (a 3000 cm⁻¹ mode at 4 K holds a mean that prints as 0 but is not 0). The rest, *"Yes, apply them"*: the two
e2e expectations of the command line's new *holds* line; the reader's message for a set of 2.81–2.84; the contract no
longer crediting `vibration` with the oscillator's mechanics; the assumptions rows' descriptions with the vibration
sentences; § 5's *does not bind again* clause; § 8's test map; the oscillator's readers in one list; science § 2b.3's
chain; normal-modes § 4c.8 pointing to science § 7; code leftovers and notes; the missing tests (a recorded state that
does not bind again — on 2.85.0 a misspelled key stopped the read with a `TypeError`; `FrameRows.meaning`; the count
words and a one-level list's words; `with_temperature`'s refusals; a hand-built event holding a value;
`bose_einstein_temperature` on its one home). **One gap recorded**: Task setup's *The citation* row, for a citation
that does not read, has no browser test; its drawing code is unchanged here — for R4 (pages and tests).

## The reader of the fixes: 5 findings

None broke a road; every reader of a worked-out state answered as 2.85.0 did, except the release's own changes; the
new tests failed on 2.85.0 where they guard a fix, and nine mutants were caught. Applied: the manual's lines on what a
worked-out state stores (a thermal start its mean, the state its temperature) — first brought to the user as a choice,
which it was not: *"i don't understand this. this sound like an invented case again"*; the case was a reviewer's chain of
calls no script makes, and the manual is made true to the code (the rule saved: a reader's probe of the API is no
question). Then, the user's *"Yes, apply them"*: the API page, the folder answer's note and the Task setup page saying
Task setup shows the frames the calculation ran; a snapshot set read with its vibration never said to be read without
it, tested; the start as typed worked out in one place and kept private; the architecture map pointing to the one list
of the oscillator's readers.

## What `level(n)` and `.add_quantum()` are (the user, 2026-11-26)

Shown the three-step script — the modes chosen, thermal equilibrium, one mode excited — the user asked what
`add_quantum()` does: *"it does not sound like add_quantum is an excitation by a single photon, it sounds more like an
arbitrary shift of all thermally equilibrium state up"*. It is one photon: of the sample before, the molecules that
absorbed one, each exactly one level up — a harmonic mode absorbs from level $n$ $n+1$ times as readily as from level
0, the jump's strength $\lvert\langle n+1\rvert Q\lvert n\rangle\rvert^2$, so the absorbers held more than the
average before. On the modes the user works with — *"we never, almost never, work with soft modes, that would be
inaccurate, error prone, and just arbitrary"* — it is `level(1)` within a few percent. The user kept the names
(*"Keep add_quantum / remove_quantum"*) and asked the documents to say it: the contract's § 2 (*A level, or a quantum
added?*, the stiff modes' table), § 3.1's example, the manual's starts and events, science § 2e.2 (where the $n+1$ comes
from, an event keeping the molecules it happened to, the stiff modes' table computed in the chapter's examples). On the
user's *"its note/meta data then should contain initial thermal status"*: a set already records the start and its
temperature (`info.state`, the `temperature_k` row) and the start's own mean (the `assumptions` row); the
`mean_quanta` row now says it is the mean after every step.

## The e2e run

One run, after the reader of the fixes: `tests/test_vibration_e2e.py`, `tests/test_siesta_vibration_e2e.py`,
`tests/test_vibration_mechanics_e2e.py` and `tests/test_spectrum_results_e2e.py` — 133 passed, one failed, in 17
minutes; `tests/test_transport_on_a_real_junction_e2e.py` alone — 230 passed in 18 minutes. The one failure was the
test's own: its listener for the folder *Write frame set…* announces, new in this release, was written as a string of
two statements, which Playwright did not run as one — observed on a probed copy: the page announced the folder by its
full path, and a listener attached as a function heard it. Written as a function, the test passed alone.

## Topic → test

| topic | test |
|---|---|
| a chain built by hand checked as its steps check; a temperature fills only an open start; a chain's record refused by name | `tests/test_oscillator_states.py::test_a_chain_built_by_hand_is_checked_as_its_steps_check`, `::test_a_temperature_fills_only_an_open_start`, `::test_a_record_that_makes_no_chain_is_refused_by_name` |
| a placed thermal state judged on its start as typed (a cold, stiff mode) | `tests/test_oscillator_states.py::test_a_quantum_taken_from_a_cold_mode_leaves_twice_its_mean`, `::test_a_quantum_taken_from_a_mode_too_cold_for_a_number_is_its_limit` |
| the oscillator's mechanics on their one home; each family's words | `tests/test_oscillator_states.py::test_a_mean_number_of_quanta_names_its_temperature`, `::test_a_distributions_words_say_its_family`, `::test_a_states_list_is_cut_for_the_largest_rule_the_doors_offer` |
| a frame set read back: one structure refused, a set that records no state read without it (moves from its geometries, no run reopened), a recorded state or motion that does not bind said | `tests/test_vibration_frames.py::test_the_reader_and_the_run_door_refuse_by_name`, `::test_a_mode_set_that_records_no_state_is_read_without_it`, `::test_a_recorded_state_that_does_not_bind_again_is_said` |
| every vibration set's assumptions row ends with the vibration's sentences; transport's five hold for any frames | `tests/test_vibration_frames.py::test_a_recipes_set_is_written_with_the_rows_of_its_kind`, `tests/test_frame_set_meanings.py::test_the_record_reads_the_weights_alone_and_says_its_own_assumptions` |
| a row's meaning as written; a calculation's cited pair changed since | `tests/test_frame_set_meanings.py::test_a_rows_meaning_is_read_as_written`, `::test_a_calculations_cited_pair_changed_since_is_said` |
| the state's and frames' words served once, printed by the command line and the page; the folder announced whole | `tests/test_vibration_mechanics_e2e.py`, `tests/test_spectrum_results_e2e.py` |
| Task setup shows the frames the calculation ran, and the pair changed since | `tests/test_transport_on_a_real_junction_e2e.py::test_a_cited_pair_edited_after_the_calculation_is_said_until_put_back` |
| what `level(n)` and `.add_quantum()` are; the stiff modes' table | `docs/science/vibrational-averaging.md` § 2e.2's examples, run with the chapter |
| end to end | `tests/test_vibration_e2e.py`, `tests/test_siesta_vibration_e2e.py` (the bundles), `tests/test_vibration_mechanics_e2e.py` (the command line), `tests/test_spectrum_results_e2e.py` (the dialog and the page), `tests/test_transport_on_a_real_junction_e2e.py` (a junction's frames on Task setup and the Results tab) |
