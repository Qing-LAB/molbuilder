*Verdict: closed in 2.77.8.* The discipline reviews of F12 fix 4, two rounds, read before its release. Every finding was fixed in 2.77.8 except these:
- The transport record reads a point's shared record again, rather than taking status's row. It is the same reader on the same file, so one source, kept.
- Three helpers state one finite-number rule: `frameset.finite_number`, `web/_shared.finite_float` and `spectra.results.finite_or_none`. They predate this release; that duplication is round 2's, by module.
- Two cases have no test:
  - a migration stopped by a write, since no road writes an older calculation in tests;
  - the average's sentence for a frame set whose junction does not read, since the shared runs hold no such frame set.

Kept as the record of what the reviews found.

# F12 fix 4: discipline reviews

## Round 1

1. **A second catch beside the launch's one door for a reader's refusal.** `_points_walk` turned any `ValueError` from a kind's `run_facts` into a `SubmitError`, beside `submit._refusal_kinds` ("one catch, not one per reader"). *Fixed: `_run_facts` raises `PrepError`, listed in `_refusal_kinds`; the walk's catch is removed.*
2. **The door's one sentence was untrue for two of its callers.**
   - "The state saved before them is the way back": migrate saves no state, and a launch again saves none.
   - "N of the plan's writes" also counted folders made, moves and removals.
   *Fixed: each verb's `way_back`; "other change(s) to the folder".*
3. **One door, three ledger shapes.** A prep cut short recorded "question, saved, stopped…, question again, preflight notes, refused". *Fixed (`PrepStopped`).*
4. **A one-way relative import in a module that travels beside a job.** *Fixed.*
5. **A private copy of the finite-number rule** in the reports reader. *Fixed: it asks `frameset.finite_number`.*
6. **The record judged a point a second time**, giving a different state from status. The added `ValueError` in the DOS join was redundant. *Fixed: one state, "finished", not done; the DOS catch reverted.*
7. **The base class was caught where a typed refusal exists** (`continuation`, `record`), and a block missing a key still escaped as `KeyError`. *Fixed: `DeckRecordUnreadable`.*
8. **Stale or untrue text** in `rung.compose_refusals`, `planned` (quoting the old § 5.0 row 6, `CutShort` naming migrate alone), `transport.md` (`_describe_reused`), `checkpointing.md` (`carry_out()` in `send_launch`), `architecture.md` (migrate's line at `CutShort`), job-system § 6.0, `PointsOf.must_hold`, `data_file_bytes`. *Fixed.*
9. **Tests.**
   - The prep case was written as a function in a file that says "prep refuses", while § 5.0's cases are rows ("a case is a row here, never a function").
   - It checked the code's phrase, not the file, the ledger's absence of "refused", or that the stage stayed not prepared.
   - Launch and migrate had no case.
   *Fixed: two rows of `prep_protocol.toml`, with new runner keys `unwritable`, `launch_unwritable` and `ledger_lacks`. Migrate is left: see the verdict.*
10. **The junction's message sent the person to the Molbuilder tab's save**, blessing a hand edit of prep's record. *Fixed: the junction's own way back.*
11. **Two small ones.** `junction_mesh_findings` caught `AttributeError` to mean "none recorded", and migrate imported `ledger` twice. *Fixed.*

## Round 2

1. **The sentence named the system's temporary file, or none on a full disk**, while the ledger named the plan's file. Message and ledger came from two sources. *Fixed: the plan operation's own path.*
2. **A second spelling of the restore command** (`Kept.way_back`, `compose`), beside `identity.checkpoint_as_typed` and `checkpoint_words`. *Fixed: both ask the one spelling.*
3. **The codec's message was cut apart by text surgery** (`split(".  ")`). *Fixed: `workingcopy_structure.geometry_changed_fact`, the codec's own fact-only door, with compose catching `GeometryChangedError`.*
4. **Code names in a person's message** ("(KeyError: 'held')"), and the missing-key branch had no test. *Fixed: "lacks its 'held' entry"; the e2e is two rows (JSON that does not parse, an entry missing), each also through `status` and `summarize task`.*
5. **A redundant catch** in `_gathered_now` once `PrepError` was in `_refusal_kinds`, and that list's docstring was narrower than the code. *Fixed.*
6. **Names and types.**
   - `PrepError`'s docstring said "refused" only.
   - `__all__` lacked the new names.
   - `deck_record.unreadable` shared its name with `continuation.unreadable()`.
   *Fixed: the docstring; `__all__`; renamed `block_unreadable`.*
7. **"Way back" values that were not one.**
   - A launch again said only "nothing was sent".
   - Migrate's typed `<name>.pre-<rule>`, and claimed renames were kept.
   *Fixed.*
8. **Chapter lines.**
   - The web-api prep row lacked the 500.
   - `transport.md` claimed every reader asks the door, though compose's own `load_compose_record` does not, and carried a tombstone ("never the Molbuilder tab's save").
   - The `carry_out` and `MigrationStopped` docstrings were stale.
   *Fixed.*
9. **Docstrings that list their callers were incomplete already**, and the frame-set module's USED-BY did not name the reports reader. *Fixed: the lists dropped for "every reader"; USED-BY names it.*
10. **One fact from two sources** in the record, and a shared-code comment citing transport's chapter. *The comment is fixed. The record's second read is left: see the verdict.*
11. **The harness.** `ledger_lacks` repeated `ledger_holds`' read, and a row named "no refusal" used the runner's `refused` key. *Fixed: one read; the row's name says "holds no refused line".*
