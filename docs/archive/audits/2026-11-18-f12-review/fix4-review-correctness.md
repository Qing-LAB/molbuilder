*Verdict: closed in 2.77.8.* The correctness reviews of F12 fix 4, two rounds, read before its release. Every finding was fixed in 2.77.8 except these:
- The shared block's way on keeps the deck record's general words. They name restoring a saved state, which works; "prepare the stage again" does not mend a run's deck, and a launch again also re-runs the point.
- `jobset migrate`'s stopped line lists every planned change's reason, as it did before this release.
- The selection PDOS of a junction deleted from the folder now refuses in words. The report still says the junction is composed at prep.

Kept as the record of what the reviews found.

# F12 fix 4: correctness reviews (a failure said in words, never a traceback)

## Round 1

1. **`jobset migrate` named a way back that does not exist.** The door's one closing clause said "the state saved before them is the way back" for every verb. Migrate saves no state: its way back is the `<name>.pre-<rule>` copies (`checkpointing.md`, the table near line 823 and § 9). *Fixed: each verb passes its own `way_back`.*
2. **A prep's failed write wrote "refused" and the question line twice in the ledger.** `prep_bench` sent the error through `_refused`. `_prep_groups` and `_record_the_order` passed `_refuse`, which writes `_asked()` again, the preflight and "refused". This contradicted § 5.0 row 6, "nothing refuses". *Fixed: `PrepStopped`; the bench prep carries its plan out after its refusal catches.*
3. **On Results, a frame set whose junction does not read showed as one structure.** The average's frame count came from `frames_of(base) if junction is not None`, so `nFrames` fell to 1 and frames 1…N-1 could not be reached. *Fixed: the count is read when the junction does not read.*
4. **The new `sent_at` guard could itself raise.** `math.isfinite(10**400)` raises `OverflowError`, and the listener keeps any int. *Fixed: `frameset.finite_number` returns None on overflow; the reader asks it.*
5. **A one-way relative import in `planned.py`**, a module that travels beside a job (`from ..runfiles import LEDGER_FILE`). *Fixed: no import; the sentence makes no claim about the ledger file.*
6. **A shared record missing a key still escaped.** `Contours.of_record` raised `KeyError`. *Fixed: `of_deck` raises `DeckRecordUnreadable`.*
7. **Prep's sentence was weaker than launch's.**
   - It did not name the saved state or the restore command.
   - It said "each written in jobset-decisions.log", which is untrue on a full disk, where `ledger.record` swallows `OSError`.
   - `CutShort.path` for a copy named the destination.
   *Fixed: `Kept.way_back`; the ledger claim is dropped.*
8. **A running point was listed as failed** when its run deck's shared record broke. *Fixed: a point not ended keeps its list.*
9. **The DOS catch was over-wide** (`except (TbtError, ValueError)`). *Fixed: back to `TbtError`.*
10. **The newly shown deck message gives one piece of advice that cannot be followed** ("or prepare the stage again"). *Left: see the verdict.*
11. **Web prep answered 400 for a server fault.** *Fixed: `PrepStopped` is a 500.*
12. **A chapter sentence was untrue**: "every reader asks `compose_refusals`" while `_describe_reused` caught `ComposeError` alone. *Fixed.*

Checked and fine (round 1):
- All five plan sites carry out through the door.
- Each surface catches its verb's error.
- `recorded_junction` raises only `ComposeError`, which every guarded reader turns into its own refusal.
- A junction of `None` downstream raises no new `TypeError`.
- Every caller of `point_done` uses `(ok, why)`.
- The JS branch runs only when `junction_file` is null.

## Round 2

Round 1's findings were closed, with two partial exceptions taken up below (2 and 4).

1. **The sentence named a temporary file, or none.**
   - Every file write goes through `persist.stage_bytes`, whose `mkstemp` raises with the temporary name.
   - A full disk's write error carries no file name.
   *Fixed: `CutShort.path` is the plan operation's own path.*
2. **The average gave a false reason when the junction did not read** ("the frames do not each state a weight"). *Fixed: `frame_average` says the junction holding the frames does not read.*
3. **`point_done` used the record's words for a point in any state.**
   - A failed point lost its engine's words.
   - A running point was told to restore the folder.
   *Fixed: the reader's words only where the run's own would say done.*
4. **A non-finite number in the shared block still escaped** (`int(inf)` raises `OverflowError`). *Fixed.*
5. **The shared-block message was unplain** ("(KeyError: 'held')"), and its "prepare the stage again" does not mend a run's deck. *The words are fixed ("lacks its 'held' entry"); the way on is left, see the verdict.*
6. **Migrate's stopped line lists every planned change**, not only those carried out. This was the same before the release. *Left.*
7. **A junction deleted after its points ran gave a 500 on the selection PDOS.** This was there before the release, next to the finding. *Fixed: refused in words.*

Checked and fine (round 2):
- **Callers of `carry_out_or_say`:** the bench prep's variables are bound on every path that reaches its carry-out; no enclosing catch writes `refused`; the question is written once.
- **The ledger cannot turn the stop into a traceback.**
- **Web prep:** `PrepStopped` answers 500.
- **Launch:** a `PrepError` among `_refusal_kinds` is recorded once.
- **The junction readers** answer as intended: the record gives 200, the PDOS 400, and `summarize`, the data file, launch and the reused page each say it in words.
- **A finished point whose shared record does not read** matches status.
- **`finite_number`** gives None on overflow.
