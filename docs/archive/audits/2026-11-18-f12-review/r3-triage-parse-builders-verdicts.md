# R3 triage — R2-7 parse (24) + R2-8 structure/builders (35) against 2.77.16

Fixed (2.77.4): C-1, C-2, C-5, C-8, C-9, C-11, C-15, C-18, T-1, T-10; partly C-4, C-21, T-5. LEAVE: #6, #12, #16, #17, #20, C-17, C-25, #18's version half. R5: #15. R4: T1, T2, T-2..T-9, TestJsonSafe.
Choices (recommendations, no question needed): #5 keep refusing an .FC overflow, unify only the D/d exponent; C-6 keep the MolView single-field cell ops as `block`s; C-4 a labelled O/N gets the same dipole warning (a correction).
DO:
- parse/types #1 parsers build TrajectoryResult directly (M, + T3); #18 one iso_z.
- #2 recorded contract set from the catalogue's citation tag, lazily (M, + T4 retire).
- #3 delete four unread FdfParams fields.
- trajectory_log/keys #4 targets_of(stage=); #13 SCF log column tuple (M) (= R2-6 F15).
- siesta_grammar #5 fortran_float with D/d; #10 FAILED tuple.
- siesta_mdnc #7 one series tuple; dirs/record #8 key tuples; #19 narrow except.
- sidecars/molstruct #9 ENVELOPE_KEYS.
- #11 TBT_CHANNEL_TAGS states the grouping (M) (= R2-5 #6 channels).
- siesta_reader #14 two methods.
- chemistry C-3 symbols from ASE; C-4 one neighbour function (M; visible correction).
- periodicity_gate C-6 one reader per field (M); C-7 Issue.to_json.
- structure C-10, C-24, C-19 concat (M).
- atom_permutation C-12, C-23.
- modify/cell C-13 one plane table (M), C-20 detect_layers, C-22, C-16 thresholds in cell.
- engine_atom_index C-14 (delete pyscf_numbering); residues C-21.
