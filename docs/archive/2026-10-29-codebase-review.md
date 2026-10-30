# The whole-codebase review of 2026-10-29, verified

**Role:** archive
**Domain:** all

*(The user, 2026-10-29: "don't limit your review to the last work"; "check
again and consolidate".)* Eight readers read every module against
`execution/architecture.md` § 2, § 3.1–§ 3.3 and § 7, `script-preparation.md`
§ 4 and `model/parse.md`; eight more then checked every finding against the
documents and the code, reading both sides — the suspect code and the
existing door it bypasses. **About 355 findings stood; 31 were rejected.**
Each confirmed finding names the existing door it should use (`USE`), the
owner module it belongs in (`MOVE`), or — where nothing answers it today —
what is missing and where it would sit (`NONE-EXISTS`). The plan
([`plans/plan.md`](?doc=plans/plan.md) § 1am) orders the work by the door,
not by the finding; the ids below are what each step closes.

Prefixes: **S** validation, spectra, PySCF · **L** jobset launch and status ·
**P** jobset prep · **W** web and the page · **R** parse · **E** SIESTA,
config, template, tables, structure stack, builders · **T** transport, run
door, paths, projects · **X** run script, monitor, run files, checkpoint, CLI,
config, server, envs, scheduler.

## Rejected, with the reason

| id | why |
|---|---|
| P6 | two renderings of the one `Resources` object, not two sources |
| P15 | answers another question (do two cards share one job) from the owners' values |
| P24 | two questions, one status door |
| L20 | migrate's rewrites name the layout they rewrite — history (§ 3.3) |
| L29 | each site asks `stage_home`, the one door |
| R6, R18, R21, R35 | naming within one owner, no second answer |
| R15 | the run door naming a run to floor 1 is the design (parse.md § 6) |
| R22, R23 | different questions (the digest's version; a data file's claim) |
| R36 | `runtime_info` is the travelling home of the job's variable names |
| R39 | the contract names the transport task reader |
| S13 | the description's own check (template.md § 5.3), named by no engine |
| S16, S57 | package-internal names, one implementation |
| S42 | the reader's check of the words the one done rule writes |
| E — | none rejected; E2 is E1 |
| T8, T36 | an engine's helper called by the kind (§ 3.3), naming only |
| T24 | the run door's own wire forms |
| T28 | `frames_of` is the one door; a re-read is a cost, not a second answer |
| T32 | both loops apply `same_volts`; frame_average stays free of transport |
| W15, W23 | the owners' public doors, underscore naming only |
| W42 | two trees, two questions (a picked path; a written citation) |
| W67 | the widget's sampler and the travelling monitor answer different questions |
| WP5 | the page's one wording of a point, by contract (web/results.md § 0.3) |
| X14 | `monitor.RUN_INDEXES` is run-reports.md § 2.8's named mechanism |
| X18 | the notify verb's own subset of the events |
| X59 | the snapshot and the live resolver are two layers |

## Confirmed, by the door each bypasses

**The engine and kind records** (A18): S1 (no engine `validator` part), S2, S3
(the ECP hint), S4, S5, S8, S9, S10, S14, P1, P8, P22, P25 = R33 (no engine
reading part for *the same calculation*), P31 = E14 (`template.engine_name`),
P32 (no label bound), L1, L2, L3, L5, L12, L16, L19, L21 (no kind `export`
part), L22, R7, R8, R11, R14, T1, T2 = E36 (no pair sort key), T7, T22, E3,
E11, E17, E18 (SIESTA_BENCH_FIELDS), E30, X1, X4, X5, X6, X7, X50 (no
environment field), W3, W13, W14, W33, W43, W44, WP1, WP1+, WP2, WP3, WP4.

**Names and records through their owners** (A14, A11, A16): P2, P11, P26, L6,
L6+, L7, L24, L27, R30, R31, T6, T13, T20, T37, T38, T40, S25, S26, S31, S33,
S35, S35+, S43, S44, S45, S46, E4, E6, E10, E15, E20, E21, E25, E31, E32,
E33, X3, X13, X19, X22, X39, X46, W35, W36, W37, W38, W50, W53, WP6, WP7, WP9.

**One reader and one writer per format**: R3, R4, R5, R12, R16, R17, R19,
R20, R24, R25, R26, R27, R28, R29, S27, S28, S29, S30, S47, T10, T11, T17,
T18, T27, E9, E22, E37a, E39, P28, X8, X16, X23, X40, X43, X45, X48, X61,
X71, X73, X76, W58, W65, W66.

**One home per vocabulary and default**: P4, P16, P17, P23, L28, T15, T16,
T21, T25, T35, T39, S6, S7, S11, S18, S19, S20, S21, S34, S37, S38, S39,
S39+, S40, S41, S48, S51, S53, S54, E16, E19, E23, E26, E41a, E42, R9, X2,
X12, X20 = W51, X29, X36, X41, X42, X51, X52, X55, X57, X57+, X64, X65, X74,
X75, W1, W12, W17, W20, W39, W45, W54, W56, W60, WP8, WP10.

**One implementation per rule**: P3, P10, P13, P14, P18, P19, P29, P30, L4,
L8, L10, L11, L13, L14, L15, L17, L18, L25 = X28, T3, T4, T9, T12, T14, T19,
T26, T29, T30, T31, T31b, T33, T34, S12, S17, S22, S23 = W16, S24, S36, S52 =
E8, S55, S56, E1, E5a, E5b, E28, E35, E38, E40, E41b, X9 = E5, X10, X15, X21,
X24, X25, X26, X27 = W59, X30, X32, X33, X34, X35, X37, X38, X44, X47, X49,
X53, X54, X56, X58, X60, X62, X63, X67, X68, X69, X72, X77, X78, W2, W4, W5,
W6, W7, W8, W9, W10, W18, W19, W21, W24, W25, W26, W27, W28, W29, W31, W32,
W34, W40, W41, W57, W62, W63.

**Floors, objects whole, private names, wrappers** (A7, A8): R1, R2, R34,
R37, P5, P7, P9, P12, P20, P21, P27, E2, E7, E12, E13, E24, E27, E29, E37b,
L9, L23, L26, S15, S49, S58, T5, X11, X31, X66, X70, X79, W11, W22, W30,
W52, W55, W61, W64, the import cycle between `diagnostics` and `envs`.

**A plain bug**: `diagnostics.local_facts` reads `caps` unbound when
`get_capabilities` raises.
