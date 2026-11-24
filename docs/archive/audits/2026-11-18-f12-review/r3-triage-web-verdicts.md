# R3 triage — R2-9 web server (34 findings) against 2.77.16

Fixed: #5 (2.77.7), #8 (2.77.4). LEAVE: #20 (not a copy), #28, #29. R5: #22 (comment). R4: T1–T4.
DO (all drift only, on the road; S unless noted):
- #1 checkpoint routes classify by `checkpoint.InputError` / NoSuchRef 404 / other 500; drop route blank checks and "set up" (Repo.init default; cli.py:3358).
- #2 modify: export `_AXES` + centre tuples; routes and CLI read them.
- #3 modify lattice: reference columns named once; `_lattice_notes` returns Issues.
- #4 selection route: read only labels/residueName/atomName/chainId; Structure fills defaults. (Moving onto the structure envelope = NEEDS USER, design; R4 with MolView.)
- #6 /api/files/write: delete overwrite/auto_rename/merge branch (M) — DELETION, show user first; page side R4.
- #7 jupyter: one notebook_env(caps), one ask_start(port); serve_daemon names START/STOP signals (M).
- #9 + #30 parse.types trajectory_refusal(parser_cls); one directory sentence in cli.
- #10 engine_table carries the form prefix.
- #11 scheduler.record label_of/name_of.
- #12 runtime_config.email_key (casefold) everywhere.
- #13 rate_limit row: type + default in one place.
- #14 monitor exports envelope keys; notify._FIELDS from it + report_fields.NAMES.
- #15 notify listener_path(seg).
- #16 references.cited_keys(text) for doc_indexes and spectra.methods.
- #17 watch `_answer(state, search_dir)`.
- #18 oauth: one record per kind (M).
- #19 drop build_google_entry's dead checks (keep the prompt).
- #21 spectra route: bibliography_keys once after the last rewrite.
- #23 _shared is_loopback(name), request_hostname() (low).
- #24 web/auth exports the callback path (low).
- #25 config_dir exports the shell spelling (low).
- #26 delete blueprints/__init__ watch re-export + stale docstring line (DELETION).
- #27 _shared bounded_int, said_yes (low).
