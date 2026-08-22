"""RETIRED 2026-08-22 (U6 close) — the stage-table widget's chromium suite.

All seventeen tests here exercised the ``stage-table`` form-schema kind
against a hand-built synthetic schema.  No producer can emit that kind:
the Python half (``_stagespec_to_field_schemas`` walking
``List[StageSpec]``) died when `stages.md` § 1.1a made a PySCF ladder N
decks (2026-08-18, StageSpec deleted with it), and the JS renderer —
recorded as reached-by-nothing in ``tests/test_stage_vocabulary.py`` —
retired 2026-08-22 with the user's cleanup ask.  A suite green against
a widget no page can reach is exactly the "139 passing tests said
nothing" class ``test_molbuilder_e2e.py``'s header documents.

The live stage table is Task setup's own, hand-rolled over ``task.json``
and pinned in ``tests/test_task_setup_tab.py``.
"""
