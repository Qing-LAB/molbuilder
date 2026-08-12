"""RETIRED 2026-08-12 (step 6 u5) — this file's nine tests all drove
``render_siesta_stage_fdfs`` / ``stages_to_jobset``, deleted with the
produce-time lifecycle they belonged to (no production caller since the
described route landed).

Where each property lives now:

  * a stage's own values reach ITS deck — test_prep_calculation.py::
    test_the_named_stages_overrides_are_what_got_rendered (rendered by
    `prep`, per element);
  * the BENCH-MARKS block records each deck's own derivation —
    test_prep_calculation.py::test_the_deck_records_the_rank_count_...;
  * per-stage wrapper env routing — `write_run_wrapper` reads the solver
    off the deck it wraps (tests/test_runwrap.py), the strongest route:
    nothing passes the routing along a parameter where it could drop;
  * rank counts ride Job.resources from the allocation, per element —
    test_prep_calculation.py::test_the_allocation_reaches_the_jobs_...
    (`mpi_np` on a PRODUCER's stage was M4's deleted contract; the one
    test here that pinned it had been failing unnoticed since step 3).

Flagged for the post-step-7 holistic review: confirm the per-stage env
routing coverage in test_runwrap.py exercises a two-solver ladder
end-to-end on the described route.
"""
