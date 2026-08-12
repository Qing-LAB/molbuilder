"""The ``jobset`` framework — a declarative model of a set of related
jobs that share a package, with engine-agnostic materialize / submit /
plan / status / summarize engines (docs/execution/job-system.md).

**Floor 3 is DERIVED, not built by producers** (R8, 2026-08-12: this
header still said "producers (bench sweep, SIESTA stage ladder) build a
JobSet" — both producers died in the 2026-08-12 fold): ``prep`` derives
the JobSet from a described calculation on the machine that will run it,
and the engines below consume it.  Import-light by design (model is pure
stdlib) so it runs on a compute target alongside the other on-target
tools.
"""
from .agreement import (DeckLaunchMismatch, LaunchAgreement,
                        check_launch_matches_deck, launch_agreement)
from .ledger import LEDGER_FILE, record
from .model import Job, JobSet, Resources, SCHEMA
from .materialize import materialize, job_dir_name, relink
from .prep import prep_jobset, prep_calculation, PrepError
from .plan import render_plan
from .submit import submit_jobset, JobResult, SubmitError
from .runstatus import jobset_status, render_status, StageStatus, JobSetStatus
from .summarize import run_summarize_jobset

__all__ = [
    "Job", "JobSet", "Resources", "SCHEMA",
    "materialize", "job_dir_name", "relink",
    "prep_jobset", "prep_calculation", "PrepError", "render_plan",
    "submit_jobset", "JobResult", "SubmitError",
    "jobset_status", "render_status", "StageStatus", "JobSetStatus",
    "LaunchAgreement", "launch_agreement", "check_launch_matches_deck",
    "DeckLaunchMismatch",
    "record", "LEDGER_FILE",
    "run_summarize_jobset",
]
