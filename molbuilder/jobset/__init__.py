"""The ``jobset`` framework — a declarative model of a set of related
jobs that share a package, with engine-agnostic materialize / submit /
plan engines (docs/protocols/staged-execution.md § 2.1).

Producers (bench sweep, SIESTA stage ladder) build a :class:`JobSet`;
the engines consume it.  Import-light by design (model is pure stdlib)
so it runs on a compute target alongside the other on-target tools.
"""
from .model import Carry, Job, JobSet, Resources, SCHEMA
from .materialize import materialize, job_dir_name, relink
from .prep import prep_jobset, PrepError
from .plan import render_plan
from .submit import submit_jobset, JobResult, SubmitError

__all__ = [
    "Carry", "Job", "JobSet", "Resources", "SCHEMA",
    "materialize", "job_dir_name", "relink",
    "prep_jobset", "PrepError", "render_plan",
    "submit_jobset", "JobResult", "SubmitError",
]
