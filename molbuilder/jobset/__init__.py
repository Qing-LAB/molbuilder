"""The ``jobset`` framework — a declarative model of a set of related
jobs that share a package, with engine-agnostic prep / launch / status /
summarize verbs (docs/execution/job-system.md).

**Floor 3 is DERIVED, not built by producers**: nothing constructs a
JobSet ahead of time and hands it down.  ``prep`` derives it from a
described calculation FOR the machine that will run it, from that machine's
record (``--target``) — prep runs here or there — so the paths in it are that
machine's, and the ranks and resources the ones the job states, checked
against that record (job-system.md § 5.0).

**This package re-exports nothing**: the floors each module sits on are
`execution/architecture.md` § 2.1, and a caller imports the module that
owns what it wants.
"""
