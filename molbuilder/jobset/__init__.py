"""The ``jobset`` framework — a declarative model of a set of related
jobs that share a package, with engine-agnostic prep / launch / status /
summarize verbs (docs/execution/job-system.md).

**Floor 3 is DERIVED, not built by producers**: nothing constructs a
JobSet ahead of time and hands it down.  ``prep`` derives it from a
described calculation on the machine that will run it — so the ranks,
resources and paths in it are the ones that machine actually has.

**This package re-exports nothing**: the floors each module sits on are
`execution/architecture.md` § 2.1, and a caller imports the module that
owns what it wants.
"""
