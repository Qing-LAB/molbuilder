"""The suite's helpers -- test code, usable ONLY inside a pytest run.

Every helper here writes where the suite's sandbox points it
(`tests/conftest.py`, `docs/process/testing.md` § 2b).  Outside pytest there
is no sandbox: the config folder a helper asks for is the person's own, and a
helper run by hand from a script once wrote the suite's test machine record
into it (2026-10-30), breaking every real run on that machine.

So this package refuses to load unless pytest is running.  pytest sets
``PYTEST_VERSION`` for the length of a run, its workers included; importing
pytest alone does not.
"""
import os

if "PYTEST_VERSION" not in os.environ:
    raise ImportError(
        "tests/ is test code: it runs only inside pytest "
        "(`python tools/testrun.py run <files>`), where the suite's sandbox "
        "keeps every write away from your home, config, state and projects "
        "folders (docs/process/testing.md § 2b) -- never import it into a "
        "script")
