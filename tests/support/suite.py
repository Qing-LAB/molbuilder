"""A small suite of test files run by the suite's own rules -- for a test of
the harness itself (how a run is spread, what is skipped and why): its
``conftest.py`` and ``pyproject.toml`` are the real suite's (links), so the
rules it is run by are the suite's own, and `support` is found beside the
real ``conftest.py`` (`tests/conftest.py`'s head).
"""
from __future__ import annotations

from pathlib import Path

from tools.testrun import REPO


def linked(where: Path, files: dict) -> Path:
    """The suite at ``where``: each ``{name: text}`` of ``files`` written as
    a test file beside the real suite's ``conftest.py`` and
    ``pyproject.toml``."""
    where.mkdir()
    (where / "conftest.py").symlink_to(Path(REPO) / "tests" / "conftest.py")
    (where / "pyproject.toml").symlink_to(Path(REPO) / "pyproject.toml")
    for name, text in files.items():
        (where / name).write_text(text)
    return where
