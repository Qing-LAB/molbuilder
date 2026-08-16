"""Canonical tab order — single source of truth.

The 5-tab nav rendered by ``_app_header.html`` and the bare-``/``
landing redirect both derive from this list, so reordering the
tabs (or adding / removing one) is a single-place change.

Each entry carries:

  * ``key``   -- matches ``{% set active_tab = "<key>" %}`` in
                  each tab's page template.
  * ``path``  -- the canonical URL the tab lives at.  The route
                  itself is registered in ``app.py`` (for
                  ``molbuilder``, ``structure-optimization``,
                  ``transport-calculation``) or in the matching
                  blueprint (``spectra.py`` for
                  ``spectrum-calculation``; ``results.py`` for
                  ``results``).  Order in this list is the order
                  shown in the nav; the routes themselves are
                  distributed across modules.
  * ``label`` -- the visible tab name.

The first entry is the canonical landing tab — the bare host
(``/``) 302-redirects to ``TABS[0]["path"]``.

Routes still match the ``path`` field 1:1; tests that exercise
the URLs pull paths from this list rather than hard-coding the
strings, so a reorder lands in ONE place.
"""
from __future__ import annotations

from typing import Final, List, TypedDict


class Tab(TypedDict):
    key:   str
    path:  str
    label: str


TABS: Final[List[Tab]] = [
    {"key": "molbuilder",
     "path": "/molbuilder",
     "label": "Molbuilder"},
    {"key": "structure-optimization",
     "path": "/structure-optimization",
     "label": "Structure optimization"},
    {"key": "spectrum-calculation",
     "path": "/spectrum-calculation",
     "label": "Spectrum calculation"},
    {"key": "transport-calculation",
     "path": "/transport-calculation",
     "label": "Transport calculation"},
    {"key": "job-prep",
     "path": "/job-prep",
     "label": "Job prep"},
    {"key": "results",
     "path": "/results",
     "label": "Results"},
    {"key": "documents",
     "path": "/documents",
     "label": "Documents"},
]


def landing_path() -> str:
    """Return the path the bare host ``/`` redirects to.

    Always the first tab's path; computed from ``TABS`` rather than
    hard-coded so a reorder picks up the new first tab automatically.
    """
    return TABS[0]["path"]
