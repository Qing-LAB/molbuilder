"""Every ``__all__`` name resolves — the whole package, one guard.

Three times in two days an edit retired a function and left its name in
``__all__`` (`validation/spectra.py` and `pyscf/vibration_emitters.py`
on 2026-08-21, `spectra/selection.py` on 2026-08-22 — the third was the
same wave that fixed the first two).  ``import *`` raises on such a
module, and nothing suite-wide checked: the one existing ``__all__``
test is deliberately scoped to a single module, and the parse-package
guard covers only ``parse``.  This walks every module under
``molbuilder`` that declares ``__all__`` and asserts each name is an
attribute — the retiring edit's forgotten half fails here by name.

Modules that cannot import in this environment (optional deps) are
skipped by name rather than silently passed.
"""
from __future__ import annotations

import importlib
import pkgutil

import pytest

import molbuilder


def _modules():
    out = []
    for m in pkgutil.walk_packages(molbuilder.__path__,
                                   prefix="molbuilder."):
        if m.name.endswith(".__main__"):
            continue          # importing it RUNS the CLI (SystemExit)
        out.append(m.name)
    return sorted(out)


@pytest.mark.parametrize("name", _modules())
def test_every_dunder_all_name_is_an_attribute(name):
    try:
        mod = importlib.import_module(name)
    except Exception as exc:                    # optional dep absent
        pytest.skip(f"{name} not importable here: {exc}")
    exported = getattr(mod, "__all__", None)
    if exported is None:
        return
    missing = [n for n in exported if not hasattr(mod, n)]
    assert not missing, (
        f"{name}.__all__ names attributes the module does not have: "
        f"{missing} — a retiring edit dropped the function and left "
        f"its export; `from {name} import *` raises.")
