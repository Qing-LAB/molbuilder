"""A JOB'S PYTHON, stood in by this suite's own -- the one way a test runs a
bundle as a compute node runs it (`configuration.md` § 2.3): an isolated
python (``-I``) that imports the standard library, the packages the bundle's
row allows (`runwrap.Bundle.packages`) with whatever they bring, and the
bundle's members -- and nothing else, molbuilder included.

**Absence enforced, never assumed.** Since 2.77.23 the host env finds this
checkout from any folder (its path file, `ops/deployment.md` § 0), so "molbuilder
is not importable here" no longer holds of the suite's python on a machine that
ran `bootstrap`: an import hook refuses every module the job's python would not
hold.
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Iterable, List, Mapping, Optional, Sequence


def lines(packages: Iterable[str], members: Iterable[str]) -> List[str]:
    """The first lines of a script that make this python refuse, from then
    on, every import but the standard library, ``packages`` -- each imported
    first, with whatever it brings -- and ``members``: a job's python
    carrying only what a bundle's row allows."""
    return [
        "import sys",
        f"for package in {list(packages)!r}:",
        "    __import__(package)",
        # WHAT THE JOB'S PYTHON HOLDS BEFORE THE BUNDLE: its own start-up
        # and the row's packages, with what each brings.
        "allowed = (set(sys.stdlib_module_names) | set(" + repr(list(members))
        + ") | {m.split('.')[0] for m in sys.modules})",
        "class Refuse:",
        "    def find_spec(self, name, path=None, target=None):",
        "        if name.split('.')[0] not in allowed:",
        "            raise ImportError(f'{name} is not the standard "
        "library, a package its row allows, or a member')",
        "sys.meta_path.insert(0, Refuse())",
    ]


def lines_without_molbuilder() -> List[str]:
    """The first lines of a script that make this python refuse molbuilder
    alone -- for a test that runs a job's whole work, not only its load, on
    an env carrying more than the bundle's row (the PySCF env's ``ase``,
    ``scipy``): molbuilder is what a compute node never holds."""
    return [
        "import sys",
        "class RefuseMolbuilder:",
        "    def find_spec(self, name, path=None, target=None):",
        "        if name.split('.')[0] == 'molbuilder':",
        "            raise ImportError('molbuilder is not on a job\\'s python')",
        "sys.meta_path.insert(0, RefuseMolbuilder())",
    ]


def run(script: Sequence[str], *, packages: Iterable[str],
        members: Iterable[str], cwd, timeout: int = 120,
        env: Optional[Mapping[str, str]] = None
        ) -> subprocess.CompletedProcess:
    """``script`` run on the job's python of :func:`lines`, in ``cwd``,
    ``PYTHONPATH`` dropped and ``env``'s variables set (a test's own
    configuration folder)."""
    env = {**{k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
           **(env or {})}
    return subprocess.run(
        [sys.executable, "-I", "-c",
         "\n".join([*lines(packages, members), *script])],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)


def bundle_members(name: str) -> List[str]:
    """The module names bundle ``name`` carries (`runwrap.members`)."""
    from molbuilder.runwrap import members
    return sorted(n[:-len(".py")] for n in members(name))


def bundle_packages(name: str) -> tuple:
    """The packages bundle ``name``'s row allows (`runwrap.Bundle.packages`)."""
    from molbuilder.runwrap import bundle
    return tuple(bundle(name).packages)
