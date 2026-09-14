"""Routing tables, Capabilities snapshot, singleton lifecycle.

# Retired 2026-09-10: `@dataclass(frozen=True)` is the enforcement, and
# CPython refuses a non-frozen subclass of a frozen one on its own.
# A test that mutates an instance to watch Python raise tests Python.

The module under test exposes three small dicts (``DEFAULT_ENV_NAMES``,
``TOOL_TO_CATEGORY``, ``EXTENSION_TO_CATEGORY``), one frozen dataclass
(:class:`Capabilities`), one probe (:func:`detect`), and four singleton
lifecycle helpers.

Singleton isolation is handled by an autouse fixture in
``tests/conftest.py`` -- every test starts and ends with the snapshot
reset.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from molbuilder import diagnostics
from molbuilder.diagnostics import (Capabilities, DEFAULT_ENV_NAMES,
                                      EXTENSION_TO_CATEGORY,
                                      TOOL_TO_CATEGORY, detect,
                                      get_capabilities, initialize,
                                      set_capabilities)


# --------------------------------------------------------------------- #
#  Routing tables                                                       #
# --------------------------------------------------------------------- #


def test_every_routed_tool_has_an_env_to_be_routed_to():
    """The rule, in place of two tests that re-typed the name list
    (retired 2026-09-14 with the notebook env, which broke them by
    existing -- a list is edited whenever the data is, which is not a
    defect being caught).

    What matters is that the two tables AGREE: a tool routed to a category
    with no env name resolves to nothing, and `run_tool` would dispatch into
    an env that cannot exist.  `molbuilder-jupyternb` is deliberately absent
    from the routing table -- nothing dispatches a command into it; the
    notebook server is launched into it -- so this is one-directional.
    """
    missing = {tool: cat for tool, cat in TOOL_TO_CATEGORY.items()
               if cat not in DEFAULT_ENV_NAMES}
    assert not missing, f"routed to a category with no env name: {missing}"


def test_every_recipe_category_has_an_env_name():
    """The same agreement from the recipe side: a recipe that declares a
    category must be able to name its env."""
    from molbuilder.envs.recipes import BUILTIN_RECIPES
    missing = [r.name for r in BUILTIN_RECIPES
               if r.category is not None and r.category not in DEFAULT_ENV_NAMES]
    assert not missing, f"recipe category names no env: {missing}"
def test_every_tool_routes_to_a_known_category():
    """Each routed tool maps to a category we know how to dispatch."""
    assert set(TOOL_TO_CATEGORY.values()) <= set(DEFAULT_ENV_NAMES)


def test_every_extension_routes_to_a_known_category():
    assert set(EXTENSION_TO_CATEGORY.values()) <= set(DEFAULT_ENV_NAMES)


def test_known_tools_present():
    """Spot-check the README-documented entries -- these are the four-env
    contract surface."""
    assert TOOL_TO_CATEGORY["tleap"]      == "mdtools"
    assert TOOL_TO_CATEGORY["siesta"]     == "siesta"
    # playwright is NOT routed: browser E2E runs under the host env.
    assert "playwright" not in TOOL_TO_CATEGORY
    assert EXTENSION_TO_CATEGORY[".fdf"]  == "siesta"
    assert EXTENSION_TO_CATEGORY[".py"]   == "pyscf"


# --------------------------------------------------------------------- #
#  Capabilities -- pure lookups                                         #
# --------------------------------------------------------------------- #


def _caps(**overrides) -> Capabilities:
    """Synthetic Capabilities for tests that want specific state."""
    defaults: Dict[str, Any] = dict(
        runtime_config = {},
        conda_binary   = "/usr/bin/conda",
        conda_envs     = frozenset(),
    )
    defaults.update(overrides)
    return Capabilities(**defaults)


def test_env_for_category_returns_default():
    caps = _caps()
    assert caps.env_for_category("siesta")  == "molbuilder-siesta"
    assert caps.env_for_category("pyscf")   == "molbuilder-pySCF"
    assert caps.env_for_category("mdtools") == "molbuilder-MDtools"
    # "tests" is not a routed category (browser E2E runs under the host env).
    assert caps.env_for_category("tests")   is None


def test_env_for_category_unknown_returns_none():
    assert _caps().env_for_category("not-a-real-category") is None


def test_env_for_category_honours_config_override():
    caps = _caps(runtime_config={
        "envs": {"siesta": "my-siesta", "mdtools": "my-amber"},
    })
    assert caps.env_for_category("siesta")  == "my-siesta"
    assert caps.env_for_category("mdtools") == "my-amber"
    # Unspecified category falls back to compiled default.
    assert caps.env_for_category("pyscf")   == "molbuilder-pySCF"


def test_env_for_tool_known():
    caps = _caps()
    assert caps.env_for_tool("tleap")      == "molbuilder-MDtools"
    assert caps.env_for_tool("siesta")     == "molbuilder-siesta"


def test_env_for_tool_playwright_not_routed():
    """``playwright`` is deliberately unrouted: browser E2E runs under
    the host env (in-process Flask app), not a dedicated conda env."""
    assert _caps().env_for_tool("playwright") is None


def test_env_for_tool_unknown_returns_none():
    assert _caps().env_for_tool("ls") is None


def test_env_for_tool_picks_up_override():
    caps = _caps(runtime_config={"envs": {"mdtools": "amber26-dac"}})
    assert caps.env_for_tool("tleap")    == "amber26-dac"
    assert caps.env_for_tool("parmchk2") == "amber26-dac"


def test_env_available():
    caps = _caps(conda_envs={"molbuilder-MDtools", "other-env"})
    assert caps.env_available("molbuilder-MDtools") is True
    assert caps.env_available("not-there")          is False


def test_routed_env_returns_name_when_routed_and_available():
    caps = _caps(conda_envs={"molbuilder-MDtools"})
    assert caps.routed_env("tleap") == "molbuilder-MDtools"


def test_routed_env_none_when_env_missing():
    caps = _caps(conda_envs=set())
    assert caps.routed_env("tleap") is None


def test_routed_env_none_when_unrouted():
    caps = _caps(conda_envs={"random-env"})
    assert caps.routed_env("ls") is None


def test_tool_available_via_routed_env():
    caps = _caps(conda_envs={"molbuilder-MDtools"})
    assert caps.tool_available("tleap") is True


def test_tool_available_via_host_path(monkeypatch):
    monkeypatch.setattr(diagnostics.shutil, "which",
                         lambda t: "/usr/bin/tleap" if t == "tleap" else None)
    caps = _caps(conda_envs=set())
    assert caps.tool_available("tleap") is True


def test_tool_available_unreachable(monkeypatch):
    monkeypatch.setattr(diagnostics.shutil, "which", lambda t: None)
    caps = _caps(conda_envs=set())
    assert caps.tool_available("tleap") is False


# --------------------------------------------------------------------- #
#  detect() -- composition of the probe                                 #
# --------------------------------------------------------------------- #


def _stub_conda_env_list(envs_list: List[str], base: str = None):
    """Return a fake subprocess.run that mimics ``conda env list --json``.

    ``base`` adds the ``envs_details`` block a real manager returns, flagging
    that prefix as its installation.  Without it the stub is a manager that
    reports only ``envs`` -- and then nothing can say which prefix is an
    installation, so none is excluded.
    """
    def fake_run(argv, *args, **kwargs):
        cp = MagicMock(spec=subprocess.CompletedProcess)
        cp.returncode = 0
        payload: Dict[str, Any] = {"envs": envs_list}
        if base is not None:
            payload["envs_details"] = {
                p: {"name": ("base" if p == base else p.rsplit("/", 1)[-1]),
                    "base": p == base}
                for p in envs_list
            }
        cp.stdout = json.dumps(payload)
        cp.stderr = ""
        return cp
    return fake_run


def test_detect_assembles_capabilities(monkeypatch):
    monkeypatch.setattr(diagnostics, "read_config",
                         lambda: {"envs": {"siesta": "my-siesta"}})
    monkeypatch.setattr(diagnostics.shutil, "which",
                         lambda t: "/usr/bin/conda" if t == "conda" else None)
    monkeypatch.setattr(diagnostics.subprocess, "run",
                         _stub_conda_env_list([
                             "/home/u/miniconda3",
                             "/home/u/miniconda3/envs/molbuilder-MDtools",
                             "/home/u/miniconda3/envs/some-other",
                         ], base="/home/u/miniconda3"))
    caps = detect()
    assert caps.conda_binary == "/usr/bin/conda"
    # The installation root is excluded by the manager's own `base` flag;
    # only true named envs make it into the snapshot.
    # `{name: prefix}` since 2026-09-12: the snapshot carries WHERE each env is,
    # so `install._env_prefix` stops paying a 1.2 s registry read per recipe.
    # Membership -- which is what every gate asks -- reads a mapping unchanged.
    assert caps.conda_envs == {
        "molbuilder-MDtools": "/home/u/miniconda3/envs/molbuilder-MDtools",
        "some-other":         "/home/u/miniconda3/envs/some-other",
    }
    assert caps.runtime_config == {"envs": {"siesta": "my-siesta"}}


def test_detect_filters_out_conda_root_installation(monkeypatch):
    """A base installation is not an env of ours, and its prefix's basename
    (``miniconda3``) is not a name conda knows at all -- the base env is called
    ``base``, so keying it by basename would INVENT a name.

    **Asserted through the mechanism that delivers it** *(2026-09-12)*: the
    manager's own ``envs_details`` block flags its installation.  This used to be
    a path rule -- "keep only prefixes whose parent is called `envs`" -- which
    excluded the base AND every env created with ``--prefix`` elsewhere, so a
    real env went missing and `probe_env_state` then called it FRESH, whereupon
    ``conda create`` would make a second env beside it.  A manager that reports
    no details is covered in
    `test_envs_one_answer_about_an_env.py`: nothing is excluded there, because
    nothing in that document says which prefix is an installation."""
    monkeypatch.setattr(diagnostics, "read_config", lambda: {})
    monkeypatch.setattr(diagnostics.shutil, "which",
                         lambda t: "/usr/bin/conda" if t == "conda" else None)
    monkeypatch.setattr(diagnostics.subprocess, "run",
                         _stub_conda_env_list(
                             ["/home/u/miniconda3", "/opt/anaconda3"],
                             base="/home/u/miniconda3"))
    caps = detect()
    # The flagged installation is gone; the other prefix is an env as far as
    # this document says -- which is the manager's statement, not a guess here.
    assert caps.conda_envs == {"anaconda3": "/opt/anaconda3"}


def test_detect_no_conda_gives_empty_envs(monkeypatch):
    monkeypatch.setattr(diagnostics, "read_config", lambda: {})
    monkeypatch.setattr(diagnostics.shutil, "which", lambda t: None)
    monkeypatch.delenv("CONDA_EXE", raising=False)
    caps = detect()
    assert caps.conda_binary is None
    assert caps.conda_envs   == {}


def test_detect_conda_failure_yields_empty_envs(monkeypatch):
    """Non-zero exit / timeout / malformed JSON -> empty set, not raise."""
    monkeypatch.setattr(diagnostics, "read_config", lambda: {})
    monkeypatch.setattr(diagnostics.shutil, "which",
                         lambda t: "/usr/bin/conda" if t == "conda" else None)
    def failing_run(argv, *a, **kw):
        cp = MagicMock(spec=subprocess.CompletedProcess)
        cp.returncode = 1
        cp.stdout = ""
        cp.stderr = "boom"
        return cp
    monkeypatch.setattr(diagnostics.subprocess, "run", failing_run)
    caps = detect()
    assert caps.conda_binary == "/usr/bin/conda"
    assert caps.conda_envs   == {}


# --------------------------------------------------------------------- #
#  Singleton lifecycle                                                  #
# --------------------------------------------------------------------- #


def test_get_capabilities_auto_initialises(monkeypatch):
    """First call to get_capabilities() runs detect; subsequent calls
    return the same snapshot."""
    monkeypatch.setattr(diagnostics, "detect",
                         lambda: _caps(conda_binary="/test/conda"))
    caps1 = get_capabilities()
    caps2 = get_capabilities()
    assert caps1 is caps2
    assert caps1.conda_binary == "/test/conda"


def test_initialize_rebinds_snapshot(monkeypatch):
    counter = {"n": 0}
    def fake_detect():
        counter["n"] += 1
        return _caps(conda_binary=f"/conda-v{counter['n']}")
    monkeypatch.setattr(diagnostics, "detect", fake_detect)

    initialize()
    assert get_capabilities().conda_binary == "/conda-v1"

    initialize()
    assert get_capabilities().conda_binary == "/conda-v2"


def test_set_capabilities_injects():
    """Direct injection -- tests, dependency injection."""
    injected = _caps(conda_binary="/injected/conda")
    set_capabilities(injected)
    assert get_capabilities() is injected


# --------------------------------------------------------------------- #
#  Env-manager autodetect (2026-06-23): mamba > micromamba > conda      #
# --------------------------------------------------------------------- #
#
# ASU supercomputer deployment + general HPC use need transparent support
# for mamba (faster solver) and micromamba (static single-binary).
# Both are drop-in replacements for ``conda create/run/env list``;
# the only thing molbuilder needs to do is pick whichever is available.
#
# Detection rule: prefer mamba > micromamba > conda on PATH; fall
# back to ``$MAMBA_EXE`` / ``$CONDA_EXE`` env vars.  Once detected,
# the chosen binary is used uniformly via ``caps.conda_binary``.


class TestEnvManagerAutodetect:

    def _set_which(self, monkeypatch, mapping):
        """Stub shutil.which: returns mapping[name] or None."""
        monkeypatch.setattr(
            diagnostics.shutil, "which",
            lambda t: mapping.get(t),
        )

    def test_mamba_preferred_when_all_three_present(self, monkeypatch):
        monkeypatch.delenv("MAMBA_EXE", raising=False)
        monkeypatch.delenv("CONDA_EXE", raising=False)
        self._set_which(monkeypatch, {
            "mamba":      "/opt/mamba/bin/mamba",
            "micromamba": "/opt/mm/bin/micromamba",
            "conda":      "/opt/conda/bin/conda",
        })
        assert diagnostics._find_conda_binary() == ("/opt/mamba/bin/mamba", "PATH (mamba)")

    def test_micromamba_preferred_over_conda(self, monkeypatch):
        monkeypatch.delenv("MAMBA_EXE", raising=False)
        monkeypatch.delenv("CONDA_EXE", raising=False)
        self._set_which(monkeypatch, {
            "micromamba": "/opt/mm/bin/micromamba",
            "conda":      "/opt/conda/bin/conda",
        })
        assert diagnostics._find_conda_binary() == ("/opt/mm/bin/micromamba", "PATH (micromamba)")

    def test_conda_only_works_as_fallback(self, monkeypatch):
        monkeypatch.delenv("MAMBA_EXE", raising=False)
        monkeypatch.delenv("CONDA_EXE", raising=False)
        self._set_which(monkeypatch, {"conda": "/opt/conda/bin/conda"})
        assert diagnostics._find_conda_binary() == ("/opt/conda/bin/conda", "PATH (conda)")

    @staticmethod
    def _exe(tmp_path, name):
        """A real executable file -- the env-var fallback checks, as
        ``shutil.which`` does for PATH and as install-env.sh's probe
        already did (``[[ -n "${v}" && -x "${v}" ]]``).  Fictional paths
        no longer stand in for one."""
        p = tmp_path / name
        p.write_text("#!/bin/sh\n")
        p.chmod(0o755)
        return str(p)

    def test_falls_back_to_mamba_exe_env_var(self, monkeypatch, tmp_path):
        """When nothing is on PATH but ``$MAMBA_EXE`` is set (mamba's
        activation hook does this), use it."""
        mamba = self._exe(tmp_path, "mamba")
        monkeypatch.setenv("MAMBA_EXE", mamba)
        monkeypatch.delenv("CONDA_EXE", raising=False)
        self._set_which(monkeypatch, {})
        assert diagnostics._find_conda_binary() == (mamba, "$MAMBA_EXE")

    def test_mamba_exe_wins_over_conda_exe(self, monkeypatch, tmp_path):
        """Both env vars set -- ``$MAMBA_EXE`` wins (faster manager,
        consistent with the PATH preference order)."""
        mamba = self._exe(tmp_path, "mamba")
        conda = self._exe(tmp_path, "conda")
        monkeypatch.setenv("MAMBA_EXE", mamba)
        monkeypatch.setenv("CONDA_EXE", conda)
        self._set_which(monkeypatch, {})
        assert diagnostics._find_conda_binary() == (mamba, "$MAMBA_EXE")

    def test_a_stale_env_var_does_not_beat_a_good_one(self, monkeypatch,
                                                       tmp_path):
        """THE hole install-env.sh's hand-off exposed.  The shim probes
        for a manager and exports the one it found -- but MAMBA_EXE is
        consulted first here, so a stale MAMBA_EXE (removed or renamed
        install) beat the correct CONDA_EXE the shim had just set.  The
        shell rejected that path as non-executable and Python accepted
        it: two probes disagreeing, which is the failure this seam
        exists to end."""
        conda = self._exe(tmp_path, "conda")
        monkeypatch.setenv("MAMBA_EXE", str(tmp_path / "removed-mamba"))
        monkeypatch.setenv("CONDA_EXE", conda)
        self._set_which(monkeypatch, {})
        assert diagnostics._find_conda_binary() == (conda, "$CONDA_EXE")

    def test_a_directory_is_not_an_env_manager(self, monkeypatch, tmp_path):
        """os.access(X_OK) is true for a directory; isfile is what makes
        the check mean 'a program I can run'."""
        monkeypatch.setenv("MAMBA_EXE", str(tmp_path))
        monkeypatch.delenv("CONDA_EXE", raising=False)
        self._set_which(monkeypatch, {})
        assert diagnostics._find_conda_binary() == (None, None)

    def test_a_non_executable_file_is_not_an_env_manager(self, monkeypatch,
                                                         tmp_path):
        p = tmp_path / "conda"
        p.write_text("#!/bin/sh\n")
        p.chmod(0o644)
        monkeypatch.setenv("CONDA_EXE", str(p))
        monkeypatch.delenv("MAMBA_EXE", raising=False)
        self._set_which(monkeypatch, {})
        assert diagnostics._find_conda_binary() == (None, None)

    def test_no_manager_returns_none(self, monkeypatch):
        monkeypatch.delenv("MAMBA_EXE", raising=False)
        monkeypatch.delenv("CONDA_EXE", raising=False)
        self._set_which(monkeypatch, {})
        assert diagnostics._find_conda_binary() == (None, None)
