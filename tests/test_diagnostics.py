"""Routing tables, Capabilities snapshot, singleton lifecycle.

The module under test exposes three small dicts (``DEFAULT_ENV_NAMES``,
``TOOL_TO_CATEGORY``, ``EXTENSION_TO_CATEGORY``), one frozen dataclass
(:class:`Capabilities`), one probe (:func:`detect`), and four singleton
lifecycle helpers.

Singleton isolation is handled by an autouse fixture in
``tests/conftest.py`` -- every test starts and ends with the snapshot
reset.
"""

from __future__ import annotations

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


def test_default_env_names_covers_the_routed_categories():
    """Four routed backend categories: siesta (precompiled CPU),
    siesta-gpu (built from source), pyscf, mdtools.  There is no
    "tests" category -- browser E2E runs under the host env."""
    assert set(DEFAULT_ENV_NAMES) == {
        "siesta", "siesta-gpu", "pyscf", "mdtools",
    }


def test_default_env_names_match_readme_install():
    """The names are what docs/ops/installation.md tells users to create."""
    assert DEFAULT_ENV_NAMES["siesta"]     == "molbuilder-siesta"
    assert DEFAULT_ENV_NAMES["siesta-gpu"] == "molbuilder-siesta-gpu"
    assert DEFAULT_ENV_NAMES["pyscf"]      == "molbuilder-pySCF"
    assert DEFAULT_ENV_NAMES["mdtools"]    == "molbuilder-MDtools"
    # No "tests" category: browser E2E runs under the host env, not a
    # dedicated conda env (the E2E fixture starts the app in-process).
    assert "tests" not in DEFAULT_ENV_NAMES


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


def test_capabilities_is_frozen():
    """No accidental attribute reassignment -- the snapshot is a stable
    contract.  (The ``runtime_config`` *dict* is mutable by Python
    semantics; treat it as read-only by convention -- see module
    docstring.)"""
    caps = _caps()
    with pytest.raises((AttributeError, Exception)):
        caps.conda_binary = "/somewhere/else"   # type: ignore[misc]


# --------------------------------------------------------------------- #
#  detect() -- composition of the probe                                 #
# --------------------------------------------------------------------- #


def _stub_conda_env_list(envs_list: List[str]):
    """Return a fake subprocess.run that mimics ``conda env list --json``."""
    def fake_run(argv, *args, **kwargs):
        cp = MagicMock(spec=subprocess.CompletedProcess)
        cp.returncode = 0
        cp.stdout = json.dumps({"envs": envs_list})
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
                         ]))
    caps = detect()
    assert caps.conda_binary == "/usr/bin/conda"
    # Conda installation root (no /envs/ parent) is filtered out;
    # only true named envs make it into the snapshot.
    assert caps.conda_envs == frozenset({"molbuilder-MDtools", "some-other"})
    assert caps.runtime_config == {"envs": {"siesta": "my-siesta"}}


def test_detect_filters_out_conda_root_installation(monkeypatch):
    """Base installation paths don't have ``/envs/`` as parent and
    aren't addressable via ``conda run -n``."""
    monkeypatch.setattr(diagnostics, "read_config", lambda: {})
    monkeypatch.setattr(diagnostics.shutil, "which",
                         lambda t: "/usr/bin/conda" if t == "conda" else None)
    monkeypatch.setattr(diagnostics.subprocess, "run",
                         _stub_conda_env_list([
                             "/home/u/miniconda3",         # base; filter out
                             "/opt/anaconda3",             # base; filter out
                         ]))
    caps = detect()
    assert caps.conda_envs == frozenset()


def test_detect_no_conda_gives_empty_envs(monkeypatch):
    monkeypatch.setattr(diagnostics, "read_config", lambda: {})
    monkeypatch.setattr(diagnostics.shutil, "which", lambda t: None)
    monkeypatch.delenv("CONDA_EXE", raising=False)
    caps = detect()
    assert caps.conda_binary is None
    assert caps.conda_envs   == frozenset()


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
    assert caps.conda_envs   == frozenset()


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
        assert diagnostics._find_conda_binary() == "/opt/mamba/bin/mamba"

    def test_micromamba_preferred_over_conda(self, monkeypatch):
        monkeypatch.delenv("MAMBA_EXE", raising=False)
        monkeypatch.delenv("CONDA_EXE", raising=False)
        self._set_which(monkeypatch, {
            "micromamba": "/opt/mm/bin/micromamba",
            "conda":      "/opt/conda/bin/conda",
        })
        assert diagnostics._find_conda_binary() == "/opt/mm/bin/micromamba"

    def test_conda_only_works_as_fallback(self, monkeypatch):
        monkeypatch.delenv("MAMBA_EXE", raising=False)
        monkeypatch.delenv("CONDA_EXE", raising=False)
        self._set_which(monkeypatch, {"conda": "/opt/conda/bin/conda"})
        assert diagnostics._find_conda_binary() == "/opt/conda/bin/conda"

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
        assert diagnostics._find_conda_binary() == mamba

    def test_mamba_exe_wins_over_conda_exe(self, monkeypatch, tmp_path):
        """Both env vars set -- ``$MAMBA_EXE`` wins (faster manager,
        consistent with the PATH preference order)."""
        mamba = self._exe(tmp_path, "mamba")
        conda = self._exe(tmp_path, "conda")
        monkeypatch.setenv("MAMBA_EXE", mamba)
        monkeypatch.setenv("CONDA_EXE", conda)
        self._set_which(monkeypatch, {})
        assert diagnostics._find_conda_binary() == mamba

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
        assert diagnostics._find_conda_binary() == conda

    def test_a_directory_is_not_an_env_manager(self, monkeypatch, tmp_path):
        """os.access(X_OK) is true for a directory; isfile is what makes
        the check mean 'a program I can run'."""
        monkeypatch.setenv("MAMBA_EXE", str(tmp_path))
        monkeypatch.delenv("CONDA_EXE", raising=False)
        self._set_which(monkeypatch, {})
        assert diagnostics._find_conda_binary() is None

    def test_a_non_executable_file_is_not_an_env_manager(self, monkeypatch,
                                                         tmp_path):
        p = tmp_path / "conda"
        p.write_text("#!/bin/sh\n")
        p.chmod(0o644)
        monkeypatch.setenv("CONDA_EXE", str(p))
        monkeypatch.delenv("MAMBA_EXE", raising=False)
        self._set_which(monkeypatch, {})
        assert diagnostics._find_conda_binary() is None

    def test_no_manager_returns_none(self, monkeypatch):
        monkeypatch.delenv("MAMBA_EXE", raising=False)
        monkeypatch.delenv("CONDA_EXE", raising=False)
        self._set_which(monkeypatch, {})
        assert diagnostics._find_conda_binary() is None
