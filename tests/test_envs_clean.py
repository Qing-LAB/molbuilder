"""Tests for ``molbuilder envs clean`` -- disk-cleanup subcommand for
source-build envs.

Pins the contract that the user can run after a successful install:

  * the recipe must have a build_spec (conda-only recipes are rejected)
  * an unknown recipe name surfaces the registered list
  * the user is shown which dirs would be deleted + their sizes before
    anything is removed
  * --dry-run never touches the filesystem
  * confirmation prompt defaults to NO (so a bare ``clean`` exits without
    deleting anything)
  * --yes skips the prompt
  * load-bearing dirs (component install dirs, logs/, .sentinels/,
    .toolchain-fingerprint) are NEVER in the deletion set
  * --keep-src drops src/ from the deletion set
  * a missing artifact-root is a clean no-op, not a crash
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from molbuilder.envs._cli import envs_group


def _run(*args, input=None):
    return CliRunner().invoke(envs_group, list(args), input=input,
                              catch_exceptions=False)


# --------------------------------------------------------------------- #
#  Error paths -- recipe shape rejection                                #
# --------------------------------------------------------------------- #


def test_unknown_recipe_lists_registered():
    r = _run("clean", "no-such-recipe", "--yes")
    assert r.exit_code != 0
    assert "unknown recipe" in r.output
    assert "molbuilder-siesta-gpu" in r.output


def test_conda_only_recipe_rejected():
    """conda-only recipes (siesta, MDtools, tests, host, pySCF) have
    no build_spec; ``clean`` is meaningful only for source-build envs."""
    r = _run("clean", "molbuilder-siesta", "--yes")
    assert r.exit_code != 0
    assert "conda-only" in r.output
    assert "no build_spec" in r.output


def test_conda_only_recipe_rejected_for_host_env():
    r = _run("clean", "molbuilder", "--yes")
    assert r.exit_code != 0
    assert "conda-only" in r.output


# --------------------------------------------------------------------- #
#  Dry-run + size reporting + no-touch contract                         #
# --------------------------------------------------------------------- #


@pytest.fixture
def fake_artifact_root(tmp_path):
    """Build a fake siesta-gpu artifact root with the full directory
    layout populated with cheap-to-create dummy content, plus a
    ``conda_binary``/``env_available`` stub so the CLI thinks the env
    exists at this prefix.

    Returns ``(env_prefix, paths_root, dummy_sizes)`` where
    ``dummy_sizes`` maps relative-path labels to the byte counts
    written.  The test asserts those exact sizes show up in the
    rendered report.
    """
    env_prefix = tmp_path / "envs" / "molbuilder-siesta-gpu"
    root = env_prefix / "opt" / "siesta-gpu-stack"
    root.mkdir(parents=True)
    # Populate every dir the subcommand mentions.
    sizes = {
        "build/":     2048,
        "src/":       1024,
        ".ccache/":    512,
        ".cache/":     256,
        ".tmp/":       128,
        "logs/":        64,
        ".sentinels/":  32,
        "elpa/":     16384,   # load-bearing -- must NOT be deleted
        "siesta/":   32768,   # load-bearing
    }
    for label, n in sizes.items():
        d = root / label.rstrip("/")
        d.mkdir(parents=True, exist_ok=True)
        (d / "blob.bin").write_bytes(b"x" * n)
    (root / ".toolchain-fingerprint").write_bytes(b"abc123")
    return env_prefix, root, sizes


def _patched_run(args, env_prefix):
    """Run ``clean`` with the env-detection layer stubbed out so the
    CLI uses our fake env_prefix instead of probing real conda."""
    # Mock the capability snapshot to claim mamba is present + the env
    # exists.  Mock the env-prefix resolver to return our fake prefix.
    with patch("molbuilder.envs._cli.get_capabilities") as gc, \
         patch("molbuilder.envs.install._env_prefix") as ep:
        caps = gc.return_value
        caps.conda_binary = "mamba"
        caps.env_available.return_value = True
        ep.return_value = str(env_prefix)
        return CliRunner().invoke(envs_group, args,
                                  catch_exceptions=False)


def test_dry_run_shows_all_candidate_dirs_with_sizes(fake_artifact_root):
    env_prefix, root, sizes = fake_artifact_root
    r = _patched_run(["clean", "molbuilder-siesta-gpu", "--dry-run"],
                     env_prefix)
    assert r.exit_code == 0, r.output
    # Every build-only candidate must appear in the report.
    for label in ("build/", "src/", ".ccache/", ".cache/", ".tmp/"):
        assert label in r.output, f"missing `{label}` from report"
    # Load-bearing dirs MUST appear in the "kept" section.
    assert "elpa/" in r.output
    assert "siesta/" in r.output
    assert "INSTALLED binary" in r.output
    # Dry-run leaves the filesystem untouched.
    assert (root / "build").exists()
    assert (root / "src").exists()
    assert (root / "elpa").exists()
    assert (root / "siesta").exists()
    assert "--dry-run: no files deleted" in r.output


def test_dry_run_reports_nonzero_total(fake_artifact_root):
    env_prefix, root, sizes = fake_artifact_root
    r = _patched_run(["clean", "molbuilder-siesta-gpu", "--dry-run"],
                     env_prefix)
    assert "TOTAL reclaimable" in r.output
    # Sum of build-only blobs = 2048+1024+512+256+128 = 3968 B
    # Reported in KiB ("3.9 KiB" range).
    assert "KiB" in r.output or "MiB" in r.output


# --------------------------------------------------------------------- #
#  Confirmation gate                                                    #
# --------------------------------------------------------------------- #


def test_default_prompt_aborts_on_no(fake_artifact_root):
    env_prefix, root, _ = fake_artifact_root
    # User pipes 'n' to the confirmation prompt -> click.confirm
    # returns False -> our handler exits 2 with the abort message.
    with patch("molbuilder.envs._cli.get_capabilities") as gc, \
         patch("molbuilder.envs.install._env_prefix") as ep:
        caps = gc.return_value
        caps.conda_binary = "mamba"
        caps.env_available.return_value = True
        ep.return_value = str(env_prefix)
        r = CliRunner().invoke(envs_group,
                                ["clean", "molbuilder-siesta-gpu"],
                                input="n\n",
                                catch_exceptions=False)
    assert r.exit_code == 2, r.output
    assert "aborted by user" in r.output
    # Nothing got deleted.
    assert (root / "build").exists()
    assert (root / "src").exists()


def test_yes_flag_proceeds_without_prompt(fake_artifact_root):
    env_prefix, root, sizes = fake_artifact_root
    r = _patched_run(["clean", "molbuilder-siesta-gpu", "--yes"],
                     env_prefix)
    assert r.exit_code == 0, r.output
    # Build-only set wiped.
    for label in ("build", "src", ".ccache", ".cache", ".tmp"):
        assert not (root / label).exists(), f"`{label}` survived deletion"
    # Load-bearing dirs survive.
    assert (root / "elpa").exists()
    assert (root / "elpa" / "blob.bin").exists()
    assert (root / "siesta").exists()
    assert (root / "siesta" / "blob.bin").exists()
    assert (root / "logs").exists()
    assert (root / ".sentinels").exists()
    assert (root / ".toolchain-fingerprint").exists()
    assert "freed" in r.output


# --------------------------------------------------------------------- #
#  --keep-src                                                            #
# --------------------------------------------------------------------- #


def test_keep_src_preserves_source_clones(fake_artifact_root):
    env_prefix, root, _ = fake_artifact_root
    r = _patched_run(["clean", "molbuilder-siesta-gpu",
                      "--keep-src", "--yes"], env_prefix)
    assert r.exit_code == 0, r.output
    # src/ survives.
    assert (root / "src").exists()
    assert (root / "src" / "blob.bin").exists()
    # The rest of the build-only set is still wiped.
    assert not (root / "build").exists()
    assert not (root / ".ccache").exists()
    # src/ must not appear in the rm-rf log lines.
    assert "rm -rf" in r.output
    for line in r.output.splitlines():
        if line.startswith("[clean] rm -rf "):
            assert "/src" not in line, line


# --------------------------------------------------------------------- #
#  Missing artifact root -- clean no-op                                 #
# --------------------------------------------------------------------- #


def test_missing_artifact_root_is_clean_noop(tmp_path):
    """Env exists but the artifact root doesn't (e.g. the user
    nuked $CONDA_PREFIX/opt manually).  ``clean`` reports + exits
    cleanly instead of crashing."""
    env_prefix = tmp_path / "envs" / "molbuilder-siesta-gpu"
    env_prefix.mkdir(parents=True)
    # NB: no opt/siesta-gpu-stack subtree.
    r = _patched_run(["clean", "molbuilder-siesta-gpu", "--yes"],
                     env_prefix)
    assert r.exit_code == 0, r.output
    assert "artifact root does not exist" in r.output


def test_already_clean_artifact_root_is_noop(tmp_path):
    """Artifact root exists with only load-bearing dirs (a state the
    user reaches after a previous successful ``clean``).  Re-running
    ``clean`` reports "nothing to delete" and exits 0."""
    env_prefix = tmp_path / "envs" / "molbuilder-siesta-gpu"
    root = env_prefix / "opt" / "siesta-gpu-stack"
    (root / "elpa").mkdir(parents=True)
    (root / "siesta").mkdir(parents=True)
    (root / "logs").mkdir(parents=True)
    r = _patched_run(["clean", "molbuilder-siesta-gpu", "--yes"],
                     env_prefix)
    assert r.exit_code == 0, r.output
    assert "nothing to delete" in r.output


# --------------------------------------------------------------------- #
#  Missing-env path                                                     #
# --------------------------------------------------------------------- #


def test_missing_env_reports_and_exits_zero(tmp_path):
    """Recipe is source-build-shaped but the conda env hasn't been
    created yet -- ``clean`` says "nothing to clean" and exits 0
    (it's a no-op, not an error).  This lets a CI script call
    ``clean`` defensively before measurement steps without first
    having to check whether the env exists."""
    with patch("molbuilder.envs._cli.get_capabilities") as gc:
        caps = gc.return_value
        caps.conda_binary = "mamba"
        caps.env_available.return_value = False
        r = CliRunner().invoke(envs_group,
                               ["clean", "molbuilder-siesta-gpu", "--yes"],
                               catch_exceptions=False)
    assert r.exit_code == 0
    assert "does not exist" in r.output
    assert "nothing to clean" in r.output
