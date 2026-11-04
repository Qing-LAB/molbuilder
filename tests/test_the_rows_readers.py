"""The readers a catalogue row names -- `job-contracts.md` § 2.2, the
``reads`` column of `runfiles.WRITTEN`.

A run output's row names the code that reads it, by question (how the run
ended, where its engine stops, its live pass, ...): `runfiles.reader` is the
one door every reader asks, on the machine and beside a job, where the
monitor's bundle carries each reader a row names (`runwrap.monitor_companions`).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from molbuilder import runfiles
from molbuilder.runfiles import READ_BESIDE_A_JOB, WRITTEN

#: Every (role, question) a row names a reader for.
ASKED = [(a.role or a.name, q) for a in WRITTEN for q, _ref in a.reads]


@pytest.mark.parametrize("role,question", ASKED,
                         ids=[f"{r} {q}" for r, q in ASKED])
def test_every_reader_a_row_names_resolves(role, question):
    """THE SILENT FAILURE: a reader renamed or moved while its row still
    names it -- found only when a job asks how its run ended, and then as
    an ending that cannot be read."""
    assert runfiles.reader(role, question) is not None


def test_the_readers_a_job_asks_beside_it_load_from_its_bundle_alone(
        tmp_path):
    """Beside a job, molbuilder is not installed: each reader a row names
    for a question the job asks there (`READ_BESIDE_A_JOB`) comes out of
    `mb_monitor.pyz`, gathered by the rows (`runwrap.monitor_companions`),
    and so does every other member -- the monitor, and each module a member
    states it needs beside it (its ``TRAVELS_WITH``), none listed by hand.
    MUTATION THIS MUST FAIL AGAINST: a member leaving out of its
    ``TRAVELS_WITH`` a module it imports as it loads, beside the job (the
    SIESTA reading pass's held-atom numbering, `engine_atom_index`) -- every
    SIESTA run's monitor then stops on its first read.  An import made only
    inside a function is review's to hold: loading proves nothing of it."""
    from molbuilder.runfiles import MONITOR_BUNDLE
    from molbuilder.runwrap import monitor_bundle, monitor_companions
    (tmp_path / MONITOR_BUNDLE).write_bytes(monitor_bundle())
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    # The condition has to be real, or this proves nothing.
    control = subprocess.run([sys.executable, "-c", "import molbuilder"],
                             cwd=tmp_path, env=env, capture_output=True,
                             text=True, timeout=120)
    assert control.returncode != 0, (
        "molbuilder is importable from the job's folder, so this test cannot "
        "reproduce a compute node.  Is it pip-installed?")
    asked = sorted(f"{a.role} {q}" for a in WRITTEN for q, _ref in a.reads
                   if q in READ_BESIDE_A_JOB)
    script = "\n".join([
        "import json, os, sys",
        f"sys.path.insert(0, {MONITOR_BUNDLE!r})",
        "import runfiles",
        "got = {}",
        "for a in runfiles.WRITTEN:",
        "    for q, ref in a.reads:",
        "        if q in runfiles.READ_BESIDE_A_JOB:",
        "            runfiles.reader(a.role, q)",
        "            mod = sys.modules[ref.split(':')[0].rsplit('.', 1)[-1]]",
        "            got[f'{a.role} {q}'] = os.path.basename(",
        "                os.path.dirname(mod.__file__))",
        "import importlib",
        f"members = {sorted(n[:-len('.py')] for n in monitor_companions())!r}",
        "loaded = {m: os.path.basename(os.path.dirname(",
        "    importlib.import_module(m).__file__)) for m in members}",
        "print(json.dumps({'readers': got, 'members': loaded}))",
    ])
    done = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                          env=env, capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, (
        f"a reader a row names does not load from {MONITOR_BUNDLE} without "
        f"molbuilder:\n{done.stdout}{done.stderr}")
    answer = json.loads(done.stdout.strip().splitlines()[-1])
    got, loaded = answer["readers"], answer["members"]
    assert sorted(got) == asked
    assert set(got.values()) == {MONITOR_BUNDLE}, got
    assert set(loaded.values()) == {MONITOR_BUNDLE}, loaded
