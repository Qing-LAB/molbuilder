"""FIELD TEST -- the [MACHINE] line names the device models this node shows, each
once, and never how many (`docs/execution/scheduler.md` R12).

Goal: a run's log says what kind of node it landed on.  The COUNT of devices a
job can see is its allocation's -- inside a scheduled job the device cgroup shows
only what was granted -- so two trials of a sweep given different counts must
still read as one machine.

What `nvidia-smi -L` prints is read where it is printed: this runs on the target,
against its real tool, and is skipped where no device is visible.  With one
device the "each once" half has nothing to tell apart; it bites on a node with
two of a model.

    MOLBUILDER_FIELD_RECORD=<the record> python tools/testrun.py run field
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from molbuilder import monitor


def test_the_machine_line_names_each_model_once_and_no_count():
    if shutil.which("nvidia-smi") is None:
        pytest.skip("no nvidia-smi on this machine")
    listed = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                            text=True, timeout=5)
    models = [line.split(":", 1)[1].split("(UUID")[0].strip()
              for line in listed.stdout.splitlines()
              if line.startswith("GPU ")]
    if listed.returncode != 0 or not models:
        pytest.skip("no device visible here")
    distinct = list(dict.fromkeys(models))
    assert monitor.machine_line().split("gpu=", 1)[1] == ", ".join(distinct), (
        "the gpu field must be exactly the distinct models, in the order they "
        "are listed: a repeat or a count is the allocation's, not the node's")
