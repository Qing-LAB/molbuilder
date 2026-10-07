"""A baked preamble must not fail as a bare bash error on the target.

**The failure this closes (Sol, 2026-08-24).** The preamble is baked
VERBATIM into the `.run.sh`, and it was baked from the machine that ran
`prep` -- today it is the target record's copy of that machine's `env_init`
(`configuration.md` § 4).  The workstation's config said

    source /home/u/miniconda3/etc/profile.d/conda.sh

so prepping from the browser — where the server runs on the workstation —
put that line into every trial's wrapper.  The bundle then travelled to
Sol, which has no `/home/u/miniconda3` and activates with
`module load mamba` instead, and every job died with

    siesta-...-run.sh: line 196: /home/u/.../conda.sh: No such file or directory

naming neither the config key that put the path there, nor the machine it
came from, nor what to do about it.

**Why the check has to be in the script.** Nothing at prep time can know:
on the prepping machine the file is right there.  The only molbuilder code
that runs on the target is the wrapper, so the wrapper checks its own
preconditions before relying on them.
"""
from __future__ import annotations


import pytest

from molbuilder.runwrap import _preamble_source_targets


class TestWhichPathsAreChecked:
    """Only ABSOLUTE `source`/`.` targets — the ones that can silently
    refer to a machine that is not this one.  A wrong guard is worse than
    none: it would refuse a run that would have worked."""

    @pytest.mark.parametrize("line,expected", [
        ("source /home/u/miniconda3/etc/profile.d/conda.sh",
         ["/home/u/miniconda3/etc/profile.d/conda.sh"]),
        ('source "/opt/conda/etc/profile.d/conda.sh"',
         ["/opt/conda/etc/profile.d/conda.sh"]),
        (". /opt/x/conda.sh", ["/opt/x/conda.sh"]),
        ("source /a/b.sh   # the hook", ["/a/b.sh"]),
        ("module load mamba", []),          # no path to check
        ("source ./local.sh", []),          # relative: author's business
        ("source $HOME/x.sh", []),          # built from a variable
        ("", []),
    ])
    def test_extractor(self, line, expected):
        assert _preamble_source_targets([("target", line)]) == expected


