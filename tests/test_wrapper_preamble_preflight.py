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


class TestOneReaderOfSlurmsGresSpelling:
    """One reader of Slurm's gres spelling: the type is read from the token,
    never matched against a list of GPU names.
    """

    @staticmethod
    def _q():
        from molbuilder.scheduler.quantities import parse_gres
        return parse_gres

    def test_the_type_is_read_from_the_token_not_guessed(self):
        q = self._q()
        assert q("gpu:gh200:1") == {"gh200": 1}          # not h200
        assert q("gpu:a100.40gb:4") == {"a100.40gb": 4}  # not a100
        assert q("gpu:h200.35gb:4") == {"h200.35gb": 4}
        assert q("gpu:hl225:8") == {"hl225": 8}          # Habana, not None

    def test_the_slurm_shapes_it_must_survive(self):
        q = self._q()
        assert q("gpu:a100:4(S:0-1)") == {"a100": 4}   # affinity tail
        assert q("gpu:a100:4,mps:400") == {"a100": 4}  # mps is not a GPU count
        assert q("gpu:4") == {"gpu": 4}                # untyped
        assert q("(null)") == {} and q("none") == {} and q("") == {}

    def test_a_partition_merged_across_node_groups_keeps_the_larger(self):
        assert self._q()("gpu:a100:2,gpu:a100:8") == {"a100": 8}

    def test_the_record_narrows_the_same_reading(self):
        """`Topology` states ONE device kind, so it narrows -- it does not
        re-read.  Untyped stays None there: the field means *which device*,
        and "gpu" answers nothing."""
        from molbuilder.scheduler.record import _parse_gres
        assert _parse_gres("gpu:gh200:1") == (1, "gh200")
        assert _parse_gres("gpu:a100.40gb:4") == (4, "a100.40gb")
        assert _parse_gres("gpu:4") == (4, None)
        assert _parse_gres("(null)") == (None, None)
