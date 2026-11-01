"""The wrapper's door onto how a run ended (`_run_ending.main`, beside a job
as ``mb_monitor.pyz ending``) -- `job-contracts.md` § 2.6, *`_mb_ending`*.
"""
from __future__ import annotations

import pytest


@pytest.mark.parametrize("asked, said", [
    (["stopped-by"], "'stopped-by' takes 1 argument(s), given 0"),
    (["stopped-by", "a", "b"], "'stopped-by' takes 1 argument(s), given 2"),
    (["nope"], "not a question: 'nope'"),
], ids=["a-marker-missing", "an-argument-too-many", "not-a-question"])
def test_a_question_asked_wrong_is_said_as_usage(asked, said, tmp_path,
                                                 capsys):
    """`job-contracts.md` § 2.6: a question that is not one, or asked with
    other than its arguments, is exit 2 said as usage before anything is
    read -- the wrapper's own rendering at fault, never an ending.  The
    output named is not there: had the door read it, it would have said the
    ending cannot be read.  THE SILENT FAILURE: a rendering mistake reported
    as an output that does not read, sending a person to the wrong file."""
    from molbuilder.parse.engines._run_ending import main
    assert main([str(tmp_path / "JOB_01_coarse.out"), *asked]) == 2
    err = capsys.readouterr().err
    assert said in err, err
    assert "cannot be read" not in err, err
