"""The wrapper's door onto how a run ended (`_run_ending.main`, beside a job
as ``mb_monitor.pyz ending``) -- `job-contracts.md` § 2.6, *`_mb_ending`*.
"""
from __future__ import annotations

import pytest


@pytest.mark.parametrize("asked, said", [
    (["out-of-time", "a"], "'out-of-time' takes no argument, given 1"),
    (["advice", "a"], "'advice' takes no argument, given 1"),
    (["nope"], "not a question: 'nope'"),
], ids=["a-question-given-an-argument", "what-is-said-given-an-argument",
        "not-a-question"])
def test_a_question_asked_wrong_is_said_as_usage(asked, said, tmp_path,
                                                 capsys):
    """`job-contracts.md` § 2.6: a question that is not one, or asked with
    an argument, is exit 2 said as usage before anything is
    read -- the wrapper's own rendering at fault, never an ending.  The
    output named is not there: had the door read it, it would have said the
    ending cannot be read.  THE SILENT FAILURE: a rendering mistake reported
    as an output that does not read, sending a person to the wrong file."""
    from molbuilder.parse.engines._run_ending import main
    assert main([str(tmp_path / "JOB_01_coarse.out"), *asked]) == 2
    err = capsys.readouterr().err
    assert said in err, err
    assert "cannot be read" not in err, err


@pytest.mark.parametrize("verdict", [("geometry", True), ("geometry", False),
                                     ("SCF", True), ("SCF", False), None],
                         ids=["geometry yes", "geometry NO", "SCF yes",
                              "SCF NO", "none"])
def test_a_verdicts_kept_words_read_back_as_the_verdict(verdict):
    """`engines/transport.md`, *What is said about how it ended*: a record
    keeps a run's verdict in its words (`verdict_words`), and the module that
    writes them reads them back (`verdict_read`) -- what the record serves a
    page as data.  THE SILENT FAILURE (to 2.69.18): the Transport report
    decided "not converged" by testing the words itself."""
    from molbuilder.parse.engines._run_ending import (verdict_read,
                                                       verdict_words)
    assert verdict_read(verdict_words(verdict)) == verdict
