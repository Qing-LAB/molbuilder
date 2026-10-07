"""The reporting policy rides ``Resources`` from `task.json` to the run
script, and survives a job-set file on the way.

**Why the policy rides ``Resources``.**  It is not a scheduler ask and
becomes no ``sbatch`` directive — like ``continue_retries``, which the class
docstring keeps there deliberately: *"this is the road every field the deck
never carries already rides… the alternative was a second, hand-maintained
road from a job to its wrapper."*  That road has lost a field to a
hand-copied argument list twice (``max_memory_mb``, then the ranks/cores
pair), which is the whole argument for not opening a third one.

**And what must never ride it: the destination.**  The URL and its
credential are the user's own file on the machine that runs the job.  A
wrapper is written to disk, copied into composed copies and read by anyone
who can see the run directory; a token in one would be a token published.
"""
from __future__ import annotations


# --------------------------------------------------------------------- #
#  which channels -- names, and the two ways of saying none              #
# --------------------------------------------------------------------- #

def test_the_names_survive_a_job_set_file(tmp_path):
    """**A tuple out, a tuple back.**

    `Resources.to_dict` is `asdict`, so a job-set file stores the names as a
    JSON array and `from_dict` hands them back as a LIST -- which never
    equals the tuple it was written from.

    It breaks QUIETLY, which is why this test exists rather than a comment:
    the names still reach the wrapper either way, and only equality lies --
    so the symptom would surface somewhere far from the cause.
    """
    import json as _json
    from molbuilder.jobset.model import Resources

    for channels in (("slack", "lab"), (), None):
        r = Resources(mpi_np=4, notify_channels=channels)
        back = Resources.from_dict(_json.loads(_json.dumps(r.to_dict())))
        assert back == r, f"{channels!r} did not survive the file"
        assert back.notify_channels == channels


def test_a_list_of_names_is_accepted_and_normalised():
    """Several roads reach `Resources` (the CLI, `prep`'s fold, a job-set
    file). A caller handing a list is not wrong; the class holds its own invariant, exactly as it does for
    `time` and `mem`."""
    from molbuilder.jobset.model import Resources
    assert Resources(notify_channels=["a", "b"]).notify_channels == ("a", "b")


# --------------------------------------------------------------------- #
#  the flags the monitor actually has                                    #
# --------------------------------------------------------------------- #


# --------------------------------------------------------------------- #
#  what the job holds, and whether it can be watched                     #
# --------------------------------------------------------------------- #

