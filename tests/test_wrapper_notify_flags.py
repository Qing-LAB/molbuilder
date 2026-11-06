"""``Resources.notify_channels`` survives a job-set file."""
from __future__ import annotations


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
