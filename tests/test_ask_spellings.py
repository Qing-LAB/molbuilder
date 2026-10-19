"""The one spelling a wall and a memory ask take on disk, and what each
edge accepts -- `scheduler/quantities.py`'s own rules (`scheduler.md` § 6a,
`stages.md` § 6.8a).  These five stood in
`test_ask_parsers_agree_across_surfaces.py` beside the parity rows that held
a JavaScript codec equal to the Python one; that codec went on 2026-10-18 (the
page writes what is typed and the save door canonicalises), the parity rows
with it, and these outcome tests moved here.
"""
from __future__ import annotations

import pytest

from molbuilder.scheduler.quantities import (canonical_mem, canonical_time,
                                            parse_duration, parse_memory)


#: Only the spellings that MEAN something -- canonicalising is what happens
#: to a value that parsed; a refusal is the reader's job.
_GOOD_TIMES = ["4h", "90m", "45", "1.5h", "4H", "7-00:00:00", "4:00:00",
               "00:15:00", "1-00:00:00", "168h", ""]
#: Includes a SUB-MEGABYTE value on purpose: the writer rounds to whole
#: megabytes, and rounding a positive ask down to 0 would hand SLURM its
#: "all the node's memory" spelling -- the smallest request becoming the
#: largest (2026-08-24).
_GOOD_MEMS = ["128G", "0.5T", "128", "512M", "1024K", "4t", "80GB",
              "128GB", "503.5G", "0.0001", "0.001", "0", ""]


def _py_canon(fn, raw):
    try:
        v = fn(raw)
    except ValueError:
        return raw.strip()          # the JS passes an unparseable through
    return v or ""


def test_canonical_is_what_sbatch_takes():
    """The point of the whole rule: `-t 4h` is not a thing SLURM accepts."""
    import re
    assert canonical_time("4h") == "0-04:00:00"
    assert re.fullmatch(r"(?:\d+-)?\d+(?::\d{2}){0,2}",
                        canonical_time("4h"))
    assert canonical_mem("80GB") == "80G"
    assert re.fullmatch(r"\d+(?:\.\d+)?[KMGT]?", canonical_mem("80GB"))


def test_canonicalising_is_idempotent():
    """A record read and rewritten must not change -- otherwise every save
    rewrites the file and no two copies of it ever match."""
    for v in _GOOD_TIMES:
        once = canonical_time(v)
        assert canonical_time(once) == once
    for v in _GOOD_MEMS:
        once = canonical_mem(v)
        assert canonical_mem(once) == once


def test_zero_memory_survives_because_slurm_means_something_by_it():
    """`--mem`'s own help says *"'0' asks for all of the node's"*, and
    `parse_memory` refuses 0 because zero gigabytes is not an amount to fit
    a queue against.  Those are two different questions and the writer must
    not inherit the reader's refusal."""
    assert canonical_mem("0") == "0"
    with pytest.raises(ValueError):
        parse_memory("0")



def test_slurms_own_spelling_is_read_by_BOTH():
    """The concrete case: it is what a queue's ceiling looks like in the
    machine record, so the tab fills a field with it -- and a value the tool
    writes must be one the tool can read back."""
    assert parse_duration("7-00:00:00") == 7 * 86400


def test_the_spelling_prep_advertises_is_accepted_by_launch():
    """`prep --mem`'s help says *"e.g. 80GB"*.  It passed the string through
    unparsed while `launch --mem` refused it."""
    assert parse_memory("80GB") == 80.0
