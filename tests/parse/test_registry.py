"""L2 tests for molbuilder.parse — the unified parse module.


Pins docs/model/parse.md (registry + dispatch) and
§ 3 (ParseResult discriminators).  A SIESTA output's detection and parse are
read off a run made on the road, `tests/test_siesta_optimization_run_e2e.py`.
"""

from __future__ import annotations

import pytest

from molbuilder.parse import UnknownFormatError, detect
from molbuilder.parse.registry import _registered_file_parsers


# Detection + dispatch ------------------------------------------------- #


def test_a_file_nothing_claims_is_told_every_format(tmp_path):
    """A file no parser claims is refused, and the refusal names every format
    molbuilder reads with its hint: it is how a person learns which file to
    pick up (`model/parse.md` § 3, `detect`).  THE SILENT FAILURE: the list
    losing its labels or its hints, the refusal still raised."""
    nonsense = tmp_path / "garbage.txt"
    nonsense.write_text("just some random text\n")
    with pytest.raises(UnknownFormatError) as ei:
        detect(nonsense)
    msg = str(ei.value)
    for parser in _registered_file_parsers():
        assert parser.label in msg
        assert getattr(parser, "hint", "") in msg


def test_detect_refuses_a_directory_by_name(tmp_path):
    """A run folder is the run door's (`model/parse.md` § 3): the registry
    says so rather than trying a file parser on a directory."""
    with pytest.raises(UnknownFormatError, match="run door"):
        detect(tmp_path)


# ---- a parser must not claim what it cannot read ---------------------- #


def test_nothing_shipped_beside_a_job_is_claimed_as_its_output(tmp_path):
    """Every attempt holds the monitor's one file, `mb_monitor.pyz` -- a zip of
    the framework modules it runs on (`runwrap.MONITOR_BUNDLE`) -- beside the
    engine's own files, and those modules QUOTE the engines' lines:
    `siesta_reader.py` names ``Begin Broyden opt. move`` in its docstring, a
    SIESTA content marker.  When the modules stood beside the deck as files
    (until 2026-09-26) the registry claimed that one as a SIESTA output and
    the Results tab listed it openable.  Neither the bundle nor any Python
    file is a run's output -- a module unpacked from it, or the
    charge-correction script prep writes beside a charged SIESTA deck, whose
    own text quotes SIESTA's cell lines.

    API-level over the registry, on the real shipped bytes and sources: the
    Results listing asks exactly this of each file.

    MUTATION THIS MUST FAIL AGAINST: drop the SIESTA sniffer's ``.py`` guard.
    """
    from molbuilder.parse import detect
    from molbuilder.parse.errors import UnknownFormatError
    from molbuilder.runwrap import (MONITOR_BUNDLE, companion_source,
                                    monitor_bundle, monitor_companions)

    (tmp_path / MONITOR_BUNDLE).write_bytes(monitor_bundle())
    with pytest.raises(UnknownFormatError):
        detect(str(tmp_path / MONITOR_BUNDLE))
    claimed = {}
    for name in monitor_companions():
        (tmp_path / name).write_text(companion_source(name), encoding="utf-8")
        try:
            claimed[name] = detect(str(tmp_path / name)).name
        except UnknownFormatError:
            pass
    # THE CORRECTION SCRIPT, as its one writer renders it beside a deck.
    from molbuilder.runfiles import MAKOV_PAYNE_SCRIPT
    from molbuilder.siesta.makov_payne import render_correction_script
    script = tmp_path / MAKOV_PAYNE_SCRIPT
    script.write_text(render_correction_script("H2", 1, stage="01_coarse"),
                      encoding="utf-8")
    try:
        claimed[script.name] = detect(str(script)).name
    except UnknownFormatError:
        pass
    assert not claimed, claimed


#: EACH XYZ FILE AND THE READER THAT IS FOR IT, by its name
#: (`model/parse.md` § 4): ``(name, reader)``.
_XYZ_READERS = [
    ("H2_initial.xyz", "pyscf-geom"),            # a PySCF deck's input, echoed back
    ("H2_optimized.xyz", "pyscf-geom"),          # ...and the geometry it reached
    ("H2_01_coarse_geom_optim.xyz", "pyscf"),    # geomeTRIC's trajectory
    ("H2.xyz", "xyz"),                           # SIESTA's own geometry file
    ("H2_01_coarse.ANI", "xyz"),                 # SIESTA's steps, as an animation
    ("H2.source.xyz", "xyz"),                    # the structure a calculation is of
    ("water.xyz", "xyz"),                        # a saved structure
]


@pytest.mark.parametrize("name,reader", _XYZ_READERS,
                         ids=[n for n, _r in _XYZ_READERS])
def test_each_xyz_is_read_by_the_reader_that_is_for_it(tmp_path, name, reader):
    """Each XYZ file is read by the reader that is for it, by its name, and
    by that one alone, so `detect`, which is exactly-one-or-raise, opens it:
    a PySCF deck's two geometries by `pyscf-geom`, geomeTRIC's trajectory by
    `pyscf`, any other XYZ by `xyz`.  The file is the structure's own
    writer's.

    THE SILENT FAILURES: the Results folder row naming no parser for a PySCF
    deck's pair, which then cannot be opened on its own; SIESTA's own
    geometry file and a calculation's structure read by geomeTRIC's reader,
    so the trajectory viewer claimed them as a run's steps -- SIESTA's
    folded into its run's progress log and never listed."""
    from molbuilder.structure import Structure
    water = Structure(elements=["O", "H", "H"],
                      positions=[[0.0, 0.0, 0.119], [0.0, 0.757, -0.477],
                                 [0.0, -0.757, -0.477]])
    path = tmp_path / name
    path.write_text(water.to_xyz())
    assert detect(str(path)).name == reader
