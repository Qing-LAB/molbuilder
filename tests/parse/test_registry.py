"""L2 tests for molbuilder.parse — the unified parse module.


Pins docs/model/parse.md (registry + dispatch) and
§ 3 (ParseResult discriminators).  A SIESTA output's detection and parse are
read off a run made on the road, `tests/test_siesta_optimization_run_e2e.py`.
"""

from __future__ import annotations

from functools import partial

import pytest

from molbuilder.atom_permutation import LISTED_BY_ENGINE, LISTED_BY_USER
from molbuilder.parse import UnknownFormatError, detect
from molbuilder.parse.registry import _registered_file_parsers
from support.road import (CO2_HELD_FIRST, H2, co2_in_engine_order,
                          road_structure)


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


#: EACH XYZ FILE AND THE READER THAT IS FOR IT, by its name and whether its
#: writer puts a sidecar beside it (`model/parse.md` § 4): ``(name, reader,
#: pair)`` -- ``pair`` the file written as the codec writes it, with its
#: sidecar; else the geometry alone, as an engine or another tool writes it.
_XYZ_READERS = [
    ("H2_initial.xyz", "pyscf-geom", True),         # a PySCF deck's input, echoed back
    ("H2_optimized.xyz", "pyscf-geom", True),       # ...and the geometry it reached
    ("H2_01_coarse_geom_optim.xyz", "pyscf", False),  # geomeTRIC's trajectory
    ("H2.xyz", "xyz", False),                       # SIESTA's own geometry file
    ("H2_01_coarse.ANI", "xyz", False),             # SIESTA's steps, as an animation
    ("H2.source.xyz", "structure-pair", True),      # the structure a calculation is of
    ("water.xyz", "structure-pair", True),          # a saved structure
]


@pytest.mark.parametrize("name,reader,pair", _XYZ_READERS,
                         ids=[n for n, _r, _p in _XYZ_READERS])
def test_each_xyz_is_read_by_the_reader_that_is_for_it(tmp_path, name, reader,
                                                       pair):
    """Each XYZ file is read by the reader that is for it, by its name, and
    by that one alone, so `detect`, which is exactly-one-or-raise, opens it:
    a PySCF deck's two geometries by `pyscf-geom`, geomeTRIC's trajectory by
    `pyscf`, a `.xyz` with its sidecar by `structure-pair`, any other XYZ by
    `xyz`.  The file is the structure's own writer's.

    THE SILENT FAILURES: the Results folder row naming no parser for a PySCF
    deck's pair, which then cannot be opened on its own; SIESTA's own
    geometry file and a calculation's structure read by geomeTRIC's reader,
    so the trajectory viewer claimed them as a run's steps -- SIESTA's
    folded into its run's progress log and never listed; a saved pair read
    by `xyz` alone, its cell and labels left behind (to 2.69.19)."""
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec
    water = Structure(elements=["O", "H", "H"],
                      positions=[[0.0, 0.0, 0.119], [0.0, 0.757, -0.477],
                                 [0.0, -0.757, -0.477]])
    path = tmp_path / name
    if pair:
        StructureCodec().write(water, path)
    else:
        path.write_text(water.to_xyz())
    assert detect(str(path)).name == reader


#: A PAIR AS ITS WRITER WRITES IT -- ``(name, made by, frames, listed by,
#: order record)``: a saved structure, the road's H2 (an atom held, a 10 Å
#: box); a frame set of it, as a script makes one through the Structure API
#: (`model/structure.md` § 2.2e); an engine's record -- the road's CO2 sorted
#: as a SIESTA vibration's decks hold it, written by the composer's one door
#: (`write(..., engine_record=True)`), its atoms listed by engine index and
#: its order record the road's.
_PAIRS = [
    ("a saved structure", partial(road_structure, H2), 1, LISTED_BY_USER, None),
    ("a frame set", partial(road_structure, H2), 3, LISTED_BY_USER, None),
    ("an engine's record", co2_in_engine_order, 1, LISTED_BY_ENGINE,
     CO2_HELD_FIRST),
]


@pytest.mark.parametrize("made_by,frames,listed_by,order",
                         [row[1:] for row in _PAIRS],
                         ids=[row[0] for row in _PAIRS])
def test_a_saved_pair_is_read_whole(tmp_path, made_by, frames, listed_by,
                                    order):
    """`model/structure.md` § 2.3: the pair is the file.  Read through the
    registry (`postprocess.parse`), a `.xyz` with its sidecar answers what
    the pair states -- every frame, its labels, its box (the result's
    `cell`, `model/parse.md` § 2), an engine's record its order record and
    its atoms listed by engine index.  THE SILENT FAILURE (to 2.69.19): the
    `xyz` reader read the `.xyz` alone, as bare frames with no box and no
    labels (the user, 2026-11-10: read the pair whole)."""
    import numpy as np
    from molbuilder.postprocess import parse
    from molbuilder.workingcopy_structure import StructureCodec
    built = made_by()
    if frames > 1:
        built = built.with_frames(
            [built.positions + 0.01 * i for i in range(frames)])
    path = tmp_path / "s.xyz"
    StructureCodec().write(built, path, engine_record=order is not None)
    read = parse(path)
    got = read.structure
    assert (got.n_frames, got.regions) == (frames, built.regions)
    assert np.allclose([got.frame_at(i).positions for i in range(frames)],
                       [built.frame_at(i).positions for i in range(frames)],
                       atol=1e-6)
    assert np.allclose(read.cell, built.cell), read.cell
    assert (got.atom_order.to_dict() if got.atom_order else None) == order
    assert read.atom_index_map.listed_by == listed_by
