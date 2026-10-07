"""P4b — prep renders the transport composite's five stages
(`engines/transport.md` § 1 + § 3, the arm in `jobset/prep.py` +
`transport/stages.py`).

"""
from __future__ import annotations

import re

import numpy as np
import pytest

from conftest import write_pseudos
from molbuilder.transport.sort import (REGION_BRIDGE, REGION_BUFFER,
                                         REGION_LEFT_ELECTRODE,
                                         REGION_RIGHT_ELECTRODE)
from support.road import jobset
from molbuilder.structure import Structure
from test_transport_compose import _BRIDGE, _LAYERS_L, _LAYERS_R

_STAGES = ("seed", "electrode_L", "electrode_R", "device", "transmission")


def _says(text: str, keyword: str, value: str) -> bool:
    """Does *text* set *keyword* to *value*?  **Whitespace-insensitive.**

    The column alignment of a deck line belongs to whoever wrote it, and
    libfdf cares about none of it.
    """
    want = (keyword + " " + value).split()
    head = re.escape(want[0])
    for line in text.splitlines():
        got = line.split()
        if not got or not re.fullmatch(head, got[0]) or len(got) != len(want):
            continue
        if all(_same_token(a, b) for a, b in zip(got[1:], want[1:])):
            return True
    return False


def _same_token(got: str, want: str) -> bool:
    """One token of a deck line, compared by VALUE where it is a number.

    `250` and `250.0` are the same mesh cutoff.

    Asserting the spelling would make a test fail for a reason that has
    nothing to do with the science, which is the same trap the column
    alignment set (see `_says`).
    """
    try:
        return float(got) == float(want)
    except ValueError:
        return got == want


#: The fixture's leads are a CHAIN, one gold atom per layer, so they are
#: ISOLATED across the transport axis in the 8 Å box they sit in -- what the
#: relaxation deck records and `compose` carries (`engines/transport.md`
#: § 6.1c).  One layer spacing is the room the transport boundary leaves.
_ACROSS = ("isolated", "isolated")
_SPACING = _LAYERS_L[1] - _LAYERS_L[0]


def _junction_struct(*, order="canonical", buffers=False, across=_ACROSS,
                     width=8.0, room=None):
    """The BDT-ish fixture sandwich; ``order="scrambled"`` writes the
    same geometry with the bridge FIRST and the leads swapped after it
    — exactly the order the emitter's preflight refuses.

    *across* and *width* are the transverse axes' kinds and length: the
    chain in an 8 Å box is a wire, isolated across; ``across=("periodic",
    "periodic"), width=_SPACING`` is the same chain as a lattice it tiles --
    the reading a relaxation deck from before the placement record gets
    (`compose._junction_axis_kind`).  *room* is what the transport boundary
    leaves, one layer spacing unless a test opens it."""
    rows = []       # (element, z, label)
    for z in _LAYERS_L:
        rows.append(("Au", z, REGION_LEFT_ELECTRODE))
    for el, z in _BRIDGE:
        rows.append((el, z, REGION_BRIDGE))
    for z in _LAYERS_R:
        rows.append(("Au", z, REGION_RIGHT_ELECTRODE))
    if buffers:
        for z in (-5.0, -2.5, 37.0, 39.5):
            rows.append(("Au", z, REGION_BUFFER))
    if order == "scrambled":
        rows = ([r for r in rows if r[2] == REGION_BRIDGE]
                + [r for r in rows if r[2] == REGION_RIGHT_ELECTRODE]
                + [r for r in rows if r[2] == REGION_LEFT_ELECTRODE]
                + [r for r in rows if r[2] == REGION_BUFFER])
    elements = [r[0] for r in rows]
    positions = np.array([[1.0, 1.0, r[1]] for r in rows])
    regions: dict = {}
    for i, r in enumerate(rows):
        regions.setdefault(r[2], []).append(i)
    frozen = [i for i, r in enumerate(rows)
              if r[2] in (REGION_LEFT_ELECTRODE, REGION_RIGHT_ELECTRODE)]
    # THE CELL MUST CONTAIN THE ATOMS (the buffer padding sits at
    # z = -5 .. 39.5), so it is sized from the geometry.
    #
    # AND IT IS THE STRUCTURE A TRANSPORT RUN CAN USE (M5 step 2, § 6.1c).
    # The leads continue through the transport boundary into the image, so
    # the room there is ONE of the lead's layer spacings.
    # And the leads are a CHAIN, one gold atom per layer, in an 8 A box: a
    # wire, isolated across transport, which is what it states -- periodic
    # there would say the chain tiles a plane it does not.
    zs = positions[:, 2]
    c = float(zs.max() - zs.min()) + (_SPACING if room is None else room)
    return Structure(elements=elements, positions=positions,
                     regions=regions, frozen_atoms=frozen,
                     cell=np.diag([width, width, c]),
                     axis_kind=(*across, "transport"))


#: THE LAUNCH SHAPE THESE DESCRIPTIONS STATE -- sixteen ranks of one thread,
#: the default test machine's width -- on the run card, as a described
#: calculation states it: a run script is not written for an unstated one
#: (`architecture.md` § 5.2).
_RUN_CARD = {"mpi_np": 16, "omp_threads": 1}


def _describe_transport(root, *, cite, bias=(0.0, 0.2)):
    from molbuilder.task import Stage, Task, derive_run, write_task
    dest = root / "J" / "transport" / "T"
    dest.mkdir(parents=True, exist_ok=True)
    write_task(dest / "task.json", Task(
        engine="siesta", shape="hierarchical",
        run=derive_run("T", cite, stage_names=_STAGES),
        structure=None, calculation="transport",
        slots={"junction": cite}, bias=bias, varies=(),
        execution=dict(_RUN_CARD),
        stages=tuple(Stage(name=n, enabled=True, overrides={})
                     for n in _STAGES)))
    # THE TEMPLATE, through the product's own doors -- `jobset init` writes
    # one for a transport description, and a fixture that skipped it would
    # stop matching what this claims to reproduce.
    from molbuilder.transport.citation_defaults import (
        transport_template_text)
    (dest / "T.template.toml").write_text(
        transport_template_text(root / cite, label="T"))
    return dest


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path_factory):
    """Same sandbox as test_prep_calculation."""
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))


class TestFormBContract:
    """4.1b: a labeled-pair citation has no deck, so the electronic
    contract is the description's own -- the contract's items are
    ordinary overrides and land in the rendered deck."""

    def test_an_open_contract_field_reaches_the_deck(self, tmp_path,
                                                     monkeypatch):
        """SCIENCE. A form-B citation carries no deck, so nothing dictates its
        electronic description — and the person must still be able to state it.

        Every transport calculation carries a TEMPLATE (`engines/transport.md`
        § 2a.7): for form A it is filled from the
        cited deck, for form B from the catalogue's own defaults, and either
        way the person may change it. So the basis is stateable for a pair —
        in the place where it applies to all five rungs at once, rather than
        on one rung where it could make the device disagree with its leads.

        Contract: `engines/transport.md` § 3.1 (form A vs form B) + § 2a.7.
        """
        from molbuilder.projects import PROJECTS_ROOT_ENV
        from molbuilder.workingcopy_structure import StructureCodec
        root = tmp_path / "projects"
        monkeypatch.setenv(PROJECTS_ROOT_ENV, str(root))
        pair = root / "J" / "structure" / "junc"
        pair.mkdir(parents=True)
        StructureCodec().write(_junction_struct(), pair / "junction.xyz")
        write_pseudos(pair, ["Au", "S", "C"])
        dest = _describe_transport(root, cite="J/structure/junc")
        # A pair has no deck, so the template starts from the catalogue's
        # defaults -- and the person changes it there.
        import dataclasses
        from molbuilder.task import read_task
        from molbuilder.template import _emit, find_template, read_template
        tmpl = find_template(dest, read_task(dest / "task.json").label)
        assert tmpl is not None, (
            "a form-B transport calculation carries a template too -- "
            "without one there would be nowhere to state the basis")
        parsed = read_template(tmpl.read_text())
        tmpl.write_text(_emit(
            [dataclasses.replace(i, value="TZP") if i.name == "basis_size"
             else i for i in parsed.items], engines=("siesta",)))
        r = jobset("prep", "run", "seed", "--bundle", dest)
        assert r.exit_code == 0, r.output
        deck = (dest / "01_seed" / "T_01_seed.fdf").read_text()
        assert _says(deck, "PAO.BasisSize", "TZP"), (
            "a pair's electronic description is the person's to state, and "
            "the template is where they state it")


