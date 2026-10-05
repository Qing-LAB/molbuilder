"""P4b — prep renders the transport composite's five stages
(`engines/transport.md` § 1 + § 3, the arm in `jobset/prep.py` +
`transport/stages.py`).

A fixture junction preps end-to-end; each gate refuses its mutation; the
emitter's own order-preflight never fires, because prep sorted first
(§ 4, *"in the composite it cannot fire in anger"*).

Properties under guard, each named for its failure:

* each stage's deck is born in its own stage directory, wrapper beside
  it, through the SHARED prep tail (job-set merge, STAGE-PLAN, run
  dirs) — no forked machinery;
* the electronic contract (basis · XC · energy shift · mesh · k ·
  electronic T) is read from the CITED attempt's own deck and lands in
  every stage's deck — one template governs (§ 5's invariant set, baked
  identically into all three fdfs; `transport/citation_defaults.py`,
  fdf-is-truth);
* the electrode deck's SystemLabel IS the ``.TSHS`` stem the device
  deck references — one spelling, both writers;
* the emitter's order-preflight never fires: a source whose atom order
  would trip it preps clean, because prep sorted first (§ 4);
* buffer atoms emit ``TS.Atoms.Buffer`` + explicit electrode positions
  (§ 4, the ``buffer`` label: with padding outermost, TranSIESTA's default
  first-N/last-N placement no longer holds);
* the composed record is written once, reused by later stages, travels
  with the folder, and a re-pointed citation recomposes;
* refusals: unnamed/unknown/disabled stage, a sweep, a moved frozen
  atom, a pseudopotential the citation cannot supply (§ 3.1 — a citation
  names a directory, so the directory must hold what the stage consumes).
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from conftest import write_pseudos
from molbuilder.transport.sort import (REGION_BRIDGE, REGION_BUFFER,
                                         REGION_LEFT_ELECTRODE,
                                         REGION_RIGHT_ELECTRODE)
from molbuilder.jobset.prep import prep_calculation as _prep_calculation
from molbuilder.structure import Structure
from test_transport_compose import _BRIDGE, _LAYERS_L, _LAYERS_R

_STAGES = ("seed", "electrode_L", "electrode_R", "device", "transmission")


def _says(text: str, keyword: str, value: str) -> bool:
    """Does *text* set *keyword* to *value*?  **Whitespace-insensitive.**

    The column alignment of a deck line belongs to whoever wrote it: the
    hand-written transport emitters padded to a fixed column, the framework's
    syntax door (`siesta/layout.py::line`) does not, and libfdf cares about
    neither.  Asserting the PAIR rather than the spacing is what lets these
    tests mean the same thing before and after a rung moves onto the seam
    (`engines/transport.md` § 3.6) -- otherwise every migrated rung breaks a
    science assertion for a reason that has nothing to do with science.
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

    `250` and `250.0` are the same mesh cutoff.  They differ because the
    framework's syntax door formats from the item's DECLARED type (a float)
    while the hand-written emitters it is replacing formatted the Python
    value they happened to hold (an int) -- so during the migration one
    rung says one and another says the other, for a value they agree on.

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
    # THE CELL MUST CONTAIN THE ATOMS, and with buffers it did not: the
    # buffer padding sits at z = -5 .. 39.5, a 44.5 A span in a 40 A box, so
    # atoms overlapped their own periodic images along the transport axis.
    # It went unnoticed because the device rung had NO settings gate until
    # TR5b put it on the seam -- the first time anything looked.  Sized from
    # the geometry so it cannot drift again.
    #
    # AND IT IS THE STRUCTURE A TRANSPORT RUN CAN USE (M5 step 2, § 6.1c).
    # The leads continue through the transport boundary into the image, so
    # the room there is ONE of the lead's layer spacings -- this fixture left
    # 5.5 A against a 2.5 A spacing, a missing layer the gate now refuses.
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


def prep_calculation(base, stage=None, **kw):
    """`prep`'s five steps, handed the run card's shape the way the prep
    entry hands it (`prep_run_inputs` -> ``chosen``, in `Resources`' own
    words) -- these tests drive the steps below the entry, which reads the
    card.  A test's own ``chosen`` speaks over it, field by field."""
    kw["chosen"] = {"mpi_np": _RUN_CARD["mpi_np"],
                    "cpus_per_task": _RUN_CARD["omp_threads"],
                    **(kw.get("chosen") or {})}
    return _prep_calculation(base, stage, **kw)


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
    # one for a transport description since 2026-09-16 (TR1), and a fixture
    # that skipped it would stop matching what this claims to reproduce.
    from molbuilder.transport.citation_defaults import (
        transport_template_text)
    (dest / "T.template.toml").write_text(
        transport_template_text(root / cite, label="T"))
    return dest


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path_factory):
    """Same sandbox as test_prep_calculation: the wrapper writer must
    read the fixture's bundle-scoped config, never this repo's."""
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))


class TestFormBContract:
    """4.1b: a labeled-pair citation has no deck, so the electronic
    contract is the description's own -- the contract's items are
    ordinary overrides and land in the rendered deck."""

    def test_an_open_contract_field_reaches_the_deck(self, tmp_path):
        """SCIENCE. A form-B citation carries no deck, so nothing dictates its
        electronic description — and the person must still be able to state it.

        **The mechanism changed on 2026-09-16 and the concern did not.** This
        test used to set `basis_size` as a STAGE OVERRIDE, because that was the
        only way in: the contract fields were sealed for form A and opened for
        form B, so the override lane was where a pair's basis could be said.
        Its worry was exact — *"if the field stayed sealed there would be no
        way to state the basis at all"*.

        `engines/transport.md` § 2a.7 answered it better. Every transport
        calculation now carries a TEMPLATE: for form A it is filled from the
        cited deck, for form B from the catalogue's own defaults, and either
        way the person may change it. So the basis is stateable for a pair —
        in the place where it applies to all five rungs at once, rather than
        on one rung where it could make the device disagree with its leads.

        Contract: `engines/transport.md` § 3.1 (form A vs form B) + § 2a.7.
        """
        from molbuilder.workingcopy_structure import StructureCodec
        root = tmp_path / "projects"
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
        prep_calculation(dest, "seed")
        deck = (dest / "01_seed" / "T_01_seed.fdf").read_text()
        assert _says(deck, "PAO.BasisSize", "TZP"), (
            "a pair's electronic description is the person's to state, and "
            "the template is where they state it")


