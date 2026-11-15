"""Every fdf label a SIESTA deck writes, checked against the BINARY --
a transport calculation's rungs, an optimization's and a vibration's calls.

`plan.md` § 5o. Seven keywords shipped that SIESTA 5.4.2 cannot read —
`TS.TBT.Emin`, `TS.TBT.Emax`, `TS.TBT.NumE`, `TS.TBT.Erange.RelToEF` and
`TS.ComplexContour.NumCircle|NumLine|Emin`, all SIESTA-3.x spellings. **fdf
ignores a label nobody queries**, so nothing failed: the run completed and
T(E) came out on tbtrans's default energy grid while the form said otherwise.

That is the whole reason this file exists. The keywords were believed correct
because a comment said so — the emitter's own docstring read *"its keyword
names didn't migrate in 4.1+"* — and a belief is what a test replaces.

**The method.** SIESTA compiles its fdf labels into the binary as literal
strings, at the `fdf_get` call sites, so *whether this installation can read
a label* is a countable fact rather than a recollection. This walks the
labels a real deck emits and greps the engine's own binary for each.

**Two rules make the comparison honest, and both were measured on 2026-09-16.**
fdf matches labels through `fdf_utils::packlabel`, which drops `.`, `-` and
`_` and ignores case — so the check packs both sides, and `WriteForces` is
recognised as the `Write.Forces` the binary compiles. And a label assembled at
runtime from a prefix does not appear whole, so the four such forms are listed
below WITH the literal the binary must carry (`TS.ChemPots` counts zero while
`ChemPots` is present) rather than excused by a stem rule, which would let
`TS.TBT.Emin`, `TS.TBT.Emax`, `TS.TBT.NumE` and `TS.ComplexContour.Emin`
through, because `Emin` and `NumE` occur inside other keywords SIESTA does know.

**What it covers.** One case per deck SHAPE of the ladder that actually runs.

**WHAT IT CANNOT DO, and the example is worth keeping.** A label being in the
binary means SIESTA can read it SOMEWHERE -- not that it reads it in the deck
you wrote. `TS.Contours.Eq.Pole.N` is in the binary, this file passes it, and
it is a real `fdf_get` (`Src/m_ts_chem_pot.F90:113`) -- and on the deck shape
this project emits it can never take effect, because the continued-fraction
branch overwrites the count from the pole ENERGY two hundred lines later
(`:319`). A deck asking for 40 poles gets the engine's 42.

And a misspelling that is the START of a keyword the binary holds passes:
SIESTA's literals run together in its strings, so a label is looked for INSIDE
them, and `ElectronicTemp` is found inside `ElectronicTemperature` (measured
2026-11-15; `ElectronTemperature`, found nowhere, fails as it should).

So a green result here means "spelled in a way this build knows", which is the
`plan.md` § 5o failure and worth catching. It does not mean "has the effect you
intended". Only the manual, the source, or a run settles that -- and for
anything with a numerical consequence, the source.

**It skips rather than fails when the env is absent**, the way every
env-dependent test here does: a laptop without `molbuilder-siesta` has
nothing to measure, and a skipped alarm is honest where a green one is not.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from _road import env_bin
from support.real_runs import CO2_OPTICAL
from support.road import CO2, H2
from support.tooling import needs

#: Labels that are NOT fdf keywords and must not be looked up: block
#: delimiters, block-interior grammar words, and comments.  Each is read by
#: the block's own parser, not by `fdf_get`.
_NOT_A_LABEL = {
    "part", "from", "to", "points", "delta", "method", "mu",
    "chem-pot", "used-atoms", "elec-pos", "bloch", "semi-inf-direction",
    "HS", "begin", "end", "GF", "GF.ReUse",
}


def _engine_binaries():
    """The molbuilder-siesta env's own `siesta` and `tbtrans`
    (`_road.env_bin`): every test reading them asks for a run made in the
    env, whose need that run states."""
    return [env_bin("molbuilder-siesta") / b for b in ("siesta", "tbtrans")]


def _strings(path: Path) -> str:
    out = subprocess.run(["strings", str(path)], capture_output=True,
                         text=True, timeout=120)
    # A binary that is not there, or not read, fails saying so -- never as
    # a deck whose every label the binary lacks.
    assert out.returncode == 0 and out.stdout, (path, out.stderr)
    return out.stdout


@pytest.fixture(scope="module")
def binary_text():
    return "\n".join(_strings(p) for p in _engine_binaries())


def _emitted_labels(text: str):
    """The fdf labels a rendered deck carries: `Label value` and `%block X`."""
    labels = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"%(?:end)?block\s+([A-Za-z][\w.\-]*)", line)
        if m:
            labels.add(m.group(1))
            continue
        m = re.match(r"([A-Za-z][\w.\-]*)\s", line)
        if m and m.group(1) not in _NOT_A_LABEL:
            labels.add(m.group(1))
    return labels


#: Labels fdf never sees whole, because the deck NAMES them and the engine
#: assembles the lookup from a prefix plus that name: `%block TS.Elec.L`,
#: `%block TS.ChemPot.Left`, `%block TBT.Contour.window`.  The prefix is the
#: part the binary must know; the tail is the deck's own word.  Four entries,
#: rather than a rule that waves through every label ending in a known word.
#:
#: Each entry maps the deck's spelling to **what the binary must contain** --
#: which is not always the prefix itself: TranSIESTA prepends `TS.` to the
#: chempot list at runtime, so the compiled literal is `ChemPots`.  The probe
#: is checked like any other label, so a typo in the fixed half (`TS.Elecs.L`)
#: is still caught.
_ASSEMBLED_PREFIXES = {
    "TS.Elec.":     "TS.Elec.",     # one block per electrode
    "TS.ChemPot.":  "TS.ChemPot.",  # one block per chemical potential
    "TBT.Contour.": "TBT.Contour.",  # one block per TBtrans energy contour
    "TS.ChemPots":  "ChemPots",     # the list block; `TS.` is added at runtime
    # THE BIAS WINDOW'S LINE (`engines/transport.md` § 2a.11): the engine
    # assembles `TS.Contours.nEq` and `TS.Contour.nEq.<name>` from a prefix
    # and the part (`m_ts_io_ctype.f90`); the literals it holds are the tails.
    "TS.Contours.nEq": "Contours.nEq",
    "TS.Contour.nEq.": "Contour.nEq.",
}


def _packed(text: str) -> str:
    """The binary's strings as fdf COMPARES them: separators dropped, lowercased.

    fdf does not match a label literally.  `fdf_utils::packlabel` strips ``.``,
    ``-`` and ``_`` and the comparison is case-insensitive, which is why
    ``SystemLabel``, ``system-label`` and ``system_label`` are one keyword.

    Without this the check reports a label that WORKS.  Measured 2026-09-16:
    every molbuilder SIESTA deck writes ``WriteForces``, the binary contains
    only ``Write.Forces``, and a stem comparison calls that missing.  Run
    against a real SIESTA it is fine -- its own fdf log answers
    ``Write.Forces  T`` with no "# default value" marker, i.e. it read the
    undotted spelling out of the deck.  A guard that cries wolf on a correct
    keyword is how a real miss gets waved through.
    """
    return re.sub(r"[.\-_]", "", text).lower()


def _refuse_unknown_labels(deck: str, binary_text: str, where: str) -> None:
    """Every label WHOLE, not by its last word.

    A stem rule (`Emin` for `TS.TBT.Emin`) is blind to the keywords in
    `plan.md` § 5o, because `Emin`, `Emax` and `NumE` all occur inside OTHER
    keywords the binary does know.
    """
    packed = _packed(binary_text)
    missing = []
    for label in sorted(_emitted_labels(deck)):
        probe = next((v for k, v in _ASSEMBLED_PREFIXES.items()
                      if label == k or label.startswith(k)), label)
        if probe and _packed(probe) not in packed:
            missing.append(f"{label}  ({probe!r} absent)")
    assert not missing, (
        f"the {where} deck writes labels the installed SIESTA cannot read, "
        f"so they are SILENTLY IGNORED -- the run completes and the setting "
        f"does nothing (`plan.md` § 5o):\n  " + "\n  ".join(missing)
        + "\n\nIf a label is genuinely assembled at runtime from a prefix, "
          "read the site and add its stem to the binary's vocabulary here "
          "with the reason.")


def _prepped_deck(pair, tmp_path, monkeypatch, rung, groups) -> str:
    """The deck `jobset prep` wrote for ``rung`` of a transport calculation
    citing the real ``pair``, with a k-point grid of its own so a grid's block
    is written."""
    from support.road import (described, prep_groups, road_tree,
                              set_by_the_person, stage_file)
    bundle = described(road_tree(tmp_path, monkeypatch, pair), name="kwcheck",
                       calculation="transport")
    set_by_the_person(bundle, {"kgrid": (2, 2, 1)})
    prep_groups(bundle, *groups)
    return stage_file(bundle, rung, ".fdf").read_text()


@pytest.mark.parametrize("rung, groups", [
    ("seed", ("1",)), ("electrode_L", ("1",)), ("device", ("1", "2")),
    ("transmission", ("1", "2", "3")),
], ids=["seed", "electrode", "device", "transmission"])
def test_every_label_a_PREPPED_RUNG_emits_is_known_to_the_binary(
        binary_text, rung, groups, real_bridged_chain, tmp_path, monkeypatch):
    """THE DECKS THAT ACTUALLY RUN.

    One case per deck SHAPE rather than per rung: `SHAPE_OF_RUNG` maps the
    five rungs onto four texts -- the device and the transmission are two,
    because two programs read them (`transport.md` § 6.1b).
    """
    deck = _prepped_deck(real_bridged_chain.pair, tmp_path, monkeypatch, rung,
                         groups)
    assert "PAO.BasisSize" in deck, (
        "this rung carries none of the engine's section set -- the test "
        "would pass vacuously on a deck that had fallen off the seam")
    _refuse_unknown_labels(deck, binary_text, rung)


def test_the_device_deck_holds_no_label_siesta_cannot_read(
        real_bridged_chain, tmp_path, monkeypatch):
    """`siesta` alone runs the device deck, so every label in it is checked
    against `siesta`'s binary ALONE (`engines/transport.md` § 6.1b).  The
    check above pools both programs' strings, which is right for asking
    whether ANY installed binary knows a label, and blind to a `TBT.*` line
    finding its way back into the device deck -- a line `tbtrans` knows and
    `siesta`, which reads this deck, never will."""
    [siesta] = [b for b in _engine_binaries() if b.name == "siesta"]
    deck = _prepped_deck(real_bridged_chain.pair, tmp_path, monkeypatch, "device",
                         ("1", "2"))
    _refuse_unknown_labels(deck, _strings(siesta), "device (siesta alone)")


@needs("molbuilder-siesta")
@pytest.mark.parametrize("calculation, structure, template", [
    ("optimization", H2, None),
    ("vibration", CO2, CO2_OPTICAL),
], ids=["optimization", "vibration"])
def test_every_label_a_siesta_optimization_or_vibration_writes_is_known(
        calculation, structure, template, tmp_path, monkeypatch):
    """THE SAME FAILURE ON THE OTHER KINDS' DECKS (the module's head: a
    label SIESTA cannot read is silently ignored): the decks `jobset prep`
    writes for an optimization's first rung, and for a vibration's three
    calls -- the relaxation, the force constants and the optical call,
    Raman asked so its deck is written (`engines/vibration.md` § 5.11
    § 2) -- each checked against `siesta`'s binary alone, the one program
    that reads them."""
    from molbuilder.jobset.naming import stage_home
    from molbuilder.parse.fdf import read_deck
    from molbuilder.siesta.vibration_deck import PARTS
    from molbuilder.task import of as task_of
    from support.road import described_and_prepared
    [siesta] = [b for b in _engine_binaries() if b.name == "siesta"]
    known = _strings(siesta)
    dest, stage, deck = described_and_prepared(
        tmp_path, monkeypatch, structure, "siesta",
        calculation=calculation, template=template)
    decks = {stage: deck}
    if calculation == "vibration":
        home = stage_home(dest, task_of(dest), stage)
        name = home.names.name(".fdf")
        decks = {part: read_deck(home.dir / part / name) for part in PARTS}
    for where, text in decks.items():
        assert "PAO.BasisSize" in text, (
            f"the {where} deck carries none of the engine's section set")
        _refuse_unknown_labels(text, known, f"{calculation} {where}")
