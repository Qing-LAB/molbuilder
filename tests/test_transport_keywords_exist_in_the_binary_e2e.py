"""Every fdf label a transport deck writes, checked against the BINARY.

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

pytestmark = pytest.mark.engine

#: Labels that are NOT fdf keywords and must not be looked up: block
#: delimiters, block-interior grammar words, and comments.  Each is read by
#: the block's own parser, not by `fdf_get`.
_NOT_A_LABEL = {
    "part", "from", "to", "points", "delta", "method", "mu",
    "chem-pot", "used-atoms", "elec-pos", "bloch", "semi-inf-direction",
    "HS", "begin", "end", "GF", "GF.ReUse",
}


def _engine_binaries():
    """The installed `siesta` + `tbtrans`, or ``None`` when absent.

    Asked through `diagnostics`, never by joining a path onto a manager
    binary — the rule `docs/ops/installation.md` M2 states and
    `feedback_ask_the_manager_never_derive_a_path` exists for.
    """
    from molbuilder.diagnostics import get_capabilities
    from molbuilder.envs.recipes import effective_name, recipe_by_name
    caps = get_capabilities()
    try:
        recipe = recipe_by_name("molbuilder-siesta")
    except Exception:                                   # noqa: BLE001
        return None
    name = effective_name(recipe, caps)
    if not caps.env_available(name):
        return None
    prefix = caps.env_prefix(name)
    if not prefix:
        return None
    found = [Path(prefix) / "bin" / b for b in ("siesta", "tbtrans")]
    return [p for p in found if p.is_file()] or None


def _strings(path: Path) -> str:
    out = subprocess.run(["strings", str(path)], capture_output=True,
                         text=True, timeout=120)
    return out.stdout


@pytest.fixture(scope="module")
def binary_text():
    bins = _engine_binaries()
    if not bins:
        pytest.skip("molbuilder-siesta is not installed -- nothing to measure")
    return "\n".join(_strings(p) for p in bins)


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
    from support.road import (describe_calculation, prep_groups,
                              set_by_the_person, stage_file)
    bundle = describe_calculation(tmp_path, monkeypatch, name="kwcheck",
                                  calculation="transport", structure=pair)
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
    bins = _engine_binaries()
    siesta = next((b for b in (bins or ()) if b.name == "siesta"), None)
    if siesta is None:
        pytest.skip("molbuilder-siesta is not installed -- nothing to measure")
    deck = _prepped_deck(real_bridged_chain.pair, tmp_path, monkeypatch, "device",
                         ("1", "2"))
    _refuse_unknown_labels(deck, _strings(siesta), "device (siesta alone)")
