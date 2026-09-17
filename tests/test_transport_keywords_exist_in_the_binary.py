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
`ChemPots` is present) rather than excused by a stem rule — which used to be
this file's answer and let `TS.TBT.Emin`, `TS.TBT.Emax`, `TS.TBT.NumE` and
`TS.ComplexContour.Emin` through, four of the seven it exists to catch,
because `Emin` and `NumE` occur inside other keywords SIESTA does know.

**What it covers.** Both the standalone `render_script` deck and, since
2026-09-16, one case per deck SHAPE of the ladder that actually runs — which
had never been checked against a binary at all.

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

#: Labels that are NOT fdf keywords and must not be looked up: block
#: delimiters, block-interior grammar words, and comments.  Each is read by
#: the block's own parser, not by `fdf_get`.
_NOT_A_LABEL = {
    "part", "from", "to", "points", "delta", "method", "mu",
    "chem-pot", "used-atoms", "elec-pos", "bloch", "semi-inf-direction",
    "HS", "begin", "end",
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


# deleted 2026-09-17 with `render_script`; the PREPPED-RUNG sibling below covers the decks that run.

def _refuse_unknown_labels(deck: str, binary_text: str, where: str) -> None:
    """Every label WHOLE, not by its last word.

    The check compared only the label's stem until 2026-09-16 — `Emin` for
    `TS.TBT.Emin` — on the reasoning that a runtime-assembled label does not
    appear whole. Measured against the seven keywords in `plan.md` § 5o that
    this file exists to catch, **the stem rule catches three of them**: `Emin`,
    `Emax` and `NumE` all occur inside OTHER keywords the binary does know, so
    `TS.TBT.Emin` — the one that silently put T(E) on the wrong energy grid —
    sailed through the guard written for it. The whole-label rule catches all
    seven, and on the five real rungs it flags exactly the four assembled
    forms above.
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


@pytest.mark.parametrize("token,rung", [("01_seed", "seed"),
                                        ("02_electrode_L", "electrode"),
                                        ("04_device", "negf")])
def test_every_label_a_PREPPED_RUNG_emits_is_known_to_the_binary(
        binary_text, token, rung):
    """THE DECKS THAT ACTUALLY RUN — the gap this file had until 2026-09-16.

    The test above renders through ``TransiestaEngine.render_script``, and
    **no rung uses it**: since the seam migration all five render through
    ``spec_for`` -> ``DeckSpec`` -> ``prepare_deck`` (`engines/transport.md`
    § 6.1a), and `render_script` survives only behind `molbuilder transport
    electrodes` and `/api/transport/render`.

    So the guard written because seven keywords shipped that SIESTA silently
    ignores was watching the one deck nobody runs, while the ~30 keywords the
    seam brought to every rung -- `MaxSCFIterations`, `DM.Tolerance`, the
    `SCF.Mixer` pair, the restart group, `Diag.ParallelOverK`, the whole
    output section, `TS.Contours.Eq.Pole.N` -- had never been checked against
    a binary at all.

    One case per deck SHAPE rather than per rung: `SHAPE_OF_RUNG` maps the
    five rungs onto three texts, and device and transmission are one text.
    """
    import numpy as np

    from molbuilder import script_emit as sc
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.siesta.input import spec_for
    from molbuilder.structure import Structure

    struct = Structure(
        elements=["Au", "Au", "S", "C", "Au", "Au"],
        positions=np.array([[0, 0, 2.0 * i] for i in range(6)], dtype=float),
        cell=np.array([[8.0, 0, 0], [0, 8.0, 0], [0, 0, 14.0]], dtype=float),
        regions={"L-electrode": [0, 1], "bridge": [2, 3],
                 "R-electrode": [4, 5]})
    cfg = SiestaConfig(system_label="kwcheck", kgrid=(2, 2, 1))
    deck = sc.render_deck(
        spec_for(struct, cfg, stage_token=token, calculation="transport"),
        struct, cfg)
    assert "MaxSCFIterations" in deck, (
        "this rung carries none of the engine's section set -- the test "
        "would pass vacuously on a deck that had fallen off the seam")
    _refuse_unknown_labels(deck, binary_text, rung)


def _junction_and_config():
    """The smallest thing that renders a device deck: a labelled junction.

    The same toy Au-S-C-Au the engine's own tests use -- geometry is
    meaningless here, because what is under test is the LABELS the deck
    writes, and those come from the region vocabulary and the config.
    """
    import numpy as np

    from molbuilder.config.transport import (REGION_BRIDGE,
                                             REGION_LEFT_ELECTRODE,
                                             REGION_RIGHT_ELECTRODE,
                                             TransportConfig)
    from molbuilder.structure import Structure
    struct = Structure(
        elements=["Au", "Au", "S", "C", "Au", "Au"],
        positions=np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0],
                            [6, 0, 0], [8, 0, 0], [10, 0, 0]], dtype=float),
        regions={REGION_LEFT_ELECTRODE: [0, 1],
                 REGION_BRIDGE: [2, 3],
                 REGION_RIGHT_ELECTRODE: [4, 5]},
    )
    return struct, TransportConfig(job_name="kwcheck")


#: The four lines the SIESTA 5.4.0 manual says a `%block TS.Elec.<name>`
#: MUST carry: "There are a few lines that must be present, HS,
#: semi-inf-dir, electrode-pos, chem-pot.  The remaining options are
#: optional."
_REQUIRED_ELECTRODE_LINES = ("HS", "semi-inf", "elec-pos", "chem-pot")


@pytest.mark.parametrize("with_buffer", [False, True])
def test_every_electrode_block_carries_the_manuals_required_lines(with_buffer):
    """SCIENCE. An electrode block missing a required line is not a deck.

    `elec-pos` sat inside `if buffer_idx:` until 2026-09-15, so an ORDINARY
    junction -- no buffer atoms -- got two electrode blocks without it
    (`engines/transport.md` 3.3). It was probably harmless, and that is
    exactly why it needs a test rather than a reading: molbuilder sorts the
    junction so the electrodes ARE the first and last atoms, which is where
    an omitted position would land anyway, so the deck relied on an
    undocumented default agreeing with the truth. Both cases are
    parametrised because only one of them was broken.

    Needs no binary, so it does not skip.
    """
    import re

    import numpy as np

    from molbuilder import script_emit as _sc
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.config.transport import (REGION_BRIDGE,
                                             REGION_LEFT_ELECTRODE,
                                             REGION_RIGHT_ELECTRODE)
    from molbuilder.siesta.input import spec_for
    from molbuilder.structure import Structure

    if with_buffer:
        n, regions = 7, {"buffer": [0], REGION_LEFT_ELECTRODE: [1, 2],
                         REGION_BRIDGE: [3, 4], REGION_RIGHT_ELECTRODE: [5, 6]}
    else:
        n, regions = 6, {REGION_LEFT_ELECTRODE: [0, 1], REGION_BRIDGE: [2, 3],
                         REGION_RIGHT_ELECTRODE: [4, 5]}
    struct = Structure(
        elements=["Au"] * n,
        positions=np.array([[0, 0, 2.0 * i] for i in range(n)], dtype=float),
        regions=regions)
    # THE LIVE DECK.  `_emit_transiesta_block` -- the emitter this checks --
    # survives and is reached through `transport/deck.py`; what went on
    # 2026-09-17 is the second writer that used to call it here.  The
    # `elec-pos begin` / `elec-pos end` pairing below is real science
    # (an off-by-one computes transmission through the wrong region and
    # converges while doing it), so the check follows the emitter rather
    # than the deleted route.
    cfg = SiestaConfig(system_label="J")
    spec = spec_for(struct, cfg, stage_token="device", calculation="transport")
    deck = _sc.render_deck(spec, struct, cfg, verbose=cfg.verbose_comments)

    blocks = re.findall(r"%block TS\.Elec\.(\w+)(.*?)%endblock", deck, re.S)
    assert len(blocks) == 2, f"expected two electrode blocks, got {len(blocks)}"
    for name, body in blocks:
        missing = [k for k in _REQUIRED_ELECTRODE_LINES if k not in body]
        assert not missing, (
            f"%block TS.Elec.{name} omits {missing}, which the SIESTA 5.4.0 "
            f"manual lists as lines that MUST be present "
            f"(`engines/transport.md` § 3.3):\n{body}")

    # AND THE RIGHT ONE, NOT MERELY ONE.  Checking only that `elec-pos` is
    # PRESENT was too weak, and the first mutation test proved it: moving the
    # left electrode's line back inside the buffer branch made the `else`
    # fire, so L got `elec-pos end` -- a WRONG position that a
    # presence-check cannot see, and the guard stayed green (caught
    # 2026-09-15 while mutation-testing this very test).
    #
    # The pairing is the fact worth pinning: the z-min electrode is anchored
    # by its FIRST atom (`begin`) and the z-max one by its LAST (`end`,
    # counted back from the end of the sorted structure).
    left, right = dict(blocks)["L"], dict(blocks)["R"]
    assert "elec-pos begin" in left, (
        f"the z-min electrode must be anchored by its first atom:\n{left}")
    assert "elec-pos end" in right, (
        f"the z-max electrode must be anchored by its last atom:\n{right}")
    assert "elec-pos end" not in left and "elec-pos begin" not in right, (
        "the two electrodes' anchors are swapped -- L is anchored from the "
        f"end or R from the beginning:\nL:{left}\nR:{right}")
