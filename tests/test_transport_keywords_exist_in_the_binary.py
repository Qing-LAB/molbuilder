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

**Its limit, stated.** A label assembled at runtime from a prefix does not
appear whole — which is why `TS.ChemPots` counts zero while `ChemPots` and
`.ChemPot.` are present, and the chempot blocks are *fine*. So the check is
on the label's STEM, and a miss is reported for a human to read rather than
treated as proof on its own. It is a smoke alarm, not a judge.

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


def _stem(label: str) -> str:
    """The most specific part of a label -- what survives prefix assembly.

    `TS.ComplexContour.NumCircle` -> `NumCircle`: if the binary does not
    contain even that, no prefix arrangement can reach it.
    """
    return label.rsplit(".", 1)[-1]


def test_every_label_a_device_deck_emits_is_known_to_the_binary(binary_text):
    """THE ASSERTION THE SEVEN DEAD KEYWORDS WOULD HAVE TRIPPED.

    Rendered through the engine's registered door, so it is the text a run
    would actually receive -- not a list of keywords maintained beside it.
    """
    from molbuilder.transport import get_engine
    rendered = get_engine("transiesta").render_script(*_junction_and_config())
    deck = ("\n".join(rendered) if isinstance(rendered, (list, tuple))
            else str(rendered))
    assert "TBT.Contour" in deck, (
        "the deck carries no TBtrans contour block -- this test would pass "
        "vacuously on a deck that had stopped emitting one")

    missing = []
    for label in sorted(_emitted_labels(deck)):
        stem = _stem(label)
        if stem and stem not in binary_text:
            missing.append(f"{label}  (stem {stem!r} absent)")
    assert not missing, (
        "the deck writes labels the installed SIESTA cannot read, so they "
        "are SILENTLY IGNORED -- the run completes and the setting does "
        "nothing (`plan.md` § 5o):\n  " + "\n  ".join(missing)
        + "\n\nIf a label is genuinely assembled at runtime from a prefix, "
          "read the site and add its stem to the binary's vocabulary here "
          "with the reason.")


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
