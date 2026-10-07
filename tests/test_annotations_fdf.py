"""model/structure-annotations.md § 4: the fdf emit-strategy registry for
extensible annotation channels (additive; built-in frozen/region untouched)."""
import numpy as np

from molbuilder.structure import Structure, AtomChannel
from molbuilder import annotations_fdf as afdf


def _struct(n=4):
    s = Structure(elements=["C"] * n,
                  positions=np.arange(n * 3, dtype=float).reshape(n, 3))
    return s


def test_register_and_emit_channel():
    afdf.register_fdf_strategy(
        "test-initspin",
        lambda ch, st: ["%block DM.InitSpin"]
        + [f"{i + 1} {ch.data[i]:+.2f}" for i in sorted(ch.data)]
        + ["%endblock DM.InitSpin"])
    s = _struct()
    s.set_channel("spin", AtomChannel("value", {0: 1.0, 2: -1.0}, fdf="test-initspin"))
    lines = afdf.emit_channels(s)
    assert lines == ["%block DM.InitSpin", "1 +1.00", "3 -1.00",
                     "%endblock DM.InitSpin"]
    assert "test-initspin" in afdf.registered_strategies()


def test_no_strategy_is_carried_not_emitted():
    s = _struct()
    s.set_channel("mytag", AtomChannel("tag", [0, 1]))                # no fdf id
    s.set_channel("orphan", AtomChannel("flag", [2], fdf="nonexistent"))  # unregistered
    assert afdf.emit_channels(s) == []                     # neither emits
    assert afdf.unregistered_channels(s) == ["orphan"]     # flagged for a warning
    # the channels still EXIST on the structure (carried, just not emitted)
    assert s.get_channel("mytag").data == [0, 1]
