"""The transport engine registry + the TransportResults round-trip.

The registry pattern with its ONE shipped backend (transiesta
registers at import), and the pre-composite wire shape's round-trip.
(The header once claimed no real backend existed and named a mirror
file, ``tests/test_spectra_engine.py``, that does not exist -- both
claims retired 2026-08-29.)
"""

from __future__ import annotations

import numpy as np
import pytest

from molbuilder.transport import (
    TransportEngine,
    TransportResults,
    UnknownEngineError,
    get_engine,
    register_engine,
    registered_engines,
    unregister_engine,
)
from molbuilder.transport.results import SCHEMA_VERSION


# --------------------------------------------------------------------- #
#  TransportResults dataclass                                            #
# --------------------------------------------------------------------- #


class TestTransportResultsShape:
    def test_default_construction(self):
        r = TransportResults()
        assert r.energy_grid_eV.shape == (0,)
        assert r.transmission.shape == (0,)
        assert r.fermi_energy_eV == 0.0
        assert r.conductance_G0 == 0.0
        assert r.pdos == {}
        assert r.bias_grid_V is None
        assert r.current_uA is None
        assert r.complete is False

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="share shape"):
            TransportResults(
                energy_grid_eV=np.zeros(5),
                transmission=np.zeros(7),
            )

    def test_pdos_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="pdos"):
            TransportResults(
                energy_grid_eV=np.zeros(5),
                transmission=np.zeros(5),
                pdos={"C-2pz": np.zeros(3)},
            )

    def test_bias_iv_paired(self):
        with pytest.raises(ValueError, match="both provided or both None"):
            TransportResults(
                bias_grid_V=np.linspace(-0.5, 0.5, 11),
            )

    def test_bias_iv_shape_mismatch(self):
        with pytest.raises(ValueError, match="share shape"):
            TransportResults(
                bias_grid_V=np.linspace(-0.5, 0.5, 11),
                current_uA=np.zeros(7),
            )

    def test_equality_refused(self):
        a = TransportResults()
        b = TransportResults()
        with pytest.raises(TypeError, match="equality is not defined"):
            a == b
        with pytest.raises(TypeError, match="unhashable"):
            hash(a)


class TestTransportResultsRoundTrip:
    def _fixture(self):
        E = np.linspace(-2.0, 2.0, 21)
        T = np.exp(-E ** 2)
        return TransportResults(
            metadata={"engine": "mock", "version": "0.0"},
            energy_grid_eV=E,
            transmission=T,
            fermi_energy_eV=0.0,
            conductance_G0=1.0,
            pdos={"C-2pz": T * 0.5, "H-1s": T * 0.1},
            bias_grid_V=np.linspace(-0.5, 0.5, 11),
            current_uA=np.zeros(11),
            methods_text="Mock methods paragraph.",
            bibliography_keys=["Datta1995"],
            complete=True,
            # Strict v2 (2026-06-26): a complete transport result must
            # carry L/M/R region assignment + frozen electrode atoms;
            # from_dict refuses an empty-regions record.
            regions={"L-electrode": [0, 1], "M-bridge": [2, 3],
                     "R-electrode": [4, 5]},
            frozen_atoms=[0, 1, 4, 5],
        )

    def test_round_trip(self):
        r = self._fixture()
        d = r.to_dict()
        assert d["schema_version"] == SCHEMA_VERSION
        r2 = TransportResults.from_dict(d)
        # Field-by-field — equality refused.
        assert r2.metadata == r.metadata
        np.testing.assert_array_equal(r2.energy_grid_eV, r.energy_grid_eV)
        np.testing.assert_array_equal(r2.transmission, r.transmission)
        assert r2.fermi_energy_eV == r.fermi_energy_eV
        assert r2.conductance_G0 == r.conductance_G0
        assert set(r2.pdos.keys()) == set(r.pdos.keys())
        for k in r.pdos:
            np.testing.assert_array_equal(r2.pdos[k], r.pdos[k])
        np.testing.assert_array_equal(r2.bias_grid_V, r.bias_grid_V)
        np.testing.assert_array_equal(r2.current_uA, r.current_uA)
        assert r2.methods_text == r.methods_text
        assert r2.bibliography_keys == r.bibliography_keys
        assert r2.complete == r.complete

    def test_unknown_schema_version_raises(self):
        r = self._fixture()
        d = r.to_dict()
        d["schema_version"] = "999"
        with pytest.raises(ValueError, match="not supported"):
            TransportResults.from_dict(d)


# --------------------------------------------------------------------- #
#  Engine registry                                                       #
# --------------------------------------------------------------------- #


class _MockTransportEngine:
    """Test-only TransportEngine implementation.  Registered in the
    test setup and unregistered in teardown so production lookups
    aren't polluted."""

    name = "mock-transport-engine"
    label = "Mock transport engine (tests only)"

    @classmethod
    def render_script(cls, struct, cfg):
        return "# mock script\nprint('hi')\n"

    @classmethod
    def parse_output(cls, path):
        return TransportResults(complete=True)

    @classmethod
    def preflight(cls, struct, cfg, prior=None):
        return []

    @classmethod
    def methods_fragment(cls, cfg, results):
        return "Mock methods."


class TestRegistry:
    def setup_method(self):
        unregister_engine(_MockTransportEngine.name)

    def teardown_method(self):
        unregister_engine(_MockTransportEngine.name)

    def test_register_and_get(self):
        register_engine(_MockTransportEngine)
        assert _MockTransportEngine.name in registered_engines()
        retrieved = get_engine(_MockTransportEngine.name)
        assert retrieved is _MockTransportEngine

    def test_protocol_isinstance(self):
        # runtime_checkable Protocol — the class itself isn't an
        # instance; an instance would be (no engine ever
        # instantiates, but the check should at least not crash).
        assert isinstance(_MockTransportEngine(), TransportEngine)

    def test_unknown_engine_raises(self):
        with pytest.raises(UnknownEngineError) as ei:
            get_engine("does-not-exist")
        assert ei.value.name == "does-not-exist"
        # Available list should be sorted.
        assert ei.value.available == sorted(ei.value.available)

    def test_no_clobber(self):
        register_engine(_MockTransportEngine)

        class _Other:
            name = _MockTransportEngine.name
            label = "Other"

            @classmethod
            def render_script(cls, s, c): return ""
            @classmethod
            def parse_output(cls, p): return TransportResults()
            @classmethod
            def preflight(cls, s, c, prior=None): return []
            @classmethod
            def methods_fragment(cls, c, r): return ""

        with pytest.raises(ValueError, match="already registered"):
            register_engine(_Other)

    def test_re_register_same_class_ok(self):
        register_engine(_MockTransportEngine)
        # Idempotent re-register of the SAME class is OK.
        register_engine(_MockTransportEngine)

    def test_register_requires_name(self):
        class _NoName:
            label = "no name"

        with pytest.raises(TypeError, match="`name`"):
            register_engine(_NoName)

    def test_config_engine_choices_match_registry(self):
        """The registry IS the source of truth for what's wired, and
        the choices must not promise more than it answers for
        (``pyscf-negf`` left 2026-08-29 — offered but never
        registered, so the form suggested a value ``get_engine``
        refused)."""
        import molbuilder.transport  # noqa: F401 -- registration side-effect
        from molbuilder.config.transport import TransportConfig
        from molbuilder.transport.engine_base import registered_engines
        fld = TransportConfig.__dataclass_fields__["engine"]
        assert set(fld.metadata["choices"]) <= set(registered_engines())
