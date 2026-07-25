"""Engine-layer tests: the registry indirection + the PySCF engine wrapper.

Spec § 3.2 + § 9 + § 11.  This file covers:

  * the engine registry (``register_engine`` / ``get_engine`` /
    ``UnknownEngineError``) -- a thin indirection so the renderer
    is engine-pluggable;
  * ``PySCFSpectraEngine`` registration + label;
  * the engine's preflight advisory layer (spec § 11.4 +
    scientific caveats from § 9.5);
  * the methods-fragment hook that the cross-engine composer
    invokes;
  * the ``_is_hybrid_functional`` helper that drives auxbasis
    selection;
  * the parse-output entry point that produces a SpectraResults
    from a job directory.

The actual emitted script text is tested in ``test_script.py``;
selector behaviour in ``test_selection.py``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from molbuilder.spectra import (
    SpectraConfig,
    SpectraResults,
)
from molbuilder.spectra.results import (
    PHASE_COMPLETE,
    PHASE_EMPTY,
    SCHEMA_VERSION,
)

from tests.spectra._helpers import _make_results, _struct_water


# --------------------------------------------------------------------- #
#  L2 engine Protocol + registry (spec § 3.2)                           #
# --------------------------------------------------------------------- #


class TestEngineRegistry:
    """The engine plug-in registry is the seam between L1 (shared
    types) and L2 (per-engine implementations).  These tests pin
    the registration semantics so the v1 PySCF engine + any future
    SIESTA engine compose correctly.
    """

    def _make_dummy_engine(self, name: str):
        """Build a minimal engine class meeting the Protocol shape
        without doing real work -- enough for registry tests."""
        # Note: SpectraEngine is the Protocol; we don't subclass it
        # explicitly (duck-typed Protocols don't need it), but the
        # isinstance test in test_protocol_runtime_checkable does
        # import it to assert structural conformance.
        class _DummyEngine:
            pass

        _DummyEngine.name  = name
        _DummyEngine.label = f"dummy ({name})"
        # Stub methods just for Protocol satisfaction; tests don't
        # exercise them.
        _DummyEngine.render_script    = classmethod(lambda c, s, cfg: "")
        _DummyEngine.parse_output     = classmethod(lambda c, p: None)
        _DummyEngine.preflight        = classmethod(
            lambda c, s, cfg, prior=None: []
        )
        _DummyEngine.methods_fragment = classmethod(lambda c, cfg, modes: "")
        return _DummyEngine

    def test_register_and_lookup(self):
        from molbuilder.spectra import (
            register_engine, get_engine, unregister_engine,
        )
        cls = self._make_dummy_engine("dummy-test-1")
        try:
            register_engine(cls)
            assert get_engine("dummy-test-1") is cls
        finally:
            unregister_engine("dummy-test-1")

    def test_unknown_engine_raises_with_available_list(self):
        from molbuilder.spectra import get_engine, UnknownEngineError
        with pytest.raises(UnknownEngineError) as exc_info:
            get_engine("not-a-real-engine-xyz")
        # The error names the requested engine + what's available
        # so a typo is actionable.
        assert "not-a-real-engine-xyz" in str(exc_info.value)
        assert exc_info.value.name == "not-a-real-engine-xyz"
        assert isinstance(exc_info.value.available, list)

    def test_duplicate_registration_rejected(self):
        """Re-registering an existing name is a programmer error
        (two engines claiming the same key); register_engine
        raises rather than silently overwriting."""
        from molbuilder.spectra import register_engine, unregister_engine
        cls1 = self._make_dummy_engine("dummy-test-dup")
        cls2 = self._make_dummy_engine("dummy-test-dup")
        try:
            register_engine(cls1)
            with pytest.raises(ValueError, match="already registered"):
                register_engine(cls2)
        finally:
            unregister_engine("dummy-test-dup")

    def test_re_registering_same_class_is_idempotent(self):
        """Importing an engine module twice (e.g., via reload
        during dev) must not raise -- the second import is the
        SAME class, no conflict."""
        from molbuilder.spectra import register_engine, unregister_engine
        cls = self._make_dummy_engine("dummy-test-idem")
        try:
            register_engine(cls)
            register_engine(cls)   # second call with the same class
            from molbuilder.spectra import get_engine
            assert get_engine("dummy-test-idem") is cls
        finally:
            unregister_engine("dummy-test-idem")

    def test_class_without_name_attribute_rejected(self):
        """An engine class without a `name` class attribute can't
        be registered -- the registry would have nothing to key on."""
        from molbuilder.spectra import register_engine

        class _NamelessEngine:
            label = "no name"

        with pytest.raises(TypeError, match="non-empty string"):
            register_engine(_NamelessEngine)

    def test_registered_engines_returns_sorted_list(self):
        from molbuilder.spectra import (
            register_engine, registered_engines, unregister_engine,
        )
        b = self._make_dummy_engine("b-engine")
        a = self._make_dummy_engine("a-engine")
        try:
            register_engine(b)
            register_engine(a)
            names = registered_engines()
            assert "a-engine" in names
            assert "b-engine" in names
            # Sorted alphabetically -- 'a' before 'b'.
            assert names.index("a-engine") < names.index("b-engine")
        finally:
            unregister_engine("a-engine")
            unregister_engine("b-engine")

    def test_protocol_runtime_checkable(self):
        """SpectraEngine is @runtime_checkable so isinstance works.
        A class meeting the Protocol shape via duck typing should
        satisfy the check; one missing required methods should not."""
        from molbuilder.spectra import SpectraEngine
        cls = self._make_dummy_engine("dummy-test-proto")
        # The dummy has all the right methods + attrs.
        assert isinstance(cls, SpectraEngine)

# --------------------------------------------------------------------- #
#  PySCFSpectraEngine (engine wrapper -- non-render_script methods)     #
#                                                                       #
#  Spec § 3.2 + § 9 + § 11.  render_script is tested separately when    #
#  the script-template module lands (next commit).                      #
# --------------------------------------------------------------------- #


class TestPySCFSpectraEngineRegistration:

    def test_registered_under_pyscf(self):
        """The engine self-registers at import time -- importing
        molbuilder.spectra (or .pyscf_engine) puts 'pyscf' in the
        registry."""
        from molbuilder.spectra import get_engine
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        assert get_engine("pyscf") is PySCFSpectraEngine

    def test_engine_metadata(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        assert PySCFSpectraEngine.name == "pyscf"
        assert "PySCF" in PySCFSpectraEngine.label


class TestPySCFEngineMethodsFragment:

    def test_basic_fragment_cites_pyscf(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig()
        frag = PySCFSpectraEngine.methods_fragment(cfg, [])
        # Pyscf citation keys present.
        assert "Sun2020" in frag
        assert "Sun2018" in frag
        # Names the analytic Hessian module.
        assert "pyscf.hessian" in frag

    def test_method_specific_hessian_module(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(method="UHF")
        frag = PySCFSpectraEngine.methods_fragment(cfg, [])
        assert "pyscf.hessian.uhf" in frag

    def test_raman_path_cites_komornicki(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(compute_raman=True)
        frag = PySCFSpectraEngine.methods_fragment(cfg, [])
        assert "Komornicki1979" in frag
        assert "polarizability" in frag.lower()

    def test_no_raman_path_omits_komornicki(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(compute_raman=False)
        frag = PySCFSpectraEngine.methods_fragment(cfg, [])
        assert "Komornicki1979" not in frag

    def test_density_fit_mentioned_when_on(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg_on  = SpectraConfig(density_fit=True)
        cfg_off = SpectraConfig(density_fit=False)
        assert "density fitting" in PySCFSpectraEngine.methods_fragment(cfg_on, []).lower()
        assert "density fitting" not in PySCFSpectraEngine.methods_fragment(cfg_off, []).lower()

    def test_grid_level_mentioned_for_dft_only(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        # DFT path mentions grid level.
        assert "grid level" in PySCFSpectraEngine.methods_fragment(
            SpectraConfig(method="RKS"), []
        ).lower()
        # HF path doesn't.
        assert "grid level" not in PySCFSpectraEngine.methods_fragment(
            SpectraConfig(method="RHF"), []
        ).lower()

    def test_fragment_composes_into_render_methods_md(self):
        """The engine's fragment flows into render_methods_md's
        output and its citations bubble up into the bibliography."""
        from molbuilder.spectra import render_methods_md
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig()
        md = render_methods_md(cfg, engine=PySCFSpectraEngine)
        assert "pyscf.hessian" in md
        assert "Sun2020" in md
        # Sun2020 appears in the trailing bibliography too.
        bib = md.split("**Bibliography**", 1)[1]
        assert "Sun2020" in bib


class TestPySCFEnginePreflight:

    def test_clean_config_has_no_issues(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig()  # defaults -- selector=none, no L3 dep
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        errors = [i for i in issues if i.severity == "error"]
        assert errors == []

    def test_hybrid_with_low_grid_warns(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(functional="B3LYP", grid_level=3)
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        warns = [i for i in issues if i.severity == "warn"]
        assert any(i.where == "config.grid_level" for i in warns)

    def test_pbe0_recognised_as_hybrid(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(functional="PBE0", grid_level=2)
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        assert any(i.where == "config.grid_level"
                   and i.severity == "warn" for i in issues)

    def test_pure_functional_no_grid_warn(self):
        """Pure PBE (LDA/GGA, no τ-dependence) shouldn't trip the
        grid-level warn -- only meta-GGAs and hybrids do."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(functional="PBE", grid_level=2)
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        assert not any(i.where == "config.grid_level" for i in issues)

    def test_meta_gga_low_grid_warns(self):
        """SCIENTIFIC-AUDIT FIX (FN-1): SCAN (meta-GGA) at grid < 4 must
        warn on the render gate -- the grid-sensitive class is meta-GGA,
        not just hybrids.  Pre-2026-07 SCAN passed silently."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(functional="SCAN", grid_level=3)
        issues = PySCFSpectraEngine.render_checks(_struct_water(), cfg)
        assert any(i.where == "config.grid_level" and i.severity == "warn"
                   for i in issues)

    def test_restricted_method_with_nonzero_spin_warns(self):
        """SCIENTIFIC-AUDIT FIX (FN-3): a restricted method (RKS/RHF)
        forces 2S=0, so spin>0 with it is a contradiction the render
        gate must flag.  Pre-2026-07 the stale 'cfg has no spin' comment
        meant this passed silently even though SpectraConfig HAS spin."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(method="RKS", spin=2)
        issues = PySCFSpectraEngine.render_checks(_struct_water(), cfg)
        assert any(i.where == "config.method" and i.severity == "warn"
                   for i in issues), (
            "RKS + spin=2 must warn (restricted method can't be open-shell)")

    def test_displacement_below_window_warns(self):
        """Window lower bound is 0.02 Å (lowered 2026-05-19 to match
        the new SpectraConfig default).  Anything smaller surfaces
        a warn so the user gets a clear signal that the FD noise
        floor may dominate."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(displacement_amplitude_ang=0.01)
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        assert any(i.where == "config.displacement_amplitude_ang"
                   and i.severity == "warn" for i in issues)

    def test_displacement_above_window_warns(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(displacement_amplitude_ang=0.25)
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        assert any(i.where == "config.displacement_amplitude_ang"
                   and i.severity == "warn" for i in issues)

    def test_default_displacement_no_warn(self):
        """Default 0.02 Å is the linear-response production value
        (lowered from 0.10 on 2026-05-19; see SpectraConfig)."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig()
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        assert not any(i.where == "config.displacement_amplitude_ang"
                       for i in issues)

    def test_compute_ir_warns_reserved(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(compute_ir=True)
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        # Plain-language: "aren't implemented" + suggestion.
        assert any(i.where == "config.compute_ir"
                   and i.severity == "warn"
                   and "implemented" in i.message for i in issues)

    def test_use_gpu_warns_when_gpu4pyscf_missing(self, monkeypatch):
        """Asking for GPU acceleration on a host where gpu4pyscf
        isn't installed should warn (not error) so the user has
        time to install it before running the generated script,
        but the generated script falls back to CPU anyway.

        Simulates the missing-package state by setting
        sys.modules['gpu4pyscf'] = None, the standard pytest trick
        for forcing an ImportError on `import gpu4pyscf` regardless
        of the installed environment.
        """
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        import sys
        monkeypatch.setitem(sys.modules, "gpu4pyscf", None)
        cfg = SpectraConfig(use_gpu=True)
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        warns = [i for i in issues
                 if i.severity == "warn" and i.where == "config.use_gpu"]
        assert len(warns) == 1
        # Message names the package + the install command.
        assert "gpu4pyscf" in warns[0].message
        assert "pip install" in warns[0].message
        # And explicitly mentions the CPU fallback so the user knows
        # this is non-fatal.
        assert "fall" in warns[0].message.lower() \
            or "cpu" in warns[0].message.lower()

    def test_use_gpu_no_warn_when_gpu4pyscf_and_modern_gpu(self, monkeypatch):
        """When gpu4pyscf is importable AND the GPU is modern enough
        (compute capability >= 7.0), no advisory should fire."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        import sys, types
        # Inject a fake gpu4pyscf so the import succeeds.
        monkeypatch.setitem(sys.modules, "gpu4pyscf",
                            types.ModuleType("gpu4pyscf"))
        # Inject a fake cupy that reports a modern GPU.
        fake_cupy = types.ModuleType("cupy")
        fake_cuda = types.ModuleType("cupy.cuda")
        fake_runtime = types.ModuleType("cupy.cuda.runtime")
        fake_runtime.getDeviceCount = lambda: 1
        fake_runtime.getDeviceProperties = lambda i: {
            "name": "Fake H100",   # modern card
            "major": 9, "minor": 0,
        }
        fake_cuda.runtime = fake_runtime
        fake_cupy.cuda = fake_cuda
        monkeypatch.setitem(sys.modules, "cupy",         fake_cupy)
        monkeypatch.setitem(sys.modules, "cupy.cuda",    fake_cuda)
        monkeypatch.setitem(sys.modules, "cupy.cuda.runtime", fake_runtime)
        cfg = SpectraConfig(use_gpu=True)
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        gpu_warns = [i for i in issues if i.where == "config.use_gpu"]
        assert gpu_warns == []

    def test_use_gpu_warns_when_gpu_too_old(self, monkeypatch):
        """Card has compute capability < 7.0 -- gpu4pyscf will fail
        with cryptic CUDA errors during the SCF.  Warn before the
        run, suggest disabling 'Use GPU'."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        import sys, types
        monkeypatch.setitem(sys.modules, "gpu4pyscf",
                            types.ModuleType("gpu4pyscf"))
        fake_cupy = types.ModuleType("cupy")
        fake_cuda = types.ModuleType("cupy.cuda")
        fake_runtime = types.ModuleType("cupy.cuda.runtime")
        fake_runtime.getDeviceCount = lambda: 1
        fake_runtime.getDeviceProperties = lambda i: {
            "name": "GTX 1080",   # Pascal, compute cap 6.1
            "major": 6, "minor": 1,
        }
        fake_cuda.runtime = fake_runtime
        fake_cupy.cuda = fake_cuda
        monkeypatch.setitem(sys.modules, "cupy",              fake_cupy)
        monkeypatch.setitem(sys.modules, "cupy.cuda",         fake_cuda)
        monkeypatch.setitem(sys.modules, "cupy.cuda.runtime", fake_runtime)
        cfg = SpectraConfig(use_gpu=True)
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        gpu_warns = [i for i in issues
                     if i.where == "config.use_gpu" and i.severity == "warn"]
        assert len(gpu_warns) == 1
        msg = gpu_warns[0].message
        # Message names the actual card + compute capability the
        # user can compare against the gpu4pyscf docs.
        assert "GTX 1080" in msg
        assert "6.1" in msg
        # ... and the minimum requirement.
        assert "7.0" in msg
        # ... and the actionable "untick" suggestion.
        assert "Use GPU" in msg

    def test_use_gpu_warns_when_no_gpu(self, monkeypatch):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        import sys, types
        monkeypatch.setitem(sys.modules, "gpu4pyscf",
                            types.ModuleType("gpu4pyscf"))
        fake_cupy = types.ModuleType("cupy")
        fake_cuda = types.ModuleType("cupy.cuda")
        fake_runtime = types.ModuleType("cupy.cuda.runtime")
        fake_runtime.getDeviceCount = lambda: 0
        fake_cuda.runtime = fake_runtime
        fake_cupy.cuda = fake_cuda
        monkeypatch.setitem(sys.modules, "cupy",              fake_cupy)
        monkeypatch.setitem(sys.modules, "cupy.cuda",         fake_cuda)
        monkeypatch.setitem(sys.modules, "cupy.cuda.runtime", fake_runtime)
        cfg = SpectraConfig(use_gpu=True)
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        gpu_warns = [i for i in issues if i.where == "config.use_gpu"]
        assert len(gpu_warns) == 1
        assert "no NVIDIA GPU" in gpu_warns[0].message

    def test_use_gpu_off_no_warn(self):
        """When the user leaves GPU off, the GPU advisory shouldn't
        fire even if gpu4pyscf isn't installed."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(use_gpu=False)
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        assert not any(i.where == "config.use_gpu" for i in issues)

    def test_few_frozen_atoms_warns_about_spurious_modes(self):
        """Fixing 1 or 2 atoms can't fully anchor the free fragment
        in space -- residual rigid-body motion leaks into the
        vibrational analysis as near-zero modes.  Warn so the user
        ignores those modes when interpreting the spectrum."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        for n in (1, 2):
            cfg = SpectraConfig(frozen_indices=list(range(n)))
            issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
            warns = [i for i in issues
                     if i.severity == "warn"
                     and i.where == "config.frozen_indices"
                     and "spurious" in i.message]
            assert len(warns) == 1, (n, issues)

    def test_three_or_more_frozen_atoms_no_spurious_warn(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(frozen_indices=[0, 1, 2])
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        assert not any("spurious" in i.message for i in issues)

    def test_element_freezing_doesnt_trigger_spurious_warn(self):
        """Element-level freezing typically pins many atoms (a whole
        metal slab); the spurious-modes warn shouldn't fire when the
        user is freezing by element rather than by index."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(frozen_elements=["O"])
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        assert not any("spurious" in i.message for i in issues)

    # ----- Scientific caveat banners -----

    def test_top_n_selector_emits_raman_vs_epc_caveat(self):
        """When user picks top_n / threshold, surface the Galperin
        caveat -- Raman brightness != EPC strength."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        # Need a prior with phase_raman=complete so the soft-dep
        # check doesn't drown out our advisory.
        prior = _make_results(complete=True)
        for sel in ("top_n", "threshold"):
            cfg = SpectraConfig(es_mode_selection=sel)
            issues = PySCFSpectraEngine.preflight(_struct_water(),
                                                  cfg, prior=prior)
            warns = [i for i in issues
                     if i.where == "config.es_mode_selection"
                     and i.severity == "warn"
                     and "electron-phonon" in i.message]
            assert len(warns) == 1, (sel, [i.message for i in issues])
            assert "Galperin2007" in warns[0].message

    def test_skip_selector_no_caveat(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(es_mode_selection="skip")
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        assert not any("electron-phonon" in i.message for i in issues)

    def test_explicit_with_empty_list_warns(self):
        """selector=explicit + empty indices = run that produces
        no orbital-energy data.  Warn so the user doesn't burn the
        run discovering this."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(es_mode_selection="explicit",
                            es_explicit_indices=[])
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        warns = [i for i in issues
                 if i.where == "config.es_explicit_indices"
                 and i.severity == "warn"
                 and "no mode indices" in i.message]
        assert len(warns) == 1

    def test_explicit_with_indices_no_warn(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(es_mode_selection="explicit",
                            es_explicit_indices=[1, 2])
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        assert not any("no mode indices" in i.message for i in issues)

    def test_large_system_suggests_freezing(self):
        """For >30 free atoms with nothing fixed, suggest freezing
        the metal slab / anchor to cut Hessian + Raman FD cost."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        from molbuilder.structure import Structure
        # 40-atom toy system, all hydrogen so SCF wouldn't matter
        # if this ever ran.  Geometry's irrelevant for the advisory.
        struct = Structure(
            elements  = ["H"] * 40,
            positions = np.array([[i * 1.0, 0.0, 0.0] for i in range(40)]),
        )
        cfg = SpectraConfig()  # no atoms fixed
        issues = PySCFSpectraEngine.preflight(struct, cfg)
        warns = [i for i in issues
                 if i.where == "config.frozen_indices"
                 and i.severity == "warn"
                 and "metal slab" in i.message]
        assert len(warns) == 1

    def test_small_system_no_freezing_suggestion(self):
        """Water (3 atoms) shouldn't trigger the large-system
        warning."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig()
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        assert not any("metal slab" in i.message for i in issues)

    def test_unsupported_method_errors(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig()
        # Sidestep the dataclass's choices validation by setting
        # the attribute directly -- the preflight is the second
        # line of defence anyway.
        cfg.method = "BOGUS"
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        assert any(i.where == "config.method"
                   and i.severity == "error" for i in issues)

    def test_out_of_range_frozen_indices_errors(self):
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(frozen_indices=[0, 1, 99])  # water has 3 atoms
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        assert any(i.where == "config.frozen_indices"
                   and i.severity == "error" for i in issues)

    def test_in_range_frozen_indices_ok(self):
        """In-range frozen_indices should NOT produce a range-check
        error.  (A separate test covers the WARN about spurious
        rigid-body modes when fewer than 3 atoms are fixed.)"""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(frozen_indices=[0, 1])  # all valid for water
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg)
        errors_from_indices = [i for i in issues
                               if i.where == "config.frozen_indices"
                               and i.severity == "error"]
        assert errors_from_indices == []

    # ------- Three-stage contract: stage-3 preflight enforcement ----- #
    # design.md "Sidecar-driven boundary conditions -- the three-stage
    # contract".  The script honors cfg.frozen_indices verbatim; the
    # preflight surfaces (A) divergence between struct.frozen_atoms
    # and cfg.frozen_indices, (B) unconsumed sidecar labels (regions)
    # so the user is never silently surprised by the boundary
    # conditions the script will use.

    def test_sidecar_frozen_divergence_warns(self):
        """Pattern A: struct.frozen_atoms has indices NOT in
        cfg.frozen_indices -> WARN.  The user is about to generate a
        script that does NOT freeze atoms the sidecar marked as
        frozen; surface it before Generate, not silently."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        struct = _struct_water()
        struct.frozen_atoms = [0]                 # sidecar says freeze atom 0
        cfg = SpectraConfig(frozen_indices=[1])   # form says freeze atom 1
        issues = PySCFSpectraEngine.preflight(struct, cfg)
        warns = [i for i in issues
                 if i.severity == "warn"
                 and i.where == "config.frozen_indices"
                 and "sidecar" in i.message
                 and "[0]" in i.message]
        assert len(warns) == 1, [i.message for i in issues]

    def test_sidecar_frozen_subset_no_warn(self):
        """Form-set ⊇ sidecar-set -> NO divergence warn.  The script
        will freeze everything the sidecar wanted plus possibly
        more; no silent omission."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        struct = _struct_water()
        struct.frozen_atoms = [0]
        cfg = SpectraConfig(frozen_indices=[0, 1])
        issues = PySCFSpectraEngine.preflight(struct, cfg)
        divergence_warns = [
            i for i in issues
            if i.where == "config.frozen_indices"
            and i.severity == "warn"
            and "sidecar" in i.message
        ]
        assert divergence_warns == []

    def test_no_sidecar_frozen_no_warn(self):
        """Struct without frozen_atoms (the common case) -> no
        divergence warn fires; regression against false positives."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        struct = _struct_water()
        cfg = SpectraConfig(frozen_indices=[0])
        issues = PySCFSpectraEngine.preflight(struct, cfg)
        divergence_warns = [
            i for i in issues
            if i.where == "config.frozen_indices"
            and i.severity == "warn"
            and "sidecar" in i.message
        ]
        assert divergence_warns == []

    def test_sidecar_regions_unrecognized_warn(self):
        """Pattern B: struct.regions is non-empty -> WARN that the
        spectra engine does not consume regions.  The user is told
        their region labels (set in /modify for transport workflows)
        are ignored here, not silently absorbed."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        struct = _struct_water()
        struct.regions = {"L-electrode": [0], "bridge": [1, 2]}
        cfg = SpectraConfig()
        issues = PySCFSpectraEngine.preflight(struct, cfg)
        notices = [i for i in issues
                   if i.where == "structure.regions"
                   and "do" in i.message.lower()
                   and "consume" in i.message.lower()]
        assert len(notices) == 1, [i.message for i in issues]
        # The notice should name the actual labels so the user
        # can see WHICH labels are being ignored.
        assert "L-electrode" in notices[0].message
        assert "bridge" in notices[0].message

    def test_no_regions_no_unrecognized_warn(self):
        """Struct without regions -> no Pattern-B notice; regression
        against false positives."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        struct = _struct_water()
        cfg = SpectraConfig()
        issues = PySCFSpectraEngine.preflight(struct, cfg)
        assert not any(
            i.where == "structure.regions" for i in issues
        )

    def test_empty_regions_no_unrecognized_warn(self):
        """A regions dict with only empty lists -> no notice
        (skipping empty entries; if there's nothing in the region,
        there's nothing to ignore)."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        struct = _struct_water()
        struct.regions = {"L-electrode": [], "bridge": []}
        cfg = SpectraConfig()
        issues = PySCFSpectraEngine.preflight(struct, cfg)
        assert not any(
            i.where == "structure.regions" for i in issues
        )

    def test_selector_top_n_without_prior_l3_errors(self):
        """top_n / threshold selectors need a prior L3 run; the
        engine's preflight delegates to selection.validate_selection
        and surfaces that as an error."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(es_mode_selection="top_n", es_top_n=5)
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg, prior=None)
        errors = [i for i in issues if i.severity == "error"]
        assert any(i.where == "config.es_mode_selection" for i in errors)

    def test_selector_top_n_with_prior_l3_ok(self):
        """Same selector + a prior result that completed L3 -> OK
        (no error from the validator)."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        cfg = SpectraConfig(es_mode_selection="top_n", es_top_n=3)
        prior = _make_results(complete=True)
        issues = PySCFSpectraEngine.preflight(_struct_water(), cfg, prior=prior)
        errs = [i for i in issues if i.severity == "error"
                and i.where == "config.es_mode_selection"]
        assert errs == []


class TestPySCFEngineIsHybridFunctional:
    """The hybrid-detection heuristic.  We accept some false
    positives (the resulting warn is benign) but want no false
    negatives for the canonical hybrid families.

    The detector moved to the ONE shared home (V4):
    ``molbuilder.validation.pyscf.is_hybrid_functional`` — the spectra
    preflight and the Build-tab PySCF validator both call it, so the
    hybrid gate can no longer drift between the two tabs."""

    @pytest.mark.parametrize("name", [
        "B3LYP", "b3lyp", "B3PW91",
        "PBE0", "pbe0",
        "M06", "M06-2X", "M06-L",
        "ωB97X-D", "wB97X",
        "CAM-B3LYP",
        "BHandH", "BHandHLYP",
        "TPSS0",
        "HSE06",
    ])
    def test_recognised_hybrids(self, name):
        from molbuilder.validation.pyscf import is_hybrid_functional
        assert is_hybrid_functional(name) is True

    @pytest.mark.parametrize("name", [
        "PBE", "BLYP", "LDA", "BP86", "TPSS", "SCAN",
    ])
    def test_pure_functionals_not_flagged(self, name):
        from molbuilder.validation.pyscf import is_hybrid_functional
        assert is_hybrid_functional(name) is False


class TestPySCFEngineParseOutput:
    """parse_output should delegate to the engine-agnostic JSON
    parser cleanly."""

    def test_parse_output_round_trips(self, tmp_path):
        from molbuilder.sidecars.spectra import dump_spectra_json
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        original = _make_results(complete=True)
        p = tmp_path / "x.spectra.json"
        dump_spectra_json(original, p)
        loaded = PySCFSpectraEngine.parse_output(str(p))
        assert loaded.engine == original.engine
        assert len(loaded.modes) == len(original.modes)

    def test_parse_output_propagates_missing_file_error(self, tmp_path):
        from molbuilder.sidecars.spectra import SpectraJsonNotFoundError
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        bad = tmp_path / "missing.spectra.json"
        with pytest.raises(SpectraJsonNotFoundError):
            PySCFSpectraEngine.parse_output(str(bad))


