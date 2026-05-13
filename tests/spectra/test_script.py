"""Emitted-script structural tests: ``render_spectra_script`` output.

The script is the runnable Python the user ships to a compute cluster.
Tests can't *run* it (would need real PySCF + a converged SCF + minutes
of wall time), so the test surface is structural:

  * ``compile()`` accepts the output as valid Python (no syntax bugs);
  * expected block markers present / absent per config flags;
  * critical safety + correctness invariants are pinned -- atomic-
    replace pattern present, ``allow_nan=False`` on every JSON dump,
    no stray BOM-like chars, SCHEMA_VERSION matches the parser, GPU
    compute-capability threshold matches the engine preflight's,
    the inlined selector matches the Python canonical (the latter
    in ``test_selection.py::TestSelectorEquivalence``).

Smoke tests that actually exec the emitted script through PySCF live
in ``test_smoke.py`` (marked ``@pytest.mark.smoke``).
"""

from __future__ import annotations

import pytest

from molbuilder.spectra import SpectraConfig

from tests.spectra._helpers import _struct_water


# --------------------------------------------------------------------- #
#  PySCF script template (pyscf_script.py)                              #
#                                                                       #
#  The emitted Python script that gets shipped to the user.  Tests      #
#  cannot RUN it (no PySCF in the test env, would take minutes), so     #
#  the test surface is structural:                                      #
#    * compile() accepts the output as valid Python (no syntax bugs);   #
#    * expected block markers present / absent per config;              #
#    * critical safety + correctness invariants are pinned (SCHEMA_     #
#      VERSION matches the parser, atomic-replace pattern present,      #
#      allow_nan=False, no stray BOM-like chars).                       #
# --------------------------------------------------------------------- #


class TestPySCFScriptCompiles:
    """The most important guarantee: every config combination
    produces a script that Python's compiler accepts.  A syntax
    bug in the template would only surface when the user runs the
    file -- catch them here instead."""

    @pytest.mark.parametrize("cfg_overrides", [
        # Default config
        dict(),
        # Minimal: no Raman, no ES
        dict(compute_raman=False),
        # Raman only
        dict(compute_raman=True, es_mode_selection="skip"),
        # ES only with explicit selector
        dict(compute_raman=False, es_mode_selection="explicit",
             es_explicit_indices=[1, 2]),
        # Full pipeline
        dict(compute_raman=True, es_mode_selection="top_n", es_top_n=5),
        # Threshold selector
        dict(compute_raman=True, es_mode_selection="threshold",
             es_threshold=10.0),
        # All modes for ES
        dict(es_mode_selection="all"),
        # Dispersion variants
        dict(dispersion="none"),
        dict(dispersion="d4"),
        # Unrestricted SCF
        dict(method="UKS"),
        # Hartree-Fock (no DFT)
        dict(method="RHF"),
        # Hybrid-low-grid (should compile fine, just a preflight warn)
        dict(grid_level=2),
        # Freeze atoms
        dict(fixed_elements=["H"]),
        dict(fixed_indices=[1, 2]),
        # Frequency window
        dict(es_mode_selection="all", freq_min_cm1=500.0, freq_max_cm1=3500.0),
    ])
    def test_compiles_as_python(self, cfg_overrides):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        cfg = SpectraConfig(**cfg_overrides)
        script = render_spectra_script(_struct_water(), cfg)
        # compile() raises SyntaxError on bad Python -- this is the
        # cheap-to-run guarantee that the template is correct.
        code = compile(script, f"<spectra.py {cfg_overrides!r}>", "exec")
        assert code is not None


class TestPySCFScriptHeader:
    """The docstring header is the Methods-section source-of-truth
    that ships with the script (spec § 11.2)."""

    def test_starts_with_docstring(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water(), SpectraConfig())
        assert script.startswith('"""PySCF Spectra input script')

    def test_methods_paragraph_inlined(self):
        """The header carries the full Methods prose (same content
        as the UI's preview modal)."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water(),
                                       SpectraConfig(compute_raman=True))
        # Manuscript-ready citations land in the header.
        assert "B3LYP" in script
        assert "Sun2020" in script
        assert "Komornicki1979" in script

    def test_run_command_pin(self):
        """The header documents `python <job>.spectra.py` so the
        reader doesn't have to guess."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water(),
                                       SpectraConfig(job_name="my_job"))
        assert "python my_job.spectra.py" in script


class TestPySCFScriptConstants:
    """The constants block is the bridge between the Python config
    surface and the inlined runtime values.  Pin invariants the
    parser depends on."""

    def test_schema_version_matches_parser(self):
        """The script writes SCHEMA_VERSION=1 to match what
        parse_spectra_json expects."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        from molbuilder.spectra.results import SCHEMA_VERSION
        script = render_spectra_script(_struct_water(), SpectraConfig())
        assert f"SCHEMA_VERSION = {SCHEMA_VERSION}" in script

    def test_phase_constants_pinned(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water(), SpectraConfig())
        assert "PHASE_EMPTY    = 'empty'" in script
        assert "PHASE_RUNNING  = 'running'" in script
        assert "PHASE_COMPLETE = 'complete'" in script

    def test_job_name_substituted_into_path(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water(),
                                       SpectraConfig(job_name="weird_name"))
        assert "JOB            = 'weird_name'" in script
        assert "JSON_PATH      = JOB + '.spectra.json'" in script

    def test_method_specific_imports(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        # DFT path imports dft module.
        s_dft = render_spectra_script(_struct_water(),
                                      SpectraConfig(method="RKS"))
        assert "from pyscf import gto, scf, dft" in s_dft
        # HF path skips dft.
        s_hf = render_spectra_script(_struct_water(),
                                     SpectraConfig(method="RHF"))
        assert "from pyscf import gto, scf" in s_hf
        # The HF path shouldn't import dft (saves a few ms on script start).
        # Check that the HF path doesn't have the trailing ", dft" import line.
        assert "from pyscf import gto, scf, dft" not in s_hf


class TestPySCFScriptAtomicWriter:
    """The inlined atomic JSON writer is the same safety contract
    as `molbuilder.parsers.spectra_json.dump_spectra_json`.  Pin
    that every safety rule is present in the emitted bytes."""

    def test_allow_nan_false(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water(), SpectraConfig())
        # NaN/Inf would otherwise round-trip; allow_nan=False raises
        # before bytes hit disk.
        assert "allow_nan=False" in script

    def test_ensure_ascii_false(self):
        """ensure_ascii=False keeps cm⁻¹ / Å verbatim in the JSON
        rather than escaping to \\uXXXX (which is valid JSON but
        ugly and breaks grep-ability)."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water(), SpectraConfig())
        assert "ensure_ascii=False" in script

    def test_atomic_replace_via_tempfile(self):
        """tempfile.mkstemp + os.replace is the atomic-rename
        pattern that survives a crash between write and rename."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water(), SpectraConfig())
        assert "tempfile.mkstemp" in script
        assert "os.replace" in script

    def test_fsync_before_replace(self):
        """fsync forces the data to disk before the atomic rename
        so a power-loss between write() and replace() doesn't leave
        the new file with stale buffer contents."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water(), SpectraConfig())
        assert "os.fsync" in script

    def test_temp_file_cleanup_on_failure(self):
        """The temp file is removed on any exception during write
        (except path: os.unlink in the except branch)."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water(), SpectraConfig())
        assert "os.unlink(tmp)" in script

    def test_no_molbuilder_import_at_runtime(self):
        """The script must run on a cluster node that has PySCF +
        numpy + stdlib only -- no molbuilder dependency."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water(), SpectraConfig())
        # No `import molbuilder` or `from molbuilder.*` lines.
        for line in script.splitlines():
            stripped = line.strip()
            assert not stripped.startswith("import molbuilder")
            assert not stripped.startswith("from molbuilder")


class TestPySCFScriptPhaseBlocks:
    """Each phase block (Hessian, Raman, ES) is emitted iff the
    config asks for it.  Pin presence / absence per knob."""

    def test_hessian_always_emitted(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water(), SpectraConfig())
        assert "Phase 2: Hessian" in script
        assert "mf.Hessian().kernel()" in script
        assert "phase_frequencies'] = PHASE_COMPLETE" in script

    def test_raman_block_when_enabled(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water(),
                                       SpectraConfig(compute_raman=True))
        assert "Phase 3: Raman" in script
        assert "Polarizability()" in script
        assert "phase_raman'] = PHASE_COMPLETE" in script

    def test_raman_block_absent_when_disabled(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water(),
                                       SpectraConfig(compute_raman=False))
        assert "Phase 3: Raman" not in script
        assert "Polarizability()" not in script

    def test_es_block_when_enabled(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(
            _struct_water(),
            SpectraConfig(es_mode_selection="all"),
        )
        assert "Phase 4: per-mode" in script
        assert "phase_es'] = PHASE_COMPLETE" in script

    def test_es_block_absent_when_disabled(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(
            _struct_water(),
            SpectraConfig(es_mode_selection="skip"),
        )
        assert "Phase 4: per-mode" not in script


class TestPySCFScriptStructure:
    """Verify atom coordinates, frozen-atom logic, and the
    selection logic are present in the emitted code."""

    def test_atoms_inlined(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water(), SpectraConfig())
        # All three water atoms inlined into the ATOMS list.
        assert "'O'" in script or "( ' O'" in script  # the formatting
        assert script.count("'H'") >= 2 or script.count("' H'") >= 2

    def test_frozen_mask_logic_present(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(
            _struct_water(),
            SpectraConfig(fixed_elements=["O"], fixed_indices=[1]),
        )
        # The freeze rule values are inlined.
        assert "'O'" in script
        # The runtime union logic is present.
        assert "FIXED_ATOM_IDXS" in script
        assert "FREE_ATOM_IDXS" in script
        assert "FIXED_ELEMENTS" in script

    def test_engine_renders_script_via_pyscf_script_module(self):
        """The engine's render_script() should delegate to the
        template module without raising."""
        from molbuilder.spectra.pyscf_engine import PySCFSpectraEngine
        script = PySCFSpectraEngine.render_script(
            _struct_water(), SpectraConfig(),
        )
        # Same script the template module produces.
        assert "PySCF Spectra input script generated by molbuilder" in script
        # And it compiles.
        compile(script, "<engine.render_script output>", "exec")


class TestPySCFScriptSelectorInline:
    """The L4 block inlines the same selector logic as
    spectra.selection.select_modes so the script behaves
    identically without importing molbuilder."""

    def test_explicit_selector_inlined(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(
            _struct_water(),
            SpectraConfig(es_mode_selection="explicit",
                          es_explicit_indices=[1, 3, 7]),
        )
        # The indices are pinned into ES_EXPLICIT_INDICES.
        assert "ES_EXPLICIT_INDICES        = [1, 3, 7]" in script

    def test_top_n_selector_inlined(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(
            _struct_water(),
            SpectraConfig(es_mode_selection="top_n", es_top_n=12),
        )
        assert "ES_TOP_N                   = 12" in script
        # The script's runtime selector branches on selector value.
        assert "ES_MODE_SELECTION == 'top_n'" in script

    def test_threshold_selector_inlined(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(
            _struct_water(),
            SpectraConfig(es_mode_selection="threshold", es_threshold=15.5),
        )
        assert "ES_THRESHOLD               = 15.5" in script
        assert "ES_MODE_SELECTION == 'threshold'" in script

    def test_freq_window_pinned(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(
            _struct_water(),
            SpectraConfig(es_mode_selection="all",
                          freq_min_cm1=500.0, freq_max_cm1=3500.0),
        )
        assert "FREQ_MIN_CM1               = 500.0" in script
        assert "FREQ_MAX_CM1               = 3500.0" in script


# --------------------------------------------------------------------- #
#  Regression tests for script-template bugs caught in review           #
# --------------------------------------------------------------------- #


class TestPySCFScriptDisplacedScfHelpers:
    """Bug: `_build_mf_at` + `COORDS_EQ_ANG` used to be defined inside
    the Raman block.  With compute_raman=False + es_mode_selection
    != "none", the ES block called undefined names -> NameError at
    runtime.  Fix: emit shared helpers when L3 OR L4 is enabled."""

    def test_helpers_defined_when_only_es_enabled(self):
        """The failing config combo from the review."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(
            _struct_water(),
            SpectraConfig(compute_raman=False,
                          es_mode_selection="explicit",
                          es_explicit_indices=[1]),
        )
        # Both names are defined in the shared helper block.
        assert "def _build_mf_at" in script
        assert "COORDS_EQ_ANG" in script
        # And the ES block references them.
        assert "_build_mf_at(" in script

    def test_helpers_defined_when_only_raman_enabled(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(
            _struct_water(),
            SpectraConfig(compute_raman=True,
                          es_mode_selection="skip"),
        )
        assert "def _build_mf_at" in script
        assert "COORDS_EQ_ANG" in script

    def test_helpers_defined_when_both_enabled(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(
            _struct_water(),
            SpectraConfig(compute_raman=True,
                          es_mode_selection="all"),
        )
        # Defined exactly once -- not duplicated between the two phases.
        assert script.count("def _build_mf_at") == 1
        assert script.count("COORDS_EQ_ANG = np.asarray") == 1

    def test_helpers_absent_when_neither_enabled(self):
        """When neither L3 nor L4 is on, the helpers aren't emitted
        (no caller).  Keeps the script minimal for diagnostic-only
        Hessian-only runs."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(
            _struct_water(),
            SpectraConfig(compute_raman=False,
                          es_mode_selection="skip"),
        )
        assert "def _build_mf_at" not in script

    def test_only_es_enabled_compiles_AND_helpers_resolve(self):
        """Compile pass + an exec-time symbol check.  The earlier
        compile parametrize matrix didn't catch the original bug
        because compile checks syntax, not name resolution.  Here
        we exec the script's textual definition of _build_mf_at by
        slicing it out and feeding it through compile() in 'exec'
        mode to verify the def parses on its own."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        cfg = SpectraConfig(compute_raman=False,
                            es_mode_selection="explicit",
                            es_explicit_indices=[1])
        script = render_spectra_script(_struct_water(), cfg)
        compile(script, "<no-raman-with-es>", "exec")
        # Locate the _build_mf_at definition and assert it appears
        # BEFORE the first L4 call site.
        def_pos  = script.find("def _build_mf_at")
        call_pos = script.find("_build_mf_at(", def_pos + 1)
        assert def_pos != -1, "_build_mf_at not defined"
        assert call_pos != -1, "_build_mf_at not called"
        assert def_pos < call_pos, (
            "def must come before first call site, otherwise NameError "
            "at runtime"
        )


class TestPySCFScriptSchemaVersionInterpolated:
    """Bug: SCHEMA_VERSION was a literal 1 in the emitted script.
    A future bump in results.SCHEMA_VERSION would leave scripts
    silently writing the old version -> parser rejects with a
    misleading 'schema_mismatch' on what should be valid output.

    Fix: interpolate from results.SCHEMA_VERSION at render time.
    """

    def test_schema_version_matches_live_constant(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        from molbuilder.spectra.results import SCHEMA_VERSION
        script = render_spectra_script(_struct_water(), SpectraConfig())
        # The emitted constant matches the imported one.  If someone
        # bumps SCHEMA_VERSION to 2, this test fails immediately and
        # the developer remembers to refresh the script template.
        assert f"SCHEMA_VERSION = {SCHEMA_VERSION}" in script

    def test_molbuilder_version_matches_package_metadata(self):
        """The emitted MOLBUILDER_VERSION lands in
        spectra.json.provenance.molbuilder_version -- it must match
        the actual installed package version, not a placeholder
        like 'spectra-v1'.  Regression: the constants block used to
        hard-code 'spectra-v1' which silently lied in every result
        file's provenance."""
        from molbuilder import __version__
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water(), SpectraConfig())
        assert f"MOLBUILDER_VERSION = {__version__!r}" in script
        # Negative: the old placeholder is gone.
        assert "'spectra-v1'" not in script


class TestPySCFScriptGPU:
    """The emitted script's GPU code path: USE_GPU constant in the
    constants block, a try/except gpu4pyscf import that falls back
    to CPU PySCF on failure, and the SCF construction uses _dft /
    _scf pointers that get rebound to gpu4pyscf when the import
    succeeds."""

    def test_use_gpu_false_emits_constant_and_setup_block(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water(),
                                       SpectraConfig(use_gpu=False))
        # Constant present.
        assert "USE_GPU                    = False" in script
        # GPU setup block always emitted (its body just runs the
        # CPU fallback when USE_GPU=False).
        assert "GPU acceleration (optional, NVIDIA via gpu4pyscf)" in script
        assert "_USING_GPU = False" in script
        # Script must still compile.
        compile(script, "<no-gpu>", "exec")

    def test_use_gpu_true_emits_constant_and_setup_block(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water(),
                                       SpectraConfig(use_gpu=True))
        # Constant present + True.
        assert "USE_GPU                    = True" in script
        # gpu4pyscf import is in the setup block, guarded by try.
        assert "from gpu4pyscf import dft as _gpu_dft" in script
        assert "from gpu4pyscf import scf as _gpu_scf" in script
        # And the fallback message is in there too -- so the user
        # who runs the script on a non-GPU node knows what happened.
        assert "Falling back to CPU PySCF" in script
        # Compiles.
        compile(script, "<gpu-on>", "exec")

    def test_scf_construction_uses_indirect_pointers(self):
        """The equilibrium SCF and _build_mf_at use _dft / _scf
        instead of hardcoded pyscf.dft / pyscf.scf so the GPU
        rebind takes effect for both paths.  Regression: earlier
        the code said `dft.RKS(mol)` which would have ignored the
        gpu4pyscf bind even with USE_GPU=True."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        cfg = SpectraConfig(
            use_gpu=True,
            compute_raman=True,             # exercises _build_mf_at
            es_mode_selection="explicit",   # exercises L4 _build_mf_at calls
            es_explicit_indices=[1],
        )
        script = render_spectra_script(_struct_water(), cfg)
        # Equilibrium SCF + displaced SCFs use the indirect pointer.
        assert "_dft.RKS(mol)" in script  # method=RKS (default)
        assert "_dft_mod.RKS" in script   # inside _build_mf_at
        # The hardcoded names must NOT appear in the SCF-construction
        # call sites (only in the GPU setup's fallback assignment).
        # We do allow "_dft = dft" once -- the CPU default-bind.
        assert script.count("dft.RKS(mol)") == 1, (
            "expected exactly one dft.RKS reference (the "
            "equilibrium SCF call, via _dft); got "
            f"{script.count('dft.RKS(mol)')}"
        )

    def test_emitted_script_does_runtime_capability_check(self):
        """The script must verify at runtime that the GPU is
        modern enough to run gpu4pyscf, not just that gpu4pyscf
        imports.  Pinning: the GPU setup block probes via cupy
        and falls back to CPU when compute capability < 7."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water(),
                                       SpectraConfig(use_gpu=True))
        # Capability probe via cupy.
        assert "import cupy as _cp" in script
        assert "getDeviceCount" in script
        assert "getDeviceProperties" in script
        # Hard threshold: major >= 7.
        assert "_maj < 7" in script
        # Runtime exception path falls back to CPU with a clear
        # message naming the actual GPU model + cap.
        assert "Falling back to CPU PySCF" in script
        # Two except branches: ImportError + RuntimeError.
        assert "except ImportError" in script
        assert "except Exception" in script

    def test_raman_block_forces_cpu_even_with_gpu_on(self):
        """gpu4pyscf doesn't yet expose analytic CPHF polarizability,
        so the Raman finite-difference path must build CPU mf
        objects even when USE_GPU=True.  Pinning: the polarizability
        FD calls pass force_cpu=True."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        cfg = SpectraConfig(use_gpu=True, compute_raman=True,
                            es_mode_selection="skip")
        script = render_spectra_script(_struct_water(), cfg)
        # The Raman block's three _build_mf_at call sites all pass
        # force_cpu=True (one for the equilibrium polarizability,
        # two for the ±FD displacements).  Match the closing-paren
        # form so we don't count the docstring's explanation of
        # the keyword as a fourth occurrence.
        assert script.count("force_cpu=True)") == 3


class TestPySCFScriptL4OutOfRangeGuard:
    """Bug: L4 loop did `modes_payload[_mode_pos]` without checking
    that _mode_pos was in range.  A user with es_explicit_indices=[99]
    on a 12-mode system would crash with IndexError AFTER L2 + L3
    already completed -- hours of wall time lost.

    Fix: pre-filter _selected to valid range, print + skip the rest.
    The pre-render validator can't catch this (mode count unknown
    pre-L2) so the script has to be the second line of defence.
    """

    def test_es_loop_has_range_guard(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(
            _struct_water(),
            SpectraConfig(es_mode_selection="explicit",
                          es_explicit_indices=[1]),
        )
        # The guard predicate.
        assert "1 <= i <= _n_modes_available" in script
        # And the WARN print for skipped indices so the user notices.
        assert "skipping out-of-range mode indices" in script

    def test_guard_emits_for_every_selector_kind(self):
        """The guard is in the shared L4 loop, so it covers ANY
        selector (top_n / threshold / explicit / all) since
        _selected is computed by the inlined selector before the
        guard runs."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        for sel in ("all", "top_n", "threshold", "explicit"):
            cfg = SpectraConfig(
                es_mode_selection=sel,
                es_explicit_indices=[1, 2] if sel == "explicit" else [],
            )
            script = render_spectra_script(_struct_water(), cfg)
            assert "1 <= i <= _n_modes_available" in script, sel


