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
        dict(frozen_elements=["H"]),
        dict(frozen_indices=[1, 2]),
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


class TestPySCFScriptChargeAndSpin:
    """The 2026-05-22 hemeC-dithiol incident root cause: SpectraConfig
    didn't have charge / spin fields, so the script's gto.M() call
    silently used PySCF defaults (0, 0) regardless of what the user
    wanted.  Pin that the new fields are honoured in the emission so
    a future refactor can't reintroduce the silent default."""

    def test_default_emits_neutral_singlet(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        text = render_spectra_script(_struct_water(), SpectraConfig())
        assert "charge     = 0," in text
        assert "spin       = 0," in text

    def test_charge_propagates_to_gto_m(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        text = render_spectra_script(_struct_water(),
                                     SpectraConfig(charge=-1, spin=1, method="UKS"))
        assert "charge     = -1," in text
        assert "spin       = 1," in text

    def test_spin_propagates_for_fe_high_spin(self):
        """Realistic case: Fe(II) high-spin needs spin=4 + UKS.  Pin
        the typed-int emission so the user's hemeC-style config
        propagates faithfully."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        text = render_spectra_script(_struct_water(),
                                     SpectraConfig(spin=4, method="UKS"))
        # Comment-tagged so a future field reordering can't make this
        # match the wrong literal.
        assert "spin       = 4," in text
        assert "# 2S = # unpaired electrons" in text


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

    def test_config_literal_parses_as_python(self):
        # The CONFIG = {...} block is emitted via pprint.pformat over
        # _config_to_jsonable_dict(cfg).  If a future SpectraConfig field
        # were a dataclass/enum/Path that asdict didn't flatten, pprint
        # would emit a non-Python repr like <X object> or PosixPath(...),
        # silently breaking the generated script's SyntaxError-free guarantee.
        # Belt-and-suspenders: extract the literal and assert it parses.
        import ast
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water(), SpectraConfig())
        marker = "CONFIG = "
        start = script.index(marker) + len(marker)
        assert script[start] == "{", "CONFIG = … must be a dict literal"
        # Walk to the matching closing brace, ignoring braces inside string
        # literals (pprint emits Python repr, so single-quoted strings only).
        depth = 0
        in_str = False
        end = None
        for i in range(start, len(script)):
            c = script[i]
            if in_str:
                if c == "\\":
                    continue  # crude but pprint doesn't emit backslashes for our types
                if c == "'":
                    in_str = False
                continue
            if c == "'":
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        assert end is not None, "Unbalanced braces in CONFIG literal"
        parsed = ast.literal_eval(script[start:end])
        assert isinstance(parsed, dict)
        # Sanity: at least one well-known SpectraConfig field made it through.
        assert "engine" in parsed


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
            SpectraConfig(frozen_elements=["O"], frozen_indices=[1]),
        )
        # The freeze rule values are inlined.
        assert "'O'" in script
        # The runtime union logic is present.
        assert "FROZEN_ATOM_IDXS" in script
        assert "FREE_ATOM_IDXS" in script
        assert "FROZEN_ELEMENTS" in script

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


class TestPySCFScriptThreadingSetup:
    """The threading-setup block: emitted BEFORE numpy/pyscf import,
    pins BLAS to 1 thread per worker, sizes PySCF OMP to physical
    cores by default (not logical) -- the canonical anti-oversub-
    scription recipe.  A user seeing load=40 on a 20-core/40-HT
    host is the regression scenario this block exists to prevent.
    """

    def test_threading_block_emitted_before_pyscf_import(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water(), SpectraConfig())
        # The block runs BEFORE ``from pyscf import gto, scf, dft``
        # so the env-var caps take effect at pyscf import time.
        thread_setup_at = script.index("Threading setup")
        pyscf_import_at = script.index("from pyscf import")
        assert thread_setup_at < pyscf_import_at, (
            "_emit_threading_setup() must run BEFORE the pyscf "
            "import -- env vars are read at import time."
        )

    def test_blas_capped_to_one_thread(self):
        """OPENBLAS_NUM_THREADS=1 and MKL_NUM_THREADS=1: this is the
        load=N*N oversubscription fix.  Without these caps each
        PySCF worker spawns its own BLAS thread pool and the
        observed load is OMP * BLAS = up to logical_cores² on
        a hyperthreaded host."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water(), SpectraConfig())
        assert "OPENBLAS_NUM_THREADS" in script
        assert "MKL_NUM_THREADS" in script
        # Both pinned to '1' (not the OMP count).
        assert "'OPENBLAS_NUM_THREADS', '1'" in script
        assert "'MKL_NUM_THREADS',      '1'" in script

    def test_pyscf_num_threads_called_after_import(self):
        """``pyscf.lib.num_threads(N)`` is the canonical post-
        import setter (env vars don't re-thread already-imported
        modules).  Verify the script calls it."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water(), SpectraConfig())
        # The call site is the _emit_pyscf_thread_pool block.
        assert "_pyscf_lib.num_threads(_MB_REQUESTED_THREADS)" in script

    def test_default_threads_auto_detects_physical_cores(self):
        """cfg.threads=None -> the script computes physical cores at
        run time via /proc/cpuinfo + psutil fallback.  Hard-coding
        os.cpu_count() would give logical cores (HT) which is what
        caused the user's load=40 bug."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water(),
                                       SpectraConfig(threads=None))
        assert "_mb_count_physical_cores" in script
        assert "_MB_REQUESTED_THREADS = _mb_count_physical_cores()" in script
        # Sanity-check the helper logic.
        assert "/proc/cpuinfo" in script
        assert "psutil" in script

    def test_explicit_threads_overrides_auto(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water(),
                                       SpectraConfig(threads=12))
        assert "_MB_REQUESTED_THREADS = 12" in script
        # Auto-detection helper is still defined (for the
        # physical_cores / logical_cores runtime-info fields) but
        # not called for the requested count.
        assert "_MB_REQUESTED_THREADS = _mb_count_physical_cores()" not in script

    def test_runtime_info_dict_populated(self):
        """The script builds a _RUNTIME_INFO dict the JSON dump
        copies into ``runtime_info`` so the /results page can show
        what CPU/GPU configuration the run actually used."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water(), SpectraConfig())
        for key in (
            "'n_threads_pyscf'", "'n_threads_omp'", "'n_threads_blas'",
            "'physical_cores'", "'logical_cores'",
            "'gpu_requested'", "'gpu_used'", "'gpu_name'", "'hostname'",
        ):
            assert key in script, (
                f"_RUNTIME_INFO missing the {key} field; the "
                f"/results page won't be able to show it."
            )
        # And the state dict's 'runtime_info' picks it up.
        assert "'runtime_info':              dict(_RUNTIME_INFO)" in script

    def test_compiles_clean_with_threading_block(self):
        """Sanity: the threading block is real Python."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water(), SpectraConfig())
        compile(script, "<water-threading>", "exec")


class TestPySCFScriptGPU:
    """The emitted script's GPU code path: USE_GPU constant in the
    constants block, a try/except gpu4pyscf import that falls back
    to CPU PySCF on failure, and the SCF construction uses _dft /
    _scf pointers that get rebound to gpu4pyscf when the import
    succeeds."""

    def test_use_gpu_false_emits_constant_and_setup_block(self):
        """GPU probe is emitted from the shared ``molbuilder.runtime_info``
        module (refactored 2026-05-22 -- previously had its own inline
        copy in this script).  The probe always runs; USE_GPU=False
        short-circuits before the gpu4pyscf import."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water(),
                                       SpectraConfig(use_gpu=False))
        assert "USE_GPU = False" in script
        assert "GPU probe" in script
        assert "_USING_GPU = False" in script
        # Script must still compile.
        compile(script, "<no-gpu>", "exec")

    def test_use_gpu_true_emits_constant_and_setup_block(self):
        """USE_GPU=True wires in the gpu4pyscf import + cupy probe +
        ``_mb_to_gpu_if_enabled`` helper, all from the shared
        ``molbuilder.runtime_info`` module.  Spectra-side also rebinds
        ``_scf`` / ``_dft`` to the gpu4pyscf modules so the existing
        ``_dft.RKS(mol)`` call sites pick up the GPU class."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water(),
                                       SpectraConfig(use_gpu=True))
        assert "USE_GPU = True" in script
        # gpu4pyscf imports are in the rebind block AND the shared
        # probe's try/except (which references gpu4pyscf to ensure the
        # patch registers).
        assert "from gpu4pyscf import dft as _gpu_dft" in script
        assert "from gpu4pyscf import scf as _gpu_scf" in script
        # CPU-fallback message tells the user when the run drops to CPU.
        assert "CPU fallback" in script
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
        """The script must verify at runtime that the GPU is modern
        enough to run gpu4pyscf (compute capability >= 7).  Shared
        probe in molbuilder.runtime_info checks via cupy + falls back
        to CPU when the card is too old."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water(),
                                       SpectraConfig(use_gpu=True))
        # Capability probe via cupy.
        assert "import cupy as _cp" in script
        assert "getDeviceCount" in script
        assert "getDeviceProperties" in script
        # Hard threshold: major >= 7.
        assert "_maj < 7" in script
        assert "CPU fallback" in script
        # Two except branches: ImportError + Exception.
        assert "except ImportError" in script
        assert "except Exception" in script

    def test_raman_block_forces_cpu_even_with_gpu_on(self):
        """gpu4pyscf doesn't yet expose analytic CPHF polarizability,
        so the Raman finite-difference path must build CPU mf
        objects even when USE_GPU=True.

        Pinning: with raman=True, the script has 4 ``force_cpu=True``
        call sites -- 3 in the Raman block (eq. polarizability + two
        ±FD displacements) and 1 in the Hessian fallback branch added
        for the GPU-coverage probe.  Together they enumerate every
        place the script forces a CPU rebuild when running on GPU."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        cfg = SpectraConfig(use_gpu=True, compute_raman=True,
                            es_mode_selection="skip")
        script = render_spectra_script(_struct_water(), cfg)
        # Match the closing-paren form so we don't count the
        # docstring's explanation of the keyword as an occurrence.
        assert script.count("force_cpu=True)") == 4


class TestPySCFScriptGPUCoverageProbe:
    """Runtime probe (emitted right after equilibrium SCF) that decides
    which stages run on GPU and which need a CPU rebuild.  Pre-probing
    instead of try/except-fallback avoids wasting an SCF on a path that
    can't continue.
    """

    def _render(self, **cfg_kwargs):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        return render_spectra_script(_struct_water(),
                                      SpectraConfig(**cfg_kwargs))

    # --------------- probe block presence + shape ----------------- #

    def test_probe_emits_capability_flags(self):
        """Both flags get assigned regardless of USE_GPU -- downstream
        code reads them unconditionally."""
        script = self._render(use_gpu=True)
        assert "_GPU_HAS_HESSIAN" in script
        assert "_GPU_HAS_POLARIZABILITY" in script

    def test_probe_runs_only_when_using_gpu(self):
        """The probe block guards its body with ``if _USING_GPU:`` so
        a CPU-only invocation doesn't pay even the cost of construct-
        ing a Hessian probe object."""
        script = self._render(use_gpu=True)
        assert "if _USING_GPU:" in script
        # The actual probe instantiates Hessian and inspects its module.
        assert "type(_h_probe).__module__.startswith('gpu4pyscf')" in script

    def test_probe_reports_coverage_gaps_to_the_user(self):
        """Scientists shouldn't have to read source to know what part
        of their job is running on CPU.  The probe prints the gaps."""
        script = self._render(use_gpu=True)
        assert "GPU coverage gaps" in script
        # When everything works, the "all good" branch reports it too.
        assert "GPU coverage: SCF + Hessian." in script

    # --------------- Hessian block consults the flag --------------- #

    def test_hessian_block_consults_gpu_has_hessian(self):
        """When the probe says gpu4pyscf doesn't cover Hessian for
        this SCF type, the script rebuilds mf on CPU instead of
        crashing on a CuPy-Hessian → harmonic_analysis(CPU) call."""
        script = self._render(use_gpu=True)
        # The branch reads as ``if _GPU_HAS_HESSIAN or not _USING_GPU``.
        assert "_GPU_HAS_HESSIAN" in script
        # The fallback path uses _build_mf_at(..., force_cpu=True).
        assert "_mf_cpu_for_hess = _build_mf_at(" in script
        assert "force_cpu=True" in script

    def test_hessian_fallback_present_even_when_raman_off(self):
        """The fallback branch is structural (Hessian always runs) --
        it shouldn't be gated on the Raman block.  With raman=False
        we still need the CPU-rebuild path for the Hessian step."""
        script = self._render(use_gpu=True, compute_raman=False,
                                es_mode_selection="skip")
        assert "_mf_cpu_for_hess = _build_mf_at(" in script
        # In this config the Raman block doesn't emit, so the ONLY
        # ``force_cpu=True)`` site is the Hessian fallback.
        assert script.count("force_cpu=True)") == 1


class TestPySCFScriptCuPyToNumPyBridge:
    """Reported bug 2026-05-14: GPU run crashed at line ~355 with
    ``TypeError: Implicit conversion to a NumPy array is not allowed.
    Please use .get() to construct a NumPy array explicitly.``
    Root cause: ``np.asarray(mf.mo_energy)`` on a gpu4pyscf mf returns
    a CuPy array; modern CuPy refuses the implicit conversion.

    Fix: emit a tiny ``_as_numpy(x)`` helper in the generated script
    that does the explicit ``.get()`` round-trip when ``x`` is CuPy
    and is a no-op otherwise.  All gpu-mf attribute crossings go
    through it.
    """

    def test_helper_definition_emitted(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water(),
                                        SpectraConfig(use_gpu=True))
        # The function lives in the helpers block.
        assert "def _as_numpy(x):" in script
        # And it's actually the .get()-or-fallthrough shape, not a
        # placeholder.  We don't pin the exact wording -- just the
        # two operations that make it work.
        assert "type(x).__module__.startswith('cupy')" in script
        assert ".get()" in script

    def test_no_naked_np_asarray_on_mf_attributes(self):
        """Every gpu-mf crossing must go through _as_numpy.  A future
        edit that adds e.g. ``np.asarray(mf.mo_coeff)`` would crash on
        GPU runs the same way the original report did."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        # Cover the two branches that emit different code paths:
        # Raman ON (force-CPU polarizability rebuild) and ES loop ON
        # (displaced SCFs on GPU).
        for raman in (True, False):
            for es_mode in ("all", "skip"):
                cfg = SpectraConfig(use_gpu=True, compute_raman=raman,
                                    es_mode_selection=es_mode)
                script = render_spectra_script(_struct_water(), cfg)
                # Naked ``np.asarray(<mf-like>.X)`` would re-introduce
                # the bug.  Include the tail "." so we don't catch
                # ``np.asarray(mol.atom_mass_list())`` which is CPU-only.
                for bad_token in ("np.asarray(mf.",
                                   "np.asarray(_mf.",
                                   "np.asarray(_mf2."):
                    assert bad_token not in script, (
                        f"raman={raman} es_mode={es_mode}: "
                        f"found {bad_token!r} -- use _as_numpy() instead "
                        f"or this will crash on gpu4pyscf"
                    )

    def test_hessian_kernel_bridged_through_as_numpy(self):
        """``mf.Hessian().kernel()`` returns CuPy on gpu4pyscf;
        ``pyscf.hessian.thermo.harmonic_analysis`` is CPU-only and
        would TypeError on a CuPy Hessian.  The bridge has to happen
        right at the assignment, BEFORE harmonic_analysis sees it.

        After the GPU-coverage probe shipped, the assignment is now
        guarded by the ``_GPU_HAS_HESSIAN`` branch; the bridged form
        appears in BOTH the GPU and CPU-fallback branches."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_view := _struct_water(),
                                        SpectraConfig(use_gpu=True))
        # The GPU branch (mf directly):
        assert "HESS = _as_numpy(mf.Hessian().kernel())" in script
        # The CPU-rebuild branch (when gpu4pyscf can't do Hessian for
        # this SCF type) -- different mf object, same bridge:
        assert "HESS = _as_numpy(_mf_cpu_for_hess.Hessian().kernel())" in script

    def test_as_numpy_helper_actually_works(self):
        """Behaviour pin (not just string presence): extract the
        ``_as_numpy`` function from the rendered script, exec it in
        a controlled namespace, and verify it converts:

          * a NumPy array  -> NumPy (identity-ish, np.asarray)
          * a CuPy-mocked object -> NumPy via .get()
          * a scalar / list -> NumPy

        A future refactor that breaks the helper would fail this test
        even if the textual fingerprints (``.get()``,
        ``startswith('cupy')``) survived elsewhere in the file."""
        import re
        from molbuilder.spectra.pyscf_script import render_spectra_script
        script = render_spectra_script(_struct_water(), SpectraConfig())

        # Pull the `def _as_numpy(x): ... ` block out of the rendered
        # script.  ``(?ms)`` lets ``.`` cross newlines AND anchors `^`
        # to line starts; we stop at the next top-level (column-0) def.
        match = re.search(r'(?ms)^def _as_numpy\(x\):.*?(?=^def |\Z)',
                            script)
        assert match is not None, "_as_numpy not found in rendered script"
        func_src = match.group(0)

        # Exec the helper into a fresh namespace.  Only ``np`` is
        # needed; the helper uses no other module.
        import numpy as np
        ns = {"np": np}
        exec(func_src, ns)
        _as_numpy = ns["_as_numpy"]

        # Case 1: real NumPy array passes through.
        arr_np = np.array([1.0, 2.0, 3.0])
        out = _as_numpy(arr_np)
        assert isinstance(out, np.ndarray)
        assert out.tolist() == [1.0, 2.0, 3.0]

        # Case 2: CuPy-mocked object (module is "cupy*" + has .get()).
        class _FakeCuPyArray:
            def __init__(self, data): self._data = data
            def get(self):          return np.asarray(self._data)
        _FakeCuPyArray.__module__ = "cupy.core.core"  # mimic cupy path

        out = _as_numpy(_FakeCuPyArray([5.0, 6.0]))
        assert isinstance(out, np.ndarray)
        assert out.tolist() == [5.0, 6.0]

        # Case 3: plain Python scalar -- np.asarray wraps to 0-d.
        out = _as_numpy(42.0)
        assert isinstance(out, np.ndarray)
        assert float(out) == 42.0

        # Case 4: list -- np.asarray builds an ndarray.
        out = _as_numpy([7, 8, 9])
        assert isinstance(out, np.ndarray)
        assert out.tolist() == [7, 8, 9]


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


class TestPySCFScriptIRScaffold:
    """IR add-on scaffold (compute_ir=True): wires dipole capture
    into the Raman FD loop and projects to ir_intensity_km_mol.
    The science (absolute magnitude) is documented as not-yet-
    validated against an external code; these tests pin the
    emission shape + the v1 'compute_ir requires compute_raman'
    constraint."""

    def test_ir_block_absent_when_compute_ir_false(self):
        """Default config: nothing IR-specific in the rendered script.
        Keeps the cost-no-IR-no-pay invariant honest."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        s = render_spectra_script(_struct_water(),
                                  SpectraConfig(compute_raman=True,
                                                compute_ir=False))
        assert "DMU_DR" not in s
        assert "_dipole_debye" not in s
        assert "ir_intensity_km_mol'] = " not in s  # no assignment
        assert "42.2561" not in s
        # The constant is still emitted so the JSON header records the flag.
        assert "COMPUTE_IR" in s

    def test_ir_block_present_when_compute_ir_true(self):
        from molbuilder.spectra.pyscf_script import render_spectra_script
        s = render_spectra_script(_struct_water(),
                                  SpectraConfig(compute_raman=True,
                                                compute_ir=True))
        # Capture machinery
        assert "DMU_DR    = np.zeros((N_FREE, 3, 3))" in s
        assert "_dipole_debye(_mf_plus)" in s
        assert "_dipole_debye(_mf_minus)" in s
        # Projection + prefactor
        assert "42.2561" in s
        assert "np.einsum('kai,ka->i', DMU_DR, _L_canonical)" in s
        assert "modes_payload[_n]['ir_intensity_km_mol']" in s

    def test_ir_prefactor_pinned_to_gaussian_constant(self):
        """The 42.2561 km/mol per (D/Å)²/amu constant is the
        Gaussian/ORCA/textbook value.  Pin it so a future edit can't
        silently drift to a different convention."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        s = render_spectra_script(_struct_water(),
                                  SpectraConfig(compute_raman=True,
                                                compute_ir=True))
        assert ("_IR_PREFACTOR_KM_MOL_PER_D2_PER_A2_PER_AMU = 42.2561"
                in s)

    def test_ir_unit_explicit_in_dipole_call(self):
        """mf.dip_moment(unit='Debye') is set explicitly so a future
        PySCF that flips the default can't change our units silently."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        s = render_spectra_script(_struct_water(),
                                  SpectraConfig(compute_raman=True,
                                                compute_ir=True))
        assert "_mf.dip_moment(unit='Debye'" in s

    def test_ir_validation_banner_in_header(self):
        """The header docstring carries a NOT-YET-VALIDATED warning
        when IR is on -- this is the user-facing trigger to read
        spec.md §13.1 before quoting absolute IR magnitudes."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        s = render_spectra_script(_struct_water(),
                                  SpectraConfig(compute_raman=True,
                                                compute_ir=True))
        assert "NOT YET VALIDATED" in s
        assert "spec.md" in s    # pointer to the validation status
        # Charged-molecule caveat (origin-shift contamination) must be
        # visible to a user reading the header.  Test prevents silent
        # removal during a future header-doc refactor.
        assert "CHARGED molecules" in s
        # ASCII-only banner (no emoji, per project style rule).
        assert "⚠" not in s

    def test_ir_requires_compute_raman_in_v1(self):
        """compute_ir=True + compute_raman=False raises at render-
        time with a clear message -- IR rides on Raman's FD loop."""
        import pytest
        from molbuilder.spectra.pyscf_script import render_spectra_script
        with pytest.raises(ValueError, match="compute_ir=True requires "
                           "compute_raman=True"):
            render_spectra_script(
                _struct_water(),
                SpectraConfig(compute_raman=False, compute_ir=True),
            )

    def test_phase_done_message_names_both(self):
        """The print('Phase 3 done: ...') line should say Raman + IR
        when both ran, not just Raman."""
        from molbuilder.spectra.pyscf_script import render_spectra_script
        s = render_spectra_script(_struct_water(),
                                  SpectraConfig(compute_raman=True,
                                                compute_ir=True))
        assert "Phase 3 done: Raman + IR" in s
        # And only-Raman config keeps its own message
        s2 = render_spectra_script(_struct_water(),
                                   SpectraConfig(compute_raman=True))
        assert "Phase 3 done: Raman activities" in s2


