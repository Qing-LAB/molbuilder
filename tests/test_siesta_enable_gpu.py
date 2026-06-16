"""L1/L2 tests for the SIESTA GPU toggle.

Covers the three-way contract introduced together:

  * ``SiestaConfig.enable_gpu``: dataclass field + form metadata.
  * ``render_fdf``: emits ``Diag.ELPA.GPU .true.`` iff the toggle is on.
  * ``runwrap._fdf_requests_gpu`` + ``runwrap.write_run_wrapper``:
    inspects the rendered .fdf for the keyword and routes the job
    into the ``molbuilder-siesta-gpu`` env when set.

The same .fdf is portable between CPU and GPU SIESTA -- the keyword
ONLY changes runtime behaviour, never the input geometry / basis /
physics.  These tests pin that contract end-to-end so a regression
in any one of the three layers (config, generator, wrapper) fails
loudly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molbuilder.config.siesta import SiestaConfig
from molbuilder.diagnostics import Capabilities, set_capabilities
from molbuilder.siesta.input import render_fdf
from molbuilder.structure import Structure
from molbuilder import runwrap as _runwrap


# --------------------------------------------------------------------- #
#  Fixtures                                                              #
# --------------------------------------------------------------------- #


def _mk_struct() -> Structure:
    """Minimal 1-atom water-stub: enough for render_fdf to succeed."""
    return Structure(
        elements      = ["H"],
        positions     = np.zeros((1, 3)),
        atom_names    = ["H1"],
        residue_ids   = [1],
        residue_names = ["UNL"],
        chain_ids     = ["A"],
    )


@pytest.fixture
def caps_with_gpu_env():
    """Bind a capabilities snapshot that has both CPU and GPU SIESTA
    envs registered, so write_run_wrapper's routing decision is
    observable in the returned text."""
    set_capabilities(Capabilities(
        runtime_config={},
        conda_binary="/usr/bin/conda",
        conda_envs=frozenset({"molbuilder-siesta", "molbuilder-siesta-gpu"}),
    ))
    yield
    set_capabilities(Capabilities(runtime_config={}))


# --------------------------------------------------------------------- #
#  L1: SiestaConfig field shape                                          #
# --------------------------------------------------------------------- #


def test_enable_gpu_default_is_off():
    """Safety default: a brand-new config does NOT request GPU.  This
    is load-bearing because the same .fdf is portable across CPU/GPU
    envs; a True default would silently route every SIESTA job into
    the GPU env regardless of whether the user has one installed."""
    cfg = SiestaConfig()
    assert cfg.enable_gpu is False


def test_enable_gpu_metadata_is_present():
    """The form schema is auto-built from dataclass metadata.  Missing
    a key here means the field renders as a default text input (or
    not at all), not the boolean checkbox we want."""
    field = SiestaConfig.__dataclass_fields__["enable_gpu"]
    md = field.metadata
    # 2026-06-16 form restructure: merged "Relaxation" + "Parallel
    # execution" into a single "Compute & budget" section so the
    # physics axis (System -> Basis -> XC -> SCF -> Spin -> Output)
    # stays compact.  See SiestaConfig._form_section_order.
    assert md["section"] == "Compute & budget"
    assert md["workflow_group"] == "budget"
    assert md["id_suffix"] == "enable-gpu"
    # The engine_key must reference the SIESTA fdf keyword so the
    # methods-text generator can cite it; an empty value would let
    # the field silently drift from the keyword the generator emits.
    assert "Diag.ELPA.GPU" in md["engine_key"]


# --------------------------------------------------------------------- #
#  L1: render_fdf emission                                               #
# --------------------------------------------------------------------- #


def test_render_fdf_omits_gpu_keyword_by_default():
    """Off by default: the rendered .fdf must NOT contain the keyword
    when enable_gpu=False, so a job rendered for a CPU-only host
    never accidentally routes to GPU on a different machine."""
    fdf = render_fdf(_mk_struct(), SiestaConfig())
    assert "Diag.ELPA.GPU" not in fdf


def test_render_fdf_emits_gpu_keyword_when_enabled():
    """``Diag.ELPA.GPU .true.`` is the modern (5.4.2) spelling per
    Src/diag_option.F90:139.  Asserting the literal value catches a
    typo like ``T``, ``Yes``, etc. -- they're ACCEPTED by fdf_get
    but the run-wrapper detector also has to accept them, and pinning
    one form keeps the contract simple."""
    cfg = SiestaConfig(enable_gpu=True)
    fdf = render_fdf(_mk_struct(), cfg)
    assert "Diag.ELPA.GPU      .true." in fdf


def test_render_fdf_emits_diag_algorithm_with_gpu():
    """When GPU is on, the generator MUST also emit ``Diag.Algorithm
    ELPA-1STAGE`` -- the GPU keyword alone is silently ignored unless
    SIESTA is routing through ELPA.  Confirmed against SIESTA source
    at Src/diag_option.F90:213-225 (default ScaLAPACK path) and the
    user-visible failure: nvidia-smi at 0% utilisation while SCF is
    iterating happily on CPU."""
    cfg = SiestaConfig(enable_gpu=True)
    fdf = render_fdf(_mk_struct(), cfg)
    assert "Diag.Algorithm     ELPA-1STAGE" in fdf


def test_render_fdf_elpa_algorithm_choice_propagates():
    """The user can switch the ELPA solver variant via the
    ``elpa_algorithm`` field (default ELPA-1STAGE -> ELPA-2STAGE for
    benchmarking or CPU-tuned configs).  Pin that the choice flows
    through to the rendered keyword."""
    cfg = SiestaConfig(enable_gpu=True, elpa_algorithm="ELPA-2STAGE")
    fdf = render_fdf(_mk_struct(), cfg)
    assert "Diag.Algorithm     ELPA-2STAGE" in fdf
    assert "Diag.Algorithm     ELPA-1STAGE" not in fdf


def test_render_fdf_omits_diag_algorithm_without_gpu():
    """When GPU is off, leave Diag.Algorithm OUT entirely so SIESTA
    falls through to its default Divide-and-Conquer ScaLAPACK path
    (which is the right CPU recipe).  The elpa_algorithm field
    setting is ignored in this case."""
    cfg = SiestaConfig(enable_gpu=False, elpa_algorithm="ELPA-2STAGE")
    fdf = render_fdf(_mk_struct(), cfg)
    assert "Diag.Algorithm" not in fdf


def test_elpa_algorithm_field_metadata():
    """Pin the form-schema metadata so the UI renders this as a
    dropdown (not a free-text field) with the correct two choices."""
    field = SiestaConfig.__dataclass_fields__["elpa_algorithm"]
    md = field.metadata
    # 2026-06-16 form restructure: merged "Relaxation" + "Parallel
    # execution" into a single "Compute & budget" section so the
    # physics axis (System -> Basis -> XC -> SCF -> Spin -> Output)
    # stays compact.  See SiestaConfig._form_section_order.
    assert md["section"] == "Compute & budget"
    assert md["workflow_group"] == "budget"
    assert md["choices"] == ("ELPA-1STAGE", "ELPA-2STAGE")
    assert md["engine_key"] == "Diag.Algorithm"
    # Default matches the GPU-preferred variant.
    default_field = SiestaConfig().elpa_algorithm
    assert default_field == "ELPA-1STAGE"


# --------------------------------------------------------------------- #
#  L1: _fdf_requests_gpu detector                                        #
# --------------------------------------------------------------------- #


def _write_fdf(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "job.fdf"
    p.write_text(body, encoding="utf-8")
    return p


def test_fdf_requests_gpu_modern_spelling(tmp_path):
    p = _write_fdf(tmp_path, "Diag.ELPA.GPU .true.\n")
    assert _runwrap._fdf_requests_gpu(p) is True


def test_fdf_requests_gpu_alias_spelling(tmp_path):
    """SIESTA 5.4.2 fdf_get accepts both ``Diag.ELPA.UseGPU`` (older)
    and ``Diag.ELPA.GPU`` (newer) for the same internal flag.  Our
    detector mirrors that so an .fdf written against either spelling
    routes correctly."""
    p = _write_fdf(tmp_path, "Diag.ELPA.UseGPU .true.\n")
    assert _runwrap._fdf_requests_gpu(p) is True


@pytest.mark.parametrize("truthy", [".true.", "true", "yes", "T", "Y", "1"])
def test_fdf_requests_gpu_truthy_values(tmp_path, truthy):
    """fdf_get's truthy alphabet -- pinning that the detector accepts
    the same set, not just the canonical ``.true.``.  Any user who
    hand-edits the file with a shorter form must route the same way
    SIESTA itself will read it."""
    p = _write_fdf(tmp_path, f"Diag.ELPA.GPU {truthy}\n")
    assert _runwrap._fdf_requests_gpu(p) is True


def test_fdf_requests_gpu_false_value(tmp_path):
    p = _write_fdf(tmp_path, "Diag.ELPA.GPU .false.\n")
    assert _runwrap._fdf_requests_gpu(p) is False


def test_fdf_requests_gpu_missing_keyword(tmp_path):
    p = _write_fdf(tmp_path, "SystemLabel siesta\nMeshCutoff 300.0 Ry\n")
    assert _runwrap._fdf_requests_gpu(p) is False


def test_fdf_requests_gpu_last_assignment_wins(tmp_path):
    """SIESTA's fdf reader takes the LAST occurrence of a duplicated
    key (read_options.F90 semantics).  An .fdf that turns GPU on then
    off must route to the CPU env -- otherwise a user disabling GPU
    via a trailing override would silently still go to the GPU env."""
    body = "Diag.ELPA.GPU .true.\nDiag.ELPA.GPU .false.\n"
    p = _write_fdf(tmp_path, body)
    assert _runwrap._fdf_requests_gpu(p) is False


def test_fdf_requests_gpu_unreadable_returns_false(tmp_path):
    """Missing file -> safe default (CPU env).  Routing must never
    raise from inside write_run_wrapper -- the wrapper is on a
    user-input boundary and an OSError here would propagate as a
    500."""
    missing = tmp_path / "absent.fdf"
    assert _runwrap._fdf_requests_gpu(missing) is False


# --------------------------------------------------------------------- #
#  L2: write_run_wrapper routing                                         #
# --------------------------------------------------------------------- #


def test_write_run_wrapper_routes_to_gpu_env_when_keyword_set(
        tmp_path, caps_with_gpu_env):
    """End-to-end: render_fdf(enable_gpu=True) -> file with keyword ->
    write_run_wrapper picks ``molbuilder-siesta-gpu``."""
    cfg = SiestaConfig(enable_gpu=True)
    fdf_text = render_fdf(_mk_struct(), cfg)
    fdf = tmp_path / "job.fdf"
    fdf.write_text(fdf_text, encoding="utf-8")
    wrapper_path = _runwrap.write_run_wrapper(fdf)
    wrapper_text = wrapper_path.read_text(encoding="utf-8")
    assert "molbuilder-siesta-gpu" in wrapper_text


def test_write_run_wrapper_routes_to_cpu_env_by_default(
        tmp_path, caps_with_gpu_env):
    """Symmetric: enable_gpu=False -> file without keyword ->
    ``molbuilder-siesta`` (CPU env)."""
    fdf_text = render_fdf(_mk_struct(), SiestaConfig())
    fdf = tmp_path / "job.fdf"
    fdf.write_text(fdf_text, encoding="utf-8")
    wrapper_path = _runwrap.write_run_wrapper(fdf)
    wrapper_text = wrapper_path.read_text(encoding="utf-8")
    assert "molbuilder-siesta-gpu" not in wrapper_text
    assert "molbuilder-siesta" in wrapper_text


def test_gpu_wrapper_keeps_siesta_template_intact(
        tmp_path, caps_with_gpu_env):
    """Regression for the 2026-06-15 ``.pyscf.log`` leak + unbalanced-
    quote shell-syntax error.

    Earlier the routing fix mutated ``category`` from ``"siesta"`` to
    ``"siesta-gpu"``, which silently disabled every downstream
    ``if category == "siesta":`` branch in ``runwrap.py``.  The wrapper
    then fell through to the PySCF code path -- emitting a
    ``-run0.pyscf.log`` filename for a SIESTA job AND an unclosed
    quote in the template (the SIESTA-specific siesta_args_block
    never rendered, so the heredoc it lived inside never closed).
    Pin both signatures here.
    """
    import subprocess
    cfg = SiestaConfig(enable_gpu=True)
    fdf_text = render_fdf(_mk_struct(), cfg)
    fdf = tmp_path / "job.fdf"
    fdf.write_text(fdf_text, encoding="utf-8")
    wrapper_path = _runwrap.write_run_wrapper(fdf)
    wrapper_text = wrapper_path.read_text(encoding="utf-8")
    # Signature 1: SIESTA log extension (.out), NOT .pyscf.log.
    assert ".pyscf.log" not in wrapper_text, (
        "wrapper fell through to the PySCF code path -- the SIESTA "
        "branch didn't render.  Check that the routing fix overrides "
        "the ENV lookup only, leaving `category` untouched."
    )
    # Signature 2: bash syntax-check the rendered wrapper.  An
    # unclosed heredoc / unbalanced quote / dangling brace would fail
    # ``bash -n`` here.  Catches the wider class of partial-template
    # bugs the .pyscf.log signature would miss when the PySCF branch
    # itself rendered cleanly.
    cp = subprocess.run(
        ["bash", "-n", str(wrapper_path)],
        capture_output=True, text=True, timeout=15,
    )
    assert cp.returncode == 0, (
        f"rendered wrapper failed bash -n syntax check.  stderr:\n"
        f"{cp.stderr}"
    )


def test_gpu_wrapper_honors_user_set_mpi_np(tmp_path, caps_with_gpu_env):
    """Regression for the 2026-06-15 silent-shadow bug: when the user
    explicitly sets ``mpi_np=4`` on the form, the rendered wrapper
    MUST bake that literal value into ``_mpi_np_default``, not defer
    to the bash-runtime ``$_gpu_mpi_np_default`` (which would
    silently use the hardware-probed policy value and ignore the
    user's choice).  Same shape as the prior bug catch: an
    explicit form-set value got silently dropped because the GPU
    code path unconditionally chose the policy default."""
    cfg = SiestaConfig(enable_gpu=True)
    fdf_text = render_fdf(_mk_struct(), cfg)
    fdf = tmp_path / "job.fdf"
    fdf.write_text(fdf_text, encoding="utf-8")
    wrapper_path = _runwrap.write_run_wrapper(fdf, mpi_np=4, omp_threads=3)
    wrapper_text = wrapper_path.read_text(encoding="utf-8")
    # Both literals must appear; the shell-var fallback names must NOT
    # be the source of either ``_*_default`` line for THIS run.
    assert "_mpi_np_default=4" in wrapper_text
    assert "_omp_threads_default=3" in wrapper_text
    assert "_mpi_np_default=$_gpu_mpi_np_default" not in wrapper_text
    assert "_omp_threads_default=$_omp_default" not in wrapper_text


def test_gpu_wrapper_uses_policy_default_when_user_unset(
        tmp_path, caps_with_gpu_env):
    """Companion test: when the user leaves mpi_np / omp_threads as
    auto (None), GPU mode SHOULD use the runtime-probed policy via
    the shell vars.  Pinning that the prior fix didn't disable the
    auto path for users who actually want it."""
    cfg = SiestaConfig(enable_gpu=True)
    fdf_text = render_fdf(_mk_struct(), cfg)
    fdf = tmp_path / "job.fdf"
    fdf.write_text(fdf_text, encoding="utf-8")
    # No mpi_np / omp_threads kwargs -> auto path.
    wrapper_path = _runwrap.write_run_wrapper(fdf)
    wrapper_text = wrapper_path.read_text(encoding="utf-8")
    assert "_mpi_np_default=$_gpu_mpi_np_default" in wrapper_text
    assert "_omp_threads_default=$_omp_default" in wrapper_text


def test_write_run_wrapper_explicit_env_overrides_detection(
        tmp_path, caps_with_gpu_env):
    """``env=...`` is the user's escape hatch and takes precedence
    over the .fdf-driven routing -- they may want to force a specific
    env for testing.  Pinning that the detector doesn't override an
    explicit user choice."""
    cfg = SiestaConfig(enable_gpu=True)
    fdf_text = render_fdf(_mk_struct(), cfg)
    fdf = tmp_path / "job.fdf"
    fdf.write_text(fdf_text, encoding="utf-8")
    wrapper_path = _runwrap.write_run_wrapper(
        fdf, env="molbuilder-siesta",
    )
    wrapper_text = wrapper_path.read_text(encoding="utf-8")
    assert "molbuilder-siesta-gpu" not in wrapper_text
    assert "molbuilder-siesta" in wrapper_text


# --------------------------------------------------------------------- #
#  GPU runtime defaults block: OMP policy + banner shape                 #
# --------------------------------------------------------------------- #


def test_gpu_runtime_defaults_block_uses_numa_aware_budget(
        tmp_path, caps_with_gpu_env):
    """Policy (2026-06-16, NUMA-aware revision):

      _gpu_budget = _cps           when NUMA-pinnable (dual+ socket,
                                   GPU NUMA known, numactl available)
                  = _phys_cores-1  otherwise
      _omp_default = _gpu_budget / _gpu_mpi_np_default

    Pin both branches of the budget decision + the divisor so a
    regression to the old (_phys_cores - 1)/np formula (which over-
    subscribed the GPU socket on dual-socket boxes when NUMA-pinned)
    surfaces loudly."""
    cfg = SiestaConfig(enable_gpu=True)
    fdf_text = render_fdf(_mk_struct(), cfg)
    fdf = tmp_path / "job.fdf"
    fdf.write_text(fdf_text, encoding="utf-8")
    wrapper_text = _runwrap.write_run_wrapper(fdf).read_text(encoding="utf-8")
    # NUMA-pin decision is a 3-condition AND.
    assert 'if [ "$_gpu_numa" != "unknown" ] && ' in wrapper_text
    assert '[ "$_n_sockets" -ge 2 ] && ' in wrapper_text
    assert 'command -v numactl >/dev/null 2>&1' in wrapper_text
    # Budget switch.
    assert "_gpu_budget=$_cps" in wrapper_text
    assert "_gpu_budget=$(( _phys_cores - 1 ))" in wrapper_text
    # OMP divisor uses the budget, NOT phys_cores - 1 directly.
    assert "_omp_default=$(( _gpu_budget / _gpu_mpi_np_default ))" in wrapper_text
    # Earlier policies MUST NOT appear.
    assert "_omp_default=$(( _cps / 2 ))" not in wrapper_text
    assert "_omp_default=2\n" not in wrapper_text
    assert ("_omp_default=$(( (_phys_cores - 1) / _gpu_mpi_np_default ))"
            not in wrapper_text)


def test_gpu_runtime_defaults_block_wraps_mpirun_in_numactl(
        tmp_path, caps_with_gpu_env):
    """The launch_cmd MUST prefix with $_numa_wrap_gpu so dual-socket
    boxes with numactl can pin all ranks to the GPU socket.  For
    single-socket / no-numactl hosts the variable is empty and the
    launch line behaves as before."""
    cfg = SiestaConfig(enable_gpu=True)
    fdf_text = render_fdf(_mk_struct(), cfg)
    fdf = tmp_path / "job.fdf"
    fdf.write_text(fdf_text, encoding="utf-8")
    wrapper_text = _runwrap.write_run_wrapper(fdf).read_text(encoding="utf-8")
    # The wrap variable is set inside the GPU defaults block.
    assert ('_numa_wrap_gpu="numactl --cpunodebind=$_gpu_numa '
            '--membind=$_gpu_numa"' in wrapper_text)
    # The launch_cmd interpolates it (both MPI branches).
    assert ('_launch_cmd="$_numa_wrap_gpu mpirun -np $_mpi_np '
            '$_mpirun_bind siesta"' in wrapper_text)
    # And the launch logic gives it a defensive default for CPU mode.
    assert '_numa_wrap_gpu="${_numa_wrap_gpu:-}"' in wrapper_text


def test_gpu_runtime_defaults_uses_effective_sockets_under_numa_pin(
        tmp_path, caps_with_gpu_env):
    """When numactl --cpunodebind restricts the cpuset to one NUMA
    node, mpirun sees only ONE package inside its cpuset.  The
    ppr:K:package mapping then needs _effective_sockets=1 -- using
    the full _n_sockets count would underfill (K ranks on socket 0,
    the remaining ranks never spawn because socket 1 is excluded).

    Pin both the variable's definition AND the conditional override
    so a regression to bare _n_sockets surfaces loudly."""
    cfg = SiestaConfig(enable_gpu=True)
    fdf_text = render_fdf(_mk_struct(), cfg)
    fdf = tmp_path / "job.fdf"
    fdf.write_text(fdf_text, encoding="utf-8")
    wrapper_text = _runwrap.write_run_wrapper(fdf).read_text(encoding="utf-8")
    # The variable is set to _n_sockets and conditionally overridden.
    assert "_effective_sockets=$_n_sockets" in wrapper_text
    assert '[ "${_numa_pinned:-0}" = 1 ] && _effective_sockets=1' in wrapper_text
    # Both branches of the placement decision use the EFFECTIVE count.
    assert 'if [ "$_mpi_np" -le "$_effective_sockets" ]; then' in wrapper_text
    assert ('_ppr=$(( (_mpi_np + _effective_sockets - 1) / '
            '_effective_sockets ))' in wrapper_text)
    # And the older bare-_n_sockets comparison MUST NOT appear.
    assert 'if [ "$_mpi_np" -le "$_n_sockets" ]; then' not in wrapper_text


def test_cpu_mode_wrapper_does_not_emit_numa_wrap_gpu_branches(
        tmp_path, caps_with_gpu_env):
    """For CPU-only SIESTA (no enable_gpu), the GPU defaults block
    isn't injected -- the launcher's defensive default
    ``_numa_wrap_gpu="${_numa_wrap_gpu:-}"`` is what keeps the
    interpolation empty.  Pin that the variable initialiser still
    appears (so the launch_cmd doesn't expand to an unbound var)."""
    cfg = SiestaConfig(enable_gpu=False)
    fdf_text = render_fdf(_mk_struct(), cfg)
    fdf = tmp_path / "job.fdf"
    fdf.write_text(fdf_text, encoding="utf-8")
    wrapper_text = _runwrap.write_run_wrapper(fdf).read_text(encoding="utf-8")
    # The GPU defaults block must NOT be injected.
    assert "_gpu_mpi_np_default" not in wrapper_text
    assert "_gpu_budget" not in wrapper_text
    # But the launch_cmd's defensive default MUST still be there.
    assert '_numa_wrap_gpu="${_numa_wrap_gpu:-}"' in wrapper_text
    # And the launch_cmd uses the (empty) interpolation.
    assert '$_numa_wrap_gpu mpirun' in wrapper_text


def test_gpu_runtime_defaults_block_emits_3_line_banner(
        tmp_path, caps_with_gpu_env):
    """C fix (2026-06-16): the GPU runtime banner now has 3 lines on
    stderr: (1) mode summary, (2) chosen arithmetic + GPU NUMA, (3)
    the advisor / override pointer.  Pin all three line headers so a
    regression in the bash heredoc surfaces loudly."""
    cfg = SiestaConfig(enable_gpu=True)
    fdf_text = render_fdf(_mk_struct(), cfg)
    fdf = tmp_path / "job.fdf"
    fdf.write_text(fdf_text, encoding="utf-8")
    wrapper_text = _runwrap.write_run_wrapper(fdf).read_text(encoding="utf-8")
    assert "molbuilder: GPU mode (ELPA-CUDA, no NCCL)" in wrapper_text
    assert "molbuilder: chosen" in wrapper_text
    assert "GPU0 NUMA=" in wrapper_text
    assert "molbuilder envs advise siesta-gpu" in wrapper_text


def test_gpu_runtime_defaults_block_probes_gpu_numa(
        tmp_path, caps_with_gpu_env):
    """C fix: the chosen-line carries GPU NUMA proximity so the user
    knows when to bind ranks to a specific socket.  Pin the awk
    extractor that reads the "NUMA Affinity" column out of
    ``nvidia-smi topo -m``."""
    cfg = SiestaConfig(enable_gpu=True)
    fdf_text = render_fdf(_mk_struct(), cfg)
    fdf = tmp_path / "job.fdf"
    fdf.write_text(fdf_text, encoding="utf-8")
    wrapper_text = _runwrap.write_run_wrapper(fdf).read_text(encoding="utf-8")
    assert "_gpu_numa=" in wrapper_text
    assert "nvidia-smi topo -m" in wrapper_text
    assert "NUMA Affinity" in wrapper_text
