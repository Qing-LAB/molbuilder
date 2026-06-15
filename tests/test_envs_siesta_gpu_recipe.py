"""Shape contract for the ``molbuilder-siesta-gpu`` recipe.

The recipe lives in :mod:`molbuilder.envs.recipes` and carries a
:class:`BuildSpec` because SIESTA-GPU is built from source rather than
installed via conda.  This file pins the exact recipe shape so a
casual edit to recipes.py can't silently desync from:

    * docs/design.md 2026-06-14 Decisions log entry (locked decisions)
    * docs/engines/siesta-gpu.md (engineering reference)
    * docs/README_install.md (user-facing instructions)
"""
from __future__ import annotations

import pytest

from molbuilder.envs.recipes import (
    BUILTIN_RECIPES,
    BuildComponent,
    BuildSpec,
    Recipe,
    recipe_by_name,
)


@pytest.fixture(scope="module")
def recipe() -> Recipe:
    r = recipe_by_name("molbuilder-siesta-gpu")
    assert r is not None, "molbuilder-siesta-gpu must be registered"
    return r


# --------------------------------------------------------------------- #
#  Top-level Recipe shape                                                #
# --------------------------------------------------------------------- #


def test_recipe_is_registered(recipe):
    """In the registry tuple AND lookup-able by name."""
    assert recipe in BUILTIN_RECIPES


def test_recipe_category(recipe):
    """The recipe sets the ``siesta-gpu`` category so doctor groups it
    correctly + so DEFAULT_ENV_NAMES routing works."""
    assert recipe.category == "siesta-gpu"


def test_recipe_carries_build_spec(recipe):
    """Source-build recipes are identified by a non-None build_spec."""
    assert recipe.build_spec is not None
    assert isinstance(recipe.build_spec, BuildSpec)


def test_recipe_uses_conda_forge_only(recipe):
    """No third-party channels for the build env -- everything in
    conda_packages must be in conda-forge."""
    assert recipe.channels == ("conda-forge",)


def test_recipe_description_mentions_gpu_and_source(recipe):
    """Description text shows up in `envs list`; it must signal both
    'GPU' and 'source build' so users don't confuse this with the
    precompiled CPU env."""
    desc = recipe.description.lower()
    assert "cuda" in desc or "gpu" in desc, (
        f"description should mention CUDA/GPU: {recipe.description!r}"
    )
    assert "source" in desc or "built" in desc or "build" in desc, (
        f"description should signal this is built from source: "
        f"{recipe.description!r}"
    )


def test_no_siesta_conda_package(recipe):
    """The recipe MUST NOT install siesta from conda -- it's built
    from source.  Adding both would silently shadow the source build."""
    for pkg in recipe.conda_packages:
        assert not pkg.startswith("siesta"), (
            f"recipe conda_packages contains `{pkg}`; SIESTA is built "
            f"from source by the build_spec, not installed via conda."
        )


# --------------------------------------------------------------------- #
#  Build toolchain in conda_packages                                     #
# --------------------------------------------------------------------- #


def test_pins_gcc_14(recipe):
    """gcc 14 is the default toolchain (locked decision #6).  The
    CUDA-gcc compat preflight refuses to start if CUDA is too old for
    gcc 14; that's the user's signal to override.  See
    docs/engines/siesta-gpu.md § 6."""
    pkgs = " ".join(recipe.conda_packages)
    assert "gcc_linux-64=14" in pkgs
    assert "gxx_linux-64=14" in pkgs
    assert "gfortran_linux-64=14" in pkgs


def test_pins_python_3_12(recipe):
    """Matches the host env + every other recipe."""
    assert "python=3.12" in recipe.conda_packages


def test_pins_cmake_geq_3_30(recipe):
    """CMake 3.30+ is required by SIESTA 5.4.2's CMakeLists."""
    assert any("cmake" in p for p in recipe.conda_packages)
    cmake_specs = [p for p in recipe.conda_packages if p.startswith("cmake")]
    assert any(">=3.30" in p for p in cmake_specs), (
        f"cmake spec should pin >=3.30: {cmake_specs!r}"
    )


def test_uses_openblas_not_mkl(recipe):
    """Locked decision: stays single-OpenMP-runtime (libgomp).  MKL
    brings libiomp5 which collides at runtime.  See
    docs/engines/siesta-gpu.md § 8."""
    pkgs_lower = [p.lower() for p in recipe.conda_packages]
    assert any("openblas" in p for p in pkgs_lower)
    assert not any("mkl" in p for p in pkgs_lower), (
        f"recipe contains MKL spec(s): {[p for p in recipe.conda_packages if 'mkl' in p.lower()]}"
    )


def test_mpi_packages_pinned_to_openmpi_variant(recipe):
    """fftw / hdf5 / netcdf-fortran must use the openmpi variant to
    match the env's OpenMPI; mismatched variants segfault at runtime."""
    for required in ("fftw", "hdf5", "netcdf-fortran"):
        matches = [p for p in recipe.conda_packages if p.startswith(required)]
        assert matches, f"no {required} pin in recipe.conda_packages"
        for spec in matches:
            assert "mpi_openmpi_" in spec, (
                f"{spec!r} should use the mpi_openmpi_* build variant "
                f"to bind to this env's OpenMPI"
            )


def test_libxc_present(recipe):
    """SIESTA links libxc for the wider functional library."""
    assert "libxc" in recipe.conda_packages


def test_no_forbidden_packages_in_conda_packages(recipe):
    """Recipe conda_packages must not declare anything that the
    build_spec.forbidden_packages list rejects."""
    assert recipe.build_spec is not None
    for forbidden in recipe.build_spec.forbidden_packages:
        forbidden_simple = forbidden.split("=")[0]
        for pkg in recipe.conda_packages:
            pkg_simple = pkg.split("=")[0]
            assert pkg_simple != forbidden_simple, (
                f"conda_packages contains forbidden pkg `{pkg}` "
                f"(forbidden pattern `{forbidden}`)"
            )


# --------------------------------------------------------------------- #
#  BuildSpec shape                                                       #
# --------------------------------------------------------------------- #


def test_build_spec_artifact_subdir(recipe):
    """Lives under $CONDA_PREFIX/opt/siesta-gpu-stack (locked decision #2)."""
    assert recipe.build_spec.artifact_subdir == "siesta-gpu-stack"


def test_build_spec_cuda_required(recipe):
    """GPU env -- preflight refuses without CUDA."""
    assert recipe.build_spec.cuda_required is True


def test_build_spec_cuda_min_version(recipe):
    """Default toolchain (gcc 14) pairs with CUDA 12.4+."""
    assert recipe.build_spec.cuda_min_version == "12.4"


def test_build_spec_omp_runtime_gomp(recipe):
    """Single OpenMP runtime, locked to libgomp (gcc's)."""
    assert recipe.build_spec.omp_runtime == "gomp"


def test_build_spec_forbids_mkl_variants(recipe):
    """MKL + intel-openmp are forbidden (libiomp5 collision)."""
    forbidden = set(recipe.build_spec.forbidden_packages)
    for required in ("mkl", "intel-openmp"):
        assert required in forbidden, (
            f"build_spec.forbidden_packages should include `{required}`; "
            f"got {sorted(forbidden)!r}"
        )


def test_build_spec_components_in_order(recipe):
    """Components MUST be listed in dependency order: elpa -> elsi -> siesta.
    The executor visits them top-to-bottom and each one's install must
    finish before the next configures (ELSI links ELPA; SIESTA links
    ELSI)."""
    names = [c.name for c in recipe.build_spec.components]
    assert names == ["elpa", "elsi", "siesta"], (
        f"components in wrong order: {names}"
    )


def test_build_spec_activate_hook_publishes_paths(recipe):
    """Activate.d hook puts siesta on PATH, ELPA/ELSI on LD_LIBRARY_PATH,
    and $CONDA_PREFIX/lib on LD_LIBRARY_PATH (where conda-installed
    libcudart / libmpi / libgomp live).  Per the 2026-06-15 design
    correction: CUDA toolkit lives IN the env, so we point at
    $CONDA_PREFIX/lib not /usr/local/cuda/lib64."""
    hook = recipe.build_spec.activate_hook
    assert '"$CONDA_PREFIX/opt/siesta-gpu-stack/siesta/bin"' in hook
    assert '"$CONDA_PREFIX/opt/siesta-gpu-stack/elpa/lib"' in hook
    assert '"$CONDA_PREFIX/opt/siesta-gpu-stack/elsi/lib"' in hook
    assert '"$CONDA_PREFIX/lib"' in hook
    # No legacy system-CUDA path
    assert "/usr/local/cuda" not in hook
    # We DO NOT export CUDA_HOME -- conda-forge's cuda-nvcc package
    # has its own activate.d that handles that; we'd conflict.
    assert "CUDA_HOME" not in hook


def test_build_spec_deactivate_hook_mirrors_activate(recipe):
    """Deactivate hook must drop exactly the paths activate added."""
    deact = recipe.build_spec.deactivate_hook
    assert '"$CONDA_PREFIX/opt/siesta-gpu-stack/siesta/bin"' in deact
    assert '"$CONDA_PREFIX/opt/siesta-gpu-stack/elpa/lib"' in deact
    assert '"$CONDA_PREFIX/opt/siesta-gpu-stack/elsi/lib"' in deact
    assert '"$CONDA_PREFIX/lib"' in deact
    assert "/usr/local/cuda" not in deact


def test_cuda_toolkit_in_conda_packages(recipe):
    """Per the 2026-06-15 design correction, CUDA toolkit lives IN
    the env via conda-forge packages (mirroring molbuilder-pySCF).
    The recipe must declare cuda-nvcc + cuda-cudart-dev + a pinned
    cuda-version; without them the env has no CUDA toolkit at all."""
    pkgs = " ".join(recipe.conda_packages)
    assert "cuda-version=" in pkgs, (
        "recipe must pin cuda-version (mirrors molbuilder-pySCF "
        "convention); got: " + pkgs
    )
    for required in ("cuda-nvcc", "cuda-cudart-dev"):
        assert required in recipe.conda_packages, (
            f"recipe must include {required!r} in conda_packages "
            f"to ship the CUDA toolkit in-env"
        )


def test_system_preconditions_mention_driver_not_toolkit(recipe):
    """System preconditions list the host-side responsibility (NVIDIA
    driver + nvidia-smi).  The CUDA toolkit is NOT a precondition --
    it's installed by the env's conda solve."""
    text = " ".join(recipe.system_preconditions).lower()
    assert "driver" in text
    assert "nvidia-smi" in text or "nvidia" in text
    # Toolkit should be either absent from preconditions OR explicitly
    # noted as "ships in env" (not as something the user provides).
    assert "ships in" in text or "toolkit" not in text or (
        "in env" in text and "/usr/local/cuda" not in text
    )


# --------------------------------------------------------------------- #
#  Per-component shape                                                   #
# --------------------------------------------------------------------- #


def _comp(recipe: Recipe, name: str) -> BuildComponent:
    for c in recipe.build_spec.components:
        if c.name == name:
            return c
    raise AssertionError(f"no component named {name!r}")


def test_elpa_component(recipe):
    """ELPA: CUDA-enabled, ENABLE_OPENMP=ON, CMAKE_CUDA_ARCHITECTURES set."""
    elpa = _comp(recipe, "elpa")
    assert elpa.needs_cuda is True
    flags = " ".join(elpa.configure_argv)
    assert "-DENABLE_NVIDIA_GPU=ON" in flags
    assert "-DENABLE_OPENMP=ON" in flags
    assert "-DCMAKE_CUDA_ARCHITECTURES={cuda_cc_numeric}" in flags
    assert "{install}" in flags  # CMAKE_INSTALL_PREFIX template
    # Upstream is MPCDF GitLab (not the GitHub mirror).
    assert "gitlab.mpcdf.mpg.de" in elpa.repo_url, (
        f"ELPA upstream should be gitlab.mpcdf.mpg.de: {elpa.repo_url!r}"
    )


def test_elsi_component(recipe):
    """ELSI: USE_EXTERNAL_ELPA=ON pointing at ELPA's install dir."""
    elsi = _comp(recipe, "elsi")
    assert elsi.needs_cuda is False
    flags = " ".join(elsi.configure_argv)
    assert "-DUSE_EXTERNAL_ELPA=ON" in flags
    assert "-DELPA_INCLUDE_DIRS={dep_elpa}/include" in flags
    assert "-DELPA_LIBRARIES={dep_elpa}/lib/libelpa.so" in flags
    # PEXSI + SIPS off (we don't ship those; they would force extra deps).
    assert "-DENABLE_PEXSI=OFF" in flags
    assert "-DENABLE_SIPS=OFF" in flags


def test_siesta_component(recipe):
    """SIESTA: TranSiesta + ELSI + libxc + netcdf, pinned to 5.4.2."""
    siesta = _comp(recipe, "siesta")
    assert siesta.needs_cuda is False
    assert siesta.ref == "5.4.2", (
        f"SIESTA ref must be 5.4.2 to match the precompiled CPU env "
        f"(locked decision #4): {siesta.ref!r}"
    )
    assert "gitlab.com/siesta-project/siesta" in siesta.repo_url
    flags = " ".join(siesta.configure_argv)
    # One binary serves siesta + transiesta + tbtrans (locked decision #7).
    assert "-DSIESTA_WITH_TRANSIESTA=ON" in flags
    assert "-DSIESTA_WITH_ELSI=ON" in flags
    assert "-DELSI_ROOT={dep_elsi}" in flags
    assert "-DSIESTA_WITH_LIBXC=ON" in flags
    assert "-DSIESTA_WITH_NETCDF=ON" in flags


# --------------------------------------------------------------------- #
#  System preconditions                                                  #
# --------------------------------------------------------------------- #


def test_system_preconditions_mention_disk(recipe):
    """User-facing prerequisites must mention disk space."""
    text = " ".join(recipe.system_preconditions).lower()
    assert "disk" in text or "gb" in text


# --------------------------------------------------------------------- #
#  Build-env isolation: cmake pins compilers + MPI + CUDA to the env    #
# --------------------------------------------------------------------- #


def _configure_flags(component_name: str, recipe: Recipe) -> str:
    """Join one component's configure_argv into a searchable string."""
    return " ".join(_comp(recipe, component_name).configure_argv)


def test_every_component_pins_cmake_prefix_path_to_env(recipe):
    """Without CMAKE_PREFIX_PATH={env_prefix}, cmake's Find* modules
    would search system paths first and could pick up system MPI/
    BLAS/HDF5/NetCDF instead of the env's pinned versions."""
    for comp in ("elpa", "elsi", "siesta"):
        flags = _configure_flags(comp, recipe)
        assert "-DCMAKE_PREFIX_PATH={env_prefix}" in flags, (
            f"{comp}: CMAKE_PREFIX_PATH must pin to the env prefix"
        )


def test_every_component_pins_mpi_compilers_to_env(recipe):
    """FindMPI walks PATH for mpicc/mpicxx/mpifort.  If system OpenMPI
    is installed via apt and PATH has /usr/bin before $CONDA_PREFIX/bin
    in some odd shell ordering, the build silently links system libmpi.
    Explicit pins make that impossible."""
    for comp in ("elpa", "elsi", "siesta"):
        flags = _configure_flags(comp, recipe)
        assert "-DMPI_C_COMPILER={env_prefix}/bin/mpicc" in flags
        assert "-DMPI_CXX_COMPILER={env_prefix}/bin/mpicxx" in flags
        assert "-DMPI_Fortran_COMPILER={env_prefix}/bin/mpifort" in flags


def test_elpa_pins_cuda_compiler_and_root_to_env(recipe):
    """FindCUDAToolkit prefers /usr/local/cuda when CUDAToolkit_ROOT
    isn't set.  Without the explicit pin, ELPA could compile against
    a different CUDA than the env's cuda-nvcc."""
    flags = _configure_flags("elpa", recipe)
    assert "-DCMAKE_CUDA_COMPILER={env_prefix}/bin/nvcc" in flags
    assert "-DCUDAToolkit_ROOT={env_prefix}" in flags


def test_no_system_paths_in_cmake_flags(recipe):
    """A regression guard.  None of the cmake flags should reference
    /usr/lib, /usr/local/cuda, /opt/cuda, or similar system locations.
    The full toolchain must live inside the env."""
    for comp in ("elpa", "elsi", "siesta"):
        flags = _configure_flags(comp, recipe)
        for forbidden in ("/usr/lib", "/usr/local/cuda", "/opt/cuda",
                          "/usr/include"):
            assert forbidden not in flags, (
                f"{comp} configure_argv references system path "
                f"{forbidden!r}: {flags}"
            )


def test_install_rpath_uses_origin_relative_path(recipe):
    """Each component bakes an $ORIGIN-relative install rpath so the
    binary can find its sibling libs (and the env's lib) at runtime
    WITHOUT depending on LD_LIBRARY_PATH being set.  $ORIGIN-relative
    so the env stays movable (rename + clone work)."""
    for comp in ("elpa", "elsi", "siesta"):
        flags = _configure_flags(comp, recipe)
        assert "CMAKE_INSTALL_RPATH=$ORIGIN" in flags, (
            f"{comp} must set CMAKE_INSTALL_RPATH with $ORIGIN-relative "
            f"path; got: {flags}"
        )
        assert "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON" in flags, (
            f"{comp} must build with install rpath so the binary in "
            f"build/ links the same way it will after install"
        )


def test_siesta_rpath_finds_elpa_elsi_and_env_lib(recipe):
    """SIESTA binary depends on libelsi, libelpa, libcudart, libmpi,
    libgomp.  Its rpath must reach all three locations from
    $CONDA_PREFIX/opt/siesta-gpu-stack/siesta/bin/siesta:

      - elsi/lib    via $ORIGIN/../../elsi/lib
      - elpa/lib    via $ORIGIN/../../elpa/lib
      - env's lib   via $ORIGIN/../../../../lib
    """
    flags = _configure_flags("siesta", recipe)
    assert "$ORIGIN/../../elsi/lib" in flags
    assert "$ORIGIN/../../elpa/lib" in flags
    assert "$ORIGIN/../../../../lib" in flags
