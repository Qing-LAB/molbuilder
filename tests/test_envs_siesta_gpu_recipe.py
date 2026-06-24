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
    """Components MUST be listed in dependency order: elpa -> siesta.

    Two-component architecture (literature + doc supported, 2026-06-15):
    ELSI + libfdf + libpsml + xmlf90 + libgridxc are SIESTA git
    submodules (per SIESTA 5.4 INSTALL.md § "Required domain-specific
    libraries") that SIESTA's cmake compiles on the fly when the
    --recurse-submodules clone populates External/.  ELPA is the only
    component we build separately because conda-forge ELPA isn't
    built with CUDA support."""
    names = [c.name for c in recipe.build_spec.components]
    assert names == ["elpa", "siesta"], (
        f"components in wrong order: {names}"
    )


def test_build_spec_activate_hook_publishes_paths(recipe):
    """Activate.d hook puts siesta on PATH, ELPA on LD_LIBRARY_PATH,
    and $CONDA_PREFIX/lib on LD_LIBRARY_PATH (where conda-installed
    libcudart / libmpi / libgomp live).  ELSI no longer has its own
    install dir since it's built into the siesta executable via the
    SIESTA submodule path."""
    hook = recipe.build_spec.activate_hook
    assert '"$CONDA_PREFIX/opt/siesta-gpu-stack/siesta/bin"' in hook
    assert '"$CONDA_PREFIX/opt/siesta-gpu-stack/elpa/lib"' in hook
    assert '"$CONDA_PREFIX/lib"' in hook
    # No separate ELSI install dir -- ELSI is a SIESTA submodule, its
    # libs end up inside the SIESTA install tree (if at all; usually
    # it's a static link).
    assert "elsi/lib" not in hook
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
    assert '"$CONDA_PREFIX/lib"' in deact
    assert "elsi/lib" not in deact
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
    """ELPA: tarball download + autotools build (verified 2026-06-15
    against upstream configure.ac across 2020.11 / 2021.11 / 2023.05 /
    2025.06: no CMakeLists.txt exists on any tag; all versions ship a
    pre-bootstrapped ``configure`` script in the tarball).

    Configure flags use the modern ``--enable-nvidia-gpu`` naming
    (introduced in ELPA 2021.x).  Older ``--enable-gpu`` syntax is NOT
    used because the default tag is 2021.11.001+.
    """
    elpa = _comp(recipe, "elpa")
    assert elpa.needs_cuda is True
    # Tarball, not git clone.
    assert elpa.tarball_url, "ELPA must use tarball download"
    assert "elpa.mpcdf.mpg.de" in elpa.tarball_url, (
        f"ELPA tarball must come from MPCDF: {elpa.tarball_url!r}"
    )
    assert elpa.tarball_inner_dir, "ELPA needs tarball_inner_dir set"
    # SHA256 pin for integrity.
    assert elpa.tarball_sha256, (
        "ELPA tarball_sha256 must be pinned for integrity verification"
    )
    assert len(elpa.tarball_sha256) == 64
    # autotools configure_argv (sh -c wrapping autotools build).
    flags = " ".join(elpa.configure_argv)
    assert elpa.configure_argv[0] == "sh"
    assert "configure" in flags
    assert "--enable-nvidia-gpu" in flags
    # 2026-06-16 (recipe bump to ELPA 2024.05.001): the cc flag now
    # uses a bash variable (CC_TAG) instead of a literal template, so
    # the SM_80 specialised kernel can force sm_80 (A100-only) while
    # everyone else uses their native cc.  See the CC_TAG block in
    # recipes.py for the design rationale.
    assert "--with-NVIDIA-GPU-compute-capability=$CC_TAG" in flags
    # The native cc still flows through via the CC_NUM/CC_TAG assignment
    # at the top of the configure prelude.
    assert "CC_TAG=sm_{cuda_cc_numeric}" in flags
    assert "--enable-openmp" in flags
    assert "--prefix={install}" in flags
    assert "--with-cuda-path={env_prefix}" in flags
    # No cmake flags from the old recipe.
    assert "cmake" not in flags.lower() or "cmake" not in flags
    assert "-DENABLE_NVIDIA_GPU" not in flags
    assert "-DCMAKE_CUDA_ARCHITECTURES" not in flags


def test_no_elsi_component(recipe):
    """ELSI is NOT a separate BuildComponent in the 2-component
    architecture -- it's a SIESTA submodule under
    External/ELSI-project/elsi_interface and SIESTA's cmake builds
    it on the fly.  Per SIESTA 5.4 INSTALL.md."""
    names = [c.name for c in recipe.build_spec.components]
    assert "elsi" not in names


def test_siesta_clones_with_recurse_submodules(recipe):
    """SIESTA must be cloned with --recurse-submodules so the four
    required ESL libraries (libfdf, libpsml, xmlf90, libgridxc) +
    ELSI come along as External/ submodules.  None of those four
    are on conda-forge; SIESTA cmake's source-step in
    FIND_METHOD=source picks them up from External/<package>."""
    siesta = _comp(recipe, "siesta")
    assert siesta.clone_recurse_submodules is True


def test_siesta_component(recipe):
    """SIESTA: TranSiesta + ELSI + libxc + netcdf, pinned to the
    ``5.4.2`` release tag.

    Upstream observation (2026-06-15 ``git ls-remote --tags``):
    gitlab.com/siesta-project/siesta DOES ship numeric 5.x release
    tags: ``5.0.0``, ``5.0.1``, ``5.0.2``, ``5.2.0``, ``5.2.1``,
    ``5.2.2``, ``5.4.0``, ``5.4.1``, ``5.4.2`` (bare numeric, no
    ``v`` / ``siesta-`` prefix).  Earlier this recipe pointed at the
    ``rel-5.4`` branch -- a moving target that drifted on every push
    and (because the old global-fingerprint scheme baked the
    resolved SHA into every sentinel) invalidated the ELPA sentinel
    just because SIESTA's upstream HEAD moved.  Pinning to ``5.4.2``
    also matches what the precompiled ``molbuilder-siesta`` env
    ships, so the ``.fdf`` input format and TranSiesta output format
    stay identical across CPU vs GPU runs.  Override via
    ``MOLBUILDER_SIESTA_TAG`` for a hotfix from the ``rel-5.4`` branch
    or a specific SHA."""
    siesta = _comp(recipe, "siesta")
    assert siesta.needs_cuda is False
    assert siesta.ref == "5.4.2", (
        f"SIESTA ref must be pinned to the 5.4.2 release tag "
        f"(see test docstring for why): {siesta.ref!r}"
    )
    assert "gitlab.com/siesta-project/siesta" in siesta.repo_url
    flags = " ".join(siesta.configure_argv)
    # One binary serves siesta + transiesta + tbtrans (locked decision #7).
    assert "-DSIESTA_WITH_TRANSIESTA=ON" in flags
    assert "-DSIESTA_WITH_ELSI=ON" in flags
    # ELSI built from SIESTA submodule -- no ELSI_ROOT; cmake's FIND_METHOD
    # source step picks it up from External/.
    assert "-DELSI_ROOT" not in flags
    assert "-DSIESTA_WITH_LIBXC=ON" in flags
    assert "-DSIESTA_WITH_NETCDF=ON" in flags
    assert "-DSIESTA_WITH_ELPA=ON" in flags


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


def test_siesta_pins_cmake_prefix_path_to_env_and_elpa(recipe):
    """SIESTA (cmake): CMAKE_PREFIX_PATH must include BOTH the env
    prefix (for conda-installed deps) AND the ELPA install dir.
    Semicolon = cmake's list separator."""
    siesta_flags = _configure_flags("siesta", recipe)
    assert "-DCMAKE_PREFIX_PATH={env_prefix};{dep_elpa}" in siesta_flags


def test_elpa_pins_env_compilers_and_cuda_path(recipe):
    """ELPA (autotools): FC/CC/CXX use the env's mpi compilers, and
    --with-cuda-path points at the env (where conda-installed cuda-nvcc
    lives).  This bypasses autoconf's PATH walk so the system openmpi
    can't be picked up by accident."""
    flags = _configure_flags("elpa", recipe)
    # MPI compilers via FC=mpifort etc. (resolves via PATH inside conda
    # activate; the autotools build doesn't take cmake's explicit
    # -DMPI_C_COMPILER pin).
    assert "FC=mpifort" in flags
    assert "CC=mpicc" in flags
    assert "CXX=mpicxx" in flags
    # CUDA path points at the env (conda-installed cuda-nvcc + libs).
    assert "--with-cuda-path={env_prefix}" in flags


def test_siesta_pins_mpi_compilers_to_env(recipe):
    """SIESTA (cmake): FindMPI walks PATH for mpicc/mpicxx/mpifort.
    If system OpenMPI is installed via apt and PATH has /usr/bin
    before $CONDA_PREFIX/bin in some odd shell ordering, the build
    silently links system libmpi.  Explicit pins make that impossible."""
    flags = _configure_flags("siesta", recipe)
    assert "-DMPI_C_COMPILER={env_prefix}/bin/mpicc" in flags
    assert "-DMPI_CXX_COMPILER={env_prefix}/bin/mpicxx" in flags
    assert "-DMPI_Fortran_COMPILER={env_prefix}/bin/mpifort" in flags


def test_no_system_paths_in_cmake_flags(recipe):
    """A regression guard.  None of the cmake flags should reference
    /usr/lib, /usr/local/cuda, /opt/cuda, or similar system locations.
    The full toolchain must live inside the env."""
    for comp in ("elpa", "siesta"):
        flags = _configure_flags(comp, recipe)
        for forbidden in ("/usr/lib", "/usr/local/cuda", "/opt/cuda",
                          "/usr/include"):
            assert forbidden not in flags, (
                f"{comp} configure_argv references system path "
                f"{forbidden!r}: {flags}"
            )


def test_siesta_install_rpath_uses_origin_relative_path(recipe):
    """SIESTA (cmake): bakes $ORIGIN-relative install rpath so the
    binary can find its sibling libs (and the env's lib) at runtime
    WITHOUT depending on LD_LIBRARY_PATH being set.  $ORIGIN-relative
    so the env stays movable (rename + clone work).

    ELPA (autotools) does NOT bake explicit rpath; instead the
    activate.d hook publishes $CONDA_PREFIX/opt/.../elpa/lib on
    LD_LIBRARY_PATH so the SIESTA binary finds libelpa.so via the
    loader's standard search path.  This is the common autotools
    pattern (cleaner than the autotools-LDFLAGS rpath hack)."""
    flags = _configure_flags("siesta", recipe)
    assert "CMAKE_INSTALL_RPATH=$ORIGIN" in flags
    assert "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON" in flags


def test_siesta_rpath_finds_elpa_and_env_lib(recipe):
    """SIESTA binary depends on libelpa, libcudart, libmpi, libgomp.
    Its rpath must reach both locations from
    $CONDA_PREFIX/opt/siesta-gpu-stack/siesta/bin/siesta:

      - elpa/lib    via $ORIGIN/../../elpa/lib
      - env's lib   via $ORIGIN/../../../../lib

    libelsi is statically linked into the siesta binary via the
    submodule build, so no separate rpath entry is needed for it.
    """
    flags = _configure_flags("siesta", recipe)
    assert "$ORIGIN/../../elpa/lib" in flags
    assert "$ORIGIN/../../../../lib" in flags
    # No ELSI lib dir -- ELSI is built into the siesta binary itself.
    assert "elsi/lib" not in flags


# --------------------------------------------------------------------- #
#  Argv-template gate -- catches the class of bug shipped in f46a9a3:    #
#                                                                        #
#  Recipe build_argv / configure_argv / install_argv strings get run     #
#  through ``str.format_map`` to substitute ``{build}`` / ``{install}``  #
#  / ``{jobs}`` etc.  When the bash payload uses literal ``{`` / ``}``   #
#  (e.g. ``{ echo X; cmd; }`` for command grouping) those collide with   #
#  the substitution and either render incorrectly OR raise               #
#  ``KeyError: 'echo X; cmd; '`` / ``ValueError: missing { in field      #
#  name`` AT RUNTIME -- only when the actual install runs, after the     #
#  user has waited for ELPA to build.                                    #
#                                                                        #
#  These tests render every BuildComponent's argv with stub              #
#  substitutions and assert:                                             #
#    (1) the template applies cleanly (no KeyError / ValueError)         #
#    (2) for ``sh -c`` / ``bash -c`` arms, ``bash -n`` accepts the       #
#        resulting script (catches shell syntax errors at test time)    #
# --------------------------------------------------------------------- #
import subprocess as _subprocess
from molbuilder.envs.builds import _apply_template as _apply_template_fn

_STUB_TEMPLATE_SUBS = {
    "src": "/stub/src",
    "build": "/stub/build",
    "install": "/stub/install",
    "env_prefix": "/stub/env",
    "jobs": "8",
    "dep_elpa": "/stub/elpa",
    "cuda_cc_numeric": "80",
    "cuda_cc_sm": "sm_80",
}


@pytest.mark.parametrize("phase", ["clone", "configure", "build", "install"])
def test_every_component_argv_renders_through_template(recipe, phase):
    """Render each phase's argv through _apply_template with stub
    subs.  If a literal ``{`` from bash command-group syntax leaked
    in, format_map fails here -- catches the class of bug that
    f46a9a3 fixed in production AFTER the user hit it during a real
    install.

    Skips when the phase has no argv (e.g. clone phase for git-based
    components -- handled by builds.py with a default git clone).
    """
    spec = recipe.build_spec
    assert spec is not None
    for comp in spec.components:
        argv_attr = f"{phase}_argv"
        argv = getattr(comp, argv_attr, None)
        if not argv:
            continue
        rendered = _apply_template_fn(argv, _STUB_TEMPLATE_SUBS)
        assert isinstance(rendered, tuple)
        assert len(rendered) == len(argv)
        # If we got here without KeyError / ValueError, the template
        # applied cleanly.  Now check the resulting shell payload
        # parses if argv[0] is ``sh`` or ``bash``.
        if argv[0] in ("sh", "bash") and len(rendered) >= 3 and argv[1] == "-c":
            payload = rendered[2]
            r = _subprocess.run(
                ["bash", "-n", "/dev/stdin"], input=payload,
                capture_output=True, text=True,
            )
            assert r.returncode == 0, (
                f"{comp.name}.{phase}_argv: bash -n rejected "
                f"the rendered payload:\n{r.stderr}\n"
                f"payload was:\n{payload}"
            )
