"""The ``molbuilder-siesta-gpu`` recipe: the failures its builds have measured,
and the lines whose loss the build does not show (`ops/installation.md` § 6).

SIESTA is built from source here, so a recipe line that goes wrong shows after
a long build, or only at a run.  The build itself checks the rest: what a test
restating a build option would add is `installation.md`'s stated reason for
leaving it out.
"""
from __future__ import annotations

import pytest

from molbuilder.envs.recipes import (
    BuildComponent,
    Recipe,
    recipe_by_name,
)


@pytest.fixture(scope="module")
def recipe() -> Recipe:
    r = recipe_by_name("molbuilder-siesta-gpu")
    assert r is not None, "molbuilder-siesta-gpu must be registered"
    return r


# --------------------------------------------------------------------- #
#  Build toolchain in conda_packages                                     #
# --------------------------------------------------------------------- #


def test_the_toolchain_is_pinned_to_a_MINOR_version(recipe):
    """**The minor is the whole point** (`ops/installation.md` § 6).

    A major-only ``gcc_linux-64=14`` is satisfied by `14.3` and `14.4`
    equally -- and **14.4's gfortran miscompiles SIESTA's `kpoint_t.F90`**:
    the build dies at its link step, near the end (`undefined reference to
    process_k_cell_`, § 6.1).

    Checked as a PROPERTY -- three packages, one version, and that version
    carrying a minor -- rather than by retyping `14.3` here: the number is
    keyed to a SIESTA tag and moves when the tag does (§ 6: *"when the
    SIESTA tag moves, re-test 14.4"*), and a test that hard-coded it would
    have to be edited by whoever re-tests, which is how the copy drifts.
    """
    from molbuilder.envs.recipes import _GCC_VERSION
    assert "." in _GCC_VERSION, (
        f"the toolchain pin is {_GCC_VERSION!r} -- a MAJOR-only pin lets "
        f"conda resolve 14.4, whose gfortran miscompiles SIESTA's "
        f"kpoint_t.F90, and the build dies at its link step "
        f"(ops/installation.md § 6.1)")
    pkgs = " ".join(recipe.conda_specs)
    for tool in ("gcc", "gxx", "gfortran"):
        want = f"{tool}_linux-64={_GCC_VERSION}"
        assert want in pkgs, (
            f"the GPU recipe does not pin {tool} to the toolchain version "
            f"the rest of the family uses ({_GCC_VERSION}): {pkgs}")


def test_uses_openblas_not_mkl(recipe):
    """THE single-OpenMP-runtime rule, and the only test of it.

    Naming `openblas` is what keeps MKL out of the solve -- scalapack and
    fftw then resolve against it, so Intel's libiomp5 never lands beside
    gcc's libgomp.  A declaration the solver enforces.

    See docs/ops/installation.md section 6."""
    pkgs_lower = [p.lower() for p in recipe.conda_specs]
    assert any("openblas" in p for p in pkgs_lower)
    assert not any("mkl" in p for p in pkgs_lower), (
        f"recipe contains MKL spec(s): {[p for p in recipe.conda_specs if 'mkl' in p.lower()]}"
    )


def test_mpi_packages_pinned_to_openmpi_variant(recipe):
    """fftw / hdf5 / libnetcdf must use the openmpi variant to match the
    env's OpenMPI; mismatched variants segfault at runtime.

    `netcdf-fortran` is built from source, so there is no conda spec to
    check.  libnetcdf, the C
    layer it links, carries the same requirement."""
    for required in ("fftw", "hdf5", "libnetcdf"):
        matches = [p for p in recipe.conda_specs if p.startswith(required)]
        assert matches, f"no {required} pin in recipe.conda_specs"
        for spec in matches:
            assert "mpi_openmpi_" in spec, (
                f"{spec!r} should use the mpi_openmpi_* build variant "
                f"to bind to this env's OpenMPI"
            )


def _comp(recipe: Recipe, name: str) -> BuildComponent:
    for c in recipe.build_spec.components:
        if c.name == name:
            return c
    raise AssertionError(f"no component named {name!r}")


def test_elpa_component(recipe):
    """ELPA's GPU path, and the GPU it is compiled for (`installation.md` § 6).

    `--enable-nvidia-gpu`: the build and its `verify` both pass without it,
    and the env has no GPU path.  `CC_TAG` is the machine's own capability,
    not a fixed `sm_80` (2026-06-15: ELPA 2024.05.001's configure refuses the
    SM80 kernel for every GPU but the A100, which broke the build on the
    others)."""
    flags = " ".join(_comp(recipe, "elpa").configure_argv)
    assert "--enable-nvidia-gpu" in flags
    assert "CC_TAG=sm_{cuda_cc_numeric}" in flags
    assert "--with-NVIDIA-GPU-compute-capability=$CC_TAG" in flags


def test_siesta_is_built_with_its_features_against_the_envs_mpi(recipe):
    """`installation.md` § 6: ELPA, ELSI, libxc and NetCDF on, against the
    env's own MPI compilers.  THE SILENT FAILURE: lose a flag and the build
    finishes and `siesta --version` passes, so the build cannot say so; and
    without the pins a machine with a system MPI first on its `PATH` links
    that `libmpi`, and nothing says so until a run."""
    argv = _comp(recipe, "siesta").configure_argv
    for flag in ("-DSIESTA_WITH_ELPA=ON", "-DSIESTA_WITH_ELSI=ON",
                 "-DSIESTA_WITH_LIBXC=ON", "-DSIESTA_WITH_NETCDF=ON",
                 "-DMPI_C_COMPILER={env_prefix}/bin/mpicc",
                 "-DMPI_CXX_COMPILER={env_prefix}/bin/mpicxx",
                 "-DMPI_Fortran_COMPILER={env_prefix}/bin/mpifort"):
        assert flag in argv, flag


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
    # derived, not listed: builds.py makes a dep_* key for EVERY component
    **{f"dep_{c.name}": f"/stub/{c.name}"
       for c in recipe_by_name("molbuilder-siesta-gpu").build_spec.components},
    "cuda_cc_numeric": "80",
    "cuda_cc_sm": "sm_80",
}


# "verify" belongs in this list.  netcdf_fortran's verify is the most
# complex `sh -c` payload in the recipe -- a printf carrying quotes and
# embedded newlines, an `||` subshell, `$FC`, and `{install}` templating.
# Measured 2026-09-19: with "verify" absent, deleting one `)` from that
# payload left the whole suite green, and the break would have surfaced at
# step 10/15 -- after ELPA had already compiled for ten minutes.
@pytest.mark.parametrize("phase",
                         ["clone", "configure", "build", "install", "verify"])
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


@pytest.mark.parametrize("preset,tmpdir", [
    ({}, "/tmp"),
    ({"TMPDIR": "/scratch/job1"}, "/scratch/job1"),
    ({"TMPDIR": "/scratch/job1", "OMPI_MCA_orte_tmpdir_base": "/mine"}, "/mine"),
], ids=["no scheduler", "the job's own TMPDIR", "the person's value"])
def test_the_hooks_publish_the_stack_and_take_back_only_what_they_set(
        tmp_path, recipe, preset, tmpdir):
    """Source the real activate and deactivate hooks in a shell and read what
    it holds (`installation.md` § 6.1).  Activated, SIESTA is on `PATH`, the
    source-built libraries come before the env's, and OpenMPI's shared memory
    goes to the job's own `TMPDIR` or `/tmp`, never onto the NFS home the env
    lives on (measured 2026-06-24: a warning at every `mpirun`, and every
    write a network trip) -- unless the person set it.  Deactivated, the shell
    is as it was.  A string check cannot see any of this: a hook whose guard
    is wrong still holds every word."""
    prefix = "/stub/env"
    stack = f"{prefix}/opt/siesta-gpu-stack"
    spec = recipe.build_spec
    activate = tmp_path / "activate.sh"
    activate.write_text(spec.activate_hook)
    deactivate = tmp_path / "deactivate.sh"
    deactivate.write_text(spec.deactivate_hook)
    show = ('printf "%s\\n" "$PATH" "${LD_LIBRARY_PATH:-}" '
            '"${OMPI_MCA_orte_tmpdir_base:-unset}"')
    r = _subprocess.run(
        ["bash", "-c", f'{show}; . "{activate}"; {show}; . "{deactivate}"; {show}'],
        env={"PATH": "/usr/bin:/bin", "CONDA_PREFIX": prefix, **preset},
        capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    before, inside, after = lines[:3], lines[3:6], lines[6:]
    path, libs, shared_memory = inside
    assert path.split(":")[0] == f"{stack}/siesta/bin"
    assert libs.split(":") == [f"{stack}/netcdf_fortran/lib", f"{stack}/elpa/lib",
                               f"{prefix}/lib"]
    assert shared_memory == tmpdir
    assert after == before


# --------------------------------------------------------------------- #
#  Header + library search paths for the BUNDLED Makefiles              #
#                                                                        #
#  Reported 2026-08-13 from a CLEAN machine (no libreadline-dev):        #
#                                                                        #
#    lua.c:82:10: fatal error: readline/readline.h:                      #
#                 No such file or directory                              #
#    make[4]: *** [<builtin>: lua.o] Error 1                             #
#    ninja: build stopped: subcommand failed.                            #
#    [ 8/10] siesta.build: FAILED                                        #
#                                                                        #
#  lua-5.3.5's src/Makefile ASSIGNS CFLAGS rather than appending it, so  #
#  conda's ``-isystem <env>/include`` never reaches it.  Its ``linux``   #
#  target then compiles with -DLUA_USE_LINUX (switching on              #
#  LUA_USE_READLINE, so lua.c includes <readline/readline.h>) and links  #
#  -lreadline, both with an empty search path.  While bare ``gcc`` was   #
#  still the HOST gcc that resolved against /usr/include by accident.    #
#  Once the bare-name shims point ``gcc`` at the conda toolchain --      #
#  whose only header dirs are its own plus the sysroot, NOT              #
#  <env>/include -- a machine without libreadline-dev fails outright.    #
#                                                                        #
#  The fix has two halves and BOTH are load-bearing:                     #
#    (a) ``readline`` DECLARED in conda_packages -- the package;         #
#    (b) C_INCLUDE_PATH / LIBRARY_PATH exported by the build step --     #
#        the search path.  gcc reads those itself, so unlike CPPFLAGS    #
#        no Makefile assignment can override them.                       #
#                                                                        #
#  Verified against the real sources before landing: the bundled         #
#  lua-5.3.5 fails with the exact error above under conda-toolchain      #
#  shims, and builds liblua.a + lua + luac against the env's readline    #
#  with the rendered prefix applied.                                     #
# --------------------------------------------------------------------- #
import os as _os


def _flook_building_component(recipe):
    """The component whose build step compiles the bundled Lua engine."""
    for comp in recipe.build_spec.components:
        if "Lua-Engine" in " ".join(comp.build_argv):
            return comp
    raise AssertionError(
        "no component builds External/Lua-Engine -- if flook was turned "
        "off, these readline guards are obsolete and should be retired, "
        "not weakened"
    )


def test_sysroot_is_pinned_not_merely_declared(recipe):
    """``sysroot_linux-64`` must carry a VERSION.

    This is the regression test for the 2026-09 ELPA failure.  The
    recipe declared the package unversioned, which reads as a decision
    and is not one: ``gcc_impl_linux-64`` depends on a bare
    ``sysroot_linux-64``, so the solver took the newest (2.39) and the
    toolchain produced binaries the host's glibc could not load.

    Note what this test can and cannot do.  It is a string assertion --
    it never solves.  It is only sufficient BECAUSE the spec is pinned:
    once the version is stated, the spec fully determines what gets
    installed.  While the spec was bare, no static assertion could have
    caught this, which is why `test_envs_solve_drift.py` solves it live.
    """
    specs = [p for p in recipe.conda_specs
             if p.split("=")[0] == "sysroot_linux-64"]
    assert specs, "recipe no longer declares sysroot_linux-64 at all"
    assert len(specs) == 1, f"sysroot declared more than once: {specs}"
    assert "=" in specs[0], (
        f"sysroot spec `{specs[0]}` carries no version.  An unversioned "
        f"spec lets the solver pick the newest sysroot, which produces a "
        f"toolchain whose binaries will not start on older hosts."
    )


def test_the_sysroot_pin_is_at_or_below_the_conda_forge_floor(recipe):
    """2.17 is conda-forge's portability floor and what every prebuilt
    package in the env already targets.  A higher default would trade
    portability for nothing measurable -- SIESTA and ELPA use no glibc
    feature past 2.17.  Sites that know their cluster is uniformly
    newer raise it via MOLBUILDER_SYSROOT, not by editing this default.
    """
    from molbuilder.envs.abi import parse_version
    spec = next(p for p in recipe.conda_specs
                if p.split("=")[0] == "sysroot_linux-64")
    assert parse_version(spec.split("=", 1)[1]) <= (2, 17)


def test_readline_is_declared_not_inherited(recipe):
    """``readline`` must be named outright.

    SIESTA's External/Lua-Engine/CMakeLists.txt does
    ``find_library(readline REQUIRED)`` -- one of only two hard-REQUIRED
    find_library calls in the whole source tree (the other is ``dl``,
    which glibc provides).  It was reaching the env only because
    ``python`` happens to pull it in transitively.
    """
    assert "readline" in recipe.conda_specs


@pytest.mark.parametrize(
    "preset, expect_c, expect_l",
    [
        (None, "{p}/include", "{p}/lib"),
        ("/preexisting", "{p}/include:/preexisting", "{p}/lib:/preexisting"),
    ],
    ids=["unset", "preserves-preexisting"],
)
def test_build_step_exports_search_paths_to_the_compiler(
        tmp_path, recipe, preset, expect_c, expect_l):
    """Run the REAL rendered build payload with ``cmake`` stubbed out,
    then read back what the compiler would actually have seen.

    Asserting that the string ``C_INCLUDE_PATH`` appears in the payload
    would still pass
    if the export sat AFTER ``cmake --build``, or were spelled so that
    no shell ever applied it.  Executing it and reading the values out
    of the stub is the only version that fails for the right reason.

    The prepend arm matters because a bare assignment would silently
    discard a search path the caller had already set, and a ``:``-
    joined empty element means "the current directory" to gcc.
    """
    env_prefix = tmp_path / "env"
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    seen = tmp_path / "seen.txt"
    stub = stub_bin / "cmake"
    stub.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n%s\\n" "$C_INCLUDE_PATH" "$LIBRARY_PATH" > "{seen}"\n'
        "exit 0\n"
    )
    stub.chmod(0o755)

    comp = _flook_building_component(recipe)
    subs = dict(_STUB_TEMPLATE_SUBS)
    subs["env_prefix"] = str(env_prefix)
    # A src that does not exist: the lua MYCFLAGS=-fPIC patch's
    # ``[ -f ... ]`` guard must skip cleanly rather than abort.
    subs["src"] = str(tmp_path / "absent-src")
    rendered = _apply_template_fn(comp.build_argv, subs)
    assert rendered[0] in ("sh", "bash") and rendered[1] == "-c"
    payload = rendered[2]

    env = dict(_os.environ)
    env["PATH"] = f"{stub_bin}:{env['PATH']}"
    env.pop("C_INCLUDE_PATH", None)
    env.pop("LIBRARY_PATH", None)
    if preset is not None:
        env["C_INCLUDE_PATH"] = preset
        env["LIBRARY_PATH"] = preset

    r = _subprocess.run(["sh", "-c", payload], env=env,
                        capture_output=True, text=True)
    assert r.returncode == 0, f"payload failed:\n{r.stderr}"
    assert seen.exists(), (
        "the stubbed cmake never ran -- the payload aborted before the "
        f"build:\n{r.stderr}"
    )
    got_c, got_l = seen.read_text().splitlines()
    assert got_c == expect_c.format(p=env_prefix), (
        f"C_INCLUDE_PATH seen by the build was {got_c!r}"
    )
    assert got_l == expect_l.format(p=env_prefix), (
        f"LIBRARY_PATH seen by the build was {got_l!r}"
    )


# --------------------------------------------------------------------- #
#  The gcc pin belongs to a SIESTA TAG, not to molbuilder               #
#                                                                       #
#  User, 2026-08-14: "note that with new siesta version this could       #
#  change so this is version specific".  A comment saying so is what     #
#  the code carried; these are the gates that make it true, because a    #
#  version stamp with no gate is worse than none.                       #
# --------------------------------------------------------------------- #

def test_the_pinned_siesta_tag_has_a_measured_gcc_row():
    """Bumping ``_SIESTA_REF`` without revisiting the pin fails HERE.

    The pin exists because SIESTA 5.4.2's ``Src/kpoint_t.F90`` miscompiles
    under gcc 14.4.  A later SIESTA may restructure that optional dummy
    procedure or fix it outright, at which point the pin is holding the
    toolchain back for a defect that no longer exists.  Nothing would say
    so on its own -- hence this test.

    If it fails: build the new tag, then check ``libsiesta.a`` for an
    undefined ``process_k_cell_``.  If it links, the tag needs no pin and
    the row records that; if it does not, the row records the gcc that
    works.  Either way the answer is MEASURED before the row is written.
    """
    from molbuilder.envs import recipes as R
    assert R._SIESTA_REF in R._GCC_PIN_BY_SIESTA_REF, (
        f"SIESTA is pinned to {R._SIESTA_REF!r} but no gcc row was measured "
        f"for it (rows: {sorted(R._GCC_PIN_BY_SIESTA_REF)}).  The gcc pin is "
        f"specific to the SIESTA source we build -- re-test before assuming "
        f"the old one carries."
    )


def test_the_default_gcc_comes_from_that_row_and_is_a_minor_pin():
    """``14`` is not a pin -- conda reads it as ``14.*``, so two machines
    installing weeks apart resolve to different compilers and only one of
    them builds.  That is the whole reason the pin is a minor version."""
    import os as _os
    from molbuilder.envs import recipes as R
    if _os.environ.get("MOLBUILDER_GCC"):
        pytest.skip("MOLBUILDER_GCC overrides the table by design")
    row = R._GCC_PIN_BY_SIESTA_REF.get(R._SIESTA_REF)
    assert row is not None, (
        f"no measured gcc row for SIESTA {R._SIESTA_REF!r} -- see the "
        f"test above, which names the re-test")
    assert R._GCC_VERSION == row
    assert R._GCC_VERSION.count(".") >= 1, (
        f"{R._GCC_VERSION!r} is a major-only spec, which conda reads as a "
        f"wildcard -- see docs/ops/installation.md 6.1")


def test_an_unmeasured_tag_falls_back_conservatively_not_to_a_wildcard():
    """A branch ref or a future release has no row.  The fallback is the
    newest MEASURED pin, still minor-pinned: an unmeasured tag is a reason
    to be conservative, not a reason to let the solver pick."""
    from molbuilder.envs import recipes as R
    assert R._GCC_WHEN_UNMEASURED in R._GCC_PIN_BY_SIESTA_REF.values()
