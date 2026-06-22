import os
import platform
import subprocess
import sys
import sysconfig
import tempfile

from setuptools import setup, Extension
from Cython.Build import cythonize

# default flags are portable (safe to ship in wheels); native tuning is opt-in
# via LATTICEPTS_NATIVE=1 since it would crash on other CPUs if shipped
compile_args = ["-O3", "-funroll-loops"]
if os.environ.get("LATTICEPTS_NATIVE") == "1":
    # arm64 uses -mcpu=native; x86 uses -march=native -mtune=native
    if platform.machine().lower() in ("arm64", "aarch64"):
        compile_args += ["-mcpu=native"]
    else:
        compile_args += ["-march=native", "-mtune=native"]

# Parallel count_only is opt-in via LATTICEPTS_OPENMP=1, which adds -fopenmp so
# the parallel path in box_enum_omp.h compiles. Without it that path is #ifdef'd
# out (it forwards to the serial box_enum.h kernel), so the default build stays
# dependency-free and behaves identically to the serial kernel
def _supports_openmp():
    """True iff the build compiler accepts -fopenmp (probe-compile a tiny TU)."""
    # CC = the C compiler (Unix convention); resolve which one to probe like the
    # build does -- explicit $CC, else Python's build-time CC, else generic "cc"
    # (that config value can carry flags like "gcc -pthread", so take token 0)
    cc = (os.environ.get("CC") or sysconfig.get_config_var("CC") or "cc").split()[0]
    # tiny probe TU (translation unit: one .c file the compiler builds as a unit)
    src = "#include <omp.h>\nint main(void){return omp_get_max_threads() > 0;}\n"
    with tempfile.TemporaryDirectory() as d:
        c = os.path.join(d, "probe.c")
        with open(c, "w") as f:
            f.write(src)
        try:
            r = subprocess.run([cc, "-fopenmp", c, "-o", os.path.join(d, "probe")],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return r.returncode == 0
        except OSError:
            return False

link_args = []
if os.environ.get("LATTICEPTS_OPENMP") == "1":
    if not _supports_openmp():
        sys.exit(
            "LATTICEPTS_OPENMP=1 was set, but this compiler does not support "
            "-fopenmp. Use a GCC/Clang toolchain with OpenMP (on macOS, Apple "
            "Clang needs `brew install libomp` plus the -Xpreprocessor flags), "
            "or unset LATTICEPTS_OPENMP to build the default (serial) extension."
        )
    compile_args += ["-fopenmp"]
    link_args += ["-fopenmp"]

setup(
    ext_modules=cythonize(
        [Extension(
            "latticepts.box_enum",
            sources=["latticepts/box_enum.pyx"],
            include_dirs=["latticepts"],
            define_macros=[("BOX_ENUM_IMPLEMENTATION", None)],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            language="c",
        )],
        compiler_directives={"language_level": "3"},
    )
)
