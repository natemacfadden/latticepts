import os
import platform

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

setup(
    ext_modules=cythonize(
        [Extension(
            "latticepts.box_enum",
            sources=["latticepts/box_enum.pyx"],
            include_dirs=["latticepts"],
            define_macros=[("BOX_ENUM_IMPLEMENTATION", None)],
            extra_compile_args=compile_args,
            language="c",
        )],
        compiler_directives={"language_level": "3"},
    )
)
