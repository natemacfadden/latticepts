from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        [Extension(
            "conevecs.box_enum",
            sources=["conevecs/box_enum.pyx"],
            include_dirs=["conevecs"],
            define_macros=[("BOX_ENUM_IMPLEMENTATION", None)],
            extra_compile_args=["-O3"],
            language="c",
        )],
        compiler_directives={"language_level": "3"},
    )
)
