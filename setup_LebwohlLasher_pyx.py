# python setup_LebwohlLasher_pyx.py build_ext -fi 
# install_name_tool -delete_rpath /opt/anaconda3/lib LebwohlLasher_pyx.cpython-312-darwin.so

from Cython.Build import cythonize

from setuptools import setup, Extension

import numpy

ext_modules = [
    Extension(
        "LebwohlLasher_pyx",
        ["LebwohlLasher_pyx.pyx"],

        include_dirs=[
            numpy.get_include()
        ],

        extra_compile_args=[
            "-O3"
        ],
    )
]

setup(
    name="LebwohlLasher_pyx",
    ext_modules=cythonize(
        ext_modules, 
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
        },
        annotate = True,
    ),
)