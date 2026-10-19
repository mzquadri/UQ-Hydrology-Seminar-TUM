"""Build the Cython extension in place.

    python setup.py build_ext --inplace

Kept as a setup.py rather than a pyproject build because the extension is built
in place and imported from this directory, not installed. The seminar code is
run as scripts from the repository root, and this follows that.
"""

from setuptools import Extension, setup

import numpy as np
from Cython.Build import cythonize

extensions = [
    Extension(
        name="hbv_cython",
        sources=["hbv_cython.pyx"],
        include_dirs=[np.get_include()],
        # Silences the deprecated-NumPy-API warning that the generated C emits.
        # It is about NumPy's own headers, not about this code.
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    name="hbv_cython",
    ext_modules=cythonize(
        extensions,
        language_level="3",
        # Directives live in the .pyx header so they travel with the source and
        # a reader sees them next to the code they affect.
        annotate=True,
    ),
    zip_safe=False,
)
