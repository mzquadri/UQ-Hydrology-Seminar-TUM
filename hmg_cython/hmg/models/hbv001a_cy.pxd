# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

import numpy as np
cimport numpy as np

from numpy cimport int64_t, npy_float32 as float32


cpdef void hbv001a_cy(
        const float32[::1] tems,
        const float32[::1] ppts,
        const float32[::1] pets,
              float32[:, ::1] otps,
              float32[::1] diss,
        const float32[::1] prms,
        const int64_t oflg,
        const float32 dslr,
        ) except +
