# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""The same HBV core loop, typed for Cython.

Line for line the arithmetic in `hbv_python.simulate`. The differences are all
declarations, and that is the point: `test_parity.py` asserts the two produce
**bit-identical** output over 12,275 real timesteps, so any change to the
arithmetic here is a test failure rather than a quiet divergence.

Why the three directives above are safe here, since each one removes a check
that normally protects you:

* `boundscheck=False` - every index is `i` from `range(n)` where `n` is the
  array length, and the arrays are memoryviews whose length is checked once in
  `simulate` before the loop starts.
* `wraparound=False` - no negative index is used anywhere in this file.
* `cdivision=True` - skips Python's division-by-zero check. The denominators are
  `p_fc` and `p_lp * p_fc`, both validated as positive in `simulate` before the
  loop. Without that guard this directive would turn a configuration error into
  an infinity propagating silently through the state.

Bit-identical output across the two is not automatic. C doubles and Python
floats are both IEEE-754 binary64, so the same operations in the same order
agree - but `x ** beta` in Python is `pow` on doubles, and writing it as
anything else here (repeated multiplication, `exp(beta * log(x))`) would round
differently. `libc.math.pow` is used for exactly that reason.
"""

import numpy as np

cimport numpy as cnp
from libc.math cimport pow as c_pow

cnp.import_array()


def simulate(
    double[::1] precipitation not None,
    double[::1] temperature not None,
    double[::1] pet not None,
    double tt,
    double cfmax,
    double sfcf,
    double cwh,
    double cfr,
    double fc,
    double lp,
    double beta,
    double perc,
    double uzl,
    double k0,
    double k1,
    double k2,
):
    """Run the model over contiguous double arrays. Returns a numpy array of discharge.

    Parameters are passed as scalars rather than as a dataclass so the loop body
    touches no Python object at all. `[::1]` requires C-contiguous input, which
    is what lets the indexing compile to pointer arithmetic.
    """
    cdef Py_ssize_t n = precipitation.shape[0]

    if temperature.shape[0] != n or pet.shape[0] != n:
        raise ValueError("precipitation, temperature and pet must be the same length")

    # Checked before the loop because cdivision=True above removes the
    # division-by-zero check that would otherwise catch this.
    if fc <= 0.0:
        raise ValueError("fc must be positive")
    if lp <= 0.0:
        raise ValueError("lp must be positive")

    cdef cnp.ndarray[double, ndim=1] out = np.zeros(n, dtype=np.float64)
    cdef double[::1] discharge = out

    cdef double snow = 0.0
    cdef double liquid = 0.0
    cdef double soil = 0.5 * fc
    cdef double upper = 0.0
    cdef double lower = 0.0

    cdef double precip, temp, rain, melt, refreeze, capacity, infiltration
    cdef double ratio, recharge, limit, actual_et, percolation, q0, q1, q2
    cdef Py_ssize_t i

    for i in range(n):
        precip = precipitation[i]
        temp = temperature[i]

        # --- snow routine -------------------------------------------------
        if temp < tt:
            snow += precip * sfcf
            rain = 0.0
        else:
            rain = precip

        if temp > tt:
            melt = cfmax * (temp - tt)
            if melt > snow:
                melt = snow
            snow -= melt
            liquid += melt
        else:
            refreeze = cfr * cfmax * (tt - temp)
            if refreeze > liquid:
                refreeze = liquid
            liquid -= refreeze
            snow += refreeze

        liquid += rain
        capacity = cwh * snow
        if liquid > capacity:
            infiltration = liquid - capacity
            liquid = capacity
        else:
            infiltration = 0.0

        # --- soil routine -------------------------------------------------
        if infiltration > 0.0:
            ratio = soil / fc
            if ratio > 1.0:
                ratio = 1.0
            # c_pow, not ratio**beta compiled some other way: the Python side
            # uses float.__pow__, which is libm pow on doubles. Repeated
            # multiplication would round differently and break parity.
            recharge = infiltration * c_pow(ratio, beta)
            soil += infiltration - recharge
        else:
            recharge = 0.0

        limit = lp * fc
        if soil < limit:
            actual_et = pet[i] * soil / limit
        else:
            actual_et = pet[i]
        if actual_et > soil:
            actual_et = soil
        soil -= actual_et

        if soil > fc:
            recharge += soil - fc
            soil = fc

        # --- response routine ---------------------------------------------
        upper += recharge

        percolation = perc if upper > perc else upper
        upper -= percolation
        lower += percolation

        q0 = k0 * (upper - uzl) if upper > uzl else 0.0
        upper -= q0
        q1 = k1 * upper
        upper -= q1
        q2 = k2 * lower
        lower -= q2

        discharge[i] = q0 + q1 + q2

    return out
