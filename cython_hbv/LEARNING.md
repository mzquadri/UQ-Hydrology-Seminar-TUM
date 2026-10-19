# The model, and the Cython

What each equation does, and the parts of the Cython that are not obvious.

---

## 1. HBV in three routines

HBV is a conceptual bucket model. It does not solve flow equations; it moves
water between a small number of stores with empirical rules. Every store is in
millimetres of water depth over the catchment, which is why discharge comes out
in mm/day rather than m3/s.

### Snow

```
T < TT   ->  snow += P * SFCF              precipitation falls as snow
T > TT   ->  melt = CFMAX * (T - TT)       degree-day melt, capped at the pack
T <= TT  ->  refreeze = CFR * CFMAX * (TT - T)
             liquid water in the pack refreezes, capped at what is there
pack holds CWH * snow of liquid water; only the excess infiltrates
```

`TT` is the rain/snow threshold, usually near 0 degrees C. `SFCF` is below 1
because precipitation gauges under-catch snow in wind - it is a measurement
correction masquerading as a model parameter.

`CFMAX` is the degree-day factor: millimetres melted per degree above `TT` per
day. The whole snow routine is one idea - warmer days melt proportionally more -
and it works well enough that physically based energy-balance melt models
struggle to beat it on daily data.

`CWH` matters more than it looks. A snowpack holds meltwater like a sponge
before releasing any, which delays the hydrograph by days. Without it, the first
warm day produces a spike that is not observed.

### Soil

```
recharge = infiltration * (SM / FC) ^ BETA
SM      += infiltration - recharge
AET      = PET * min(SM / (LP * FC), 1)
SM      -= AET
```

This is the non-linearity that makes HBV work. When the soil is dry
(`SM/FC` small), `(SM/FC)^BETA` is near zero, so almost all rain is absorbed and
nothing reaches the streams. When it is near saturation the ratio approaches 1
and nearly all rain becomes recharge. `BETA` controls how sharp that transition
is: `BETA = 1` makes it linear, higher values keep the catchment unresponsive
until it is wet.

That single term is why the same storm produces a trickle in August and a flood
in November.

`LP` is the fraction of `FC` above which plants transpire at the full potential
rate. Below it, evapotranspiration is limited by water availability rather than
by energy.

### Response

```
SUZ += recharge
PERC  = min(SUZ, PERC)          percolation to the lower zone
SUZ  -= PERC ;  SLZ += PERC
Q0    = K0 * max(SUZ - UZL, 0)  fast outlet, only above UZL
Q1    = K1 * SUZ                upper zone recession
Q2    = K2 * SLZ                baseflow
Q     = Q0 + Q1 + Q2
```

Two linear reservoirs with `K2 < K1 < K0`. The lower zone drains slowly and
produces baseflow that persists through dry spells; the upper zone responds
within days; `Q0` only exists above the threshold `UZL` and is what produces a
flood peak. Three timescales from three coefficients.

### Potential evapotranspiration

Oudin et al. (2005), chosen because it needs only temperature and day of year:

```
PET = Ra * (T + 5) / 100     for T > -5, else 0
```

`Ra` is extraterrestrial radiation, computed from latitude and solar
declination - pure geometry, no observations. The paper's finding is that this
simple form performs as well as Penman-Monteith *inside a rainfall-runoff
model*, which is the relevant comparison: a more sophisticated PET needs
humidity, wind and radiation, and the DWD series here has temperature and
precipitation.

The `acos` argument is clamped to `[-1, 1]`. At the latitude of Munich it never
leaves that range, but at high latitudes it does during polar day and night, and
an unclamped `acos` would raise there.

---

## 2. Why this loop cannot be vectorised

The obvious NumPy answer does not apply. Every timestep reads the snowpack, soil
moisture and reservoir levels that the previous timestep wrote:

```
snow[i] depends on snow[i-1], which depends on snow[i-2], ...
```

That is a sequential dependency, not an array operation. There is no NumPy
expression for it that is not itself a Python loop. Which is exactly the shape
where Cython earns its keep, and why this is the piece worth typing.

Measured: 8.11 microseconds per timestep in pure Python, 0.0714 in Cython. About
thirty float operations per step, so Python was spending roughly 8 microseconds
on interpreter overhead for about 30 nanoseconds of arithmetic.

---

## 3. What the Cython is actually doing

**Typed memoryviews.** `double[::1] precipitation` declares a one-dimensional,
**C-contiguous** array of doubles. The `[::1]` is the contiguity: it lets
`precipitation[i]` compile to a pointer offset rather than a strided lookup. The
cost is that the caller must pass contiguous arrays, which is why `benchmark.py`
and the test call `np.ascontiguousarray`.

**`cdef double` locals.** Every intermediate - `melt`, `recharge`, `ratio` - is
declared. Without that, each one would be a Python float object: heap-allocated,
reference-counted, and unboxed on every operation.

**Parameters as scalars, not a dataclass.** `simulate` takes sixteen doubles
rather than a `Parameters` object. Reading `p.beta` inside the loop would be an
attribute lookup on a Python object 12,275 times. Passing them flat is uglier at
the call site and the reason the loop body touches no Python object at all.

**`libc.math.pow`, deliberately.** Python's `ratio ** beta` is `float.__pow__`,
which calls libm `pow` on doubles. Writing the Cython side as repeated
multiplication would be faster and would round differently, which is the kind of
"optimisation" that breaks a parity test for a reason nobody enjoys finding.

### The directives, and what each one gives up

| Directive | Removes | Safe because |
|---|---|---|
| `boundscheck=False` | index range check | every index is `i` from `range(n)`; the three lengths are compared once before the loop |
| `wraparound=False` | negative-index support | no negative index appears in the file |
| `cdivision=True` | division-by-zero check | the only denominators are `fc` and `lp*fc`, validated positive before the loop |

`cdivision=True` is the one with teeth. Python raises `ZeroDivisionError`; C
produces `inf` or `nan` and carries on. An `fc` of zero would put an infinity
into `soil` and every subsequent timestep would be garbage, with no exception
anywhere. Hence the explicit `if fc <= 0.0: raise ValueError`, and a test for it.

---

## 4. Proving the two agree

Exact equality, not a tolerance. Both are IEEE-754 binary64 doing the same
operations in the same order, so anything other than exact equality means the
arithmetic changed.

```python
mismatches = [i for i in range(n) if expected[i] != float(actual[i])]
assert not mismatches
```

Repeated for three further parameter sets, varying `BETA` on purpose because it
is the `pow` exponent.

**A parity test can pass for the wrong reason.** Two implementations that both
return zeros agree perfectly. So:

- `test_the_model_actually_does_something` asserts the output takes more than a
  thousand distinct values, stays non-negative, and that total discharge is less
  than total precipitation. That last one is a real mass-balance check: with
  `SFCF < 1` and evapotranspiration removing water, output must be strictly less
  than input over a long series.
- `test_snow_accumulates_below_the_threshold` drives 30 cold wet days then 30
  warm dry ones and asserts **exactly zero** flow in the first half, then flow in
  the second. That is the snow routine working rather than precipitation passing
  straight through.
- `test_the_series_is_the_one_expected` pins the row count at 12,275, so a
  replaced data file fails there rather than surfacing as a parity difference.

---

## 5. What would break this

- **Rewriting `c_pow` as multiplication.** Faster, different rounding, parity
  test fails. The test is the only thing standing between a reasonable-looking
  optimisation and a silent divergence.
- **Passing a non-contiguous array.** A slice like `arr[::2]` is not
  C-contiguous and the memoryview will reject it. That is a loud failure, which
  is the right one.
- **Setting `fc` to zero with `cdivision=True` and no guard.** Infinities
  through the whole state, no exception. The guard exists for this.
- **A DWD file with gaps read as zeros.** The seven `-999` days are dropped
  during extraction. Reading them as 0.0 mm would invent dry days and change
  every subsequent state, in both implementations identically - so the parity
  test would still pass while the model was wrong. Provenance is not something
  a parity test can check for you.
