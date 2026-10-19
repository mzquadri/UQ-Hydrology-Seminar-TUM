# A Cython HBV core loop, held to the Python one

A conceptual HBV rainfall-runoff model written twice: once in pure Python, once
in Cython. The two produce **bit-identical** discharge over 12,275 real daily
timesteps, and the Cython version runs **113x** faster.

```
timesteps          12275
pure Python        99.51 ms  (8.1069 us/step)
Cython              0.88 ms  (0.0714 us/step)
speedup            113.49x
bit-identical      True
```

Measured output in [`results/benchmark.json`](results/benchmark.json), produced
by [`benchmark.py`](benchmark.py). Python 3.11.17, NumPy 2.2.6, x86_64 Linux,
best of 9 repetitions.

## Why the model is written here rather than ported

The seminar assignments in `../code/` call `HBV001A` from the course-provided
`hmg` package. **That package is not in this repository and is not
redistributable** - the repository README says so, and the same goes for the
catchment forcing and area inputs. There is no HBV source here to optimise.

So this is a standalone implementation of the published HBV structure, written
to be the thing the Cython version is checked against. It follows HBV-light as
documented by Seibert and Vis (2012), which is the HBV of Bergstrom (1976):

- **snow** - precipitation falls as snow below `TT`, with a snowfall correction;
  melt is a degree-day function limited by the pack; the pack refreezes liquid
  water and holds a fraction `CWH` of it before releasing any
- **soil** - recharge grows as `(SM/FC)^BETA` of infiltration; actual
  evapotranspiration is potential ET scaled by storage up to `LP*FC`
- **response** - two linear reservoirs, the upper with a fast outlet above
  `UZL`, separated by a percolation limit `PERC`

Potential evapotranspiration uses Oudin et al. (2005), which needs only
temperature and day of year. That is what the available data supports.

**This model is not calibrated.** The parameters are mid-range defaults from the
HBV-light documentation, not a fit to any catchment, and the discharge it
produces is not a prediction of anything. What it is, is a fixed deterministic
time-stepping loop over real data - which is what both the parity test and the
benchmark need.

## The data

Daily observations from **DWD station 01262, Muenchen-Stadt**.

| | |
|---|---|
| Source | Deutscher Wetterdienst, Climate Data Center |
| Path | `climate_environment/CDC/observations_germany/climate/daily/kl/historical/` |
| File | `tageswerte_KL_01262_19920517_20251231_hist.zip`, SHA-256 `338fa1204729c280250970fd738963a1026812b74ffe3ddb147b9d579a397817` |
| Downloaded | 2026-10-19 |
| Licence | Open data. DWD publishes its CDC observations for free reuse under the GeoNutzV ordinance, with the source acknowledged |
| Attribution | Deutscher Wetterdienst (DWD), Climate Data Center, station 01262 |

Extracted to [`data/dwd_01262_daily.csv`](data/dwd_01262_daily.csv): `date`,
`precip_mm` (DWD field `RSK`), `tmean_c` (field `TMK`). 1992-05-17 to
2025-12-31, SHA-256 `a71bd8b296729d537cb5df1f5d38dbb52a29f64b2f6a593de5f5e3228ebe03e6`.

**12,275 rows of 12,282.** Seven days carry DWD's `-999` missing-value code and
are dropped rather than read as zero rainfall, which would have invented seven
dry days.

## Running it

A C compiler is required. The pinned image in
[`Dockerfile.bench`](Dockerfile.bench) is what produced the numbers above:

```bash
docker build -f Dockerfile.bench -t hbv-bench .
docker run --rm -v "$PWD":/work -w /work hbv-bench bash -c "
  python setup.py build_ext --inplace &&
  python -m pytest test_parity.py -q &&
  python benchmark.py --repeats 9 --out results/benchmark.json"
```

Locally, with a toolchain already present:

```bash
pip install cython numpy setuptools pytest
python setup.py build_ext --inplace
python -m pytest test_parity.py -v
python benchmark.py --out results/benchmark.json
```

`test_parity.py` skips rather than fails when the extension has not been built,
so it does not make a compiler a prerequisite for the rest of the repository.

## Why the two agree exactly

Bit-identical is a stronger claim than "close", and it is not automatic.

C doubles and Python floats are both IEEE-754 binary64, so the same operations
in the same order give the same answer. The Cython file is therefore a
line-for-line transcription of the Python loop, differing only in declarations.
One choice matters more than the rest: `ratio ** beta` in Python is
`float.__pow__`, which is libm `pow` on doubles, so the Cython side calls
`libc.math.pow` rather than expanding it. Repeated multiplication, or
`exp(beta * log(x))`, would round differently and the test would fail.

`test_parity.py` asserts exact equality across the whole series and repeats it
for three other parameter sets, varying `BETA` specifically because it is the
exponent where rounding could diverge.

Two tests exist to stop the parity test passing vacuously:

- `test_the_model_actually_does_something` - a loop returning zeros everywhere
  would match perfectly. It asserts the output varies over more than a thousand
  distinct values, stays non-negative, and respects mass balance against total
  precipitation.
- `test_snow_accumulates_below_the_threshold` - a cold wet spell must produce no
  flow at all, then release it once the temperature rises above `TT`.

## Where the speedup comes from

The loop is sequential: every timestep reads the snowpack, soil moisture and
reservoir levels left by the one before it. That rules out vectorising it in
NumPy, which is the usual first answer, and leaves a genuine Python-level loop
over 12,275 iterations with roughly thirty float operations in each.

What Cython removes is per-operation interpreter overhead: boxed floats, dynamic
dispatch on every arithmetic operator, and list indexing through
`PyObject_GetItem`. With `double` locals and typed memoryviews the body compiles
to straight C on unboxed doubles.

Three directives in the `.pyx` header each remove a safety check, and each is
justified in place rather than copied in:

| Directive | Why it is safe here |
|---|---|
| `boundscheck=False` | every index is `i` from `range(n)`, and the three lengths are compared once before the loop |
| `wraparound=False` | no negative index appears in the file |
| `cdivision=True` | the only denominators are `fc` and `lp*fc`, both validated positive before the loop |

That last one is why `simulate` raises on `fc <= 0` explicitly: with
`cdivision=True` a zero denominator would otherwise produce an infinity that
propagates silently through the state instead of an exception.
`test_non_positive_fc_is_rejected_by_cython` pins it.

See [LEARNING.md](LEARNING.md) for the HBV equations in full and what each
parameter does.
