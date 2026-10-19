"""The Cython loop must reproduce the Python loop exactly, not approximately.

Run after building the extension:

    python setup.py build_ext --inplace
    python -m pytest test_parity.py -v

`hbv_cython` is skipped rather than failed when it has not been built, so the
rest of the repository's test run does not depend on a compiler being present.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import hbv_python as py

hbv_cython = pytest.importorskip(
    "hbv_cython",
    reason="extension not built; run `python setup.py build_ext --inplace`",
)

PARAMS = py.Parameters()


def run_cython(precipitation, temperature, pet, p=PARAMS):
    return hbv_cython.simulate(
        np.ascontiguousarray(precipitation, dtype=np.float64),
        np.ascontiguousarray(temperature, dtype=np.float64),
        np.ascontiguousarray(pet, dtype=np.float64),
        p.tt, p.cfmax, p.sfcf, p.cwh, p.cfr, p.fc, p.lp,
        p.beta, p.perc, p.uzl, p.k0, p.k1, p.k2,
    )


@pytest.fixture(scope="module")
def forcing():
    return py.load_forcing()


def test_the_series_is_the_one_expected(forcing):
    precipitation, temperature, pet = forcing
    # Pinned so a silently replaced data file fails here rather than showing up
    # as a parity difference, which would be a confusing place to find it.
    assert len(precipitation) == 12275
    assert len(temperature) == 12275
    assert len(pet) == 12275
    assert all(p >= 0.0 for p in precipitation)
    assert all(e >= 0.0 for e in pet)


def test_bit_identical_over_the_whole_series(forcing):
    """The headline assertion: equal doubles, not merely close ones."""
    precipitation, temperature, pet = forcing

    expected = py.simulate(precipitation, temperature, pet, PARAMS)
    actual = run_cython(precipitation, temperature, pet)

    assert len(actual) == len(expected)

    # Exact equality, no tolerance. Both sides are IEEE-754 binary64 doing the
    # same operations in the same order, so anything else means the arithmetic
    # was changed.
    mismatches = [
        (i, expected[i], float(actual[i]))
        for i in range(len(expected))
        if expected[i] != float(actual[i])
    ]
    assert not mismatches, f"{len(mismatches)} of {len(expected)} differ; first: {mismatches[:3]}"


def test_the_model_actually_does_something(forcing):
    """Guards against both implementations agreeing on nothing.

    A loop that returned zeros everywhere would pass the parity test perfectly,
    so the output has to be shown to vary and to stay physically plausible.
    """
    precipitation, temperature, pet = forcing
    discharge = py.simulate(precipitation, temperature, pet, PARAMS)

    assert all(q >= 0.0 for q in discharge), "discharge cannot be negative"
    assert max(discharge) > 0.1, "the model produced essentially no flow"
    assert len(set(round(q, 6) for q in discharge)) > 1000, "output barely varies"

    # Mass balance: total discharge cannot exceed total input. Snowfall
    # correction is below 1 and evapotranspiration removes water, so output is
    # strictly less than precipitation over a long series.
    assert sum(discharge) < sum(precipitation)


@pytest.mark.parametrize(
    "beta,fc,k0",
    [(1.0, 100.0, 0.1), (3.5, 400.0, 0.5), (2.0, 250.0, 0.25)],
)
def test_parity_holds_for_other_parameters(forcing, beta, fc, k0):
    """Parity must be a property of the code, not of one parameter set.

    `beta` is the interesting one: it is the exponent passed to pow, which is
    where the two languages could most easily round differently.
    """
    precipitation, temperature, pet = forcing
    p = py.Parameters(beta=beta, fc=fc, k0=k0)

    expected = py.simulate(precipitation, temperature, pet, p)
    actual = run_cython(precipitation, temperature, pet, p)

    assert all(expected[i] == float(actual[i]) for i in range(len(expected)))


def test_snow_accumulates_below_the_threshold():
    """A cold wet spell must produce no flow, then release it when it warms."""
    cold = [10.0] * 30 + [0.0] * 30
    temps = [-5.0] * 30 + [10.0] * 30
    pet = [0.0] * 60

    discharge = py.simulate(cold, temps, pet, PARAMS)

    assert sum(discharge[:30]) == 0.0, "precipitation fell as snow and should be held"
    assert sum(discharge[30:]) > 0.0, "the pack should melt once above TT"
    assert all(py.simulate(cold, temps, pet, PARAMS)[i] == float(run_cython(cold, temps, pet)[i])
               for i in range(60))


def test_length_mismatch_is_rejected_by_both():
    with pytest.raises(ValueError):
        py.simulate([1.0, 2.0], [1.0], [1.0, 2.0], PARAMS)
    with pytest.raises(ValueError):
        run_cython([1.0, 2.0], [1.0], [1.0, 2.0])


def test_non_positive_fc_is_rejected_by_cython():
    """cdivision=True removes the division-by-zero check, so this guard matters."""
    with pytest.raises(ValueError, match="fc must be positive"):
        run_cython([1.0], [1.0], [1.0], py.Parameters(fc=0.0))


def test_pet_formula_matches_a_hand_computation():
    """One value of Oudin PET checked against the formula written out separately."""
    doy, temp, lat = 180, 20.0, py.STATION_LATITUDE_DEG

    latitude = math.radians(lat)
    declination = 0.409 * math.sin(2.0 * math.pi * doy / 365.0 - 1.39)
    sunset = math.acos(max(-1.0, min(1.0, -math.tan(latitude) * math.tan(declination))))
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 365.0)
    ra = 15.392 * dr * (
        sunset * math.sin(latitude) * math.sin(declination)
        + math.cos(latitude) * math.cos(declination) * math.sin(sunset)
    )

    assert py.oudin_pet(doy, temp, lat) == pytest.approx(ra * 25.0 / 100.0)
    # The formula's own cut-off, not a choice made in this repository.
    assert py.oudin_pet(doy, -5.0, lat) == 0.0
    assert py.oudin_pet(doy, -20.0, lat) == 0.0
