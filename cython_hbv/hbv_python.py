"""A conceptual HBV rainfall-runoff model: the reference pure-Python implementation.

Why this file exists rather than a port of the course model: the seminar's
assignments call `HBV001A` from the course-provided `hmg` package, which is not
redistributable and is therefore not in this repository (see the README). There
is no HBV source here to optimise. So this is a standalone implementation of the
published HBV structure, written to be the thing the Cython version is checked
against.

Structure follows HBV-light as documented by Seibert and Vis (2012), which is
itself the HBV of Bergstrom (1976, 1992):

  snow      P is snow when T < TT, accumulating with a snowfall correction;
            melt is a degree-day function, limited by the pack
  soil       recharge grows as a power of relative storage; actual
             evapotranspiration is potential ET scaled by storage up to LP*FC
  response   two linear reservoirs, the upper with a fast outlet above UZL,
             separated by a percolation limit

Potential evapotranspiration uses Oudin et al. (2005), which needs only
temperature and day of year. That matters here because the DWD series carries
precipitation and temperature and nothing else.

What this is **not**: calibrated. No parameter here was fitted to a catchment,
and the discharge it produces is not a prediction of anything. The point is a
fixed, deterministic time-stepping loop that the Cython version must reproduce
exactly, and a benchmark over 12,275 real timesteps.

References
----------
Bergstrom, S. (1976). Development and application of a conceptual runoff model
    for Scandinavian catchments. SMHI Report RHO 7.
Seibert, J. and Vis, M. J. P. (2012). Teaching hydrological modeling with a
    user-friendly catchment-runoff-model software package. Hydrology and Earth
    System Sciences 16, 3315-3325.
Oudin, L. et al. (2005). Which potential evapotranspiration input for a lumped
    rainfall-runoff model? Journal of Hydrology 303, 290-306.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "dwd_01262_daily.csv"

#: Latitude of DWD station 01262, Muenchen-Stadt, in degrees. Used only by the
#: Oudin PET formula, which needs extraterrestrial radiation.
STATION_LATITUDE_DEG = 48.1632


@dataclass(frozen=True)
class Parameters:
    """HBV parameters, in the units HBV-light uses.

    The values are mid-range defaults from the HBV-light documentation, not a
    calibration. They are frozen so the Python and Cython runs are comparing the
    same model rather than the same code with different numbers.
    """

    tt: float = 0.0        # threshold temperature for snow, degrees C
    cfmax: float = 3.0     # degree-day melt factor, mm / (degree C * day)
    sfcf: float = 0.9      # snowfall correction factor, dimensionless
    cwh: float = 0.1       # water holding capacity of the snowpack, fraction
    cfr: float = 0.05      # refreezing coefficient, dimensionless
    fc: float = 250.0      # maximum soil moisture storage, mm
    lp: float = 0.7        # soil moisture above which AET reaches PET, fraction of FC
    beta: float = 2.0      # shape of the recharge function, dimensionless
    perc: float = 1.5      # maximum percolation to the lower zone, mm/day
    uzl: float = 20.0      # storage threshold for the fast outlet Q0, mm
    k0: float = 0.25       # fast recession coefficient, 1/day
    k1: float = 0.1        # upper zone recession coefficient, 1/day
    k2: float = 0.02       # lower zone recession coefficient, 1/day


def oudin_pet(day_of_year: int, temperature_c: float, latitude_deg: float) -> float:
    """Potential evapotranspiration in mm/day, after Oudin et al. (2005).

    Returns 0 below -5 degrees C, which is the formula's own cut-off rather than
    a choice made here.
    """
    latitude = math.radians(latitude_deg)

    # Solar declination and the sunset hour angle, for extraterrestrial radiation.
    declination = 0.409 * math.sin(2.0 * math.pi * day_of_year / 365.0 - 1.39)
    x = -math.tan(latitude) * math.tan(declination)
    # Clamped because at high latitudes the argument leaves [-1, 1] during polar
    # day and night, where acos is undefined.
    x = max(-1.0, min(1.0, x))
    sunset_angle = math.acos(x)

    inverse_distance = 1.0 + 0.033 * math.cos(2.0 * math.pi * day_of_year / 365.0)
    # 15.392 converts to mm/day equivalent; 0.0820 MJ m-2 min-1 is the solar constant.
    radiation = (
        15.392
        * inverse_distance
        * (
            sunset_angle * math.sin(latitude) * math.sin(declination)
            + math.cos(latitude) * math.cos(declination) * math.sin(sunset_angle)
        )
    )

    if temperature_c + 5.0 <= 0.0:
        return 0.0
    return radiation * (temperature_c + 5.0) / 100.0


def simulate(
    precipitation: list[float],
    temperature: list[float],
    pet: list[float],
    p: Parameters,
) -> list[float]:
    """Run the model and return simulated discharge in mm/day, one value per step.

    State is five scalars carried across the loop: snowpack, liquid water in the
    pack, soil moisture, upper zone and lower zone. Every step depends on the
    previous one, which is why this is a loop and not a vectorised expression -
    and why it is worth handing to Cython.
    """
    n = len(precipitation)
    if not (len(temperature) == len(pet) == n):
        raise ValueError("precipitation, temperature and pet must be the same length")

    snow = 0.0
    liquid = 0.0
    soil = 0.5 * p.fc          # start half full, a conventional warm-up state
    upper = 0.0
    lower = 0.0

    discharge = [0.0] * n

    for i in range(n):
        precip = precipitation[i]
        temp = temperature[i]

        # --- snow routine -------------------------------------------------
        if temp < p.tt:
            snow += precip * p.sfcf
            rain = 0.0
        else:
            rain = precip

        if temp > p.tt:
            melt = p.cfmax * (temp - p.tt)
            if melt > snow:
                melt = snow
            snow -= melt
            liquid += melt
        else:
            refreeze = p.cfr * p.cfmax * (p.tt - temp)
            if refreeze > liquid:
                refreeze = liquid
            liquid -= refreeze
            snow += refreeze

        liquid += rain
        # The pack releases only what it cannot hold.
        capacity = p.cwh * snow
        if liquid > capacity:
            infiltration = liquid - capacity
            liquid = capacity
        else:
            infiltration = 0.0

        # --- soil routine -------------------------------------------------
        if infiltration > 0.0:
            ratio = soil / p.fc
            if ratio > 1.0:
                ratio = 1.0
            recharge = infiltration * (ratio ** p.beta)
            soil += infiltration - recharge
        else:
            recharge = 0.0

        limit = p.lp * p.fc
        if soil < limit:
            actual_et = pet[i] * soil / limit
        else:
            actual_et = pet[i]
        if actual_et > soil:
            actual_et = soil
        soil -= actual_et

        # Soil cannot exceed its capacity; the excess joins recharge.
        if soil > p.fc:
            recharge += soil - p.fc
            soil = p.fc

        # --- response routine ---------------------------------------------
        upper += recharge

        percolation = p.perc if upper > p.perc else upper
        upper -= percolation
        lower += percolation

        q0 = p.k0 * (upper - p.uzl) if upper > p.uzl else 0.0
        upper -= q0
        q1 = p.k1 * upper
        upper -= q1
        q2 = p.k2 * lower
        lower -= q2

        discharge[i] = q0 + q1 + q2

    return discharge


def load_forcing(path: Path = DATA) -> tuple[list[float], list[float], list[float]]:
    """Read the DWD series and derive PET. Returns (precipitation, temperature, pet)."""
    precipitation: list[float] = []
    temperature: list[float] = []
    pet: list[float] = []

    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            precipitation.append(float(row["precip_mm"]))
            temp = float(row["tmean_c"])
            temperature.append(temp)

            stamp = row["date"]
            year, month, day = int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8])
            # Day of year without the datetime module, so the Cython translation
            # of this loop has nothing Python-specific to reproduce.
            cumulative = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)
            doy = cumulative[month - 1] + day
            leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
            if leap and month > 2:
                doy += 1

            pet.append(oudin_pet(doy, temp, STATION_LATITUDE_DEG))

    return precipitation, temperature, pet
