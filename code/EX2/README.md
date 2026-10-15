# Exercise 2 — parallel DREAM calibration of a MODFLOW flood model

Bayesian calibration of a MODFLOW groundwater model with DREAM, run in parallel across
four MPI ranks. The sampler is `spotpy.algorithms.dream`; the chain it writes is
[`results/EX2/DREAM_modflow.csv`](../../results/EX2/DREAM_modflow.csv).

| File | What it does |
| --- | --- |
| `run_dream.py` | Configures and runs the sampler. Writes `DREAM_modflow.csv` via spotpy's `dbname='DREAM_modflow'`, `dbformat='csv'`, `db_precision=float32`. |
| `spot_setup_modflow.py` | The spotpy setup: the five conductance parameters, the simulation call into MODFLOW, and the objective. |
| `visualization.py` | Reads `DREAM_modflow.csv` and draws the parameter-uncertainty figures. |
| `additional_functions.py` | Helpers, including the semicolon-delimited CSV readers for the observation files. |
| `1_modflow.ipynb` | The MODFLOW notebook the exercise was built from. |
| `inputs/` | `mask_new.npy`, `obs_flood_multi.pkl`, `4_riv_stress_period_data.csv`, `5_head_data_layer5.csv`. |

## Running it needs more than this directory

The MODFLOW input decks and the solver binaries are **not** here. They are in the release
asset, because they are bulk rather than work:

**[`ex2-dream-run-v1`](https://github.com/mzquadri/UQ-Hydrology-Seminar-TUM/releases/tag/ex2-dream-run-v1)**
— the full run directory, 372 MB compressed, 3.7 GB unpacked.

It contains the `flood_riv_dream*` directories with `chd6.dat`, `lpf6.dat`, `discret.dat`,
`hk`, `zone.dat`, `pmr_ip_bas.dat`, `oc.dat`, `pcg2.dat` and `flood.nam`, plus the USGS
`mf2005.exe` and `mt3dms5b.exe` binaries. Unpack it beside this directory and the scripts
have what they need.

## What the chain is

`DREAM_modflow.csv` is spotpy's own output format: one row per evaluation, `like1` the
objective, `parparcond_1..5` the sampled conductance parameters, and `simulation_*` the
simulated heads at the observation points. 15 MB.

A second copy exists in the release with those columns renamed to `parcond_1..5`. It is the
same data. `run_dream.py` writes the original name and `visualization.py` reads it, so the
original is the one tracked here.

## Scope

Group seminar coursework, Group B, WS 2025/26. The figures and the chain are the measured
output of one calibration run. Nothing here was re-run for this repository, and no number
from this exercise appears in the seminar README's results table.
