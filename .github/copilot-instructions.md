# AI Coding Instructions for HBV Hydrology Model Project

## Project Overview

This is a **hydrological modeling and uncertainty quantification** project implementing the HBV (Hydrologiska Byråns Vattenbalansavdelning) rainfall-runoff model. The project consists of three major assignments analyzing the HBV-001A model variant through parameter optimization and sensitivity analysis.

**Core Structure:**
- `hmg/` - Pure Python implementation of HBV model with assignment scripts
- `hmg_cython/` - Cython-optimized version for computationally intensive operations (Assignment 3)
- Input data expected at `D:\Python Projects\hmg\data\` (user-specific, needs path adjustment)

## HBV Model Architecture

### The HBV001A Model Class

The model is a **lumped conceptual hydrological model** with 18 parameters controlling snow accumulation/melt, soil moisture, and reservoir routing. Access via:

```python
from hmg import HBV001A  # Import from package root

# Standard workflow (see hmg/test/aa_run_model.py)
model = HBV001A()
model.set_inputs(temps, precip, pet)  # 1D numpy arrays, same length
model.set_outputs(n_timesteps)
model.set_discharge_scaler(catchment_area / (3600 * 1000))  # mm/hr → m³/s
model.set_parameters(params)  # 18-element float32 array
model.run_model()
discharge = model.get_discharge()  # Simulated runoff
```

**Critical Implementation Notes:**
1. **Parameter order is FIXED** - indices defined in `models/hbv001a_py.py` lines 62-98
2. **Units matter**: Temperature (K/°C/°F), Precipitation (mm/hr), PET (mm/hr), Discharge (m³/s after scaling)
3. **Discharge scaler** converts mm/hr to m³/s: `dslr = catchment_area_m2 / (3600 * 1000)`
4. Use `np.float32` for all arrays to match Cython implementation
5. The model has both Python (`hbv001a_py.py`) and Cython (`hbv001a_cy.pyx`) implementations - the class automatically uses Cython if available

### Parameter Naming Convention

Parameters use a **3-letter prefix system** (defined in comments at top of model files):

- **`snw_*`** - Snow module (4 params: initial depth, temperature thresholds, melt factors)
- **`sl0_*`** - Soil moisture (4 params: depth, PWP, field capacity, beta exponent)
- **`urr_*`** - Upper reservoir (8 params: initial depth, routing, thresholds, recession)
- **`lrr_*`** - Lower reservoir (2 params: recession rate, cutoff threshold)

**Example:** `urr_ulc` = Upper Reservoir → Upper-Lower Connection constant

See `hmg/models/hbv001a_py.py` lines 20-60 for complete symbol glossary.

## Assignment-Specific Patterns

### Assignment 1: Parameter Optimization (Differential Evolution)

**File:** `hmg/Assignment1/Model_Parameter_Optimisation.py`

**Pattern: DERecorder Class for Tracking Optimization**
```python
class DERecorder:
    """Wraps model evaluation and tracks optimization history"""
    def __init__(self, inputs, dslr, tsps, obs_q, model_class=HBV001A, pop_sz=15):
        self.input = inputs  # (temps, precip, pet) tuple
        self.eval_params = []  # History of all evaluated parameter sets
        self.eval_objs = []    # Corresponding objective values (1-NSE)
        self.gen_bounds = []   # Track generation boundaries
```

**Key Workflow:**
1. SciPy's `differential_evolution` optimizer called with custom callback
2. Each generation stores parameter sets + objectives in `DERecorder`
3. **Objective function:** `1.0 - NSE` where NSE is Nash-Sutcliffe Efficiency
4. Generates 25 plots per run (convergence, parameters, discharge comparison)

**Critical Constants:**
- `PENALTY = 5.0` for invalid/failed runs
- Default bounds: see lines 426-444 (wide ranges for global search)
- Population size: 15-20 (configurable)
- Max generations: 80 (typical)

### Assignment 2: Local Sensitivity Analysis

**File:** `hmg/Assignment2/Local_Sensitivity_Analysis.py`

**Pattern: One-at-a-Time (OAT) Perturbation**
```python
# For each parameter, perturb by ±10%, ±20%, ±30%
perturbations = [-30, -20, -10, 10, 20, 30]  # Percent changes
for param_name in PARAM_NAMES_OPT:
    for perturb_pct in perturbations:
        perturbed_value = base_value * (1 + perturb_pct/100)
        # Clip to parameter bounds
        # Run model, compute NSE change
```

**Results Organization:**
- `results/plus_10_percent/`, `results/minus_20_percent/`, etc.
- Separate subdirectories for each perturbation level
- CSV files track NSE changes: `(NSE_perturbed - NSE_base) / NSE_base * 100`

**Physical Constraint Checking:**
- Parameters hitting bounds → stored in `summary_outside_boundaries.txt`
- Common issue: `sl0_fcy` (field capacity) clipped at 100mm lower bound

### Assignment 3: Global Sensitivity Analysis (Sobol Indices)

**File:** `hmg_cython/Assignment3/Global_Sensitivity_Analysis_using_Sobol_Indices.py`

**Pattern: Saltelli Sampling for Sobol Indices**
```python
# Use scipy.stats.qmc for low-discrepancy sampling
from scipy.stats import qmc
sampler = qmc.Sobol(d=17, scramble=True, seed=123)  # 17 variable params
sample = sampler.random_base2(m=9)  # N=2^9=512 samples

# Saltelli matrices: A, B, and C_i for each parameter
# Total runs = N × (2 × D + 2) where D=17 → ~9,728 evaluations
```

**Critical Implementation Details:**
1. **`snw_dth` excluded** from Sobol analysis (fixed at 0.0) → only 17 variable params
2. **Parallel evaluation** using `joblib.Parallel(n_jobs=4)` for 4-8x speedup
3. **Failure tracking** separate from objective value (see lines 30-32 comments)
4. **Objective capping** at 5.0 to prevent outlier distortion
5. **Jansen 1999 estimator** for total effects (lower variance than Saltelli)

**Narrow vs Wide Bounds:**
- Lines 53-78: Full parameter space (wide exploration)
- Lines 82-103: Narrow range around calibrated optimum (local uncertainty)

## Data Requirements & Conventions

### Input Data Files

**Expected location:** `D:\Python Projects\hmg\data\` (hardcoded, must update per user)

**Required files:**
1. `time_series___24163005.csv` - Meteorological + discharge data
   - Columns: `tavg__ref`, `pptn__ref`, `petn__ref`, `diso__ref`
   - Index: DateTime format `%Y-%m-%d-%H` (hourly resolution)
   - Separator: semicolon (`;`)

2. `area___24163005.csv` - Catchment area in m²
   - Single value used for discharge scaling

**Data loading pattern (used in ALL scripts):**
```python
main_dir = Path(r'D:\Python Projects\hmg\data')  # UPDATE THIS PATH
inp_dfe = pd.read_csv(main_dir / 'time_series___24163005.csv', sep=';', index_col=0)
inp_dfe.index = pd.to_datetime(inp_dfe.index, format='%Y-%m-%d-%H')
ccaa = pd.read_csv(main_dir / 'area___24163005.csv', sep=';', index_col=0).values[0, 0]
```

### Output Organization

Each assignment has its own `results/` subdirectory:
- `hmg/Assignment1/results/` - Optimization convergence plots, CSV histories
- `hmg/Assignment2/results/` - Perturbation-specific folders with sensitivity plots
- `hmg_cython/Assignment3/results/` - Sobol indices CSV + visualization

**Standard output pattern:**
```python
out_dir = Path(__file__).parent / 'results'
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / 'plot_name.png', bbox_inches='tight', dpi=150)
```

## Performance & Optimization

### When to Use Cython Version

- **Assignment 3 REQUIRES Cython** (`hmg_cython/` folder) due to ~10,000 model runs
- Cython compilation: `python setup.py build_ext --inplace` (if needed)
- Speedup: 10-50x for computational loops
- Import path unchanged: `from hmg import HBV001A` auto-detects compiled version

### Typical Runtimes

- Assignment 1: ~15-30 minutes (differential evolution, 20k+ evaluations)
- Assignment 2: ~7-10 minutes (2,160 perturbation runs)
- Assignment 3: ~40-60 minutes with Cython + parallel (N=512 Sobol samples)

### Parallelization

**Assignment 3 uses joblib:**
```python
from joblib import Parallel, delayed
results = Parallel(n_jobs=4)(delayed(run_model)(params) for params in param_sets)
```

DO NOT parallelize Assignments 1-2 (sequential optimization/perturbation logic).

## Common Issues & Solutions

1. **Path errors:** All scripts hardcode `D:\Python Projects\hmg\data` - must update for local environment
2. **Import errors:** Run scripts from workspace root OR add parent to `sys.path` (see `hmg/test/aa_run_model.py` lines 24-26)
3. **Parameter bound violations:** Check `get_abds_prms_py()` for absolute limits before optimization
4. **Memory usage:** Assignment 3 with N=2048 samples → ~40k runs → use optimization flag in model
5. **NSE = NaN:** Usually means zero variance in observations or all-NaN comparison

## Project-Specific Conventions

- **Float precision:** Always `np.float32` for model arrays (Cython compatibility)
- **Index-based parameter access:** Use `get_parameter_labels()` dict, not hardcoded integers
- **Objective convention:** Minimization of `1 - NSE` (not maximization of NSE)
- **Plotting style:** Consistent use of `alpha=0.75`, `dpi=150`, `bbox_inches='tight'`
- **Documentation language:** Comments mix English and Urdu/Hindi (student notes in markdown files)

## Testing & Validation

**Basic model test:** Run `hmg/test/aa_run_model.py` to verify:
- Model loads and runs without errors
- Discharge values are finite and physically reasonable
- Water balance (`mod_bal` output) near zero

**Parameter validation:**
```python
model.verify_parameters(params)  # Raises AssertionError if out of bounds
```

## Key Files for Understanding

1. `hmg/models/hbv001a_py.py` - Model implementation + parameter definitions
2. `hmg/__init__.py` - Package entry point (imports HBV001A)
3. `hmg/test/aa_run_model.py` - Simplest working example
4. `hmg/Assignment*/PRESENTATION_GUIDE.md` - Detailed explanations for each assignment

---

**When making changes:** Always verify parameter order/indices match the model definition. The HBV model is sensitive to parameter sequence errors.
