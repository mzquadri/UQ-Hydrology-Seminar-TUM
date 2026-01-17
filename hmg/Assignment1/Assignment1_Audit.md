# Assignment 1: Parameter Optimization - Technical Audit

## Summary

**Status**: Script runs and produces results. Key fixes applied for reproducibility and plotting compliance.

**PDF Note**: The PDF file `MMUQ_L02_asgt_01.pdf` is binary and cannot be directly read by text tools. Audit based on user-provided requirements and standard MMUQ course expectations.

---

## 1. What Matches Requirements

| Requirement | Status | Notes |
|-------------|--------|-------|
| Uses HBV001A model from repo | ✅ OK | Correctly imports from `hmg` |
| Objective function: 1 - NSE | ✅ OK | `obj_nse()` returns `1.0 - NSE` |
| Uses Differential Evolution | ✅ OK | SciPy `differential_evolution` |
| Data path: repo-local `./data` | ✅ OK | `find_data_dir()` walks up from script |
| Discharge scaler: hourly | ✅ OK | `dslr = area / (3600 * 1000)` |
| 18 parameters | ✅ OK | Bounds list has 18 entries |
| No warmup removal | ✅ OK | Uses full time series (2880 timesteps) |
| Parameter order matches model | ✅ OK | `PARAM_NAMES_OPT` order matches `hbv001a_py.py` |
| Last parameter is `lrr_lct` | ✅ OK | Verified in model file and script |
| No OFV clipping in scatter plots | ✅ OK | `set_ylim()` removed/commented out |

---

## 2. Issues Found and Fixed

### 2.1 FIXED: Missing Random Seed
- **Problem**: `differential_evolution()` had no `seed=` argument
- **Impact**: Results not reproducible across runs
- **Fix**: Added `seed=42`

### 2.2 FIXED: Y-axis Clipping in Convergence Plots
- **Problem**: `ax.set_ylim(0.0, 3.0)` clipped objective values above 3.0
- **Impact**: Violated requirement "Do NOT clip objective values for plots"
- **Fix**: Removed `set_ylim()` from main convergence plot; kept one zoomed view

### 2.3 FIXED: Summary Output Format
- **Problem**: Summary file had inconsistent formatting
- **Fix**: Added seed, generations, popsize to `final_debug_summary.txt`

### 2.4 ADDED: FAST_DEBUG Mode
- **Purpose**: Quick pipeline verification before full run
- **Usage**: Set `FAST_DEBUG = True` for 5 generations test

---

## 3. Parameter Verification

### 3.1 Parameter Names and Indices (from `hmg/models/hbv001a_py.py`)

| Index | Parameter | Description |
|-------|-----------|-------------|
| 0 | snw_dth | Snow initial depth |
| 1 | snw_att | Snow air temperature threshold |
| 2 | snw_pmf | Precipitation melt factor |
| 3 | snw_amf | Air melt factor |
| 4 | sl0_dth | Soil initial depth |
| 5 | sl0_pwp | Permanent wilting point |
| 6 | sl0_fcy | Field capacity |
| 7 | sl0_bt0 | Beta exponent |
| 8 | urr_dth | Upper reservoir initial depth |
| 9 | lrr_dth | Lower reservoir initial depth |
| 10 | urr_wsr | Water split ratio |
| 11 | urr_ulc | URR-LRR connection constant |
| 12 | urr_tdh | Threshold depth |
| 13 | urr_tdr | Threshold drainage rate |
| 14 | urr_ndr | Non-threshold drainage rate |
| 15 | urr_uct | URR cutoff threshold |
| 16 | lrr_dre | LRR drainage rate |
| 17 | lrr_lct | LRR cutoff threshold |

**Verification**: Script `PARAM_NAMES_OPT` list matches this order exactly ✅

### 3.2 Output Labels (for internal state plots)

| Label | Description |
|-------|-------------|
| snw_dth | Snow depth |
| sl0_dth | Soil depth |
| sl0_etn | Evapotranspiration |
| urr_dth | Upper reservoir depth |
| lrr_dth | Lower reservoir depth |
| urr_urf | Upper reservoir runoff |
| lrr_lrf | Lower reservoir runoff |
| chn_pow | Channel power/surface flow |
| mod_bal | Mass balance (should be ~0) |

---

## 4. Configuration

| Setting | Value | Notes |
|---------|-------|-------|
| Generations (`maxiter`) | 80 | Full run |
| Population multiplier (`popsize`) | 16 | |
| Actual population per gen | 16 × 18 = 288 | SciPy: popsize × ndim |
| Total evaluations (approx) | ~23,040 | 288 × 80 |
| Strategy | `best1bin` | |
| Tolerance (`atol`) | 1e-3 | |
| Polish | False | |
| Seed | 42 | For reproducibility |

---

## 5. Bounds Used

```python
bounds_opt = [
    (0.0, 0.0),      # snw_dth - fixed at 0
    (-2.0, 3.0),     # snw_att
    (0.0, 3.0),      # snw_pmf
    (0.0, 10.0),     # snw_amf
    (0.0, 100.0),    # sl0_dth
    (5.0, 700.0),    # sl0_pwp
    (100.0, 700.0),  # sl0_fcy
    (0.01, 10.0),    # sl0_bt0
    (0.0, 20.0),     # urr_dth
    (0.0, 100.0),    # lrr_dth
    (0.0, 1.0),      # urr_wsr
    (0.0, 1.0),      # urr_ulc
    (0.0, 200.0),    # urr_tdh
    (0.01, 1.0),     # urr_tdr
    (0.0, 1.0),      # urr_ndr
    (0.0, 1.0),      # urr_uct
    (0.0, 1.0),      # lrr_dre
    (0.0, 1.0),      # lrr_lct
]
```

---

## 6. Commands

### Run from repo root:
```bash
python hmg/Assignment1/Model_Parameter_Optimisation.py
```

### Quick test (edit script first):
Set `FAST_DEBUG = True` in the script, then run same command.

---

## 7. Output Files

Located in `hmg/Assignment1/results/`:

| File | Description |
|------|-------------|
| `final_debug_summary.txt` | Best parameters, NSE, OFV, config |
| `optimization_history_evals.csv` | All evaluated parameter sets + objectives |
| `optimization_gen_summary.csv` | Per-generation statistics |
| `observed_vs_simulated_best.png` | Discharge comparison plot |
| `convergence_per_generation.png` | OFV convergence (full range) |
| `convergence_per_generation_ytotal.png` | OFV convergence (zoomed) |
| `param_scatter_*.png` | Parameter vs OFV scatter plots |
| `params_normalized_best.png` | Normalized best parameters |
| `params_evolution_per_generation.png` | Parameter evolution |
| `Inputs, and internally simulated variables of HBV.png` | Internal states |

---

## 8. Sanity Checks

After full run, verify:
- [ ] NSE > 0.5 (reasonable fit)
- [ ] OFV < 0.5 (1 - NSE)
- [ ] No NaN/Inf in simulated discharge
- [ ] Mass balance (`mod_bal`) stays near zero
- [ ] Convergence plot shows decreasing trend
- [ ] Scatter plots show full OFV range (not clipped)
