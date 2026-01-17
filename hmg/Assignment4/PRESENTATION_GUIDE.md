# Assignment 4: Input Uncertainty Propagation - Presentation Guide

## Overview

This assignment implements **Monte Carlo simulation** to propagate precipitation uncertainty through the HBV001A hydrological model. The goal is to understand how input errors (measurement uncertainty in precipitation) affect model outputs (discharge predictions).

## Methodology

### 1. Precipitation Perturbation

Each timestep's precipitation is scaled by a random factor:

$$P_{perturbed,t} = P_{original,t} \times C_t$$

where $C_t \sim \mathcal{N}(1, 0.05)$ (normal distribution with mean=1, σ=0.05)

This represents ±5% measurement uncertainty in precipitation gauges.

### 2. Monte Carlo Simulation (N=2000 iterations)

For each iteration:
1. **Perturb precipitation**: Generate new random scaling factors
2. **Reference evaluation**: Run model with calibrated (A1) parameters + perturbed PPT
3. **Recalibration**: Re-optimize parameters using perturbed PPT as "true" input
4. **Record**: Store NSE values and recalibrated parameters

### 3. Analysis

Compare:
- **NSE(ref + pert)**: Performance with fixed reference parameters under precipitation uncertainty
- **NSE(recalib)**: Performance after recalibration to each perturbed realization

## Key Findings (Expected)

1. **Reference parameters are sensitive to input uncertainty**
   - When using calibrated parameters with perturbed precipitation, NSE varies significantly
   - This shows how measurement errors propagate through the model

2. **Recalibration compensates for input errors**
   - After recalibration, NSE is typically higher and more stable
   - However, parameter values become uncertain (equifinality)

3. **Parameter uncertainty is induced by input uncertainty**
   - Even with identical model structure, precipitation uncertainty causes parameter scatter
   - Some parameters are more sensitive than others

## Output Files

| File | Description |
|------|-------------|
| `mc_results.csv` | All iteration results: NSE values and recalibrated parameters |
| `nse_distributions.png` | Histograms and CDFs of NSE values |
| `parameter_boxplots.png` | Distribution of recalibrated parameter values |
| `parameter_scatter.png` | Parameter values vs iteration with reference lines |
| `discharge_uncertainty.png` | Discharge uncertainty bands (5-95 percentile) |
| `summary_report.txt` | Statistical summary of all results |
| `sim_q_*.npy` | Raw discharge arrays for further analysis |

## Running the Script

### Quick Test Mode (N=20, ~5 minutes)

Edit line 37 in `Input_Uncertainty_Propagation.py`:
```python
QUICK_TEST = True
```

### Full Run (N=2000, ~60-120 minutes)

```python
QUICK_TEST = False
```

### Command

```bash
cd hmg/Assignment4
python Input_Uncertainty_Propagation.py
```

## Interpretation Tips

### NSE Distribution Plots

- **Histogram spread** shows sensitivity to input uncertainty
- **CDF** shows probability of achieving different NSE levels
- Compare blue (ref params) vs green (recalibrated) distributions

### Parameter Boxplots

- **Red dashed line** = reference value from Assignment 1
- **Box width** = parameter sensitivity to precipitation uncertainty
- Narrow boxes = robust parameters, wide boxes = sensitive parameters

### Discharge Uncertainty Bands

- **Gray shading** = 5-95 percentile range
- **Narrow bands** = low uncertainty propagation
- **Wide bands** = high uncertainty (typically during peak flows)

## Connection to Previous Assignments

| Assignment | Focus | Relationship |
|------------|-------|--------------|
| A1 | Parameter calibration | Provides reference parameters used here |
| A2 | Local sensitivity | OAT perturbation of parameters |
| A3 | Global sensitivity (Sobol) | Parameter importance ranking |
| **A4** | **Input uncertainty** | **Precipitation → discharge uncertainty** |

## Mathematical Framework

### Nash-Sutcliffe Efficiency (NSE)

$$NSE = 1 - \frac{\sum_{t=1}^{T}(Q_{obs,t} - Q_{sim,t})^2}{\sum_{t=1}^{T}(Q_{obs,t} - \bar{Q}_{obs})^2}$$

### Monte Carlo Uncertainty Bounds

For any output variable $Y$, after $N$ simulations:
- Mean: $\bar{Y} = \frac{1}{N}\sum_{i=1}^{N} Y_i$
- Standard deviation: $\sigma_Y = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(Y_i - \bar{Y})^2}$
- Confidence interval: $[\bar{Y} - 1.96\sigma_Y, \bar{Y} + 1.96\sigma_Y]$ (95% CI)

## Common Issues

1. **Long runtime**: Use `QUICK_TEST = True` for debugging
2. **Memory**: N=2000 iterations with full discharge arrays requires ~500MB RAM
3. **Random seed**: Set `RANDOM_SEED = 42` for reproducibility

## Presentation Talking Points

1. **Why input uncertainty matters**: "Real precipitation measurements have ±5-10% errors from gauge under-catch, spatial interpolation, etc."

2. **Key insight**: "Even perfect model structure cannot overcome input uncertainty without recalibration."

3. **Equifinality**: "Multiple parameter sets can achieve similar NSE when inputs are uncertain."

4. **Practical implication**: "Uncertainty bounds on flood predictions should account for rainfall measurement errors, not just model parameters."
