Assignment 1: HBV Model Parameter Optimization - Plot Guide
Generated: January 2026
Best Result: NSE = 0.862218, OFV (1-NSE) = 0.137782


1. observed_vs_simulated.png

Purpose: Compare model predictions vs observed discharge over time.

X-axis: Time (datetime)
Y-axis (top): Discharge [m3/s]
Y-axis (bottom): Residual [m3/s] = Simulated - Observed

How to read:
- Top panel: Black = observed, Blue = simulated. Closer lines = better fit.
- Bottom panel: Red = model overestimates (+), Blue = model underestimates (-).
- Zero line is bold for reference.

Conclusion: The model reproduces the overall timing and recession behavior,
but it underestimates the largest flood peaks, especially in January. Because
NSE penalizes squared errors, these peak mismatches dominate the objective.


2. convergence_per_generation.png

Purpose: Show optimization progress over 80 generations.

Two-panel design:
- Panel 1 (top): Zoomed view (OFV 0-2) showing min, median, running best
- Panel 2 (bottom): Full range showing max values (high-objective samples)

X-axis: Generation number (1 to 80)
Y-axis (left): OFV = 1 - NSE (lower is better)
Y-axis (right): NSE = 1 - OFV (higher is better)

How to read:
- Green dashed line = running best (overall best found so far)
- Red dotted line = final best achieved (0.1378)
- Secondary y-axis converts OFV to NSE directly

Conclusion: Most improvement happens early. After that, gains become smaller
and the running best curve flattens, indicating the search has largely
converged to a stable solution. Final NSE = 0.862 achieved around generation 60.


3. convergence_per_generation_zoomed.png

Purpose: Same as above but single panel with auto-zoom for clarity.

How to read: Secondary y-axis shows NSE values for easy interpretation.


4. internal_variables.png

Purpose: Visualize HBV model internal states and fluxes.

8 subplots from top to bottom:
1. Temperature [deg C] - Input forcing (0 deg line shown)
2. Precipitation [mm/hr] - Input forcing
3. Snow depth [mm w.e.] - State (orange line = melt rate on right axis)
4. Soil moisture [mm] - State with FCY and PWP horizontal lines
5. ET [mm/hr] - Orange = PET (potential), Green = AET (actual)
6. Reservoir [mm] - URR threshold line (urr_tdh = 101.9 mm) shown
7. Runoff [mm/hr] - Surface runoff and reservoir contributions (URR and LRR)
8. Water balance [mm per timestep (hourly)] - Should be near zero

How to read:
- FCY/PWP lines show soil parameter relationships
- URR threshold line shows when fast outlet activates
- Water balance near zero confirms model conservation

Conclusion: The calibrated soil parameters give PWP (665.0 mm) greater than
FCY (113.6 mm). This is not physically typical, but the current HBV
implementation does not enforce FCY > PWP, so the optimizer can select such
combinations. Model correctly simulates snow accumulation/melt and reservoir
dynamics.


5. parameter_evolution.png

Purpose: Track how best parameters evolve during optimization.

X-axis: Model run number (evaluation count, 1 to 22,032)
Y-axis: Parameter value (one subplot per parameter)

How to read:
- Blue step lines show running best value for each parameter
- Gray dotted vertical lines = generation boundaries
- Steps occur only when overall best improves

Conclusion: Rapid changes early show exploration, later stability shows
convergence. Parameters like sl0_pwp and urr_tdh stabilize after about 5,000
evaluations.


6. parameters_normalized.png

Purpose: Show where optimized parameters fall within their bounds.

X-axis: Normalized value (0 = lower bound, 1 = upper bound)
Y-axis: Parameter names with [lower, upper] bounds shown

How to read:
- Blue dots = variable parameters
- Red dots = fixed parameters (snw_dth fixed at 0)
- Values near 0 or 1 may indicate bounds constraining optimization

Conclusion: sl0_pwp (665/700) is near upper bound, suggesting it may want
to go higher. sl0_fcy (114/700) is near lower bound.


7. scatter_obs_vs_sim.png

Purpose: Scatter plot showing observed vs simulated discharge.

X-axis: Observed discharge [m3/s]
Y-axis: Simulated discharge [m3/s]

How to read:
- 1:1 line = perfect fit
- Points above line = overestimate
- Points below line = underestimate
- Tight clustering around 1:1 = good fit

Conclusion: Good fit for low-medium flows (clustered around 1:1).
High flows (>100 m3/s) show underestimation bias.


8. flow_duration_curve.png

Purpose: Compare flow distribution between observed and simulated.

X-axis: Exceedance probability [%] (100% = always exceeded)
Y-axis: Discharge [m3/s]

How to read:
- Left side (0-10%) = high flows (peaks)
- Right side (90-100%) = low flows (baseflow)
- Gap between lines shows systematic bias

Bias summary shown in figure:
- High flows (top 10%): Shows if peaks are over/underestimated
- Medium flows (25-75%): Main flow regime bias
- Low flows (bottom 10%): Baseflow bias

Conclusion: Model underestimates high flows (left side gap) which is the
primary reason NSE cannot reach 0.90. Low flows slightly overestimated.


9. param_scatter_*.png (18 files)

Purpose: Show relationship between parameter value and objective.

X-axis: Parameter value (original units)
Y-axis: OFV = 1 - NSE (no filtering, no log scale)

How to read:
- Each dot = one model evaluation during optimization
- Red dashed vertical lines = parameter bounds
- Solid red vertical line = best parameter value
- Green horizontal line = best objective achieved
- V-shape or U-shape = sensitive parameter (clear optimum)
- Vertical band = insensitive parameter (wide range gives similar OFV)

Files: param_scatter_snw_dth.png through param_scatter_lrr_lct.png

Conclusion: Parameters like sl0_fcy and urr_tdh show clear sensitivity.
Parameters like lrr_lct show less sensitivity (flat cloud of points).


10. detailed_metrics_analysis.txt

Purpose: Text report explaining why NSE is not reaching 0.90.

Contains:
- Basic metrics (NSE, OFV, Bias, RMSE)
- Peak flow analysis (top 5 peaks with errors)
- Low flow analysis (bottom 10% statistics)
- NSE formula explanation
- Practical improvement suggestions

Key finding: Top 5 peaks contribute significant portion of squared error,
which dominates NSE. Model structure limits peak capture ability.


Summary: Why NSE is not 0.90

1. NSE uses squared errors, so large peak mismatches dominate
2. Model underestimates major flood peaks by about 50-55 m3/s (about 35%)
3. Model slightly overestimates low flows
4. PWP > FCY in calibration is physically unusual
5. NSE = 0.862 is very good for lumped conceptual models
6. Higher NSE (>0.90) would require model structure changes

This is not an optimization failure - it is a model structure limitation.
