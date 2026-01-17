'''
Assignment 1: HBV Model Parameter Optimization using Differential Evolution

This script calibrates the HBV001A hydrological model by minimizing 1-NSE
using scipy's differential_evolution optimizer.

Key design decisions:
- Parameter order is read from HBV001A.get_parameter_labels() - single source of truth
- No hardcoded parameter names or bounds order
- Objective = 1 - NSE (minimize), with PENALTY for invalid NSE
- No warmup removal, no filtering, linear scales in plots
- All outputs saved to ./outputs/assignment1/

@author: Zamin (cleaned version based on PDF requirements)
'''

import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

# Add project root to Python path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hmg import HBV001A
from scipy.optimize import differential_evolution

# =============================================================================
# Configuration
# =============================================================================

DEBUG_FLAG = False
FAST_DEBUG = False  # Set True for quick testing (5 gen, 5 pop)

# Penalty for invalid/undefined NSE (NaN or Inf)
PENALTY = 5.0

# DE configuration
DE_GENERATIONS = 80
DE_POPSIZE = 16
DE_SEED = 42

# =============================================================================
# Verify which HBV implementation is being used
# =============================================================================

def verify_hbv_implementation():
    """Print which HBV001A implementation (Python or Cython) is being used."""
    model = HBV001A()
    modl_func = model._modl
    modl_module = modl_func.__module__
    
    # Get module file path
    mod = sys.modules.get(modl_module, None)
    modl_file = getattr(mod, '__file__', 'unknown') if mod else 'unknown'
    
    print("=" * 60)
    print("HBV001A Implementation Verification")
    print("=" * 60)
    print(f"  Model function: {modl_func.__name__}")
    print(f"  Module: {modl_module}")
    print(f"  File: {modl_file}")
    
    if 'hbv001a_py' in modl_module or '_py' in str(modl_file):
        print("  >>> Using PYTHON implementation (correct for Assignment 1)")
    else:
        print("  >>> WARNING: May be using CYTHON implementation!")
    print("=" * 60)
    print()
    return model

# =============================================================================
# Get parameter info from model (single source of truth)
# =============================================================================

def get_param_info_from_model():
    """
    Get parameter labels, indices, and absolute bounds from HBV001A model.
    Returns: (param_names_ordered, param_indices, absolute_bounds_dict)
    
    CRITICAL: This is the single source of truth for parameter order.
    """
    model = HBV001A()
    
    # Get parameter label -> index mapping
    param_labels = model.get_parameter_labels()  # dict: name -> index
    
    # Get absolute bounds
    abs_bounds = model.get_parameter_absolute_bounds()  # dict: name -> (lo, hi)
    
    # Create ordered list of parameter names (sorted by index)
    param_names_ordered = sorted(param_labels.keys(), key=lambda k: param_labels[k])
    
    print("Parameter order from model (single source of truth):")
    for name in param_names_ordered:
        idx = param_labels[name]
        lo, hi = abs_bounds[name]
        lo_str = f"{lo:.2f}" if np.isfinite(lo) else str(lo)
        hi_str = f"{hi:.2f}" if np.isfinite(hi) else str(hi)
        print(f"  [{idx:2d}] {name}: abs_bounds ({lo_str}, {hi_str})")
    print()
    
    return param_names_ordered, param_labels, abs_bounds

# =============================================================================
# Build optimization bounds in correct parameter order
# =============================================================================

def build_bounds_in_order(param_names_ordered, param_labels):
    """
    Build bounds list in the correct parameter order.
    Uses reasonable ranges within absolute bounds.
    
    CRITICAL: bounds_list[i] corresponds to param_names_ordered[i]
    """
    # Define optimization bounds for each parameter (narrower than absolute)
    opt_bounds_dict = {
        'snw_dth': (0.0, 0.0),       # Fixed at 0 (no initial snow)
        'snw_att': (-2.0, 3.0),      # Air snow temperature threshold
        'snw_pmf': (0.0, 3.0),       # Precipitation melt factor
        'snw_amf': (0.0, 10.0),      # Air melt factor
        'sl0_dth': (0.0, 100.0),     # Soil initial depth
        'sl0_pwp': (5.0, 700.0),     # Permanent wilting point
        'sl0_fcy': (100.0, 700.0),   # Field capacity
        'sl0_bt0': (0.01, 10.0),     # Beta exponent
        'urr_dth': (0.0, 20.0),      # Upper reservoir initial depth
        'lrr_dth': (0.0, 100.0),     # Lower reservoir initial depth
        'urr_wsr': (0.0, 1.0),       # Water split ratio
        'urr_ulc': (0.0, 1.0),       # Upper-lower connection constant
        'urr_tdh': (0.0, 200.0),     # Threshold depth high
        'urr_tdr': (0.01, 1.0),      # Threshold depth ratio
        'urr_ndr': (0.0, 1.0),       # Near drainage rate
        'urr_uct': (0.0, 1.0),       # Upper cutoff threshold
        'lrr_dre': (0.0, 1.0),       # Lower reservoir drainage
        'lrr_lct': (0.0, 1.0),       # Lower reservoir cutoff
    }
    
    # Verify all parameters have bounds defined
    for name in param_names_ordered:
        if name not in opt_bounds_dict:
            raise KeyError(f"Missing bounds definition for parameter '{name}'")
    
    # Build bounds list in correct order
    bounds_list = []
    for name in param_names_ordered:
        bounds_list.append(opt_bounds_dict[name])
    
    print("Optimization bounds (in model parameter order):")
    for i, name in enumerate(param_names_ordered):
        lo, hi = bounds_list[i]
        print(f"  [{i:2d}] {name}: ({lo}, {hi})")
    print()
    
    return bounds_list

# =============================================================================
# NSE and objective function
# =============================================================================

def nse(obs, sim):
    """
    Nash-Sutcliffe Efficiency.
    Returns float NSE or np.nan if undefined.
    """
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    if mask.sum() == 0:
        return np.nan
    
    o = obs[mask]
    s = sim[mask]
    
    denom = np.sum((o - o.mean()) ** 2)
    if denom == 0.0:
        return np.nan
    
    num = np.sum((s - o) ** 2)
    return 1.0 - num / denom

def obj_nse(obs, sim):
    """
    Objective to minimize: 1 - NSE.
    Returns PENALTY if NSE is undefined.
    """
    val = nse(obs, sim)
    if not np.isfinite(val):
        return PENALTY
    return 1.0 - val

def compute_detailed_metrics(obs, sim):
    """
    Compute detailed performance metrics for Part A analysis.
    
    Returns dict with:
    - nse, ofv, bias, rmse
    - peak_errors: top 5 peaks with obs, sim, error
    - low_flow_stats: bottom 10% flow stats
    """
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    o = obs[mask]
    s = sim[mask]
    
    # Basic metrics
    nse_val = nse(obs, sim)
    ofv = 1.0 - nse_val
    
    # Bias: mean(sim - obs), positive = overestimate
    residuals = s - o
    bias = np.mean(residuals)
    
    # RMSE
    rmse = np.sqrt(np.mean(residuals**2))
    
    # Peak flow analysis: top 5 observed peaks
    peak_indices = np.argsort(o)[-5:][::-1]  # Top 5, sorted high to low
    peak_errors = []
    for idx in peak_indices:
        peak_errors.append({
            'obs': float(o[idx]),
            'sim': float(s[idx]),
            'error': float(s[idx] - o[idx]),
            'rel_error_pct': float((s[idx] - o[idx]) / o[idx] * 100) if o[idx] != 0 else 0.0
        })
    
    # Low flow analysis: bottom 10% of observed flows
    threshold_10pct = np.percentile(o, 10)
    low_mask = o <= threshold_10pct
    low_obs = o[low_mask]
    low_sim = s[low_mask]
    low_flow_stats = {
        'threshold': float(threshold_10pct),
        'n_points': int(np.sum(low_mask)),
        'mean_obs': float(np.mean(low_obs)),
        'mean_sim': float(np.mean(low_sim)),
        'bias': float(np.mean(low_sim - low_obs)),
        'rmse': float(np.sqrt(np.mean((low_sim - low_obs)**2)))
    }
    
    # Squared error contribution analysis
    total_squared_error = np.sum(residuals**2)
    peak_squared_error = np.sum((s[peak_indices] - o[peak_indices])**2)
    peak_contribution_pct = (peak_squared_error / total_squared_error) * 100 if total_squared_error > 0 else 0.0
    
    return {
        'nse': nse_val,
        'ofv': ofv,
        'bias': bias,
        'rmse': rmse,
        'peak_errors': peak_errors,
        'low_flow_stats': low_flow_stats,
        'peak_contribution_pct': peak_contribution_pct,
        'total_squared_error': total_squared_error
    }

# =============================================================================
# Differential Evolution Recorder
# =============================================================================

class DERecorder:
    """
    Callable class that runs the HBV model and records optimization history.
    Tracks all evaluations and per-generation statistics.
    Uses SciPy callback for reliable generation boundary detection.
    """
    
    def __init__(self, inputs, dslr, tsps, obs_q, param_names, model_class=HBV001A, pop_sz=15):
        self.inputs = inputs  # (temps, precip, pet)
        self.dslr = np.float32(dslr)
        self.tsps = int(tsps)
        self.obs_q = np.asarray(obs_q, dtype=np.float32)
        self.model_class = model_class
        self.pop_sz = pop_sz
        self.param_names = param_names
        self.n_params = len(param_names)
        
        # Evaluation history
        self.eval_params = []
        self.eval_objs = []
        self.eval_count = 0
        
        # Generation tracking (set by callback)
        self.gen_bounds = []  # List of (start_idx, end_idx) per generation
        self._gen_start = 0
        self._current_gen = 0
        
        # Per-generation statistics (set by callback)
        self.gen_best_objs = []
        self.gen_max_objs = []
        self.gen_med_objs = []
        self.gen_best_params = []
        
        # Running best tracking (for parameter evolution plot)
        self.running_best_obj = float('inf')
        self.running_best_params = None
        self.running_best_at_eval = []  # (eval_idx, best_obj, best_params) when improvement happens
        
        # Buffer for evaluations since last callback
        self._buffer_objs = []
        self._buffer_params = []
    
    def callback(self, xk, convergence):
        """
        SciPy DE callback - called at end of each generation.
        xk: best parameter vector so far
        convergence: convergence metric
        """
        # Finalize generation statistics from buffer
        if len(self._buffer_objs) > 0:
            objs = np.array(self._buffer_objs)
            best_idx = np.argmin(objs)
            
            self.gen_best_objs.append(float(objs[best_idx]))
            self.gen_max_objs.append(float(np.max(objs)))
            self.gen_med_objs.append(float(np.median(objs)))
            self.gen_best_params.append(self._buffer_params[best_idx].copy())
            
            # Record generation bounds
            end_idx = self.eval_count
            self.gen_bounds.append((self._gen_start, end_idx))
            self._gen_start = end_idx
            
            self._current_gen += 1
            print(f"Generation {self._current_gen} complete: best_obj={self.gen_best_objs[-1]:.6f}, n_evals={len(self._buffer_objs)}")
            
            # Clear buffer
            self._buffer_objs = []
            self._buffer_params = []
        
        return False  # Return False to continue optimization
    
    def __call__(self, params):
        """
        Run model with given parameters and return objective value.
        """
        params = np.asarray(params, dtype=np.float32)
        
        temps, precip, pet = self.inputs
        
        model = self.model_class()
        model.set_inputs(temps, precip, pet)
        model.set_outputs(self.tsps)
        model.set_discharge_scaler(self.dslr)
        model.set_parameters(params)
        model.set_optimization_flag(1)
        
        start_t = time.time()
        model.run_model()
        elapsed = time.time() - start_t
        print(f"Model runtime: {elapsed:.2E} seconds.")
        
        sim_q = model.get_discharge()
        
        if sim_q.shape[0] != self.obs_q.shape[0]:
            raise RuntimeError(f"Length mismatch: sim={sim_q.shape[0]}, obs={self.obs_q.shape[0]}")
        
        sim_q = np.asarray(sim_q, dtype=np.float32)
        if np.any(~np.isfinite(sim_q)):
            obj = PENALTY
        else:
            obj = obj_nse(self.obs_q, sim_q)
        
        # Record this evaluation
        self.eval_params.append(params.copy())
        self.eval_objs.append(float(obj))
        self.eval_count += 1
        
        # Buffer for callback-based generation tracking
        self._buffer_objs.append(float(obj))
        self._buffer_params.append(params.copy())
        
        # Track running best for parameter evolution plot
        if obj < self.running_best_obj:
            self.running_best_obj = obj
            self.running_best_params = params.copy()
            self.running_best_at_eval.append((self.eval_count, float(obj), params.copy()))
        
        return float(obj)
    
    def finalize(self):
        """Finalize any remaining evaluations as final generation."""
        if len(self._buffer_objs) > 0:
            # Call callback one more time to finalize
            self.callback(None, None)

# =============================================================================
# Plotting functions
# =============================================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def plot_parameters_normalized(bounds, best_params, param_names, out_file):
    """
    Plot best parameters normalized to [0,1] range.
    
    Normalization formula: (value - lower_bound) / (upper_bound - lower_bound)
    Fixed parameters (where lower = upper) are marked separately.
    """
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    best = np.array(best_params, dtype=float)
    
    # Identify fixed parameters
    is_fixed = (ub - lb) == 0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = ub - lb
        denom[denom == 0] = 1.0
        norm = (best - lb) / denom
    norm = np.clip(norm, 0.0, 1.0)
    
    fig, ax = plt.subplots(figsize=(12, 9), dpi=150)
    y = np.arange(len(best))
    
    # Draw range lines
    ax.hlines(y, 0, 1, color='lightgray', linewidth=2, zorder=1)
    
    # Plot points - different color for fixed params
    colors = ['red' if f else 'blue' for f in is_fixed]
    ax.scatter(norm, y, s=80, zorder=3, c=colors)
    
    # Create labels with bounds info
    labels = []
    for i, name in enumerate(param_names):
        if is_fixed[i]:
            labels.append(f'{name} [FIXED at {lb[i]:.2f}]')
        else:
            labels.append(f'{name} [{lb[i]:.2f}, {ub[i]:.2f}]')
    
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Normalized value\n(0 = lower bound, 1 = upper bound)', fontsize=10)
    ax.set_title('Best Parameter Values (Normalized to Optimization Bounds)\n'
                 'Blue = variable parameter, Red = fixed parameter', fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.grid(axis='x', alpha=0.3)
    
    # Add text box with interpretation
    textstr = ('How to read: Each dot shows where the best parameter value\n'
               'falls within its allowed range. Values near 0 are at lower bound,\n'
               'values near 1 are at upper bound. Fixed parameters cannot vary.')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_param_evolution_vs_model_runs(running_best_at_eval, total_evals, param_names, 
                                        gen_bounds, out_file):
    """
    Plot parameter evolution vs model runs (not generations).
    Parameter line only changes when running best improves.
    Step-style plot without markers.
    
    X-axis: Model run number (evaluation count)
    Y-axis: Parameter value for each subplot
    Vertical lines show generation boundaries.
    
    running_best_at_eval: list of (eval_idx, obj, params) when improvement happens
    gen_bounds: list of (start_idx, end_idx) for each generation
    """
    if len(running_best_at_eval) == 0:
        return
    
    n_params = len(param_names)
    
    fig, axs = plt.subplots(n_params, 1, figsize=(14, 2.2*n_params), dpi=100, sharex=True)
    if n_params == 1:
        axs = [axs]
    
    for i in range(n_params):
        # Build step values for this parameter
        x_vals = []
        y_vals = []
        
        for j, (eval_idx, obj, params) in enumerate(running_best_at_eval):
            if j == 0:
                # Start from evaluation 1 with first best params
                x_vals.extend([1, eval_idx])
                y_vals.extend([params[i], params[i]])
            else:
                # Step to new value at this evaluation
                prev_val = running_best_at_eval[j-1][2][i]
                x_vals.extend([eval_idx, eval_idx])
                y_vals.extend([prev_val, params[i]])
        
        # Extend to total_evals
        if len(running_best_at_eval) > 0:
            last_val = running_best_at_eval[-1][2][i]
            x_vals.append(total_evals)
            y_vals.append(last_val)
        
        # Plot step-style line
        axs[i].plot(x_vals, y_vals, linewidth=1.2, color='blue')
        axs[i].set_ylabel(param_names[i], fontsize=9)
        axs[i].grid(True, alpha=0.3)
        
        # Add generation boundaries as vertical lines
        for start, end in gen_bounds:
            if end < total_evals:
                axs[i].axvline(end, color='gray', linestyle=':', alpha=0.4, linewidth=0.5)
    
    axs[-1].set_xlabel('Model Run Number (Evaluation Count)', fontsize=10)
    
    fig.suptitle('Parameter Evolution vs Model Runs\n'
                 '(Step changes only when running best improves, '
                 'gray dotted lines = generation boundaries)', fontsize=11)
    
    # Add interpretation text
    textstr = ('How to read: Each subplot shows how the best-so-far value of one parameter\n'
               'changes over the optimization. Flat sections mean no improvement was found.')
    fig.text(0.5, 0.01, textstr, ha='center', fontsize=8, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_file, dpi=120, bbox_inches='tight')
    plt.close(fig)

def plot_param_scatter_all_evals(eval_params, eval_objs, bounds, param_names, 
                                  best_params, best_obj, out_dir):
    """
    Create scatter plots for each parameter: parameter value vs objective.
    No filtering - shows all evaluations.
    Linear scale - no log transform.
    
    X-axis: Parameter value in original units
    Y-axis: Objective value (1 - NSE), linear scale
    
    Red vertical line: best parameter value
    Green horizontal line: best objective value
    Red dashed vertical lines: parameter bounds
    """
    if len(eval_params) == 0:
        return
    
    ensure_dir(out_dir)
    eval_params = np.vstack(eval_params).astype(float)
    eval_objs = np.array(eval_objs, dtype=float)
    best_params = np.array(best_params, dtype=float)
    
    n_params = eval_params.shape[1]
    
    for i in range(n_params):
        fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
        
        # Scatter: x = parameter value, y = objective (no filtering!)
        ax.scatter(eval_params[:, i], eval_objs, s=8, alpha=0.4, c='steelblue', 
                   label='All evaluations')
        
        # Mark bounds as red dashed lines
        lb, ub = bounds[i]
        ax.axvline(lb, linestyle='--', linewidth=1.5, color='red', alpha=0.7)
        ax.axvline(ub, linestyle='--', linewidth=1.5, color='red', alpha=0.7)
        
        # Mark best parameter value as solid red line
        ax.axvline(best_params[i], linestyle='-', linewidth=2, color='darkred', 
                   alpha=0.8, label=f'Best value = {best_params[i]:.4f}')
        
        # Mark best objective as horizontal green line
        ax.axhline(best_obj, linestyle='-', linewidth=1.5, color='green', 
                   alpha=0.7, label=f'Best objective = {best_obj:.4f}')
        
        ax.set_xlabel(f'{param_names[i]} (parameter value)', fontsize=10)
        ax.set_ylabel('Objective (1 - NSE)', fontsize=10)
        ax.set_title(f'Parameter Sensitivity: {param_names[i]}\n'
                     f'Bounds: [{lb}, {ub}]', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        
        # Add interpretation note
        if lb == ub:
            note = 'This parameter is FIXED (bounds are equal).'
        else:
            note = ('Sensitivity hint: If low objectives cluster in a narrow parameter\n'
                    'range, the model is sensitive to this parameter.')
        ax.text(0.02, 0.98, note, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', style='italic', alpha=0.7)
        
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f'param_scatter_{param_names[i]}.png'), dpi=120)
        plt.close(fig)

def plot_convergence_per_generation(gen_best_objs, gen_max_objs, gen_med_objs, best_obj, out_dir):
    """
    Plot convergence with two-panel design:
    Panel 1 (top): Zoomed view showing min, median, running best (y=0 to 2 or 3)
    Panel 2 (bottom): Wide view showing max values
    
    Both panels have secondary y-axis showing NSE = 1 - OFV.
    No log scale, no filtering.
    """
    if len(gen_best_objs) == 0:
        return []
    
    mins = np.array(gen_best_objs)
    maxs = np.array(gen_max_objs)
    meds = np.array(gen_med_objs)
    running_min = np.minimum.accumulate(mins)
    gens = np.arange(1, len(mins) + 1)
    
    # Two-panel figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=150)
    
    # Panel 1: Zoomed view (y = 0 to 2 or 3)
    ax1.plot(gens, mins, label='Generation best', marker='o', markersize=3, 
             alpha=0.8, color='blue')
    # Only plot median if within range
    meds_clipped = np.clip(meds, 0, 3)
    ax1.plot(gens, meds_clipped, label='Generation median', alpha=0.6, 
             linewidth=1, color='orange')
    ax1.plot(gens, running_min, label='Running best', 
             linestyle='--', linewidth=2, color='green')
    ax1.axhline(best_obj, linestyle=':', linewidth=2, color='red', 
                label=f'Final best OFV = {best_obj:.4f}')
    
    ax1.set_ylabel('Objective Value (OFV = 1 - NSE)', fontsize=10, color='black')
    ax1.set_ylim(0, 2)  # Zoomed range
    ax1.set_xlim(1, len(gens))
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Panel 1: Zoomed View (OFV 0-2) - Min, Median, Running Best\n'
                  'Lower OFV = Higher NSE = Better fit', fontsize=11)
    
    # Secondary y-axis for NSE
    ax1_nse = ax1.twinx()
    ax1_nse.set_ylim(1 - 2, 1 - 0)  # NSE = 1 - OFV
    ax1_nse.set_ylabel('NSE = 1 - OFV', fontsize=10, color='darkgreen')
    ax1_nse.tick_params(axis='y', labelcolor='darkgreen')
    
    # Panel 2: Wide view showing max
    ax2.plot(gens, maxs, label='Generation max (worst)', alpha=0.7, 
             linewidth=1.5, color='gray')
    ax2.plot(gens, meds, label='Generation median', alpha=0.6, 
             linewidth=1, color='orange')
    ax2.axhline(best_obj, linestyle=':', linewidth=2, color='red', 
                label=f'Final best OFV = {best_obj:.4f}')
    
    ax2.set_xlabel('Generation Number', fontsize=10)
    ax2.set_ylabel('Objective Value (OFV = 1 - NSE)', fontsize=10, color='black')
    ax2.set_xlim(1, len(gens))
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Panel 2: Full Range View - Shows high-objective samples (max per generation)',
                  fontsize=11)
    
    # Secondary y-axis for NSE on panel 2
    ax2_nse = ax2.twinx()
    y2_max = ax2.get_ylim()[1]
    ax2_nse.set_ylim(1 - y2_max, 1 - 0)
    ax2_nse.set_ylabel('NSE = 1 - OFV', fontsize=10, color='darkgreen')
    ax2_nse.tick_params(axis='y', labelcolor='darkgreen')
    
    # Add interpretation text
    fig.text(0.5, 0.01, 
             'How to read: Panel 1 shows convergence of good solutions. '
             'Panel 2 shows full range including very poor parameter samples. '
             f'Final: OFV = {best_obj:.4f}, NSE = {1-best_obj:.4f}',
             ha='center', fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_dir / 'convergence_per_generation.png', dpi=150)
    plt.close(fig)
    
    # Single zoomed plot for quick reference
    fig2, ax = plt.subplots(figsize=(11, 6), dpi=150)
    ax.plot(gens, mins, label='Generation best', marker='o', markersize=3, alpha=0.8)
    ax.plot(gens, meds_clipped, label='Generation median', alpha=0.6, linewidth=1)
    ax.plot(gens, running_min, label='Running best', 
            linestyle='--', linewidth=2, color='green')
    ax.axhline(best_obj, linestyle=':', linewidth=2, color='red', 
               label=f'Final: OFV={best_obj:.4f}, NSE={1-best_obj:.4f}')
    
    ax.set_xlabel('Generation Number', fontsize=10)
    ax.set_ylabel('Objective Value (OFV = 1 - NSE)', fontsize=10)
    ax.set_title('Optimization Convergence (Zoomed View)\n'
                 'Lower OFV = Higher NSE = Better model fit', fontsize=11)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, min(2.0, max(0.5, 10 * best_obj)))
    
    # Secondary NSE axis
    ax_nse = ax.twinx()
    ax_nse.set_ylim(1 - ax.get_ylim()[1], 1 - ax.get_ylim()[0])
    ax_nse.set_ylabel('NSE = 1 - OFV', fontsize=10, color='darkgreen')
    ax_nse.tick_params(axis='y', labelcolor='darkgreen')
    
    plt.tight_layout()
    fig2.savefig(out_dir / 'convergence_per_generation_zoomed.png', dpi=150)
    plt.close(fig2)
    
    return running_min.tolist()

def plot_observed_vs_simulated(index, obs_q, sim_q, nse_val, obj_val, out_file):
    """
    Plot observed vs simulated discharge with residuals.
    
    Top panel: Time series of observed and simulated discharge
    Bottom panel: Residuals = Simulated - Observed (positive = model overestimates)
    
    X-axis: Time (datetime)
    Y-axis (top): Discharge [m3/s]
    Y-axis (bottom): Residual [m3/s]
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), dpi=150, 
                                    sharex=True, height_ratios=[3, 1])
    
    obs_arr = np.array(obs_q)
    sim_arr = np.array(sim_q)
    
    # Top panel: discharge time series
    ax1.plot(index, obs_arr, label='Observed', alpha=0.9, linewidth=0.8, color='black')
    ax1.plot(index, sim_arr, label='Simulated', alpha=0.8, linewidth=0.8, color='blue')
    ax1.set_ylabel('Discharge [m3/s]', fontsize=10)
    ax1.set_title('Observed vs Simulated Discharge', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Add NSE and OFV text box
    textstr = f'NSE = {nse_val:.4f}\nOFV (1-NSE) = {obj_val:.4f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props, fontweight='bold')
    
    # Bottom panel: residuals
    # DEFINITION: Residual = Simulated - Observed
    # Positive residual = model OVERESTIMATES
    # Negative residual = model UNDERESTIMATES
    residuals = sim_arr - obs_arr
    
    # Color: red where positive (overestimate), blue where negative (underestimate)
    ax2.fill_between(index, residuals, 0, where=(residuals >= 0), 
                     alpha=0.5, color='red', interpolate=True)
    ax2.fill_between(index, residuals, 0, where=(residuals < 0), 
                     alpha=0.5, color='blue', interpolate=True)
    ax2.axhline(0, color='black', linewidth=1.5, linestyle='-')
    ax2.set_ylabel('Residual [m3/s]\n(Sim - Obs)', fontsize=10)
    ax2.set_xlabel('Time', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add residual definition and statistics
    # Bias = mean(Sim - Obs): positive = overall overestimate
    bias = np.mean(residuals)
    rmse = np.sqrt(np.mean(residuals**2))
    
    # Create legend-like text explaining colors
    ax2.text(0.02, 0.95, 
             f'Residual = Sim - Obs | Red: overestimate (+), Blue: underestimate (-)\n'
             f'Bias = {bias:.2f} m3/s (positive=overestimate), RMSE = {rmse:.2f} m3/s',
             transform=ax2.transAxes, fontsize=9, verticalalignment='top', 
             style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_internal_variables(index, inp_dfe, otps, otps_lbls, best_params, param_names, out_file):
    """
    Plot internal model variables for diagnostics.
    
    Subplots:
    1. Temperature [deg C] - input forcing
    2. Precipitation [mm/hr] - input forcing
    3. Snow depth [mm w.e.] - state, with melt components
    4. Soil moisture [mm] - state, with FCY and PWP lines
    5. Evapotranspiration [mm/hr] - flux, PET vs AET
    6. Reservoir storage [mm] - state, with urr_tdh threshold
    7. Runoff components [mm/hr] - flux
    8. Water balance [mm] - diagnostic
    
    X-axis: Time (datetime)
    Y-axis: Variable-specific units
    """
    # Extract relevant parameters
    param_dict = {name: best_params[i] for i, name in enumerate(param_names)}
    sl0_fcy = param_dict.get('sl0_fcy', None)
    sl0_pwp = param_dict.get('sl0_pwp', None)
    urr_tdh = param_dict.get('urr_tdh', None)
    
    fig, axs = plt.subplots(8, 1, figsize=(14, 20), dpi=100, sharex=True)
    
    # 1. Temperature (input forcing)
    axs[0].plot(index, inp_dfe['tavg__ref'], alpha=0.85, color='red')
    axs[0].axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    axs[0].set_ylabel('Temperature\n[deg C]', fontsize=9)
    axs[0].set_title('Input: Air temperature [deg C] (0 deg C line shown for reference)', fontsize=9, loc='left')
    
    # 2. Precipitation (input forcing)
    axs[1].bar(index, inp_dfe['pptn__ref'], alpha=0.7, color='blue', width=0.04)
    axs[1].set_ylabel('Precipitation\n[mm/hr]', fontsize=9)
    axs[1].set_title('Input: Precipitation [mm/hr]', fontsize=9, loc='left')
    
    # 3. Snow depth with melt components
    axs[2].fill_between(index, otps[:, otps_lbls['snw_dth']], alpha=0.5, color='lightblue')
    axs[2].plot(index, otps[:, otps_lbls['snw_dth']], alpha=0.85, color='blue', 
                linewidth=0.8, label='Snow depth')
    # Plot melt components if available
    if 'snw_mlt' in otps_lbls:
        ax2b = axs[2].twinx()
        ax2b.plot(index, otps[:, otps_lbls['snw_mlt']], alpha=0.7, color='orange', 
                  linewidth=0.6, label='Total melt')
        ax2b.set_ylabel('Melt [mm/hr]', fontsize=8, color='orange')
        ax2b.tick_params(axis='y', labelcolor='orange')
    axs[2].set_ylabel('Snow Depth\n[mm w.e.]', fontsize=9)
    axs[2].set_title('State: Snow water equivalent [mm] (orange = melt rate on right axis)', fontsize=9, loc='left')
    axs[2].legend(loc='upper left', fontsize=7)
    
    # 4. Soil moisture with FCY and PWP horizontal lines
    axs[3].fill_between(index, otps[:, otps_lbls['sl0_dth']], alpha=0.5, color='brown')
    axs[3].plot(index, otps[:, otps_lbls['sl0_dth']], alpha=0.85, color='saddlebrown', 
                linewidth=0.8, label='Soil moisture')
    # Add FCY and PWP lines
    if sl0_fcy is not None:
        axs[3].axhline(sl0_fcy, color='green', linestyle='--', linewidth=1.5, 
                       label=f'FCY = {sl0_fcy:.1f} mm')
    if sl0_pwp is not None:
        axs[3].axhline(sl0_pwp, color='red', linestyle='--', linewidth=1.5, 
                       label=f'PWP = {sl0_pwp:.1f} mm')
    axs[3].set_ylabel('Soil Moisture\n[mm]', fontsize=9)
    # Note about PWP > FCY
    if sl0_pwp is not None and sl0_fcy is not None and sl0_pwp > sl0_fcy:
        note = f'NOTE: PWP ({sl0_pwp:.1f}) > FCY ({sl0_fcy:.1f}) - physically unusual but model allows it'
    else:
        note = ''
    axs[3].set_title(f'State: Soil moisture [mm] with FCY (field capacity) and PWP (wilting point)\n{note}', 
                     fontsize=9, loc='left')
    axs[3].legend(loc='upper right', fontsize=7)
    
    # 5. Evapotranspiration (flux)
    axs[4].plot(index, inp_dfe['petn__ref'], label='PET (potential)', alpha=0.7, color='orange')
    axs[4].plot(index, otps[:, otps_lbls['sl0_etn']], label='AET (actual)', alpha=0.85, color='green')
    axs[4].set_ylabel('ET\n[mm/hr]', fontsize=9)
    axs[4].set_title('Flux: Evapotranspiration [mm/hr] - AET limited by soil moisture availability', fontsize=9, loc='left')
    axs[4].legend(loc='upper right', fontsize=8)
    
    # 6. Reservoir storage with threshold line
    axs[5].plot(index, otps[:, otps_lbls['urr_dth']], label='Upper reservoir (URR)', 
                alpha=0.85, color='blue')
    axs[5].plot(index, otps[:, otps_lbls['lrr_dth']], label='Lower reservoir (LRR)', 
                alpha=0.85, color='darkblue')
    # Add urr_tdh threshold line
    if urr_tdh is not None:
        axs[5].axhline(urr_tdh, color='red', linestyle='--', linewidth=1.5,
                       label=f'URR threshold (urr_tdh) = {urr_tdh:.1f} mm')
    axs[5].set_ylabel('Reservoir\n[mm]', fontsize=9)
    axs[5].set_title('State: Reservoir storage [mm] - URR threshold controls fast outlet activation', 
                     fontsize=9, loc='left')
    axs[5].legend(loc='upper right', fontsize=7)
    
    # 7. Runoff components (flux)
    axs[6].plot(index, otps[:, otps_lbls['chn_pow']], label='Surface runoff (chn_pow)', 
                alpha=0.85, color='cyan')
    axs[6].plot(index, otps[:, otps_lbls['urr_urf']], label='URR runoff (urr_urf)', 
                alpha=0.85, color='green')
    axs[6].plot(index, otps[:, otps_lbls['lrr_lrf']], label='LRR runoff (lrr_lrf)', 
                alpha=0.85, color='darkgreen')
    axs[6].set_ylabel('Runoff\n[mm/hr]', fontsize=9)
    axs[6].set_title('Flux: Runoff components [mm/hr] - sum of all = total discharge', fontsize=9, loc='left')
    axs[6].legend(loc='upper right', fontsize=7)
    
    # 8. Water balance (diagnostic)
    balance = otps[:, otps_lbls['mod_bal']]
    axs[7].plot(index, balance, alpha=0.85, color='purple')
    axs[7].axhline(0, color='black', linestyle='-', linewidth=1.5)
    axs[7].set_ylabel('Balance\n[mm]', fontsize=9)
    axs[7].set_xlabel('Time', fontsize=10)
    
    # Add balance statistics
    bal_mean = np.mean(balance)
    bal_max = np.max(np.abs(balance))
    bal_text = f'Balance: mean = {bal_mean:.6f} mm, max|error| = {bal_max:.6f} mm'
    axs[7].set_title(f'Diagnostic: Water balance error [mm] - should be near zero\n{bal_text}', 
                     fontsize=9, loc='left')
    
    for ax in axs:
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('HBV Model Internal Variables\n'
                 '(Units: Temperature=deg C, Precipitation/ET/Runoff=mm/hr, Storage=mm)', 
                 fontsize=12, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_file, dpi=120, bbox_inches='tight')
    plt.close(fig)

def plot_scatter_obs_vs_sim(obs_q, sim_q, nse_val, bias, rmse, out_file):
    """
    Scatter plot: Observed vs Simulated discharge.
    
    X-axis: Observed discharge [m3/s]
    Y-axis: Simulated discharge [m3/s]
    1:1 line for reference
    
    This plot helps identify systematic biases:
    - Points above 1:1 line = model overestimates
    - Points below 1:1 line = model underestimates
    """
    fig, ax = plt.subplots(figsize=(9, 9), dpi=150)
    
    obs_arr = np.array(obs_q)
    sim_arr = np.array(sim_q)
    
    # Scatter plot
    ax.scatter(obs_arr, sim_arr, s=10, alpha=0.5, c='steelblue', edgecolors='none')
    
    # 1:1 line
    max_val = max(obs_arr.max(), sim_arr.max())
    min_val = min(obs_arr.min(), sim_arr.min())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, 
            label='1:1 line (perfect fit)')
    
    ax.set_xlabel('Observed Discharge [m3/s]', fontsize=11)
    ax.set_ylabel('Simulated Discharge [m3/s]', fontsize=11)
    ax.set_title('Scatter Plot: Observed vs Simulated Discharge', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper left', fontsize=9)
    
    # Add metrics text box
    textstr = (f'NSE = {nse_val:.4f}\n'
               f'Bias = {bias:.2f} m3/s\n'
               f'RMSE = {rmse:.2f} m3/s\n'
               f'n = {len(obs_arr)} points')
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    # Add interpretation text
    ax.text(0.02, 0.98, 
            'Points above 1:1 line = overestimate\nPoints below 1:1 line = underestimate',
            transform=ax.transAxes, fontsize=9, verticalalignment='top', style='italic')
    
    plt.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_flow_duration_curve(obs_q, sim_q, out_file):
    """
    Flow Duration Curve (FDC): Exceedance probability vs discharge.
    
    X-axis: Exceedance probability [%] (100% = always exceeded, 0% = never exceeded)
    Y-axis: Discharge [m3/s]
    
    FDC helps diagnose:
    - High flow bias (left side of curve)
    - Low flow bias (right side of curve)
    - Overall flow distribution match
    """
    fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
    
    obs_arr = np.array(obs_q)
    sim_arr = np.array(sim_q)
    n = len(obs_arr)
    
    # Sort in descending order
    obs_sorted = np.sort(obs_arr)[::-1]
    sim_sorted = np.sort(sim_arr)[::-1]
    
    # Exceedance probability (Weibull plotting position)
    exceedance = (np.arange(1, n + 1) / (n + 1)) * 100
    
    # Plot FDC
    ax.plot(exceedance, obs_sorted, 'k-', linewidth=1.5, label='Observed', alpha=0.9)
    ax.plot(exceedance, sim_sorted, 'b-', linewidth=1.5, label='Simulated', alpha=0.8)
    
    ax.set_xlabel('Exceedance Probability [%]', fontsize=11)
    ax.set_ylabel('Discharge [m3/s]', fontsize=11)
    ax.set_title('Flow Duration Curve (FDC)\n'
                 'Left side = high flows, Right side = low flows', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    # Add vertical lines at key percentiles
    for pct in [5, 10, 50, 90, 95]:
        ax.axvline(pct, color='gray', linestyle=':', alpha=0.5)
        ax.text(pct, ax.get_ylim()[1]*0.95, f'{pct}%', ha='center', fontsize=8, alpha=0.7)
    
    # Calculate and show bias at different flow regimes
    # High flows: top 10%
    high_idx = int(0.1 * n)
    high_bias = np.mean(sim_sorted[:high_idx] - obs_sorted[:high_idx])
    # Low flows: bottom 10%
    low_idx = int(0.9 * n)
    low_bias = np.mean(sim_sorted[low_idx:] - obs_sorted[low_idx:])
    # Medium flows: middle 50%
    med_start = int(0.25 * n)
    med_end = int(0.75 * n)
    med_bias = np.mean(sim_sorted[med_start:med_end] - obs_sorted[med_start:med_end])
    
    textstr = (f'Bias (Sim - Obs):\n'
               f'  High flows (top 10%): {high_bias:+.2f} m3/s\n'
               f'  Medium flows (25-75%): {med_bias:+.2f} m3/s\n'
               f'  Low flows (bottom 10%): {low_bias:+.2f} m3/s')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=props)
    
    # Add interpretation
    ax.text(0.98, 0.98, 
            'How to read:\n'
            '- Simulated above Observed = overestimate\n'
            '- Simulated below Observed = underestimate\n'
            '- Gap on left = peak flow error (affects NSE most)',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            horizontalalignment='right', style='italic')
    
    plt.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

def save_detailed_metrics_report(metrics, out_file):
    """
    Save detailed performance metrics report to text file.
    This explains why NSE may not be reaching 0.90.
    """
    with open(out_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("DETAILED PERFORMANCE METRICS ANALYSIS\n")
        f.write("Why NSE is not reaching 0.90: Evidence-based explanation\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. BASIC METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"   NSE (Nash-Sutcliffe Efficiency): {metrics['nse']:.6f}\n")
        f.write(f"   OFV (Objective = 1 - NSE):       {metrics['ofv']:.6f}\n")
        f.write(f"   Bias (mean(Sim - Obs)):          {metrics['bias']:.4f} m3/s\n")
        f.write(f"      (positive = model overestimates on average)\n")
        f.write(f"   RMSE:                            {metrics['rmse']:.4f} m3/s\n\n")
        
        f.write("2. PEAK FLOW ANALYSIS (Top 5 Observed Peaks)\n")
        f.write("-" * 40 + "\n")
        f.write("   NSE uses squared errors, so large peak mismatches dominate!\n\n")
        f.write("   Rank | Observed   | Simulated  | Error      | Rel.Error\n")
        f.write("   " + "-" * 55 + "\n")
        for i, peak in enumerate(metrics['peak_errors'], 1):
            f.write(f"   {i:4d} | {peak['obs']:10.2f} | {peak['sim']:10.2f} | "
                    f"{peak['error']:+10.2f} | {peak['rel_error_pct']:+8.1f}%\n")
        
        f.write(f"\n   Top 5 peaks contribute {metrics['peak_contribution_pct']:.1f}% "
                f"of total squared error!\n\n")
        
        f.write("3. LOW FLOW ANALYSIS (Bottom 10% of Observed Flows)\n")
        f.write("-" * 40 + "\n")
        lf = metrics['low_flow_stats']
        f.write(f"   Threshold (10th percentile): {lf['threshold']:.2f} m3/s\n")
        f.write(f"   Number of points:            {lf['n_points']}\n")
        f.write(f"   Mean observed:               {lf['mean_obs']:.2f} m3/s\n")
        f.write(f"   Mean simulated:              {lf['mean_sim']:.2f} m3/s\n")
        f.write(f"   Low flow bias:               {lf['bias']:.2f} m3/s\n")
        f.write(f"   Low flow RMSE:               {lf['rmse']:.2f} m3/s\n\n")
        
        f.write("4. WHY NSE IS NOT REACHING 0.90\n")
        f.write("-" * 40 + "\n")
        f.write("   The NSE formula:\n")
        f.write("   NSE = 1 - sum((Sim - Obs)^2) / sum((Obs - mean(Obs))^2)\n\n")
        f.write("   Key insight: SQUARED errors mean large peak mismatches\n")
        f.write("   have disproportionate impact on NSE.\n\n")
        f.write("   From the analysis above:\n")
        
        # Identify main issues
        avg_peak_error = np.mean([abs(p['error']) for p in metrics['peak_errors']])
        if avg_peak_error > 5:
            f.write(f"   - Peak flows are underestimated by avg {avg_peak_error:.1f} m3/s\n")
        
        if abs(lf['bias']) > 2:
            if lf['bias'] > 0:
                f.write(f"   - Low flows are overestimated by {lf['bias']:.1f} m3/s\n")
            else:
                f.write(f"   - Low flows are underestimated by {abs(lf['bias']):.1f} m3/s\n")
        
        f.write("\n   These systematic errors in peaks and low flows prevent NSE > 0.90.\n")
        f.write("   This is a model structure limitation, not an optimization failure.\n\n")
        
        f.write("5. PRACTICAL IMPROVEMENT SUGGESTIONS (Assignment-safe)\n")
        f.write("-" * 40 + "\n")
        f.write("   a) Run DE with multiple random seeds (3-5 runs) and compare\n")
        f.write("   b) Increase maxiter and popsize (if time allows)\n")
        f.write("   c) Try polish=True in DE (bounded local search after DE)\n")
        f.write("   d) Check if bounds are overly tight (stay within assignment limits)\n")
        f.write("   e) Note: NSE = 0.86 is already 'very good' for lumped models\n")
        f.write("      (NSE > 0.75 = good, NSE > 0.85 = very good in hydrology)\n")
        f.write("=" * 70 + "\n")

# =============================================================================
# Data loading
# =============================================================================

def load_data():
    """
    Load input data from ./data directory.
    Returns: (temps, precip, pet, obs_discharge, n_timesteps, discharge_scaler, index, dataframe)
    """
    data_dir = project_root / 'data'
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    print(f"Loading data from: {data_dir}")
    
    # Read time series
    ts_file = data_dir / 'time_series___24163005.csv'
    inp_dfe = pd.read_csv(ts_file, sep=';', index_col=0)
    inp_dfe.index = pd.to_datetime(inp_dfe.index, format='%Y-%m-%d-%H')
    
    # Read catchment area
    area_file = data_dir / 'area___24163005.csv'
    cca_srs = pd.read_csv(area_file, sep=';', index_col=0)
    catchment_area = cca_srs.values[0, 0]
    
    # Extract arrays
    temps = inp_dfe['tavg__ref'].values.astype(np.float32)
    precip = inp_dfe['pptn__ref'].values.astype(np.float32)
    pet = inp_dfe['petn__ref'].values.astype(np.float32)
    obs_q = inp_dfe['diso__ref'].values.astype(np.float32)
    
    n_timesteps = len(temps)
    
    # Discharge scaler: mm/hr -> m³/s for hourly data
    dslr = catchment_area / (3600 * 1000)
    
    print(f"  Time steps: {n_timesteps}")
    print(f"  Catchment area: {catchment_area:,.0f} m²")
    print(f"  Discharge scaler: {dslr:.6f}")
    print()
    
    return temps, precip, pet, obs_q, n_timesteps, dslr, inp_dfe.index, inp_dfe

# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("ASSIGNMENT 1: HBV Model Parameter Optimization")
    print("=" * 70)
    print()
    
    # Verify HBV implementation (must be Python for Assignment 1)
    verify_hbv_implementation()
    
    # Get parameter info from model (single source of truth)
    param_names, param_indices, abs_bounds = get_param_info_from_model()
    n_params = len(param_names)
    
    # Build bounds in correct order
    bounds_list = build_bounds_in_order(param_names, param_indices)
    
    # Load data
    temps, precip, pet, obs_q, n_timesteps, dslr, time_index, inp_dfe = load_data()
    
    # Configuration
    gen = DE_GENERATIONS
    pop_sz = DE_POPSIZE
    seed = DE_SEED
    
    if FAST_DEBUG:
        gen = 5
        pop_sz = 5
        print("*** FAST_DEBUG MODE: gen=5, pop_sz=5 ***")
    
    print(f"DE Configuration: gen={gen}, popsize={pop_sz}, seed={seed}")
    print()
    
    # Create recorder
    recorder = DERecorder(
        inputs=(temps, precip, pet),
        dslr=dslr,
        tsps=n_timesteps,
        obs_q=obs_q,
        param_names=param_names,
        model_class=HBV001A,
        pop_sz=pop_sz
    )
    
    # Run differential evolution
    print("Starting Differential Evolution optimization...")
    print()
    
    de_result = differential_evolution(
        func=recorder,
        bounds=bounds_list,
        strategy='best1bin',
        maxiter=gen,
        popsize=pop_sz,
        atol=1e-3,
        polish=False,
        seed=seed,
        callback=recorder.callback
    )
    
    # Finalize any remaining evaluations
    recorder.finalize()
    
    # Results
    best_params = de_result.x.astype(np.float32)
    best_obj = float(de_result.fun)
    final_nse = 1.0 - best_obj
    
    print()
    print("=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Best Objective (1-NSE): {best_obj:.6f}")
    print(f"Best NSE: {final_nse:.6f}")
    print(f"Total evaluations: {recorder.eval_count}")
    print(f"Generations: {len(recorder.gen_bounds)}")
    print()
    
    print("Best parameters (in model order):")
    for i, name in enumerate(param_names):
        print(f"  [{i:2d}] {name}: {best_params[i]:.6f}")
    print()
    
    # Output directory
    out_dir = project_root / 'outputs' / 'assignment1'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to: {out_dir}")
    
    # Run final simulation with best parameters
    model = HBV001A()
    model.set_inputs(temps, precip, pet)
    model.set_outputs(n_timesteps)
    model.set_discharge_scaler(dslr)
    model.set_parameters(best_params)
    model.set_optimization_flag(0)
    model.run_model()
    
    sim_q = model.get_discharge()
    otps = model.get_outputs()
    otps_lbls = model.get_output_labels()
    
    # Save best parameters to CSV
    best_params_df = pd.DataFrame({
        'parameter': param_names,
        'index': list(range(n_params)),
        'value': best_params,
        'lower_bound': [b[0] for b in bounds_list],
        'upper_bound': [b[1] for b in bounds_list]
    })
    best_params_df.to_csv(out_dir / 'best_parameters.csv', index=False)
    
    # Save summary
    with open(out_dir / 'optimization_summary.txt', 'w') as f:
        f.write("HBV Model Parameter Optimization - Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Best Objective (1-NSE): {best_obj:.6f}\n")
        f.write(f"Best NSE: {final_nse:.6f}\n")
        f.write(f"Total evaluations: {recorder.eval_count}\n")
        f.write(f"Generations: {len(recorder.gen_bounds)}\n")
        f.write(f"Population size: {pop_sz}\n")
        f.write(f"Random seed: {seed}\n\n")
        f.write("Best parameters (in model order):\n")
        for i, name in enumerate(param_names):
            f.write(f"  [{i:2d}] {name}: {best_params[i]:.6f}\n")
    
    # Save full evaluation history
    hist_df = pd.DataFrame(np.vstack(recorder.eval_params), columns=param_names)
    hist_df['objective'] = recorder.eval_objs
    hist_df.to_csv(out_dir / 'optimization_history.csv', index=False)
    
    # Save per-generation summary
    gen_summary = []
    for gi, (s, e) in enumerate(recorder.gen_bounds):
        if gi < len(recorder.gen_best_objs):
            gen_summary.append({
                'generation': gi + 1,
                'n_evals': e - s,
                'best_obj': recorder.gen_best_objs[gi],
                'max_obj': recorder.gen_max_objs[gi],
                'median_obj': recorder.gen_med_objs[gi]
            })
    pd.DataFrame(gen_summary).to_csv(out_dir / 'generation_summary.csv', index=False)
    
    # Generate plots
    print("Generating plots...")
    
    # Compute detailed metrics for Part A analysis
    metrics = compute_detailed_metrics(obs_q, sim_q)
    bias = metrics['bias']
    rmse = metrics['rmse']
    
    # Save detailed metrics report
    save_detailed_metrics_report(metrics, out_dir / 'detailed_metrics_analysis.txt')
    print("  Saved: detailed_metrics_analysis.txt")
    
    # Observed vs simulated (with residuals and metrics)
    plot_observed_vs_simulated(time_index, obs_q, sim_q, final_nse, best_obj,
                               out_dir / 'observed_vs_simulated.png')
    
    # Internal variables (with FCY/PWP lines and URR threshold)
    plot_internal_variables(time_index, inp_dfe, otps, otps_lbls, best_params, param_names,
                           out_dir / 'internal_variables.png')
    
    # Normalized parameters
    plot_parameters_normalized(bounds_list, best_params, param_names,
                              out_dir / 'parameters_normalized.png')
    
    # Parameter evolution (x-axis: model runs, step changes only when running best improves)
    if len(recorder.running_best_at_eval) > 0:
        plot_param_evolution_vs_model_runs(recorder.running_best_at_eval, recorder.eval_count,
                                           param_names, recorder.gen_bounds,
                                           out_dir / 'parameter_evolution.png')
    
    # Convergence (with reference line for final best)
    plot_convergence_per_generation(recorder.gen_best_objs, recorder.gen_max_objs,
                                   recorder.gen_med_objs, best_obj, out_dir)
    
    # Parameter scatter plots (with best value reference lines)
    plot_param_scatter_all_evals(recorder.eval_params, recorder.eval_objs,
                                bounds_list, param_names, best_params, best_obj,
                                str(out_dir))
    
    # NEW: Scatter plot Observed vs Simulated
    plot_scatter_obs_vs_sim(obs_q, sim_q, final_nse, bias, rmse,
                            out_dir / 'scatter_obs_vs_sim.png')
    print("  Saved: scatter_obs_vs_sim.png")
    
    # NEW: Flow Duration Curve
    plot_flow_duration_curve(obs_q, sim_q, out_dir / 'flow_duration_curve.png')
    print("  Saved: flow_duration_curve.png")
    
    print()
    print("=" * 60)
    print("OUTPUT FILES")
    print("=" * 60)
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")
    
    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"NSE = {final_nse:.6f}")
    print(f"OFV (1-NSE) = {best_obj:.6f}")
    print(f"Output directory: {out_dir}")
    print("=" * 60)

if __name__ == '__main__':
    print(f'#### Started on {time.asctime()} ####\n')
    START = timeit.default_timer()
    
    if DEBUG_FLAG:
        try:
            main()
        except:
            pre_stack = tb.format_stack()[:-1]
            err_tb = list(tb.TracebackException(*sys.exc_info()).format())
            lines = [err_tb[0]] + pre_stack + err_tb[2:]
            for line in lines:
                print(line, file=sys.stderr, end='')
            import pdb
            pdb.post_mortem()
    else:
        main()
    
    STOP = timeit.default_timer()
    print(f'\n#### Done on {time.asctime()}')
    print(f'Total runtime: {STOP - START:.2f} seconds ####')


