"""
Group B

Assignment 4: Model Input Uncertainty Analysis
==============================================

This script analyzes the impact of input uncertainty (precipitation) on model
output and calibrated parameters.

Tasks (from PDF):
1. Create 2000 perturbed precipitation series using Gaussian noise C = N(1, 0.05)
2. Compare CDFs of perturbed vs original precipitation
3. Run model with reference parameters for all perturbed series
4. Recalibrate model for all perturbed series (optional, computationally heavy)
5. Compare parameter CDFs and OFV distributions
6. Scatter plots: Mean absolute relative PPT change vs OFV

Reference: Oudin et al. 2006 - Impact of biased and randomly corrupted inputs
on the efficiency and the parameters of watershed models

Usage:
------
# Quick debug (no recalibration)
python hmg/Assignment4/Input_Uncertainty_Analysis.py --n 5 --no-recalibrate

# Quick recalibration test
python hmg/Assignment4/Input_Uncertainty_Analysis.py --n 3 --recalibrate --maxiter 5 --popsize 3

# Full run (2000 perturbations with recalibration, multipliers enforced [0.75,1.25] by default)
python hmg/Assignment4/Input_Uncertainty_Analysis.py --n 2000 --recalibrate --maxiter 40 --popsize 10

# Disable multiplier bounds (allows >25% change, NOT recommended for assignment)
python hmg/Assignment4/Input_Uncertainty_Analysis.py --n 2000 --recalibrate --no-clip-multipliers

# Diagnostic convergence test (K random series with higher DE settings)
python hmg/Assignment4/Input_Uncertainty_Analysis.py --n 100 --recalibrate --diagnostic-k 5
"""

import sys
import argparse
from pathlib import Path

# Add parent directories to path for imports
script_dir = Path(__file__).resolve().parent
hmg_dir = script_dir.parent
workspace_dir = hmg_dir.parent

for p in [str(workspace_dir), str(hmg_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd

# Set matplotlib backend before importing pyplot (avoid tkinter issues in parallel)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution
from scipy import stats
from joblib import Parallel, delayed
import time
from datetime import datetime
import os
import platform
import warnings
from contextlib import contextmanager

# Suppress excessive model runtime prints
warnings.filterwarnings('ignore')


@contextmanager
def suppress_output(enabled=True):
    """
    Suppress both Python-level and C-level stdout/stderr.
    
    The HBV model backend uses Python print() for timing output,
    so we need to redirect sys.stdout in addition to file descriptors.
    """
    if not enabled:
        yield
        return
    
    # Save original Python streams
    old_stdout_py = sys.stdout
    old_stderr_py = sys.stderr
    
    # Redirect Python streams to devnull
    with open(os.devnull, 'w') as devnull_file:
        sys.stdout = devnull_file
        sys.stderr = devnull_file
        try:
            yield
        finally:
            # Restore Python streams
            sys.stdout = old_stdout_py
            sys.stderr = old_stderr_py


# =============================================================================
# BACKEND DETECTION
# =============================================================================

def detect_and_import_hbv():
    """
    Detect which HBV001A backend is available and import it.
    Returns: HBV001A class and backend info string
    """
    backend_info = []
    HBV001A = None
    
    # Try Cython version first
    try:
        from hmg_cython.hmg import HBV001A as HBV_Cython
        HBV001A = HBV_Cython
        backend_info.append("Cython (hmg_cython)")
    except ImportError:
        pass
    
    # Try standard hmg package
    if HBV001A is None:
        try:
            from hmg import HBV001A as HBV_Standard
            HBV001A = HBV_Standard
            
            # Check if it's using compiled backend
            model = HBV_Standard()
            modl_func = model._modl
            modl_module = modl_func.__module__
            mod = sys.modules.get(modl_module, None)
            modl_file = getattr(mod, '__file__', 'unknown') if mod else 'unknown'
            
            if '.pyd' in str(modl_file) or '.so' in str(modl_file):
                backend_info.append(f"Compiled ({modl_file})")
            else:
                backend_info.append(f"Python ({modl_file})")
        except ImportError:
            pass
    
    # Try direct import
    if HBV001A is None:
        try:
            from models.hbv001a_py import HBV001A as HBV_Direct
            HBV001A = HBV_Direct
            backend_info.append("Direct Python import")
        except ImportError:
            raise ImportError("Could not import HBV001A from any source!")
    
    return HBV001A, backend_info[0] if backend_info else "Unknown"


# Import HBV001A
HBV001A, BACKEND_INFO = detect_and_import_hbv()


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

# Gaussian perturbation parameters: C = N(mean, std)
PERTURB_MEAN = 1.0
PERTURB_STD = 0.083 # Results in ~5% std, up to ~25% at 3-sigma

# Random seed for reproducibility
MASTER_SEED = 42

# Penalty for failed runs
PENALTY = 5.0

# Parallel processing
N_JOBS = 4  # Number of parallel jobs


# =============================================================================
# PARAMETER BOUNDS (same as Assignment 1)
# =============================================================================

PARAM_BOUNDS = [
    (0.0, 0.0),      # snw_dth - FIXED at 0
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

PARAM_NAMES = [
    'snw_dth', 'snw_att', 'snw_pmf', 'snw_amf',
    'sl0_dth', 'sl0_pwp', 'sl0_fcy', 'sl0_bt0',
    'urr_dth', 'lrr_dth', 'urr_wsr', 'urr_ulc',
    'urr_tdh', 'urr_tdr', 'urr_ndr', 'urr_uct',
    'lrr_dre', 'lrr_lct'
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load input data and reference calibration results."""
    
    # Data paths - use workspace data directory
    workspace_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = workspace_dir / 'data'
    
    data_dir = Path(os.environ.get("HYDROLOGY_DATA_DIR", data_dir))
    if not data_dir.is_dir():
        raise FileNotFoundError(
            "Hydrology input data was not found. Set HYDROLOGY_DATA_DIR to an "
            "authorized directory containing time_series___24163005.csv and "
            "area___24163005.csv."
        )
    
    out_dir_a1 = workspace_dir / 'outputs' / 'assignment1'
    ref_dir = Path(__file__).resolve().parent.parent / 'Assignment1' / 'results'
    
    # Load time series
    inp_dfe = pd.read_csv(data_dir / 'time_series___24163005.csv', 
                          sep=';', index_col=0)
    inp_dfe.index = pd.to_datetime(inp_dfe.index, format='%Y-%m-%d-%H')
    
    # Load catchment area
    ccaa = pd.read_csv(data_dir / 'area___24163005.csv', 
                       sep=';', index_col=0).values[0, 0]
    
    # Extract arrays
    temps = inp_dfe['tavg__ref'].values.astype(np.float32)
    precip = inp_dfe['pptn__ref'].values.astype(np.float32)
    pet = inp_dfe['petn__ref'].values.astype(np.float32)
    obs_q = inp_dfe['diso__ref'].values.astype(np.float32)
    
    # Discharge scaler
    dslr = ccaa / (3600 * 1000)
    
    # Load reference parameters from Assignment 1
    ref_params = None
    for try_dir in [out_dir_a1, ref_dir]:
        try:
            ref_params_df = pd.read_csv(try_dir / 'best_parameters.csv')
            ref_params = ref_params_df['value'].values.astype(np.float32)
            print(f"  Loaded reference parameters from: {try_dir}")
            break
        except FileNotFoundError:
            continue
    
    if ref_params is None:
        raise FileNotFoundError("Could not find best_parameters.csv from Assignment 1")
    
    return temps, precip, pet, obs_q, dslr, ref_params, inp_dfe.index


# =============================================================================
# PERTURBATION FUNCTIONS
# =============================================================================

def create_perturbed_series(precip, n_series, mean=1.0, std=0.05, 
                            clip_multipliers=True, seed=None):
    """
    Create n_series perturbed precipitation series.
    
    Each value is scaled by C = N(mean, std).
    Assignment requirement: "up to 25% each value" => clip to [0.75, 1.25].
    All values are kept non-negative.
    
    Parameters
    ----------
    precip : array
        Original precipitation series
    n_series : int
        Number of perturbed series to create
    mean : float
        Mean of Gaussian multiplier (default 1.0)
    std : float
        Standard deviation of Gaussian multiplier (default 0.05)
    clip_multipliers : bool
        If True (DEFAULT), clip multipliers to [0.75, 1.25] (up to 25% change)
        This satisfies the assignment requirement.
        If False, use pure Gaussian (can exceed 25%).
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    perturbed : array of shape (n_series, len(precip))
        Perturbed precipitation series
    multipliers : array of shape (n_series, len(precip))
        The random multipliers used (after any clipping)
    """
    #rng = np.random.default_rng(seed)
    
    n_timesteps = len(precip)
    
    # Generate random multipliers C = N(mean, std)
    #multipliers = rng.normal(mean, std, size=(n_series, n_timesteps))
    multipliers = np.random.normal(mean, std, size=(n_series, n_timesteps))
    
    # Enforce assignment requirement: up to 25% change per timestep
    # Default is ON to satisfy grading requirements
    #if clip_multipliers:
        #multipliers = np.clip(multipliers, 0.75, 1.25)
    
    # Apply to precipitation
    perturbed = precip[np.newaxis, :] * multipliers
    
    # Keep all non-negative
    perturbed = np.maximum(perturbed, 0.0)
    
    return perturbed.astype(np.float32), multipliers


def compute_mean_abs_relative_change(original, perturbed):
    """
    Compute mean absolute relative change between original and perturbed.
    
    MARC = mean(|perturbed - original| / original) for original > 0
    
    Handles zero PPT safely by only computing over timesteps where PPT_ref > 0.
    """
    # Avoid division by zero - only consider non-zero original values
    mask = original > 0
    if not np.any(mask):
        return 0.0
    
    rel_change = np.abs(perturbed[mask] - original[mask]) / original[mask]
    return np.mean(rel_change)


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def compute_nse(obs, sim):
    """Compute Nash-Sutcliffe Efficiency."""
    obs_mean = np.mean(obs)
    ss_res = np.sum((obs - sim) ** 2)
    ss_tot = np.sum((obs - obs_mean) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def run_model_with_params(params, temps, precip, pet, dslr, n_steps, verbose=False):
    """Run HBV model with given parameters and inputs."""
    model = HBV001A()
    model.set_inputs(temps, precip, pet)
    model.set_outputs(n_steps)
    model.set_discharge_scaler(dslr)
    model.set_parameters(params)
    # The optimization flag doesn't control timing output in current backend
    model.set_optimization_flag(0 if not verbose else 1)
    
    # Suppress Python print() from model backend (timing messages)
    with suppress_output(enabled=not verbose):
        model.run_model()
    
    return model.get_discharge()


def objective_function(params_variable, temps, precip, pet, obs_q, dslr, n_steps):
    """
    Objective function for optimization: OFV = 1 - NSE
    
    params_variable contains only the variable parameters (excluding fixed snw_dth)
    """
    # Reconstruct full parameter array
    params = np.zeros(18, dtype=np.float32)
    params[0] = 0.0  # snw_dth fixed
    params[1:] = params_variable
    
    try:
        sim_q = run_model_with_params(params, temps, precip, pet, dslr, n_steps, verbose=False)
        nse = compute_nse(obs_q, sim_q)
        
        if np.isnan(nse) or np.isinf(nse):
            return PENALTY
        
        # Clamp OFV to [0, PENALTY] to handle NSE>1 numerical issues
        ofv = np.clip(1.0 - nse, 0.0, PENALTY)
        return ofv
        
    except Exception:
        return PENALTY
class DERecorder:
    """
    Model and input needed for optimization.
    created memory for interim results of optimization run
    """
    
    def __init__(self, inputs, dslr, tsps, obs_q, model_class=HBV001A, pop_sz=15):
    
        self.input = inputs #(tupel temp., prec., ETP)
        self.dslr = np.float32(dslr)
        self.tsps = int(tsps)
        self.obs_q = np.asarray(obs_q, dtype=np.float32)
        self.model_class = model_class
        self.pop_sz = pop_sz
        
        # history
        self.eval_params = []   # list of numpy arrays (optimized subset)
        self.eval_objs = []     # matching list of objective values (float)
        self.eval_count = 0
        
        # generation boundaries: list of (start_idx, end_idx) per generation
        self.gen_bounds = []
        self._last_gen_end = 0
        
        
        
    def __call__(self, prms):
        """
        Set up and run the model; return optional function value.
        """
        # Reconstruct full parameter array
        params = np.zeros(18, dtype=np.float32)
        params[0] = 0.0  # snw_dth fixed
        params[1:] = prms
            
        tems, ppts, pets = self.input
        modl_objt = self.model_class()
        modl_objt.set_inputs(tems, ppts, pets)
        modl_objt.set_outputs(self.tsps)
        modl_objt.set_discharge_scaler(self.dslr)
        modl_objt.set_parameters(params)
        modl_objt.set_optimization_flag(0)
        # Suppress Python print() from model backend (timing messages)
            
        modl_objt.run_model()
        diss = modl_objt.get_discharge()
            
        nse = compute_nse(self.obs_q, diss)
        obj = np.clip(1.0 - nse, 0.0, PENALTY)
            
        return float(obj)

def calibrate_model(temps, precip, pet, obs_q, dslr, maxiter, popsize, seed=None,
                    init_population=None):
    """
    Calibrate model using Differential Evolution.
    
    Parameters
    ----------
    init_population : array-like, optional
        Initial population for warm-start. Shape: (popsize * n_params, n_params)
        where n_params = 17. SciPy DE uses popsize as a multiplier, so actual
        population size = popsize * len(bounds).
        If provided, seeds the DE search near these values.
    
    Returns best parameters and best OFV.
    """
    n_steps = len(temps)
    
    # Variable bounds (excluding fixed snw_dth at index 0)
    var_bounds = [PARAM_BOUNDS[i] for i in range(1, 18)]
    
        
    recorder = DERecorder(
        inputs = (temps, precip, pet),
        dslr = dslr,
        tsps = n_steps,
        obs_q = obs_q,
        model_class=HBV001A,
        pop_sz = popsize
        )
    result = differential_evolution(
        func = recorder,
        bounds = var_bounds,
        strategy='best1bin',
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-3,
        atol=0,
        polish=False,
        workers=1,  # Single worker since we parallelize at series level
        updating='deferred'
        )
    
    # Reconstruct full parameters
    
    best_params = np.zeros(18, dtype=np.float32)
    best_params[0] = 0.0
    best_params[1:] = result.x.astype(np.float32)
    best_obj = float(result.fun)
    
    return best_params, best_obj


# =============================================================================
# DIAGNOSTIC FUNCTIONS
# =============================================================================

def run_diagnostic_convergence_test(k_series, perturbed_ppt, temps, pet, obs_q, 
                                     dslr, ref_params, args, out_dir):
    """
    Run diagnostic test to check if optimizer under-convergence explains poor recalibration.
    
    Picks K random perturbed series and runs recalibration with higher DE settings,
    then compares to normal settings.
    """
    print(f"\n[DIAGNOSTIC CONVERGENCE TEST]", flush=True)
    print(f"  Testing {k_series} random series with higher DE settings...", flush=True)
    
    rng = np.random.default_rng(args.seed + 9999)
    test_indices = rng.choice(len(perturbed_ppt), size=min(k_series, len(perturbed_ppt)), replace=False)
    
    # Higher settings: 3x iterations, 2x population
    high_maxiter = args.maxiter * 3
    high_popsize = args.popsize * 2
    
    print(f"  Normal DE: maxiter={args.maxiter}, popsize={args.popsize}", flush=True)
    print(f"  High DE: maxiter={high_maxiter}, popsize={high_popsize}", flush=True)
    
    ref_params_variable = ref_params[1:].copy()  # Exclude snw_dth
    n_steps = len(temps)
    
    results = []
    for i, idx in enumerate(test_indices):
        ppt = perturbed_ppt[idx]
        seed_base = args.seed + idx
        
        # Compute OFV with reference params on this perturbed series
        sim_ref = run_model_with_params(ref_params, temps, ppt, pet, dslr, n_steps, verbose=False)
        nse_ref = compute_nse(obs_q, sim_ref)
        # Clamp OFV to [0, PENALTY] to handle NSE>1 numerical issues
        ofv_ref_on_perturbed = np.clip(1.0 - nse_ref, 0.0, PENALTY) if not np.isnan(nse_ref) else PENALTY
        
        # Normal recalibration
        _, _, ofv_normal = calibrate_single_series(
            idx, ppt, temps, pet, obs_q, dslr, 
            args.maxiter, args.popsize, seed_base, 
            ref_params_variable=ref_params_variable, warmstart=args.warmstart
        )
        
        # High-effort recalibration
        _, _, ofv_high = calibrate_single_series(
            idx, ppt, temps, pet, obs_q, dslr,
            high_maxiter, high_popsize, seed_base,
            ref_params_variable=ref_params_variable, warmstart=True  # Always warmstart for diagnostic
        )
        
        results.append({
            'series': idx,
            'ofv_ref_params': ofv_ref_on_perturbed,
            'ofv_normal_recal': ofv_normal,
            'ofv_high_recal': ofv_high,
            'improvement': ofv_normal - ofv_high
        })
        
        print(f"  Series {idx}: RefParams={ofv_ref_on_perturbed:.4f}, "
              f"NormalDE={ofv_normal:.4f}, HighDE={ofv_high:.4f}, "
              f"Diff={ofv_normal - ofv_high:.4f}", flush=True)
    
    # Summarize
    improvements = [r['improvement'] for r in results]
    mean_improvement = np.mean(improvements)
    
    print(f"\n  DIAGNOSTIC SUMMARY:", flush=True)
    print(f"  - Mean OFV improvement with higher DE: {mean_improvement:.4f}", flush=True)
    
    if mean_improvement > 0.01:
        print(f"  - WARNING: Optimizer likely under-converged. Consider increasing maxiter/popsize.", flush=True)
    else:
        print(f"  - Optimizer convergence appears adequate.", flush=True)
    
    # Check if reference params beat recalibration
    ref_better = sum(1 for r in results if r['ofv_ref_params'] < r['ofv_normal_recal'])
    print(f"  - Reference params better than recal: {ref_better}/{len(results)}", flush=True)
    
    # Save diagnostic results
    diag_df = pd.DataFrame(results)
    diag_df.to_csv(out_dir / 'diagnostic_convergence.csv', index=False)
    print(f"  Saved: diagnostic_convergence.csv", flush=True)
    
    return results


# =============================================================================
# PARALLEL PROCESSING FUNCTIONS
# =============================================================================

def evaluate_single_series_reference(idx, perturbed_precip, temps, pet, obs_q, 
                                      dslr, ref_params):
    """Evaluate a single perturbed series with reference parameters."""
    n_steps = len(temps)
    
    try:
        sim_q = run_model_with_params(ref_params, temps, perturbed_precip, 
                                       pet, dslr, n_steps, verbose=False)
        nse = compute_nse(obs_q, sim_q)
        if np.isnan(nse) or np.isinf(nse):
            ofv = PENALTY
        else:
            # Clamp OFV to [0, PENALTY] to handle NSE>1 numerical issues
            ofv = np.clip(1.0 - nse, 0.0, PENALTY)
    except Exception:
        ofv = PENALTY
    
    return idx, ofv


def calibrate_single_series(idx, perturbed_precip, temps, pet, obs_q, dslr, 
                            maxiter, popsize, seed, ref_params_variable=None,
                            warmstart=False):
    """
    Calibrate model for a single perturbed series.
    
    Parameters
    ----------
    ref_params_variable : array, optional
        Reference parameters (excluding snw_dth) for warm-start
    warmstart : bool
        If True and ref_params_variable provided, seed initial population near reference
    
    Note on popsize:
        SciPy's differential_evolution uses popsize as a MULTIPLIER.
        Actual population = popsize * n_params (17 here).
        The init array must have shape (popsize * n_params, n_params).
    """
    
    try:
        # Build initial population for warm-start if requested
        init_pop = None
        if warmstart and ref_params_variable is not None:
            # Create initial population: reference + random perturbations around it
            var_bounds = [PARAM_BOUNDS[i] for i in range(1, 18)]
            rng = np.random.default_rng(seed)
            n_params = len(ref_params_variable)  # 17
            
            # CRITICAL: actual_pop_size = popsize * n_params (SciPy convention)
            actual_pop_size = popsize * n_params
            init_pop = np.zeros((actual_pop_size, n_params))
            init_pop[0] = ref_params_variable  # First member is reference
            
            for j in range(1, actual_pop_size):
                # Perturb reference by ±10% within bounds
                perturbed = ref_params_variable * (1 + rng.uniform(-0.1, 0.1, n_params))
                for k, (lo, hi) in enumerate(var_bounds):
                    perturbed[k] = np.clip(perturbed[k], lo, hi)
                init_pop[j] = perturbed
        
        best_params, best_ofv = calibrate_model(temps, perturbed_precip, pet, 
                                                 obs_q, dslr, maxiter, popsize, 
                                                 seed=seed, init_population=init_pop)
    except Exception:
        best_params = np.full(18, np.nan, dtype=np.float32)
        best_ofv = PENALTY
    
    return idx, best_params, best_ofv


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_ppt_cdf_comparison(original, perturbed_all, out_dir):
    """
    Plot CDF comparison of original vs perturbed precipitation.
    Shows [%] percentile envelope band + original as thick line.
    Does NOT plot all individual CDFs (avoids clutter).
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # Compute CDF percentiles across all perturbed series
    # For each quantile level, find the 5th and 95th percentile of PPT values
    n_quantiles = 200  # Resolution for CDF
    quantile_levels = np.linspace(0, 1, n_quantiles)
    
    # Get quantile values for each perturbed series (vectorized)
    # np.quantile with axis=1 computes quantiles along each series
    perturbed_quantiles = np.quantile(perturbed_all, quantile_levels, axis=1).T
    # Compute envelope (5th and 95th percentile across series at each quantile level)
    envelope_lo = np.percentile(perturbed_quantiles, 5, axis=0)
    envelope_hi = np.percentile(perturbed_quantiles, 95, axis=0)
    envelope_median = np.median(perturbed_quantiles, axis=0)
    
    # Plot envelope band
    ax.fill_betweenx(quantile_levels, envelope_lo, envelope_hi, 
                     alpha=0.3, color='blue', label='5 - 95% enverlope (perturbed)')
    ax.plot(envelope_median, quantile_levels, color='blue', linewidth=1, 
            linestyle='--', alpha=0.7, label='Median perturbed')
    
    # Plot original CDF
    sorted_orig = np.sort(original)
    cdf_orig = np.arange(1, len(sorted_orig) + 1) / len(sorted_orig)
    ax.plot(sorted_orig, cdf_orig, color='black', linewidth=2.5, 
            label='Original PPT')
    
    ax.set_xlabel('Precipitation [mm/hr]', fontsize=11)
    ax.set_ylabel('Cumulative Probability', fontsize=11)
    ax.set_title('CDF Comparison: Original vs Perturbed Precipitation\n'
                 f'(Envelope from {len(perturbed_all)} perturbed series)',
                 fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    fig.savefig(out_dir / 'ppt_cdf_comparison.png', dpi=150)
    plt.close(fig)
    print("  Saved: ppt_cdf_comparison.png")

def plot_ppt_cdf_comparison_two(original, perturbed_all, out_dir):
    """
    Plot CDF comparison of original vs perturbed precipitation.
    Shows [%] percentile envelope band + original as thick line.
    Does NOT plot all individual CDFs (avoids clutter).
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    perturbed_ppt = np.stack([np.cumsum(n) for n in perturbed_all])
    
    cdf_orig = np.cumsum(original)
    tsp = range(0,len(original))
    
    for n in perturbed_ppt:
        ax.plot(tsp, n , alpha = 0.5, color='lightskyblue')
    ax.plot(tsp, cdf_orig, color='black', linewidth=1.5, 
            label='original PPT')
    ax.set_xlabel('time steps [d]', fontsize=11)
    ax.set_ylabel('Cumulative Precipitation', fontsize=11)
    ax.set_title('Comparison: Original vs Perturbed Precipitation\n'
                 f'(Envelope from {len(perturbed_all)} perturbed series)',
                 fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    fig.savefig(out_dir / 'ppt_cdf_comparison_two.png', dpi=150)
    plt.close(fig)
    print("  Saved: ppt_cdf_comparison_two.png")

def plot_parameter_cdfs(ref_params, recalib_params_all, ofv_recalibrated, out_dir):
    """
    Plot CDF of each recalibrated parameter with reference value marked.
    """
    # Filter valid results using consistent mask (same as save_results_summary)
    valid_mask = ~np.isnan(ofv_recalibrated) & (ofv_recalibrated < PENALTY)
    valid_params = recalib_params_all[valid_mask]
    
    if len(valid_params) == 0:
        print("  WARNING: No valid recalibration results for parameter CDF plot")
        return
    
    fig, axes = plt.subplots(6, 3, figsize=(14, 16), dpi=150)
    axes = axes.flatten()
    
    param_idx = 0
    for i in range(18):
        if i == 0:  # Skip fixed snw_dth
            continue
        
        ax = axes[param_idx]
        
        # Get recalibrated values for this parameter
        recalib_vals = valid_params[:, i]
        
        # Plot CDF
        sorted_vals = np.sort(recalib_vals)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf, color='blue', linewidth=1.5, 
                label='Recalibrated')
        
        # Mark reference value
        ref_val = ref_params[i]
        ax.axvline(ref_val, color='red', linestyle='--', linewidth=2,
                   label=f'Reference: {ref_val:.4f}')
        
        # Find percentile of reference in recalibrated distribution
        percentile = np.mean(recalib_vals <= ref_val) * 100
        
        ax.set_xlabel(PARAM_NAMES[i], fontsize=9)
        ax.set_ylabel('CDF', fontsize=9)
        ax.set_title(f'{PARAM_NAMES[i]} (ref at {percentile:.0f}th %ile)', 
                     fontsize=10)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
        
        param_idx += 1
    
    # Hide unused subplot
    axes[-1].set_visible(False)
    
    fig.suptitle('Parameter CDFs: Recalibrated vs Reference\n'
                 f'(n = {len(valid_params)} successful recalibrations)',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / 'parameter_cdfs.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: parameter_cdfs.png")


def plot_ofv_cdfs(ref_ofv, ofv_reference_params, ofv_recalibrated, out_dir, 
                  has_recalibration=True):
    """
    Plot OFV CDFs comparing reference params vs recalibrated.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # Filter valid OFVs
    valid_ref = ofv_reference_params[ofv_reference_params < PENALTY]
    
    # Plot CDF for reference parameters
    sorted_ref = np.sort(valid_ref)
    cdf_ref = np.arange(1, len(sorted_ref) + 1) / len(sorted_ref)
    ax.plot(sorted_ref, cdf_ref, color='blue', linewidth=2, 
            label=f'With reference params (n={len(valid_ref)})')
    
    # Plot CDF for recalibrated (if available)
    if has_recalibration and ofv_recalibrated is not None:
        valid_recalib = ofv_recalibrated[ofv_recalibrated < PENALTY]
        if len(valid_recalib) > 0:
            sorted_recalib = np.sort(valid_recalib)
            cdf_recalib = np.arange(1, len(sorted_recalib) + 1) / len(sorted_recalib)
            ax.plot(sorted_recalib, cdf_recalib, color='green', linewidth=2,
                    label=f'Recalibrated (n={len(valid_recalib)})')
    
    # Mark reference OFV
    ax.axvline(ref_ofv, color='red', linestyle='--', linewidth=2,
               label=f'Reference OFV: {ref_ofv:.4f}')
    
    ax.set_xlabel('OFV (1 - NSE)', fontsize=11)
    ax.set_ylabel('Cumulative Probability', fontsize=11)
    ax.set_title('OFV Distribution: Reference Params vs Recalibrated\n'
                 'Using Perturbed Precipitation Inputs', fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Add secondary axis for NSE
    ax2 = ax.twiny()
    xlim = ax.get_xlim()
    ax2.set_xlim(1 - xlim[0], 1 - xlim[1])
    ax2.set_xlabel('NSE = 1 - OFV', fontsize=10, color='darkgreen')
    ax2.tick_params(axis='x', labelcolor='darkgreen')
    
    plt.tight_layout()
    fig.savefig(out_dir / 'ofv_cdfs.png', dpi=150)
    plt.close(fig)
    print("  Saved: ofv_cdfs.png")


def plot_scatter_relative_change_vs_ofv(original_ppt, perturbed_all, 
                                         ofv_reference, ofv_recalibrated,
                                         ref_ofv, out_dir, has_recalibration=True):
    """
    Scatter plot: Mean absolute relative PPT change vs OFV.
    Shows both reference params and recalibrated results.
    """
    # Compute mean absolute relative change for each perturbed series
    marc_values = []
    for i in range(len(perturbed_all)):
        marc = compute_mean_abs_relative_change(original_ppt, perturbed_all[i])
        marc_values.append(marc)
    marc_values = np.array(marc_values)
    
    # Filter valid points
    valid_ref = ofv_reference < PENALTY
    
    if has_recalibration and ofv_recalibrated is not None:
        valid_recalib = ofv_recalibrated < PENALTY
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6), dpi=150)
        axes = [axes]
    
    # Left: Reference parameters
    ax1 = axes[0]
    ax1.scatter(marc_values[valid_ref] * 100, ofv_reference[valid_ref], 
                alpha=0.3, s=10, color='blue')
    ax1.axhline(ref_ofv, color='red', linestyle='--', linewidth=2,
                label=f'Reference OFV: {ref_ofv:.4f}')
    ax1.axvline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Mean Absolute Relative PPT Change [%]', fontsize=11)
    ax1.set_ylabel('OFV (1 - NSE)', fontsize=11)
    ax1.set_title('With Reference Parameters', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Add correlation
    if np.sum(valid_ref) > 1:
        corr_ref = np.corrcoef(marc_values[valid_ref], ofv_reference[valid_ref])[0, 1]
        ax1.text(0.95, 0.95, f'r = {corr_ref:.3f}', transform=ax1.transAxes,
                 ha='right', va='top', fontsize=10, 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Right: Recalibrated (if available)
    if has_recalibration and ofv_recalibrated is not None and len(axes) > 1:
        ax2 = axes[1]
        ax2.scatter(marc_values[valid_recalib] * 100, ofv_recalibrated[valid_recalib],
                    alpha=0.3, s=10, color='green')
        ax2.axhline(ref_ofv, color='red', linestyle='--', linewidth=2,
                    label=f'Reference OFV: {ref_ofv:.4f}')
        ax2.axvline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax2.set_xlabel('Mean Absolute Relative PPT Change [%]', fontsize=11)
        ax2.set_ylabel('OFV (1 - NSE)', fontsize=11)
        ax2.set_title('With Recalibrated Parameters', fontsize=12)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Add correlation
        if np.sum(valid_recalib) > 1:
            corr_recalib = np.corrcoef(marc_values[valid_recalib], 
                                       ofv_recalibrated[valid_recalib])[0, 1]
            ax2.text(0.95, 0.95, f'r = {corr_recalib:.3f}', transform=ax2.transAxes,
                     ha='right', va='top', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle('OFV vs Mean Absolute Relative Precipitation Change', 
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / 'scatter_ppt_change_vs_ofv.png', dpi=150, 
                bbox_inches='tight')
    plt.close(fig)
    print("  Saved: scatter_ppt_change_vs_ofv.png")
    
    return marc_values


def plot_scatter_ofv_original_vs_recalibrated(ofv_reference, ofv_recalibrated,
                                               ref_ofv, out_dir):
    """
    Scatter plot: OFV with reference params vs OFV with recalibrated params.
    Shows how much recalibration improves/worsens results.
    Required by Assignment 4 lecture slides.
    """
    # Filter valid points (both must be valid)
    valid_mask = (ofv_reference < PENALTY) & (ofv_recalibrated < PENALTY)
    
    if np.sum(valid_mask) == 0:
        print("  WARNING: No valid points for OFV scatter plot")
        return
    
    ofv_ref_valid = ofv_reference[valid_mask]
    ofv_recal_valid = ofv_recalibrated[valid_mask]
    
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    
    # Scatter plot
    ax.scatter(ofv_ref_valid, ofv_recal_valid, alpha=0.4, s=20, color='blue',
               label=f'Perturbed series (n={len(ofv_ref_valid)})')
    
    # 1:1 line
    min_val = min(ofv_ref_valid.min(), ofv_recal_valid.min())
    max_val = max(ofv_ref_valid.max(), ofv_recal_valid.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, 
            label='1:1 line (no change)')
    
    # Reference OFV lines
    ax.axhline(ref_ofv, color='red', linestyle=':', linewidth=1.5, alpha=0.7,
               label=f'Reference OFV: {ref_ofv:.4f}')
    ax.axvline(ref_ofv, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Color points by improvement
    improved = ofv_recal_valid < ofv_ref_valid
    worsened = ofv_recal_valid > ofv_ref_valid
    
    # Count statistics
    n_improved = np.sum(improved)
    n_worsened = np.sum(worsened)
    n_better_than_ref = np.sum(ofv_recal_valid < ref_ofv)
    n_ref_better_than_ref = np.sum(ofv_ref_valid < ref_ofv)
    
    ax.set_xlabel('OFV with Reference Parameters', fontsize=12)
    ax.set_ylabel('OFV with Recalibrated Parameters', fontsize=12)
    ax.set_title('OFV Comparison: Reference Params vs Recalibrated\n'
                 f'(Using Perturbed Precipitation Inputs)', fontsize=13)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Add statistics text box
    stats_text = (
        f'Recalibration improved: {n_improved}/{len(ofv_ref_valid)} ({100*n_improved/len(ofv_ref_valid):.1f}%)\n'
        f'Recalibration worsened: {n_worsened}/{len(ofv_ref_valid)} ({100*n_worsened/len(ofv_ref_valid):.1f}%)\n'
        f'Better than reference OFV:\n'
        f'  - With ref params: {n_ref_better_than_ref} ({100*n_ref_better_than_ref/len(ofv_ref_valid):.1f}%)\n'
        f'  - After recalib: {n_better_than_ref} ({100*n_better_than_ref/len(ofv_ref_valid):.1f}%)'
    )
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
            family='monospace')
    
    # Add secondary axes for NSE
    ax2 = ax.secondary_xaxis('top', functions=(lambda x: 1-x, lambda x: 1-x))
    ax2.set_xlabel('NSE with Reference Parameters', fontsize=10, color='darkgreen')
    ax3 = ax.secondary_yaxis('right', functions=(lambda y: 1-y, lambda y: 1-y))
    ax3.set_ylabel('NSE with Recalibrated Parameters', fontsize=10, color='darkgreen')
    
    plt.tight_layout()
    fig.savefig(out_dir / 'scatter_ofv_ref_vs_recalibrated.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  Saved: scatter_ofv_ref_vs_recalibrated.png")
    
    return {
        'n_improved': n_improved,
        'n_worsened': n_worsened,
        'n_better_than_ref': n_better_than_ref,
        'n_ref_better_than_ref': n_ref_better_than_ref,
        'total': len(ofv_ref_valid)
    }


def save_results_summary(ref_params, ref_ofv, ofv_reference, ofv_recalibrated,
                          recalib_params_all, marc_values, out_dir, args,
                          has_recalibration=True, scatter_stats=None):
    """Save summary statistics to text file with assignment questions answered."""
    
    valid_ref = ofv_reference[ofv_reference < PENALTY]
    
    # Count how often results got better by chance (using reference params)
    n_better_by_chance_ref = np.sum(valid_ref < ref_ofv)
    pct_better_by_chance_ref = 100 * n_better_by_chance_ref / len(valid_ref) if len(valid_ref) > 0 else 0
    
    with open(out_dir / 'uncertainty_analysis_summary.txt', 'w') as f:
        f.write("Assignment 4: Model Input Uncertainty Analysis\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Number of perturbed series: {args.n}\n")
        f.write(f"  Perturbation: C = N({PERTURB_MEAN}, {PERTURB_STD})\n")
        f.write(f"  Multiplier clipping [0.75, 1.25]: {args.clip_multipliers}\n")
        f.write(f"  Recalibration enabled: {args.recalibrate}\n")
        if args.recalibrate:
            f.write(f"  DE iterations: {args.maxiter}\n")
            f.write(f"  DE population size: {args.popsize}\n")
        f.write(f"  Random seed: {args.seed}\n\n")
        
        f.write("Reference Calibration (from Assignment 1):\n")
        f.write(f"  Reference OFV: {ref_ofv:.6f}\n")
        f.write(f"  Reference NSE: {1-ref_ofv:.6f}\n\n")
        
        f.write("Results with Reference Parameters:\n")
        f.write(f"  Samples below penalty threshold: {len(valid_ref)} / {args.n}\n")
        if len(valid_ref) > 0:
            f.write(f"  OFV mean: {np.mean(valid_ref):.6f}\n")
            f.write(f"  OFV std: {np.std(valid_ref):.6f}\n")
            f.write(f"  OFV min: {np.min(valid_ref):.6f}\n")
            f.write(f"  OFV max: {np.max(valid_ref):.6f}\n")
            f.write(f"  OFV 5th percentile: {np.percentile(valid_ref, 5):.6f}\n")
            f.write(f"  OFV 95th percentile: {np.percentile(valid_ref, 95):.6f}\n")
            f.write(f"  NSE mean: {1-np.mean(valid_ref):.6f}\n")
            f.write(f"  Better than reference OFV: {n_better_by_chance_ref}/{len(valid_ref)} ({pct_better_by_chance_ref:.1f}%)\n\n")
        else:
            f.write("  WARNING: All samples at or above penalty threshold\n\n")
        
        if has_recalibration and ofv_recalibrated is not None:
            # Filter valid samples (exclude NaN and >= PENALTY)
            valid_mask_recalib = ~np.isnan(ofv_recalibrated) & (ofv_recalibrated < PENALTY)
            valid_recalib = ofv_recalibrated[valid_mask_recalib]
            # Use same validity mask for params (not just NaN check on params)
            valid_params = recalib_params_all[valid_mask_recalib]
            
            # Count how often recalibration is better than reference
            n_better_by_chance_recal = np.sum(valid_recalib < ref_ofv)
            pct_better_by_chance_recal = 100 * n_better_by_chance_recal / len(valid_recalib) if len(valid_recalib) > 0 else 0
            
            f.write("Results with Recalibration:\n")
            f.write(f"  Samples below penalty threshold: {len(valid_recalib)} / {args.n}\n")
            n_at_penalty = args.n - len(valid_recalib)
            if n_at_penalty > 0:
                f.write(f"  Samples at/above penalty threshold (OFV >= {PENALTY}): {n_at_penalty}\n")
            
            if len(valid_recalib) > 0:
                f.write(f"  OFV mean: {np.mean(valid_recalib):.6f}\n")
                f.write(f"  OFV std: {np.std(valid_recalib):.6f}\n")
                f.write(f"  OFV min: {np.min(valid_recalib):.6f}\n")
                f.write(f"  OFV max: {np.max(valid_recalib):.6f}\n")
                f.write(f"  OFV 5th percentile: {np.percentile(valid_recalib, 5):.6f}\n")
                f.write(f"  OFV 95th percentile: {np.percentile(valid_recalib, 95):.6f}\n")
                f.write(f"  NSE mean: {1-np.mean(valid_recalib):.6f}\n")
                f.write(f"  Better than reference OFV: {n_better_by_chance_recal}/{len(valid_recalib)} ({pct_better_by_chance_recal:.1f}%)\n\n")
                
                f.write("Parameter Statistics (Recalibrated):\n")
                f.write("-" * 70 + "\n")
                f.write(f"{'Parameter':<12} {'Reference':>12} {'Mean':>12} {'Std':>12} {'5th%':>10} {'95th%':>10}\n")
                f.write("-" * 70 + "\n")
                for i in range(18):
                    if i == 0:
                        continue
                    vals = valid_params[:, i]
                    f.write(f"{PARAM_NAMES[i]:<12} {ref_params[i]:>12.4f} "
                            f"{np.mean(vals):>12.4f} {np.std(vals):>12.4f} "
                            f"{np.percentile(vals, 5):>10.4f} {np.percentile(vals, 95):>10.4f}\n")
                f.write("-" * 70 + "\n\n")
            else:
                f.write("  WARNING: All recalibration samples at or above penalty threshold\n\n")
        
        f.write("Mean Absolute Relative PPT Change:\n")
        f.write(f"  Mean: {np.mean(marc_values)*100:.2f}%\n")
        f.write(f"  Std: {np.std(marc_values)*100:.2f}%\n")
        f.write(f"  Min: {np.min(marc_values)*100:.2f}%\n")
        f.write(f"  Max: {np.max(marc_values)*100:.2f}%\n\n")
        
        # =================================================================
        # ASSIGNMENT QUESTIONS (from lecture slides)
        # =================================================================
        f.write("=" * 70 + "\n")
        f.write("ASSIGNMENT QUESTIONS & ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        
        # Q1: How often did the results get better by chance?
        f.write("Q1: How often did the results get better by chance (compared to reference)?\n")
        f.write("-" * 70 + "\n")
        f.write(f"  With reference parameters: {n_better_by_chance_ref}/{len(valid_ref)} "
                f"({pct_better_by_chance_ref:.1f}%)\n")
        if has_recalibration and ofv_recalibrated is not None:
            f.write(f"  After recalibration: {n_better_by_chance_recal}/{len(valid_recalib)} "
                    f"({pct_better_by_chance_recal:.1f}%)\n")
            if scatter_stats:
                f.write(f"  Recalibration improved OFV: {scatter_stats['n_improved']}/{scatter_stats['total']} "
                        f"({100*scatter_stats['n_improved']/scatter_stats['total']:.1f}%)\n")
        f.write("\n")
        
        # Q2: How much can recalibration compensate for?
        f.write("Q2: How much can recalibration compensate for input uncertainty?\n")
        f.write("-" * 70 + "\n")
        if has_recalibration and ofv_recalibrated is not None:
            mean_ofv_ref_params = np.mean(valid_ref)
            mean_ofv_recalib = np.mean(valid_recalib)
            
            # Compute loss metrics
            mean_loss_refparams = mean_ofv_ref_params - ref_ofv
            mean_loss_recalib = mean_ofv_recalib - ref_ofv
            compensation = mean_ofv_ref_params - mean_ofv_recalib
            
            f.write(f"  Reference OFV (original data): {ref_ofv:.4f}\n")
            f.write(f"  Mean OFV with reference params (perturbed): {mean_ofv_ref_params:.4f}\n")
            f.write(f"  Mean OFV after recalibration (perturbed): {mean_ofv_recalib:.4f}\n\n")
            
            f.write(f"  Mean loss (ref params on perturbed vs original): {mean_loss_refparams:.4f}\n")
            f.write(f"  Mean loss (recalib on perturbed vs original): {mean_loss_recalib:.4f}\n")
            f.write(f"  Compensation (improvement from recalibration): {compensation:.4f}\n")
            
            # Compensation fraction - use adaptive threshold to avoid misleading ratios
            # Threshold scales with the spread of reference OFV distribution
            ref_spread = np.percentile(valid_ref, 95) - np.percentile(valid_ref, 5)
            DENOM_EPS = max(1e-3, 0.05 * ref_spread)  # 5% of ref 90% range, min 0.001
            
            if abs(mean_loss_refparams) < DENOM_EPS:
                f.write(f"  Compensation fraction: N/A (input-induced mean loss ~ 0, ratio unstable)\n")
                f.write(f"    Note: Because the perturbation caused negligible mean performance loss,\n")
                f.write(f"    the improvement after recalibration ({compensation:.4f}) cannot be interpreted\n")
                f.write(f"    as 'compensation' for input uncertainty; it more likely reflects a better\n")
                f.write(f"    optimum found by the optimizer (relative to the Assignment-1 reference)\n")
                f.write(f"    and/or series-specific overfitting.\n")
            else:
                comp_fraction = compensation / mean_loss_refparams
                f.write(f"  Compensation fraction: {comp_fraction:.2%} of input-induced loss recovered\n")
            
            # Summary interpretation
            if mean_ofv_recalib < mean_ofv_ref_params:
                f.write(f"\n  => Recalibration IMPROVES mean performance on perturbed inputs.\n")
            else:
                f.write(f"\n  => Recalibration does NOT improve (may indicate under-convergence).\n")
            
            # Range analysis
            ref_range = np.percentile(valid_ref, 95) - np.percentile(valid_ref, 5)
            recal_range = np.percentile(valid_recalib, 95) - np.percentile(valid_recalib, 5)
            f.write(f"  OFV 90% range (5th-95th): Reference params={ref_range:.4f}, Recalibrated={recal_range:.4f}\n")
        else:
            f.write("  Recalibration not performed in this run.\n")
        f.write("\n")
        
        # Q3: Does it make sense to use input uncertainty during calibration?
        f.write("Q3: Does it make sense to use input uncertainty during calibration?\n")
        f.write("-" * 70 + "\n")
        f.write("  Analysis:\n")
        if has_recalibration and ofv_recalibrated is not None:
            # Compare parameter uncertainty using IQR (robust metric)
            param_uncertainty = []
            for i in range(1, 18):  # Skip snw_dth
                vals = valid_params[:, i]
                ref_val = ref_params[i]
                p5, p95 = np.percentile(vals, [5, 95])
                iqr_90 = p95 - p5  # 90% interquartile range
                abs_std = np.std(vals)
                # Relative std (with warning for near-zero ref)
                if abs(ref_val) > 0.01:
                    rel_std = abs_std / abs(ref_val) * 100
                    rel_std_str = f"{rel_std:.1f}%"
                else:
                    rel_std_str = "N/A (ref≈0)"
                param_uncertainty.append((PARAM_NAMES[i], iqr_90, abs_std, rel_std_str, ref_val))
            
            # Sort by IQR (robust measure of spread)
            param_uncertainty.sort(key=lambda x: x[1], reverse=True)
            
            f.write("  - Most uncertain parameters (ranked by IQR = P95 - P5):\n")
            f.write(f"    {'Parameter':<12} {'IQR':>10} {'Std':>10} {'Rel.Std':>12} {'Ref.Val':>10}\n")
            for name, iqr, std, rel_std_str, ref_val in param_uncertainty[:5]:
                f.write(f"    {name:<12} {iqr:>10.4f} {std:>10.4f} {rel_std_str:>12} {ref_val:>10.4f}\n")
            
            f.write("\n  Conclusion:\n")
            f.write("  - Input uncertainty causes significant parameter uncertainty.\n")
            f.write("  - Accounting for input uncertainty during calibration could:\n")
            f.write("    * Provide more robust parameter estimates\n")
            f.write("    * Better quantify prediction uncertainty bounds\n")
            f.write("    * Identify parameters that are most affected by input errors\n")
        else:
            f.write("  Recalibration needed to fully answer this question.\n")
        f.write("\n")
        
        # Q4: Difference to Oudin et al. 2006
        f.write("Q4: Difference to Oudin et al. 2006?\n")
        f.write("-" * 70 + "\n")
        f.write("  Oudin et al. 2006 study:\n")
        f.write("  - Analyzed biased AND randomly corrupted inputs\n")
        f.write("  - Used multiple rainfall-runoff models (GR4J, TOPMO, HBV)\n")
        f.write("  - Found that model efficiency is more affected by random errors\n")
        f.write("    than by systematic bias\n")
        f.write("  - Showed that recalibration can partially compensate for input errors\n")
        f.write("\n")
        f.write("  Our analysis (Assignment 4):\n")
        f.write("  - Uses only random perturbations C = N(1, 0.05) (no systematic bias)\n")
        f.write("  - Single model (HBV001A)\n")
        f.write("  - Similar findings expected: random noise affects model performance\n")
        f.write("  - Recalibration can compensate to some extent by adjusting parameters\n")
        f.write("\n")
    
    print("  Saved: uncertainty_analysis_summary.txt")


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Assignment 4: Model Input Uncertainty Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast test (no recalibration)
  python %(prog)s --n 100 --no-recalibrate
  
  # Fast recalibration test
  python %(prog)s --n 50 --recalibrate --maxiter 20 --popsize 8
  
  # Full run (2000 perturbations, multiplier clipping ON by default)
  python %(prog)s --n 2000 --recalibrate --maxiter 40 --popsize 10
  
  # Disable multiplier clipping (NOT recommended for assignment)
  python %(prog)s --n 2000 --recalibrate --no-clip-multipliers
        """
    )
    
    parser.add_argument('--n', type=int, default=2000,
                        help='Number of perturbed series (default: 2000)')
    parser.add_argument('--recalibrate', action='store_true', default=True,
                        help='Enable recalibration (computationally heavy)')
    parser.add_argument('--no-recalibrate', action='store_true', default=False,
                        help='Disable recalibration (only reference params)')
    parser.add_argument('--maxiter', type=int, default=600,
                        help='DE max iterations for recalibration (default: 100)')
    parser.add_argument('--popsize', type=int, default=10,
                        help='DE population size for recalibration (default: 8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--no-clip-multipliers', action='store_true', default=False,
                        help='Disable multiplier clipping. Default enforces [0.75, 1.25] per assignment.')
    parser.add_argument('--jobs', type=int, default=4,
                        help='Number of parallel jobs (default: 4)')
    parser.add_argument('--diagnostic-k', type=int, default=0,
                        help='Run diagnostic convergence test on K random series with higher DE settings.')
    parser.add_argument('--warmstart', action='store_true', default=False,
                        help='Include reference parameters in DE initial population (warm start).')
    
    args = parser.parse_args()
    
    # Handle recalibrate/no-recalibrate logic
    if args.no_recalibrate:
        args.recalibrate = False
    
    # Handle clip multipliers (default is ON, use --no-clip-multipliers to disable)
    args.clip_multipliers = not args.no_clip_multipliers
    
    return args


def ensure_directory_exists(dir_path):
    """
    Robustly create directory, handling OneDrive sync issues.
    """
    dir_path = Path(dir_path)
    for attempt in range(3):
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            # Verify it actually exists
            if dir_path.exists() and dir_path.is_dir():
                return True
            time.sleep(0.5)  # Brief wait for filesystem sync
        except Exception as e:
            print(f"  Directory creation attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(1.0)
    
    # Final fallback: use os.makedirs
    try:
        os.makedirs(str(dir_path), exist_ok=True)
        return dir_path.exists()
    except Exception as e:
        print(f"  FATAL: Could not create directory {dir_path}: {e}", flush=True)
        return False


def load_checkpoint(out_dir, n_total, has_recalibration=False):
    """
    Load existing results CSV if it exists for checkpoint/resume.
    
    Returns
    -------
    completed_ref : set
        Series IDs that have reference param evaluation completed
    completed_recalib : set
        Series IDs that have recalibration completed (if applicable)
    checkpoint_df : DataFrame or None
        The loaded checkpoint data
        
    Note: Counts are printed in main() after intersection with current run range
    """
    csv_path = out_dir / 'uncertainty_results.csv'
    if not csv_path.exists():
        return set(), set(), None
    
    try:
        df = pd.read_csv(csv_path)
        
        # Reference evaluation: any row with valid series_id and ofv_reference_params
        completed_ref = set()
        completed_recalib = set()
        
        for _, row in df.iterrows():
            sid = int(row['series_id'])
            # Check if reference eval is done
            if pd.notna(row.get('ofv_reference_params', np.nan)):
                completed_ref.add(sid)
            # Check if recalibration is done
            if has_recalibration and pd.notna(row.get('ofv_recalibrated', np.nan)):
                completed_recalib.add(sid)
        
        # Just note that checkpoint was loaded (counts will be shown after intersection)
        print(f"  [CHECKPOINT] Loaded from: {csv_path.name}", flush=True)
        
        return completed_ref, completed_recalib, df
    except Exception as e:
        print(f"  [CHECKPOINT] Could not load: {e}", flush=True)
        return set(), set(), None


def save_checkpoint(out_dir, series_ids, marc_values, ofv_reference, 
                    ofv_recalibrated=None, recalib_params_all=None):
    """
    Save checkpoint with correct series_id mapping.
    
    Parameters
    ----------
    series_ids : array-like
        Actual series indices (may be non-contiguous if resuming)
    """
    results_dict = {
        'series_id': series_ids,
        'mean_abs_rel_change': marc_values,
        'ofv_reference_params': ofv_reference,
    }
    
    if ofv_recalibrated is not None:
        results_dict['ofv_recalibrated'] = ofv_recalibrated
        if recalib_params_all is not None:
            for i, name in enumerate(PARAM_NAMES):
                results_dict[f'recalib_{name}'] = recalib_params_all[:, i]
    
    df = pd.DataFrame(results_dict)
    df.to_csv(out_dir / 'uncertainty_results.csv', index=False)
    return df


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main execution function."""
    
    # Parse arguments
    args = parse_arguments()
    
    print("\n" + "=" * 70, flush=True)
    print("ASSIGNMENT 4: MODEL INPUT UNCERTAINTY ANALYSIS", flush=True)
    print("=" * 70, flush=True)
    
    # Print backend and environment info
    print(f"\n[ENVIRONMENT]", flush=True)
    print(f"  Backend: {BACKEND_INFO}", flush=True)
    print(f"  Platform: {platform.system()} {platform.release()}", flush=True)
    print(f"  Python: {platform.python_version()}", flush=True)
    
    # Print configuration
    print(f"\n[CONFIGURATION]", flush=True)
    print(f"  Number of perturbations: {args.n}", flush=True)
    print(f"  Recalibration: {'ENABLED' if args.recalibrate else 'DISABLED'}", flush=True)
    if args.recalibrate:
        print(f"  DE maxiter: {args.maxiter}", flush=True)
        print(f"  DE popsize: {args.popsize}", flush=True)
    print(f"  Multiplier clipping [0.75, 1.25]: {args.clip_multipliers}", flush=True)
    print(f"  Random seed: {args.seed}", flush=True)
    print(f"  Parallel jobs: {args.jobs}", flush=True)
    
    start_time = time.time()
    
    # Output directory - robust creation
    out_dir = Path(__file__).resolve().parent / 'outputs' / 'assignment4'
    print(f"\n[DIRECTORY SETUP]", flush=True)
    print(f"  Target: {out_dir}", flush=True)
    if not ensure_directory_exists(out_dir):
        print("  FATAL: Could not create output directory. Exiting.", flush=True)
        sys.exit(1)
    print(f"  Directory ready: {out_dir.exists()}", flush=True)
    
    # Load data
    print("\n[LOADING DATA]", flush=True)
    temps, precip, pet, obs_q, dslr, ref_params, time_index = load_data()
    n_steps = len(temps)
    print(f"  Time steps: {n_steps}", flush=True)
    
    # Compute reference OFV
    print("\n[REFERENCE CALIBRATION]", flush=True)
    ref_sim = run_model_with_params(ref_params, temps, precip, pet, dslr, n_steps, verbose=True)
    ref_nse = compute_nse(obs_q, ref_sim)
    # Clamp OFV to [0, PENALTY] for consistency
    ref_ofv = np.clip(1.0 - ref_nse, 0.0, PENALTY) if not np.isnan(ref_nse) else PENALTY
    print(f"  Reference OFV: {ref_ofv:.6f} (NSE: {ref_nse:.6f})", flush=True)
    
    # Create perturbed precipitation series
    print(f"\n[PERTURBATION GENERATION]", flush=True)
    print(f"  Creating {args.n} perturbed precipitation series...", flush=True)
    print(f"  Perturbation: C = N({PERTURB_MEAN}, {PERTURB_STD})", flush=True)
    clip_mult = args.clip_multipliers
    print(f"  Multiplier clipping: {clip_mult}")
    
    perturbed_ppt, multipliers = create_perturbed_series(
        precip, args.n, PERTURB_MEAN, PERTURB_STD, 
        clip_multipliers=clip_mult, seed=args.seed
    )
    print(f"  Done. Shape: {perturbed_ppt.shape}", flush=True)
    
    # Plot PPT CDF comparison
    print("\n[PPT CDF VERIFICATION]", flush=True)
    plot_ppt_cdf_comparison(precip, perturbed_ppt, out_dir)
    plot_ppt_cdf_comparison_two(precip, perturbed_ppt, out_dir)
    
    # Check for existing checkpoint (early, for reference eval)
    print(f"\n[REFERENCE PARAMETER EVALUATION]", flush=True)
    completed_ref_early, completed_recalib_early, checkpoint_df_early = load_checkpoint(
        out_dir, args.n, has_recalibration=args.recalibrate
    )
    
    # Compute intersection with current run range
    indices_in_run = set(range(args.n))
    completed_ref_in_run = completed_ref_early & indices_in_run
    
    # Initialize results array with NaN (for checkpoint logic - NaN = not yet computed)
    ofv_reference = np.full(args.n, np.nan, dtype=float)
    
    # Restore from checkpoint
    if checkpoint_df_early is not None and len(completed_ref_in_run) > 0:
        for _, row in checkpoint_df_early.iterrows():
            sid = int(row['series_id'])
            if sid in completed_ref_in_run:
                ofv_reference[sid] = row['ofv_reference_params']
    
    # Determine which indices still need reference evaluation
    indices_to_eval_ref = [i for i in range(args.n) if i not in completed_ref_in_run]
    n_to_eval = len(indices_to_eval_ref)
    
    if n_to_eval < args.n:
        print(f"  [CHECKPOINT] Skipping {args.n - n_to_eval}/{args.n} (already done)", flush=True)
    
    if n_to_eval > 0:
        print(f"  Evaluating {n_to_eval} series with reference parameters...", flush=True)
        print(f"  Using {args.jobs} parallel workers...", flush=True)
        eval_start = time.time()
        
        results_ref = Parallel(n_jobs=args.jobs, verbose=1)(
            delayed(evaluate_single_series_reference)(
                i, perturbed_ppt[i], temps, pet, obs_q, dslr, ref_params
            ) for i in indices_to_eval_ref
        )
        
        # Collect results
        for idx, ofv in results_ref:
            ofv_reference[idx] = ofv
        
        eval_elapsed = time.time() - eval_start
        print(f"  Elapsed: {eval_elapsed:.1f}s ({eval_elapsed/n_to_eval:.2f}s per series)", flush=True)
    else:
        print(f"  All {args.n} series already evaluated (from checkpoint)", flush=True)
    
    # Count valid samples (NaN comparisons return False, so NaN entries excluded)
    valid_count = np.sum(~np.isnan(ofv_reference) & (ofv_reference < PENALTY))
    print(f"  Samples below penalty threshold: {valid_count} / {args.n}", flush=True)
    
    # Run diagnostic convergence test if requested
    if args.diagnostic_k > 0:
        run_diagnostic_convergence_test(
            args.diagnostic_k, perturbed_ppt, temps, pet, obs_q, 
            dslr, ref_params, args, out_dir
        )
    
    # Initialize recalibration arrays
    recalib_params_all = None
    ofv_recalibrated = None
    
    # Recalibrate for each perturbed series (if enabled)
    if args.recalibrate:
        print(f"\n[RECALIBRATION PHASE]", flush=True)
        print(f"  Recalibrating {args.n} series...", flush=True)
        print(f"  DE settings: maxiter={args.maxiter}, popsize={args.popsize}", flush=True)
        print(f"  Using {args.jobs} parallel workers...", flush=True)
        print(f"  Warm-start from reference: {args.warmstart}", flush=True)
        print(f"  ESTIMATED TIME: {args.n * args.maxiter * args.popsize * 0.05 / args.jobs / 60:.1f} minutes (rough)", flush=True)
        
        recalib_start = time.time()
        
        # Generate seeds for reproducibility
        rng = np.random.default_rng(args.seed + 1000)
        seeds = rng.integers(0, 2**31, size=args.n)
        
        # Reference parameters (excluding fixed snw_dth) for warm-start
        ref_params_variable = ref_params[1:].copy()
        
        # Reuse checkpoint from earlier (completed_ref_in_run, completed_recalib_early)
        completed_recalib_in_run = completed_recalib_early & indices_in_run
        
        # Use batch processing with checkpointing for large runs
        batch_size = min(25, args.n)  # Checkpoint every 25 runs
        recalib_params_all = np.full((args.n, 18), np.nan, dtype=np.float32)
        ofv_recalibrated = np.full(args.n, np.nan)
        
        # Resume from checkpoint if available
        if checkpoint_df_early is not None and len(completed_recalib_in_run) > 0:
            for _, row in checkpoint_df_early.iterrows():
                sid = int(row['series_id'])
                if sid in completed_recalib_in_run:
                    ofv_recalibrated[sid] = row['ofv_recalibrated']
                    for i, name in enumerate(PARAM_NAMES):
                        recalib_params_all[sid, i] = row.get(f'recalib_{name}', np.nan)
        
        # Get indices still to process
        indices_to_process = [i for i in range(args.n) if i not in completed_recalib_in_run]
        n_to_process = len(indices_to_process)
        
        if n_to_process < args.n:
            print(f"  [CHECKPOINT] Skipping {args.n - n_to_process}/{args.n} recalibrations (already done)", flush=True)
        
        if n_to_process > 0:
            completed = args.n - n_to_process
            for batch_start in range(0, n_to_process, batch_size):
                batch_end = min(batch_start + batch_size, n_to_process)
                batch_indices = indices_to_process[batch_start:batch_end]
                
                print(f"  [BATCH {batch_start//batch_size + 1}] Processing {len(batch_indices)} series...", flush=True)
                batch_start_time = time.time()
                
                results_batch = Parallel(n_jobs=args.jobs, verbose=10)(
                    delayed(calibrate_single_series)(
                        i, perturbed_ppt[i], temps, pet, obs_q, dslr,
                        args.maxiter, args.popsize, seeds[i],
                        ref_params_variable=ref_params_variable,
                        warmstart=args.warmstart
                    ) for i in batch_indices
                )
                
                # Collect batch results
                for idx, params, ofv in results_batch:
                    recalib_params_all[idx] = params
                    ofv_recalibrated[idx] = ofv
                
                completed += len(batch_indices)
                batch_time = time.time() - batch_start_time
                total_elapsed = time.time() - recalib_start
                rate = completed / total_elapsed if total_elapsed > 0 else 0
                remaining = (args.n - completed) / rate if rate > 0 else 0
                
                print(f"  [PROGRESS] {completed}/{args.n} done ({100*completed/args.n:.1f}%) | "
                       f"Batch: {batch_time:.1f}s | ETA: {remaining/60:.1f} min", flush=True)
                
                # Save checkpoint after each batch (with correct series_id mapping)
                if completed < args.n:
                    # Build checkpoint with ALL completed series (correct IDs)
                    # Use ~np.isnan to detect completed (more robust than !=0)
                    completed_ids = [i for i in range(args.n) if not np.isnan(ofv_recalibrated[i]) or i in completed_recalib_in_run]
                    if len(completed_ids) > 0:
                        marc_completed = [compute_mean_abs_relative_change(precip, perturbed_ppt[i]) for i in completed_ids]
                        save_checkpoint(
                            out_dir, completed_ids, marc_completed,
                            ofv_reference[completed_ids],
                            ofv_recalibrated[completed_ids],
                            recalib_params_all[completed_ids]
                        )
        else:
            print(f"  All {args.n} recalibrations already done (from checkpoint)", flush=True)
        
        recalib_elapsed = time.time() - recalib_start
        # Count valid samples (exclude NaN and >= PENALTY)
        valid_recalib = np.sum(~np.isnan(ofv_recalibrated) & (ofv_recalibrated < PENALTY))
        print(f"  Samples below penalty threshold: {valid_recalib} / {args.n}", flush=True)
        print(f"  Total recalibration time: {recalib_elapsed/60:.1f} minutes", flush=True)
    
    # Generate plots
    print("\n[GENERATING PLOTS]", flush=True)
    
    # Parameter CDFs (only if recalibration done)
    if args.recalibrate and recalib_params_all is not None:
        print("  Creating parameter CDFs...", flush=True)
        plot_parameter_cdfs(ref_params, recalib_params_all, ofv_recalibrated, out_dir)
    
    # OFV CDFs
    print("  Creating OFV CDFs...", flush=True)
    plot_ofv_cdfs(ref_ofv, ofv_reference, ofv_recalibrated, out_dir, 
                  has_recalibration=args.recalibrate)
    
    # Scatter plot: MARC vs OFV
    print("  Creating MARC vs OFV scatter...", flush=True)
    marc_values = plot_scatter_relative_change_vs_ofv(
        precip, perturbed_ppt, ofv_reference, ofv_recalibrated, ref_ofv, out_dir,
        has_recalibration=args.recalibrate
    )
    
    # Scatter plot: OFV Reference vs Recalibrated (only if recalibration done)
    scatter_stats = None
    if args.recalibrate and ofv_recalibrated is not None:
        print("  Creating OFV scatter (ref vs recalibrated)...", flush=True)
        scatter_stats = plot_scatter_ofv_original_vs_recalibrated(
            ofv_reference, ofv_recalibrated, ref_ofv, out_dir
        )
    
    # Save results
    print("\n[SAVING RESULTS]", flush=True)
    
    # Save raw results to CSV
    results_df = pd.DataFrame({
        'series_id': range(args.n),
        'mean_abs_rel_change': marc_values,
        'ofv_reference_params': ofv_reference,
    })
    
    # Add recalibrated results if available
    if args.recalibrate and ofv_recalibrated is not None:
        results_df['ofv_recalibrated'] = ofv_recalibrated
        for i, name in enumerate(PARAM_NAMES):
            results_df[f'recalib_{name}'] = recalib_params_all[:, i]
    
    results_df.to_csv(out_dir / 'uncertainty_results.csv', index=False)
    print("  Saved: uncertainty_results.csv", flush=True)
    
    # Save summary with analysis questions
    save_results_summary(ref_params, ref_ofv, ofv_reference, ofv_recalibrated,
                          recalib_params_all, marc_values, out_dir, args,
                          has_recalibration=args.recalibrate, 
                          scatter_stats=scatter_stats)
    
    # Final summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS COMPLETE", flush=True)
    print("=" * 70, flush=True)
    print(f"Total runtime: {elapsed/60:.1f} minutes ({elapsed:.1f} seconds)", flush=True)
    print(f"Output directory: {out_dir}", flush=True)
    print("\nOutput files:", flush=True)
    for f in sorted(out_dir.glob('*')):
        print(f"  {f.name}", flush=True)
    
    print(f"\n#### Done on {datetime.now().strftime('%c')}", flush=True)
    print(f"Total runtime: {elapsed:.2f} seconds ####\n", flush=True)

if __name__ == '__main__':
    main()
