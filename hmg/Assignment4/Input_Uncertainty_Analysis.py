"""
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
# Fast test (no recalibration)
python hmg/Assignment4/Input_Uncertainty_Analysis.py --n 100 --no-recalibrate

# Fast recalibration test
python hmg/Assignment4/Input_Uncertainty_Analysis.py --n 50 --recalibrate --maxiter 20 --popsize 8

# Full run (2000 perturbations with recalibration)
python hmg/Assignment4/Input_Uncertainty_Analysis.py --n 2000 --recalibrate --maxiter 40 --popsize 10

# Enable multiplier clipping [0.75, 1.25] (optional, off by default)
python hmg/Assignment4/Input_Uncertainty_Analysis.py --n 2000 --clip-multipliers
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

# Suppress excessive model runtime prints
warnings.filterwarnings('ignore')


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
PERTURB_STD = 0.05  # Results in ~5% std, up to ~25% at 3-sigma

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
    
    # Data paths
    data_dir = Path(r'D:\Python Projects\hmg\data')
    out_dir_a1 = Path(__file__).resolve().parent.parent.parent / 'outputs' / 'assignment1'
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
                            clip_multipliers=False, seed=None):
    """
    Create n_series perturbed precipitation series.
    
    Each value is scaled by C = N(mean, std).
    All values are kept positive (clipped to 0).
    
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
        If True, clip multipliers to [0.75, 1.25] (up to 25% change)
        If False, use pure Gaussian (default - can exceed 25% but rare)
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    perturbed : array of shape (n_series, len(precip))
        Perturbed precipitation series
    multipliers : array of shape (n_series, len(precip))
        The random multipliers used
    """
    rng = np.random.default_rng(seed)
    
    n_timesteps = len(precip)
    
    # Generate random multipliers C = N(mean, std)
    multipliers = rng.normal(mean, std, size=(n_series, n_timesteps))
    
    # Optionally clip multipliers (off by default - pure Gaussian)
    if clip_multipliers:
        multipliers = np.clip(multipliers, 0.75, 1.25)
    
    # Apply to precipitation
    perturbed = precip[np.newaxis, :] * multipliers
    
    # Keep all positive
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
    # Use optimization flag 1 to suppress verbose output during batch runs
    model.set_optimization_flag(1 if not verbose else 0)
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
        
        ofv = 1.0 - nse
        return min(ofv, PENALTY)
        
    except Exception:
        return PENALTY


def calibrate_model(temps, precip, pet, obs_q, dslr, maxiter, popsize, seed=None):
    """
    Calibrate model using Differential Evolution.
    
    Returns best parameters and best OFV.
    """
    n_steps = len(temps)
    
    # Variable bounds (excluding fixed snw_dth at index 0)
    var_bounds = [PARAM_BOUNDS[i] for i in range(1, 18)]
    
    result = differential_evolution(
        objective_function,
        var_bounds,
        args=(temps, precip, pet, obs_q, dslr, n_steps),
        strategy='best1bin',
        maxiter=maxiter,
        popsize=popsize,
        tol=0.001,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=seed,
        workers=1,  # Single worker since we parallelize at series level
        updating='deferred',
        disp=False
    )
    
    # Reconstruct full parameters
    best_params = np.zeros(18, dtype=np.float32)
    best_params[0] = 0.0
    best_params[1:] = result.x
    
    return best_params, result.fun


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
        ofv = 1.0 - nse if not np.isnan(nse) else PENALTY
    except Exception:
        ofv = PENALTY
    
    return idx, ofv


def calibrate_single_series(idx, perturbed_precip, temps, pet, obs_q, dslr, 
                            maxiter, popsize, seed):
    """Calibrate model for a single perturbed series."""
    
    try:
        best_params, best_ofv = calibrate_model(temps, perturbed_precip, pet, 
                                                 obs_q, dslr, maxiter, popsize, 
                                                 seed=seed)
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
    Shows envelope and sample of perturbed CDFs with original as thick line.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # Plot subset of perturbed CDFs for visibility
    n_plot = min(100, len(perturbed_all))
    indices = np.linspace(0, len(perturbed_all)-1, n_plot, dtype=int)
    
    for i in indices:
        sorted_vals = np.sort(perturbed_all[i])
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf, color='blue', alpha=0.1, linewidth=0.5)
    
    # Plot original CDF
    sorted_orig = np.sort(original)
    cdf_orig = np.arange(1, len(sorted_orig) + 1) / len(sorted_orig)
    ax.plot(sorted_orig, cdf_orig, color='black', linewidth=2, 
            label='Original PPT')
    
    ax.set_xlabel('Precipitation [mm/hr]', fontsize=11)
    ax.set_ylabel('Cumulative Probability', fontsize=11)
    ax.set_title('CDF Comparison: Original vs Perturbed Precipitation\n'
                 f'({n_plot} of {len(perturbed_all)} perturbed series shown)',
                 fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    fig.savefig(out_dir / 'ppt_cdf_comparison.png', dpi=150)
    plt.close(fig)
    print("  Saved: ppt_cdf_comparison.png")


def plot_parameter_cdfs(ref_params, recalib_params_all, out_dir):
    """
    Plot CDF of each recalibrated parameter with reference value marked.
    """
    # Filter valid results
    valid_mask = ~np.isnan(recalib_params_all[:, 1])
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


def save_results_summary(ref_params, ref_ofv, ofv_reference, ofv_recalibrated,
                          recalib_params_all, marc_values, out_dir, args,
                          has_recalibration=True):
    """Save summary statistics to text file."""
    
    valid_ref = ofv_reference[ofv_reference < PENALTY]
    
    with open(out_dir / 'uncertainty_analysis_summary.txt', 'w') as f:
        f.write("Assignment 4: Model Input Uncertainty Analysis\n")
        f.write("=" * 60 + "\n\n")
        
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
        f.write(f"  Valid runs: {len(valid_ref)} / {args.n}\n")
        f.write(f"  OFV mean: {np.mean(valid_ref):.6f}\n")
        f.write(f"  OFV std: {np.std(valid_ref):.6f}\n")
        f.write(f"  OFV min: {np.min(valid_ref):.6f}\n")
        f.write(f"  OFV max: {np.max(valid_ref):.6f}\n")
        f.write(f"  OFV 5th percentile: {np.percentile(valid_ref, 5):.6f}\n")
        f.write(f"  OFV 95th percentile: {np.percentile(valid_ref, 95):.6f}\n")
        f.write(f"  NSE mean: {1-np.mean(valid_ref):.6f}\n\n")
        
        if has_recalibration and ofv_recalibrated is not None:
            valid_recalib = ofv_recalibrated[ofv_recalibrated < PENALTY]
            valid_params = recalib_params_all[~np.isnan(recalib_params_all[:, 1])]
            
            f.write("Results with Recalibration:\n")
            f.write(f"  Successful recalibrations: {len(valid_recalib)} / {args.n}\n")
            f.write(f"  OFV mean: {np.mean(valid_recalib):.6f}\n")
            f.write(f"  OFV std: {np.std(valid_recalib):.6f}\n")
            f.write(f"  OFV min: {np.min(valid_recalib):.6f}\n")
            f.write(f"  OFV max: {np.max(valid_recalib):.6f}\n")
            f.write(f"  OFV 5th percentile: {np.percentile(valid_recalib, 5):.6f}\n")
            f.write(f"  OFV 95th percentile: {np.percentile(valid_recalib, 95):.6f}\n")
            f.write(f"  NSE mean: {1-np.mean(valid_recalib):.6f}\n\n")
            
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
        
        f.write("Mean Absolute Relative PPT Change:\n")
        f.write(f"  Mean: {np.mean(marc_values)*100:.2f}%\n")
        f.write(f"  Std: {np.std(marc_values)*100:.2f}%\n")
        f.write(f"  Min: {np.min(marc_values)*100:.2f}%\n")
        f.write(f"  Max: {np.max(marc_values)*100:.2f}%\n")
    
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
  
  # Full run (2000 perturbations)
  python %(prog)s --n 2000 --recalibrate --maxiter 40 --popsize 10
  
  # Enable multiplier clipping [0.75, 1.25] (optional)
  python %(prog)s --n 2000 --clip-multipliers
        """
    )
    
    parser.add_argument('--n', type=int, default=2000,
                        help='Number of perturbed series (default: 2000)')
    parser.add_argument('--recalibrate', action='store_true', default=False,
                        help='Enable recalibration (computationally heavy)')
    parser.add_argument('--no-recalibrate', action='store_true', default=False,
                        help='Disable recalibration (only reference params)')
    parser.add_argument('--maxiter', type=int, default=40,
                        help='DE max iterations for recalibration (default: 40)')
    parser.add_argument('--popsize', type=int, default=10,
                        help='DE population size for recalibration (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--clip-multipliers', action='store_true', default=False,
                        help='Clip multipliers to [0.75, 1.25]. Default is OFF (pure Gaussian).')
    parser.add_argument('--jobs', type=int, default=4,
                        help='Number of parallel jobs (default: 4)')
    
    args = parser.parse_args()
    
    # Handle recalibrate/no-recalibrate logic
    if args.no_recalibrate:
        args.recalibrate = False
    
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


def load_checkpoint(out_dir, n_total):
    """
    Load existing results CSV if it exists for checkpoint/resume.
    Returns: completed_indices (set), partial_results (dict) or (empty set, None)
    """
    csv_path = out_dir / 'uncertainty_results.csv'
    if not csv_path.exists():
        return set(), None
    
    try:
        df = pd.read_csv(csv_path)
        completed = set(df['series_id'].values)
        print(f"  [CHECKPOINT] Found {len(completed)}/{n_total} completed runs", flush=True)
        return completed, df
    except Exception as e:
        print(f"  [CHECKPOINT] Could not load: {e}", flush=True)
        return set(), None


def save_partial_results(out_dir, results_df):
    """Save partial results for checkpoint."""
    try:
        results_df.to_csv(out_dir / 'uncertainty_results.csv', index=False)
    except Exception as e:
        print(f"  [CHECKPOINT SAVE ERROR] {e}", flush=True)


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
    out_dir = Path(__file__).resolve().parent.parent.parent / 'outputs' / 'assignment4'
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
    ref_ofv = 1.0 - ref_nse
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
    
    # Evaluate with reference parameters
    print(f"\n[REFERENCE PARAMETER EVALUATION]", flush=True)
    print(f"  Evaluating {args.n} series with reference parameters...", flush=True)
    print(f"  Using {args.jobs} parallel workers...", flush=True)
    eval_start = time.time()
    
    results_ref = Parallel(n_jobs=args.jobs, verbose=1)(
        delayed(evaluate_single_series_reference)(
            i, perturbed_ppt[i], temps, pet, obs_q, dslr, ref_params
        ) for i in range(args.n)
    )
    
    # Collect results
    ofv_reference = np.zeros(args.n)
    for idx, ofv in results_ref:
        ofv_reference[idx] = ofv
    
    eval_elapsed = time.time() - eval_start
    valid_count = np.sum(ofv_reference < PENALTY)
    print(f"  Valid evaluations: {valid_count} / {args.n}", flush=True)
    print(f"  Elapsed: {eval_elapsed:.1f}s ({eval_elapsed/args.n:.2f}s per series)", flush=True)
    
    # Initialize recalibration arrays
    recalib_params_all = None
    ofv_recalibrated = None
    
    # Recalibrate for each perturbed series (if enabled)
    if args.recalibrate:
        print(f"\n[RECALIBRATION PHASE]", flush=True)
        print(f"  Recalibrating {args.n} series...", flush=True)
        print(f"  DE settings: maxiter={args.maxiter}, popsize={args.popsize}", flush=True)
        print(f"  Using {args.jobs} parallel workers...", flush=True)
        print(f"  ESTIMATED TIME: {args.n * args.maxiter * args.popsize * 0.05 / args.jobs / 60:.1f} minutes (rough)", flush=True)
        
        recalib_start = time.time()
        
        # Generate seeds for reproducibility
        rng = np.random.default_rng(args.seed + 1000)
        seeds = rng.integers(0, 2**31, size=args.n)
        
        # Use batch processing with checkpointing for large runs
        batch_size = min(25, args.n)  # Checkpoint every 25 runs
        recalib_params_all = np.zeros((args.n, 18), dtype=np.float32)
        ofv_recalibrated = np.zeros(args.n)
        
        completed = 0
        for batch_start in range(0, args.n, batch_size):
            batch_end = min(batch_start + batch_size, args.n)
            batch_indices = list(range(batch_start, batch_end))
            
            print(f"  [BATCH {batch_start//batch_size + 1}] Processing series {batch_start}-{batch_end-1}...", flush=True)
            batch_start_time = time.time()
            
            results_batch = Parallel(n_jobs=args.jobs, verbose=0)(
                delayed(calibrate_single_series)(
                    i, perturbed_ppt[i], temps, pet, obs_q, dslr,
                    args.maxiter, args.popsize, seeds[i]
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
            
            # Save checkpoint after each batch
            if completed < args.n:
                # Build partial dataframe
                partial_df = pd.DataFrame({
                    'series_id': range(completed),
                    'ofv_reference_params': ofv_reference[:completed],
                    'ofv_recalibrated': ofv_recalibrated[:completed],
                })
                for i, name in enumerate(PARAM_NAMES):
                    partial_df[f'recalib_{name}'] = recalib_params_all[:completed, i]
                save_partial_results(out_dir, partial_df)
        
        recalib_elapsed = time.time() - recalib_start
        valid_recalib = np.sum(ofv_recalibrated < PENALTY)
        print(f"  Successful recalibrations: {valid_recalib} / {args.n}", flush=True)
        print(f"  Total recalibration time: {recalib_elapsed/60:.1f} minutes", flush=True)
    
    # Generate plots
    print("\n[GENERATING PLOTS]", flush=True)
    
    # Parameter CDFs (only if recalibration done)
    if args.recalibrate and recalib_params_all is not None:
        print("  Creating parameter CDFs...", flush=True)
        plot_parameter_cdfs(ref_params, recalib_params_all, out_dir)
    
    # OFV CDFs
    print("  Creating OFV CDFs...", flush=True)
    plot_ofv_cdfs(ref_ofv, ofv_reference, ofv_recalibrated, out_dir, 
                  has_recalibration=args.recalibrate)
    
    # Scatter plot
    print("  Creating MARC vs OFV scatter...", flush=True)
    marc_values = plot_scatter_relative_change_vs_ofv(
        precip, perturbed_ppt, ofv_reference, ofv_recalibrated, ref_ofv, out_dir,
        has_recalibration=args.recalibrate
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
    
    # Save summary
    save_results_summary(ref_params, ref_ofv, ofv_reference, ofv_recalibrated,
                          recalib_params_all, marc_values, out_dir, args,
                          has_recalibration=args.recalibrate)
    
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
