"""
Assignment 3: Sobol Sensitivity Analysis (Corrected & Complete)
Based on Chris's initial implementation + fixes + Saltelli et al. (2010)

Key Features:
1. Unified Sobol sampling (single sequence for all parameters, not separate)
2. Proper Saltelli matrices A, B, C_i with low-discrepancy properties
3. Parallel model evaluation (4-8x speedup with joblib)
4. Jansen 1999 estimator for total effects (low variance, f_A vs f_C_i where C_i = A_B^(i))
5. Objective function: 1-NSE (capped at 5.0 to prevent outlier distortion)
6. FIXED: Failure tracking (separate flags, not by objective value)
7. Reproducible (seed=123)
8. Comprehensive visualization and results export

Critical Fix (Dec 2024):
- Failures tracked with separate boolean flag (is_fail)
- Valid objective=1.0 (NSE=0) is NO LONGER treated as failure
- Capped runs (is_capped) counted separately from failures
- Failure rate computed from failure flag only, not objective value

Methodology:
- Saltelli et al. (2010): "Variance based sensitivity analysis of model output"
- Jansen (1999): Low-variance total effect estimator
- N=512 base samples → 9,728 total model evaluations (17 params × (17+2) × 512)

Author: Zamin (with corrections to Chris's code)
Date: December 2024
"""

import os
import sys
import time
import timeit
from pathlib import Path
from scipy.stats import qmc 

# Add project root to Python path so 'hmg' module can be found
# For Assignment 3, we use hmg_cython/hmg/ (Cython-optimized version)
script_dir = Path(__file__).resolve().parent     # hmg_cython/Assignment3/
hmg_cython_root = script_dir.parent              # hmg_cython/
project_root = hmg_cython_root.parent            # zamin-uq-hydralogy/
sys.path.insert(0, str(hmg_cython_root))         # For hmg_cython/hmg
sys.path.insert(0, str(project_root))            # Fallback to hmg/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed 

from hmg import HBV001A

# ============================================================================
# GLOBAL CONSTANTS
# ============================================================================

# Fixed parameter (not sampled)
SNW_DTH_FIXED = 0.0

# Variable parameters (17 total - snw_dth excluded as it's constant)
PARAM_NAMES_FULL = [
    'snw_att', 'snw_pmf', 'snw_amf',
    'sl0_dth', 'sl0_pwp', 'sl0_fcy', 'sl0_bt0',
    'urr_dth', 'lrr_dth', 'urr_wsr', 'urr_ulc',
    'urr_tdh', 'urr_tdr', 'urr_ndr', 'urr_uct',
    'lrr_dre', 'lrr_lct'
]
#full range:

# BOUNDS_FULL = np.array([
#     (-2.0, 3.0),     # snw_att
#     (0.0, 3.0),      # snw_pmf
#     (0.0, 10.0),     # snw_amf
#     (0.0, 100.0),    # sl0_dth
#     (5.0, 700.0),    # sl0_pwp
#     (100.0, 700.0),  # sl0_fcy
#     (0.01, 10.0),    # sl0_bt0
#     (0.0, 20.0),     # urr_dth
#     (0.0, 100.0),    # lrr_dth
#     (0.0, 1.0),      # urr_wsr
#     (0.0, 1.0),      # urr_ulc
#     (0.0, 200.0),    # urr_tdh
#     (0.01, 1.0),     # urr_tdr
#     (0.0, 1.0),      # urr_ndr
#     (0.0, 1.0),      # urr_uct
#     (0.0, 1.0),      # lrr_dre
#     (0.0, 1.0),      # lrr_lct
# ], dtype=np.float32)

# narrow range: 
    
BOUNDS_FULL = np.array([
    (-0.23393099941313267, 0.26606900058686733),     # snw_att
    (0.3527601182460785, 0.6551259338855744),      # snw_pmf
    (0.036014999449253085, 0.06688499897718429),     # snw_amf
    (35.045325469970706, 65.08417587280273),    # sl0_dth
    (277.46546936035156, 515.2930145263672),    # sl0_pwp
    (100.0, 141.96853561401366),  # sl0_fcy
    (1.4277963638305664, 2.6516218185424805),    # sl0_bt0
    (0.4346475064754486, 0.8072025120258332),     # urr_dth
    (0.9422154068946839, 1.749828612804413),    # lrr_dth
    (0.5120226919651032, 0.9508992850780487),      # urr_wsr
    (0.12625760436058045, 0.23447840809822082),      # urr_ulc
    (60.46181106567383, 112.28622055053711),    # urr_tdh
    (0.11332999914884567, 0.21046999841928482),     # urr_tdr
    (8.610000368207693e-05, 0.00015990000683814288),      # urr_ndr
    (4.9000000672094755e-06, 9.100000124817597e-06),      # urr_uct
    (0.006135499943047762, 0.011394499894231557),      # lrr_dre
    (1.049999973474769e-05, 1.9499999507388565e-05),      # lrr_lct
], dtype=np.float32)
# ============================================================================
# OBJECTIVE FUNCTIONS
# ============================================================================

def nse(obs, sim):
    """
    Calculate Nash-Sutcliffe Efficiency (NSE).
    
    NSE = 1 - SS_res / SS_tot
    Range: (-∞, 1], where 1 = perfect, 0 = as good as mean
    """
    obs = np.asarray(obs, dtype=np.float32)
    sim = np.asarray(sim, dtype=np.float32)
    
    # Remove NaN values
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    if mask.sum() == 0:
        return np.nan
    
    o = obs[mask]
    s = sim[mask]
    
    # Calculate NSE
    denom = np.sum((o - o.mean(dtype=np.float32)) ** 2, dtype=np.float32)
    if denom == 0.0:
        return np.nan
    
    num = np.sum((s - o) ** 2, dtype=np.float32)
    nse_val = float(1.0 - num / denom)
    
    return nse_val


def calculate_lnnse(obs, sim):
    """
    Calculate Log Nash-Sutcliffe Efficiency (LnNSE).
    Better for evaluating low-flow (baseflow) performance.
    """
    obs = np.asarray(obs, dtype=np.float32)
    sim = np.asarray(sim, dtype=np.float32)
    
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    if mask.sum() == 0:
        return np.nan
    
    o = obs[mask]
    s = sim[mask]
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-6
    log_o = np.log(o + epsilon)
    log_s = np.log(s + epsilon)
    
    # Calculate LnNSE
    numerator = np.sum((log_o - log_s) ** 2, dtype=np.float32)
    denominator = np.sum((log_o - log_o.mean(dtype=np.float32)) ** 2, dtype=np.float32)
    
    if denominator == 0.0:
        return np.nan
    
    lnnse = float(1.0 - numerator / denominator)
    return lnnse

# ============================================================================
# MODEL EXECUTION
# ============================================================================

def run_hbv_model(params, tems, ppts, pets, dslr, tsps):
    """
    Run HBV model with given parameters and return simulated discharge.
    
    Parameters:
    -----------
    params : array-like, shape (17,)
        Model parameters (variable parameters only, snw_dth excluded)
    tems, ppts, pets : array-like
        Input time series
    dslr : float
        Discharge scaler
    tsps : int
        Number of time steps
        
    Returns:
    --------
    discharge : ndarray
        Simulated discharge time series
    
    Notes:
    ------
    Parameter order verified from hbv001a_py.py:
      Index 0: snw_dth (FIXED at 0.0)
      Index 1: snw_att (variable)
      Index 2: snw_pmf (variable)
      ... etc
    This function inserts snw_dth at position 0 before passing to HBV001A.
    """
    # Insert fixed snw_dth parameter at index 0 (verified from model source)
    full_params = np.concatenate([[SNW_DTH_FIXED], params])
    
    # Sanity check: should have 18 parameters total
    assert len(full_params) == 18, f"Expected 18 params, got {len(full_params)}"
    
    model = HBV001A()
    model.set_inputs(tems, ppts, pets)
    model.set_outputs(tsps)
    model.set_discharge_scaler(dslr)
    model.set_parameters(full_params)
    model.set_optimization_flag(0)
    model.run_model()
    
    return model.get_discharge()

# ============================================================================
# SALTELLI SAMPLING - CORRECTED VERSION
# ============================================================================

def generate_saltelli_samples_corrected(bounds, N=512):
    """
    Generate Saltelli samples using Sobol quasi-random sequences.
    
    *** CORRECTED VERSION ***
    
    Key Fix: Generates ONE unified Sobol sequence for ALL parameters together,
    not separate sequences per parameter (which breaks low-discrepancy property).
    
    Following Saltelli et al. (2010):
    - Creates matrices A and B from Sobol sequence
    - Creates k matrices C_i where C_i = A with column i replaced by B[:,i]
    - Total samples: N(k+2) where k = number of parameters
    
    Parameters:
    -----------
    bounds : array-like, shape (k, 2)
        Parameter bounds [(lower, upper), ...]
    N : int
        Base sample size (must be power of 2 for Sobol)
        Recommended: N ≥ 500 (Saltelli et al. 2010)
        
    Returns:
    --------
    A : ndarray, shape (N, k)
        Sample matrix (first k columns of Sobol sequence)
    B : ndarray, shape (N, k)
        Re-sample matrix (last k columns of Sobol sequence)
    C_matrices : list of ndarrays
        List of k matrices A_B^(i), each shape (N, k)
    """
    k = len(bounds)  # Number of parameters
    
    # Calculate m such that 2^m = N
    m = int(np.log2(N))
    if 2**m != N:
        raise ValueError(f"N must be power of 2. Given N={N}, nearest: 2^{m}={2**m}")
    
    print(f"\n{'='*70}")
    print("GENERATING SALTELLI SAMPLES (CORRECTED)")
    print(f"{'='*70}")
    print(f"Number of parameters (k): {k}")
    print(f"Base sample size (N): {N} = 2^{m}")
    print(f"Total dimensions for Sobol sequence: 2k = {2*k}")
    print(f"Total parameter sets needed: N(k+2) = {N}×{k+2} = {N*(k+2)}")
    print(f"\nFollowing Saltelli et al. (2010) methodology:")
    print(f"  - Matrix A: Sample matrix")
    print(f"  - Matrix B: Re-sample matrix")
    print(f"  - Matrices C_i (i=1..k): A with column i from B")
    
    # ========================================================================
    # CRITICAL FIX: Generate ONE Sobol sequence for ALL parameters
    # ========================================================================
    
    print(f"\n*** CORRECTION ***")
    print(f"Generating ONE unified Sobol sequence in {2*k} dimensions")
    print(f"(NOT separate sequences per parameter)")
    
    # Create Sobol sampler for 2k dimensions (k for A, k for B)
    # scramble=True adds randomization while preserving low-discrepancy
    # seed=123 ensures reproducibility
    sampler = qmc.Sobol(d=2*k, scramble=True, seed=123)
    
    # Generate N samples in 2k dimensions
    samples = sampler.random_base2(m=m)  # Shape: (N, 2k)
    
    print(f"Generated Sobol sequence: shape {samples.shape}")
    print(f"  This ensures low-discrepancy across ALL parameters simultaneously")
    
    # ========================================================================
    # Scale to parameter bounds
    # ========================================================================
    
    print(f"\nScaling samples to parameter bounds...")
    
    # Prepare bounds for scaling (repeat bounds twice for A and B)
    l_bounds = np.array([b[0] for b in bounds] * 2, dtype=np.float32)
    u_bounds = np.array([b[1] for b in bounds] * 2, dtype=np.float32)
    
    # Scale samples to parameter ranges
    samples_scaled = qmc.scale(samples, l_bounds, u_bounds)
    
    # ========================================================================
    # Split into A and B matrices
    # ========================================================================
    
    A = samples_scaled[:, :k].astype(np.float32)     # First k columns
    B = samples_scaled[:, k:].astype(np.float32)     # Last k columns
    
    print(f"Matrix A created: shape {A.shape}")
    print(f"Matrix B created: shape {B.shape}")
    
    # ========================================================================
    # Create C matrices (A_B^(i))
    # ========================================================================
    
    print(f"\nCreating C matrices (A_B^(i))...")
    
    C_matrices = []
    for i in range(k):
        C_i = A.copy()
        C_i[:, i] = B[:, i]  # Replace column i with B's column i
        C_matrices.append(C_i)
    
    print(f"Created {len(C_matrices)} C matrices")
    for i in range(min(3, k)):
        print(f"  C_{i}: A with column {i} ({PARAM_NAMES_FULL[i]}) from B")
    if k > 3:
        print(f"  ... and {k-3} more")
    
    print(f"\n{'='*70}")
    print(f"SAMPLING COMPLETE")
    print(f"{'='*70}")
    print(f"Total parameter sets to evaluate:")
    print(f"  f(A):   {N} evaluations")
    print(f"  f(B):   {N} evaluations")
    print(f"  f(C_i): {k} × {N} = {k*N} evaluations")
    print(f"  TOTAL:  {N + N + k*N} = {N*(k+2)} evaluations")
    print(f"{'='*70}\n")
    
    return A, B, C_matrices

# ============================================================================
# SOBOL INDEX CALCULATION
# ============================================================================

def compute_sobol_indices(f_A, f_B, f_C_list, param_names):
    """
    Compute Sobol sensitivity indices using Saltelli et al. (2010) formulas.
    
    Uses Jansen 1999 estimator (Formula f in Table 2) for total effects:
    - Most efficient estimator for ST_i
    - Lower variance than alternatives
    - Uses f_A vs f_C_i (since C_i = A_B^(i), matches our matrix construction)
    
    Parameters:
    -----------
    f_A : ndarray, shape (N,)
        Model outputs for matrix A (objective = 1-NSE, lower is better)
    f_B : ndarray, shape (N,)
        Model outputs for matrix B (objective = 1-NSE, lower is better)
    f_C_list : list of ndarrays
        Model outputs for each C_i matrix, each shape (N,) (objective values)
    param_names : list of str
        Parameter names
        
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'S1': First-order indices (direct effects on objective)
        - 'ST': Total-order indices (direct + interactions on objective)
        - 'interaction': ST - S1 (pure interaction effects)
        - 'names': Parameter names
        - 'V_Y': Total output variance
    
    Note: Higher S1/ST means parameter has larger effect on objective (1-NSE).
    """
    N = len(f_A)
    k = len(f_C_list)
    
    print(f"\n{'='*70}")
    print("COMPUTING SOBOL INDICES")
    print(f"{'='*70}")
    print(f"Method: Jansen 1999 estimator (Saltelli et al. 2010, Formula f)")
    print(f"  - Most efficient for total effects (ST_i)")
    print(f"  - Uses f_A vs f_C_i (since C_i = A_B^(i)) for lower variance")
    print(f"\nSample size N: {N}")
    print(f"Number of parameters k: {k}")
    
    # ========================================================================
    # Total variance (using all available samples)
    # ========================================================================
    
    all_samples = np.concatenate([f_A, f_B] + f_C_list)
    
    # All values should be finite (failures mapped to cap_obj)
    n_invalid = np.sum(~np.isfinite(all_samples))
    if n_invalid > 0:
        raise ValueError(
            f"Found {n_invalid} NaN/Inf values in outputs. "
            "This should not happen with failures mapped to cap_obj."
        )
    
    V_Y = np.var(all_samples, ddof=1)
    
    print(f"\nTotal output variance V(Y): {V_Y:.6f}")
    print(f"  (Objective function: 1-NSE, lower is better)")
    
    if V_Y == 0:
        raise ValueError(
            "Zero variance detected - model output is constant. "
            "Check parameter bounds and model implementation."
        )
    
    # ========================================================================
    # Initialize arrays
    # ========================================================================
    
    S1 = np.zeros(k)
    ST = np.zeros(k)
    
    print(f"\nComputing indices for {k} parameters...")
    
    # ========================================================================
    # Compute indices for each parameter
    # ========================================================================
    
    for i in range(k):
        f_C_i = f_C_list[i]
        
        # ====================================================================
        # First-order index (Saltelli estimator)
        # V_i = (1/N) * sum(f_B * (f_C_i - f_A))
        # 
        # This estimates: V_X_i[E_X_~i(Y|X_i)]
        # Interpretation: Variance due to X_i alone
        # Higher S1 → parameter has strong direct effect on objective (1-NSE)
        # ====================================================================
        
        term1 = f_B * (f_C_i - f_A)
        V_i = np.mean(term1)
        S1[i] = V_i / V_Y
        
        # ====================================================================
        # Total effect index (Jansen 1999 - low variance form)
        # Since C_i = A_B^(i) (A with column i from B):
        # ST_i = mean((f_A - f_C_i)^2) / (2 * V_Y)
        # 
        # This estimates: E_X_~i[V_X_i(Y|X_~i)]
        # Interpretation: Variance remaining when all except X_i are fixed
        # Higher ST → parameter important (direct + interactions)
        # Must use f_A vs f_C_i because (A, C_i) share all variables except X_i
        # ====================================================================
        
        term2 = (f_A - f_C_i) ** 2
        ST[i] = np.mean(term2) / (2.0 * V_Y)
    
    print(f"Indices computed successfully")
    
    # ========================================================================
    # Create results dictionary
    # ========================================================================
    
    results = {
        'S1': S1,
        'ST': ST,
        'interaction': ST - S1,
        'names': param_names,
        'V_Y': V_Y,
        'sum_S1': np.sum(S1),
        'sum_ST': np.sum(ST),
        'mean_interaction': np.mean(ST - S1)
    }
    
    print(f"\nSummary statistics:")
    print(f"  Sum of S1 (first-order):     {results['sum_S1']:.4f}")
    print(f"  Sum of ST (total):           {results['sum_ST']:.4f}")
    print(f"  Total interaction effect:    {results['sum_ST'] - results['sum_S1']:.4f}")
    print(f"  Mean interaction per param:  {results['mean_interaction']:.4f}")
    
    return results

# ============================================================================
# RESULTS DISPLAY & EXPORT
# ============================================================================

def print_sobol_results(results, top_n=10):
    """Print formatted Sobol sensitivity analysis results."""
    
    print(f"\n{'='*70}")
    print("SOBOL SENSITIVITY ANALYSIS RESULTS")
    print(f"{'='*70}")
    
    # Create DataFrame
    results_df = pd.DataFrame({
        'Parameter': results['names'],
        'S1': results['S1'],
        'ST': results['ST'],
        'Interaction': results['interaction']
    })
    
    # Sort by total effect
    results_df = results_df.sort_values('ST', ascending=False)
    
    print(f"\nVariance Decomposition:")
    print(f"  Total variance V(Y):         {results['V_Y']:.6f}")
    print(f"  Sum of first-order effects:  {results['sum_S1']:.4f}")
    print(f"  Sum of total effects:        {results['sum_ST']:.4f}")
    print(f"  Total interaction effect:    {results['sum_ST'] - results['sum_S1']:.4f}")
    
    print(f"\nAll Parameters (sorted by Total Effect):")
    print(f"{'='*70}")
    print(f"{'Parameter':<12s} {'S1 (First)':>12s} {'ST (Total)':>12s} {'Interaction':>12s}")
    print(f"{'-'*70}")
    
    for _, row in results_df.iterrows():
        print(f"{row['Parameter']:<12s} {row['S1']:12.6f} {row['ST']:12.6f} {row['Interaction']:12.6f}")
    
    print(f"\n{'='*70}")
    print(f"TOP {top_n} MOST SENSITIVE PARAMETERS (by Total Effect)")
    print(f"{'='*70}\n")
    
    for idx, (_, row) in enumerate(results_df.head(top_n).iterrows(), 1):
        print(f"{idx:2d}. {row['Parameter']:12s}  ST = {row['ST']:.6f}  "
              f"(S1 = {row['S1']:.6f}, Int = {row['Interaction']:.6f})")
    
    print(f"\n{'='*70}\n")
    
    return results_df


def save_results(results, results_df, out_dir, label=''):
    """Save results to CSV and text files."""
    
    # Save DataFrame
    csv_path = out_dir / f'sobol_indices{label}.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"Saved indices to: {csv_path}")
    
    # Save summary text file
    txt_path = out_dir / f'sobol_summary{label}.txt'
    with open(txt_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SOBOL SENSITIVITY ANALYSIS SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total variance V(Y): {results['V_Y']:.6f}\n")
        f.write(f"Sum of S1: {results['sum_S1']:.6f}\n")
        f.write(f"Sum of ST: {results['sum_ST']:.6f}\n")
        f.write(f"Total interaction: {results['sum_ST'] - results['sum_S1']:.6f}\n\n")
        f.write("Top 10 most sensitive parameters:\n")
        f.write("-"*70 + "\n")
        for idx, (_, row) in enumerate(results_df.head(10).iterrows(), 1):
            f.write(f"{idx:2d}. {row['Parameter']:12s}  ST = {row['ST']:.6f}\n")
    
    print(f"Saved summary to: {txt_path}")

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_sobol_indices(results, out_dir, label=''):
    """Create comprehensive visualization of Sobol indices."""
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Sort by ST
    sort_idx = np.argsort(results['ST'])[::-1]
    sorted_names = [results['names'][i] for i in sort_idx]
    sorted_s1 = results['S1'][sort_idx]
    sorted_st = results['ST'][sort_idx]
    sorted_int = results['interaction'][sort_idx]
    
    # ========================================================================
    # Plot 1: Bar chart (S1 vs ST)
    # ========================================================================
    
    ax1 = fig.add_subplot(gs[0, 0])
    
    x = np.arange(len(sorted_names))
    width = 0.35
    
    ax1.bar(x - width/2, sorted_s1, width, label='S₁ (First-order)', 
            alpha=0.8, color='steelblue', edgecolor='black', linewidth=0.5)
    ax1.bar(x + width/2, sorted_st, width, label='Sᴛ (Total)', 
            alpha=0.8, color='coral', edgecolor='black', linewidth=0.5)
    
    ax1.set_ylabel('Sobol Index', fontsize=11, fontweight='bold')
    ax1.set_title('Sobol Sensitivity Indices', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=9)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, max(sorted_st) * 1.15])
    
    # ========================================================================
    # Plot 2: Horizontal bar (sorted by ST)
    # ========================================================================
    
    ax2 = fig.add_subplot(gs[0, 1])
    
    y_pos = np.arange(len(sorted_names))
    ax2.barh(y_pos, sorted_st, alpha=0.7, color='coral', edgecolor='black', linewidth=0.5)
    ax2.barh(y_pos, sorted_s1, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sorted_names, fontsize=9)
    ax2.set_xlabel('Sobol Index', fontsize=11, fontweight='bold')
    ax2.set_title('Parameter Ranking', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()
    
    # ========================================================================
    # Plot 3: Interaction effects
    # ========================================================================
    
    ax3 = fig.add_subplot(gs[0, 2])
    
    colors = ['red' if val > 0.1 else 'steelblue' for val in sorted_int]
    y_pos_int = np.arange(len(sorted_names))
    ax3.barh(y_pos_int, sorted_int, color=colors, alpha=0.7, 
             edgecolor='black', linewidth=0.5)
    
    ax3.set_yticks(y_pos_int)
    ax3.set_yticklabels(sorted_names, fontsize=9)
    ax3.set_xlabel('Interaction (Sᴛ - S₁)', fontsize=11, fontweight='bold')
    ax3.set_title('Interaction Effects', fontsize=12, fontweight='bold')
    ax3.axvline(0.1, color='red', linestyle='--', linewidth=2, label='Threshold (0.1)')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.invert_yaxis()
    
    # ========================================================================
    # Plot 4: Pie chart of top contributors
    # ========================================================================
    
    ax4 = fig.add_subplot(gs[1, 0])
    
    # Top 5 + Others
    top_5_st = sorted_st[:5]
    top_5_names = sorted_names[:5]
    others_st = np.sum(sorted_st[5:])
    
    pie_values = list(top_5_st) + [others_st]
    pie_labels = top_5_names + ['Others']
    #colors_pie = plt.cm.Set3(range(len(pie_values)))
    colors_pie = plt.cm.get_cmap('Set3', len(pie_values))(range(len(pie_values)))
    
    ax4.pie(pie_values, labels=pie_labels, autopct='%1.1f%%', 
            colors=colors_pie, startangle=90)
    ax4.set_title('Top 5 Contributors (Total Effect)', fontsize=12, fontweight='bold')
    
    # ========================================================================
    # Plot 5: S1 vs ST scatter
    # ========================================================================
    
    ax5 = fig.add_subplot(gs[1, 1])
    
    ax5.scatter(sorted_s1, sorted_st, s=100, alpha=0.6, color='purple', edgecolor='black')
    
    # Add parameter labels for top 5
    for i in range(min(5, len(sorted_names))):
        ax5.annotate(sorted_names[i], (sorted_s1[i], sorted_st[i]), 
                    fontsize=8, xytext=(5, 5), textcoords='offset points')
    
    # Add diagonal line (where S1 = ST, no interaction)
    max_val = max(max(sorted_s1), max(sorted_st))
    ax5.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='S₁ = Sᴛ (no interaction)')
    
    ax5.set_xlabel('S₁ (First-order)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Sᴛ (Total)', fontsize=11, fontweight='bold')
    ax5.set_title('First-order vs Total Effect', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # ========================================================================
    # Plot 6: Summary statistics
    # ========================================================================
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    summary_text = f"""
SUMMARY STATISTICS
{'='*40}

Total Variance:        {results['V_Y']:.6f}

Sum of S₁:            {results['sum_S1']:.4f}
Sum of Sᴛ:            {results['sum_ST']:.4f}

Total Interaction:     {results['sum_ST'] - results['sum_S1']:.4f}
Mean Interaction:      {results['mean_interaction']:.4f}

TOP 5 PARAMETERS:
{'-'*40}
"""
    
    for i in range(min(5, len(sorted_names))):
        summary_text += f"{i+1}. {sorted_names[i]:<10s}  {sorted_st[i]:.4f}\n"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ========================================================================
    # Save figure
    # ========================================================================
    
    plt.suptitle(f'Sobol Sensitivity Analysis Results{label}', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    out_path = out_dir / f'sobol_analysis{label}.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {out_path}")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function.
    Performs Sobol sensitivity analysis with corrected Saltelli sampling.
    
    Key Fix: Failures tracked separately (not by objective value).
    Valid objective=1.0 (NSE=0) is no longer treated as failure.
    """
    
    print("\n" + "="*80)
    print(" "*20 + "ASSIGNMENT 3: SOBOL SENSITIVITY ANALYSIS")
    print(" "*25 + "(Corrected & Complete Version)")
    print("="*80)
    print("\nBased on:")
    print("  - Saltelli et al. (2010) - Computer Physics Communications")
    print("  - Jansen (1999) estimator for total effects")
    print("  - Corrected implementation with proper failure tracking")
    print("="*80)
    
    # ========================================================================
    # SETUP: Paths and check data availability
    # ========================================================================
    
    # Find data directory relative to repository root
    # Data format notes:
    # - CSV files use semicolon (;) as separator
    # - Index is datetime: YYYY-MM-DD-HH (hourly resolution)
    # - Columns: tavg__ref, pptn__ref, petn__ref, diso__ref
    # - area___24163005.csv provides catchment area in m² (959,000,000 m²)
    # - Discharge scaler for hourly data: area_m2 / (3600 * 1000)
    def find_data_dir():
        current = Path(__file__).resolve().parent
        for _ in range(10):
            candidate = current / 'data'
            if candidate.is_dir():
                return candidate
            current = current.parent
        raise FileNotFoundError("Could not find 'data' directory in repository")
    
    main_dir = find_data_dir()
    
    # Output directory (in Assignment3 folder)
    try:
        script_dir = Path(__file__).parent
    except NameError:
        script_dir = Path.cwd()
    
    out_dir = script_dir / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nData directory: {main_dir}")
    print(f"Output directory: {out_dir}")
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("LOADING INPUT DATA")
    print(f"{'='*70}")
    
    try:
        inp_dfe = pd.read_csv(main_dir / 'time_series___24163005.csv', sep=';', index_col=0)
        inp_dfe.index = pd.to_datetime(inp_dfe.index, format='%Y-%m-%d-%H')
        
        cca_srs = pd.read_csv(main_dir / 'area___24163005.csv', sep=';', index_col=0)
        ccaa = cca_srs.values[0, 0]
        
        print(f"Loaded time series data")
    except FileNotFoundError as e:
        print(f"\nERROR: Required data file not found!")
        print(f"   {e}")
        return
    
    # Extract variables
    tems = inp_dfe.loc[:, 'tavg__ref'].values
    ppts = inp_dfe.loc[:, 'pptn__ref'].values
    pets = inp_dfe.loc[:, 'petn__ref'].values
    diso = inp_dfe.loc[:, 'diso__ref'].values
    
    tsps = tems.shape[0]
    dslr = ccaa / (3600 * 1000)
    
    print(f"  Time steps: {tsps}")
    print(f"  Catchment area: {ccaa:.2f} m²")
    print(f"  Discharge scaler: {dslr:.6f}")
    print(f"  Date range: {inp_dfe.index[0]} to {inp_dfe.index[-1]}")
    
    # ========================================================================
    # GENERATE SALTELLI SAMPLES (CORRECTED)
    # ========================================================================
    
    # N = 512      # 2^9  - Quick test (9,728 runs, ~5-10 min)
    # N = 1024     # 2^10 - Standard (19,456 runs, ~15-20 min)
    # N = 2048     # 2^11 - Good quality (38,912 runs, ~30-40 min)
    # N = 4096       # 2^12 - Publication quality (77,824 runs, ~60-90 min)  RECOMMENDED
    # N = 8192     # 2^13 - High precision (155,648 runs, ~2-3 hours)
    # N = 524288   # 2^19 - OVERKILL! (9,961,472 runs, ~15-20 hours!)
    N = 512  # Using N=512 for reasonable runtime
    
    try:
        A, B, C_matrices = generate_saltelli_samples_corrected(BOUNDS_FULL, N=N)
    except Exception as e:
        print(f"\nERROR in sample generation: {e}")
        return
    
    # ========================================================================
    # EVALUATE MODEL: f(A), f(B), f(C_i) - PARALLEL EXECUTION
    # ========================================================================
    
    cap_obj = 5.0  # Cap for objective = 1 - NSE
    
    def evaluate_model_wrapper(params, tems, ppts, pets, dslr, tsps, obs_data):
        """
        Wrapper for parallel execution.
        Returns tuple: (objective, is_failure, is_capped)
        
        KEY FIX: Failures tracked separately, not by objective value.
        Valid objective=1.0 (NSE=0) is no longer treated as failure.
        
        Returns:
        --------
        obj : float
            Objective function value = 1 - NSE (lower is better)
        is_fail : bool
            True if model crashed or NSE is NaN/Inf
        is_capped : bool
            True if objective was capped at cap_obj (extreme negative NSE)
        
        Objective mapping:
          NSE = 1.0 (perfect)     → obj = 0.0 (best)
          NSE = 0.0 (baseline)    → obj = 1.0 (NOT a failure!)
          NSE < 0.0 (poor)        → obj > 1.0 (capped at 5.0)
          Invalid/failed run      → obj = cap_obj, is_fail=True
        """
        try:
            sim = run_hbv_model(params, tems, ppts, pets, dslr, tsps)
            #nse_val = nse(obs_data, sim)
            nse_val = calculate_lnnse(obs_data, sim)
            
            if not np.isfinite(nse_val):
                # Invalid NSE (NaN/Inf) - this is a failure
                return cap_obj, True, False
            
            # Valid NSE - compute objective
            obj = 1.0 - nse_val
            
            # Check if capping needed (extreme negative NSE)
            is_capped = obj >= (cap_obj - 1e-9)
            obj = min(cap_obj, obj)
            
            return obj, False, is_capped
            
        except Exception:
            # Model crashed - this is a failure
            return cap_obj, True, False
    
    k = len(PARAM_NAMES_FULL)
    total_runs = N + N + k * N
    
    print(f"\n{'='*70}")
    print("EVALUATING MODEL (PARALLEL)")
    print(f"{'='*70}")
    print(f"Total model runs required: {total_runs}")
    print(f"  f(A):   {N:>6d} runs")
    print(f"  f(B):   {N:>6d} runs")
    print(f"  f(C_i): {k:>2d} × {N} = {k*N:>6d} runs")
    print(f"\nUsing parallel execution with all available CPU cores")
    print(f"Estimated time: ~{total_runs * 0.05 / 60:.1f} minutes (serial estimate)")
    print(f"Actual time will be significantly faster with parallelization")
    
    n_jobs = -1  # Use all available cores
    start_eval = time.time()
    
    # ====================================================================
    # Evaluate f(A) - Parallel
    # ====================================================================
    
    print(f"\n[1/3] Evaluating f(A) - {N} runs (parallel)...")
    resA = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(evaluate_model_wrapper)(row, tems, ppts, pets, dslr, tsps, diso) 
        for row in A
    )
    f_A, fail_A, cap_A = map(np.array, zip(*resA))
    print(f"f(A) complete - Mean objective (1-NSE): {np.mean(f_A):.4f} ({fail_A.sum()} failures, {cap_A.sum()} capped)")
    
    # ====================================================================
    # Evaluate f(B) - Parallel
    # ====================================================================
    
    print(f"\n[2/3] Evaluating f(B) - {N} runs (parallel)...")
    resB = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(evaluate_model_wrapper)(row, tems, ppts, pets, dslr, tsps, diso) 
        for row in B
    )
    f_B, fail_B, cap_B = map(np.array, zip(*resB))
    print(f"f(B) complete - Mean objective (1-NSE): {np.mean(f_B):.4f} ({fail_B.sum()} failures, {cap_B.sum()} capped)")
    
    # ====================================================================
    # Evaluate f(C_i) for each parameter - Parallel
    # ====================================================================
    
    print(f"\n[3/3] Evaluating f(C_i) for {k} parameters - {k*N} runs (parallel)...")
    f_C_list = []
    fail_C_list = []
    cap_C_list = []
    
    for idx, C_i in enumerate(C_matrices):
        param_name = PARAM_NAMES_FULL[idx]
        print(f"\n  Parameter {idx+1}/{k}: {param_name}")
        
        resC = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(evaluate_model_wrapper)(row, tems, ppts, pets, dslr, tsps, diso) 
            for row in C_i
        )
        f_C_i, fail_C_i, cap_C_i = map(np.array, zip(*resC))
        f_C_list.append(f_C_i)
        fail_C_list.append(fail_C_i)
        cap_C_list.append(cap_C_i)
        print(f"  C_{idx} ({param_name}) complete - Mean objective (1-NSE): {np.mean(f_C_i):.4f} ({fail_C_i.sum()} failures, {cap_C_i.sum()} capped)")
    
    total_eval_time = time.time() - start_eval
    
    # ====================================================================
    # Compute failure and capping statistics
    # ====================================================================
    
    total_failures = fail_A.sum() + fail_B.sum() + sum(f.sum() for f in fail_C_list)
    failure_rate = 100 * total_failures / total_runs
    n_capped = cap_A.sum() + cap_B.sum() + sum(c.sum() for c in cap_C_list)
    
    print(f"\n{'='*70}")
    print(f"MODEL EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total evaluation time: {total_eval_time/60:.2f} minutes")
    print(f"Average time per run: {total_eval_time/total_runs:.3f} seconds")
    print(f"Speedup from parallelization: ~{(total_runs * 0.05) / total_eval_time:.1f}x")
    print(f"\nObjective function statistics (1-NSE, capped at {cap_obj}):")
    print(f"  f(A):   Mean = {np.mean(f_A):.4f}, Std = {np.std(f_A):.4f}, Max = {np.max(f_A):.4f}")
    print(f"  f(B):   Mean = {np.mean(f_B):.4f}, Std = {np.std(f_B):.4f}, Max = {np.max(f_B):.4f}")
    
    print(f"\nFailure tracking:")
    print(f"  Total failures: {total_failures} / {total_runs} runs ({failure_rate:.2f}%)")
    print(f"  Capped runs (obj at {cap_obj}): {n_capped} runs")
    print(f"\nNote: Objective = 1-NSE (lower is better):")
    print(f"  0.0 = perfect (NSE=1), 1.0 = baseline (NSE=0), >{cap_obj} = poor")
    print(f"  Valid obj=1.0 (NSE=0) is NOT counted as failure")
    
    if failure_rate > 10.0:
        print(f"  CRITICAL: Very high failure rate (>{failure_rate:.1f}%)")
        print(f"     -> Check parameter bounds - may include physically impossible combinations")
        print(f"     -> Consider narrowing bounds or reviewing model constraints")
    elif failure_rate > 5.0:
        print(f"  WARNING: Elevated failure rate (>{failure_rate:.1f}%)")
        print(f"     -> Some parameter combinations may be unrealistic")
        print(f"     -> Results are still valid but consider reviewing bounds")
    else:
        print(f"  Acceptable failure rate (<5%)")
    
    # ========================================================================
    # COMPUTE SOBOL INDICES
    # ========================================================================
    
    try:
        results = compute_sobol_indices(f_A, f_B, f_C_list, PARAM_NAMES_FULL)
    except Exception as e:
        print(f"\nERROR in Sobol computation: {e}")
        return
    
    # ========================================================================
    # DISPLAY & SAVE RESULTS
    # ========================================================================
    
    results_df = print_sobol_results(results, top_n=10)
    
    save_results(results, results_df, out_dir, label='_corrected')
    
    plot_sobol_indices(results, out_dir, label='_corrected')
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print(f"\n{'='*80}")
    print(" "*30 + "ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {out_dir}")
    print(f"\nFiles created:")
    print(f"  - sobol_indices_corrected.csv")
    print(f"  - sobol_summary_corrected.txt")
    print(f"  - sobol_analysis_corrected.png")
    
    print(f"\n{'='*80}")
    print("KEY FINDINGS:")
    print(f"{'='*80}")
    
    # Top 5 sensitive parameters
    print(f"\nTop 5 most sensitive parameters (by Total Effect):")
    for idx, row in results_df.head(5).iterrows():
        print(f"  {row['Parameter']:12s}  ST = {row['ST']:.6f}")
    
    # Parameters with strong interactions
    strong_int = results_df[results_df['Interaction'] > 0.1]
    if len(strong_int) > 0:
        print(f"\nParameters with strong interactions (Interaction > 0.1):")
        for idx, row in strong_int.iterrows():
            print(f"  {row['Parameter']:12s}  Interaction = {row['Interaction']:.6f}")
    else:
        print(f"\nNo parameters show strong interactions (all < 0.1)")
    
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n{'='*80}")
        print("FATAL ERROR!")
        print(f"{'='*80}")
        print(f"{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    STOP = timeit.default_timer()
    print(f'\n#### Completed on {time.asctime()} ####')
    print(f'#### Total runtime: {(STOP - START)/60:.2f} minutes ####\n')
