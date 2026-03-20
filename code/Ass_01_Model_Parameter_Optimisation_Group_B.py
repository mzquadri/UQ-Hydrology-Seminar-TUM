'''
Group B 

finding optimal model Parameter values with the differential evolution algorithm
'''

import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hmg import HBV001A
from scipy.optimize import differential_evolution

DEBUG_FLAG = False

# Add project root to Python path so 'hmg' module can be found
project_root = Path(__file__).resolve().parent  
sys.path.insert(0, str(project_root))

# Penalty for invalid/undefined NSE (e.g., NaN or Inf)
PENALTY = 5.0

PARAM_NAMES_OPT = [
    'snw_dth', 'snw_att', 'snw_pmf', 'snw_amf',
    'sl0_dth', 'sl0_pwp', 'sl0_fcy', 'sl0_bt0',
    'urr_dth', 'lrr_dth', 'urr_wsr', 'urr_ulc',
    'urr_tdh', 'urr_tdr', 'urr_ndr', 'urr_uct',
    'lrr_dre', 'lrr_lct'
    ]
 
def nse(obs, sim):
    """
    Nash-Sutcliffe Efficiency (robust).
    Returns: float NSE or np.nan if undefined (e.g. zero variance or no valid data).
    """
    obs = np.asarray(obs, dtype=np.float64) 
    sim = np.asarray(sim, dtype=np.float64)
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    if mask.sum() == 0:
        return np.nan
    o = obs[mask]
    s = sim[mask]
    denom = np.sum((o - o.mean(dtype=np.float64)) ** np.float64(2.0), dtype=np.float64)
    if denom == 0.0:
        return np.nan
    num = np.sum((s - o) ** np.float64(2.0), dtype=np.float64)
    return float(np.float64(1.0) - num / denom)

def obj_nse(obs, sim):
    """
    Objective to minimize: 1 - NSE. If NSE undefined, return PENALTY.
    """
    val = nse(obs, sim)
    if not np.isfinite(val):
        return PENALTY
    return 1.0 - float(val)

# ===========================
# Differential Evolution Recorder (objective + history)
# ===========================

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
        
        # per-generation bests
        self.gen_objs = []
        self.gen_params = []
        self.gen_best_objs = []
        self.gen_max_objs = []
        self.gen_med_obj = []
        self.gen_best_params = []
        self.gen_params_dvt = [] # save one the best parms per gen if new best objectiv function is better 
        self.gen_cnt = 1
        
    def _assemble_full_params(self, prms_opt):
        """
        Build the full parameter vector (length = MODEL_PARAM_COUNT),
        inserting fixed params and the optimized subset according to opt_index_order.
        Returns float64 array (is needed if atol < 1e-7).
        """
        full = np.zeros((PARAM_NAMES_OPT,), dtype=np.float64)
        # place fixed first
        for idx, val in self.fixed_params.items():
            full[idx] = np.float64(val)
        # then place optimized entries in order
        prms_opt = np.asarray(prms_opt, dtype=np.float64)
        for k, idx in enumerate(self.opt_index_order):
            full[idx] = prms_opt[k]
        return full
        
    def __call__(self, prms):
        """
        Set up and run the model; return optional function value.
        """
        
        tems, ppts, pets = self.input
        modl_objt = self.model_class()
        modl_objt.set_inputs(tems, ppts, pets)
        modl_objt.set_outputs(self.tsps)
        modl_objt.set_discharge_scaler(self.dslr)
        modl_objt.set_parameters(prms)
        modl_objt.set_optimization_flag(1)
        modl_objt.run_model()
        diss = modl_objt.get_discharge()
       
        if diss.shape[0] != self.obs_q.shape[0]:
            raise RuntimeError(f"Sim length {diss.shape[0]} != obs length {self.obs_q.shape[0]}")
        
        diss = np.asarray(diss, dtype=np.float32)
        if np.isnan(diss).any() or np.isinf(diss).any():
            raise RuntimeError("Simulation produced NaN/Inf in discharge.")

        
        obj = obj_nse(self.obs_q, diss)
        
        if self.eval_count == self.pop_sz*len(prms) or self.eval_count == self.pop_sz*len(prms)*self.gen_cnt:
            print(f'start Generation {self.gen_cnt+1}:')
            dfe = pd.DataFrame([self.gen_objs, self.gen_params]).T
            dfe_min = dfe.loc[dfe[0] == min(dfe[0])]
        
            self.gen_best_objs.append(dfe_min[0])
            self.gen_max_objs.append(max(dfe[0]))
            self.gen_med_obj.append(np.median(dfe[0]))
            self.gen_best_params.append(dfe_min[1])
            
            self.gen_objs = []
            self.gen_params = []
            self.gen_cnt += 1
            
            start = self._last_gen_end
            end = self.eval_count
            self.gen_bounds.append((start, end))
            self._last_gen_end = end
            
            if len(self.gen_best_objs) >1 and self.gen_best_objs[-1].iloc[0] < self.gen_best_objs[-2].iloc[0]:
                self.gen_params_dvt.append(dfe_min[1])
            elif len(self.gen_best_objs) == 1:
                self.gen_params_dvt.append(dfe_min[1])
            else:
                self.gen_params_dvt.append(self.gen_params_dvt[-1])
                
        
        # record evaluation
        self.eval_params.append(prms.astype(float).copy())
        self.eval_objs.append(float(obj))
        self.gen_objs.append(float(obj))
        self.gen_params.append(prms.astype(float).copy())
        self.eval_count += 1
        
        return float(obj)
        
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def plot_parameters_normalized(bounds, best_params, out_file, param_names):
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    best = np.array(best_params, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = (ub - lb)
        denom[denom == 0] = 1.0
        norm = (best - lb) / denom
    norm = np.clip(norm, 0.0, 1.0)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    y = np.arange(len(best))
    ax.hlines(y, 0, 1, color='lightgray', linewidth=2, zorder=1)
    ax.scatter(norm, y, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(param_names)
    ax.set_xlabel('Normalized value (0..1)')
    ax.set_title('Final best parameter values (normalized)')
    ax.grid(axis='x')
    plt.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)

def plot_param_evolution_per_generation(pop_sz, run_min, eval_params, gen_best_parms, best_params_opt, params_dvt, out_dir , param_names): #hier gen_best_params
    if len(eval_params) == 0:
        return
    
    params_dvt_dfe = [] 
    for i in params_dvt: 
        if isinstance(i, pd.Series): 
            params_dvt_dfe.append(i.values.tolist()) 
    params_dvt_dfe = pd.DataFrame(params_dvt_dfe)
    new_colum = {f"{i}": params_dvt_dfe[0].apply(lambda lst: lst[i]) for i in range( len(params_dvt_dfe.iloc[0, 0]))}
    new_df = pd.DataFrame(new_colum)
    
    x = np.arange(0, len(params_dvt), 1)
    for i in range(0,len(param_names)):
        fig, ax = plt.subplots()
        ax.plot(x, new_df.iloc[:, i])
        ax.set_xlabel('Generation')
        ax.set_ylabel('Parameter best-values evolution across generations')
        nam = param_names[i]
        out_file=str(out_dir / f'params_evolution_per_generation_{nam}.png')
        fig.savefig(out_file)
        plt.close(fig)
        
        
    

def plot_param_scatter_all_evals(eval_params, eval_objs, bounds, out_dir, param_names):
    """
    Scatter plots for each parameter:
      - X-axis: raw parameter values (original scale from bounds)
      - Y-axis: normalized objective values (0 = best, 1 = worst)
    """
    if len(eval_params) == 0:
        return

    ensure_dir(out_dir)
    eval_params = np.vstack(eval_params).astype(float)
    eval_objs = np.array(eval_objs, dtype=float)
    
    dfe_sct = pd.DataFrame([eval_objs, eval_params]).T
    dfe_sct_clip = dfe_sct.loc[(dfe_sct[0] >= 0) & (dfe_sct[0] <= 1)]
    
    if dfe_sct_clip.shape[0] > 0:
        eval_params = np.vstack(dfe_sct_clip[1]).astype(float)
        eval_objs = np.array(dfe_sct_clip[0], dtype=float)

    # Normalize objective (min -> 0, max -> 1); safe if constant
    if np.all(np.isnan(eval_objs)):
        obj_rng = np.zeros_like(eval_objs, dtype=float)
    else:
        obj_rng = eval_objs  

    n_params = eval_params.shape[1]

    for i in range(n_params):
        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)

        # X: raw parameter values; Y: normalized objective
        ax.scatter(eval_params[:, i], obj_rng, s=10, alpha=0.6)
            
        #ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(f'{param_names[i]} value')
        ax.set_ylabel('Objective function value (0 = best, 1 = worst)')
        ax.set_title(f'{param_names[i]}: sampled values vs objective function')
        ax.grid(True)

        # show parameter bounds on x-axis
        lb, ub = bounds[i]
        ax.axvline(lb, linestyle='--', linewidth=1)
        ax.axvline(ub, linestyle='--', linewidth=1)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f'param_scatter_{param_names[i]}.png'))
        plt.close(fig)

def plot_convergence_per_generation(gen_max_obj, gen_best_obj, gen_med_obj, out_dir):
    
    mins = np.array(gen_best_obj) 
    maxs = np.array(gen_max_obj) 
    meds = np.array(gen_med_obj)
    
    running_min = []
    for i in range(0, len(mins)):
        if i == 0:
            running_min.append(mins[0])
        else:
            if mins[i] < mins[i-1] and mins[i] < running_min[i-1]:
                running_min.append(mins[i])
            else:
                running_min.append(running_min[i-1])
                
    gens = np.arange(1, len(mins) + 1)
    
    fig1, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(gens, mins, label='min')
    ax.plot(gens, maxs, label='max')
    ax.plot(gens, meds, label='median')
    ax.plot(gens, running_min, label='running best (min)', linestyle='--', linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Objective (1 - NSE)')
    ax.set_title('Convergence per generation')
    ax.legend()
    ax.grid(True)
    ax.set_ylim(0.0, 3.0)
    
    
    plt.tight_layout()
    out_file=str(out_dir / 'convergence_per_generation.png')
    fig1.savefig(out_file)
    plt.close(fig1)
    
    fig2, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(gens, mins, label='min')
    ax.plot(gens, maxs, label='max')
    ax.plot(gens, meds, label='median')
    ax.plot(gens, running_min, label='running best (min)', linestyle='--', linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Objective (1 - NSE)')
    ax.set_title('Convergence per generation')
    ax.legend()
    ax.grid(True)
    ax.set_ylim(0.0, 10.0)
    
    plt.tight_layout()
    out_file=str(out_dir / 'convergence_per_generation_y10.png')
    fig2.savefig(out_file)
    plt.close(fig2)
    
    fig3, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(gens, mins, label='min')
    ax.plot(gens, maxs, label='max')
    ax.plot(gens, meds, label='median')
    ax.plot(gens, running_min, label='running best (min)', linestyle='--', linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Objective (1 - NSE)')
    ax.set_title('Convergence per generation')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    out_file=str(out_dir / 'convergence_per_generation_ytotal.png')
    fig3.savefig(out_file)
    plt.close(fig3)
    
    return running_min

    
def main():
    # Absolute path to the directory where the input data lies.
    main_dir = project_root / 'data'
    os.chdir(main_dir)

    # Read input text time series as a pandas Dataframe object and
    # cast the index to a datetime object.
    inp_dfe = pd.read_csv(r'time_series___24163005.csv', sep=';', index_col=0)
    inp_dfe.index = pd.to_datetime(inp_dfe.index, format='%Y-%m-%d-%H')

    # Read the catcment area in meters squared. The first value is needed
    # only.
    cca_srs = pd.read_csv(r'area___24163005.csv', sep=';', index_col=0)
    ccaa = cca_srs.values[0, 0]
    
    tems = inp_dfe.loc[:, 'tavg__ref'].values  # Temperature.
    ppts = inp_dfe.loc[:, 'pptn__ref'].values  # Preciptiation.
    pets = inp_dfe.loc[:, 'petn__ref'].values  # PET.
    diso = inp_dfe.loc[:, 'diso__ref'].values  # Observed discharge.

    tsps = tems.shape[0]  # Number of time steps.

    # Conversion constant for mm/hour to m3/s.
    dslr = ccaa / (3600 * 1000)  # For daily res. multiply denominator with 24.
    
    # Bounds for optimized subset (corrected, excluding snw_dth)
    bounds_opt = {
        'snw_dth': (0.00, 0.0),  # Initial.
        'snw_att': (-2.0, 3.0),  # Critical.
        'snw_pmf': (0.0, 3.0),  # Optional.
        'snw_amf': (0.0, 10.0),  # Critical.

        'sl0_dth': (0.00, 100),  # Initial.
        'sl0_pwp': (5.00, 700.0),  # Critical.
        'sl0_fcy': (100.0, 700.0),  # Critical.
        'sl0_bt0': (0.01, 10.0),  # Critical.

        'urr_dth': (0.0, 20.0),  # Initial.
        'lrr_dth': (0.0, 100.0),  # Initial.

        'urr_wsr': (0.00, 1.00),  # Optional.
        'urr_ulc': (0.00, 1.00),  # Critical.
        'urr_tdh': (0.00, 200.0),  # Critical.
        'urr_tdr': (0.01, 1.00),  # Critical.
        'urr_ndr': (0.00, 1.00),  # Critical.
        'urr_uct': (0.00, 1.00),  # Critical.

        'lrr_dre': (0.00, 1.00),  # Critical.
        'lrr_lct': (0.00, 1.00),  # Critical.
    }
   
    #==========================================================================
    modl_objt = HBV001A()
    modl_objt.get_parameter_bounds_in_correct_order(bounds_opt)
    bounds_opt = list(bounds_opt.values())
    
    gen = 250
    pop_sz = 10
    
    recorder = DERecorder(
        inputs = (tems, ppts, pets),
        dslr = dslr,
        tsps = tsps,
        obs_q = diso,
        model_class=HBV001A,
        pop_sz = pop_sz
        )
    
    de_result = differential_evolution(
        func = recorder,
        bounds = bounds_opt,
        strategy='best1bin',
        maxiter=gen,
        popsize=pop_sz,
        tol=1e-3,
        atol=0,
        polish=False,
        updating='deferred'
        )
   
    
    best_params_opt = de_result.x.astype(np.float64)
    print('')
    best_obj = float(de_result.fun)
    print(f'best objectiv funcion value: {best_obj}')
        
    # Initiate an empty model object. To understand what each method call
    # used below is for, please take a look at the files inside models
    # directory.
    modl_objt = HBV001A()

    # Set the above defined inputs.
    modl_objt.set_inputs(tems, ppts, pets)

    # Pass the number of time steps to the model object here. It creates the
    # ouputs array(s) with the proper shape.
    modl_objt.set_outputs(tsps)

    # Set the constant that will convert units from those of precipitation
    # to those of measured discharge.
    modl_objt.set_discharge_scaler(dslr)
    #==========================================================================
    # Outputs & plots - use repo-local path ./outputs/assignment1/
    out_dir = project_root / 'outputs' / 'assignment1'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Show the parameters against their names as a check.
    print('')

    print('Model parameters:')
    final_nse = 1 - best_obj
    with open(out_dir/'final_debug_summary.txt', 'w') as txt_hdl:
        txt_hdl.write('\n')
        
    for prm_lbl, i in modl_objt.get_parameter_labels().items():
        print(f'{prm_lbl}:', round(best_params_opt[i], 6))
        with open(out_dir/'final_debug_summary.txt', 'a') as txt_hdl:
            txt_hdl.write(f'{prm_lbl}: {round(best_params_opt[i], 6)}\n')
            
    with open(out_dir/'final_debug_summary.txt', 'a') as txt_hdl:
        txt_hdl.write(f' best_obj:{float(best_obj)}\n'\
                      f'final_nse: {final_nse}\n'\
                      f'n_evals: {recorder.eval_count}\n'\
                      )

    print('')

    # Get a dictionary that links an output labe to its column index in the
    # outputs array.
    otps_lbls = modl_objt.get_output_labels()

    # Pass the parameters.
    modl_objt.set_parameters(best_params_opt)

    # Tell the model object that the simulation is a not an optimization.
    modl_objt.set_optimization_flag(0)

    # Run the model for the given inputs, constants and parameters.
    modl_objt.run_model()

    # Read the internal ouputs and simulated discharge.
    otps = modl_objt.get_outputs()
    diss = modl_objt.get_discharge()
    #==========================================================================
    
    
    print('start generating plots:')
    
    if True:
        # Show a figure of the observed vs. simulated river flow.
        fig = plt.figure()

        plt.plot(inp_dfe.index, diso, label='REF', alpha=0.75)
        plt.plot(inp_dfe.index, diss, label='SIM', alpha=0.75)

        plt.grid()
        plt.legend()

        plt.xticks(rotation=45)

        plt.xlabel('Time [hr]')
        plt.ylabel('Discharge\n[$m^3.s^{-1}$]')

        plt.title('Observed vs. Simulated RIver Flow')

        fig.savefig(out_dir / f'observed_vs_simulated_best', bbox_inches='tight', dpi=150)
        plt.close(fig)
    #===========================================================================

    # Show a figure of some of the internally simulated variables of the model.
    # This also serves as a diagnostic tool to check whether what is simulated
    # makes sense or not.
    fig, axs = plt.subplots(8, 1, figsize=(4, 8), dpi=120, sharex=True)

    (axs_tem,
     axs_ppt,
     axs_snw,
     axs_sl0,
     axs_etn,
     axs_rrr,
     axs_rnf,
     axs_bal) = axs
    #===========================================================================

    # Inputs.
    axs_tem.plot(inp_dfe['tavg__ref'], alpha=0.85)
    axs_tem.set_ylabel('TEM\n[°C]')

    axs_ppt.plot(inp_dfe['pptn__ref'], alpha=0.85)
    axs_ppt.set_ylabel('PPT\n[mm]')
    #===========================================================================

    # Snow depth.
    axs_snw.plot(inp_dfe.index, otps[:, otps_lbls['snw_dth']], alpha=0.85)
    axs_snw.set_ylabel('SNW\n[mm]')
    #===========================================================================

    # Mositure level in both soil layers.
    axs_sl0.plot(inp_dfe.index, otps[:, otps_lbls['sl0_dth']], alpha=0.85)
    axs_sl0.set_ylabel('SL0\n[mm]')
    #===========================================================================

    # Potential and simulated evapotranspiration.
    axs_etn.plot(inp_dfe.index, inp_dfe['petn__ref'], label='PET', alpha=0.85)

    axs_etn.plot(
        inp_dfe.index, otps[:, otps_lbls['sl0_etn']], label='ETN', alpha=0.85)

    axs_etn.set_ylabel('ETN\n[mm]')
    axs_etn.legend()
    #===========================================================================

    # Depth of water in the upper and lower reservoirs.
    axs_rrr.plot(
        inp_dfe.index, otps[:, otps_lbls['urr_dth']], label='URR', alpha=0.85)

    axs_rrr.plot(
        inp_dfe.index, otps[:, otps_lbls['lrr_dth']], label='LRR', alpha=0.85)

    axs_rrr.set_ylabel('DTH\n[mm]')
    axs_rrr.legend()
    #===========================================================================

    # Surface and underground runoff.
    axs_rnf.plot(
        inp_dfe.index,
        otps[:, otps_lbls['chn_pow']],
        label='SFC',
        alpha=0.85)

    axs_rnf.plot(
        inp_dfe.index,
        otps[:, otps_lbls['urr_urf']] + otps[:, otps_lbls['lrr_lrf']],
        label='GND',
        alpha=0.85)

    axs_rnf.set_ylabel('RNF\n[mm]')
    axs_rnf.legend()
    #===========================================================================

    # Water balance time series at each time step.
    # Should be close to zero.
    axs_bal.plot(inp_dfe.index, otps[:, otps_lbls['mod_bal']], alpha=0.85)
    axs_bal.set_ylabel('BAL\n[mm]')
    #===========================================================================

    # Some other makeup.
    for ax in axs: ax.grid()

    axs[-1].set_xlabel('Time [hr]')

    plt.xticks(rotation=45)

    plt.suptitle('Inputs, and internally simulated variables of HBV')
    fig.savefig(out_dir / f'Inputs, and internally simulated variables of HBV', bbox_inches='tight', dpi=150)
    plt.close(fig)
    #===========================================================================
    

    # Normalized final best (optimized params only)
    plot_parameters_normalized(bounds_opt, best_params_opt,
                               out_file=str(out_dir / 'params_normalized_best.png'),
                               param_names=PARAM_NAMES_OPT)

    # Scatter for each parameter (all samples vs normalized objective)
    print('- scatter plots')
    plot_param_scatter_all_evals(recorder.eval_params, recorder.eval_objs, bounds_opt,
                                 out_dir=str(out_dir), param_names=PARAM_NAMES_OPT)
    
    #ab hier funktionier noch nicht 
    # # Convergence per generation
    print('- plots to show convergence per generation')
    if len(bounds_opt) > 0:
        run_min = plot_convergence_per_generation(recorder.gen_max_objs, recorder.gen_best_objs, recorder.gen_med_obj,
                                         out_dir)
    
    #Evolution of best-per-generation
    if len(recorder.eval_params) > 0: # gen_best_params
        plot_param_evolution_per_generation(pop_sz, run_min, recorder.eval_params,recorder.gen_best_params, best_params_opt, recorder.gen_params_dvt,
                                            out_dir ,
                                            param_names=PARAM_NAMES_OPT)
    

    # Save optimization history (optimized subset)
    if len(recorder.eval_params) > 0:
        hist_df = pd.DataFrame(np.vstack(recorder.eval_params), columns=PARAM_NAMES_OPT)
        hist_df['objective'] = recorder.eval_objs
        # also record fixed param value (as a column)
        #hist_df['snw_dth'] = FIXED_PARAMS[0]
        hist_df.to_csv(out_dir / 'optimization_history_evals.csv', index=False)
    
    # Per-generation summary
    gen_summary = []
    for gi, (s, e) in enumerate(recorder.gen_bounds, start=1):
        if e > s:
            vals = np.array(recorder.eval_objs[s:e])
            gen_summary.append({
                'generation': gi,
                'n_eval': int(e - s),
                'obj_min': float(np.nanmin(vals)),
                'obj_median': float(np.nanmedian(vals)),
                'obj_max': float(np.nanmax(vals)),
                'best_obj_in_generation': float(recorder.gen_best_objs[gi-1].iloc[0]) if gi-1 < len(recorder.gen_best_objs) else np.nan
            })
        else:
            gen_summary.append({
                'generation': gi,
                'n_eval': 0,
                'obj_min': np.nan,
                'obj_median': np.nan,
                'obj_max': np.nan,
                'best_obj_in_generation': float(recorder.gen_best_objs[gi-1]) if gi-1 < len(recorder.gen_best_objs) else np.nan
            })
    pd.DataFrame(gen_summary).to_csv(out_dir / 'optimization_gen_summary.csv', index=False)

    return

if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack,
    # 2. "up" move the stack up to an older frame,
    # 3. "down" move the stack down to a newer frame, and
    # 4. "interact" start an interactive interpreter.
    #==========================================================================

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
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
