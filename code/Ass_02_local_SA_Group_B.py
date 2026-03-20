'''
Group B 

local Sensitivity Analysis

'''

import os
import sys
import time
import timeit
import math
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hmg import HBV001A

DEBUG_FLAG = False

# Add project root to Python path so 'hmg' module can be found
project_root = Path(__file__).resolve().parent  
sys.path.insert(0, str(project_root))

def nse(obs, sim):
    """
    Nash-Sutcliffe Efficiency (robust).
    Returns: float NSE or np.nan if undefined (e.g. zero variance or no valid data).
    """
    obs = np.asarray(obs, dtype=np.float32)
    sim = np.asarray(sim, dtype=np.float32)
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    if mask.sum() == 0:
        return np.nan
    o = obs[mask]
    s = sim[mask]
    denom = np.sum((o - o.mean(dtype=np.float32)) ** np.float32(2.0), dtype=np.float32)
    if denom == 0.0:
        return np.nan
    num = np.sum((s - o) ** np.float32(2.0), dtype=np.float32)
    return float(np.float32(1.0) - num / denom)

def obj_nse_pzt(obj_nse, y0):
    """
    Objective to minimize: 1 - NSE. If NSE undefined, return PENALTY.
    """

    dta_y = obj_nse - y0
    dta_y_pzt = (dta_y / y0) * 100
    
    return dta_y_pzt

def obj_nse(obs, sim):
    """
    Objective to minimize: 1 - NSE. If NSE undefined, return PENALTY.
    """
    val = nse(obs, sim)
    
    return 1.0 - float(val)

def plot_diss(inp_dfe, diso, diss, out_dir, change):
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

        #plt.show()
        fig.savefig(out_dir / f'observed_vs_simulated_best_{change}', bbox_inches='tight', dpi=150)
        plt.close(fig)
        
def plot_input(inp_dfe, otps, otps_lbls, out_dir, change):
    
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
    fig.savefig(out_dir / f'Inputs, and internally simulated variables of HBV {change}', bbox_inches='tight', dpi=150)
    #plt.show()

    plt.close(fig)
    #===========================================================================
    
def plot_param_scatter(X, Y, out_dir, param_names):
    """
    Scatter plots for each parameter:
      - X-axis: raw parameter values (original scale from bounds)
      - Y-axis: normalized objective values (0 = best, 1 = worst)
    """
    if len(X) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)

    # X: raw parameter values; Y: normalized objective
    ax.scatter(X, Y, s=10, alpha=0.6)
    
    ax.set_xlabel(f'{param_names}: rel. value change in [%]')
    ax.set_ylabel('rel. change of Objective function value in [%]')
    ax.set_title(f'relative sampled values change vs relative objective function change')
    ax.grid(True)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'param_scatter_{param_names}.png'))
    plt.close(fig)
    #===========================================================================
    
def plot_param_scatter_T(X, Y, out_dir, param_names):
    """
    Scatter plots for each parameter:
      - X-axis: raw parameter values (original scale from bounds)
      - Y-axis: normalized objective values (0 = best, 1 = worst)
    """
    if len(X) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)

    # X: raw parameter values; Y: normalized objective
    ax.scatter(X, Y, s=10, alpha=0.6)
            
    #ax.set_yscale('log')
    ax.set_xlabel(f'{param_names}: value change in Temperature [K]')
    ax.set_ylabel('rel. change of Objective function value in [%]')
    ax.set_title(f'relative sampled values change vs relative objective function change')
    ax.grid(True)
        
    # lock y-range to [0, 1] explicitly
    #ax.set_ylim(0.0, 1.0)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'param_scatter_{param_names}.png'))
    plt.close(fig)
    #===========================================================================

def main():

    # Absolute path to the directory where the input data lies.
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
    
    PARAM_NAMES_OPT = [
    'snw_dth', 'snw_att', 'snw_pmf', 'snw_amf',
    'sl0_dth', 'sl0_pwp', 'sl0_fcy', 'sl0_bt0',
    'urr_dth', 'lrr_dth', 'urr_wsr', 'urr_ulc',
    'urr_tdh', 'urr_tdr', 'urr_ndr', 'urr_uct',
    'lrr_dre', 'lrr_lct'
    ]
    
    bounds_opt = [
        (0.0, 0.0),      # snw_dth
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
    
    # Read model and related files in the models directory for more info.
    # Correct sequence must be followed. Values that are out of
    # the absolute parameter range will result in an AssertionError.
    prms = np.array([
        0.00,  # 'snw_dth'
        -0.006978000048547983,  # 'snw_att'
        0.5291069746017456,  # 'snw_pmf'
        0.04326999932527542,  # 'snw_amf'

        97.91517639160156,  # 'sl0_dth'
        612.0751953125,  # 'sl0_pwp'
        163.34249877929688,  # 'sl0_fcy'
        2.9130570888519287,  # 'sl0_bt0'

        0.9689339995384216,  # 'urr_dth'
        1.6860179901123047,  # 'lrr_dth'

        0.9866669774055481,  # 'urr_wsr'
        0.18052799999713898,  # 'urr_ulc'
        40.24491882324219,  # 'urr_tdh'
        0.6858029961585999,  # 'urr_tdr'
        1.9999999494757503e-05,  # 'urr_ndr'
        0.00011000000085914508,  # 'urr_uct'

        0.009052000008523464,  # 'lrr_dre'
        4.999999873689376e-06,  # 'lrr_lct'
        ], dtype=np.float32)
    #==========================================================================
    
    out_dir_0 = project_root / 'outputs' / 'assignment2'
    out_dir_0.mkdir(parents=True, exist_ok=True)
            
    # Loop through the model parameters (X)
    #==========================================================================
    X = []
    x = np.linspace(-30, 30, 120)
    x_T = np.linspace(-0.25, 0.25, 120)
 
    for i in range(0, len(prms)): 
        if i != 1: # i=1 is prms: snw_att 
            for j in np.linspace(-30, 30, 120):
                if j < 0: 
                    x_lpv = prms[i]*(100 + j)/100
                    X.append(max(x_lpv,bounds_opt[i][0]))
                    
                else: 
                    x_lpv = prms[i]*(100+ j)/100
                    X.append(min(x_lpv,bounds_opt[i][1]))    
                    
        else:
            for j in np.linspace(-0.25, 0.25, 120):
            
                x_lpv = prms[i] + j
                if j < 0: 
                    X.append(max(x_lpv,bounds_opt[i][0]))
                else:
                    X.append(min(x_lpv,bounds_opt[i][1]))  
                
    # document at which point the new relative calculated values exceed the parameter boundaries 
    with open(out_dir_0/'summary_outside_boundaries.txt', 'w') as txt_hdl:
        txt_hdl.write(f'\n')
    for i in range(0, len(prms)): 
        with open(out_dir_0/'summary_outside_boundaries.txt', 'a') as txt_hdl:
            txt_hdl.write(f'\n for {PARAM_NAMES_OPT[i]} following relative values [X] are outside the boundaries:\n')
        for j in range(0,119):
            if X[i*120 +j] <= bounds_opt[i][0]:
                with open(out_dir_0/'summary_outside_boundaries.txt', 'a') as txt_hdl:
                    txt_hdl.write(f'True (lower B.) index:{i*120 +j}, X: {x[j]}%, value: {X[i*120 +j]}\n')
    
            elif X[i*120 +j] >= bounds_opt[i][1]:
                with open(out_dir_0/'summary_outside_boundaries.txt', 'a') as txt_hdl:
                    txt_hdl.write(f'True (upper B.) index:{i*120 +j}, X: {x[j]}%, value: {X[i*120 +j]}\n')
            

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
            
    # Show the parameters against their names as a check.
    print('')
            
    print('Model parameters:')
    for prm_lbl, i in modl_objt.get_parameter_labels().items():
        print(f'{prm_lbl}:', round(prms[i], 6))
            
    print('')
    
    # Run model with optimized parameter to generate Y_0 as reference         
    # Get a dictionary that links an output labe to its column index in the
    # ouputs array.
    otps_lbls = modl_objt.get_output_labels()
    
    # Pass the parameters.
    modl_objt.set_parameters(prms)

    # Tell the model object that the simulation is a not an optimization.
    modl_objt.set_optimization_flag(0)
            
    # Run the model for the given inputs, constants and parameters.
    modl_objt.run_model()

    # Read the internal ouputs and simulated discharge.
    otps = modl_objt.get_outputs()
    diss = modl_objt.get_discharge()
    
    Y_0 = obj_nse(diso, diss)
    Y = []
    
    out_dir_1 = out_dir_0 / 'plot' / 'SA'
    out_dir_1.mkdir(parents=True, exist_ok=True)
            
    with open(out_dir_1/'summary_values_the_perc_range.txt', 'w') as txt_hdl:
        txt_hdl.write(f'\n')
    
    for i in range(0, len(X)):
        
        j = math.floor(i/120) #to get the current parameter 
        print(f'pars:{j+1}')
        print(f'run: {i}')
        prms_lpv = prms.copy()
        prms_lpv[j] = X[i]
        
        # Pass the parameters.
        modl_objt.set_parameters(prms_lpv)

        # Tell the model object that the simulation is a not an optimization.
        modl_objt.set_optimization_flag(0)
            
        # Run the model for the given inputs, constants and parameters.
        modl_objt.run_model()

        # Read the internal ouputs and simulated discharge.
        otps = modl_objt.get_outputs()
        diss = modl_objt.get_discharge()
        
        nse_test = obj_nse(diso, diss)
        y = obj_nse_pzt(nse_test, Y_0)
        print(f'NSE = {1-nse_test}')
        Y.append(y)
                
                
        if i == j*120 + 0:
            print('-30')
            out_dir = out_dir_0 / 'plot' / 'SA' / 'minus_30_percent'
            out_dir.mkdir(parents=True, exist_ok=True)
            out_dir_1 = out_dir_0 / 'plot' / 'SA'
            out_dir_1.mkdir(parents=True, exist_ok=True)
            
            with open(out_dir_1/'summary_values_the_perc_range.txt', 'a') as txt_hdl:
                txt_hdl.write(f'\n'\
                              f'-30% {PARAM_NAMES_OPT[j]}: {X[i]}')
     
            plot_diss(inp_dfe, diso, diss, out_dir, str(f'{PARAM_NAMES_OPT[j]}_minus_30_percent'))
            plot_input(inp_dfe, otps, otps_lbls, out_dir, str(f'{PARAM_NAMES_OPT[j]}_minus_30_percent'))

        elif i == j*120 + 19: 
            print('-20')
            out_dir = out_dir_0 / 'plot' / 'SA' / 'minus_20_percent'
            out_dir.mkdir(parents=True, exist_ok=True)
     
            plot_diss(inp_dfe, diso, diss, out_dir, str(f'{PARAM_NAMES_OPT[j]}_minus_20_percent'))
            plot_input(inp_dfe, otps, otps_lbls, out_dir, str(f'{PARAM_NAMES_OPT[j]}_minus_20_percent'))
            
        elif i == j*120 +39: 
            print('-10')
            out_dir = out_dir_0 / 'plot' / 'SA' / 'minus_10_percent'
            out_dir.mkdir(parents=True, exist_ok=True)
     
            plot_diss(inp_dfe, diso, diss, out_dir, str(f'{PARAM_NAMES_OPT[j]}_minus_10_percent'))
            plot_input(inp_dfe, otps, otps_lbls, out_dir, str(f'{PARAM_NAMES_OPT[j]}_minus_10_percent'))
                
        elif i == j*120 +79: 
            print('+10')
            out_dir = out_dir_0 / 'plot' / 'SA' / 'plus_10_percent'
            out_dir.mkdir(parents=True, exist_ok=True)
     
            plot_diss(inp_dfe, diso, diss, out_dir, str(f'{PARAM_NAMES_OPT[j]}_plus_10_percent'))
            plot_input(inp_dfe, otps, otps_lbls, out_dir, str(f'{PARAM_NAMES_OPT[j]}_plus_10_percent'))
               
        elif i == j*120 +99: 
            print('+20')
            out_dir = out_dir_0 / 'plot' / 'SA' / 'plus_20_percent'
            out_dir.mkdir(parents=True, exist_ok=True)
     
            plot_diss(inp_dfe, diso, diss, out_dir, str(f'{PARAM_NAMES_OPT[j]}_plus_20_percent'))
            plot_input(inp_dfe, otps, otps_lbls, out_dir, str(f'{PARAM_NAMES_OPT[j]}_plus_20_percent'))
                
        elif i == j*120 + 119: 
            print('+30')
            out_dir = out_dir_0 / 'plot' / 'SA' / 'plus_30_percent'
            out_dir.mkdir(parents=True, exist_ok=True)
            
            out_dir_1 = out_dir_0 / 'plot' / 'SA'
            out_dir_1.mkdir(parents=True, exist_ok=True)
            
            with open(out_dir_1/'summary_values_the_perc_range.txt', 'a') as txt_hdl:
                txt_hdl.write(f'\n'\
                              f'+30% {PARAM_NAMES_OPT[j]}: {X[i]}')
     
            plot_diss(inp_dfe, diso, diss, out_dir, str(f'{PARAM_NAMES_OPT[j]}_plus_30_percent'))
            plot_input(inp_dfe, otps, otps_lbls, out_dir, str(f'{PARAM_NAMES_OPT[j]}_plus_30_percent'))
                
        prms_lpv  = []
    #==========================================================================
    out_dir = out_dir_0 / 'plot' / 'SA' 
    out_dir.mkdir(parents=True, exist_ok=True)
    
   
    for i in range(0, len(prms)):
        if i !=1:        
            j = i*120
            plot_param_scatter(x, Y[j : (j + 120)], out_dir, PARAM_NAMES_OPT[i])
        else:
            plot_param_scatter_T(x_T, Y[j : (j + 120)], out_dir, PARAM_NAMES_OPT[i])
        
    
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
