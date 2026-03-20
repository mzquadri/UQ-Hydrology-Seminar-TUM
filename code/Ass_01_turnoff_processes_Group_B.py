'''
Group B
Assignment 1 - turn off Processes 

using found optimal model Parameter
'''
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hmg import HBV001A

DEBUG_FLAG = False

# Add project root to Python path so 'hmg' module can be found
project_root = Path(__file__).resolve().parent  
sys.path.insert(0, str(project_root))
print(project_root)

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

def obj_nse(obs, sim):
    """
    Objective to minimize: 1 - NSE. If NSE undefined, return PENALTY.
    """
    val = nse(obs, sim)
    
    return 1.0 - float(val)

def plot_diss(inp_dfe, diso, diss_urr, diss_lrr, diss_snw, out_dir):
    # Show a figure of the observed vs. simulated river flow.
        fig = plt.figure()

        plt.plot(inp_dfe.index, diso, label='REF', alpha=0.75)
        #plt.plot(inp_dfe.index, diss_run, label='100% runoff', alpha=0.75)
        plt.plot(inp_dfe.index, diss_urr, label='tunoff: urr', alpha=0.75)
        plt.plot(inp_dfe.index, diss_lrr, label='tunoff: lrr', alpha=0.75)
        #plt.plot(inp_dfe.index, diss_GW, label='tunoff: GW', alpha=0.75)
        plt.plot(inp_dfe.index, diss_snw, label='tunoff: snw', alpha=0.75)

        plt.grid()
        plt.legend()

        plt.xticks(rotation=45)

        plt.xlabel('Time [hr]')
        plt.ylabel('Discharge\n[$m^3.s^{-1}$]')

        plt.title('Observed and turned off Processes')

        #plt.show()
        fig.savefig(out_dir / f'observed_with_turnoff', bbox_inches='tight', dpi=150)
        plt.close(fig)
        
def plot_otps(inp_dfe, otps, otps_lbls, out_dir, name):
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
    # #===========================================================================

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
    #plt.show()
    fig.savefig(out_dir / f'otps_with_turnoff_{name}', bbox_inches='tight', dpi=150)
    plt.close(fig)
    

def run_model(tems, ppts, pets, diso, tsps, dslr, prms):
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
    nse_test = nse(diso, diss)
    print(nse_test)
    
    return diss, otps, nse_test, otps_lbls
    #==========================================================================


print('start')
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
     
    # opt. model parameter + tunroff verions 
prms = np.array([
    0.00,  # 'snw_dth'
    -0.006045000161975622,  # 'snw_att'
    0.5223780274391174,  # 'snw_pmf'
    0.04626699909567833,  # 'snw_amf'

    96.13312530517578,  # 'sl0_dth'
    695.4364624023438,  # 'sl0_pwp'
    161.8762969970703,  # 'sl0_fcy'
    2.799398899078369,  # 'sl0_bt0'

    0.45157501101493835,  # 'urr_dth'
    2.01528000831604,  # 'lrr_dth'

    0.9972169995307922,  # 'urr_wsr'
    0.1752610057592392,  # 'urr_ulc'
    37.125091552734375,  # 'urr_tdh'
    0.10983400046825409,  # 'urr_tdr'
    1.8000000636675395e-05,  # 'urr_ndr'
    2.099999983329326e-05,  # 'urr_uct'

    0.009208999574184418,  # 'lrr_dre'
    9.999999974752427e-07,  # 'lrr_lct'
    ], dtype=np.float32)
    #=========================
    
prms_100runoff = np.array([
        0.00,  # 'snw_dth'
        -0.006045000161975622,  # 'snw_att'
        0.5223780274391174,  # 'snw_pmf'
        0.04626699909567833,  # 'snw_amf'

        96.13312530517578,  # 'sl0_dth'
        695.4364624023438,  # 'sl0_pwp'
        1.0e-10,  # 'sl0_fcy'
        2.799398899078369,  # 'sl0_bt0'

        0.45157501101493835,  # 'urr_dth'
        2.01528000831604,  # 'lrr_dth'

        0.9972169995307922,  # 'urr_wsr'
        0.1752610057592392,  # 'urr_ulc'
        37.125091552734375,  # 'urr_tdh'
        0.10983400046825409,  # 'urr_tdr'
        1.8000000636675395e-05,  # 'urr_ndr'
        2.099999983329326e-05,  # 'urr_uct'

        0.009208999574184418,  # 'lrr_dre'
        9.999999974752427e-07,  # 'lrr_lct'
        ], dtype=np.float32)
    
prms_trunoff_urr = np.array([
    0.00,  # 'snw_dth'
    -0.006045000161975622,  # 'snw_att'
    0.5223780274391174,  # 'snw_pmf'
    0.04626699909567833,  # 'snw_amf'

    96.13312530517578,  # 'sl0_dth'
    695.4364624023438,  # 'sl0_pwp'
    161.8762969970703,  # 'sl0_fcy'
    2.799398899078369,  # 'sl0_bt0'

    0.45157501101493835,  # 'urr_dth'
    2.01528000831604,  # 'lrr_dth'

    0.9972169995307922,  # 'urr_wsr'
    0.1752610057592392,  # 'urr_ulc'
    37.125091552734375,  # 'urr_tdh'
    0.0,  # 'urr_tdr'
    0.0,  # 'urr_ndr'
    0.0,  # 'urr_uct'

    0.009208999574184418,  # 'lrr_dre'
    9.999999974752427e-07,  # 'lrr_lct'
    ], dtype=np.float32)
    
prms_trunoff_Irr = np.array([
    0.00,  # 'snw_dth'
    -0.006045000161975622,  # 'snw_att'
    0.5223780274391174,  # 'snw_pmf'
    0.04626699909567833,  # 'snw_amf'

    96.13312530517578,  # 'sl0_dth'
    695.4364624023438,  # 'sl0_pwp'
    161.8762969970703,  # 'sl0_fcy'
    2.799398899078369,  # 'sl0_bt0'

    0.45157501101493835,  # 'urr_dth'
    2.01528000831604,  # 'lrr_dth'

    0.9972169995307922,  # 'urr_wsr'
    0.1752610057592392,  # 'urr_ulc'
    37.125091552734375,  # 'urr_tdh'
    0.10983400046825409,  # 'urr_tdr'
    1.8000000636675395e-05,  # 'urr_ndr'
    0.0,  # 'urr_uct'
    0.0,  # 'lrr_dre'
    9.999999974752427e-07,  # 'lrr_lct'
    ], dtype=np.float32)
    
prms_trunoff_GW = np.array([
       0.00,  # 'snw_dth'
    -0.006045000161975622,  # 'snw_att'
    0.5223780274391174,  # 'snw_pmf'
    0.04626699909567833,  # 'snw_amf'

    96.13312530517578,  # 'sl0_dth'
    695.4364624023438,  # 'sl0_pwp'
    161.8762969970703,  # 'sl0_fcy'
    2.799398899078369,  # 'sl0_bt0'

    0.45157501101493835,  # 'urr_dth'
    2.01528000831604,  # 'lrr_dth'

    0.9972169995307922,  # 'urr_wsr'
    0.1752610057592392,  # 'urr_ulc'
    37.125091552734375,  # 'urr_tdh'
    0.10983400046825409,  # 'urr_tdr'
    1.8000000636675395e-05,  # 'urr_ndr'
    0.0,  # 'urr_uct'

    0.009208999574184418,  # 'lrr_dre'
    0.0,  # 'lrr_lct'
    ], dtype=np.float32)
    
prms_trunoff_snw = np.array([
    0.00,  # 'snw_dth'
    -100.0,  # 'snw_att'
    0.5223780274391174,  # 'snw_pmf'
    0.04626699909567833,  # 'snw_amf'

    96.13312530517578,  # 'sl0_dth'
    695.4364624023438,  # 'sl0_pwp'
    161.8762969970703,  # 'sl0_fcy'
    2.799398899078369,  # 'sl0_bt0'

    0.45157501101493835,  # 'urr_dth'
    2.01528000831604,  # 'lrr_dth'

    0.9972169995307922,  # 'urr_wsr'
    0.1752610057592392,  # 'urr_ulc'
    37.125091552734375,  # 'urr_tdh'
    0.10983400046825409,  # 'urr_tdr'
    1.8000000636675395e-05,  # 'urr_ndr'
    2.099999983329326e-05,  # 'urr_uct'

    0.009208999574184418,  # 'lrr_dre'
    9.999999974752427e-07,  # 'lrr_lct'
    ], dtype=np.float32)
    
# creat output directory
out_dir = project_root / 'outputs' / 'assignment1' / 'turnoff'
out_dir.mkdir(parents=True, exist_ok=True)
print(f'output directory: {out_dir}')
    #==========================================================================
print('')
print('run model')
diss, otps, nse_test, otps_lbls = run_model(tems, ppts, pets, diso, tsps, dslr, prms)
print('')
print('run with turned off urr')   
diss_urr, otps_urr, nse_test_urr, otps_lbls_urr = run_model(tems, ppts, pets, diso, tsps, dslr, prms_trunoff_urr)
print('')
print('run with turned off lrr') 
diss_lrr, otps_lrr, nse_test_lrr, otps_lbls_lrr = run_model(tems, ppts, pets, diso, tsps, dslr, prms_trunoff_Irr)
print('')
print('run with turned off GW-flow') 
diss_GW, otps_GW, nse_test_GW, otps_lbls_GW = run_model(tems, ppts, pets, diso, tsps, dslr, prms_trunoff_GW)
print('')
print('run with turned off snw') 
diss_snw, otps_snw, nse_test_snw, otps_lbls_snw = run_model(tems, ppts, pets, diso, tsps, dslr, prms_trunoff_snw)
print('')
print('create plots:')
print('- diss')

plot_diss(inp_dfe, diso, diss_urr, diss_lrr, diss_snw, out_dir)
print('- otps')
plot_otps(inp_dfe, otps, otps_lbls, out_dir, 'unchagned')
plot_otps(inp_dfe, otps_urr, otps_lbls, out_dir, 'urr')
plot_otps(inp_dfe, otps_lrr, otps_lbls, out_dir, 'lrr')
plot_otps(inp_dfe, otps_GW, otps_lbls, out_dir, 'GW')
plot_otps(inp_dfe, otps_snw, otps_lbls, out_dir, 'snw')
print('')
print('save NSE values')
print('')
with open(out_dir /'NSE_values_tunroff.txt', 'w') as f:
        f.write("NSE values of tured off processes\n")
        f.write(f"NSE without change = {nse_test}\n")
        f.write(f"NSE_urr = {nse_test_urr}\n")
        f.write(f"NSE_lrr = {nse_test_lrr}\n")
        f.write(f"NSE_GW = {nse_test_GW}\n")
        f.write(f"NSE_snw = {nse_test_snw}\n")

print('everythink done')
