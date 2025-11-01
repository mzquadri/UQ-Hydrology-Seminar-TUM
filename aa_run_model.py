# -*- coding: utf-8 -*-

'''
@author: Faizan-TU Munich

21.10.2025

09:31:38

Description: Demonstration of 1D 001A variant of HBV.

Keywords:

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

DEBUG_FLAG = False


def main():

    # Absolute path to the directory where the input data lies.
    main_dir = Path(r'D:\Portable_Python_Setup\EclipsePortable426\Data\workspace\TEST_Codes\math_methods\data')
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
    
    # Read model and related files in the models directory for more info.
    # Correct sequence must be followed. Values that are out of
    # the absolute parameter range will result in an AssertionError.
    prms = np.array([
        0.00,  # 'snw_dth'
        +0.1,  # 'snw_att'
        0.01,  # 'snw_pmf'
        0.10,  # 'snw_amf'

        0.00,  # 'sl0_dth'
        300.,  # 'sl0_pwp'
        70.0,  # 'sl0_fcy'
        2.50,  # 'sl0_bt0'

        0.00,  # 'urr_dth'
        0.00,  # 'lrr_dth'

        1.00,  # 'urr_wsr'
        0.01,  # 'urr_ulc'
        30.0,  # 'urr_tdh'
        0.00,  # 'urr_tdr'
        0.10,  # 'urr_ndr'
        0.01,  # 'urr_uct'

        1e-3,  # 'lrr_dre'
        1e-5,  # 'lrr_dct'
        ], dtype=np.float32)
    #==========================================================================

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
    #==========================================================================

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

        plt.show()
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
    plt.show()

    plt.close(fig)
    #===========================================================================
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
