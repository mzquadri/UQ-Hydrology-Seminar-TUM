# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

import numpy as np
cimport numpy as np

# To avoid the annoying type not found warning from PyDev.
from numpy cimport (
    int8_t, int64_t, npy_float32 as float32, npy_float64 as float64)

# To avoid the annoying type not found warning from PyDev.
cdef extern from 'Python.h':
    ctypedef Py_ssize_t Py_ssize_t

cdef extern from 'math.h' nogil:
    cdef float32 powf(float32 x, float32 e)

cdef:
    float32 PINF = +np.float32(np.inf)
    float32 NINF = -np.float32(np.inf)

#==============================================================================
# Variables for the model below.
#
# Variable symbols and their meaning.
# ARF: Actual runoff.
# BAS: Bias.
# CST: Constant.
# DCS: Discretizations.
# DCT: Direct.
# DIS: Discharge or runoff.
# DTH: Depth.
# EXP: Exponent.
# FCY: Field capacity.
# FRZ: Freezing.
# FTR: Factor.
# HRT: Horizontal.
# IDH: Initial depth.
# MDH: Maximum depth.
# PRF: Potential runoff.
# PPT: Precipitation (any form).
# PWP: Permanent wilting point.
# RDN: Radiation.
# RNF: Runoff or discharge.
# RTG: Routing.
# RTO: Ratio.
# SBN: Sublimation.
# SLR: Scaler.
# SRO: Split ratio.
# TEM: Temperature.
# THD: Threshold.
# VRT: Vertical.
# WTR: Water.
#
# Dimension symbols and their meanings.
# L = Length.
# T = Time.
# K = Temperature.
# - = No dimension, or a ratio only.
#==============================================================================

cdef:
    #==========================================================================
    # Input indices.
    #==========================================================================

    Py_ssize_t tem_i = 0, ppt_i = 1, pet_i = 2

    #==========================================================================
    # Parameter indices.
    # (initial conditions, process parameters, total).
    #==========================================================================

    # Snow (1, 3, 4).
    Py_ssize_t prm_snw_dth_i = 0                  # SNW IDH [L].
    Py_ssize_t prm_snw_att_i = prm_snw_dth_i + 1  # Air SNW TEM [K].
    Py_ssize_t prm_snw_pmf_i = prm_snw_att_i + 1  # PPT MLT TEM [L].
    Py_ssize_t prm_snw_amf_i = prm_snw_pmf_i + 1  # Air MLT TEM [L].

    # Soil (1, 3, 4).
    Py_ssize_t prm_sl0_dth_i = prm_snw_amf_i + 1  # IDH [L].
    Py_ssize_t prm_sl0_pwp_i = prm_sl0_dth_i + 1  # PWP [L].
    Py_ssize_t prm_sl0_fcy_i = prm_sl0_pwp_i + 1  # FCY [L].
    Py_ssize_t prm_sl0_bt0_i = prm_sl0_fcy_i + 1  # EXP RNF [-].

    # Reservoir initial conditions (2, 0, 2).
    Py_ssize_t prm_urr_dth_i = prm_sl0_bt0_i + 1  # URR IDH [L].
    Py_ssize_t prm_lrr_dth_i = prm_urr_dth_i + 1  # LRR IDH [L].

    # Upper reservoir (0, 6, 6).
    Py_ssize_t prm_urr_wsr_i = prm_lrr_dth_i + 1  # WTR SRO [-].
    Py_ssize_t prm_urr_ulc_i = prm_urr_wsr_i + 1  # URR-LRR PLN CST [-].
    Py_ssize_t prm_urr_tdh_i = prm_urr_ulc_i + 1  # THD DTH [L].
    Py_ssize_t prm_urr_tdr_i = prm_urr_tdh_i + 1  # THD DRO [-].
    Py_ssize_t prm_urr_ndr_i = prm_urr_tdr_i + 1  # DIS RTO [-].
    Py_ssize_t prm_urr_uct_i = prm_urr_ndr_i + 1  # RNF CST URR-to-URR [-].

    # Lower reservoir (0, 2, 2).
    Py_ssize_t prm_lrr_dre_i = prm_urr_uct_i + 1  # DIS RTO [-].
    Py_ssize_t prm_lrr_lct_i = prm_lrr_dre_i + 1  # RNF CST LRR-to-LRR [-].

    #==========================================================================
    # Output indices. Some are essential for computing water balance.
    #==========================================================================

    # Snow.
    Py_ssize_t out_snw_dth_i = 0                  # Depth [L].
    Py_ssize_t out_snw_pim_i = out_snw_dth_i + 1  # PPT induced melt [L].
    Py_ssize_t out_snw_aim_i = out_snw_pim_i + 1  # Air induced melt [L].
    Py_ssize_t out_snw_mlt_i = out_snw_aim_i + 1  # Total melt [L].

    # Soil.
    Py_ssize_t out_sl0_dth_i = out_snw_mlt_i + 1  # DTH [L].
    Py_ssize_t out_sl0_etn_i = out_sl0_dth_i + 1  # ETN [L]

    # Upper reservoir.
    Py_ssize_t out_urr_dth_i = out_sl0_etn_i + 1  # DTH [L].
    Py_ssize_t out_urr_urf_i = out_urr_dth_i + 1  # URR-to-URR RNF [L].

    # Lower reservoir.
    Py_ssize_t out_lrr_dth_i = out_urr_urf_i + 1  # DTH [L].
    Py_ssize_t out_lrr_lrf_i = out_lrr_dth_i + 1  # LRR-to-LRR RNF [L].

    # Surface flow.
    Py_ssize_t out_chn_pow_i = out_lrr_lrf_i + 1  # Surface RNF [L].

    # Mass balance.
    Py_ssize_t out_mod_bal_i = out_chn_pow_i + 1  # Water balance [L/T].
#==============================================================================

#==============================================================================
# Functions for indices outside cython.
#==============================================================================


def get_idxs_prms_cy():

    cmbs = {
        'snw_dth': prm_snw_dth_i,
        'snw_att': prm_snw_att_i,
        'snw_pmf': prm_snw_pmf_i,
        'snw_amf': prm_snw_amf_i,

        'sl0_dth': prm_sl0_dth_i,
        'sl0_pwp': prm_sl0_pwp_i,
        'sl0_fcy': prm_sl0_fcy_i,
        'sl0_bt0': prm_sl0_bt0_i,

        'urr_dth': prm_urr_dth_i,
        'lrr_dth': prm_lrr_dth_i,

        'urr_wsr': prm_urr_wsr_i,
        'urr_ulc': prm_urr_ulc_i,
        'urr_tdh': prm_urr_tdh_i,
        'urr_tdr': prm_urr_tdr_i,
        'urr_ndr': prm_urr_ndr_i,
        'urr_uct': prm_urr_uct_i,

        'lrr_dre': prm_lrr_dre_i,
        'lrr_lct': prm_lrr_lct_i,
    }

    return cmbs


def get_idxs_otps_cy():

    cmbs = {
        'snw_dth': out_snw_dth_i,
        'snw_pim': out_snw_pim_i,
        'snw_aim': out_snw_aim_i,
        'snw_mlt': out_snw_mlt_i,

        'sl0_dth': out_sl0_dth_i,
        'sl0_etn': out_sl0_etn_i,

        'urr_dth': out_urr_dth_i,
        'lrr_dth': out_lrr_dth_i,

        'urr_urf': out_urr_urf_i,

        'lrr_lrf': out_lrr_lrf_i,

        'chn_pow': out_chn_pow_i,

        'mod_bal': out_mod_bal_i,
    }

    return cmbs


def get_abds_prms_cy():

    '''
    Absolute bounds on each parameter.

    End dtypes are all floats.
    '''

    buds = {
        'snw_dth': (0.00, PINF),  # Initial.
        'snw_att': (NINF, PINF),  # Critical.
        'snw_pmf': (0.00, PINF),  # Optional.
        'snw_amf': (0.00, PINF),  # Critical.

        'sl0_dth': (0.00, PINF),  # Initial.
        'sl0_pwp': (0.00, PINF),  # Critical.
        'sl0_fcy': (0.00, PINF),  # Critical.
        'sl0_bt0': (0.00, PINF),  # Critical.

        'urr_dth': (0.00, PINF),  # Initial.
        'lrr_dth': (0.00, PINF),  # Initial.

        'urr_wsr': (0.00, 1.00),  # Optional.
        'urr_ulc': (0.00, 1.00),  # Critical.
        'urr_tdh': (0.00, PINF),  # Critical.
        'urr_tdr': (0.00, 1.00),  # Critical.
        'urr_ndr': (0.00, 1.00),  # Critical.
        'urr_uct': (0.00, 1.00),  # Critical.

        'lrr_dre': (0.00, 1.00),  # Critical.
        'lrr_lct': (0.00, 1.00),  # Critical.
    }

    return buds
#==============================================================================

#==============================================================================
# The model.
#==============================================================================

cpdef void hbv001a_cy(
        const float32[::1] tems,
        const float32[::1] ppts,
        const float32[::1] pets,
              float32[:, ::1] otps,
              float32[::1] diss,
        const float32[::1] prms,
        const int64_t oflg,
        const float32 dslr,
        ) except +:

    _hbv001a(tems, ppts, pets, otps, diss, prms, oflg, dslr)
    return


cdef void _hbv001a(
        const float32[::1] tems,
        const float32[::1] ppts,
        const float32[::1] pets,
              float32[:, ::1] otps,
              float32[::1] diss,
        const float32[::1] prms,
        const int64_t oflg,
        const float32 dslr,
        ) nogil except +:

    '''
    A lumped conceptual HBV.

    Variant: 001a.
             4 initial values.
             14 process parameters.
             18 parameters in total.

    Parameters:
        tems: Temperature [time]
        ppts: Precipitation [time]
        pets: Potential evapotranspiration [time]
        otps: Outputs [time, variable]
        diss: Discharge that flows out on surface [time]
        prms: Model parameters [parameter]
        oflg: Optimization flag. When True, the otps is only a single step.
              Which is overwritten at each iteration. This spares memory.
    '''

    cdef:
        # Counters.
        Py_ssize_t t, tt

        Py_ssize_t nt = otps.shape[0]
        Py_ssize_t np = prms.shape[1]

        # Loop variables. Some are also parameters that are used as initial
        # values.
        float32 tem, ppt, pet

        float32 lpv_snw_dth, lpv_snw_mlt, lpv_snw_lpt, lpv_snw_pim, lpv_snw_aim

        float32 lpv_sl0_prf, lpv_sl0_dth, lpv_sl0_ero, lpv_sl0_etn 
        float32 lpv_sl0_arf, lpv_sl0_rrm, lpv_sl0_ror, lpv_sl0_iln

        float32 lpv_urr_dth, lpv_lrr_dth

        float32 lpv_urr_pln, lpv_urr_pow

        float32 lpv_lrr_pow

        float32 lpv_chn_pow

        float32 lpv_rnf_gnd
    #==========================================================================

    cdef:
        # Parameters, excluding initial values. 
        float32 prm_snw_att = prms[prm_snw_att_i]
        float32 prm_snw_pmf = prms[prm_snw_pmf_i]
        float32 prm_snw_amf = prms[prm_snw_amf_i]

        float32 prm_sl0_pwp = prms[prm_sl0_pwp_i]
        float32 prm_sl0_fcy = prms[prm_sl0_fcy_i]
        float32 prm_sl0_bt0 = prms[prm_sl0_bt0_i]

        float32 prm_urr_wsr = prms[prm_urr_wsr_i]
        float32 prm_urr_ulc = prms[prm_urr_ulc_i]
        float32 prm_urr_tdh = prms[prm_urr_tdh_i]
        float32 prm_urr_tdr = prms[prm_urr_tdr_i]
        float32 prm_urr_ndr = prms[prm_urr_ndr_i]
        float32 prm_urr_uct = prms[prm_urr_uct_i]  # Updated in another loop.

        float32 prm_lrr_dre = prms[prm_lrr_dre_i]
        float32 prm_lrr_lct = prms[prm_lrr_lct_i]  # Updated in another loop.
    #==========================================================================

    for t in range(nt):

        if oflg:
            tt = 0

        else:
            tt = t
        #======================================================================

        # Inputs.
        tem = tems[t]
        ppt = ppts[t]
        pet = pets[t]
        #======================================================================

        #======================================================================
        # Snow Module (SNW).
        #======================================================================

        if t == 0:
            lpv_snw_dth = prms[prm_snw_dth_i]

        elif oflg:
            lpv_snw_dth = otps[tt, out_snw_dth_i]

        else:
            lpv_snw_dth = otps[t - 1, out_snw_dth_i]
        #======================================================================

        lpv_snw_pim = 0
        lpv_snw_aim = 0
        lpv_snw_mlt = 0
        lpv_snw_lpt = 0
        #======================================================================

        #======================================================================
        # Snowmelt.
        #======================================================================

        if (lpv_snw_dth > 0) and (tem > prm_snw_att):

            lpv_snw_lpt = ppt

            #==================================================================
            # Precipitation induced snow melt.
            #==================================================================

            if (ppt > 0) and (prm_snw_pmf > 0):

                lpv_snw_pim = (tem - prm_snw_att) * prm_snw_pmf
                lpv_snw_pim *= ppt
                lpv_snw_pim = min(lpv_snw_dth, lpv_snw_pim)
                #==============================================================

                if lpv_snw_pim > 0:

                    lpv_snw_dth -= lpv_snw_pim
                    lpv_snw_mlt += lpv_snw_pim
            #==================================================================

            #==================================================================
            # Air induced snow melt.
            #==================================================================

            if lpv_snw_dth > 0:

                lpv_snw_aim = (tem - prm_snw_att) * prm_snw_amf
                lpv_snw_aim = min(lpv_snw_dth, lpv_snw_aim)

                if lpv_snw_aim > 0:

                    lpv_snw_dth -= lpv_snw_aim
                    lpv_snw_mlt += lpv_snw_aim
            #==================================================================

        #======================================================================
        # Snowfall.
        #======================================================================

        elif (tem <= prm_snw_att) and (ppt > 0):

            lpv_snw_dth += ppt

        #======================================================================
        # Rain.
        #======================================================================

        elif (tem > prm_snw_att) and (ppt > 0):

            lpv_snw_lpt = ppt
        #======================================================================

        otps[tt, out_snw_dth_i] = lpv_snw_dth
        otps[tt, out_snw_pim_i] = lpv_snw_pim
        otps[tt, out_snw_aim_i] = lpv_snw_aim
        otps[tt, out_snw_mlt_i] = lpv_snw_mlt
        #======================================================================

        #======================================================================
        # Evapotranspiration, infiltration, soil moisture and runoff.
        #======================================================================

        # Previous soil moisture.
        if t == 0:
            lpv_sl0_dth = prms[prm_sl0_dth_i]

        elif oflg:
            lpv_sl0_dth = otps[tt, out_sl0_dth_i]

        else:
            lpv_sl0_dth = otps[t - 1, out_sl0_dth_i]
        #======================================================================

        # Potential runoff from snow melt and liquid water.
        lpv_sl0_prf = lpv_snw_lpt + lpv_snw_mlt

        # Remaining runoff.
        lpv_sl0_rrm = lpv_sl0_prf

        # Actual runoff.
        lpv_sl0_arf = 0

        if lpv_sl0_rrm:

            # Relative amount that becomes runoff.
            lpv_sl0_ror = powf(lpv_sl0_dth / prm_sl0_fcy, prm_sl0_bt0)

            if lpv_sl0_ror > 1: lpv_sl0_ror = 1

            # Infiltration.
            lpv_sl0_iln = min(
                lpv_sl0_rrm * (<float32> 1 - lpv_sl0_ror),
                prm_sl0_fcy - lpv_sl0_dth)

            lpv_sl0_dth += lpv_sl0_iln
            lpv_sl0_rrm -= lpv_sl0_iln

            lpv_sl0_arf += lpv_sl0_rrm
            lpv_sl0_rrm = 0
        #======================================================================

        # Evapotranspiration.
        if prm_sl0_pwp > 0:
            lpv_sl0_ero = lpv_sl0_dth / prm_sl0_pwp

        else:
            lpv_sl0_ero = 0

        if lpv_sl0_ero > 1: lpv_sl0_ero = 1

        if (tem > 0) and (lpv_snw_dth == 0):

            lpv_sl0_etn = lpv_sl0_ero * pet
            lpv_sl0_etn = min(lpv_sl0_dth, lpv_sl0_etn)

            lpv_sl0_dth -= lpv_sl0_etn

        else:
            lpv_sl0_etn = 0
        #======================================================================

        otps[tt, out_sl0_dth_i] = lpv_sl0_dth
        otps[tt, out_sl0_etn_i] = lpv_sl0_etn
        #======================================================================

        #======================================================================
        # Runoff routing within cell/catchment (non-channel).
        #======================================================================

        if t == 0:
            lpv_urr_dth = prms[prm_urr_dth_i]
            lpv_lrr_dth = prms[prm_lrr_dth_i]

        elif oflg:
            lpv_urr_dth = otps[tt, out_urr_dth_i]
            lpv_lrr_dth = otps[tt, out_lrr_dth_i]

        else:
            lpv_urr_dth = otps[t - 1, out_urr_dth_i]
            lpv_lrr_dth = otps[t - 1, out_lrr_dth_i]

        lpv_chn_pow = 0
        #======================================================================

        # Add part of incoming water.
        lpv_urr_dth += lpv_sl0_arf * (<float32> 1 - prm_urr_wsr)
        #======================================================================

        #======================================================================
        # Upper reservoir. Sequence of outflows based on speed.
        #======================================================================

        # URR upper-outlet discharge.
        if lpv_urr_dth > prm_urr_tdh:

            lpv_urr_pow = (lpv_urr_dth - prm_urr_tdh) * prm_urr_tdr

            lpv_chn_pow += lpv_urr_pow
            lpv_urr_dth -= lpv_urr_pow
            #==================================================================

        # Discharge to channel.
        lpv_urr_pow = min(lpv_urr_dth, prm_urr_tdh) * prm_urr_ndr

        lpv_chn_pow += lpv_urr_pow
        lpv_urr_dth -= lpv_urr_pow

        # URR-to-LRR transfer in same cell.
        lpv_urr_pln = lpv_urr_dth * prm_urr_ulc
        lpv_urr_dth -= lpv_urr_pln
        #======================================================================

        #======================================================================
        # Lower reservoir. Sequence of outflows based on speed.
        #======================================================================

        # Add incoming water.
        lpv_lrr_dth += lpv_urr_pln

        # Discharge to channel.
        lpv_lrr_pow = lpv_lrr_dth * prm_lrr_dre

        lpv_chn_pow += lpv_lrr_pow
        lpv_lrr_dth -= lpv_lrr_pow
        #======================================================================

        #======================================================================
        # Surface discharge routing (1D case).
        #======================================================================

        otps[tt, out_chn_pow_i] = lpv_chn_pow

        #======================================================================
        # Sub-surface discharge routing (1D case).
        #======================================================================

        # URR-to-URR.
        otps[tt, out_urr_urf_i] = lpv_urr_dth * prm_urr_uct

        lpv_urr_dth -= otps[tt, out_urr_urf_i]

        # LRR-to-LRR.
        otps[tt, out_lrr_lrf_i] = lpv_lrr_dth * prm_lrr_lct

        lpv_lrr_dth -= otps[tt, out_lrr_lrf_i]
        #======================================================================

        # URR remaining runoff update (1D case).
        lpv_urr_dth += lpv_sl0_arf * prm_urr_wsr

        otps[tt, out_urr_dth_i] = lpv_urr_dth
        otps[tt, out_lrr_dth_i] = lpv_lrr_dth
        #======================================================================

        # River discharge.
        diss[t] = lpv_chn_pow * dslr
        #======================================================================

        #======================================================================
        # Mass Balance
        #======================================================================

        lpv_rnf_gnd = otps[tt, out_urr_urf_i] + otps[tt, out_lrr_lrf_i]

        # A full length otps array is required, so this only takes place
        # when oflag is False. Values in this column should all be zeros.
        if oflg == 0:
            otps[t, out_mod_bal_i] = (
                ppt - (lpv_sl0_etn + lpv_chn_pow + lpv_rnf_gnd))

            if t == 0:
                otps[t, out_mod_bal_i] += (
                    prms[prm_snw_dth_i] +
                    prms[prm_sl0_dth_i] +
                    prms[prm_urr_dth_i] +
                    prms[prm_lrr_dth_i])

                otps[t, out_mod_bal_i] -= (
                    lpv_snw_dth +
                    lpv_sl0_dth +
                    lpv_urr_dth +
                    lpv_lrr_dth)

            else:
                otps[t, out_mod_bal_i] -= (
                    (lpv_snw_dth - otps[t - 1, out_snw_dth_i]) +
                    (lpv_sl0_dth - otps[t - 1, out_sl0_dth_i]) +
                    (lpv_urr_dth - otps[t - 1, out_urr_dth_i]) +
                    (lpv_lrr_dth - otps[t - 1, out_lrr_dth_i]))
        #======================================================================
    return
