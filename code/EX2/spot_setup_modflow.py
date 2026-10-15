'''
Code implemented after an example from Tobias Houska (2018), 
part of Statistical Parameter Estimation Tool (SPOTPY).
:author: Alexandra Geist, modified by Tanu Singh
This example implements the python version of MODFLOW into SPOTPY.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import numpy as np
import pandas as pd
import spotpy
from spotpy.parameter import Uniform

from additional_functions import load_modflow_model, load_heads, get_head_ts, HiddenPrints, process_model_path


class spot_setup(object):

    def __init__(self, _used_algorithm):
        self._used_algorithm = _used_algorithm
        # !!! Specify the prior !!!!!

        # This setup expects the parameters to be uniform distributed in [log10(min), log10(max)]
        # on a logarithmic scale.
        # the 10^x is done in simulation
        self.params = [Uniform("parcond_1", low=-5, high=-1),
                       Uniform("parcond_2", low=-5, high=-1),
                       Uniform("parcond_3", low=-7, high=-2),
                       Uniform("parcond_4", low=-7, high=-2),
                       Uniform("parcond_5", low=-7, high=-2)]


    def parameters(self):
        return spotpy.parameter.generate(self.params)

    @staticmethod
    def simulation(x):
        # with HiddenPrints():

        print(f'run Model', flush=True)
        model_name = "flood.nam"
        model_path = os.path.join("..", "flood_riv_dream")
        ml = load_modflow_model(model_name, model_path)
        #print('test', flush=True)
        cond_mask = np.load('mask_new.npy')  # load a mask that defines where are cond_1,_2,_3,_4,_5
        cond_mask[cond_mask == 1] = 10 ** (x[0])  # where the mask equals 1 -> parameter 1
        cond_mask[cond_mask == 2] = 10 ** (x[1])  # where the mask equals 2 -> parameter 2
        cond_mask[cond_mask == 3] = 10 ** (x[2])  # where the mask equals 3 -> parameter 3
        cond_mask[cond_mask == 4] = 10 ** (x[3])  # where the mask equals 4 -> parameter 4
        cond_mask[cond_mask == 5] = 10 ** (x[4])  # where the mask equals 5 -> parameter 5
        #print(cond_mask, flush=True)
        for i in range(0, len(ml.riv.stress_period_data.data), 1):
            ml.riv.stress_period_data.data[i]['cond'] = cond_mask
        ml.riv.write_file()
        #print('test_02', flush=True)
        ml.run_model(silent=True)

        heads, hfile = load_heads(process_model_path(model_path))
        ts_1 = get_head_ts(157, 158, hfile)
        ts_2 = get_head_ts(220, 99, hfile)
        ts_3 = get_head_ts(127, 161, hfile)
        ts_4 = get_head_ts(41, 125, hfile)
        sim = ts_1.append(ts_2).append(ts_3).append(ts_4)

        return list(sim[1])

    def evaluation(self):
        path = r'C:\Users\chris\Documents\TUM\Master\WS25_26\Math_Meth\MOOC\Exercise_02\Ex 2 parallel DREAM-20260111'
        file_path = os.path.join(path, "obs_flood_multi.pkl")
        #obs = pd.read_pickle("obs_flood_multi.pkl") # ursprüngliche version
        obs = pd.read_pickle(file_path)
        # obs = obs.reset_index(drop = True)
        return obs[0]

    def objectivefunction(self, simulation, evaluation, params=None):
        like = spotpy.likelihoods.gaussianLikelihoodMeasErrorOut(evaluation, simulation)
        return like
