'''
Code implemented after an example from Tobias Houska (2018), 
part of Statistical Parameter Estimation Tool (SPOTPY).
:author: Alexandra Geist, modified by Tanu Singh
This example implements the python version of MODFLOW into SPOTPY.
'''

import spotpy
import numpy as np
from spot_setup_modflow import spot_setup
import random
from mpi4py import MPI
from additional_functions import process_model_path

import pickle as pkl

if __name__ == "__main__":
    # parallel = 'seq'

    # to run the DREAM in parallel using mpi, start this script from the
    # commandline with
    # $ mpiexec -n 1 python run_dream.py
    # with as many cores you want to use.
    # If you have enabled hyper-threading you can use the option --use-hwthread-cpus
    # to start 2 processes on every physical core.
    parallel = 'mpi' #'seq' #'mpi'

    import os
    model_name = "flood.nam"
    #model_path = os.path.join("..", "flood_riv_dream")
    model_path = r"C:\Users\chris\Documents\TUM\Master\WS25_26\Math_Meth\MOOC\Exercise_02\Ex 2 parallel DREAM-20260111\flood_riv_dream"

    # make separated modflow working dir for every process.
    comm = MPI.COMM_WORLD
    if not os.path.exists(process_model_path(model_path)):
        from distutils.dir_util import copy_tree
        # copy flood dir data to process dir
        fromDirectory = model_path
        toDirectory = process_model_path(model_path)
        copy_tree(fromDirectory, toDirectory)

    print("run_dream with parallel=" + parallel, flush=True)
    spot_setup = spot_setup(_used_algorithm='dream')
    # !!! Specify the information !!!!!
    # Select number of maximum repetitions
    rep = 2500 #14 * 240  # ?

    # Select number of chains and set the Gelman-Rubin convergence limit
    nChains = 7 # 7  # <= 2*(numb. of chains per proposal) + 1 = 9, 4 cores= numb. of chains per proposal;  2*dim, dim = number of model parameter
    convergence_limit = 1.2
    runs_after_convergence = 100 #10000

    # Set the random state. Additionally random state is set in spotpy.parameter.Uniform !
    random_state = 5
    # random_state = None
    if random_state is None:
        random_state = np.random.randint(low=0, high=2 ** 30)
    np.random.seed(random_state)
    random.seed(random_state)

    sampler = spotpy.algorithms.dream(spot_setup, dbname='DREAM_modflow', dbformat='csv', db_precision=np.float32,
                                      save_sim=True, random_state=random_state, parallel=parallel)

    r_hat = sampler.sample(rep, nChains=nChains, convergence_limit=convergence_limit,
                           runs_after_convergence=runs_after_convergence)

    # save r_hat with pickle
    with open('r_hat.pkl', 'wb') as f:
        pkl.dump(r_hat, f, pkl.HIGHEST_PROTOCOL)

    results = sampler.getdata()  # Load the likelihood, and the parameter values of all simulations
