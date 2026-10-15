# -*- coding: utf-8 -*-
"""
Visualization file
"""

import pandas as pd
import numpy as np
import pylab as plb
import spotpy
import seaborn as sns
import matplotlib.pyplot as plt
import math
from statistics import stdev 
import shutil
from spotpy.analyser import plot_parameter_trace
from spotpy.analyser import plot_posterior_parameter_histogram
from spot_setup_modflow import spot_setup
import os
#os.chdir('C:/Users/Gheorghe Pascu/OneDrive - tum.de/WiSe 21-22/Uncertainty Quantification in Hydrology - MOOC(2)/block_2/flood/code_dream')
os.chdir(r'C:\Users\chris\Documents\TUM\Master\WS25_26\Math_Meth\MOOC\Exercise_02\Ex 2 parallel DREAM-20260111\dream_bl2')
cwd = os.getcwd()


date = "DREAM_modflow.csv"

parnames = ['CRIV-1', 'CRIV-2',
            'CRIV-3', 'CRIV-4',
            'CRIV-5']
ranges = [(np.log10(1e-5),np.log10(1e-1)),(np.log10(1e-5),np.log10(1e-1)),
          (np.log10(1e-7),np.log10(1e-2)),(np.log10(1e-7),np.log10(1e-2)),
          (np.log10(1e-7),np.log10(1e-2))]

results = pd.read_csv(date)
evaluation = pd.read_pickle("obs_flood_multi.pkl")
evaluation = evaluation.reset_index(drop = True)
fields=[word for word in results.columns if word.startswith('sim')]
Fields = [fields[0:300], fields[300:600], fields[600:900], fields[900:1200]]
i = 0
for fields in Fields:
    fig= plt.figure(figsize=(8, 4.5))
    ax = plt.subplot(1,1,1) 
    q5,q25,q75,q95=[],[],[],[]
    std = []
    mean = []
    for field in fields:
        #q5.append(np.percentile(results[field][-500:-1],2.5))
        #q95.append(np.percentile(results[field][-500:-1],97.5))
        std.append(np.std(results[field][-500:-1]))
        mean.append(np.mean(results[field][-500:-1]))
    #ax.plot(q5,color='dimgrey',linestyle='solid')
    #ax.plot(q95,color='dimgrey',linestyle='solid')
    std = np.array(std)
    mean = np.array(mean)
    #ax.plot(mean,color='black',linestyle='solid')
    ax.plot(mean + std,color='lightblue',linestyle='solid')
    ax.plot(mean - std,color='lightblue',linestyle='solid')
    ax.fill_between(np.arange(0,len(std),1),list(mean + std),list(mean - std),facecolor='lightblue',zorder=0,
                    linewidth=0,label = 'parameter uncertainty')  

    if i==0:
        ax.plot(evaluation.values[0:300],'r.',label='data')
        ax.set_xlabel('Stress period')
        ax.set_ylabel('Groundwater heads [m.a.s.l.]')
        ax.legend()
        ax.set_title('Alzpitz')
        plt.show()
        fig.savefig('parameter_uncertainty_Alzpitz.png',dpi=300)
    if i==1:
        ax.plot(evaluation.values[300:600],'r.',label='data')
        ax.set_xlabel('Stress period')
        ax.set_ylabel('Groundwater heads [m.a.s.l.]')
        ax.legend()
        ax.set_title('B1')
        plt.show()
        fig.savefig('parameter_uncertainty_B1.png',dpi=300)
    if i==2:
        ax.plot(evaluation.values[600:900],'r.',label='data')
        ax.set_xlabel('Stress period')
        ax.set_ylabel('Groundwater heads [m.a.s.l.]')
        ax.legend()
        ax.set_title('B3')
        plt.show()
        fig.savefig('parameter_uncertainty_B3.png',dpi=300)
    if i==3:
        ax.plot(evaluation.values[900:1200],'r.',label='data')
        ax.set_xlabel('Stress period')
        ax.set_ylabel('Groundwater heads [m.a.s.l.]')
        ax.legend()
        ax.set_title('B4')
        plt.show()
        fig.savefig('parameter_uncertainty_B4.png',dpi=300)
    i += 1

spot_setup = spot_setup(_used_algorithm='dream')
#spotpy.analyser.plot_gelman_rubin(results, r_hat, fig_name='python_hydrology_convergence.png') #we don't have r_hat
#parameters = spotpy.parameter.get_parameters_array(spot_setup)
parameters = spotpy.parameter.get_parameters_array(spot_setup)

fig, ax = plt.subplots(nrows=5, ncols=2, figsize = (8, 4.5))

for par_id in range(len(parameters)):
    plot_parameter_trace(ax[par_id][0], results, parameters[par_id])
    plot_posterior_parameter_histogram(ax[par_id][1], results, parameters[par_id])

ax[-1][0].set_xlabel('Iterations')
ax[-1][1].set_xlabel('Parameter range')

plt.show()
fig.savefig('MODFLOW_parameters.png',dpi=300)


print("Means:")
print(" ")
print(np.mean(results['parcond_1']), np.mean(results['parcond_2']), np.mean(results['parcond_3']), np.mean(results['parcond_4']), np.mean(results['parcond_5']))
print(" ")
print("Stds:")
print(" ")
print(np.std(results['parcond_1']), np.std(results['parcond_2']), np.std(results['parcond_3']), np.std(results['parcond_4']), np.std(results['parcond_5']))

# from spot_setup_modflow import spot_setup
# spot_setup = spot_setup(_used_algorithm = 'fast')
# M, d, k = 4, 2, 4 #https://spotpy.readthedocs.io/en/latest/Sensitivity_analysis_with_FAST/
# rep = (1 + 4 * M ** 2 * (1 + (k - 2) * d)) * k
# sampler = spotpy.algorithms.fast(spot_setup,  dbname='FAST_hydrology',  dbformat='csv')
# sampler.sample(repetitions = rep)
# results_FAST = spotpy.analyser.load_csv_results('FAST_hydrology')
# spotpy.analyser.plot_fast_sensitivity(results_FAST, number_of_sensitiv_pars=5, fig_name = 'FAST_sensitivity.png')







