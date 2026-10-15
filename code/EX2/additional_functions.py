# first test of flopy: load an existing model and visualize inputs and outputs; read and edit parameters
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import flopy
import datetime
import pylab as plb
import csv

from mpi4py import MPI


def process_model_path(model_path):
    # return model_path + "_rank" + str(MPI.COMM_WORLD.Get_rank()) # error
    suffix = "_rank" + str(MPI.COMM_WORLD.Get_rank())
    if model_path.endswith(suffix):
        return model_path
    return model_path + suffix

    
def load_modflow_model(model_name, model_path): # load an existing model
    model_path = process_model_path(model_path)
    print(model_path)
    ml = flopy.modflow.Modflow.load(model_name, model_ws=model_path, verbose=False,
                               check=False, exe_name=r"C:\Users\chris\Documents\TUM\Master\WS25_26\Math_Meth\MOOC\Exercise_02\Ex 2 parallel DREAM-20260111\2_mf2005.exe")
    return ml

def load_heads(model_path): # Load the output (formatted) heads with flopy
    fpth = os.path.join(model_path, "heads.dat")
    hfile = flopy.utils.formattedfile.FormattedHeadFile(fpth)
    heads = hfile.get_alldata()
    return heads, hfile 

def get_head_ts(row,col,hfile): # Extract head as a time series in one cell
    idx = (0, row, col)
    ts = hfile.get_ts(idx)
    df = pd.DataFrame(ts)
    df = df.drop(columns = [0])
#    start = datetime.datetime(2013,5,2) # Enter date of model run
#    df = pd.DataFrame(ts, columns = ['time','value'])
#    df['time'] = start + pd.to_timedelta(df['time'], unit='s')
#    df = df.set_index('time')
#    df = df.reset_index(drop = True)
    return df

def plot_head_ts(df,row,col): # Plot the head versus time in a specific cell
    idx = (0, row, col)
    plt.subplot(1, 1, 1)
    ttl = 'Head at cell ({0},{1},{2})'.format(idx[0] + 1, idx[1] + 1, idx[2] + 1)
    plt.title(ttl)
    plt.xlabel('time')
    plt.ylabel('head')
    plt.plot(df,'bo-')
    
def plot_head_layer(hfile): # Plot the head of each layer after all stress periods
    times = hfile.get_times()
    hfile.plot(totim=times[-1], masked_values=[-9999.0], colorbar=True)
    
def import_observation(filename):
    dates = []
    scores = []
    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile, delimiter = ';')
        for row in csvReader:
            dates.append(row[0])
            scores.append(row[1])   
    dates[0] = dates[0][3:]
    for i in range(len(dates)):
        dates[i] = datetime.datetime.strptime(dates[i], '%d-%b-%Y %H:%M:%S')
        scores[i] = float(scores[i])
    obs = pd.DataFrame(scores, dates)
    return obs

def change_cond(ml, cond_1, cond_2, cond_3): # Change the riverbed conductance in the RIV file
    cond_mask_box = [cond_1, cond_1, cond_1, cond_1, cond_1, cond_1, cond_1, cond_1, cond_1,
       cond_1, cond_1, cond_1, cond_1, cond_1, cond_1, cond_1, cond_1, cond_1,
       cond_1, cond_1, cond_1, cond_2, cond_2, cond_2, cond_2, cond_1, cond_1,
       cond_1, cond_2, cond_2, cond_2, cond_2, cond_1, cond_1, cond_1, cond_2,
       cond_2, cond_2, cond_2, cond_2, cond_2, cond_2, cond_2, cond_2, cond_2,
       cond_2, cond_2, cond_2, cond_2, cond_2, cond_2, cond_2, cond_2, cond_2,
       cond_2, cond_2, cond_2, cond_3, cond_3, cond_3, cond_2, cond_2, cond_2,
       cond_2, cond_2, cond_2, cond_3, cond_3, cond_3, cond_2, cond_2, cond_2,
       cond_2, cond_2, cond_2, cond_3, cond_3, cond_3, cond_2, cond_2, cond_2,
       cond_2, cond_2, cond_2, cond_3, cond_3, cond_3, cond_3, cond_3, cond_3,
       cond_3, cond_3, cond_3, cond_3, cond_3, cond_3, cond_3, cond_3, cond_3,
       cond_3, cond_3, cond_3, cond_3, cond_3, cond_3]
    cond_mask_flood = np.load('mask')
    for i in range(0,len(ml.riv.stress_period_data.data),1):
        ml.riv.stress_period_data.data[i]['cond'] = cond_mask_box
    ml.riv.write_file()
        
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def get_maxlikeindex(results,verbose=True):
    """
    Get the maximum objectivefunction of your result array
    :results: Expects an numpy array which should of an index "like" for objectivefunctions
    :type: array
    :return: Index of the position in the results array with the maximum objectivefunction
        value and value of the maximum objectivefunction of your result array
    :rtype: int and float
    """
    try:
        likes=results['like']
    except ValueError:
        likes=results['like1']
    maximum=np.nanmax(likes)
    value=str(round(maximum,4))
    text=str('Run number ' )
    index=np.where(likes==maximum)
    text2=str(' has the highest objectivefunction with: ')
    textv=text+str(index[0][0])+text2+value
    if verbose:
        print(textv, flush=True)
    return index, maximum

def get_simulation_fields(results):
    header = get_header(results)
    fields=[word for word in header if word.startswith('sim')]
    return fields

def get_header(results):
    return results.dtype.names

def plot_parameter_distribution(results):
    header = get_header(results)
    fields=[word for word in header if word.startswith('par')]
    for field in fields:
        dist = pd.DataFrame(results[field])
        fig, ax = plt.subplots(figsize=(10,7))
        hist, bins = np.histogram(results[field], density=True, bins=20)
        # Width of each bin
        bins_w = np.diff(bins)
        # Compute proportion of sample in each bin
        hist_p = hist * bins_w
        # Plot histogram
        plt.bar(bins[:-1], hist_p, width=bins_w, align='edge')
        ax.set_ylabel('Marginal probability')
        ax.set_xlabel(field)
        ax.grid(axis='y')
        ax.set_facecolor('#d8dcd6')

def plot_bestmodelrun(results,evaluation,fig_name ='Best_model_run.png'):
    """
    Get a plot with the maximum objectivefunction of your simulations in your result
    array.
    The plot will be saved as a .png file.
    :results: Expects an numpy array which should of an index "like" for
              objectivefunctions and "sim" for simulations.
     type: Array
     :evaluation: Should contain the values of your observations. Expects that this list has the same lenght as the number of simulations in your result array.
     :type: list
    Returns:
        figure. Plot of the simulation with the maximum objectivefunction value in the result array as a blue line and dots for the evaluation data.
    """
    fig= plt.figure(figsize=(12,7))
#    for i in range(len(evaluation)):
#        if evaluation[i] == -9999:
#            evaluation[i] = np.nan
    plb.plot(evaluation,'ro',markersize=3, label='Observation data')
    simulation_fields = get_simulation_fields(results)
    bestindex,bestobjf = get_maxlikeindex(results,verbose=False)
    plb.plot(list(results[simulation_fields][bestindex][0]),'b-',label='Obj='+str(round(bestobjf,2)))
    plb.xlabel('Number of Observation Points')
    plb.ylabel ('Simulated value')
    plb.legend(loc='upper right')
    fig.savefig(fig_name,dpi=300)
    text='A plot of the best model run has been saved as '+fig_name
    print(text, flush=True)
    
def plot_parameter_uncertainty_box(posterior_results, evaluation, obs, fig_name='Posterior_parameter_uncertainty.png'):
    simulation_fields = get_simulation_fields(posterior_results)
    if obs == 1:
        simulation_fields = simulation_fields[:6]
        evaluation = evaluation[:6].reset_index(drop = True)
    elif obs == 2:
        simulation_fields = simulation_fields[6:12]
        evaluation = evaluation[6:12].reset_index(drop = True)
    elif obs == 3:
        simulation_fields = simulation_fields[-6:]
        evaluation = evaluation[-6:].reset_index(drop = True)
    else:
        simulation_fields = simulation_fields
    
    fig= plb.figure(figsize=(14,9))
#    for i in range(len(evaluation)):
#        if evaluation[i] == -9999:
#            evaluation[i] = np.nan
    ax = plb.subplot(1,1,1)
    q5,q95=[],[]
    for field in simulation_fields:
        q5.append(np.percentile(list(posterior_results[field]),2.5))
        q95.append(np.percentile(list(posterior_results[field]),97.5))
    ax.plot(q5,color='lightgrey',linestyle='solid')
    ax.plot(q95,color='lightgrey',linestyle='solid')
    ax.fill_between(np.arange(0,len(q5),1),list(q5),list(q95),facecolor='lightgrey',zorder=0,
                    linewidth=0,label='parameter uncertainty')
    ax.plot(evaluation,'r.',markersize=8, label='Observation data')

    bestindex,bestobjf = get_maxlikeindex(posterior_results,verbose=False)
    plb.plot(list(posterior_results[simulation_fields][bestindex][0]),'b-',label='Obj='+str(round(bestobjf,4)))
    plb.xlabel('Time')
    plb.ylabel ('Simulated value')
    plb.legend(loc='upper right')
    fig.savefig(fig_name,dpi=300)
    text='A plot of the parameter uncertainty has been saved as '+fig_name
    print(text, flush=True)
    
def plot_parameter_uncertainty_flood(posterior_results, evaluation, fig_name='Posterior_parameter_uncertainty.png'):
    simulation_fields = get_simulation_fields(posterior_results)
    nPoints = 4
    
    ev_1 = evaluation[:300]
    ev_2 = evaluation[300:600]
    ev_3 = evaluation[600:900]
    ev_4 = evaluation[-300:]
    
    sim_1 = simulation_fields[:300]
    sim_2 = simulation_fields[300:600]
    sim_3 = simulation_fields[600:900] 
    sim_4 = simulation_fields[-300:]
    
    simdata=np.column_stack((sim_1,sim_2))
    simdata=np.column_stack((simdata,sim_3))
    simdata=np.column_stack((simdata,sim_4))
    evdata=np.column_stack((ev_1,ev_2))
    evdata=np.column_stack((evdata,ev_3))
    evdata=np.column_stack((evdata,ev_4))
    
    #fig= plb.figure(figsize=(14,9))
    #ax = plb.subplot(1,1,1)
    
    for i in range(nPoints):
        fig, ax = plb.subplots(figsize=(12,7))        
        evaluation = evdata[:,i]
        simulation = simdata[:,i]       
        q5,q95=[],[]
        for field in simulation:
            q5.append(np.percentile(list(posterior_results[field]),2.5))
            q95.append(np.percentile(list(posterior_results[field]),97.5))
        ax.plot(q5,color='lightgrey',linestyle='solid')
        ax.plot(q95,color='lightgrey',linestyle='solid')
        ax.fill_between(np.arange(0,len(q5),1),list(q5),list(q95),facecolor='lightgrey',zorder=0,
                    linewidth=0,label='parameter uncertainty')
        ax.plot(evaluation,'r.',markersize=8, label='Observation data')
    
        bestindex,bestobjf = get_maxlikeindex(posterior_results,verbose=False)
        plb.plot(list(posterior_results[simulation][bestindex][0]),'b-',label='Obj='+str(round(bestobjf,4)))
        plb.xlabel('Time')
        plb.ylabel ('Simulated value')
        plb.legend(loc='upper right')
        name = str(i)+fig_name
        fig.savefig(name,dpi=300)
        text='A plot of the parameter uncertainty has been saved as '+name
        print(text,flush=True)