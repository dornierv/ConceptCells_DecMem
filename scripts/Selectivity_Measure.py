'''
Created by VD (22/07/2025)


Script to compute the selectivity measure of each neurons in the different regions
It is based on the method described in Quiroga et al. (2007)

This script is used to do Fig. 5e
'''

# Import libraries needed

import json 
import re 
import numpy as np
import os
import pynapple as nap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from scipy import io as io
from scipy import stats as stats



# Inputs
path2data = 'C:/Users/dornier/GitHub/ConceptCells_DecMem/'
path2fig = 'C:/Users/dornier/GitHub/ConceptCells_DecMem/Selectivity/'


#######################
# Functions 
#######################


def read_lines(filename, removeEndLines=False, str2Split='\n', dtype=str):
    """
    return lines
    """
    lines=[]
    with open(filename, 'rb') as f:
        for line in f:
            if removeEndLines:
                line=line.decode(errors='ignore')
                line=line.split('\n')[0]
            else:
                line=line.decode(errors='ignore')
            lines.append(dtype(line))
    return lines

def load_spikes(data_path):
    # Load cluster file from Klusters (you should have first transformed your 
    #Klusters files into nwb files)
    data = nap.load_folder(data_path)
    #data.view

    nwb = data["spikes"]["saved_file"]


    spikes = nwb["units"]



    try:
        spikes = spikes[spikes['quality']=='good']
    except:
        ok = 1

    return spikes


def get_firingrate(perievent):
    bin_size = 0.2  # 200ms bin size
    step_size = 0.01  # 10ms step size, to make overlapping bins
    winsize = int(bin_size / step_size)  # Window size


    counts_im = perievent.count(step_size)

    counts_im = (
        counts_im.as_dataframe()
        .rolling(winsize, win_type="gaussian", min_periods=1, center=True, axis=0)
        .mean(std=0.2 * winsize)
        )
    
    fr_im = counts_im * winsize

    fr = fr_im._values

    fr = fr.T

    return fr


def selectivity_neurons(path_json,M,response_type='all'):
    # Open the dictionary containing all single-units registered in the temporal pole
    with open(path_json, "r") as f:
        neurons_tp = json.load(f)


    # Extract list of patients from dictionary
    list_patient = list(neurons_tp.keys())


    # General variable
    sr = 32768 # Sampling rate of Neuralynx system

    # Initialize list to store the result
    selectivity = list()

    # Loop over patients having single-units in the temporal pole
    for iPatient,i in zip(list_patient,range(len(list_patient))):

        # Extract sessions from the dictionnary
        list_session = list(neurons_tp[iPatient][0].keys())

        bsnm=iPatient # ID of the patient
        
        # Loop over sessions containing units in the TP
        for iSession  in list_session:

            # Only keep the number of the session
            session = re.sub('sess-','',iSession) # Remove sess- from sess-1 to only keep the number of the session


            # Path where data is stored
            data_path = path2data+'data/Examples_Session/'+bsnm+'/sess-'+str(session)
            path_images = data_path+'/stimuli/' 

            # Load logfile
            logfname=data_path+'/lfps/'+bsnm+'_ses-01_task-Screening_run-01_ieeg_log.txt'

            logLines=np.array(read_lines(logfname, removeEndLines=True))

            stream=np.arange(len(logLines)/25, dtype=int)*25+1
            chRegs=np.array([line.split('.')[0] for line in logLines[stream]])
            print(chRegs,len(chRegs))

            # Load units from nwb
            spikes = load_spikes(data_path)


            # Load TTLs & dat files
            folder_TTL = data_path+'/ttl/'
            TTLvals = io.loadmat(folder_TTL+bsnm+'_TTLvals_tot.mat')['TTLvals_tot'][0]
            TS = io.loadmat(folder_TTL+bsnm+'_TS_tot.mat')['TS_tot'][0]

            # Load timestamps
            tsname=data_path+'/lfps/'+bsnm+'_ses-01_task-Screening_run-01_timestamps.dat'
            TS_stream=np.memmap(tsname, mode='r', dtype=float, order='F')


            # Get TTLs sync with EEG
            TS_32768 = np.searchsorted(TS_stream, TS, side='left') # index temps absolu
            TS_32768[TS_32768 ==len(TS_stream)] = len(TS_stream)-1


            # Regroup TTLs by class of stimulus presented
            # Check that all different TTLs correspond to images
            # Otherwise, remove bad TTLs
            imgs=np.unique(TTLvals)
            imgIndices=[np.where(TTLvals==img_)[0] for img_ in imgs]


            imgTS=[TS_32768[imgIndice] for imgIndice in imgIndices]

            # Suppress the first value that correspond to TTL = 0 (i.e., fixation cross)
            imgTS = imgTS[1:]

            list_neurons = neurons_tp[iPatient][0][iSession]

            # Loop over neurons of interest
            if list_neurons:
                for iNeuron in list_neurons:
                    max_fr_neuron = list()
                    count_spike_bl = list()
                    count_spike_trial = list()
                    for TimingImage in imgTS:
                        Image_Second = TimingImage/sr

                        # Calcul firing rate
                        TS_Index = nap.Ts(Image_Second)


                        # Get spikes in the trial
                        Trial_peth = nap.compute_perievent(spikes[iNeuron], TS_Index, minmax=(0,1))
                        All_Trial_peth = nap.compute_perievent(spikes[iNeuron], TS_Index, minmax=(-1,1))
                        
                        # Test for statistical difference between baseline and trial period
                        ep_trial = nap.IntervalSet(start=0, end=1, time_units='s')
                        ep_baseline = nap.IntervalSet(start=-1,end=0,time_units='s')
                        spike_trial = All_Trial_peth.restrict(ep_trial)
                        spike_baseline = All_Trial_peth.restrict(ep_baseline)
                        

                        count_spike_bl.append(spike_baseline.count().d[0])
                        count_spike_trial.append(spike_trial.count().d[0])

                        


                        firing_rate = get_firingrate(Trial_peth)

                        all_images_fr = np.mean(firing_rate,axis=0)

                        max_fr_neuron.append(np.max(all_images_fr))

                    if response_type == 'increasing':
                        if np.sum(np.concatenate(count_spike_trial)) > np.sum(np.concatenate(count_spike_bl)):
                            selectivity_neuron = calcul_selectivity(max_fr_neuron,M)

                            selectivity.append(selectivity_neuron)
                        else:
                            ok=1
                    elif response_type == 'all':
                        selectivity_neuron = calcul_selectivity(max_fr_neuron,M)

                        selectivity.append(selectivity_neuron)
                    else:
                        ok = 1
                    
    return selectivity



def calcul_R(firing_rate,threshold):
    '''
    Compute selectivity index R 

    Voir Equation (2) dans Quiroga et al. (2007)

    Parameters
    -------------------
    
    firing_rate : list in array shape
        list containing maximum firing rate of a neuron in response to each stimuli employed during screening sessions

    threshold : float
        threshold to test how many responses are above

    Output
    ---------------------
    R : float
        values containing the normalized number of responses, i.e. the relative number of stimuli with firing larger than threshold
    '''
    theta = 0

    nb_stim = np.shape(firing_rate)[0]
    for iStim in range(nb_stim):
        if firing_rate[iStim] > threshold:
            theta +=1
    
    R = (1/nb_stim) * theta

    return R



def calcul_selectivity(firing_rate,M):
    '''
    Compute the area under the curve of normalized response 

    For details see Equation (3) in Quiroga et al. (2007)

    Parameters 
    ------------------
    firing_rate : list like array
        list containing float number corresponding to the maximum firing rate of a neuron to each stimulus presented in the screening session

    M : int
        Number corresponding to the number of steps to create a distribution
    
    Output
    ------------------
    A : float
        Area under the curve
    '''
    fmin = np.min(firing_rate)

    fmax = np.max(firing_rate)

    array_threshold = np.linspace(fmin,fmax,M)

    # Initialize R to zero
    R = 0

    for iThresh in range(M):

        # Get the sum of R values for each threshold
        R = R +  calcul_R(firing_rate,array_threshold[iThresh])
    
    # Compute area under the curve
    A = (1/M) * R


    # Compute selectivity (see Eq.4 in Quiroga et al., 2007)
    S = 1 - 2*A

    return S




###########################
# Main script
###########################

# Compute selectivity of neurons in the temporal pole
try:
    selectivity_tp = np.load(path2data+'data/Selectivity/selectivity_tp.npy')
except:
    selectivity_tp = selectivity_neurons(r'C:\Users\dornier\GitHub\ConceptCells_TP/dictionary_singleunits_tp.json',1000,response_type='increasing')
    np.save(path2data+'data/Selectivity/selectivity_tp.npy',selectivity_tp)


# Compute selectivity of neurons in the hippocampus
try:
    selectivity_hippocampus = np.load(path2data+'data/Selectivity/selectivity_hippocampus.npy')
except:
    selectivity_hippocampus = selectivity_neurons(r'C:\Users\dornier\GitHub\ConceptCells_TP/dictionary_singleunits_hippocampus.json',1000,response_type='increasing')
    np.save(path2data+'data/Selectivity/selectivity_hippocampus.npy',selectivity_hippocampus)



# Compute selectivity of neurons in the parahippocampal regions
try:
    selectivity_parahippocampal = np.load(path2data+'data/Selectivity/selectivity_parahippocampal.npy')
except: 
    selectivity_parahippocampal = selectivity_neurons(r'C:\Users\dornier\GitHub\ConceptCells_TP/dictionary_singleunits_parahippocampal.json',1000,response_type='increasing')
    np.save(path2data+'data/Selectivity/selectivity_parahippocampal.npy',selectivity_parahippocampal)

# Compute selectivity of neurons in the PCC regions
try:
    selectivity_pcc = np.load(path2data+'data/Selectivity/selectivity_pcc.npy')
except: 
    selectivity_pcc = selectivity_neurons(r'C:\Users\dornier\GitHub\ConceptCells_TP/dictionary_singleunits_pcc.json',1000,response_type='increasing')
    np.save(path2data+'data/Selectivity/selectivity_pcc.npy',selectivity_pcc)




####################
# Plot part
####################



fig, ax = plt.subplots(layout='constrained',figsize=(10,5))
ax.set_ylabel('Selectivity index',fontsize=15)



colors = ['palevioletred','darkslateblue','sandybrown','cornflowerblue']
labels = ['Temporal pole','Hippocampus','Parahippocampal','PCC']



bplot = ax.boxplot((selectivity_tp,selectivity_hippocampus,selectivity_parahippocampal,selectivity_pcc),
                patch_artist=True)  
    # will be used to label x-ticks
ax.tick_params(axis='x',labelrotation=45)
ax.set_xticklabels(labels,fontsize=11)
ax.set_title('Selectivity across ROIs',pad=15,fontsize=16)
# fill with colors
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)

for median in bplot['medians']:
    median.set_color('black')


ax.scatter(np.random.normal(0 + 1, 0.04, len(selectivity_tp)),selectivity_tp,color='grey',alpha=0.4)
ax.scatter(np.random.normal(1 + 1, 0.04, len(selectivity_hippocampus)),selectivity_hippocampus,color='grey',alpha=0.4)
ax.scatter(np.random.normal(2 + 1, 0.04, len(selectivity_parahippocampal)),selectivity_parahippocampal,color='grey',alpha=0.4)
ax.scatter(np.random.normal(3 + 1, 0.04, len(selectivity_pcc)),selectivity_pcc,color='grey',alpha=0.4)


plt.savefig(path2fig+'Boxplot_Selectivity_All_ROI.svg')
plt.show()

plt.close()

