'''
Created by VD (13/06/2025)

In this script we will determine the proportion of cells that respond to at least one stimulus.

This script has been used to Fig. 2b, Fig. 4d
'''

# Import libraries needed
import warnings
warnings.filterwarnings("ignore")
import json 
import re 
import numpy as np
import os
import pynapple as nap
import matplotlib.pyplot as plt
import mne
from scipy import io as io
from scipy import stats as stats


# Inputs 
path2data = 'C:/Users/dornier/GitHub/ConceptCells_DecMem/'

###########################################################################
############ FUNCTIONS USED LATER IN THE MAIN SCRIPT ######################
###########################################################################


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
    data.view

    nwb = data["spikes"]["saved_file"]


    spikes = nwb["units"]



    try:
        spikes = spikes[spikes['quality']=='good']
    except:
        print('Made with SC1')

    return spikes

def get_firingrate(perievent):
    # Parameters for firing rate estimation 
    # Parameters are from pynapple documentation
    bin_size = 0.2  
    step_size = 0.01  
    winsize = int(bin_size / step_size)  


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

def average_neurons(path_json):
    '''
    Get the number of cells responding to at least one picture.
    '''
    # Open the dictionary containing all single-units recorded
    with open(path_json, "r") as f:
        neurons_tp = json.load(f)


    # Extract list of patients from dictionary
    list_patient = list(neurons_tp.keys())


    # General variable
    sr = 32768 # Sampling rate of Neuralynx system

    # Initialize count of neurons
    nb_neuron = 0
    nb_neuron_responsive = 0

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
            data_path = path2data+'data/Examples_Session/'+bsnm+'/sess-'+str(session)+'/'
            path_images = data_path+'/stimuli/' 
            

            # Load logfile
            logfname=data_path+'/lfps/'+bsnm+'_ses-01_task-Screening_run-01_ieeg_log.txt'
            logLines=np.array(read_lines(logfname, removeEndLines=True))
            stream=np.arange(len(logLines)/25, dtype=int)*25+1
            chRegs=np.array([line.split('.')[0] for line in logLines[stream]])
            print(chRegs,len(chRegs))


            # Load spiking activity from .nwb
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

            # Get the sample in which stimuli were presented
            imgTS=[TS_32768[imgIndice] for imgIndice in imgIndices]

            # Suppress the first value that correspond to TTL = 0 (i.e., fixation cross)
            imgTS = imgTS[1:]

            # Extract list of neurons
            list_neurons = neurons_tp[iPatient][0][iSession]

            # Loop over neurons of interest
            if list_neurons:
                for iNeuron in list_neurons:

                    # Initialize the neuron as non responsive first
                    is_neuron_responsive = 0

                    # Loop over all stimuli presented
                    for TimingImage in imgTS:
                        Image_Second = TimingImage/sr

                        # Calcul firing rate
                        TS_Index = nap.Ts(Image_Second)


                        # Get spikes in the trial
                        Trial_peth = nap.compute_perievent(spikes[iNeuron], TS_Index, minmax=(-1,1))

                        # Get the firing rate
                        fr_im = get_firingrate(Trial_peth)

                        # Extract firing rate for trial and baseline periods
                        firing_rate_trial = fr_im[:,100:200]
                        baseline_rate_trial = fr_im[:,0:100]


                        # Cluster-based permutation test to assess responsiveness
                        F_obs, clusters, clusters_pv,H0 = mne.stats.permutation_cluster_test([baseline_rate_trial,firing_rate_trial])

                        # Get lowest p-value of clusters detected
                        try:
                            p_value_final = np.min(clusters_pv)
                        except:
                            p_value_final = 1
                        
                        # If inferior to .05 then neuron turn responsive
                        if p_value_final < 0.05 :

                            is_neuron_responsive = 1

                    # Always add one to neuron count 
                    nb_neuron += 1

                    # If the neuron is responsive add one to responsive count
                    if is_neuron_responsive == 1:
                        nb_neuron_responsive +=1

                     
    return nb_neuron, nb_neuron_responsive



###########################################################
##################### MAIN SCRIPT #########################
###########################################################

# Get the mean firing rate for all neurons in the region of interest - Note that here is only for one session so not entire database
nb_neuron, nb_neuron_responsive = average_neurons(path2data+"/data/Examples_Session/dictionary_singleunits_example.json")


print('There are: '+str(nb_neuron) +' neurons')
print('\nWith '+str(nb_neuron_responsive)+' that respond to at least one stimulus')
print('\nSo: '+str(nb_neuron_responsive/nb_neuron)+' percent of neurons')