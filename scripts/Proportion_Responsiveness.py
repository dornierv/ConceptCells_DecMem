'''
Created by VD (13/06/2025)

In this script we will compute for each responsive cell the proportion of stimuli that elicit a response.

This script has been used to do Fig. 2c and Fig. 4e
'''
# Import libraries needed
import json 
import re 
import numpy as np
import os
import pynapple as nap
import matplotlib.pyplot as plt
import mne
import warnings
warnings.filterwarnings("ignore")
from scipy import io as io
from scipy import stats as stats



# Path where to save fig
path2fig = 'C:/Users/dornier/PhD/Article/Concept_Cells_temporal_pole/Figures/Figure1/Panel_Selectivity'
ROI = 'tp'

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

def proportion_response(path_json):
    '''
    Get for each unit in your region of interest the proportion of stimuli that elicit a response
    '''
    # Open the dictionary containing all single-units registered in the temporal pole
    with open(path_json, "r") as f:
        neurons_tp = json.load(f)


    # Extract list of patients from dictionary
    list_patient = list(neurons_tp.keys())


    # General variable
    sr = 32768 # Sampling rate of Neuralynx system


    # Initialize list to store data
    nb_response_per_neuron = list()

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
            data_path = r'D:\Screening\database/'+bsnm+'/sess-'+str(session)
            path_images = data_path+'/stimuli/' #"E:/screening hors eeg/Screening images/Pool images/"
            

            # Load logfile
            logfname=data_path+'/lfps/'+bsnm+'_ses-01_task-Screening_run-01_ieeg_log.txt'
            logLines=np.array(read_lines(logfname, removeEndLines=True))
            stream=np.arange(len(logLines)/25, dtype=int)*25+1
            chRegs=np.array([line.split('.')[0] for line in logLines[stream]])
            print(chRegs,len(chRegs))

            # Load spiking activity from .nwb file
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
                    is_neuron_responsive = 0
                    count_image_responsive = 0

                    

                    # Loop over all stimuli
                    for TimingImage in imgTS:
                        Image_Second = TimingImage/sr

                        # Calcul firing rate
                        TS_Index = nap.Ts(Image_Second)


                        # Get spikes in the trial
                        Trial_peth = nap.compute_perievent(spikes[iNeuron], TS_Index, minmax=(-1,1))

                        # Extract firing rate
                        fr_im = get_firingrate(Trial_peth)
                        firing_rate_trial = fr_im[:,100:200] 
                        baseline_rate_trial = fr_im[:,0:100]

                        # Cluster-based permutation test
                        F_obs, clusters, clusters_pv,H0 = mne.stats.permutation_cluster_test([baseline_rate_trial,firing_rate_trial])

                        try:
                            p_value_final = np.min(clusters_pv)
                        except:
                            p_value_final = 1
                        
                        # If response significant neuron gets responsive and add 1 to count of picture responsive
                        if p_value_final < 0.05 :
                            is_neuron_responsive = 1

                            count_image_responsive += 1
                
                    
                    # Determine the percentage of responsive pictures
                    percent_response = count_image_responsive/len(imgTS)

                    
                    # If neuron responsive store percentage of response
                    if is_neuron_responsive == 1:
                        nb_response_per_neuron.append(percent_response)

                     
    return nb_response_per_neuron



###########################################################
##################### MAIN SCRIPT #########################
###########################################################

# Get the mean firing rate for all neurons in TP responsive
try:
    nb_response_neuron = np.load('C:/Users/dornier/PhD/Article/Concept_Cells_temporal_pole/Figures/Figure1/Panel_Selectivity/Nb_Response_TP_Pourcentage.npy')
except:
    nb_response_neuron = proportion_response("C:/Users/dornier/GitHub/ConceptCells_TP/dictionary_singleunits_tp.json")
    np.save('C:/Users/dornier/PhD/Article/Concept_Cells_temporal_pole/Figures/Figure1/Panel_Selectivity/Nb_Response_TP_Pourcentage.npy',nb_response_neuron)

# Plot
fig,ax = plt.subplots(figsize=(12,6),layout='constrained')
ax.hist(nb_response_neuron*100,bins=45,weights=np.ones(len(nb_response_neuron)) / len(nb_response_neuron)*100,color='palevioletred',alpha=0.8,ec='k')
ax.set_title('Selectivity of neurons in the temporal pole',pad=15,fontsize=25)
ax.set_xlabel('Percentage of pictures eliciting a response',labelpad=15,fontsize=25)
ax.set_ylabel('Proportion of neurons (%)',labelpad=15,fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=18)



plt.savefig(path2fig+'/Proportion_Responsiveness_Neuron'+ROI+'.svg')

plt.show()
plt.close()



