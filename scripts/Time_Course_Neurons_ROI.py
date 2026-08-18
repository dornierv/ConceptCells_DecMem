'''
Created by VD (13/06/2025)

In this script we will plot the mean firing rate of each region (baseline corrected)
To obtain informations about the time course of neuronal activity in each region

This script is used to do Fig. 5a
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
path2database = 'C:/Users/dornier/GitHub/ConceptCells_DecMem/'
path2figure = 'C:/Users/dornier/GitHub/ConceptCells_DecMem/data/Firing_Rate/'

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
    '''
    Load spikes information from the .nwb file
    '''
    # Load cluster file from Klusters (you should have first transformed your 
    #Klusters files into nwb files)
    data = nap.load_folder(data_path)
    data.view

    nwb = data["spikes"]["saved_file"]


    spikes = nwb["units"]


    # Exception for the only patient for who we used SC2
    try:
        spikes = spikes[spikes['quality']=='good']
    except:
        print('Made with SC1')

    return spikes

def get_firingrate(perievent):
    '''
    Obtain firing rate from spike trains
    '''
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

def average_neurons(path_json):
    '''
    Get the time course of firing rate
    '''
    # Open the dictionary containing all single-units registered in the temporal pole
    with open(path_json, "r") as f:
        neurons_tp = json.load(f)


    # Extract list of patients from dictionary
    list_patient = list(neurons_tp.keys())


    # General variable
    sr = 32768 # Sampling rate of Neuralynx system

    # Initialize lists to save data
    all_fr_images = list()
    fr_neurons = list()
    fr_neurons_image = np.zeros((1,200))

    # Loop over patients included
    for iPatient,i in zip(list_patient,range(len(list_patient))):

        # Extract sessions from the dictionnary
        list_session = list(neurons_tp[iPatient][0].keys())

        bsnm=iPatient # ID of the patient
        
        # Loop over sessions containing units in the TP
        for iSession  in list_session:

            # Only keep the number of the session
            session = re.sub('sess-','',iSession) # Remove sess- from sess-1 to only keep the number of the session


            # Path where data is stored
            data_path = path2database+'data/Examples_Session/'+bsnm+'/sess-'+str(session)
            path_images = data_path+'/stimuli/' #"E:/screening hors eeg/Screening images/Pool images/"
            

            # Load logfile and infos about the recording
            logfname=data_path+'/lfps/'+bsnm+'_ses-01_task-Screening_run-01_ieeg_log.txt'

            logLines=np.array(read_lines(logfname, removeEndLines=True))

            stream=np.arange(len(logLines)/25, dtype=int)*25+1
            chRegs=np.array([line.split('.')[0] for line in logLines[stream]]) # Name of channels, e.g., dtb1
           
            
            print(chRegs,len(chRegs))

            # Load spiking activity
            spikes = load_spikes(data_path)

            # Load Run Lists and get TTL associated with each image
            test_mat_Run1 = io.loadmat(path_images+'run-01.mat') 
            my_array_Run1 = test_mat_Run1['trial']
            keys_TTL = [my_array_Run1[0,i][1][0][0] for i in range(my_array_Run1.shape[1])] 
            values_images = [my_array_Run1[0,i][2][0] for i in range(my_array_Run1.shape[1])] 
            

            


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

            # Get the list of single-units from the dictionnary
            list_neurons = neurons_tp[iPatient][0][iSession]

            # Loop over neurons of interest
            if list_neurons:
                for iNeuron in list_neurons:

                    all_fr_images = list()
                    for TimingImage in imgTS:

                        # Transform onset from sample to second
                        Image_Second = TimingImage/sr

                        # Calcul firing rate
                        TS_Index = nap.Ts(Image_Second)


                        # Get spikes in the trial - from -1second to + second
                        Trial_peth = nap.compute_perievent(spikes[iNeuron], TS_Index, minmax=(-1,1))


                        # Extract spiking activity from baseline and from trial
                        ep_trial = nap.IntervalSet(start=0, end=1, time_units='s')
                        ep_baseline = nap.IntervalSet(start=-1,end=0,time_units='s')
                        spike_trial = Trial_peth.restrict(ep_trial)
                        spike_baseline = Trial_peth.restrict(ep_baseline)
                        count_spike_bl = spike_baseline.count().d[0]
                        count_spike_trial = spike_trial.count().d[0]

                        

                        # Get the firing rate
                        fr_im = get_firingrate(Trial_peth)
                        firing_rate_trial = fr_im[:,100:200] 

                        baseline_rate_trial = fr_im[:,0:100]

                        # Cluster-based permutation test between baseline and trial
                        F_obs, clusters, clusters_pv,H0 = mne.stats.permutation_cluster_test([baseline_rate_trial,firing_rate_trial])

                        
                        try:
                            p_value_final = np.min(clusters_pv)
                        except:
                            p_value_final = 1 # If no clusters then p = 1
                        
                        
                        # If the neuron is responsive and with increased activity
                        if p_value_final < 0.05 and np.sum(count_spike_trial) > np.sum(count_spike_bl):

                            

                            # Get the mean firing rate across all trials for a picture
                            mean_fr_image = np.mean(fr_im,axis=0)


                            # With shape X x Y x Z with X = image, Y = time
                            all_fr_images.append(mean_fr_image)

                    # Get the mean activity of the neuron
                    mean_neuron = np.mean(all_fr_images,axis=0)
                    
                     
                    if np.isnan(mean_neuron).any() == False:
                        fr_neurons.append(mean_neuron)  
                    if all_fr_images:
                        fr_neurons_image = np.vstack((fr_neurons_image,all_fr_images)) 
                
                
                
                
                     

                         
    return fr_neurons,fr_neurons_image



###########################################################
##################### MAIN SCRIPT #########################
###########################################################

# Get the mean firing rate for all neurons in TP responsive
try:
    firingrate_tp = np.load(path2database+'data/Firing_Rate/FR_TP.npy')
    neuron_tp = np.load(path2database+'data/Firing_Rate/FR_Neuron_TP.npy')
except:
    neuron_tp,firingrate_tp = average_neurons(r"C:\Users\dornier\GitHub\ConceptCells_TP/dictionary_singleunits_tp.json")
    np.save(path2database+'data/Firing_Rate/FR_TP.npy',firingrate_tp) # Firing rate for each image
    np.save(path2database+'data/Firing_Rate/FR_Neuron_TP.npy',neuron_tp) # Mean firing rate for each neuron

# Get the mean firing rate for all neurons in hippocampus responsive
try:
    firingrate_hippocampus = np.load(path2database+'data/Firing_Rate/FR_Hippocampus.npy')
    neuron_hippocampus = np.load(path2database+'data/Firing_Rate/FR_Neuron_Hippocampus.npy')
except:
    neuron_hippocampus,firingrate_hippocampus = average_neurons(r"C:\Users\dornier\GitHub\ConceptCells_TP/dictionary_singleunits_hippocampus.json")
    np.save(path2database+'data/Firing_Rate/FR_Hippocampus.npy',firingrate_hippocampus)
    np.save(path2database+'data/Firing_Rate/FR_Neuron_Hippocampus.npy',neuron_hippocampus)
    



# Get the mean firing rate for all neurons in parahippocampal regions
try:
    firingrate_parahippocampal = np.load(path2database+'data/Firing_Rate/FR_Parahippocampal.npy')
    neuron_parahippocampal = np.load(path2database+'data/Firing_Rate/FR_Neuron_Parahippocampal.npy')
except:
    neuron_parahippocampal, firingrate_parahippocampal = average_neurons(r"C:\Users\dornier\GitHub\ConceptCells_TP/dictionary_singleunits_parahippocampal.json")
    np.save(path2database+'data/Firing_Rate/FR_Parahippocampal.npy',firingrate_parahippocampal)
    np.save(path2database+'data/Firing_Rate/FR_Neuron_Parahippocampal.npy',neuron_parahippocampal)


# Get the mean firing rate for all neurons in posterior cingulate cortex
try:
    firingrate_pcc = np.load(path2database+'data/Firing_Rate/FR_PCC.npy')
    neuron_pcc = np.load(path2database+'data/Firing_Rate/FR_Neuron_PCC.npy')
except:
    neuron_pcc,firingrate_pcc = average_neurons(r"C:\Users\dornier\GitHub\ConceptCells_TP/dictionary_singleunits_pcc.json")
    np.save(path2database+'data/Firing_Rate/FR_PCC.npy',firingrate_pcc)
    np.save(path2database+'data/Firing_Rate/FR_Neuron_PCC.npy',neuron_pcc)




# Normalize data with baseline
fr_tp_norm = (np.mean(firingrate_tp,axis=0)-np.mean(firingrate_tp[:,0:100]))/np.mean(firingrate_tp[:,0:100])
fr_parahippocampal_norm = (np.mean(firingrate_parahippocampal,axis=0)-np.mean(firingrate_parahippocampal[:,0:100]))/np.mean(firingrate_parahippocampal[:,0:100])
fr_hippocampus_norm = (np.mean(firingrate_hippocampus,axis=0) - np.mean(firingrate_hippocampus[:,0:100]))/np.mean(firingrate_hippocampus[:,0:100])
fr_pcc_norm = (np.mean(firingrate_pcc,axis=0) - np.mean(firingrate_pcc[:,0:100]))/np.mean(firingrate_pcc[:,0:100])



# Create axis
x = np.linspace(-1,1,200)
fig, ax = plt.subplots(figsize=(9,4),layout='constrained')
ax.plot(x,fr_parahippocampal_norm,color='sandybrown')
ax.plot(x,fr_tp_norm,color='palevioletred')  
ax.plot(x,fr_hippocampus_norm,color='darkslateblue') 
ax.plot(x,fr_pcc_norm,color='cornflowerblue') 
  



# Find maximum for each ROI
idx_max_tp = np.argmax(np.mean(firingrate_tp,axis=0))
idx_max_parahipp = np.argmax(np.mean(firingrate_parahippocampal,axis=0))
idx_max_hippocampus = np.argmax(np.mean(firingrate_hippocampus,axis=0))
idx_max_pcc = np.argmax(np.mean(firingrate_pcc,axis=0))


# Plot vertical line where the maximum value is
ax.vlines(x[idx_max_tp],ymin =0, ymax=np.max(fr_tp_norm),color='palevioletred',linestyle='--',linewidth=2)
ax.vlines(x[idx_max_parahipp],ymin =0, ymax=np.max(fr_parahippocampal_norm),color='sandybrown',linestyle='--',linewidth=2)
ax.vlines(x[idx_max_hippocampus],ymin =0, ymax=np.max(fr_hippocampus_norm),color='darkslateblue',linestyle='--',linewidth=2)
ax.vlines(x[idx_max_pcc],ymin =0, ymax=np.max(fr_pcc_norm),color='cornflowerblue',linestyle='--',linewidth=2)

 

ax.legend(['Parahippocampal (n = '+ str(np.shape(neuron_parahippocampal)[0]) + ')',
           'Temporal pole (n = ' + str(np.shape(neuron_tp)[0]) + ')',
           'Hippocampus (n = '+ str(np.shape(neuron_hippocampus)[0]) + ')',
           'PCC (n = '+ str(np.shape(neuron_pcc)[0]) + ')'],
           loc='upper left',
           fontsize=12)


# Add SEM around mean activity
ax.fill_between(x,fr_tp_norm+stats.sem(firingrate_tp),
                fr_tp_norm-stats.sem(firingrate_tp),
                color='palevioletred',alpha=0.4)

ax.fill_between(x,fr_hippocampus_norm+stats.sem(firingrate_hippocampus),
                fr_hippocampus_norm-stats.sem(firingrate_hippocampus),
                color='darkslateblue',alpha=0.4)



ax.fill_between(x,fr_parahippocampal_norm+stats.sem(firingrate_parahippocampal),
                fr_parahippocampal_norm-stats.sem(firingrate_parahippocampal),
                color='sandybrown',alpha=0.4)



ax.fill_between(x,fr_pcc_norm+stats.sem(firingrate_pcc),
                fr_pcc_norm-stats.sem(firingrate_pcc),
                color='cornflowerblue',alpha=0.4)
ax.set_xlim(-0.5,1)
ax.set_title('Grand average of firing rates',fontsize=16)

ax.set_xlabel('Time (s)',fontsize=14)
ax.set_ylabel('Neuronal activity - Baseline corrected',fontsize=14)
plt.savefig(path2figure+'/FiringRate_AllNeurons_ROI.svg')
plt.show()

# Print timing of maximum firing rate
print('Max TP = '+str(x[idx_max_tp]))
print('\nMax Parahippocampal = '+str(x[idx_max_parahipp]))
print('\nMax Hippocampus = '+str(x[idx_max_hippocampus]))
print('\nMax PCC = '+str(x[idx_max_pcc]))
