'''
Created by VD on 23/09/2025

In this script we will want to test if neuronal responses are modulated by repetition of stimuli

Concept cells in the MTL tend to have a repetition suppression effect (they decreased their number of spikes
with the increasing number of presentations of a stimulus) (see Pedreira et al., 2010)

For now I only keep sessions where 8 stimuli were presented to have consistency

This script has been used to do Fig. 4g, Extended Data Fig. 5h
'''


import warnings
warnings.filterwarnings("ignore")

# Import libraries needed
import json 
import re 
import numpy as np
import os
import pynapple as nap
import matplotlib.pyplot as plt
import mne
import seaborn as sns
from statsmodels.sandbox.stats.multicomp import multipletests


from scipy import io as io
from scipy import stats as stats

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

def get_repetition_spikes(path_json):
    # Open the dictionary containing all single-units registered in the temporal pole
    with open(path_json, "r") as f:
        neurons_tp = json.load(f)


    # Extract list of patients from dictionary
    list_patient = list(neurons_tp.keys())


    # General variable
    sr = 32768 # Sampling rate of Neuralynx system

    all_fr_images = list()
    fr_neurons = list()
    fr_neurons_image = np.zeros((1,200))

    repetition_score = []

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
            data_path = r'F:\Screening\database/'+bsnm+'/sess-'+str(session)
            path_images = data_path+'/stimuli/' #"E:/screening hors eeg/Screening images/Pool images/"
            path_results = 'F:/Screening/results/Plot_Population_TP/'


            # Check whether the specified path exists or not
            isExist = os.path.exists(path_results)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(path_results)

            # Load logfile
            logfname=data_path+'/lfps/'+bsnm+'_ses-01_task-Screening_run-01_ieeg_log.txt'

            logLines=np.array(read_lines(logfname, removeEndLines=True))

            stream=np.arange(len(logLines)/25, dtype=int)*25+1
            chRegs=np.array([line.split('.')[0] for line in logLines[stream]])
            nCh=chRegs.shape[0]
            intermed_=logLines[6].split(' ')[1]
            print(chRegs,len(chRegs))


            spikes = load_spikes(data_path)

            # Load Run Lists and get TTL associated with each image
            test_mat_Run1 = io.loadmat(path_images+'run-01.mat') # et concaténer les 4 runs
            my_array_Run1 = test_mat_Run1['trial']
            keys_TTL = [my_array_Run1[0,i][1][0][0] for i in range(my_array_Run1.shape[1])] 
            values_images = [my_array_Run1[0,i][2][0] for i in range(my_array_Run1.shape[1])] 
            dict_TTL2Image = {k: v for k, v in zip(keys_TTL, values_images)}

            


            # Load TTLs & dat files
            folder_ttl = data_path+'/ttl'
            #folder_TTL = dataFolder[:32]+'saveTTL/_TTLs4Python/s'+session+'/'
            folder_TTL = folder_ttl+'/'
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

                    all_fr_images = list()
                    for TimingImage in imgTS:
                        Image_Second = TimingImage/sr

                        # Calcul firing rate
                        TS_Index = nap.Ts(Image_Second)


                        # Get spikes in the trial
                        Trial_peth = nap.compute_perievent(spikes[iNeuron], TS_Index, minmax=(-1,1))


                        # Test for statistical difference between baseline and trial period
                        ep_trial = nap.IntervalSet(start=0, end=1, time_units='s')
                        ep_baseline = nap.IntervalSet(start=-1,end=0,time_units='s')
                        spike_trial = Trial_peth.restrict(ep_trial)
                        spike_baseline = Trial_peth.restrict(ep_baseline)
                        

                        count_spike_bl = spike_baseline.count().d[0]
                        count_spike_trial = spike_trial.count().d[0]

                        # Compute Wilcoxon signed-rank test between the 100 ms bin window and baseline (-400 ; 0)
                        diff_bl_trial = np.array(count_spike_trial)-np.array(count_spike_bl)
                        
                        count_zero = not np.any(diff_bl_trial)

                        # If no difference between baseline and trial wilcoxon can't be compute so had to add this if
                        if count_zero == False:
                            result_stats = stats.wilcoxon(count_spike_trial,count_spike_bl)
                            p_value=result_stats.pvalue


                        fr_im = get_firingrate(Trial_peth)
                        firing_rate_trial = fr_im[:,100:200] # Get the values and transpose to get 8 * 100 format 

                        baseline_rate_trial = fr_im[:,0:100]

                        F_obs, clusters, clusters_pv,H0 = mne.stats.permutation_cluster_test([baseline_rate_trial,firing_rate_trial])

                        try:
                            p_value_final = np.min(clusters_pv)
                        except:
                            p_value_final = 1
                        
                        
                        if p_value_final < 0.05 and firing_rate_trial.shape[0]==8:
                            normalized_count_spike = count_spike_trial / np.max(count_spike_trial)
                            

                            try:
                                repetition_score.append(normalized_count_spike)
                            except:
                                a = 1
                            

                                
                


                            
                            

    repetition_score = np.array(repetition_score)
                         
    return repetition_score



# Get the number of spikes across repetitions
try:
    repetition_tp = np.load('C:/Users/dornier/GitHub/ConceptCells_TP/Semantic_Coding/repetition_effect/repetition_tp.npy')
except:
    repetition_tp = get_repetition_spikes('C:/Users/dornier/GitHub/ConceptCells_TP/dictionary_singleunits_tp.json')
    np.save('C:/Users/dornier/GitHub/ConceptCells_TP/Semantic_Coding/repetition_effect/repetition_tp.npy',repetition_tp)

# Hstack repetition value, with first trial values first
repetition_tp_stack = np.hstack(repetition_tp.T)

# Create a vector corresponding to the # of the trial in array above
id_trial = np.hstack(np.repeat(np.arange(8),np.shape(repetition_tp)[0]))

# Statistical test
result = stats.linregress(repetition_tp_stack,id_trial)

er
print('Effet du numéro d essai: '+str(pvalue))
fig, ax1 = plt.subplots(1,1,layout='constrained',figsize=(3.5,5))
ax1 = sns.pointplot(repetition_tp,color='palevioletred')
ax1.set_xlabel('Trial number',fontsize=14)
ax1.set_xticklabels(labels=['1','2','3','4','5','6','7','8'],fontsize=10)
ax1.set_ylabel('Normalized spike count',fontsize=14)
plt.savefig('C:/Users/dornier/PhD/Article/Concept_Cells_temporal_pole/Figures/Figure4/Repetition_Suppression/Norm_Spikes_Trial_TP_v2.svg')
plt.show()
plt.close()