'''
Created by VD (28/11/2025)

In this script we will perform spike-train reliability across trials

Our hypothesis is that spike train in TP should be more reliable than in the hippocampus
because responses in TP are more evoked and linked with visual areas than the hippocampus

The closer the permutation entropy normalized is to 1 the noisier it is

The method is based on this paper : 
Waschke, L., Kamp, F., van den Elzen, E., Krishna, S., Lindenberger, U., Rutishauser, U., & Garrett, D. D. (2025). 
Single-neuron spiking variability in hippocampus dynamically tracks sensory content during memory formation in humans. 
Nature Communications, 16(1), 236. https://doi.org/10.1038/s41467-024-55406-4

For the parameters of the permutation entropy I used parameters used in the same study, they can be find here:
https://github.com/LNDG/SpikeVar/blob/v1.0.0/neural/b_compute_spike_pe.m


References about permutation entropy
https://www.aptech.com/blog/permutation-entropy/ 

This script is used to generate Extended Data Fig. 7a

'''
# Import libraries
import EntropyHub as entropy
import pynapple as nap
import json
import os
import numpy as np
import scipy.io as io
import re
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import math
np.math = math

# Inputs
path2data = 'C:/Users/dornier/GitHub/ConceptCells_DecMem/'
path2save = 'C:/Users/dornier/GitHub/ConceptCells_DecMem/data/Spiking_Variability/'


###################
#### FUNCTIONS ####
###################

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


def get_permutation_entropy(path_json):
    # Open the dictionary containing all single-units registered in the temporal pole
    with open(path_json, "r") as f:
        neurons_tp = json.load(f)


    # Extract list of patients from dictionary
    list_patient = list(neurons_tp.keys())


    # General variable
    sr = 32768 # Sampling rate of Neuralynx system

    # Initialize lists to store data
    PEnorm2 = []
    PEnorm3 = []
    PEnorm4 = []
    pe4_meanNeuron = []

    

    # Loop over patients having single-units in the temporal pole
    for iPatient in list_patient:

        # Extract sessions from the dictionnary
        list_session = list(neurons_tp[iPatient][0].keys())

        bsnm=iPatient # ID of the patient
        
        # Loop over sessions containing units in the TP
        for iSession  in list_session:

           


            # Path where data is stored
            data_path = path2data+'/data/Examples_Session/'+bsnm+'/'+iSession
            path_images = data_path+'/stimuli/' #"E:/screening hors eeg/Screening images/Pool images/"
            

            # Load logfile
            logfname=data_path+'/lfps/'+bsnm+'_ses-01_task-Screening_run-01_ieeg_log.txt'

            logLines=np.array(read_lines(logfname, removeEndLines=True))
            stream=np.arange(len(logLines)/25, dtype=int)*25+1
            chRegs=np.array([line.split('.')[0] for line in logLines[stream]]) # Name of the channels

            print(chRegs,len(chRegs))

            # Load spiking activity
            spikes = load_spikes(data_path)

            # Load Run Lists and get TTL associated with each image
            test_mat_Run1 = io.loadmat(path_images+'run-01.mat') # et concaténer les 4 runs
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

            # Extract list of neurons from the session analyzed
            list_neurons = neurons_tp[iPatient][0][iSession]

            # Loop over neurons of interest
            if list_neurons:
                for iNeuron in list_neurons:
                    PE4 = []

                    # Loop over all stimuli presented
                    for TimingImage in imgTS:

                        # Transform sample into second
                        Image_Second = TimingImage/sr

                        # Calcul firing rate
                        TS_Index = nap.Ts(Image_Second)


                        # Get spikes in the trial
                        Trial_peth = nap.compute_perievent(spikes[iNeuron], TS_Index, minmax=(0,1))

                        # Extract the number of spikes for 10 ms non-overlapping bins for each trial
                        bincount1 = [Trial_peth[iTrial].count(0.01).d for iTrial in range(len(Image_Second))]

                        


                        try:
                            bincount = np.array(bincount1)
                        
                        except:
                            list2 = [x for x in bincount1 if x.any()]
                            bincount = np.array(list2)

                        
                        # Get the permutation entropy
                        # If trials have no spike skip it
                        if bincount.any():
                            for iii in range(len(bincount)):
                                Perm, Pnorm, cPe = entropy.PermEn(bincount[iii],m=4,tau=1,Norm=True)

                                PEnorm2.append(Pnorm[1])
                                PEnorm3.append(Pnorm[2])
                                PEnorm4.append(Pnorm[3])

                                PE4.append(Pnorm[3])
                    
                    # Get the mean entropy of the neuron
                    pe4_meanNeuron.append(np.nanmean(PE4))
                    
                        
    return pe4_meanNeuron
                        

            
# Compute the permutation entropy for temporopolar regions 
try: 

    tp_pe = np.load(path2data+'/data/Spiking_Variability/Permutation_Entropy_TP.npy')
except:
    tp_pe = get_permutation_entropy(r'C:\Users\dornier\GitHub\ConceptCells_TP/dictionary_singleunits_tp.json')
    np.save(path2data+'/data/Spiking_Variability/Permutation_Entropy_TP.npy',tp_pe)



# Compute the permutation entropy for hippocampus
try: 
    hipp_pe = np.load(path2data+'/data/Spiking_Variability/Permutation_Entropy_Hippocampus.npy')
except:
    hipp_pe = get_permutation_entropy(r'C:\Users\dornier\GitHub\ConceptCells_TP/dictionary_singleunits_hippocampus.json')
    np.save(path2data+'/data/Spiking_Variability/Permutation_Entropy_Hippocampus.npy',hipp_pe)





# Compute the permutation entropy for parahippocampal cortex
try: 
    gph_pe = np.load(path2data+'/data/Spiking_Variability/Permutation_Entropy_Parahippocampal.npy')
except:
    gph_pe = get_permutation_entropy(r'C:\Users\dornier\GitHub\ConceptCells_TP/dictionary_singleunits_parahippocampal.json')
    np.save(path2data+'/data/Spiking_Variability/Permutation_Entropy_Parahippocampal.npy',gph_pe)



# Compute the permutation entropy for pcc
try: 
    pcc_pe = np.load(path2data+'/data/Spiking_Variability/Permutation_Entropy_PCC.npy')
except:
    pcc_pe = get_permutation_entropy(r'C:\Users\dornier\GitHub\ConceptCells_TP/dictionary_singleunits_pcc.json')
    np.save(path2data+'/data/Spiking_Variability/Permutation_Entropy_PCC.npy',pcc_pe)






# Plot part
# Plot
fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4,layout='constrained',figsize=(7,5),sharex=True)


ax1 = sns.violinplot(tp_pe,ax=ax1,color='palevioletred')
ax2 = sns.violinplot(hipp_pe,ax=ax2,color='darkslateblue')
ax3 = sns.violinplot(gph_pe,ax=ax3,color='sandybrown')
ax4 = sns.violinplot(pcc_pe,ax=ax4,color='cornflowerblue')



ax1.set_xlabel('Temporal pole')
ax2.set_xlabel('Hippocampus')
ax3.set_xlabel('Parahippocampal')
ax4.set_xlabel('PCC')


ax1.set_ylim(-0.1,1)
ax2.set_ylim(-0.1,1)
ax3.set_ylim(-0.1,1)
ax4.set_ylim(-0.1,1)

# remove borders, axis ticks, and labels
ax2.set_yticklabels([])
ax3.set_yticklabels([])
ax4.set_yticklabels([])



ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax2.set_yticks([])
ax3.set_yticks([])
ax4.set_yticks([])


spines = ["top","right","left"]
for s in spines:
    ax2.spines[s].set_visible(False)
    ax3.spines[s].set_visible(False)
    ax4.spines[s].set_visible(False)



ax1.set_ylabel('Permutation entropy value normalized',fontsize=12)

plt.savefig(path2save+'/Permutation_Entropy_All_ROI.svg')

plt.show()






