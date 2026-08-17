'''
Author : Dornier Vincent (25/09/2025)


This script will be used to compute the onset latencies of response of neurons
This method is based on the one described in Reddy et al. (2015)

References : 
Reddy, L., Poncet, M., Self, M. W., Peters, J. C., Douw, L., van Dellen, E., Claus, S., 
Reijneveld, J. C., Baayen, J. C., & Roelfsema, P. R. (2015). Learning of anticipatory responses 
in single neurons of the human medial temporal lobe. Nature Communications, 6, 8556. https://doi.org/10.1038/ncomms9556

This script is used to do Fig. 5b
'''

import scipy.signal as sig




# Import libraries
import pynapple as nap
import json
import re 
import numpy as np
import os
import pynapple as nap
import matplotlib.pyplot as plt
import mne
import seaborn as sns
import math
import scipy
import astropy
from scipy.ndimage import gaussian_filter
from astropy.convolution import Gaussian1DKernel, convolve
import itertools
import seaborn as sns
import pandas as pd

from scipy import io as io
from scipy import stats as stats

################################################################
############## FUNCTIONS USED LATER IN THE SCRIPT ##############
################################################################

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



def onset_neurons(path_json):
    # Open the dictionary containing all single-units registered in the temporal pole
    with open(path_json, "r") as f:
        neurons_tp = json.load(f)


    # Extract list of patients from dictionary
    list_patient = list(neurons_tp.keys())


    # General variable
    sr = 32768 # Sampling rate of Neuralynx system


    onset_neuron_list = list()

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
                    


                    

                    onset_image = list()
                    for TimingImage in imgTS:
                        Image_Second = TimingImage/sr

                        # Calcul firing rate
                        TS_Index = nap.Ts(Image_Second)


                        # Get spikes in the trial
                        Trial_peth = nap.compute_perievent(spikes[iNeuron], TS_Index, minmax=(-0.5,1))
                        Trial_peth2 = nap.compute_perievent(spikes[iNeuron], TS_Index, minmax=(-1,1))



                        # Compute the firing rate of each trial
                        firing_rate = get_firingrate(Trial_peth2)

                        firing_rate_trial = firing_rate[:,100:200] 

                        baseline_rate_trial = firing_rate[:,0:100]
                        

                        F_obs, clusters, clusters_pv,H0 = mne.stats.permutation_cluster_test([baseline_rate_trial,firing_rate_trial]) 


                        try:
                            p_value_final = np.min(clusters_pv)
                        except:
                            p_value_final = 1                   


                        # Test for statistical difference between baseline and trial period
                        ep_trial = nap.IntervalSet(start=0, end=1, time_units='s')
                        ep_baseline = nap.IntervalSet(start=-1,end=0,time_units='s')
                        spike_trial = Trial_peth.restrict(ep_trial)
                        spike_baseline = Trial_peth.restrict(ep_baseline)
                        

                        count_spike_bl = spike_baseline.count().d[0]
                        count_spike_trial = spike_trial.count().d[0]

                        
                        # Si la réponse est significative à l'image
                        if p_value_final < 0.05 and np.sum(count_spike_trial) > np.sum(count_spike_bl):
                            # Count the number of spikes in 1-ms window (obtain a nTrials x nSamples array-like)
                            count_spike = Trial_peth.count(bin_size=1,time_units='ms').d.T


                            # Gaussian kernel with sd = 100 ms
                            gauss_kernel = Gaussian1DKernel(100)

                            smoothed_data_gauss = [convolve(count_spike[iTrial], gauss_kernel) for iTrial in range(count_spike.shape[0])]

                            firing_rate = np.array(smoothed_data_gauss)


                            # Get the ISI 
                            ISI = np.mean(firing_rate[:,0:500],axis=1)

                            # Trial
                            trial_rate = firing_rate[:,500:-1]

                            # Compare each sample with baseline
                            p_values = []
                            for iSample in range(trial_rate.shape[1]):

                                result = stats.ttest_rel(trial_rate[:,iSample],ISI)

                                p_values.append(result.pvalue)
                            
                            
                            p_values = np.array(p_values)
                            

                            # Create an array of 0 and 1 where significant
                            sig = np.zeros(1000)
                            sig[np.argwhere(p_values<0.05)] = 1

                            

                            # runs get the length of blocks of same values
                            runs = [len(list(g)) for _,g in itertools.groupby(sig)]

                            # As runs get length of blocks it can only start with zeros
                            # So to get the length of 0s and 1s we first need to detect by which value it starts

                            if sig[0] == 0:
                                # If starts with zeros then runs[0] = 0 and runs[1] = 1, runs[2] = 0 eventually runs[3] = 1
                                length_1 = runs[1::2]
                            else:
                                # If starts with zeros then runs[0] = 1 and runs[1] = 0, runs[2] = 1 eventually runs[3] = 0
                                length_1 = runs[0::2]
                            
                            try:
                                # Get the index in runs where is the maximul length
                                idx_debut_max = np.where(runs==np.max(length_1))

                                # Extract values from runs before the consecutive trains 
                                # Then we sum these values to get the first index of trains
                                runs_before_onset = runs[0:idx_debut_max[0][0]]

                                # This is the onset in ms
                                idx_onset = np.sum(runs_before_onset)
                            
                                onset_image.append(idx_onset)
                            except:
                                a=1
                    
                    onset_neuron_list.append(np.mean(onset_image))
    
    onset_neuron_list = np.array(onset_neuron_list)
    onset_neuron_list = onset_neuron_list[~np.isnan(onset_neuron_list)]

    
    return onset_neuron_list
                        


try:
    poisson_tp = np.load('C:/Users/dornier/PhD/Article/Concept_Cells_temporal_pole/Figures/Figure5/Latency/Latency_Leila_TP.npy')
except:
    poisson_tp = onset_neurons(r"C:\Users\dornier\GitHub\ConceptCells_TP/dictionary_singleunits_tp_latency.json")
    np.save('C:/Users/dornier/PhD/Article/Concept_Cells_temporal_pole/Figures/Figure5/Latency/Latency_Leila_TP.npy',poisson_tp)


# For neurons in the hippocampus
try:
    poisson_hipp = np.load('C:/Users/dornier/PhD/Article/Concept_Cells_temporal_pole/Figures/Figure5/Latency/Latency_Leila_Hippocampus.npy')
except:
    poisson_hipp = onset_neurons(r"C:\Users\dornier\GitHub\ConceptCells_TP/dictionary_singleunits_hippocampus.json")
    np.save('C:/Users/dornier/PhD/Article/Concept_Cells_temporal_pole/Figures/Figure5/Latency/Latency_Leila_Hippocampus.npy',poisson_hipp)


# For neurons in GPH
try:
    poisson_gph = np.load('C:/Users/dornier/PhD/Article/Concept_Cells_temporal_pole/Figures/Figure5/Latency/Latency_Leila_GPH.npy')
except:
    poisson_gph = onset_neurons(r"C:\Users\dornier\GitHub\ConceptCells_TP/dictionary_singleunits_parahippocampal.json")
    np.save('C:/Users/dornier/PhD/Article/Concept_Cells_temporal_pole/Figures/Figure5/Latency/Latency_Leila_GPH.npy',poisson_gph)


# For neurons in rhinal cortex
try: 
    poisson_rhinal = np.load('C:/Users/dornier/PhD/Article/Concept_Cells_temporal_pole/Figures/Figure5/Latency/Latency_Leila_Rhinal.npy')
except: 
    poisson_rhinal = onset_neurons(r"C:\Users\dornier\GitHub\ConceptCells_TP/dictionary_singleunits_rhinal_cortex.json")
    np.save('C:/Users/dornier/PhD/Article/Concept_Cells_temporal_pole/Figures/Figure5/Latency/Latency_Leila_Rhinal.npy',poisson_rhinal)


# For neurons in amygdala
try: 
    poisson_amygdala = np.load('C:/Users/dornier/PhD/Article/Concept_Cells_temporal_pole/Figures/Figure5/Latency/Latency_Leila_Amygdala.npy')
except: 
    poisson_amygdala = onset_neurons(r"C:\Users\dornier\GitHub\ConceptCells_TP/dictionary_singleunits_amygdala.json")
    np.save('C:/Users/dornier/PhD/Article/Concept_Cells_temporal_pole/Figures/Figure5/Latency/Latency_Leila_Amygdala.npy',poisson_amygdala)


# For neurons in pcc
try: 
    poisson_pcc = np.load('C:/Users/dornier/PhD/Article/Concept_Cells_temporal_pole/Figures/Figure5/Latency/Latency_Leila_PCC.npy')
except: 
    poisson_pcc = onset_neurons(r"C:\Users\dornier\GitHub\ConceptCells_TP/dictionary_singleunits_pcc.json")
    np.save('C:/Users/dornier/PhD/Article/Concept_Cells_temporal_pole/Figures/Figure5/Latency/Latency_Leila_PCC.npy',poisson_pcc)


# Statistical part
kruskal_wallis = stats.kruskal(poisson_tp,poisson_hipp,poisson_rhinal,poisson_gph, poisson_amygdala, poisson_pcc)
er

print('Kruskal-wallis = '+str(kruskal_wallis.statistic)+', p = '+str(kruskal_wallis.pvalue))

# two-by-two comparisons
tp_hippocampus = stats.mannwhitneyu(poisson_tp,poisson_hipp)
print('\nTP vs Hippocampus = '+str(tp_hippocampus.statistic)+', p = '+str(tp_hippocampus.pvalue))


tp_gph = stats.mannwhitneyu(poisson_tp,poisson_gph)
print('\nTP vs GPH = '+str(tp_gph.statistic)+', p = '+str(tp_gph.pvalue))


tp_rhinal = stats.mannwhitneyu(poisson_tp,poisson_rhinal)
print('\nTP vs Rhinal = '+str(tp_rhinal.statistic)+', p = '+str(tp_rhinal.pvalue))

tp_amygdala = stats.mannwhitneyu(poisson_tp,poisson_amygdala)
print('\nTP vs Amygdala = '+str(tp_amygdala.statistic)+', p = '+str(tp_amygdala.pvalue))

tp_pcc = stats.mannwhitneyu(poisson_tp,poisson_pcc)
print('\nTP vs PCC = '+str(tp_pcc.statistic)+', p = '+str(tp_pcc.pvalue))

# Plot 
# Plot part


plot = 'ridgeline'

if plot == 'ridgeline':

    # Transform our arrays into pandas dataframe
    df_tp = pd.DataFrame({'TP': poisson_tp})
    df_gph = pd.DataFrame({'Parahippocampal': poisson_gph})
    df_rhinal = pd.DataFrame({'Rhinal': poisson_rhinal})
    df_hippocampus = pd.DataFrame({'Hippocampus': poisson_hipp})
    df_amygdala = pd.DataFrame({'Amygdala': poisson_amygdala})
    df_pcc = pd.DataFrame({'PCC': poisson_pcc})

    
    df_roi = pd.concat([df_tp,df_gph,df_rhinal,df_hippocampus,df_amygdala,df_pcc], axis=1) 
    df_roi = pd.concat([df_tp,df_gph,df_hippocampus,df_amygdala,df_pcc], axis=1) 


    #fig, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(6,1,figsize=(8,3),sharex=True)
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,figsize=(8,3),sharex=True)



    fig.subplots_adjust(hspace=-0.7)

    ax1 = sns.kdeplot(data=df_roi, x='Parahippocampal',fill=True,ax=ax1,color='sandybrown')
    ax1.set_axis_off()


    ax2 = sns.kdeplot(data=df_roi, x='TP',fill=True,ax=ax2,color='palevioletred')

    ax2.set_axis_off()

    # ax3 = sns.kdeplot(data=df_roi, x='Rhinal',fill=True,ax=ax3,color='seagreen')
    # ax3.set_axis_off()

    ax3 = sns.kdeplot(data=df_roi, x='Hippocampus',fill=True,ax=ax3,color='darkslateblue')
    ax3.set_axis_off()

    ax4 = sns.kdeplot(data=df_roi, x='PCC',fill=True,ax=ax4,color='cornflowerblue')
    #ax4.set_axis_off()


    #ax5 = sns.kdeplot(data=df_roi, x='PCC',fill=True,ax=ax5,color='cornflowerblue')
    

    back6 = ax4.patch
    back6.set_alpha(0)
    # remove borders, axis ticks, and labels
    ax4.set_yticklabels([])
    ax4.set_ylabel('')
    ax4.set_xlabel('Onset time (ms)')



    spines = ["top","right","left"]
    for s in spines:
        ax4.spines[s].set_visible(False)

    fig.legend(labels=['Parahippocampal','TP','Hippocampus','PCC'])
    ax1.set_title('Onset time for each region')
    
    plt.savefig('C:/Users/dornier/PhD/Article/Concept_Cells_temporal_pole/Figures/Figure5/Latency/Ridgelineplot_OnsetLatencies_All_Regions_v3.svg')


elif plot == 'histogram':

    

    fig, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(1,6,layout='constrained', figsize=(16,7.5), sharey=True) 
    ax2.hist(poisson_tp,bins=20,color='palevioletred',range=(0,1000),alpha=0.6)
    ax2.set_title('Temporal pole',fontsize=17)
    ax1.set_xlabel('Onset time (ms)',fontsize=15)
    ax1.set_ylabel('Number of units',fontsize=15)
    ax4.hist(poisson_hipp,bins=20,color='darkslateblue',range=(0,1000),alpha=0.6)
    ax4.set_title('Hippocampus',fontsize=17)
    ax3.hist(poisson_rhinal,bins=20,color='seagreen',range=(0,1000),alpha=0.6)
    ax3.set_title('Rhinal cortex',fontsize=17)
    ax1.hist(poisson_gph,bins=20,color='sandybrown',range=(0,1000),alpha=0.9)
    ax1.set_title('Parahippocampal',fontsize=17)
    ax5.hist(poisson_amygdala,bins=20,color='saddlebrown',range=(0,1000),alpha=0.6)
    ax5.set_title('Amygdala',fontsize=17)
    ax6.hist(poisson_pcc,bins=20,color='cornflowerblue',range=(0,1000),alpha=0.6)
    ax6.set_title('PCC',fontsize=17)
    plt.savefig('C:/Users/dornier/PhD/Article/Concept_Cells_temporal_pole/Figures/Figure5/Latency/Distribution_Leila_Latencies_v4.svg')
    plt.show()
    plt.close()

    print('ALL DONE')
