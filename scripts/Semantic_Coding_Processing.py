'''
Author : Vincent Dornier

In this script we will compute the semantic proximity between all stimuli with word2vec
Then computing the correlation of the firing rate for each stimuli
To then compare the two to see if regions linked to semantic proximity

This script has been used to perform analyses of Semantic coding presented in Fig. 3e, f, h and Fig. 5f
'''
# Import libraries
import gensim
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pynapple as nap
from scipy import io as io
import scipy.stats as stats
import json
import random
import time
import mne
from scipy.spatial.distance import cosine
from gensim.models import Word2Vec

# Path where word2vec by Google is stored
path = 'C:/Users/dornier/PhD/ConceptCells_Analysis/Semantic_Coding/word2vec/GoogleNews-vectors-negative300.bin.gz'

# Load model (here word2vec)
model = KeyedVectors.load_word2vec_format(path, binary=True)



#########################################
############### Functions ###############
#########################################

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
        a=1

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

def word2vec_correlation(id_patient,id_session,model):
    '''
    With this function we compute the similarity between words using word2vec
    We order word by groups of labels and then return the correlation matrix and the order of the words
    So the firing rate correlation then could be compute in the same order

    Info about word2vec Mikolov et al. (2013): https://arxiv.org/abs/1301.3781 

    We use the gensim package (https://github.com/piskvorky/gensim)


    Parameters 
    -------------
    id_patient : int
        Numero of the patient you want to analyze

    id_session : int
        Numero of the session you want to analyze
    
    Output
    -------------
    corr : array-like
        Array-like containing the cosine similarity between labels

    label_order : array
        Array containing the label of the words sorted by groups
    '''
    # Path to the database where words labels are stored
    path_database = 'C:/Users/dornier/GitHub/ConceptCells_TP/Semantic_Coding/database/'+id_patient+'/'+id_session+'/'


    # Load the labels of each pictures in a dataframe
    try:
        semantic = pd.read_csv(path_database+'Semantic_Category_Stimuli.csv',sep=';',header=0)


        # Keep only the word labels that we use for word2vec
        try:
            label_word2vec = semantic['word2vec'].values
        except:
            print('No column word2vec, are you sure it exists?')

        label_word2vec_sorted = np.array(sorted(label_word2vec))


    except:
        semantic = pd.read_csv(path_database+'Semantic_Category_Stimuli.csv',sep=',',header=0)


        # Keep only the word labels that we use for word2vec
        try:
            label_word2vec = semantic['word2vec'].values
        except:
            print('No column word2vec, are you sure it exists?')
    
    label_word2vec_sorted = np.array(sorted(label_word2vec))
    
    
    # Initialize the matrix of correlation with the appropriate shape (nWords x nWords)
    corr = np.zeros((len(label_word2vec_sorted),len(label_word2vec_sorted)))


    # Compute the correlation between all words with nested loop
    for i in range(len(label_word2vec_sorted)):
        for j in range(len(label_word2vec_sorted)):
            
            # Compute the cosine similarity according to pre-trained word2vec
            corr[i,j] = model.similarity(label_word2vec_sorted[i], label_word2vec_sorted[j])

    # Replace higher triangle with zeros
    corr_low_triangle = np.tril(corr)

    # Only keep lower triangle   
    corr_withoutzeros = corr_low_triangle[corr_low_triangle!=0]

    return corr, label_word2vec_sorted, corr_withoutzeros


def corr_firingrate(id_patient,id_session,id_neuron):
    '''
    This function compute the firing rate of a neuron Y for each pair of picture
    It computes it in the same order as the vector for word2vec correlation 
    As they are in the same order we can easily compare them later


    Parameters
    ------------
    id_patient : int
        Number of the patient in your database

    id_session : int
        Number of the session to analyze
    
    id_neurone : int
        Number of the neurone of interest
    
    Output
    ------------
    corr_fr : array-like
        Array-like containing the correlation of firing rate for each pair of picture

    '''
    # Get the correlation between firing rate 

    # Path where your neuronal data are stored
    data_path = 'F:/Screening/database/'+id_patient+'/'+id_session+'/'

    # Path where stimuli and run lists are stored
    path_images = data_path+'stimuli/'

 


    # Load logfile
    logfname=data_path+'lfps/'+id_patient+'_ses-01_task-Screening_run-01_ieeg_log.txt'
    logLines=np.array(read_lines(logfname, removeEndLines=True))
    stream=np.arange(len(logLines)/25, dtype=int)*25+1
    chRegs=np.array([line.split('.')[0] for line in logLines[stream]])



    # Load spikes data from nwb file after cleaning with Klusters
    spikes = load_spikes(data_path)

    # Load Run Lists and get TTL associated with each image
    test_mat_Run1 = io.loadmat(path_images+'run-01.mat') # et concaténer les 4 runs
    my_array_Run1 = test_mat_Run1['trial']
    keys_TTL = [my_array_Run1[0,i][1][0][0] for i in range(my_array_Run1.shape[1])] 
    values_images = [my_array_Run1[0,i][2][0] for i in range(my_array_Run1.shape[1])] 
    dict_TTL2Image = {k: v for k, v in zip(keys_TTL, values_images)}


    # Load label used for word2vec to get ttl associated
    path_database_word2vec = 'C:/Users/dornier/GitHub/ConceptCells_TP/Semantic_Coding/database/'+id_patient+'/'+id_session +'/'

    # Open csv file containing word2vec labels
    try:
        df_semantic = pd.read_csv(path_database_word2vec+'Semantic_Category_Stimuli.csv',sep=';',header=0)

        df_semantic_ordered = df_semantic.sort_values('word2vec',ignore_index=True)
    
    except:
        df_semantic = pd.read_csv(path_database_word2vec+'Semantic_Category_Stimuli.csv',sep=',',header=0)

        df_semantic_ordered = df_semantic.sort_values('word2vec',ignore_index=True)

    

    # Load TTLs & dat files
    folder_ttl = data_path+'ttl/'
    TTLvals = io.loadmat(folder_ttl+id_patient+'_TTLvals_tot.mat')['TTLvals_tot'][0]
    TS = io.loadmat(folder_ttl+id_patient+'_TS_tot.mat')['TS_tot'][0]

    # Load timestamps
    tsname=data_path+'/lfps/'+id_patient+'_ses-01_task-Screening_run-01_timestamps.dat'
    TS_stream=np.memmap(tsname, mode='r', dtype=float, order='F')


    # Get TTLs sync with EEG
    TS_32768 = np.searchsorted(TS_stream, TS, side='left') # index temps absolu
    TS_32768[TS_32768 ==len(TS_stream)] = len(TS_stream)-1


    # Regroup TTLs by class of stimulus presented
    # Check that all different TTLs correspond to images
    # Otherwise, remove bad TTLs
    imgs=np.unique(TTLvals)

    # Suppress the first value (= 0) because correspond to fixation cross
    imgs = imgs[1:]

    
    # Initialize lists to store data
    fr_pictures = list()
    is_responsive = list()

    # Loop over all stimuli presented
    for iImage in range(len(imgs)):
        

        try:

            # Get the image to analyze to have the firing activity in the same order than word2vec matrix
            img2analyze = df_semantic_ordered['namefile'][iImage]

            # Find TTL for the image to analyze first
            keys = np.array([key for key, val in dict_TTL2Image.items() if val == img2analyze][0])

            # Get the index where the image of interest was displayed
            imgIndices=np.where(TTLvals==keys)

            # Get the timestamps corresponding to the onset of the picture
            imgTS=TS_32768[imgIndices]

            # Transform the timing from sample to second
            Image_Second = imgTS/32768

            # Calcul firing rate
            TS_Index = nap.Ts(Image_Second)

            # Get spikes in the trial and baseline
            Trial_stimulus = nap.compute_perievent(spikes[id_neuron], TS_Index, minmax=(0,1))
            trial_baseline = nap.compute_perievent(spikes[id_neuron], TS_Index, minmax=(-1,0))
            

            # Get the firing rate in trial and baseline
            firing_rate_stimulus = get_firingrate(Trial_stimulus)
            firing_rate_baseline = get_firingrate(trial_baseline)


            # Compute cluster-based permutation test to assess significance
            F_obs, clusters, clusters_pv,H0 = mne.stats.permutation_cluster_test([firing_rate_baseline,firing_rate_stimulus])

            try:
                p_value_final = np.min(clusters_pv)
            except:
                p_value_final = 1

            # Store 1 if significant and zero if not (we will be able to determine index in the matrice of response)
            if p_value_final < 0.05 :
                is_responsive.append(1)
            else:
                is_responsive.append(0)
                

            

            # Get the mean for one particular image (i.e., across 8 presentations of the stimulus)
            firing_image = np.mean(firing_rate_stimulus,axis=0)

            # Store into a list the firing rate of each picture (in the same order as word2vec corr)
            fr_pictures.append(firing_image)
        
        except:

            a=1
        
        

    try:
        # Transform firing rate (Hz) into z-score
        fr_pictures = stats.zscore(np.array(fr_pictures),axis=0)
    
        # Initialize lists with zeros
        corr_fr = np.zeros((fr_pictures.shape[0],fr_pictures.shape[0]))
        cosine_similarity = np.zeros((fr_pictures.shape[0],fr_pictures.shape[0]))


        
        # Loop over each pair of stimuli
        for i in range(fr_pictures.shape[0]):
            for j in range(fr_pictures.shape[0]):
                
                
                corr_fr[i,j] = stats.pearsonr(fr_pictures[i],fr_pictures[j]).statistic
                # Here we compute the cosine distance so need to do 1 - cosine_distance
                # But it is the cosine similarity
                cosine_similarity[i,j] = 1 - cosine(fr_pictures[i],fr_pictures[j])
    except:
        a=1

    # Keep only lower triangle         
    corr_low_triangle = np.tril(cosine_similarity)

    # Keep only lower triangle for cosine similarity
    cosine_similarity_low = cosine_similarity[corr_low_triangle !=0]

    return cosine_similarity,cosine_similarity_low,fr_pictures, is_responsive



def reorder_fr_category(id_patient,id_session,firingrate_og,name_image):
    '''
    This function is used to re-order the firing rates of all trials by semantic category


    Parameters
    -------------
    id_patient : string
        ID of the patient (format asked : 'sub-118' for example)

    id_session : int
        Numero of the session you want to analyze
    
    firingrate_og : array-like
        Array-like with shape Ntrials x Nsamples
    
    name_image : array of string
        Array of the name of all images displayed during the experiment
    
    Output
    -------------
    fr_ordered_cat : array-like
        Array_like of the firing rates reordered by semantic category
    
    length_category : array
        Array containing the number of stimuli per semantic category
    
    '''
    # Path where you create your database with semantic conditions
    path_semantic_vector = 'C:/Users/dornier/GitHub/ConceptCells_TP/Semantic_Coding/database/'+id_patient+'/sess-'+str(id_session)+'/'

    # Load the semantic vector into a dataframe
    semantic_vector = pd.read_csv(path_semantic_vector+'Semantic_Category_Stimuli.csv',header=0,sep=';')


    # Get the category associated with each image
    category_image = list()
    for iImage in range(len(name_image)):
        test = semantic_vector.index[semantic_vector['namefile'] == name_image[iImage]]

        cat = semantic_vector['semantic'][test[0]]

        category_image.append(cat)

    category_image = np.array(category_image)

    # Get a list of unique semantic category
    category = semantic_vector['semantic'].values

    category_unique = np.unique(category)

    fr_ordered_cat = np.empty((0,200))
    
    # Initialize index of beginning and end of each category
    idx_debut = list()
    idx_fin = list()

    # Set beginning of plot to zero
    idx_legend = 0
    for iCategory in category_unique:
        idx_cat = np.where(category_image==iCategory)


        fr_ordered_cat = np.vstack((fr_ordered_cat,firingrate_og[idx_cat]))
        idx_debut.append(idx_legend)
        idx_legend = idx_legend + len(idx_cat[0])
        idx_fin.append(idx_legend)
        
    idx_fin = np.array(idx_fin)-0.5
    idx_debut = np.array(idx_debut)


    return fr_ordered_cat, idx_debut,idx_fin,category_unique
        


#####################
#### MAIN SCRIPT ####
#####################

# Inputs
ROI = 'GPH' # Or TP, Hipp, PCC
plot=1 # Put = 0 if don't want to plot 

# Open the dictionary containing all single-units registered in ROI
path_json = 'C:/Users/dornier/GitHub/ConceptCells_TP/Semantic_Coding/dictionary_singleunits_parahippocampal_semanticcoding.json'
with open(path_json, "r") as f:
    neurons_tp = json.load(f)


# Initialize list to store stats results

d = []

# Extract list of patients from dictionary
list_patient = list(neurons_tp.keys())

# Loop over all patients with units
for patient in list_patient:

    id_patient = patient # ID of the patient

    # Extract sessions from the dictionnary
    list_session = list(neurons_tp[patient][0].keys())
    

    
    # Loop over all the session performed by the patient
    for session in list_session:

        id_session = session # Number of the session

        # Extract list of neurons
        list_neuron = neurons_tp[id_patient][0][id_session]

        start_time = time.time()
            

        # Get the semantic similarity between our concepts presented during the session
        semantic_correlation, label_ordered, corr_w2v_lowdiagonal = word2vec_correlation(id_patient,id_session,model)


        # Loop over the neurons in the session of interest
        for iNeurone in list_neuron:


            # Get the cosine similarity of firing rate for each pair of picture and the firing rate of each pictures
            cosine_distance, cosine_distance_lowdiagonal,firingrate,responsive_stimuli = corr_firingrate(id_patient,id_session,iNeurone)

            # Extract index where responsive stimuli
            idx_response = np.argwhere(responsive_stimuli)

            # If more than one responsive stimulus do the analyses otherwise skip because no sense
            if len(idx_response) > 1:

                # Extract rows (so output with shape nResponse x nAllTrials)
                cosine_intermed = np.squeeze(cosine_distance[idx_response])

                # Extract columns now - so shape nResponse x nResponse
                cosine_responsive = np.squeeze(cosine_intermed[:,idx_response])

                # Extract only low diagonal
                corr_low_triangle_responsive = np.tril(cosine_responsive)

                # Suppress the zeros from above diagonal
                cosine_similarity_low_responsive = cosine_responsive[corr_low_triangle_responsive !=0]
                

                # Extract w2vec rows (so output with shape nResponse x nAllTrials)
                cosine_intermed_w2vec = np.squeeze(semantic_correlation[idx_response])

                # Extract columns now - so shape nResponse x nResponse
                cosine_responsive_w2vec = np.squeeze(cosine_intermed_w2vec[:,idx_response])

                # Suppress the zeros from above diagonal
                cosine_low_responsive_w2vec = cosine_responsive_w2vec[corr_low_triangle_responsive !=0]


                # Extract only trials responsive in firing rate
                firingrate_responsive = np.squeeze(firingrate[idx_response])
                

                # From the firing rate of each picture we will shuffle the order of trials

                # Initialize the list to store permuted matrice
                cosine_shuffled = list()

                # Loop over the number of permutations wanted
                for i in range(1000):

                    # Copy the original matrice that we will shuffle
                    shuffled_firingrate = np.copy(firingrate_responsive)

                    # We shuffled it
                    np.random.shuffle(shuffled_firingrate)

                    # Initialize to get the cosine similarity between all trials
                    cosine_similarity_fr = np.zeros((shuffled_firingrate.shape[0],shuffled_firingrate.shape[0]))

            
                    # Loop over all pairs of trials
                    for i in range(shuffled_firingrate.shape[0]):
                        for j in range(shuffled_firingrate.shape[0]):
                            
                            # Here we compute the cosine distance so need to do 1 - cosine_distance
                            # But it is the cosine similarity
                            cosine_similarity_fr[i,j] = 1 - cosine(shuffled_firingrate[i],shuffled_firingrate[j])


                    # We want to perform RSA only on low diagonal of the matrix
                    # Replace by zeros what's above diagonal
                    corr_low_triangle = np.tril(cosine_similarity_fr)

                    # Suppress the zeros from above diagonal
                    cosine_similarity_low = cosine_similarity_fr[corr_low_triangle !=0]


                    
                    # Store the shuffled trials into a list - shape nPermutations x nTrials x nSamples
                    cosine_shuffled.append(cosine_similarity_low)
                

                
                
                # Compute the cosine similarity between Word2Vec and neuronal activity

                # Handle errors and replace stored value with nan if needed
                try:
                
                    # RSA between Word2Vec and original cosine distance of FRs
                    r_original = stats.spearmanr(cosine_low_responsive_w2vec,cosine_similarity_low_responsive)

                    # RSA between Word2Vec and all shuffled cosine similarities matrix
                    r_shuffled = [stats.spearmanr(cosine_low_responsive_w2vec,cosine_shuffled[i]).statistic for i in range(len(cosine_shuffled))]

                    # Get the value corresponding to the 95th percentile
                    percentile_threshold = np.percentile(r_shuffled,95)

                    # Transform into a z-score the r - original
                    z_original = (r_original.statistic - np.mean(r_shuffled)) / np.std(r_shuffled)
                    
                    d.append(
                        {
                            'ID Patient': id_patient,
                            'ID Session': id_session,
                            'ID Neurone': iNeurone,                        
                            'r value -original': r_original.statistic,
                            'p-value-cosine': r_original.pvalue,
                            'r values - shuffle': r_shuffled,
                            '95th percentile': percentile_threshold,
                            'Z-score': z_original
                        }
                    )

                # If error appears then store nan to the neuron corresponding
                except:
                    d.append(
                        {
                        'ID Patient': id_patient,
                            'ID Session': id_session,
                            'ID Neurone': iNeurone,                        
                            'r value -original': np.nan,
                            'p-value-cosine': np.nan,
                            'r values - shuffle': np.nan,
                            '95th percentile': np.nan,
                            'Z-score': np.nan

                        }
                    )


                # Plot the correlation of word2vec and correlation of the firing rate
                if plot == 1:

                    try:

                        fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(12,5),layout='constrained')

                        pos1 = ax1.imshow(cosine_responsive,cmap='GnBu',vmin=-1,vmax= 1,aspect='auto')
                        ax1.set_title('FR correlation',pad=30,fontsize=22)



                        fig.colorbar(pos1,ax=ax1)



                        pos2 = ax2.imshow(cosine_responsive_w2vec,cmap='GnBu',vmin=-1,vmax=1,aspect='auto')
                        ax2.set_title('Word2Vec',pad=30,fontsize=22)

                        fig.colorbar(pos2,ax=ax2)


                        ax3.hist(r_shuffled,50,color='grey',alpha=0.6)
                        ax3.axvline(r_original.statistic,color='palevioletred',linewidth=3)
                        
                        if r_original.statistic > percentile_threshold:
                            ax3.set_title('*')
                        else:
                            ax3.set_title('n.s.')



                        fig.get_layout_engine().set(wspace=0.2)

                        plt.suptitle("r="+str(r_original.statistic)+"p="+str(r_original.pvalue))

                    




                        plt.savefig('C:/Users/dornier/PhD/Article/Concept_Cells_temporal_pole/Figures/Figure3/Correlation_Ordered/V1_Responsive/gph/Corr_W2V_Cosine_'+
                                    str(id_patient)+'_'+str(id_session)+'_neuron-'+str(iNeurone)+'.svg')
                        
                        plt.savefig('C:/Users/dornier/PhD/Article/Concept_Cells_temporal_pole/Figures/Figure3/Correlation_Ordered/V1_Responsive/gph/AAA_Corr_W2V_Cosine_'+
                                    str(id_patient)+'_'+str(id_session)+'_neuron-'+str(iNeurone)+'.jpg')

                        plt.close()
                    
                    except:

                        a=1
            else:
                d.append(
                    {
                        'ID Patient': id_patient,
                        'ID Session': id_session,
                        'ID Neurone': iNeurone,                        
                        'r value -original': np.nan,
                        'p-value-cosine': np.nan,
                        'r values - shuffle': np.nan,
                        '95th percentile': np.nan,
                        'Z-score': np.nan

                    }
                )

            


            



            

            



# Save the data from our entire database
df = pd.DataFrame(d)
df.to_csv('C:/Users/dornier/PhD/Article/Concept_Cells_temporal_pole/Figures/Figure3/Correlation_Ordered/Correlation_Cosine_word2vec_AllDatabase_v1_Responsive_'+ROI+'.csv')


print("All done")