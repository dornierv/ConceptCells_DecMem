'''
Author : Vincent Dornier
Made the 27/08/2025


In this script we will compute decoding approach.
We will decode the picture presented to the patient based on the firing activity

Runtime: 6 minutes per session approximately
'''
import numpy as np
import pandas as pd
import pynapple as nap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec # affichage rasters
import statsmodels.stats.multitest as sm
import statsmodels.discrete.discrete_model as stats_model
import math
from scipy import io as io
from scipy import stats as stats
import os
from mne.decoding import (SlidingEstimator, LinearModel)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV, BayesianRidge, Perceptron, PoissonRegressor, RidgeClassifierCV, SGDClassifier, SGDRegressor, LassoCV
from sklearn.metrics import make_scorer, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from mne import decoding
import warnings
import mne.stats as mne_stats
import matplotlib.spines as spines
import json

from mne.decoding import (
    CSP,
    GeneralizingEstimator,
    LinearModel,
    Scaler,
    SlidingEstimator,
    Vectorizer,
    cross_val_multiscore,
    get_coef,
)

# Inputs
path2data = 'C:/Users/dornier/GitHub/ConceptCells_DecMem/'
path2figure = 'C:/Users/dornier/GitHub/ConceptCells_DecMem/data/Decoding/'

###############################
########## FUNCTIONS ##########
###############################


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
    Load spikes from a .nwb file (https://alleninstitute.github.io/nwb-api/)
    I personnaly used NeuroConv to convert Kluster file into nwb file (https://github.com/catalystneuro/neuroconv)
    Here I open nwb file with Pynapple (see https://pynapple.org/)

    Parameters
    -----------------
    data_path : string
        Path where the nwb file is store

    Output
    -----------------
    spikes : structure
        Structure containing all informations that we will be used later with pynapple
    '''
    # Load the folder where nwb file is store with pynapple
    data = nap.load_folder(data_path)
    
    # Get the file of interest
    nwb = data["spikes"]["saved_file"]

    # Get the informations about the neurons identified
    spikes = nwb["units"]


    # Here for sub-118 as spike sorting made with SC2 nwb file has different structure
    try:
        # Here try to open only good units identified with SC2 (in SC1 no label 'quality')
        spikes = spikes[spikes['quality']=='good']
    except:
        print("Made with SC1")


    # Return the structure to be loaded in pynapple
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


def get_epochs(path,subject,session,neurons,tmin,tmax,sr=32768):
    '''
    This function will create the array-like containing firing rate of all neurons for all pictures.

    Parameters
    ---------------
    path : string
        Path where your database is stored
    
    subject : string
        ID of the subject to analyze
    
    session : string
        ID of the session to analyze
    
    neurons : list, array-like
        List of the number of the neurons in the nwb file
    
    tmin : int
        Beginning of the epoch compare to onset of stimuli (in seconds) 
        e.g. if tmin = 0 then take from the onset of picture, if tmin = -1 take one second before onset
    
    tmax : int
        Same as tmin but for the end of the epoch compare to onset of stimuli

    sr : int
        Sampling rate of your acquisition system, Default = 32768 because sampling rate of neuralynx system
    
    Output
    ---------------
    epoch : array-like
        Matrix containing firing rate of all neurons for all trials 
        Shape : M x N x P with M = number of trials, N = number of neurons and P = number of samples
    
    '''
    # Pynapple produce a lot of warning when computing the firing rate we don't want them to be print
    warnings.filterwarnings('ignore')

    path_data = path+subject+'/'+session+'/'

    # Path where data is stored
    data_path = path_data+"lfps/"
    path_runList = path_data+"stimuli/"

    # Load spike information from the nwb file
    spikes = load_spikes(path_data)

    # Load Run Lists and get TTL associated with each image
    test_mat_Run1 = io.loadmat(path_runList+'run-01.mat') # et concaténer les 4 runs
    my_array_Run1 = test_mat_Run1['trial']
    keys_TTL = [my_array_Run1[0,i][1][0][0] for i in range(my_array_Run1.shape[1])] 
    values_images = [my_array_Run1[0,i][2][0] for i in range(my_array_Run1.shape[1])] 
    dict_TTL2Image = {k: v for k, v in zip(keys_TTL, values_images)}


    # Load TTLs & dat files
    folder_TTL = path_data+'ttl/'
    TTLvals = io.loadmat(folder_TTL+subject+'_TTLvals_tot.mat')['TTLvals_tot'][0]
    TS = io.loadmat(folder_TTL+subject+'_TS_tot.mat')['TS_tot'][0]


    # Load timestamps from neuralynx acquisition system
    tsname=data_path+'/'+subject+'_ses-01_task-SCREENING_run-01_timestamps.dat'
    TS_stream=np.memmap(tsname, mode='r', dtype=float, order='F')

    # Get TTLs sync with EEG
    TS_32768 = np.searchsorted(TS_stream, TS, side='left') # index temps absolu
    TS_32768[TS_32768 ==len(TS_stream)] = len(TS_stream)-1

    

    # Get only TTL associated with pictures, sometimes more TTL than pictures (e.g. TTL send during plugging)
    TTL_picture = dict_TTL2Image.keys()

    # Get indices where TTL are from picture
    imgIndices3 = np.sort(np.hstack([np.where(TTLvals==img_ttl)[0] for img_ttl in TTL_picture]))
    
    # Get the timestamps associated with onset of pictures
    imgTS=[TS_32768[imgIndice] for imgIndice in imgIndices3]


    ####################################################################
    # Prepare the matrix epoch to get the shape desired for the output #
    ####################################################################

    # Determine the number of samples in your epoch
    nSamples = int((tmax - tmin) * 100)

    # Create matrix with shape :  Mtrials x Nneurons x Psamples
    epoch = np.zeros((len(imgIndices3),len(neurons),nSamples))

    
    ################################################################
    # Now we fill the matrix with the firing rates of each stimuli #
    ################################################################


    # Initialize neurone count to fill the matrix
    idx_neuron = 0

    # Loop over all neurons of interest in the session
    for iNeuron in neurons:

        # Loop over all trials
        for iTrials in range(len(imgIndices3)):
            
            
                

            # Get the time in seconds so it runs in Pynapple
            time_fr = (imgTS[iTrials])/sr


            # Create pynapple structure
            Trial_fr = nap.Ts(time_fr)
            
            # Compute perievent 
            Trial_perievent = nap.compute_perievent(spikes[iNeuron], Trial_fr, minmax=(tmin,tmax))


            # Get the firing rate
            firing_rate = get_firingrate(Trial_perievent)
            

            # Store the firing rate of the trial in the epoch matrix
            epoch[iTrials][idx_neuron] = firing_rate[0]
            

        
        # When looping over all trials get to the next neuron
        idx_neuron+=1

    

    return epoch


def create_image_vector(path,subject,session):
    '''
    Create an array containing the labels of all the pictures presented during the session
    This label will be used lated to decode the stimuli presented

    Parameters
    --------------
    path : string
        Path where your database is stored

    subject : string
        ID of the subject you want to analyze
    
    session : string
        ID of the session to analyze

    Output
    ------------
    image_vector : array-like
        Array containing the name of all pictures presented
        Shape : nTrials 
    '''

    path_session = path+subject+'/'+session


    # Load Run Lists and get TTL associated with each image
    test_mat_Run1 = io.loadmat(path_session+'/stimuli/run-01.mat') # et concaténer les 4 runs
    my_array_Run1 = test_mat_Run1['trial']
    keys_TTL = [my_array_Run1[0,i][1][0][0] for i in range(my_array_Run1.shape[1])] 
    values_images = [my_array_Run1[0,i][2][0] for i in range(my_array_Run1.shape[1])]
    dict_TTL2Image = {k: v for k, v in zip(keys_TTL, values_images)}


    # Load TTLs & dat files
    folder_TTL = path_session+'/ttl/'
    TTLvals = io.loadmat(folder_TTL+subject+'_TTLvals_tot.mat')['TTLvals_tot'][0]


    idx_image = np.where(TTLvals !=0)

    TTLimage = TTLvals[idx_image]


    # Initialize the output
    image_vector = []


    # Loop over all trials
    for iTrials in range(len(TTLimage)):
        
        try:
            image_vector.append(dict_TTL2Image[TTLimage[iTrials]])
        except:
            a=1


    # Transform list into array
    image_vector = np.array(image_vector)

    return image_vector

            
################################
############# MAIN #############
################################


# Input
path_database = path2data+'/data/Examples_Session/' # Change path2folder by the path where is your unzipped folder

path_json = path_database+'dictionary_singleunits_example.json'

path_results = path2figure # Replace path2save by the path where you want to store results


# Check whether the specified path exists or not
isExist = os.path.exists(path_results)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(path_results)

with open(path_json, "r") as f:
    neurons_tp = json.load(f)

# Initialize list to store stats results
d = []

# Extract list of patients from dictionary
list_patient = list(neurons_tp.keys())


# Loop over all patients included
for patient in list_patient:

    id_patient = patient # ID of the patient

    # Extract sessions from the dictionnary
    list_session = list(neurons_tp[patient][0].keys())
    

    
    # Loop over all sessions performed by the patient
    for session in list_session:

        # Extract list of neurons
        list_neurons = neurons_tp[patient][0][session]
        

        # Get the matrix of firing rate
        epoch = get_epochs(path_database,patient,session, list_neurons,-0.5,2)

        
        # Get the label vector
        vector_label = create_image_vector(path_database,patient,session)



        # Transform name of pictures (string) into int
        le = LabelEncoder()
        le.fit(vector_label)
        y = le.transform(vector_label)


        # Determine the number of presentation (if less than 8 we still want to do cross-validation but not on 7-fold)
        nb_pres = len(np.where(y==np.max(y))[0])

        # We use Logistic Regression normally for binary classification but solver='lbfgs' handle multi-class problem
        # See here for documentation https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        clf = make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs"))


        time_decod = SlidingEstimator(clf, n_jobs=None, scoring="accuracy", verbose=True)

        scores = cross_val_multiscore(time_decod, epoch, y, cv=nb_pres-1, n_jobs=None)

        # Run permutation t-test on score to see if scores is significant
        random = 1/(np.max(y)+1)

        t_obs, p_values,H0 = mne_stats.permutation_t_test(scores-random)

        

        # Mean scores across cross-validation splits
        scores_m = np.mean(scores, axis=0)
        sem= stats.sem(scores,axis=0)

        # Plot
        fig, ax = plt.subplots()
        times = np.linspace(-0.5,2,num=250)
        ax.plot(times, scores_m, label="score",color='cornflowerblue')
        ax.fill_between(times,scores_m+sem,scores_m-sem,color='cornflowerblue',alpha=0.4)

        ax.set_xlabel("Time (s)",fontsize=14)
        ax.set_ylabel("Decoder accuracy (%)",fontsize=14)  # Area Under the Curve
        ax.legend()
        ax.axvline(0.0, color="k", linestyle="-")




        # Get the random value by dividind 1 by the number of pictures presented
        
        plt.axhline(random, color='k', linestyle='--')
        plt.text(x=-0.8,y=random,s='Random',color='k')

        idx_sample_sig = np.where(p_values < 0.05)
        plt.plot(times[idx_sample_sig[0]],np.zeros(len(idx_sample_sig[0])),linestyle=' ',marker='o',color='k',alpha=0.1)
        
        plt.suptitle('Decoding image with firing rate')
        plt.title('Number neurons : '+str(epoch.shape[1]))
        

        plt.savefig(path_results+'Decoding_score_'+patient+'_'+session+'_2sec.svg')
        plt.savefig(path_results+'Decoding_score_'+patient+'_'+session+'_2sec.jpg')

        plt.close()


        


        # Store results in a dictionary to save later in a csv file
        # Append the values of interest
        d.append(
            {
                'ID Patient': patient,
                'ID Session': session,
                'Nombre neurones': epoch.shape[1],
                'Nombre images': np.max(y)+1,
                'Decoding max (%)': scores_m.max()*100,
                'Time peak decoding (ms)': times[np.argmax(scores_m)]*1000
            }
        )

        


# Save the data from our entire database
df = pd.DataFrame(d)
df.to_csv(path_results+'Performance_Decoding_All_Database_2sec.csv')


print("All done")
