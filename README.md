# Concept Cells - Declarative Memory

This repository contains code accompanying the following paper:


**A distributed and hierarchical organization of concept cells in the human declarative memory system**

Authors: Vincent DORNIER (1)*, Leila REDDY (1), Aube DARVES-BORNOZ (1), Adrien A. CAUSSE (2), Zoé DARRASSE (1), Luc VALTON (1,3), Marie DENUELLE (1,3), Jean-Albert LOTTERIE (4,5), Amaury DE BARROS (4,5), Annabelle GOUJON (6), Jonathan CUROT (1,3) & Emmanuel J. BARBEAU (1)


Affiliations: 
1. University of Toulouse, CNRS, CerCo, Toulouse, France.
2. Medical Research Council Brain Network Dynamics Unit, Nuffield Department of Clinical Neurosciences, University of Oxford, Oxford, United Kingdom;
Medical Research Council Centre of Research Excellence in Restorative Neural Dynamics, United Kingdom.
3. Brain Electrophysiology, Epilepsy and Sleep Unit, Neurology Department, Toulouse University Hospital, Toulouse, France.
4. University of Toulouse, INSERM, ToNiC, Toulouse, France.
5. Department of Neurology and Neurosurgery, Toulouse University Hospital, Toulouse, France.
6. Marie & Louis Pasteur University, UMR 1322 INSERM, LINC, Besançon, France.


If you have questions please contact: vincent.dornier@cnrs.fr


## Requirements

- Python 3 (3.12)
- Numpy (1.26.2)
- Scipy (1.13.1)
- Matplotlib (3.10.0)
- Pandas (2.2.3)
- Pynapple (0.8.5)
- MNE (1.9.0)
- Gensim (4.3.3)

We recommend creating a new environment (e.g., decmem) with conda by typing for example in the anaconda prompt:

```
conda create -n decmem python=3.12.0
```

Then you can install each specific version of the library with pip (with numpy here as an example):
```
pip install numpy==1.26.2
```


## Code usage

To reproduce analyses performed in the manuscript please download the folder as a ZIP file on your computer.
Then unzip the file and launch script from the folder scripts with Python.

In the script you want to launch replace the path at the beginning by the path where you stored the folder unzipped.

### Main function


#### Description neuronal pattern region

- **Proportion_Responsiveness.py**: Get the proportion of stimuli that elicited a response for a given region (Fig. 2c and Fig. 4e)
- **Percentage_Neurons_Responsive.py**: Get the proportion of neurons in a given region that significantly respond to at least one stimulus (Fig. 2b and Fig. 4d)


#### Semantic coding

- **Semantic_Coding_Processing.py"**: Compute RSA between neuronal activity and embedding of concept from Word2vec (Fig. 3e, f, h and Fig. 5f)
- **Notebook_SemanticCoding_All_ROI.ipynb**: Compare semantic coding across regions included (Fig. 5f)

#### Comparisons across regions

- **Time_Course_Neurons_ROI.py**: Mean firing rate across time for all regions included (Fig. 5a)
- **Stimulus_Repetition_Effect.py**: Effect of stimulus repetition on the firing activity (Fig. 4g & Extended Data Fig. 5h)
- **Spike_Train_Reliability.py**: Permutation entropy to determine spiking variability across regions included (Extended Data Fig. 7a)
- **Selectivity_Measure.py**: Compute selectivity of units recorded in all regions included (Fig. 5e)
- **Onset_Latency.py**: Get the onset latencies of neuronal response after the onset of stimuli (Fig. 5b)




