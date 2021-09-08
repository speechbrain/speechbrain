# P300 decoding from single EEG trials using ERP CORE dataset
# Task description
The P300 is an attention-dependant response occurring when infrequent stimuli are presented to the user immersed into a sequence of more frequent background stimuli. 
This response peaks between 300-500 ms after the infrequent stimulus onset and is mostly distributed on the parietal area. Due to the low signal-to-noise ratio of the electroencephalogram (EEG), P300 only emerges after an averaging procedure of EEG signals across several responses to stimuli (i.e., EEG trials) and across subjects. 
Therefore, the decoding of the P300 event at the level of every single trial is a very challenging task. 
P300 not only is of particular relevance as a control signal to guide Brain-Computer Interfaces (e.g., P300 spellers) but also as a biomarker in psychiatric disorders (e.g., schizophrenia, depression, etc.).

This folder contains the scripts to train a P300 decoder with EEG signals collected in the ERP CORE dataset using a compact convolutional neural network based on EEGNet.

ERP CORE is an open collection of event-related potentials available at: https://osf.io/thsqg/

The objective decoding task is the classification of the absence vs. presence of the P300 event (binary classification) from single EEG trials (i.e., single-trial P300 decoding) using signals from each subject separately. 
This is necessary due to the high subject-to-subject variability in the EEG. 
Therefore, subject-specific decoders are trained and due to the resulting compact dataset (consisting of 200 EEG trials per subject using a 10-fold cross-validation scheme is adopted).

# How to run
Before running an experiment, make sure the extra-dependencies reported in the file `extra_requirements.txt` are installed in your environment.
Note that this code requires mne==0.22.1.

Download the dataset with: \
\>>> python download_required_data.py --data_folder /path/to/ERPCore_P3 

Perform training on a subject (e.g., subject ID 4='sub-004'): \
\>>> python train.py train.yaml --sbj_id 'sub-004' --data_folder '/path/to/ERPCore_P3'

If you want to run the full training on all the subjets, you can loop over all of then with a simple bash script:

```
#!/bin/bash

data_folder=$1
for sub in $data_folder/sub*; do
    sub_id=$(basename $sub)
    echo "processing $sub_id..."
    python train.py train.yaml --sbj_id $sub_id --data_folder $1
done
```


# Results
For each subject-specific decoder and within each fold, AUROCs and F1 scores were computed on the test set. 
These metrics are stored in the pickle file "metrics.pkl" (containing a ndarray with loss, F1 and AUROC for each fold within each row, with this order). 

Performance metrics were averaged across folds (subject-level metrics). 
In the following table, the subject-level metrics are reported (mean ± standard error of the mean across subjects).

| Release | Hyperparams file | Test F1 score | Test AUROCs |  GPUs |
|:-------------:|:---------------------------:| -----:| -----:| :-----------:|
| 21-07-13 | train.yaml |  45.7±2.0% | 73.2±1.8% | 1xTITAN V 12GB |

You can find the full output folder containing the pre-trained models and logs here (sub-004):
https://drive.google.com/drive/folders/1MCy-fvUFwFx9sXU_AFI_-_HZnZv6dZKF?usp=sharing


# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```

# **Additional References**
Please refer also to the study that collected the ERP CORE dataset and proposed EEGNet.
```bibtex
@article{ERPCORE,
title = {ERP CORE: An open resource for human event-related potential research},
author = {Emily S. Kappenman and Jaclyn L. Farrens and Wendy Zhang and Andrew X. Stewart and Steven J. Luck},
journal = {NeuroImage},
volume = {225},
pages = {117465},
year = {2021},
}
```
```bibtex
@article{EEGNet,
title={EEGNet: a compact convolutional neural network for EEG-based brain--computer interfaces},
author={Vernon J. Lawhern and Amelia J. Solon and Nicholas R. Waytowich and Stephen M. Gordon and Chou P. Hung and Brent J. Lance},
journal={Journal of neural engineering},
volume={15},
number={5},
pages={056013},
year={2018},
}
```


