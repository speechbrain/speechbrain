# Single EEG trial decoding using MOABB datasets
# Task description
The objective decoding task is to classify different brain states (e.g., different motor imagery conditions, event-related potential components, etc.) using electroencephalographic (EEG) signals at single-trial level belonging to Mother Of All BCI Benchmark (MOABB) datasets.

# How to run
Before running an experiment, make sure the extra-dependencies reported in the file `extra_requirements.txt` are installed in your environment.

In addition to network and optimization hyper-parameters, define in the hparams file (yaml file in hparams folder) a target MOABB dataset to decode (see datasets at http://moabb.neurotechx.com/docs/datasets.html) and pre-processing hyper-parameters for that specific dataset, defining the paradigm object of MOABB (see paradigms at http://moabb.neurotechx.com/docs/paradigms.html).

Train a neural networks to decode single EEG trials using different training strategies: leave-one-session-out, leave-one-subject-out.

These training strategies are defined as follows:
* Leave-one-session-out (within-subject, cross-session and session-agnostic decoders).
    For each subject, one session was held back as test set and the remaining ones were used to train neural networks.

* Leave-one-subject-out (cross-subject, cross-session and subject-agnostic decoders).
    One subject was held back as test set and the remaining ones were used to train neural networks.

E.g., to train EEGNet to decode motor imagery on BNCI2014001:\
\>>> python train.py hparams/EEGNet_BNCI2014001.yaml --data_folder '/path/to/BNCI2014001'

The dataset will be automaticallt downloaded in the specified folder.

# Notes on MNE
Moab depends on MNE. By default the latter store a config file in the home at `$HOME/.mne/mne-python.json` and downloads data to `$HOME/mne-data`. In some cases, the home does not exist, it might have storage limitations, or it might be in a shared filesystem where data reading and writing operations might be discouraged by the system admim. If you want to set up a different folder:
1- `export _MNE_FAKE_HOME_DIR='your/folder'`  (in your bash shell)
2- Go to python and type `import mne`. This will create `your/folder/.mne/mne-python.json`.
3- Open this file and set `MNE_DATA` and `MNE_DATASETS_BNCI_PATH` with the folders you prefer.


# Results
After training, you can aggregate and visualize the performance of all the experiments with:
\>>> python parse_results.py results/MOABB/EEGNet_BNCI2014001/ test_metrics.pkl acc f1

To see the results on the validation set use:
\>>> python parse_results.py results/MOABB/EEGNet_BNCI2014001/ test_metrics.pkl acc f1

In the following some results are provided, obtained with a leave-one-session-out strategy.
Here, performance metrics were computed on each held-out session (stored in the metrics.pkl file). Then, these metrics were reported for each held-out session (average value ± standard deviation across subjects).

| Release | Task | Hyperparams file | Training strategy | Session | Key loaded model | Performance (test set) |  GPUs | RAM
|:-------------:|:-------------:|:---------------------------:|:---------------------------:|  -----:|-----:| -----:| :-----------:| -----:|
| 23-07-31 | Motor imagery | EEGNet_BNCI2014001.yaml | leave-one-session-out | session_E | 'acc'| 0.7465±0.0660 | 1xNVIDIA A100 (40 GB) | 12 GB |
| 23-07-31 | Motor imagery | EEGNet_BNCI2014001.yaml | leave-one-session-out | session_T | 'acc'| 0.7585±0.0710 | 1xNVIDIA A100 (40 GB) | 12 GB |
| 23-07-31 | P300 | EEGNet_EPFLP300.yaml | leave-one-session-out | session_1 | 'f1'| 0.6332±0.1146 | 1xNVIDIA A100 (40 GB) | 12 GB |
| 23-07-31 | P300 | EEGNet_EPFLP300.yaml | leave-one-session-out | session_2 | 'f1'| 0.6566±0.0944 | 1xNVIDIA A100 (40 GB) | 12 GB |
| 23-07-31 | P300 | EEGNet_EPFLP300.yaml | leave-one-session-out | session_3 | 'f1'| 0.6600±0.1242 | 1xNVIDIA A100 (40 GB) | 12 GB|
| 23-07-31 | P300 | EEGNet_EPFLP300.yaml | leave-one-session-out | session_4 | 'f1'| 0.6526±0.1218 | 1xNVIDIA A100 (40 GB) | 12 GB|
| 23-07-31 | SSVEP | EEGNet_Lee2019_SSVEP.yaml | leave-one-session-out | session_1 | 'acc'| 0.9370±0.1170 | 1xNVIDIA A100 (40 GB) | 12 GB|
| 23-07-31 | SSVEP | EEGNet_Lee2019_SSVEP.yaml | leave-one-session-out | session_2 | 'acc'| 0.9287±0.1157 | 1xNVIDIA A100 (40 GB) | 12 GB|


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
Please also refer to the reference paper of MOABB benchmark dataset.
```bibtex
@article{moabb,
	doi = {10.1088/1741-2552/aadea0},
	url = {https://doi.org/10.1088/1741-2552/aadea0},
	year = 2018,
	month = {sep},
	publisher = {{IOP} Publishing},
	volume = {15},
	number = {6},
	pages = {066011},
	author = {Vinay Jayaram and Alexandre Barachant},
	title = {{MOABB}: trustworthy algorithm benchmarking for {BCIs}},
	journal = {Journal of Neural Engineering},}
```
