# Single EEG trial decoding using MOABB datasets
# Task description
The objective decoding task is to classify different brain states (e.g., different motor imagery conditions, event-related potential components, etc.) using electroencephalographic (EEG) signals at single-trial level belonging to Mother Of All BCI Benchmark (MOABB) datasets.

# How to run
Before running an experiment, make sure the extra-dependencies reported in the file `extra_requirements.txt` are installed in your environment.

Define a target MOABB dataset to decode (see datasets at http://moabb.neurotechx.com/docs/datasets.html) and pre-processing hyper-parameters for that specific dataset, defining the paradigm object of MOABB (see paradigms at http://moabb.neurotechx.com/docs/paradigms.html).

Train a neural networks to decode single EEG trials using different training strategies: within-session, cross-session, leave-one-session-out, leave-one-subject-out.
These training strategies were defined as follows:
* Within-session (within-subject and within-session decoders).
    For each subject and for each session, the training and test sets were defined using a stratified cross-validation partitioning.
    
* Leave-one-session-out (within-subject, cross-session and session-agnostic decoders).
    For each subject, one session was held back as test set and the remaining ones were used to train neural networks.
    
* Cross-session (within-subject and cross-session decoders).
    For each subject, all session' signals were merged together.
    Training and test sets were defined using a stratified cross-validation partitioning.

* Leave-one-subject-out (cross-subject, cross-session and subject-agnostic decoders).
    One subject was held back as test set and the remaining ones were used to train neural networks.
    
For all these strategies, the validation set was sampled from the training set using a fixed validation ratio (20%).
All sets were extracted balanced across subjects, sessions and classes.

E.g., to train EEGNet to decode motor imagery on BNCI2014001:\
\>>> python train.py hparams/EEGNet_BNCI2014001.yaml --data_folder '/path/to/BNCI2014001'

The dataset will be automaticallt downloaded in the specified folder.


# Results
After training, you can aggregate and visualize the performance of all the experiments with:
\>>> python parse_results.py results/MOABB/EEGNet_BNCI2014001/1234 acc loss f1


In the following, results are reported for each MOABB dataset and each architecture using different training strategies.

The model scoring the optimal value on the validation set for a target key  was loaded (e.g., max accuracy). Then, results were aggregated as follows:
* Within-session. For each subject and for each session, performance metrics were computed on each test set of each cross-validation fold (stored in the metrics.pkl file). Then, these were averaged across folds. The so averaged metrics were reported for each session (average value ± standard deviation across subjects).
* Leave-one-session-out. For each subject, performance metrics were computed on each held-out session (stored in the metrics.pkl file). Then, these metrics were reported for each held-out session (average value ± standard deviation across subjects).
* Cross-session. For each subject, performance metrics were computed on each test set of each cross-validation fold (stored in the metrics.pkl file). Then, these metrics were reported (average value ± standard deviation across subjects).
* Leave-one-subject-out. Performance metrics were computed on each held-out subject (stored in the metrics.pkl file). Then, these metrics were reported (average value ± standard deviation across subjects).

| Release | Hyperparams file | Training strategy | Session | Key loaded model | Test Accuracy |  GPUs |
|:-------------:|:---------------------------:|:---------------------------:|  -----:|-----:| -----:| :-----------:|
| 21-10-30 | EEGNet_BNCI2014001.yaml | within-session | session_T | 'acc'|62.92±15.43% | 1xTITAN V 12GB |
| 21-10-30 | EEGNet_BNCI2014001.yaml | within-session | session_E | 'acc'|61.46±18.88% | 1xTITAN V 12GB |
| 21-10-30 | EEGNet_BNCI2014001.yaml | leave-one-session-out | session_T | 'acc'|62.89±18.31% | 1xTITAN V 12GB |
| 21-10-30 | EEGNet_BNCI2014001.yaml | leave-one-session-out | session_E | 'acc'|64.04±13.66% | 1xTITAN V 12GB |
| 21-10-30 | EEGNet_BNCI2014001.yaml | leave-one-session-out | session_E | 'loss'|66.13±13.63% | 1xTITAN V 12GB |
| 21-10-30 | EEGNet_BNCI2014001.yaml | cross-session | - | 'acc'|69.06±17.90% | 1xTITAN V 12GB |
| 21-10-30 | EEGNet_BNCI2014001.yaml | leave-one-subject-out | - | 'acc'|37.77±11.65% | 1xTITAN V 12GB |

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
