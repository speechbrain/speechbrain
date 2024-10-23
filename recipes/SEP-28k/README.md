# Stuttering Event Detection experiments with SEP-28k (speech only)
This folder contains scripts for running stuttering event detection experiments with the SEP-28k dataset (https://github.com/apple/ml-stuttering-events-dataset). The partitioning follows the suggestion of SEP-28k-E (https://github.com/th-nuernberg/ml-stuttering-events-dataset-extended).

# Training
Run the following command to train the model:
`python train.py hparams/fluency.yaml`

Note that this is a minimal working example. The model and training parameters should be modified accordingly.

# Note on Data Preparation

Our `sep28k_prepare.py` will:
1. Download the dataset (including deleted podcasts impossible to download from the original script).
2. Prepare train/valid/test with partitioning suggested by https://github.com/th-nuernberg/ml-stuttering-events-dataset-extended. By default, it follows the "SEP-28k-E" partitioning.

# **About SEP-28k and SEP-28k-E**

```bibtex
@misc{lea:2021,
    author       = {Colin Lea AND Vikramjit Mitra AND Aparna Joshi AND Sachin Kajarekar AND Jeffrey P. Bigham},
    title        = {{SEP-28k}: A Dataset for Stuttering Event Detection from Podcasts with People Who Stutter},
    howpublished = {ICASSP 2021},
}
```
```bibtex
@incollection{bayerl_sep28k_E_2022,
	title = {The {Influence} of {Dataset-Partitioning} on {Dysfluency} {Detection} {Systems}},
	booktitle = {Text, {Speech}, and {Dialogue}},
	author = {Bayerl, Sebastian P. and Wagner, Dominik and Bocklet, Tobias and Riedhammer, Korbinian},
	year = {2022},
}
```

# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/

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

