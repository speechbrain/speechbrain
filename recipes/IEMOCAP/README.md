# Emotion recognition experiments with IEMOCAP (speech only)
This folder contains scripts for running emotion recognition experiments with the IEMOCAP dataset (https://sail.usc.edu/iemocap/).
Get the IEMOCAP dataset from https://sail.usc.edu/iemocap/iemocap_release.htm and put it in the same folder as `iemocap_prepare.py` under the name `IEMOCAP_processed.tar.gz`.

# Training ECAPA-TDNN
Run the following command to train the model:
`python train.py hparams/train.yaml`

# Results
| Release | hyperparams file | Val. Acc. | Test Acc. | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| -----:| --------:| :-----------:|
| 2021-07-04 | train.yaml |  65.3 | 65.7 | https://drive.google.com/drive/folders/1U9SiO4KkCNBKfxilXzJqBZ_k-vHz4ltV?usp=sharing | 1xV100 16GB |
| 2021-09-12 | train_with_wav2vec2.yaml |  best 77.8 | best: 79.1 (avg 77.02) | Gdrive download | 1xV100 32GB |
# Training Time
About 40 sec for each epoch with a TESLA V100 (with ECAPA-TDNN).
About 2min 50 sec for each epoch with a TESLA V100 (with wav2vec2 encoder).

# Note on Data Preparation
We here use only the audio part of the dataset. The assumpion is that the data folder is structured as:

    ```<session_id>/<emotion>/<file:name>.wav```

e.g. ```session1/ang/psno1_ang_s084_orgn.wav```

Please, process the original IEMOCAP folder to match the expected folder structure.


# **About IEMOCAP**

```bibtex
@article{Busso2008IEMOCAPIE,
  title={IEMOCAP: interactive emotional dyadic motion capture database},
  author={C. Busso and M. Bulut and Chi-Chun Lee and Ebrahim Kazemzadeh and Emily Mower Provost and Samuel Kim and J. N. Chang and Sungbok Lee and Shrikanth S. Narayanan},
  journal={Language Resources and Evaluation},
  year={2008},
  volume={42},
  pages={335-359}
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

