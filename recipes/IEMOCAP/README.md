# Emotion recognition experiments with IEMOCAP (speech only)
This folder contains scripts for running emotion recognition experiments with the IEMOCAP dataset (https://paperswithcode.com/dataset/iemocap).

# Training ECAPA-TDNN/wav2vec 2.0/HuBERT
Run the following command to train the model:
`python train.py hparams/train.yaml`
or with wav2vec2 model:
`python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml`

# Results
The results reported here use random splits
| Release | hyperparams file | Val. Acc. | Test Acc. | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| -----:| --------:| :-----------:|
| 2021-07-04 | train.yaml |  65.3 | 65.7 | [model](https://www.dropbox.com/sh/o72ex46i49qgdm0/AABxsuG0EEqTLgzWwrkYQzu_a?dl=0) | 1xV100 16GB |
| 2021-10-17 | train_with_wav2vec2.yaml (wav2vec2 base) |  best 78.1 | best: 78.7 (avg 77.0) | [model](https://www.dropbox.com/sh/lmebg4li83sgkhg/AACooPKbNlwd-7n5qSJMbc7ya?dl=0) | 1xV100 32GB |
| 2021-10-17 | train_with_wav2vec2.yaml (voxpopuli base) |  best 73.3 | best: 73.3 (avg 70.5) | [model](https://www.dropbox.com/sh/ikjwnwebekf2xx2/AADyaJKPiaR0_iO0nntucH5pa?dl=0) | 1xV100 32GB |
| 2021-10-17 | train_with_wav2vec2.yaml (hubert base) |  best 74.9  | best: 79.1 (avg 76,6) | [model](https://www.dropbox.com/sh/ke4fxiry97z58m8/AACPEOM5bIyxo9HxG2mT9v_aa?dl=0) | 1xV100 32GB |

# Training Time
About 40 sec for each epoch with a TESLA V100 (with ECAPA-TDNN).
About 3min 14 sec for each epoch with a TESLA V100 (with wav2vec2 BASE encoder).

# Note on Data Preparation
We here use only the audio part of the dataset.

Our `iemocap_prepare.py` will:
1. Do labelling transformation to 4 emitions [neural, happy, sad, anger]
2. Prepare IEMOCAP data with random split if different_speakers is False. (Note for becnhmarking: you need to run 5 folds)
3. Prepare IEMOCAP data with speaker-independent split if different_speakers is True. (Note for becnhmarking: you need to run 10 folds with test_spk_id from 1 to 10)


# PreTrained Model + Easy-Inference
You can find the wav2vec2 pre-trained model with an easy-inference function on HuggingFace:
- https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP/


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

