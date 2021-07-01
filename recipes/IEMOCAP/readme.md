# Emotion recognition experiments with IEMOCAP.
This folder contains scripts for running emotion recognition experiments with the IEMOCAP dataset (https://sail.usc.edu/iemocap/).
Get the IEMOCAP dataset from https://sail.usc.edu/iemocap/iemocap_release.htm and put it in the same folder as `iemocap_prepare.py` under the name `IEMOCAP_processed.tar.gz`.

# Training ECAPA-TDNN
Run the following command to train the model:
`python train.py hparams/train.yaml`

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

