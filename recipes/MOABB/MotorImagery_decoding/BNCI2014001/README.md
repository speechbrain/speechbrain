# Motor imagery decoding from single EEG trials using BNCI2014001 dataset (MOABB benchmark)
# Task description
todo
# How to run
Before running an experiment, make sure the extra-dependencies reported in the file `extra_requirements.txt` are installed in your environment.
Note that this code requires mne==0.20.7.

Perform training (including within-session, cross-session, leave-one-subject-out iterations) with: \
\>>> python train.py train.yaml --data_folder '/path/to/MOABB_BNCI2014001'

# Results


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
Please refer also to ....
