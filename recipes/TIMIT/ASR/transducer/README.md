# TIMIT ASR with Transducer models.
This folder contains the scripts to train an RNNT system using TIMIT.
TIMIT is a speech dataset available from LDC: https://catalog.ldc.upenn.edu/LDC93S1


# Extra-Dependencies
Before running this recipe, make sure numba is installed. Otherwise, run:
```
pip install numba
```

# How to run
Update the path to the dataset in the yaml config file and run the following.
```
python train.py hparams/train.yaml
```

# Results

| Release | hyperparams file | Val. PER | Test PER | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| -----:| --------:| :-----------:|
| 2021-02-06 | train.yaml |  13.11 | 14.12 | https://drive.google.com/drive/folders/1g3T6zK2o9XTEa_GTw0aoAkRqhg1_BVQ3?usp=sharing | 1xRTX6000 24GB |
| 21-04-16 | train_wav2vec2.yaml |  7.97 | 8.91 | https://drive.google.com/drive/folders/1z8Ox3q2ntnnnh3PPk_eOcKhGeFgVeRcD?usp=sharing | 1xRTX6000 24Gb |

# Training Time
About 2 min and 40 sec for each epoch with a  RTX 6000.

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

