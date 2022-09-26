# mTEDx ASR with Transducer models.
This folder contains scripts necessary to run an ASR experiment with the mTEDx
dataset. Before running this recipe, make sure to read this
[README](../../README.md) file first.

# Extra-Dependencies
This recipe supports three implementations of the transducer loss, see
`framework` arg in the yaml file:
1. Transducer loss from torchaudio (this requires torchaudio version >= 0.10.0)
(Default).
2. Speechbrain implementation using Numba. To use it, please set
`framework=speechbrain` in the yaml file. This version is implemented within
SpeechBrain and  allows you to directly access the python code of the
transducer loss (and directly modify it if needed).
3. FastRNNT (pruned / unpruned) loss function.
  - To use the un-pruned loss function, please set `framework=fastrnnt`.
  - To use the pruned loss function, please change the whole `transducer_cost`
  yaml variable.

If you are planning to use speechbrain RNNT loss function, install `numba`:
```
pip install numba
```

If you are planning to use FastRNNT loss function, install `FastRNNT`:
```
pip install FastRNNT
```

# How to run it

To run Transducer experiments
```bash
# CRDNN Transducer + PyTorch RNNT loss
$ python train.py hparams/train.yaml

# CRDNN Transducer + FastRNNT unpruned loss
$ python train.py hparams/train_unpruned.yaml

# CRDNN Transducer + FastRNNT pruned loss
$ python train.py hparams/train_pruned.yaml

# Wav2vec Transducer + FastRNNT pruned loss
$ python train_wav2vec.py hparams/train_wav2vec_pruned.yaml
```

# mTEDx French Results

| Release | hyperparams file | Val. CER | Val. WER | Test WER | Model link | GPUs |
|:-------------:|:---------------------------:| ------:| :-----------:| :------------------:| --------:| :-----------:|
| 2022-08-10 | train_wav2vec_pruned.yaml |  4.49 | GS: 10.66 | GS: 12.59 | Not Available | 4xV100 32GB |
| 2022-07-20 | train_unpruned.yaml | 21.22 | GS: 47.04 | BS (beam=4): 57.24 | Not Available | 4xV100 32GB |


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
