# LibriSpeech ASR with Transducer models.
This folder contains scripts necessary to run an ASR experiment with the LibriSpeech dataset;
Before running this recipe, make sure numba is installed (pip install numba)
You can download LibriSpeech at http://www.openslr.org/12

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
python train.py train/train.yaml

# Librispeech 100H Results

| Release | hyperparams file | Val. CER | Val. WER | Test WER (test clean) | Model link | GPUs |
|:-------------:|:---------------------------:| ------:| :-----------:| :------------------:| --------:| :-----------:|
| 2020-10-22 | train.yaml |  5.2 | GS: 11.45 | BS (beam=4): 11.03 | Not Available | 1xRTX-8000 48GB |

The output folder with the checkpoints and training logs is available [here](https://drive.google.com/drive/folders/17kEW0crU3tyP-8-u5TeoFom4ton_B-j2?usp=sharing).


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
