# CommonVoice ASR with Transducers.
This folder contains scripts necessary to run an ASR experiment with the CommonVoice dataset: [CommonVoice Homepage](https://commonvoice.mozilla.org/)

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

# How to run
python train.py hparams/{hparam_file}.py

# Data preparation
It is important to note that CommonVoice initially offers mp3 audio files at 42Hz. Hence, audio files are downsampled on the fly within the dataio function of the training script.

# Languages
Here is a list of the different languages that we tested within the CommonVoice dataset
with our transducers:
- French

# Results

| Language | Release | hyperparams file | LM | Val. CER | Val. WER | Test CER | Test WER | Model link | GPUs |
| ------------- |:-------------:|:---------------------------:| -----:| -----:| -----:| -----:| -----:| :-----------:| :-----------:|
| French | 2020-06-22 | train_fr.yaml | No | 6.70 | 18.97 | 7.41 | 20.18 | [model](https://drive.google.com/drive/folders/1ZwY2FaRl1gfFbupodph_xRiGj4h25I08?usp=sharing) | 2xV100 16GB |

The output folders with checkpoints and logs can be found [here](https://drive.google.com/drive/folders/11NMzY0zV-NqJmPMyZfC3RtT64bYe-G_O?usp=sharing).

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
