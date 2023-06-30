# LibriSpeech ASR with Transducer models.
This folder contains scripts necessary to run an ASR experiment with the LibriSpeech dataset;
Before running this recipe, make sure numba is installed (pip install numba)
You can download LibriSpeech at http://www.openslr.org/12

# Extra-Dependencies
This recipe supports two implementations of the transducer loss, see `use_torchaudio` arg in the yaml file:
1. Transducer loss from torchaudio (this requires torchaudio version >= 0.10.0) (Default).
2. Speechbrain implementation using Numba. To use it, please set `use_torchaudio=False` in the yaml file. This version is implemented within SpeechBrain and  allows you to directly access the python code of the transducer loss (and directly modify it if needed).

Note: Before running this recipe, make sure numba is installed. Otherwise, run:
```
pip install numba
```

# How to run it
```shell
python train.py train/train.yaml
```

# Librispeech Results

Dev. clean is evaluated with Greedy Decoding while the test sets are using Greedy Decoding OR a RNNLM + Beam Search.

| Release | hyperparams file | Dev. Clean | Test-clean Greedy | Test-other Greedy | Test-clean BS+RNNLM| Test-other BAS+RNNLM | Model link | GPUs |
|:-------------:|:---------------------------:| :------:| :-----------:| :------------------:| :------------------:| :------------------:| :--------:| :-----------:|
| 2020-10-22 | conformer_transducer.yaml | 2.8 | 3.0 | 7.1 | 2.8 | 6.8 | Not Available | 4xA100 80GB |



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
