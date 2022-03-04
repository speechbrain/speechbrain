# CommonVoice ASR with Transducers.
This folder contains scripts necessary to run an ASR experiment with the CommonVoice dataset: [CommonVoice Homepage](https://commonvoice.mozilla.org/)

# Extra-Dependencies
This recipe support two implementation of Transducer loss, see `use_torchaudio` arg in Yaml file:
1- Transducer loss from torchaudio (if torchaudio version >= 0.10.0) (Default)
2- Speechbrain Implementation using Numba lib. (this allow you to have a direct access in python to the Transducer loss implementation)
Note: Before running this recipe, make sure numba is installed. Otherwise, run:
```
pip install numba
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