# CommonVoice ASR with Transducers.
This folder contains scripts necessary to run an ASR experiment with the CommonVoice 14.0 dataset: [CommonVoice Homepage](https://commonvoice.mozilla.org/) and pytorch 2.0

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
- Italian
- German

# Results

| Language | Release | hyperparams file | LM | Val. CER | Val. WER | Test CER | Test WER | Model link | GPUs |
| ------------- |:-------------:|:---------------------------:| -----:| -----:| -----:| -----:| -----:| :-----------:| :-----------:|
| French | 2023-08-15 | train_fr.yaml | No | 5.75 | 14.53 | 7.61 | 17.58 | [model](https://huggingface.co/speechbrain/asr-transducer-commonvoice-14-fr) | [model](https://www.dropbox.com/sh/nv2pnpo5n3besn3/AADZ7l41oLt11ZuOE4MqoJhCa?dl=0) | 1xV100 16GB |
| Italian | 2023-08-15 | train_it.yaml | No | 4.66 | 14.08 | 5.11 | 14.88 | [model](https://huggingface.co/speechbrain/asr-transducer-commonvoice-14-it) | [model](https://www.dropbox.com/sh/ksm08x0wwiomrgs/AABnjPePWGPxqIqW7bJHp1jea?dl=0) | 1xV100 16GB |
| German | 2023-08-15 | train_de.yaml | No | 4.32 | 13.09 | 5.43 | 15.25 | [model](https://huggingface.co/speechbrain/asr-transducer-commonvoice-14-de) | [model](https://www.dropbox.com/sh/jfge6ixbtoje64t/AADeAgL5un0A8uEjPSM84ex8a?dl=0) | 1xV100 16GB |

The output folders with checkpoints and logs can be found [here](https://www.dropbox.com/sh/852eq7pbt6d65ai/AACv4wAzk1pWbDo4fjVKLICYa?dl=0).

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
