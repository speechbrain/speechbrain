# CommonVoice ASR with CTC + Attention based Seq2Seq models.
This folder contains scripts necessary to run an ASR experiment with the CommonVoice 14.0 dataset: [CommonVoice Homepage](https://commonvoice.mozilla.org/) and pytorch 2.0
# How to run
python train.py hparams/{hparam_file}.py

Make sure you have "transformers" installed if you use the wav2vec2 fine-tuning model.

# Data preparation
It is important to note that CommonVoice initially offers mp3 audio files at 42Hz. Hence, audio files are downsampled on the fly within the dataio function of the training script.

# Languages
Here is a list of the different languages that we tested within the CommonVoice dataset:
- French
- Kinyarwanda
- Italian
- English
- German
- Spanish

# Results

| Language | CommonVoice Release | hyperparams file | LM | Val. CER | Val. WER | Test CER | Test WER | HuggingFace link | Model link | GPUs |
| ------------- |:-------------:|:---------------------------:| -----:| -----:| -----:| -----:| -----:| :-----------:| :-----------:| :-----------:|
| French | 2023-08-15 | train_fr.yaml | No | 4.40 | 12.17 | 5.93 | 14.88 | [model](https://huggingface.co/speechbrain/asr-crdnn-commonvoice-14-fr) | [model](https://www.dropbox.com/sh/07a5lt21wxp98x5/AABhNwmWFaNFyA734bNZUO03a?dl=0) | 1xV100 32GB |
| Kinyarwanda | 2023-08-15 | train_rw.yaml | No | 6.75 | 23.66 | 10.80 | 29.22 | [model](https://huggingface.co/speechbrain/asr-crdnn-commonvoice-14-rw) | [model](https://www.dropbox.com/sh/i1fv4f8miilqgii/AAB3gE97kmFDA0ISkIDSUW_La?dl=0) | 1xV100 32GB |
| English | 2023-08-15 | train_en.yaml | No | 9.75 | 20.23 | 12.76 | 23.88 | [model](https://huggingface.co/speechbrain/asr-crdnn-commonvoice-14-en) | [model](https://www.dropbox.com/sh/h8ged0yu3ztypkh/AAAu-12k_Ceg-tTjuZnrg7dza?dl=0) | 1xV100 32GB |
| Italian | 2023-08-15 | train_it.yaml | No | 5.89 | 15.99 | 6.27 | 17.02 | [model](https://huggingface.co/speechbrain/asr-crdnn-commonvoice-14-it) | [model](https://www.dropbox.com/sh/ss59uu0j5boscvp/AAASsiFhlB1nDWPkFX410bzna?dl=0) | 1xV100 32GB |
| German | 2023-08-15 | train_de.yaml | No | 2.90 | 10.21 | 3.82 | 12.25 | [model](https://huggingface.co/speechbrain/asr-crdnn-commonvoice-14-de) | [model](https://www.dropbox.com/sh/zgatirb118f79ef/AACmjh-D94nNDWcnVI4Ef5K7a?dl=0) | 1xV100 32GB |
| Spanish | 2023-08-15 | train_es.yaml | No | 4.10 | 14.10 | 4.68 | 14.77 | [model](https://huggingface.co/speechbrain/asr-crdnn-commonvoice-14-es) | [model](https://www.dropbox.com/sh/r3w0b2tm1p73vft/AADCxdhUwDN6j4PVT9TYe-d5a?dl=0) | 1xV100 32GB |


## How to simply use pretrained models to transcribe my audio file?

SpeechBrain provides a simple interface to transcribe audio files with pretrained models. All the necessary information can be found on the different HuggingFace repositories (see the results table above) corresponding to our different models for CommonVoice.


# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{speechbrainV1,
  title={Open-Source Conversational AI with SpeechBrain 1.0},
  author={Mirco Ravanelli and Titouan Parcollet and Adel Moumen and Sylvain de Langen and Cem Subakan and Peter Plantinga and Yingzhi Wang and Pooneh Mousavi and Luca Della Libera and Artem Ploujnikov and Francesco Paissan and Davide Borra and Salah Zaiem and Zeyu Zhao and Shucong Zhang and Georgios Karakasidis and Sung-Lin Yeh and Pierre Champion and Aku Rouhe and Rudolf Braun and Florian Mai and Juan Zuluaga-Gomez and Seyed Mahed Mousavi and Andreas Nautsch and Xuechen Liu and Sangeet Sagar and Jarod Duret and Salima Mdhaffar and Gaelle Laperriere and Mickael Rouvier and Renato De Mori and Yannick Esteve},
  year={2024},
  eprint={2407.00463},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2407.00463},
}
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
