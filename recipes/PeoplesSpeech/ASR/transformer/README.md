# How to run an ASR experiment with People's Speech

This recipe provides the necessary recipe to train a speech recognizer on the People's Speech dataset.

## Downloading the dataset

The full dataset will occupy around 3 TB of storage and must be obtained following the standard HuggingFace
dataset process in the [corresponding HuggingFace people's speech repository](https://huggingface.co/datasets/MLCommons/peoples_speech).

By default, our recipe set the HuggingFace environmental variable *HF_DATASETS_OFFLINE* to disable the download
of the dataset. This is because we ask the user to download it before, like for any other recipe. Indeed, we do
not want a recipe script, potentially run on GPUs nodes, to take hours downloading a dataset first.

## Important note on the data

[People's speech](https://arxiv.org/pdf/2111.09344) is a very challenging dataset containing two main sets 'clean' and 'dirty' totalising 28,000 hours of error-prone transcribed speech. From our experience, it is most likely that this dataset should not be utilised alone, as the training material is very hard. It mostly contains spontaneous speech, with a few errors in transcription alignments and a lot of transcription inconsistency e.g. sometimes transcribing filler words, sometimes not - sometimes transcribing repetition, sometimes not. This makes the models trained on this data fairly hard to evaluate... The provided validation and test sets seem to be also a bit out of domain.

However, this dataset remain very valuable due to the high quantity of spontaneous speech provided.

## Results

It is tricky to evaluate this dataset. Here we provide the results on the official validation
and test split as well as Voxpopuli test set and the test-clean from LibriSpeech. It is important
to remember that the results for the two last sets are out-of-domain and vocabulary for the
tokenizer.

| hyperparams file | validation WER | test WER | VoxPopuli | LibriSpeech test-clean WER | GPUs |
|:-------------:|:-------------:|:-------------:|:---------------------------:| :-----:| :-----:|
| conformer_large.yaml | 28.44 | 31.04 | 20.0 | 9.45 | 8xA100 80GB |

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

