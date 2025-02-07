# Libriheavy Dataset
This folder contains the scripts to train a Transformer-based speech recognizer.

1. Please download Libri-Light at https://github.com/facebookresearch/libri-light/tree/main/data_preparation
After this step, please make sure you have all the splits (small, medium, and large) in one folder.
Please note if you want to use the large split, the large.tar file is 3.05TB. Also, the download can take quite a while.

2. Please git clone the repo https://github.com/k2-fsa/libriheavy, and follow the repo's instruction to prepare Libriheavy manifests.
After this step, please make sure you have all the "jsonl.gz" Libriheavy manifest files in one folder.

**Note 1:** This recipe relies on the `soundfile` backend for fast audio processing. Libriheavy comes with long audio files, and we need to read them in chunks. In our experiments, we found that `soundfile` was the only audio backend fast enough to read these long audio files. You can dynamically change the backend through the `--audio_backend` parameter in the YAML file.

**Note 2:** If you don't have the `large` folder but want to run this recipe with the `small` and/or `medium` splits, you need to download the official `dev` and `test` splits from the LibriSpeech dataset. This is necessary because the `dev` and `test` splits for Libriheavy are located in the `large` folder. You can download LibriSpeech at http://www.openslr.org/12 and run the `librispeech_prepare.py` script from the `recipes/LibriSpeech/` folder. Then, specify the `dev_splits` and `test_splits` parameters in the YAML file.

# How to run
```shell
python train.py hparams/transformer.yaml --data_folder=/path/to/Libri-Light --manifest_folder=/path/to/Libriheavy
```

# LibriSpeech Dev/Test Results
Results of trained with the Libriheavy large split and tested with LibriSpeech dev/test sets.

| Release | hyperparams file | Dev Clean WER (Transformer LM) | Test Clean WER (Transformer LM) | Test Other WER (Transformer LM) | HuggingFace link | Model link | GPUs |
|:-------------:|:-------------:|:-------------:|:---------------------------:| :-----:| :-----:| :-----:| :--------:|
| 24-12-09 | conformer_large.yaml | 1.58 | 1.74 | 3.92 | Not Avail. | Not Avail. | 8xA100 80GB |


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
