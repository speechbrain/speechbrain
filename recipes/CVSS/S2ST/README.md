# Speech-to-Speech Translation (with CVSS)
This folder contains the recipe for training a speech-to-unit translation (S2UT) model using a pre-trained Wav2Vec 2.0 encoder and a transformer decoder on the CVSS dataset.
The implementation is based on [Textless Speech-to-Speech Translation](https://arxiv.org/abs/2112.08352) and [Enhanced Direct Speech-to-Speech Translation Using Self-supervised Pre-training and Data Augmentation](https://arxiv.org/abs/2204.02967) papers.

## Dataset
[CVSS](https://github.com/google-research-datasets/cvss) is a massively multilingual-to-English speech-to-speech translation corpus. It covers pairs from 21 languages into English. CVSS is derived from the Common Voice speech corpus and the CoVoST 2 speech-to-text translation corpus.
The CVSS dataset includes two versions of spoken translation: CVSS-C and CVSS-T. While both versions can be utilized to train the S2UT model, we recommend using CVSS-C because of its superior speech quality.

The first step is to select a source language and download [Common Voice (version 4)](https://commonvoice.mozilla.org/en/datasets) for the chosen language code. In this recipe, we've chosen French as the source language.
The next step is to pair translation audio clips with the source speech by downloading the corresponding subset of the [CVSS dataset](https://github.com/google-research-datasets/cvss). In our case, we have to download the French CVSS-C subset, which corresponds to the English translation of the French portion of the Common Voice dataset.

At this point, you should have two distinct folders: the first one containing the Common Voice data and the second one containing the CVSS data.

> Note: In the recipe, we frequently employ the terms `src_data` and `tgt_data`.
> `src_data` refers to the source language data (Common Voice).
> `tgt_data` refers to the target language data (CVSS).

## Installing Extra Dependencies
Before proceeding, ensure you have installed the necessary additional dependencies. To do this, simply run the following command in your terminal:
```
pip install -r extra_requirements.txt
```

## How to Run
Before training the speech-to-unit translation (S2UT) model, we have to quantize the target speech into discrete speech units. This is achieved by training a k-means model on raw speech features, which will then serve as the target for training the S2UT model. By default, we use a pre-trained model with `k=100` trained on the 6th layer of HuBERT. For instructions on training a new quantization model, please refer to `recipes/LJSpeech/quantization`.

To train the S2UT model on French-English, simply run the following command:
```
python train.py hparams/train_fr-en.yaml --src_data_folder=/corpus/CommonVoice/fr --tgt_data_folder=/corpus/CVSS/fr --bfloat16_mix_prec
```

>  Dynamic batch settings are optimized for a 40GB VRAM GPU. Don't hesitate to adjust max_batch_len and max_batch_len_val to fit your GPU's capabilities.


# Performance summary
Results are reported in terms of sacrebleu.

| hyperparams file | valid | test   | Model      | Training logs | GPUs       |
|:----------------:|:-----:| :-----:|:-------:   | :-----------: |:---------: |
| train_fr-en.yaml | 24.25   | 24.47    | [dropbox](https://www.dropbox.com/sh/woz4i1p8pkfkqhf/AACmOvr3sS7p95iXl3twCj_xa?dl=0) | [wandb](https://wandb.ai/jar0d/s2ut_cvss_sb/runs/uh4tvc8c?workspace=user-jar0d)    |1xA100 80GB |

Training requires about 1 hour and 5 minutes for each epoch on an NVIDIA A100 GPU. A total of 30 epochs are needed.

To synthesize speech from the predicted speech units, you need to train a unit-based HiFi-GAN vocoder. If you haven't done this already, please refer to the `LJSpeech/TTS/vocoder/unit_hifi_gan` recipe.

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
