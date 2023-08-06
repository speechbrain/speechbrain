# Speech-to-Speech Translation (with CVSS)
This folder contains the recipes for training speech-to-unit translation model with the CVSS dataset.
CVSS is a massively multilingual-to-English speech-to-speech translation corpus, covering pairs from 21 languages into English. CVSS is derived from the Common Voice speech corpus and the CoVoST 2 speech-to-text translation corpus.

To train the unit-based HiFi-GAN vocoder, please refer to the LJSpeech/S2ST recipe.

## Dataset
The dataset can be downloaded from here:
https://github.com/google-research-datasets/cvss

## Installing Extra Dependencies
Before proceeding, ensure you have installed the necessary additional dependencies. To do this, simply run the following command in your terminal:
```
pip install -r extra_requirements.txt
```

## How to run
To train the S2UT model on French-English, simply run the following command:
```
python train.py hparams/train_fr-en.yaml --src_data_folder=/corpus/CV4/fr --tgt_data_folder=/corpus/CV4/fr-en --bfloat16_mix_prec
```

>  Dynamic batch settings are optimized for a 40GB VRAM GPU. Don't hesitate to adjust max_batch_len and max_batch_len_val to fit your GPU's capabilities.

# Performance summary
Results are reported in terms of sacrebleu.

| hyperparams file | valid | test   | Model      | Training logs | GPUs       |
|:----------------:|:-----:| :-----:|:-------:   | :-----------: |:---------: |
| train_fr-en.yaml | 0.0   | 0.0    | Not Avail. | Not Avail.    |1xA100 40GB |

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