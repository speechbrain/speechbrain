# Speech-to-Speech Translation (with CVSS)
This folder contains the recipe for training a speech-to-unit translation (S2UT) model using a pre-trained Wav2Vec 2.0 encoder and a transformer decoder on the CVSS dataset.  
The implementation is based on [Textless Speech-to-Speech Translation](https://arxiv.org/abs/2112.08352) and [Enhanced Direct Speech-to-Speech Translation Using Self-supervised Pre-training and Data Augmentatio](https://arxiv.org/abs/2204.02967) papers.


## Dataset
CVSS is a massively multilingual-to-English speech-to-speech translation corpus. It covers pairs from 21 languages into English. CVSS is derived from the Common Voice speech corpus and the CoVoST 2 speech-to-text translation corpus.  
The CVSS dataset includes two versions of spoken translation: CVSS-C and CVSS-T. While both versions can be utilized to train the S2UT model, we recommend using CVSS-C because of its superior speech quality.  
The final step is to pair the translation audio clips and translation texts with the source speech by downloading Common Voice (version 4) for the selected language code.

CVSS dataset can be downloaded from [here](https://github.com/google-research-datasets/cvss).  
Common Voice can be downloaded from [here](https://commonvoice.mozilla.org/en/datasets).  


## Installing Extra Dependencies
Before proceeding, ensure you have installed the necessary additional dependencies. To do this, simply run the following command in your terminal:
```
pip install -r extra_requirements.txt
```

## How to Run
Before training the speech-to-unit translation (S2UT) model, we have to quantize the target speech into discrete speech units. This is achieved by training a k-means model on raw speech features, which will then serve as the target for training the S2UT model. By default, we use a pre-trained model with `k=100` trained on the 6th layer of HuBERT. For instructions on training a new quantization model, please refer to `recipes/LJSpeech/TTS/quantization`.

To train the S2UT model on French-English, simply run the following command:
```
python train.py hparams/train_fr-en.yaml --src_data_folder=/corpus/CV4/fr --tgt_data_folder=/corpus/CV4/fr-en --bfloat16_mix_prec
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
