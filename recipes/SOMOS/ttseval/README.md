# MOS Estimation (with SOMOS)
This folder contains the recipes for training TTS evaluation systems trained on LJSpeech using the Samsung Open MOS Dataset (SOMOS)

# Dataset
The dataset can be downloaded from here:
https://zenodo.org/records/7378801

# Installing Extra Dependencies

Before proceeding, ensure you have installed the necessary additional dependencies. To do this, simply run the following command in your terminal:

```
pip install -r extra_requirements.txt
```
# TTS Evaluation Model

The model is based loosely on the baseline model described in the VoiceMOS 2022 challenge described in the paper below:
https://arxiv.org/pdf/2203.11389.pdf

It is based on the same principle: the weights of the base self-supervised representation model are updated in order
to fine-tune it to the quality assessment task. Linear regression between human and model ratings is used for
assessment.

Additional enhancements have been added. The updated model featured a shallow encoder-only Tranformer before the pooling
layer in order to introduce an attention mechanism.

To run this recipe, run the following command

```
python train.py --device=cuda:0  --data_folder=/your_folder/SOMOS hparams/train_ssl_wavlm_xformer.yaml
```

# Training Results
| Release     | Model             | hyperparams file | Val R | Test R | HuggingFace Link                                                    | Model Link                           | GPUs        |
| ----------- |:-----------------:| ----------------:|:--------------:|:-------------------------------------------------------------------:|:------------------------------------:|:-----------:|
| 2024-02-26  | WavLM Transformer | train.yaml       | TBD   | TBD    | [model](https://huggingface.co/flexthink/ttseval-wavlm-transformer) | [model](https://www.dropbox.com/tbd) | 1xV100 32GB |


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

