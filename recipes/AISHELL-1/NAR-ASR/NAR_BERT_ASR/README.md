# AISHELL-1 Non-Autoregressive ASR with [NAR-BERT-ASR](https://arxiv.org/abs/2104.04805) models.


# Introduction
This folder contains recipes for [NAR-BERT-ASR](https://arxiv.org/abs/2104.04805) with [AISHELL-1](https://www.openslr.org/33/), a 150-hour Chinese ASR dataset.


# How to run
Step 1: Install extra dependencies.
```
pip install transformers
```

Step 2: Pretrain the NAR-BERT-ASR-Encoder.
```
python pretrain_encoder.py hparams/pretrain_encoder.yaml --data_folder=/localscratch/aishell/
```

Step 3: Fine-tune the NAR-BERT-ASR.
```
python train_nar_bert_asr.py hparams/train_nar_bert_asr.yaml --data_folder=/localscratch/aishell/ --pretrain_weights_dir=results/Pretrain_Encoder/save/{CKPT+XXXX-XX-XX+XX-XX-XX+XX}
```


# Performance summary
Results are reported in terms of Character Error Rate (CER).

| hyperparams file | Test CER | Dev CER | GPUs |
|:--------------------------:|:-----:|:-----:|:-----:|
| pretrain_encoder.yaml | 7.94 | 6.95 | 1xRTX 2080 Ti 11GB |
| train_nar_bert_asr.yaml | 5.26 | 4.64 | 1xTITAN RTX 24GB |

You can checkout our results (models, training logs, etc,) here:
https://drive.google.com/drive/folders/1zdUUp-hdbpVvpV3jUCgBqet04HjqZfRy?usp=sharing
 

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