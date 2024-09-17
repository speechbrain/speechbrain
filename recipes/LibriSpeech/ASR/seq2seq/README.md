# LibriSpeech ASR with seq2seq models (CTC + attention).
This folder contains the scripts to train a seq2seq CNN-RNN-based system using LibriSpeech.
You can download LibriSpeech at http://www.openslr.org/12

# How to run
```shell
python train.py hparams/file.yaml
```

# Results

| Release | hyperparams file | Test Clean WER | HuggingFace link | Full model link | GPUs |
|:-------------:|:---------------------------:| :-----:| :-----:| :-----:| :--------:|
| 01-03-21 | train_BPE_1000.yaml | 3.16 | [HuggingFace](https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech) | [Model](https://www.dropbox.com/sh/1ycv07gyxdq8hdl/AABUDYzza4SLYtY45RcGf2_0a?dl=0)| 1xV100 32GB |
| 01-03-21 | train_BPE_5000.yaml | 2.89 | [HuggingFace](https://huggingface.co/speechbrain/asr-crdnn-transformerlm-librispeech) | [Model](https://www.dropbox.com/sh/a39wq3h60luv552/AABBnCM2Uf-CNax_cgMWdqDda?dl=0) | 1xV100 32GB |

# Training Time
It takes about 5 hours for each epoch on a NVIDIA V100 (32GB).

# PreTrained Model + Easy-Inference
You can find the pre-trained model with an easy-inference function on HuggingFace:
- https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech
- https://huggingface.co/speechbrain/asr-crdnn-transformerlm-librispeech
- https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech

You can find the full experiment folder (i.e., checkpoints, logs, etc) here:
https://www.dropbox.com/sh/a39wq3h60luv552/AABBnCM2Uf-CNax_cgMWdqDda?dl=0


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
