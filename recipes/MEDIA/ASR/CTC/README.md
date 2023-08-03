# Media ASR with CTC + Wav2Vec 2.0.
This folder contains scripts necessary to run an ASR experiment with the Media French dataset: [Media ASR (ELRA-S0272)](https://catalogue.elra.info/en-us/repository/browse/ELRA-S0272/), [Media SLU (ELRA-E0024)](https://catalogue.elra.info/en-us/repository/browse/ELRA-E0024/) both needed for the task. Please also download the 2 csv files given [here](https://www.dropbox.com/sh/y7ab0lktbylz647/AADMsowYHmNYwaoL_hQt7NMha?dl=0) and place them in the `../../MEDIA` directory.

This recipe has been implemented following the paper of G. Laperrière, V. Pelloin, A. Caubriere, S. Mdhaffar, N. Camelin, S. Ghannay, B. Jabaian, Y. Estève, [The Spoken Language Understanding MEDIA Benchmark Dataset in the Era of Deep Learning: data updates, training and evaluation tools](https://aclanthology.org/2022.lrec-1.171).

# How to run
Do not forget to process the dataset and change the `!PLACEHOLDER` in the yaml file.

```bash
python train_hf_wav2vec.py hparams/train_hf_wav2vec.yaml
```

# Data preparation
It is important to note that Media initially offers audio files at 8kHz. Hence, audio files are upsampled on the fly within the preparation script to 16kHz.

# Results

| Media Release | hyperparams | Test ChER | Wav2Vec | Training time | HuggingFace link | Model link |
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| 2008-03-27 | train_with_wav2vec.yaml | 4.78 | [LeBenchmark wav2vec2-FR-3K-large](https://huggingface.co/LeBenchmark/wav2vec2-FR-3K-large) | 12m30s per epoch | [here](https://huggingface.co/speechbrain/asr-wav2vec2-ctc-MEDIA) | Not Avail. |

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
