# Media SLU with CTC + Seq2Seq models +/- Wav2Vec 2.0.

### Contributors : 
- Gaëlle Laperrière, Avignon University, LIA
- Yannick Estève, Avignon University, LIA 
- Bassam Jabaian, Avignon University, LIA
- Sahar Ghannay, Paris-Saclay University, CNRS, LISN
- Salima Mdhaffar, Avignon University, LIA 
- Antoine Caubrière, Avignon University, LIA 

This folder contains scripts necessary to run an SLU experiment with the Media French dataset: [Media ASR (ELRA-S0272)](https://catalogue.elra.info/en-us/repository/browse/ELRA-S0272/), [Media SLU (ELRA-E0024)](https://catalogue.elra.info/en-us/repository/browse/ELRA-E0024/) both needed for the task.

# How to run 
Do not forget to process the dataset and change the `!PLACEHOLDER` in the yaml file.

```bash
python train.py hparams/{hparam_file}.yaml
python train_with_wav2vec.py hparams/{hparam_file}.yaml
```

# Data preparation
It is important to note that Media initially offers audio files at 8kHz. Hence, audio files are upsampled on the fly within the preparation script to 16kHz.

# Results 

| Media Release | hyperparams file | Dev ChER | Dev CER | Dev CVER | Test ChER | Test CER | Test CVER | Wav2Vec |
|:-------------:|:-------------------------:|:----:|:----:|:----:|:----:|:----:|:----:|:------------------------------------:|
| 2008-03-27 | train_full.yaml | 52.61 | 77.41 | 84.27 | 53.98 | 75.03 | 82.07 | [LeBenchmark wav2vec2-FR-3K-large](https://huggingface.co/LeBenchmark/wav2vec2-FR-3K-large) |
| 2008-03-27 | train_relax.yaml | 52.16 | 76.97 | 83.92 | 53.29 | 73.32 | 80.99 | [LeBenchmark wav2vec2-FR-3K-large](https://huggingface.co/LeBenchmark/wav2vec2-FR-3K-large) |
| 2008-03-27 | train_with_wav2vec_full.yaml | 8.14 | 30.07 | 33.00 | 7.90 | 27.42 | 30.59 | [LeBenchmark wav2vec2-FR-3K-large](https://huggingface.co/LeBenchmark/wav2vec2-FR-3K-large) |
| 2008-03-27 | train_with_wav2vec_relax.yaml | 7.81 | 22.89 | 28.79 | 7.52 | 20.71 | 27.02 | [LeBenchmark wav2vec2-FR-3K-large](https://huggingface.co/LeBenchmark/wav2vec2-FR-3K-large) |

## How to simply use pretrained models?

SpeechBrain provides a simple interface to make SLU for audio files with pretrained models. All the necessary information can be found on the different HuggingFace repositories (see the results table above) corresponding to our different models for Media.

# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/

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
