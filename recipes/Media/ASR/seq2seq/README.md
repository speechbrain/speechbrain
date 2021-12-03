# Media ASR with CTC + Seq2Seq models +/- Wav2Vec 2.0.

### Contributors :
- Gaëlle Laperrière, Avignon University, LIA
- Yannick Estève, Avignon University, LIA
- Bassam Jabaian, Avignon University, LIA
- Sahar Ghannay, Paris-Saclay University, CNRS, LISN
- Salima Mdhaffar, Avignon University, LIA
- Antoine Caubrière, Avignon University, LIA

This folder contains scripts necessary to run an ASR experiment with the Media French dataset: [Media ASR (ELRA-S0272)](https://catalogue.elra.info/en-us/repository/browse/ELRA-S0272/), [Media SLU (ELRA-E0024)](https://catalogue.elra.info/en-us/repository/browse/ELRA-E0024/) both needed for the task.

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
