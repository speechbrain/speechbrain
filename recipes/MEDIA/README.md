# Media data preparation.

### Contributors :
- Gaëlle Laperrière, Avignon University, LIA
- Yannick Estève, Avignon University, LIA
- Bassam Jabaian, Avignon University, LIA
- Sahar Ghannay, Paris-Saclay University, CNRS, LISN
- Antoine Caubrière, Avignon University, LIA
- Valentin Pelloin, Le Mans University, LIUM
- Nathalie Camelin, Le Mans University, LIUM

The `parseXMLtoSB.py` script allows to prepare the Media French dataset for experiments. You need both [Media ASR (ELRA-S0272)](https://catalogue.elra.info/en-us/repository/browse/ELRA-S0272/) and [Media SLU (ELRA-E0024)](https://catalogue.elra.info/en-us/repository/browse/ELRA-E0024/) to run the script.

# How to run
```
python parseXMLtoSB.py [-w] [-r | -f] (-s | -a) data_folder wav_folder csv_folder
```
With :
- `data_folder` : Path where folders S0272 and E0024 are stored.
- `wav_folder` : Path where the wavs will be stored.
- `csv_folder` : Path where the csv will be stored.
- `-w` / `--skip_wav` : Skip the wav files storing if already done before.
- `-r` / `--relax` or `-f` / `--full` (by default) : Remove (relax) or keep (full) specifiers in concepts.
- `-s` / `--slu` or `-a` / `--asr` : Choose the task.

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
