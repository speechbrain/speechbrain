# Media data preparation.
The `media_prepare.py` script allows to prepare the Media French dataset for experiments. You need both [Media ASR (ELRA-S0272)](https://catalogue.elra.info/en-us/repository/browse/ELRA-S0272/) and [Media SLU (ELRA-E0024)](https://catalogue.elra.info/en-us/repository/browse/ELRA-E0024/) to run the script. Please also download the 2 csv files given [here](https://www.dropbox.com/sh/y7ab0lktbylz647/AADMsowYHmNYwaoL_hQt7NMha?dl=0) and place them in the `MEDIA` directory.

The recipes have been implemented following the paper of G. Laperrière, V. Pelloin, A. Caubriere, S. Mdhaffar, N. Camelin, S. Ghannay, B. Jabaian, Y. Estève, [The Spoken Language Understanding MEDIA Benchmark Dataset in the Era of Deep Learning: data updates, training and evaluation tools](https://aclanthology.org/2022.lrec-1.171).

**The results obtained with the different models can be found in the corresponding sub-directories!**

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
