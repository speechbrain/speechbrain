# **Speech enhancement with Microsoft DNS dataset**
This folder contains the recipe for speech enhancement on Deep Noise Suppression (DNS) Challenge 4 (ICASSP 2022) dataset using SepFormer.

Install additional dependencies
```
pip install mir_eval
pip install pyroomacoustics==0.3.1
```
To start training (two setups- 8K and 16K)
```
python train.py hparams/sepformer-dns-8k.yaml
python train.py hparams/sepformer-dns-16k.yaml
```

# **Results**
TBD

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


**Citing SepFormer**
```bibtex
@inproceedings{subakan2021attention,
      title={Attention is All You Need in Speech Separation},
      author={Cem Subakan and Mirco Ravanelli and Samuele Cornell and Mirko Bronzi and Jianyuan Zhong},
      year={2021},
      booktitle={ICASSP 2021}
}
```