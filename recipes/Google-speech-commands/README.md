# Google Speech Command v0.02 Dataset
This folder contains recipes for command recognition with [Google Speech Command Dataset](https://www.tensorflow.org/datasets/catalog/speech_commands), including a sample recipe for the recent [LEAF audio frontend](https://openreview.net/forum?id=jM76BCb6F9m).
The recipes supports 12 or 35 commands.

# How to run
To run it, please type:

```
python train.py hparams/xvect.yaml --data_folder=/path_to_/GSC (V12 task)
python train.py hparams/xvect.yaml --data_folder=/path_to_/GSC --seed=1234  --number_of_commands=35 --percentage_unknown=0 --percentage_silence=0 (v35 task)

# for leaf
python train.py hparams/xvect_leaf.yaml --data_folder=/path_to_/GSC --seed=1234  --number_of_commands=35 --percentage_unknown=0 --percentage_silence=0 (v35 task)
```

# Performance summary

[Command accuracy on Google Speech Commands]
| System | Accuracy |
|----------------- | ------------ |
| xvector + augment v12 | 98.14% |
| xvector + augment v35 | 97.43% |
| xvector + augment + LEAF v35 | 96.79% |



# PreTrained Model + Easy-Inference
You can find the pre-trained model with an easy-inference function on HuggingFace:
- https://huggingface.co/speechbrain/google_speech_command_xvector

You can find the full experiment folder (i.e., checkpoints, logs, etc) here:
- xvector v12: https://www.dropbox.com/sh/9n9q42pugbx0g7a/AADihpfGKuWf6gkwQznEFINDa?dl=0
- xvector leaf v35: https://www.dropbox.com/sh/r63w4gytft4s1x6/AAApP8-pp179QKGCZHV_OuD8a?dl=0


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
