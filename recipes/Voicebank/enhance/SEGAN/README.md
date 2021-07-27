
# VoiceBank Speech Enhancement with SEGAN
This recipe implements a speech enhancement system based on the SEGAN architecture
with the VoiceBank dataset.
(based on the paper: Pascual et al. https://arxiv.org/pdf/1703.09452.pdf).

# How to run
python train.py hparams/train.yaml

# Results
| Release | hyperparams file | Test PESQ | Test STOI | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| -----:| --------:| :-----------:|
| 2021-07-10 | train.yaml |  2.38 | 0.923 | https://drive.google.com/drive/folders/1gLxbH59LpMJFhvGHLPsVlX_MP2lcwVC8?usp=sharing | 1xV100 16GB |

# Training Time
About 2 min and 30 sec for each epoch with a TESLA V100.



# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


# ***Citing SEGAN**

```bibtex
@inproceedings{DBLP:conf/interspeech/PascualBS17,
  author    = {Santiago Pascual and
               Antonio Bonafonte and
               Joan Serr{\`{a}}},
  title     = {{SEGAN:} Speech Enhancement Generative Adversarial Network},
  booktitle = {Proc. of Interspeech},
  pages     = {3642--3646},
  year      = {2017},
}
```

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
