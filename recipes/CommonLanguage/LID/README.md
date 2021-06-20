# Language Identification experiments with CommonLanguage.
This folder contains scripts for running language identificationexperiments with the [CommonLanguage](https://drive.google.com/uc?id=1Vzgod6NEYO1oZoz_EcgpZkUO9ohQcO1F) dataset. These experiments were highly inspired by Speaker Identification tasks on VoxCeleb and follow a similar path.

# Training [X-Vectors](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf)
Run the following command to train xvectors:

`python train.py hparams/train_xvectors.yaml`

The experiment is fine-tuning of the trained speaker embeddings done for Speaker Identification task on VoxCeleb, which can be found on [HuggingFace](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb). Therefore, most of the architecture choices come from that task.

Data augmentation and environmental corruption are done by concatenating waveforms, dropout, speed change, reverberation, noise, and noise+rev. The batch is double size of the original one. This may lead to
better performance, at the cost of longer training time and higher compute resourses.

# Training [ECAPA-TDNN](https://arxiv.org/abs/2005.07143)
Similar to the X-Vector a bigger and more powerful ECAPA-TDNN model can be used.

`python train.py hparams/train_ecapa_tdnn.yaml`

The experiment is also fine-tuning of the trained speaker embeddings done for Speaker Identification task on VoxCeleb, and can be accessed on [HuggingFace](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb). Therefore, most of the architecture choices come from that task.

Data augmentation and environmental corruption are done by concatenating waveforms, dropout, speed change, reverberation, noise, and noise+rev. The batch is double size of the original one. This may lead to
better performance, at the cost of longer training time and higher compute resourses.

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


