# Language Model with LibriSpeech
This folder contains recipes for training language models for the LibriSpeech Dataset.
It supports both an RNN-based LM and a Transformer-based LM. 
The scripts rely on the HuggingFace dataset, which manages data reading and loading from
large text corpora. 

You can download LibriSpeech at http://www.openslr.org/12

# Extra Dependency:
Make sure you have the HuggingFace dataset installed. If not, type:
pip install datasets

# How to run:
python train.py hparams/RNNLM.yaml
python train.py hparams/transformer.yaml

| Release | hyperparams file | Test PP | Model link | GPUs |
| :---     | :---: | :---: | :---: | :---: |
| 20-05-22 | RNNLM.yaml (1k BPE) | --.-- | [link](https://drive.google.com/drive/folders/1CCsGfq0mbHTvOVL7cJRl6hwmXDQB2Xcy?usp=sharing) | 1xV100 32GB |
| 20-05-22 | RNNLM.yaml (5k BPE) | --.-- | [link](https://drive.google.com/drive/folders/17Qa2-3Q9KF-8huxxH_oZGdEwz4igCJ4o?usp=sharing) | 1xV100 32GB |
| 20-05-22 | transformer.yaml | --.-- | [link](https://drive.google.com/drive/folders/1oCEAjYUyummzcQSkhCbl_3Vf2ozy0BXp?usp=sharing) | 1xV100 32GB |


# Training time
Training a LM takes a lot of time. In our case, it take 3/4 weeks on 4 TESLA V100. Use the pre-trained model to avoid training it from scratch


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