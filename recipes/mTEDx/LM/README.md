# Language Model with mTEDx
This folder contains recipes for training language models for the mTEDx Dataset.
This recipe supports only an RNN-based LM.
The scripts rely on the HuggingFace dataset, which manages data reading and
loading from large text corpora.

Before running this recipe, make sure to read the mTEDx
[README](../../README.md) file first.

# Extra Dependency:
Make sure you have the HuggingFace dataset installed. If not, run the following
command:
```bash
pip install -r extra_requirements.txt
```

# How to run:

To train a RNNLM on single/multiple language(s), run the following command:

```bash
python train.py hparams/train.yaml
```


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