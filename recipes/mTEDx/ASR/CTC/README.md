# mTEDx ASR with CTC models.
This folder contains the scripts to train a wav2vec based system using mTEDx. You can train either a single-language wav2vec model or multilingual
wav2vec model. Before running this recipe, make sure to read this [README](../../README.md) file first.

**Note:**\
Wav2vec model used in this recipe is pre-trained on the French language.
In order to use another language, don't forget to change the `wav2vec2_hub`
in the `train_wav2vec.yaml` YAML file. 


# How to run

To train a single-language wav2vec model, run:
```bash
$ python train.py hparams/train_wav2vec.yaml
```

To train a multilingual wav2vec model, run:
```bash
$ python train.py hparams/train_xlsr.yaml
```

# Results

TODO


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
