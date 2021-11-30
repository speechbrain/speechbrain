# Grapheme-to-phoneme (G2P).
This folder contains the scripts to train a grapheme-to-phoneme system
that converts characters in input to phonemes in output. It used the
lexicon of the LibriSpeech dataset

You can download LibriSpeech at http://www.openslr.org/12

# How to run
python train.py train/train.yaml

# Results

| Release | hyperparams file | Test PER | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| -----:| --------:| :-----------:|
| 20-05-22 | train.yaml | 7.28% | https://drive.google.com/drive/folders/1nk9ms8cQ5N07wOG4oTi9h5a1dmiPmvnv?usp=sharing | 1xV100 32GB |


# Training Time
About 2 minutes for each epoch with a TESLA V100.


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
