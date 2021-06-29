# Recipes for ASR with AISHELL-1
This folder contains recipes for tokenization and speech recognition with [AISHELL-1](https://www.openslr.org/33/), a 150-hour Chinese ASR dataset.

### Base recipe
The recipe uses an attention-based CRDNN sequence-to-sequence model with an auxiliary CTC loss.
To train a full recipe:

1- Train a tokenizer. The tokenizer takes in input the training transcripts and determines the subword units that will be used for both acoustic and language model training.

```
cd ../../Tokenizer
python train.py hparams/tokenizer_bpe5000.yaml --data_folder=/localscratch/aishell/
```
If not present in the specified data_folder, the dataset will be automatically downloaded there.
This step is not mandatory. We will use the official tokenizer downloaded from the web if you do not 
specify a different tokenizer in the speech recognition recipe. 

2- Train the speech recognizer
```
python train.py hparams/train.yaml --data_folder=/localscratch/aishell/
```

# Performance summary
Results are reported in terms of Character Error Rate (CER). It is not clear from published results whether spaces are kept or removed when computing CER; we therefore report results for both settings. (This can be controlled using `python train.py hparams/train.yaml --remove_spaces={True,False}`.)

[Test Character Error Rate (CER).]
| System | Test CER |
|----------------- | ------|
| Base (remove spaces) | 7.76 |
| Base (keep spaces) | 7.51 |

You can checkout our results (models, training logs, etc,) here:
https://drive.google.com/drive/folders/1zlTBib0XEwWeyhaXDXnkqtPsIBI18Uzs?usp=sharing

# Training Time
It takes about 1h 30 minutes on a NVIDIA V100 (32GB).

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