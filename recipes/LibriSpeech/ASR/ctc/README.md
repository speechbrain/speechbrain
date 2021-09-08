# LibriSpeech ASR with wav2vec + ctc.
This folder contains the scripts to train a wav2vec based system using LibriSpeech.
You can download LibriSpeech at http://www.openslr.org/12

# How to run
python train_with_wav2vec.py hparams/file.yaml

# Results

| Release | hyperparams file | Test Clean WER | HuggingFace link | Full model link | GPUs |
|:-------------:|:---------------------------:| :-----:| :-----:| :-----:| :--------:|
| 09-09-21 | train_with_wav2vec.yaml | 1.90 | [HuggingFace](https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech) | [Link](https://drive.google.com/drive/folders/1uKpnllZRM3fs5KVAR7HiFlaNhefGdwLb) | 1xV100 32GB |

# Training Time
It takes about 3 hours for an epoch on a rtx8000 (48 GB).

You can find the full experiment folder (i.e., checkpoints, logs, etc) here:
https://drive.google.com/drive/folders/1uKpnllZRM3fs5KVAR7HiFlaNhefGdwLb


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
