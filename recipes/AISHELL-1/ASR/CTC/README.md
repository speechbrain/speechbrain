# AISHELL-1 ASR with CTC.
This folder contains a CTC-wav2vec2 recipe for speech recognition with [AISHELL-1](https://www.openslr.org/33/), a 150-hour Chinese ASR dataset.

### How to run
1- Tokenizer/Dataset
A pretrained tokenizer from [huggingface](https://huggingface.co/bert-base-chinese) is used and can be downloaded
automatically.

This step is not mandatory. We will use the official tokenizer downloaded from the web if you do not
specify a different tokenizer in the speech recognition recipe.

2- Train the speech recognizer
```
python train_with_wav2vec.py hparams/train_with_wav2vec.yaml
```

Make sure to have "transformers" installed.

# Performance summary
Results are reported in terms of Character Error Rate (CER).

| hyperparams file | LM | Test CER | Dev CER | GPUs |
|:--------------------------:|:-----:| :-----:| :-----:| :-----: |
| train_with_wav2vec.yaml | No | 5.06 | 4.52 | 1xRTX 8000 Ti 48GB |

You can checkout our results (models, training logs, etc,) [here](https://www.dropbox.com/sh/e4bth1bylk7c6h8/AADFq3cWzBBKxuDv09qjvUMta?dl=0)

# Training Time
It takes about 2h on 1 RTX 8000 (48GB)

# PreTrained Model + Easy-Inference
You can find the pre-trained model with an easy-inference function on HuggingFace
- https://huggingface.co/speechbrain/asr-wav2vec2-ctc-aishell

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
