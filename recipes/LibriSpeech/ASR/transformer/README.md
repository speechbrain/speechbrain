# LibriSpeech ASR with Transformers.
This folder contains the scripts to train a Transformer-based speech recognizer
using LibriSpeech.

You can download LibriSpeech at http://www.openslr.org/12


# How to run
python train.py train/train.yaml

# Results

| Release | hyperparams file | Test Clean WER | HuggingFace link | Model link | GPUs |
|:-------------:|:---------------------------:| :-----:| :-----:| :-----:| :--------:|
| 20-05-22 | transformer.yaml | 2.46 | [HuggingFace](https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech) | [GoogleDrive](https://drive.google.com/drive/folders/1ZudxqMWb8VNCJKvY2Ws5oNY3WI1To0I7?usp=sharing) | 2xRTX8000 42GB |

# Training Time
It takes about 1 hour 35 minutes for each epoch on 4 NVDIA V100 (32GB).


# PreTrained Model + Easy-Inference
You can find the pre-trained model with an easy-inference function on HuggingFace:
- https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech
- https://huggingface.co/speechbrain/asr-crdnn-transformerlm-librispeech
- https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech

You can find the full experiment folder (i.e., checkpoints, logs, etc) here:
https://drive.google.com/drive/folders/15uUZ21HYnw4KyOPW3tx8bLrS9RoBZfS7?usp=sharing

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
