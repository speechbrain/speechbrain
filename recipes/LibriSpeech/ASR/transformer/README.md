# LibriSpeech ASR with Transformers or Whisper models.
This folder contains the scripts to train a Transformer-based speech recognizer or the scripts to fine-tune the Whisper encoder-decoder model.

You can download LibriSpeech at http://www.openslr.org/12

# How to run
```shell
python train_with_whisper.py hparams/train_hf_whisper.yaml
```

```shell
python train.py hparams/transformer.yaml
```

**If using a HuggingFace pre-trained model, please make sure you have "transformers"
installed in your environment (see extra-requirements.txt)**
# Results

| Release | hyperparams file | Test Clean WER | HuggingFace link | Model link | GPUs |
|:-------------:|:---------------------------:| :-----:| :-----:| :-----:| :--------:|
| 24-03-22 | transformer.yaml | 2.27 | [HuggingFace](https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech) | [GoogleDrive](https://drive.google.com/drive/folders/1Nv1OLbHLqVeShyZ8LY9gjhYGE1DBFzFf?usp=sharing) | 4xV100 32GB |
| 24-03-22 | conformer_small.yaml | 2.49 (**only 13.3M parameters**) | [HuggingFace](https://huggingface.co/speechbrain/asr-conformersmall-transformerlm-librispeech) | [GoogleDrive](https://drive.google.com/drive/folders/1I4qntoodHCcj1JNbDrfwFHYcLyu1S5-l?usp=sharing) | 1xV100 32GB |
| 06-12-23 | train_hf_whisper.yaml | 3.60 | Not Avail. | Not Avail. | 1xA100 40GB |

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
