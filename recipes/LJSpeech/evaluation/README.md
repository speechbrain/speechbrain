# Text-to-Speech (with LJSpeech)
This folder contains the recipes for evaluation of existing pretrained text-to-speech systems using ASR-based evaluators and MOS estimation

By default, MOS evaluation is performed using a pretrained Transformer model, as defined in `recipes/SOMOS/ttseval/hparams/train.yaml` and available in pre-trained form on HuggingFace in
https://huggingface.co/flexthink/ttseval-wavlm-transformer

ASR evaluation is performed using the bundled Transformer ASR : https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech

# Tacotron 2
The recipe contains hyperparameters for the evaluation of Tacotron2 in `hparams/tacotron2.yaml`

To perform evaluation, run the following script
```
python evaluate.py --data_folder=/your_folder/LJSpeech-1.1 hparams/tacotron.yaml
```


# FastSpeech2
The recipe contains hyperparameters for the evaluation of FastSpeech2 in `hparams/fastspeech2.yaml`

```
python train.py --data_folder=/your_folder/LJSpeech-1.1 hparams/fastspeech2.yaml
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

