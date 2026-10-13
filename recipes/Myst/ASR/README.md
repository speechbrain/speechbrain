# Myst ASR with Transformers or Whisper models.
This folder contains the scripts to train a Transformer-based speech recognizer or the scripts to fine-tune the Whisper model for the My Science Tutor (MyST).
MyST is one of the largest publicly accessible collections of English children’s speech, comprising approximately 400 hours. It encompasses dialogues between
children and a virtual tutor across eight scientific domains, involving 1,372 students in grades three to five. The corpus is pre-partitioned, ensuring equitable
representation of scientific domains and unique student occurrences within each partition. However, only 45% of utterances are transcribed at the word level.

You can find Myst dataset at https://catalog.ldc.upenn.edu/LDC2021S05

# How to run
```shell
python train_with_whisper.py hparams/train_hf_whisper.yaml # Finetune Whisper
python train_with_whisper.py hparams/train_whisper_lora.yaml # Use LoRa to finetune Whisper
python train.py hparams/transformer.yaml # Train from scratch Transformer model

```

# How to run on test sets only
If you want to run it on the test sets only, you can add the flag `--test_only` to the following command:

```shell
python train_with_whisper.py hparams/train_hf_whisper.yaml --test_only
python train_with_whisper.py hparams/train_whisper_lora.yaml --test_only
python train.py hparams/transformer.yaml --test_only
```

**If using a HuggingFace pre-trained model, please make sure you have "transformers"
installed in your environment (see extra-requirements.txt)**


# Note about data preparation

In accordance with the methodology presented in [1], we offer an optional WER filtering mechanism. This filters out all utterances that exceed a specified threshold, which may result in a longer data preparation time, as every file must be decoded using a pre-trained Whisper model. We highly recommend running the data preparation process only once and saving the resulting CSV files for future use.

Set `enable_wer_filter: False` in the recipe YAML to skip this step while keeping the same preparation script and config file.

Note that this data filtering will take couple of hours to run.

[1] A. A. Attia et al., “Kid-whisper: Towards bridging the performance gap in automatic speech recognition for children vs. adults,” in *Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society*, vol. 7, 2024, pp. 74–80.

# Results

## Whisper Finetuning Result:

Following table contains whisper-finetuning results for 1 epoch using Whisper model, using different configurations.

| Release | Model | Configuration  | hyperparams file | LM | WER | Model link |
| -------------| :-------------|:-------------| :-------------| :-------------| :-------------| :-------------
2025-11-13 | large-v3 | Decoder | train_hf_whisper.yaml | No | 8.36% | [Save](https://cloud.inesc-id.pt/s/eknR4y73RHKSB7F) |
2025-11-13 | medium.en | Decoder | train_hf_whisper.yaml | No | 8.50% | [Save](https://cloud.inesc-id.pt/s/oJeyJCM7R2tGmPG) |
2025-11-13 | medium.en | Encoder + Decoder | train_hf_whisper.yaml |No | 8.75% |[Save](https://cloud.inesc-id.pt/s/px3KWAditRo7wHH) |
2025-11-13 | medium.en | LoRA (r=16) in Decoder | train_whisper_lora.yaml | No | 9.38% | [Save](https://cloud.inesc-id.pt/s/6YrRKPjNpKdMgoW)|





## Transformers

| Release | Model |  hyperparams file | LM | WER | Model link |
| -------------| :-------------| :-------------| :-------------| :-------------| :-------------
2025-11-15 | Transformer | transformer.yaml | LibriSpeech LM | 12.95% | [Save](https://cloud.inesc-id.pt/s/ooG53HSjsTJTZPY) |



# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{speechbrainV1,
  title={Open-Source Conversational AI with SpeechBrain 1.0},
  author={Mirco Ravanelli and Titouan Parcollet and Adel Moumen and Sylvain de Langen and Cem Subakan and Peter Plantinga and Yingzhi Wang and Pooneh Mousavi and Luca Della Libera and Artem Ploujnikov and Francesco Paissan and Davide Borra and Salah Zaiem and Zeyu Zhao and Shucong Zhang and Georgios Karakasidis and Sung-Lin Yeh and Pierre Champion and Aku Rouhe and Rudolf Braun and Florian Mai and Juan Zuluaga-Gomez and Seyed Mahed Mousavi and Andreas Nautsch and Xuechen Liu and Sangeet Sagar and Jarod Duret and Salima Mdhaffar and Gaelle Laperriere and Mickael Rouvier and Renato De Mori and Yannick Esteve},
  year={2024},
  eprint={2407.00463},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2407.00463},
}
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
