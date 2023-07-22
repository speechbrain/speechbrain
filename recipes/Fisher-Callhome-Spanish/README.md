# Speech Translation on Fisher-Callhome Spanish recipe
This folder contains recipes for tokenization and speech translation with [Fisher-Callhome Spanish](https://catalog.ldc.upenn.edu/LDC2014T23), a 160-hour Spanish-English ST dataset.

## Installing Extra Dependencies

Before proceeding, ensure you have installed the necessary additional dependencies. To do this, simply run the following command in your terminal:

```
pip install -r extra_requirements.txt
```

### How to run
1- Train a tokenizer. The tokenizer takes in input the training translations and determines the subword units that will be used for the ST task, the auxiliary MT task.

```
cd Tokenizer
python train.py hparams/train_bpe_1k.yaml
```

2- Train the speech translator
```
cd ST/transformer
python train.py hparams/transformer.yaml
```

# Performance summary
Results are reported in terms of sacrebleu.

| hyperparams file | dev   | dev2   | test   | ctc_weight | asr_weight | mt_weight | Model | GPUs               |
|:----------------:|:-----:| :-----:| :-----:| :--------: | :--------: | :-------: | :-------: | :----------------: |
| transformer.yaml | 40.67 | 41.51  | 40.30  | 0          | 0          | 0         | Not Avail. | 2xRTX 2080 Ti 11GB |
| transformer.yaml | 47.50 | 48.33  | 47.31  | 1          | 0.3        | 0         | [Model](https://www.dropbox.com/sh/tmh7op8xwthdta0/AACuU9xHDHPs8ToxIIwoTLB0a?dl=0) | 2xRTX 2080 Ti 11GB |
| transformer.yaml | 46.10 | 46.56  | 46.79  | 1          | 0.2        | 0.2       | Not Avail. | 2xRTX 2080 Ti 11GB |
| conformer.yaml   | 46.37 | 47.07  | 46.10  | 0          | 0          | 0         | Not Avail. | 2xRTX 2080 Ti 11GB |
| conformer.yaml   | 48.09 | 48.19  | 48.04  | 1          | 0.3        | 0         | [Model](https://www.dropbox.com/sh/qz33qjr10y351gk/AADApachs3WtDXx67pIz5fCZa?dl=0) | 1xTesla A100 (works with 2xRTX 2080 Ti) |

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
