# Speech Translation on MuST-C version 1 recipe
This folder contains recipes for tokenization and speech translation with [Must-C version 1](https://ict.fbk.eu/must-c/), a multilingual speech translation corpus whose size and quality facilitates the training of end-to-end systems for speech translation from English into several languages.

### How to run
0- Install extra dependencies
```
pip install -r extra_requirements.txt
```

1- Train a tokenizer. The tokenizer takes in input the training translations and determines the subword units that will be used for the ST task, the auxiliary MT task.

```
cd Tokenizer
python train.py hparams/train_bpe_8k.yaml
```

2- Train the speech translator
```
cd ST/transformer
python train.py hparams/transformer.yaml
```

# Performance summary
Results are reported in terms of sacrebleu.

| hyperparams file | language | tst_com | tst_he | ctc_weight | asr_weight | mt_weight | Model |        GPUs        |
|:----------------:|:--------:|:-------:|:------:|:----------:|:----------:|:---------:|:-----:|:------------------:|
| transformer.yaml |    de    |  19.47  | 17.77  |      0     |      0     |      0    | Not Avail. | 1xRTX 3090 Ti 25GB |
| transformer_wav.yaml |    de    |  22.07  | 20.95  |      0     |      0     |      0    | Not Avail. | 1xRTX 3090 Ti 25GB |

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
2
