# Language Model with MATBN
This folder contains recipes for training language models for the MATBN Dataset.
It supports both an RNN-based LM and a Transformer-based LM.

# How to run:
```
python train.py hparams/RNNLM.yaml --tokenizer_file=<tokenizer ckpt location> --data_folder=<prepared data location>
python train.py hparams/TransformerLM.yaml --tokenizer_file=<tokenizer ckpt location>  --data_folder=<prepared data location><prepared data location>
```

| hyperparams file | Test PPL |  GPUs | Training time |
| :--- | :---: | :---: | :---: |
| RNNLM.yaml | 5.78 | 1xGTX1080 8G | 1 hours 43 mins |
| TransformerLM.yaml | 5.78 | 1xGTX1080 8G | 1 hours 31 mins |

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