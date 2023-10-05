# Tedlium2 ASR with Transformers
This folder contains the scripts to train a Transformer-based speech recognizer

You can download Tedlium2 at https://lium.univ-lemans.fr/ted-lium2/

Please first check ../../Tokenizer. It will prepare the Tedlium2 dataset by splitting the whole Ted recording into utterances-level recorderings, and create 
the csv files for train/dev/test. Then, please point to the csv files in hparams/branchformer.yaml. The data preparition will only run once in ../../Tokenizer,
since it will write additional data (utterances-level recorderings) to the disk. 

# How to run
```shell
python ../../Tokenizer/train.py ../../Tokenizer/hparams/tedlium2_500_bpe.yaml
python train.py hparams/branchformer.yaml

```

# Results

| Release | hyperparams file |  Test WER (No LM) | HuggingFace link | Model link | GPUs |
|:-------------:|:-------------:|:-------------:|:---------------------------:| :-----:| :-----:|
| 23-05-23 | branchformer_large.yaml | 7.9 | Not Avail. | Not Avail. | 4xA100 80GB |


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
