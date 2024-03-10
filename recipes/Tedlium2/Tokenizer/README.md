# Tokenizer.
This folder contains the scripts to train a tokenizer using SentencePiece (https://github.com/google/sentencepiece).
The tokenizer is trained on the top of the Tedlium2 training transcriptions.

You can download Tedlium2 at https://lium.univ-lemans.fr/ted-lium2/


# How to Run

To run the training script, follow these steps:

1. Run the following command, replacing `--data_folder` with the path to your downloaded and unpacked Tedlium2 dataset:

```python
python train.py hparams/tedlium2_500_bpe.yaml --data_folder=/path/to/TEDLIUM --clipped_utt_folder=/path/where/to/store/clipped/TEDLIUM
```

**IMPORTANT**: Please utilize **absolute paths** for both the `data_folder` and the `clipped_utt_folder` because the generated CSV files will be employed in training the ASR model.


2. The script will automatically process the dataset and store a modified version of it in the directory specified by `--clipped_utt_folder`. This modified dataset contains recordings split into individual utterances, making it suitable for Automatic Speech Recognition (ASR) training. You can now use this processed dataset for ASR training as described in the `../ASR/README.md` file.

Make sure to adjust the paths and filenames as needed to match your specific setup and dataset location.

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
