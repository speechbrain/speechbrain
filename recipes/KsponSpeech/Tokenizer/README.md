# Tokenizer
This folder contains the scripts to train a tokenizer using SentencePiece (https://github.com/google/sentencepiece). The tokenizer is trained on the top of the KsponSpeech training transcriptions.

# How to run
```
python train.py train/5K_unigram_subword_bpe.yaml
```
# Model link
- 5K unigram model: [HuggingFace](https://huggingface.co/ddwkim/asr-conformer-transformerlm-ksponspeech/blob/main/tokenizer.ckpt)

The output folder with the logs and the tokenizers is available [here](https://drive.google.com/drive/folders/1zNGKDvHlLjQdUPrqP66vpD5RN9IIX6RC?usp=sharing).

# About SpeechBrain
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


# Citing SpeechBrain
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
