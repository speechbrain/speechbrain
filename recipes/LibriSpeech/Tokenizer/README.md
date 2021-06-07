# Tokenizer.
This folder contains the scripts to train a tokenizer using SentencePiece (https://github.com/google/sentencepiece).
The tokenizer is trained on the top of the librispeech training transcriptions.

You can download LibriSpeech at http://www.openslr.org/12


# How to run
python train.py train/1K_unigram_subword_bpe.yaml  
python train.py train/5K_unigram_subword_bpe.yaml
