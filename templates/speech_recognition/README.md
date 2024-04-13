# Training a Speech Recognizer

This template implements a simple speech recognizer trained on mini-librispeech.  In particular,  it implements an offline end-to-end attention-based speech recognizer.  A tokenizer is used to detect the word token to estimate. Search replies on beam search coupled with an RNN language model.

Training such a system requires the following steps:

1. Train a tokenizer.
Given the training transcriptions, the tokenizers decide which word pieces allocate for training. Most atomic units are character,  the least atomic units are words.  Most of the time, it is convenient to use tokens that are something in between characters and full words.
SpeechBrain relies on the popular [SentencePiece](https://github.com/google/sentencepiece) for tokenization. To train the tokenizer:

```
cd Tokenizer
python train.py tokenizer.yaml
```

2. Train a LM
After having our target tokens, we can train a language model on top of that. To do it, we need some large text corpus (better if the language domain is the same as the one of your target application). In this example, we simply train the LM on top of the training transcriptions:

```
cd ../LM
python train.py RNNLM.yaml
```

In a real case, training LM is extremely computational demanding. It is thus a good practice to reuse existing LM or fine-tune them.

3. Train the speech recognizer
At this point, we can train our speech recognizer. In this case, we are using a simple CRDNN model with an autoregressive GRU decoder. An attention mechanism is employed between encoding and decoder. The final sequence of words is retrieved with beamsearch coupled with the RNN LM trained in the previous step. To train the ASR:

```
cd ../ASR
python train.py train.yaml
```

This template can help you figure out how to set speechbrain for implementing an e2e speech recognizer. However, in a real case, the system must be trained with much more data to provide acceptable performance. For a competitive recipe with more data, see for instance our recipes on LibriSpeech (https://github.com/speechbrain/speechbrain/tree/develop/recipes/LibriSpeech/ASR).

[For more information, please take a look into the "ASR from scratch" tutorial](https://colab.research.google.com/drive/1aFgzrUv3udM_gNJNUoLaHIm78QHtxdIz)
