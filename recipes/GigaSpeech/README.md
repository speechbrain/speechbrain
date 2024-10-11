# Experimenting with the GigaSpeech dataset

GigaSpeech is an evolving, multi-domain English speech recognition corpus with 10,000 hours of high quality labeled audio suitable for supervised training, and 40,000 hours of total audio suitable for semi-supervised and unsupervised training (this implementation contains only labelled data for now). However, the data access is gated, meaning, you need to request access to it.

# Data access and download

SpeechBrain supports two ways of dealing with the GigaSpeech dataset:
1. [HuggingFace dataset](https://huggingface.co/datasets/speechcolab/gigaspeech/). For HuggingFace note that **you must use** the HuggingFace client to log in first before running the recipe.
2. [Original Github](https://github.com/SpeechColab/GigaSpeech).

You simply need to follow the instructions on either of the above links. **We strongly
recomment using HuggingFace as the download speed for people outside of China is
much quicker**.