# Grapheme-to-phoneme (G2P).
The following models are available:
* RNN-based (LSTM-encoder, GRU-decoder), with an attention mechanism
  * `hparams/hparams_attnrnn_librig2p_nostress.yaml`: LibriG2P (no stress markers)
  * `hparams/hparams_attnrnn_librig2p_nostress_tok.yaml`: LibriG2P (no stress markers, tokenization)
* Convolutional
  * `hparams/hparams_conv_librig2p_nostress.yaml`
* Transformer
  * `hparams/hparams_transformer_librig2p_nostress.yaml`: LibriG2P (no stress markers)
  * `hparams/hparams_transformer_librig2p_nostress_tok.yaml`: LibriG2P (no stress markers, tokenization)

The datasets used here are available at the following locations:

* LibriG2P (no stress markers): https://github.com/flexthink/librig2p-nostress
* LibriG2P (no stress markers, spaces preserved, with homographs): https://github.com/flexthink/librig2p-nostress

Decoding is performed with a beamsearch, optionally enhanced with language models.

To run this recipe, do the following:
> python train.py <hyperparameter file>
Example:
> python train.py hparams/hparams_attnrnn_librig2p_nostress.yaml

RNN Model
---------
With the default hyperparameters, the system employs an LSTM encoder.
The decoder is based on a standard  GRU. The neural network is trained with
negative-log.

Transformer Model
-----------------
With the default hyperparameters, the system employs a Conformer architecture
with a convolutional encoder and a standard Transformer decoder.

The choice of Conformer vs Transformer is controlled by the
transformer_encoder_module parameter.

Homograph Disambiguation
------------------------
Both RNN-based and Transformer-based models are capable of sentence-level
hyperparameter disambiguation. Fine-tuning on homographs relies on an additional
weighted loss computed only on the homograph and, therefore, requires a dataset
in which the homograph is labelled, such as LibriG2P.

The experiment file is flexible enough to support a large variety of
different systems. By properly changing the parameter files, you can try
different encoders, decoders,  and many other possible variations.

Hyperparameter Optimization
---------------------------
This recipe supports hyperparameter optimization via Oríon or other similar tools.
For details on how to set up hyperparameter optimization, refer to the
"Hyperparameter Optimization" tutorial in the Advanced Tutorials section
on the SpeechBrian website:

https://speechbrain.github.io/tutorial_advanced.html

A supplemental hyperparameter file is provided for hyperparameter optimiszation,
which will turn off checkpointing and limit the number of epochs:

hparams/hpfit.yaml

You can download LibriSpeech at http://www.openslr.org/12

# Training Time
All reference times are given for a Quattro P5000 GPU. These are rough estimations only - exact 
training times will vary depending on the hyperparameters chosen and system configuration

## RNN Models
* **Lexicon**: approx. 6 minutes/epoch for training, < 1min/epoch for evaluation
* **Sentence**: approx. 50 minutes/epoch for training, 55 minutes/epoch for evaluation
## Transformer Models

* **Lexicon**: approx. 7 minutes/epoch for training, approx. 1/2 minutes/epoch for evaluation
* **Sentence**: 25-30 minutes/epoch for training, ~1 hour and 45 minutes/epoch for evaluation

**Note**: To speed up evaluation with the Transformer model, consider reducing the beam size. For fastest
evaluation, use `--beam_search_beam_size 1`.

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
