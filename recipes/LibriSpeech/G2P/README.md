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

# How to run
python train.py train/train.yaml

# Results

| Release | hyperparams file | Test PER | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| -----:| --------:| :-----------:|
| 20-05-22 | train.yaml | 7.28% | https://drive.google.com/drive/folders/1nk9ms8cQ5N07wOG4oTi9h5a1dmiPmvnv?usp=sharing | 1xV100 32GB |


# Training Time
About 2 minutes for each epoch with a TESLA V100.


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
