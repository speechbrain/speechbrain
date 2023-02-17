# Grapheme-to-phoneme (G2P).
The following models are available:

* `hparams/hparams_g2p_rnn.yaml`: RNN-based (LSTM-encoder, GRU-decoder), with an attention mechanism
Transformer
* `hparams/hparams_g2p_transformer.yaml`: Transformer/Conformer model

The datasets used here are available at the following locations:

* LibriG2P (no stress markers): https://huggingface.co/datasets/flexthink/librig2p-nostress
* LibriG2P (no stress markers, spaces preserved, with homographs): https://huggingface.co/datasets/flexthink/librig2p-nostress-space

The datasets are derived from the LibriSpeech-Alignments dataset (https://zenodo.org/record/2619474#.YbwRGi_73JM), optimized for training G2P models.

Decoding is performed with a beamsearch, optionally enhanced with language models.

To run this recipe, do the following:
> python train.py &lt;hyperparameter file&gt;

Example:
```shell
python train.py hparams/hparams_g2p_transformer.yaml
```

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

Language Models
---------------
The provided G2P model allows for the optional use of a language model trained on
the phoneme space integrated with a beam search in order to obtain a modest improvement
in the results.

To train a language model, use the `train_lm.py` script provided.

For an RNN-based language model:
> python train_lm.py hparams/hparams_lm_rnn.yaml

For a transformer-based language modle:
> python train_lm.py hparams/hparams_lm_transformer.yaml

To use a language model during training or inference
* Copy the language model file from the location indicated by `<save_folder>/<checkpoint>/model.ckpt`
to `<pretrained_path>/lm.ckpt`
* Add`--use_language_model true` to the command line.


Hyperparameter Optimization
---------------------------
This recipe supports hyperparameter optimization via Oríon or other similar tools.
For details on how to set up hyperparameter optimization, refer to the
"Hyperparameter Optimization" tutorial in the Advanced Tutorials section
on the SpeechBrian website:

https://speechbrain.github.io/tutorial_advanced.html

A supplemental hyperparameter file is provided for hyperparameter optimiszation,
which will turn off checkpointing and limit the number of epochs:

hparams/hpopt.yaml

You can download LibriSpeech at http://www.openslr.org/12

Pretrained Models
-----------------
| Release       | hyperparams file           | Sentence Test PER | Homograph % | Model link                                                                           |
|:-------------:|:--------------------------:| --------:| --------------------------------------------------------------------------------------------------:|
| 0.5.12        | train_g2p_rnn.yaml         | 2.72               |  94%        | https://drive.google.com/drive/folders/1jpVDz6Kqtl4qp3_dsuK767mjNlqkIxTH?usp=sharing |
| 0.5.12        | train_g2p_transformer.yaml | 2.89               |  92%        | https://drive.google.com/drive/folders/1lbSjCKUit8H3FCzaDJmfBDJOkcDRH3XI?usp=sharing |

NOTE: Sentence PER is reported as achieved at the end of the sentence training step. Nominal PER on
librispeech data may increase post fine-tuning due to a distribution shift in labeling, if reevaluated.
To replicate the result exactly, train with --homograph_epochs=0.


Pretrained language models can be found at the following URLs:
* **RNN**: https://drive.google.com/drive/folders/1Zv8SNYIXzboFatSRpmoNgRyVXl_6ucir?usp=sharing
* **Transformer**: https://drive.google.com/drive/folders/1MPceslDRVKW7sk1Q6W6nSaWETEAqp5t5?usp=sharing

Training Time
-------------
All reference times are given for a Quattro P5000 GPU. These are rough estimations only - exact training times will vary depending on the hyperparameters chosen and system configuration.

## RNN Models
* **Lexicon**: approx. 6 minutes/epoch for training, < 1min/epoch for evaluation
* **Sentence**: approx. 50 minutes/epoch for training, 55 minutes/epoch for evaluation
## Transformer Models

* **Lexicon**: approx. 7 minutes/epoch for training, approx. 1/2 minutes/epoch for evaluation
* **Sentence**: 25-30 minutes/epoch for training, ~1 hour and 45 minutes/epoch for evaluation

**Note**: To speed up evaluation with the Transformer model, consider reducing the beam size. For fastest
evaluation, use `--beam_search_beam_size 1`.

# Pretrained Models
Pretrained models can be found on the following Google drive:
https://drive.google.com/drive/folders/1nk9ms8cQ5N07wOG4oTi9h5a1dmiPmvnv?usp=sharing


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