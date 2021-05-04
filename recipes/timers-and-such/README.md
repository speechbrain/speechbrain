# SLU recipes for Timers and Such v1.0
This folder contains recipes for spoken language understanding (SLU) with [Timers and Such v1.0](https://zenodo.org/record/4623772#.YGeMMHVKg5k), an SLU dataset with a (train/dev/test) set of synthetic speech and a (train/dev/test) set of real speech.

### LM recipe
This recipe trains a language model (LM) on Timers and Such transcripts. (It is not necessary to run this before running the other recipes, as they download a trained checkpoint.)

### Decoupled recipe
The "decoupled" recipe uses an ASR model (using the LibriSpeech seq2seq recipe) to map speech to text and a separate NLU model, trained on the true transcripts rather than the ASR output, to map text to semantics using an attention-based seq2seq model.
The ASR model uses either the LibriSpeech LM (`train_LS_LM.yaml`) or the Timers and Such LM (`train_TAS_LM.yaml`).

```
cd decoupled
python train.py hparams/{train_LS_LM, train_TAS_LM}.yaml
```

### Multistage recipe
The "multistage" recipe is similar to the decoupled recipe, but instead of using the true transcripts to train the NLU model, we use transcripts predicted by the ASR model, again using either the LibriSpeech LM (`train_LS_LM.yaml`) or the Timers and Such LM (`train_TAS_LM.yaml`).

```
cd multistage
python train.py hparams/{train_LS_LM, train_TAS_LM}.yaml
```

### Direct recipe
The "direct" maps the input speech to directly to semantics using a seq2seq model. The encoder is pre-trained using the LibriSpeech seq2seq recipe.

```
cd direct
python train.py hparams/train.yaml
```

The recipe can also be used using a pretrained wav2vec 2.0 model (finetuned for ASR using LibriSpeech) as an encoder:

```
cd direct
python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml
```

# Performance summary

The table below reports the performance on the test-real and test-synth subsets achieved when training with both real and synthetic data (train-real+train-synth), as well as performance on the all-real subset when training with train-synth only.
(Additional results using train-real only and train-synth only can be found in the paper linked below.)

[Sentence accuracy on Timers and Such v1.0, measured using 5 random seeds.]
| System | test-real | test-synth | all-real
|----------------- | ------------ | ------|------|
| Decoupled (LibriSpeech LM) | 31.4% ± 4.3% | 22.5% ± 2.1% | 26.8% ± 3.3% |
| Decoupled (Timers and Such LM) | 46.8% ± 2.1% | 38.4% ± 1.3% | 44.6% ± 2.4% |
| Multistage (LibriSpeech LM) | 67.8% ± 1.4% | 79.4% ± 0.4% | 64.6% ± 0.7% |
| Multistage (Timers and Such LM) | 72.6% ± 1.6% | 85.4% ± 0.2% | 69.9% ± 6.0% |
| Direct | 77.5% ± 1.6% | 96.7% ± 0.3% | 68.9% ± 5.4% |

Additionally, we report three improved results with the direct recipe trained only on the train-real subset. 
The first is identical to the baseline CRDNN direct model, except it uses a smaller batch size and is trained for 80 epochs. 
The second uses the "Base" unsupervised wav2vec 2.0 model as an encoder, with the transformer layers unfrozen and the initial CNN layers frozen.
The third uses the "960 Hr" variant of wav2vec 2.0, which is finetuned on LibriSpeech using ASR labels.

| Encoder | test-real | test-synth
|----------------- | ------------ | ------|
| CRDNN (batch size 8) | 89.2% ± 0.8% | 79.6% ± 2.9% |
| wav2vec 2.0 "Base" | 92.7% ± 1.0% | none |
| wav2vec 2.0 "960 Hr" | 94.0% ± 1.2% | none |

You can find the output folders (model, logs, etc) here: https://drive.google.com/drive/folders/1x2crmemZj2uxdzyOM_nlfuHxlTCP-9_-?usp=sharing


# Citation

The dataset and baseline models are described in the paper below. If you found this code or the dataset useful, you can use this bibtex entry to cite the paper:

[Timers and Such: A Practical Benchmark for Spoken Language Understanding with Numbers](https://arxiv.org/abs/2104.01604)

```
@misc{lugosch2021timers,
      title={Timers and Such: A Practical Benchmark for Spoken Language Understanding with Numbers},
      author={Lugosch, Loren and Papreja, Piyush and Ravanelli, Mirco and Heba, Abdelwahab and Parcollet, Titouan},
      year={2021},
      eprint={2104.01604},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
