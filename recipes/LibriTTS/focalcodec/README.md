# FocalCodec: Low-Bitrate Speech Coding via Focal Modulation Networks

**Project Page**: https://lucadellalib.github.io/focalcodec-web/

This folder contains recipes for training FocalCodec on LibriTTS. You can download LibriTTS from https://www.openslr.org/60/.
FocalCodec is a low-bitrate single-codebook speech codec based on [focal modulation](https://arxiv.org/abs/2203.11926).

For more information, check our papers:

- [FocalCodec: Low-Bitrate Speech Coding via Focal Modulation Networks](https://arxiv.org/abs/2502.04465)

- [FocalCodec-Stream: Streaming Low-Bitrate Speech Coding via Causal Distillation](https://arxiv.org/abs/2509.16195)

<img src="https://raw.githubusercontent.com/lucadellalib/focalcodec/refs/heads/main/focalcodec.png" width="700">

---------------------------------------------------------------------------------------------------------

## Installing Extra Dependencies

Before proceeding, ensure you have installed the necessary additional dependencies.
To do so, simply run the following command in your terminal:

```bash
pip install -r extra_requirements.txt
```

---------------------------------------------------------------------------------------------------------

## Running an Experiment

Training FocalCodec is a two-stage process:

1. **Train the decoder** to reconstruct waveforms from continuous speech representations (WavLM6 in our case).
2. **Train the quantization pipeline** (compressor, quantizer, decompressor) using the same representations.

---------------------------------------------------------------------------------------------------------

### 1. Train the Decoder

```bash
python train_decoder.py hparams/vocos.yaml --data_folder <path-to-dataset>
```

This step trains a decoder to map encoder features back into high-quality audio.
UTMOS, dWER, and speaker similarity are computed on test set to assess the resynthesis performance.

---------------------------------------------------------------------------------------------------------

### 2. Train the Quantization Pipeline

```bash
python train_quantizer.py hparams/bsq.yaml --data_folder <path-to-dataset>
```

This stage trains the compressor, quantizer, and decompressor.
Note that it can be run in parallel with decoder training, since both stages operate on the same continuous encoder representations.

To monitor the end-to-end resynthesis performance during training, you can provide the previously trained decoder checkpoint:

```bash
python train_quantizer.py hparams/bsq.yaml --data_folder <path-to-dataset> --decoder_checkpoint <path-to-decoder-checkpoint>
```

---------------------------------------------------------------------------------------------------------

## Results

Note that this is a SpeechBrain adaptation of the original training code.
Some implementation details may differ, which can lead to slightly different results compared to the original implementation.

For reference, we include the resynthesis results from the paper, obtained on **LibriSpeech test-clean**:

|                                        Checkpoint                                       |  Train Data  | Sample<br/>Rate (kHz) | Token<br/>Rate (Hz) | Codebooks | Bitrate<br/>(kbps) | UTMOS | dWER (%) | Sim  |
| :-------------------------------------------------------------------------------------: | :----------: |:---------------------:|:-------------------:| :-------: |:------------------:| :---: | :------: |:----:|
|   [lucadellalib/focalcodec_50hz](https://huggingface.co/lucadellalib/focalcodec_50hz)   | LibriTTS-960 |          16           |        50.0         |   1x8192  |        0.65        |  4.05 |   2.18   | 97.4 |
|   [lucadellalib/focalcodec_25hz](https://huggingface.co/lucadellalib/focalcodec_25hz)   | LibriTTS-960 |          16           |        25.0         |   1x8192  |        0.33        |  4.14 |   3.30   | 96.3 |
| [lucadellalib/focalcodec_12_5hz](https://huggingface.co/lucadellalib/focalcodec_12_5hz) | LibriTTS-960 |          16           |        12.5         |   1x8192  |        0.16        |  4.22 |   7.94   | 93.9 |

The original training logs can be found at: [https://www.dropbox.com/scl/fo/o652m0qow1hs428ppocx3/ABiZp8xIK4d6iTcl-JXbn0s?rlkey=6cka0iabo2kzjg44if2kdgsvu&st=yqwv7x0w&dl=0](https://www.dropbox.com/scl/fo/o652m0qow1hs428ppocx3/ABiZp8xIK4d6iTcl-JXbn0s?rlkey=6cka0iabo2kzjg44if2kdgsvu&st=yqwv7x0w&dl=0).

The original checkpoints can be found at: [https://huggingface.co/collections/lucadellalib/focalcodec](https://huggingface.co/collections/lucadellalib/focalcodec).

The inference code can be found at: [https://github.com/lucadellalib/focalcodec](https://github.com/lucadellalib/focalcodec).

---------------------------------------------------------------------------------------------------------

## About SpeechBrain

- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/

---------------------------------------------------------------------------------------------------------

## Citing FocalCodec

Please, cite FocalCodec if you use it for your research or business.

```bibtex
@inproceedings{dellalibera2025focalcodec,
    title     = {{FocalCodec}: Low-Bitrate Speech Coding via Focal Modulation Networks},
    author    = {Luca {Della Libera} and Francesco Paissan and Cem Subakan and Mirco Ravanelli},
    booktitle = {Advances in Neural Information Processing Systems},
    year      = {2025},
}
```

```bibtex
@article{dellalibera2025focalcodecstream,
    title   = {{FocalCodec-Stream}: Streaming Low-Bitrate Speech Coding via Causal Distillation},
    author  = {Luca {Della Libera} and Cem Subakan and Mirco Ravanelli},
    journal = {arXiv preprint arXiv:2509.16195},
    year    = {2025},
}
```

---------------------------------------------------------------------------------------------------------

## Citing SpeechBrain

Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@article{speechbrainV1,
  author  = {Mirco Ravanelli and Titouan Parcollet and Adel Moumen and Sylvain de Langen and Cem Subakan and Peter Plantinga and Yingzhi Wang and Pooneh Mousavi and Luca {Della Libera} and Artem Ploujnikov and Francesco Paissan and Davide Borra and Salah Zaiem and Zeyu Zhao and Shucong Zhang and Georgios Karakasidis and Sung-Lin Yeh and Pierre Champion and Aku Rouhe and Rudolf Braun and Florian Mai and Juan Zuluaga-Gomez and Seyed Mahed Mousavi and Andreas Nautsch and Ha Nguyen and Xuechen Liu and Sangeet Sagar and Jarod Duret and Salima Mdhaffar and Ga{{\"e}}lle Laperri{{\`e}}re and Mickael Rouvier and Renato De Mori and Yannick Est{{\`e}}ve},
  title   = {Open-Source Conversational {AI} with {SpeechBrain} 1.0},
  journal = {Journal of Machine Learning Research},
  year    = {2024},
  volume  = {25},
  number  = {333},
  pages   = {1--11},
  url     = {http://jmlr.org/papers/v25/24-0991.html}
}
```

```bibtex
@article{ravanelli2021speechbrain,
  author  = {Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  title   = {{SpeechBrain}: A General-Purpose Speech Toolkit},
  journal = {arXiv preprint arXiv:2106.04624},
  year    = {2021},
  url     = {https://arxiv.org/abs/2106.04624},
}
```
