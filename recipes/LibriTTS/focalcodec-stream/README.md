# FocalCodec-Stream: Streaming Low-Bitrate Speech Coding via Causal Distillation

**Project Page**: https://lucadellalib.github.io/focalcodec-stream-web/

This folder contains recipes for training FocalCodec-Stream on LibriTTS. You can download LibriTTS from https://www.openslr.org/60/.

Note that for simplicity we use only LibriTTS here. However, the original model was trained on a combination of LibriTTS and LibriLight.
For downloading and preparing LibriLight, see the recipes under Libri-Light.

FocalCodec-Stream is a low-bitrate single-codebook **streaming** speech codec based on [focal modulation](https://arxiv.org/abs/2203.11926).

For more information, check our papers:

- [FocalCodec: Low-Bitrate Speech Coding via Focal Modulation Networks](https://arxiv.org/abs/2502.04465)

- [FocalCodec-Stream: Streaming Low-Bitrate Speech Coding via Causal Distillation](https://arxiv.org/abs/2509.16195)

<img src="https://cas-bridge.xethub.hf.co/xet-bridge-us/68cec895158e9d135dba9625/0b3a9404850c0081f006cb8568029cd7fbd78e2adad1312ab3902da71e6d294c?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=cas%2F20260420%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260420T034348Z&X-Amz-Expires=3600&X-Amz-Signature=9d5b5b2683f7ef336b20929f033eec484e199ceaccb86dd42f664446acb4a75f&X-Amz-SignedHeaders=host&X-Xet-Cas-Uid=63195d0582e7eec0eac040e3&response-content-disposition=inline%3B+filename*%3DUTF-8%27%27focalcodec-stream.png%3B+filename%3D%22focalcodec-stream.png%22%3B&response-content-type=image%2Fpng&x-amz-checksum-mode=ENABLED&x-id=GetObject&Expires=1776660228&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc3NjY2MDIyOH19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2FzLWJyaWRnZS54ZXRodWIuaGYuY28veGV0LWJyaWRnZS11cy82OGNlYzg5NTE1OGU5ZDEzNWRiYTk2MjUvMGIzYTk0MDQ4NTBjMDA4MWYwMDZjYjg1NjgwMjljZDdmYmQ3OGUyYWRhZDEzMTJhYjM5MDJkYTcxZTZkMjk0YyoifV19&Signature=umOK5Wo0NLTdlBNFfn7TYppcyo7g8d4A91-p1BAaAkY3Qza1BZEZjXbqGWY7xzBEKr-tRtGjxqjnHYx4wYK27yBYSuYLL2Jc2nEm9PSRjb%7EMKvSobFBnb4pqgZJ7sRdg5DeQXSxoxg9dTqQc9LcUwGIo4ywMehaX3yJg3yg5%7E7QVmoVVtqH7NMiiM0tMorOqvcUbn%7E2SuUBf7KZswCk2apXLLhyK3JVHYusLmrNybrt1CP54rcr3t4XTaSJWDBQ6vdQ5IBm9JbFs-988Vzu3Stla14hEP3HUGgzAiKfAhLYE8jkqfd3sPhuXieZo0rMALACUbJFQzn99mHmPF2WxFQ__&Key-Pair-Id=K2L8F4GPSG1IFC" width="700">

---------------------------------------------------------------------------------------------------------

## Installing Extra Dependencies

Before proceeding, ensure you have installed the necessary additional dependencies.
To do so, simply run the following command in your terminal:

```bash
pip install -r extra_requirements.txt
```

---------------------------------------------------------------------------------------------------------

## Running an Experiment

Training FocalCodec-Stream is a five-stage process:

1. **Train the decoder** to reconstruct waveforms from continuous speech representations (WavLM6 in our case).
2. **Distill the positional embeddings** to obtain causal positional embeddings from full-context ones.
3. **Distill the encoder** to obtain a causal encoder from the full-context encoder.
4. **Train the quantization pipeline** (compressor, quantizer, decompressor) using the distilled encoder representations.
5. **Train jointly** the encoder, compressor, quantizer, decompressor, and refiner modules.

---------------------------------------------------------------------------------------------------------

### 1. Train the Decoder

```bash
python train_decoder.py hparams/wavenext.yaml --data_folder <path-to-dataset>
```

This stage trains a causal decoder to map encoder features back into high-quality audio.
UTMOS, dWER, and speaker similarity are computed on test set to assess the resynthesis performance.

---------------------------------------------------------------------------------------------------------

### 2. Distill the Positional Embeddings

```bash
python train_posemb.py hparams/conv_posemb.yaml --data_folder <path-to-dataset>
```

This stage distills the full-context positional embeddings into causal ones while keeping the rest of the encoder fixed.
Note that it can be run in parallel with decoder training, since both stages operate on the same continuous encoder representations.

---------------------------------------------------------------------------------------------------------

### 3. Distill the Encoder

```bash
python train_encoder.py hparams/wavlm.yaml --data_folder <path-to-dataset> \
--student_encoder_posemb_checkpoint <path-to-distilled-posemb-checkpoint>
```

This stage distills the full-context encoder into a causal encoder using the same continuous speech representations.
Note that it can be run in parallel with decoder training, since both stages operate on the same continuous encoder representations.

---------------------------------------------------------------------------------------------------------

### 4. Train the Quantization Pipeline

```bash
python train_quantizer.py hparams/bsq.yaml --data_folder <path-to-dataset> \
--encoder_checkpoint <path-to-distilled-encoder-checkpoint>
```

This stage trains the compressor, quantizer, and decompressor.
Note that it can be run in parallel with decoder training, since both stages operate on the same continuous encoder representations.

To monitor the end-to-end resynthesis performance during training, you can provide the previously trained decoder checkpoint:

```bash
python train_quantizer.py hparams/bsq.yaml --data_folder <path-to-dataset> \
--decoder_checkpoint <path-to-decoder-checkpoint>
```

---------------------------------------------------------------------------------------------------------

### 5. Joint Fine-Tuning

```bash
python train_joint.py hparams/joint.yaml --data_folder <path-to-dataset> \
--encoder_checkpoint <path-to-distilled-encoder-checkpoint> \
--compressor_checkpoint <path-to-compressor-checkpoint> \
--quantizer_checkpoint <path-to-quantizer-checkpoint> \
--decompressor_checkpoint <path-to-decompressor-checkpoint>
```

This stage reduces the distribution shift between the causally distilled encoder features and the decoder trained on full-context representations.
Note that it can be run in parallel with decoder training, since both stages operate on the same continuous encoder representations.

To monitor the end-to-end resynthesis performance during training, you can provide the previously trained decoder checkpoint:

```bash
python train_joint.py hparams/joint.yaml --data_folder <path-to-dataset> \
--encoder_checkpoint <path-to-distilled-encoder-checkpoint> \
--compressor_checkpoint <path-to-compressor-checkpoint> \
--quantizer_checkpoint <path-to-quantizer-checkpoint> \
--decompressor_checkpoint <path-to-decompressor-checkpoint> \
--decoder_checkpoint <path-to-decoder-checkpoint>
```

---------------------------------------------------------------------------------------------------------

## Results

Note that this is a SpeechBrain adaptation of the original training code.
Some implementation details may differ, which can lead to slightly different results compared to the original implementation.

For reference, we include the resynthesis results from the paper, obtained on **LibriSpeech test-clean**:

|                                                Checkpoint                                                 | Train Data  | Sample<br/>Rate (kHz) | Token<br/>Rate (Hz) | Latency<br/>(ms) | Codebooks | Bitrate<br/>(kbps) | UTMOS | dWER (%) | Sim  |
|:---------------------------------------------------------------------------------------------------------:|:-----------:|:---------------------:|:-------------------:|:----------------:|:---------:|:------------------:|:-----:|:--------:|:----:|
| [lucadellalib/focalcodec_50hz_2k_causal](https://huggingface.co/lucadellalib/focalcodec_50hz_2k_causal)   | LibriLight  |          24           |        50.0         |        80        |  1x2048   |        0.55        | 3.88  |   4.63   | 96.1 |
| [lucadellalib/focalcodec_50hz_4k_causal](https://huggingface.co/lucadellalib/focalcodec_50hz_4k_causal)   | LibriLight  |          24           |        50.0         |        80        |  1x4096   |        0.60        | 3.87  |   4.39   | 96.3 |
| [lucadellalib/focalcodec_50hz_65k_causal](https://huggingface.co/lucadellalib/focalcodec_50hz_65k_causal) | LibriLight  |          24           |        50.0         |        80        |  1x65536  |        0.80        | 3.85  |   3.68   | 97.0 |

The original training logs can be found at: TODO.

The original checkpoints can be found at: [https://huggingface.co/collections/lucadellalib/focalcodec](https://huggingface.co/collections/lucadellalib/focalcodec).

The inference code can be found at: [https://github.com/lucadellalib/focalcodec](https://github.com/lucadellalib/focalcodec).

---------------------------------------------------------------------------------------------------------

## About SpeechBrain

- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/

---------------------------------------------------------------------------------------------------------

## Citing FocalCodec-Stream

Please, cite FocalCodec-Stream if you use it for your research or business.

```bibtex
@inproceedings{dellalibera2025focalcodec,
    title     = {{FocalCodec}: Low-Bitrate Speech Coding via Focal Modulation Networks},
    author    = {Luca {Della Libera} and Francesco Paissan and Cem Subakan and Mirco Ravanelli},
    booktitle = {Advances in Neural Information Processing Systems},
    year      = {2025},
}
```

```bibtex
@inproceedings{dellalibera2026focalcodecstream,
    title     = {{FocalCodec-Stream}: Streaming Low-Bitrate Speech Coding via Causal Distillation},
    author    = {Luca {Della Libera} and Cem Subakan and Mirco Ravanelli},
    booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
    year      = {2026},
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
