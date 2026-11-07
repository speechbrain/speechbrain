# VoiceBank Speech Enhancement with MP-SENet

This recipe trains MP-SENet ([Lu et al., Interspeech 2023](https://arxiv.org/abs/2305.13686)) on VoiceBank-DEMAND at 16 kHz.
MP-SENet denoises the magnitude and phase spectra in parallel. It is trained adversarially with a metric discriminator that learns to predict PESQ.

The model is in `speechbrain/lobes/models/MPSENet.py`. Its time-frequency blocks take any sequence model, set with `seq_block` in the hparams.
This recipe uses Conformer blocks, as described in the paper.

## Results

Experiment Date | PESQ | STOI
-|-|-
TODO | TODO | TODO

The full experiment folder (checkpoints, logs, enhanced files): TODO

## How to Run

```bash
pip install -r extra_requirements.txt
python train.py hparams/train.yaml --data_folder /path/to/noisy-vctk-16k
```

`data_folder` must have the 16 kHz layout produced by `download_vctk()` in `voicebank_prepare.py`:

```bash
python -c "from voicebank_prepare import download_vctk; download_vctk('/path/to/data')"
```

Enhanced test files are written to `results/MP-SENet/<seed>/enhanced_wavs`.

Training follows the original setup: 2 s random crops, batch size 4, AdamW (lr 5e-4, betas 0.8/0.99), and an LR decay of 0.99 per epoch.
A discriminator step and a generator step run on every batch.
The discriminator targets need PESQ of each enhanced crop, computed on CPU. This makes training slower than the GPU forward pass alone.

## Differences from the original implementation

- Validation uses held-out training speakers (from `voicebank_prepare.py`). The original code validates on the test set.
- The attention and GRU layers of the original code run along the wrong tensor axis (missing `batch_first=True`). They are fixed here; see the docstring of `MPSENet.py`.
- The authors' released checkpoints use an attention + GRU block (`GRUTransformerBlock`), not the Conformer from the paper.
  They can be loaded with `seq_block: !name:speechbrain.lobes.models.MPSENet.GRUTransformerBlock` and `batch_first: False`, after renaming their keys.
  Converted this way, the released VoiceBank checkpoint produces the same output as the original code.

## About SpeechBrain

- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/

## Citing

```bibtex
@inproceedings{lu2023mp,
  title={{MP-SENet}: A Speech Enhancement Model with Parallel Denoising of Magnitude and Phase Spectra},
  author={Lu, Ye-Xin and Ai, Yang and Ling, Zhen-Hua},
  booktitle={Proc. Interspeech},
  pages={3834--3838},
  year={2023}
}

@misc{speechbrainV1,
  title={Open-Source Conversational AI with SpeechBrain 1.0},
  author={Mirco Ravanelli and Titouan Parcollet and Adel Moumen and Sylvain de Langen and Cem Subakan and Peter Plantinga and Yingzhi Wang and Pooneh Mousavi and Luca Della Libera and Artem Ploujnikov and Francesco Paissan and Davide Borra and Salah Zaiem and Zeyu Zhao and Shucong Zhang and Georgios Karakasidis and Sung-Lin Yeh and Pierre Champion and Aku Rouhe and Rudolf Braun and Florian Mai and Juan Zuluaga-Gomez and Seyed Mahed Mousavi and Andreas Nautsch and Xuechen Liu and Sangeet Sagar and Jarod Duret and Salima Mdhaffar and Gaelle Laperriere and Mickael Rouvier and Renato De Mori and Yannick Esteve},
  year={2024},
  eprint={2407.00264},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2407.00264},
}
```
