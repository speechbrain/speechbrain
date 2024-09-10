# VoiceBank ASR with CTC models

This folder contains the scripts to train a CTC acoustic model using Voicebank.

Use the `download_vctk()` function in the `voicebank_prepare.py` file to
download and resample the dataset.

## How to run

```bash
python train.py hparams/train.yaml --data_folder=your/data/folder --jit
```

**Note on Compilation**:
Enabling the just-in-time (JIT) compiler significantly improves code performance, resulting in a 50-60% speed boost. We highly recommend utilizing the JIT compiler for optimal results.
This speed improvement is observed specifically when using the CRDNN model.

## Results

| Release  | hyperparams file | input type  | Test PER | Model link    | GPUs        |
|:--------:|:----------------:|:-----------:|:--------:|:-------------:|:-----------:|
| 21-02-09 | train.yaml       | `clean_wav` | 10.12    | Not Available | 1xV100 32GB |

You can find the output folders with the training logs and checkpoints [here](https://www.dropbox.com/sh/w4j0auezgmmo005/AAAjKcoJMdLDp0Pqe3m7CLVaa?dl=0)

## Training Time

About 4 mins for each epoch with a TESLA V100.

# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{speechbrainV1,
  title={Open-Source Conversational AI with SpeechBrain 1.0},
  author={Mirco Ravanelli and Titouan Parcollet and Adel Moumen and Sylvain de Langen and Cem Subakan and Peter Plantinga and Yingzhi Wang and Pooneh Mousavi and Luca Della Libera and Artem Ploujnikov and Francesco Paissan and Davide Borra and Salah Zaiem and Zeyu Zhao and Shucong Zhang and Georgios Karakasidis and Sung-Lin Yeh and Pierre Champion and Aku Rouhe and Rudolf Braun and Florian Mai and Juan Zuluaga-Gomez and Seyed Mahed Mousavi and Andreas Nautsch and Xuechen Liu and Sangeet Sagar and Jarod Duret and Salima Mdhaffar and Gaelle Laperriere and Mickael Rouvier and Renato De Mori and Yannick Esteve},
  year={2024},
  eprint={2407.00463},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2407.00463},
}
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
