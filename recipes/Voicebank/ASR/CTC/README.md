# VoiceBank ASR with CTC models

This folder contains the scripts to train a CTC acoustic model using Voicebank.

Use the `download_vctk()` function in the `voicebank_prepare.py` file to
download and resample the dataset.

## How to run

```bash
python train.py hparams/train.yaml
```

## Results

| Release  | hyperparams file | input type  | Test PER | Model link    | GPUs        |
|:--------:|:----------------:|:-----------:|:--------:|:-------------:|:-----------:|
| 21-02-09 | train.yaml       | `clean_wav` | 10.12    | Not Available | 1xV100 32GB |

## Training Time

About 4 mins for each epoch with a TESLA V100.

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
