# LibriSpeech ASR with Transducer models.
This folder contains scripts necessary to run an ASR experiment with the LibriSpeech dataset;
Before running this recipe, make sure numba is installed (pip install numba)
You can download LibriSpeech at http://www.openslr.org/12

# How to run it
python train.py train/train.yaml

# Librispeech 100H Results

| Release | hyperparams file | Val. CER | Val. WER | Test WER (test clean) | Model link | GPUs |
|:-------------:|:---------------------------:| ------:| :-----------:| :------------------:| --------:| :-----------:|
| 2020-10-22 | train.yaml |  5.2 | GS: 11.45 | BS (beam=4): 11.03 | Not Available | 1xRTX-8000 48GB |



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