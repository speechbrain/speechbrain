# SpeechEnhancement with the DNS dataset (spectral map).
This folder contains the scripts to train a speech enhancement system with spectral map.
You can download the dataset from here: https://github.com/microsoft/DNS-Challenge

# How to run
python train.py train/params_CNNTransformer.yaml  
python train.py train/params_CNN.yaml 

# Results
| Release | hyperparams file | STOI | PESQ | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| -----:| --------:| :-----------:|
| 20-05-22 |  params_CNN.yaml |  --.- | --.- | Not Available | 1xV100 32GB |
| 20-05-22 |  params_CNNTransformer.yaml |  --.- | --.- | Not Available | 1xV100 32GB |


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
