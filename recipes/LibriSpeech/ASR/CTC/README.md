# LibriSpeech ASR with CTC and pre-trained wav2vec2 models.
This folder contains the scripts to finetune a wav2vec2 based system using LibriSpeech.
You can download LibriSpeech at http://www.openslr.org/12.

**Supported pre-trained wav2vec2:** [SpeechBrain]() and [HuggingFace]()

# How to run
python train_with_wav2vec.py hparams/file.yaml

**If using a HuggingFace pre-trained model, please make sure you have "transformers"
installed in your environment (see extra-requirements.txt)**

# Results

| Release | Hyperparams file | Finetuning Split | Test Clean WER | HuggingFace link | Full model link | GPUs |
|:-------------:|:---------------------------:| :-----:| :-----:| :-----:| :-----:| :--------:|
| 09-09-21 | train_hf_wav2vec.yaml | 960h | 1.90 | Not Avail. | [Link](https://drive.google.com/drive/folders/1pg0QzW-LqAISG8Viw_lUTGjXwOqh7gkl?usp=sharing) | 1xRTX8000 48GB |
| 22-09-22 | train_sb_wav2vec.yaml | 100h | 7.X | Not Avail. | Not Avail. | 1xTesla V100 32GB |

# Training Time
TO UPDATE.

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
