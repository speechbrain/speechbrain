# MultiWOZ Response Generation with GPT2 Model.
This folder contains the scripts to finetune a gpt based system using MultiWOZ for response generation task.
You can download MultiWOZ at https://github.com/budzianowski/multiwoz.
The data will be automatically download in the specified data_folder.


## Installing Extra Dependencies

Before proceeding, ensure you have installed the necessary additional dependencies. To do this, simply run the following command in your terminal:

```
pip install -r extra_requirements.txt
```

# How to run
```
python train_with_gpt.py hparams/train_gpt.yaml --data_folder=/your/data/folder
```
The data will be automatically download in the specified data_folder.


# Results

| Model | Release | Hyperparams file | Test Cross-entropy Loss | Test PPL | Test BLEU 4| HuggingFace link | Full model link | GPUs |
|:-------------:|:-------------:|:---------------------------:| :-----:| :-----:| :-----:| :-----:| :--------:|:--------:|
| GPT2 | 2023-08-15 | train_gpt.yaml |  1.39 |  4.01 | 2.54e-04 |[model](https://huggingface.co/speechbrain/MultiWOZ-GPT-Response_Generation) | [model](https://www.dropbox.com/sh/vm8f5iavohr4zz9/AACrkOxXuxsrvJy4Cjpih9bQa?dl=0) | 1xV100 16GB |




# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/

# **Citing**
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
