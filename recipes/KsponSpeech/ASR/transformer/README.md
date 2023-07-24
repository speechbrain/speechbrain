# KsponSpeech ASR with Transformers

This folder contains the scripts to train a Transformer-based speech recognizer using KsponSpeech.

You can download KsponSpeech at [Link](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=123)

# How to run
Before start training, set pretrained lm and tokenizer path in the YAML file to the proper path (i.e. the directory where trained tokenizer and language model exist)

This is set to huggingface repository as a default. Pretrained models will be downloaded from the repository.

```YAML
pretrained_lm_tokenizer_path: /path/to/pretrained/models
```

Also, data_foler in the YAML file should point to the results of ksponspeech_prepare.py
```YAML
data_folder: /path/to/data/prep/results
```
Run the following to start training
```bash
python train.py hparams/conformer_medium.yaml
```

# Results
| Release  |   hyperparams file    | eval clean WER | eval other WER | eval clean CER | eval other CER |                                   HuggingFace link                                   |                                               Model link                                                |    GPUs     |  Training time  |
| :------: | :-------------------: | :------------: | :------------: | :------------: | :------------: | :----------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: | :---------: | :-------------: |
| 01-23-23 | conformer_medium.yaml |     20.47%     |     25.18%     |     7.33%      |     7.99%      | [HuggingFace](https://huggingface.co/speechbrain/asr-conformer-transformerlm-ksponspeech) | [DropBox](https://www.dropbox.com/sh/uibokbz83o8ybv3/AACtO5U7mUbu_XhtcoOphAjza?dl=0) | 6xA100 80GB | 2 days 13 hours |

# PreTrained Model + Easy-Inference
You can find the pre-trained model with an easy-inference function on HuggingFace: [HuggingFace](https://huggingface.co/speechbrain/asr-conformer-transformerlm-ksponspeech)

# About SpeechBrain
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


# Citing SpeechBrain
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
