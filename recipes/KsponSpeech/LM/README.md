# Language Model with KsponSpeech

This folder contains recipes for training language models for the KsponSpeech Dataset. It supports a Transformer-based LM.

You can download KsponSpeech at https://aihub.or.kr/aidata/105/download

# How to run
Set tokenizer_file in the yaml files to the directory where the trained tokenizer is located. This is set to huggingface repository as a default.

Also, set data_folder in the yaml file to the result of ksponspeech_prepare.py.

Run the following to start training the language model.

```bash
python train.py hparams/transformer.yaml
```
# Results

| Release | hyperparams file | eval clean loss | eval other loss | Model link | GPUs |Training time|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|09-05-21|transformer.yaml|4.41|4.68|[GoogleDrive](https://drive.google.com/drive/folders/1NmpE7aThLogxVhPrpWqFcw1dzd-fqzVA?usp=sharing)|1xA100 80GB|5 hours 35 mins|

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
