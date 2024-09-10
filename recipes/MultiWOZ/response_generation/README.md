# MultiWOZ Response Generation with LLM Model.
This folder contains the scripts to finetune a LLM using MultiWOZ for the response generation task.
You can download MultiWOZ at https://github.com/budzianowski/multiwoz.
The data will be automatically downloaded in the specified data_folder.
Supported LLM models are:
 - GPT
 - LLAMA2


## Installing Extra Dependencies

Before proceeding, ensure you have installed the necessary additional dependencies. To do this, simply run the following command in your terminal:
> **Note**
> For the Llama2 recipe, transformers and peft libraries should follow the versions mentioned in the extra_requirements.

```
cd recipes/MultiWOZ/response_generation/[LLM_model]
pip install -r extra_requirements.txt
```
> **Note**
> “Llama 2 is licensed under the LLAMA 2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.”
>
> Use of the llama2 model is governed by the Meta license. In order to download the model weights and tokenizer, please visit the [website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and accept the License before starting training the llama2 model. Getting access to the original weights is usually very fast. Sometimes, It took longer to get access to the HF repo. Before proceeding, make sure that you have access to the HF repo.

After getting access to the HF repo, you should log in to your HF generate a new token, and use this token to :
```
pip install huggingface_hub
python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('Your_TOKEN)"
```

# How to run
```
cd recipes/MultiWOZ/response_generation/[LLM_model]
python train_with_[LLM_model].py hparams/train_[LLM_model].yaml --data_folder=/your/data/folder
```
The data will be automatically downloaded in the specified data_folder.


# Results

| Model | Release | Hyperparams file | Test Cross-entropy Loss | Test PPL | Test BLEU 4| HuggingFace link | Full model link | GPUs |
|:-------------:|:-------------:|:---------------------------:| :-----:| :-----:| :-----:| :-----:| :--------:|:--------:|
| GPT2 | 2023-08-15 | train_gpt.yaml |  1.39 |  4.01 | 2.54e-04 |[model](https://huggingface.co/speechbrain/MultiWOZ-GPT-Response_Generation) | [model](https://www.dropbox.com/sh/vm8f5iavohr4zz9/AACrkOxXuxsrvJy4Cjpih9bQa?dl=0) | 1xV100 16GB |
| LLAMA2 | 2023-10-15 | train_llama2.yaml |  1.13 |  2.90 | 7.45e-04 |[model](https://huggingface.co/speechbrain/MultiWOZ-Llama2-Response_Generation) | [model](https://www.dropbox.com/sh/d093vsje1d7ijj9/AAA-nHEd_MwNEFJfBGLmXxJra?dl=0) | 1xV100 16GB |




# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/

# **Citing**
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
