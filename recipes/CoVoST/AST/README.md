# CoVoST speech to text translation

This folder contains script necessary to run automatic speech translation with the [CoVoST dataset](https://github.com/facebookresearch/covost) based on [CommonVoice](https://commonvoice.mozilla.org/en/datasets).

Two heuristics are available:
1. Training from scratch with a conformer encoder-decoder model and multitask speech recognition plus speech translation training.
2. SpeechLLM fine-tuning based on SSL speech encoders and LLaMA large language models (with and without adapters).

# How to run
```shell
python train{_xlsr_llama}.py hparams/{hparam_file}.py
```

# Data preparation
It is important to note that CommonVoice initially offers mp3 audio files. It is feasible to convert these files to .wav during data preparation, this will speed up training but also make the first data preparation to be pretty slow. Audio files are downsampled on the fly within the dataio function of the training script.

# Languages
While CoVoST offers multiple languages, this recipe only was tested on English to German translation. However, there is nothing special to do to select another language pair aside from adding a proper text normalisation on the covost_prepary.py file.

# Results
| Language | hyperparams file | Encoder | LLM | Test BLEU | Hugging Face link | Model link | GPUs |
| ------------- |:-------------:|:---------------------------:| -----:| -----:| -----:| -----:| -----:|
| English - German | conformer.yaml | conformer | None | 13.9 | None | None | 2x A40 |
| English - German | w2v2_llama3.yaml| wavlm-large | LLaMA 3.1 7B | 27.2 | None | None | 2x A100 |

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
