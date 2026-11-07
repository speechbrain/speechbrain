# CoVoST speech to text translation

This folder contains script necessary to run automatic speech translation with the [CoVoST dataset](https://github.com/facebookresearch/covost) based on [CommonVoice](https://commonvoice.mozilla.org/en/datasets).

Three heuristics are available:
1. Training from scratch with a conformer encoder-decoder model and multitask speech recognition plus speech translation training.
2. SpeechLLM fine-tuning based on SSL speech encoders and LLaMA large language models (with and without adapters).
3. Fine-tuning of SeamlessM4T (speech encoder and text decoder), e.g. `python train_with_seamless.py hparams/seamless.yaml`.

# How to run
```shell
python train{_xlsr_llama}.py hparams/{hparam_file}.py
```

# Data preparation
It is important to note that CommonVoice initially offers mp3 audio files. It is feasible to convert these files to .wav during data preparation, this will speed up training but also make the first data preparation to be pretty slow. Audio files are downsampled on the fly within the dataio function of the training script.

# Languages
While CoVoST offers multiple languages, the conformer and SpeechLLM recipes were only tested on English to German translation, and the SeamlessM4T recipe on French to English translation. However, there is nothing special to do to select another language pair aside from adding a proper text normalisation on the covost_prepary.py file.

# SeamlessM4T
`hparams/seamless.yaml` fine-tunes SeamlessM4T for French to English translation (CoVoST 2 `fr_en`, whose train split has 264 hours of speech; the training utterances longer than 10 s are removed). Set `src_language` and `tgt_language` for CoVoST and `target_lang` to the SeamlessM4T code of the target language (e.g. `eng`, `deu`). French normalisation is available in `covost_prepare.py`; other source languages may need their own character set there.
The parts of the model to fine-tune are chosen in the yaml file with `freeze_encoder`, `freeze_feature_projection`, `freeze_adapter`, `freeze_decoder` and `freeze_embeddings` (or `freeze_seamless` for the whole model).

# Results
| Language | hyperparams file | Encoder | LLM | Test BLEU | Hugging Face link | Model link | GPUs |
| ------------- |:-------------:|:---------------------------:| -----:| -----:| -----:| -----:| -----:|
| English - German | conformer.yaml | conformer | None | 13.9 | None | None | 2x A40 |
| English - German | w2v2_llama3.yaml| wavlm-large | LLaMA 3.1 7B | 27.2 | None | None | 2x A100 |
| French - English | seamless.yaml | SeamlessM4T v2-large | None | 45.00* | None | None | 2x A100 |

*The CoVoST 2 `fr_en` splits were built with `get_covost_splits.py` from the Common Voice 8.0 French `validated.tsv`, which contains 99.9% of the CoVoST 2 clips (Common Voice 4.0 is no longer distributed). As for the other rows, BLEU is computed on the text normalised by `covost_prepare.py` (lower case, no punctuation) and on the utterances it keeps, so it is not comparable to the cased and punctuated BLEU reported in the SeamlessM4T papers.

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
