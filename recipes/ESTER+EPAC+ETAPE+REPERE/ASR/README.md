# *ESTER+EPAC+ETAPE+REPERE* (ELRA) CTC ASR with pre-trained wav2vec2.

Information about the datasets here:
 - ESTER1: https://catalogue.elra.info/en-us/repository/browse/ELRA-S0241
 - ESTER2: https://catalogue.elra.info/en-us/repository/browse/ELRA-S0338
 - ETAPE:  https://catalogue.elra.info/en-us/repository/browse/ELRA-E0046
 - EPAC:   https://catalogue.elra.info/en-us/repository/browse/ELRA-S0305
 - REPERE: https://catalogue.elra.info/en-us/repository/browse/ELRA-E0044

**Supported pre-trained wav2vec2 from LeBenchmark:** [HuggingFace](https://huggingface.co/LeBenchmark)

## Installing Extra Dependencies

Before proceeding, ensure you have installed the necessary additional dependencies. To do this, simply run the following command in your terminal:

```
pip install -r extra_requirements.txt
```

For `k2`-based recipes, you will first need to install `k2`. The integration has been tested with `k2==1.24.4` and `torch==2.0.1`, although it should also work with any `torch` version as long as `k2` supports it (compatibility list [here](https://k2-fsa.github.io/k2/installation/pre-compiled-cuda-wheels-linux/index.html)). You can install `k2` by following the instructions [here](https://k2-fsa.github.io/k2/installation/from_wheels.html#linux-cuda-example).


# WFST-based CTC training/inference
To fine-tune a wav2vec 2.0 model with the WFST-based CTC loss, you can use the `train_with_wav2vec_k2.py` script. This will create a `lang` directory inside your output folder, which will contain the files required to build a lexicon FST. The tokenization method used here is a very basic character-based tokenization (e.g. `hello -> h e l l o`).

Using a lexicon FST (L) while training can help guide the model to better predictions. When decoding, you can either use a simple HL decoding graph (where H is the ctc topology), or use an HLG graph (where G is usually a 3-gram language model) to further improve the results. In addition, whole lattice rescoring is also supported. This typically happens with a 4-gram language model. See `hparams/train_with_wav2vec_k2.yaml`` for more details.

If you choose to use a 3-gram or a 4-gram language model, you can either supply pre-existing ARPA LMs for both cases, including the option to train your own, or you can specify the name in the YAML docstring for automatic downloading. Comprehensive instructions are provided in `train_hf_wav2vec_k2.yaml`.

For those interested in training their own language model, please consult our recipe at ESTER+EPAC+ETAPE+REPERE/LM/train_ngram.py.

Example usage:
```
 python3 train_with_wav2vec_ctc_k2.py hparams/train_with_wav2vec_ctc_k2_phone.yaml --data_folder=/path/to/ESTER+EPAC+ETAPE+REPERE/parent_dir
```

To use the HLG graph (instead of the default HL), pass `--compose_HL_with_G=True`. To use the 4-gram LM for rescoring, pass the `--decoding_method=whole-lattice-rescoring` argument. Note that this will require more memory, as the whole lattice will be kept in memory during the decoding. In this recipe, the `lm_scale` used by default is 0.4.

| Release    | Hyperparams file                     | Decoding method                   | Text Normalization for scoring  |  EPAC WER  | ESTER1 WER | ESTER2 WER | ETAPE | REPERE |
|:----------:|:------------------------------------:|:---------------------------------:|:-------------------------------:|:----------:|:----------:|:----------:|:-----:|:-------|
| 21/05/2024 | train_with_wav2vec_ctc_k2_phone.yaml | k2CTC + HL graph + 1best decoding | No                              | 14.41      | 12.81      | 13.66      | 24.90 | 13.95  |
| 21/05/2024 | train_with_wav2vec_ctc_k2_char.yaml  | k2CTC + HL graph + 1best decoding | No                              | 15.17      | 13.18      | 14.21      | 26.16 | 14.74  |
| 30/05/2024 | train_with_wav2vec_ctc_k2_phone.yaml | k2CTC + HL graph + 1best decoding | Yes                             | 9.49       | 10.19      | 11.36      | 23.01 | 11.58  |
| 30/05/2024 | train_with_wav2vec_ctc_k2_char.yaml  | k2CTC + HL graph + 1best decoding | Yes                             | 10.96      | 11.00      | 12.39      | 24.83 | 12.88  |

## Text normalization for scoring
The script for text normalization scoring is available at: https://github.com/pchampio/UB-WER-NORM_ESTER-EPAC-ETAPE-REPERE/

# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/

# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{ravanelli2024opensourceconversationalaispeechbrain,
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
