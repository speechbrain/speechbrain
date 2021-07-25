# SLU recipes for SLURP
This folder contains recipes for spoken language understanding (SLU) with [SLURP](https://zenodo.org/record/4274930#.YEFCYHVKg5k).

### Direct recipe
The "direct" maps the input speech directly to semantics using a seq2seq model.

```
cd direct
python train.py hparams/train.yaml
```

### Tokenizer recipe
(You don't need to run this because the direct recipe downloads a tokenizer, but you can if you'd like to train a new tokenizer for SLURP.)

Run this to train the tokenizer:

```
cd Tokenizer
python train.py hparams/tokenizer_bpe51.yaml
```

### NLU recipe
The "NLU" recipe takes the true transcript as input rather than speech and trains a seq2seq model to map the transcript to the semantics.

```
cd NLU
python train.py hparams/train.yaml
```


# Performance summary
Note: SLURP comes with a tool for measuring SLU-F1 and other metrics.
The recipes here dump the model outputs to a file called `predictions.jsonl` in the `results` folder.
You can compute the metrics by feeding `predictions.jsonl` into the [SLURP evaluation tool](https://github.com/pswietojanski/slurp/tree/master/scripts/evaluation).

The following results were obtained on a 48 GB RTX 8000 (the recipe has also been successfully tested on a 12 GB Tesla K80):

| Model	| scenario (accuracy) | action (accuracy) | intent (accuracy) | Word-F1 | Char-F1 | SLU-F1 | Training time |
|:------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Direct | 81.73 | 77.11 | 75.05 | 61.24 | 65.42 | 63.26 | 1 hour per epoch |

| Model	| scenario (accuracy) | action (accuracy) | intent (accuracy) | Training time |
|:---:|:-----:|:-----:|:-----:|:-----:|
| NLU | 90.81 | 88.29 | 87.28 | 40 min per epoch |


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


