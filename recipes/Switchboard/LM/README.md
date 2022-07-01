# Language Model with Switchboard
This folder contains recipes for finetuning language models for the Switchboard dataset.
It supports both an RNN-based LM and a Transformer-based LM.

You can download the Switchboard data at https://catalog.ldc.upenn.edu/LDC97S62.

The eval2000/Hub5 English test set can be found at:
- Speech data: https://catalog.ldc.upenn.edu/LDC2002S09
- Transcripts: https://catalog.ldc.upenn.edu/LDC2002T43

Part 1 and part 2 of the Fisher corpus are available at:
- https://catalog.ldc.upenn.edu/LDC2004T19
- https://catalog.ldc.upenn.edu/LDC2005T19

As in Kaldi's [swbd/s5c](https://github.com/kaldi-asr/kaldi/tree/master/egs/swbd/s5c) recipe,
the Fisher transcripts can be used as an additional resource for training Tokenizer and LM.

# How to run:
```
python train.py hparams/transformer.yaml
python train.py hparams/transformer_finetune.yaml
```

| Release | hyperparams file | Test PP | Model link | GPUs |
| :---     | :---: | :---: | :---: | :---: |
| 01-07-22 | transformer.yaml | --.-- | n.a. | 1xA100 40GB |
| 01-07-22 | transformer_finetune.yaml | --.-- | n.a. | 1xA100 40GB |


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