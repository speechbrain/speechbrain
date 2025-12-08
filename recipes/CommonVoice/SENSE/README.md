# Multilingual SENSE on Common Voice

This folder contains scripts to train **SENSE** models that align a self-supervised speech encoder (wav2vec-BERT) with multilingual text embeddings (BGE-M3) on the [Common Voice](https://commonvoice.mozilla.org/en/datasets) corpus. The resulting speech embeddings live in a shared semantic space, similar in spirit to **SAMU-XLSR** and **Meta's SONAR** multilingual speech–text encoders. They follow the SENSE framework (*"SENSE models: an open source solution for multilingual and multimodal semantic-based tasks"*) described in [https://arxiv.org/abs/2509.12093](https://arxiv.org/abs/2509.12093).


A **single multilingual model** is trained on **90 Common Voice languages**.

Two components are used:
1. A **student audio encoder**: wav2vec-BERT followed by attention pooling and a linear projection.
2. A **teacher text encoder**: BGE-M3 sentence embeddings computed on-the-fly from the reference transcripts.

# How to run
```shell
python train.py hparams/train_sense.yaml
```

# Data

This recipe uses the multilingual **Common Voice** dataset. A large number of languages is selected, and all of them are merged into a single multilingual training setup.

Any Common Voice language can be used by editing the `languages` field in `hparams/train_sense.yaml`.
In this configuration, the model was trained on the following **90 Common Voice languages**:

`af`, `am`, `ar`, `as`, `ast`, `az`, `ba`, `be`, `bg`, `bn`, `br`, `ca`,
`ckb`, `cs`, `cv`, `cy`, `da`, `de`, `dv`, `el`, `en`, `eo`, `es`, `et`,
`fa`, `fi`, `fr`, `fy-NL`, `ga-IE`, `gl`, `gn`, `he`, `hi`, `hsb`, `ht`,
`hu`, `ia`, `id`, `is`, `it`, `ja`, `ka`, `kab`, `kk`, `ko`, `ky`, `lt`,
`lo`, `lv`, `ml`, `mn`, `mhr`, `mk`, `mr`, `mt`, `ne-NP`, `nl`, `nn-NO`,
`oc`, `or`, `os`, `pa-IN`, `pl`, `ps`, `pt`, `ro`, `ru`, `sah`, `sc`,
`sk`, `sl`, `sr`, `sv-SE`, `sw`, `ta`, `te`, `th`, `ti`, `tk`, `tr`, `tt`, <!-- codespell:ignore -->
`ug`, `uk`, `ur`, `uz`, `vi`, `yi`, `yo`, `zh-HK`, `zu`.

## Multilingual sampling ratios

Common Voice languages do **not** have the same number of utterances: some are very high-resource, others much smaller. To avoid that high-resource languages dominate the training batches, we compute a **sampling ratio** for each language.

This multilingual smoothing strategy follows the rebalancing scheme introduced in **SAMU-XLSR** (“SAMU-XLSR: Semantically-Aligned Multimodal Utterance-level Cross-Lingual Speech Representation”, Khurana et al., 2022, Eq. (3), see https://arxiv.org/abs/2205.08180).

For the train split, let:

- $N_l$ be the number of utterances in language $l$,
- $N_{\text{total}} = \sum_l N_l$ be the total number of utterances over all languages,
- $p_l = \frac{N_l}{N_{\text{total}}}$ be the empirical probability of language $l$.

The sampling ratio $r_l$ used by the sampler is then defined as:

$$
r_l = \frac{1}{p_l} \cdot \frac{p_l^\alpha}{\sum_k p_k^\alpha},
$$

where $\alpha$ is the hyperparameter `sampling_alpha` (e.g. $\alpha = 0.05$).

- $p_l$ reflects how frequent a language is in the corpus.
- $r_l$ is the **sampling ratio** used as a weight in the sampler:
  - high-resource languages (large $p_l$) are down-weighted,
  - low-resource languages (small $p_l$) are up-weighted.


These ratios are saved to `<output_folder>/language_ratios.json` and stored in a `ratio` column inside `train.csv`. During training, `ReproducibleWeightedRandomSampler` uses this `ratio` column to build multilingual batches where smaller languages are seen more often and larger languages are not over-represented.

## Pretrained checkpoint

Download SENSE model:
[https://github.com/MaryemBouziane/SENSE](https://github.com/MaryemBouziane/SENSE)
(Trained on 32×A100 GPUs)


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
