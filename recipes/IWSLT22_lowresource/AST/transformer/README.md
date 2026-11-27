# IWSLT 2022 Low-resource Task: Tamasheq-French end-to-end Speech Translation


## Description

This is the recipe for the best system from the IWSLT 2022 low-resource task, as described in the original paper.
The speech translation model comprises a wav2vec 2.0 encoder and a Transformer decoder. It is trained end-to-end without any auxiliary loss. The recipe allows for removing the last layers of the Transformer Encoder inside the wav2vec 2.0 in order to reduce the number of training parameters.

This recipe also provides a flexible use of text-based sequence-to-sequence models, such as [mBART](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt) or [NLLB](https://huggingface.co/facebook/nllb-200-1.3B) model, to initialize the decoder of the speech translation model. This practice has been proven more effective in a wide range of settings in comparison with the randomly initialized decoder.

An update to this recipe adds support for SpeechT5. It is not part of the original contribution and is meant to serve as an example of usage of this model for speech to text.

## Data Downloading

For downloading the dataset used for this experiment, please run the following command.

```
git clone https://github.com/mzboito/IWSLT2022_Tamasheq_data.git
```

## Installing Extra Dependencies

Before proceeding, ensure you have installed the necessary additional dependencies. To do this, simply run the following command in your terminal:

```
pip install -r extra_requirements.txt
```

## Training

For training the model, please update the variables at hparams/train_w2v2_st.yaml.

Note that in order to drop the last layers of the wav2vec 2.0 module, it is necessary to update the parameter "keep_n_layers".
For instance: Using ``keep_n_layers: 10'' means that only the first 10 layers inside the wav2vec 2.0 Transformer encoder will be used for training. The remaining layers are removed.

For launching training:
```
python train.py hparams/train_w2v2_st.yaml --root_data_folder=your/data/path # e.g., /workspace/speechbrain/recipes/IWSLT22_lowresource/IWSLT2022_Tamasheq_data/taq_fra_clean/

```

## Training with mBART/NLLB

For training the model with the mBART/NLLB model, please refer to the hparams/train_w2v2_mbart_st.yaml or hparams/train_w2v2_nllb_st.yaml file.

For launching training:
```
python train_with_w2v_mbart.py hparams/train_w2v2_mbart_st.yaml --root_data_folder=your/data/path # e.g., /workspace/speechbrain/recipes/IWSLT22_lowresource/IWSLT2022_Tamasheq_data/taq_fra_clean
```

One should change hparams/train_w2v2_mbart_st.yaml to hparams/train_w2v2_nllb_st.yaml in the above training command for using NLLB model instead.

## Pre-training Semantically-Aligned Multimodal Utterance-level (SAMU) wav2vec

Inspired by [SAMU-XLSR](https://arxiv.org/abs/2205.08180), a model that unifies speech and text modality for making the pre-trained speech foundation model more semantically aware, we introduce here a recipe for fine-tuning a pre-trained wav2vec 2.0 model in the same manner. Training data can be paired speech/text data of the kind used by ASR or AST. In this recipe, we use directly the IWSLT2022_Tamasheq_data AST data.

For launching SAMU training:
```
python train_samu.py hparams/train_samu.yaml --root_data_folder=your/data/path # e.g., /workspace/speechbrain/recipes/IWSLT22_lowresource/IWSLT2022_Tamasheq_data/taq_fra_clean
```

After the SAMU model is pre-trained, one can use it in the same manner as wav2vec 2.0 model. We found that using SAMU model as speech encoder coupled with a decoder from mBART or NLLB helps further improve BLEU scores on this challenging dataset.

For launching AST training:
```
train_with_samu_mbart.py hparams/train_samu_mbart_st.yaml --root_data_folder=your/data/path --pre_trained_samu=your/samu/ckpt
```

Examples of the two parameters:
--root_data_folder=/workspace/speechbrain/recipes/IWSLT22_lowresource/IWSLT2022_Tamasheq_data/taq_fra_clean
--pre_trained_samu=/workspace/speechbrain/recipes/IWSLT22_lowresource/results/samu_pretraining/7777/save/CKPT+checkpoint_epoch100/wav2vec2.ckpt

One should change hparams/train_samu_mbart_st.yaml to hparams/train_samu_nllb_st.yaml in the above training command for using NLLB model instead.

## Training with SpeechT5

To train the model, please update the variables at hparams/train_speecht5_st.yaml.

To launch the training training:
```bash
python train.py hparams/train_speecht5_st.yaml
```
If you are using distributed training, use the following:
```bash
 torchrun --nproc_per_node=your_number train.py hparams/train_speecht5_st.yaml --find_unused_parameters
 ```


# Results

| No. | hyperparams file |  dev BLEU | test BLEU | Model Link |
| --- |:----------------:|:---------:|:--------:|:--------:|
| 1 | train_w2v2_st.yaml | 7.63 | 5.38 | Not avail. | Not avail. |
| 2 | train_w2v2_mbart_st.yaml | 9.62 | 7.73 | [DropBox](https://www.dropbox.com/sh/xjo0ou739oksnus/AAAgyrCwywmDRRuUiDnUva2za?dl=0) |
| 3 | train_w2v2_nllb_st.yaml | 11.09 | 8.70 | [DropBox](https://www.dropbox.com/sh/spp2ijgfdbzuz26/AABkJ97e72D7aKzNLTm1qmWEa?dl=0) |
| 4 | train_samu_mbart_st.yaml | 13.41 | 10.28 | [DropBox](https://www.dropbox.com/sh/98s1xyc3chreaw6/AABom3FnwY5SsIvg4en9tWC2a?dl=0) |
| 5 | train_samu_nllb_st.yaml | 13.89 | 11.32 | [DropBox](https://www.dropbox.com/sh/ekkpl9c3kxsgllj/AABa0q2LrJe_o7JF-TTbfxZ-a?dl=0) |
| 6 | train_speecht5_st.yaml | 6.00 | 5.28 | [DropBox](https://www.dropbox.com/scl/fo/q5zx8ah7rzeoz0fg6ea62/h?rlkey=y68eo4faog0nz4t9c4lyxoh4x&dl=0) |

## Citation
```
@inproceedings{boito-etal-2022-trac,
    title = "{ON}-{TRAC} Consortium Systems for the {IWSLT} 2022 Dialect and Low-resource Speech Translation Tasks",
    author = {Boito, Marcely Zanon  and
      Ortega, John  and
      Riguidel, Hugo  and
      Laurent, Antoine  and
      Barrault, Lo{\"\i}c  and
      Bougares, Fethi  and
      Chaabani, Firas  and
      Nguyen, Ha  and
      Barbier, Florentin  and
      Gahbiche, Souhir  and
      Est{\`e}ve, Yannick},
    booktitle = "Proceedings of the 19th International Conference on Spoken Language Translation (IWSLT 2022)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland (in-person and online)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.iwslt-1.28",
    doi = "10.18653/v1/2022.iwslt-1.28",
    pages = "308--318"
}
```
