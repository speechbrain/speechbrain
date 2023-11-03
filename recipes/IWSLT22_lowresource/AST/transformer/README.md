# IWSLT 2022 Low-resource Task: Tamasheq-French end-to-end Speech Translation


## Description

This is the recipe for the best system from the IWSLT 2022 low-resource task, as described in the original paper.
The speech translation model comprises a wav2vec 2.0 encoder and a Transformer decoder. It is trained end-to-end without any auxiliary loss. The recipe allows for removing the last layers of the Transformer Encoder inside the wav2vec 2.0 in order to reduce the number of training parameters.

This recipe also provides a flexible use of text-based sequence-to-sequence models, such as [mBART](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt) or [NLLB](https://huggingface.co/facebook/nllb-200-1.3B) model, to initialize the decoder of the speech translation model. This pratice has been proven more effective in a wide range of settings in comparison with the randomly initialized decoder.

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

# Results

| No. | hyperparams file |  dev BLEU | test BLEU | Model Link |
| --- |:----------------:|:---------:|:--------:|:--------:|
| 1 | train_w2v2_st.yaml | 7.63 | 5.38 | Not avail. | Not avail. |
| 2 | train_w2v2_mbart_st.yaml | 9.62 | 7.73 | [DropBox](https://www.dropbox.com/sh/xjo0ou739oksnus/AAAgyrCwywmDRRuUiDnUva2za?dl=0) |
| 3 | train_w2v2_nllb_st.yaml | 11.09 | 8.70 | [DropBox](https://www.dropbox.com/sh/spp2ijgfdbzuz26/AABkJ97e72D7aKzNLTm1qmWEa?dl=0) |

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
