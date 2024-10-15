# IWSLT 2022 Low-resource Task: Tamasheq-French end-to-end Speech Translation


## Description
This file describes the SpeechT5 recipe for the end-to-end speech translation task from Tamsheq to French using the IWSLT 2022 Low-resource Tamasheq-French dataset. 
This recipe is not part of the original work. It is a contribution to serve as an example for using the Speechbrain SpeechT5 for speech to text integration. 

For more details about the dataset, the task or the orignial submission by the authors of the original recipes, please refer to the `README.md` file in this same directory.

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

| hyperparams file |  dev BLEU | test BLEU | Model Link |
|:----------------:|:---------:|:--------:|:--------:|
| train_speecht5_st.yaml | 6.00 | 5.28 | [DropBox](https://www.dropbox.com/scl/fo/q5zx8ah7rzeoz0fg6ea62/h?rlkey=y68eo4faog0nz4t9c4lyxoh4x&dl=0) |
