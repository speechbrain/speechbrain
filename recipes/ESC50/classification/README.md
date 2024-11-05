# Sound Classification - ESC50 Dataset

This recipe trains a classifier for the ESC50 multiclass sound classification dataset.

The task involves classifying audio sounds into 50 different categories. These categories are divided into the following groups:

- Animals
- Natural soundscapes and water sounds
- Human, non-speech sounds
- Interior/domestic sounds
- Exterior/urban noises

The scripts offer the possibility to train both with log-spectra and log-mel audio features.

## Dataset Download

The ESC50 dataset will be automatically downloaded when running the recipe. If you prefer to download it manually, please visit: [https://github.com/karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50)


---------------------------------------------------------------------------------------------------------

## Installing Extra Dependencies

Before proceeding, make sure you have installed the necessary additional dependencies.

To do this, simply run the following command in your terminal:

```shell
pip install -r extra_requirements.txt
```

---------------------------------------------------------------------------------------------------------

## Supported Models

### CNN14

This script trains a [CNN14 model](https://arxiv.org/abs/1912.10211) on the ESC50 dataset. To run this, you can use the command:

```shell
python train.py hparams/cnn14.yaml --data_folder /yourpath/ESC50
```

The dataset will be automatically download at the specified data folder.

---------------------------------------------------------------------------------------------------------

### Conv2D

This script trains a simple convolutional model on the ESC50 dataset. To run this, you can use the command:

```shell
python train.py hparams/conv2d.yaml --data_folder /yourpath/ESC50
````

---------------------------------------------------------------------------------------------------------

### FocalNet

This script trains a FocalNet model on the ESC50 dataset. To run this, you can use the command:

```shell
python train.py hparams/focalnet.yaml --data_folder /yourpath/ESC50
```

---------------------------------------------------------------------------------------------------------

### ViT

This script trains a ViT model on the ESC50 dataset. To run this, you can use the command:

```shell
python train.py hparams/vit.yaml --data_folder /yourpath/ESC50
```

---------------------------------------------------------------------------------------------------------

### To train with WHAM! noise

In order to train the classifier with WHAM! noise, you need to download the  WHAM! noise dataset from [here](http://wham.whisper.ai/).
Then, you can train your classifier with the following command:

```shell
python train.py hparams/modelofchoice.yaml --data_folder /yourpath/ESC50 --add_wham_noise True --wham_folder /yourpath/wham_noise
```


## Results

| Hyperparams file | Accuracy (%) |   Training time    |                        HuggingFace link                         |                                                         Model link                                                         |    GPUs     |
|:----------------:|:------------:|:------------------:|:---------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------:|:-----------:|
|    cnn14.yaml    |     82.0     | 11 seconds / epoch |     [model](https://huggingface.co/speechbrain/cnn14-esc50)     |                     [model](https://www.dropbox.com/sh/fbe7l14o3n8f5rw/AACABE1BQGBbX4j6A1dIhBcSa?dl=0)                     |  RTX 3090   |
|   conv2d.yaml    |     75.0     | 15 seconds / epoch |      [model](https://huggingface.co/speechbrain/PIQ-ESC50)      |                     [model](https://www.dropbox.com/sh/tl2pbfkreov3z7e/AADwwhxBLw1sKvlSWzp6DMEia?dl=0)                     |  RTX 3090   |
|  focalnet.yaml   |     77.4     | 60 seconds / epoch | [model](https://huggingface.co/speechbrain/focalnet-base-esc50) | [model](https://www.dropbox.com/scl/fo/zk101h5xypgi56d777yp5/AGVIfoe56OWInxWf6F57JyQ?rlkey=hmme5c8rnu2sok3jnwbanw7eq&dl=0) | 1xV100 32GB |
|     vit.yaml     |     73.6     | 56 seconds / epoch |   [model](https://huggingface.co/speechbrain/vit-base-esc50)    | [model](https://www.dropbox.com/scl/fo/af59l6mtm0ytqyhz3l7ib/ADGklBYXxil1DWKv5CSMDGk?rlkey=wk5tdh0h26f61e1tn3bh80vys&dl=0) | 1xV100 32GB |

---------------------------------------------------------------------------------------------------------

## How to Run on Test Sets Only

If you want to run it on the test sets only, you can add the flag `--test_only` to the following command:

```shell
python train.py hparams/<config>.yaml --data_folder /yourpath/ESC50 --test_only
```

---------------------------------------------------------------------------------------------------------

## Notes

- The recipe automatically downloads the ESC50 dataset. You only need to specify the path to which you would like to download it.

- All the necessary models are downloaded automatically for each training script.

---------------------------------------------------------------------------------------------------------

## Citing

If you find this recipe useful, please cite:

```bibtex
@article{Wang_2022,
    doi = {10.1109/lsp.2022.3229643},
    url = {https://doi.org/10.1109%2Flsp.2022.3229643},
    year = 2022,
    publisher = {Institute of Electrical and Electronics Engineers ({IEEE})},
    volume = {29},
    pages = {2607--2611},
    author = {Zhepei Wang and Cem Subakan and Xilin Jiang and Junkai Wu and Efthymios Tzinis and Mirco Ravanelli and Paris Smaragdis},
    title = {Learning Representations for New Sound Classes With Continual Self-Supervised Learning},
    journal = {{IEEE} Signal Processing Letters}
}
```

```bibtex
@inproceedings{dellalibera2024focal,
    title={Focal Modulation Networks for Interpretable Sound Classification},
    author={Luca Della Libera and Cem Subakan and Mirco Ravanelli},
    booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) XAI-SA Workshop},
    year={2024},
}
```

If you use **SpeechBrain**, please cite:

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

---------------------------------------------------------------------------------------------------------

## About SpeechBrain

- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/

---------------------------------------------------------------------------------------------------------
