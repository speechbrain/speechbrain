## UrbanSound8k  multi-class audio classification

[This recipe and description has been adapted from the SpeechBrain "VoxCeleb" recipe example] 

This recipe contains scripts for multi-class audio classification experiments with the UrbanSound8k dataset (https://urbansounddataset.weebly.com/urbansound8k.html). While publicly available, a request must be made before a download link for the dataset will be provided by the authors (https://urbansounddataset.weebly.com/download-urbansound8k.html). 

UrbanSound8k is divided into 10 classes, one of which (engine_idling) receives special attention in our experiments below.

```
0 = dog_bark
1 = children_playing
2 = air_conditioner
3 = street_music
4 = gun_shot
5 = siren
6 = engine_idling
7 = jackhammer
8 = drilling
9 = car_horn
```

# Multiclass Classification

Run the following command to train using the ECAPA-TDNN network architecture:

`python train.py hparams/train_ecapa_tdnn.yaml`

Note that data-augmentations listed in the `augment_pipeline`,  list will be applied when running. The simplest way to enable/disable augmentation is by simply
enabling/commenting out the list items. If all are enabled, we concatenate waveform dropout, speed change, reverberation, noise, and noise+rev. The batch is 6 times larger than the original one. This normally leads to a performance improvement at the cost of longer training time and higher memory use.

# 10-fold Cross Validation
Per the authors of the UrbanSound8k dataset, some of the pre-defined data folds are much "easier" test sets than others. For a true measure of the quality of a network architecture, we must perform a 10-fold cross validation.

Run the following commands to perform a 10-fold cross validation:

`python train.py hparams/train_ecapa_tdnn.yaml --data_folder=/localscratch/UrbanSound8K/ --train_fold_nums=[2, 3, 4, 5, 6, 7, 8, 9, 10] --valid_fold_nums=[1] --test_fold_nums=[1] --output_folder=./results/urban_sound/fold_1

python train.py hparams/train_ecapa_tdnn.yaml --data_folder=/localscratch/UrbanSound8K/ --train_fold_nums=[1, 3, 4, 5, 6, 7, 8, 9, 10] --valid_fold_nums=[2] --test_fold_nums=[2] --output_folder=./results/urban_sound/fold_2

python train.py hparams/train_ecapa_tdnn.yaml --data_folder=/localscratch/UrbanSound8K/ --train_fold_nums=[1, 2, 4, 5, 6, 7, 8, 9, 10] --valid_fold_nums=[3] --test_fold_nums=[3] --output_folder=./results/urban_sound/fold_3

python train.py hparams/train_ecapa_tdnn.yaml --data_folder=/localscratch/UrbanSound8K/ --train_fold_nums=[1, 2, 3, 5, 6, 7, 8, 9, 10] --valid_fold_nums=[4] --test_fold_nums=[4] --output_folder=./results/urban_sound/fold_4

python train.py hparams/train_ecapa_tdnn.yaml --data_folder=/localscratch/UrbanSound8K/ --train_fold_nums=[1, 2, 3, 4, 6, 7, 8, 9, 10] --valid_fold_nums=[5] --test_fold_nums=[5] --output_folder=./results/urban_sound/fold_5

python train.py hparams/train_ecapa_tdnn.yaml --data_folder=/localscratch/UrbanSound8K/ --train_fold_nums=[1, 2, 3, 4, 5, 7, 8, 9, 10] --valid_fold_nums=[6] --test_fold_nums=[6] --output_folder=./results/urban_sound/fold_6

python train.py hparams/train_ecapa_tdnn.yaml --data_folder=/localscratch/UrbanSound8K/ --train_fold_nums=[1, 2, 3, 4, 5, 6, 8, 9, 10] --valid_fold_nums=[7] --test_fold_nums=[7] --output_folder=./results/urban_sound/fold_7

python train.py hparams/train_ecapa_tdnn.yaml --data_folder=/localscratch/UrbanSound8K/ --train_fold_nums=[1, 2, 3, 4, 5, 6, 7, 9, 10] --valid_fold_nums=[8] --test_fold_nums=[8] --output_folder=./results/urban_sound/fold_8

python train.py hparams/train_ecapa_tdnn.yaml --data_folder=/localscratch/UrbanSound8K/ --train_fold_nums=[1, 2, 3, 4, 5, 6, 7, 8, 10] --valid_fold_nums=[9] --test_fold_nums=[9] --output_folder=./results/urban_sound/fold_9

python train.py hparams/train_ecapa_tdnn.yaml --data_folder=/localscratch/UrbanSound8K/ --train_fold_nums=[1, 2, 3, 4, 5, 6, 7, 8, 9] --valid_fold_nums=[10] --test_fold_nums=[10] --output_folder=./results/urban_sound/fold_10
`

Note that the results for 10-fold must be compiled from the output folders and averaged manually.

# Performance (single fold)
test loss: 4.15, test acc: 7.55e-01, test error: 2.46e-01

Per Class Accuracy: 
0: 0.850
1: 0.670
2: 0.600
3: 0.800
4: 0.812
5: 0.554
6: 0.753
7: 0.906
8: 0.790
9: 0.939, 

 Confusion Matrix: 
[[85  1  2  3  0  1  1  0  1  6]
 [ 2 67  5  9  0  3  5  2  6  1]
 [ 0  3 60  1  1  0 16 16  3  0]
 [ 0 10  3 80  0  1  0  0  6  0]
 [ 2  0  0  0 26  4  0  0  0  0]
 [15 14  7  0  0 46  0  0  1  0]
 [ 0  0 16  0  0  0 70  6  1  0]
 [ 0  0  0  0  2  0  3 87  4  0]
 [ 3  1  4  2  0  1  3  6 79  1]
 [ 1  0  0  0  0  0  0  0  1 31]]

Please, take a look [here](https://drive.google.com/drive/folders/1sItfg_WNuGX6h2dCs8JTGq2v2QoNTaUg?usp=sharing) for the full experiment folder (with pre-trained models).


Classification performance and f-scores are output to the console and log file for each epoch using a passed validation set, and after training using the passed test set.

The default hyperparameter settings will output Tensorboard logs to `<output_folder>/tb_logs/` and can be viewed simply using:

 `tensorboard --logdir=<YOUR_PATH_TO_OUTPUT_FOLDER>/tb_logs/`


# PreTrained Model + Easy-Inference
You can find the pre-trained model with an easy-inference function on HuggingFace:
- https://huggingface.co/speechbrain/urbansound8k_ecapa
You can find the full experiment folder (i.e., checkpoints, logs, etc) here:
- https://drive.google.com/drive/folders/1sItfg_WNuGX6h2dCs8JTGq2v2QoNTaUg?usp=sharing



# UrbanSound8k Download and Use

1. Download UrbanSound8k
You can request a download link at the form, here: https://urbansounddataset.weebly.com/download-urbansound8k.html

2. Setting train, validation and test folds
UrbanSound8k comes split into 10 predefined data folds. It is VERY important that data from these folds not be shuffled together, as folds contain related data.

All experiments described above (except 10-fold CV) are analogous to only a single one of these 10-fold experiments, which are insufficient for publication.

From the authors: (https://urbansounddataset.weebly.com/urbansound8k.html#10foldCV):

Since releasing the dataset we have noticed a couple of common mistakes that could invalidate your results, potentially leading to manuscripts being rejected or the publication of incorrect results. To avoid this, please read the following carefully:

1. Don't reshuffle the data! Use the predefined 10 folds and perform 10-fold (not 5-fold) cross validation
The experiments conducted by vast majority of publications using UrbanSound8K (by ourselves and others)  evaluate classification models via 10-fold cross validation using the predefined splits*. We strongly recommend following this procedure.

Why?
If you reshuffle the data (e.g. combine the data from all folds and generate a random train/test split) you will be incorrectly placing related samples in both the train and test sets, leading to inflated scores that don't represent your model's performance on unseen data. Put simply, your results will be wrong.
Your results will NOT be comparable to previous results in the literature, meaning any claims to an improvement on previous research will be invalid. Even if you don't reshuffle the data, evaluating using different splits (e.g. 5-fold cross validation) will mean your results are not comparable to previous research.

2. Don't evaluate just on one split! Use 10-fold (not 5-fold) cross validation and average the scores
We have seen reports that only provide results for a single train/test split, e.g. train on folds 1-9, test on fold 10 and report a single accuracy score. We strongly advise against this. Instead, perform 10-fold cross validation using the provided folds and report the average score.

Why?
Not all the splits are as "easy". That is, models tend to obtain much higher scores when trained on folds 1-9 and tested on fold 10, compared to (e.g.) training on folds 2-10 and testing on fold 1. For this reason, it is important to evaluate your model on each of the 10 splits and report the average accuracy.
Again, your results will NOT be comparable to previous results in the literature.



  

While all of the above hyperparameter files listed above (except the 10-fold-cv) accept as lists the train, valid and test  



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