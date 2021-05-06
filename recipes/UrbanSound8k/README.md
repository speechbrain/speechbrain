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

Run the following command to loop through, holding out a single fold for validation/test while trainging on the others:

`python train_10_fold.py hparams/train_ecapa_tdnn.yaml`

Note that the train_fold_nums, valid_fold_nums and test_fold_nums hyperparameters are IGNORED by this script.

Note that the results for 10-fold must be compiled from the output folders and averaged manually.

# Results

Classification performance and f-scores are output to the console and log file for each epoch using a passed validation set, and after training using the passed test set.

The default hyperparameter settings will output Tensorboard logs to `<output_folder>/tb_logs/` and can be viewed simply using:

 `tensorboard --logdir=<YOUR_PATH_TO_OUTPUT_FOLDER>/tb_logs/`

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
