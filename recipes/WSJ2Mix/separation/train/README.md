# Speech separation with WSJMIX
This folder contains some popular recipes for the WSJMIX task (2/3 sources).

# SepFormer recipe

# DualPath RNN recipe

# ConvTasnet Recipe

* This recipe supports train with several source separation models on WSJ2 - Mix, including[Sepformer](https: // arxiv.org / abs / 2010.13154), [DPRNN](https: // arxiv.org / abs / 1910.06379), [ConvTasnet](https: // arxiv.org / abs / 1809.07454), [DPTNet](https: // arxiv.org / abs / 2007.13975)

Additional dependency:
```
pip install mir_eval
```

To run it:

```
python train.py hyperparams/sepformer.yaml
```
Make sure you modified the paths inside the parameter file before running the recipe.


# WSJ0-2mix dataset creation
* The best way to create this dataset is using the original matlab script. This script and the associated meta data can be obtained through the following [link](https://www.dropbox.com/s/gg524noqvfm1t7e/create_mixtures_wsj023mix.zip?dl=1).
* The dataset creation script assumes that the original WSJ0 files in the sphere format are already converted to .wav .


# Dynamic Mixing:

* This recipe supports dynamic mixing where the training data is dynamically created in order to obtain new utterance combinations during training. For this you need to have the WSJ0 dataset (available though LDC at `https://catalog.ldc.upenn.edu/LDC93S6A`). After this you need to run the preprocessing script under `recipes/WSJ2Mix/meta/preprocess_dynamic_mixing.py`. Then you need to specify the path to the output folder of this script through the `wsj0_tr` variable in the variable. This script converts the recordings into 8kHz, and runs the level normalization script.

This script utilises octave to be able to call the matlab function `activlev.m` for level normalization. Depending on your octave version, you might observe the following warning:
```
error: called from graphics_toolkit at line 81 column 5
graphics_toolkit: = toolkit is not available
```
This is in essence a warning and does not affect the results of this script.

# Results

Here are the SI - SNRi results on the test set of WSJ0 - 2Mix dataset with SepFormer:

| | SepFormer |
|--- | --- |
|NoAugment | 20.4 |
|SpeedPerturb | 21.8 |
|DynamicMixing + SpeedPerturb | 22.3 |
