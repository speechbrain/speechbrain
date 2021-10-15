# The Blind SI-SNR estimation recipe

* The goal of this recipe is to train a neural network to be able to estimate the scale-invariant source-to-noise ratio (SI-SNR) from the separated signals.

* This model is developed to estimate source separation performance on the REAL-M dataset which consists of real life mixtures.

## Example call for running the training script

```python train.py hparams/pool_sisnrestimator.yaml --data_folder /yourLibri2Mixpath --base_folder_dm /yourLibriSpeechpath --rir_path /yourpathforwhamrRIRs --dynamic_mixing True --use_whamr_train True --test_only False```
