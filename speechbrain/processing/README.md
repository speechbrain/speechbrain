# Feature extraction
This section reports some examples of feature extraction performed with SpeechBrain.
The feature extraction process is extremely efficient for standard features, especially if performed on the GPUs. For this reason, we suggest to do **feature extraction on-the-fly** and to consider it just as any other speech processing module. The on-the-fly feature computation has the following advantages:

1- The speech processing pipeline is cleaner and does not require a feature computation step before starting the processing.
2- It is more compliant with on-the-fly data augmentation that can be used to significantly improve the system performance in many speech applications.

Note that the standard feature extraction pipelines (e.g, MFCCS or FBANKs) are **fully differentiable** and we can backpropagate the gradient through them if needed. Thanks to this property, we can **learn** (when requested by the users) some of the **parameters** related to the feature extraction such as the filter frequencies and bands (similarly to what done in SincNet).

## Short-time Fourier transform (STFT)
We start from the most basic speech features i.e. the *Short-Time Fourier transform (STFT)*. The STFT computes the FFT transformation using sliding windows with a certain length (win_length) and a certain hop size (hop_length).

## Filter banks (FBANKs)
Mel filters average the frequency axis of the spectrogram with a set of filters (usually with a triangular shape)  that cover the full band. The filters, however, are not equally distributed, but we allocated more "narrow-band" filters in the lower part of the spectrum and fewer "large-band" filters for higher frequencies.  This processing is inspired by our auditory system, which is much more sensitive to low frequencies rather than high ones. Let's compute mel-filters by running:

## Mel Frequency Cepstral Coefficients (MFCCs)
Beyond FBANKs, other very popular features are the Mel-Frequency Cepstral Coefficients (MFCCs). **MFCCs are built on the top of the FBANK feature by applying a Discrete Cosine Transformation (DCT)**.
DCT is just a linear transformation that fosters the coefficients to be less correlated. These features were extremely useful before neural networks (e.g, in the case of Gaussian Mixture Models).
Neural networks, however, work very well also when the input features are highly correlated and for this reason, in standard speech processing pipeline FBANKs and MFCCs often provide similar performance. To compute MFCCs, you can run:

The compute_mfccs function takes in input the FBANKs and outputs the MFCCs after applying the DCT transform and selecting n_mfcc coefficients.

## Derivatives
A standard practice is to compute the derivatives of the speech features over the time axis to embed a bit of local context. This is done with **compute_deltas function**.

## Context Window
When processing the speech features with feedforward networks, another standard **approach to embedding a larger context consists of gathering more local frames using a context window**.

The context_window function takes in input a tensor and returns the expanded tensor. The only two hyperparameters are *left* and *right*, that corresponds to the number of past and future frames to add, respectively.

Note that delta and context window can be used for any kind of feature (e.g, FBANKs) and not only for MFCCs.



# Data Augmentation

Similarly to generating features on-the-fly, there are advantages to augmenting data on-the-fly rather than pre-computing augmented data. Our augmentation module is designed to be efficient so you can dynamically apply the augmentations during training, rather than multiplying the size of the dataset on the disk. Besides the disk and speed savings, this will improve the training process by presenting different examples each epoch rather than relying on a fixed set of examples.


The [`speechbrain/processing/speech_augmentation.py`](speechbrain/processing/speech_augmentation.py) file defines the set of augmentations for increasing the robustness of machine learning models, and for creating datasets for speech enhancement and other environment-related tasks. The current list of enhancements is below, with a link for each to an example of a config file with all options specified:

 * Adding noise
 * Adding reverberation
 * Adding babble
 * Speed perturbation
 * Dropping a frequency
 * Dropping chunks
 * Clipping

In order to use these augmentations, a function is defined and used in the same way as the feature generation functions. More details about some important augmentations follows:

## Adding Noise or Reverberation

In order to add pre-recorded noise or reverberation to a dataset, one needs to specify the relevant files in the same way that the speech files are specified: with a csv file. An example, found at `samples/noise_samples/noise_rel.csv`, is reproduced below:

```
ID, duration, wav, wav_format, wav_opts

noise1, 33.12325, $noise_folder/noise1.wav, wav,
noise2, 5.0, $noise_folder/noise2.wav, wav,
noise3, 1.0, $noise_folder/noise3.wav, wav, start:0 stop:16000
noise4, 17.65875, $noise_folder/noise4.wav, wav,
noise5, 13.685625, $noise_folder/noise5.wav, wav,
```

The add_noise function can be defined in yaml:

```
# hyperparams.yaml
noise_folder: samples/noise_samples
add_noise: !speechbrain.processing.speech_augmentation.AddNoise
    csv_file: !ref <noise_folder>/noise_rel.csv
    replacements:
        $noise_folder: !ref <noise_folder>
```

The `.csv` file is passed to this function through the csv_file parameter. This file will be processed in the same way that speech is processed, with ordering, batching, and caching options. When loaded, this function can be simply used to add noise:

```
hyperparams = load_hyperpyyaml(open("hyperparams.yaml"))
noisy_wav = hyperparams.add_noise(wav)
```

Adding noise has additional options that are not available to adding reverberation. The `snr_low` and `snr_high` parameters define a range of SNRs from which this function will randomly choose an SNR for mixing each sample. If the `pad_noise` parameter is `True`, any noise samples that are shorter than their respective speech clips will be replicated until the whole speech signal is covered.

## Speed perturbation

Speed perturbation is a data augmentation strategy popularized by Kaldi. We provide it here with defaults that are similar to Kaldi's implementation. Our implementation is based on the included `resample` function, which comes from torchaudio. Our investigations showed that the implementation is efficient, since it is based on a polyphase filter that computes no more than the necessary information, and uses `conv1d` for fast convolutions.

```
# hyperparams.yaml
speed_perturb: !speechbrain.processing.speech_augmentation.SpeedPerturb
    speeds: [9, 10, 11]
```

The `speeds` parameter takes a list of integers, which are divided by 10 to determine a fraction of the original speed. Of course the `resample` method can be used for arbitrary changes in speed, but simple ratios are more efficient. Passing 9, 10, and 11 for the `speeds` parameter (the default) mimics Kaldi's functionality.


## Other augmentations

The remaining augmentations: dropping a frequency, dropping chunks, and clipping are straightforward. They augment the data by removing portions of the data so that a learning model does not rely too heavily on any one type of data. In addition, dropping frequencies and dropping chunks can be combined with speed perturbation to create an augmentation scheme very similar to SpecAugment. An example would be a config file like the following:

```
# hyperparams.yaml
speed_perturb: !speechbrain.processing.speech_augmentation.speed_perturb
drop_freq: !speechbrain.processing.speech_augmentation.drop_freq
drop_chunk: !speechbrain.processing.speech_augmentation.drop_chunk
compute_stft: !speechbrain.processing.features.STFT
compute_spectrogram: !speechbrain.processing.features.spectrogram
```

```
# experiment.py
hyperparams = load_hyperpyyaml(open("hyperparams.yaml"))

def spec_augment(wav):
    feat = speed_perturb(wav)
    feat = drop_freq(feat)
    feat = drop_chunk(feat)
    feat = compute_stft(feat)
    return compute_spectrogram(feat)
```
