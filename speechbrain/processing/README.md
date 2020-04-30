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

 * Adding noise - [white noise example](cfg/minimal_examples/basic_processing/save_signals_with_noise.cfg) or [noise from csv file example](cfg/minimal_examples/basic_processing/save_signals_with_noise_csv.cfg)
 * Adding reverberation - [reverb example](cfg/minimal_examples/basic_processing/save_signals_with_reverb.cfg)
 * Adding babble - [babble example](cfg/minimal_examples/basic_processing/save_signals_with_babble.cfg)
 * Speed perturbation - [perturbation example](cfg/minimal_examples/basic_processing/save_signals_with_speed_perturb.cfg)
 * Dropping a frequency - [frequency drop example](cfg/minimal_examples/basic_processing/save_signals_with_drop_freq.cfg)
 * Dropping chunks - [chunk drop example](cfg/minimal_examples/basic_processing/save_signals_with_drop_chunk.cfg)
 * Clipping - [clipping example](cfg/minimal_examples/basic_processing/save_signals_with_clipping.cfg)

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

The function can then be defined in a configuration file to load data from the locations listed in this file. A simple example config file can be found at `cfg/minimal_examples/basic_processing/save_signals_with_noise_csv.cfg`, reproduced below:

```
[functions]
    [add_noise]
        class_name=speechbrain.processing.speech_augmentation.add_noise
        csv_file=$noise_folder/noise_rel.csv
        order=descending
        batch_size=2
        do_cache=True
        snr_low=-6
        snr_high=10
        pad_noise=True
        mix_prob=0.8
        random_seed=0
    [/add_noise]
    [save]
        class_name=speechbrain.data_io.data_io.save
        sample_rate=16000
        save_format=flac
        parallel_write=True
    [/save]
[/functions]


[computations]
    id,wav,wav_len,*_=get_input_var()
    wav_noise=add_noise(wav,wav_len)
    save(wav_noise,id,wav_len)
[/computations]
```

The `.csv` file is passed to this function through the csv_file parameter. This file will be processed in the same way that speech is processed, with ordering, batching, and caching options.

Adding noise has additional options that are not available to adding reverberation. The `snr_low` and `snr_high` parameters define a range of SNRs from which this function will randomly choose an SNR for mixing each sample. If the `pad_noise` parameter is `True`, any noise samples that are shorter than their respective speech clips will be replicated until the whole speech signal is covered.

## Adding Babble

Babble can be automatically generated by rotating samples in a batch and adding the samples at a high SNR. We provide this functionality, with similar SNR options to adding noise. The example from `cfg/minimal_examples/basic_processing/save_signals_with_babble.cfg` is reproduced here:

```
[functions]
    [add_babble]
        class_name=speechbrain.processing.speech_augmentation.add_babble
        speaker_count=4
        snr_low=0
        snr_high=10
        mix_prob=0.8
        random_seed=0
    [/add_babble]
    [save]
        class_name=speechbrain.data_io.data_io.save
        sample_rate=16000
        save_format=flac
        parallel_write=True
    [/save]
[/functions]


[computations]
    id,wav,wav_len,*_=get_input_var()
    wav_noise=add_babble(wav,wav_len)
    save(wav_noise,id,wav_len)
[/computations]
```

The `speaker_count` option determines the number of speakers that are added to the mixture, before the SNR is determined. Once the babble mixture has been computed, a random SNR between `snr_low` and `snr_high` is computed, and the mixture is added at the appropriate level to the original speech. The batch size must be larger than the `speaker_count`.


## Speed perturbation

Speed perturbation is a data augmentation strategy popularized by Kaldi. We provide it here with defaults that are similar to Kaldi's implementation. Our implementation is based on the included `resample` function, which comes from torchaudio. Our investigations showed that the implementation is efficient, since it is based on a polyphase filter that computes no more than the necessary information, and uses `conv1d` for fast convolutions. The example config is reproduced below:

```
[functions]
    [speed_perturb]
        class_name=speechbrain.processing.speech_augmentation.speed_perturb
        orig_freq=$sample_rate
        speeds=8,9,11,12
        perturb_prob=0.8
        random_seed=0
    [/speed_perturb]
    [save]
        class_name=speechbrain.data_io.data_io.save
        sample_rate=16000
        save_format=flac
        parallel_write=True
    [/save]
[/functions]


[computations]
    id,wav,wav_len,*_=get_input_var()
    wav_perturb=speed_perturb(wav)
    save(wav_perturb,id,wav_len)
[/computations]
```

The `speeds` parameter takes a list of integers, which are divided by 10 to determine a fraction of the original speed. Of course the `resample` method can be used for arbitrary changes in speed, but simple ratios are more efficient. Passing 9, 10, and 11 for the `speeds` parameter (the default) mimics Kaldi's functionality.


## Other augmentations

The remaining augmentations: dropping a frequency, dropping chunks, and clipping are straightforward. They augment the data by removing portions of the data so that a learning model does not rely too heavily on any one type of data. In addition, dropping frequencies and dropping chunks can be combined with speed perturbation to create an augmentation scheme very similar to SpecAugment. An example would be a config file like the following:

```
[functions]
    [speed_perturb]
        class_name=speechbrain.processing.speech_augmentation.speed_perturb
    [/speed_perturb]
    [drop_freq]
        class_name=speechbrain.processing.speech_augmentation.drop_freq
    [/drop_freq]
    [drop_chunk]
        class_name=speechbrain.processing.speech_augmentation.drop_chunk
    [/drop_chunk]
    [compute_STFT]
        class_name=speechbrain.processing.features.STFT
    [/compute_STFT]
    [compute_spectrogram]
        class_name=speechbrain.processing.features.spectrogram
    [/compute_spectrogram]
[/functions]

[computations]
    id,wav,wav_len,*_=get_input_var()
    wav_perturb=speed_perturb(wav)
    wav_drop=drop_freq(wav_perturb)
    wav_chunk=drop_chunk(wav_drop,wav_len)
    stft=compute_stft(wav_chunk)
    augmented_spec=compute_spectrogram(stft)
[/computations]
```
