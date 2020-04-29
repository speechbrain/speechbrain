# The SpeechBrain Toolkit

[![](https://speechbrain.github.io/assets/logo_noname_rounded_small.png)](https://speechbrain.github.io/)

SpeechBrain is an **open-source** and **all-in-one** speech toolkit based on PyTorch.

The goal is to create a **single**, **flexible**, and **user-friendly** toolkit that can be used to easily develop **state-of-the-art speech technologies**, including systems for **speech recognition**, **speaker recognition**, **speech enhancement**, **multi-microphone signal processing** and many others. 

*SpeechBrain is currently under development*.

# Table of Contents
- [Basics](#basics)
  * [License](#license)
  * [Requirements](#requirements)
  * [Test installation](#test-installation)
  * [Folder Structure](#folder-structure)
  * [How to run an experiment](#how-to-run-an-experiment)
  * [Configuration files](#configuration-files)
  	* [Global section](#global-section)
	* [Function section](#function-section)
	* [Computation section](#computation-section)
  	* [Hierarchical Configuration Files](#hierarchical-configuration-files)
  * [Data reading and writing](#data-reading-and-writing)
  	* [CSV file format](#csv-file-format)
	* [Audio File Reader](#audio-file-reader)
	* [Pkl file reader](#pkl-file-reader)
	* [String Labels](#string-labels)
	* [Data Writing](#data-writing)
	* [Copying Data Locally](#copying-data-locally)
  * [Execute computation class](#execute-computation-class)
	* [Data Loop](#data-loop)
   	* [Minibatch Creation](#minibatch-creation)
	* [Data Sorting](#data-sorting)
	* [Data Loader](#data-loader)
	* [Data Caching](#data-caching)
	* [Output Variables](#output-variables)
	* [Device Selection](#device-selection)
    * [Multi-GPU parallelization](#multi-gpu-parallelization)
  * [Tensor format](#tensor-format)
  * [Data preparation](#data-preparation)
- [Feature extraction](#feature-extraction)
  * [Short-time Fourier transform (STFT)](#short-time-fourier-transform-stft)
  * [Spectrograms](#spectrograms)
  * [Filter banks (FBANKs)](#filter-banks-fbanks)
  * [Mel Frequency Cepstral Coefficients (MFCCs)](#mel-frequency-cepstral-coefficients-mfccs)
  * [Derivatives](#derivatives)
  * [Context Window](#context-window)
- [Data augmentation](#data-augmentation)
  * [Adding noise](#adding-noise)
  * [Adding reverberation](#adding-reverberation)
  * [Speed perturbation](#speed-perturbation)
  * [Other augmentations](#other-augmentations)
- [Neural Networks](#neural-networks)
  * [Data splits](#data-splits)
  * [Training and Validation Loops](#training-and-validation-loops)
	* [Training](#training)
	* [Validation](#validation)
	* [Test](#test)
  * [Saving checkpoints](#saving-checkpoints)
  * [Architectures](#architectures)
  * [Replicate computations](#replicate-computations)
  * [Residual, Skip, and Dense connections](#residual-skip-and-dense-connections)
  * [Normalization](#normalization)
  * [Losses](#losses)
  * [Optimizers](#optimizers)
  * [Learning rate scheduler](#learning-rate-scheduler)
  * [Classification examples](#classification-examples)
	* [Single Prediction Problems](#single-prediction-problems)
	* [Sequence Prediction Problems](#sequence-prediction-problems)
        * [HMM-DNN ASR Example](#hmm-dnn-asr-example)
		* [CTC Example](#ctc-example)
  * [Regression example](#regression-example)
- [HMM-DNN Speech Recognition](#hmm-dnn-speech-recognition)
- [End-to-End Speech Recognition](#end-to-end-speech-recognition)
- [Speech enhancement](#speech-enhancement)
- [Speaker recognition and diarization](#speaker-recognition-and-diarization)
- [Multi-microphone processing](#multi-microphone-processing)
- [Developer Guidelines](#developer-guidelines)
   * [General Guidelines](#general-guidelines)
   * [Python styleguide](#python-style)
   * [Documentation](#documentation)
   * [How to write a processing class](#how-to-write-a-processing-class)
	* [Initialization Method](#initialization-method)
	* [Call Method](#call-method)
	* [Example of Processing Class](#example-of-processing-class)
   * [Merge Requests](#merge-requests)

# Basics
In the following sections, the basic functionalities of SpeechBrain are described. 

## License
SpeechBrain is licensed under the [Apache License v2.0](https://tldrlegal.com/license/apache-license-2.0-(apache-2.0)) (i.e., the same as the popular Kaldi toolkit).

## Development requirements
```
pip install -r requirements.txt
pip install --editable .
```
## Test Installation
Please, run the following script to make sure your installation is working:
```
pytest tests
pytest --doctest-modules speechbrain
```
 
## Folder Structure
The current version of Speechbrain has the following folder/file organization:
- **speechbrain**: The core library
- **recipes**: Experiment scripts and configurations
- **exp**: Top directory under which to save experiment output (by convention)
- **samples**: Some toy data for debugging and testing
- **tools**: Additional, runnable utility script

## How to run an experiment
In SpeechBrain an experiment can be simply run in this way:

```
python recipes/<path>/<to>/<specific>/experiment.py
```
 
The output will be saved in *exp/<outdir>* (relative to current working dir). 
Both detailed logs and experiment output are saved there. 
Furthermore, less detailed are output to stdout
The experiment script and configuration (including possible command-line 
overrides are also copied to the output directory.


## Data Reading and Writing
SpeechBrain is designed to read/write different audio formats (e.g., wav, flac, pcm). Moreover, the toolkit can read data entries (e.g, features or labels) stored in pkl files or directly specified as a string (e.g, spk_ids). All the data entries used in an experiment must be itemized in a comma-separated values (CSV) file, whose format is described in the following sub-section.

### CSV file format
The CSV files must itemize all the sentences, features, and labels to use in an experiment.  For instance, let's open the CSV file in   ```samples/audio_samples/csv_example.csv ``` . The file can be better rendered when opening it with a standard CSV reader, but let's open it as a raw text file for now:

 ```
ID, duration, wav, wav_format, wav_opts, spk_id, spk_id_format, spk_id_opts

example1, 3.260, $data_folder/example1.wav, wav, , spk01, string, 
example2, 2.068, $data_folder/example2.flac, flac, , spk02, string,
example3, 2.890, $data_folder/example3.sph, wav, , spk03, string,
example4, 2.180, $data_folder/example4.raw, raw, samplerate:16000 subtype:PCM_16 endian:LITTLE channels:1, spk04, string,
example5, 1.000, $data_folder/example5.wav, wav, start:10000 stop:26000, spk05, string,
 ```

The first line shows the fields that are specified in the CSV file. The mandatory fields are the following:
- **ID**: it is the unique sentence identifier. It must be different for each sentence. 
- **duration**: it reports how much each sentence lasts in seconds. As we will see later, this can be useful for processing the sentence in ascending or descending order according to their lengths. 

After these two mandatory fields, the CSV contains a variable number of optional fields that depend on the features and labels
we want to read. Each feature/label takes exactly three columns: **[data_name][data_format][data_opts]**. 

- **data_name** is the name given to the data-entry to read (e.g, wav) and contains the paths where the current element is saved (for wav and pkl format) or the label string (for string format). 
- **data_format** specifies the format of the data_name. The currently supported formats are wav, pkl, or string.
- **data_opts** specifies additional options that are passed to the data_reader.

Each line contains a different sentence. For instance, the first line contains:
 ```
example1, 3.260, $data_folder/example1.wav, wav, , spk01, string, 
 ```
For the sentence called *example1*, we have two elements: the wav features read from  $data_folder/example1.wav and the 
corresponding spk_id label "spk01" which is directly specified as a string.  

### Audio file reader
The audio files are read with [**soundfile**](https://pypi.org/project/SoundFile/), and we thus support all its audio formats.  We indeed can read files in *wav*, *flac*, *sphere*, and *raw* formats. Note that audio data in *raw* format are stored without header and thus some options such as *samplerate*, *dtype*, *endian*, and *channels* must be specified in the field data_opts.  

In some cases, users might want to just read a portion of the audio file. It is possible to specify the portion to read by properly setting the data_ops as shown in the following line:

```
example5, 1.000, $data_folder/example5.wav, wav, start:10000 stop:26000, spk05, string,
```
In this case, we read the audio from sample 10000 to sample 26000. 

Soundfile can also read multi-channel data, as you can see in *samples/audio_samples/csv_example_multichannel.csv*

### Pkl file reader
SpeechBrain can read data in pkl format as well. This way we can read additional non-audio speech features, as shown in the following example:
```
ID, duration, pkl_fea, pkl_fea_format, pkl_fea_opts

example1, 3.260, $data_folder/your_pkl_file, pkl,
```
The content of the pkl file must be numpy array. 

### String labels
In speechbrain labels can be specified in a pkl file or, more directly, in a simple text string.
In the example reported in the CSV file section, we report speaker-identities as strings (e.g, example1 => spk01, 
example2 => spk02, ..). 

The big advantage of this modality is that users do not have to necessarily convert all these labels into the corresponding integer numbers (as required for instance during neural network training). Speechbrain will do it for you by automatically creating a label dictionary, as described in the following sub-section.

### Label dictionary
When reading a string label, a label dictionary is automatically created. In particular, during the initialization of the data_loader, we read all the data and we automatically create a dictionary that maps each label into a corresponding integer. 

To take a look into that, let's run 
```
python spbrain.py  cfg/minimal_examples/features/compute_fbanks_example.cfg
```
and open the `label_dict.pkl` that has been created in `/exp/minimal/compute_fbanks/`. We can read it with the following command:
```
from speechbrain.data_io.data_io import load_pkl
abel_dict=load_pkl('exp/minimal/compute_fbanks/label_dict.pkl')
print(label_dict)
```
And you will see the following:
```
>>> label_dict
{'spk_id': {'counts': {'spk05': 1, 'spk02': 1, 'spk04': 1, 'spk03': 1, 'spk01': 1}, 'lab2index': {'spk01': 0, 'spk02': 1, 'spk03': 2, 'spk04': 3, 'spk05': 4}, 'index2lab': {0: 'spk01', 1: 'spk02', 2: 'spk03', 3: 'spk04', 4: 'spk05'}}}
```
In practice the label_dict contains the following information:
- **counts**: for each label entry we provide the number of times it appears
- **lab2index**: it is a mapping from the string label to the corresponding id
- **index2lab**: it is the mapping from indexes to labels 

Note that we also support a list of labels for each sentence. This can be useful to set up phonemes or word labels for speech recognition. In this case, it is simply necessary to add a space between the labels, as shown in this example:
```
ID,duration,wav,wav_format,wav_opts, phn,phn_format,phn_opts
fpkt0_sx188,2.2208125,TIMIT/test/dr3/fpkt0/sx188.wav,wav,, sil hh uw ao th er ay z dh iy ix n l ix m ix dx ih dx ix cl k s cl p eh n s ix cl k aw n cl sil, string
```

If you do not wish to have your string added to a label_dict and converted to an integer, you can turn this behavior off by adding `label:False` to the `xxx_opts` column. For example,

```
ID, duration, wav, wav_format, wav_opts, spk_id, spk_id_format, spk_id_opts
example1, 1.0, xx.wav, wav, , spk1, string, label:False
```

### Data writing
Speechbrain can store tensors that are created during the computations into the disk.
This operation is perfomed by the `speechbrain.data_io.data_io.save` function.
The current implementation can store tensors in the following formats:
- **wav**: wav file format. It used the sound file writers and supports all its formats.
- **pkl**: python pkl format.
- **txt**: text format.
- **ark**: Kaldi binary format.
- **png**: image format (useful to store image-like tensors like spectrograms).
- **std_out**: the output is directly printed into the standard output.

 See for instance `cfg/minimal_examples/features/FBANKs.cfg` to see how one can use it within a configuration file.  The latter is called by the root config file `cfg/minimal_examples/features/compute_fbanks_example.cfg` and saves the FBANK features (using the pkl format) in `exp/minimal/compute_fbanks/save` folder.

### Data sorting

Batches can be created differently based on how the field sentence_sorting is set.
This parameter specifies how to sort the data before the batch creation:
- **ascending**: sorts the data in ascending order using the "duration" field in the csv file.
- **descending**  sorts the data in descending order using the "duration" field in the csv file.
- **random**  sort the data randomly
- **original**  keeps the original sequence of data defined in the csv file. 

Note that if the data are sorted in ascending or descending order the batches will approximately have the same size and the need for zero paddings is minimized. Instead, if sentence_sorting is set to random, the batches might be composed of both short and long sequences and several zeros might be added in the batch. When possible, it is desirable to
sort the data. This way, we use more efficiently the computational resources, without wasting time on processing time steps composed on zeros only. 


### Data Loader
SpeechBrain uses the pytorch dataloader. All the features and labels reported in the csv files are read in parallel using different workers. The option **num_workers** in exec_computations sets the number of workers used to read the data from disk and form the related batch of data. When num_workers=0, the data are not read in parallel.
Please, see the [pytorch documentation on the data loader](https://pytorch.org/docs/stable/data.html) for more details.

### Data caching
SpeechBrain also supports a **caching mechanism** that allows users to store data in the RAM instead of reading them from disk every time.   If we set ```cache=True``` in execute_computation function, we activate this option.
Data are stored while the total RAM used in less or equal than cache_ram_percent. For instance, if we set ```cache_ram_percent=75``` it means that we will keep storing data until the RAM available is less than 75%. This helps when processing the same data multiple times (i.e, when n_loops>1). The first time the data are read from the disk and stored into a variable called **self.cache** in the data loader (see create_dataloader in data_io.py ). **From the second time on, we read the data from the cache when possible**.

## Tensor format.
All the tensors within SpeechBrain are formatted using the following convention:
```
tensor=(batch, time_steps, channels[optional])
```
**The batch is always the first element, while time_steps is always the last one. In the middle, you might have a variable number of channels**.

*Why we need tensor with the same format?*
It is crucial to have a shared format for all the classes that process data and all the processing functions must be designed considering it. In SpeechBrain we might have pipelines of modules and if each module was based on different tensor formats, exchanging data between processing units would have been painful. Many formats are possible. For SpeechBrain we selected this one because
it is commonly used with recurrent layers, which are common in speech applications. 

The format is very **flexible** and allows users to read different types of data. As we have seen, for **single-channel** raw waveform signals, the tensor will be ```tensor=(batch, time_steps)```, while for **multi-channel** raw waveform it will be ```tensor=(batch, time_steps, n_channel)```. Beyond waveforms, this format is used for any tensor in the computation pipeline. For instance,  fbank features that are formatted in this way:
```
(batch, time_step, n_filters)
```
The Short-Time Fourier Transform (STFT) the tensor, instead, will be:
```
(batch, time_step, n_fft, 2)
```
where the "2" is because STFT is based on complex numbers with a real and imaginary part.
We can also read multi-channel SFT data, that will be formatted in this way:
```
(batch, time_step, n_fft, 2, n_audio_channels)
```

## Data preparation
**The data_preparation is the process of creating the CSV file starting from a certain dataset**.  
Since every dataset is formatted in a different way, typically a different data_preparation script must be designed. Even though we will provide data_preparation scripts for several popular datasets (see data_preparation.py), in general, this part should be **done by the users**. In *samples/audio_samples* you can find some examples of CSV files from which it is possible to easily infer the type of information required in input by speechbrain.

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


# Neural Networks
Speechbrain supports different typologies of neural networks, that can be applied to a large variety of speech processing problems. Some minimal working examples of toy tasks for autoencoder, speech recognition (both end-to-end and HMM-DNN), and spk_id can be found in *cfg/minimal_examples/neural_networks/*.
For instance, to run the minimal spk_id experiment, you can type:

```
python spbrain.py cfg/minimal_examples/neural_networks/spk_id/spk_id_example.cfg
```

Similarly to the other examples, the root config file calls the *training loop* (that manages training and validation phases) and finally tests the performance of the network using test data. The set of computations is organized using three configuration files organized with the following hierarchy:


├──  `cfg/minimal_examples/neural_networks/spk_id/spk_id_example.cfg ` (*root config file)*

|--------└── `cfg/minimal_examples/neural_networks/spk_id/training_loop.cfg` (*training/validation  loop)

|----------------└── `cfg/minimal_examples/neural_networks/spk_id/basic_MLP.cfg` (*neural computations*)

In the following subsections, we provide some examples of how a neural speech processing pipeline can be set up within SpeechBrain.

## Data Splits
To set up a neural speech processing pipeline, we need to split our dataset into three different chunks:
- **Training set**: it is used to train the parameters of the neural network.
- **Validation set**: it is used to monitor the performance of the neural network on held-out data during the training phase. The validation phase can be also used for different other purposes, such as hyperparameter tuning, early stopping, or learning rate annealing.
- **Test set**: it is used to check the final performance of the neural network.  

For instance, let's take a look into the following configuration file: `cfg/minimal_examples/neural_networks/spk_id/spk_id_example.cfg`. The configuration files implement a tiny toy example whose goal ss to classify whether a sentence belongs to a speaker 1 or a speaker 2. To do it, we have the following data:
- *samples/audio_samples/nn_training_sample/train.csv*: is it composed of 8 speech signals (4 from speaker1 and 4 from speaker 2). The speaker identities are annotated in the column *spk_id*. This dataset will be used to train the neural network.
- *samples/audio_samples/nn_training_sample/dev.csv*: is it composed of 2 speech signals (1 from speaker1 and 1 from speaker 2). It will be used for validation.
- *samples/audio_samples/nn_training_sample/test.csv*: is it composed of 2 speech signals (1 from speaker1 and 1 from speaker 2). It will be used for test purposes.

## Training and Validation Loops
Let's now take a look into the root config file `cfg/minimal_examples/neural_networks/spk_id/spk_id_example.cfg`.
This configuration file defines and runs two functions executed in sequence:

- **training_validation**: it manages the main training loop that alternates training and validation phases for N epochs.
- **test**: it manages the test phase, that is used to check the performance of the system after training.

The *training_validation* function takes in input the following parameters:
```
[training_validation]
    class_name=speechbrain.core.execute_computations
    cfg_file=$training_computations
    device=$device
    n_loops=$N_epochs
    recovery=True
[/training_validation]
```        

The *training_validation* function runs the computations reported in another config file (i.e., ``cfg/minimal_examples/neural_networks/spk_id/training_loop.cfg``). As we will see, the latter manages both training and validation phases. We also specify the device where all the computations must be executed and the number of training iterations (N_epohcs). Finally, we set ``` recovery=True```  to make sure the training can be resumed from the last epoch correctly executed if training is interrupted for some reason.

### Training
Let's now take a look into the file ``cfg/minimal_examples/neural_networks/spk_id/training_loop.cfg` that is called from the root config file to manage training and validations loops. It defines and runs two functions:
- **training_loop**: it loops over all the training data and trains the neural architecture specified in the configuration file. 
- **validation_loop**: it loops over all the test data check the performance after each training epoch.

The training loop function takes the following arguments:

```
[training_loop]
    class_name=speechbrain.core.execute_computations
    cfg_file=$neural_computations
    csv_file=$csv_train
    csv_read=wav,spk_id
    batch_size=$N_batch
    sentence_sorting=ascending
    stop_at=optimizer
    out_var=loss,error
    accum_type=average,average
    eval_mode=False
[/training_loop]
```

The function runs the computations specified in `cfg/minimal_examples/neural_networks/spk_id/Basic_MLP.cfg`, which implements a simple single-layer MLP. The difference with the previous *traning/validation* loop is that here we specify the data in csv_file (i.e, *csv_file=samples/audio_samples/nn_training_samples/train.csv*). We thus loop over the train data and we create mini-batches, as discussed in the [basics/execute_computations] section. In particular, we create batches for the data entries specified in csv_read:  wav (i.e, the audio waveform) and spk_id (i.e., the speaker label).

The batches are creating using a batch size=2 and the sentences are sorted in ascending order before creating the batches. The latter can be useful to minimize the need for zero paddings. 

Let's now take a look into the variables returned by the training_loop (see  stop_at, out_var, and  accum_type fields). To better understand them, let's open the child configuration file that `cfg/minimal_examples/neural_networks/spk_id/Basic_MLP.cfg` and take a look into the related computation section:

```
[computations]
 
    id,wav,wav_len, spk_id,spk_id_len,batch_id,loop_id,mode,*_=get_input_var()
    
    feats=compute_features(wav)
    feats = mean_var_norm(feats,wav_len)
    out=linear1(feats)
    out=activation(out)
    out=linear2(out)
    out=torch.mean(out,dim=-1).unsqueeze(-1)
    pout=softmax(out)
    
    if mode=='valid' or mode=='train':
        loss,error=compute_cost(pout,spk_id,spk_id_len)
    if mode == 'train':
        loss.backward()
        optimizer(linear1,linear2)
    if mode=='test':
        print_predictions(id,pout,wav_len)
[/computations]
```
As specified in the *stop_at* variable, we stop the computations when the function *optimizer*
is met. Then, we return the variables *loss* , *error* and the average them before finally giving then in output.
In practice, when calling the training_loop function we got the average training loss and classifications errors:

```
# Training Loop
mode='train'
avg_loss,avg_err=training_loop(mode)
```

Note that *training loop* also takes an additional input called *mode*, that is used to discriminate between training and test phases (as you can see from `cfg/minimal_examples/neural_networks/spk_id/Basic_MLP.cfg`, the training phase requires computing the gradient and calling the optimizer, while validation and test phases do not require it).

Finally, the training loop sets the flag `eval_mode=False` to make sure that all the computations are performed with the training flag active (that might impact techniques such as dropout or batch normalization, whose behavior is different from training and test phases).

### Validation
The validation loop is performed after the training one to progressively monitor the performance evolution of the system.  The validation loop takes in input exactly the same neural computations specified in `cfg/minimal_examples/neural_networks/spk_id/Basic_MLP.cfg`. The main difference are the following:
- the loop is performed on the test data (csv_file=samples/audio_samples/nn_training_samples/dev.csv).
- we stop the computations when we meet the *loss* variable (i.e we avoid the optimization step).
- we run the computations with the flag `torch_no_grad=True` active (to avoid computing gradient buffers)
and with `eval_mode=True` (to perform computations in eval modality).

### Test
The final loop over the test data is performed only after having completed the neural training part and it is used to check the final performance of the neural network. The test loop in  `cfg/minimal_examples/neural_networks/spk_id/spk_id_example.cfg` takes the same parameters of the validation loop. The only difference is that the function *test* is called with `mode='test'`:

```
[computations]
    # training/validation epochs
    training_validation()
    
    # test
    mode='test'
    test(mode)
[/computations]
```
As you can see from `cfg/minimal_examples/neural_networks/spk_id/Basic_MLP.cfg`, thisflag activates the function 
*print_predictions* that prints the predictions performed by the neural networks on the test data.

## Saving checkpoints
After each training/validation loop, it is possible to save a checkpoint of the current status of the system. 
The saved checkpoint can be used to resume the training loop when needed. 
Checkpoints can be saved with speechbrain.data_io.data_io.save_ckpt, as shown in  `cfg/minimal_examples/neural_networks/spk_id/training_loop.cfg`. The checkpoint saves all the neural parameters, the optimization status, and the current performance.  All the information needed to recover an experiment is stored in the recovery.pkl dictionary saved in the output folder. See the function documentation for more information on this functionality.

## Architectures
The library in *speechbrain.nnet/architectures.py* contains different types of neural networks, that can
be used to implement **fully-connected**, **convolutional**, and **recurrent models**.  All the models are designed to be combined to create more complex neural architectures. 
Please, take a look into the following configuration files to see an example different architectures:

| Architecture   |      Configuration file      | 
|----------|:-------------:|
| MLP |  *cfg/minimal_examples/neural_networks/spk_id/Basic_MLP.cfg* | 
| CNN |    -  |
| SincNet |    -  |
| RNN | - |
| GRU | - |
| Li-GRU | - |
| LSTM | - |
| CNN+MLP+RNN | - |

In the spk_id example introduced in the previous section, one can simply switch from architecture to another. 

## Replicate computations
Many neural networks are composed of the same basic computation blocks replicated for several different layers.
For instance, the basic computation block for standard MLP can be the following:
```
[functions]    
        
        [linear]
        class_name=neural_networks.linear
        n_neurons=1024
            bias = False
        [/linear]


        [batch_norm]
        class_name=neural_networks.normalize
            norm_type=batchnorm
        [/batch_norm]

        [activation]
        class_name=neural_networks.activation
            act_type=leaky_relu
        [/activation]

        [dropout]
        class_name=neural_networks.dropout
            drop_rate=0.15
        [/dropout]

[/functions]


[computations]
 
    fea,*_=get_input_var()
    out=linear(fea)
    out=batch_norm(out)
    out=activation(out)
    out=dropout(out)

[/computations]
```
In this case, we have in input some features, we perform a linear transformation, followed by batch normalization, application of the non-linearity and dropout. When we add multiple layers, we just have to replicate the same type of computations N times. Instead of requiring users to do by themselves this operation for every single model they implement, Speechrain offers the possibility to grow neural architectures automatically. This is done using the replicate field in the execute_computation functions. 
Do an example of replication

## Residual, Skip, and Dense connections
The advantage of growing neural networks automatically is that we can automatically add additional connections such as residual, skip, and dense connections when required by the user. The current version of SpeechBrain supports the following additional connections:
- **residual connections**: They are created between adjacent blocks
- **Skip connections**: they are created between each block and the final one. 
- **Dense connections**:  They are created between the current block and all the previous ones.
The typology of new connections to create can be selected with the parameter *add_connections* of the execute_computations function.

The new connections can be combined with the existing ones in different ways:
- **sum**: the activations are summed together. This can be done only if the tensors have the same dimensionality (e.g., when the number of neurons is the same for all the hidden layers)
- **diff**: the new activations are the difference between the original ones and the new ones. This can be done only if the tensors have the same dimensionality.
- **average**: the new activations are the average between the original ones and the new ones. This can be done only if the tensors have the same dimensionality.
- **linear_comb**: the tensor corresponding to the new activations are linearly combined with the original ones. The weights of the linear combination are learned. This operation can be done also for tensors with different dimensionality, and the final dimensionality corresponds to the original one.

The modality with which the new connections are combined can be specified with the *connection_merge* parameter of the *execute_computations* function.
Let's now see an example.

Other examples can be found here:

| Architecture   |      Configuration file      | 
|----------|:-------------:|
| MLP |  *cfg/minimal_examples/neural_networks/spk_id/Basic_MLP.cfg* | 
| MLP + Residual Connection |    -  |
| MLP + Skip Connection |    -  |
| MLP + Dense Connection |    -  |

Note that any architecture, including convolutional and recurrent neural networks, can be replicated with the additional connections described in this sub-section.

## Normalization
SpeechBrain implements different modalities to normalize neural networks in speechbrain/speechbrain/nnet/normalization.py. The function currently supports:
- **batchnorm**: it applies the standard batch normalization by normalizing the mean and std of the input tensor over the batch axis.
- **layernorm**: it applies the standard layer normalization by normalizing the mean and std of the input tensor over the neuron axis.
- **groupnorm"**: it applies group normalization over a mini-batch of inputs. See torch.nn documentation for more info.

- **instancenorm**: it applies instance norm over a mini-batch of inputs. It is similar to layernorm, but different statistics for each channel are computed.

- **localresponsenorm**: it applies local response normalization over an input signal composed of several input planes. See torch.nn documentation for more info.

An example of an MLP coupled with batch normalization can be found here:

## Losses
The loss is a measure that estimates "how far" are the target labels from the current network output.
The loss (also known as *cost function* or *objective*) are scalars from which the gradient is computed. 

In Speechbrain the cost functions are implemented in *speechbrain/speechbrain/nnet/losses.py*. Currently, the following losses are supported:
- **nll**: it is the standard negative log-likelihood cost (categorical cross-entropy).
- **mse**: it is the mean squared error between the prediction and the target.
- **l1**:  it is the l1 distance between the prediction and the target.
- **ctc**:  it is the CTC function used for sequence-to-sequence learning. It sums p over all the possible alignments between targets and predictions.
- **error**:  it is the standard classification error.

Depending on the typology of the problem to address, a different loss function must be used. For instance, regression problems typically use *l1* or *mse* losses, while classification problems require often *nll*. 

An important option is *avoid_pad*. when True, the time steps corresponding to zero-padding are not included in the cost function computation.

Please, see the following configuration files to see some examples with different loss functions:

| Loss   |      Configuration file      | 
|----------|-------------|
| mse |  *cfg/minimal_examples/neural_networks/autoencoder/basic_MLP.cfg* | 
| nll |    *cfg/minimal_examples/neural_networks/spk_id/basic_MLP.cfg*  |
| nnl |    *cfg/minimal_examples/neural_networks/DNN_HMM_ASR/basic_MLP.cfg*  |
| ctc |    *cfg/minimal_examples/neural_networks/E2E_ASR/basic_MLP.cfg*  |

As you can see from  *cfg/minimal_examples/neural_networks/autoencoder/basic_MLP.cfg*, the cost function is defined in this way:

```
[compute_cost]
    class_name=speechbrain.nnet.losses.compute_cost
    cost_type=nll,error
[/compute_cost]
```
and it called in the computation section as follows:
```
    if mode=='valid' or mode=='train':
        loss,error=compute_cost(pout,spk_id,spk_id_len)
```
The gradient of the neural parameters over the loss can be computed in this way:
```
    if mode =='train':
        loss.backward()
```
After computing the gradient, the parameters of the neural network can be updated with an optimization algorithm, as described in the following section.

# Optimizers
In SpeechBrain the optimizers are implemented in *speechbrain/speechbrain/nnet/optimizers.py*. All the basic optimizers such as *Adam*, *SGD*, *Rmseprop* (along with their most popular variations) are implemented. See the function documentation for a complete list of all the available alternatives. 

In *cfg/minimal_examples/neural_networks/autoencoder/basic_MLP.cfg*, we used a standard Adam optimizer defined in this way:

```
[optimizer]
    class_name=speechbrain.nnet.optimizers.optimize
    recovery=True
    optimizer_type=adam
    learning_rate=$lr
[/optimizer]
```
The parameter updates is performed by calling the optimizer after the gradient computation:

```
    if mode == 'train':
        loss.backward()
        optimizer(linear1,linear2)
 ```
 As you can see, the optimizer takes in input the names of the functions contaning the parameters.
 After updating the parameters, the optimizer automatically calls the function *zero_grad()* to add zeros in the gradient buffers (and avoid its accumulation in the next batch of data).

## Learning rate scheduler
During neural network training, it is very convenient to change the values of the learning rate. Typically, the learning rate is annealed during the training phase. In *speechbrain/speechbrain/nnet/lr_scheduling.py* we implemented some learning rate schedules.  In particular, the following strategies are currently available:

- **newbob**: the learning rate is annealed based on the validation performance. In particular:
    ```
    if (past_loss-current_loss)/past_loss < impr_threshold:
    lr=lr * annealing_factor
    ```                                              
                                              
- **time_decay**: linear decay over the epochs:
    ```
    lr=lr_ini/(epoch_decay*(epoch)) 
    ```

- **step_decay**:  decay over the epochs with the selected epoch_decay factor
     ```
    lr=self.lr_int*epoch_decay/((1+epoch)/self.epoch_drop)
     ```

- **exp_decay**:   exponential decay over the epochs selected epoch_decay factor
    ```
    lr=lr_ini*exp^(-self.exp_decay*epoch)
    ```
- **custom**:   the learning rate is set by the user with an external array (with length equal to the number of epochs)

In you can find an example of, ...


## Classification examples
Neural speech processing pipelines can be divided into two different categories based on the typologies of the problem that neural networks solve:
- **Classification problems**: The output of this system is a single prediction or a sequence of predictions. *Speech recognition*, *speaker recognition*, *emotion recognition* are all types of problems that fall into this category

- **Regression problems**: the output of this system is a continuous value such as another speech signal. Systems for speech enhancement and speech separation falls into this category.


Let's focus here on classification problems. Based on the output predicted, this problem can be classified in turn into the following categories:

- *Single Prediction problems*: For each sentence, a single prediction is required. **Speaker Recognition** or other tasks that perform high-level global classification over the entire sentence such **language identification**, **emotion recognition**, **accent classification** falls into this category.

- *Sequence Prediction problems*: in this case for each sentence a sequence of predictions is generated. Speech recognition is an example of such type of problems. In particular, **HMM-DNN speech recognition** is a task that requires one phone prediction at every time-steps. This means that the length of the input is equal to the length of the output label. **End-to-end speech recognition**, instead, can predict an output sequence of arbitrary size and input and labels might have different lengths.

### Single Prediction problems
Let's see an example of a single prediction problem, by taking another loop into the toy speaker classification task.
We can run it by typing:
```
python spbrain.py cfg/minimal_examples/neural_networks/spk_id/spk_id_example.cfg
```
The roor config files call the **training_loop*, which in turn performs the computation reported in 
``cfg/minimal_examples/neural_networks/spk_id/spk_id_example.cfg``.  Let's take a look into the first part of the computation section of the latter file:

```
    feats=compute_features(wav)
    feats = mean_var_norm(feats,wav_len)

    out=linear1(feats)
    out=activation(out)
    out=linear2(out)

  
    out=torch.mean(out,dim=-1).unsqueeze(-1)

    pout=softmax(out)

        
    if mode=='valid' or mode=='train':
        loss,error=compute_cost(pout,spk_id,spk_id_len)
```
The neural model takes in input some speech features and process them with a couple of linear transformations and activation functions. Note that after the second linear transformation (```out=linear2(out)```), we have an output vector with the following dimensionality:
```
torch.Size([2, 2, 189])
```
We thus have 2 batches, 2 neurons in output (we have two speakers), and 189 time-steps. 
In practice, we still have one vector for each time-step, while we only have a single label for all the utterances.
One possible approach (not, of course, the only one) is to average all the time steps using the following command:

```
out=torch.mean(out,dim=-1).unsqueeze(-1)
```

After this operation, the dimensionality of the out tensor is

```
torch.Size([2, 2, 1])
```
since we squeeze all the time steps into a single one. We can then apply the softmax function to this single vector, compute the loss, the related gradients and finally update the neural layers:

```
    pout=softmax(out)

        
    if mode=='valid' or mode=='train':
        loss,error=compute_cost(pout,spk_id,spk_id_len)
    
    if mode == 'train':
        loss.backward()
        optimizer(linear1,linear2)
 ```
When we run the root configuration file, the following ouput is produced:
```
    epoch 0: loss_tr=0.8686 err_tr=0.5000 loss_valid=0.6290 err_valid=0.5000 lr_optimizer=0.00040000
    epoch 1: loss_tr=0.4430 err_tr=0.0000 loss_valid=0.3849 err_valid=0.0000 lr_optimizer=0.00040000
    epoch 2: loss_tr=0.2147 err_tr=0.0000 loss_valid=0.2236 err_valid=0.0000 lr_optimizer=0.00040000
    epoch 3: loss_tr=0.1140 err_tr=0.0000 loss_valid=0.1357 err_valid=0.0000 lr_optimizer=0.00040000

Predictions:
id              prob    prediction
spk2_snt6       0.938   spk2
spk1_snt6       0.959   spk1
```

As you can see, the task is extremely easy and can be solved quickly even with a tiny dataset composed of 4 sentences only. The classification error in the validation set is zero, and the same happens on the test set (see the predictions printed). The results are also stored in the output folder:  ```exp/minimal/spk_id_minimal/nnets ```.
This folder contains the last and the best neural models as well as a *res.res* file that reports the evolution of the performance during training.

Even though more sophisticated techniques for more challenging problems, several popular speech processing pipelines based on a single prediction can be approached in the way described in this section.

### Sequence Prediction problems
Let's now see some examples of sequence prediction problems such as speech recognition. 

#### HMM-DNN ASR example
Let's start with an HMM-DNN speech recognition example, where the neural network must provide a prediction for each time step. In a real application, an alignment step over HMM states representing phone-like units must be performed, as will be shown in the HMM-DNN section.  For now, we assume that this step is already done and we read the alignments (i.e, the labels for each time step). In particular, *cfg/minimal_examples/neural_networks/DNN_HMM_ASR/ASR_example.cfg* imports the labels form the pkl files derived from the "ali" column of the csv file (*samples/audio_samples/nn_training_samples/*.csv*). The neural computations are performed in *cfg/minimal_examples/neural_networks/DNN_HMM_ASR/basic_MLP.cfg* and are very similar to that reported in the previous example. The main difference is that in this case, we do not perform any time step squeezing because a prediction in each time step is required.

We can run this example by typing:

```
python spbrain.py cfg/minimal_examples/neural_networks/DNN_HMM_ASR/ASR_example.cfg
```

The results are the following:
```
    epoch 0: loss_tr=3.7217 err_tr=0.9474 loss_valid=3.5079 err_valid=0.8885 lr_optimizer=0.00040000
    epoch 1: loss_tr=3.1739 err_tr=0.6314 loss_valid=3.2666 err_valid=0.7577 lr_optimizer=0.00040000
    epoch 2: loss_tr=2.7578 err_tr=0.5369 loss_valid=3.0712 err_valid=0.7462 lr_optimizer=0.00040000
    epoch 3: loss_tr=2.4065 err_tr=0.4679 loss_valid=2.9143 err_valid=0.7327 lr_optimizer=0.00040000
    epoch 4: loss_tr=2.1042 err_tr=0.3956 loss_valid=2.7846 err_valid=0.7154 lr_optimizer=0.00040000
    epoch 5: loss_tr=1.8383 err_tr=0.3358 loss_valid=2.6772 err_valid=0.6788 lr_optimizer=0.00040000
    epoch 6: loss_tr=1.6040 err_tr=0.2856 loss_valid=2.5920 err_valid=0.6731 lr_optimizer=0.00040000
    epoch 7: loss_tr=1.3985 err_tr=0.2354 loss_valid=2.5244 err_valid=0.6788 lr_optimizer=0.00040000
    epoch 8: loss_tr=1.2182 err_tr=0.1971 loss_valid=2.4688 err_valid=0.6596 lr_optimizer=0.00040000
    epoch 9: loss_tr=1.0596 err_tr=0.1545 loss_valid=2.4227 err_valid=0.6481 lr_optimizer=0.00040000
    epoch 10: loss_tr=0.9204 err_tr=0.1228 loss_valid=2.3857 err_valid=0.6269 lr_optimizer=0.00040000
    epoch 11: loss_tr=0.7990 err_tr=0.0944 loss_valid=2.3570 err_valid=0.6058 lr_optimizer=0.00040000
    ....................
    epoch 24: loss_tr=0.1578 err_tr=0.0029 loss_valid=2.3773 err_valid=0.6154 lr_optimizer=0.00040000
```    

This time a tiny dataset composed on 4 sentences only is not enough to provide good performance in speech recognition (which is notoriously a challenging problem). From these results, it emerges that we can perfectly **overfit the training set** (i.e, training error = 0%), but as expected we  *fail to generalize* well on the validation set (err_valid=61%). The fact that we can overfit a tiny dataset, however, is an important **sanity check**. Implementations containing some bugs, in fact, usually struggle to overfit on tiny datasets like this one. When implementing a neural model, we strongly recommend to always perform an overfitting test as an important debug step.

### CTC example

Let's now see an example that predicts sequences of arbitrary length, as commonly done in the so-called end-to-end speech recognition. In  *cfg/minimal_examples/neural_networks/E2E_ASR/CTC/CTC_example.cfg* we report an example with a technique called Connectionist Temporal Classification (CTC). 
As you can see from the neural computations reported in *cfg/minimal_examples/neural_networks/E2E_ASR/CTC/basic_GRU.cfg* this technique requires a recurrent neural network to work properly. A key difference with the previous examples lies in the cost function. CTC utilizes a special cost function that takes in input the current set of predictions and the target phoneme sequence:

```  
loss=compute_cost(pout,phn,[wav_len,phn_len])
```  

CTC introduces a special *blank* symbol that is used internally by CTC to explore all the possible alignments between the neural predictions and the target labels. 
The cost function exploits a dynamic programming technique very similar to the forward algorithm used in the context of Hidden Markov Models (HMMS).

The minimal CTC example (standard CTC+ greedy decoder) can be executed by typing:

```
python spbrain.py cfg/minimal_examples/neural_networks/E2E_ASR/CTC/CTC_example.cfg
```
The results are the following:

```
    epoch 0: loss_tr=13.0512 wer_tr=93.1280 loss_valid=9.2014 wer_valid=92.7910 lr_optimizer=0.00400000
    epoch 1: loss_tr=6.4152 wer_tr=88.2063 loss_valid=4.7612 wer_valid=94.6429 lr_optimizer=0.00400000
    epoch 2: loss_tr=4.6752 wer_tr=91.7736 loss_valid=4.3100 wer_valid=98.1481 lr_optimizer=0.00400000
    epoch 3: loss_tr=3.5873 wer_tr=90.3361 loss_valid=4.1391 wer_valid=92.7249 lr_optimizer=0.00400000
    epoch 4: loss_tr=2.9066 wer_tr=75.6517 loss_valid=4.5876 wer_valid=94.5106 lr_optimizer=0.00400000
    epoch 5: loss_tr=2.3078 wer_tr=68.7263 loss_valid=4.3725 wer_valid=92.6587 lr_optimizer=0.00400000
    epoch 6: loss_tr=1.7537 wer_tr=53.4681 loss_valid=4.6528 wer_valid=94.5106 lr_optimizer=0.00400000
    epoch 7: loss_tr=1.2776 wer_tr=40.2720 loss_valid=4.5146 wer_valid=90.8730 lr_optimizer=0.00400000
    epoch 8: loss_tr=0.9414 wer_tr=32.6076 loss_valid=4.9586 wer_valid=92.7249 lr_optimizer=0.00400000
    epoch 9: loss_tr=0.6902 wer_tr=24.2274 loss_valid=5.1190 wer_valid=92.7249 lr_optimizer=0.00400000
```

As expected on such a tiny dataset, the neural network overfits the training data but cannot generalize well on the validation set.

### Attention-based example

# Regression example
Some speech processing tasks require the estimation of continuous values. In speech enhancement, we have a noisy speech signal in input and we try to output a clean version of the input waveform. A standard autoencoder is another example of a regression problem. Regression neural networks are normally trained with the Mean-Squared-Error (MSE) or the L1 metrics as losses. 

In *cfg/minimal_examples/neural_networks/autoencoder/autoencoder_example.cfg*, we provide an example of an autoencoder, that takes in input the speech features, it applies a bottleneck, and then tries to predict in the output the original output. The main difference with the previous examples lies in the labels (that are not categorical but contain real values) and the cost functions (in this case we use the standard MSE).

This examples can be executed with the following command:

```
python spbrain.py cfg/minimal_examples/neural_networks/autoencoder/autoencoder_example.cfg
```

The result is the following:

```
  epoch 0: loss_tr=1.0349 loss_valid=0.8065 lr_optimizer=0.00400000
  epoch 1: loss_tr=0.7715 loss_valid=0.6327 lr_optimizer=0.00400000
  epoch 2: loss_tr=0.6052 loss_valid=0.5134 lr_optimizer=0.00400000
..................
  epoch 25: loss_tr=0.1103 loss_valid=0.1627 lr_optimizer=0.00400000
..................
  epoch 50: loss_tr=0.0835 loss_valid=0.1478 lr_optimizer=0.00400000
..................
  epoch 75: loss_tr=0.0760 loss_valid=0.1392 lr_optimizer=0.00400000
..................
  epoch 99: loss_tr=0.0678 loss_valid=0.1297 lr_optimizer=0.00400000
```

Similar to the previous example, we tend to overfit the small training dataset used in this experiment, as witnessed by the consistent difference between training and validation losses.

# HMM-DNN Speech Recognition
# End-to-End Speech Recognition
# Speech enhancement
# Speaker recognition and diarization
# Multi-microphone processing

# Developer Guidelines
The goal is to write a set of libraries that process audio and speech in several different ways. The goal is to build a set of homogeneous libraries that are all compliant with the guidelines described in the following sub-sections.

## General Guidelines
SpeechBrain could be used for *research*, *academic*, *commercial*,*non-commercial* purposes. Ideally, the code should have the following features:
- **Simple:**  the code must be easy to understand even by students or by users that are not professional programmers or speech researchers. Try to design your code such that it can be easily read. Given alternatives with the same level of performance, code the simplest one. (the most explicit and straightforward manner is preferred)
- **Readable:** SpeechBrain mostly adopts the code style conventions in PEP8. The code written by the users must be compliant with that. Please use  *pycodestyle* to check the compatibility with PEP8 guidelines.
We also suggest using *pylint* for further checks (e.g, to find typos in comments or other suggestions).

- **Efficient**: The code should be as efficient as possible. When possible, users should maximize the use of pytorch native operations.  Remember that in generally very convenient to process in parallel multiple signals rather than processing them one by one (e.g try to use *batch_size > 1* when possible). Test the code carefully with your favorite profiler (e.g, torch.utils.bottleneck https://pytorch.org/docs/stable/bottleneck.html ) to make sure there are no bottlenecks if your code.  Since we are not working in *c++* directly, performance can be an issue. Despite that, our goal is to make SpeechBrain as fast as possible. 
- **modular:** Write your code such that is is very modular and fits well with the other functionalities of the toolkit. The idea is to develop a bunch of models that can be naturally interconnected with each other to implement complex modules.
- **well documented:**  Given the goals of SpeechBrain, writing a rich a good documentation is a crucial step. Many existing toolkits are not well documented, and we have to succeed in that to make the difference.
This aspect will be better described in the following sub-section.

## Python 

### Version
SpeechBrain targets Python >= 3.7.

### Formatting
To settle code formatting, SpeechBrain adopts the [black](https://black.readthedocs.io/en/stable/) code formatter. Before submitting merge requests, please run the black formatter on your code.

### Adding dependencies
In general, we strive to have as few dependencies as possible. However, we will debate dependencies on a case-by-case basis. We value easy installability via pip.

### Testing
We are adopting unit-tests using [pytest](https://docs.pytest.org/en/latest/contents.html)


## Documentation
In SpeechBrain, we plan to provide documentation at different levels:

-  **Website documentation**. In the SpeechBrain website, we will put detailed documentation where we put both the written tutorials and descriptions of all the functionalities of the toolkit. 

-  **The SpeechBrain book**: Similarly to HTK (an old HMM-based speech toolkit developed by Cambridge) we plan to have a book that summarized the functionalities of speechbrain. The book will be mainly based on the website documentation, but also summarizing everything in a book, make it simpler to cite us.

-  **Video tutorial**: For each important topic (e.g, speech recognition, speaker recognition, speech enhancement) we plan to have some video tutorials.
  
-  **Function/class header**: For each class/function in the repository, there should a header that properly describes its functionality, inputs, and outputs. It is also crucial to provide an example that shows how it can be used as a stand-alone function.  For each function there must be a header formatted in the following way:

    ```
    library.class/function name (authors:Author1, Author2,..)
    
    Description: function description
     
    Input: input description
     
    Output: output description
    
    Example: an example
    ```

-  **Comments**: We encourage developers to comments well on the lines of code. Please, take a look at some classes/functions in the available SpeechBrain libraries.


### How to write a processing class
The processing classes are those that can be called directly within a configuration file (e.g., all the classes in *processing* and *nnet* libraries). **All these classes must share the input arguments and general architecture**.

**Suggestion**: *The best way to write a novel processing class is to copy and paste all the content of an existing one and do the needed changes.*

The processing classes must have an initialization method followed by a __call__ or **forward** method. 

#### Initialization method
The **initialization** method initializes all the parameters and performs **all the computations that need to be done only once**. 

The only mandatory argument to initialize the class is the following:

- **config** (dict, mandatory): it is a dictionary containing the hyperparameters needed by the current class. For instance, if the config file contains a section like this:
    ```
        [linear1]
            class_name=speechbrain.nnet.architectures.linear
            n_neurons=1024
            bias = False
        [/linear1]
    ```
    The function *speechbrain.nnet.architectures.linear* will receive the following dictionary:
    ```
    config={'class_name':'speechbrain.nnet.architectures.linear',   
            'n_neurons':'1024',
            'bias':'True'}
    ```
    Note that all the values of the dictionary are string at this point of the code (they will cast later in the code).

Beyond the mandatory fields, other optional fields can be given:

- **funct_name** (str, optional, default:None): It is the name of the current function. For instance, in the previous example, it will be  "linear1".

-  **global_config** (dict, optional, default:None): This is the dictionary containing the global variables. If the [global] section of the config  file is the following:
    ```
    [global]
        verbosity=2
        output_folder=exp/read_mininal_example2
    [/global]
     ```

    This dictionary is the following way:
    ```
    global_config={'verbosity':'2', output_folder:'exp/read_mininal_example2'}
    ```
    These variables can be used within the code. 

- **logger** (logger, optional, default: None): This is the logger that must be used to write error, info, and debug messages. In SpeechBrain all the log information is stored in a single file that is placed in $output_folder/log.log

- **first_input** : (list, optional, default: None): This is a list containing the first input provided when calling the class for the first time. In SpeechBrain, **we initialize the class only the first time we call it** (see execute_computations class) and we thus have the chance to analyze the input and make sure it is the expected one (e.g, analyzing number, type, and shapes of the input list).

Note that **all these arguments are automatically passed to the class when executing the related configuration file** withing SpeechBrain.

The initialization method then goes ahead with the definition of the expected inputs. Let's discuss this part more in detail in the next subsection:

##### Expected options
The variable *self.expected_options* should be defined for all the classes that SpeechBrain can execute. As you can see, in *speechbrain.nnet.architectures.linear* it is defined in this way:
``` 
        self.expected_options = {
            "class_name": ("str", "mandatory"),
            "recovery": ("bool", "optional", "True"),
            "initialize_with": ("str", "optional", "None"),
            "n_neurons": ("int(1,inf)", "mandatory"),
            "bias": ("bool", "optional", "True"),
        }
``` 
The *self.expected_variables* is a simple python dictionary that collects all the options expected by this class. Some options are **mandatory** (e.g, *class_name* or *n_neurons*) and an error will be raised if they are missing in the config file. Other options (e.g, *recovery*, *initialize_with*, *bias* ) are **optional** and a **default value** is assigned to them when not specified in the config file.

The  *self.expected_options* dictionary is organized in this way: the key of the dictionary represents the options to check. For each key, a tuple of 2 or 3 elements is expected for mandatory and optional parameters, respectively:

``` 
param_name: (parameter_type, 'mandatory')
param_name: (parameter_type, 'optional', default_value)
``` 

The current version of SpeechBrain supports the following types (defined in the function *check_and_cast_type* of *utils.py* ):
- **"str"**: a single string (e.g, *modality=training* )
- **"str_list"**: a list of strings. Each element must be separated by a comma (e.g., *scp_read=wav,spk_id*)
- **"int"**: a single integer (e.g., *batch_size=4*). For integers it is also possible to specified a valide range. For 'batch_size': ('int(1,inf)','optional','1') means that the specified value of batch_size must be an integer between 1 and infinite.
- **"int_lst**: a list of integers separated by commas (e.g., *layers=1024,1024,512*)
- **"float"**: a single float (e.g, learning_rate=0.001). Similarly to integers, a range can be specified (e.g. *float(0,1.0)*)
- **float_lst**: a list of floats (e.g., *dropout_layers=0.8,0.5,0.0*)
- **bool**: a single boolean (e.g., *drop_last=False*)
- **file**: a single file (e.g, *processing_cfg=cfg/features/features.cfg*)
- **file_lst**: a list of files (e.g, *models=exp/exp1/mdl1.pkl,exp/exp1/mdl2.pkl*)
- **directory**: a single directory (e.g., *save_folder=exp/exp_speech*)
- **directory_lst**: a list of directory (e.g., *save_folders=exp/exp_speech*, *save_folders=exp/exp_speech2*)
- **"one_of"** : it can be used when the selection is limited to a fixed number of options. For instance, the sentence order can be *ascending* or *descending* or *random*, or *original* (i.e, *one_of(ascending,descending,random,original)*). 

##### Checking the options
After defining the self.expected_option dictionary, we can run the *check_opts* function.
The function **check_opts checks if the options set in the config file by the users corresponding to the type expected**. If not, an error will be raised. 
Moreover, the function assigns the specified default value for the optional parameters.  In the current example, for instance, the variable "bias" is not defined and it will be automatically set to True.
The  check_opts not only performs a check over the parameters specified by the users but also **creates and casts the needed parameters**.  For instance, after calling *check_opts* the variable *self.n_neurons* will be created and will contain an integer (self.n_neurons=1024). **The variables created in this case, can be used in the rest of the code**.

In the *speechbrain.nnet.architectures.linear*, the check_opts creates the following variables
``` 
self.recovery=True
self.initialize_with=None
self.n_neurons= 1024
self.bias = True
``` 

Feel free to add some prints after *check_opts* (e.g, *print(self.batch_size)*) to verify it.

##### Checking the first input
As outlined before, we can also have the chance to analyze the first input given when calling the class for the first time.
In  *speechbrain.nnet.architectures.linear*, we expect in input a single tensor and we can thus perform the check-in the following way:
```python 
        # Definition of the expected input
        self.expected_inputs = ["torch.Tensor"]

        # Check the first input
        check_inputs(
            self.conf, self.expected_inputs, first_input, logger=self.logger
        )
```

The check_inputs function from *speechbrain.utils.input_validation* makes sure that the first input is of the expected type.

The initialization then goes ahead with additional operations and performs all computations that are class-specific.For instance *speechbrain.nnet.architectures.linear*, defines the output folder,  performs additional checks in the input shapes and initializes the parameters of the linear transformation.

#### Call Method
The __call__ (or forward method), instead, can be called only after initializing the class. By default, it receives a list in input and returns another list. The input received here is the one reported in the [computation] of the config file. 
For instance, in */cfg/minimal_examples/neural_networks/spk_id/basic_MLP.cfg*, the comoputations section contains:
```
out=linear1(feats)
```
The input_list given when calling the loop function will be thus:

```
# Reading input _list
x = input_lst[0]
```

### Example of Processing Class
To clarify how to write a data_processing class, let's write a very minimal processing function. For instance, let's write a function that takes in input an audio signal and normalizes its amplitude. For a class like this, we can think about the following parameters:
```
        [normalize]
        class_name=speechbrain.data_io.data_processing.normalize_amp
        normalize=True
        [/normalize]
```
If normalize is True we return the signal with a maximum amplitude of 1 or -1, if not we return the original input signal.
Let's now go in *speechbrain.data_io.data_processing.py* and create a new class called normalize_amp:
```python
class normalize_amp(nn.Module):
    """
     -------------------------------------------------------------------------
     data_processing.normalize_amp (author: add here the author name)

     Description:  Add here the function description.
     Input (init): Add here the description of the parameters (see other classes).
     Input (call): Add here the description of the inputs expected when calling the function.
 output(call ): Add there the description of the outputs expected when calling the function.
 -------------------------------------------------------------------------
"""
    def __init__(
        self,
        config,
        funct_name=None,
        global_config=None,
        logger=None,
        first_input=None,
    ):
        super(normalize_amp, self).__init__()
        
# let's add here the initialization part

def forward(self, input_lst):
# let's add here the output class

```
The initialization method will be like this:
```python
     self.logger=logger
      self.expected_options = {
            "class_name": ("str", "mandatory"),
            "normalize": ("bool", "optional", "True"),
        }

        # Check, cast , and expand the options
        self.conf = check_opts(
            self, self.expected_options, config, self.logger
        )

        # Definition of the expected input
        self.expected_inputs = ["torch.Tensor"]

        # Check the first input
        check_inputs(
            self.conf, self.expected_inputs, first_input, logger=self.logger
        )
```

The forward method will simply be:
```python
    def forward(self, input_lst):

        # Reading input _list
        x = input_lst[0]

        # Normalization
    if self.normalize:
        x= x/x.abs().max()
      return [x]
```
This function operates at the signal level and must be called within a data loop. The root config file (that should be saved somewhere, e.g, in *cfg/minimal_examples/basic_processing/normalize_loop.cfg*) will be like this:

```
[global]
    verbosity=2
    output_folder=exp/minimal/normalize_amplitude
    data_folder=samples/audio_samples
    csv_file=samples/audio_samples/scp_example.csv
[/global]

[functions]    
        
        [normalize_loop]
        class_name=core.loop
        csv_file=$scp_file
        processing_cfg=cfg/minimal_examples/basic_processing/normalize_sig.cfg
        [/normalize_loop]

[/functions]

[computations]

    # loop over data batches
    normalize_loop()
    
[/computations]
```

while the processing_cfg (*cfg/minimal_examples/basic_processing/normalize_sig.cfg*) must be created like this:

```
[global]
    device=cuda
[/global]

[functions]

    [normalize]
        class_name=speechbrain.data_io.data_processing.normalize_amp
         normalize=True
    [/normalize]

[/functions]


[computations]
    id,wav,wav_len=get_input_var()
    norm_sig=normalize(wav)
[/computations]
```

You can now run the experiment in the following way:
```
python speechbrain.py cfg/minimal_examples/basic_processing/normalize_loop.cfg
```

Please, take a look at the other data processing class to figure out how to implement more complex functionalities.   

### Merge Requests:

Before doing a pull request:
1. Make sure that the group leader or the project leader is aware and accepts the functionality that you want to implement.
2. Test your code accurately and make sure there are no performance bottlenecks.
3. Make sure the code is formatted as outlined  before (do not forget the header of the functions and the related comments)
4. Run ```python test.py``` to make sure your code doesn't harm some core functionalities. Note that the checks done by test.py are rather superficial and it is thus necessary to test your code in a very deep way before asking for a pull request.
5. Ask for the pull request and review it according to the group leader/project leader feedback. 


