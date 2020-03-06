# The SpeechBrain Toolkit

[![](https://speechbrain.github.io/assets/logo_noname_rounded_small.png)](https://speechbrain.github.io/)

SpeechBrain is an **open-source** and **all-in-one** speech toolkit based on PyTorch.

The goal is to create a **single**, **flexible**, and **user-friendly** toolkit that can be used to easily develop **state-of-the-art speech technologies**, including systems for **speech recognition**, **speaker recognition**, **speech enhancement**, **multi-microphone signal processing** and many others. 

SpeechBrain is currently under development.

# Table of Contents
- [Basics](#basics)
  * [License](#license)
  * [Requirements](#requirements)
  * [Test installation](#test-installation)
  * [Folder Structure](#folder-structure)
  * [How to run an experiment](#how-to-run-an-experiment)
  * [Configuration files](#configuration-files)
  * [Data reading and writing](#data-reading-and-writing)
  * [Execute computation class](#execute-computation-class)
  * [Tensor format](#tensor-format)
  * [Data preparation](#data-preparation)
- [Feature extraction](#feature-extraction)
  * [Short-time Fourier transform (STFT)](#short-time-fourier-transform-(stft))
  * [Spectrograms](#spectrograms)
  * [Filter banks (FBANKs)](#filter-banks-(fbanks))
  * [Mel Frequency Cepstral Coefficients (MFCCs)](#mel-frequency-cepstral-coefficients-mfccs)
- [Data augmentation](#data-augmentation)
- [Neural Networks](#neural-networks)
  * [Training](#training)
  * [Validation](#validation)
  * [Test](#test)
  * [Saving checkpoints](#saving-checkpoints)
  * [Architectures](#architectures)
  * [Normalization](#normalization)
  * [Losses](#losses)
  * [Optimizers](#optimizers)
  * [Learning rate scheduler](#learning-rate-scheduler)
  * [Replicate computations](#replicate-computations)
  * [Residual, Skip, and Dense connections](#replicate-computations)
  * [Classification example](#losses)
  * [Sequence-to-sequence example](#losses)
  * [Regression example](#losses)
- [HMM-DNN speech recogntition](#hmm-dnn-speech-recogntition)
- [End-to-end speech recogntition](#end-to-end-speech-recogntition)
- [Speech enhancement](#speech-enhancement)
- [Speaker recognition and diarization](#speaker-recognition-and-diarization)
- [Multi-microphone processing](#multi-microphone-processing)
- [Developer Guidelines](#developer-guidelines)

# Basics
In the following sections, the basic functionalities of SpeechBrain are described. 

## License
SpeechBrain is licensed under the [Apache License v2.0](https://tldrlegal.com/license/apache-license-2.0-(apache-2.0)) (i.e., the same as the popular Kaldi toolkit).

## Installing Requirements
 ```
cd SpeechBrain
pip install -r requirements.txt
```
## Test Installation
Please, run the following script to make sure your installation is working:
 ```
python test.py
 ```
 
## Folder Structure
The current version of Speechbrain has the following folder/file organization:
- **cfg**: it contains the configuration files of different experiments. 
- **exp**: it contains the results of the experiments run with SpeechBrain
- **speechbrain**: it contains the python libraries
- **samples**: it contains few data samples useful for running debugging experiments
- **tools**:  it contains additional scripts that can be useful for different purposes (e.g, running debugging tests).
- **speechbrain.py**: it is the python script used to run the experiments

## How to run an experiment
In SpeechBrain an experiment can be simply run in this way:

 ```
python spbrain.py config_file
 ```
 
where `config_file`  is a configuration file (formatted as described below).
For instance, the config file *cfg/minimal_examples/features/compute_fbanks_example.cfg* loops over the speech data itemized in *samples/audio_samples/csv_example.csv* and computes the corresponding fbank features.  To run this experiment you can simply type:

 ```
python spbrain.py  cfg/minimal_examples/features/compute_fbanks_example.cfg
 ```
 
The output will be saved in *exp/compute_fbanks* (i.e, the output_folder specified in the config file). As you can see, this folder contains a file called *log.log*, which reports useful information on the computations that have been performed. 
Errors will appear both in the log file and in the standard error. The output folder also contains all the configuration files used during the computations. The file *cfg/minimal_examples/features/compute_fbanks_example.cfg* is called root config file, but other config files might be used while performing the various processing steps. All of them will be saved here to allow users to have a  clear idea about all the computations performed. The fbank features, finally, are stored in the *save* folder (in this case are saved in pkl format).

## Configuration Files
SpeechBrain can be used in two different ways by the users. Users, for instance, can directly import the functions coded in the speechbrain libraries in their own python scripts.  To make this simpler, the functions in our libraries are **properly documented**, also reporting **examples** on how they can be used as stand-alone functions. 

The preferred modality in SpeechBrain, however, is to use the **speechbrain configuration files**. As we will see in this section, the configuration files define an *elegant*, *clean*, *transparent*, and *standard* framework where all the speechbrain modules can be **naturally integrated and combined**. The modularity and flexibility offered by this environment can allow users to code only minimal parts of the speech processing system, leaving unchanged the other parts. In the cfg folder, several examples of configuration files implementing recipes for many different speech processing systems are reported.

Standard configuration files are *flat* and *back-box* list of hyperparameters. Conversely, the SpeechBrain configuration files are designed such that it is immediately clear which function uses a certain hyperparameter. The configuration files also specify how the main functions are combined, thus offering a **"big picture" of the main computations** performed in a relatively compact way.

More specifically, the configuration files are organized ib the following way:

1. First, we **define a set of functions** to be used (with the related hyper-parameters)
2. Then we **define how these functions are combined** to implement the desired functionality. 

To better understand the logic behind the current version, let's take a look into  ```cfg/minimal_examples/features/FBANKs.cfg ```.
This config file is designed to compute *FBANK* features:
 
 ```
[global]
    sample_rate=$sample_rate
    n_fft=1024
    n_mels=40
    left_frames=5
    right_frames=5
    freeze=True
[/global]

[functions]

    [compute_STFT]
        class_name=speechbrain.processing.features.STFT
        sample_rate=$sample_rate
        win_length=25
        hop_length=10
        n_fft=$n_fft
        window_type=hamming    
    [/compute_STFT]

    [compute_spectrogram]
        class_name=speechbrain.processing.features.spectrogram
        power_spectrogram=2     
    [/compute_spectrogram]

    [compute_fbanks]
        class_name=speechbrain.processing.features.FBANKs
            n_mels=$n_mels
        log_mel=True
        filter_shape=triangular
        f_min=0
        f_max=8000          
            freeze=$freeze
        n_fft=$n_fft     
    [/compute_fbanks]

    [compute_deltas]
        class_name=speechbrain.processing.features.deltas
        der_win_length=5
    [/compute_deltas]

    [context_window]
        class_name=speechbrain.processing.features.context_window
        left_frames=$left_frames
        right_frames=$right_frames
    [/context_window]

    [save]
        class_name=speechbrain.processing.features.save
        save_format=pkl    
    [/save]


[/functions]


[computations]
    wav,*_=get_input_var()
    
    # mfcc computation pipeline
    STFT=compute_STFT(wav)
    spectr=compute_spectrogram(STFT)
    FBANKs=compute_fbanks(spectr)

    # computing derivatives
    delta1=compute_deltas(FBANKs)
    delta2=compute_deltas(delta1)

    # concatenate mfcc+delta1+delta2
    fbank_with_deltas=torch.cat([FBANKs,delta1,delta2],dim=-2)

    # applying the context window
    fbank_cw=context_window(fbank_with_deltas)

[/computations]
 ```

### Global section
All the config files have three main sections: **global**, **functions**, and **computations**.  

The *[global]* section **declares some variables** that could be reused in the following part of the config file. For instance, one might want to set a variable called (*device* or *n_fft*) once and use it in other parts of the config file. 

### Function section
The section *[functions]* **defines the set of functions** used in the experiment with the related hyper-parameters (or arguments). Each function must start with the mandatory field *"class_name="* that specifies where the current function is implemented. The other fields correspond to function-specific arguments (e.g, *batch_size*, *win_length*, *hop_length*, etc.). The list of the required hyperparameters of each function can be found in the documentation of the function itself. In this case, we can open the *lib.processing.features.STFT* and take a look into the documentation (or into the self.expected_options) for a list of the expected arguments. As you can see, some arguments are mandatory, while others are optional (in this case the specified default value is taken).
The functions called in the config file are actually classes, that must have an **init** and **call** method (see class *STFT* in *data_processing.py for an example*). 

The initialization method takes in input the parameters, it checks them and then performs all the **computations that need to be done only once** and not be repeated every time the function is called. 

The __call__ method, instead, takes in input a list of input and gives in output a list as well. This method should be designed to perform all the **computations that cannot be done only once** (e.g., updating the parameters of a neural network). Note that for neural network-based classes (like the STFT function in the example), we have a forward method instead of a __call__method.

### Computation section
The section *[computations]* finally defines **how the functions are combined** to implement the desired functionality. In this specific example, we implement the standard pipeline for computing *FBANK* features, where we first compute the Short-Term Fourier Transform, followed by a spectrogram and filter-bank computation. The computation section is a python shell, where simple python commands can be run as well. 

### Hierarchical configuration files
Standard speech processing pipelines are rather complex. In many cases, it could be better to organize the computations in a  more structured way rather than writing a single big configuration file. For instance, one might want to gather the computations for feature extraction in a config file and the neural network computations in another one.
SpeechBrain supports hierarchical configuration files, i.e, configuration files that can call other configuration files. The first configuration file is the root one, while the others are child configuration files. The latter can be called from the root one using the execute_computations function, as we will be described in detail below.


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
we want to read. Each feature/label takes exactly three colums: **[data_name][data_format][data_opts]**. 

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
and open the `label_dict.pkl` that has been created in `/exp/compute_fbanks/`. We can read it with the following command:
```
from speechbrain.data_io.data_io import load_pkl
abel_dict=load_pkl('exp/compute_fbanks/label_dict.pkl')
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
fpkt0_sx188,2.2208125,TIMIT/test/dr3/fpkt0/sx188.wav,wav,, sil hh uw ao th er ay z dh iy ix n l ix m ix dx ih dx ix cl k s cl p eh n s ix cl k aw n cl sil,string
```
### Data writing
Speechbrain can store tensors that are created during the computations into disk.
This operation is perfomed by the `speechbrain.data_io.data_io.save` function.
The current implementation can store tensors in the following formats:
- **wav**: wav file format. It used the sound file writers and supports all its formats.
- **pkl**: python pkl format.
- **txt**: text format.
- **ark** : kaldi binary format.
- **png**: image format (useful to store image-like tensors like spectrograms).
- **std_out**: the output is directly printed into the standard output.

 See for instance `cfg/minimal_examples/features/FBANKs.cfg` to see how one can use it within a configuration file.  The latter is called by the root config file `cfg/minimal_examples/features/compute_fbanks_example.cfg` and saves the FBANK features (using the pkl format) in `exp/compute_fbanks/save` folder.

### Copying data locally
**SpeechBrain is designed to read data on-the-fly from disk**. Copying your data in the local computation node is of crucial importance to reading them quickly. For instance, in most HPC clusters reading several small files from the shared filesystem can be slow and can even slow down the entire cluster (e.g., think about a lustre file system that is designed to read and write large files only, but it is very inefficient to read/write several small files). 

The solution is thus to always read data from the local disk (e.g, in SLURM the local disk is in *$SLURM_TMPDIR*). To do it, we suggest to do the following steps:

1. *If not already compressed, compress your dataset.*
2. *Copy the compressed file in the local disk.*
3. *Uncompress the dataset.*
4. *Process it with SpeechBrain.*

To help users with this operation, we created a function called copy_locally (see lib.data_io.data_io.copy_data_locally), which automatically implements steps 2 and 3. In particular, we can initialize this class with the following parameters:

```
    [copy_locally]
            class_name=data_preparation.copy_data_locally
            data_file=$data_file
            local_folder=$local_data_folder
    [/copy_locally]
```

where data_file must be a single file (e.g. a *tar.gz* file that contains your dataset) and local_folder is the folder on the local computation node where you want to copy and uncompress the data. The function is automatically skipped when the tar.gz file is already present in the local folder. 

## Execute computation class
The *execute_computations* class is the most important core function within speechbrain and we require users to familiarize themselves with it.

The execute_computation class **reads a configuration file and executes the corresponding computations**.
As you can see from the function documentation in *speechbrain.core.py*, this function has only one mandatory argument:

- **cfg_file**: it is the path of the config file that we want to execute. The computations reported in the [computation] section will be executed.

The numerous optional arguments can be used to implement more advanced functionalities, such *data loops*, *minibatch_creation*, *data caching*. We will discuss here the main functionalities. More advanced functionalities will be described in the following sections and in the function documentation.

### Data Loop and minibatch creation
When a csv_file (i.e, the file containing the list of sentences to process) is passed as an argument, execute_computations automatically create mini-batches and loops over all the data. 

For instance, let's take a look into the execute_computations function called in this minimal example: `cfg/minimal_examples/data_reading/read_write_data.cfg`:
```
[global]
    verbosity=2
    output_folder=exp/read_write_mininal
    data_folder=samples/audio_samples
    csv_file=$data_folder/csv_example.csv
[/global]

[functions]    
        
        [loop]
            class_name=speechbrain.core.execute_computations
            cfg_file=cfg/minimal_examples/data_reading/save_signals.cfg
            csv_file=$csv_file
            batch_size=2
            csv_read=wav,spk_id
            sentence_sorting=ascending
            num_workers=2
            cache=True
            cache_ram_percent=75
        [/loop]

[/functions]

[computations]
    
    # process the specified dataset
    loop()    

[/computations]
```

This root config file reads all the data itemized in `samples/audio_samples/sv_example.csv`, creates mini-batches of two sentences (see batch_size argument), and process them as specified in the child config file `cfg/minimal_examples/data_reading/save_signals.cfg`. The field `csv_read=wav,spk_id` can be use to select which set of features/labels read from the CSV file. If not specified, it automatically creates batches for all the features/labels available, otherwise, it creates batches for the subset of data specified (in this case, only wav features).

### Minibatch creation
To better explain how mini-batches are created within speechbrain, let's take a look into the child config file `cfg/minimal_examples/data_reading/save_signals.cfg`:

```
[global]
[/global]

[functions]
    [save]
        class_name=speechbrain.data_io.data_io.save
        sample_rate=16000
        save_format=flac    
    [/save]
[/functions]


[computations]
    id,wav,wav_len,*_=get_input_var()
    save(wav,id,wav_len)
[/computations]
```
This config file defines the function save, that is used to store the output on the disk (in this case using the flac format). The most peculiar line is the following:
```
    id,wav,wav_len,it_id=get_input_var()
```
The special function *get_input_var()* returns by default the following elements: 
- **id**: it is a list containing all the ids of all the sentences in the current batch
- **data_batch** it is a torch.tensor containing the batch of the data specified in the csv_read file
- **data_batch_len** it is a torch.tensor containing the relative length of the batches.
- **it_id** it is an integer containing the iteration id. It will be 0 for the first iteration, 1 for the second and so on.
- **epoch_id** it is the epoch_id (it is non-zero when looping over the same data multiple times).

In this case, we asked our execute_computations functions to only create mini-batches for the wav files and *get_input_var*
thus returns `id, wav, wav_len,it_id`.

To help users better understand the loop over the mini-batches works, let's add some prints into the computation section.
Remember that the computation section is just a python shell and we can use it to add some prints and other python commands useful to debug. 
Let's change the computation section of the child configuration file in the following way:
```
[computations]
    id,wav,wav_len,it_id,epoch_id=get_input_var()
    print(id)
    print(wav.shape)
    print(wav_len)
    print(it_id)
    print(epoch_id)
    sys.exit(0)
    save(wav,id,wav_len)
[/computations]
```
In this case, we print the first batch of data created from the CSV file `samples/audio_samples/sv_example.csv` and we can see the following output:
```
['example5', 'example2']
torch.Size([2, 33088])
tensor([0.4836, 1.0000])
0
0
```
Note that wav is a torch.tensor composed of two signals (batch_size=2).  
Signals are in general of different lengths and to create batches with the same size we have to perform zero paddings. This operation is automatically performed within speechbrain. It is, however, important to keep track of the original length of the signals (e.g, this could be important when we want to save non-padded signals, or when we want to compute neural network cost on valid time steps only). For this reason we also automatically return wav_len, that is a tensor containing the relative lengths of the sentences within the batch (in this case the length of the first sentence is 0.48 times the length of the second sentence)

We return the relative information because it is more flexible than the absolute one. In a standard feature processing pipeline, we can change the temporal resolution of our tensors. This happens, for instance, when we compute the FBANK features from a raw waveform using windows every 10 ms. With the relative measure, we are still able to save the valid time steps, as done in the current example.

If we remove `sys.exit(0)` we will see the following:

```
['example5', 'example2']
torch.Size([2, 33088])
tensor([0.4836, 1.0000])
0
0
['example4', 'example3']
torch.Size([2, 46242])
tensor([0.7571, 1.0000])
1
0
['example1']
torch.Size([1, 52173])
tensor([1.])
2
0
```
As you can see, the CSV file is composed of 5 sentences and we thus have three batches (two composed of two sentences and the last one composed of a single sentences).

### Data sorting

Batches can be created differently based on how the field sentence_sorting is set.
This parameter specifies how to sort the data before the batch creation:
- **ascending**: sorts the data in ascending order using the "duration" field in the csv file.
- **descending**  sorts the data in descending order using the "duration" field in the csv file.
- **random**  sort the data randomly
- **original**  keeps the original sequence of data defined in the csv file. 

Note that if the data are sorted in ascending or descending order the batches will approximately have the same size and the need for zero paddings is minimized. Instead, if sentence_sorting is set to random, the batches might be composed of both short and long sequences and several zeros might be added in the batch. When possible, it is desirable to
sort the data. This way, we use more efficiently the computational resources, without wasting time on processing time steps composed on zeros only. 

### Data caching

### Output variables

# Data augmentation

In addition to adding noise, the [```speechbrain/processing/speech_augmentation.py```](speechbrain/processing/speech_augmentation.py) file defines a set of augmentations for increasing the robustness of machine learning models, and for creating datasets for speech enhancement and other environment-related tasks. The current list of enhancements follows, with links to sample files of each:

 * Adding noise - [white noise example](cfg/minimal_examples/basic_processing/save_signals_with_noise.cfg) or [noise from csv file example](cfg/minimal_examples/basic_processing/save_signals_with_noise_csv.cfg)
 * Adding reverberation - [reverb example](cfg/minimal_examples/basic_processing/save_signals_with_reverb.cfg)
 * Adding babble - [babble example](cfg/minimal_examples/basic_processing/save_signals_with_babble.cfg)
 * Speed perturbation - [perturbation example](cfg/minimal_examples/basic_processing/save_signals_with_speed_perturb.cfg)
 * Dropping a frequency - [frequency drop example](cfg/minimal_examples/basic_processing/save_signals_with_drop_freq.cfg)
 * Dropping chunks - [chunk drop example](cfg/minimal_examples/basic_processing/save_signals_with_drop_chunk.cfg)
 * Clipping - [clipping example](cfg/minimal_examples/basic_processing/save_signals_with_clipping.cfg)

These augmentations are designed to be efficient, so that you can use them on data during training without worrying about saving the augmented data to disk. This also allows using a dynamic training set that can change from epoch to epoch, rather than relying on a static set. In addition, all augmentations should be differentiable, since they are implemented as ```nn.Module```s. Finally, all augmentations have a ```random_seed``` parameter, to ensure that the augmentations are repeatable, and your results are comparable from experiment to experiment.

Aside from adding noise, all augmentations work on a batch level for the sake of efficiency. This means that for smaller datasets and larger batch sizes, the diversity of augmentations applied may be limited. However, the fact that these augmentations can be different for different epochs can make up for this fact.


