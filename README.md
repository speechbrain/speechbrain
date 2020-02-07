# The SpeechBrain Toolkit

<img src="https://speechbrain.github.io/assets/logo_noname_rounded_small.png" img align="left" width="350" height="300" />


[SpeechBrain](speechbrain.github.io/) is an open-source and all-in-one speech toolkit relying on PyTorch.

The goal is to create a **single**, **flexible**, and **user-friendly** toolkit that can be used to easily develop **state-of-the-art speech technologies**, including systems for **speech recognition** (both **end-to-end** and **HMM-DNN**), **speaker recognition**, **speech separation**, **multi-microphone signal processing** (e.g, beamforming), **self-supervised and unsupervised learning**, speech contamination / augmentation, and many others. The toolkit will be designed to be a stand-alone framework, but simple interfaces with well-known toolkits, such as Kaldi will also be implemented.

SpeechBrain is currently under development.

# Disclaimer:
The following documentation is not the final documentation that will be released. It is mainly thought for speechbrain developers that must understand in detail how the toolkit works. Normal users do not have to dive into so many details and before the first release, we will re-organize everything to be much more user-friendly. 

# Table of Contents
- [License](#license)
- [Requirements](#installing-requirements)
- [Test](#test)
- [How to run an experiment](#how-to-run-an-experiment)
- [General Architecture](#general-architecture)
  * [Configuration files](#configuration-files)
  * [Execute Computations](#execute-computations)
- [Data Reading and Writing](#data-reading-and-writing)
	* [Scp file format](#scp-file-format)
	* [Data loop](#data-loop)
	* [Multi-Channel audio](#multi-channel-audio)
	* [Tensor Format](#tensor-format)
	* [Reading non-audio files](#reading-non-audio-files)
	* [Writing Example](#writing-example)
- [Basic processing](#basic-processing)
	* [Adding Noise](#adding-noise)
        * [Other augmentations](#other-augmentations)
- [Feature extraction](#feature-extraction)
	* [Computing STFT](#computing-the-short-time-fourier-transform)
	* [Computing Spectrograms](#computing-spectrograms)
	* [Computing FBANKs](#computing-filter-banks)
	* [Computing MFCCs](#computing-mel-frequency-cepstral-coefficients)
	* [Features for multi-channel audio](#features-for-multi-channel-audio)
- [Data preparation](#data-preparation)
	* [Copy your data locally](#copy-your-data-locally)
	* [LibriSpeech data preparation](#librispeech-data-preparation)
- [Developer Guidelines](#developer-guidelines)
	* [General Guidelines](#general-guidelines)
	* [Folder Structure](#folder-structure)
	* [How to write a processing class](#how-to-write-a-processing-class)
	* [Pull Requests](#pull-requests)
- [Team leader guidelines](#team-leader-guidelines)


## License
SpeechBrain is licensed under the [Apache License v2.0](https://tldrlegal.com/license/apache-license-2.0-(apache-2.0)) (i.e., the same as the popular Kaldi toolkit).

## Installing Requirements
 ```
cd SpeechBrain
pip install -r requirements.txt
```
## Test
 ```
python test.py
```

## How to run an experiment
In SpeechBrain an experiment can be simply run in this way:

 ```
python speechbrain.py config_file
 ```
 
where `config_file`  is a configuration file (formatted as described below).
For instance, the config file *cfg/minimal_examples/features/compute_fbanks_example.cfg* loops over the speech data itemized in *samples/audio_samples/scp_example.scp* and computes the corresponding fbank features.  To run this experiment you can simply type:

 ```
python speechbrain.py  cfg/minimal_examples/features/compute_fbanks_example.cfg
 ```
 
The output will be saved in *exp/compute_fbanks* (i.e, the output_folder specified in the config file). As you can see this folder contains a file called *log.log*, which reports useful information on the computations that have been performed. 
Errors will appear both in the log file and in the standard error. The folder also contains all the configuration files used during the computations. The file *cfg/minimal_examples/features/compute_fbanks_example.cfg* is called root config file, but other config files might be used while performing the various processing steps. All of them will be saved here to allow users having a  clear idea about all the computations performed. The fbank features, finally, are stored in the *save* folder (in this case are saved in pkl format).

Let's now move a bit on trying to better discuss the general architecture of SpeechBrain.  

## General Architecture
SpeechBrain is a toolkit that should be designed to address many different speech tasks such as **speech recognition**, **speaker recognition**, **speech enhancement**, **speech separation**, **multi-microphone signal processing**, and many others.  The general architecture should be suitable for managing such very different applications and should have the following characteristics:
- *Flexibility*
- *Simplicity*
- *Efficiency*

To fulfill these requirements, the current version of the toolkit is simply based on configuration files that work in the following way:

1. First we **define a set of functions** to be used (with the related parameters)
2. Then we **define how these functions are combined** to implement the desired functionality. 

## Configuration Files
To better understand the logic behind the current version, let's take a look into a possible configuration file for computing *FBANK* features:
 ```
[global]
    device=$device
    sample_rate=$sample_rate
    n_fft=400
    n_mels=23
    n_mfcc=20
    left_frames=5
    right_frames=5
    freeze=True
[/global]

[functions]
    [compute_STFT]
        class_name=data_processing.STFT
        device=$device
        sample_rate=$sample_rate
        win_length=25
        hop_length=10
        n_fft=$n_fft
        window_type=hamming    
    [/compute_STFT]

    [compute_spectrogram]
        class_name=data_processing.spectrogram
        power_spectrogram=2     
    [/compute_spectrogram]

    [compute_fbanks]
            class_name=data_processing.FBANKs
            n_mels=$n_mels
           log_mel=True
           filter_shape=triangular
           f_min=0
           f_max=8000          
           freeze=$freeze
           n_fft=$n_fft     
    [/compute_fbanks]

[/functions]

[computations]
     # reading input variables
    id,wav,wav_len=get_input_var()

    # FBANK computation pipeline
    STFT=compute_STFT(wav)
    spectr=compute_spectrogram(STFT)
    FBANKs=compute_fbanks(spectr)
[/computations]
 ```

The config file has three main sections: **global**, **functions**, and **computations**.  

The *[global]* section **declares some variables** that could be reused in the following part of the config file. For instance, one might want to set a variable called (*device* or *n_fft*) once and use it in other parts of the config file. Users can also use the [global] section to highlight at the very beginning of the config file the most important hyper-parameters of the processing algorithm described in the configuration file.

The section *[functions]* **defines the set of functions** used in the experiment with the related hyper-parameters (or arguments). Each function must start with the mandatory field *"class_name="* that specifies where the current function is implemented. The other fields correspond to function-specific arguments (e.g, *batch_size*, *win_length*, *hop_length*, etc.). The functions called in the config file are actually classes, that must have an **init** and **call** method (see class *STFT* in *data_processing.py for an example*). 
The initialization method takes in input the configuration file and it should be used for **setting the specified hyperparameters** and for doing all the **computations that need to be done only once** and not be repeated every time the function is called. 
The __call__ method, instead, takes in input a list of input and gives in output a list as well. This method should be designed to perform all the **computations that cannot be done only once** (e.g., updating the parameters of a neural network). Note that for neural network-based classes (like the STFT function in the example), we have a forward method instead of a __call__method.

The section *[computations]* finally defines **how the functions are combined** to implement the desired functionality. In this specific example, we implement the standard pipeline for computing *FBANK* features. We thus have a first step that computed the Short-Term Fourier Transform, followed by a spectrogram and filter computation. 
SpeechBrain can interpret this simple meta-language and execute the specified computations. This is performed by the execute_computation function in *core.py*. This function reads all the commands and calls the related functions. The functions are initialized only the first time they are called. This has several advantages:
1. We initialize only the functions that we actually use.
2. We can check the first input in the initialization method and raise errors if it is not in the expected format (this is the purpose of  first_input argument in the __init_ method).
3. For neural networks, we can take a look into the shape of the first input and initialize the parameters based on that. This means that users don't have to necessary specified the input dimensionality every time, because it is automatically inferred from the first input.

Note that in more realistic applications (think about a speech recognition system for instance), the configuration files are more complex.  We typically have several high-level steps (e.g, *data_preparation*, *training_nn*, *decoding*, *scoring*) and each step might have several hyperparameters.  To make the configuration files less dense and more readable, SpeechBrain supports a **hierarchy of configuration files**. As we will see later, the main configuration (called *root_cfg*) file might call other configuration files describing phases that require several functions and parameters.

#### Reproducibility:
The global section can contain an optional field called *seed* (e.g, ```seed=1234```).

If the seed is set to an integer, all the operations that involve randomness (e.g, shuffling the lists, initializing the neural networks) will be **done with the same seed**. This means that if we run multiple times an experiment that involves some form of randomness (e.g, neural network training) in the same machine,  it will always produce the same results (note that is is not true if we change machine). If the seed is not set, the random experiment will produce different results every time we run it.

### Execute Computations
Let's describe now the execute_computation class in core.py. This function reads the config file and execute the computations reported in its [computation] section.

This class expects in input the following arguments:

 - **cfg_file**: the config file that contains the computations to execute.
 - **cfg_change**: can be used to change some of the parameters of the cfg_file (e.g, ```cfg_change=--global,device=cuda```).
 - **stop_at**: it is used to stop the computations when the variable reported in stop_at is encountered in the computation section.
 - **root_cfg**: is a flag that indicates in the current config file is the root one or not.

The *call* method of execute_computations reads the computation sections and execute the corresponding functions. **All the functions in the computation section must be previously defined in the section functions**. The only exception are the special function **get_input_var** that returns the variables given in input when calling the execute function and the **pycmd** one that is used to execute raw python code. 

The main script speechbrain.py is extremely simple since it just initializes the execute_computation class with the root config file provided by the user through the command line and executes the computations. Note that some of the processing functions (e.g. core.loop) take in input another config file and execute the related computations by running execute_computations. In practice, this means that **SpeechBrain supports hierarchical computations** (i.e,  sub-computations within other computations).

## Data Reading and Writing
This examples reported so far were only intended to give some insights into the general architecture of the toolkit. In this section, we will introduce a set of more practical minimal examples. The examples will be discussed in order of complexity to allow users to gradually familiarize with the framework.

### Scp file format:
Even though users can define their data loaders, SpeechBrain is designed to read external files (e.g, audio files, features, etc) with a **specific formalism**.
To have a more concreate idea about it, let's open *samples/audio_samples/scp_example.scp*:

 ```
ID=example1 duration=3.260 wav=($data_folder/example1.wav,wav) spk_id=(spk01,string)
ID=example2 duration=2.068 wav=($data_folder/example2.flac,flac) spk_id=(spk02,string)
ID=example3 duration=2.890 wav=($data_folder/example3.sph,wav) spk_id=(spk03,string)
ID=example4 duration=2.180 wav=($data_folder/example4.raw,raw,samplerate:16000,subtype:PCM_16,endian:LITTLE,channels:1) spk_id=(spk04,string)
ID=example5 duration=1.00 wav=($data_folder/example5.wav,wav,start:10000, stop:26000) spk_id=(spk05,string)
```

**Each line corresponds to a different sentence**. The first field (ID) is a mandatory sentence identifier that should be different for each sentence. We then have another mandatory field called *duration*, which reports how much each sentence lasts in seconds. As we will see later, this can be useful for processing the sentence in ascending or descending according to their lengths.  ID and duration are the only mandatory fields, while the following ones are optional and depend on the information needed by the processing algorithm.  

The optional fields formatted in the following way: 
 
 ```
data_name=(data_source, format, [options])
```

For instance, in our scp file the first line   ```"wav=($data_folder/example1.wav,wav)"```  reads the file ```$data_folder/example1.wav```. 
The audio files are read with [**soundfile**](https://pypi.org/project/SoundFile/), and we thus support all its audio formats. In the following lines, we indeed read files in *flac*, *sphere*, and *raw* formats. Note that audio data in *raw* format are stored without header and thus some options such as *samplerate*, *dtype*, *endian*, and *channels* must be specified externally.  The last line shows how to read just a segment of  the entire file:
```
wav=($data_folder/example5.wav,wav,start:10000, stop:26000) 
```
In this case, we read the audio from sample 10000 to sample 26000.  This feature can be useful when the dataset has long files and only some segments of it should be processed. Soundfile can also read multi-channel data, as you can see in *samples/audio_samples/scp_example_multichannel.scp*

Moreover, data can be read in non-audio formats. For instance, feature vectors can also be read in pkl format in the following way:
```
feat=(file_vector.pkl,pkl)) 
```
One can simply read strings in this way:
```
spk_id=(spk01,string)
```
In the latter case, we simply associate the string *"spk01"* to the data_name *"spk_id"*. Beyond that, the string format might be used to report the text transcription of a speech file:
```
transcription=("hello_how_are_you", string)
```
Note that words must be separated by the special character *"_"*. Adding blank spaces between words will cause an error because spaces are special characters used in this formalism to separate data_entries (e.g, ```ID=... duration=... wav=... transcription=... spk_id=...```) 

Note that the data types (e.g, *wav*, *spkid*) that should be specified in the scp files depends on the specific speech application that needs to be addressed.
For instance, if we would like to do end-to-end speech recognition there could be a field called wav (or any other name) to read the waveform and a field called transcription as a text label. If you want to address speaker recognition there could be wav, and spk_id field that reports the speech label. For unsupervised learning (e.g. a waveform autoencoder), only the field wav could be present.  
**The scp files must be created by the users**. Datasets, in fact, are formatted in different ways and users will have to retrieve the needed information with some specific scripts (see data_preparation part). A good idea is to retrieve all the relevant information from your dataset (e.g, waveforms, speaker_id, text) and create **a single scp file useful for multiple tasks**. As we will see later,  in the configuration file we can define which data of the scp file we want to read without necessarily reading all the data_entries. 

### Data loop
Once the scp file is ready, *how this is used within the SpeechBrain toolkit to actually read the data*? To better understand this aspect, let's open the following configuration file:
```
cfg/minimal_examples/data_reading/read_data_example.cfg
```
```
[global]
    verbosity=2
    output_folder=exp/read_mininal_example
    data_folder=samples/audio_samples
    scp_file=samples/audio_samples/scp_example.scp
[/global]

[functions]    
        
        [loop]
        class_name=core.loop
        scp=$scp_file
        [/loop]

[/functions]

[computations]

    # loop over data batches
    loop()    
    
[/computations]
```

This very minimal example just **reads the data itemized in the scp file and loops over them**.  To run it, you have to type:
```
python speechbrain.py cfg/minimal_examples/data_reading/read_data_example.cfg
```

The script  speechbrain.py  will simply execute the computations specified in the config file (using the function *execute_computations* that will be discussed more in detail later). As we will see, the current config file just loops over the specified scp files without doing any additional processing.  Note that is is possible to change the parameters of the config file with the command line. For instance:

``` 
python speechbrain.py  cfg/minimal_examples/data_reading/read_data_example.cfg global,data_folder=/new_samples/samples
```

will change the parameter *data_folder* with */new_samples/samples*, while
``` 
python speechbrain.py  cfg/minimal_examples/data_reading/read_data_example.cfg functions,loop,scp_file=new_file.cfg
``` 
will change *scp_file=$scp_file* with new_file.cfg.

As you can see, the global section contains some variables. For a root config file like this one, the fields *verbosity* and *ouput_folders* are mandatory. *Verbosity* is a number between 0 (no prints) and 2 (highest verbosity). *Output_folder* instead is the reference folder where all the results and log files will be saved.   For instance, ....

The data_folder reports the folder where the data are stored, while scp_file is the list of the files that will be used for this experiment.  Note that the scp file contains filename like this:
``` 
$data_folder/example1.wav
``` 
where the variable $data_folder will be automatically replaced with data_folder path reported in the global section (the scp file reader has access to the global variables defined in the config file by the user). This is a pretty useful feature because one **can change the path of the dataset without changing every time the paths within the scp file**. As we will see later, SpeechBrain is designed to read data from the disk of the local computation node. In a standard HPC cluster this node changes every time and to read data from its **local disk**, we only have to change the variable $data_folder in the global section. 
The data loop itself is implemented in the *loop class* of the *core.py* library (as one can see from its "class_name" field).  



#### The loop class
Let's take a more detailed look into this very important class.
The **loop** class implements several useful functionalities. Let's thus describe its parameters more in detail:

- **scp** (file, mandatory): it is the scp file containing the list of data 
- **processing_cfg** (file, optional, None): this field might contain another config file that will be used to process the input data. If not specified, we will just loop over the scp data without doing any additional processing. The computation section of the processing_cfg file can contain a special function called **get_inp_var()** (see for instance cfg/minimal_examples/features/STFT.cfg). This function returns the batch list passed in ``` self.loop_computation(batch)``` . The list is compososed of the sentence_id followed by data, data_len of the data reported in the scp file and read with the scp variable (see description later). Finally the list contain the batch_id, the iteration_id, as well as the inputs specified when calling the loop class.
- **cfg_change** (str, optional, None): this option can be used to change from the current config file the options of the *processing_cfg* config file. The syntax to change the options is the same as the one used from the command line (see the example of command line arguments before). For instance, if I have to set *cfg_change =global,glob_opt=new_value, functions,funct1,field=new_value2* to change the options (global,glob_opt and functions,funct1,field) in the processing_cfg file. If the processing_cfg file is not specified this functionality has no effect.
- **torch_eval** (type:bool, optional, default:False): if True, the computations will be performed with the flag ```torch.no_grad()``` as required in the test/validation modality.
- **stop_at** (type: str_list,optional, default: None): when the processing_cfg file is set, stop_at stops the execution of the processing function when the given variables or function names are encountered (by default, we return the values observed the last time the variable is assigned. It can be useful when we have to run only a part of the computations reported in the  processing_cfg.
- **out_var** (type: str_list, optional, default: None): it is used to define the variables of computation section that will be returned when calling execute_computation.
- **accum_type** ('list','sum','average','last', optional, default: None): this variable defines the way the *out_var** variables are accumulated over the different iterations of the loop. 
If set to '*list*', the output variables will be accumulated into a list. The list has the same number of elements of *out_var*. Each element is composed of another list that contains each returned ouput variable (i.e, each element will contain a list of  *n_loop*data_size* elements). If set to '*sum*', the elements are summed up, while if set to 'average' the elements are averaged (this is useful, for instance, when computing the loss at each iteration and I want to return the average loss over all the iterations). If set to '*last*', only the last returned  element is saved (this can be useful for  instance when we want to return the final model at the end of the training loop).
- **batch_size**: (int(1,inf),optional,1): the data itemized in the scp file are automatically organized in batches. In the case of variable size tensors, zero padding is performed. When *batch_size=1*, the data are simply processed one by one without the creation of batches.
- **sentence_sorting** ('ascending,descending,random,original', optional, 'original'):  This parameter specifies how to sort the data before the batch creation. *Ascending* and *descending* values sort the data using the "duration" field in the scp files. *Random* sort the data randomly, while *original* (the default option) keeps the original sequence of data defined in the scp file. Note that this option affects the batch creation.  If the data are sorted in ascending or descending order the batches will approximately have the same size and the need for zero padding is minimized. Instead, if sentence_sorting is set to random, the batches might be composed of both short and long sequences and several zeros might be added in the batch. **When possible, it is desirable to sort the data**. This way, we use more efficiently the computational resources, without wasting time on processing time steps composed on zeros only. Note that is the data are sorted in ascending/descending errors the same batches will be created every time we want to loop over the dataset, while if we set a random order the batches will be different every time we loop over the dataset.
- **scp_read** (str_list,optional,None): this option can be used to read only some data_entries of the scp file. For instance, in the aforementioned scp file you can set it to *scp_read=wav* if you only want to read wav signals or *scp_read=wav,spk_id* if you also wanna read spk_id labels. When not specified, it automatically reads all the data entries (in our case it will read both wav and spk_id labels).
- **select_n_sentence**s (int(1,inf),optional,None): this option can be used to read-only n sentences from the scp file. This option can be useful to debug the code, when instead of running an experiment of a full set of data I might just want to run it with a little about of data.
- **num_workers** (int(0,inf),optional,0): data are read using the pytorch data_loader. This option set the number of workers used to read the data from disk and form the related batch of data. Please, see the pytorch documentation on the data loader for more details.
- **cache(bool,optional,False)**: When set to true, this option stores the input data in a variable called self.cache (see create_dataloader in data_io.py). In practice, the first time the data are read from the disk, they are stored in the cpu RAM. If the data needs to be used again (e.g. when loops>1) the data will be read from the RAM directly. If False, data are read from the disk every time.  Data are stored until a certain percentage of the total ram available is reached (see cache_ram_percent below)
- **cache_ram_percent (int(0,100),optional,75)**: If cache if True, the data will be stored in the cpu RAM until the total RAM occupation is less or equal than the specified threshold (by default 75%). In practice, if a lot of RAM is available several data will be stored in memory, otherwise, most of them will be read from the disk directly.
- **drop_last (bool,optional,False)**: this is an option directly passed to the pytorch dataloader (see the related documentation for more details). When True, it skips the last batch of data if contains fewer samples than the other ones. 

Most of the functionalities itemized here will be clarified with the minimal examples reported in the following.

After checking and casting the parameters, the **init** method creates the dataloader using *create_dataloader* from *data_io.py*. As you can see from the latter, the dataloader first reads the scp file and covert in into a dictionary (see *data_dictionary* created by *self.generate_data_dict*). The relative dataset and dataloaders are then initialize using the corresponding pytorch function. Finally, the init method of loop create a dictionary (called exec_config) that is used to initialize the execute function for the processing computations reported in the processing_cfg file.

The method **call** of the class loop (from *core.py*) just loops over the data specified in the scp file  n_loop times:

 
```python
        for i in range(self.n_loops):
            # processing the specified data 
            for data_list in zip(*self.dataloader):
                
                # process data list to have a list formatted in this way
                # [snt_id,data1,data1_len,data2,data2_len,..]
                batch=self.prepare_data_list(data_list)
                
                
                # now, you can process the batch as specified in self.processing_cfg
                if self.processing_cfg!=None:
                    result=self.loop_computation(batch)
```
As you can see, when the *processing_cfg* file is specified, the batches created by the dataloader are processed.  In the example reported in *cfg/minimal_examples/data_reading/read_data_example.cfg* the field *processing_cfg* is not specified and the latter step is skipped. 

Let's not take a closer look into the batches by adding the following print after the variable batch in the code snippet before:
```python
                print(batch)
                print(type(batch))
                print(len(batch))
```

You should see the following output:
```python
[['example1'], tensor([[2.4414e-04, 1.8311e-04, 1.2207e-04,  ..., 9.1553e-05, 4.5776e-04,
         1.8311e-04]]), tensor([1.]), [['spk01']], tensor([1.])]
<class 'list'>
5
[['example2'], tensor([[-0.0009, -0.0005, -0.0008,  ..., -0.0027, -0.0031, -0.0027]]), tensor([1.]), [['spk02']], tensor([1.])]
<class 'list'>
5
[['example3'], tensor([[-0.0078, -0.0078, -0.0078,  ..., -0.0078,  0.0000,  0.0000]]), tensor([1.]), [['spk03']], tensor([1.])]
<class 'list'>
5
[['example4'], tensor([[0.0036, 0.0034, 0.0038,  ..., 0.0032, 0.0027, 0.0022]]), tensor([1.]), [['spk04']], tensor([1.])]
<class 'list'>
5
[['example5'], tensor([[ 1.5259e-04, -9.1553e-05,  6.1035e-05,  ..., -1.1597e-03,
         -1.1902e-03, -1.3123e-03]]), tensor([1.]), [['spk05']], tensor([1.])]
<class 'list'>
5
```

As you can see the variable batch is not a tensor, but a list that contains the following information:
```
[ID, data, data_len, data, data_len,...]
```

By default, the first element of the list is the list of sentence IDs present in the batch. In this case, the *batch_size* is 1 and the list is composed of a single ID (e.g., for the first batch we have *['example1']*, for the second *['example_2']*,...). The following elements are the data in the order reported in the scp file:
For instance, in the */samples/audio_samples/scp_example.scp*, the data order is the ```wav, spk_id ```.
Consequently,  the batch list will be formatted in the following way:
```
[ID, wav, wav_len, spk_id, spk_id_len]
```
Note that after each data, we also report the time duration of it. This information is more useful when we have multiple batches and zero padding is performed. With this information, we can retrieve the original data without zero padding. The information reported here is actually this one:
```
wav_len=time_steps_sentence/time_steps_batch
`````
If the batch_size is 1 like in this case, wav_len will always be one. 

When the number of batches is larger than one it will be a number between 0 and 1.  We decided to report this information rather than directly the number of time_steps because the processing algorithm might change the temporal resolution. 

For instance, let's assume we have a speech signal that lasts 16000 samples that is embedded in a batch of sentences that lasts 32000 samples (i.e. the length of the longest sentence in the batch). Let's compute a set of MFCCs on this batch every 160 samples. The batch size will now be 200 (i.e, 32000/160), while the current sentence will last 100 (i.e, 16000/160). If we want to save the MFCC we only have to save 100 steps and not 200. 

The *len* information output by the data_loader for this sentence will be 0.50 (i.e, 16000/32000). To save the correct number of MFCC steps we just have to multiply the time steps on the MFCC batch (i.e, 200) by the length factor (i.e, 0.5). This gives 100, that are exactly the time steps that we want to save for that sentence. In short, giving the relative length rather than the absolute one allows you to have the **right number of times steps even when the time resolution is changed**. 

Let's now try to play with some parameters of the *loop* function. For instance, let's sort the data in ascending order by adding ```sentence_sorting=ascending``` in the loop section of the config file *cfg/minimal_examples/data_reading/read_data_example.cfg*. 

If we run again the script we can see that sentences are read in the following order:
```
['example5']
['example2']
['example4']
['example3']
['example1']
```

Data are ordered from the shortest to the longest sentence (based on the duration field reported in the scp file). By setting ```sentence_sorting=descending```, the order will be exactly the opposite:

```
['example1']
['example3']
['example4']
['example2']
['example5']
```

Let's now set  *batch_size=2* (with "sentence_sorting=ascending). As you can see from the output the batch variable  will now contain two examples (exept for the last batch):

```
batch 1
[['example5', 'example2'], tensor([[ 1.5259e-04, -9.1553e-05,  6.1035e-05,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        [-8.8501e-04, -4.5776e-04, -8.2397e-04,  ..., -2.6550e-03,
         -3.1128e-03, -2.7466e-03]]), tensor([0.4836, 1.0000]), [['spk05'], ['spk02']], tensor([1., 1.])]

batch 2
[['example4', 'example3'], tensor([[ 0.0036,  0.0034,  0.0038,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0078, -0.0078, -0.0078,  ..., -0.0078,  0.0000,  0.0000]]), tensor([0.7571, 1.0000]), [['spk04'], ['spk03']], tensor([1., 1.])]

batch 3

[['example1'], tensor([[2.4414e-04, 1.8311e-04, 1.2207e-04,  ..., 9.1553e-05, 4.5776e-04,
         1.8311e-04]]), tensor([1.]), [['spk01']], tensor([1.])]
```

The total number of batches can be computed as ```N_sentences/batch_size (in this case, 5/2=2.5)```. Since this ratio in not an integer, the last batch will be composed of fewer samples (exactly one in this case). To avoid computing the last batch when it contains fewer samples that the other ones, the user has just to set drop_last=True in the config file (try to check it).  

Let's now take a look into the dimensionality of the tensors/lists returned by the dataloader. As outlined before, the first element returned is a list containing the sentence IDs of the data composing the minibatch (e.g, ['example5', 'example2'] for the batch 1). We then return a tensor that contains the original waveform. If you print its dimensionality (add print(batch[1].shape in the __call__ method of the loop class) you can see the following 
shapes:
```
torch.Size([2, 33088])
torch.Size([2, 46242])
torch.Size([1, 52173])
```

As we will discuss more in detail the "tensor format" section, all the returned tensors will have be formatted in the following way:

```
tensor=[n_batches, channels[optional], time_steps]
```

In this case, we are reading single-channel waveform and the number of channels is equal to 1 (thus skipped).  This convention is very important to keep in mind when developing new processing functions (**batch_size always the first, time_steps always the last, in the middle a variable number of optional channels**).

As outlined before, after returning this data tensors, the dataloader returns the corresponding normalized lengths. For the first batch we have (add  ```print(batch[2])``` in loop):
```
tensor([0.4836, 1.0000])
```
This means that the first sentence (i.e, *example5*), the first 48% time steps of the *batch_size* contain the actual sentence, while the last 52% contain zeros. In this case, the flag *sentence_sorting=ascending* is still active and the last sample *example2* of bach  is always the longest one (i.e, no zero padding). 
The dataloader then returns the *spk_id*. Please, add ```print(batch[3]``` in loop) to take a look into it:

```
[['spk05'], ['spk02']]
[['spk04'], ['spk03']]
[['spk01']] 
```

In this case, the data is not a tensor, but a list. This is because the data "spk_id" are of type "string" in the scp file. **By default this data has dimension=1**. Thus is we take a look into spk_len by adding print(batch[4] in loop) we will see the following output:

```
tensor([1., 1.])
tensor([1., 1.])
tensor([1.])
```

Let's try now to loop multiple times over the dataset by setting *n_loops=4*. This might be useful, for instance, in the context of neural networks where the same dataset is processed multiple times (i.e., for multiple epochs) before converging to the final solution. If you run the config file with these options active, you will  read the following data:
```
loop 1
['example5', 'example2']
['example4', 'example3']
['example1']

loop 2
['example5', 'example2']
['example4', 'example3']
['example1']

loop 3
['example5', 'example2']
['example4', 'example3']
['example1']

loop 4
['example5', 'example2']
['example4', 'example3']
['example1']
```

Note that in the current version of the config file the field *scp_read* is not specified and by default, we read all the data entries in the scp file. As we have seen, in fact, the dataloader reads both wav and spk_id data.  Let's assume now that we only have to process the original waveform indicated as "wav" in the scp file.
If we set ```scp_read=wav```, the dataloader will return a list composed in this way:
```
[ID, wav, wav_len]
```
If you add these options and you ```print(len(batch))``` in the loop you will see that a list composed of three elements is returned in this case.
If we set  ```scp_read=spk_id``` the dataloader will output:
```
[ID, spk_id spk_id_len]
```
while if we set scp_read=wav,spk_id the list will be again:
```
[ID, wav, wav_len, spk_id spk_id_len]
```
Note that the order reported in *scp_read* matters. For instance, if you set *scp_read=spk_id,wav* the list returned by the dataloader will be:
```
[ID, spk_id spk_id_len,wav, wav_len]
```
As outlined before, if scp read is not specified the dataloader will automatically read all the data entries and will return a list whose elements are composed based on the order of data reported in the scp file (e.g. if we have wav= and then spk_id= we will return ```[ID, wav,wav_len, spk_id,spk_id_len]```)

The scp_read option can be really useful to read only the needed data, **without wasting computational resources to read data entries that are not used**.

Let's now set ```select_n_sentences=2``` in the config file. This option will just read the first two lines of the scp file. If you print the read ID is the data_io class (```print(batch[0])```), you can see that we are only reading:

```
['example2', 'example1']
```

which corresponds to the first 2 sentences reported in the scp file.

This option can be used to read a subset of the input and could be useful for debugging purposes (we want to make sure that the code runs with few sentences only).

Le'ts now open the *cfg/minimal_examples/data_reading/read_data_example2.cfg**  to take a look into other options of the loop class.

We can run it in the following way:
```
python speechbrain.py cfg/minimal_examples/data_reading/read_data_example2.cfg
```

In this case, we ```set num_workers=2```. This means that the reading we read data from the disk with multiple processes.  According to our experience, it is not alway advantageous setting a high number of workers.  SpeechBrain also supports a **caching mechanism** that allows users to store data in the RAM instead of reading them from disk every time.   If we set ```cache=True``` in the config file, we activate this option.
data are stored while the total RAM used in less or equal than cache_ram_percent. For instance, if we set ```cache_ram_percent=75``` it means that we will keep storing data until the RAM available is less than 75%. This helps when processing the same data multiple times (i.e, when n_loops>1). The first time the data are read from the disk and stored into a variable called **self.cache** in the dataloader (see create_dataloader in data_io.py ). **From the second time on, we read the data from the cache when possible**.

The loop class described before loops over the scp_file and creates the corresponding batches. SpeechBrain supports also a more basic form of loop, that just repeats the computations reported in the cfg_file n_loops times. You can run this very basic example in the following way:

```
python speechbrain.py cfg/minimal_examples/data_reading/loop_example.cfg.cfg
```

As you can see from the config file, the class that is called now is **core.loop**.
The loop class is slightly different from the loop one. The latter, in fact, reads the scp data and creates the batches, while loop only replicates the computations reported in the processing cfg_file multiple times. It can be useful, for instance, when we want to run the same set of computations for multiple times.
Think for instance to a training loop of a neural network, where we execute exactly the same computations for a certain number of epochs.




### Multi-Channel audio
SpeechBrain supports multi-channel data. As an example try to open the file *samples/audio_samples/scp_example_multichannel.scp*:
```
ID=example1 duration=3.260 wav=($data_folder/example_multichannel.wav,wav) spk_id=(spk01,string)
```
This file contains a single sentence that is composed of two channels. 

Let's now replace ```scp_file=$data_folder/scp_example.scp``` with  ```scp_file=$data_folder/scp_example_multichannel.scp``` in the  *cfg/minimal_examples/data_reading/read_data_example2.cfg* and run the it again.
If we add  ```print(batch[1].shape)``` in the __call__ method of the loop class we will see the dimension of the multi-channel tensor:

```
torch.Size([1, 2, 33882])
```
Differently from before, we have now a tensor composed of three elements: **batch, channels, time_steps**.

### Reading non-audio files
Even though so far we only have read audio waveforms, **the dataloader can read any kind of data**. As an exercise try to read the features computed so far (e.g, STFT, FBANKs, and MFCCs). In the previous examples, the option create_scp was set to true. All you have to do is to change the field "scp=" with the new scp_file and change "scp_read=" with the corresponding feature (e.g, stft)

In general, **we encourage users to directly feed the raw waveform directly in SpeechBrain and compute the speech features on-the-fly**. The computation of the standard features reported below is extremely computationally efficient, especially if performed on gpu with batch_size >1.  The computation of standard features on-the-fly thus leads to a very minimal computational overhead, but has several advantages:
1. **We can tune some important parameters of the feature extraction**, such as the central frequency and the band of each filter
2. **We can perform data augmentation on-the-fly** by contaminating the speech waveform with artifacts that are different every time. We can backpropagate the gradient from the output to the input samples, offering the possibility to draw and analyze adversarial examples directly in the sample domain.

### Tensor Format
So far, we only read audio data. SpeechBrain can read any other kind of tensors stored in *pk*l format.
All the tensors within SpeechBrain are read and processed using the following convention:
```
tensor=(batch, channels[optional], time_steps).
```
**The batch is always the first element, while time_steps is always the last one. In the middle, you might have a variable number of channels**.

##### *Why we need tensor with the same format?*
It is crucial to have a shared format for all the classes that process data and all the processing functions must be designed considering it. In SpeechBrain we might have pipelines of modules and if each module is based on different tensor formats, exchanging data between processing units will be painful. Many formats are possible. For SpeechBrain we selected this one because it is the same used in torch.audio. 

The format is actually very **flexible** and allows users to read different types of data. As we have seen, for single-channel raw waveform signal, the tensor will be ```tensor=(batch, time_steps)```, while for multi-channel raw waveform it will be ```tensor=(batch, n_channel, time_steps)```. Beyond waveform, Speechbrain can read other kinds of data. For instance, if we read fbank features the tensor should be formatted in this way:
```
(batch, n_filters, time_step)
```
If we wanna read a Short-Time Fourier Transform (STFT), the tensor will be:
```
(batch, n_fft, 2, time_step)
```
where the "2" is because STFT is based on complex numbers with a real and imaginary part.
We can also read multi-channel SFT data, that will be formatted in this way:
```
(batch, n_channels,n_fft, 2, time_step)
```
**This format must be used for all the processing functions**. For instance, a neural network (or any other processing module) will be feed with this formalism and should produce outputs exactly with the same tensor formalism.

### Writing Example
So far, we only read some data. Let's try now to do a step ahead by also **writing each sentence** in another folder (with a different format). 
Let's open *cfg/minimal_examples/data_reading/read_write_data.cfg*.
As you can see, the field *processing_cfg* is now defined (```processing_cfg=cfg/minimal_examples/save_signals.cfg```).  The *processing_cfg* file contains a list of computations to perform for each sentence.  Saving a sentence in a different format like in this case can be seen a very simple form of processing. Let's thus open the processing file *cfg/minimal_examples/save_signals.cfg*.  

The only function that we define here is called "save" and it is implemented in "data_processing.save". The function takes the sample rate (in this case, sample_rate=16000) and save_format (in this case, save_format=flac). Other options that can be set can be seen in  self.expected_variables of the class save in data_processing.  

As reported in the root config file *cfg/minimal_examples/data_reading/read_write_data.cfg*, the *processing_cfg* file is called by the *loop* class. of *core.p*y.  

*So, which data are we processing with that?*

As you can see in the **call** method of *loop*, we call **execute_computation** giving in input the batch variable. As we have seen before, the batch variable is the list of data returned by the data loader for each batch.
```
[computations]
    id,wav,wav_len=get_input_var()
    save(wav,id,wav_len)
[/computations]
```
In the computation section of the *processing_cfg file*   *cfg/minimal_examples/save_signals.cfg*, the first line just read this list of data returned by the *data_loader*. This is done with the special function **get_input_var()** that is the only function that can appear in the computation sections without being defined functions. 

This function simply returns the input list and make available the data that will be processed by the following commands.  
When calling **loop()**, the *get_input_var()* function returns the batch list defined in the call method and given in input to  ```self.loop_computation(batch)```. By default this list is composed of the sentence_id, data read from the scp file and their corresponding lengths. The list then contains the *batch_id* and the *iteration_id* (the latter will be zero if n_loops=1). Finally, the possible inputs given when calling the *loop* class are appended.

The second line, simply calls the save function with the expected argument.

*Ok, but how can I know the expected variables to provide in input when calling a function?* 
One way is to read the class description that appears before all the processing classes. **This description should clearly report the input expected by this function along with the related outputs**. 
Another way is to take a look into the corresponding *call* function (in this case the *call* method of save in *data_processing.py*). **By default, the input to of all the processing functions is always a list**. In the save __call__ method we can see:
``` pyhon
# reading input arguments
data,data_id,data_len=input_lst
```
this means that we would expect in input a  list of 3 elements (data,data_id,data_len). In this particular case, data are the data to store, data_id is the sentence id (each sentence is stored separately with its data_id as filename), while data_len is used to store to actual tensor without the zeros added during zero padding.

Similarly, to figure out the expected output one can read the class description or take a look at what the __call__ method returns. In general, **the output is a list of elements**. The save function reported here does not return anything because we only have to save some data on disk.  

Let's run now the root config file by typing:

```
python speechbrain.py  cfg/minimal_examples/data_reading/read_write_data.cfg
```

the data will be *exp/read_write_mininal/save*. As you can see all the data are stored in the flac format as specified.  Waveforms writing is performed with soundfile and we support here all the audio formats of soundfile. Beyond that, we can save files in pkl format, plain text, kaldi-binary formats, as well as png for 2-d tensors (see get_supported_formats in the class save of data_processing.py). 

Try to change the *save_format variable* in the *processing_cfg file* to change the output format. As you might have noticed, the data are saved by default is *$output_foler/save*. If you want to save data in another folder, you can set *save_folder* in the *processing_cfg* file.

If we set *save_scp=True* in the save function of the processing_cfg file, we will automatically save in the save_folder an scp file containing the list of files written on disk:

```
ID=example5 duration=1.000000 data=(exp/read_write_mininal/save/example5.flac,flac)
ID=example2 duration=2.068000 data=(exp/read_write_mininal/save/example2.flac,flac)
ID=example4 duration=2.188250 data=(exp/read_write_mininal/save/example4.flac,flac)
ID=example3 duration=2.890125 data=(exp/read_write_mininal/save/example3.flac,flac)
ID=example1 duration=3.260813 data=(exp/read_write_mininal/save/example1.flac,flac)
```

The scp file can be read back by speechbrain (just replace scp= in the root config file). This feature might be useful when we want to process the signals once, save some kind of output (e.g, speech features) and then feed the saved feature to another config file. 

## Basic processing
So far we have only seen very basic examples, where we just read and write some data. Let's now progressively dive into more realistic cases where we perform some kind of processing of the input signals.

### Adding Noise
Let's start with */cfg/minimal_examples/basic_processing/minimal_processing_read_write_example_noise.cfg*. In this case, we read data, we add some random noise, and we save the noisy data.

The root config file defines a loop functions that process the data with the processing_cfg *cfg/minimal_examples/save_signals_with_noise.cfg*.

The latter is very similar to the one described before. First of all we read the input data with the function *get_inp_var*,  then we apply some kind of processing, and finally, we save them.
In a general case, the processing function should be defined in functions and called in the computation section. For very basic processing functions like adding a random sequence of noise to the input data, we can use the special function pycmd:

```
 pycmd(noise_amp=0.005;noise=torch.randn_like(wav))
 pycmd(wav_noise=wav+noise_amp*noise)
```

**pycmd is a special function that can be used to directly run simple python commands in the computation section**. In this particular example, the first function define a noise amplitude noise_amp and generate a sequence of noise with the same shape of wav.
As you can see, the variable defined before this line are recognized and can be used within pycmd. The second line simply add the noise sequence to the original signals. 

The function *pycmd* can be extremely useful for debugging purposes as well. For instance, one can stop the execution of the code after printing the first sentence_id, the wav tensor (along with its shape) by adding the following commands:


```
pycmd(print(id);print(wav_noise);print(wav_noise.shape);sys.exit(0))
``` 


Finally, we save the noisy waveform with the specified format:
```
save(wav_noise,id,wav_len)
```

Note that the variables defined within *pycmd* can be recognized also outside. In this case wav_noise is defined withing pycmd, but can be used without problems within the save function.

Let's now run our simple example:
```
python speechbrain.py /cfg/minimal_examples/basic_processing/minimal_processing_read_write_example_noise.cfg
```

The noisy data will be saved in *exp/read_write_mininal_noise*. As you can hear, the speech data contains the noise added with the previous script.

Let's now open  */cfg/minimal_examples/minimal_processing_read_write_example_noise_parallel.cfg* to take a look about another interesting options supported in SpeechBrain. 

So far we only have seen computations that must be executed in cascade. We can anyway optimize the performance by doing some **computations in parallel** (when possible).  The root config file 
*/cfg/minimal_examples/minimal_processing_read_write_example_noise_parallel.cfg* calls a processing_cfg  *cfg/minimal_examples/basic_processing/save_signals_with_noise_parallel.cfgt* hat performs some computations in parallel:

```
    id,wav,wav_len=get_input_var()

    \parallel{
        pycmd(noise_amp=0.005;noise=torch.randn_like(wav))
    pycmd(val,ind=torch.max(torch.abs(wav),dim=-1); wav_norm=wav/val.unsqueeze(1))
        }

    pycmd(wav_noise=wav_norm+noise_amp*noise)

    save(wav_noise,id,wav_len)
```

**The ```\parallel{}``` statement can be used to run parallel computations** (each line is executed separately on a different process). In this case, the commands within the ```\parallel{}``` statement are independent can be run in parallel. The first line estimates the noise,  while the second normalizes the amplitude of the signal. When all these processes are finished, we can add the noise to the normalize sequence and save it as we have seen before.

#### Note of caution:
Use the ```\parallel{}``` statement at your own risk. Before using it you have to make sure that the computations are independent. The current version **does not support parallel computations on the same gpu**. According to our experience, parallelizing over the same gpu is critical and often the time took by parallel processes is larger than two processes in sequence.

### Other augmentations
In addition to adding noise, the [```data_augmentations.py```](data_augmentations.py) file defines a set of augmentations for increasing the robustness of machine learning models, and for creating datasets for speech enhancement and other environment-related tasks. The current list of enhancements follows, with links to sample files of each:

 * Adding noise - [white noise example](cfg/minimal_examples/basic_processing/save_signals_with_noise.cfg) or [noise from scp file example](cfg/minimal_examples/basic_processing/save_signals_with_noise_scp.cfg)
 * Adding reverberation [reverb example](cfg/minimal_examples/basic_processing/save_signals_with_reverb.cfg)
 * Adding babble [babble example](cfg/minimal_examples/basic_processing/save_signals_with_babble.cfg)
 * Speed perturbation [perturbation example](cfg/minimal_examples/basic_processing/save_signals_with_speed_perturb.cfg)
 * Dropping a frequency [frequency drop example](cfg/minimal_examples/basic_processing/save_signals_with_drop_freq.cfg)
 * Dropping chunks [chunk drop example](cfg/minimal_examples/basic_processing/save_signals_with_drop_chunk.cfg)
 * Clipping [clipping example](cfg/minimal_examples/basic_processing/save_signals_with_clipping.cfg)

These augmentations are designed to be efficient, so that you can use them on data during training without worrying about saving the augmented data to disk. In addition, all augmentations should be differentiable, since they are implemented as ```nn.Module```s. Finally, all augmentations have a ```random_seed``` parameter, to ensure that the augmentations are repeatable, and your results are comparable from experiment to experiment.

Aside from adding noise, all augmentations work on a batch level for the sake of efficiency. This means that for smaller datasets and larger batch sizes, the diversity of augmentations applied may be limited. However, the fact that these augmentations can be different for different epochs can make up for this fact.

## Feature extraction
Let's now move on by discussing some more realistic examples of speech feature extraction. 

#### Computing the Short-Time Fourier transform
We start from the most basic speech features i.e. the *Short-Time Fourier transform (STFT)*. The STFT computes the FFT transformation using sliding windows with a certain length (win_length) and a certain hop size (hop_length). 

Let's open *cfg/minimal_examples/features$ more compute_stft_example.cfg*. Also in this case, this root config file just call a *loop function* that processes the data with the processing_cfg *cfg/minimal_examples/features/STFT.cfg*. The latter processing STFT define a function called compute STFT:
```
    [compute_STFT]
        class_name=data_processing.STFT
        device=$device
        sample_rate=$sample_rate
        win_length=25
        hop_length=10
        n_fft=400
        window_type=hamming
        normalized_stft=False
        center=True
        pad_mode=reflect
        onesided=True
        amin= 1e-10
        ref_value=1.0
        top_db=80     
    [/compute_STFT]
Please, see the description of the class STFT in data_processing.py from more details on the parameters.
The computations section is the following:
[computations]
    id,wav,wav_len=get_input_var()
    STFT=compute_STFT(wav)
    save(STFT,id,wav_len)
[/computations]
```
After reading the input data, we run the *STFT* function that takes in input only the speech waveform as one can see from the *STFT* description or from the relative forward method ():

```python
# reading input _list
x=input_lst[0]
```
Note that here **feature extraction is implemented with a neural network class** (nn.Module) rather than a standard class. In SpeechBrain the feature extraction is seen as just a neural network that processes the input signals. This way we can backpropagate the gradient from the output of the network to the input samples and we can easily implement solutions like tunable filter-banks instead of the classical frozen features.

Once computed the *STFT*, we save it in pkl format with the function save. 

Let's now run the STFT feature extraction with the following command:
```
python speechbrain.py cfg/minimal_examples/features/compute_stft_example.cfg.
```

Once run, you can find the STFT tensors in *exp/compute_stft/save*.

You can open and inspect them from a python shell this way:

```python
>>> from data_io import read_pkl
>>> stft_tensor=read_pkl('exp/compute_stft/save/example1.pkl')
>>> stft_tensor
tensor([[[ 3.8108e-03,  5.7270e-04,  3.9190e-03,  ...,  4.1250e-04,
          -1.6833e-03, -1.2516e-03],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
           0.0000e+00,  0.0000e+00]],

        [[-4.9213e-03,  1.9049e-03,  3.3945e-03,  ...,  3.7572e-03,
           2.6394e-05, -5.7473e-03],
         [ 4.7684e-10,  8.7459e-04, -1.3462e-03,  ..., -1.2747e-03,
          -7.4749e-03,  2.2892e-03]],

        [[ 8.6700e-03,  2.0907e-03,  4.8991e-03,  ..., -1.2160e-02,
           1.9242e-04,  8.6598e-03],
         [ 5.0885e-11, -1.0420e-03,  2.4084e-04,  ...,  6.0171e-03,
           1.3465e-02, -2.5232e-03]],

        ...,

        [[ 2.3675e-03, -1.8950e-04,  1.5863e-03,  ..., -3.3456e-04,
           1.7539e-03, -1.7516e-03],
         [ 2.8372e-10,  2.3330e-03,  3.6345e-04,  ...,  9.0855e-04,
           2.0035e-03, -2.6378e-04]],

        [[-2.3171e-04,  1.5940e-03, -2.0379e-03,  ...,  2.0709e-03,
          -2.2090e-03,  2.7626e-03],
         [-4.5449e-10, -2.2351e-03,  1.4587e-04,  ...,  2.9044e-04,
          -1.2591e-03,  7.4596e-04]],

        [[-2.7688e-03, -1.6443e-03,  2.3796e-03,  ..., -1.4062e-03,
           1.3985e-03, -1.1332e-03],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
          -0.0000e+00, -0.0000e+00]]])
>>> stft_tensor.shape
torch.Size([201, 2, 327])
```

As you can see we have 201 points (i.e, n_fft/2+1), a real plus and imaginary part (2) and 327 time steps (i.e, sig_len/hop_length).

Let's take this opportunity to better describe how the data_processing classes like the STFT one should be implemented within SpeechBrain. All the data processing classes in speechbrain should be structured similarly.  Generally speaking, the data processing classes will have two mandatory methods:
1.  an **init** method  that sets the parameters and performs the **computations that should be done only once**.
2.  a  **call** or **forward** method (the latter for neural computations) that takes in input a list and performs some kind of **processing**. 

More precisely, the *init* method takes in input the configuration file and does the following:
1. **Defines the expected options** within the variable self.expected_options (see the variable self.expected_options into the STFT class defined in data_processing.py.)
2. **Checks if the options contained in the config dictionary matches the expected types**. This operation is performed by the check_opts function (see line    self.conf=check_opts(self,self.expected_options,config,logger=self.logger)).
3. **Defines the expected input types** when calling the function using the variable  self.expected_inputs. 
4. **Checks if the first input matches with the expected type** using the function check_inputs. Since we initialize the class only the first time we call the function, we have the chance to analyze the first input that is passed with the variable first_inp:      
    ``` python
        self.expected_inputs=['torch.Tensor']
        # check the first input     
        check_inputs(self.conf,self.expected_inputs,first_input,logger=self.logger)
    ```
    The last line just makes sure that the input is of the type expected. In this case, the *STFT* expects a single input of type tensor (that represents the batch containing the waveform tensors). 
    
    If we try to feed the *STFT* function with another type (e.g, a string, an integer, or something else) an error will be raised. Try for instance  to replace ```STFT=compute_STFT(wav)``` with ```STFT=compute_STFT(id)``` in the processing_cfg file   *cfg/minimal_examples/features/STFT.cfg* you will see the following error:

    ``` 
    ERROR: the input 0 of the function data_processing.STFT must be a torch.Tensor (got <class 'list'>)
    ``` 
    Then we do a shape check on the first input to make sure the shape is the one expected:
    ``` python        
            if first_input!=None:
                # shape check
                if len(first_input[0].shape)>3 or len(first_input[0].shape)<1:
                    err_msg='The input of STFT must be a tensor with one of the  following dimensions: [time] or [batch,time] or [batch,channels,time]. Got %s '%(str(first_input[0].shape))
                    logger_write(err_msg,logfile=logger)
    ``` 
    These lines just make sure that the shape of tensor is the one expected and raises an error otherwise.
    
    In this case, we can compute the STFT only if the input is formatted in this way:
    ``` 
    (time_steps) => 1-D tensor
    (batch,time_steps) => 2-D tensor
    (batch, channel, steps) => 3-D tensor
     ``` 

    The aforementioned check make sure that the we are in one of this case.

5. **Definition of other variables**. For instance in the STFT we define
    ``` python 
    # convert win_length and hop_length from ms to samples
            self.win_length=int(round((self.sample_rate/1000.0)*self.win_length))
            self.hop_length=int(round((self.sample_rate/1000.0)*self.hop_length))
    ``` 

    to covert *win_length* and *hop_length* from ms to samples.

6. **Perform the computations that need to be done only once**. In the STFT case we compute the window function just at the beginning:
    ``` python 
        # window creation
        self.window=self.create_window()
    ``` 
    Note tha it doesn't make sense to recompute the window for every input batch.

The __call__ or forward method should to the following operations:

1. **Read the input list**. In the STFT forward method we simply have:
    ``` python 
        # reading input _list
        x=input_lst[0]
    ```
Note that here we expect a single tensor in input (i.e. we have to select the first element of the input_lst).

2. **Process the data**. For instance in the STFT case, we have:
    ``` python 
        # adding signal to gpu or cpu
        x=x.to(self.device)
                
        # managing multi-channel stft:
        or_shape=x.shape
            
        # reshape tensor (batch*channel,time)
        if len(or_shape)==3:
            x=x.view(or_shape[0]*or_shape[1],or_shape[2])

        stft=torch.stft(
        x,
        self.n_fft,
        self.hop_length,
        self.win_length,
        self.window,
        self.center,
        self.pad_mode,
        self.normalized_stft,
        self.onesided,
        )
        
        # retrieving the original dimensionality (batch,channel,time)
        if len(or_shape)==3:
            stft=stft.reshape(or_shape[0],or_shape[1],stft.shape[1],stft.shape[2],stft.shape[3])
            
        # batch first dim, time last
        stft=stft.transpose(-2,-1)
    ```
        

3. **Return a list containing the outputs**. In the SFTT function, we return [stft] because the STFT vector is the only one that should be returned. 

Now, let's add a print into the forward method of STFT to take a look into the input and output dimensionality. In particular, let's add ```print(x.shape)``` right after reading the input  list and ```print(stft.shape)``` right before the return. You should see the following shapes:
```
batch 1
x torch.Size([2, 33088])
stft torch.Size([2, 201, 2, 207])

batch 2
x torch.Size([2, 46242])
stft torch.Size([2, 201, 2, 290])

batch 3
x torch.Size([1, 52173])
stft torch.Size([1, 201, 2, 327])
```

As you can see the input is formated with **[batch_size, n_samples]**, while the stft has **[batch,n_fft,2, time_steps]**.

#### Computing spectrograms
Let's now add another function of the top of STFT to compute the spectrogram. The spectrogram can is simply the module of the complex stft function (it is thus a real number).

To compute the spectrogram let's just run the following config files:
```
python speechbrain.py cfg/minimal_examples/features/compute_spectrogram_example.cfg
```
This root config file processes the data with  the computations specified in  *cfg/minimal_examples/features/spectrogram.cfg*. The computations section execute the following functions:

```
[computations]
    id,wav,wav_len=get_input_var()
    STFT=compute_STFT(wav)
    spectr=compute_spectrogram(STFT)
    save(spectr,id,wav_len)
[/computations]
```
As you can see, the function *compute_spectrogram* (implemented in *data_processing.spectrogram*)  takes in input an STFT tensor and returns a spectrogram by taking its module. Feel free to add prints within the call method of the spectrogram class to check the tensor dimensionalities. This time we save the spectr tensor in pkl format. To visualize the spectrograms you can type:

```
python tools/visualize_pkl.py \
              exp/compute_spectrogram/save/example1.pkl \
              exp/compute_spectrogram/save/example2.pkl \
```

The spectrogram is one of the most popular feature that can feed a neural speech processing system. The spectrogram, however, is a very high-dimensional representation of an audio signal and many times the frequency resolution is reduced by applying mel-filters. 

#### Computing Filter Banks
Mel filters average the frequency axis of the spectrogram with a set of filters (usually with a triangular shape)  that cover the full band. The filters, however, are not equally distributed, but we allocated more "narrow-band" filters in the lower part of the spectrum and fewer "large-band" filters for higher frequencies.  This processing is inspired by our auditory system, which is much more sensitive to low frequencies rather than high ones. Let's compute mel-filters by running:

```
python speechbrain.py  cfg/minimal_examples/features/compute_fbanks_example.cfg
```

The root_config file calls *cfg/minimal_examples/features/FBANKS.cfg*. The latter is very similar to the one discussed before for the spectrogram computation, where a function compute_fbanks is added to compute the filterbanks. 
**This function takes in input the spectrogram and averages in with the set of mel filters**. See the FBANK class description for more details. One important parameter is ```freeze=True```. In this case, freeze is set to true and the filters will remain always the same every time we call the function. 
If we set ```freeze=False```, **the central frequency and the band of each filter become learnable parameters** and can be changed by an optimizer.  In practice, if "freeze=False", this function can be seen as **a layer of a neural network where we can learn two parameters for each filter: the central frequency and the band**.

#### Computing Mel-Frequency Cepstral Coefficients
Beyond FBANKs, other very popular features are the Mel-Frequency Cepstral Coefficients (MFCCs). **MFCCs are build on the top of the FBANK feature by applying a Discrete Cosine Transformation (DCT)**. 
DCT is just a linear transformation that fosters the coefficients to be less correlated. These features were extremely useful before neural networks (e.g, in the case of Gausian Mixture Models). 
Neural networks, however, work very well also when the input features are highly correlated and for this reason, in standard speech processing pipeline FBANKs and MFCCs often provide similar performance. To compute MFCCs, you can run:

```
python speechbrain.py  cfg/minimal_examples/features/compute_mfccs_example.cfg
```

As you can see from the processing_cfg file *cfg/minimal_examples/features/compute_mfccs_example.cfg*,  the computation pipeline is now the following:

```
[computations]
    id,wav,wav_len=get_input_var()
    
    # mfcc computation pipeline
    STFT=compute_STFT(wav)
    spectr=compute_spectrogram(STFT)
    FBANKs=compute_fbanks(spectr)
    MFCCs=compute_mfccs(FBANKs)

    # computing derivatives
    delta1=compute_deltas(MFCCs)
    delta2=compute_deltas(delta1)

    # concatenate mfcc+delta1+delta2
    pycmd(mfcc_with_deltas=torch.cat([MFCCs,delta1,delta2],dim=-2))

    # applying the context window
    mfcc_cw=context_window(mfcc_with_deltas)

    save(mfcc_cw,id,wav_len)

[/computations]
```

The compute_mfccs function takes in input the FBANKs and gives in output the MFCCs after applying the DCT transform and selecting n_mfcc coefficients.  

A standard practice with MFCCs is to compute the derivatives over the time axis to embed a bit of local context. This is done with **compute_deltas function** (implemented data_processing.deltas). As commonly done, we  here compute the first and the second derivative  (```delta1=compute_deltas(MFCCs)``` and ```delta2=compute_deltas(delta1)```)
and the then concatenate them with the static coefficients:
``` python
  # concatenate mfcc+delta1+delta2
   pycmd(mfcc_with_deltas=torch.cat([MFCCs,delta1,delta2],dim=-2))
```

When processing the speech features with feedforward networks, another standard **approach to embed a larger context consists of gathering more local frames using a context window**.  This operation is performed with the function **context_window** (implemented in data_processing.context_window). Please, add some prints in mfcc, delta, and context window forward methods to check the dimensionalities of the tensors. Note that delta and context window can be used for any kind of feature (e.g, FBANKs) and not only for MFCCs.

In many cases, the feature pipeline is quite standard and shouldn't be changed every time. It could thus make sense to save the feature pipeline in a file (*e.g. cfg/features/mfcc_features.cfg*) and, when needed, change the parameters form the root config file rather than changing the rather "stable" *cfg/features/mfcc_features.cfg file*. To do it, we can now introduce the *"cfg_change*" option of loop that we didn't discuss before.

Let's assume for instance that in the particular experiment that we are going to run we want ```n_fft=1024```, ```n_mels=40 ```and ```n_mfcc=13```. To do it without we can set ```cfg_change=--global,nfft=1024 --global,n_mels=40 --global,n_mfcc=1```3 as done in  cfg/minimal_examples/features/compute_mfccs_example2.cfg. 

The cfg_change variable uses the same syntax previously described to change root config files from the command line.  **The advantage is that I only have to modify the root config file, without necessarily changing the more "stable" feature computation file**.

The  *cfg/minimal_examples/features/compute_mfccs_example2.cfg* config file actually shows another important parameter of the loop class. The field stop_at is in fact set to ```stop_at=MFCC```s.  The stop_at stops the execution of the processing_cfg file when this variable is encountered. In this case the computations are stopped when the line MFCCs=compute_mfccs(FBANKs)  in *cfg/features/mfcc_features.cfg*. is executed and the variable MFCCs is returned (skipping all delta and context window computations). To convince yourself about it, add a print (e.g., ```print(self.class_name)```) within the forward methods of STFT, spectr,  FBANKs, MFCCs,deltas, and context _windows. 

If the stop_at is not set, we execute by default all the processing computations and you will thus see something like this:
```
data_processing.STFT
data_processing.spectr
data_processing.FBANKs
data_processing.MFCCs
data_processing.delta
data_processing.delta
data_processing.context_window
```
While if you set stop_at=MFCCs:
```
data_processing.STFT
data_processing.spectr
data_processing.FBANKs
data_processing.MFCCs
```
or stop_at=STFT:
```
data_processing.STFT
```
Note that if we set *stop_at* (e.g, ```stop_at=MFCCcs```) with a variable name not defined in *cfg/features/mfcc_features.cfg*, an error is raised:

```
ERROR: the output variables (stop_at=['MFCCsv']) defined in execute_computations are not defined in the section [computation] of the config file!
```

This feature can be extremely useful when only some of the computations reported in the processing_cfg file should be executed.


Let's conclude the feature example part with a little bit more complex example. As you can see from the configuration files, the number of steps involved in the feature computation could be rather high, but rather "stable" too. 

Imagine now that we feed this feature into a neural network that also performs several layers of computations of the top of it. To avoid producing computation sections that are too long one can use the approach implemented in *cfg/minimal_examples/features/compute_mfccs_example3.cfg*. In this case, the root config file process the data with cfg/minimal_examples/features/compute_and_process_features.cfg. 

This processing function defines a function *compute_features* that execute (using core.execute_computations) the computations reported in *cfg/features/features.cfg*. The stop_at is set to ``` stop_at=MFCCs,FBANK```s and thus when compute_features is called, the computation within *cfg/features/features.cfg* will return both MFCCs and FBANKs. 

The computation section of *cfg/minimal_examples/features/compute_and_process_features.cfg*  can now be the following:

```
[computations]
    id,wav,wav_len=get_input_var()
    
    # mfcc computation pipeline
    mfcc_fea,fbank_fea=compute_features(id,wav,wav_len)

    # do other computations here
    pycmd(features=torch.cat([mfcc_fea,fbank_fea],dim=-2))

    #save(features,id,wav_len)
[/computations]
```
where the multiple feature computations are just replaced with the line  ```mfcc_fea,fbank_fea=compute_features(id,wav,wav_len)```

After that, any kind of processing (e.g, neural networks) can be performed.

### Features for multi-channel audio
The same processing pipeline can be employed for multi-channel data.
In cfg/minimal_examples/multichannel* you can find some example of feature computations for multi-channel audio. Note that the config files (except for the scp field that now points to multi-channel audios) are exactly the same.  One difference lies in the dimensionality of the data tensor that will now be:
```
wav=(batch, channel, time_steps)
```
rather then:
```
wav=(batch, time_steps)
```

**All the processing functions e.g., STFT, spectrogram, FBANKs, MFCCs are designed to support multi-channel audio**. As an exercise, feel free to add some prints of the tensor shapes in the __call__ functions.


## Data Preparation
So far, we  only used already-prepared scp files.  
**The data_preparation is the process of creating the scp file starting from a speech/audio dataset**.  
Since every dataset is formatted in a different way, typically a different  data_preparation script must be designed. Even though we will provide data_preparation scripts for several popular datasets (see data_preparation.py), in general, this part should be done by the users.
In *cfg/minimal_examples/data_preparation/LibriSpeech* you can find a couple of examples where we perform data_preparation before looping over the data. 

The config file *cfg/minimal_examples/data_preparation/minimal_processing_librispeech_read.cfg*  simply loops over the data without doning nothing else, while *minimal_processing_librispeech_mfcc.cfg* computes the MFCCs feature of the training data. 
Let's open one of the two config file. As you can see we here define two new functions i.e, **copy_locally** and **prepare_librispeech**. The first one copy the dataset from the original data_folder  to a local folder, the second one generate the scp file for the data chunks reported in split. 

### Copy your data locally!
**SpeechBrain is designed to form batches of data by ready these data on-the-fly from disk**. 
This is true especially when the dataset is very big and the option cache=True can only store in RAM a small fraction of the data. Reading data from disk can be critical if the operation is not performed on the local disk of the machine used to perform the computation. For instance, in a standard HPC cluster reading several small files from the shared filesystem can be extremely slow and can even slow down the entire cluster (e.g., think about a lustre file system that is designed to read and write large files only, but it is very inefficient to read/write several small files). 

The solution is to always read data from the local disk (e.g, in SLURM the local disk is in *$SLURM_TMPDIR*). To do it, we suggest to do the following steps:

1. *If not already compressed, compress your dataset.*
2. *Copy the file in the local disk.*
3. *Uncompress the dataset.*
4. *Process it with SpeechBrain.*

To help users with this operation, we created a function called copy_locally (see data_preparation.copy_data_locally), which automatically implements steps 2 and 3. In the particular case we initialize this class with the following parameters:

```
    [copy_locally]
            class_name=data_preparation.copy_data_locally
            data_file=$data_file
          local_folder=$local_data_folder
    [/copy_locally]
```

where data_file must be a single file (e.g. a *tar.gz* file that contains your dataset) and local_folder is the folder on the local computation node where you want to copy and uncompress the data. 
Note that this function makes sense only if the computations are done in a HPC cluster or if the dataset is stored in a shared filesystem different from the local one. When we call this function in the computation section we just copy the data_file in the local folder and uncompress it there.  The function is automatically skipped when the tar.gz file are already present in the local folder. 

### LibriSpeech data preparation
The following data preparation function (i.e, prepare_librispeech) analyze the dataset and creates the related scp file:
```
    [prepare_librispeech]
            class_name=data_preparation.librispeech_prepare
           data_folder=$local_data_folder
          splits=train-clean-100,dev-clean,test-clean
    [/prepare_librispeech]
```

In data_folder you should add the main librispeech folder (in this case the one stored locally), while splits is a list of librispeech splits for which we want to create in scp files. By default (if not specified differently with the save_folder filed of  data_preparation.librispeech_prepare), the scp files are save in *$output_folder/prepare_librispeech/*.scp*

For instance, try to open *train-clean-100.sc*p that is composed of  *28539* sentences.

Note that the data preparation step can be quite computational demanding for large datasets (we have to open all the audio files, check them, compute their duration, etc). The data_preparation is thus automatically skipped in the needed scp that has been already created.

Once finished the data preparation we can loop over the data and process them as we have seen in the previous examples.

## Developer Guidelines
The goal is to write a set of libraries that process audio and speech in several different ways. Developers should be read the tutorial before to familiarize themselves with the general architecture of the toolkit. 
### General Guidelines
SpeechBrain could be used for *research*, *academic*, *commercial*,*non-commercial* purposes. Ideally, the code should have the following features:
- **Simple:**  the code must be easy to understand even by students or by users that are not professional programmers or speech researchers. Try to design your code such that it can be easily read. Given alternatives with the same level of performance, code the simplest one. (the most explicit and straightforward manner is preferred)
- **Readable:** SpeechBrain adopts the code style conventions in PEP8. The code written by the users must be compliant with that. Please use  *pycodestyle* to check the compatibility with PEP8 guidelines.
We also suggest to use *pylint* for further checks (e.g, to find typos in comments or other suggestions).

- **Efficient**: The code should be as efficient as possible. When possible, users should maximize the use of pytorch native operations.  Remember that in generally very convenient to process in parallel multiple signals rather than processing them one by one (e.g try to use *batch_size > 1* when possible). Test the code carefully with your favorite profiler (e.g, torch.utils.bottleneck https://pytorch.org/docs/stable/bottleneck.html ) to make sure there are no bottlenecks if your code.  Since we are not working in c++ directly, performance can be an issue. Despite that, our goal is to make SpeechBrain as fast as possible. 
- **modular:** Write your code such that is is very modular and fits well with the other functionalities of the toolkit.
 
**It is or crucial important to properly comment each of the functions and classes of SpeechBrain**. Take a look into one of the classes/functions in *core.py*, *data_processing.py*, *data_preparation.py*, or *utils.py*. For each function there must be a header formatted in the following way:

```
library.class/function name (authors:Author1, Author2,..)

Description: function description
 
Input: input description
 
Output: output description

Example: an example
```

For each function/class,  **a description that summarizsd the functionalities implemented must be written**. Input and output should also be described with their corresponding types. Last but not least, **an example of the use of this function/class must be provided**. Examples are often useful to clarify how to use a certain function and might help to learn how to use the SpeechBrain libraries also as stand-alone functions. 

Moreover, each meaningful line must contain a corresponding comment. Please, take a look at some classes/functions in the aforementioned libraries.

### Folder Structure
The current version of the project is organized in the following way:
- **speechbrain.py:** *it is the main file of SpeechBrain and it is used to run the computations described in a config file*.
- **core.py:** *it contains core functionalities such as the execute_computations or the loop class.* 
- **neural_networks.py:** *it contains all the classes that implement neural architectures.*
- **data_processing.py:** *it contains all the classes for data processing.*
- **data_preparation.py:** *it contains all the classes for data_preparation.*
- **data_io.py:** *it contains a set of functions useful for reading/writing operations.*
- **utils.py:** *it contains a set of support functions.*
- **test.py:** *it is used to run basic tests to make sure the most important functionalities are working.*
- **cfg:** *it is a folder containing the configuration files.*
- **tools:** *it is a folder where adding additional tools.*
- **samples:** *it contains some samples (useful for debugging).*
- **ReadMe.md:** *it is the file that contains the current documentation.*

The developers must write their codes within one of the aforementioned libraries.  If you think that your code doesn't fit well in any of the current libraries,  please discuss it with your group supervisor or with one of the project leaders. The idea is to organize the code in a limited number of well-organized libraries to avoid confusion. 

### How to write a processing class
The processing classes can be found in *data_processing.py*, *core.py*, *data_preparaionn.py*, or *neural_networks.py*. 
All the classes reported in these libraries can be called within the configuration file and are executed by the execute_computations class. **These classes must share the input arguments and general architecture**.

The processing classes must have an initialization method followed by a __call__ or **forward** method.

#### Initialization method
The **initialization** method initializes all the parameters and performs all the computations that need to be done only once. This method takes in input the following arguments:

- **config** (dict, mandatory): it is the part of the config file containing the hyperparameters needed by the current class. For instance, if the config file contains a section like this:
    ```
            [loop]
            class_name=core.loop
            scp=samples/audio_samples/scp_example.scp
    	    drop_last=True
            [/loop]
    ```
    The function *core.loop* will receive the following dictionary:
    ```
    config={'class_name':'core.loop', 'scp':'samples/audio_samples/scp_example.scp','drop_last':'True'}
    ```
    Note that all the values of the dictionary are string at this point of the code (they will cast later in the code).

- **funct_name** (str, optional, default:None): It is the name of the current function. For instance in the previous example it will be  "loop". It can be used to raise more precise errors (e.g, ERROR in function "loop").

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

- **first_input** : (list, optional, default: None): This is a list containing the first input provided when calling the class for the first time. In SpeechBrain **we initialize the class only the first time we call it** (see execute_computation class) and we thus have the chance to analyze the input and make sure it is the expected one (e.g, analyzing number, type, and shapes of the input list).
All these arguments are automatically passed to the class after calling execute_computations. As an exercise try to open one of the processing classes (e.g. core.data_io) and print the arguments (prints will be shown after running the script using python speechbrain.py cfg_file).

The initialization method then goes ahead with the definition of the expected inputs. Let's discuss this part more in detail in the next subsection:

##### Expected options
The variable *self.expected_options* should be defined for all the classes that SpeechBrain can execute. As you can see, in this case it is defined in this way:
``` 
        self.expected_options={
        'scp': ('file','mandatory'),
        'processing_cfg': ('file','optional','None'),
        'cfg_change': ('str','optional','None'),
        'stop_at': ('str_list','optional','None'),
        'batch_size': ('int(1,inf)','optional','1'),
        'scp_read': ('str_list','optional','None'),
        'sentence_sorting':('one_of(ascending,descending,random,original)','optional','original'),
        'num_workers': ('int(0,inf)','optional','0'),
        'cache': ('bool','optional','False'),
        'cache_ram_percent': ('int(0,100)','optional','75'),
        'select_n_sentences': ('int(1,inf)','optional','None'),
        'drop_last': ('bool','optional','False'),
        }
``` 
The *self.expected_variables* is a simple python dictionary that collects all the options expected by this class. Some options are **mandatory** (e.g, *scp*) and an error will be raised if they are missing in the config file. Other options (e.g, *batch_size*, *sentence sorting* ) are **optional** and a **default value** is assigned to them when not specified in the config file.

The dictionary is organized in this way: the key of the dictionary represents the options to check. For each key, a tuple of 2 or 3 elements is expected for mandatory and optional parameters, respectively:

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


The function **check_opts checks if the options set in the config file by the users correspond to the type expected**. If not, an error will be raised. 
Moreover, the function assigns the specified default value for the optional parameters.  In the current example, for instance, the variable "batch_size" is not defined and it will be automatically set to 1.
The  check_opts not only performs a check over the parameters specified by the users, but also **creates and casts the needed parameters**.  For instance, after calling *check_opts* the variable *self.scp* or  *self.sentence_sorting* will be created and can be used in the rest of the code. 

In the current example, we will have the following variables:
``` 
        self.scp='samples/audio_samples/scp_example.scp'
        self.processing_cfg=None
        self.cfg_change=None
        self. stop_at=None
        self.batch_size=1
        scp_read=None'
        self.sentence_sorting='original' 
        self.num_workers=0
        self.cache=False
        self.cache_ram_percent=75
        select_n_sentences=None
        drop_last=False
``` 

Feel free to add some prints after *check_opts* (e.g, *print(self.batch_size)*) to verify it.

As outlined before, we can also have the chance to analyze the first input given when calling the class for the first time.
If  the function expects no inputs (i.e, the input_list is empty) the code is simply the following:
```python 
        # Expected inputs when calling the class (no inputs in this case)
        self.expected_inputs = []

        # Checking the first input
        check_inputs(
            self.conf, self.expected_inputs, first_input, logger=self.logger
        )
```

While if the input is a *torch.Tensor* like in data_processing.STFT,  the code should be the following:
```python 
        # Definition of the expected input
        self.expected_inputs = ["torch.Tensor"]

        # Check the first input
        check_inputs(
            self.conf, self.expected_inputs, first_input, logger=self.logger
        )
```
The check_inputs function from utils make sure that the first input is of the expected type.

The initialization then goes ahead with variable initialization and computations that are class-specific and varies depending on the specific functionality implemented within the class.

#### Call Method
The __call__ (or forward method), instead, can be called only after initializing the class. By default, it receives in input a list and returns another list. The input received here is the one reported in the [computation] of the config file. 
For instance, if the computation section is the following:

```
[computations]

    # loop over data batches
    loop()    
    
[/computations] 
```

the input_list given when calling the loop function will be an empty list:

```
inp_lst=[]
```

If the computation section  is the following (cfg/minimal_examples/features/STFT.cfg):
```
[computations]
    id,wav,wav_len=get_input_var()
    STFT=compute_STFT(wav)
    save(STFT,id,wav_len)
[/computations]
```
We will receive ```input_lst=[wav]``` when calling the *data_processing.STFT* function.

Add some prints in the various __call__ function to make sure about that. The __call__ (or forward) methods then processing the input and produces an output. 

The output should be organized in a list. For instance, in the previous example, the output will be ```out_lst=[STFT]```. If no output is returned by that function, ```None``` can be returned.

When writing a new class, the encourage users to take a look into the already existing processing classes and organize the code similarly (e.g, one way is to copy and paste one class and then modify only some parts of it).

### Example of Processing Class
To clarify how to write a data_processing class, let's write a very minimal processing function. For instance, let's write a function that takes in input an audio signal and normalizes its amplitude. For a class like this, we can think about the following parameters:
```
        [normalize]
        class_name=data_processing.normalize_amp
        normalize=True
        [/normalize]
```
If normalize is True we return the signal with a maximum amplitude of 1 or -1, if not we return the original input signal.
Let's now go in data_processing.py and create a new class called normalize_amp:
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
This function operates at the signal level and must be called within a loop. The root config file (that should be saved somewhere, e.g, in *cfg/minimal_examples/basic_processing/normalize_loop.cfg*) will be like this:

```
[global]
    verbosity=2
    output_folder=exp/read_mininal_example
    data_folder=samples/audio_samples
    scp_file=samples/audio_samples/scp_example.scp
[/global]

[functions]    
        
        [loop]
        class_name=core.loop
        scp=$scp_file
	processing_cfg=cfg/minimal_examples/basic_processing/normalize_sig.cfg
        [/loop]

[/functions]

[computations]

    # loop over data batches
    loop()    
    
[/computations]
```

while the processing_cfg (*cfg/minimal_examples/basic_processing/normalize_sig.cfg*) must be created like this:

```
[global]
    device=cuda
[/global]

[functions]

    [normalize]
        class_name=data_processing.normalize_amp
     	normalize=True
    [/normalize]

[/functions]


[computations]
    id,wav,wav_len=get_input_var()
    norm_sig=normalize(wav)
[/computations]
```

You can now run the experiment in the following way:
python speechbrain.py *cfg/minimal_examples/basic_processing/normalize_loop.cfg*

Please, take a look into the other data processing class to figure out how to implement more complex functionalities.   

### Pull Requests:

Before doing a pull request:
1. Make sure that the group leader or the project leader is aware and accepts the functionality that you want to implement.
2. Test your code accurately and make sure there are no performance bottlenecks.
3. Make sure the code is formatted as outlined  before (do not forget the header of the functions and the related comments)
4. Run ```python test.py``` to make sure your code doesn't harm some core functionalities. Note that the checks done by test.py are rather superficial and it is thus necessary to test your code in a very deep way before asking for a pull request.
5. Ask for the pull request and review it according to the group leader/project leader feedback. 

The list of the current list of team leaders and developers can be found here.


## Team leader guidelines
For each speech application that will be developed, a team leader is assigned. The team leaders, that can also be a developer, have the following responsibilities:
1. Deciding (together with the project leaders) the functionalities that should be implemented and the related priorities.
2. Monitoring the evolution of the specific functionality and report the status of the project periodically to the project leaders
3. Coordinating the work of the developers and assigned them tasks.
4. Reviewing the pull request done by the users, test them, and making sure they are compliant with all the developer guidelines
The list of the current list of team leaders and developers can be found here.



