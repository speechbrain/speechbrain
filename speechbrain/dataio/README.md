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


## Data preparation
**The data_preparation is the process of creating the CSV file starting from a certain dataset**.
Since every dataset is formatted in a different way, typically a different data_preparation script must be designed. Even though we will provide data_preparation scripts for several popular datasets (see data_preparation.py), in general, this part should be **done by the users**. In *samples/audio_samples* you can find some examples of CSV files from which it is possible to easily infer the type of information required in input by speechbrain.
