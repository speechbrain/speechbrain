# Running an experiment

In SpeechBrain experiments can be run from anywhere, but the experimental `results/` directory will be created relative to the directory you are in. The most common pattern for running experiments is as follows:

```
> cd recipes/<dataset>/<task>/
> python experiment.py params.yaml
```

At the top of the `experiment.py` file, the function
`sb.core.create_experiment_directory()` is called to create an output directory
(by default: `<cwd>/results/`). Both detailed logs and experiment outputs are saved there. Furthermore, less detailed logs are output to stdout. The experiment script and configuration (including possible command-line overrides) are also copied to the output directory.

## YAML basics

Also have a look at the YAML files in recipe directories. The YAML files
specify the hyperparameters of the recipes. The syntax is explained in
`speechbrain.utils.data_utils` in the docstring of `load_extended_yaml`.

A quick look at the extended YAML features, using an example:
```
seed: !PLACEHOLDER
output_dir: !ref results/vgg_blstm/<seed>
save_dir: !ref <output_dir>/save
data_folder: !PLACEHOLDER # e.g. /path/to/TIMIT

model: !new:speechbrain.lobes.models.CRDNN.CRDNN
    output_size: 40 # 39 phonemes + 1 blank symbol
    cnn_blocks: 2
    dnn_blocks: 2
```
- `!new:speechbrain.lobes.models.CRDNN.CRDNN` creates a `CRDNN` instance
  from the module `speechbrain.lobes.models.CRDNN`
- The indented keywords (`output_size` etc.) after it are passed as keyword
  arguments.
- `!ref <output_dir>/save` evaluates the part in angle brackets,
  referencing the YAML itself.
- `!PLACEHOLDER` simply errors out when loaded; it should be replaced by
  every user either by editing the yaml, or with an override (passed to
  `load_extended_yaml`).

For more details on YAML and our extensions, please see our dedicated tutorial: [amazing YAML tutorial](#)

## Running arguments
We define a set of running arguments in SpeechBrain, these arguments can be set from the command line args or an hparams file.
- `device`: set the device to be used for computation.
- `data_parallel_backend`: default False, if True, use `data_parallel` for multigpu training on a single machine.
- `data_parallel_count`: default "-1" (use all gpus), if > 0, use a subset of gpus available [0, 1, ..., data_parallel_count].
- `distributed_launch`: default False, if True, we assume that we already use `torch.distributed.launch` for multiGPU training. the `local_rank` and `rank` UNIX arguments are then parsed for running multigpu training.
- `distributed_backend`: default "nccl", options: ["nccl", "gloo", "mpi"], this backend will be used as a DDP communication protocol. See Pytorch Doc for more details.
- Additional runtime arguments are documented in the Brain class.

Please note that we provide a dedicated tutorial to document the different
multi-gpu training strategies: [amazing multi-gpu tutorial](#)

You can also override parameters in YAML by passing the name of the parameter as an argument on the command line. For example:

```
> python experiment.py params.yaml --seed 1234 --data_folder /path/to/folder --num_layers 5
```

This call would override hyperparameters `seed` and `data_folder` and `num_layers`.

Important:
- The command line args will always override the hparams file args.

## Tensor format
All the tensors within SpeechBrain are formatted using the following convention:
```
tensor=(batch, time_steps, channels[optional])
```
**The batch is always the first element, and time_steps is always the second one.
The rest of the dimensions are as many channels as you need**.

*Why do we need all tensors to have the same format?*
It is crucial to have a shared format for all the classes that process data and all the processing functions must be designed considering it. In SpeechBrain we might have pipelines of modules and if each module was based on different tensor formats, exchanging data between processing units would have been painful. Many formats are possible. For SpeechBrain we selected this one because
it is commonly used with recurrent layers, which are common in speech applications.

The format is very **flexible** and allows users to read different types of data. As we have seen, for **single-channel** raw waveform signals, the tensor will be ```tensor=(batch, time_steps)```, while for **multi-channel** raw waveform it will be ```tensor=(batch, time_steps, n_channel)```. Beyond waveforms, this format is used for any tensor in the computation pipeline. For instance,  fbank features that are formatted in this way:
```
(batch, time_step, n_filters)
```
The Short-Time Fourier Transform (STFT) tensor, instead, will be:
```
(batch, time_step, n_fft, 2)
```
where the "2" is because STFT is based on complex numbers with a real and imaginary part.
We can also read multi-channel SFT data, that will be formatted in this way:
```
(batch, time_step, n_fft, 2, n_audio_channels)
```
