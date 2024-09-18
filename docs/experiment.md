# Running an experiment
In SpeechBrain, you can run experiments in this way:

```
> cd recipes/<dataset>/<task>/
> python experiment.py params.yaml
```

The results will be saved in the `output_folder` specified in the yaml file.
The folder is created by calling `sb.core.create_experiment_directory()` in `experiment.py`. Both detailed logs and experiment outputs are saved there. Furthermore, less verbose logs are output to stdout.

## YAML basics

The YAML syntax offers an elegant way to specify the hyperparameters of a recipe.
In SpeechBrain, the YAML file is not a plain list of parameters, but for each parameter, we specify the function (or class) that is using it.
This not only makes the specification of the parameters more transparent but also allows us to properly initialize all the entries by simply calling the `load_extended_yaml` (in `speechbrain.utils.data_utils`).

### Security note
Loading HyperPyYAML allows arbitrary code execution.
This is a feature: HyperPyYAML allows you to construct *anything* and *everything*
you need in your experiment.
However, take care to verify any untrusted recipes' YAML files just as you would verify the Python code.

### Features

Let's now take a quick look at the extended YAML features, using an example:

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

For more details on YAML and our extensions, please see our dedicated [tutorial](https://speechbrain.readthedocs.io/en/latest/tutorials/basics/hyperpyyaml.html).

## Running arguments
SpeechBrain defines a set of running arguments that can be set from the command line args (or within the YAML file).
- `device`: set the device to be used for computation.
- `debug`: a flag that enables debug mode, only running a few iterations to verify that program won't crash.
- `data_parallel_backend`: a flag that enables `data_parallel` for multigpu training on a single machine.
- `data_parallel_count`: default "-1" (use all gpus), if > 0, use a subset of gpus available `[0, 1, ..., data_parallel_count]`.
- `distributed_backend`: default "nccl", options: `["nccl", "gloo", "mpi"]`, this backend will be used as a DDP communication protocol. See PyTorch documentation for more details.
- Additional runtime arguments are documented in the Brain class.

Please note that we provide [documentation](https://speechbrain.readthedocs.io/en/latest/multigpu.html) on multi-GPU training strategies.

You can also override parameters in YAML in this way:

```
> python experiment.py params.yaml --seed 1234 --data_folder /path/to/folder --num_layers 5
```

This call would override hyperparameters `seed` and `data_folder` and `num_layers`.

*Important*:
- The command line args will always override the hparams file args.

## Tensor format
All the tensors within SpeechBrain are formatted using the following convention:
```
tensor=(batch, time_steps, channels[optional])
```
**The batch is always the first element, and time_steps is always the second one. The remaining optional dimensions are channels. (there might be as many channels as you need)**.

*Why do we need all tensors to have the same format?*
It is crucial to have a shared format for all the classes and functions. This makes model combination easier.
Many formats are possible. For SpeechBrain we selected this one because it is commonly used in recurrent neural networks.

The adopted format is very flexible and allows users to read different types of data. For instance, with single-channel raw waveform signals, the tensor will be tensor=(batch, time_steps), while for multi-channel raw waveform it will be tensor=(batch, time_steps, n_channel). Beyond waveforms, this format is used for any tensor in the computation pipeline. For instance, fbank features that are formatted in this way:
```
(batch, time_step, n_filters)
```
The Short-Time Fourier Transform (STFT) tensor, instead, will be:
```
(batch, time_step, n_fft, 2)
```
where the “2” corresponds to the real and imaginary parts of the STFT.
We can also read multi-channel SFT data, that will be formatted in this way:
```
(batch, time_step, n_fft, 2, n_audio_channels)
```

## Reproducibility

To improve reproducibility across experiments, SpeechBrain supports its own seeding function located in `speechbrain.utils.seed.seed_everything`. This function sets the seed for various generators such as NumPy, PyTorch, and Python, following the [PyTorch recommendations](https://pytorch.org/docs/stable/notes/randomness.html).

However, due to the differences in how GPU and CPU executions work, results may not be fully reproducible even with identical seeds, especially when training models. This issue primarily affects training experiments.

On the other hand, when preparing data using data preparation scripts, the output of these scripts is independent of the global seeds. This ensures that you will get identical outputs on different setups, even if different seeds are used.

In distributed experiments, reproducibility becomes more complex as different seeds (offset by the rank) will be set on different machines or processes. This primarily impacts operations that rely on randomness, such as data augmentations. Since each process in a distributed setup is assigned its own seed, the randomness applied to data (e.g., augmentations) can differ between processes, even though the global seed is the same across machines.

It’s important to note that this variance in seeding does not affect certain elements of the experiment. For instance, initial model parameters are broadcast to all processes from the main process in distributed training. Similarly, components like data loaders, which shuffle data, will be affected by per-process seeds, but the underlying data pipeline remains synchronized across processes.
