# Running an experiment
In SpeechBrain, you can train most models in recipes like this:

```
> cd recipes/<dataset>/<task>/<model>
> python train.py hparams/hyperparams.yaml
```

Follow the steps in the README of each recipe for more details.

The results will be saved in the `output_folder` specified in the yaml file.
The folder is created by calling `sb.core.create_experiment_directory()` in `train.py`. Both detailed logs and experiment outputs are saved there. Furthermore, less verbose logs are output to stdout.

## YAML basics

SpeechBrain uses an extended variant of YAML named HyperPyYAML. It offers an elegant way to specify the hyperparameters of a recipe.

In SpeechBrain, the YAML file is not a plain list of parameters, but for each parameter, we specify the function (or class) that is using it.
This not only makes the specification of the parameters more transparent but also allows us to properly initialize all the entries by simply calling `load_hyperpyyaml` (from HyperPyYAML).

### Security warning

Loading HyperPyYAML allows **arbitrary code execution**.
This is a feature: HyperPyYAML allows you to construct *anything* and *everything*
you need in your experiment.
However, take care to verify any untrusted recipes' YAML files just as you would verify the Python code.

### Features

Let's now take a quick tour of the extended YAML features, using an example:

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
  every user either by using the commandline (which passes an override to
  `load_hyperpyyaml`), or by manually editing the `.yaml` if necessary.

[**Learn more with the dedicated HyperPyYAML tutorial!**](https://speechbrain.readthedocs.io/en/latest/tutorials/basics/hyperpyyaml.html)

## Running arguments

SpeechBrain defines a set of running arguments that can be set from the command line args (or within the YAML file), e.g.:

- `device`: set the device to be used for computation.
- `debug`: a flag that enables debug mode, only running a few iterations to verify that program won't crash.
- Additional runtime arguments are documented in the Brain class.

If you want to train using multiple GPUs, please follow the [**multi-GPU training guide**](https://speechbrain.readthedocs.io/en/latest/multigpu.html).

You can also override parameters from the YAML in this way:

```
> python experiment.py params.yaml --seed 1234 --data_folder /path/to/folder --num_layers 5
```

This call would override hyperparameters `seed` and `data_folder` and `num_layers`.

*Important*:
- The command line args will always override the hparams file args.

## Tensor format

Tensors in SpeechBrain follow a batch-time-channels convention:

- **The batch dimension is always the first dimension (even if it is `1`).**
- **The time step dimension is always the second one.**
- **The remaining optional dimensions are channels (however many dimensions you need)**.

In other words, a tensor will look like any of these:

```
(batch_size, time_steps)
(batch_size, time_steps, channel0)
(batch_size, time_steps, channel0, channel1, ...)
```

It is crucial to have a shared format for all the classes and functions. This makes model combination easier.
Many formats are possible. For SpeechBrain we selected this one because it is commonly used in recurrent neural networks.

For waveforms, we generally choose to squeeze the final dimension (i.e. it there is _no_ channel dimension for mono audio).

Simple waveform examples:

- A waveform of 3 seconds sampled at 16kHz in mono: `(1, 3*16000)`
- A waveform of 3 seconds sampled at 16kHz in stereo: `(1, 3*16000, 2)`

Beyond waveforms, this format is used for any tensor in the computation pipeline. For instance...

- The [Short-Time Fourier Transform (STFT)](https://speechbrain.readthedocs.io/en/develop/tutorials/preprocessing/fourier-transform-and-spectrograms.html) for mono audio would follow this shape, where `2` corresponds to the real and imaginary parts of the STFT (complex number):

```
(batch_size, time_steps, n_fft, 2)
```

- If we were to process the STFT of multi-channel audio (e.g. stereo), it would look like this:

```
(batch_size, time_steps, n_fft, 2, n_audio_channels)
```

- For [Filter Banks (FBanks)](https://speechbrain.readthedocs.io/en/develop/tutorials/preprocessing/speech-features.html), the shape would be:

```
(batch_size, time_steps, n_filters)
```

## Modified PyTorch globals and GPU quirks

For various reasons, SpeechBrain modifies some PyTorch global configuration to work around issues or improve execution speed, sometimes depending on GPU configuration.
We do so when we consider that some modified defaults make more sense given our usecases than PyTorch's defaults. For instance, we very commonly encounter dynamic tensor shapes, which comes at odds with certain auto-tuning methods.

These changes are applied in a standardized location, [`quirks.py`](https://github.com/speechbrain/speechbrain/tree/develop/speechbrain/utils/quirks.py). They are logged when starting an experiment.

The `SB_DISABLE_QUIRKS` environment variable lets you disable quirks easily. For instance, to disable TensorFloat32 and re-enable JIT profiling, you would use `SB_DISABLE_QUIRKS=allow_tf32,disable_jit_profiling`.

## Reproducibility

To improve reproducibility across experiments, SpeechBrain supports its own seeding function located in `speechbrain.utils.seed.seed_everything`. This function sets the seed for various generators such as NumPy, PyTorch, and Python, following the [PyTorch recommendations](https://pytorch.org/docs/stable/notes/randomness.html).

However, due to the differences in how GPU and CPU executions work, results may not be fully reproducible even with identical seeds, especially when training models. This issue primarily affects training experiments.

On the other hand, when preparing data using data preparation scripts, the output of these scripts is independent of the global seeds. This ensures that you will get identical outputs on different setups, even if different seeds are used.

In distributed experiments, reproducibility becomes more complex as different seeds (offset by the rank) will be set on different machines or processes. This primarily impacts operations that rely on randomness, such as data augmentations. Since each process in a distributed setup is assigned its own seed, the randomness applied to data (e.g., augmentations) can differ between processes, even though the global seed is the same across machines.

It’s important to note that this variance in seeding does not affect certain elements of the experiment. For instance, initial model parameters are broadcast to all processes from the main process in distributed training. Similarly, components like data loaders, which shuffle data, will be affected by per-process seeds, but the underlying data pipeline remains synchronized across processes.
