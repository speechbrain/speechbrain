# Template for Speech Enhancement

This folder provides a working, well-documented example for training
a speech enhancement model from scratch, based on a few hours of
data. The data we use is from Mini Librispeech + OpenRIR.

There are four files here:

* `train.py`: the main code file, outlines entire training process.
* `train.yaml`: the hyperparameters file, sets all parameters of execution.
* `custom_model.py`: A file containing the definition of a PyTorch module.
* `mini_librispeech_prepare.py`: If necessary, downloads and prepares data
    manifests.

To train an enhancement model, just execute the following on the command-line:

```bash
python train.py train.yaml --data_folder /path/to/save/mini_librispeech
```

This will automatically download and prepare the data manifest for mini
librispeech, and then train a model with dynamically generated noisy
samples, using noise, reverberation, and babble.

More details about what each file does and how to make modifications
are found within each file. The whole folder can be copied and used
as a starting point for developing recipes doing regression tasks
similar to speech enhancement. Please reach out to the SpeechBrain
team if any errors are found or clarification is needed about how
parts of the template work. Good Luck!

[For more information, please take a look into the "Speech Enhancement from scratch" tutorial](https://colab.research.google.com/drive/18RyiuKupAhwWX7fh3LCatwQGU5eIS3TR?usp=sharing)
