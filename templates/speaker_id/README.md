# Template for Speaker Identification
  
This folder provides a working, well-documented example for training
a speaker identification model from scratch, based on a few hours of
data. The data we use is from Mini Librispeech + OpenRIR.

There are four files here:

* `train.py`: the main code file, outlines the entire training process.
* `train.yaml`: the hyperparameters file, sets all parameters of execution.
* `custom_model.py`: A file containing the definition of a PyTorch module.
* `mini_librispeech_prepare.py`: If necessary, downloads and prepares data manifests.

To train the speaker-id model, just execute the following on the command-line:

```bash
python train.py train.yaml
```

This will automatically download and prepare the data manifest for mini
librispeech, and then train a model with dynamically augmented samples.

More details about what each file does and how to make modifications
are found within each file. The whole folder can be copied and used
as a starting point for developing recipes doing classification tasks
similar to speech speaker-id (e.g, language-id, emotion classification, ..).
Please reach out to the SpeechBrain
team if any errors are found or clarification is needed about how
parts of the template work. Good Luck!

[For more information, please take a look into the "speaker-id from scratch" tutorial](https://colab.research.google.com/drive/1UwisnAjr8nQF3UnrkIJ4abBMAWzVwBMh?usp=sharing)
