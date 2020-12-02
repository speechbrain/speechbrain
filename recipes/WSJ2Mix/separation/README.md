# ConvTasnet Recipe

* This recipe is able to train several different source separation models on WSJ2-Mix, including [Sepformer](https://arxiv.org/abs/2010.13154), [DPRNN](https://arxiv.org/abs/1910.06379) , [ConvTasnet](https://arxiv.org/abs/1809.07454), [DPTNet](https://arxiv.org/abs/2007.13975)

To run it:

```
python experiment.py hyperparams/convtasnet.yaml
```
Make sure you modified the paths inside the parameter file before running the recipe.

## WSJ0-2mix dataset creation
* If not available in the "wsj0mixpath" variable of the parameter file, this recipe creates the standard WSJ0-2Mix dataset from scratch.  To generate it you need to own the standard WSJ0 dataset (available though LDC at https://catalog.ldc.upenn.edu/LDC93S6A).
* The dataset creation script assumes that the original WSJ0 files in the sphere format are already converted to .wav .
* The data creation script for the WSJ-2Mix dataset uses an Octave, which is called through a python interface (oct2py). Therefore you need to have octave installed.

Depending on your octave version, you might observe the following error:
```
error: called from graphics_toolkit at line 81 column 5
 graphics_toolkit: = toolkit is not available
This is in essence a warning and does not affect dataset creation.
```

##  Results

Here are the SI-SNRi results on WSJ0-2Mix:

|   |SepFormer   |DPRNN | DPTNet   | ConvTasnet |
|---|---|---|---|---|
|NoAugment| 20.1  |   |   |   |
|WaveformDrop|   |   |   |   |
|WaveformDrop+SpeedPerturb| 20.9 |   |   |   |
|DynamicMixing   |   |   |   |   |




