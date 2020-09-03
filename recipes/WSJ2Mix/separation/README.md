# ConvTasnet Recipe

* This recipe trains a [conv-tasnet model](https://arxiv.org/pdf/1809.07454.pdf) on the WSJ0-2mix dataset as defined [in this paper](https://arxiv.org/pdf/1508.04306.pdf)

To run it:

```
python experiment.py hyperparameters/convtasnet.yaml
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
The recipe achieves a SDRi=15.6 dB on the test data.



