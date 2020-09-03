# ConvTasnet Recipe

* This recipe trains a [conv-tasnet model](https://arxiv.org/pdf/1809.07454.pdf) on the WSJ0-2mix dataset as defined [in this paper](https://arxiv.org/pdf/1508.04306.pdf).

* This recipe first tries to create the WSJ0-2Mix dataset if it is not already present on the path `wsj0mixpath`. If the WS20-Mix dataset is not present, then you need the have WSJ0 dataset, the path for which is specified with `wsj0path`. The dataset creation script assumes that the original WSj0 files in the sphere format are already converted to .wav .

* The data creation script for the WSJ-2Mix dataset uses a Matlab script, which is called through a python interface (oct2py) for octave. Therefore you need to have octave installed.

* Depending on your octave version, you might observe the following error:
```
error: called from graphics_toolkit at line 81 column 5
 graphics_toolkit: = toolkit is not available
```
This is in essence a warning and does not affect dataset creation.



