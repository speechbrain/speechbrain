UrbanSound8K
============

Created By
----------

Justin Salamon*^, Christopher Jacoby* and Juan Pablo Bello*
* Music and Audio Research Lab (MARL), New York University, USA
^ Center for Urban Science and Progress (CUSP), New York University, USA
http://serv.cusp.nyu.edu/projects/urbansounddataset
http://marl.smusic.nyu.edu/
http://cusp.nyu.edu/

Version 1.0


Description
-----------

This dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes: air_conditioner, car_horn, 
children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, and street_music. The classes are 
drawn from the urban sound taxonomy described in the following article, which also includes a detailed description of 
the dataset and how it was compiled:

J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research", 
22nd ACM International Conference on Multimedia, Orlando USA, Nov. 2014.

All excerpts are taken from field recordings uploaded to www.freesound.org. The files are pre-sorted into ten folds
(folders named fold1-fold10) to help in the reproduction of and comparison with the automatic classification results
reported in the article above.

In addition to the sound excerpts, a CSV file containing metadata about each excerpt is also provided.


Audio Files Included
--------------------

8732 audio files of urban sounds (see description above) in WAV format. The sampling rate, bit depth, and number of 
channels are the same as those of the original file uploaded to Freesound (and hence may vary from file to file).


Meta-data Files Included
------------------------

UrbanSound8k.csv

This file contains meta-data information about every audio file in the dataset. This includes:

* slice_file_name: 
The name of the audio file. The name takes the following format: [fsID]-[classID]-[occurrenceID]-[sliceID].wav, where:
[fsID] = the Freesound ID of the recording from which this excerpt (slice) is taken
[classID] = a numeric identifier of the sound class (see description of classID below for further details)
[occurrenceID] = a numeric identifier to distinguish different occurrences of the sound within the original recording
[sliceID] = a numeric identifier to distinguish different slices taken from the same occurrence

* fsID:
The Freesound ID of the recording from which this excerpt (slice) is taken

* start
The start time of the slice in the original Freesound recording

* end:
The end time of slice in the original Freesound recording

* salience:
A (subjective) salience rating of the sound. 1 = foreground, 2 = background.

* fold:
The fold number (1-10) to which this file has been allocated.

* classID:
A numeric identifier of the sound class:
0 = air_conditioner
1 = car_horn
2 = children_playing
3 = dog_bark
4 = drilling
5 = engine_idling
6 = gun_shot
7 = jackhammer
8 = siren
9 = street_music

* class:
The class name: air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, 
siren, street_music.


Please Acknowledge UrbanSound8K in Academic Research
----------------------------------------------------

When UrbanSound8K is used for academic research, we would highly appreciate it if scientific publications of works 
partly based on the UrbanSound8K dataset cite the following publication:

J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research", 
22nd ACM International Conference on Multimedia, Orlando USA, Nov. 2014.

The creation of this dataset was supported by a seed grant by NYU's Center for Urban Science and Progress (CUSP).


Conditions of Use
-----------------

Dataset compiled by Justin Salamon, Christopher Jacoby and Juan Pablo Bello. All files are excerpts of recordings
uploaded to www.freesound.org. Please see FREESOUNDCREDITS.txt for an attribution list.
 
The UrbanSound8K dataset is offered free of charge for non-commercial use only under the terms of the Creative Commons
Attribution Noncommercial License (by-nc), version 3.0: http://creativecommons.org/licenses/by-nc/3.0/
 
The dataset and its contents are made available on an "as is" basis and without warranties of any kind, including 
without limitation satisfactory quality and conformity, merchantability, fitness for a particular purpose, accuracy or 
completeness, or absence of errors. Subject to any liability that may not be excluded or limited by law, NYU is not 
liable for, and expressly excludes, all liability for loss or damage however and whenever caused to anyone by any use of
the UrbanSound8K dataset or any part of it.


Feedback
--------

Please help us improve UrbanSound8K by sending your feedback to: justin.salamon@nyu.edu or justin.salamon@gmail.com
In case of a problem report please include as many details as possible.
