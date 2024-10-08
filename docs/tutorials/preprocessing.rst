Speech Preprocessing
====================

..
   Originally generated with https://gist.github.com/asumagic/19f9809480b62bfd16094fb5c844a564 but OK to edit in repo now.
   Please ensure for each tutorial that you are adding it to the hidden toctree at the end of the file!

.. toctree::
   :hidden:

   preprocessing/speech-augmentation.ipynb
   preprocessing/fourier-transform-and-spectrograms.ipynb
   preprocessing/speech-features.ipynb
   preprocessing/environmental-corruption.ipynb
   preprocessing/multi-microphone-beamforming.ipynb


.. rubric:: `🔗 Speech Augmentation <preprocessing/speech-augmentation.html>`_
   :heading-level: 2

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 0

   * - Ravanelli M.
     - Jan. 2021
     - Difficulty: easy
     - Time: 20min
     - `🔗 Google Colab <https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/preprocessing/speech-augmentation.ipynb>`__


A popular saying in machine learning is "there is no better data than more data". However, collecting new data can be expensive
and we must cleverly use the available dataset. One popular technique is called speech augmentation. The idea is to artificially
corrupt the original speech signals to give the network the "illusion" that we are processing a new signal. This acts as a powerful regularizer,
that normally helps neural networks improving generalization and thus achieve better performance on test data.

.. rubric:: `🔗 Fourier Transforms and Spectrograms <preprocessing/fourier-transform-and-spectrograms.html>`_
   :heading-level: 2

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 0

   * - Ravanelli M.
     - Jan. 2021
     - Difficulty: easy
     - Time: 20min
     - `🔗 Google Colab <https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/preprocessing/fourier-transform-and-spectrograms.ipynb>`__


In speech and audio processing, the signal in the time-domain is often transformed into another domain.
But why do we need to transform an audio signal? This is because some speech characteristics/patterns of the signal (e.g, pitch, formats)
might not be very evident when looking at the audio in the time-domain. With properly designed transformations,
it might be easier to extract the needed information from the signal itself.

The most popular transformation is the
Fourier Transform, which turns the time-domain signal into an equivalent representation in the frequency domain.
In the following sections, we will describe the Fourier transforms along with other related transformations such as
Short-Term Fourier Transform (STFT) and spectrograms.

.. rubric:: `🔗 Speech Features <preprocessing/speech-features.html>`_
   :heading-level: 2

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 0

   * - Ravanelli M.
     - Jan. 2021
     - Difficulty: easy
     - Time: 20min
     - `🔗 Google Colab <https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/preprocessing/speech-features.ipynb>`__


Speech is a very high-dimensional signal. For instance, when the sampling frequency is 16 kHz,
we have 16000 samples for each second. Working with such very high dimensional data can be critical from a machine learning perspective.
The goal of feature extraction is to find more compact ways to represent speech.

.. rubric:: `🔗 Environmental Corruption <preprocessing/environmental-corruption.html>`_
   :heading-level: 2

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 0

   * - Ravanelli M.
     - Feb. 2021
     - Difficulty: medium
     - Time: 20min
     - `🔗 Google Colab <https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/preprocessing/environmental-corruption.ipynb>`__


In realistic speech processing applications, the signal recorded by the microphone is corrupted by noise and reverberation.
This is particularly harmful in distant-talking (far-field) scenarios, where the speaker and the reference microphone are distant
(think about popular devices such as Google Home, Amazon Echo, Kinect, and similar devices).

.. rubric:: `🔗 Multi-microphone Beamforming <preprocessing/multi-microphone-beamforming.html>`_
   :heading-level: 2

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 0

   * - Grondin F. & Aris W.
     - Jan. 2021
     - Difficulty: medium
     - Time: 20min
     - `🔗 Google Colab <https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/preprocessing/multi-microphone-beamforming.ipynb>`__


Using a microphone array can be very handy to improve the signal quality
(e.g. reduce reverberation and noise) prior to performing speech recognition tasks.
Microphone arrays can also estimate the direction of arrival of a sound source, and this information can later
be used to "listen" in the direction of the source of interest.
