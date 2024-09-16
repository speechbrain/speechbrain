Speech Processing Tasks
=======================

..
   Originally generated with https://gist.github.com/asumagic/19f9809480b62bfd16094fb5c844a564 but OK to edit in repo now.
   Please ensure for each tutorial that you are adding it to the hidden toctree at the end of the file!

.. toctree::
   :hidden:

   tasks/speech-recognition-from-scratch.ipynb
   tasks/source-separation.ipynb
   tasks/speech-enhancement-from-scratch.ipynb
   tasks/speech-classification-from-scratch.ipynb
   tasks/voice-activity-detection.ipynb

.. rubric:: `🔗 Speech Recognition From Scratch <tasks/speech-recognition-from-scratch.html>`_
   :heading-level: 2

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 0

   * - Ravanelli M. & Parcollet T.
     - Apr. 2021
     - Difficulty: medium
     - Time: 45min
     - `🔗 Google Colab <https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/tutorials/asr/speech-recognition-from-scratch.ipynb>`__


Do you want to figure out how to implement your speech recognizer with SpeechBrain? Look no further, you're in the right place. This tutorial will walk you through all the steps needed to implement an offline end-to-end attention-based speech recognizer. This is a self-contained tutorial that will help you "connecting the dots" across all the steps needed to train a modern speech recognizer. We will address data preparation, tokenizer training, language model, ASR model, and inference. We will explain how to train your model on your data.

.. rubric:: `🔗 Source Separation <tasks/source-separation.html>`_
   :heading-level: 2

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 0

   * - Subakan C.
     - Jan. 2021
     - Difficulty: medium
     - Time: 30min
     - `🔗 Google Colab <https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/tutorials/separation/source-separation.ipynb>`__


In source separation, the goal is to be able to separate out the sources from an observed mixture signal
which consists of superposition of several sources. In this tutorial, we cover few examples of performing source separation with SpeechBrain.

.. rubric:: `🔗 Speech Enhancement From Scratch <tasks/speech-enhancement-from-scratch.html>`_
   :heading-level: 2

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 0

   * - Plantinga P.
     - Feb. 2021
     - Difficulty: medium
     - Time: 30min
     - `🔗 Google Colab <https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/tutorials/enhancement/speech-enhancement-from-scratch.ipynb>`__


So you want to do regression tasks with speech? Look no further, you're in the right place.
This tutorial will walk you through a basic speech enhancement template with SpeechBrain to
show all the components needed for making a new recipe.

.. rubric:: `🔗 Speech Classification From Scratch <tasks/speech-classification-from-scratch.html>`_
   :heading-level: 2

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 0

   * - Ravanelli M.
     - Jan. 2021
     - Difficulty: medium
     - Time: 30min
     - `🔗 Google Colab <https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/tutorials/classification/speech-classification-from-scratch.ipynb>`__


In this tutorial, we show how to use SpeechBrain to implement an utterance-level speech classifier.
It might help if you want to develop systems for speaker-id, language-id, emotion recognition, sound classification, keyword spotting, and many 					     	     other tasks.

.. rubric:: `🔗 Voice Activity Detection <tasks/voice-activity-detection.html>`_
   :heading-level: 2

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 0

   * - Ravanelli M.
     - Sept. 2021
     - Difficulty: easy
     - Time: 15min
     - `🔗 Google Colab <https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/tutorials/classification/voice-activity-detection.ipynb>`__


In this tutorial, we show how to use SpeechBrain for voice activity detection. The tutorial will describe how to train a neural VAD and use it for inference on long audio recordings.
