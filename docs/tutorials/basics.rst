SpeechBrain Basics
==================

..
   Originally generated with https://gist.github.com/asumagic/19f9809480b62bfd16094fb5c844a564 but OK to edit in repo now.
   Please ensure for each tutorial that you are adding it to the hidden toctree at the end of the file!

.. toctree::
   :hidden:

   basics/introduction-to-speechbrain.ipynb
   basics/what-can-i-do-with-speechbrain.ipynb
   basics/brain-class.ipynb
   basics/hyperpyyaml.ipynb
   basics/data-loading-pipeline.ipynb
   basics/checkpointing.ipynb


.. rubric:: `🔗 Introduction to SpeechBrain <basics/introduction-to-speechbrain.html>`_
   :heading-level: 2

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 0

   * - Ravanelli M.
     - Feb. 2021
     - Difficulty: easy
     - Time: 10min
     - `🔗 Google Colab <https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/basics/introduction-to-speechbrain.ipynb>`__


SpeechBrain is an open-source all-in-one speech toolkit based on PyTorch.
It is designed to make the research and development of speech technology easier. Alongside with our documentation
this tutorial will provide you all the very basic elements needed to start using SpeechBrain for your projects.

.. rubric:: `🔗 What can I do with SpeechBrain? <basics/what-can-i-do-with-speechbrain.html>`_
   :heading-level: 2

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 0

   * - Ravanelli M.
     - Jan. 2021
     - Difficulty: easy
     - Time: 10min
     - `🔗 Google Colab <https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/basics/what-can-i-do-with-speechbrain.ipynb>`__


In this tutorial, we provide a high-level description of the speech tasks currently supported by SpeechBrain.
We also show how to perform inference on speech recognition, speech separation, speaker verification, and other applications.


.. rubric:: `🔗 The Brain Class <basics/brain-class.html>`_
   :heading-level: 2

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 0

   * - Plantinga P.
     - Jan. 2021
     - Difficulty: easy
     - Time: 10min
     - `🔗 Google Colab <https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/basics/brain-class.ipynb>`__


One key component of deep learning is iterating the dataset multiple times and performing parameter updates.
This process is sometimes called the "training loop" and there are usually many stages to this loop.

SpeechBrain provides a convenient framework for organizing the training loop, in the form of a class known as the "Brain" class,
implemented in `speechbrain/core.py`. In each recipe, we sub-class this class and override the methods for which the default
implementation doesn't do what is required for that particular recipe.

.. rubric:: `🔗 HyperPyYAML Tutorial <basics/hyperpyyaml.html>`_
   :heading-level: 2

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 0

   * - Plantinga P.
     - Jan. 2021
     - Difficulty: easy
     - Time: 15min
     - `🔗 Google Colab <https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/basics/hyperpyyaml.ipynb>`__


An essential part of any deep learning pipeline is the definition of hyperparameters and other metadata.
These data in conjunction with the deep learning algorithms control the various aspects of the pipeline,
such as model architecture, training, and decoding.

At SpeechBrain, we decided that the distinction between
hyperparameters and learning algorithms ought to be evident in the structure of our toolkit, so we split our
recipes into two primary files: `train.py` and `hyperparams.yaml`. The `hyperparams.yaml` file is in a
SpeechBrain-developed format, which we call "HyperPyYAML". We chose to extend YAML since it is a highly
readable format for data serialization. By extending an already useful format, we were able to create an
expanded definition of hyperparameter, keeping our actual experimental code small and highly readable.

.. rubric:: `🔗 Data Loading <basics/data-loading-pipeline.html>`_
   :heading-level: 2

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 0

   * - Cornell S. & Rouhe A.
     - Jan. 2021
     - Difficulty: medium
     - Time: 20min
     - `🔗 Google Colab <https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/basics/data-loading-pipeline.ipynb>`__


Setting up an efficient data loading pipeline is often a tedious task which involves creating the examples,
defining your torch.utils.data.Dataset class as well as different data sampling and augmentations strategies.
In SpeechBrain, we provide efficient abstractions to simplify this time-consuming process without sacrificing
flexibility. In fact our data pipeline is built around the Pytorch one.

.. rubric:: `🔗 Checkpointing <basics/checkpointing.html>`_
   :heading-level: 2

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 0

   * - Rouhe A.
     - Feb. 2021
     - Difficulty: easy
     - Time: 15min
     - `🔗 Google Colab <https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/basics/checkpointing.ipynb>`__


By checkpointing, we mean saving the model and all the other necessary state information
(like optimizer parameters, which epoch and which iteration), at a particular point in time.
