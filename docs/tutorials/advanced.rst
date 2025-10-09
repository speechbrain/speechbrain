SpeechBrain Advanced
====================

..
   Originally generated with https://gist.github.com/asumagic/19f9809480b62bfd16094fb5c844a564 but OK to edit in repo now.
   Please ensure for each tutorial that you are adding it to the hidden toctree at the end of the file!

.. toctree::
   :hidden:

   advanced/profiling-and-benchmark.ipynb
   advanced/dynamic-batching.ipynb
   advanced/hyperparameter-optimization.ipynb
   advanced/federated-speech-model-training-via-speechbrain-and-flower.ipynb
   advanced/inferring-on-your-own-speechbrain-models.ipynb
   advanced/pre-trained-models-and-fine-tuning-with-huggingface.ipynb
   advanced/data-loading-for-big-datasets-and-shared-filesystems.ipynb
   advanced/text-tokenizer.ipynb
   advanced/model-quantization.ipynb


.. rubric:: `🔗 Performance Profiling <advanced/profiling-and-benchmark.html>`_
   :heading-level: 2

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 0

   * - Nautsch  A.
     - June. 2022
     - Difficulty: medium
     - Time: 45min
     - `🔗 Google Colab <https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/advanced/profiling-and-benchmark.ipynb>`__


Profiling and benchmark of SpeechBrain models can serve different purposes and look at different angles. Performance requirements are highly particular to the use case with that one desires to use SpeechBrain. This provides means to comprehensive self-learning as a starting point to individual growth beyond the provided.

.. rubric:: `🔗 Dynamic Batching: What is it and why it is necessary sometimes <advanced/dynamic-batching.html>`_
   :heading-level: 2

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 0

   * - Nautsch  A. and Cornell S.
     - Nov. 2021
     - Difficulty: medium
     - Time: 25min
     - `🔗 Google Colab <https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/advanced/dynamic-batching.ipynb>`__


Do you want to speed up training or make it less memory-demanding? One possible solution could be dynamic batching. With this approach, you can dynamically sample batches composed of a variable number of sentences. In this tutorial, we show how to use this technique within SpeechBrain.

.. rubric:: `🔗 Hyperparameter Optimization <advanced/hyperparameter-optimization.html>`_
   :heading-level: 2

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 0

   * - Ploujnikov A.
     - Dec. 2021
     - Difficulty: medium
     - Time: 25min
     - `🔗 Google Colab <https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/advanced/hyperparameter-optimization.ipynb>`__


Do you want to optimize the hyperparameters of your model? Are you tired of doing it by hand? This tutorial will describe how you can optimize the hyperparameter of your SpeechBrain model using the Orion toolkit.

.. rubric:: `🔗 Federated Speech Model Training via SpeechBrain and Flower <advanced/federated-speech-model-training-via-speechbrain-and-flower.html>`_
   :heading-level: 2

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 0

   * - Gao Y. & Parcollet T.
     - Nov. 2021
     - Difficulty: high
     - Time: 45min
     - `🔗 Google Colab <https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/advanced/federated-speech-model-training-via-speechbrain-and-flower.ipynb>`__


Are you interested in both federated learning (FL) and speech, but worried about the proper tools to run experiments? Today you will get the answer.
This tutorial introduces how to integrate Flower and SpeechBrain to achieve federated speech model training.

.. rubric:: `🔗 Inferring on your trained SpeechBrain model <advanced/inferring-on-your-own-speechbrain-models.html>`_
   :heading-level: 2

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 0

   * - Parcollet T.
     - Sept.. 2021
     - Difficulty: medium
     - Time: 30min
     - `🔗 Google Colab <https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/advanced/inferring-on-your-own-speechbrain-models.ipynb>`__


In this tutorial, we will learn the three different ways of inferring on a trained model.
This is particularly useful to debug your pipeline or to deploy a model in a production context.

.. rubric:: `🔗 Pre-trained Models and Fine-Tuning with HuggingFace <advanced/pre-trained-models-and-fine-tuning-with-huggingface.html>`_
   :heading-level: 2

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 0

   * - Cornell S. & Parcollet T.
     - Mar. 2021
     - Difficulty: medium
     - Time: 30min
     - `🔗 Google Colab <https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/advanced/pre-trained-models-and-fine-tuning-with-huggingface.ipynb>`__


Training DNN models is often very time-consuming and expensive.
For this reason, whenever it is possible, using off-the-shelf pretrained
models can be convenient in various scenarios.
We provide a simple and straightforward way to download and instantiate a
state-of-the-art pretrained-model from HuggingFace HuggingFace HuggingFace and use it either for direct inference or
or fine-tuning/knowledge distillation or whatever new fancy technique you can come up with!

.. rubric:: `🔗 Data Loading for Big Datasets and Shared Filesystems <advanced/data-loading-for-big-datasets-and-shared-filesystems.html>`_
   :heading-level: 2

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 0

   * - Rouhe A.
     - Feb. 2021
     - Difficulty: medium
     - Time: 15min
     - `🔗 Google Colab <https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/advanced/data-loading-for-big-datasets-and-shared-filesystems.ipynb>`__


Do you have a huge dataset stored in a shared file system? This tutorial will show you how to load large datasets from the shared file system and use them for training a neural network with SpeechBrain.
In particular, we describe a solution based on the WebDataset library, that is easy to integrate within the SpeechBrain toolkit.

.. rubric:: `🔗 Text Tokenization <advanced/text-tokenizer.html>`_
   :heading-level: 2

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 0

   * - Heba A. & Parcollet T.
     - Feb. 2021
     - Difficulty: easy
     - Time: 20min
     - `🔗 Google Colab <https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/advanced/text-tokenizer.ipynb>`__


Machine Learning tasks that process text may contain thousands of vocabulary
words which leads to models dealing with huge embeddings as input/output
(e.g. for one-hot-vectors and ndim=vocabulary_size). This causes an important consumption of memory,
complexe computations, and more importantly, sub-optimal learning due to extremely sparse and cumbersome
one-hot vectors. In this tutorial, we provide all the basics needed to correctly use the SpeechBrain Tokenizer relying
on SentencePiece (BPE and unigram).

.. rubric:: `🔗 Applying Quantization to a Speech Recognition Model <advanced/model-quantization.html>`_
   :heading-level: 2

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 0

   * - Lam J.
     - Apr. 2024
     - Difficulty: medium
     - Time: 30min
     - `🔗 Google Colab <https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/advanced/model-quantization.ipynb>`__


Quantization is a necessary step for many deep neural networks, particularly for tasks requiring low latency and efficient memory usage like real-time automatic speech recognition. This tutorial will introduce the problem of quantization and explain how to perform quantization using SpeechBrain.
