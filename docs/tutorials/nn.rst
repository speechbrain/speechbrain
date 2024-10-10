Neural Architectures
====================

..
   Originally generated with https://gist.github.com/asumagic/19f9809480b62bfd16094fb5c844a564 but OK to edit in repo now.
   Please ensure for each tutorial that you are adding it to the hidden toctree at the end of the file!

.. toctree::
   :hidden:

   nn/using-wav2vec-2.0-hubert-wavlm-and-whisper-from-huggingface-with-speechbrain.ipynb
   nn/neural-network-adapters.ipynb
   nn/complex-and-quaternion-neural-networks.ipynb
   nn/recurrent-neural-networks-and-speechbrain.ipynb
   nn/conformer-streaming-asr.ipynb


.. rubric:: `🔗 Fine-tuning or using Whisper, wav2vec2, HuBERT and others with SpeechBrain and HuggingFace <nn/using-wav2vec-2.0-hubert-wavlm-and-whisper-from-huggingface-with-speechbrain.html>`_
   :heading-level: 2

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 0

   * - Parcollet T. & Moumen A.
     - Dec. 2022
     - Difficulty: medium
     - Time: 20m
     - `🔗 Google Colab <https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/nn/using-wav2vec-2.0-hubert-wavlm-and-whisper-from-huggingface-with-speechbrain.ipynb>`__


This tutorial describes how to combine (use and finetune) pretrained models
coming from HuggingFace. Any wav2vec 2.0 / HuBERT / WavLM or Whisper model integrated to the transformers interface of HuggingFace can be then plugged to
SpeechBrain to approach a speech-related task: automatic speech recognition, speaker recognition,
spoken language understanding ...

.. rubric:: `🔗 Neural Network Adapters for faster low-memory fine-tuning <nn/neural-network-adapters.html>`_
   :heading-level: 2

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 0

   * - Plantinga P.
     - Sept. 2024
     - Difficulty: easy
     - Time: 20m
     - `🔗 Google Colab <https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/nn/neural-network-adapters.ipynb>`__


This tutorial covers the SpeechBrain implementation of adapters such as LoRA. This includes how to integrate either SpeechBrain implemented adapters, custom adapters, and adapters from libraries such as PEFT into a pre-trained model.

.. rubric:: `🔗 Complex and Quaternion Neural Networks <nn/complex-and-quaternion-neural-networks.html>`_
   :heading-level: 2

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 0

   * - Parcollet T.
     - Feb. 2021
     - Difficulty: medium
     - Time: 30min
     - `🔗 Google Colab <https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/nn/complex-and-quaternion-neural-networks.ipynb>`__


This tutorial demonstrates how to use the SpeechBrain implementation of complex-valued and quaternion-valued neural networks
for speech technologies. It covers the basics of highdimensional representations and the associated neural layers :
Linear, Convolution, Recurrent and Normalisation.

.. rubric:: `🔗 Recurrent Neural Networks <nn/recurrent-neural-networks-and-speechbrain.html>`_
   :heading-level: 2

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 0

   * - Ravanelli M.
     - Feb. 2021
     - Difficulty: easy
     - Time: 30min
     - `🔗 Google Colab <https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/nn/recurrent-neural-networks-and-speechbrain.ipynb>`__


Recurrent Neural Networks (RNNs) offer a natural way to process sequences.
This tutorial demonstrates how to use the SpeechBrain implementations of RNNs including LSTMs, GRU, RNN and LiGRU a specific recurrent cell designed
for speech-related tasks. RNNs are at the core of many sequence to sequence models.


.. rubric:: `🔗 Streaming Speech Recognition with Conformers <nn/conformer-streaming-asr.html>`_
   :heading-level: 2

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 0

   * - de Langen S.
     - Sep. 2024
     - Difficulty: medium
     - Time: 60min+
     - `🔗 Google Colab <https://colab.research.google.com/github/speechbrain/speechbrain/blob/develop/docs/tutorials/nn/conformer-streaming-asr.ipynb>`__


Automatic Speech Recognition (ASR) models are often only designed to transcribe an entire large chunk of audio and are unsuitable for usecases like live stream transcription, which requires low-latency, long-form transcription.

This tutorial introduces the Dynamic Chunk Training approach and architectural changes you can apply to make the Conformer model streamable. It introduces the tooling for training and inference that SpeechBrain can provide for you.
This might be a good starting point if you're interested in training and understanding your own streaming models, or even if you want to explore improved streaming architectures.
