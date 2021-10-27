.. SpeechBrain documentation master file, created by
   sphinx-quickstart on Tue Apr  7 20:07:28 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SpeechBrain
=======================================

.. image:: images/logo_noname_rounded_big.png
  :width: 400
  :align: center

SpeechBrain is an open-source and all-in-one speech toolkit based on PyTorch.
This documentation is intended to give SpeechBrain users all the API
information necessary to develop their projects. For tutorials,
please refer to the official `Github <https://github.com/speechbrain/speechbrain>`_
or the official `Website <https://speechbrain.github.io>`


Licence
--------

SpeechBrain is released under the Apache license, version 2.0. The Apache license is a popular BSD-like license.
SpeechBrain can be redistributed for free, even for commercial purposes, although you can not take off the license headers (and under some circumstances you may have to distribute a license document).
Apache is not a viral license like the GPL, which forces you to release your modifications to the source code. Also note that this project has no connection to the Apache Foundation, other than that we use the same license terms.

It is a community project, which means that discussions are engaged community-wide while decisions are taken by Dr. Ravanelli and Dr. Parcollet with respect to the community views.
There is no legal institution associated as an owner of SpeechBrain. Furthermore, and due to the Apache Licence, anyone that would disagree with the way the project is being run can fork it and start a new toolkit.

Referencing SpeechBrain
--------
.. code-block:: txt

  @misc{speechbrain,
      title={SpeechBrain: A General-Purpose Speech Toolkit}, 
      author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
      year={2021},
      eprint={2106.04624},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}


.. toctree::
   :maxdepth: 1
   :caption: Getting started:

   installation.md
   experiment.md
   multigpu.md
   tutorials.md
   contributing.md

API Documentation
--------

.. toctree::
   :caption: API Documentation:
   :hidden:
   :maxdepth: 3

   Core library (speechbrain) <API/speechbrain>
   HyperPyYAML (hyperpyyaml) <API/hyperpyyaml>

.. autosummary::

   speechbrain
   speechbrain.alignment
   speechbrain.dataio
   speechbrain.decoders
   speechbrain.lm
   speechbrain.lobes
   speechbrain.nnet
   speechbrain.processing
   speechbrain.tokenizers
   speechbrain.utils
