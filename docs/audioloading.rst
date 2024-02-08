=============================
Audio loading troubleshooting
=============================

This page is intended to document how to install torchaudio backends and
provides troubleshooting steps for your audio loading troubles.

Introduction
============

SpeechBrain relies on
`torchaudio <https://pytorch.org/audio/stable/index.html>`_
for loading audio files in most cases. Please first try to **update torchaudio**
if you are encountering issues. Please also ensure that you are using the
correct PyTorch version for your installed torchaudio version.

As of torchaudio `2.2.0`, three backends are supported: ``ffmpeg``, ``sox`` and
``soundfile``. torchaudio documents how their backends are found in their
`optional dependency docs <https://pytorch.org/audio/stable/installation.html#optional-dependencies>`_.

You can determine which backends are available in your environment by running
:func:`torchaudio.list_audio_backends`.

.. warning::
    **A backend can *silently* fail to load** if initialization failed and will be
    omitted from this list.

.. warning::
    **Not every backend can support any codec.** For instance, at the time of
    writing, the torchaudio SoX backend cannot handle MP3 and the SoundFile
    backend cannot handle AAC (usually ``.m4a``), both of which are found in
    certain popular speech datasets.
    However, most common formats are typically well supported by all backends
    (``.wav``/``.ogg`` vorbis/opus/``.flac``).

Recommended install steps
=========================

Often, torchaudio will work out of the box. On certain systems, there might not
be a working backend installed. We recommend you try if any of those steps fixes
your issue:

- On Linux, if you have superuser rights, install ffmpeg and/or libsndfile
  and/or SoX through your distribution's package manager.

- On Windows/Linux/macOS, you can try installing ffmpeg through Conda
  (see `ffmpeg`_), which does not require superuser rights (provided Conda is
  available).

- On macOS, alternatively, it appears to be possible to install ffmpeg through
  Homebrew. Make sure that you are installing a version compatible with
  torchaudio (see `ffmpeg`_).

- On Windows/Linux/macOS, `SoundFile <https://pypi.org/project/soundfile/>`_
  has started shipping with a prebuilt ``libsndfile``, which does not require
  admin rights. Try installing or updating it. See the linked page for more
  details.

Note for developers & breaking torchaudio `2.x` changes
=======================================================

With torchaudio `<2.x`, backends were selected through
``torchaudio.set_audio_backend``. This function was deprecated and then
removed in the `2.x` branch of torchaudio and is no longer used in SpeechBrain.
Since then, the backend is (optionally) selected through the ``backend``
argument of :func:`torchaudio.load` and :func:`torchaudio.info`.

Installing/troubleshooting backends
===================================

ffmpeg
------

torchaudio compiles their ffmpeg backend for a **specific range** of ffmpeg
versions.

ffmpeg is commonly already installed on common Linux distributions.
On Ubuntu, it can be installed through ``sudo apt install ffmpeg``.

Depending on your OS version, it is possible that your installed ffmpeg version
is not supported by torchaudio (if too recent or too old).
If you believe this to be the case, you can try installing a specific version
of the ``ffmpeg`` package as supplied by
`conda-forge <https://anaconda.org/conda-forge/ffmpeg>`_.

See `torchaudio documentation on optional dependencies <https://pytorch.org/audio/stable/installation.html#optional-dependencies>`_ for more details.

SoundFile
---------

torchaudio can use `soundfile <https://pypi.org/project/soundfile/>`_ as an
audio backend, which depends on ``libsndfile``.

Starting with SoundFile 0.12.0, this package bundles a prebuilt ``libsndfile``
for a number of platforms. Refer to the project page for more details.

SoX
---

Starting with torchaudio 0.12.0, the SoX backend no longer supports mp3 files.

Starting with torchaudio 2.1.0, torchaudio no longer compiles and bundles SoX
by itself, and expects it to be provided by the system.

If you have upgraded from an earlier version and can no longer load audio files,
it may be due to this. In this case, you may need to install SoX or use a
different backend.
