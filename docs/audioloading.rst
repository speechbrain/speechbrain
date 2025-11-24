=============================
Audio loading troubleshooting
=============================

This page is intended to document how to install audio backends and
provides troubleshooting steps for your audio loading troubles.

Introduction
============

SpeechBrain now uses `soundfile <https://pypi.org/project/soundfile/>`_ as the
sole supported audio I/O backend through the :mod:`speechbrain.dataio.audio_io` module.

The soundfile backend supports most common audio formats including:
``wav``, ``flac``, and ``mp3``. For advanced format support or issues,
please refer to the sections below.

.. note::
   **Legacy torchaudio backends**: SpeechBrain previously used torchaudio for
   audio I/O, which supported three backends: ``ffmpeg``, ``sox`` and ``soundfile``.
   However, torchaudio 2.9 deprecated all audio I/O support so SpeechBrain
   now relies on ``soundfile`` directly for audio I/O.

Recommended install steps
=========================

The pip package `soundfile` is a dependency of SpeechBrain and should be automatically
installed when you install SpeechBrain.
Starting with SoundFile 0.12.0, the pip package bundles a prebuilt ``libsndfile``
for most platforms (Windows, macOS, Linux), so it typically works out of the box
when installed via pip.

If you encounter issues with audio loading:

- **Update soundfile**: Try running ``pip install --upgrade soundfile`` to get
  the latest version with updated ``libsndfile`` binaries.

- **On Linux with superuser rights**: Install ``libsndfile`` through your
  distribution's package manager (e.g., ``sudo apt install libsndfile1`` on
  Ubuntu/Debian).

- **For advanced codec support**: If you need to work with formats not supported
  by soundfile (e.g., AAC/M4A), you may need to convert your audio files
  to a supported format like WAV or FLAC using external tools such as ``ffmpeg``.

- **Check installation**: You can verify soundfile is working by running:

  .. code-block:: python

      import soundfile as sf
      print(sf.__version__)
      print(sf.available_formats())

SpeechBrain Audio I/O API
==========================

SpeechBrain provides its own audio I/O interface through the
:mod:`speechbrain.dataio.audio_io` module. Usage example:

.. code-block:: python

    from speechbrain.dataio import audio_io

    # Load audio file
    audio, sample_rate = audio_io.load("path/to/audio.wav")

    # Get audio metadata
    info = audio_io.info("path/to/audio.wav")
    print(info.sample_rate, info.duration, info.channels)

    # Save audio file
    audio_io.save("output.wav", audio, sample_rate)

This API is compatible with the previous torchaudio-based interface, making
migration straightforward.
