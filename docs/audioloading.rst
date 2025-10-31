=============================
Audio loading troubleshooting
=============================

This page is intended to document how to install audio backends and
provides troubleshooting steps for your audio loading troubles.

Introduction
============

SpeechBrain uses `soundfile <https://pypi.org/project/soundfile/>`_ as the
primary audio I/O backend through the :mod:`speechbrain.dataio.audio_io` module.
For audio processing operations (e.g., resampling), SpeechBrain still relies on
`torchaudio <https://pytorch.org/audio/stable/index.html>`_.

The soundfile backend supports most common audio formats including:
``wav``, ``flac``, ``ogg`` (vorbis/opus). For advanced format support or issues,
please refer to the sections below.

.. note::
    **Legacy torchaudio backends**: SpeechBrain previously used torchaudio for
    audio I/O. As of torchaudio `2.2.0`, three backends were supported: ``ffmpeg``,
    ``sox`` and ``soundfile``. While SpeechBrain now uses soundfile directly,
    torchaudio is still used for audio processing operations like resampling.

.. warning::
    **Not every backend can support any codec.** For instance, soundfile cannot
    handle AAC (usually ``.m4a``) or MP3 files by default, both of which are
    found in certain popular speech datasets. However, most common formats are
    well supported (``.wav``/``.ogg`` vorbis/opus/``.flac``).

Recommended install steps
=========================

SpeechBrain uses soundfile for audio I/O, which is automatically installed when
you install SpeechBrain. On most systems, soundfile will work out of the box
as it ships with a prebuilt ``libsndfile`` library.

If you encounter issues with audio loading:

- **Update soundfile**: Try running ``pip install --upgrade soundfile`` to get
  the latest version with updated ``libsndfile`` binaries.

- **On Linux with superuser rights**: Install ``libsndfile1`` through your
  distribution's package manager (e.g., ``sudo apt install libsndfile1`` on
  Ubuntu/Debian).

- **For advanced codec support**: If you need to work with formats not supported
  by soundfile (e.g., MP3, AAC/M4A), you may need to convert your audio files
  to a supported format like WAV or FLAC using external tools such as ``ffmpeg``.

- **Check installation**: You can verify soundfile is working by running:

  .. code-block:: python

      import soundfile as sf
      print(sf.__version__)
      print(sf.available_formats())

SpeechBrain Audio I/O API
==========================

SpeechBrain provides its own audio I/O interface through the
:mod:`speechbrain.dataio.audio_io` module:

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

Note for developers & breaking torchaudio `2.x` changes
=======================================================

**SpeechBrain now uses soundfile for audio I/O** instead of torchaudio. The
:mod:`speechbrain.dataio.audio_io` module provides load, save, and info
functions that replace the previous torchaudio equivalents.

Legacy note: With torchaudio `<2.x`, backends were selected through
``torchaudio.set_audio_backend``. This function was deprecated and then
removed in the `2.x` branch of torchaudio.

Installing/troubleshooting soundfile
=====================================

SoundFile backend
-----------------

SpeechBrain uses `soundfile <https://pypi.org/project/soundfile/>`_ for audio
I/O operations, which depends on ``libsndfile``.

Starting with SoundFile 0.12.0, this package bundles a prebuilt ``libsndfile``
for many platforms (Windows, macOS, Linux), so it typically works out of the box
when installed via pip.

If you encounter issues:

1. **Update soundfile**: ``pip install --upgrade soundfile``
2. **Check version**: Ensure you have soundfile >= 0.12.1
3. **Manual libsndfile installation** (if needed):

   - **Ubuntu/Debian**: ``sudo apt install libsndfile1``
   - **Fedora/RHEL**: ``sudo dnf install libsndfile``
   - **macOS**: ``brew install libsndfile``
   - **Windows**: The prebuilt binaries should work automatically

Refer to the `soundfile project page <https://pypi.org/project/soundfile/>`_
for more details.

Supported formats
-----------------

Soundfile supports the following formats through libsndfile:

- **Well supported**: WAV, FLAC, OGG (Vorbis/Opus), AIFF
- **Not supported**: MP3, AAC (M4A), WMA

For unsupported formats, you'll need to convert your audio files to a supported
format. You can use ffmpeg for conversion:

.. code-block:: bash

    # Convert MP3 to WAV
    ffmpeg -i input.mp3 output.wav

    # Convert M4A to FLAC
    ffmpeg -i input.m4a output.flac

Legacy torchaudio backends (for reference)
===========================================

For users coming from torchaudio-based workflows, note that SpeechBrain
previously supported torchaudio backends (ffmpeg, sox, soundfile). These are
no longer used for audio I/O in SpeechBrain, though torchaudio is still used
for audio processing operations.

If you need to work with the formats that require these backends (MP3, AAC),
we recommend converting your audio files to WAV or FLAC format as described above.
