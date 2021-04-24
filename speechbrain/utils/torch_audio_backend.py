import platform


def get_torchaudio_backend():
    """Get the backend for torchaudio between soundfile and sox_io accoding to the os.

    Allow users to use soundfile or sox_io according to their os.

    Returns
    -------
    str
        The torchaudio backend to use.
    """
    current_system = platform.system()
    if current_system == "Windows":
        return "soundfile"
    else:
        return "sox_io"
