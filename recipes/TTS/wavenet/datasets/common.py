import speechbrain as sb
from torchaudio import transforms


def wrap_transform(transform_type, takes=None, provides=None):
    """
    Wraps a Torch transform for the pipeline, returning a
    decorator
    """
    default_takes = takes
    default_provides = provides
    def decorator(takes=None, provides=None, *args, **kwargs):
        transform = transform_type(*args, **kwargs)
        @sb.utils.data_pipeline.takes(takes or default_takes)
        @sb.utils.data_pipeline.provides(provides or default_provides)
        def f(*args, **kwargs):
            return transform(*args, **kwargs)
        return f

    return decorator


@sb.utils.data_pipeline.takes("wav")
@sb.utils.data_pipeline.provides("sig")
def audio_pipeline(file_name: str):
    """
    A pipeline function that reads a single file and reads
    audio from it into a tensor
    """
    return sb.dataio.dataio.read_audio(file_name)


resample = wrap_transform(transforms.Resample, takes="sig", provides="sig_resampled")
mel_spectrogram = wrap_transform(transforms.MelSpectrogram, takes="sig", provides="mel")