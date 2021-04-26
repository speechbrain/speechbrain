import speechbrain as sb
from torchaudio import transforms
from speechbrain.dataio.encoder import CategoricalEncoder


def wrap_transform(transform_type, takes=None, provides=None):
    """
    Wraps a Torch transform for the pipeline, returning a
    decorator

    Arguments
    ---------
    transform_type: torch.nn.Module
        a Torch transform (from torchaudio.Transforms) to be wrapped
    takes: str
        the name of the pipeline input
    provides: str
        the name of the pipeline output

    Arguments
    ---------
    takes: str
        the name of the pipeline input
    provides: str
        the name of the pipeline output

    Returns
    -------
    result: DynamicItem
      A wrapped transformation function    
    """
    default_takes = takes
    default_provides = provides
    def decorator(takes=None, provides=None, *args, **kwargs):
        transform = transform_type(*args, **kwargs)
        @sb.utils.data_pipeline.takes(takes or default_takes)
        @sb.utils.data_pipeline.provides(provides or default_provides)
        def f(*args, **kwargs):            
            return transform.to(args[0].device)(*args, **kwargs)
        return f

    return decorator


def categorical(takes, provides):
    """
    A pipeline function that encodes categorical information,
    such as speaker IDs

    Arguments
    ---------
    takes: str
        the name of the pipeline input
    provides: str
        the name of the pipeline output

    Returns
    -------
    result: DynamicItem
      A wrapped transformation function        
    """
    encoder = CategoricalEncoder()
    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)
    def f(label):
        encoder.ensure_label(label)
        return encoder.ensure_label(label)
    
    return f


@sb.utils.data_pipeline.takes("wav")
@sb.utils.data_pipeline.provides("sig")
def audio_pipeline(file_name: str):
    """
    A pipeline function that reads a single file and reads
    audio from it into a tensor
    """
    return sb.dataio.dataio.read_audio(file_name)


def transpose_spectrogram(takes, provides):
    """
    A pipeline function that transposes a spectrogram along the
    last two axes

    Arguments
    ---------
    takes: str
        the name of the pipeline input
    provides: str
        the name of the pipeline output

    Returns
    -------
    result: DymamicItem    
    """
    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)
    def f(spectrogram):
        return spectrogram.transpose(-1, -2)
    return f


resample = wrap_transform(transforms.Resample, takes="sig", provides="sig_resampled")
mel_spectrogram = wrap_transform(transforms.MelSpectrogram, takes="sig", provides="mel")
spectrogram = wrap_transform(transforms.Spectrogram, takes="sig", provides="spectrogram")
inverse_spectrogram = wrap_transform(transforms.GriffinLim, takes="spectrogram", provides="sig")