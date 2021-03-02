import speechbrain as sb
from torchaudio import transforms

@sb.utils.data_pipeline.takes("wav")
@sb.utils.data_pipeline.provides("sig")
def audio_pipeline(file_name: str):
    """
    A pipeline function that reads a single file and reads
    audio from it into a tensor
    """
    return sb.dataio.dataio.read_audio(file_name)


def mel_spectrogram(*args, **kwargs):
    """
    A pipeline wrapper function for torchaudio.transforms.MelSpectrogram,
    which produces a MEL spectrogram out of a raw waveform
    """
    mel = transforms.MelSpectrogram(*args, **kwargs)

    @sb.utils.data_pipeline.takes("sig")
    @sb.utils.data_pipeline.provides("mel")
    def f(sig):
        return mel(sig)

    return f
    