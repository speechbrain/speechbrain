"""Defines interfaces for simple inference with pretrained models"""
from speechbrain.dataio.preprocess import AudioNormalizer
import torchaudio


class ASR:
    """General interface for Automatic Speech Recognition"""

    def __init__(self, normalizer=None):
        if normalizer is None:
            normalizer = AudioNormalizer()
        self.normalizer = normalizer

    def load_audio(self, path):
        """Load an audio file with this model's input spec

        When using an ASR model, it is important to use the same type of data,
        as was used to train the model. This means for example using the same
        sampling rate and number of channels. It is, however, possible to
        convert a file from a higher sampling rate to a lower one (downsampling).
        Similarly, it is simple to downmix a stereo file to mono.
        """
        signal, sr = torchaudio.load(path, channels_first=False)
        return self.normalizer(signal, sr)

    def transcribe_file(self, path):
        waveform = self.load_audio(path)
        return self.transcribe(waveform)

    def transcribe(self, waveform):
        MSG = "Each ASR model should implement the transcribe() method."
        raise NotImplementedError(MSG)
