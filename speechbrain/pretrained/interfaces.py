"""Defines interfaces for simple inference with pretrained models"""
import torch
import torchaudio
from types import SimpleNamespace
from hyperpyyml import load_hyperpyyaml
from speechbrain.pretrained.fetching import fetch
from speechbrain.dataio.preprocess import AudioNormalizer


class Heart:
    """Inference with learned systems - what is known by Heart

    Arguments
    ---------
    modules : dict of str:torch.nn.Module pairs
        The Torch modules that make up the learned system. These can be treated
        in special ways (put on the right device, frozen, etc.)
    hparams : dict
        Each key:value pair should consist of a string key and a hyperparameter
        that is used within the overridden methods. These will
        be accessible via an ``hparams`` attribute, using "dot" notation:
        e.g., self.hparams.model(x).
    eval_mode : bool
        To freeze (requires_grad=False) parameters or not. Normally in inference
        you want to freeze the params. Also calls .eval() on all modules.
    """

    def __init__(
        self, modules=None, hparams=None, freeze_params=True, device="cpu"
    ):

        self.device = device

        # Put modules on the right device, accessible with dot notation
        self.modules = torch.nn.ModuleDict(modules).to(self.device)

        # Make hyperparams available with dot notation too
        if hparams is not None:
            self.hparams = SimpleNamespace(**hparams)

        # If we don't want to backprop, freeze the pretrained parameters
        if freeze_params:
            self.modules.eval()
            for p in self.modules.parameters():
                p.requires_grad = False

    @classmethod
    def from_hparams(
        cls, source, hparams_file="hyperparams.yaml", overrides={}
    ):
        """Fetch and load based from outside source based on HyperPyYAML file

        The source can be a location on the filesystem or online/huggingface

        The hyperparams file should contain a "modules" key, which is a
        speechbrain.utils.parameter_transfer.Pretrainer

        The hyperparams file should contain a "pretrainer" key, which is a
        speechbrain.utils.parameter_transfer.Pretrainer

        """
        hparams_local_path = fetch(source, hparams_file)

        # Load the modules:
        with open(hparams_local_path) as fin:
            hparams = load_hyperpyyaml(fin, overrides)

        # Pretraining:
        pretrainer = hparams["pretrainer"]
        pretrainer.fetch_and_load(source)

        # Now return the system
        return cls(hparams["modules"], hparams)


class ASRInterface(Heart):
    """General interface for Automatic Speech Recognition"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: How to integrate this better?
        if not hasattr(self, "normalizer"):
            self.normalizer = AudioNormalizer()

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


class EncoderDecoderASR(ASRInterface):
    """A ready-to-use Encoder-Decoder ASR model

    The class can be used either to run only the encoder (encode()) to extract
    features or to run the entire encoder-decoder model
    (transcribe()) to transcribe speech.

    Relies on a few keys in the modules dictionary, as follows.
        compute_features
        normalize
        asr_encoder
        beam_searcher
    ```
    TODO: Make this list sensible and minimal

    """

    def __init__(self,):
        super().__init__()

        # TODO: Don't rely on LM!!
        # The tokenizer is the one used by the LM
        self.tokenizer = self.lm_model.tokenizer

    def encode_batch(self, wavs, wav_lens):
        """Encodes the input audio into a sequence of hidden states"""
        wavs = wavs.float()
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        feats = self.mod.compute_features(wavs)
        feats = self.mod.normalize(feats, wav_lens)
        encoder_out = self.mod.asr_encoder(feats)
        return encoder_out

    def transcribe_batch(self, wavs, wav_lens):
        """Transcribes the input audio into a sequence of words"""
        with torch.no_grad():
            wav_lens = wav_lens.to(self.device)
            encoder_out = self.encode_batch(wavs, wav_lens)
            predicted_tokens, scores = self.mod.beam_searcher(
                encoder_out, wav_lens
            )

            predicted_words = [
                self.tokenizer.decode_ids(predicted_tokens[i])
                for i in range(len(predicted_tokens))
            ]
        return predicted_words, predicted_tokens

    def transcribe(self, waveform):
        # Fake a batch:
        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        predicted_words, predicted_tokens = self.transcribe_batch(
            batch, rel_length
        )
        return predicted_words[0]
