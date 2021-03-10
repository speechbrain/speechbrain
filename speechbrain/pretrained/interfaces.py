"""Defines interfaces for simple inference with pretrained models

Authors:
 * Aku Rouhe 2021
 * Peter Plantinga 2021
 * Loren Lugosch 2020
 * Mirco Ravanelli 2020
 * Titouan Parcollet 2021
"""
import torch
import torchaudio
from types import SimpleNamespace
from torch.nn import SyncBatchNorm
from torch.nn import DataParallel as DP
from hyperpyyaml import load_hyperpyyaml
from speechbrain.pretrained.fetching import fetch
from speechbrain.dataio.preprocess import AudioNormalizer
from torch.nn.parallel import DistributedDataParallel as DDP
from speechbrain.utils.distributed import run_on_main


class Pretrained:
    """Takes a trained model and makes predictions on new data.

    This is a base class which handles some common boilerplate.
    It intentionally has an interface similar to ``Brain`` - these base
    classes handle similar things.

    Subclasses of Pretrained should implement the actual logic of how
    the pretrained system runs, and add methods with descriptive names
    (e.g. transcribe_file() for ASR).

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
    run_opts : dict
        Options parsed from command line. See ``speechbrain.parse_arguments()``.
        List that are supported here:
         * device
         * data_parallel_count
         * data_parallel_backend
         * distributed_launch
         * distributed_backend
         * jit_module_keys
    freeze_params : bool
        To freeze (requires_grad=False) parameters or not. Normally in inference
        you want to freeze the params. Also calls .eval() on all modules.
    """

    def __init__(
        self, modules=None, hparams=None, run_opts=None, freeze_params=True
    ):

        # Arguments passed via the run opts dictionary. Set a limited
        # number of these, since some don't apply to inference.
        run_opt_defaults = {
            "device": "cpu",
            "data_parallel_count": -1,
            "data_parallel_backend": False,
            "distributed_launch": False,
            "distributed_backend": "nccl",
            "jit_module_keys": None,
        }
        for arg, default in run_opt_defaults.items():
            if run_opts is not None and arg in run_opts:
                setattr(self, arg, run_opts[arg])
            else:
                # If any arg from run_opt_defaults exist in hparams and
                # not in command line args "run_opts"
                if hparams is not None and arg in hparams:
                    setattr(self, arg, hparams[arg])
                else:
                    setattr(self, arg, default)

        # Put modules on the right device, accessible with dot notation
        self.modules = torch.nn.ModuleDict(modules).to(self.device)

        # Make hyperparams available with dot notation too
        if hparams is not None:
            self.hparams = SimpleNamespace(**hparams)

        # Prepare modules for computation, e.g. jit
        self._prepare_modules(freeze_params)

    def _prepare_modules(self, freeze_params):
        """Prepare modules for computation, e.g. jit.

        Arguments
        ---------
        freeze_params : bool
            Whether to freeze the parameters and call ``eval()``.
        """

        # Make jit-able
        self._compile_jit()
        self._wrap_distributed()

        # If we don't want to backprop, freeze the pretrained parameters
        if freeze_params:
            self.modules.eval()
            for p in self.modules.parameters():
                p.requires_grad = False

    def _compile_jit(self):
        """Compile requested modules with ``torch.jit.script``."""
        if self.jit_module_keys is None:
            return

        for name in self.jit_module_keys:
            if name not in self.modules:
                raise ValueError(
                    "module " + name + " cannot be jit compiled because "
                    "it is not defined in your hparams file."
                )
            module = torch.jit.script(self.modules[name])
            self.modules[name] = module.to(self.device)

    def _wrap_distributed(self):
        """Wrap modules with distributed wrapper when requested."""
        if not self.distributed_launch and not self.data_parallel_backend:
            return
        elif self.distributed_launch:
            for name, module in self.modules.items():
                if any(p.requires_grad for p in module.parameters()):
                    # for ddp, all module must run on same GPU
                    module = SyncBatchNorm.convert_sync_batchnorm(module)
                    module = DDP(module, device_ids=[self.device])
                    self.modules[name] = module
        else:
            # data_parallel_backend
            for name, module in self.modules.items():
                if any(p.requires_grad for p in module.parameters()):
                    # if distributed_count = -1 then use all gpus
                    # otherwise, specify the set of gpu to use
                    if self.data_parallel_count == -1:
                        module = DP(module)
                    else:
                        module = DP(
                            module,
                            [i for i in range(self.data_parallel_count)],
                        )
                    self.modules[name] = module

    @classmethod
    def from_hparams(
        cls,
        source,
        hparams_file="hyperparams.yaml",
        overrides={},
        savedir="./data",
        **kwargs,
    ):
        """Fetch and load based from outside source based on HyperPyYAML file

        The source can be a location on the filesystem or online/huggingface

        The hyperparams file should contain a "modules" key, which is a
        dictionary of torch modules used for computation.

        The hyperparams file should contain a "pretrainer" key, which is a
        speechbrain.utils.parameter_transfer.Pretrainer

        Arguments
        ---------
        source : str
            The location to use for finding the model. See
            ``speechbrain.pretrained.fetching.fetch`` for details.
        hparams_file : str
            The name of the hyperparameters file to use for constructing
            the modules necessary for inference. Must contain two keys:
            "modules" and "pretrainer", as described.
        overrides : dict
            Any changes to make to the hparams file when it is loaded.
        savedir : str or Path
            Where to put the pretraining material.
        """
        hparams_local_path = fetch(hparams_file, source, savedir)

        # Load the modules:
        with open(hparams_local_path) as fin:
            hparams = load_hyperpyyaml(fin, overrides)

        # Pretraining:
        pretrainer = hparams["pretrainer"]
        pretrainer.set_collect_in(savedir)
        # For distributed setups, have this here:
        run_on_main(pretrainer.collect_files, kwargs={"default_source": source})
        # Load on the CPU. Later the params can be moved elsewhere by specifying
        # run_opts={"device": ...}
        pretrainer.load_collected(device="cpu")

        # Now return the system
        return cls(hparams["modules"], hparams, **kwargs)


class EncoderDecoderASR(Pretrained):
    """A ready-to-use Encoder-Decoder ASR model

    The class can be used either to run only the encoder (encode()) to extract
    features or to run the entire encoder-decoder model
    (transcribe()) to transcribe speech.

    Example
    -------
    >>> from speechbrain.pretrained import EncoderDecoderASR
    >>> asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech")
    >>> asr_model.transcribe_file("samples/audio_samples/example2.flac'")
    ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: How to integrate this better?
        if not hasattr(self, "normalizer"):
            self.normalizer = AudioNormalizer()

        self.tokenizer = self.hparams.tokenizer

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
        """Transcribes the given audiofile into a sequence of words.

        Arguments
        ---------
        path : str
            Path to audio file which to transcribe.

        Returns
        -------
        str
            The audiofile transcription produced by this ASR system.
        """
        waveform = self.load_audio(path)
        # Fake a batch:
        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        predicted_words, predicted_tokens = self.transcribe_batch(
            batch, rel_length
        )
        return predicted_words[0]

    def encode_batch(self, wavs, wav_lens):
        """Encodes the input audio into a sequence of hidden states

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = EncoderDecoderASR.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

        Arguments
        ---------
        wavs : torch.tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model.
        wav_lens : torch.tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        tensor
            The encoded batch

        """
        wavs = wavs.float()
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        feats = self.modules.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)
        encoder_out = self.modules.asr_encoder(feats)
        return encoder_out

    def transcribe_batch(self, wavs, wav_lens):
        """Transcribes the input audio into a sequence of words

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = EncoderDecoderASR.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

        Arguments
        ---------
        wavs : torch.tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model.
        wav_lens : torch.tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        list
            Each waveform in the batch transcribed.
        tensor
            Each predicted token id.
        """
        with torch.no_grad():
            wav_lens = wav_lens.to(self.device)
            encoder_out = self.encode_batch(wavs, wav_lens)
            predicted_tokens, scores = self.modules.beam_searcher(
                encoder_out, wav_lens
            )
            predicted_words = [
                self.tokenizer.decode_ids(token_seq)
                for token_seq in predicted_tokens
            ]
        return predicted_words, predicted_tokens


class SpeakerRecognition(Pretrained):
    """A ready-to-use model for speaker recognition. It can be used to
    compute the speaker embeddings as well.

    The class can be used either to run only the encoder (encode_batch()) to
    extract embeddings or to run the entire verification system
    (verify()) to check speaker identifies.
    ```

    Example
    -------
    >>> import torchaudio
    >>> from speechbrain.pretrained import SpeakerRecognition
    >>> # Model is downloaded from the speechbrain HuggingFace repo
    >>> verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

    >>> # Compute embeddings
    >>> signal, fs =torchaudio.load('samples/audio_samples/example1.wav')
    >>> embeddings = verification.encode(signal)

    >>> # Speaker Verification
    >>> signal2, fs = torchaudio.load('samples/audio_samples/example2.flac')
    >>> score, prediction = verification.verify(signal, signal2)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not hasattr(self, "normalizer"):
            self.normalizer = AudioNormalizer()

        self.similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    def encode(self, wavs, wav_lens=None, normalize=False):
        """Encodes the input audio into a single vector embedding.

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = TransformerASR.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

        Arguments
        ---------
        wavs : torch.tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model. Make sure the sample rate is fs=16000 Hz.
        wav_lens : torch.tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.
        normalize : bool
            If True, it normalizes the embeddings with the statistics
            contained in mean_var_norm_emb.

        Returns
        -------
        tensor
            The encoded batch

        """
        # Manage single waveforms in input
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)

        # Storing waveform in the specified device
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        wavs = wavs.float()

        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, wav_lens)
        embeddings = self.modules.embedding_model(feats, wav_lens)
        if normalize:
            embeddings = self.modules.mean_var_norm_emb(
                embeddings, torch.ones(embeddings.shape[0], device=self.device)
            )
        return embeddings

    def verify(
        self, wavs1, wavs2, wav1_lens=None, wav2_lens=None, threshold=0.25
    ):
        """Performs speaker verification with cosine distance.
        It returns the score and the decision (0 different speakers,
        1 same speakers).

        Arguments
        ---------
        wavs1 : Torch.Tensor
                Tensor containing the speech waveform1 (batch, time).
                Make sure the sample rate is fs=16000 Hz.
        wavs2 : Torch.Tensor
                Tensor containing the speech waveform2 (batch, time).
                Make sure the sample rate is fs=16000 Hz.
        wav1_lens: Torch.Tensor
                Tensor containing the relative length for each sentence
                in the length (e.g., [0.8 0.6 1.0])
        wav2_lens: Torch.Tensor
                Tensor containing the relative length for each sentence
                in the length (e.g., [0.8 0.6 1.0])
        threshold: Float
                Threshold applied to the cosine distance to decide if the
                speaker is different (0) or the same (1).

        Returns
        -------
        score
            The score associated to the binary verification output
            (cosine distance).
        prediction
            The prediction is 1 if the two signals in input are from the same
            speaker and 0 otherwise.
        """
        emb1 = self.encode(wavs1, wav1_lens, normalize=True)
        emb2 = self.encode(wavs2, wav2_lens, normalize=True)
        score = self.similarity(emb1, emb2)
        return score, score > threshold


class separator(Pretrained):
    """A "ready-to-use" speech separation model for 2 or 3 speaker mixtures.
    This pretrained model is based on the WSJ0Mix recipe.

    The SepFormer system achieves an SI-SNR=22.5dB on WSJ0-2Mix test set.

    Example
    -------
    >>> from speechbrain.pretrained import separator
    >>> model = separator.from_hparams(source="speechbrain/sepformer-wsj02mix")
    >>> mix = torch.randn(1, 400)
    >>> est_sources = model.separate(mix)
    >>> print(est_sources.shape)
    torch.Size([1, 400, 2])
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def separate(self, mix):
        import torch.nn.functional as F

        # Separation
        mix_w = self.modules.encoder(mix)
        est_mask = self.modules.masknet(mix_w)
        mix_w = torch.stack([mix_w] * self.hparams.num_spks)
        sep_h = mix_w * est_mask

        # Decoding
        est_source = torch.cat(
            [
                self.modules.decoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.hparams.num_spks)
            ],
            dim=-1,
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]
        return est_source
