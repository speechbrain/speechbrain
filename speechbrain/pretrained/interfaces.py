"""Defines interfaces for simple inference with pretrained models

Authors:
 * Aku Rouhe 2021
 * Peter Plantinga 2021
 * Loren Lugosch 2020
 * Mirco Ravanelli 2020
 * Titouan Parcollet 2021
 * Abdel Heba 2021
 * Andreas Nautsch 2022
"""
import logging
import hashlib
import sys
import speechbrain
import torch
import torchaudio
import sentencepiece
from types import SimpleNamespace
from torch.nn import SyncBatchNorm
from torch.nn import DataParallel as DP
from hyperpyyaml import load_hyperpyyaml
from speechbrain.pretrained.fetching import fetch
from speechbrain.dataio.preprocess import AudioNormalizer
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from speechbrain.utils.data_utils import split_path
from speechbrain.utils.distributed import run_on_main
from speechbrain.dataio.batch import PaddedBatch, PaddedData
from speechbrain.utils.data_pipeline import DataPipeline
from speechbrain.utils.callchains import lengths_arg_exists
from speechbrain.utils.superpowers import import_from_path

logger = logging.getLogger(__name__)


def foreign_class(
    source,
    hparams_file="hyperparams.yaml",
    pymodule_file="custom.py",
    classname="CustomInterface",
    overrides={},
    savedir=None,
    use_auth_token=False,
    download_only=False,
    **kwargs,
):
    """Fetch and load an interface from an outside source

    The source can be a location on the filesystem or online/huggingface

    The pymodule file should contain a class with the given classname. An
    instance of that class is returned. The idea is to have a custom Pretrained
    subclass in the file. The pymodule file is also added to the python path
    before the Hyperparams YAML file is loaded, so it can contain any custom
    implementations that are needed.

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
    pymodule_file : str
        The name of the Python file that should be fetched.
    classname : str
        The name of the Class, of which an instance is created and returned
    overrides : dict
        Any changes to make to the hparams file when it is loaded.
    savedir : str or Path
        Where to put the pretraining material. If not given, will use
        ./pretrained_models/<class-name>-hash(source).
    use_auth_token : bool (default: False)
        If true Hugginface's auth_token will be used to load private models from the HuggingFace Hub,
        default is False because the majority of models are public.
    download_only : bool (default: False)
        If true, class and instance creation is skipped.

    Returns
    -------
    object
        An instance of a class with the given classname from the given pymodule file.
    """
    if savedir is None:
        savedir = f"./pretrained_models/{classname}-{hashlib.md5(source.encode('UTF-8', errors='replace')).hexdigest()}"
    hparams_local_path = fetch(
        filename=hparams_file,
        source=source,
        savedir=savedir,
        overwrite=False,
        save_filename=None,
        use_auth_token=use_auth_token,
        revision=None,
    )
    pymodule_local_path = fetch(
        filename=pymodule_file,
        source=source,
        savedir=savedir,
        overwrite=False,
        save_filename=None,
        use_auth_token=use_auth_token,
        revision=None,
    )
    sys.path.append(str(pymodule_local_path.parent))

    # Load the modules:
    with open(hparams_local_path) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Pretraining:
    pretrainer = hparams["pretrainer"]
    pretrainer.set_collect_in(savedir)
    # For distributed setups, have this here:
    run_on_main(pretrainer.collect_files, kwargs={"default_source": source})
    # Load on the CPU. Later the params can be moved elsewhere by specifying
    if not download_only:
        # run_opts={"device": ...}
        pretrainer.load_collected(device="cpu")

        # Import class and create instance
        module = import_from_path(pymodule_local_path)
        cls = getattr(module, classname)
        return cls(modules=hparams["modules"], hparams=hparams, **kwargs)


class Pretrained(torch.nn.Module):
    """Takes a trained model and makes predictions on new data.

    This is a base class which handles some common boilerplate.
    It intentionally has an interface similar to ``Brain`` - these base
    classes handle similar things.

    Subclasses of Pretrained should implement the actual logic of how
    the pretrained system runs, and add methods with descriptive names
    (e.g. transcribe_file() for ASR).

    Pretrained is a torch.nn.Module so that methods like .to() or .eval() can
    work. Subclasses should provide a suitable forward() implementation: by
    convention, it should be a method that takes a batch of audio signals and
    runs the full model (as applicable).

    Arguments
    ---------
    modules : dict of str:torch.nn.Module pairs
        The Torch modules that make up the learned system. These can be treated
        in special ways (put on the right device, frozen, etc.). These are available
        as attributes under ``self.mods``, like self.mods.model(x)
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

    HPARAMS_NEEDED = []
    MODULES_NEEDED = []

    def __init__(
        self, modules=None, hparams=None, run_opts=None, freeze_params=True
    ):
        super().__init__()
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
        self.mods = torch.nn.ModuleDict(modules)
        for module in self.mods.values():
            if module is not None:
                module.to(self.device)

        # Check MODULES_NEEDED and HPARAMS_NEEDED and
        # make hyperparams available with dot notation
        if self.HPARAMS_NEEDED and hparams is None:
            raise ValueError("Need to provide hparams dict.")
        if hparams is not None:
            # Also first check that all required params are found:
            for hp in self.HPARAMS_NEEDED:
                if hp not in hparams:
                    raise ValueError(f"Need hparams['{hp}']")
            self.hparams = SimpleNamespace(**hparams)

        # Prepare modules for computation, e.g. jit
        self._prepare_modules(freeze_params)

        # Audio normalization
        self.audio_normalizer = hparams.get(
            "audio_normalizer", AudioNormalizer()
        )

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
            self.mods.eval()
            for p in self.mods.parameters():
                p.requires_grad = False

    def load_audio(self, path, savedir="."):
        """Load an audio file with this model's input spec

        When using a speech model, it is important to use the same type of data,
        as was used to train the model. This means for example using the same
        sampling rate and number of channels. It is, however, possible to
        convert a file from a higher sampling rate to a lower one (downsampling).
        Similarly, it is simple to downmix a stereo file to mono.
        The path can be a local path, a web url, or a link to a huggingface repo.
        """
        source, fl = split_path(path)
        path = fetch(fl, source=source, savedir=savedir)
        signal, sr = torchaudio.load(str(path), channels_first=False)
        return self.audio_normalizer(signal, sr)

    def _compile_jit(self):
        """Compile requested modules with ``torch.jit.script``."""
        if self.jit_module_keys is None:
            return

        for name in self.jit_module_keys:
            if name not in self.mods:
                raise ValueError(
                    "module " + name + " cannot be jit compiled because "
                    "it is not defined in your hparams file."
                )
            module = torch.jit.script(self.mods[name])
            self.mods[name] = module.to(self.device)

    def _wrap_distributed(self):
        """Wrap modules with distributed wrapper when requested."""
        if not self.distributed_launch and not self.data_parallel_backend:
            return
        elif self.distributed_launch:
            for name, module in self.mods.items():
                if any(p.requires_grad for p in module.parameters()):
                    # for ddp, all module must run on same GPU
                    module = SyncBatchNorm.convert_sync_batchnorm(module)
                    module = DDP(module, device_ids=[self.device])
                    self.mods[name] = module
        else:
            # data_parallel_backend
            for name, module in self.mods.items():
                if any(p.requires_grad for p in module.parameters()):
                    # if distributed_count = -1 then use all gpus
                    # otherwise, specify the set of gpu to use
                    if self.data_parallel_count == -1:
                        module = DP(module)
                    else:
                        module = DP(
                            module, [i for i in range(self.data_parallel_count)]
                        )
                    self.mods[name] = module

    @classmethod
    def from_hparams(
        cls,
        source,
        hparams_file="hyperparams.yaml",
        pymodule_file="custom.py",
        overrides={},
        savedir=None,
        use_auth_token=False,
        revision=None,
        download_only=False,
        **kwargs,
    ):
        """Fetch and load based from outside source based on HyperPyYAML file

        The source can be a location on the filesystem or online/huggingface

        You can use the pymodule_file to include any custom implementations
        that are needed: if that file exists, then its location is added to
        sys.path before Hyperparams YAML is loaded, so it can be referenced
        in the YAML.

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
        pymodule_file : str
            A Python file can be fetched. This allows any custom
            implementations to be included. The file's location is added to
            sys.path before the hyperparams YAML file is loaded, so it can be
            referenced in YAML.
            This is optional, but has a default: "custom.py". If the default
            file is not found, this is simply ignored, but if you give a
            different filename, then this will raise in case the file is not
            found.
        overrides : dict
            Any changes to make to the hparams file when it is loaded.
        savedir : str or Path
            Where to put the pretraining material. If not given, will use
            ./pretrained_models/<class-name>-hash(source).
        use_auth_token : bool (default: False)
            If true Hugginface's auth_token will be used to load private models from the HuggingFace Hub,
            default is False because the majority of models are public.
        revision : str
            The model revision corresponding to the HuggingFace Hub model revision.
            This is particularly useful if you wish to pin your code to a particular
            version of a model hosted at HuggingFace.
        download_only : bool (default: False)
            If true, class and instance creation is skipped.
        """
        if savedir is None:
            clsname = cls.__name__
            savedir = f"./pretrained_models/{clsname}-{hashlib.md5(source.encode('UTF-8', errors='replace')).hexdigest()}"
        hparams_local_path = fetch(
            filename=hparams_file,
            source=source,
            savedir=savedir,
            overwrite=False,
            save_filename=None,
            use_auth_token=use_auth_token,
            revision=revision,
        )
        try:
            pymodule_local_path = fetch(
                filename=pymodule_file,
                source=source,
                savedir=savedir,
                overwrite=False,
                save_filename=None,
                use_auth_token=use_auth_token,
                revision=revision,
            )
            sys.path.append(str(pymodule_local_path.parent))
        except ValueError:
            if pymodule_file == "custom.py":
                # The optional custom Python module file did not exist
                # and had the default name
                pass
            else:
                # Custom Python module file not found, but some other
                # filename than the default was given.
                raise

        # Load the modules:
        with open(hparams_local_path) as fin:
            hparams = load_hyperpyyaml(fin, overrides)

        # Pretraining:
        pretrainer = hparams["pretrainer"]
        pretrainer.set_collect_in(savedir)
        # For distributed setups, have this here:
        run_on_main(pretrainer.collect_files, kwargs={"default_source": source})
        # Load on the CPU. Later the params can be moved elsewhere by specifying
        if not download_only:
            # run_opts={"device": ...}
            pretrainer.load_collected(device="cpu")

            # Now return the system
            return cls(hparams["modules"], hparams, **kwargs)


class EndToEndSLU(Pretrained):
    """An end-to-end SLU model.

    The class can be used either to run only the encoder (encode()) to extract
    features or to run the entire model (decode()) to map the speech to its semantics.

    Example
    -------
    >>> from speechbrain.pretrained import EndToEndSLU
    >>> tmpdir = getfixture("tmpdir")
    >>> slu_model = EndToEndSLU.from_hparams(
    ...     source="speechbrain/slu-timers-and-such-direct-librispeech-asr",
    ...     savedir=tmpdir,
    ... )
    >>> slu_model.decode_file("tests/samples/single-mic/example6.wav")
    "{'intent': 'SimpleMath', 'slots': {'number1': 37.67, 'number2': 75.7, 'op': ' minus '}}"
    """

    HPARAMS_NEEDED = ["tokenizer", "asr_model_source"]
    MODULES_NEEDED = ["slu_enc", "beam_searcher"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = self.hparams.tokenizer
        self.asr_model = EncoderDecoderASR.from_hparams(
            source=self.hparams.asr_model_source,
            run_opts={"device": self.device},
        )

    def decode_file(self, path):
        """Maps the given audio file to a string representing the
        semantic dictionary for the utterance.

        Arguments
        ---------
        path : str
            Path to audio file to decode.

        Returns
        -------
        str
            The predicted semantics.
        """
        waveform = self.load_audio(path)
        waveform = waveform.to(self.device)
        # Fake a batch:
        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        predicted_words, predicted_tokens = self.decode_batch(batch, rel_length)
        return predicted_words[0]

    def encode_batch(self, wavs, wav_lens):
        """Encodes the input audio into a sequence of hidden states

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        torch.Tensor
            The encoded batch
        """
        wavs = wavs.float()
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        ASR_encoder_out = self.asr_model.encode_batch(wavs.detach(), wav_lens)
        encoder_out = self.mods.slu_enc(ASR_encoder_out)
        return encoder_out

    def decode_batch(self, wavs, wav_lens):
        """Maps the input audio to its semantics

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        list
            Each waveform in the batch decoded.
        tensor
            Each predicted token id.
        """
        with torch.no_grad():
            wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
            encoder_out = self.encode_batch(wavs, wav_lens)
            predicted_tokens, scores = self.mods.beam_searcher(
                encoder_out, wav_lens
            )
            predicted_words = [
                self.tokenizer.decode_ids(token_seq)
                for token_seq in predicted_tokens
            ]
        return predicted_words, predicted_tokens

    def forward(self, wavs, wav_lens):
        """Runs full decoding - note: no gradients through decoding"""
        return self.decode_batch(wavs, wav_lens)


class EncoderDecoderASR(Pretrained):
    """A ready-to-use Encoder-Decoder ASR model

    The class can be used either to run only the encoder (encode()) to extract
    features or to run the entire encoder-decoder model
    (transcribe()) to transcribe speech. The given YAML must contain the fields
    specified in the *_NEEDED[] lists.

    Example
    -------
    >>> from speechbrain.pretrained import EncoderDecoderASR
    >>> tmpdir = getfixture("tmpdir")
    >>> asr_model = EncoderDecoderASR.from_hparams(
    ...     source="speechbrain/asr-crdnn-rnnlm-librispeech",
    ...     savedir=tmpdir,
    ... )
    >>> asr_model.transcribe_file("tests/samples/single-mic/example2.flac")
    "MY FATHER HAS REVEALED THE CULPRIT'S NAME"
    """

    HPARAMS_NEEDED = ["tokenizer"]
    MODULES_NEEDED = ["encoder", "decoder"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = self.hparams.tokenizer

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
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        torch.Tensor
            The encoded batch
        """
        wavs = wavs.float()
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        encoder_out = self.mods.encoder(wavs, wav_lens)
        return encoder_out

    def transcribe_batch(self, wavs, wav_lens):
        """Transcribes the input audio into a sequence of words

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = EncoderDecoderASR.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model.
        wav_lens : torch.Tensor
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
            predicted_tokens, scores = self.mods.decoder(encoder_out, wav_lens)
            predicted_words = [
                self.tokenizer.decode_ids(token_seq)
                for token_seq in predicted_tokens
            ]
        return predicted_words, predicted_tokens

    def forward(self, wavs, wav_lens):
        """Runs full transcription - note: no gradients through decoding"""
        return self.transcribe_batch(wavs, wav_lens)


class WaveformEncoder(Pretrained):
    """A ready-to-use waveformEncoder model

    It can be used to wrap different embedding models such as SSL ones (wav2vec2)
    or speaker ones (Xvector) etc. Two functions are available: encode_batch and
    encode_file. They can be used to obtain the embeddings directly from an audio
    file or from a batch of audio tensors respectively.

    The given YAML must contain the fields specified in the *_NEEDED[] lists.

    Example
    -------
    >>> from speechbrain.pretrained import WaveformEncoder
    >>> tmpdir = getfixture("tmpdir")
    >>> ssl_model = WaveformEncoder.from_hparams(
    ...     source="speechbrain/ssl-wav2vec2-base-libri",
    ...     savedir=tmpdir,
    ... ) # doctest: +SKIP
    >>> ssl_model.encode_file("samples/audio_samples/example_fr.wav") # doctest: +SKIP
    """

    MODULES_NEEDED = ["encoder"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode_file(self, path):
        """Encode the given audiofile into a sequence of embeddings.

        Arguments
        ---------
        path : str
            Path to audio file which to encode.

        Returns
        -------
        torch.Tensor
            The audiofile embeddings produced by this system.
        """
        waveform = self.load_audio(path)
        # Fake a batch:
        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        results = self.encode_batch(batch, rel_length)
        return results["embeddings"]

    def encode_batch(self, wavs, wav_lens):
        """Encodes the input audio into a sequence of hidden states

        The waveforms should already be in the model's desired format.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        torch.Tensor
            The encoded batch
        """
        wavs = wavs.float()
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        encoder_out = self.mods.encoder(wavs, wav_lens)
        return encoder_out

    def forward(self, wavs, wav_lens):
        """Runs the encoder"""
        return self.encode_batch(wavs, wav_lens)


class EncoderASR(Pretrained):
    """A ready-to-use Encoder ASR model

    The class can be used either to run only the encoder (encode()) to extract
    features or to run the entire encoder + decoder function model
    (transcribe()) to transcribe speech. The given YAML must contain the fields
    specified in the *_NEEDED[] lists.

    Example
    -------
    >>> from speechbrain.pretrained import EncoderASR
    >>> tmpdir = getfixture("tmpdir")
    >>> asr_model = EncoderASR.from_hparams(
    ...     source="speechbrain/asr-wav2vec2-commonvoice-fr",
    ...     savedir=tmpdir,
    ... ) # doctest: +SKIP
    >>> asr_model.transcribe_file("samples/audio_samples/example_fr.wav") # doctest: +SKIP
    """

    HPARAMS_NEEDED = ["tokenizer", "decoding_function"]
    MODULES_NEEDED = ["encoder"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tokenizer = self.hparams.tokenizer
        self.decoding_function = self.hparams.decoding_function

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
        return str(predicted_words[0])

    def encode_batch(self, wavs, wav_lens):
        """Encodes the input audio into a sequence of hidden states

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = EncoderASR.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        torch.Tensor
            The encoded batch
        """
        wavs = wavs.float()
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        encoder_out = self.mods.encoder(wavs, wav_lens)
        return encoder_out

    def transcribe_batch(self, wavs, wav_lens):
        """Transcribes the input audio into a sequence of words

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = EncoderASR.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model.
        wav_lens : torch.Tensor
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
            predictions = self.decoding_function(encoder_out, wav_lens)
            if isinstance(
                self.tokenizer, speechbrain.dataio.encoder.CTCTextEncoder
            ):
                predicted_words = [
                    "".join(self.tokenizer.decode_ndim(token_seq))
                    for token_seq in predictions
                ]
            elif isinstance(
                self.tokenizer, sentencepiece.SentencePieceProcessor
            ):
                predicted_words = [
                    self.tokenizer.decode_ids(token_seq)
                    for token_seq in predictions
                ]
            else:
                sys.exit(
                    "The tokenizer must be sentencepiece or CTCTextEncoder"
                )

        return predicted_words, predictions

    def forward(self, wavs, wav_lens):
        """Runs the encoder"""
        return self.encode_batch(wavs, wav_lens)


class EncoderClassifier(Pretrained):
    """A ready-to-use class for utterance-level classification (e.g, speaker-id,
    language-id, emotion recognition, keyword spotting, etc).

    The class assumes that an encoder called "embedding_model" and a model
    called "classifier" are defined in the yaml file. If you want to
    convert the predicted index into a corresponding text label, please
    provide the path of the label_encoder in a variable called 'lab_encoder_file'
    within the yaml.

    The class can be used either to run only the encoder (encode_batch()) to
    extract embeddings or to run a classification step (classify_batch()).
    ```

    Example
    -------
    >>> import torchaudio
    >>> from speechbrain.pretrained import EncoderClassifier
    >>> # Model is downloaded from the speechbrain HuggingFace repo
    >>> tmpdir = getfixture("tmpdir")
    >>> classifier = EncoderClassifier.from_hparams(
    ...     source="speechbrain/spkrec-ecapa-voxceleb",
    ...     savedir=tmpdir,
    ... )

    >>> # Compute embeddings
    >>> signal, fs = torchaudio.load("tests/samples/single-mic/example1.wav")
    >>> embeddings = classifier.encode_batch(signal)

    >>> # Classification
    >>> prediction = classifier.classify_batch(signal)
    """

    MODULES_NEEDED = [
        "compute_features",
        "mean_var_norm",
        "embedding_model",
        "classifier",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode_batch(self, wavs, wav_lens=None, normalize=False):
        """Encodes the input audio into a single vector embedding.

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = <this>.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model. Make sure the sample rate is fs=16000 Hz.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.
        normalize : bool
            If True, it normalizes the embeddings with the statistics
            contained in mean_var_norm_emb.

        Returns
        -------
        torch.Tensor
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

        # Computing features and embeddings
        feats = self.mods.compute_features(wavs)
        feats = self.mods.mean_var_norm(feats, wav_lens)
        embeddings = self.mods.embedding_model(feats, wav_lens)
        if normalize:
            embeddings = self.hparams.mean_var_norm_emb(
                embeddings, torch.ones(embeddings.shape[0], device=self.device)
            )
        return embeddings

    def classify_batch(self, wavs, wav_lens=None):
        """Performs classification on the top of the encoded features.

        It returns the posterior probabilities, the index and, if the label
        encoder is specified it also the text label.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model. Make sure the sample rate is fs=16000 Hz.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        out_prob
            The log posterior probabilities of each class ([batch, N_class])
        score:
            It is the value of the log-posterior for the best class ([batch,])
        index
            The indexes of the best class ([batch,])
        text_lab:
            List with the text labels corresponding to the indexes.
            (label encoder should be provided).
        """
        emb = self.encode_batch(wavs, wav_lens)
        out_prob = self.mods.classifier(emb).squeeze(1)
        score, index = torch.max(out_prob, dim=-1)
        text_lab = self.hparams.label_encoder.decode_torch(index)
        return out_prob, score, index, text_lab

    def classify_file(self, path):
        """Classifies the given audiofile into the given set of labels.

        Arguments
        ---------
        path : str
            Path to audio file to classify.

        Returns
        -------
        out_prob
            The log posterior probabilities of each class ([batch, N_class])
        score:
            It is the value of the log-posterior for the best class ([batch,])
        index
            The indexes of the best class ([batch,])
        text_lab:
            List with the text labels corresponding to the indexes.
            (label encoder should be provided).
        """
        waveform = self.load_audio(path)
        # Fake a batch:
        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        emb = self.encode_batch(batch, rel_length)
        out_prob = self.mods.classifier(emb).squeeze(1)
        score, index = torch.max(out_prob, dim=-1)
        text_lab = self.hparams.label_encoder.decode_torch(index)
        return out_prob, score, index, text_lab

    def forward(self, wavs, wav_lens=None):
        """Runs the classification"""
        return self.classify_batch(wavs, wav_lens)


class SpeakerRecognition(EncoderClassifier):
    """A ready-to-use model for speaker recognition. It can be used to
    perform speaker verification with verify_batch().

    ```
    Example
    -------
    >>> import torchaudio
    >>> from speechbrain.pretrained import SpeakerRecognition
    >>> # Model is downloaded from the speechbrain HuggingFace repo
    >>> tmpdir = getfixture("tmpdir")
    >>> verification = SpeakerRecognition.from_hparams(
    ...     source="speechbrain/spkrec-ecapa-voxceleb",
    ...     savedir=tmpdir,
    ... )

    >>> # Perform verification
    >>> signal, fs = torchaudio.load("tests/samples/single-mic/example1.wav")
    >>> signal2, fs = torchaudio.load("tests/samples/single-mic/example2.flac")
    >>> score, prediction = verification.verify_batch(signal, signal2)
    """

    MODULES_NEEDED = [
        "compute_features",
        "mean_var_norm",
        "embedding_model",
        "mean_var_norm_emb",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    def verify_batch(
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
        emb1 = self.encode_batch(wavs1, wav1_lens, normalize=True)
        emb2 = self.encode_batch(wavs2, wav2_lens, normalize=True)
        score = self.similarity(emb1, emb2)
        return score, score > threshold

    def verify_files(self, path_x, path_y):
        """Speaker verification with cosine distance

        Returns the score and the decision (0 different speakers,
        1 same speakers).

        Returns
        -------
        score
            The score associated to the binary verification output
            (cosine distance).
        prediction
            The prediction is 1 if the two signals in input are from the same
            speaker and 0 otherwise.
        """
        waveform_x = self.load_audio(path_x)
        waveform_y = self.load_audio(path_y)
        # Fake batches:
        batch_x = waveform_x.unsqueeze(0)
        batch_y = waveform_y.unsqueeze(0)
        # Verify:
        score, decision = self.verify_batch(batch_x, batch_y)
        # Squeeze:
        return score[0], decision[0]


class VAD(Pretrained):
    """A ready-to-use class for Voice Activity Detection (VAD) using a
    pre-trained model.

    Example
    -------
    >>> import torchaudio
    >>> from speechbrain.pretrained import VAD
    >>> # Model is downloaded from the speechbrain HuggingFace repo
    >>> tmpdir = getfixture("tmpdir")
    >>> VAD = VAD.from_hparams(
    ...     source="speechbrain/vad-crdnn-libriparty",
    ...     savedir=tmpdir,
    ... )

    >>> # Perform VAD
    >>> boundaries = VAD.get_speech_segments("tests/samples/single-mic/example1.wav")
    """

    HPARAMS_NEEDED = ["sample_rate", "time_resolution", "device"]

    MODULES_NEEDED = ["compute_features", "mean_var_norm", "model"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_resolution = self.hparams.time_resolution
        self.sample_rate = self.hparams.sample_rate
        self.device = self.hparams.device

    def get_speech_prob_file(
        self,
        audio_file,
        large_chunk_size=30,
        small_chunk_size=10,
        overlap_small_chunk=False,
    ):
        """Outputs the frame-level speech probability of the input audio file
        using the neural model specified in the hparam file. To make this code
        both parallelizable and scalable to long sequences, it uses a
        double-windowing approach.  First, we sequentially read non-overlapping
        large chunks of the input signal.  We then split the large chunks into
        smaller chunks and we process them in parallel.

        Arguments
        ---------
        audio_file: path
            Path of the audio file containing the recording. The file is read
            with torchaudio.
        large_chunk_size: float
            Size (in seconds) of the large chunks that are read sequentially
            from the input audio file.
        small_chunk_size:
            Size (in seconds) of the small chunks extracted from the large ones.
            The audio signal is processed in parallel within the small chunks.
            Note that large_chunk_size/small_chunk_size must be an integer.
        overlap_small_chunk: bool
            True, creates overlapped small chunks. The probabilities of the
            overlapped chunks are combined using hamming windows.

        Returns
        -------
        prob_vad: torch.Tensor
            Tensor containing the frame-level speech probabilities for the
            input audio file.
        """
        # Getting the total size of the input file
        sample_rate, audio_len = self._get_audio_info(audio_file)

        if sample_rate != self.sample_rate:
            raise ValueError(
                "The detected sample rate is different from that set in the hparam file"
            )

        # Computing the length (in samples) of the large and small chunks
        long_chunk_len = int(sample_rate * large_chunk_size)
        small_chunk_len = int(sample_rate * small_chunk_size)

        # Setting the step size of the small chunk (50% overlapping windows are supported)
        small_chunk_step = small_chunk_size
        if overlap_small_chunk:
            small_chunk_step = small_chunk_size / 2

        # Computing the length (in sample) of the small_chunk step size
        small_chunk_len_step = int(sample_rate * small_chunk_step)

        # Loop over big chunks
        prob_chunks = []
        last_chunk = False
        begin_sample = 0
        while True:

            # Reading the big chunk
            large_chunk, fs = torchaudio.load(
                audio_file, frame_offset=begin_sample, num_frames=long_chunk_len
            )
            large_chunk = large_chunk.to(self.device)

            # Manage padding of the last small chunk
            if last_chunk or large_chunk.shape[-1] < small_chunk_len:
                padding = torch.zeros(
                    1, small_chunk_len, device=large_chunk.device
                )
                large_chunk = torch.cat([large_chunk, padding], dim=1)

            # Splitting the big chunk into smaller (overlapped) ones
            small_chunks = torch.nn.functional.unfold(
                large_chunk.unsqueeze(1).unsqueeze(2),
                kernel_size=(1, small_chunk_len),
                stride=(1, small_chunk_len_step),
            )
            small_chunks = small_chunks.squeeze(0).transpose(0, 1)

            # Getting (in parallel) the frame-level speech probabilities
            small_chunks_prob = self.get_speech_prob_chunk(small_chunks)
            small_chunks_prob = small_chunks_prob[:, :-1, :]

            # Manage overlapping chunks
            if overlap_small_chunk:
                small_chunks_prob = self._manage_overlapped_chunks(
                    small_chunks_prob
                )

            # Prepare for folding
            small_chunks_prob = small_chunks_prob.permute(2, 1, 0)

            # Computing lengths in samples
            out_len = int(
                large_chunk.shape[-1] / (sample_rate * self.time_resolution)
            )
            kernel_len = int(small_chunk_size / self.time_resolution)
            step_len = int(small_chunk_step / self.time_resolution)

            # Folding the frame-level predictions
            small_chunks_prob = torch.nn.functional.fold(
                small_chunks_prob,
                output_size=(1, out_len),
                kernel_size=(1, kernel_len),
                stride=(1, step_len),
            )

            # Appending the frame-level speech probabilities of the large chunk
            small_chunks_prob = small_chunks_prob.squeeze(1).transpose(-1, -2)
            prob_chunks.append(small_chunks_prob)

            # Check stop condition
            if last_chunk:
                break

            # Update counter to process the next big chunk
            begin_sample = begin_sample + long_chunk_len

            # Check if the current chunk is the last one
            if begin_sample + long_chunk_len > audio_len:
                last_chunk = True

        # Converting the list to a tensor
        prob_vad = torch.cat(prob_chunks, dim=1)
        last_elem = int(audio_len / (self.time_resolution * sample_rate))
        prob_vad = prob_vad[:, 0:last_elem, :]

        return prob_vad

    def _manage_overlapped_chunks(self, small_chunks_prob):
        """This support function manages overlapped the case in which the
        small chunks have a 50% overlap."""

        # Weighting the frame-level probabilities with a hamming window
        # reduces uncertainty when overlapping chunks are used.
        hamming_window = torch.hamming_window(
            small_chunks_prob.shape[1], device=self.device
        )

        # First and last chunks require special care
        half_point = int(small_chunks_prob.shape[1] / 2)
        small_chunks_prob[0, half_point:] = small_chunks_prob[
            0, half_point:
        ] * hamming_window[half_point:].unsqueeze(1)
        small_chunks_prob[-1, 0:half_point] = small_chunks_prob[
            -1, 0:half_point
        ] * hamming_window[0:half_point].unsqueeze(1)

        # Applying the window to all the other probabilities
        small_chunks_prob[1:-1] = small_chunks_prob[
            1:-1
        ] * hamming_window.unsqueeze(0).unsqueeze(2)

        return small_chunks_prob

    def get_speech_prob_chunk(self, wavs, wav_lens=None):
        """Outputs the frame-level posterior probability for the input audio chunks
        Outputs close to zero refers to time steps with a low probability of speech
        activity, while outputs closer to one likely contain speech.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model. Make sure the sample rate is fs=16000 Hz.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        torch.Tensor
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

        # Computing features and embeddings
        feats = self.mods.compute_features(wavs)
        feats = self.mods.mean_var_norm(feats, wav_lens)
        outputs = self.mods.cnn(feats)

        outputs = outputs.reshape(
            outputs.shape[0],
            outputs.shape[1],
            outputs.shape[2] * outputs.shape[3],
        )

        outputs, h = self.mods.rnn(outputs)
        outputs = self.mods.dnn(outputs)
        output_prob = torch.sigmoid(outputs)

        return output_prob

    def apply_threshold(
        self, vad_prob, activation_th=0.5, deactivation_th=0.25
    ):
        """Scans the frame-level speech probabilities and applies a threshold
        on them. Speech starts when a value larger than activation_th is
        detected, while it ends when observing a value lower than
        the deactivation_th.

        Arguments
        ---------
        vad_prob: torch.Tensor
            Frame-level speech probabilities.
        activation_th:  float
            Threshold for starting a speech segment.
        deactivation_th: float
            Threshold for ending a speech segment.

        Returns
        -------
        vad_th: torch.Tensor
            Tensor containing 1 for speech regions and 0 for non-speech regions.
       """
        vad_activation = (vad_prob >= activation_th).int()
        vad_deactivation = (vad_prob >= deactivation_th).int()
        vad_th = vad_activation + vad_deactivation

        # Loop over batches and time steps
        for batch in range(vad_th.shape[0]):
            for time_step in range(vad_th.shape[1] - 1):
                if (
                    vad_th[batch, time_step] == 2
                    and vad_th[batch, time_step + 1] == 1
                ):
                    vad_th[batch, time_step + 1] = 2

        vad_th[vad_th == 1] = 0
        vad_th[vad_th == 2] = 1
        return vad_th

    def get_boundaries(self, prob_th, output_value="seconds"):
        """Computes the time boundaries where speech activity is detected.
        It takes in input frame-level binary decisions
        (1 for speech, 0 for non-speech) and outputs the begin/end second
        (or sample) of each detected speech region.

        Arguments
        ---------
        prob_th: torch.Tensor
            Frame-level binary decisions (1 for speech frame, 0 for a
            non-speech one).  The tensor can be obtained from apply_threshold.
        output_value: 'seconds' or 'samples'
            When the option 'seconds' is set, the returned boundaries are in
            seconds, otherwise, it reports them in samples.

        Returns
        -------
        boundaries: torch.Tensor
            Tensor containing the start second (or sample) of speech segments
            in even positions and their corresponding end in odd positions
            (e.g, [1.0, 1.5, 5,.0 6.0] means that we have two speech segment;
             one from 1.0 to 1.5 seconds and another from 5.0 to 6.0 seconds).
       """
        # Shifting frame-levels binary decision by 1
        # This allows detecting changes in speech/non-speech activities
        prob_th_shifted = torch.roll(prob_th, dims=1, shifts=1)
        prob_th_shifted[:, 0, :] = 0
        prob_th = prob_th + prob_th_shifted

        # Needed to first and last time step
        prob_th[:, 0, :] = (prob_th[:, 0, :] >= 1).int()
        prob_th[:, -1, :] = (prob_th[:, -1, :] >= 1).int()

        # Fix edge cases (when a speech starts in the last frames)
        if (prob_th == 1).nonzero().shape[0] % 2 == 1:
            prob_th = torch.cat(
                (prob_th, torch.Tensor([1.0]).unsqueeze(0).unsqueeze(2)), dim=1
            )

        # Where prob_th is 1 there is a change
        indexes = (prob_th == 1).nonzero()[:, 1].reshape(-1, 2)

        # Remove 1 from end samples
        indexes[:, -1] = indexes[:, -1] - 1

        # From indexes to samples
        seconds = (indexes * self.time_resolution).float()
        samples = (self.sample_rate * seconds).round().int()

        if output_value == "seconds":
            boundaries = seconds
        else:
            boundaries = samples
        return boundaries

    def merge_close_segments(self, boundaries, close_th=0.250):
        """Merges segments that are shorter than the given threshold.

        Arguments
        ---------
        boundaries : str
            Tensor containing the speech boundaries. It can be derived using the
            get_boundaries method.
        close_th: float
            If the distance between boundaries is smaller than close_th, the
            segments will be merged.

        Returns
        -------
        new_boundaries
            The new boundaries with the merged segments.
        """

        new_boundaries = []

        # Single segment case
        if boundaries.shape[0] == 0:
            return boundaries

        # Getting beg and end of previous segment
        prev_beg_seg = boundaries[0, 0].float()
        prev_end_seg = boundaries[0, 1].float()

        # Process all the segments
        for i in range(1, boundaries.shape[0]):
            beg_seg = boundaries[i, 0]
            segment_distance = beg_seg - prev_end_seg

            # Merging close segments
            if segment_distance <= close_th:
                prev_end_seg = boundaries[i, 1]

            else:
                # Appending new segments
                new_boundaries.append([prev_beg_seg, prev_end_seg])
                prev_beg_seg = beg_seg
                prev_end_seg = boundaries[i, 1]

        new_boundaries.append([prev_beg_seg, prev_end_seg])
        new_boundaries = torch.FloatTensor(new_boundaries).to(boundaries.device)
        return new_boundaries

    def remove_short_segments(self, boundaries, len_th=0.250):
        """Removes segments that are too short.

        Arguments
        ---------
        boundaries : torch.Tensor
            Tensor containing the speech boundaries. It can be derived using the
            get_boundaries method.
        len_th: float
            If the length of the segment is smaller than close_th, the segments
            will be merged.

        Returns
        -------
        new_boundaries
            The new boundaries without the short segments.
        """
        new_boundaries = []

        # Process the segments
        for i in range(boundaries.shape[0]):
            # Computing segment length
            seg_len = boundaries[i, 1] - boundaries[i, 0]

            # Accept segment only if longer than len_th
            if seg_len > len_th:
                new_boundaries.append([boundaries[i, 0], boundaries[i, 1]])
        new_boundaries = torch.FloatTensor(new_boundaries).to(boundaries.device)

        return new_boundaries

    def save_boundaries(
        self, boundaries, save_path=None, print_boundaries=True, audio_file=None
    ):
        """Saves the boundaries on a file (and/or prints them)  in a readable format.

        Arguments
        ---------
        boundaries: torch.Tensor
            Tensor containing the speech boundaries. It can be derived using the
            get_boundaries method.
        save_path: path
            When to store the text file containing the speech/non-speech intervals.
        print_boundaries: Bool
            Prints the speech/non-speech intervals in the standard outputs.
        audio_file: path
            Path of the audio file containing the recording. The file is read
            with torchaudio. It is used here to detect the length of the
            signal.
        """
        # Create a new file if needed
        if save_path is not None:
            f = open(save_path, mode="w", encoding="utf-8")

        # Getting the total size of the input file
        if audio_file is not None:
            sample_rate, audio_len = self._get_audio_info(audio_file)
            audio_len = audio_len / sample_rate

        # Setting the rights format for second- or sample-based boundaries
        if boundaries.dtype == torch.int:
            value_format = "% i"
        else:
            value_format = "% .2f "

        # Printing speech and non-speech intervals
        last_end = 0
        cnt_seg = 0
        for i in range(boundaries.shape[0]):
            begin_value = boundaries[i, 0]
            end_value = boundaries[i, 1]

            if last_end != begin_value:
                cnt_seg = cnt_seg + 1
                print_str = (
                    "segment_%03d " + value_format + value_format + "NON_SPEECH"
                )
                if print_boundaries:
                    print(print_str % (cnt_seg, last_end, begin_value))
                if save_path is not None:
                    f.write(print_str % (cnt_seg, last_end, begin_value) + "\n")

            cnt_seg = cnt_seg + 1
            print_str = "segment_%03d " + value_format + value_format + "SPEECH"
            if print_boundaries:
                print(print_str % (cnt_seg, begin_value, end_value))
            if save_path is not None:
                f.write(print_str % (cnt_seg, begin_value, end_value) + "\n")

            last_end = end_value

        # Managing last segment
        if audio_file is not None:
            if last_end < audio_len:
                cnt_seg = cnt_seg + 1
                print_str = (
                    "segment_%03d " + value_format + value_format + "NON_SPEECH"
                )
                if print_boundaries:
                    print(print_str % (cnt_seg, end_value, audio_len))
                if save_path is not None:
                    f.write(print_str % (cnt_seg, end_value, audio_len) + "\n")

        if save_path is not None:
            f.close()

    def energy_VAD(
        self,
        audio_file,
        boundaries,
        activation_th=0.5,
        deactivation_th=0.0,
        eps=1e-6,
    ):
        """Applies energy-based VAD within the detected speech segments.The neural
        network VAD often creates longer segments and tends to merge segments that
        are close with each other.

        The energy VAD post-processes can be useful for having a fine-grained voice
        activity detection.

        The energy VAD computes the energy within the small chunks. The energy is
        normalized within the segment to have mean 0.5 and +-0.5 of std.
        This helps to set the energy threshold.

        Arguments
        ---------
        audio_file: path
            Path of the audio file containing the recording. The file is read
            with torchaudio.
        boundaries : torch.Tensor
            Tensor containing the speech boundaries. It can be derived using the
            get_boundaries method.
        activation_th: float
            A new speech segment is started it the energy is above activation_th.
        deactivation_th: float
            The segment is considered ended when the energy is <= deactivation_th.
        eps: float
            Small constant for numerical stability.


        Returns
        -------
        new_boundaries
            The new boundaries that are post-processed by the energy VAD.
        """

        # Getting the total size of the input file
        sample_rate, audio_len = self._get_audio_info(audio_file)

        if sample_rate != self.sample_rate:
            raise ValueError(
                "The detected sample rate is different from that set in the hparam file"
            )

        # Computing the chunk length of the energy window
        chunk_len = int(self.time_resolution * sample_rate)
        new_boundaries = []

        # Processing speech segments
        for i in range(boundaries.shape[0]):
            begin_sample = int(boundaries[i, 0] * sample_rate)
            end_sample = int(boundaries[i, 1] * sample_rate)
            seg_len = end_sample - begin_sample

            # Reading the speech segment
            segment, _ = torchaudio.load(
                audio_file, frame_offset=begin_sample, num_frames=seg_len
            )

            # Create chunks
            segment_chunks = self.create_chunks(
                segment, chunk_size=chunk_len, chunk_stride=chunk_len
            )

            # Energy computation within each chunk
            energy_chunks = segment_chunks.abs().sum(-1) + eps
            energy_chunks = energy_chunks.log()

            # Energy normalization
            energy_chunks = (
                (energy_chunks - energy_chunks.mean())
                / (2 * energy_chunks.std())
            ) + 0.5
            energy_chunks = energy_chunks.unsqueeze(0).unsqueeze(2)

            # Apply threshold based on the energy value
            energy_vad = self.apply_threshold(
                energy_chunks,
                activation_th=activation_th,
                deactivation_th=deactivation_th,
            )

            # Get the boundaries
            energy_boundaries = self.get_boundaries(
                energy_vad, output_value="seconds"
            )

            # Get the final boundaries in the original signal
            for j in range(energy_boundaries.shape[0]):
                start_en = boundaries[i, 0] + energy_boundaries[j, 0]
                end_end = boundaries[i, 0] + energy_boundaries[j, 1]
                new_boundaries.append([start_en, end_end])

        # Convert boundaries to tensor
        new_boundaries = torch.FloatTensor(new_boundaries).to(boundaries.device)
        return new_boundaries

    def create_chunks(self, x, chunk_size=16384, chunk_stride=16384):
        """Splits the input into smaller chunks of size chunk_size with
        an overlap chunk_stride. The chunks are concatenated over
        the batch axis.

        Arguments
        ---------
        x: torch.Tensor
            Signal to split into chunks.
        chunk_size : str
            The size of each chunk.
        chunk_stride:
            The stride (hop) of each chunk.


        Returns
        -------
        x: torch.Tensor
            A new tensors with the chunks derived from the input signal.

        """
        x = x.unfold(1, chunk_size, chunk_stride)
        x = x.reshape(x.shape[0] * x.shape[1], -1)
        return x

    def _get_audio_info(self, audio_file):
        """Returns the sample rate and the length of the input audio file"""

        # Getting the total size of the input file
        metadata = torchaudio.info(audio_file)
        sample_rate = metadata.sample_rate
        audio_len = metadata.num_frames
        return sample_rate, audio_len

    def upsample_VAD(self, vad_out, audio_file, time_resolution=0.01):
        """Upsamples the output of the vad to help visualization. It creates a
        signal that is 1 when there is speech and 0 when there is no speech.
        The vad signal has the same resolution as the input one and can be
        opened with it (e.g, using audacity) to visually figure out VAD regions.

        Arguments
        ---------
        vad_out: torch.Tensor
            Tensor containing 1 for each frame of speech and 0 for each non-speech
            frame.
        audio_file: path
            The original audio file used to compute vad_out
        time_resolution : float
            Time resolution of the vad_out signal.

        Returns
        -------
        vad_signal
            The upsampled version of the vad_out tensor.
        """

        # Getting the total size of the input file
        sample_rate, sig_len = self._get_audio_info(audio_file)

        if sample_rate != self.sample_rate:
            raise ValueError(
                "The detected sample rate is different from that set in the hparam file"
            )

        beg_samp = 0
        step_size = int(time_resolution * sample_rate)
        end_samp = step_size
        index = 0

        # Initialize upsampled signal
        vad_signal = torch.zeros(1, sig_len, device=vad_out.device)

        # Upsample signal
        while end_samp < sig_len:
            vad_signal[0, beg_samp:end_samp] = vad_out[0, index, 0]
            index = index + 1
            beg_samp = beg_samp + step_size
            end_samp = beg_samp + step_size
        return vad_signal

    def upsample_boundaries(self, boundaries, audio_file):
        """Based on the input boundaries, this method creates a signal that is 1
        when there is speech and 0 when there is no speech.
        The vad signal has the same resolution as the input one and can be
        opened with it (e.g, using audacity) to visually figure out VAD regions.

        Arguments
        ---------
        boundaries: torch.Tensor
            Tensor containing the boundaries of the speech segments.
        audio_file: path
            The original audio file used to compute vad_out

        Returns
        -------
        vad_signal
            The output vad signal with the same resolution of the input one.
        """

        # Getting the total size of the input file
        sample_rate, sig_len = self._get_audio_info(audio_file)

        if sample_rate != self.sample_rate:
            raise ValueError(
                "The detected sample rate is different from that set in the hparam file"
            )

        # Initialization of the output signal
        vad_signal = torch.zeros(1, sig_len, device=boundaries.device)

        # Composing the vad signal from boundaries
        for i in range(boundaries.shape[0]):
            beg_sample = int(boundaries[i, 0] * sample_rate)
            end_sample = int(boundaries[i, 1] * sample_rate)
            vad_signal[0, beg_sample:end_sample] = 1.0
        return vad_signal

    def double_check_speech_segments(
        self, boundaries, audio_file, speech_th=0.5
    ):
        """Takes in input the boundaries of the detected speech segments and
        double checks (using the neural VAD) that they actually contain speech.

        Arguments
        ---------
        boundaries: torch.Tensor
            Tensor containing the boundaries of the speech segments.
        audio_file: path
            The original audio file used to compute vad_out.
        speech_th: float
            Threshold on the mean posterior probability over which speech is
            confirmed. Below that threshold, the segment is re-assigned to a
            non-speech region.

        Returns
        -------
        new_boundaries
            The boundaries of the segments where speech activity is confirmed.
        """

        # Getting the total size of the input file
        sample_rate, sig_len = self._get_audio_info(audio_file)

        # Double check the segments
        new_boundaries = []
        for i in range(boundaries.shape[0]):
            beg_sample = int(boundaries[i, 0] * sample_rate)
            end_sample = int(boundaries[i, 1] * sample_rate)
            len_seg = end_sample - beg_sample

            # Read the candidate speech segment
            segment, fs = torchaudio.load(
                audio_file, frame_offset=beg_sample, num_frames=len_seg
            )
            speech_prob = self.get_speech_prob_chunk(segment)
            if speech_prob.mean() > speech_th:
                # Accept this as a speech segment
                new_boundaries.append([boundaries[i, 0], boundaries[i, 1]])

        # Convert boundaries from list to tensor
        new_boundaries = torch.FloatTensor(new_boundaries).to(boundaries.device)
        return new_boundaries

    def get_segments(
        self, boundaries, audio_file, before_margin=0.1, after_margin=0.1
    ):
        """Returns a list containing all the detected speech segments.

        Arguments
        ---------
        boundaries: torch.Tensor
            Tensor containing the boundaries of the speech segments.
        audio_file: path
            The original audio file used to compute vad_out.
        before_margin: float
            Used to cut the segments samples a bit before the detected margin.
        after_margin: float
            Use to cut the segments samples a bit after the detected margin.

        Returns
        -------
        segments: list
            List containing the detected speech segments
        """
        sample_rate, sig_len = self._get_audio_info(audio_file)

        if sample_rate != self.sample_rate:
            raise ValueError(
                "The detected sample rate is different from that set in the hparam file"
            )

        segments = []
        for i in range(boundaries.shape[0]):
            beg_sample = boundaries[i, 0] * sample_rate
            end_sample = boundaries[i, 1] * sample_rate

            beg_sample = int(max(0, beg_sample - before_margin * sample_rate))
            end_sample = int(
                min(sig_len, end_sample + after_margin * sample_rate)
            )

            len_seg = end_sample - beg_sample
            vad_segment, fs = torchaudio.load(
                audio_file, frame_offset=beg_sample, num_frames=len_seg
            )
            segments.append(vad_segment)
        return segments

    def get_speech_segments(
        self,
        audio_file,
        large_chunk_size=30,
        small_chunk_size=10,
        overlap_small_chunk=False,
        apply_energy_VAD=False,
        double_check=True,
        close_th=0.250,
        len_th=0.250,
        activation_th=0.5,
        deactivation_th=0.25,
        en_activation_th=0.5,
        en_deactivation_th=0.0,
        speech_th=0.50,
    ):
        """Detects speech segments within the input file. The input signal can
        be both a short or a long recording. The function computes the
        posterior probabilities on large chunks (e.g, 30 sec), that are read
        sequentially (to avoid storing big signals in memory).
        Each large chunk is, in turn, split into smaller chunks (e.g, 10 seconds)
        that are processed in parallel. The pipeline for detecting the speech
        segments is the following:
            1- Compute posteriors probabilities at the frame level.
            2- Apply a threshold on the posterior probability.
            3- Derive candidate speech segments on top of that.
            4- Apply energy VAD within each candidate segment (optional).
            5- Merge segments that are too close.
            6- Remove segments that are too short.
            7- Double check speech segments (optional).


        Arguments
        ---------
        audio_file : str
            Path to audio file.
        large_chunk_size: float
            Size (in seconds) of the large chunks that are read sequentially
            from the input audio file.
        small_chunk_size: float
            Size (in seconds) of the small chunks extracted from the large ones.
            The audio signal is processed in parallel within the small chunks.
            Note that large_chunk_size/small_chunk_size must be an integer.
        overlap_small_chunk: bool
            If True, it creates overlapped small chunks (with 50% overlap).
            The probabilities of the overlapped chunks are combined using
            hamming windows.
        apply_energy_VAD: bool
            If True, a energy-based VAD is used on the detected speech segments.
            The neural network VAD often creates longer segments and tends to
            merge close segments together. The energy VAD post-processes can be
            useful for having a fine-grained voice activity detection.
            The energy thresholds is  managed by activation_th and
            deactivation_th (see below).
        double_check: bool
            If True, double checks (using the neural VAD) that the candidate
            speech segments actually contain speech. A threshold on the mean
            posterior probabilities provided by the neural network is applied
            based on the speech_th parameter (see below).
        activation_th:  float
            Threshold of the neural posteriors above which starting a speech segment.
        deactivation_th: float
            Threshold of the neural posteriors below which ending a speech segment.
        en_activation_th: float
            A new speech segment is started it the energy is above activation_th.
            This is active only if apply_energy_VAD is True.
        en_deactivation_th: float
            The segment is considered ended when the energy is <= deactivation_th.
            This is active only if apply_energy_VAD is True.
        speech_th: float
            Threshold on the mean posterior probability within the candidate
            speech segment. Below that threshold, the segment is re-assigned to
            a non-speech region. This is active only if double_check is True.
        close_th: float
            If the distance between boundaries is smaller than close_th, the
            segments will be merged.
        len_th: float
            If the length of the segment is smaller than close_th, the segments
            will be merged.

        Returns
        -------
        boundaries: torch.Tensor
            Tensor containing the start second of speech segments in even
            positions and their corresponding end in odd positions
            (e.g, [1.0, 1.5, 5,.0 6.0] means that we have two speech segment;
             one from 1.0 to 1.5 seconds and another from 5.0 to 6.0 seconds).
        """

        # Fetch audio file from web if not local
        source, fl = split_path(audio_file)
        audio_file = fetch(fl, source=source)

        # Computing speech vs non speech probabilities
        prob_chunks = self.get_speech_prob_file(
            audio_file,
            large_chunk_size=large_chunk_size,
            small_chunk_size=small_chunk_size,
            overlap_small_chunk=overlap_small_chunk,
        )

        # Apply a threshold to get candidate speech segments
        prob_th = self.apply_threshold(
            prob_chunks,
            activation_th=activation_th,
            deactivation_th=deactivation_th,
        ).float()

        # Compute the boundaries of the speech segments
        boundaries = self.get_boundaries(prob_th, output_value="seconds")

        # Apply energy-based VAD on the detected speech segments
        if apply_energy_VAD:
            boundaries = self.energy_VAD(
                audio_file,
                boundaries,
                activation_th=en_activation_th,
                deactivation_th=en_deactivation_th,
            )

        # Merge short segments
        boundaries = self.merge_close_segments(boundaries, close_th=close_th)

        # Remove short segments
        boundaries = self.remove_short_segments(boundaries, len_th=len_th)

        # Double check speech segments
        if double_check:
            boundaries = self.double_check_speech_segments(
                boundaries, audio_file, speech_th=speech_th
            )

        return boundaries

    def forward(self, wavs, wav_lens=None):
        """Gets frame-level speech-activity predictions"""
        return self.get_speech_prob_chunk(wavs, wav_lens)


class SepformerSeparation(Pretrained):
    """A "ready-to-use" speech separation model.

    Uses Sepformer architecture.

    Example
    -------
    >>> tmpdir = getfixture("tmpdir")
    >>> model = SepformerSeparation.from_hparams(
    ...     source="speechbrain/sepformer-wsj02mix",
    ...     savedir=tmpdir)
    >>> mix = torch.randn(1, 400)
    >>> est_sources = model.separate_batch(mix)
    >>> print(est_sources.shape)
    torch.Size([1, 400, 2])
    """

    MODULES_NEEDED = ["encoder", "masknet", "decoder"]

    def separate_batch(self, mix):
        """Run source separation on batch of audio.

        Arguments
        ---------
        mix : torch.Tensor
            The mixture of sources.

        Returns
        -------
        tensor
            Separated sources
        """

        # Separation
        mix = mix.to(self.device)
        mix_w = self.mods.encoder(mix)
        est_mask = self.mods.masknet(mix_w)
        mix_w = torch.stack([mix_w] * self.hparams.num_spks)
        sep_h = mix_w * est_mask

        # Decoding
        est_source = torch.cat(
            [
                self.mods.decoder(sep_h[i]).unsqueeze(-1)
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

    def separate_file(self, path, savedir="."):
        """Separate sources from file.

        Arguments
        ---------
        path : str
            Path to file which has a mixture of sources. It can be a local
            path, a web url, or a huggingface repo.
        savedir : path
            Path where to store the wav signals (when downloaded from the web).
        Returns
        -------
        tensor
            Separated sources
        """
        source, fl = split_path(path)
        path = fetch(fl, source=source, savedir=savedir)

        batch, fs_file = torchaudio.load(path)
        batch = batch.to(self.device)
        fs_model = self.hparams.sample_rate

        # resample the data if needed
        if fs_file != fs_model:
            print(
                "Resampling the audio from {} Hz to {} Hz".format(
                    fs_file, fs_model
                )
            )
            tf = torchaudio.transforms.Resample(
                orig_freq=fs_file, new_freq=fs_model
            ).to(self.device)
            batch = batch.mean(dim=0, keepdim=True)
            batch = tf(batch)

        est_sources = self.separate_batch(batch)
        est_sources = (
            est_sources / est_sources.abs().max(dim=1, keepdim=True)[0]
        )
        return est_sources

    def forward(self, mix):
        """Runs separation on the input mix"""
        return self.separate_batch(mix)


class SpectralMaskEnhancement(Pretrained):
    """A ready-to-use model for speech enhancement.

    Arguments
    ---------
    See ``Pretrained``.

    Example
    -------
    >>> import torch
    >>> from speechbrain.pretrained import SpectralMaskEnhancement
    >>> # Model is downloaded from the speechbrain HuggingFace repo
    >>> tmpdir = getfixture("tmpdir")
    >>> enhancer = SpectralMaskEnhancement.from_hparams(
    ...     source="speechbrain/metricgan-plus-voicebank",
    ...     savedir=tmpdir,
    ... )
    >>> enhanced = enhancer.enhance_file(
    ...     "speechbrain/metricgan-plus-voicebank/example.wav"
    ... )
    """

    HPARAMS_NEEDED = ["compute_stft", "spectral_magnitude", "resynth"]
    MODULES_NEEDED = ["enhance_model"]

    def compute_features(self, wavs):
        """Compute the log spectral magnitude features for masking.

        Arguments
        ---------
        wavs : torch.Tensor
            A batch of waveforms to convert to log spectral mags.
        """
        feats = self.hparams.compute_stft(wavs)
        feats = self.hparams.spectral_magnitude(feats)
        return torch.log1p(feats)

    def enhance_batch(self, noisy, lengths=None):
        """Enhance a batch of noisy waveforms.

        Arguments
        ---------
        noisy : torch.Tensor
            A batch of waveforms to perform enhancement on.
        lengths : torch.Tensor
            The lengths of the waveforms if the enhancement model handles them.

        Returns
        -------
        torch.Tensor
            A batch of enhanced waveforms of the same shape as input.
        """
        noisy = noisy.to(self.device)
        noisy_features = self.compute_features(noisy)

        # Perform masking-based enhancement, multiplying output with input.
        if lengths is not None:
            mask = self.mods.enhance_model(noisy_features, lengths=lengths)
        else:
            mask = self.mods.enhance_model(noisy_features)
        enhanced = torch.mul(mask, noisy_features)

        # Return resynthesized waveforms
        return self.hparams.resynth(torch.expm1(enhanced), noisy)

    def enhance_file(self, filename, output_filename=None):
        """Enhance a wav file.

        Arguments
        ---------
        filename : str
            Location on disk to load file for enhancement.
        output_filename : str
            If provided, writes enhanced data to this file.
        """
        noisy = self.load_audio(filename)
        noisy = noisy.to(self.device)

        # Fake a batch:
        batch = noisy.unsqueeze(0)
        if lengths_arg_exists(self.enhance_batch):
            enhanced = self.enhance_batch(batch, lengths=torch.tensor([1.0]))
        else:
            enhanced = self.enhance_batch(batch)

        if output_filename is not None:
            torchaudio.save(output_filename, enhanced, channels_first=False)

        return enhanced.squeeze(0)


class EncodeDecodePipelineMixin:
    """
    A mixin for pretrained models that makes it possible to specify an encoding pipeline and a decoding pipeline
    """

    def create_pipelines(self):
        """
        Initializes the encode and decode pipeline
        """
        self._run_init_steps(self.hparams.encode_pipeline)
        self._run_init_steps(self.hparams.decode_pipeline)
        self.encode_pipeline = DataPipeline(
            static_data_keys=self.INPUT_STATIC_KEYS,
            dynamic_items=self.hparams.encode_pipeline["steps"],
            output_keys=self.hparams.encode_pipeline["output_keys"],
        )
        self.decode_pipeline = DataPipeline(
            static_data_keys=self.hparams.model_output_keys,
            dynamic_items=self.hparams.decode_pipeline["steps"],
            output_keys=self.OUTPUT_KEYS,
        )

    def _run_init_steps(self, pipeline_definition):
        """Encode/decode pipelines may include initialization
        steps, such as filling text encoders with tokens. Calling
        this method will run them, if defined"""
        steps = pipeline_definition.get("init", [])
        for step in steps:
            step_func = step.get("func")
            if not step_func or not callable(step_func):
                raise ValueError("Invalid pipeline init definition")
            step_func()

    def _run_pipeline(self, pipeline, input, batch):
        if batch:
            output = pipeline(input)
        else:
            output = [pipeline(item) for item in input]
        return output

    def _get_encode_pipeline_input(self, input):
        return input if self.batch_inputs else self._itemize(input)

    def _get_decode_pipeline_input(self, model_output):
        model_output_keys = getattr(self.hparams, "model_output_keys", None)
        pipeline_input = model_output
        if len(model_output_keys) == 1:
            pipeline_input = (pipeline_input,)
        # The input to a pipeline is a dictionary. If model_output_keys
        # is provided, the output of the model is assumed to be a collection
        # (e.g. a list or a tuple).
        if model_output_keys:
            pipeline_input = dict(zip(model_output_keys, pipeline_input))

        # By default, the pipeline will be applied to in batch mode
        # to the entire model input
        if not self.batch_outputs:
            pipeline_input = self._itemize(pipeline_input)
        return pipeline_input

    def _itemize(self, pipeline_input):
        first_item = next(iter(pipeline_input.values()))
        keys, values = pipeline_input.keys(), pipeline_input.values()
        batch_length = len(first_item)
        return [
            dict(zip(keys, [value[idx] for value in values]))
            for idx in range(batch_length)
        ]

    def to_dict(self, data):
        """
        Converts padded batches to dictionaries, leaves
        other data types as is

        Arguments
        ---------
        data: object
            a dictionary or a padded batch

        Returns
        -------
        results: dict
            the dictionary
        """
        if isinstance(data, PaddedBatch):
            data = {
                key: self._get_value(data, key)
                for key in self.hparams.encode_pipeline["output_keys"]
            }
        return data

    def _get_value(self, data, key):
        """
        Retrieves the value associated with the specified key, dereferencing
        .data where applicable

        Arguments
        ---------
        data: PaddedBatch
            a padded batch
        key: str
            the key

        Returns
        -------
        result: object
            the result
        """
        value = getattr(data, key)
        if not self.input_use_padded_data and isinstance(value, PaddedData):
            value = value.data
        return value

    @property
    def batch_inputs(self):
        """
        Determines whether the input pipeline
        operates on batches or individual examples
        (true means batched)

        Returns
        -------
        batch_inputs: bool
        """
        return self.hparams.encode_pipeline.get("batch", True)

    @property
    def input_use_padded_data(self):
        """
        If turned on, raw PaddedData instances will be passed to
        the model. If turned off, only .data will be used

        Returns
        -------
        result: bool
            whether padded data is used as is
        """
        return self.hparams.encode_pipeline.get("use_padded_data", False)

    @property
    def batch_outputs(self):
        """
        Determines whether the output pipeline
        operates on batches or individual examples
        (true means batched)

        Returns
        -------
        batch_outputs: bool
        """
        return self.hparams.decode_pipeline.get("batch", True)

    def _collate(self, data):
        if not self.batch_inputs:
            collate_fn = getattr(self.hparams, "collate_fn", PaddedBatch)
            data = collate_fn(data)
        return data

    def encode_input(self, input):
        """
        Encodes the inputs using the pipeline

        Arguments
        ---------
        input: dict
            the raw inputs

        Returns
        -------
        results: object

        """
        pipeline_input = self._get_encode_pipeline_input(input)
        model_input = self._run_pipeline(
            pipeline=self.encode_pipeline,
            input=pipeline_input,
            batch=self.batch_inputs,
        )
        model_input = self._collate(model_input)
        if hasattr(model_input, "to"):
            model_input = model_input.to(self.device)
        return self.to_dict(model_input)

    def decode_output(self, output):
        """
        Decodes the raw model outputs

        Arguments
        ---------
        output: tuple
            raw model outputs

        Returns
        -------
        result: dict or list
            the output of the pipeline
        """
        pipeline_input = self._get_decode_pipeline_input(output)
        return self._run_pipeline(
            pipeline=self.decode_pipeline,
            input=pipeline_input,
            batch=self.batch_outputs,
        )


class GraphemeToPhoneme(Pretrained, EncodeDecodePipelineMixin):
    """
    A pretrained model implementation for Grapheme-to-Phoneme (G2P) models
    that take raw natural language text as an input and

    Example
    -------
    >>> text = ("English is tough. It can be understood "
    ...         "through thorough thought though")
    >>> from speechbrain.pretrained import GraphemeToPhoneme
    >>> tmpdir = getfixture('tmpdir')
    >>> g2p = GraphemeToPhoneme.from_hparams('path/to/model', savedir=tmpdir) # doctest: +SKIP
    >>> phonemes = g2p.g2p(text) # doctest: +SKIP
    """

    INPUT_STATIC_KEYS = ["txt"]
    OUTPUT_KEYS = ["phonemes"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.create_pipelines()
        self.load_dependencies()

    @property
    def phonemes(self):
        """Returns the available phonemes"""
        return self.hparams.phonemes

    @property
    def language(self):
        """Returns the language for which this model is available"""
        return self.hparams.language

    def g2p(self, text):
        """Performs the Grapheme-to-Phoneme conversion

        Arguments
        ---------
        text: str or list[str]
            a single string to be encoded to phonemes - or a
            sequence of strings

        Returns
        -------
        result: list
            if a single example was provided, the return value is a
            single list of phonemes
        """
        single = isinstance(text, str)
        if single:
            text = [text]

        model_inputs = self.encode_input({"txt": text})
        self._update_graphemes(model_inputs)
        model_outputs = self.mods.model(**model_inputs)
        decoded_output = self.decode_output(model_outputs)
        phonemes = decoded_output["phonemes"]
        if single:
            phonemes = phonemes[0]
        return phonemes

    def _update_graphemes(self, model_inputs):
        grapheme_sequence_mode = getattr(self.hparams, "grapheme_sequence_mode")
        if grapheme_sequence_mode and grapheme_sequence_mode != "raw":
            grapheme_encoded_key = f"grapheme_encoded_{grapheme_sequence_mode}"
            if grapheme_encoded_key in model_inputs:
                model_inputs["grapheme_encoded"] = model_inputs[
                    grapheme_encoded_key
                ]

    def load_dependencies(self):
        """Loads any relevant model dependencies"""
        deps_pretrainer = getattr(self.hparams, "deps_pretrainer", None)
        if deps_pretrainer:
            deps_pretrainer.collect_files()
            deps_pretrainer.load_collected(device=self.device)

    def __call__(self, text):
        """A convenience callable wrapper - same as G2P

        Arguments
        ---------
        text: str or list[str]
            a single string to be encoded to phonemes - or a
            sequence of strings

        Returns
        -------
        result: list
            if a single example was provided, the return value is a
            single list of phonemes
        """
        return self.g2p(text)

    def forward(self, noisy, lengths=None):
        """Runs enhancement on the noisy input"""
        return self.enhance_batch(noisy, lengths)


class WaveformEnhancement(Pretrained):
    """A ready-to-use model for speech enhancement.

    Arguments
    ---------
    See ``Pretrained``.

    Example
    -------
    >>> from speechbrain.pretrained import WaveformEnhancement
    >>> # Model is downloaded from the speechbrain HuggingFace repo
    >>> tmpdir = getfixture("tmpdir")
    >>> enhancer = WaveformEnhancement.from_hparams(
    ...     source="speechbrain/mtl-mimic-voicebank",
    ...     savedir=tmpdir,
    ... )
    >>> enhanced = enhancer.enhance_file(
    ...     "speechbrain/mtl-mimic-voicebank/example.wav"
    ... )
    """

    MODULES_NEEDED = ["enhance_model"]

    def enhance_batch(self, noisy, lengths=None):
        """Enhance a batch of noisy waveforms.

        Arguments
        ---------
        noisy : torch.Tensor
            A batch of waveforms to perform enhancement on.
        lengths : torch.Tensor
            The lengths of the waveforms if the enhancement model handles them.

        Returns
        -------
        torch.Tensor
            A batch of enhanced waveforms of the same shape as input.
        """
        noisy = noisy.to(self.device)
        enhanced_wav, _ = self.mods.enhance_model(noisy)
        return enhanced_wav

    def enhance_file(self, filename, output_filename=None):
        """Enhance a wav file.

        Arguments
        ---------
        filename : str
            Location on disk to load file for enhancement.
        output_filename : str
            If provided, writes enhanced data to this file.
        """
        noisy = self.load_audio(filename)

        # Fake a batch:
        batch = noisy.unsqueeze(0)
        enhanced = self.enhance_batch(batch)

        if output_filename is not None:
            torchaudio.save(output_filename, enhanced, channels_first=False)

        return enhanced.squeeze(0)

    def forward(self, noisy, lengths=None):
        """Runs enhancement on the noisy input"""
        return self.enhance_batch(noisy, lengths)


class SNREstimator(Pretrained):
    """A "ready-to-use" SNR estimator.
    """

    MODULES_NEEDED = ["encoder", "encoder_out"]
    HPARAMS_NEEDED = ["stat_pooling", "snrmax", "snrmin"]

    def estimate_batch(self, mix, predictions):
        """Run SI-SNR estimation on the estimated sources, and mixture.

        Arguments
        ---------
        mix : torch.Tensor
            The mixture of sources of shape B X T
        predictions : torch.Tensor
            of size (B x T x C),
            where B is batch size
                  T is number of time points
                  C is number of sources

        Returns
        -------
        tensor
            Estimate of SNR
        """

        predictions = predictions.permute(0, 2, 1)
        predictions = predictions.reshape(-1, predictions.size(-1))

        if hasattr(self.hparams, "separation_norm_type"):
            if self.hparams.separation_norm_type == "max":
                predictions = (
                    predictions / predictions.max(dim=1, keepdim=True)[0]
                )
                mix = mix / mix.max(dim=1, keepdim=True)[0]

            elif self.hparams.separation_norm_type == "stnorm":
                predictions = (
                    predictions - predictions.mean(dim=1, keepdim=True)
                ) / predictions.std(dim=1, keepdim=True)
                mix = (mix - mix.mean(dim=1, keepdim=True)) / mix.std(
                    dim=1, keepdim=True
                )

        min_T = min(predictions.shape[1], mix.shape[1])
        assert predictions.shape[1] == mix.shape[1], "lengths change"

        mix_repeat = mix.repeat(2, 1)
        inp_cat = torch.cat(
            [
                predictions[:, :min_T].unsqueeze(1),
                mix_repeat[:, :min_T].unsqueeze(1),
            ],
            dim=1,
        )

        enc = self.mods.encoder(inp_cat)
        enc = enc.permute(0, 2, 1)
        enc_stats = self.hparams.stat_pooling(enc)

        # this gets the SI-SNR estimate in the compressed range 0-1
        snrhat = self.mods.encoder_out(enc_stats).squeeze()

        # get the SI-SNR estimate in the true range
        snrhat = self.gettrue_snrrange(snrhat)
        return snrhat

    def forward(self, mix, predictions):
        """Just run the batch estimate"""
        return self.estimate_batch(mix, predictions)

    def gettrue_snrrange(self, inp):
        """Convert from 0-1 range to true snr range"""
        rnge = self.hparams.snrmax - self.hparams.snrmin
        inp = inp * rnge
        inp = inp + self.hparams.snrmin
        return inp


class Tacotron2(Pretrained):
    """
    A ready-to-use wrapper for Tacotron2 (text -> mel_spec).

    Arguments
    ---------
    hparams
        Hyperparameters (from HyperPyYAML)

    Example
    -------
    >>> tmpdir_vocoder = getfixture('tmpdir') / "vocoder"
    >>> tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir=tmpdir_vocoder)
    >>> mel_output, mel_length, alignment = tacotron2.encode_text("Mary had a little lamb")
    >>> items = [
    ...   "A quick brown fox jumped over the lazy dog",
    ...   "How much wood would a woodchuck chuck?",
    ...   "Never odd or even"
    ... ]
    >>> mel_outputs, mel_lengths, alignments = tacotron2.encode_batch(items)

    >>> # One can combine the TTS model with a vocoder (that generates the final waveform)
    >>> # Initialize the Vocoder (HiFIGAN)
    >>> tmpdir_tts = getfixture('tmpdir') / "tts"
    >>> hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir=tmpdir_tts)
    >>> # Running the TTS
    >>> mel_output, mel_length, alignment = tacotron2.encode_text("Mary had a little lamb")
    >>> # Running Vocoder (spectrogram-to-waveform)
    >>> waveforms = hifi_gan.decode_batch(mel_output)
    """

    HPARAMS_NEEDED = ["model", "text_to_sequence"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_cleaners = getattr(
            self.hparams, "text_cleaners", ["english_cleaners"]
        )
        self.infer = self.hparams.model.infer

    def text_to_seq(self, txt):
        """Encodes raw text into a tensor with a customer text-to-sequence function
        """
        sequence = self.hparams.text_to_sequence(txt, self.text_cleaners)
        return sequence, len(sequence)

    def encode_batch(self, texts):
        """Computes mel-spectrogram for a list of texts

        Texts must be sorted in decreasing order on their lengths

        Arguments
        ---------
        texts: List[str]
            texts to be encoded into spectrogram

        Returns
        -------
        tensors of output spectrograms, output lengths and alignments
        """
        with torch.no_grad():
            inputs = [
                {
                    "text_sequences": torch.tensor(
                        self.text_to_seq(item)[0], device=self.device
                    )
                }
                for item in texts
            ]
            inputs = speechbrain.dataio.batch.PaddedBatch(inputs)

            lens = [self.text_to_seq(item)[1] for item in texts]
            assert lens == sorted(
                lens, reverse=True
            ), "input lengths must be sorted in decreasing order"
            input_lengths = torch.tensor(lens, device=self.device)

            mel_outputs_postnet, mel_lengths, alignments = self.infer(
                inputs.text_sequences.data, input_lengths
            )
        return mel_outputs_postnet, mel_lengths, alignments

    def encode_text(self, text):
        """Runs inference for a single text str"""
        return self.encode_batch([text])

    def forward(self, texts):
        "Encodes the input texts."
        return self.encode_batch(texts)


class HIFIGAN(Pretrained):
    """
    A ready-to-use wrapper for HiFiGAN (mel_spec -> waveform).

    Arguments
    ---------
    hparams
        Hyperparameters (from HyperPyYAML)

    Example
    -------
    >>> tmpdir_vocoder = getfixture('tmpdir') / "vocoder"
    >>> hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir=tmpdir_vocoder)
    >>> mel_specs = torch.rand(2, 80,298)
    >>> waveforms = hifi_gan.decode_batch(mel_specs)

    >>> # You can use the vocoder coupled with a TTS system
    >>>	# Initialize TTS (tacotron2)
    >>> tmpdir_tts = getfixture('tmpdir') / "tts"
    >>>	tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir=tmpdir_tts)
    >>>	# Running the TTS
    >>>	mel_output, mel_length, alignment = tacotron2.encode_text("Mary had a little lamb")
    >>>	# Running Vocoder (spectrogram-to-waveform)
    >>>	waveforms = hifi_gan.decode_batch(mel_output)
    """

    HPARAMS_NEEDED = ["generator"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.infer = self.hparams.generator.inference
        self.first_call = True

    def decode_batch(self, spectrogram):
        """Computes waveforms from a batch of mel-spectrograms

        Arguments
        ---------
        spectrogram: torch.Tensor
            Batch of mel-spectrograms [batch, mels, time]

        Returns
        -------
        waveforms: torch.Tensor
            Batch of mel-waveforms [batch, 1, time]

        """
        # Prepare for inference by removing the weight norm
        if self.first_call:
            self.hparams.generator.remove_weight_norm()
            self.first_call = False
        with torch.no_grad():
            waveform = self.infer(spectrogram.to(self.device))
        return waveform

    def decode_spectrogram(self, spectrogram):
        """Computes waveforms from a single mel-spectrogram

        Arguments
        ---------
        spectrogram: torch.Tensor
            mel-spectrogram [mels, time]

        Returns
        -------
        waveform: torch.Tensor
            waveform [1, time]

        audio can be saved by:
        >>> waveform = torch.rand(1, 666666)
        >>> sample_rate = 22050
        >>> torchaudio.save(str(getfixture('tmpdir') / "test.wav"), waveform, sample_rate)
        """
        if self.first_call:
            self.hparams.generator.remove_weight_norm()
            self.first_call = False
        with torch.no_grad():
            waveform = self.infer(spectrogram.unsqueeze(0).to(self.device))
        return waveform.squeeze(0)

    def forward(self, spectrogram):
        "Decodes the input spectrograms"
        return self.decode_batch(spectrogram)
