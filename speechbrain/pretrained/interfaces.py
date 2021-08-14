"""Defines interfaces for simple inference with pretrained models

Authors:
 * Aku Rouhe 2021
 * Peter Plantinga 2021
 * Loren Lugosch 2020
 * Mirco Ravanelli 2020
 * Titouan Parcollet 2021
 * Artem Ploujnikov 2021
"""
import os
import logging
import shlex
import sys
import torch
import torchaudio
import torch.nn.functional as F

from copy import Error
from importlib import import_module
from speechbrain.utils.superpowers import run_shell
from types import SimpleNamespace
from torch.nn import SyncBatchNorm
from torch.nn import DataParallel as DP
from hyperpyyaml import load_hyperpyyaml
from speechbrain.pretrained.fetching import fetch
from speechbrain.dataio.preprocess import AudioNormalizer
from torch.nn.parallel import DistributedDataParallel as DDP
from speechbrain.utils.data_pipeline import DataPipeline
from speechbrain.utils.data_utils import split_path
from speechbrain.utils.distributed import run_on_main
from speechbrain.dataio.batch import PaddedBatch


logger = logging.getLogger(__name__)


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

    HPARAMS_NEEDED = []
    MODULES_NEEDED = []

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
        self.modules = torch.nn.ModuleDict(modules)
        for mod in self.modules:
            self.modules[mod].to(self.device)

        for mod in self.MODULES_NEEDED:
            if mod not in modules:
                raise ValueError(f"Need modules['{mod}']")

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
            self.modules.eval()
            for p in self.modules.parameters():
                p.requires_grad = False

    def load_audio(self, path, savedir="."):
        """Load an audio file with this model"s input spec

        When using a speech model, it is important to use the same type of data,
        as was used to train the model. This means for example using the same
        sampling rate and number of channels. It is, however, possible to
        convert a file from a higher sampling rate to a lower one (downsampling).
        Similarly, it is simple to downmix a stereo file to mono.
        The path can be a local path, a web url, or a link to a huggingface repo.
        """
        source, fl = split_path(path)
        path = fetch(fl, source=source, savedir=savedir)
        signal, sr = torchaudio.load(path, channels_first=False)
        return self.audio_normalizer(signal, sr)

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
        savedir=None,
        use_auth_token=False,
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
            Where to put the pretraining material. If not given, will use
            ./pretrained_models/<class-name>-hash(source).
        use_auth_token : bool (default: False)
            If true Hugginface's auth_token will be used to load private models from the HuggingFace Hub,
            default is False because majority of models are public.
        """
        if savedir is None:
            clsname = cls.__name__
            savedir = f"./pretrained_models/{clsname}-{hash(source)}"
        hparams_local_path = fetch(
            hparams_file, source, savedir, use_auth_token
        )

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


class EndToEndSLU(Pretrained):
    """A end-to-end SLU model.

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
    >>> slu_model.decode_file("samples/audio_samples/example6.wav")
    "{'intent': 'SimpleMath', 'slots': {'number1': 37.67, 'number2': 75.7, 'op': ' minus '}}"
    """

    HPARAMS_NEEDED = ["tokenizer", "asr_model_source"]
    MODULES_NEEDED = [
        "slu_enc",
        "beam_searcher",
    ]

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
        torch.tensor
            The encoded batch
        """
        wavs = wavs.float()
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        with torch.no_grad():
            ASR_encoder_out = self.asr_model.encode_batch(
                wavs.detach(), wav_lens
            )
        encoder_out = self.modules.slu_enc(ASR_encoder_out)
        return encoder_out

    def decode_batch(self, wavs, wav_lens):
        """Maps the input audio to its semantics

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
            Each waveform in the batch decoded.
        tensor
            Each predicted token id.
        """
        with torch.no_grad():
            wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
            encoder_out = self.encode_batch(wavs, wav_lens)
            predicted_tokens, scores = self.modules.beam_searcher(
                encoder_out, wav_lens
            )
            predicted_words = [
                self.tokenizer.decode_ids(token_seq)
                for token_seq in predicted_tokens
            ]
        return predicted_words, predicted_tokens


class EncoderDecoderASR(Pretrained):
    """A ready-to-use Encoder-Decoder ASR model

    The class can be used either to run only the encoder (encode()) to extract
    features or to run the entire encoder-decoder model
    (transcribe()) to transcribe speech. The given YAML must contains the fields
    specified in the *_NEEDED[] lists.

    Example
    -------
    >>> from speechbrain.pretrained import EncoderDecoderASR
    >>> tmpdir = getfixture("tmpdir")
    >>> asr_model = EncoderDecoderASR.from_hparams(
    ...     source="speechbrain/asr-crdnn-rnnlm-librispeech",
    ...     savedir=tmpdir,
    ... )
    >>> asr_model.transcribe_file("samples/audio_samples/example2.flac")
    "MY FATHER HAS REVEALED THE CULPRIT'S NAME"
    """

    HPARAMS_NEEDED = ["tokenizer"]
    MODULES_NEEDED = [
        "encoder",
        "decoder",
    ]

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
        torch.tensor
            The encoded batch
        """
        wavs = wavs.float()
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        encoder_out = self.modules.encoder(wavs, wav_lens)
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
            predicted_tokens, scores = self.modules.decoder(
                encoder_out, wav_lens
            )
            predicted_words = [
                self.tokenizer.decode_ids(token_seq)
                for token_seq in predicted_tokens
            ]
        return predicted_words, predicted_tokens


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
    >>> signal, fs = torchaudio.load("samples/audio_samples/example1.wav")
    >>> embeddings =  classifier.encode_batch(signal)

    >>> # Classification
    >>> prediction =  classifier .classify_batch(signal)
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
        torch.tensor
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
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, wav_lens)
        embeddings = self.modules.embedding_model(feats, wav_lens)
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
        wavs : torch.tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model. Make sure the sample rate is fs=16000 Hz.
        wav_lens : torch.tensor
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
        out_prob = self.modules.classifier(emb).squeeze(1)
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
        out_prob = self.modules.classifier(emb).squeeze(1)
        score, index = torch.max(out_prob, dim=-1)
        text_lab = self.hparams.label_encoder.decode_torch(index)
        return out_prob, score, index, text_lab


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
    >>> signal, fs = torchaudio.load("samples/audio_samples/example1.wav")
    >>> signal2, fs = torchaudio.load("samples/audio_samples/example2.flac")
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
        mix : torch.tensor
            The mixture of sources.

        Returns
        -------
        tensor
            Separated sources
        """

        # Separation
        mix = mix.to(self.device)
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
            )
            batch = batch.mean(dim=0, keepdim=True)
            batch = tf(batch)

        est_sources = self.separate_batch(batch)
        est_sources = est_sources / est_sources.max(dim=1, keepdim=True)[0]
        return est_sources


class SpectralMaskEnhancement(Pretrained):
    """A ready-to-use model for speech enhancement.

    Arguments
    ---------
    See ``Pretrained``.

    Example
    -------
    >>> import torchaudio
    >>> from speechbrain.pretrained import SpectralMaskEnhancement
    >>> # Model is downloaded from the speechbrain HuggingFace repo
    >>> tmpdir = getfixture("tmpdir")
    >>> enhancer = SpectralMaskEnhancement.from_hparams(
    ...     source="speechbrain/mtl-mimic-voicebank",
    ...     savedir=tmpdir,
    ... )
    >>> noisy, fs = torchaudio.load("samples/audio_samples/example_noisy.wav")
    >>> # Channel dimension is interpreted as batch dimension here
    >>> enhanced = enhancer.enhance_batch(noisy)
    """

    HPARAMS_NEEDED = ["compute_stft", "spectral_magnitude", "resynth"]
    MODULES_NEEDED = ["enhance_model"]

    def compute_features(self, wavs):
        """Compute the log spectral magnitude features for masking.

        Arguments
        ---------
        wavs : torch.tensor
            A batch of waveforms to convert to log spectral mags.
        """
        feats = self.hparams.compute_stft(wavs)
        feats = self.hparams.spectral_magnitude(feats)
        return torch.log1p(feats)

    def enhance_batch(self, noisy, lengths=None):
        """Enhance a batch of noisy waveforms.

        Arguments
        ---------
        noisy : torch.tensor
            A batch of waveforms to perform enhancement on.
        lengths : torch.tensor
            The lengths of the waveforms if the enhancement model handles them.

        Returns
        -------
        torch.tensor
            A batch of enhanced waveforms of the same shape as input.
        """
        noisy = noisy.to(self.device)
        noisy_features = self.compute_features(noisy)

        # Perform masking-based enhancement, multiplying output with input.
        if lengths is not None:
            mask = self.modules.enhance_model(noisy_features, lengths=lengths)
        else:
            mask = self.modules.enhance_model(noisy_features)
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
        enhanced = self.enhance_batch(batch)

        if output_filename is not None:
            torchaudio.save(output_filename, enhanced, channels_first=False)

        return enhanced.squeeze(0)


class SpeechSynthesizer(Pretrained):
    HPARAMS_NEEDED = ["model", "encode_pipeline", "decode_pipeline"]
    INPUT_STATIC_KEYS = ["txt"]
    OUTPUT_KEYS = ["wav"]

    """
    A friendly wrapper for speech synthesis models

    Arguments
    ---------
    hparams
        Hyperparameters (from HyperPyYAML)

    Example
    -------
    >>> synthesizer = SpeechSynthesizer.from_hparams('path/to/model')
    >>> waveform = synthesizer.tts("Mary had a little lamb")
    >>> items = [
    ...   "A quick brown fox jumped over the lazy dog",
    ...   "How much wood would a woodchuck chuck?",
    ...   "Never odd or even"
    ... ]
    >>> waveforms = synthesizer.tts(items)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        # NOTE: Some models will provide a custom, model-specific inference
        # function, others can be called directly
        self.infer = getattr(self.hparams, "infer", None)
        if not self.infer:
            self.infer = lambda model, **kwargs: model(**kwargs)
        self._transfer_external()

    def tts(self, text):
        """Computes the waveform for the provided example or batch
        of examples

        Arguments
        ---------
        text: str or List[str]
            the text to be translated into speech

        Returns
        -------
        a single waveform if a single example is provided - or
        a list of waveform tensors if multiple examples are provided
        """
        # Create a single batch if provided a single example
        single = isinstance(text, str)
        if single:
            text = [text]
        pipeline_input = self._get_encode_pipeline_input(text)
        model_input = self._run_pipeline(
            pipeline=self.encode_pipeline,
            input=pipeline_input,
            batch=self.batch_inputs,
        )
        model_input = self._collate(model_input)
        model_output = self.compute_forward(model_input)
        pipeline_input = self._get_decode_pipeline_input(model_output)
        decoded_output = self._run_pipeline(
            pipeline=self.decode_pipeline,
            input=pipeline_input,
            batch=self.batch_outputs,
        )
        waveform = decoded_output.get("wav")
        if waveform is None:
            raise ValueError("The output pipeline did not output a waveform")
        if single:
            waveform = waveform[0]
        return waveform

    def _run_pipeline(self, pipeline, input, batch):
        if batch:
            output = pipeline(input)
        else:
            output = [pipeline(item) for item in input]
        return output

    def _get_encode_pipeline_input(self, text):
        pipeline_input = {"txt": text}
        if not self.batch_inputs:
            pipeline_input = self._itemize(pipeline_input)
        return pipeline_input

    def _get_decode_pipeline_input(self, model_output):
        model_output_keys = getattr(self.hparams, "model_output_keys", None)
        pipeline_input = model_output
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

    def _to_dict(self, data):
        if isinstance(data, PaddedBatch):
            data = {
                key: getattr(data, key).data
                for key in self.hparams.encode_pipeline["output_keys"]
            }
        return data

    def _to_device(self, data):
        return {
            key: value.to(self.device)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in data.items()
        }

    @property
    def batch_inputs(self):
        """Determines whether the input pipeline
        operates on batches or individual examples
        (true means batched)

        Returns
        -------
        batch_intputs: bool
        """
        return self.hparams.encode_pipeline.get("batch", True)

    @property
    def batch_outputs(self):
        """Determines whether the output pipeline
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

    def __call__(self, text):
        """Calls tts(text)
        """
        return self.tts(text)

    def compute_forward(self, data):
        """Computes the forward pass of the model. This method can be overridden
        in the implementation, if needed

        Arguments
        ---------
        data
            the raw inputs to the model

        Returns
        -------
        The raw output of the model (the exact output
        depends on the implementation)
        """
        data_dict = self._to_dict(data)
        data_dict = self._to_device(data_dict)
        return self.infer(model=self.modules.model, **data_dict)

    def _transfer_external(self):
        """Transfers all external models to the same device as this
        model"""

        external_models = getattr(self.hparams, "external", {})
        for model in external_models.values():
            model.to(self.device)


class RepositoryCloneError(Error):
    """Exception thrown when a repository cannot be cloned

    Arguments
    ---------
    repo_path: str
        the path to the Git repository
    local_path: str
        the path to the local directory to which the repository
        was going to be cloned
    returncode: int
        the return code from the external command
    output: str
        the command output
    err: str
        the error output
    """

    def __init__(self, repo, local_path, returncode, output, err):
        self.repo = repo
        self.local_path = local_path
        self.returncode = returncode
        self.output = output
        self.err = err
        super().__init__(self._format_message())

    def _format_message(self):
        return f"""Unable to clone {self.repo} into {self.local_path}
Return code {self.returncode}
Standard Output:
{self.output}

Errors:
{self.err}"""


class ExternalModelSourceError(Exception):
    def __init__(self, src):
        self.src = src
        super().__init__(f"External model source not found at {src}")


class ExternalModelWrapper(Pretrained):
    """A Pretrainer wrapper for pretrained third-party models that
    are not part of SpeechBrain but are written in Python. This
    class is meant to be used as a base class for wrappers for specific
    models
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ready = False

    def _ensure_ready(self):
        """Ensures that the model is ready, loads it
           if it is not"""
        if not self.ready:
            self.load()
            self.ready = True

    def load(self):
        """Downloads the source code (if available)"""
        git_repo = getattr(self.hparams, "git_repo", None)
        if git_repo and not os.path.exists(self.hparams.model_src):
            self._clone()
        if not os.path.exists(self.hparams.model_src):
            raise ExternalModelSourceError(self.hparams.model_src)
        if self.hparams.model_src not in sys.path:
            sys.path.append(self.hparams.model_src)
        self._download_files()
        self._load_model()

    def _load_model(self):
        if hasattr(self.hparams, "model"):
            self._load_model_module()
        elif hasattr(self.hparams, "model_file"):
            self._load_model_file()
        if isinstance(self.model, torch.nn.Module):
            self.model = self.model.to(self.device)
        self.adapter = getattr(self.hparams, "adapter", None)
        if self.adapter is None:
            self.adapter = lambda model, *args, **kwargs: model(*args, **kwargs)

    def _load_model_module(self):
        model_components = self.hparams.model.split(".")
        model_module = ".".join(model_components[:-1])
        model_obj = model_components[-1]
        self.model_module = import_module(model_module)
        model = getattr(self.model_module, model_obj)
        self.model = model

    def _load_model_file(self):
        model = torch.load(self.hparams.model_file)
        model_key = getattr(self.hparams, "model_key", None)
        if model_key:
            model = model[model_key]
        self.model = model

    def _download_files(self):
        downloads = getattr(self.hparams, "downloads")
        if downloads:
            logger.info("Downloading files")
            for download in downloads:
                downloader = download["downloader"]
                downloader.check_prerequisites()
                existence = {
                    file["destination"]: os.path.exists(file["destination"])
                    for file in download["files"]
                }
                for file_path, exists in existence.items():
                    if exists:
                        logging.info("File %s already exists", file_path)
                missing_files = [
                    file
                    for file in download["files"]
                    if not existence[file["destination"]]
                ]
                downloader.download(missing_files)

    def _clone(self):
        logger.info("Cloning %s into %s")
        clone_cmd = (
            f"git clone {shlex.quote(self.hparams.git_repo)} "
            f"{shlex.quote(self.hparams.model_src)}"
        )
        output, err, returncode = run_shell(clone_cmd)
        if returncode != 0:
            raise RepositoryCloneError(
                repo=self.hparam.git_repo,
                local_path=self.hparams.model_src,
                returncode=returncode,
                output=output,
                err=err,
            )
        if hasattr(self.hparams, "git_revision"):
            self._get_revision()

    def _get_revision(self, revision):
        co_cmd = f"git checkout -C {shlex.quote(self.hparams.model_src)} checkout {shlex.quote(self.hparams.git_revision)}"
        output, err, returncode = run_shell(co_cmd)
        if returncode != 0:
            raise RepositoryCloneError(
                repo=self.hparam.git_repo,
                local_path=self.hparams.model_src,
                returncode=returncode,
                output=output,
                err=err,
            )

    def __call__(self, *args, **kwargs):
        """Calls the underlying model, loading it if it has not been already
        loaded. Any arguments will be passed through 'as is'"""
        self._ensure_ready()
        return self.adapter(self.model, *args, **kwargs)

    @classmethod
    def from_hparams(
        cls, source=None, hparams_file=None, overrides={}, **kwargs
    ):
        """Fetch and load from external source based on HyperPyYAML file

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
        """
        if not source:
            source = "."
        if not hparams_file:
            hparams_file = "hyperparams.yaml"
        if not os.path.isabs(hparams_file):
            hparams_file = os.path.join(source, hparams_file)
        if not os.path.exists(hparams_file):
            raise ValueError(f"Unable to find hparams file {hparams_file}")
        with open(hparams_file) as fin:
            hparams = load_hyperpyyaml(fin, overrides)
        return cls(hparams=hparams, **kwargs)

    def to(self, device):
        """
        Transfers this model to another device

        Arguments
        ---------
        device: str or torch device
            the torch device
        """
        self._ensure_ready()
        self.device = device
        self.model = self.model.to(device)


class VocoderWrapper(ExternalModelWrapper):
    def synthesize(self, inputs):
        """Synthesizes audio (usually speech) from inputs (usually a spectrogram)

        Arguments
        ---------
        inputs: torch.Tensor
            Vocoder inputs
        """
        return self.__call__(inputs)
