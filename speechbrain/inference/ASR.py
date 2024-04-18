""" Specifies the inference interfaces for Automatic speech Recognition (ASR) modules.

Authors:
 * Aku Rouhe 2021
 * Peter Plantinga 2021
 * Loren Lugosch 2020
 * Mirco Ravanelli 2020
 * Titouan Parcollet 2021
 * Abdel Heba 2021
 * Andreas Nautsch 2022, 2023
 * Pooneh Mousavi 2023
 * Sylvain de Langen 2023, 2024
 * Adel Moumen 2023, 2024
 * Pradnya Kandarkar 2023
"""

import functools
import itertools
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import sentencepiece
import torch
import torchaudio

import speechbrain
from speechbrain.inference.interfaces import Pretrained
from speechbrain.utils.data_utils import split_path
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig
from speechbrain.utils.fetching import fetch
from speechbrain.utils.streaming import split_fixed_chunks


class EncoderDecoderASR(Pretrained):
    """A ready-to-use Encoder-Decoder ASR model

    The class can be used either to run only the encoder (encode()) to extract
    features or to run the entire encoder-decoder model
    (transcribe()) to transcribe speech. The given YAML must contain the fields
    specified in the *_NEEDED[] lists.

    Arguments
    ---------
    *args : tuple
    **kwargs : dict
        Arguments are forwarded to ``Pretrained`` parent class.

    Example
    -------
    >>> from speechbrain.inference.ASR import EncoderDecoderASR
    >>> tmpdir = getfixture("tmpdir")
    >>> asr_model = EncoderDecoderASR.from_hparams(
    ...     source="speechbrain/asr-crdnn-rnnlm-librispeech",
    ...     savedir=tmpdir,
    ... )  # doctest: +SKIP
    >>> asr_model.transcribe_file("tests/samples/single-mic/example2.flac")  # doctest: +SKIP
    "MY FATHER HAS REVEALED THE CULPRIT'S NAME"
    """

    HPARAMS_NEEDED = ["tokenizer"]
    MODULES_NEEDED = ["encoder", "decoder"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = self.hparams.tokenizer
        self.transducer_beam_search = False
        self.transformer_beam_search = False
        if hasattr(self.hparams, "transducer_beam_search"):
            self.transducer_beam_search = self.hparams.transducer_beam_search
        if hasattr(self.hparams, "transformer_beam_search"):
            self.transformer_beam_search = self.hparams.transformer_beam_search

    def transcribe_file(self, path, **kwargs):
        """Transcribes the given audiofile into a sequence of words.

        Arguments
        ---------
        path : str
            Path to audio file which to transcribe.
        **kwargs : dict
            Arguments forwarded to ``load_audio``.

        Returns
        -------
        str
            The audiofile transcription produced by this ASR system.
        """
        waveform = self.load_audio(path, **kwargs)
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
        if self.transformer_beam_search:
            encoder_out = self.mods.transformer.encode(encoder_out, wav_lens)
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
            if self.transducer_beam_search:
                inputs = [encoder_out]
            else:
                inputs = [encoder_out, wav_lens]
            predicted_tokens, _, _, _ = self.mods.decoder(*inputs)
            predicted_words = [
                self.tokenizer.decode_ids(token_seq)
                for token_seq in predicted_tokens
            ]
        return predicted_words, predicted_tokens

    def forward(self, wavs, wav_lens):
        """Runs full transcription - note: no gradients through decoding"""
        return self.transcribe_batch(wavs, wav_lens)


class EncoderASR(Pretrained):
    """A ready-to-use Encoder ASR model

    The class can be used either to run only the encoder (encode()) to extract
    features or to run the entire encoder + decoder function model
    (transcribe()) to transcribe speech. The given YAML must contain the fields
    specified in the *_NEEDED[] lists.

    Arguments
    ---------
    *args : tuple
    **kwargs : dict
        Arguments are forwarded to ``Pretrained`` parent class.

    Example
    -------
    >>> from speechbrain.inference.ASR import EncoderASR
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
        self.set_decoding_function()

    def set_decoding_function(self):
        """Set the decoding function based on the parameters defined in the hyperparameter file.

        The decoding function is determined by the `decoding_function` specified in the hyperparameter file.
        It can be either a functools.partial object representing a decoding function or an instance of
        `speechbrain.decoders.ctc.CTCBaseSearcher` for beam search decoding.

        Raises:
            ValueError: If the decoding function is neither a functools.partial nor an instance of
                        speechbrain.decoders.ctc.CTCBaseSearcher.

        Note:
            - For greedy decoding (functools.partial), the provided `decoding_function` is assigned directly.
            - For CTCBeamSearcher decoding, an instance of the specified `decoding_function` is created, and
            additional parameters are added based on the tokenizer type.
        """
        # Greedy Decoding case
        if isinstance(self.hparams.decoding_function, functools.partial):
            self.decoding_function = self.hparams.decoding_function
        # CTCBeamSearcher case
        else:
            # 1. check if the decoding function is an instance of speechbrain.decoders.CTCBaseSearcher
            if issubclass(
                self.hparams.decoding_function,
                speechbrain.decoders.ctc.CTCBaseSearcher,
            ):
                # If so, we need to retrieve the vocab list from the tokenizer.
                # We also need to check if the tokenizer is a sentencepiece or a CTCTextEncoder.
                if isinstance(
                    self.tokenizer, speechbrain.dataio.encoder.CTCTextEncoder
                ):
                    ind2lab = self.tokenizer.ind2lab
                    vocab_list = [ind2lab[x] for x in range(len(ind2lab))]
                elif isinstance(
                    self.tokenizer, sentencepiece.SentencePieceProcessor
                ):
                    vocab_list = [
                        self.tokenizer.id_to_piece(i)
                        for i in range(self.tokenizer.vocab_size())
                    ]
                else:
                    raise ValueError(
                        "The tokenizer must be sentencepiece or CTCTextEncoder"
                    )

                # We can now instantiate the decoding class and add all the parameters
                if hasattr(self.hparams, "test_beam_search"):
                    opt_beam_search_params = self.hparams.test_beam_search
                    # check if the kenlm_model_path is provided and fetch it if necessary
                    if "kenlm_model_path" in opt_beam_search_params:
                        source, fl = split_path(
                            opt_beam_search_params["kenlm_model_path"]
                        )
                        kenlm_model_path = str(
                            fetch(fl, source=source, savedir=".")
                        )
                        # we need to update the kenlm_model_path in the opt_beam_search_params
                        opt_beam_search_params["kenlm_model_path"] = (
                            kenlm_model_path
                        )
                else:
                    opt_beam_search_params = {}
                self.decoding_function = self.hparams.decoding_function(
                    **opt_beam_search_params, vocab_list=vocab_list
                )
            else:
                raise ValueError(
                    "The decoding function must be an instance of speechbrain.decoders.CTCBaseSearcher"
                )

    def transcribe_file(self, path, **kwargs):
        """Transcribes the given audiofile into a sequence of words.

        Arguments
        ---------
        path : str
            Path to audio file which to transcribe.
        **kwargs : dict
            Arguments forwarded to ``load_audio``.

        Returns
        -------
        str
            The audiofile transcription produced by this ASR system.
        """
        waveform = self.load_audio(path, **kwargs)
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
            is_ctc_text_encoder_tokenizer = isinstance(
                self.tokenizer, speechbrain.dataio.encoder.CTCTextEncoder
            )
            if isinstance(self.hparams.decoding_function, functools.partial):
                if is_ctc_text_encoder_tokenizer:
                    predicted_words = [
                        "".join(self.tokenizer.decode_ndim(token_seq))
                        for token_seq in predictions
                    ]
                else:
                    predicted_words = [
                        self.tokenizer.decode_ids(token_seq)
                        for token_seq in predictions
                    ]
            else:
                predicted_words = [hyp[0].text for hyp in predictions]

        return predicted_words, predictions

    def forward(self, wavs, wav_lens):
        """Runs the encoder"""
        return self.encode_batch(wavs, wav_lens)


class WhisperASR(Pretrained):
    """A ready-to-use Whisper ASR model

    The class can be used  to  run the entire encoder-decoder whisper model
    (transcribe()) to transcribe speech. The given YAML must contains the fields
    specified in the *_NEEDED[] lists.

    Arguments
    ---------
    *args : tuple
    **kwargs : dict
        Arguments are forwarded to ``Pretrained`` parent class.

    Example
    -------
    >>> from speechbrain.inference.ASR import WhisperASR
    >>> tmpdir = getfixture("tmpdir")
    >>> asr_model = WhisperASR.from_hparams(source="speechbrain/asr-whisper-medium-commonvoice-it", savedir=tmpdir,) # doctest: +SKIP
    >>> asr_model.transcribe_file("speechbrain/asr-whisper-medium-commonvoice-it/example-it.wav")  # doctest: +SKIP
    """

    HPARAMS_NEEDED = ["language"]
    MODULES_NEEDED = ["whisper", "decoder"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = self.hparams.whisper.tokenizer
        self.tokenizer.set_prefix_tokens(
            self.hparams.language, "transcribe", False
        )
        self.hparams.decoder.set_decoder_input_tokens(
            self.tokenizer.prefix_tokens
        )

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
        return " ".join(predicted_words[0])

    def encode_batch(self, wavs, wav_lens):
        """Encodes the input audio into a sequence of hidden states

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = EncoderDecoderASR.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

        Arguments
        ---------
        wavs : torch.tensor
            Batch of waveforms [batch, time, channels].
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
        encoder_out = self.mods.whisper.forward_encoder(wavs)
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
            Batch of waveforms [batch, time, channels].
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
            predicted_tokens, _, _, _ = self.mods.decoder(encoder_out, wav_lens)
            predicted_words = self.tokenizer.batch_decode(
                predicted_tokens, skip_special_tokens=True
            )
            if self.hparams.normalized_transcripts:
                predicted_words = [
                    self.tokenizer._normalize(text).split(" ")
                    for text in predicted_words
                ]

        return predicted_words, predicted_tokens

    def forward(self, wavs, wav_lens):
        """Runs full transcription - note: no gradients through decoding"""
        return self.transcribe_batch(wavs, wav_lens)


@dataclass
class ASRStreamingContext:
    """Streaming metadata, initialized by
    :meth:`~StreamingASR.make_streaming_context` (see there for details on
    initialization of fields here).

    This object is intended to be mutate: the same object should be passed
    across calls as streaming progresses (namely when using the lower-level
    :meth:`~StreamingASR.encode_chunk`, etc. APIs).

    Holds some references to opaque streaming contexts, so the context is
    model-agnostic to an extent."""

    config: DynChunkTrainConfig
    """Dynamic chunk training configuration used to initialize the streaming
    context. Cannot be modified on the fly."""

    fea_extractor_context: Any
    """Opaque feature extractor streaming context."""

    encoder_context: Any
    """Opaque encoder streaming context."""

    decoder_context: Any
    """Opaque decoder streaming context."""

    tokenizer_context: Optional[List[Any]]
    """Opaque streaming context for the tokenizer. Initially `None`. Initialized
    to a list of tokenizer contexts once batch size can be determined."""


class StreamingASR(Pretrained):
    """A ready-to-use, streaming-capable ASR model.

    Arguments
    ---------
    *args : tuple
    **kwargs : dict
        Arguments are forwarded to ``Pretrained`` parent class.

    Example
    -------
    >>> from speechbrain.inference.ASR import StreamingASR
    >>> from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig
    >>> tmpdir = getfixture("tmpdir")
    >>> asr_model = StreamingASR.from_hparams(source="speechbrain/asr-conformer-streaming-librispeech", savedir=tmpdir,) # doctest: +SKIP
    >>> asr_model.transcribe_file("speechbrain/asr-conformer-streaming-librispeech/test-en.wav", DynChunkTrainConfig(24, 8)) # doctest: +SKIP
    """

    HPARAMS_NEEDED = [
        "fea_streaming_extractor",
        "make_decoder_streaming_context",
        "decoding_function",
        "make_tokenizer_streaming_context",
        "tokenizer_decode_streaming",
    ]
    MODULES_NEEDED = ["enc", "proj_enc"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.filter_props = self.hparams.fea_streaming_extractor.properties

    def _get_audio_stream(
        self, streamer: torchaudio.io.StreamReader, frames_per_chunk: int
    ):
        """From a :class:`torchaudio.io.StreamReader`, identifies the audio
        stream and returns an iterable stream of chunks (after resampling and
        downmixing to mono).

        Arguments
        ---------
        streamer : torchaudio.io.StreamReader
            The stream object. Must hold exactly one source stream of an
            audio type.
        frames_per_chunk : int
            The number of frames per chunk. For a streaming model, this should
            be determined from the DynChunkTrain configuration.

        Yields
        ------
        chunks from streamer
        """

        stream_infos = [
            streamer.get_src_stream_info(i)
            for i in range(streamer.num_src_streams)
        ]

        audio_stream_infos = [
            (i, stream_info)
            for i, stream_info in enumerate(stream_infos)
            if stream_info.media_type == "audio"
        ]

        if len(audio_stream_infos) != 1:
            raise ValueError(
                f"Expected stream to have only 1 stream (with any number of channels), got {len(audio_stream_infos)} (with streams: {stream_infos})"
            )

        # find the index of the first (and only) audio stream
        audio_stream_index = audio_stream_infos[0][0]

        # output stream #0
        streamer.add_basic_audio_stream(
            frames_per_chunk=frames_per_chunk,
            stream_index=audio_stream_index,
            sample_rate=self.audio_normalizer.sample_rate,
            format="fltp",  # torch.float32
            num_channels=1,
        )

        for (chunk,) in streamer.stream():
            chunk = chunk.squeeze(-1)  # we deal with mono, remove that dim
            chunk = chunk.unsqueeze(0)  # create a fake batch dim
            yield chunk

    def transcribe_file_streaming(
        self,
        path,
        dynchunktrain_config: DynChunkTrainConfig,
        use_torchaudio_streaming: bool = True,
        **kwargs,
    ):
        """Transcribes the given audio file into a sequence of words, in a
        streaming fashion, meaning that text is being yield from this
        generator, in the form of strings to concatenate.

        Arguments
        ---------
        path : str
            URI/path to the audio to transcribe. When
            ``use_torchaudio_streaming`` is ``False``, uses SB fetching to allow
            fetching from HF or a local file. When ``True``, resolves the URI
            through ffmpeg, as documented in
            :class:`torchaudio.io.StreamReader`.
        dynchunktrain_config : DynChunkTrainConfig
            Streaming configuration. Sane values and how much time chunks
            actually represent is model-dependent.
        use_torchaudio_streaming : bool
            Whether the audio file can be loaded in a streaming fashion. If not,
            transcription is still performed through chunks of audio, but the
            entire audio file is fetched and loaded at once.
            This skips the usual fetching method and instead resolves the URI
            using torchaudio (via ffmpeg).
        **kwargs : dict
            Arguments forwarded to ``load_audio``

        Yields
        ------
        generator of str
            An iterator yielding transcribed chunks (strings). There is a yield
            for every chunk, even if the transcribed string for that chunk is an
            empty string.
        """

        chunk_size = self.get_chunk_size_frames(dynchunktrain_config)

        if use_torchaudio_streaming:
            streamer = torchaudio.io.StreamReader(path)
            chunks = self._get_audio_stream(streamer, chunk_size)
        else:
            waveform = self.load_audio(path, **kwargs)
            batch = waveform.unsqueeze(0)  # create batch dim
            chunks = split_fixed_chunks(batch, chunk_size)

        rel_length = torch.tensor([1.0])
        context = self.make_streaming_context(dynchunktrain_config)

        final_chunks = [
            torch.zeros((1, chunk_size), device=self.device)
        ] * self.hparams.fea_streaming_extractor.get_recommended_final_chunk_count(
            chunk_size
        )

        for chunk in itertools.chain(chunks, final_chunks):
            predicted_words = self.transcribe_chunk(context, chunk, rel_length)
            yield predicted_words[0]

    def transcribe_file(
        self,
        path,
        dynchunktrain_config: DynChunkTrainConfig,
        use_torchaudio_streaming: bool = True,
    ):
        """Transcribes the given audio file into a sequence of words.

        Arguments
        ---------
        path : str
            URI/path to the audio to transcribe. When
            ``use_torchaudio_streaming`` is ``False``, uses SB fetching to allow
            fetching from HF or a local file. When ``True``, resolves the URI
            through ffmpeg, as documented in
            :class:`torchaudio.io.StreamReader`.
        dynchunktrain_config : DynChunkTrainConfig
            Streaming configuration. Sane values and how much time chunks
            actually represent is model-dependent.
        use_torchaudio_streaming : bool
            Whether the audio file can be loaded in a streaming fashion. If not,
            transcription is still performed through chunks of audio, but the
            entire audio file is fetched and loaded at once.
            This skips the usual fetching method and instead resolves the URI
            using torchaudio (via ffmpeg).

        Returns
        -------
        str
            The audio file transcription produced by this ASR system.
        """

        pred = ""

        for text_chunk in self.transcribe_file_streaming(
            path, dynchunktrain_config, use_torchaudio_streaming
        ):
            pred += text_chunk

        return pred

    def make_streaming_context(self, dynchunktrain_config: DynChunkTrainConfig):
        """Create a blank streaming context to be passed around for chunk
        encoding/transcription.

        Arguments
        ---------
        dynchunktrain_config : DynChunkTrainConfig
            Streaming configuration. Sane values and how much time chunks
            actually represent is model-dependent.

        Returns
        -------
        ASRStreamingContext
        """

        return ASRStreamingContext(
            config=dynchunktrain_config,
            fea_extractor_context=self.hparams.fea_streaming_extractor.make_streaming_context(),
            encoder_context=self.mods.enc.make_streaming_context(
                dynchunktrain_config
            ),
            decoder_context=self.hparams.make_decoder_streaming_context(),
            tokenizer_context=None,
        )

    def get_chunk_size_frames(
        self, dynchunktrain_config: DynChunkTrainConfig
    ) -> int:
        """Returns the chunk size in actual audio samples, i.e. the exact
        expected length along the time dimension of an input chunk tensor (as
        passed to :meth:`~StreamingASR.encode_chunk` and similar low-level
        streaming functions).

        Arguments
        ---------
        dynchunktrain_config : DynChunkTrainConfig
            The streaming configuration to determine the chunk frame count of.

        Returns
        -------
        chunk size
        """

        return (self.filter_props.stride - 1) * dynchunktrain_config.chunk_size

    @torch.no_grad()
    def encode_chunk(
        self,
        context: ASRStreamingContext,
        chunk: torch.Tensor,
        chunk_len: Optional[torch.Tensor] = None,
    ):
        """Encoding of a batch of audio chunks into a batch of encoded
        sequences.
        For full speech-to-text offline transcription, use `transcribe_batch` or
        `transcribe_file`.
        Must be called over a given context in the correct order of chunks over
        time.

        Arguments
        ---------
        context : ASRStreamingContext
            Mutable streaming context object, which must be specified and reused
            across calls when streaming.
            You can obtain an initial context by calling
            `asr.make_streaming_context(config)`.

        chunk : torch.Tensor
            The tensor for an audio chunk of shape `[batch size, time]`.
            The time dimension must strictly match
            `asr.get_chunk_size_frames(config)`.
            The waveform is expected to be in the model's expected format (i.e.
            the sampling rate must be correct).

        chunk_len : torch.Tensor, optional
            The relative chunk length tensor of shape `[batch size]`. This is to
            be used when the audio in one of the chunks of the batch is ending
            within this chunk.
            If unspecified, equivalent to `torch.ones((batch_size,))`.

        Returns
        -------
        torch.Tensor
            Encoded output, of a model-dependent shape."""

        if chunk_len is None:
            chunk_len = torch.ones((chunk.size(0),))

        chunk = chunk.float()
        chunk, chunk_len = chunk.to(self.device), chunk_len.to(self.device)

        assert chunk.shape[-1] <= self.get_chunk_size_frames(context.config)

        x = self.hparams.fea_streaming_extractor(
            chunk, context=context.fea_extractor_context, lengths=chunk_len
        )
        x = self.mods.enc.forward_streaming(x, context.encoder_context)
        x = self.mods.proj_enc(x)
        return x

    @torch.no_grad()
    def decode_chunk(
        self, context: ASRStreamingContext, x: torch.Tensor
    ) -> Tuple[List[str], List[List[int]]]:
        """Decodes the output of the encoder into tokens and the associated
        transcription.
        Must be called over a given context in the correct order of chunks over
        time.

        Arguments
        ---------
        context : ASRStreamingContext
            Mutable streaming context object, which should be the same object
            that was passed to `encode_chunk`.

        x : torch.Tensor
            The output of `encode_chunk` for a given chunk.

        Returns
        -------
        list of str
            Decoded tokens of length `batch_size`. The decoded strings can be
            of 0-length.
        list of list of output token hypotheses
            List of length `batch_size`, each holding a list of tokens of any
            length `>=0`.
        """
        tokens = self.hparams.decoding_function(x, context.decoder_context)

        # initialize token context for real now that we know the batch size
        if context.tokenizer_context is None:
            context.tokenizer_context = [
                self.hparams.make_tokenizer_streaming_context()
                for _ in range(len(tokens))
            ]

        words = [
            self.hparams.tokenizer_decode_streaming(
                self.hparams.tokenizer, cur_tokens, context.tokenizer_context[i]
            )
            for i, cur_tokens in enumerate(tokens)
        ]

        return words, tokens

    def transcribe_chunk(
        self,
        context: ASRStreamingContext,
        chunk: torch.Tensor,
        chunk_len: Optional[torch.Tensor] = None,
    ):
        """Transcription of a batch of audio chunks into transcribed text.
        Must be called over a given context in the correct order of chunks over
        time.

        Arguments
        ---------
        context : ASRStreamingContext
            Mutable streaming context object, which must be specified and reused
            across calls when streaming.
            You can obtain an initial context by calling
            `asr.make_streaming_context(config)`.
        chunk : torch.Tensor
            The tensor for an audio chunk of shape `[batch size, time]`.
            The time dimension must strictly match
            `asr.get_chunk_size_frames(config)`.
            The waveform is expected to be in the model's expected format (i.e.
            the sampling rate must be correct).
        chunk_len : torch.Tensor, optional
            The relative chunk length tensor of shape `[batch size]`. This is to
            be used when the audio in one of the chunks of the batch is ending
            within this chunk.
            If unspecified, equivalent to `torch.ones((batch_size,))`.

        Returns
        -------
        str
            Transcribed string for this chunk, might be of length zero.
        """

        if chunk_len is None:
            chunk_len = torch.ones((chunk.size(0),))

        chunk = chunk.float()
        chunk, chunk_len = chunk.to(self.device), chunk_len.to(self.device)

        x = self.encode_chunk(context, chunk, chunk_len)
        words, _tokens = self.decode_chunk(context, x)

        return words
