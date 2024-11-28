"""This lobe enables the integration of huggingface pretrained SpeechT5 models for Automatic Speech Recognition.
Reference: https://aclanthology.org/2022.acl-long.393.pdf

Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Haroun Elleuch 2024
"""

import gc
import logging
from typing import Tuple

import torch
from transformers import (
    GenerationConfig,
    SpeechT5Config,
    SpeechT5ForSpeechToText,
)

from speechbrain.lobes.models.huggingface_transformers.huggingface import (
    HFTransformersInterface,
)
from speechbrain.utils.checkpoints import (
    mark_as_loader,
    mark_as_saver,
    register_checkpoint_hooks,
)
from speechbrain.utils.fetching import fetch

logger = logging.getLogger(__name__)


@register_checkpoint_hooks
class SpeechT5ForASR(HFTransformersInterface):
    """This lobe enables the integration of HuggingFace and SpeechBrain
    pretrained SpeechT5 models for Automatic Speech Recognition.

    Source paper SpeechT5: https://aclanthology.org/2022.acl-long.393.pdf
    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html

    It will download automatically the model from HuggingFace or use a local path.

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "microsoft/speecht5_asr"
    save_path : str
        save directory of the downloaded model.
    cache_dir: str
        cache directory used by HuggingFace transformers.
    freeze : bool (default: True)
        If `True`, the model is frozen. If `False`, the model will be trained
        alongside with the rest of the pipeline.
    freeze_encoder : bool (default: False)
        If `True`, the model encoder is frozen. If the model is set to encoder only,
        using the freeze parameter will also freeze the encoder.
    freeze_feature_extractor: bool (default: False)
        Whether or not to freeze the feature extractor of SpeechT5.
    sampling_rate: int (default: 16000)
        Sampling rate of the audio inputs.
    encoder_only : bool (default: False)
        If `True`, will only load the (speech) encoder of the SpeechT5 model.
    output_attentions: bool (default: True)
        Whether to output the attentions from the encoder.
    output_all_hiddens: bool (default: True)
        Whether or not to output all the hidden states from the encoder. They will
        be stacked if returned.
    device : any, optional
        Device to migrate the model to.
    *args : tuple
        Variable-length positional arguments. Each argument can be of any type.
    **kwargs : dict
        Variable-length keyword arguments. Each argument name is a string, and
        each value can be of any type.

    Example
    -------
    >>> model_hub = "microsoft/speecht5_asr"
    >>> save_path = "tmp"
    >>> model = SpeechT5ForASR(model_hub, save_path=save_path) # doctest: +SKIP
    """

    def __init__(
        self,
        source: str,
        save_path: str = "",
        cache_dir: str = "",
        freeze: bool = False,
        freeze_encoder: bool = False,
        freeze_feature_extractor: bool = False,
        sampling_rate: int = 16_000,
        encoder_only: bool = False,
        output_attentions: bool = False,
        output_all_hiddens: bool = False,
        device=None,
        *args,
        **kwargs,
    ) -> None:

        if encoder_only and freeze:
            freeze_encoder = True

        if encoder_only and freeze_encoder:
            freeze = True

        super().__init__(
            source=source,
            save_path=save_path,
            freeze=freeze,
            cache_dir=cache_dir,
            device=device,
            *args,
            **kwargs,
        )

        self.sampling_rate = sampling_rate
        self.encoder_only = encoder_only
        self.freeze_encoder = freeze_encoder
        self.freeze_feature_extractor = freeze_feature_extractor
        self.output_attentions = output_attentions
        self.output_all_hiddens = output_all_hiddens
        self.generation_config = None

        self.load_tokenizer(source=source, **kwargs)

        config = SpeechT5Config.from_pretrained(
            source, cache_dir=cache_dir, **kwargs
        )
        self.config = self.override_config(config)

        if self.encoder_only:
            logger.warning(
                f"{self.__class__.__name__}: Using an encoder-only SpeechT5 for speech to text."
            )
            del self.model.speecht5.decoder
            del self.model.text_decoder_postnet

        if self.freeze_feature_extractor:
            self.model.freeze_feature_encoder()

    def forward(self, wav, decoder_input_ids=None):
        """
        One forward step of the SpeechT5 model ** using the manually implemented methods ** by calling
        the `forward_encoder()` and `forward_decoder()` methods.

        Arguments
        ----------
            wav (raw waveform to be fed to the model.): torch.Tensor or numpy.ndarray
            decoder_input_ids (If using a decoder, those would be the tokenized text targets.): list

        Returns
        -------
            Encoder-only: Either the encoder hidden layers (last or all internal stacked if `self.output_all_hiddens` is `True`)
            Encoder-Decoder: The encoder hidden states (see encoder-only), the logits, and the attentions
        """

        encoder_output = self.forward_encoder(wav)
        if self.output_all_hiddens:
            encoder_output = encoder_output[-1]

        if self.encoder_only:
            return encoder_output

        # In the same fashion as mBART:
        # We replace 0 elements by pax_idx as pad_idx of SpeechT5 is 1
        decoder_input_ids = custom_padding(
            x=decoder_input_ids,
            org_pad=0,
            custom_pad=self.model.speecht5.decoder.config.pad_token_id,
        )

        if self.freeze:
            with torch.no_grad():
                logits, attentions = self.forward_decoder(
                    audio_features=encoder_output,
                    decoder_input_ids=decoder_input_ids,
                )
        else:
            logits, attentions = self.forward_decoder(
                audio_features=encoder_output,
                decoder_input_ids=decoder_input_ids,
            )

        return encoder_output, logits, attentions

    def forward_encoder(self, wav) -> torch.Tensor:
        """Forward step of the SpeechT5 encoder.

        Arguments
        ----------
            wav : torch.Tensor
                Input tensor.

        Returns
        -------
            torch.Tensor
                Encoder embeddings.
        """
        if self.output_all_hiddens:
            if self.freeze_encoder:
                with torch.no_grad():
                    states = self.model.speecht5.encoder(
                        input_values=wav,
                        output_hidden_states=self.output_all_hiddens,
                    )
            else:
                states = self.model.speecht5.encoder(
                    input_values=wav,
                    output_hidden_states=self.output_all_hiddens,
                )
            return torch.stack(states.hidden_states)

        else:
            if self.freeze_encoder:
                with torch.no_grad():
                    states = self.model.speecht5.encoder(
                        input_values=wav,
                        output_hidden_states=self.output_all_hiddens,
                    )
            else:
                states = self.model.speecht5.encoder(
                    input_values=wav,
                    output_hidden_states=self.output_all_hiddens,
                )
            return states.last_hidden_state

    def forward_decoder(
        self, audio_features, decoder_input_ids
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one step of the SpeechT5 decoder.
        For more details or go to theseq2seq2.py file in SpeechBrain to see how to generate
        the tokens with Greedy Search and/or Beam Search.

        Arguments
        ---------
        audio_features : torch.Tensor
            A batch of audio features (SpeechT5 encoding).
        decoder_input_ids : torch.Tensor
            A batch of decoder inputs tokens.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            logits: output of the model's text decoder post-net
            attentions: decoder self-attentions
        """

        output_states = self.model.speecht5.decoder(
            encoder_hidden_states=audio_features,
            input_values=decoder_input_ids,
            output_attentions=self.output_attentions,
        )

        if self.output_attentions:
            attentions = output_states.attentions[-1]
            attentions = attentions.view(
                attentions.shape[0] * attentions.shape[1], *attentions.shape[2:]
            )
        else:
            attentions = None

        logits = self.model.text_decoder_postnet(
            output_states.last_hidden_state
        )

        return logits, attentions

    @torch.no_grad()
    def decode(
        self, tgt, encoder_out, enc_len=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """This method implements a decoding step for the transformer model.

        Arguments
        ---------
        tgt : torch.Tensor
            The sequence to the decoder.
        encoder_out : torch.Tensor
            Hidden output of the encoder.
        enc_len : torch.LongTensor, optional
            Length of encoder states, by default None

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            _description_
        """

        if tgt.dtype not in [torch.long, torch.int64]:
            tgt = tgt.long()

        tgt_mask = torch.ones(tgt.size(), device=tgt.device)

        output = self.model.speecht5.decoder(
            encoder_hidden_states=encoder_out,
            input_ids=tgt,
            attention_mask=tgt_mask,
            output_attentions=self.output_attentions,
        )

        return (
            self.model.text_decoder_postnet(output.last_hidden_state),
            output.cross_attentions[-1],
        )

    def _from_pretrained(self, source: str, save_path: str, cache_dir: str):
        """This methods loads a SpeechT5 checkpoint according to its origin.

        Arguments
        ----------
        source : str
            Used as a source for the model. can be either a local path or a Huggingface directory.
        save_path : str
            Location of the saved checkpoint.
        cache_dir : str
            Cache directory.
        """

        is_sb, ckpt_file, _ = self._check_model_source(source, save_path)

        if is_sb or self.for_pretraining:
            self.model = SpeechT5ForSpeechToText._from_config(self.config)

        if is_sb:
            self.model.gradient_checkpointing_disable()  # Required by DDP
            # fetch the checkpoint file
            ckpt_full_path = fetch(
                filename=ckpt_file,
                source=source,
                savedir=save_path,
                huggingface_cache_dir=cache_dir,
            )
            # We transfer the parameters from the checkpoint.
            self._load_sb_pretrained_parameters(
                path=ckpt_full_path,
            )
        elif not self.for_pretraining:
            self.model = SpeechT5ForSpeechToText.from_pretrained(
                source,
                config=self.config,
                cache_dir=cache_dir,
                quantization_config=self.quantization_config,
                ignore_mismatched_sizes=True,
            )

    def _free_memory(self):
        """Frees the memory by deleting the model."""
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_generation_config(self) -> GenerationConfig:
        """Returns the HuggingFace Transformers generation config associated with SpeechT5.
        If the generation config is None, it will create a new one.

        Returns
        -------
        GenerationConfig
            Model generation config.
        """
        if self.generation_config is None:
            self.generation_config = GenerationConfig(
                decoder_start_token_id=2,
                eos_token_id=self.model.config.eos_token_id,
                pad_token=self.model.config.pad_token_id,
                max_length=450,
            )

        return self.generation_config

    @mark_as_loader
    def load_checkpoint_hf(self, path, end_of_epoch) -> None:
        """Custom checkpoint loading hook used to avoid the model size mismatch when using Pytorch's format.
        This function will call the transformers library implementations of `from_pretrained()` to load :
            - The model
            - The model configuration
            - The generation configuration

        Arguments
        ----------
        path : os.PathLike or str
            Path of the checkpoint directory.
        end_of_epoch : bool
            Whether or not the checkpoint was saved duraing the end of an epoch.
        """
        self._free_memory()
        self.config = SpeechT5Config.from_pretrained(path)

        if torch.cuda.is_available():
            self.model = SpeechT5ForSpeechToText.from_pretrained(
                path,
                config=self.config,
                local_files_only=True,
                ignore_mismatched_sizes=True,
                use_safetensors=True,
            ).cuda()
        else:
            self.model = SpeechT5ForSpeechToText.from_pretrained(
                path,
                config=self.config,
                local_files_only=True,
                ignore_mismatched_sizes=True,
                use_safetensors=True,
            )

        self.generation_config = GenerationConfig.from_pretrained(
            path, "generation_config.json"
        )

    @mark_as_saver
    def save_checkpoint_hf(self, path):
        """Custom checkpoint saving hook used to avoid the model size mismatch when using Pytorch's format.
        This function will call the transformers library implementationsof `save_pretrained()` to save :
            - The model and its configuration
            - The generation configuration

        Arguments
        ----------
        path :
            Path where the checkpoint will be saved
        """
        self.model.save_pretrained(
            path, from_pt=True, state_dict=self.model.state_dict()
        )
        self.get_generation_config().save_pretrained(
            path, "generation_config.json"
        )

    def _modify_state_dict(self, path):
        """A custom loading ensures SpeechBrain compatibility for Pretrain and model
        de/serialization. Here, the scope is to remove '.model' before loading.

        Arguments
        ---------
        path : str
            Checkpoint path, file name relative to the repo root.

        Returns
        -------
        modified_state_dict : see torch.load
            SpeechBrain-valid deserialized pretrained model.
        """
        modified_state_dict = {}
        orig_state_dict = torch.load(path, map_location="cpu")

        # We remove the .model in the state dict.
        for key, params in orig_state_dict.items():
            save_key = key.replace("model.", "")
            modified_state_dict[save_key] = params
        return modified_state_dict


def custom_padding(x, org_pad, custom_pad):
    """
    Copied from the mBART integration in Speechbrain.
    This function customizes the padding.
    Default pad_idx of SpeechBrain is 0.
    However, it happens that some text-based models like mBART reserves 0 for something else,
    and are trained with specific pad_idx.
    This function change org_pad to custom_pad

    Arguments
    ---------
    x : torch.Tensor
      Input tensor with original pad_idx
    org_pad : int
      Original pad_idx
    custom_pad : int
      Custom pad_idx

    Returns
    -------
    torch.Tensor:
        Padded input tensor.
    """
    out = x.clone()
    out[x == org_pad] = custom_pad

    return out
