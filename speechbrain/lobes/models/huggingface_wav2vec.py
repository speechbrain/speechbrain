"""This lobe enables the integration of huggingface pretrained wav2vec2/hubert/wavlm models.

Reference: https://arxiv.org/abs/2006.11477
Reference: https://arxiv.org/abs/1904.05862
Reference: https://arxiv.org/abs/2110.13900
Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Titouan Parcollet 2021
 * Boumadane Abdelmoumene 2021
"""

import os
import torch
import logging
import pathlib
import numpy as np
import torch.nn.functional as F
from torch import nn
from huggingface_hub import model_info
from speechbrain.pretrained.fetching import fetch
from speechbrain.dataio.dataio import length_to_mask

# We check if transformers is installed.
try:
    import transformers
    from transformers import AutoModel
    from transformers import Wav2Vec2Model, HubertModel, WavLMModel
    from transformers import Wav2Vec2Config, HubertConfig, WavLMConfig
    from transformers import Wav2Vec2FeatureExtractor
    from transformers import Wav2Vec2ForPreTraining
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        _compute_mask_indices,
    )

except ImportError:
    MSG = "Please install transformers from HuggingFace to use wav2vec2 / Hubert\n"
    MSG += "E.G. run: pip install transformers"
    raise ImportError(MSG)

logger = logging.getLogger(__name__)

HF_models = {
    "wav2vec2": Wav2Vec2Model,
    "hubert": HubertModel,
    "wavlm": WavLMModel,
}

HF_config = {
    "wav2vec2": Wav2Vec2Config,
    "hubert": HubertConfig,
    "wavlm": WavLMConfig,
}


class HuggingFaceWav2Vec2(nn.Module):
    """This lobe enables the integration of HuggingFace and SpeechBrain
    pretrained wav2vec2.0/Hubert models.

    Source paper wav2vec2.0: https://arxiv.org/abs/2006.11477
    Source paper Hubert: https://arxiv.org/abs/2106.07447
    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html

    The model can be used as a fixed feature extractor or can be finetuned. It
    will download automatically the model from HuggingFace or use a local path.

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "facebook/wav2vec2-large-lv60"
    save_path : str
        Path (dir) of the downloaded model.
    output_norm : bool (default: True)
        If True, a layer_norm (affine) will be applied to the output obtained
        from the wav2vec model.
    freeze : bool (default: True)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    freeze_feature_extractor :  bool (default: False)
        When freeze = False and freeze_feature_extractor True, the featue_extractor module of the model is Frozen. If False
        all the wav2vec model will be trained including featue_extractor module.
    apply_spec_augment : bool (default: False)
        If True, the model will apply spec augment on the output of feature extractor
        (inside huggingface Wav2VecModel() class).
        If False, the model will not apply spec augment. We set this to false to prevent from doing it twice.
    output_all_hiddens : bool (default: False)
        If True, the forward function outputs the hidden states from all transformer layers.
        For example wav2vec2-base has 12 transformer layers and the output is of shape (13, B, T, C),
        where a projection of the CNN output is added to the beginning.
        If False, the forward function outputs the hidden states only from the last transformer layer.

    Example
    -------
    >>> inputs = torch.rand([10, 600])
    >>> model_hub = "facebook/wav2vec2-base-960h"
    >>> save_path = "savedir"
    >>> model = HuggingFaceWav2Vec2(model_hub, save_path)
    >>> outputs = model(inputs)
    """

    def __init__(
        self,
        source,
        save_path,
        output_norm=False,
        freeze=False,
        freeze_feature_extractor=False,
        apply_spec_augment=False,
        output_all_hiddens=False,
    ):
        super().__init__()

        # Download the extractor from HuggingFace.
        # The extractor is only used to retrieve the normalisation information
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            source, cache_dir=save_path
        )

        # Select specific self-supervised loader (eg. Wav2Vec2, Hubert)
        if "hubert" in source:
            config = HF_config.get("hubert")
            model = HF_models.get("hubert")
        elif "wavlm" in source:
            config = HF_config.get("wavlm")
            model = HF_models.get("wavlm")
        else:
            config = HF_config.get("wav2vec2")
            model = HF_models.get("wav2vec2")

        # Download and load the model
        self._from_pretrained(
            source, config=config, model=model, save_path=save_path
        )

        self.model.config.apply_spec_augment = apply_spec_augment

        # We check if inputs need to be normalized w.r.t pretrained wav2vec2
        self.normalize_wav = self.feature_extractor.do_normalize

        self.freeze = freeze
        self.freeze_feature_extractor = freeze_feature_extractor
        self.output_norm = output_norm
        if self.freeze:
            logger.warning(
                "speechbrain.lobes.models.huggingface_wav2vec - wav2vec 2.0 is frozen."
            )
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model.train()
            if self.freeze_feature_extractor:
                logger.warning(
                    "speechbrain.lobes.models.huggingface_wav2vec - wav2vec 2.0 feature extractor is frozen."
                )
                self.model.feature_extractor.eval()
                for param in self.model.feature_extractor.parameters():
                    param.requires_grad = False
        self.output_all_hiddens = output_all_hiddens

    def _from_pretrained(self, source, config, model, save_path):
        """This function manages the source checking and loading of the params.
        # 1. Is the model from HF or a local path
        # 2. Is the model pretrained with HF or SpeechBrain
        # 3. Download (if appropriate) and load with respect to 1. and 2.
        """

        is_sb, ckpt_file, is_local = self._check_model_source(source, save_path)
        if is_sb:
            config = config.from_pretrained(source, cache_dir=save_path)
            self.model = model(config)
            self.model.gradient_checkpointing_disable()  # Required by DDP
            # fetch the checkpoint file
            ckpt_full_path = fetch(
                filename=ckpt_file, source=source, savedir=save_path
            )
            # We transfer the parameters from the checkpoint.
            self._load_sb_pretrained_w2v2_parameters(ckpt_full_path)
        else:
            self.model = model.from_pretrained(
                source, cache_dir=save_path, local_files_only=is_local
            )

    def _load_sb_pretrained_w2v2_parameters(self, path):
        """Loads the parameter of a w2v2 model pretrained with SpeechBrain and the
        HuggingFaceWav2Vec2Pretrain Object. It is necessary to perform a custom
        loading because HuggingFace adds a level to the checkpoint when storing
        the model breaking the compatibility between HuggingFaceWav2Vec2Pretrain
        and HuggingFaceWav2Vec2.

        In practice a typical HuggingFaceWav2Vec2 checkpoint for a given parameter
        would be: model.conv.weight.data while for HuggingFaceWav2Vec2Pretrain it
        is: model.wav2vec2.weight.data (wav2vec2 must be removed before loading).
        """

        modified_state_dict = {}
        orig_state_dict = torch.load(path, map_location="cpu")

        # We remove the .wav2vec2 in the state dict.
        for key, params in orig_state_dict.items():
            if "wav2vec2." in key:
                save_key = key.replace("model.wav2vec2.", "")
                modified_state_dict[save_key] = params

        incompatible_keys = self.model.load_state_dict(
            modified_state_dict, strict=False
        )
        for missing_key in incompatible_keys.missing_keys:
            logger.warning(
                f"During parameter transfer to {self.model} loading from "
                + f"{path}, the transferred parameters did not have "
                + f"parameters for the key: {missing_key}"
            )
        for unexpected_key in incompatible_keys.unexpected_keys:
            logger.warning(
                f"The param with the key: {unexpected_key} is discarded as it "
                + "is useless for wav2vec 2.0 finetuning."
            )

    def _check_model_source(self, path, save_path):
        """Checks if the pretrained model has been trained with SpeechBrain and
        is hosted locally or on a HuggingFace hub.
        Called as static function in HuggingFaceTransformer._from_pretrained.
        Arguments
        ---------
        path : str
            Used as "source"; local path or HuggingFace hub name: e.g "facebook/wav2vec2-large-lv60"
        save_path : str
            norm_output (dir) of the downloaded model.
        Returns
        -------
        is_sb : bool
            Whether/not the model is deserializable w/ SpeechBrain or not (then, model conversion is needed).
        checkpoint_filename : str
            as of HuggingFace documentation: file name relative to the repo root (guaranteed to be here).
        """
        checkpoint_filename = ""
        source = pathlib.Path(path)
        is_local = True

        # If path is a huggingface hub.
        if not source.exists():
            is_local = False

        # Check if source is downloaded already
        sink = pathlib.Path(
            save_path + "/models--" + path.replace("/", "--") + "/snapshots"
        )
        if sink.exists():
            sink = (
                sink / os.listdir(str(sink))[0]
            )  # there's a hash-id subfolder
            if any(
                File.endswith(".bin") or File.endswith(".ckpt")
                for File in os.listdir(str(sink))
            ):
                is_local = True
                local_path = str(sink)
            else:
                local_path = path
        else:
            local_path = path

        if is_local:
            # Test for HuggingFace model
            if any(File.endswith(".bin") for File in os.listdir(local_path)):
                is_sb = False
                return is_sb, checkpoint_filename, is_local

            # Test for SpeechBrain model and get the filename.
            for File in os.listdir(local_path):
                if File.endswith(".ckpt"):
                    checkpoint_filename = os.path.join(path, File)
                    is_sb = True
                    return is_sb, checkpoint_filename, is_local
        else:
            files = model_info(
                path
            ).siblings  # get the list of files of the Hub

            # Test if it's an HuggingFace model or a SB one
            for File in files:
                if File.rfilename.endswith(".ckpt"):
                    checkpoint_filename = File.rfilename
                    is_sb = True
                    return is_sb, checkpoint_filename, is_local

            for File in files:
                if File.rfilename.endswith(".bin"):
                    checkpoint_filename = File.rfilename
                    is_sb = False
                    return is_sb, checkpoint_filename, is_local

        err_msg = f"{path} does not contain a .bin or .ckpt checkpoint !"
        raise FileNotFoundError(err_msg)

    def forward(self, wav, wav_lens=None):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        wav_len : tensor
            The relative length of the wav given in SpeechBrain format.
        """

        # If we freeze, we simply remove all grads from the graph.
        if self.freeze:
            with torch.no_grad():
                return self.extract_features(wav, wav_lens)

        return self.extract_features(wav, wav_lens)

    def extract_features(self, wav, wav_lens=None):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        wav_len : tensor
            The relative length of the wav given in SpeechBrain format.
        """

        padding_mask = self.make_masks(wav, wav_len=wav_lens)

        if self.normalize_wav:
            wav = F.layer_norm(wav, wav.shape[1:])

        # Extract wav2vec output
        out = self.model(
            wav,
            attention_mask=padding_mask,
            output_hidden_states=self.output_all_hiddens,
        )

        if self.output_all_hiddens:
            out = torch.stack(list(out.hidden_states), dim=0)
            norm_shape = out.shape[-3:]
        else:
            out = out.last_hidden_state
            norm_shape = out.shape

        # We normalize the output if required
        if self.output_norm:
            out = F.layer_norm(out, norm_shape[1:])

        return out

    def make_masks(self, src, wav_len=None, pad_idx=0):
        """This method generates the padding masks.
        Arguments
        ---------
        src : tensor
            The sequence to the encoder (required).
        wav_len : tensor
            The relative length of the wav given in SpeechBrain format.
        pad_idx : int
            The index for <pad> token (default=0).
        """
        src_key_padding_mask = None
        if wav_len is not None:
            abs_len = torch.round(wav_len * src.shape[1])
            src_key_padding_mask = length_to_mask(abs_len).bool()

        return src_key_padding_mask


class HuggingFaceWav2Vec2Pretrain(nn.Module):
    """This lobe enables the integration of HuggingFace
     wav2vec2.0 models to be pretrained.

    Source paper: https://arxiv.org/abs/2006.11477
    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html

    The return is an HuggingFace format and the mask indices that contains:
    https://huggingface.co/transformers/model_doc/wav2vec2.html#wav2vec2forpretraining

    For instance, it returns the loss that can be accessed with .loss

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "facebook/wav2vec2-large-lv60"
    save_path : str
        Path (dir) of the downloaded model.
    mask_prob : float (default: 0.65)
        Probability of masking a given frame. Default is taken from the paper.
    mask_length : float (default: 10)
        Length (i.e. number of consecutive masked frames). Default is taken from
        the paper.
    Example
    -------
    >>> inputs = torch.rand([10, 32000])
    >>> model_hub = "facebook/wav2vec2-base-960h"
    >>> save_path = "savedir"
    >>> model = HuggingFaceWav2Vec2Pretrain(model_hub, save_path)
    >>> outputs, _ = model(inputs, wav_lens=None)
    """

    def __init__(
        self,
        source,
        save_path,
        mask_prob=0.65,
        mask_length=10,
        normalize_wav=True,
    ):
        super().__init__()

        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.normalize_wav = normalize_wav

        # Download the config of the model from HuggingFace.
        self.config = Wav2Vec2Config.from_pretrained(
            source, cache_dir=save_path
        )
        self.config.output_hidden_states = (
            True  # We want the hidden states as well!
        )

        self.model = Wav2Vec2ForPreTraining(self.config)
        self.model.gradient_checkpointing_disable()  # Required by DDP
        self.model.train()

        # We check if inputs need to be normalized w.r.t pretrained wav2vec2

    def forward(self, wav, wav_lens=None):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        wav_len : tensor
            The relative length of the wav given in SpeechBrain format.
        """
        batch_size, raw_sequence_length = wav.shape

        if self.normalize_wav:
            wav = F.layer_norm(wav, wav.shape)

        sequence_length = self.model._get_feat_extract_output_lengths(
            raw_sequence_length
        ).item()

        # 1. Compute the indices that will be masked
        mask_time_indices = _compute_mask_indices(
            (batch_size, sequence_length),
            mask_prob=self.mask_prob,
            mask_length=self.mask_length,
        )
        torch_mask_time_indices = torch.tensor(
            mask_time_indices, device=wav.device, dtype=torch.long,
        )
        padding_mask = self.make_padding_masks(wav, wav_len=wav_lens)

        # 2. Sample the negative samples from the entire sequence.
        # Fairseq does it only on the masked indices, but this only work if you
        # have long sentences. For more versatily, we sample on the entire sequence.
        # value.
        full_sentence_indices = np.ones((batch_size, sequence_length))

        # print(np.sum(mask_time_indices, axis=1))
        negative_sample_indices = torch.tensor(
            transformers.models.wav2vec2.modeling_wav2vec2._sample_negative_indices(
                (batch_size, sequence_length),
                num_negatives=self.config.num_negatives,
                mask_time_indices=full_sentence_indices,
            ),
            device=wav.device,
            dtype=torch.long,
        )

        return (
            self.model(
                wav,
                mask_time_indices=torch_mask_time_indices,
                sampled_negative_indices=negative_sample_indices,
                attention_mask=padding_mask,
            ),
            torch_mask_time_indices,
        )

    def make_padding_masks(self, src, wav_len=None, pad_idx=0):
        """This method generates the padding masks.
        Arguments
        ---------
        src : tensor
            The sequence to the encoder (required).
        wav_len : tensor
            The relative length of the wav given in SpeechBrain format.
        pad_idx : int
            The index for <pad> token (default=0).
        """
        src_key_padding_mask = None
        if wav_len is not None:
            abs_len = torch.round(wav_len * src.shape[1])
            src_key_padding_mask = length_to_mask(abs_len).bool()

        return src_key_padding_mask


class WeightedSSLModel(torch.nn.Module):
    """This lobe enables the integration of use of weighted sum representations
    from different layers in a SSL encoder.

    The model can be used as a fixed feature extractor for SSL benchmarking. It
    will download automatically the model from HuggingFace or use a local path.

    More details in recipes/SSL_benchmark

    Arguments
    ---------
    hub : str
        HuggingFace hub name: e.g "facebook/wav2vec2-large-lv60"
    num_layers: int
        Number of internal layers: e.g 13 for "Base" models.
    layernorm: bool
        Whether layer representations should be layernormed before sum
    Example
    -------
    >>> inputs = torch.rand([10, 600])
    >>> model_hub = "facebook/wav2vec2-base-960h"
    >>> num_layers = 13
    >>> model = WeightedSSLModel(model_hub, num_layers)
    >>> outputs = model(inputs)
    """

    def __init__(self, hub, num_layers, layernorm=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(hub, output_hidden_states=True)
        self.num_layers = num_layers
        # Initializing the learnable weights
        zero_init = torch.cat([torch.zeros(self.num_layers)])
        self.weights = torch.nn.Parameter(zero_init, requires_grad=True)
        self.layernorm = layernorm

    def forward(self, wav, wav_lens=None):
        """This method outputs a weighted sum of the layers representations of the SSL encoder
        Arguments
        ---------
        wav : tensor
            The wavs
        """

        feats = self.encoder(wav)
        hidden_states = torch.stack(feats.hidden_states, dim=0).detach()
        # First dimension should be equal to the number of layers in the hparams
        assert (
            self.num_layers == hidden_states.shape[0]
        ), "Num layers not equal to num hidden states"
        norm_weights = torch.nn.functional.softmax(self.weights, dim=-1)
        # Layernorming the layers representations if asked
        if self.layernorm:
            hidden_states = [
                F.layer_norm(t, (t.shape[-1],)) for t in hidden_states
            ]
        # Summing the weighted layers
        weighted_feats = hidden_states[0] * norm_weights[0]
        for i in range(1, len(hidden_states)):
            weighted_feats += hidden_states[i] * norm_weights[i]

        return weighted_feats
