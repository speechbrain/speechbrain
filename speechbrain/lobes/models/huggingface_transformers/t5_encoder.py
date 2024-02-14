# Software Name : E2E-DST
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Authors: Lucas Druart

"""
This lobe is a wrapper for the T5Decoder model for Dialogue Understanding in SpeechBrain
"""

import logging
import pathlib
from typing import Optional, Tuple, Union
import os
import torch
from torch import nn

from speechbrain.utils.fetching import fetch


# We check if transformers is installed.
try:
    from transformers import T5Config, AutoModelForSeq2SeqLM
    from transformers.modeling_outputs import Seq2SeqLMOutput
    from huggingface_hub import model_info
except ImportError:
    MSG = "Please install transformers from HuggingFace to use T5\n"
    MSG += "E.G. run: pip install transformers"
    raise ImportError(MSG)

logger = logging.getLogger(__name__)


class T5EncoderModelForDialogueUnderstanding(nn.Module):
    _keys_to_ignore_on_load_missing = []

    def __init__(self, source, save_path, freeze=False):
        super().__init__()

        config = T5Config
        model = AutoModelForSeq2SeqLM

        # Download and load the model
        self._from_pretrained(
            source, config=config, model=model, save_path=save_path
        )

        self.freeze = freeze
        if self.freeze:
            logger.warning(
                "speechbrain.lobes.models.huggingface_T5Encoder - T5 Encoder is frozen."
            )
            self.model.eval()
            for param in self.model.encoder.parameters():
                param.requires_grad = False
        else:
            self.model.train()

    def _check_model_source(self, path):
        """
        Checks if the pretrained model has been trained with SpeechBrain and
        is hosted locally or on a HuggingFace hub.
        """
        checkpoint_filename = ""
        source = pathlib.Path(path)
        is_local = True
        is_sb = True

        # If path is a huggingface hub.
        if not source.exists():
            is_local = False

        if is_local:
            # Test for HuggingFace model
            if any(File.endswith(".bin") for File in os.listdir(path)):
                is_sb = False
                return is_sb, checkpoint_filename

            # Test for SpeechBrain model and get the filename.
            for File in os.listdir(path):
                if File.endswith(".ckpt"):
                    checkpoint_filename = os.path.join(path, File)
                    is_sb = True
                    return is_sb, checkpoint_filename
        else:
            files = model_info(
                path
            ).siblings  # get the list of files of the Hub

            # Test if it's an HuggingFace model or a SB one
            for File in files:
                if File.rfilename.endswith(".ckpt"):
                    checkpoint_filename = File.rfilename
                    is_sb = True
                    return is_sb, checkpoint_filename

            for File in files:
                if File.rfilename.endswith(".bin"):
                    checkpoint_filename = File.rfilename
                    is_sb = False
                    return is_sb, checkpoint_filename

        err_msg = f"{path} does not contain a .bin or .ckpt checkpoint !"
        raise FileNotFoundError(err_msg)

    def _load_sb_pretrained_t5_parameters(self, path):
        """Loads the parameter of a w2v2 (=T5) model pretrained with SpeechBrain and the
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
            if "t5." in key:
                save_key = key.replace("model.t5.", "")
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
                + "is useless for T5 finetuning."
            )

    def _from_pretrained(self, source, config, model, save_path):
        """This function manages the source checking and loading of the parameters.
        # 1. Is the model from HF or a local path
        # 2. Is the model pretrained with HF or SpeechBrain
        # 3. Download (if appropriate) and load with respect to 1. and 2.
        """

        is_sb, ckpt_file = self._check_model_source(source)
        if is_sb:
            config = config.from_pretrained(source, cache_dir=save_path)
            self.model = model(config)
            self.model.gradient_checkpointing_disable()  # Required by DDP
            # fetch the checkpoint file
            ckpt_full_path = fetch(
                filename=ckpt_file, source=source, savedir=save_path
            )
            # We transfer the parameters from the checkpoint.
            self._load_sb_pretrained_bart_parameters(ckpt_full_path)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                source, cache_dir=save_path
            )
            del self.model.decoder

    def forward(
        self,
        # the input ids are always already encoded
        input_ids: Optional[torch.LongTensor] = None,
        # attention_mask: Optional[torch.FloatTensor] = None,
        # decoder_input_ids: Optional[torch.LongTensor] = None,
        # decoder_attention_mask: Optional[torch.BoolTensor] = None,
        # head_mask: Optional[torch.FloatTensor] = None,
        # decoder_head_mask: Optional[torch.FloatTensor] = None,
        # cross_attn_head_mask: Optional[torch.Tensor] = None,
        # encoder_hidden_states: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        # past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        # inputs_embeds: Optional[torch.FloatTensor] = None,
        # decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        # The labels are handled with the loss and we expect the decoder_input_ids to be provided
        # labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        # return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        # We assume the encoder outputs are always given

        if self.freeze:
            with torch.no_grad():
                enc_out = self.model.encoder(
                    input_ids=input_ids
                ).last_hidden_state.detach()
                return enc_out

        enc_out = self.model.encoder(input_ids=input_ids).last_hidden_state
        return enc_out
