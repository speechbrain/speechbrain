"""
This lobe is a wrapper for the Encoder of the T5 model.

Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Author : Lucas Druart 2024
"""

import logging
from typing import Optional, Tuple

import torch

from speechbrain.lobes.models.huggingface_transformers.huggingface import (
    HFTransformersInterface,
)

logger = logging.getLogger(__name__)


class T5(HFTransformersInterface):
    """
    This lobe enables integration of the t5 model into SpeechBrain.

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "facebook/wav2vec2-large-lv60"
    save_path : str
        save directory of the downloaded model.
    freeze : bool (default: False)
        If True, the entire model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    freeze_encoder : bool (default: False)
        If False, the encoder weights are frozen.
    """

    def __init__(
        self, source, save_path="", freeze=False, freeze_encoder=False
    ):
        super().__init__(
            source=source, save_path=save_path, freeze=freeze, seq2seqlm=True
        )
        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            logger.warning(
                "speechbrain.lobes.models.huggingface_transformers.t5 - encoder is frozen."
            )
            self.model.encoder.eval()
            for param in self.model.encoder.parameters():
                param.requires_grad = False

    def forward_encoder(
        self, input_ids: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        """
        Performs the forward pass of T5's encoder.

        Arguments
        ---------
        input_ids : torch.LongTensor
            A batch of input token ids. The input is supposed to already be padded.

        Returns
        -------
        torch.Tensor
            Final layer hidden state of the encoder.
        """
        if self.freeze_encoder:
            with torch.no_grad():
                enc_out = self.model.encoder(
                    input_ids=input_ids
                ).last_hidden_state.detach()
                return enc_out

        enc_out = self.model.encoder(input_ids=input_ids).last_hidden_state
        return enc_out

    def forward_decoder(
        self,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        """
        Performs the forward pass of T5's decoder.

        Arguments
        ---------
        decoder_input_ids : torch.LongTensor
            A batch of decoder input token ids. The input is supposed to already be padded.
        encoder_hidden_states : torch.LongTensor
            A batch of encoder hidden states to attend to in the decoder's cross-attention.
        output_hidden_states : bool
            Whether to output all hidden states or just the last ones.

        Returns
        -------
        torch.Tensor
            Output logits from the decoder.
        """
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = decoder_outputs.last_hidden_state

        lm_logits = self.model.lm_head(sequence_output)

        return lm_logits

    @torch.no_grad()
    def decode(
        self, memory, enc_states, enc_lens=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a single decoding step for this transformer model.
        It is required for the searchers to decode step by step.

        Arguments
        ---------
        memory : torch.LongTensor
            The sequence of already decoded.
        enc_states : torch.LongTensor
            Hidden output of the encoder.
        enc_lens : torch.LongTensor
            The actual length of encoder states.

        Returns
        -------
        torch.Tensor
            Output logits from the decoder.
        torch.Tensor
            Last layer's cross-attention weights.
        """

        output = self.model.decoder(
            input_ids=memory.int(),
            encoder_hidden_states=enc_states,
            output_attentions=True,
        )

        return (
            self.model.lm_head(output.last_hidden_state),
            output.cross_attentions[-1],
        )
