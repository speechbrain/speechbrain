"""This lobe enables the integration of huggingface pretrained mBART models.
Reference: https://arxiv.org/abs/2001.08210

Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Ha Nguyen 2023
"""

import torch

from speechbrain.lobes.models.huggingface_transformers.huggingface import (
    HFTransformersInterface,
)
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class mBART(HFTransformersInterface):
    """This lobe enables the integration of HuggingFace and SpeechBrain
    pretrained mBART models.

    Source paper mBART: https://arxiv.org/abs/2001.08210
    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html

    The model is normally used as a text decoder of seq2seq models. It
    will download automatically the model from HuggingFace or use a local path.

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "facebook/mbart-large-50-many-to-many-mmt"
    save_path : str
        Path (dir) of the downloaded model.
    freeze : bool (default: True)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    target_lang: str (default: fra_Latn (a.k.a French)
        The target language code according to NLLB model.
    decoder_only : bool (default: True)
        If True, only take the decoder part (and/or the lm_head) of the model.
        This is useful in case one wants to couple a pre-trained speech encoder (e.g. wav2vec)
        with a text-based pre-trained decoder (e.g. mBART, NLLB).
    share_input_output_embed : bool (default: True)
        If True, use the embedded layer as the lm_head.

    Example
    -------
    >>> src = torch.rand([10, 1, 1024])
    >>> tgt = torch.LongTensor([[250008,    313,     25,    525,    773,  21525,   4004,      2]])
    >>> model_hub = "facebook/mbart-large-50-many-to-many-mmt"
    >>> save_path = "savedir"
    >>> model = mBART(model_hub, save_path) # doctest: +SKIP
    >>> outputs = model(src, tgt) # doctest: +SKIP
    """

    def __init__(
        self,
        source,
        save_path,
        freeze=True,
        target_lang="fr_XX",
        decoder_only=True,
        share_input_output_embed=True,
    ):
        super().__init__(
            source=source,
            save_path=save_path,
            freeze=freeze,
            seq2seqlm=True,
        )

        self.target_lang = target_lang
        self.decoder_only = decoder_only
        self.share_input_output_embed = share_input_output_embed

        self.load_tokenizer(source=source, pad_token=None, tgt_lang=target_lang)

        if share_input_output_embed:
            self.model.lm_head.weight = (
                self.model.model.decoder.embed_tokens.weight
            )
            self.model.lm_head.requires_grad = False
            self.model.model.decoder.embed_tokens.requires_grad = False

        if decoder_only:
            # When we only want to use the decoder part
            del self.model.model.encoder

        for k, p in self.model.named_parameters():
            # It is a common practice to only fine-tune the encoder_attn and layer_norm layers of this model.
            if "encoder_attn" in k or "layer_norm" in k:
                p.requires_grad = True
            else:
                p.requires_grad = False

    def forward(self, src, tgt, pad_idx=0):
        """This method implements a forward step for mt task using a wav2vec encoder
        (same than above, but without the encoder stack)

        Arguments
        ---------
        src : tensor
            output features from the w2v2 encoder (transcription)
        tgt : tensor
            The sequence to the decoder (translation) (required).
        pad_idx : int
            The index for <pad> token (default=0).

        Returns
        -------
        dec_out : torch.Tensor
            Decoder output.
        """

        # should we replace 0 elements by pax_idx as pad_idx of mbart model seems to be different from 0?
        tgt = self.custom_padding(
            tgt, 0, self.model.model.decoder.config.pad_token_id
        )

        if self.freeze:
            with torch.no_grad():
                if hasattr(self.model.model, "encoder"):
                    src = self.model.model.encoder(
                        inputs_embeds=src
                    ).last_hidden_state.detach()
                dec_out = self.model.model.decoder(
                    input_ids=tgt, encoder_hidden_states=src
                ).last_hidden_state.detach()
                dec_out = self.model.lm_head(dec_out).detach()
                return dec_out

        if hasattr(self.model.model, "encoder"):
            src = self.model.model.encoder(inputs_embeds=src).last_hidden_state
        dec_out = self.model.model.decoder(
            input_ids=tgt, encoder_hidden_states=src
        ).last_hidden_state
        dec_out = self.model.lm_head(dec_out)
        return dec_out

    @torch.no_grad()
    def decode(self, tgt, encoder_out, enc_len=None):
        """This method implements a decoding step for the transformer model.

        Arguments
        ---------
        tgt : torch.Tensor
            The sequence to the decoder.
        encoder_out : torch.Tensor
            Hidden output of the encoder.
        enc_len : torch.LongTensor
            The actual length of encoder states.

        Returns
        -------
        output : torch.Tensor
            Output of transformer.
        cross_attention : torch.Tensor
            Attention value.
        """

        if tgt.dtype not in [torch.long, torch.int64]:
            tgt = tgt.long()

        tgt_mask = torch.ones(tgt.size(), device=tgt.device)

        output = self.model.model.decoder(
            input_ids=tgt,
            encoder_hidden_states=encoder_out,
            attention_mask=tgt_mask,
            output_attentions=True,
        )

        return (
            self.model.lm_head(output.last_hidden_state),
            output.cross_attentions[-1],
        )

    def custom_padding(self, x, org_pad, custom_pad):
        """This method customizes the padding.
        Default pad_idx of SpeechBrain is 0.
        However, it happens that some text-based models like mBART reserves 0 for something else,
        and are trained with specific pad_idx.
        This method change org_pad to custom_pad

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
        out : torch.Tensor
            Padded outputs.
        """
        out = x.clone()
        out[x == org_pad] = custom_pad

        return out

    def override_config(self, config):
        """If the config needs to be overridden, here is the place.

        Arguments
        ---------
        config : MBartConfig
            The original config needs to be overridden.

        Returns
        -------
        Overridden config
        """
        config.decoder_layerdrop = 0.05
        return config
