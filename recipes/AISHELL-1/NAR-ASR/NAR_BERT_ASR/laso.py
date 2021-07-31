import torch
import torch.nn as nn
import speechbrain as sb
from speechbrain.nnet.containers import ModuleList
from speechbrain.lobes.models.transformer.Transformer import (
    get_lookahead_mask,
    get_key_padding_mask,
)
from speechbrain.dataio.dataio import length_to_mask
from transformers import BertModel
from typing import Optional
import math


class Conv2dSubsampling(nn.Module):
    def __init__(self, out_channels, kernel_size, stride):
        super(Conv2dSubsampling, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        """Subsample x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
        Returns:
            torch.Tensor: Subsampled tensor (#batch, time/4, (idim/4)*out_channels)
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = x.transpose(1, 2).contiguous().view(b, t, c * f)

        return x


class PositionalwiseFeedForward(nn.Module):
    """The class implements the positional-wise feed forward module in
    “Attention Is All You Need”.
    Arguments
    ----------
    d_ffn: int
        Dimension of representation space of this positional-wise feed
        forward module.
    input_shape : tuple
        Expected shape of the input. Alternatively use ``input_size``.
    input_size : int
        Expected size of the input. Alternatively use ``input_shape``.
    dropout: float
        Fraction of outputs to drop.
    activation: torch class
        activation functions to be applied (Recommendation: GLU).
    Example
    -------
    >>> inputs = torch.rand([8, 60, 512])
    >>> net = PositionalwiseFeedForward(256, input_size=inputs.shape[-1])
    >>> outputs = net(inputs)
    >>> outputs.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        d_ffn,
        input_shape=None,
        input_size=None,
        dropout=0.1,
        activation=nn.GLU,
    ):
        super().__init__()

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")

        if input_size is None:
            input_size = input_shape[-1]

        self.ffn = nn.Sequential(
            nn.Linear(input_size, d_ffn),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn // 2, input_size),
        )

    def forward(self, x):
        x = x.permute(1, 0, 2)  # give a tensor of shap (time, batch, fea)
        x = self.ffn(x)
        x = x.permute(1, 0, 2)  # reshape the output back to (batch, time, fea)

        return x


class AttentionBlock(nn.Module):
    """This class implements the AttentionBlock of LASO.
    Arguments
    ----------
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    nhead : int
        Number of attention heads.
    d_model : int
        Dimension of the model.
    kdim : int
        Dimension for key (optional).
    vdim : int
        Dimension for value (optional).
    dropout : float
        Dropout for the decoder (optional).
    """

    def __init__(
        self,
        d_ffn,
        nhead,
        d_model,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation=nn.GLU,
    ):
        super().__init__()
        self.mutihead_attn = sb.nnet.attention.MultiheadAttention(
            nhead=nhead, d_model=d_model, kdim=kdim, vdim=vdim, dropout=dropout,
        )
        self.pos_ffn = PositionalwiseFeedForward(
            d_ffn=d_ffn,
            input_size=d_model,
            dropout=dropout,
            activation=activation,
        )

        # normalization layers
        self.norm = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self, query, key, value, attn_mask=None, key_padding_mask=None,
    ):
        output, multihead_attention = self.mutihead_attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        output = query + self.dropout1(output)

        output = self.norm(output)
        output = output + self.dropout2(self.pos_ffn(output))

        return output, multihead_attention


class TransformerEncoderLayer(nn.Module):
    """This is an implementation of self-attention encoder layer.
    Arguments
    ----------
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    nhead : int
        Number of attention heads.
    d_model : int
        The expected size of the input embedding.
    kdim : int
        Dimension of the key (Optional).
    vdim : int
        Dimension of the value (Optional).
    dropout : float
        Dropout for the encoder (Optional).
    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = TransformerEncoderLayer(512, 8, d_model=512)
    >>> output = net(x)
    >>> output[0].shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        d_ffn,
        nhead,
        d_model,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation=nn.GLU,
    ):
        super().__init__()

        self.attention_block = AttentionBlock(
            d_ffn=d_ffn,
            nhead=nhead,
            d_model=d_model,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
            activation=activation,
        )
        self.norm = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Arguments
        ----------
        src : tensor
            The sequence to the encoder layer (required).
        src_mask : tensor
            The mask for the src sequence (optional).
        src_key_padding_mask : tensor
            The mask for the src keys per batch (optional).
        """

        src = self.norm(src)
        output, multihead_attention = self.attention_block(
            query=src,
            key=src,
            value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )

        return output, multihead_attention


class TransformerEncoder(nn.Module):
    """This class implements the transformer encoder.
    Arguments
    ---------
    num_layers : int
        Number of transformer layers to include.
    nhead : int
        Number of attention heads.
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    input_shape : tuple
        Expected shape of an example input.
    d_model : int
        The dimension of the input embedding.
    kdim : int
        Dimension for key (Optional).
    vdim : int
        Dimension for value (Optional).
    dropout : float
        Dropout for the encoder (Optional).
    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = TransformerEncoder(1, 8, 512, d_model=512)
    >>> output, _ = net(x)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        input_shape=None,
        d_model=None,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation=nn.GLU,
    ):
        super().__init__()

        if input_shape is None and d_model is None:
            raise ValueError("Expected one of input_shape or d_model")

        if input_shape is not None and d_model is None:
            if len(input_shape) == 3:
                msg = "Input shape of the Transformer must be (batch, time, fea). Please revise the forward function in TransformerInterface to handle arbitrary shape of input."
                raise ValueError(msg)
            d_model = input_shape[-1]

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Arguments
        ----------
        src : tensor
            The sequence to the encoder layer (required).
        src_mask : tensor
            The mask for the src sequence (optional).
        src_key_padding_mask : tensor
            The mask for the src keys per batch (optional).
        """
        output = src
        attention_lst = []
        for enc_layer in self.layers:
            output, attention = enc_layer(
                src=output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
            )
            attention_lst.append(attention)
        output = self.norm(output)

        return output, attention_lst


class PDSLayer(nn.Module):
    """This class implements the LASO PDS layer.
    Arguments
    ----------
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    nhead : int
        Number of attention heads.
    d_model : int
        Dimension of the model.
    kdim : int
        Dimension for key (optional).
    vdim : int
        Dimension for value (optional).
    dropout : float
        Dropout for the decoder (optional).
    """

    def __init__(
        self,
        d_ffn,
        nhead,
        d_model,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation=nn.GLU,
    ):
        super().__init__()

        self.attention_block = AttentionBlock(
            d_ffn=d_ffn,
            nhead=nhead,
            d_model=d_model,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
            activation=activation,
        )
        self.norm = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)

    def forward(
        self, tgt, memory, memory_mask=None, memory_key_padding_mask=None
    ):
        """
        Arguments
        ----------
        tgt: tensor
            The sequence to the decoder layer (required).
        memory: tensor
            The sequence from the last layer of the encoder (required).
        memory_mask: tensor
            The mask for the memory sequence (optional).
        memory_key_padding_mask: tensor
            The mask for the memory keys per batch (optional).
        """
        tgt = self.norm(tgt)
        output, multihead_attention = self.attention_block(
            query=tgt,
            key=memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )

        return output, multihead_attention


class PDS(nn.Module):
    """This class implements the LASO PDS.
    Arguments
    ----------
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    nhead : int
        Number of attention heads.
    d_model : int
        Dimension of the model.
    kdim : int
        Dimension for key (Optional).
    vdim : int
        Dimension for value (Optional).
    dropout : float
        Dropout for the decoder (Optional).
    """

    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        d_model,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation=nn.GLU,
    ):
        super().__init__()

        self.attention_block = AttentionBlock(
            d_ffn=d_ffn,
            nhead=nhead,
            d_model=d_model,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
            activation=activation,
        )
        self.layers = nn.ModuleList(
            [
                PDSLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, tgt, memory, memory_mask=None, memory_key_padding_mask=None
    ):
        """
        Arguments
        ----------
        tgt : tensor
            The sequence to the decoder layer (required).
        memory : tensor
            The sequence from the last layer of the encoder (required).
        tgt_mask : tensor
            The mask for the tgt sequence (optional).
        memory_mask : tensor
            The mask for the memory sequence (optional).
        tgt_key_padding_mask : tensor
            The mask for the tgt keys per batch (optional).
        memory_key_padding_mask : tensor
            The mask for the memory keys per batch (optional).
        """
        output, multihead_attention = self.attention_block(
            query=tgt,
            key=memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        multihead_attns = []
        for pds_layer in self.layers:
            output, multihead_attn = pds_layer(
                tgt=output,
                memory=memory,
                memory_mask=memory_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            multihead_attns.append(multihead_attn)

        return output, multihead_attns


class TransformerDecoder(nn.Module):
    """This class implements the LASO decoder.
    Arguments
    ---------
    num_layers : int
        Number of transformer layers to include.
    nhead : int
        Number of attention heads.
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    d_model : int
        The dimension of the input embedding.
    kdim : int
        Dimension for key (Optional).
    vdim : int
        Dimension for value (Optional).
    dropout : float
        Dropout for the encoder (Optional).
    """

    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        d_model,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation=nn.GLU,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        """
        Arguments
        ----------
        tgt : tensor
            The sequence to the decoder layer (required).
        tgt_mask : tensor
            The mask for the tgt sequence (optional).
        tgt_key_padding_mask : tensor
            The mask for the tgt keys per batch (optional).
        """
        output = tgt
        attention_lst = []
        for dec_layer in self.layers:
            output, attention = dec_layer(
                src=output,
                src_mask=tgt_mask,
                src_key_padding_mask=tgt_key_padding_mask,
            )
            attention_lst.append(attention)

        return output, attention_lst


class PositionalEncoding(nn.Module):
    """This class implements the positional encoding function.
    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    Arguments
    ---------
    max_len : int
        Max length of the input sequences (default 2500).
    Example
    -------
    >>> a = torch.rand((8, 120, 512))
    >>> enc = PositionalEncoding(input_size=a.shape[-1])
    >>> b = enc(a)
    >>> b.shape
    torch.Size([1, 120, 512])
    """

    def __init__(self, input_size, max_len=2500):
        super().__init__()
        self.max_len = max_len
        pe = torch.zeros(self.max_len, input_size, requires_grad=False)
        positions = torch.arange(0, self.max_len).unsqueeze(1).float()
        denominator = torch.exp(
            torch.arange(0, input_size, 2).float()
            * -(math.log(10000.0) / input_size)
        )

        pe[:, 0::2] = torch.sin(positions * denominator)
        pe[:, 1::2] = torch.cos(positions * denominator)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments
        ---------
        x : tensor
            Input feature shape (batch, time, fea)
        """
        return self.pe[:, : x.size(1)].clone().detach()


class LASO(nn.Module):
    """This is an implementation of NAR-BERT-ASR-Encoder.
    Arguments
    ----------
    d_model : int
        The number of expected features in the encoder/decoder inputs.
    nhead : int
        The number of heads in the multi-head attention models.
    num_encoder_layers : int
        The number of sub-encoder-layers in the encoder.
    num_decoder_layers : int
        The number of sub-decoder-layers in the decoder.
    dim_ffn : int
        The dimension of the feedforward network model.
    dropout : int
        The dropout value (default=0.1).
    activation : torch class
        The activation function of encoder/decoder intermediate layer.
    """

    def __init__(
        self,
        tgt_vocab,
        fbank_dim=80,
        out_channels=32,
        kernel_size=3,
        stride=2,
        d_model=256,
        bert_d_model=768,
        nhead=8,
        num_encoder_layers=6,
        num_pds_layers=4,
        num_decoder_layers=6,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.GLU,
        bert_model_name="bert-base-chinese",
    ):
        super().__init__()

        assert (
            num_encoder_layers + num_decoder_layers > 0
        ), "number of encoder layers and number of decoder layers cannot both be 0!"

        self.conv2d_subsampling = Conv2dSubsampling(
            out_channels=out_channels, kernel_size=kernel_size, stride=stride
        )
        self.positional_encoding = PositionalEncoding(d_model)
        self.custom_src_module = ModuleList(
            nn.Linear(
                in_features=(((fbank_dim - 1) // 2 - 1) // 2) * out_channels,
                out_features=d_model,
            ),
            nn.Dropout(dropout),
        )
        self.encoder = TransformerEncoder(
            nhead=nhead,
            num_layers=num_encoder_layers,
            d_ffn=d_ffn,
            d_model=d_model,
            dropout=dropout,
            activation=activation,
        )
        self.pds = PDS(
            num_layers=num_pds_layers,
            nhead=nhead,
            d_ffn=d_ffn,
            d_model=d_model,
            dropout=dropout,
            activation=activation,
        )
        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            nhead=nhead,
            d_ffn=d_ffn,
            d_model=d_model,
            dropout=dropout,
            activation=activation,
        )
        self.mlp = nn.Linear(in_features=d_model, out_features=bert_d_model)
        self.classifier = nn.Linear(
            in_features=bert_d_model, out_features=tgt_vocab, bias=False
        )

        self.bert_model_name = bert_model_name

        # reset parameters using xavier_normal_
        self._init_params()

    def forward(
        self, src, tgt, wav_len=None, pad_idx=0,
    ):
        """
        Arguments
        ----------
        src : tensor
            The sequence to the encoder (required).
        tgt : tensor
            The sequence to the decoder (required).
        pad_idx : int
            The index for <pad> token (default=0).
        """

        encode_result = self.encode(
            src=src, tgt=tgt, wav_len=wav_len, pad_idx=pad_idx
        )
        output = self.classifier(encode_result)

        return output

    def encode(
        self, src, tgt, wav_len=None, pad_idx=0,
    ):
        """
        Arguments
        ----------
        src : tensor
            The sequence to the encoder (required).
        tgt : tensor
            The sequence to the decoder (required).
        pad_idx : int
            The index for <pad> token (default=0).
        """

        src = self.conv2d_subsampling(src)

        (
            src_key_padding_mask,
            tgt_key_padding_mask,
            src_mask,
            tgt_mask,
        ) = self.make_masks(src, tgt, wav_len, pad_idx=pad_idx)

        src = self.custom_src_module(src)
        src = src + self.positional_encoding(src)
        encoder_out, encoder_attentions = self.encoder(
            src=src,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        tgt, pds_attentions = self.pds(tgt=tgt, memory=encoder_out)
        decoder_out, decoder_attentions = self.decoder(tgt=tgt)
        encode_result = self.mlp(decoder_out)

        return encode_result

    def make_masks(self, src, tgt, wav_len=None, pad_idx=0):
        """This method generates the masks for training the transformer model.
        Arguments
        ---------
        src : tensor
            The sequence to the encoder (required).
        tgt : tensor
            The sequence to the decoder (required).
        pad_idx : int
            The index for <pad> token (default=0).
        """
        src_key_padding_mask = None
        if wav_len is not None and self.training:
            abs_len = torch.round(wav_len * src.shape[1])
            src_key_padding_mask = (1 - length_to_mask(abs_len)).bool()
        tgt_key_padding_mask = get_key_padding_mask(tgt, pad_idx=pad_idx)

        src_mask = None
        tgt_mask = get_lookahead_mask(tgt)
        return src_key_padding_mask, tgt_key_padding_mask, src_mask, tgt_mask

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

        # load Hugging Face BERT word_embeddings pretrained weights
        self.classifier.load_state_dict(
            BertModel.from_pretrained(
                self.bert_model_name
            ).embeddings.word_embeddings.state_dict()
        )
        # don't update the classifier weights
        for param in self.classifier.parameters():
            param.requires_grad = False
