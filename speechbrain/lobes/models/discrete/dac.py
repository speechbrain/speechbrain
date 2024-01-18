"""
This lobe enables the integration of pretrained discrete DAC model.
Reference: http://arxiv.org/abs/2306.06546
Reference: https://descript.notion.site/Descript-Audio-Codec-11389fce0ce2419891d6591a68f814d5
Reference: https://github.com/descriptinc/descript-audio-codec

Author
 * Shubham Gupta 2023

"""

import math
from pathlib import Path
from typing import List, Union
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Note: The path torch.nn.utils.parametrizations may not be available
# in older PyTorch versions, such as 1.13.1. To ensure compatibility,
# it is recommended to check and use the appropriate import statement.

# Attempt to import the preferred module for parametrizations in newer PyTorch versions
try:
    from torch.nn.utils.parametrizations import weight_norm

# If the preferred import fails, fallback to the alternative import for compatibility
except ImportError:
    from torch.nn.utils import weight_norm

logger = logging.getLogger(__name__)

SUPPORTED_VERSIONS = ["1.0.0"]


__MODEL_LATEST_TAGS__ = {
    ("44khz", "8kbps"): "0.0.1",
    ("24khz", "8kbps"): "0.0.4",
    ("16khz", "8kbps"): "0.0.5",
    ("44khz", "16kbps"): "1.0.0",
}


__MODEL_URLS__ = {
    (
        "44khz",
        "0.0.1",
        "8kbps",
    ): "https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.1/weights.pth",
    (
        "24khz",
        "0.0.4",
        "8kbps",
    ): "https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.4/weights_24khz.pth",
    (
        "16khz",
        "0.0.5",
        "8kbps",
    ): "https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.5/weights_16khz.pth",
    (
        "44khz",
        "1.0.0",
        "16kbps",
    ): "https://github.com/descriptinc/descript-audio-codec/releases/download/1.0.0/weights_44khz_16kbps.pth",
}


def WNConv1d(*args, **kwargs):
    """
    Apply weight normalization to a 1D convolutional layer.

    Parameters
    ----------
    *args
        Variable length argument list for nn.Conv1d.
    **kwargs
        Arbitrary keyword arguments for nn.Conv1d.

    Returns
    -------
    torch.nn.Module
        The weight-normalized nn.Conv1d layer.
    """
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    """
    Apply weight normalization to a 1D transposed convolutional layer.

    Parameters
    ----------
    *args
        Variable length argument list for nn.ConvTranspose1d.
    **kwargs
        Arbitrary keyword arguments for nn.ConvTranspose1d.

    Returns
    -------
    torch.nn.Module
        The weight-normalized nn.ConvTranspose1d layer.
    """
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def init_weights(m):
    """
    Initialize the weights of a 1D convolutional layer.
    """
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


def download(
    model_type: str = "44khz",
    model_bitrate: str = "8kbps",
    tag: str = "latest",
    local_path: Path = None,
):
    """
    Downloads a specified model file based on model type, bitrate, and tag, saving it to a local path.

    Parameters
    ----------
    model_type : str, optional
        The type of model to download. Can be '44khz', '24khz', or '16khz'. Default is '44khz'.
    model_bitrate : str, optional
        The bitrate of the model. Can be '8kbps' or '16kbps'. Default is '8kbps'.
    tag : str, optional
        A specific version tag for the model. Default is 'latest'.
    local_path : Path, optional
        The local file path where the model will be saved. If not provided, a default path will be used.

    Returns
    -------
    Path
        The local path where the model is saved.

    Raises
    ------
    ValueError
        If the model type or bitrate is not supported, or if the model cannot be found or downloaded.
    """

    model_type = model_type.lower()
    tag = tag.lower()

    assert model_type in [
        "44khz",
        "24khz",
        "16khz",
    ], "model_type must be one of '44khz', '24khz', or '16khz'"

    assert model_bitrate in [
        "8kbps",
        "16kbps",
    ], "model_bitrate must be one of '8kbps', or '16kbps'"

    if tag == "latest":
        tag = __MODEL_LATEST_TAGS__[(model_type, model_bitrate)]

    download_link = __MODEL_URLS__.get((model_type, tag, model_bitrate), None)
    logger.info(f"Download link: {download_link}")

    if download_link is None:
        raise ValueError(
            f"Could not find model with tag {tag} and model type {model_type}"
        )

    if local_path is None:
        local_path = (
            Path.home()
            / f".cache/descript/dac/weights_{model_type}_{model_bitrate}_{tag}.pth"
        )

    if not local_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Download the model
        import requests

        response = requests.get(download_link)

        if response.status_code != 200:
            raise ValueError(
                f"Could not download model. Received response code {response.status_code}"
            )
        local_path.write_bytes(response.content)

    return local_path


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    """
    Applies the 'snake' activation function on the input tensor.

    This function reshapes the input tensor, applies a modified sine function to it, and then reshapes it back
    to its original shape.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor to which the snake activation function will be applied.
    alpha : float
        A scalar value that modifies the sine function within the snake activation.

    Returns
    -------
    torch.Tensor
        The transformed tensor after applying the snake activation function.
    """
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class VectorQuantize(nn.Module):
    """
    An implementation for Vector Quantization
    """

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        """
        Implementation of VQ similar to Karpathy's repo:
        https://github.com/karpathy/deep-vector-quantization
        Additionally uses following tricks from Improved VQGAN
        (https://arxiv.org/pdf/2110.04627.pdf):
            1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
                for improved codebook usage
            2. l2-normalized codes: Converts euclidean distance to cosine similarity which
                improves training stability
        """
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z: torch.Tensor):
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """

        # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
        z_e = self.in_proj(z)  # z_e : (B x D x T)
        z_q, indices = self.decode_latents(z_e)

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean(
            [1, 2]
        )
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean(
            [1, 2]
        )

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id: torch.Tensor):
        """
        Embeds an ID using the codebook weights.

        This method utilizes the codebook weights to embed the given ID.

        Parameters
        ----------
        embed_id : torch.Tensor
            The tensor containing IDs that need to be embedded.

        Returns
        -------
        torch.Tensor
            The embedded output tensor after applying the codebook weights.
        """
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id: torch.Tensor):
        """
        Decodes the embedded ID by transposing the dimensions.

        This method decodes the embedded ID by applying a transpose operation to the dimensions of the
        output tensor from the `embed_code` method.

        Parameters
        ----------
        embed_id : torch.Tensor
            The tensor containing embedded IDs.

        Returns
        -------
        torch.Tensor
            The decoded tensor
        """
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents: torch.Tensor):
        """
        Decodes latent representations into discrete codes by comparing with the codebook.

        Parameters
        ----------
        latents : torch.Tensor
            The latent tensor representations to be decoded.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the decoded latent tensor (`z_q`) and the indices of the codes.
        """
        encodings = latents.permute(0, 2, 1).reshape(-1, latents.size(1))
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )

        # indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))

        max_indices = (-dist).max(dim=1)[1]
        b = latents.size(0)
        t = max_indices.numel() // b
        indices = max_indices.view(b, t)
        z_q = self.decode_code(indices)
        return z_q, indices


class ResidualVectorQuantize(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312


    Example
    -------
    Using a pretrained RVQ unit.

    >>> dac = DAC(load_pretrained=True, model_type="44KHz", model_bitrate="8kbps", tag="latest")
    >>> quantizer = dac.quantizer
    >>> continuous_embeddings = torch.randn(1, 1024, 100) # Example shape: [Batch, Channels, Time]
    >>> discrete_embeddings, codes, _, _, _ = quantizer(continuous_embeddings)

    """

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
    ):
        """
        Initializes the ResidualVectorQuantize

        Parameters
        ----------
        input_dim : int, optional, by default 512
        n_codebooks : int, optional, by default 9
        codebook_size : int, optional, by default 1024
        codebook_dim : Union[int, list], optional,  by default 8
        quantizer_dropout : float, optional, by default 0.0
        """
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(input_dim, codebook_size, codebook_dim[i])
                for i in range(n_codebooks)
            ]
        )
        self.quantizer_dropout = quantizer_dropout

    def forward(self, z, n_quantizers: int = None):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            No. of quantizers to use
            (n_quantizers < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.
        Returns
        -------
        z : Tensor[B x D x T]
            Quantized continuous representation of input
        codes : Tensor[B x N x T]
            Codebook indices for each codebook
            (quantized discrete representation of input)
        latents : Tensor[B x N*D x T]
            Projected latents (continuous representation of input before quantization)
        vq/commitment_loss : Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        vq/codebook_loss : Tensor[1]
            Codebook loss to update the codebook
        """
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break

            (
                z_q_i,
                commitment_loss_i,
                codebook_loss_i,
                indices_i,
                z_e_i,
            ) = quantizer(residual)

            # Create mask to apply quantizer dropout
            mask = (
                torch.full((z.shape[0],), fill_value=i, device=z.device)
                < n_quantizers
            )
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            # Sum losses
            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)

        return z_q, codes, latents, commitment_loss, codebook_loss

    def from_codes(self, codes: torch.Tensor):
        """Given the quantized codes, reconstruct the continuous representation
        Parameters
        ----------
        codes : Tensor[B x N x T]
            Quantized discrete representation of input
        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        """
        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        return z_q, torch.cat(z_p, dim=1), codes

    def from_latents(self, latents: torch.Tensor):
        """Given the unquantized latents, reconstruct the
        continuous representation after quantization.

        Parameters
        ----------
        latents : Tensor[B x N x T]
            Continuous representation of input after projection

        Returns
        -------
        Tensor[B x D x T]
            Quantized representation of full-projected space
        Tensor[B x D x T]
            Quantized representation of latent space
        """
        z_q = 0
        z_p = []
        codes = []
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(
            axis=0, keepdims=True
        )[0]
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            z_p_i, codes_i = self.quantizers[i].decode_latents(
                latents[:, j:k, :]
            )
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)


class Snake1d(nn.Module):
    """
    A PyTorch module implementing the Snake activation function in 1D.

    Parameters
    ----------
    channels : int
        The number of channels in the input tensor.
    """

    def __init__(self, channels):
        """
        Initializes Snake1d
        Parameters
        ----------
        channels : int
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        """

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return snake(x, self.alpha)


class ResidualUnit(nn.Module):
    """
    A residual unit module for convolutional neural networks.

    Parameters
    ----------
    dim : int, optional
        The number of channels in the input tensor. Default is 16.
    dilation : int, optional
        The dilation rate for the convolutional layers. Default is 1.

    """

    def __init__(self, dim: int = 16, dilation: int = 1):
        """
        Initializes the ResidualUnit
        Parameters
        ----------
        dim : int, optional, by default 16
        dilation : int, optional, by default 1
        """
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Parameters
        ----------
        x : torch.tensor

        Returns
        -------
        torch.tensor
        """
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    """
    An encoder block module for convolutional neural networks.

    This module constructs an encoder block consisting of a series of ResidualUnits and a final Snake1d
    activation followed by a weighted normalized 1D convolution. This block can be used as part of an
    encoder in architectures like autoencoders.

    Parameters
    ----------
    dim : int, optional
        The number of output channels. Default is 16.
    stride : int, optional
        The stride for the final convolutional layer. Default is 1.
    """

    def __init__(self, dim: int = 16, stride: int = 1):
        """
        Initializes the EncoderBlock
        Parameters
        ----------
        dim : int, optional, by default 16
        stride : int, optional, by default 1
        """
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x: torch.tensor):
        """
        Parameters
        ----------
        x : torch.tensor

        Returns
        -------
        torch.tensor
        """
        return self.block(x)


class Encoder(nn.Module):
    """
    A PyTorch module for the Encoder part of DAC.

    Parameters
    ----------
    d_model : int, optional
        The initial dimensionality of the model. Default is 64.
    strides : list, optional
        A list of stride values for downsampling in each EncoderBlock. Default is [2, 4, 8, 8].
    d_latent : int, optional
        The dimensionality of the output latent space. Default is 64.

    Example
    -------
    Creating an Encoder instance
    >>> encoder = Encoder()
    >>> audio_input = torch.randn(1, 1, 44100) # Example shape: [Batch, Channels, Time]
    >>> continuous_embedding = encoder(audio_input)

    Using a pretrained encoder.

    >>> dac = DAC(load_pretrained=True, model_type="44KHz", model_bitrate="8kbps", tag="latest")
    >>> encoder = dac.encoder
    >>> audio_input = torch.randn(1, 1, 44100) # Example shape: [Batch, Channels, Time]
    >>> continuous_embeddings = encoder(audio_input)
    """

    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
    ):
        """
        Initializes the Encoder

        Parameters
        ----------
        d_model : int, optional, by default 64
        strides : list, optional, by default [2, 4, 8, 8]
        d_latent : int, optional, by default 64
        """
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.tensor

        Returns
        -------
        torch.tensor
        """
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    A PyTorch module representing a block within the Decoder architecture.

    Parameters
    ----------
    input_dim : int, optional
        The number of input channels. Default is 16.
    output_dim : int, optional
        The number of output channels. Default is 8.
    stride : int, optional
        The stride for the transposed convolution, controlling the upsampling. Default is 1.
    """

    def __init__(
        self, input_dim: int = 16, output_dim: int = 8, stride: int = 1
    ):
        """
        Initializes the DecoderBlock

        Parameters
        ----------
        input_dim : int, optional, by default 16
        output_dim : int, optional, by default 8
        stride : int, optional, by default 1
        """
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        """

        Parameters
        ----------
        x : torch.tensor

        Returns
        -------
        torch.tensor
        """
        return self.block(x)


class Decoder(nn.Module):
    """
    A PyTorch module for the Decoder part of DAC.

    Parameters
    ----------
    input_channel : int
        The number of channels in the input tensor.
    channels : int
        The base number of channels for the convolutional layers.
    rates : list
        A list of stride rates for each decoder block
    d_out: int
        The out dimension of the final conv layer, Default is 1.

    Example
    -------
    Creating a Decoder instance

    >>> decoder = Decoder(256, 1536,  [8, 8, 4, 2])
    >>> discrete_embeddings = torch.randn(2, 256, 200) # Example shape: [Batch, Channels, Time]
    >>> recovered_audio = decoder(discrete_embeddings)

    Using a pretrained decoder. Note that the actual input should be proper discrete representation.
    Using randomly generated input here for illustration of use.

    >>> dac = DAC(load_pretrained=True, model_type="44KHz", model_bitrate="8kbps", tag="latest")
    >>> decoder = dac.decoder
    >>> discrete_embeddings = torch.randn(1, 1024, 500) # Example shape: [Batch, Channels, Time]
    >>> recovered_audio = decoder(discrete_embeddings)
    """

    def __init__(
        self,
        input_channel: int,
        channels: int,
        rates: List[int],
        d_out: int = 1,
    ):
        """Initializes Decoder

        Parameters
        ----------
        input_channel : int
        channels : int
        rates : List[int]
        d_out : int, optional, by default 1
        """
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2 ** i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """

        Parameters
        ----------
        x : torch.tensor

        Returns
        -------
        torch.tensor
        """
        return self.model(x)


class DAC(nn.Module):
    """
    Discrete Autoencoder Codec (DAC) for audio data encoding and decoding.

    This class implements an autoencoder architecture with quantization for efficient audio processing.
    It includes an encoder, quantizer, and decoder for transforming audio data into a compressed latent representation and reconstructing it back into audio.
    This implementation supports both initializing a new model and loading a pretrained model.

    Parameters
    ----------
    encoder_dim : int
        Dimensionality of the encoder.
    encoder_rates : List[int]
        Downsampling rates for each encoder layer.
    latent_dim : int, optional
        Dimensionality of the latent space, automatically calculated if None.
    decoder_dim : int
        Dimensionality of the decoder.
    decoder_rates : List[int]
        Upsampling rates for each decoder layer.
    n_codebooks : int
        Number of codebooks for vector quantization.
    codebook_size : int
        Size of each codebook.
    codebook_dim : Union[int, list]
        Dimensionality of each codebook entry.
    quantizer_dropout : bool
        Whether to use dropout in the quantizer.
    sample_rate : int
        Sample rate of the audio data.
    model_type : str
        Type of the model to load (if pretrained).
    model_bitrate : str
        Bitrate of the model to load (if pretrained).
    tag : str
        Specific tag of the model to load (if pretrained).
    load_path : str, optional
        Path to load the pretrained model from, automatically downloaded if None.
    strict : bool
        Whether to strictly enforce the state dictionary match.
    load_pretrained : bool
        Whether to load a pretrained model.

    Example
    -------
    Creating a new DAC instance:

    >>> dac = DAC()
    >>> audio_data = torch.randn(1, 1, 16000) # Example shape: [Batch, Channels, Time]
    >>> tokens, embeddings = dac(audio_data)


    Loading a pretrained DAC instance:

    >>> dac = DAC(load_pretrained=True, model_type="44KHz", model_bitrate="8kbps", tag="latest")
    >>> audio_data = torch.randn(1, 1, 16000) # Example shape: [Batch, Channels, Time]
    >>> tokens, embeddings = dac(audio_data)

    The tokens and the discrete embeddings obtained above or from other sources can be decoded:

    >>> dac = DAC(load_pretrained=True, model_type="44KHz", model_bitrate="8kbps", tag="latest")
    >>> audio_data = torch.randn(1, 1, 16000) # Example shape: [Batch, Channels, Time]
    >>> tokens, embeddings = dac(audio_data)
    >>> decoded_audio = dac.decode(embeddings)
    """

    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: bool = False,
        sample_rate: int = 44100,
        model_type: str = "44khz",
        model_bitrate: str = "8kbps",
        tag: str = "latest",
        load_path: str = None,
        strict: bool = False,
        load_pretrained: bool = False,
    ):
        """ Initializes DAC

        Parameters
        ----------
        encoder_dim : int, optional, by default 64
        encoder_rates : List[int], optional, by default [2, 4, 8, 8]
        latent_dim : int, optional, by default None
        decoder_dim : int, optional, by default 1536
        decoder_rates : List[int], optional, by default [8, 8, 4, 2]
        n_codebooks : int, optional, by default 9
        codebook_size : int, optional, by default 1024
        codebook_dim : Union[int, list], optional, by default 8
        quantizer_dropout : bool, optional, by default False
        sample_rate : int, optional, by default 44100
        model_type : str, optional, by default "44khz"
        model_bitrate : str, optional, by default "8kbps"
        tag : str, optional, by default "latest"
        load_path : str, optional, by default None
        strict : bool, optional, by default False
        load_pretrained : bool, optional
             If True, then a pretrained model is loaded, by default False
        """
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.latent_dim = latent_dim
        self.quantizer_dropout = quantizer_dropout

        if load_pretrained:
            if not load_path:
                load_path = download(
                    model_type=model_type, model_bitrate=model_bitrate, tag=tag
                )
                logger.info(f"Obtained load path as: {load_path}")
            model_dict = torch.load(load_path, "cpu")
            metadata = model_dict["metadata"]
            for key, value in metadata["kwargs"].items():
                setattr(self, key, value)

        self.hop_length = np.prod(self.encoder_rates)
        if self.latent_dim is None:
            self.latent_dim = self.encoder_dim * (2 ** len(self.encoder_rates))
        self.encoder = Encoder(
            self.encoder_dim, self.encoder_rates, self.latent_dim
        )
        self.quantizer = ResidualVectorQuantize(
            input_dim=self.latent_dim,
            n_codebooks=self.n_codebooks,
            codebook_size=self.codebook_size,
            codebook_dim=self.codebook_dim,
            quantizer_dropout=self.quantizer_dropout,
        )
        self.decoder = Decoder(
            self.latent_dim, self.decoder_dim, self.decoder_rates,
        )
        self.apply(init_weights)

        if load_pretrained:
            self.load_state_dict(model_dict["state_dict"], strict=strict)
            self.metadata = metadata

    def encode(
        self, audio_data: torch.Tensor, n_quantizers: int = None,
    ):
        """Encode given audio data and return quantized latent codes

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        n_quantizers : int, optional
            Number of quantizers to use, by default None
            If None, all quantizers are used.

        Returns
        -------
        "z" : Tensor[B x D x T]
            Quantized continuous representation of input
        "codes" : Tensor[B x N x T]
            Codebook indices for each codebook
            (quantized discrete representation of input)
        "latents" : Tensor[B x N*D x T]
            Projected latents (continuous representation of input before quantization)
        "vq/commitment_loss" : Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        "vq/codebook_loss" : Tensor[1]
            Codebook loss to update the codebook
        "length" : int
            Number of samples in input audio
        """
        z = self.encoder(audio_data)
        z, codes, latents, commitment_loss, codebook_loss = self.quantizer(
            z, n_quantizers
        )
        return z, codes, latents, commitment_loss, codebook_loss

    def decode(self, z: torch.Tensor):
        """Decode given latent codes and return audio data

        Parameters
        ----------
        z : Tensor[B x D x T]
            Quantized continuous representation of input
        length : int, optional
            Number of samples in output audio, by default None

        Returns
        -------
        torch.Tensor: shape B x 1 x length
            Decoded audio data.
        """
        return self.decoder(z)

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
    ):
        """Model forward pass

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        sample_rate : int, optional
            Sample rate of audio data in Hz, by default None
            If None, defaults to `self.sample_rate`
        n_quantizers : int, optional
            Number of quantizers to use, by default None.
            If None, all quantizers are used.

        Returns
        -------
        "tokens" : Tensor[B x N x T]
            Codebook indices for each codebook
            (quantized discrete representation of input)
        "embeddings" : Tensor[B x D x T]
            Quantized continuous representation of input
        """
        # Preprocess the audio data to have the right padded lengths
        length = audio_data.shape[-1]
        right_pad = (
            math.ceil(length / self.hop_length) * self.hop_length - length
        )
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        z, codes, _, _, _ = self.encode(audio_data, n_quantizers)
        return codes, z
