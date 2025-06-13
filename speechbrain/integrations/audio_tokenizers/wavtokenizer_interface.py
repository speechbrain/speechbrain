"""This lobe enables the integration of pretrained WavTokenizer.

Note that you need to pip install `git+https://github.com/Tomiinek/WavTokenizer` to use this module.

Repository: https://github.com/jishengpeng/WavTokenizer/
Paper: https://arxiv.org/abs/2408.16532

Authors
 * Pooneh Mousavi 2024
"""

import os

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download


class WavTokenizer(nn.Module):
    """This lobe enables the integration of pretrained WavTokenizer model, a discrete codec models with single codebook for Audio Language Modeling.

    Source paper:
        https://arxiv.org/abs/2408.16532

    You need to pip install `git+https://github.com/Tomiinek/WavTokenizer` to use this module.

    The code is adapted from the official WavTokenizer repository:
    https://github.com/jishengpeng/WavTokenizer/

    Arguments
    ---------
    source : str
        A HuggingFace repository identifier or a path
    save_path : str
        The location where the pretrained model will be saved
    config : str
        The name of the HF config file.
    checkpoint : str
        The name of the HF checkpoint file.
    sample_rate : int (default: 24000)
        The audio sampling rate
    freeze : bool
        whether the model will be frozen (e.g. not trainable if used
        as part of training another model)

    Example
    -------
    >>> model_hub = "novateur/WavTokenizer"
    >>> save_path = "savedir"
    >>> config="wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    >>> checkpoint="WavTokenizer_small_600_24k_4096.ckpt"
    >>> model = WavTokenizer(model_hub, save_path,config=config,checkpoint=checkpoint)
    >>> audio = torch.randn(4, 48000)
    >>> length = torch.tensor([1.0, .5, .75, 1.0])
    >>> tokens, embs= model.encode(audio)
    >>> tokens.shape
    torch.Size([4, 1, 80])
    >>> embs.shape
    torch.Size([4, 80, 512])
    >>> rec = model.decode(tokens)
    >>> rec.shape
    torch.Size([4, 48000])
    """

    def __init__(
        self,
        source,
        save_path=None,
        config="wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
        checkpoint="WavTokenizer_small_600_24k_4096.ckpt",
        sample_rate=24000,
        freeze=True,
    ):
        # Lazy import to avoid circular dependency issues
        try:
            import wavtokenizer

            self.wavtokenizer = wavtokenizer
        except ImportError:
            raise ImportError(
                "Please install the WavTokenizer module using: "
                "`pip install git+https://github.com/Tomiinek/WavTokenizer`"
            )

        super().__init__()

        path = snapshot_download(repo_id=source, cache_dir=save_path)
        checkpoint_path = os.path.join(path, checkpoint)
        config_path = os.path.join(path, config)
        self.model = self.wavtokenizer.WavTokenizer.from_pretrained0802(
            config_path, checkpoint_path
        )
        self.embeddings = self._compute_embedding()
        self.sample_rate = sample_rate

    def forward(self, inputs):
        """Encodes the input audio as tokens and embeddings and  decodes audio from tokens

        Arguments
        ---------
        inputs : torch.Tensor
            A (Batch x Samples)
            tensor of audio
        Returns
        -------
        tokens : torch.Tensor
            A (Batch x Tokens x Heads) tensor of audio tokens
        emb : torch.Tensor
            Raw vector embeddings from the model's
            quantizers
        audio : torch.Tensor
            the reconstructed audio
        """

        tokens, embedding = self.encode(inputs)
        audio = self.decode(tokens)

        return tokens, embedding, audio

    @torch.no_grad()
    def _compute_embedding(self):
        embs = self.model.feature_extractor.encodec.quantizer.vq.layers[
            0
        ].codebook
        return embs

    def encode(self, inputs):
        """Encodes the input audio as tokens and embeddings

        Arguments
        ---------
        inputs : torch.Tensor
            A (Batch x Samples) or (Batch x Channel x Samples)
            tensor of audio

        Returns
        -------
        tokens : torch.Tensor
            A (Batch x NQ x Length) tensor of audio tokens
        emb : torch.Tensor
            Raw vector embeddings from the model's
            quantizers
        """
        emb, tokens = self.model.encode(inputs, bandwidth_id=0)
        return tokens.movedim(0, 1), emb.movedim(1, -1)

    def decode(
        self,
        tokens,
    ):
        """Decodes audio from tokens

        Arguments
        ---------
        tokens : torch.Tensor
            A (Batch x NQ x Length) tensor of audio tokens
        Returns
        -------
        audio : torch.Tensor
            the reconstructed audio
        """
        feats = self.model.codes_to_features(tokens.movedim(1, 0))
        sig = self.model.decode(
            feats, bandwidth_id=torch.tensor(0, device=tokens.device)
        )
        return sig
