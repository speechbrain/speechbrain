"""This lobe enables the integration of fairseq pretrained wav2vec models.

Reference: https://arxiv.org/abs/2006.11477
Reference: https://arxiv.org/abs/1904.05862
FairSeq needs to be installed: https://fairseq.readthedocs.io/en/latest/

Authors
 * Titouan Parcollet 2021
 * Salima Mdhaffar 2021
"""

import torch
import torch.nn.functional as F
from torch import nn
from speechbrain.utils.data_utils import download_file

# We check if fairseq is installed.
try:
    import fairseq
except ImportError:
    print("Please install Fairseq to use pretrained wav2vec!")


class FairseqWav2Vec2(nn.Module):
    """This lobe enables the integration of fairseq pretrained wav2vec2.0 models.

    Source paper: https://arxiv.org/abs/2006.11477
    FairSeq needs to be installed: https://fairseq.readthedocs.io/en/latest/

    The model can be used as a fixed features extractor or can be finetuned. It
    will download automatically the model if a url is given (e.g FairSeq
    repository from GitHub).

    Arguments
    ---------
    pretrained_path : str
        Path of the pretrained wav2vec2 model. It can be a url or a local path.
    save_path : str
        Path and filename of the downloaded model.
    output_norm : bool (default: True)
        If True, a layer_norm (affine) will be applied to the output obtained
        from the wav2vec model.
    freeze : bool (default: True)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    pretrain : bool (default: True)
        If True, the model is pretrained with the specified source.
        If False, the randomly-initialized model is instantiated.

    Example
    -------
    >>> inputs = torch.rand([10, 600])
    >>> model_url = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt"
    >>> save_path = "models_checkpoints/wav2vec2.pt"
    >>> model = FairseqWav2Vec2(model_url, save_path)
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([10, 100,  768])
    """

    def __init__(
        self,
        pretrained_path,
        save_path,
        output_norm=True,
        freeze=True,
        pretrain=True,
    ):
        super().__init__()

        # Download the pretrained wav2vec2 model. It can be local or online.
        download_file(pretrained_path, save_path)

        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([save_path])

        # wav2vec pretrained models may need the input waveform to be normalized
        # Hence, we check if the model has be trained with or without it.
        self.normalize = cfg.normalize
        model = model[0]
        self.model = model
        self.freeze = freeze
        self.output_norm = output_norm
        if self.freeze:
            model.eval()

        # Randomly initialized layers if pretrain is False
        if not (pretrain):
            self.reset_layer(self.model)

    def forward(self, wav):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """

        # If we freeze, we simply remove all grads and features from the graph.
        if self.freeze:
            with torch.no_grad():
                return self.extract_features(wav).detach()

        return self.extract_features(wav)

    def extract_features(self, wav):
        """Extracts the wav2vect embeddings"""
        # We normalize the input signal if needed.
        if self.normalize:
            wav = F.layer_norm(wav, wav.shape)

        # Extract wav2vec output
        out = self.model.extract_features(wav, padding_mask=None, mask=False)[0]

        # We normalize the output if required
        if self.output_norm:
            out = F.layer_norm(out, out.shape)

        return out

    def reset_layer(self, model):
        """Reinitializes the parameters of the network"""
        if hasattr(model, "reset_parameters"):
            model.reset_parameters()
        for child_layer in model.children():
            if model != child_layer:
                self.reset_layer(child_layer)


class FairseqWav2Vec1(nn.Module):
    """This lobes enables the integration of fairseq pretrained wav2vec1.0 models.

    Arguments
    ---------
    pretrained_path : str
        Path of the pretrained wav2vec1 model. It can be a url or a local path.
    save_path : str
        Path and filename of the downloaded model.
    output_norm : bool (default: True)
        If True, a layer_norm (affine) will be applied to the output obtained
        from the wav2vec model.
    freeze : bool (default: True)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    pretrain : bool (default: True)
        If True, the model is pretrained with the specified source.
        If False, the randomly-initialized model is instantiated.

    Example
    -------
    >>> inputs = torch.rand([10, 600])
    >>> model_url = ""
    >>> save_path = "models_checkpoints/wav2vec.pt"
    >>> model = FairseqWav2Vec1(model_url, save_path)
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([10, 100, 512])
    """

    def __init__(
        self,
        pretrained_path,
        save_path,
        output_norm=True,
        freeze=True,
        pretrain=True,
    ):
        super().__init__()
        self.freeze = freeze
        self.output_norm = output_norm

        # Download the pretrained wav2vec1 model. It can be local or online.
        download_file(pretrained_path, save_path)

        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [pretrained_path]
        )

        self.model = model
        self.model = self.model[0]
        if self.freeze:
            model.eval()

        # Randomly initialized layers if pretrain is False
        if not (pretrain):
            self.reset_layer(self.model)

    def forward(self, wav):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """

        # If we freeze, we simply remove all grads and features from the graph.
        if self.freeze:
            with torch.no_grad():
                return self.extract_features(wav).detach()

        return self.extract_features(wav)

    def extract_features(self, wav):
        """Extracts the wav2vect embeddings"""

        out = self.model.feature_extractor(wav)
        out = self.model.feature_aggregator(out).squeeze(0)
        out = out.transpose(2, 1)

        # We normalize the output if required
        if self.output_norm:
            out = F.layer_norm(out, out.shape)

        return out

    def reset_layer(self, model):
        """Reinitializes the parameters of the network"""
        if hasattr(model, "reset_parameters"):
            model.reset_parameters()
        for child_layer in model.children():
            if model != child_layer:
                self.reset_layer(child_layer)
