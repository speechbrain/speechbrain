"""This lobes enables the integration of fairseq pretrained wav2vec2.0 models.

Reference: https://arxiv.org/abs/2006.11477
FairSeq needs to be installed: https://fairseq.readthedocs.io/en/latest/

Authors
 * Titouan Parcollet 2021
 * Salima Mdhaffar 2021
"""

import fairseq
from torch import nn
import torch.nn.functional as F
from speechbrain.utils.data_utils import download_file


class FairseqWav2Vec2(nn.Module):
    """This lobes enables the integration of fairseq pretrained wav2vec2.0 models.

    Source paper: https://arxiv.org/abs/2006.11477
    FairSeq needs to be installed: https://fairseq.readthedocs.io/en/latest/

    The model can be used as a fixed features extractor or can be finetuned. It
    will download automatically the model if a url is given (e.g FairSeq
    repository).

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

    Example
    -------
    >>> inputs = torch.rand([10, 600])
    >>> model_url = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt"
    >>> save_path = "models_checkpoints/wav2vec2.pt"
    >>> model = FairseqWav2Vec2(model_url, save_path)
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([10, 768])
    """

    def __init__(
        self, pretrained_path, save_path, output_norm=True, freeze=True
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
        # Hence, we check of the model has be trained with or without it.
        self.normalize = cfg.normalize
        model = model[0]
        self.model = model
        self.freeze = freeze
        self.output_norm = output_norm
        if self.freeze:
            model.eval()

    def forward(self, x):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        x : torch.Tensor (signal)

        Outputs
        --------
        x : torch.Tensor [batch, time, features]

        """

        # If freeze is specified, we remove the model from the gradient graph.
        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad = True

        # We normalize the input signal if needed.
        if self.normalize:
            x = F.layer_norm(x, x.shape)

        # Extract wav2vec output
        out = self.model.extract_features(x, padding_mask=None, mask=False)[0]

        # We normalize the output if required
        if self.output_norm:
            out = F.layer_norm(out, out.shape)

        return out
