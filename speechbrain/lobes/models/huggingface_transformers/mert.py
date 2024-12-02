"""This lobe enables the integration of huggingface pretrained MERT models, an acoustic Music Understanding Model with Large-Scale Self-supervised Training.

Reference: https://arxiv.org/abs/2306.00107

Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Pooneh Mousavi 2024
"""

import logging

from speechbrain.lobes.models.huggingface_transformers.wav2vec2 import Wav2Vec2

logger = logging.getLogger(__name__)


class MERT(Wav2Vec2):
    """
    A class for integrating HuggingFace and SpeechBrain pretrained MERT models, enabling
    usage as a feature extractor or for fine-tuning purposes.

    Source paper MERT: https://arxiv.org/abs/2306.00107
    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html

    The model can be used as a fixed feature extractor or can be finetuned. It
    will download automatically the model from HuggingFace or use a local path.

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "m-a-p/MERT-v1-330M"
    save_path : str
        Path (dir) of the downloaded model.
    output_norm : bool (default: True)
        If True, a layer_norm (affine) will be applied to the output obtained
        from the mert model.
    freeze : bool (default: True)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    freeze_feature_extractor :  bool (default: False)
        When freeze = False and freeze_feature_extractor True, the feature_extractor module of the model is Frozen. If False
        all the mert model will be trained including feature_extractor module.
    apply_spec_augment : bool (default: False)
        If True, the model will apply spec augment on the output of feature extractor
        (inside huggingface mertModel() class).
        If False, the model will not apply spec augment. We set this to false to prevent from doing it twice.
    output_all_hiddens : bool (default: False)
        If True, the forward function outputs the hidden states from all transformer layers.
        For example MERT-v1-95M has 12 transformer layers and the output is of shape (13, B, T, C),
        where a projection of the CNN output is added to the beginning.
        If False, the forward function outputs the hidden states only from the last transformer layer.

    Example
    -------
    >>> import torch
    >>> inputs = torch.rand([10, 600])
    >>> model_hub = "m-a-p/MERT-v1-95M"
    >>> save_path = "savedir"
    >>> model = MERT(model_hub, save_path)  # doctest:+ELLIPSIS
    WARNING: ...
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([10, 1, 768])
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
        super().__init__(
            source=source,
            save_path=save_path,
            output_norm=output_norm,
            freeze=freeze,
            freeze_feature_extractor=freeze_feature_extractor,
            apply_spec_augment=apply_spec_augment,
            output_all_hiddens=output_all_hiddens,
            trust_remote_code=True,
        )
