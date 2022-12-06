"""This lobe enables the integration of huggingface pretrained wav2vec2/hubert/wavlm models.

Reference: https://arxiv.org/abs/2006.11477
Reference: https://arxiv.org/abs/1904.05862
Reference: https://arxiv.org/abs/2110.13900
Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Titouan Parcollet 2021
 * Boumadane Abdelmoumene 2021
 * Andreas Nautsch 2022
"""
import logging
from functools import partial
from speechbrain.lobes.models.huggingface.transformers import (
    HuggingFaceTransformer,
)
from speechbrain.lobes.models.huggingface.overrides import (
    modify_state_dict_wav2vec2,
    config_return_hidden_states,
    model_set_spectral_augmentation,
)
from speechbrain.lobes.models.huggingface.forward import (
    wav2vec2,
    wav2vec2_pretraining,
)

# We check if transformers is installed.
try:
    import transformers
    from transformers import Wav2Vec2ForPreTraining

except ImportError:
    MSG = "Please install transformers from HuggingFace to use wav2vec2 / Hubert\n"
    MSG += "E.G. run: pip install transformers"
    raise ImportError(MSG)

logger = logging.getLogger(__name__)


class HuggingFaceWav2Vec2(HuggingFaceTransformer):
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
    cache_dir: str or Path (default: None)
        Location of HuggingFace cache for storing pre-trained models, to which symlinks are created.

    Example
    -------
    >>> import torch
    >>> inputs = torch.rand([10, 600])
    >>> model_hub = "facebook/wav2vec2-base-960h"
    >>> save_path = "tmp"
    >>> model = HuggingFaceWav2Vec2(model_hub, save_path)
    >>> outputs = model(inputs)
    """

    def __init__(
        self,
        source,
        save_path,
        output_norm=True,
        freeze=True,
        freeze_feature_extractor=False,
        apply_spec_augment=False,
        output_all_hiddens=False,
        cache_dir=None,
    ):
        super().__init__(
            source,
            save_path=save_path,
            freeze=freeze,
            modify_state_dict_partial_fn=partial(modify_state_dict_wav2vec2),
            forward_partial_fn=partial(
                wav2vec2,
                output_all_hiddens=output_all_hiddens,
                output_norm=output_norm,  # TODO drop: superfluous
                normalize_wav=transformers.Wav2Vec2FeatureExtractor.from_pretrained(  # TODO drop: superfluous
                    source, cache_dir=save_path
                ).do_normalize,
            ),
            override_hf_model_partial_fn=partial(
                model_set_spectral_augmentation,
                apply_spec_augment=apply_spec_augment,
            ),
            freeze_nested_models_their_calls="feature_extractor._freeze_parameters"
            if freeze_feature_extractor
            else None,
            cache_dir=cache_dir,
        )


class HuggingFaceWav2Vec2Pretrain(HuggingFaceTransformer):
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
    mask_length : int (default: 10)
        Length (i.e. number of consecutive masked frames). Default is taken from
        the paper.
    cache_dir: str or Path (default: None)
        Location of HuggingFace cache for storing pre-trained models, to which symlinks are created.

    Example
    -------
    >>> import torch
    >>> inputs = torch.rand([10, 32000])
    >>> model_hub = "facebook/wav2vec2-base-960h"
    >>> save_path = "tmp"
    >>> model = HuggingFaceWav2Vec2Pretrain(model_hub, save_path)
    >>> outputs, _ = model(inputs)
    """

    def __init__(
        self,
        source,
        save_path,
        mask_prob=0.65,
        mask_length=10,
        normalize_wav=True,
        cache_dir=None,
    ):
        super().__init__(
            source,
            save_path=save_path,
            for_pretraining_cls=Wav2Vec2ForPreTraining,
            override_hf_config_partial_fn=partial(config_return_hidden_states),
            forward_partial_fn=partial(
                wav2vec2_pretraining,
                mask_prob=mask_prob,
                mask_length=mask_length,
                normalize_wav=normalize_wav,  # TODO drop: superfluous
            ),
            freeze=False,
            cache_dir=cache_dir,
        )
