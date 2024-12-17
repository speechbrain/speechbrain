"""This lobe enables the integration of huggingface pretrained NLLB models.
Reference: https://arxiv.org/abs/2207.04672

Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Ha Nguyen 2023
"""

from speechbrain.lobes.models.huggingface_transformers.mbart import mBART
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class NLLB(mBART):
    """This lobe enables the integration of HuggingFace and SpeechBrain
    pretrained NLLB models.

    Source paper NLLB: https://arxiv.org/abs/2207.04672
    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html

    The model is normally used as a text decoder of seq2seq models. It
    will download automatically the model from HuggingFace or use a local path.

    For now, HuggingFace's NLLB model can be loaded using the exact code for mBART model.
    For this reason, NLLB can be fine inheriting the mBART class.

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "facebook/nllb-200-1.3B"
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
    >>> import torch
    >>> src = torch.rand([10, 1, 1024])
    >>> tgt = torch.LongTensor([[256057,    313,     25,    525,    773,  21525,   4004,      2]])
    >>> model_hub = "facebook/nllb-200-distilled-600M"
    >>> save_path = "savedir"
    >>> model = NLLB(model_hub, save_path)
    >>> outputs = model(src, tgt)
    """

    def __init__(
        self,
        source,
        save_path,
        freeze=True,
        target_lang="fra_Latn",
        decoder_only=True,
        share_input_output_embed=True,
    ):
        super().__init__(
            source=source,
            save_path=save_path,
            freeze=freeze,
            target_lang=target_lang,
            decoder_only=decoder_only,
            share_input_output_embed=share_input_output_embed,
        )
