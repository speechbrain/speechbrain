"""
Pre-trained Tokenizer for inference.

Authors
 * Mirco Ravanelli 2020
 * Abdel Heba 2020
"""
import os
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.data_utils import download_from_huggingface
import sentencepiece as spm


class tokenizer:
    """ This class provides two possible way of using pretrained tokenizers:
    1. Downloads from HuggingFace and loads the pretrained tokenizer.
    2. Downloads from the web (or copy locally) and loads a pretrained tokenizer
    model if the checkpoint isn't stored on HuggingFace. This is particularly
    useful for tokenizers not ditributed by SpeechBrain. WARNING: When using
    custom tokenizers, make sure that your acoustic models and language models
    are train with the same tokenizer!!

    Arguments
    ---------
    tokenizer_file : str
        Path corresponding either to the HuggingFace path of the checkpoint
        within the model directory or to a custom user model path.
    model_name : str, optional
        HuggingFace model name (to be found on the SpeechBrain HuggingFace
        organization).
    save_folder : str
        Path where the tokenizer will be saved (default 'model_checkpoints')
    save_filename : str
        New filename of the model that has been downloaded.

    """

    def __init__(
        self,
        tokenizer_file,
        model_name=None,
        save_folder="model_checkpoints",
        save_filename="tok.model",
    ):
        super().__init__()

        save_file = os.path.join(save_folder, save_filename)

        if model_name is not None:
            download_from_huggingface(
                model_name, tokenizer_file, save_folder, save_filename
            )
        else:
            download_file(
                source=tokenizer_file, dest=save_file, replace_existing=False,
            )

        # Defining tokenizer and loading it
        self.spm = spm.SentencePieceProcessor()
        self.spm.load(save_file)
