"""
Pre-trained Tokenizer for inference.

Authors
 * Mirco Ravanelli 2020
 * Abdel Heba 2020
"""
import os
import tempfile
from speechbrain.utils.data_utils import download_file
import sentencepiece as spm


class tokenizer:
    """Downloads and loads the pretrained tokenizer.

    Arguments
    ---------
    tokenizer_file : str
        Path where the tokenizer is stored. If it is an url, the
        tokenizer is downloaded.
    save_folder : str
        Path where the tokenizer will be saved (default 'model_checkpoints')
    Examples
    -------
    >>> from pretrained import tokenizer
    >>> token_file = 'pretrained_tok/1000_unigram.model'
    >>> save_dir = 'model_checkpoints'
    >>> tokenizer = tokenizer(token_file, save_dir)
    >>> text = "THE CAT IS ON THE TABLE"
    >>> print(tokenizer.spm.encode(text))
    >>> print(tokenizer.spm.encode(text, out_type='str'))
    """

    def __init__(self, tokenizer_file, save_folder='model_checkpoints'):
        super().__init__()

        save_file = os.path.join(save_folder, "tok.model")
        download_file(
            source=tokenizer_file, dest=save_file, replace_existing=False,
        )

        # Defining tokenizer and loading it
        self.spm = spm.SentencePieceProcessor()
        self.spm.load(save_file)
