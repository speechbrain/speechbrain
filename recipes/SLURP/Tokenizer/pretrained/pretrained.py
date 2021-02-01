"""
Pre-trained Tokenizer for inference.

Authors
 * Mirco Ravanelli 2020
 * Abdel Heba 2020
"""
import os
from speechbrain.utils.data_utils import download_file
import sentencepiece as spm


class tokenizer:
    """Downloads and loads the pretrained tokenizer.

    Arguments
    ---------
    tokenizer_file : str
        Path where the tokenizer is stored. If it is an url, the
        tokenizer is downloaded.
    save_folder: str
        Folder where the learned tokenizer will be stored

    Example
    -------
    >>> from pretrained import tokenizer
    >>> token_file = 'pretrained_tok/51_unigram.model'
    >>> tokenizer = tokenizer(token_file)
    >>> text = "{'intent': 'SetAlarm'| 'slots': {'am_or_pm': 'AM'}"
    >>> print(tokenizer.spm.encode(text))
    >>> print(tokenizer.spm.encode(text, out_type='str'))
    """

    def __init__(self, tokenizer_file, save_folder="tokenizer_model"):
        super().__init__()

        save_file = os.path.join(save_folder, "tok.model")
        download_file(
            source=tokenizer_file, dest=save_file, replace_existing=True,
        )
        tokenizer_file = save_file

        # Defining tokenizer and loading it
        self.spm = spm.SentencePieceProcessor()
        self.spm.load(tokenizer_file)
