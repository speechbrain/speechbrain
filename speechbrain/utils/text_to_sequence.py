"""from https://github.com/keithito/tacotron"""

# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import re

from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)

valid_symbols = [
    "AA",
    "AA0",
    "AA1",
    "AA2",
    "AE",
    "AE0",
    "AE1",
    "AE2",
    "AH",
    "AH0",
    "AH1",
    "AH2",
    "AO",
    "AO0",
    "AO1",
    "AO2",
    "AW",
    "AW0",
    "AW1",
    "AW2",
    "AY",
    "AY0",
    "AY1",
    "AY2",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "EH0",
    "EH1",
    "EH2",
    "ER",
    "ER0",
    "ER1",
    "ER2",
    "EY",
    "EY0",
    "EY1",
    "EY2",
    "F",
    "G",
    "HH",
    "IH",
    "IH0",
    "IH1",
    "IH2",
    "IY",
    "IY0",
    "IY1",
    "IY2",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OW0",
    "OW1",
    "OW2",
    "OY",
    "OY0",
    "OY1",
    "OY2",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UH0",
    "UH1",
    "UH2",
    "UW",
    "UW0",
    "UW1",
    "UW2",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
]


"""
Defines the set of symbols used in text input to the model.
The default is a set of ASCII characters that works well for English. For other data, you can modify _characters. See TRAINING_DATA.md for details.
"""


_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same
# as uppercase letters):
_arpabet = ["@" + s for s in valid_symbols]

# Export all symbols:
symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")


# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "missus"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]


def expand_abbreviations(text):
    """Expand abbreviations pre-defined"""
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


# def expand_numbers(text):
#  return normalize_numbers(text)


def lowercase(text):
    """Lowercase the text"""
    return text.lower()


def collapse_whitespace(text):
    """Replaces whitespace by " " in the text"""
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    """Converts text to ascii"""
    text_encoded = text.encode("ascii", "ignore")
    return text_encoded.decode()


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def german_cleaners(text):
    """Pipeline for German text, that collapses whitespace without transliteration."""
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    """Pipeline for English text, including number and abbreviation expansion."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text


def text_to_sequence(text, cleaner_names):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Arguments
    ---------
    text : str
        string to convert to a sequence
    cleaner_names : list
        names of the cleaner functions to run the text through

    Returns
    -------
    sequence : list
        The integers corresponding to the symbols in the text.
    """
    sequence = []

    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)
        if not m:
            sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)

    return sequence


def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == "@":
                s = "{%s}" % s[1:]
            result += s
    return result.replace("}{", " ")


def _clean_text(text, cleaner_names):
    """Apply different cleaning pipeline according to cleaner_names"""
    for name in cleaner_names:
        if name == "english_cleaners":
            cleaner = english_cleaners
        if name == "transliteration_cleaners":
            cleaner = transliteration_cleaners
        if name == "basic_cleaners":
            cleaner = basic_cleaners
        if name == "german_cleaners":
            cleaner = german_cleaners
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols):
    """Convert symbols to sequence"""
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    """Prepend "@" to ensure uniqueness"""
    return _symbols_to_sequence(["@" + s for s in text.split()])


def _should_keep_symbol(s):
    """Whether to keep a certain symbol"""
    return s in _symbol_to_id and s != "_" and s != "~"


def _g2p_keep_punctuations(g2p_model, text):
    """Do grapheme to phoneme and keep the punctuations between the words

    Arguments
    ---------
    g2p_model: speechbrain.inference.text.GraphemeToPhoneme
        Model to apply to the given text while keeping punctuation.
    text: string
        the input text.

    Returns
    -------
    The text string's corresponding phoneme symbols with punctuation symbols.

    Example
    -------
    >>> from speechbrain.inference.text import GraphemeToPhoneme
    >>> g2p_model = GraphemeToPhoneme.from_hparams("speechbrain/soundchoice-g2p") # doctest: +SKIP
    >>> from speechbrain.utils.text_to_sequence import _g2p_keep_punctuations # doctest: +SKIP
    >>> text = "Hi, how are you?" # doctest: +SKIP
    >>> _g2p_keep_punctuations(g2p_model, text) # doctest: +SKIP
    ['HH', 'AY', ',', ' ', 'HH', 'AW', ' ', 'AA', 'R', ' ', 'Y', 'UW', '?']
    """
    # find the words where a "-" or "'" or "." or ":" appears in the middle
    special_words = re.findall(r"\w+[-':\.][-':\.\w]*\w+", text)

    # remove intra-word punctuations ("-':."), this does not change the output of speechbrain g2p
    for special_word in special_words:
        rmp = special_word.replace("-", "")
        rmp = rmp.replace("'", "")
        rmp = rmp.replace(":", "")
        rmp = rmp.replace(".", "")
        text = text.replace(special_word, rmp)

    # keep inter-word punctuations
    all_ = re.findall(r"[\w]+|[-!'(),.:;? ]", text)
    try:
        phonemes = g2p_model(text)
    except RuntimeError:
        logger.info(f"error with text: {text}")
        quit()
    word_phonemes = "-".join(phonemes).split(" ")

    phonemes_with_punc = []
    count = 0
    try:
        # if the g2p model splits the words correctly
        for i in all_:
            if i not in "-!'(),.:;? ":
                phonemes_with_punc.extend(word_phonemes[count].split("-"))
                count += 1
            else:
                phonemes_with_punc.append(i)
    except IndexError:
        # sometimes the g2p model cannot split the words correctly
        logger.warning(
            f"Do g2p word by word because of unexpected outputs from g2p for text: {text}"
        )

        for i in all_:
            if i not in "-!'(),.:;? ":
                p = g2p_model.g2p(i)
                p_without_space = [i for i in p if i != " "]
                phonemes_with_punc.extend(p_without_space)
            else:
                phonemes_with_punc.append(i)

    while "" in phonemes_with_punc:
        phonemes_with_punc.remove("")
    return phonemes_with_punc
