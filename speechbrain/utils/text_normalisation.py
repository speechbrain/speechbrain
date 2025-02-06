"""Utilities to normalise text for speech recognition.

Authors
* Titouan Parcollet 2024
"""

import re

from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class TextNormaliser:
    """Used to normalise text with custom rules. Note that we do not provide any numeral conversion. This must be done before hand with, for instance the Nemo text processing tool."""

    def english_specific_preprocess(
        self, sentence, upper_case=True, symbols_limit=4
    ):
        """
        Preprocess English text. This function relies on different tools to convert numerals and special symbols. This also removes various punctuation and treats it as word boundaries. It normalises and retains various apostrophes (’‘´) between letters, but not other ones, which are probably quotation marks. It capitalises all text. This function may error out if new characters show up in the given sentence.

        Parameters
        ----------
        sentence : str
            The string to modify.
        upper_case : bool
            Whether to upper case (if True) or lower case (if False) the string.
        symbols_limit : int
            If a sentence contains more than symbols_limit, it will not be normalised and skipped. This is because in most case, the pronunciation will not be certain enough.
        Returns
        -------
        str
            The normalised sentence. Returns None if it was not possible to
            normalise the sentence.

        Example
        -------
        >>> norm = TextNormaliser()
        >>> txt = norm.english_specific_preprocess("Over the Rainbow! How are you today? Good + one hundred %")
        >>> print(txt)
        OVER THE RAINBOW HOW ARE YOU TODAY GOOD PLUS ONE HUNDRED PERCENT
        >>> txt = norm.english_specific_preprocess("Over the Rainbow! How are you today? Good + 100 %")
        >>> print(txt)
        None
        """

        # These characters mean we should discard the sentence, because the
        # pronunciation will be too uncertain.
        # if the sentence contains number we simply remove it.
        # This is because we expect the user to provide only text and symbols.
        # Numerals can be converted using the NeMo text processing tool.
        stop_characters = (
            "["
            "áÁàăâåäÄãÃāảạæćčČçÇðéÉèÈêěëęēəğíîÎïīịıłṃńňñóÓòôőõøØōŌœŒřšŠşșȘúÚûūụýžþ"
            # Suggests the sentence is not English but German.
            "öÖßüÜ"
            # All sorts of languages: Greek, Arabic...
            "\u0370-\u1AAF"
            # Chinese/Japanese/Korean.
            "\u4E00-\u9FFF"
            # Technical symbols.
            "\u2190-\u23FF"
            # Symbols that could be pronounced in various ways.
            "]"
        )
        if not re.search(stop_characters, sentence) is None:
            return None

        # encoding goes brrrrr
        sentence = self.clean_text(sentence)

        # These characters mark word boundaries.
        split_character_regex = '[ ",:;!?¡\\.…()\\-—–‑_“”„/«»]'

        # These could all be used as apostrophes in the middle of words.
        # If at the start or end of a word, they will be removed.
        apostrophes_or_quotes = "['`´ʻ‘’]"

        # Just in case Nemo missed it...
        sentence_level_mapping = {
            "&": " and ",
            "+": " plus ",
            "ﬂ": "fl",
            "%": " percent ",
            "=": " equal ",
            "@": " at ",
            "#": " hash ",
            "$": " dollar ",
            "}": "",
            "{": "",
            "\\": "",
            "|": "",
            "[": "",
            "]": "",
            "~": "",
            "^": "",
            "*": "",
            "•": "",
        }

        # Remove sentences that contain too many symbols.
        symbol_list = list(sentence_level_mapping.keys())
        if self.count_symbols_in_str(sentence, symbol_list) >= symbols_limit:
            return None

        final_characters = set(" ABCDEFGHIJKLMNOPQRSTUVWXYZ'")

        sentence_mapped = sentence
        if any((source in sentence) for source in sentence_level_mapping):
            for source, target in sentence_level_mapping.items():
                sentence_mapped = sentence_mapped.replace(source, target)

        # Some punctuation that indicates a word boundary.
        words_split = re.split(split_character_regex, sentence_mapped)
        words_quotes = [
            # Use ' as apostrophe.
            # Remove apostrophes at the start and end of words (probably quotes).
            # Word-internal apostrophes, even where rotated, are retained.
            re.sub(apostrophes_or_quotes, "'", word).strip("'")
            for word in words_split
        ]

        # Processing that does not change the length.
        if upper_case:
            words_upper = [word.upper() for word in words_quotes]
        else:
            words_upper = [word.lower() for word in words_quotes]

        words_mapped = [
            # word.translate(character_mapping)
            word
            for word in words_upper
            # Previous processing may have reduced words to nothing.
            # Remove them.
            if word != ""
        ]

        result = " ".join(words_mapped)
        character_set = set(result)

        if not character_set <= final_characters:
            logger.warning(
                "Sentence not properly normalised and removed: " + result
            )
            return None
        else:
            return result

    def clean_text(self, text):
        """Some sentences are poorly decoded from people's speech or yodas. This
        removes these char in the text.

        """
        unwanted_char = "\u0159\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x7f\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f\x90\x91\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f\xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf\xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf\xc0\xc1\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xcb\xcc\xcd\xce\xcf\xd0\xd1\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xdb\xdc\xdd\xde\xdf\xe0\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xee\xef\xf0\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xfb\xfc\xfd\xfe\xff"
        text = "".join(
            [
                (" " if n in unwanted_char else n)
                for n in text
                if n not in unwanted_char
            ]
        )
        return text

    def count_symbols_in_str(self, sentence, symbols):
        """Count the total number of symbols occurring in a string from a list of
        symbols

        Parameters
        ----------
        sentence : str
            The string to check.
        symbols : list
            List of symbols to count.

        Returns
        -------
        int
            The total count


        """
        cpt = 0

        for symbol in symbols:
            cpt += sentence.count(symbol)

        return cpt
