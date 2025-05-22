"""Dictionary utilities, e.g. synonym dictionaries.

Authors
 * Sylvain de Langen 2024"""

import json
from collections import defaultdict
from typing import Iterable


class SynonymDictionary:
    """Loads sets of synonym words and lets you look up if two words are
    synonyms.

    This could, for instance, be used to check for equality in the case of two
    spellings of the same word when normalization might be unsuitable.

    Synonyms are not considered to be transitive:
    If A is a synonym of B and B is a synonym of C, then A is NOT considered a
    synonym of C unless they are added in the same synonym set."""

    def __init__(self):
        self.word_map = defaultdict(set)

    @staticmethod
    def from_json_file(file) -> "SynonymDictionary":
        """Parses an opened file as JSON, where the top level structure is a
        list of sets of synonyms (i.e. words that are all synonyms with each
        other), e.g. `[ ["hello", "hi"], ["say", "speak", "talk"] ]`.

        Arguments
        ---------
        file : file object
            File object that supports reading (e.g. an `open`ed file)

        Returns
        -------
        SynonymDictionary
            Synonym dictionary frm the parsed JSON file with all synonym sets
            added.
        """
        d = json.load(file)

        synonym_dict = SynonymDictionary()

        for entry in d:
            if isinstance(entry, list):
                synonym_dict.add_synonym_set(entry)
            else:
                raise ValueError(
                    f"Unexpected entry type {type(entry)} in synonyms JSON (expected list)"
                )

        return synonym_dict

    @staticmethod
    def from_json_path(path) -> "SynonymDictionary":
        """Opens a file and parses it as JSON, with otherwise the same semantics
        as :meth:`~SynonymDictionary.from_json_file`, which uses an opened file.

        Arguments
        ---------
        path : str
            Path to the JSON file

        Returns
        -------
        SynonymDictionary
            Synonym dictionary frm the parsed JSON file with all synonym sets
            added.
        """
        with open(path, "r", encoding="utf8") as f:
            return SynonymDictionary.from_json_file(f)

    def add_synonym_set(self, words: Iterable[str]) -> None:
        """Add a set of words that are all synonyms with each other.

        Arguments
        ---------
        words : Iterable[str]
            List of words that should be defined as synonyms to each other"""

        word_set = set(words)

        for word in word_set:
            self.word_map[word].update(word_set - {word})

    def __call__(self, a: str, b: str) -> bool:
        """Check for the equality or synonym equality of two words.

        Arguments
        ---------
        a : str
            First word to compare. May be outside of the known dictionary.
        b : str
            Second word to compare. May be outside of the known dictionary.
            The order of arguments does not matter.

        Returns
        -------
        bool
            Whether `a` and `b` should be considered synonyms. Not transitive,
            see the main class documentation."""

        return (a == b) or (b in self.word_map[a])

    def get_synonyms_for(self, word: str) -> set:
        """Returns the set of synonyms for a given word.

        Arguments
        ---------
        word : str
            The word to look up the synonyms of. May be outside of the known
            dictionary.

        Returns
        -------
        set of str
            Set of known synonyms for this word. Do not mutate (or copy it
            prior). May be empty if the word has no known synonyms."""

        return self.word_map.get(word, set())
