"""
Data pipeline elements for the G2P pipeline

Authors
 * Loren Lugosch 2020
 * Mirco Ravanelli 2020
 * Artem Ploujnikov 2021 (minor refactoring only)
"""

from functools import reduce
from speechbrain.wordemb.util import expand_to_chars
import speechbrain as sb
import torch
import re

RE_MULTI_SPACE = re.compile(r"\s{2,}")


def clean_pipeline(txt, graphemes):
    """
    Cleans incoming text, removing any characters not on the
    accepted list of graphemes and converting to uppercase

    Arguments
    ---------
    txt: str
        the text to clean up
    graphemes: list
        a list of graphemes

    Returns
    -------
    item: DynamicItem
        A wrapped transformation function
    """
    result = txt.upper()
    result = "".join(char for char in result if char in graphemes)
    result = RE_MULTI_SPACE.sub(" ", result)
    return result


def grapheme_pipeline(char, grapheme_encoder=None, uppercase=True):
    """Encodes a grapheme sequence

    Arguments
    ---------
    graphemes: list
        a list of available graphemes
    grapheme_encoder: speechbrain.dataio.encoder.TextEncoder
        a text encoder for graphemes. If not provided,
    takes: str
        the name of the input
    uppercase: bool
        whether or not to convert items to uppercase

    Returns
    -------
    grapheme_list: list
        a raw list of graphemes, excluding any non-matching
        labels
    grapheme_encoded_list: list
        a list of graphemes encoded as integers
    grapheme_encoded: torch.Tensor
    """
    if uppercase:
        char = char.upper()
    grapheme_list = [
        grapheme for grapheme in char if grapheme in grapheme_encoder.lab2ind
    ]
    yield grapheme_list
    grapheme_encoded_list = grapheme_encoder.encode_sequence(grapheme_list)
    yield grapheme_encoded_list
    grapheme_encoded = torch.LongTensor(grapheme_encoded_list)
    yield grapheme_encoded


def tokenizer_encode_pipeline(
    seq,
    tokenizer,
    tokens,
    wordwise=True,
    word_separator=" ",
    token_space_index=512,
    char_map=None,
):
    """A pipeline element that uses a pretrained tokenizer

    Arguments
    ---------
    tokenizer: speechbrain.tokenizer.SentencePiece
        a tokenizer instance
    tokens: str
        available tokens
    takes: str
        the name of the pipeline input providing raw text
    provides_prefix: str
        the prefix used for outputs
    wordwise: str
        whether tokenization is peformed on the whole sequence
        or one word at a time. Tokenization can produce token
        sequences in which a token may span multiple words
    token_space_index: int
        the index of the space token
    char_map: dict
        a mapping from characters to tokens. This is used when
        tokenizing sequences of phonemes rather than sequences
        of characters. A sequence of phonemes is typically a list
        of one or two-character tokens (e.g. ["DH", "UH", " ", "S", "AW",
        "N", "D"]). The character map makes it possible to map these
        to arbitrarily selected characters

    Returns
    -------
    token_list: list
        a list of raw tokens
    encoded_list: list
        a list of tokens, encoded as a list of integers
    encoded: torch.Tensor
        a list of tokens, encoded as a tensor
    """
    token_list = [token for token in seq if token in tokens]
    yield token_list
    tokenizer_input = "".join(
        _map_tokens_item(token_list, char_map)
        if char_map is not None
        else token_list
    )

    if wordwise:
        encoded_list = _wordwise_tokenize(
            tokenizer(), tokenizer_input, word_separator, token_space_index
        )
    else:
        encoded_list = tokenizer().sp.encode_as_ids(tokenizer_input)
    yield encoded_list
    encoded = torch.LongTensor(encoded_list)
    yield encoded


def _wordwise_tokenize(tokenizer, sequence, input_separator, token_separator):
    """Tokenizes a sequence wordwise

    Arguments
    ---------
    tokenizer: speechbrain.tokenizers.SentencePiece.SentencePiece
        a tokenizer instance
    sequence: iterable
        the original sequence
    input_separator: str
        the separator used in the input seauence
    token_separator: str
        the token separator used in the output sequence

    Returns
    -------
    result: str
        the resulting tensor
    """

    if input_separator not in sequence:
        return tokenizer.sp.encode_as_ids(sequence)
    words = list(_split_list(sequence, input_separator))
    encoded_words = [
        tokenizer.sp.encode_as_ids(word_tokens) for word_tokens in words
    ]
    sep_list = [token_separator]
    return reduce((lambda left, right: left + sep_list + right), encoded_words)


def _wordwise_detokenize(tokenizer, sequence, output_separtor, token_separator):
    """Detokenizes a sequence wordwise

    Arguments
    ---------
    tokenizer: speechbrain.tokenizers.SentencePiece.SentencePiece
        a tokenizer instance
    sequence: iterable
        the original sequence
    output_separator: str
        the separator used in the output seauence
    token_separator: str
        the token separator used in the output sequence

    Returns
    -------
    result: torch.Tensor
        the result

    """
    if isinstance(sequence, str) and sequence == "":
        return ""
    if token_separator not in sequence:
        sequence_list = (
            sequence if isinstance(sequence, list) else sequence.tolist()
        )
        return tokenizer.sp.decode_ids(sequence_list)
    words = list(_split_list(sequence, token_separator))
    encoded_words = [
        tokenizer.sp.decode_ids(word_tokens) for word_tokens in words
    ]
    return output_separtor.join(encoded_words)


def _split_list(items, separator):
    """
    Splits a sequence (such as a tensor) by the specified separator

    Arguments
    ---------
    items: sequence
        any sequence that supports indexing

    Results
    -------
    separator: str
        the separator token
    """
    if items is not None:
        last_idx = -1
        for idx, item in enumerate(items):
            if item == separator:
                yield items[last_idx + 1 : idx]
                last_idx = idx
        if last_idx < idx - 1:
            yield items[last_idx + 1 :]


def enable_eos_bos(tokens, encoder, bos_index, eos_index):
    """
    Initializs the phoneme encoder with EOS/BOS sequences

    Arguments
    ---------
    tokens: list
        a list of tokens
    encoder: speechbrain.dataio.encoder.TextEncoder.
        a text encoder instance. If none is provided, a new one
        will be instantiated
    bos_index: int
        the position corresponding to the Beginning-of-Sentence
        token
    eos_index: int
        the position corresponding to the End-of-Sentence

    Returns
    -------
    encoder: speechbrain.dataio.encoder.TextEncoder
        an encoder

    """
    if encoder is None:
        encoder = sb.dataio.encoder.TextEncoder()
    if bos_index == eos_index:
        if "<eos-bos>" not in encoder.lab2ind:
            encoder.insert_bos_eos(
                bos_label="<eos-bos>",
                eos_label="<eos-bos>",
                bos_index=bos_index,
            )
    else:
        if "<bos>" not in encoder.lab2ind:
            encoder.insert_bos_eos(
                bos_label="<bos>",
                eos_label="<eos>",
                bos_index=bos_index,
                eos_index=eos_index,
            )
    if "<unk>" not in encoder.lab2ind:
        encoder.add_unk()
    encoder.update_from_iterable(tokens, sequence_input=False)
    return encoder


def phoneme_pipeline(phn, phoneme_encoder=None):
    """Encodes a sequence of phonemes using the encoder
    provided

    Arguments
    ---------
    phoneme_encoder: speechbrain.datio.encoder.TextEncoder
        a text encoder instance (optional, if not provided, a new one
        will be created)

    Returns
    -------
    phn: list
        the original list of phonemes
    phn_encoded_list: list
        encoded phonemes, as a list
    phn_encoded: torch.Tensor
        encoded phonemes, as a tensor
    """

    yield phn
    phn_encoded_list = phoneme_encoder.encode_sequence(phn)
    yield phn_encoded_list
    phn_encoded = torch.LongTensor(phn_encoded_list)
    yield phn_encoded


def add_bos_eos(seq=None, encoder=None):
    """Adds BOS and EOS tokens to the sequence provided

    Arguments
    ---------
    seq: torch.Tensor
        the source sequence
    encoder: speechbrain.dataio.encoder.TextEncoder
        an encoder instance


    Returns
    -------
    seq_eos: torch.Tensor
        the sequence, with the EOS token added
    seq_bos: torch.Tensor
        the sequence, with the BOS token added
    """
    seq_bos = encoder.prepend_bos_index(seq)
    if not torch.is_tensor(seq_bos):
        seq_bos = torch.tensor(seq_bos)
    yield seq_bos.long()
    yield torch.tensor(len(seq_bos))
    seq_eos = encoder.append_eos_index(seq)
    if not torch.is_tensor(seq_eos):
        seq_eos = torch.tensor(seq_eos)
    yield seq_eos.long()
    yield torch.tensor(len(seq_eos))


def beam_search_pipeline(char_lens, encoder_out, beam_searcher):
    """Performs a Beam Search on the phonemes. This function is
    meant to be used as a component in a decoding pipeline

    Arguments
    ---------
    char_lens: torch.Tensor
        the length of character inputs
    encoder_out: torch.Tensor
        Raw encoder outputs
    beam_searcher: speechbrain.decoders.seq2seq.S2SBeamSearcher
        a SpeechBrain beam searcher instance

    Returns
    -------
    hyps: list
        hypotheses
    scores: list
        confidence scores associated with each hypotheses
    """
    return beam_searcher(encoder_out, char_lens)


def phoneme_decoder_pipeline(hyps, phoneme_encoder):
    """Decodes a sequence of phonemes

    Arguments
    ---------
    hyps: list
        hypotheses, the output of a beam search
    phoneme_encoder: speechbrain.datio.encoder.TextEncoder
        a text encoder instance

    Returns
    -------
    phonemes: list
        the phoneme sequence
    """
    return phoneme_encoder.decode_ndim(hyps)


def char_range(start_char, end_char):
    """Produces a list of consequtive characters

    Arguments
    ---------
    start_char: str
        the starting character
    end_char: str
        the ending characters

    Returns
    -------
    char_range: str
        the character range
    """
    return [chr(idx) for idx in range(ord(start_char), ord(end_char) + 1)]


def build_token_char_map(tokens):
    """Builds a map that maps arbitrary tokens to arbitrarily chosen characters.
    This is required to overcome the limitations of SentencePiece.

    Arguments
    ---------
    tokens: list
        a list of tokens for which to produce the map

    Returns
    -------
    token_map: dict
        a dictionary with original tokens as keys and
        new mappings as values
    """
    chars = char_range("A", "Z") + char_range("a", "z")
    values = list(filter(lambda chr: chr != " ", tokens))
    token_map = dict(zip(values, chars[: len(values)]))
    token_map[" "] = " "
    return token_map


def flip_map(map_dict):
    """Exchanges keys and values in a dictionary

    Arguments
    ---------
    map_dict: dict
        a dictionary

    Returns
    -------
    reverse_map_dict: dict
        a dictioanry with keys and values flipped
    """
    return {value: key for key, value in map_dict.items()}


def text_decode(seq, encoder):
    """Decodes a sequence using a tokenizer.
    This function is meant to be used in hparam files

    Arguments
    ---------
    seq: torch.Tensor
        token indexes
    encoder: sb.dataio.encoder.TextEncoder
        a text encoder instance

    Returns
    -------
    output_seq: list
        a list of lists of tokens
    """
    return encoder.decode_ndim(seq)


def char_map_detokenize(
    char_map, tokenizer, token_space_index=None, wordwise=True
):
    """Returns a function that recovers the original sequence from one that has been
    tokenized using a character map

    Arguments
    ---------
    char_map: dict
        a character-to-output-token-map
    tokenizer: speechbrain.tokenizers.SentencePiece.SentencePiece
        a tokenizer instance
    token_space_index: int
        the index of the "space" token

    Returns
    -------
    f: callable
        the tokenizer function

    """

    def detokenize_wordwise(item):
        """Detokenizes the sequence one word at a time"""
        return _wordwise_detokenize(tokenizer(), item, " ", token_space_index)

    def detokenize_regular(item):
        """Detokenizes the entire sequence"""
        return tokenizer().sp.decode_ids(item)

    detokenize = detokenize_wordwise if wordwise else detokenize_regular

    def f(tokens):
        """The tokenizer function"""
        decoded_tokens = [detokenize(item) for item in tokens]
        mapped_tokens = _map_tokens_batch(decoded_tokens, char_map)
        return mapped_tokens

    return f


def _map_tokens_batch(tokens, char_map):
    """Performs token mapping, in batch mode

    Arguments
    ---------
    tokens: iterable
        a list of token sequences
    char_map: dict
        a token-to-character mapping

    Returns
    -------
    result: list
        a list of lists of characters
    """
    return [[char_map[char] for char in item] for item in tokens]


def _map_tokens_item(tokens, char_map):
    """Maps tokens to characters, for a single item

    Arguments
    ---------
    tokens: iterable
        a single token sequence
    char_map: dict
        a token-to-character mapping

    Returns
    -------
    result: list
        a list of tokens

    """
    return [char_map[char] for char in tokens]


def lazy_init(init):
    """A wrapper to ensure that the specified object is initialzied
    only once (used mainly for tokenizers that train when the
    constructor is called

    Arguments
    ---------
    init: callable
        a constructor or function that creates an object

    Returns
    -------
    instance: object
        the object instance
    """
    instance = None

    def f():
        """The initializer function"""
        nonlocal instance
        if instance is None:
            instance = init()
        return instance

    return f


def get_sequence_key(key, mode):
    """Determines the key to be used for sequences (e.g. graphemes/phonemes)
    based on the naming convention

    Arguments
    ---------
    key: str
        the key (e.g. "graphemes", "phonemes")
    mode:
        the mode/sufix (raw, eos/bos)
    """
    return key if mode == "raw" else f"{key}_{mode}"


def phonemes_to_label(phns, decoder):
    """Converts a batch of phoneme sequences (a single tensor)
    to a list of space-separated phoneme label strings,
    (e.g. ["T AY B L", "B UH K"]), removing any special tokens

    Arguments
    ---------
    phn: sequence
        a batch of phoneme sequences

    Returns
    -------
    result: list
        a list of strings corresponding to the phonemes provided"""

    phn_decoded = decoder(phns)
    return [" ".join(remove_special(item)) for item in phn_decoded]


def remove_special(phn):
    """Removes any special tokens from the sequence. Special tokens are delimited
    by angle brackets.

    Arguments
    ---------
    phn: list
        a list of phoneme labels

    Returns
    -------
    result: list
        the original list, without any special tokens
    """
    return [token for token in phn if "<" not in token]


def word_emb_pipeline(
    txt,
    grapheme_encoded,
    grapheme_encoded_len,
    grapheme_encoder=None,
    word_emb=None,
    use_word_emb=None,
):
    """Applies word embeddings, if applicable. This function is meant
    to be used as part of the encoding pipeline

    Arguments
    ---------
    txt: str
        the raw text
    grapheme_encoded: torch.tensor
        the encoded graphemes
    grapheme_encoded_len: torch.tensor
        encoded grapheme lengths
    grapheme_encoder: speechbrain.dataio.encoder.TextEncoder
        the text encoder used for graphemes
    word_emb: callable
        the model that produces word embeddings
    use_word_emb: bool
        a flag indicated if word embeddings are to be applied

    Returns
    -------
    char_word_emb: torch.tensor
        Word embeddings, expanded to the character dimension
    """
    char_word_emb = None

    if use_word_emb:
        raw_word_emb = word_emb().embeddings(txt)
        word_separator_idx = grapheme_encoder.lab2ind[" "]
        char_word_emb = expand_to_chars(
            emb=raw_word_emb.unsqueeze(0),
            seq=grapheme_encoded.unsqueeze(0),
            seq_len=grapheme_encoded_len.unsqueeze(0),
            word_separator=word_separator_idx,
        ).squeeze(0)

    return char_word_emb
