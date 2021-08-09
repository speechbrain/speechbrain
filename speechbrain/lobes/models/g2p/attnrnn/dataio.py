"""
Data pipeline elements for the G2P pipeline

Authors
 * Loren Lugosch 2020
 * Mirco Ravanelli 2020
 * Artem Ploujnikov 2021 (minor refactoring only)
"""
from functools import reduce
import speechbrain as sb
import torch


def clean_pipeline(graphemes, takes="txt", provides="txt_cleaned"):
    """
    Creates a pipeline element that cleans incoming text, removing
    any characters not on the accepted list of graphemes and converting
    to uppercase

    Arguments
    ---------
    graphemes: list
        a list of graphemes
    takes: str
        the source pipeline element
    provides: str
        the pipeline element to output

    Returns
    -------
    item: DynamicItem
        A wrapped transformation function
    """
    grapheme_set = set(graphemes)

    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)
    def f(txt):
        txt_upper = txt.upper()
        return "".join(char for char in txt_upper if char in grapheme_set)

    return f


def grapheme_pipeline(
    graphemes,
    grapheme_encoder=None,
    space_separated=False,
    uppercase=True,
    takes="char",
):
    """Creates a pipeline element for grapheme encoding

    Arguments
    ---------
    graphemes: list
        a list of available graphemes
    takes: str
        the name of the input
    space_separated: bool
        wether inputs are space-separated

    Returns
    -------
    result: DymamicItem
        a pipeline element
    """
    if grapheme_encoder is None:
        grapheme_encoder = sb.dataio.encoder.TextEncoder()
    grapheme_encoder.update_from_iterable(graphemes)
    grapheme_set = set(graphemes)

    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(
        "grapheme_list", "grapheme_encoded_list", "grapheme_encoded"
    )
    def f(char):
        if uppercase:
            char = char.upper()
        grapheme_list = char.strip().split(" ") if space_separated else char
        grapheme_list = [
            grapheme for grapheme in grapheme_list if grapheme in grapheme_set
        ]
        yield grapheme_list
        grapheme_encoded_list = grapheme_encoder.encode_sequence(grapheme_list)
        yield grapheme_encoded_list
        grapheme_encoded = torch.LongTensor(grapheme_encoded_list)
        yield grapheme_encoded

    return f


def tokenizer_encode_pipeline(
    tokenizer,
    tokens,
    takes="char",
    provides_prefix="grapheme",
    wordwise=True,
    word_separator=" ",
    token_space_index=512,
    space_separated=True,
    char_map=None,
):
    """A pipeline element that uses a pretrained tokenizer

    Arguments
    ---------
    tokenizer: speechbrain.tokenizer.SentencePiece
        a tokenizer instance
    """
    token_set = set(tokens)

    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(
        f"{provides_prefix}_list",
        f"{provides_prefix}_encoded_list",
        f"{provides_prefix}_encoded",
    )
    def f(seq):
        token_list = seq.strip().split(" ") if space_separated else seq
        token_list = [token for token in token_list if token in token_set]
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

    return f


def _add_bos(encoder, seq):
    return torch.LongTensor(encoder.prepend_bos_index(seq))


def _add_eos(encoder, seq):
    return torch.LongTensor(encoder.append_eos_index(seq))


def _wordwise_tokenize(tokenizer, sequence, input_separator, token_separator):
    if input_separator not in sequence:
        return tokenizer.sp.encode_as_ids(sequence)
    words = list(_split_list(sequence, input_separator))
    encoded_words = [
        tokenizer.sp.encode_as_ids(word_tokens) for word_tokens in words
    ]
    sep_list = [token_separator]
    return reduce((lambda left, right: left + sep_list + right), encoded_words)


def _wordwise_detokenize(tokenizer, sequence, output_separtor, token_separator):
    if token_separator not in sequence:
        return tokenizer.sp.decode_ids(sequence)
    words = list(_split_list(sequence, token_separator))
    encoded_words = [
        tokenizer.sp.decode_ids(word_tokens) for word_tokens in words
    ]
    return output_separtor.join(encoded_words)


def _split_list(items, separator):
    if items is not None:
        last_idx = -1
        for idx, item in enumerate(items):
            if item == separator:
                yield items[last_idx + 1 : idx]
                last_idx = idx
        if last_idx < idx - 1:
            yield items[last_idx + 1 :]


def _enable_eos_bos(tokens, encoder, bos_index, eos_index):
    """
    Initializs the phoneme encoder
    """
    if encoder is None:
        encoder = sb.dataio.encoder.TextEncoder()
    if bos_index == eos_index:
        encoder.insert_bos_eos(
            bos_label="<eos-bos>", eos_label="<eos-bos>", bos_index=bos_index,
        )
    else:
        encoder.insert_bos_eos(
            bos_label="<bos>",
            eos_label="<eos>",
            bos_index=bos_index,
            eos_index=eos_index,
        )
    encoder.add_unk()
    encoder.update_from_iterable(tokens, sequence_input=False)
    return encoder


def phoneme_pipeline(phoneme_encoder=None, space_separated=True):
    """Creates a pipeline element for phoneme encoding

    Arguments
    ---------
    phoneme_encoder: speechbrain.datio.encoder.TextEncoder
        a text encoder instance (optional, if not provided, a new one
        will be created)

    Returns
    -------
    result: DymamicItem
        a pipeline element
    """

    @sb.utils.data_pipeline.takes("phn")
    @sb.utils.data_pipeline.provides(
        "phn_list", "phn_encoded_list", "phn_encoded"
    )
    def f(phn):
        phn_list = phn.strip().split(" ") if space_separated else phn
        yield phn_list
        phn_encoded_list = phoneme_encoder.encode_sequence(phn_list)
        yield phn_encoded_list
        phn_encoded = torch.LongTensor(phn_encoded_list)
        yield phn_encoded

    return f


def add_bos_eos(tokens, encoder, bos_index=0, eos_index=0, prefix="phn"):
    """Creates a pipeline that takes {prefix}_list (e.g. "phn_list")
    and yields {prefix}_encoded_eos, {prefix}_encoded_eos with
    an EOS token appended and a BOS token prepended, respectively

    Arguments
    ---------
    phonemes: list
        a list of tokens to be used
    bos_index: int
        the index of the BOS token
    eos_index: int
        the index of the EOS token
    prefix: str
        the prefix to the pipeline items


    Returns
    -------
    encoder: speechbrain.dataio.encoder.TextEncoder
        an encoder for tokens
    result: DymamicItem
        a pipeline element
    """
    phoneme_encoder = _enable_eos_bos(tokens, encoder, bos_index, eos_index)

    @sb.utils.data_pipeline.takes(f"{prefix}_encoded_list")
    @sb.utils.data_pipeline.provides(
        f"{prefix}_encoded_eos", f"{prefix}_encoded_bos",
    )
    def f(seq):
        yield _add_eos(encoder, seq)
        yield _add_bos(encoder, seq)

    return f


def beam_search_pipeline(beam_searcher):
    """
    Performs a Beam Search on the phonemes
    """

    @sb.utils.data_pipeline.takes("char_lens", "encoder_out")
    @sb.utils.data_pipeline.provides("hyps", "scores")
    def f(char_lens, encoder_out):
        hyps, scores = beam_searcher(encoder_out, char_lens)
        return hyps, scores

    return f


def phoneme_decoder_pipeline(
    phonemes, phoneme_encoder=None, bos_index=0, eos_index=0
):
    """Creates a pipeline element for grapheme encoding

    Arguments
    ---------
    phonemes: list
        a list of available phonemes
    phoneme_encoder: speechbrain.datio.encoder.TextEncoder
        a text encoder instance (optional, if not provided, a new one
        will be created)
    bos_index: int
        the index of the BOS token
    eos_index: int

    Returns
    -------
    result: DymamicItem
        a pipeline element
    """
    phoneme_encoder = _enable_eos_bos(
        phonemes, phoneme_encoder, bos_index, eos_index
    )

    @sb.utils.data_pipeline.takes("hyps")
    @sb.utils.data_pipeline.provides("phonemes")
    def f(hyps):
        return phoneme_encoder.decode_ndim(hyps)

    return f


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
    tokenizer, char_map, token_space_index=None, wordwise=True
):
    """Returns a function that recovers the original sequence from one that has been
    tokenized using a character map

    Arguments
    ---------
    phn: torch.Tensor
        token indexes
    tokenizer: speechbrain.tokenizers.SentencePiece.SentencePiece
        a tokenizer instance
    char_map: dict
        a character-to-output-token-map
    token_space_index: int
        the index of the "space" token

    Returns
    -------
    f: callable
        the tokenizer function

    """
    if wordwise:
        detokenize = lambda item: _wordwise_detokenize(
            tokenizer(), item, " ", token_space_index
        )
    else:
        detokenize = lambda item: tokenizer().sp.decode_ids(item)

    def f(tokens):
        decoded_tokens = [detokenize(item) for item in tokens]
        mapped_tokens = _map_tokens_batch(decoded_tokens, char_map)
        return mapped_tokens

    return f


def _map_tokens_batch(tokens, char_map):
    return [[char_map[char] for char in item] for item in tokens]


def _map_tokens_item(tokens, char_map):
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
        nonlocal instance
        if instance is None:
            instance = init()
        return instance

    return f
