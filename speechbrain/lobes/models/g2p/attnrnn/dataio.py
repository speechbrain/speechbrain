"""
Data pipeline elements for the G2P pipeline

Authors
 * Loren Lugosch 2020
 * Mirco Ravanelli 2020
 * Artem Ploujnikov 2021 (minor refactoring only)
"""
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
    graphemes, grapheme_encoder=None, space_separated=False, takes="char"
):
    """
    Creates a pipeline element for grapheme encoding

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

    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(
        "grapheme_list", "grapheme_encoded_list", "grapheme_encoded"
    )
    def f(char):
        grapheme_list = char.strip().split(" ") if space_separated else char
        yield grapheme_list
        grapheme_encoded_list = grapheme_encoder.encode_sequence(grapheme_list)
        yield grapheme_encoded_list
        grapheme_encoded = torch.LongTensor(grapheme_encoded_list)
        yield grapheme_encoded

    return f


def _init_phoneme_encoder(phonemes, phoneme_encoder, bos_index, eos_index):
    """
    Initializs the phoneme encoder
    """
    if phoneme_encoder is None:
        phoneme_encoder = sb.dataio.encoder.TextEncoder()
    phoneme_encoder.update_from_iterable(phonemes, sequence_input=False)
    if bos_index == eos_index:
        phoneme_encoder.insert_bos_eos(
            bos_label="<eos-bos>", eos_label="<eos-bos>", bos_index=bos_index,
        )
    else:
        phoneme_encoder.insert_bos_eos(
            bos_label="<bos>",
            eos_label="<eos>",
            bos_index=bos_index,
            eos_index=eos_index,
        )
    return phoneme_encoder


def phoneme_pipeline(phonemes, phoneme_encoder=None, bos_index=0, eos_index=0,
                     space_separated=True):
    """
    Creates a pipeline element for phoneme encoding

    Arguments
    ---------
    phonemes: list
        a list fo phonemes to be used
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

    phoneme_encoder = _init_phoneme_encoder(
        phonemes, phoneme_encoder, bos_index, eos_index
    )

    @sb.utils.data_pipeline.takes("phn")
    @sb.utils.data_pipeline.provides(
        "phn_list",
        "phn_encoded_list",
        "phn_encoded",
        "phn_encoded_eos",
        "phn_encoded_bos",
    )
    def f(phn):
        phn_list = phn.strip().split(" ") if space_separated else phn
        yield phn_list
        phn_encoded_list = phoneme_encoder.encode_sequence(phn_list)
        yield phn_encoded_list
        phn_encoded = torch.LongTensor(phn_encoded_list)
        yield phn_encoded
        phn_encoded_eos = torch.LongTensor(
            phoneme_encoder.append_eos_index(phn_encoded_list)
        )
        yield phn_encoded_eos
        phn_encoded_bos = torch.LongTensor(
            phoneme_encoder.prepend_bos_index(phn_encoded_list)
        )
        yield phn_encoded_bos

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
    """
    Creates a pipeline element for grapheme encoding

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
    phoneme_encoder = _init_phoneme_encoder(
        phonemes, phoneme_encoder, bos_index, eos_index
    )

    @sb.utils.data_pipeline.takes("hyps")
    @sb.utils.data_pipeline.provides("phonemes")
    def f(hyps):
        return phoneme_encoder.decode_ndim(hyps)

    return f
