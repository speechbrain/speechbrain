"""
Data pipeline elements for the G2P pipeline

Authors
 * Loren Lugosch 2020
 * Mirco Ravanelli 2020
 * Artem Ploujnikov 2021
"""
import speechbrain as sb
import torch

def grapheme_pipeline():
    """
    Creates a pipeline element for grapheme encoding

    Returns
    -------
    result: DymamicItem
        a pipeline element
    """
    grapheme_encoder = sb.dataio.encoder.TextEncoder()

    @sb.utils.data_pipeline.takes("char")
    @sb.utils.data_pipeline.provides(
        "grapheme_list", "grapheme_encoded_list", "grapheme_encoded"
    )
    def f(char):
        grapheme_list = char.strip().split(" ")
        yield grapheme_list
        grapheme_encoded_list = grapheme_encoder.encode_sequence(grapheme_list)
        yield grapheme_encoded_list
        grapheme_encoded = torch.LongTensor(grapheme_encoded_list)
        yield grapheme_encoded

    return f


def phoneme_pipeline():
    """
    Creates a pipeline element for phoneme encoding

    Returns
    -------
    result: DymamicItem
        a pipeline element
    """
    phoneme_encoder = sb.dataio.encoder.TextEncoder()

    @sb.utils.data_pipeline.takes("phn")
    @sb.utils.data_pipeline.provides(
        "phn_list",
        "phn_encoded_list",
        "phn_encoded",
        "phn_encoded_eos",
        "phn_encoded_bos",
    )
    def f(phn):
        phn_list = phn.strip().split(" ")
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

def phoneme_decoder_pipeline(phonemes):
    """
    Creates a pipeline element for grapheme encoding

    Arguments
    ---------
    phonemes: list
        a list of available phonemes

    Returns
    -------
    result: DymamicItem
        a pipeline element
    """
    def f(phoneme_indices, char_lens):
        return [
            [phonemes[phoneme_index] for phoneme_index in phoneme_indices[:length]]
            for phoneme_indices, length in zip(phonemes, char_lens)
        ]

    return f