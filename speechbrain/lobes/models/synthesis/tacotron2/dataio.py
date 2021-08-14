"""
 Data preprocessing for Tacotron2

 Authors
 * Georges Abous-Rjeili 2021
 * Artem Ploujnikov 2021

"""

import os
import speechbrain as sb
import torch
from speechbrain.lobes.models.synthesis.dataio import load_datasets
from torchaudio import transforms
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.lobes.models.synthesis.tacotron2.text_to_sequence import (
    text_to_sequence,
)

DEFAULT_TEXT_CLEANERS = ["english_cleaners"]


def encode_text(
    text_cleaners=None,
    takes="txt",
    provides=["text_sequences", "input_lengths"],
):
    """
    A pipeline function that encodes raw text into a tensor

    Arguments
    ---------
    text_cleaners: list
        an list of text cleaners to use
    takes: str
        the name of the pipeline input
    provides: str
        the name of the pipeline output

    Returns
    -------
    result: DymamicItem
        a pipeline element
    """
    text_cleaners = DEFAULT_TEXT_CLEANERS

    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(*provides)
    def f(txt):
        sequence = text_to_sequence(txt, text_cleaners)
        yield torch.tensor(sequence, dtype=torch.int32)
        yield torch.tensor(len(sequence), dtype=torch.int32)

    return f


def _dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    Arguments
    ---------
    C: int
        compression factor

    clip_val: float
        the minimum value below which x values wil be clipped

    return: torch.tensor
        compressed results
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def _dynamic_range_decompression(x, C=1):
    """
    Arguments
    ----------
    C: int
        compression factor used to compress
    """
    return torch.exp(x) / C


def dynamic_range_decompression(C=1, takes="mel", provides="mel_decompressed"):
    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(provides)
    def f(x):
        return _dynamic_range_decompression(x, C)

    return f


# TODO: Modularize and decouple this
def audio_pipeline(hparams):
    """
    A pipeline function that provides text sequences, a mel spectrogram
    and the text length

    Arguments
    ---------
    hparams: dict
        model hyperparameters

    Returns
    -------
    result: DymamicItem
        a pipeline element
    """
    audio_to_mel = transforms.MelSpectrogram(
        sample_rate=hparams["sample_rate"],
        hop_length=hparams["hop_length"],
        win_length=hparams["win_length"],
        n_fft=hparams["n_fft"],
        n_mels=hparams["n_mel_channels"],
        f_min=hparams["mel_fmin"],
        f_max=hparams["mel_fmax"],
        power=hparams["power"],
        normalized=hparams["mel_normalized"],
        norm=hparams["norm"],
        mel_scale=hparams["mel_scale"],
    )

    preprocessing_mode = hparams.get("preprocessing_mode")
    input_encoder = hparams.get("input_encoder")
    input_bos_eos = hparams.get("input_eos_bos", False)
    if input_encoder is not None:
        if not input_encoder.lab2ind:
            if input_bos_eos:
                input_encoder.insert_bos_eos(
                    bos_label="<bos>", eos_label="<eos>"
                )
            input_encoder.update_from_iterable(
                hparams["input_tokens"], sequence_input=False
            )
    elif preprocessing_mode != "nvidia":
        ValueError(
            "An input_encoder is required except when"
            "preprocessing_mode=nvidia"
        )

    wav_folder = hparams.get("wav_folder")

    @sb.utils.data_pipeline.takes("wav", "label")
    @sb.utils.data_pipeline.provides("mel_text_pair")
    def f(file_path, words):
        if preprocessing_mode == "nvidia":
            text_seq = torch.IntTensor(
                text_to_sequence(words, hparams["text_cleaners"])
            )
        else:
            text_seq = input_encoder.encode_sequence_torch(words[0])
            if input_bos_eos:
                text_seq = input_encoder.prepend_bos_index(text_seq)
                text_seq = input_encoder.append_eos_index(text_seq)
            text_seq = text_seq.int()

        if wav_folder:
            file_path = os.path.join(wav_folder, file_path)
        audio = sb.dataio.dataio.read_audio(file_path)

        mel = audio_to_mel(audio)
        if hparams["dynamic_range_compression"]:
            mel = _dynamic_range_compression(mel)
        len_text = len(text_seq)
        yield text_seq, mel, len_text

    return f


def dataset_prep(dataset, hparams):
    """
    Adds pipeline elements for Tacotron to a dataset and
    wraps it in a saveable data loader

    Arguments
    ---------
    dataset: DynamicItemDataSet
        a raw dataset

    Returns
    -------
    result: SaveableDataLoader
        a data loader
    """
    dataset.add_dynamic_item(audio_pipeline(hparams))
    dataset.set_output_keys(["mel_text_pair", "wav", "label"])
    dataset.data_ids = dataset.data_ids[:128]
    return SaveableDataLoader(
        dataset,
        batch_size=hparams["batch_size"],
        collate_fn=TextMelCollate(),
        drop_last=hparams.get("drop_last", False),
    )


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.

    Arguments
    ---------
    hparams: dict
        model hyperparameters

    Returns
    -------
    datasets: tuple
        a tuple of data loaders (train_data_loader, valid_data_loader, test_data_loader)
    """

    return load_datasets(hparams, dataset_prep)


class TextMelCollate:
    """ Zero-pads model inputs and targets based on number of frames per step

    Arguments
    ---------
    n_frames_per_step: int
        the number of frames per step

    Returns
    -------
    result: tuple
        a tuple of tensors to be used as inputs/targets
        (
            text_padded,
            input_lengths,
            mel_padded,
            gate_padded,
            output_lengths,
            len_x
        )
    """

    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step

    # TODO: Make this more intuitive, use the pipeline
    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        Arguments
        ---------
        batch: list
            [text_normalized, mel_normalized]
        """

        # TODO: Remove for loops and this dirty hack
        raw_batch = list(batch)
        for i in range(
            len(batch)
        ):  # the pipline return a dictionary wiht one elemnent
            batch[i] = batch[i]["mel_text_pair"]

        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True
        )
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, : text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += (
                self.n_frames_per_step - max_target_len % self.n_frames_per_step
            )
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        labels, wavs = [], []
        for i in range(len(ids_sorted_decreasing)):
            idx = ids_sorted_decreasing[i]
            mel = batch[idx][1]
            mel_padded[i, :, : mel.size(1)] = mel
            gate_padded[i, mel.size(1) - 1 :] = 1
            output_lengths[i] = mel.size(1)
            labels.append(raw_batch[idx]["label"])
            wavs.append(raw_batch[idx]["wav"])

        # count number of items - characters in text
        len_x = [x[2] for x in batch]
        len_x = torch.Tensor(len_x)
        return (
            text_padded,
            input_lengths,
            mel_padded,
            gate_padded,
            output_lengths,
            len_x,
            labels,
            wavs,
        )
