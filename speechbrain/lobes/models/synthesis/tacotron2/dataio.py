"""
 Data preprocessing for Tacotron2

 Authors
 * Georges Abous-Rjeili 2020

"""

import speechbrain as sb
import torch
from torchaudio import transforms
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.lobes.models.synthesis.tacotron2.text_to_sequence import (
    text_to_sequence,
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

    data_folder = hparams["data_folder"]

    train_data = DynamicItemDataset.from_json(
        json_path=hparams["json_train"],
        replacements={"data_root": data_folder},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["json_valid"],
        replacements={"data_root": data_folder},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["json_test"], replacements={"data_root": data_folder},
    )

    datasets = [train_data, valid_data, test_data]

    audio_toMel = transforms.MelSpectrogram(
        sample_rate=hparams["sampling_rate"],
        hop_length=hparams["hop_length"],
        win_length=hparams["win_length"],
        n_fft=hparams["n_fft"],
        n_mels=hparams["n_mel_channels"],
        f_min=hparams["mel_fmin"],
        f_max=hparams["mel_fmax"],
        normalized=hparams["mel_normalized"],
    )

    #  Define audio and text pipeline:
    @sb.utils.data_pipeline.takes("file_path", "words")
    @sb.utils.data_pipeline.provides("mel_text_pair")
    def audio_pipeline(file_path, words):
        text_seq = torch.IntTensor(
            text_to_sequence(words, hparams["text_cleaners"])
        )
        audio = sb.dataio.dataio.read_audio(file_path)
        mel = audio_toMel(audio)
        len_text = len(text_seq)
        yield text_seq, mel, len_text

        # set outputs

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    sb.dataio.dataset.set_output_keys(
        datasets, ["mel_text_pair"],
    )
    # create dataloaders that are passed to the model.
    train_data_loader = SaveableDataLoader(
        train_data,
        batch_size=hparams["batch_size"],
        collate_fn=TextMelCollate(),
        drop_last=True,
    )
    valid_data_loader = SaveableDataLoader(
        valid_data,
        batch_size=hparams["batch_size"],
        collate_fn=TextMelCollate(),
        drop_last=True,
    )
    test_data_loader = SaveableDataLoader(
        test_data,
        batch_size=hparams["batch_size"],
        collate_fn=TextMelCollate(),
        drop_last=True,
    )

    return train_data_loader, valid_data_loader, test_data_loader


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

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        Arguments
        ---------
        batch: list
            [text_normalized, mel_normalized]
        """
        for i in range(
            len(batch)
        ):  # the pipline return a dictionary wiht one elemnent
            batch[i] = list(batch[i].values())[0]

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
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, : mel.size(1)] = mel
            gate_padded[i, mel.size(1) - 1 :] = 1
            output_lengths[i] = mel.size(1)

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
        )
