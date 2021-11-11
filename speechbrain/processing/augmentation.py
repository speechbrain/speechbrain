"""Classes for implemeting data augmentation pipelines.

Authors
 * Mirco Ravanelli 2020
"""

import torch
from speechbrain.utils.callchains import lengths_arg_exists


class Augmenter(torch.nn.Module):
    """Applies pipelines of data augmentation.

    Arguments
    ---------
    pipeline : mapping
        Mapping from pipeline key to object. This connects the keys to
        the actual augmentation object instances.
    parallel_augment: If False, the augmentations are applied sequentially with
        the order specified in the pipeline argument (one orignal  input, one
        augmented output).
        When True, all the N augmentations are concatenated in the output
        on the batch axis (one orignal  input, N augmented output)
    concat_original: if True, the original input is concatenated with the
        augmented outputs (on the batch axis).

    Example
    -------
    >>> from speechbrain.processing.speech_augmentation import DropFreq, DropChunk
    >>> freq_dropper = DropFreq()
    >>> chunk_dropper = DropChunk(drop_start=100, drop_end=16000)
    >>> pipeline={'freq_drop': freq_dropper, 'chunk_dropper': chunk_dropper}
    >>> augment = Augmenter(pipeline=pipeline, parallel_augment=False, concat_original=False)
    >>> signal = torch.rand([4, 16000])
    >>> output_signal, lenghts = augment(signal, lengths=torch.tensor([0.2,0.5,0.7,1.0]))
    """

    def __init__(
        self, pipeline=None, parallel_augment=False, concat_original=False
    ):
        super().__init__()

        self.parallel_augment = parallel_augment
        self.concat_original = concat_original
        self.pipeline = {}

        if pipeline is not None:
            self.pipeline.update(pipeline)

        # Check if augmentation modules need the length argument
        self.require_lengths = {}
        for aug_key, aug_fun in self.pipeline.items():
            self.require_lengths[aug_key] = lengths_arg_exists(aug_fun.forward)

    def forward(self, x, lengths):
        """Applies data augmentation. Each function takes in input a tensor and
        a length (optional). The output is an augmented tensor (with length
        optionally returned).
        """
        next_input = x
        next_lengths = lengths
        output = []
        output_lengths = []
        out_lengths = lengths

        for augment_name, augment_fun in self.pipeline.items():

            # Check input arguments
            if self.require_lengths[augment_name]:
                out = augment_fun(next_input, lengths=next_lengths)
            else:
                out = augment_fun(next_input)

            # Check output arguments
            if isinstance(out, tuple):
                if len(out) == 2:
                    out, out_lengths = out
                else:
                    raise ValueError(
                        "The function must return max two arguments (Tensor, Length[optional])"
                    )

            # Manage sequential or parallel augmentation
            if not self.parallel_augment:
                next_input = out
                next_lengths = out_lengths
            else:
                output.append(out)
                output_lengths.append(out_lengths)

        if self.parallel_augment:
            # Concatenate all the augmented data
            output = torch.cat(output, dim=0)
            output_lengths = torch.cat(output_lengths, dim=0)
        else:
            # Take the last agumented signal of the pipeline
            output = out
            output_lengths = out_lengths

        # Concatenate the original signal if required
        if self.concat_original:
            output = torch.cat([x, output], dim=0)
            output_lengths = torch.cat([lengths, output_lengths], dim=0)
        return output, output_lengths
