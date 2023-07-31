"""Classes for implementing data augmentation pipelines.

Authors
 * Mirco Ravanelli 2020
"""

import torch
import random
from speechbrain.utils.callchains import lengths_arg_exists


class Augmenter(torch.nn.Module):
    """Applies pipelines of data augmentation.

    Arguments
    ---------
    **augmentations: dict
        The inputs are treated as a dictionary containing the name assigned to
        the augmentation and the corresponding objects.
        The augmentations are applied in sequence (or parallel).
    parallel_augment: bool
        If False, the augmentations are applied sequentially with
        the order specified in the pipeline argument (one orignal  input, one
        augmented output).
        When True, all the N augmentations are concatenated in the output
        on the batch axis (one orignal  input, N augmented output)
    parallel_augment_fixed_bs: bool
        If False, each augmenter (performed in parallel) generates a number of
        augmented examples equal to the batch size. Thus, overall, with this option N*batch size artificial data are
        generated, where N is the number of augmenters.
        When True, the number of total augmented examples is kept fixed at
        the batch size, thus, for each augmenter, fixed at batch size // N examples.
        This option is useful to keep controlled the number of synthetic examples
        with respect to the original data distribution, as it keep always
        50% of original data, and 50% of augmented data.
    concat_original: bool
        if True, the original input is concatenated with the
        augmented outputs (on the batch axis).
    min_augmentations: int
        The number of augmentations applied to the input signal is randomly
        sampled between min_augmentations and max_augmentations. For instance,
        if the augmentation dict contains N=6 augmentations and we set
        select min_augmentations=1 and max_augmentations=4 we apply up to
        M=4 augmentations. The selected augmentations are applied in the order
        specified in the augmentations dict. If shuffle_augmentations = True,
        a random set of M augmentations is selected.
    max_augmentations: int
        Maximum number of augmentations to apply. See min_augmentations for
        more details.
    shuffle_augmentations:  bool
        If True, it shuffles the entries of the augmentations dictionary.
        The effect is to randomply select the order of the augmentations
        to apply.
    repeat_augment: int
        Applies the augmentation algorithm N times. This can be used to
        perform more data augmentation.


    Example
    -------
    >>> from speechbrain.processing.speech_augmentation import DropFreq, DropChunk
    >>> freq_dropper = DropFreq()
    >>> chunk_dropper = DropChunk(drop_start=100, drop_end=16000)
    >>> augment = Augmenter(parallel_augment=False, concat_original=False, freq_dropper=freq_dropper,  chunk_dropper= chunk_dropper)
    >>> signal = torch.rand([4, 16000])
    >>> output_signal, lenghts = augment(signal, lengths=torch.tensor([0.2,0.5,0.7,1.0]))
    """

    def __init__(
        self,
        parallel_augment=False,
        parallel_augment_fixed_bs=False,
        concat_original=False,
        min_augmentations=None,
        max_augmentations=None,
        shuffle_augmentations=False,
        repeat_augment=1,
        **augmentations,
    ):
        super().__init__()
        self.parallel_augment = parallel_augment
        self.parallel_augment_fixed_bs = parallel_augment_fixed_bs
        self.concat_original = concat_original
        self.augmentations = augmentations
        self.min_augmentations = min_augmentations
        self.max_augmentations = max_augmentations
        self.shuffle_augmentations = shuffle_augmentations
        self.repeat_augment = repeat_augment
        # Check min and max augmentations
        self.check_min_max_augmentations()

        # Check repeat augment arguments
        if not isinstance(self.repeat_augment, int):
            raise ValueError("repeat_augment must be an integer.")

        if self.repeat_augment < 0:
            raise ValueError("repeat_augment must be greater than 0.")

        # Check if augmentation modules need the length argument
        self.require_lengths = {}
        for aug_key, aug_fun in self.augmentations.items():
            self.require_lengths[aug_key] = lengths_arg_exists(aug_fun.forward)

    def augment(self, x, lengths, selected_augmentations):
        """Applies data augmentation on the seleted augmentations.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to augment.
        lengths : torch.Tensor
            The length of each sequence in the batch.
        selected_augmentations: dict
            Dictionary containg the selected augmentation to apply.
        """
        next_input = x
        next_lengths = lengths
        output = []
        output_lengths = []
        out_lengths = lengths

        for k, augment_name in enumerate(selected_augmentations):
            augment_fun = self.augmentations[augment_name]

            idx = torch.arange(x.shape[0])
            if self.parallel_augment and self.parallel_augment_fixed_bs:
                idx_startstop = torch.linspace(
                    0, x.shape[0], len(selected_augmentations) + 1
                ).to(torch.int)
                idx_start = idx_startstop[k]
                idx_stop = idx_startstop[k + 1]
                idx = idx[idx_start:idx_stop]

            # Check input arguments
            if self.require_lengths[augment_name]:
                out = augment_fun(
                    next_input[idx, ...], lengths=next_lengths[idx]
                )
            else:
                out = augment_fun(next_input[idx, ...])

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
                next_lengths = out_lengths[idx]
            else:
                output.append(out)
                output_lengths.append(out_lengths[idx])

        if self.parallel_augment:
            # Concatenate all the augmented data
            output = torch.cat(output, dim=0)
            output_lengths = torch.cat(output_lengths, dim=0)
        else:
            # Take the last agumented signal of the pipeline
            output = out
            output_lengths = out_lengths

        return output, output_lengths

    def forward(self, x, lengths):
        """Applies data augmentation.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to augment.
        lengths : torch.Tensor
            The length of each sequence in the batch.
        """
        # Select the number of augmentations to apply
        N_augment = torch.randint(
            low=self.min_augmentations,
            high=self.max_augmentations + 1,
            size=(1,),
        )

        # Get augmentations list
        augmentations_lst = list(self.augmentations.keys())

        # No augmentation
        if (
            self.repeat_augment == 0
            or N_augment == 0
            or len(self.augmentations) == 0
        ):
            return x, lengths

        # Shuffle augmentation
        if self.shuffle_augmentations:
            random.shuffle(augmentations_lst)
        # Select the augmentations to apply
        selected_augmentations = augmentations_lst[0:N_augment]

        # # Select the augmentations to apply
        # selected_augmentations = list(self.augmentations.keys())[0:N_augment]
        #
        # # Shuffle augmentation
        # if self.shuffle_augmentations:
        #     random.shuffle(selected_augmentations)

        # Lists to collect the outputs
        output_lst = []
        output_len_lst = []

        # Concatenate the original signal if required
        if self.concat_original:
            output_lst.append(x)
            output_len_lst.append(lengths)

        # Perform augmentations
        for i in range(self.repeat_augment):
            output, output_lengths = self.augment(
                x, lengths, selected_augmentations
            )
            output_lst.append(output)
            output_len_lst.append(output_lengths)

        # Concatenate the final outputs
        output = torch.cat(output_lst, dim=0)
        output_lengths = torch.cat(output_len_lst, dim=0)

        return output, output_lengths

    def check_min_max_augmentations(self):
        """Checks the min_augmentations and max_augmentations arguments.
        """
        if self.min_augmentations is None:
            self.min_augmentations = 1
        if self.max_augmentations is None:
            self.max_augmentations = len(self.augmentations)
        if self.max_augmentations > len(self.augmentations):
            self.max_augmentations = len(self.augmentations)
        if self.min_augmentations > len(self.augmentations):
            self.min_augmentations = len(self.augmentations)

        if self.max_augmentations < self.min_augmentations:
            raise ValueError(
                "max_augmentations cannot be smaller than min_augmentations "
            )
        if self.min_augmentations < 0:
            raise ValueError("min_augmentations cannot be smaller than 0.")
        if self.max_augmentations < 0:
            raise ValueError("max_augmentations cannot be smaller than 0.")
