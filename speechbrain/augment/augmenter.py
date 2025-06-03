"""Classes for implementing data augmentation pipelines.

Authors
 * Mirco Ravanelli 2022
"""

import random

import torch
import torch.nn.functional as F

from speechbrain.utils.callchains import lengths_arg_exists
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class Augmenter(torch.nn.Module):
    """Applies pipelines of data augmentation.

    Arguments
    ---------
    parallel_augment: bool
        If False, the augmentations are applied sequentially with
        the order specified in the pipeline argument.
        When True, all the N augmentations are concatenated in the output
        on the batch axis.
    parallel_augment_fixed_bs: bool
        If False, each augmenter (performed in parallel) generates a number of
        augmented examples equal to the batch size. Thus, overall, with this
        option N*batch size artificial data are
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
    augment_start_index: int
        The index of the first element in the input batch from which data
        augmentation should begin.
        This argument allows you to specify the starting point for applying
        data augmentation.
    augment_end_index: int
        The index of the last element in the input batch at which data
        augmentation should stop.
        You can use this argument to define the endpoint for applying data
        augmentation within the batch.
    concat_start_index: int
        If `concat_original` is set to True, you can specify a subpart of the
        original batch to concatenate in the output.
        Use this argument to select the index of the first element from the
        original input batch to start copying from.
    concat_end_index: int
        If `concat_original` is set to True, you can specify a subpart of the
        original batch to concatenate in the output. Use this argument to select
        the index of the last element from the original input batch to end the
        copying process.
    augment_prob: float
        The probability (0.0 to 1.0) of applying data augmentation. When set to 0.0,
        the original signal is returned without any augmentation. When set to 1.0,
        augmentation is always applied. Values in between determine the likelihood
        of augmentation.
    augmentations: list
        List of augmentater objects to combine to perform data augmentation.
    enable_augmentations: list
        A list of booleans used to selectively enable or disable specific augmentation
        techniques within the 'augmentations' list.
        Each boolean corresponds to an augmentation object in the 'augmentations' list
        and should be of the same length and order.
        This feature is useful for performing ablations on augmentation techniques to
        tailor them for a specific task.

    Example
    -------
    >>> from speechbrain.augment.time_domain import DropFreq, DropChunk
    >>> freq_dropper = DropFreq()
    >>> chunk_dropper = DropChunk(drop_start=100, drop_end=16000)
    >>> augment = Augmenter(parallel_augment=False, concat_original=False, augmentations=[freq_dropper, chunk_dropper])
    >>> signal = torch.rand([4, 16000])
    >>> output_signal, lengths = augment(signal, lengths=torch.tensor([0.2,0.5,0.7,1.0]))
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
        augment_start_index=0,
        augment_end_index=None,
        concat_start_index=0,
        concat_end_index=None,
        augment_prob=1.0,
        augmentations=list(),
        enable_augmentations=None,
    ):
        super().__init__()
        self.parallel_augment = parallel_augment
        self.parallel_augment_fixed_bs = parallel_augment_fixed_bs
        self.concat_original = concat_original
        self.augmentations = augmentations
        self.min_augmentations = min_augmentations
        self.max_augmentations = max_augmentations
        self.shuffle_augmentations = shuffle_augmentations
        self.augment_start_index = augment_start_index
        self.augment_end_index = augment_end_index
        self.concat_start_index = concat_start_index
        self.concat_end_index = concat_end_index
        self.repeat_augment = repeat_augment
        self.augment_prob = augment_prob
        # Check min and max augmentations
        self.check_min_max_augmentations()

        # This variable represents the total number of augmentations to perform for each signal,
        # including the original signal in the count.
        self.num_augmentations = None
        self.do_augment = True

        # Check repeat augment arguments
        if not isinstance(self.repeat_augment, int):
            raise ValueError("repeat_augment must be an integer.")

        if self.repeat_augment < 0:
            raise ValueError("repeat_augment must be greater than 0.")

        if self.augment_end_index is not None:
            if self.augment_end_index < self.augment_start_index:
                raise ValueError(
                    "augment_end_index must be smaller or equal to augment_start_index."
                )

        if self.concat_end_index is not None:
            if self.concat_end_index < self.concat_start_index:
                raise ValueError(
                    "concat_end_index must be smaller or equal to concat_start_index."
                )

        # Managing enable augmentations
        if enable_augmentations is None:
            enable_augmentations = [True] * len(augmentations)
        elif not isinstance(enable_augmentations, list):
            raise ValueError("enable_augmentations must be a list.")
        elif len(enable_augmentations) != len(augmentations):
            raise ValueError(
                "enable_augmentations must have the same length as augmentations."
            )
        else:
            augmentations = [
                aug
                for aug, enabled in zip(augmentations, enable_augmentations)
                if enabled
            ]

        # Turn augmentations into a dictionary
        self.augmentations = {
            augmentation.__class__.__name__ + str(i): augmentation
            for i, augmentation in enumerate(augmentations)
        }

        if len(self.augmentations) == 0:
            logger.warning(
                "No augmentation is applied because the augmentation list is empty."
            )

        # Check min and max augmentations
        if self.max_augmentations <= 0:
            logger.warning(
                "No augmentations applied because max_augmentations is non-positive."
            )
        if self.min_augmentations < 0:
            self.min_augmentations = 0
            logger.warning(
                "min_augmentations is negative. Modified to be non-negative."
            )
        if self.min_augmentations > self.max_augmentations:
            logger.warning(
                "min_augmentations is greater than max_augmentations. min_augmentations set to max_augmentations."
            )
            self.max_augmentations = self.min_augmentations

        # Check if augmentation modules need the length argument
        self.require_lengths = {}
        for aug_key, aug_fun in self.augmentations.items():
            self.require_lengths[aug_key] = lengths_arg_exists(aug_fun.forward)

    def augment(self, x, lengths, selected_augmentations):
        """Applies data augmentation on the selected augmentations.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to augment.
        lengths : torch.Tensor
            The length of each sequence in the batch.
        selected_augmentations: dict
            Dictionary containing the selected augmentation to apply.

        Returns
        -------
        output : torch.Tensor
            Augmented outputs.
        output_lengths : torch.Tensor
            The corresponding length of each output.
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
                output_lengths.append(out_lengths)

        if self.parallel_augment:
            # Concatenate all the augmented data
            output, output_lengths = self.concatenate_outputs(
                output, output_lengths
            )
        else:
            # Take the last augmented signal of the pipeline
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

        Returns
        -------
        output : torch.Tensor
            Augmented outputs.
        output_lengths : torch.Tensor
            The corresponding length of each output.
        """

        # Determine whether to apply data augmentation
        self.do_augment = True
        if random.random() > self.augment_prob:
            self.do_augment = False
            return x, lengths

        x_original = x
        len_original = lengths

        # Determine the ending index for augmentation, considering user-specified or default values.
        self.augment_end_index_batch = (
            min(self.augment_end_index, x.shape[0])
            if self.augment_end_index is not None
            else x.shape[0]
        )

        # If the augmentation starting index is beyond the size of the data, return the original data.
        if self.augment_start_index >= x.shape[0]:
            self.do_augment = False
            logger.warning(
                "No augmentation is applied because the augmentation start index is greater than or equal to the number of examples in the input batch."
            )
            return x, lengths

        # Select the number of augmentations to apply
        self.N_augment = torch.randint(
            low=self.min_augmentations,
            high=self.max_augmentations + 1,
            size=(1,),
            device=x.device,
        )

        # Get augmentations list
        augmentations_lst = list(self.augmentations.keys())

        # No augmentation
        if (
            self.repeat_augment == 0
            or self.N_augment == 0
            or len(augmentations_lst) == 0
        ):
            self.do_augment = False
            return x, lengths

        # Shuffle augmentation
        if self.shuffle_augmentations:
            random.shuffle(augmentations_lst)

        # Select the augmentations to apply
        selected_augmentations = augmentations_lst[0 : self.N_augment]

        # Select the portion of the input to augment and update lengths accordingly.
        x = x[self.augment_start_index : self.augment_end_index_batch]
        lengths = lengths[
            self.augment_start_index : self.augment_end_index_batch
        ]

        # Lists to collect the outputs
        output_lst = []
        output_len_lst = []

        # Concatenate the original signal if required
        self.skip_concat = not (self.concat_original)
        if self.concat_original:

            # Check start index
            if self.concat_start_index >= x_original.shape[0]:
                self.skip_concat = True
                pass
            else:
                self.skip_concat = False
                # Determine the ending index for concatenation, considering user-specified or default values.
                self.concat_end_index_batch = (
                    min(self.concat_end_index, x_original.shape[0])
                    if self.concat_end_index is not None
                    else x_original.shape[0]
                )

                output_lst.append(
                    x_original[
                        self.concat_start_index : self.concat_end_index_batch
                    ]
                )
                output_len_lst.append(
                    len_original[
                        self.concat_start_index : self.concat_end_index_batch
                    ]
                )

        # Perform augmentations
        for i in range(self.repeat_augment):
            output, output_lengths = self.augment(
                x, lengths, selected_augmentations
            )
            output_lst.append(output)
            output_len_lst.append(output_lengths)

        # Concatenate the final outputs while handling scenarios where
        # different temporal dimensions may arise due to augmentations
        # like speed change.
        output, output_lengths = self.concatenate_outputs(
            output_lst, output_len_lst
        )

        return output, output_lengths

    def concatenate_outputs(self, augment_lst, augment_len_lst):
        """
        Concatenate a list of augmented signals, accounting for varying temporal lengths.
        Padding is applied to ensure all signals can be concatenated.

        Arguments
        ---------
        augment_lst : List of torch.Tensor
            List of augmented signals to be concatenated.
        augment_len_lst : List of torch.Tensor
            List of lengths corresponding to the augmented signals.

        Returns
        -------
        concatenated_signals : torch.Tensor
            A tensor containing the concatenated signals.
        concatenated_lengths : torch.Tensor
            A tensor containing the concatenated signal lengths.

        Notes
        -----
        This function takes a list of augmented signals, which may have different temporal
        lengths due to variations such as speed changes. It pads the signals to match the
        maximum temporal dimension found among the input signals and rescales the lengths
        accordingly before concatenating them.
        """

        # Find the maximum temporal dimension (batch length) among the sequences
        max_len = max(augment.shape[1] for augment in augment_lst)

        # Rescale the sequence lengths to adjust for augmented batches with different temporal dimensions.
        augment_len_lst = [
            length * (output.shape[1] / max_len)
            for length, output in zip(augment_len_lst, augment_lst)
        ]

        # Pad sequences to match the maximum temporal dimension.
        # Note that some augmented batches, like those with speed changes, may have different temporal dimensions.
        augment_lst = [
            F.pad(output, (0, max_len - output.shape[1]))
            for output in augment_lst
        ]

        # Concatenate the padded sequences and rescaled lengths
        output = torch.cat(augment_lst, dim=0)
        output_lengths = torch.cat(augment_len_lst, dim=0)

        return output, output_lengths

    def replicate_multiple_labels(self, *args):
        """
        Replicates the labels along the batch axis a number of times that
        corresponds to the number of augmentations. Indeed parallel and
        concatenation augmentations alter the time dimension.

        Arguments
        ---------
        *args : tuple
            Input label tensors to be replicated. Can be a uniq or a list of
            torch.Tensors.

        Returns
        -------
        augmented_labels: torch.Tensor
            Labels corresponding to the augmented input. Returns as many torch.Tensor
            as given in input.
        """

        # Determine whether to apply data augmentation
        if not self.do_augment:
            return args

        list_of_augmented_labels = []

        for labels in args:
            list_of_augmented_labels.append(self.replicate_labels(labels))

        return list_of_augmented_labels

    def replicate_labels(self, labels):
        """
        Replicates the labels along the batch axis a number of times that
        corresponds to the number of augmentations. Indeed parallel and
        concatenation augmentations alter the time dimension.

        Arguments
        ---------
        labels : torch.Tensor
            Input label tensors to be replicated.

        Returns
        -------
        augmented_labels: torch.Tensor
            Labels corresponding to the augmented input. Returns as many torch.Tensor
            as given in input.
        """

        # Determine whether to apply data augmentation
        if not self.do_augment:
            return labels

        augmented_labels = []
        if self.concat_original and not (self.skip_concat):
            augmented_labels = [
                labels[self.concat_start_index : self.concat_end_index_batch]
            ]
        selected_labels = labels[
            self.augment_start_index : self.augment_end_index_batch
        ]

        if self.parallel_augment:
            selected_labels = torch.cat(
                [selected_labels] * self.N_augment, dim=0
            )

        augmented_labels = (
            augmented_labels + [selected_labels] * self.repeat_augment
        )

        augmented_labels = torch.cat(augmented_labels, dim=0)

        return augmented_labels

    def check_min_max_augmentations(self):
        """Checks the min_augmentations and max_augmentations arguments."""
        if self.min_augmentations is None:
            self.min_augmentations = 1
        if self.max_augmentations is None:
            self.max_augmentations = len(self.augmentations)
        if self.max_augmentations > len(self.augmentations):
            self.max_augmentations = len(self.augmentations)
        if self.min_augmentations > len(self.augmentations):
            self.min_augmentations = len(self.augmentations)
