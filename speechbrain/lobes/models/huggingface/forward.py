"""Helpers library to provide common forward functions (reduces boilerplate code).

Authors
 * Titouan Parcollet 2021
 * Boumadane Abdelmoumene 2021
 * Andreas Nautsch 2022
"""
import torch
import logging
import numpy as np

# from typing import List
import torch.nn.functional as F

# We check if transformers is installed.
try:
    import transformers
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        _compute_mask_indices,
    )

except ImportError:
    MSG = "Please install transformers from HuggingFace to use wav2vec2 / Hubert\n"
    MSG += "E.G. run: pip install transformers"
    raise ImportError(MSG)

logger = logging.getLogger(__name__)


def default(model, data):
    """Takes input data and returns its forward pass of a given model.

    Default for HuggingFaceTransformer init argument:
        `forward_partial_fn: Union[Callable, None] = None`
    as it is invoked by:
        `self.forward_partial_fn = partial(default_forward, model=self.model)`.

        Note: `model` is a required parameter, and handled in the init function scope.

    Called in: HuggingFaceTransformer.forward - invoked by:
        `return self.forward_partial_fn(data=data)`.

    If you have another forward function:
     * create your function in your recipe (copy this one and modify as you see fit)
     * pass it to HuggingFaceTransformer init as a partial callable
    This function serves as a reference example to your implementation, only.
    Check out `wav2vec2_forward` and `wav2vec2_pretraining_forward` below for more reference examples.

    Arguments
    ---------
    model : transformers.AutoModel
        A valid HuggingFace transformers model.
    data : torch.Tensor (signal)
        A batch of audio signals to transform to features.

    Returns
    -------
    out : torch.Tensor
        Batch of depending model outputs
    """
    out = model(data)
    return out


# TODO drop normalize_wav & output_norm arguments
def wav2vec2(
    model,
    data,
    output_all_hiddens=False,
    normalize_wav=False,
    output_norm=False,
):
    """Takes an input waveform and return its corresponding wav2vec encoding.

    Used in `HuggingFaceWav2Vec2(HuggingFaceTransformer)` init when calling `super().__init__` as:
        forward_partial_fn=partial(
                wav2vec2_forward,
                output_all_hiddens=output_all_hiddens  # here (default) values are assigned to this partial
            )
    Then, `forward_partial_fn` is handled in the HuggingFaceTransformer init by:
        self.forward_partial_fn = forward_partial_fn
        self.forward_partial_fn.keywords["model"] = self.model

    If you have another forward function:
     * create your function in your recipe (copy this one and modify as you see fit)
     * pass it to HuggingFaceTransformer init as a partial callable
    This function serves as a reference example to your implementation, only.

    See above `default_forward` documentation.

    Arguments
    ---------
    model : transformers.AutoModel
        A valid HuggingFace transformers model.
    data : torch.Tensor (signal)
        A batch of audio signals to transform to features.
    output_all_hiddens : bool (default: False)
        If True, the forward function outputs the hidden states from all transformer layers.
        For example wav2vec2-base has 12 transformer layers and the output is of shape (13, B, T, C),
        where a projection of the CNN output is added to the beginning.
        If False, the forward function outputs the hidden states only from the last transformer layer.

    Returns
    -------
    out : torch.Tensor
        Batch of depending model outputs
    norm_shape : List[int]
        Shape to be used in layer norm.
    """
    # TODO drop this; it's ontologically superfluous
    if normalize_wav:
        data = F.layer_norm(data, data.shape)

    # Extract wav2vec output
    out = model(data, output_hidden_states=True)

    if output_all_hiddens:
        out = torch.stack(list(out.hidden_states), dim=0)
        norm_shape = out.shape[-3:]
    else:
        out = out.last_hidden_state
        norm_shape = out.shape

    # TODO drop this; it's ontologically superfluous
    if output_norm:
        out = F.layer_norm(out, norm_shape)

    return out


# TODO drop normalize_wav argument
def wav2vec2_pretraining(
    model, data, mask_prob, mask_length, normalize_wav=False
):
    """Takes an input waveform and return its corresponding wav2vec encoding.

    Used in `HuggingFaceWav2Vec2Pretrain(HuggingFaceTransformer)` init when calling `super().__init__` as:
        forward_partial_fn=partial(
                wav2vec2_pretraining_forward,
                mask_prob=mask_prob,  # here (default) values are assigned to this partial
                mask_length=mask_length,  # here (default) values are assigned to this partial
            )
    Then, `forward_partial_fn` is handled in the HuggingFaceTransformer init by:
        self.forward_partial_fn = forward_partial_fn
        self.forward_partial_fn.keywords["model"] = self.model

    If you have another forward function:
     * create your function in your recipe (copy this one and modify as you see fit)
     * pass it to HuggingFaceTransformer init as a partial callable
    This function serves as a reference example to your implementation, only.

    See above `default_forward` documentation.

    Parameters
    ----------
    model : transformers.AutoModel
        A valid HuggingFace transformers model.
    data : torch.Tensor (signal)
        A batch of audio signals to transform to features.
    mask_prob : float
        Probability of masking a given frame. Default is taken from the paper.
    mask_length : int
        Length (i.e. number of consecutive masked frames). Default is taken from
        the paper.

    Returns
    -------
    out : torch.Tensor
        Batch of depending model outputs
    torch_mask_time_indices : List[int]
        Shape to be used in layer norm. # TODO check doc
    """
    batch_size, raw_sequence_length = data.shape
    # TODO drop this; it's ontologically superfluous
    if normalize_wav:
        data = F.layer_norm(data, data.shape)

    sequence_length = model._get_feat_extract_output_lengths(
        raw_sequence_length
    )

    # 1. Compute the indices that will be masked
    mask_time_indices = _compute_mask_indices(
        (batch_size, sequence_length),
        mask_prob=mask_prob,
        mask_length=mask_length,
    )
    torch_mask_time_indices = torch.tensor(
        mask_time_indices, device=data.device, dtype=torch.long,
    )

    # 2. Sample the negative samples from the entire sequence.
    # Fairseq does it only on the masked indices, but this only work if you
    # have long sentences. For more versatily, we sample on the entire sequence.
    # value.
    full_sentence_indices = np.ones((batch_size, sequence_length))
    sampled_negative_indices = transformers.models.wav2vec2.modeling_wav2vec2._sample_negative_indices(
        (batch_size, sequence_length.numpy()),
        num_negatives=model.config.num_negatives,
        mask_time_indices=full_sentence_indices,
    )

    negative_sample_indices = torch.tensor(
        sampled_negative_indices, device=data.device, dtype=torch.long,
    )

    # 3. prepare the output
    out = model(
        data,
        mask_time_indices=torch_mask_time_indices,
        sampled_negative_indices=negative_sample_indices,
    )

    return out, torch_mask_time_indices
