"""Provides a function to compute the BERTscore metric.

Authors
* Sylvain de Langen 2024
"""

from dataclasses import dataclass
from typing import Optional
import torch

from speechbrain.utils.distances import cosine_similarity_matrix

@dataclass
class BERTScoreOutput:
    """Output of an evaluation of BERTScore for a batch of refs and hypothesis.
    """

    similarity_matrix: torch.Tensor
    """Cosine similarity matrix of shape `[batch, ref_len, hyp_len]`."""

    selected_values: torch.Tensor
    """Selected values in the similarity matrix after optional weighting.
    Of shape `[batch, ref_len]`"""

    selected_indices: torch.Tensor
    """Selected indices in the similarity matrix. May be used to map which
    contextualized token in the hypothesis were matched with which
    contextualized token in the reference text."""

    weight_sum: float
    """Sum of the weights within the selected path. If no token weights were
    specified, this will be the number of reference tokens."""

    # FIXME: NEED TO HANDLE THE FACT WE'RE NOT GOING TO HAVE A SINGLE BERTSCOREOUTPUT FOR ALL THE CORPUS
    # can we reuse the existing statistics class for recall? that'd probably be a ton better
    def recall(self, eps: float = 1.0e-8) -> float:
        """Computes the recall metric for this output.

        Arguments
        ---------
        eps : float, optional
            Epsilon value; used to avoid a division by zero."""

        sum = self.selected_values.sum(dim=-1).item()
        return sum / max(self.weight_sum, eps)


def bert_score(
        ref_hidden: torch.Tensor,
        hyp_hidden: torch.Tensor,
        ref_mask: Optional[torch.BoolTensor] = None,
        hyp_mask: Optional[torch.BoolTensor] = None,
        ref_inputs: Optional[torch.Tensor] = None,
        token_weights: Optional[torch.Tensor] = None,):
    """Computes the BERTScore evaluation metric, as described in the paper
    `BERTScore: Evaluating Text Generation with BERT <https://arxiv.org/abs/1904.09675>`_.

    BERTScore operates over contextualized tokens (e.g. the output of BERT, but
    many other models would work). See the linked resources for more details.

    Authors' own implementation of the metric can be found
    `here <https://github.com/Tiiiger/bert_score>`_. The linked page extensively
    describes the approach and compares how the BERTScore relates to human
    evaluation with many different models.

    Arguments
    ---------
    ref_hidden : torch.Tensor
        Outputs of the LM encoder for the reference text, i.e. contextualized
        tokens.
    hyp_hidden : torch.Tensor
        Outputs of the LM encoder for the hypothesis text, i.e. contextualized
        tokens.
    ref_mask : Optional[torch.BoolTensor]
        Boolean mask for the reference hidden states where `True` is a token
        that should **not** be ignored. Should be specified if all texts in the
        batch are not the same length. When provided, the shape must strictly
        match `ref_hidden.shape[:-1]`.
    hyp_mask : Optional[torch.BoolTensor]
        Boolean mask for the hypothesis hidden states where `True` is a token
        that should **not** be ignored. Should be specified if all texts in the
        batch are not the same length. When provided, the shape must strictly
        match `hyp_hidden.shape[:-1]`.
    ref_inputs : Optional[torch.Tensor]
        Tokenized reference text (usually a tensor of an integral type).
        Normally unused but must be specified if `token_weights` is provided.
        When provided, the shape must strictly match `ref_hidden.shape[:-1]`.
    token_weights : Optional[torch.Tensor]
        Token weights, of shape `[vocab_size]`. When specified, all tokens
        referenced by `ref_inputs` must be a valid index in `token_weights`.
        Otherwise, all tokens in the reference are weighted equally.

    Returns
    -------
    BERTScoreOutput
        A class containing sufficient state to directly derive useful metrics
        (e.g. recall).
    """

    # TODO: compare results with official implementation
    # TODO: implement idf calculation generically
    # TODO: add HF interface
    # TODO: add wrapper to do this with a batch size; maybe at metric class level?
    # ref_tokens = tokenizer(ref_text, return_tensors="pt", padding=True)
    # hyp_tokens = tokenizer(hyp_text, return_tensors="pt", padding=True)
    # ref_hidden = lm(**ref_tokens).last_hidden_state
    # hyp_hidden = lm(**hyp_tokens).last_hidden_state

    # shape [batch, ref dim, hyp dim]
    similarity_matrix = cosine_similarity_matrix(ref_hidden, hyp_hidden)

    # zero out similarity for any pair where either of the tokens is padding
    if ref_mask is not None:
        similarity_matrix[~ref_mask, :] = 0.0  # mask [B, X, :]
    if hyp_mask is not None:
        similarity_matrix.transpose(1, 2)[~hyp_mask, :] = 0.0  # mask [B, :, Y]

    # greedily select the "closest" hyp token for every ref token
    # thus of shape [batch, ref dim]
    selected_values, selected_indices = torch.max(similarity_matrix, dim=-1)

    if token_weights is not None:
        if ref_inputs is None:
            raise ValueError("Must specify ref_inputs when weighting tokens!")

        # ref_inputs -> weight for every ref input sampled from token_weights
        # the result is a weight tensor with the same shape as the inputs
        ref_weights = torch.index_select(
            input=token_weights,
            dim=0,
            index=ref_inputs.flatten()
        ).reshape(ref_inputs.shape)

        if ref_mask is not None:
            ref_weights = torch.masked_fill(ref_weights, ~ref_mask, 0.0)

        weight_sum = torch.sum(ref_weights, dim=-1).item()

        # weight the cosine distance for every ref token
        selected_values *= ref_weights
    else:
        if ref_mask is not None:
            weight_sum = torch.sum(ref_mask, dim=-1).item()
        else:
            weight_sum = float(selected_values.size(-1))

    return BERTScoreOutput(
        similarity_matrix=similarity_matrix,
        selected_values=selected_values,
        selected_indices=selected_indices,
        weight_sum=weight_sum
    )

