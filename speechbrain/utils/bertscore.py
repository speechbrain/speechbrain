"""Provides a function to compute the BERTscore metric.

Authors
* Sylvain de Langen 2024
"""

from dataclasses import dataclass
from typing import Optional, List
import torch

from speechbrain.utils.distances import cosine_similarity_matrix
from speechbrain.utils.metric_stats import MetricStats

FIXME
#    SEP and CLS tokens should be zeroed
#    so token_weights should be edited accordingly even if non present to set it
#    to zero. also are we supposed to truncate and/or weight the tokens as they
#    are in the hyps?? because otherwise it could match against SEP/CLS...

FIXME
# why is the 3rd example so broken


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

    weight_sum: torch.Tensor
    """Sum of the weights within the selected path of shape `[batch]`. If no
    token weights were specified, this will be the number of reference tokens.
    """


def bert_score(
    ref_hidden: torch.Tensor,
    hyp_hidden: torch.Tensor,
    ref_mask: Optional[torch.BoolTensor] = None,
    hyp_mask: Optional[torch.BoolTensor] = None,
    ref_inputs: Optional[torch.Tensor] = None,
    token_weights: Optional[torch.Tensor] = None,
):
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
        tokens. May include special tokens (such as `[CLS]` and `[SEP]` tokens)
        in which case they are taken into account.
    hyp_hidden : torch.Tensor
        Outputs of the LM encoder for the hypothesis text, i.e. contextualized
        tokens. May include special tokens (such as `[CLS]` and `[SEP]` tokens)
        in which case they are taken into account.
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
        Token weighting is performed at reference word level.
        BERTScore typically uses the Inverse Document Frequency (IDF) metric.

    Returns
    -------
    BERTScoreOutput
        A class containing sufficient state to directly derive useful metrics
        (e.g. recall).
    """

    # TODO: in token_weights link to IDF function
    # TODO: compare results with official implementation
    # TODO: implement idf calculation generically

    # shape [batch, ref dim, hyp dim]
    similarity_matrix = cosine_similarity_matrix(ref_hidden, hyp_hidden)

    print(similarity_matrix)

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
            input=token_weights, dim=0, index=ref_inputs.flatten()
        ).reshape(ref_inputs.shape)

        if ref_mask is not None:
            ref_weights = torch.masked_fill(ref_weights, ~ref_mask, 0.0)

        weight_sum = torch.sum(ref_weights, dim=-1)

        # weight the cosine distance for every ref token
        selected_values *= ref_weights
    else:
        if ref_mask is not None:
            weight_sum = torch.sum(ref_mask, dim=-1)
        else:
            weight_sum = float(selected_values.size(-1))

    return BERTScoreOutput(
        similarity_matrix=similarity_matrix,
        selected_values=selected_values,
        selected_indices=selected_indices,
        weight_sum=weight_sum,
    )


class BERTScoreStats(MetricStats):
    """Computes BERTScore with a provided tokenizer and LM.
    See :func:`speechbrain.utils.bertscore.bert_score` for details.

    Arguments
    ---------
    model : transformers.AutoModelForTextEncoding
        Transformers text encoder. May live on a non-CPU device.
    tokenizer : transformers.AutoTokenizer
        Transformers tokenizer.
    batch_size : int
        How many pairs of utterances should be considered at once. Higher is
        faster but may result in OOM.
    """

    def __init__(self, lm, tokenizer, batch_size: int):
        self.clear()
        self.lm = lm
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def clear(self):
        """Clears the collected statistics"""
        self.ids = []
        self.predictions = []
        self.targets = []
        self.summary = {}

    def append(self, ids, predictions, targets):
        """
        Appends inputs, predictions and targets to internal
        lists

        Arguments
        ---------
        ids: list
            the string IDs for the samples
        predictions: list
            the model's predictions in tokenizable format
        targets: list
            the ground truths in tokenizable format
        """
        self.ids.extend(ids)
        self.predictions.extend(predictions)
        self.targets.extend(targets)

    def summarize(self, field=None):
        """Summarize the classification metric scores

        The following statistics are computed:

        TODO

        Arguments
        ---------
        field : str
            If provided, only returns selected statistic. If not,
            returns all computed statistics.

        Returns
        -------
        float or dict
            Returns a float if ``field`` is provided, otherwise
            returns a dictionary containing all computed stats.
        """

        # TODO: calculate IDF weights

        selected_sum = 0.0
        weight_sum = 0.0

        for chunk_idx in range(0, len(self.predictions), self.batch_size):
            chunk_refs = self.targets[chunk_idx:chunk_idx+self.batch_size]
            chunk_hyps = self.predictions[chunk_idx:chunk_idx+self.batch_size]

            with torch.no_grad():
                chunk_stats = self._get_batch_stats(chunk_refs, chunk_hyps)

                selected_sum += chunk_stats.selected_values.sum().item()
                weight_sum += chunk_stats.weight_sum.sum().item()

                print(chunk_stats.selected_values)
                #chunk_stats.selected_values.sum()

        self.summary["bertscore-recall"] = selected_sum / weight_sum

        if field is not None:
            return self.summary[field]
        else:
            return self.summary


    def _get_batch_stats(self, ref_text: List[str], hyp_text: List[str]):
        ref_tokens = self.tokenizer(ref_text, return_tensors="pt", padding=True)
        hyp_tokens = self.tokenizer(hyp_text, return_tensors="pt", padding=True)
        ref_hidden = self.lm(**ref_tokens).last_hidden_state
        hyp_hidden = self.lm(**hyp_tokens).last_hidden_state

        return bert_score(
            ref_hidden=ref_hidden,
            hyp_hidden=hyp_hidden,
            ref_mask=ref_tokens["attention_mask"].bool(),
            hyp_mask=hyp_tokens["attention_mask"].bool(),
            ref_inputs=ref_tokens["input_ids"],
            # token_weights=TODO
        )