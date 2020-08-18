"""
Equal Error Rate (EER)

Authors
 * Mirco Ravanelli 2020
"""
import torch


def EER(positive_scores, negative_scores):
    """Computes the Equal Error Rate.

    Arguments
    ---------
    positive_scores : torch.tensor
        The scores from entiries of the same class.
    negative_scores : torch.tensor
        The scores from entiries of different classes.

    Outputs
    -------
    EER: float

    Example
    -------
    >>> postive_scores=torch.tensor([0.6, 0.7, 0.8, 0.5])
    >>> negative_scores=torch.tensor([0.6, 0.4, 0.3, 0.2])
    >>> EER(postive_scores,negative_scores)
    0.25
    """

    # Computing candidate thresholds
    positive_scores = positive_scores.float()
    negative_scores = negative_scores.float()
    thresholds, _ = torch.sort(torch.cat([positive_scores, negative_scores]))
    thresholds = torch.unique(thresholds)

    # Adding intermediate thresholds
    interm_thresholds = (thresholds[0:-1] + thresholds[1:]) / 2
    thresholds, _ = torch.sort(torch.cat([thresholds, interm_thresholds]))

    # Computing False Rejection Rate
    positive_scores = torch.cat(
        len(thresholds) * [positive_scores.unsqueeze(0)]
    )
    positive_scores_threshold = positive_scores.transpose(0, 1) <= thresholds
    FRR = (positive_scores_threshold.sum(0)).float() / positive_scores.shape[1]
    del positive_scores
    del positive_scores_threshold

    # Computing False Aceptance Rate
    negative_scores = torch.cat(
        len(thresholds) * [negative_scores.unsqueeze(0)]
    )
    negative_scores_threshold = negative_scores.transpose(0, 1) > thresholds

    FAR = (negative_scores_threshold.sum(0)).float() / negative_scores.shape[1]
    del negative_scores
    del negative_scores_threshold

    # Finding the threshold for EER
    min_index = (FAR - FRR).abs().argmin()

    # It is possible that eer != fpr != fnr. We return (FAR  + FRR) / 2 as EER.
    EER = (FAR[min_index] + FRR[min_index]) / 2

    return float(EER)
