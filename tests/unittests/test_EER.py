import torch


def test_EER():
    from speechbrain.utils.EER import EER

    positive_scores = torch.tensor([0.1, 0.2, 0.3])
    negative_scores = torch.tensor([0.4, 0.5, 0.6])
    eer = EER(positive_scores, negative_scores)
    assert eer == 1.0

    positive_scores = torch.tensor([0.4, 0.5, 0.6])
    negative_scores = torch.tensor([0.3, 0.2, 0.1])
    eer = EER(positive_scores, negative_scores)
    assert eer == 0.0

    positive_scores = torch.tensor([0.0, 1.0])
    negative_scores = torch.tensor([0.0, 1.0])
    eer = EER(positive_scores, negative_scores)
    assert eer == 0.5
