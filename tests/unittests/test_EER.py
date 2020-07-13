import torch
import torch.nn


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

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    input1 = torch.randn(1000, 64)
    input2 = torch.randn(1000, 64)
    positive_scores = cos(input1, input2)

    input1 = torch.randn(1000, 64)
    input2 = torch.randn(1000, 64)
    negative_scores = cos(input1, input2)

    eer = EER(positive_scores, negative_scores)

    assert eer < 0.53 and eer > 0.47
