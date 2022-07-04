import torch
import torch.nn
import math


def test_metric_stats(device):
    from speechbrain.utils.metric_stats import MetricStats
    from speechbrain.nnet.losses import l1_loss

    l1_stats = MetricStats(metric=l1_loss)
    l1_stats.append(
        ids=["utterance1", "utterance2"],
        predictions=torch.tensor([[0.1, 0.2], [0.1, 0.2]], device=device),
        targets=torch.tensor([[0.1, 0.3], [0.2, 0.3]], device=device),
        length=torch.ones(2, device=device),
        reduction="batch",
    )
    summary = l1_stats.summarize()
    assert math.isclose(summary["average"], 0.075, rel_tol=1e-5)
    assert math.isclose(summary["min_score"], 0.05, rel_tol=1e-5)
    assert summary["min_id"] == "utterance1"
    assert math.isclose(summary["max_score"], 0.1, rel_tol=1e-5)
    assert summary["max_id"] == "utterance2"


def test_error_rate_stats(device):
    from speechbrain.utils.metric_stats import ErrorRateStats

    wer_stats = ErrorRateStats()
    i2l = {1: "hello", 2: "world", 3: "the"}

    def mapper(batch):
        return [[i2l[int(x)] for x in seq] for seq in batch]

    wer_stats.append(
        ids=["utterance1", "utterance2"],
        predict=[[3, 2, 1], [2, 3]],
        target=torch.tensor([[3, 2, 0], [2, 1, 0]], device=device),
        target_len=torch.tensor([0.67, 0.67], device=device),
        ind2lab=mapper,
    )
    summary = wer_stats.summarize()
    assert summary["WER"] == 50.0
    assert summary["insertions"] == 1
    assert summary["substitutions"] == 1
    assert summary["deletions"] == 0
    assert wer_stats.scores[0]["ref_tokens"] == ["the", "world"]
    assert wer_stats.scores[0]["hyp_tokens"] == ["the", "world", "hello"]


def test_binary_metrics(device):
    from speechbrain.utils.metric_stats import BinaryMetricStats

    binary_stats = BinaryMetricStats()
    binary_stats.append(
        ids=["utt1", "utt2", "utt3", "utt4", "utt5", "utt6"],
        scores=torch.tensor([0.1, 0.4, 0.8, 0.2, 0.3, 0.6], device=device),
        labels=torch.tensor([1, 0, 1, 0, 1, 0], device=device),
    )
    summary = binary_stats.summarize(threshold=0.5)
    assert summary["TP"] == 1
    assert summary["TN"] == 2
    assert summary["FP"] == 1
    assert summary["FN"] == 2

    summary = binary_stats.summarize(threshold=None)
    assert summary["threshold"] >= 0.3 and summary["threshold"] < 0.4

    summary = binary_stats.summarize(threshold=None, max_samples=1)
    assert summary["threshold"] >= 0.1 and summary["threshold"] < 0.2


def test_EER(device):
    from speechbrain.utils.metric_stats import EER

    positive_scores = torch.tensor([0.1, 0.2, 0.3], device=device)
    negative_scores = torch.tensor([0.4, 0.5, 0.6], device=device)
    eer, threshold = EER(positive_scores, negative_scores)
    assert eer == 1.0
    assert threshold > 0.3 and threshold < 0.4

    positive_scores = torch.tensor([0.4, 0.5, 0.6], device=device)
    negative_scores = torch.tensor([0.3, 0.2, 0.1], device=device)
    eer, threshold = EER(positive_scores, negative_scores)
    assert eer == 0
    assert threshold > 0.3 and threshold < 0.4

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    input1 = torch.randn(1000, 64, device=device)
    input2 = torch.randn(1000, 64, device=device)
    positive_scores = cos(input1, input2)

    input1 = torch.randn(1000, 64, device=device)
    input2 = torch.randn(1000, 64, device=device)
    negative_scores = cos(input1, input2)

    eer, threshold = EER(positive_scores, negative_scores)

    correct = (positive_scores > threshold).nonzero(as_tuple=False).size(0) + (
        negative_scores < threshold
    ).nonzero(as_tuple=False).size(0)

    assert correct > 900 and correct < 1100


def test_minDCF(device):
    from speechbrain.utils.metric_stats import minDCF

    positive_scores = torch.tensor([0.1, 0.2, 0.3], device=device)
    negative_scores = torch.tensor([0.4, 0.5, 0.6], device=device)
    min_dcf, threshold = minDCF(positive_scores, negative_scores)
    assert (0.01 - min_dcf) < 1e-4
    assert threshold >= 0.6

    positive_scores = torch.tensor([0.4, 0.5, 0.6], device=device)
    negative_scores = torch.tensor([0.1, 0.2, 0.3], device=device)
    min_dcf, threshold = minDCF(positive_scores, negative_scores)
    assert min_dcf == 0
    assert threshold > 0.3 and threshold < 0.4


def test_classification_stats():
    import pytest
    from speechbrain.utils.metric_stats import ClassificationStats

    stats = ClassificationStats()
    stats.append(ids=["1", "2"], predictions=["B", "A"], targets=["B", "A"])
    stats.append(ids=["3", "4"], predictions=["A", "B"], targets=["B", "C"])

    summary = stats.summarize()
    assert pytest.approx(summary["accuracy"], 0.01) == 0.5
    classwise_accuracy = summary["classwise_accuracy"]
    assert pytest.approx(classwise_accuracy["A"]) == 1.0
    assert pytest.approx(classwise_accuracy["B"]) == 0.5
    assert pytest.approx(classwise_accuracy["C"]) == 0.0


def test_categorized_classification_stats():
    import pytest
    from speechbrain.utils.metric_stats import ClassificationStats

    stats = ClassificationStats()
    stats.append(
        ids=["1", "2"],
        predictions=["B", "A"],
        targets=["B", "A"],
        categories=["C1", "C2"],
    )
    stats.append(
        ids=["3", "4"],
        predictions=["A", "B"],
        targets=["B", "C"],
        categories=["C2", "C1"],
    )
    stats.append(
        ids=["5", "6"],
        predictions=["A", "C"],
        targets=["B", "C"],
        categories=["C2", "C1"],
    )

    summary = stats.summarize()
    assert pytest.approx(summary["accuracy"], 0.01) == 0.5
    classwise_accuracy = summary["classwise_accuracy"]
    assert pytest.approx(classwise_accuracy["C1", "B"]) == 1.0
    assert pytest.approx(classwise_accuracy["C1", "C"]) == 0.5
    assert pytest.approx(classwise_accuracy["C2", "A"]) == 1.0
    assert pytest.approx(classwise_accuracy["C2", "B"]) == 0.0


def test_classification_stats_report():
    from io import StringIO
    from speechbrain.utils.metric_stats import ClassificationStats

    stats = ClassificationStats()
    stats.append(ids=["1", "2"], predictions=["B", "A"], targets=["B", "A"])
    stats.append(ids=["3", "4"], predictions=["A", "B"], targets=["B", "C"])
    report_file = StringIO()
    stats.write_stats(report_file)
    report_file.seek(0)
    report = report_file.read()
    ref_report = """Overall Accuracy: 50%

Class-Wise Accuracy
-------------------
A: 1 / 1 (100.00%)
B: 1 / 2 (50.00%)
C: 0 / 1 (0.00%)

Confusion
---------
Target: A
  -> A: 1 / 1 (100.00%)
Target: B
  -> A: 1 / 2 (50.00%)
  -> B: 1 / 2 (50.00%)
Target: C
  -> B: 1 / 1 (100.00%)
"""
    assert report == ref_report
