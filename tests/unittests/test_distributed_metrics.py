import math
import os

import torch
import torch.multiprocessing as mp
import torch.nn

import speechbrain as sb


def _test_ddp(tmpdir, rank, size, backend="gloo"):  # noqa
    """Initialize the distributed environment."""
    os.environ["WORLD_SIZE"] = f"{size}"
    os.environ["RANK"] = f"{rank}"
    os.environ["LOCAL_RANK"] = f"{rank}"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    run_opts = dict()
    run_opts["distributed_launch"] = True
    run_opts["distributed_backend"] = backend

    sb.utils.distributed.ddp_init_group(run_opts)

    def test_gather_tensor():
        tensor = torch.tensor([rank])
        gathered = sb.utils.distributed_metrics.gather(tensor)
        assert (gathered == torch.tensor(list(range(size)))).all()

        a = torch.tensor([rank])
        b = torch.tensor([rank + 1])
        a_, b_ = sb.utils.distributed_metrics.gather((a, b))
        assert (a_ == torch.tensor(list(range(size)))).all()
        assert (b_ == torch.tensor(list(range(1, size + 1)))).all()

    test_gather_tensor()

    def test_gather_object_list():
        obj = [{"rank": rank, "tensor": torch.tensor([rank])}]
        gathered = sb.utils.distributed_metrics.gather_object(obj)
        assert gathered == [
            {"rank": i, "tensor": torch.tensor([i])} for i in range(size)
        ]

        obj = [{"test": [{"rank": rank}]}]
        gathered = sb.utils.distributed_metrics.gather_object(obj)
        assert gathered == [{"test": [{"rank": i}]} for i in range(size)]

    test_gather_object_list()

    def test_gather_object_and_tensor():
        obj = [{"rank": rank, "tensor": torch.tensor([rank])}]
        gathered = sb.utils.distributed_metrics.gather_for_metrics(obj)

        assert gathered == [
            {"rank": i, "tensor": torch.tensor([i])} for i in range(size)
        ]

    test_gather_object_and_tensor()

    def test_metric_stats():
        from speechbrain.nnet.losses import l1_loss
        from speechbrain.utils.metric_stats import MetricStats

        l1_stats = MetricStats(metric=l1_loss)
        l1_stats.append(
            ids=[f"utterance1_{rank}", f"utterance2_{rank}"],
            predictions=torch.tensor([[0.1, 0.2], [0.1, 0.2]]) * rank,
            targets=torch.tensor([[0.1, 0.3], [0.2, 0.3]]) * rank,
            length=torch.ones(2),
            reduction="batch",
        )

        assert l1_stats.ids == [
            "utterance1_0",
            "utterance2_0",
            "utterance1_1",
            "utterance2_1",
        ]
        summary = l1_stats.summarize()

        assert math.isclose(summary["average"], 0.0375, rel_tol=1e-5)
        assert math.isclose(summary["min_score"], 0, rel_tol=1e-5)
        assert summary["min_id"] == "utterance1_0"
        assert math.isclose(summary["max_score"], 0.1, rel_tol=1e-5)
        assert summary["max_id"] == "utterance2_1"

    test_metric_stats()

    def test_error_rate_stats():
        from speechbrain.utils.metric_stats import ErrorRateStats

        wer_stats = ErrorRateStats()
        i2l = {1: "hello", 2: "world", 3: "the"}

        def mapper(batch):
            return [[i2l[int(x)] for x in seq] for seq in batch]

        wer_stats.append(
            ids=[f"utterance1_{rank}", f"utterance2_{rank}"],
            predict=[[3, 2, 1], [2, 3]],
            target=torch.tensor([[3, 2, 0], [2, 1, 0]]),
            target_len=torch.tensor([0.67, 0.67]),
            ind2lab=mapper,
        )
        assert wer_stats.ids == [
            "utterance1_0",
            "utterance2_0",
            "utterance1_1",
            "utterance2_1",
        ]

        summary = wer_stats.summarize()
        assert summary["WER"] == 50.0
        assert summary["insertions"] == 2
        assert summary["substitutions"] == 2
        assert summary["deletions"] == 0
        assert wer_stats.scores[0]["ref_tokens"] == ["the", "world"]
        assert wer_stats.scores[0]["hyp_tokens"] == ["the", "world", "hello"]

    test_error_rate_stats()

    def test_weighted_error_rate_stats():
        from speechbrain.utils.metric_stats import (
            ErrorRateStats,
            WeightedErrorRateStats,
        )

        # simple example where a and a' substitution get matched as similar
        def test_cost(edit, a, b):
            if edit != "S":
                return 1.0

            a_syms = ["a", "a'"]
            if a in a_syms and b in a_syms:
                return 0.1  # high similarity
            return 1.0  # low similarity

        wer_stats = ErrorRateStats()
        weighted_wer_stats = WeightedErrorRateStats(
            wer_stats, cost_function=test_cost
        )

        predict = [["d", "b", "c"], ["a'", "b", "c"]]
        refs = [["a", "b", "c"]] * 2

        wer_stats.append(
            ids=[f"utterance1_{rank}", f"utterance2_{rank}"],
            predict=predict,
            target=refs,
        )
        summary = weighted_wer_stats.summarize()
        assert weighted_wer_stats.base_stats.ids == [
            "utterance1_0",
            "utterance2_0",
            "utterance1_1",
            "utterance2_1",
        ]
        assert math.isclose(summary["weighted_wer"], 18.33333, abs_tol=1e-3)
        assert math.isclose(
            summary["weighted_substitutions"], (1.0 + 0.1) * size
        )

    test_weighted_error_rate_stats()

    def test_embedding_error_rate_stats(device):
        from speechbrain.utils.metric_stats import EmbeddingErrorRateSimilarity

        def test_word_embedding(sentence):
            if sentence == "a":
                return torch.tensor([1.0, 0.0], device=device)
            if sentence == "b":
                return torch.tensor([0.0, 1.0], device=device)
            if sentence == "c":
                return torch.tensor([0.9, 0.1], device=device)

        ember = EmbeddingErrorRateSimilarity(test_word_embedding, 1.0, 0.1, 0.4)

        assert ember("S", "a", "b") == 1.0  # low similarity
        assert ember("S", "a", "c") == 0.1  # high similarity

    # in ddp, this function doesn't change
    test_embedding_error_rate_stats("cpu")

    def test_binary_metrics(device):
        from speechbrain.utils.metric_stats import BinaryMetricStats

        binary_stats = BinaryMetricStats()
        binary_stats.append(
            ids=[
                f"utt1_{rank}",
                f"utt2_{rank}",
                f"utt3_{rank}",
                f"utt4_{rank}",
                f"utt5_{rank}",
                f"utt6_{rank}",
            ],
            scores=torch.tensor([0.1, 0.4, 0.8, 0.2, 0.3, 0.6], device=device),
            labels=torch.tensor([1, 0, 1, 0, 1, 0], device=device),
        )

        assert binary_stats.ids == [
            "utt1_0",
            "utt2_0",
            "utt3_0",
            "utt4_0",
            "utt5_0",
            "utt6_0",
            "utt1_1",
            "utt2_1",
            "utt3_1",
            "utt4_1",
            "utt5_1",
            "utt6_1",
        ]

        summary = binary_stats.summarize(threshold=0.5)
        assert summary["TP"] == 2
        assert summary["TN"] == 4
        assert summary["FP"] == 2
        assert summary["FN"] == 4

        summary = binary_stats.summarize(threshold=None)
        assert summary["threshold"] >= 0.3 and summary["threshold"] < 0.4

        summary = binary_stats.summarize(threshold=None, max_samples=1)
        assert summary["threshold"] >= 0.1 and summary["threshold"] < 0.2

    test_binary_metrics("cpu")

    def test_categorized_classification_stats():
        import pytest

        from speechbrain.utils.metric_stats import ClassificationStats

        stats = ClassificationStats()
        stats.append(
            ids=[f"1_{rank}", f"2_{rank}"],
            predictions=["B", "A"],
            targets=["B", "A"],
            categories=["C1", "C2"],
        )
        stats.append(
            ids=[f"3_{rank}", f"4_{rank}"],
            predictions=["A", "B"],
            targets=["B", "C"],
            categories=["C2", "C1"],
        )
        stats.append(
            ids=[f"5_{rank}", f"6_{rank}"],
            predictions=["A", "C"],
            targets=["B", "C"],
            categories=["C2", "C1"],
        )

        assert stats.ids == [
            "1_0",
            "2_0",
            "1_1",
            "2_1",
            "3_0",
            "4_0",
            "3_1",
            "4_1",
            "5_0",
            "6_0",
            "5_1",
            "6_1",
        ]
        summary = stats.summarize()
        assert pytest.approx(summary["accuracy"], 0.01) == 0.5
        classwise_accuracy = summary["classwise_accuracy"]
        assert pytest.approx(classwise_accuracy["C1", "B"]) == 1.0
        assert pytest.approx(classwise_accuracy["C1", "C"]) == 0.5
        assert pytest.approx(classwise_accuracy["C2", "A"]) == 1.0
        assert pytest.approx(classwise_accuracy["C2", "B"]) == 0.0

    test_categorized_classification_stats()

    def test_classification_stats_report():
        from io import StringIO

        from speechbrain.utils.metric_stats import ClassificationStats

        stats = ClassificationStats()
        stats.append(
            ids=[f"1_{rank}", f"2_{rank}"],
            predictions=["B", "A"],
            targets=["B", "A"],
        )
        stats.append(
            ids=[f"3_{rank}", f"4_{rank}"],
            predictions=["A", "B"],
            targets=["B", "C"],
        )

        assert stats.ids == [
            "1_0",
            "2_0",
            "1_1",
            "2_1",
            "3_0",
            "4_0",
            "3_1",
            "4_1",
        ]

        report_file = StringIO()
        stats.write_stats(report_file)
        report_file.seek(0)
        report = report_file.read()
        ref_report = """Overall Accuracy: 50%

Class-Wise Accuracy
-------------------
A: 2 / 2 (100.00%)
B: 2 / 4 (50.00%)
C: 0 / 2 (0.00%)

Confusion
---------
Target: A
  -> A: 2 / 2 (100.00%)
Target: B
  -> A: 2 / 4 (50.00%)
  -> B: 2 / 4 (50.00%)
Target: C
  -> B: 2 / 2 (100.00%)
"""
        assert report == ref_report

    test_classification_stats_report()

    def test_SemDistStats():
        from speechbrain.lobes.models.huggingface_transformers import (
            TextEncoder,
        )
        from speechbrain.utils.semdist import SemDistStats

        semdist_model_name = "almanach/camembert-base"
        semdist_stats = SemDistStats(
            lm=TextEncoder(
                source=semdist_model_name,
                save_path=tmpdir,
                device="cpu",
            ),
            method="meanpool",
        )

        semdist_stats.clear()
        semdist_stats.append(
            ids=[f"1_{rank}", f"2_{rank}"],
            predict=["bonjor", "mondo"],
            target=["bonjour", "monde"],
        )

        assert semdist_stats.ids == [
            "1_0",
            "2_0",
            "1_1",
            "2_1",
        ]

        summary = semdist_stats.summarize()

        assert math.isclose(summary["semdist"], 15.41, rel_tol=0.01)

    test_SemDistStats()

    def test_BleuStats():
        from speechbrain.utils.bleu import BLEUStats

        bleu = BLEUStats()
        i2l = {0: "a", 1: "b"}
        bleu.append(
            ids=[f"utterance_{rank}"],
            predict=[[0, 1, 1]],
            targets=[[[0, 1, 0]], [[0, 1, 1]], [[1, 1, 0]]],
            ind2lab=lambda batch: [[i2l[int(x)] for x in seq] for seq in batch],
        )

        assert bleu.ids == ["utterance_0", "utterance_1"]
        assert len(bleu.predicts) == 2
        assert len(bleu.targets) == 6

        stats = bleu.summarize()

        assert stats["BLEU"] == 0.0

    test_BleuStats()

    def test_BERTScoreStats():
        from speechbrain.lobes.models.huggingface_transformers import (
            TextEncoder,
        )
        from speechbrain.utils.bertscore import BERTScoreStats

        bert_model_name = "almanach/camembert-base"
        bertscore_stats = BERTScoreStats(
            lm=TextEncoder(
                source=bert_model_name,
                save_path=tmpdir,
                device="cpu",
                num_layers=2,
            ),
        )

        bertscore_stats.clear()
        bertscore_stats.append(
            ids=[f"1_{rank}", f"2_{rank}"],
            predict=["bonjor", "mondo"],
            target=["bonjour", "monde"],
        )

        assert bertscore_stats.ids == [
            "1_0",
            "2_0",
            "1_1",
            "2_1",
        ]

        summary = bertscore_stats.summarize()

        assert math.isclose(summary["bertscore-recall"], 0.8472, rel_tol=0.01)
        assert math.isclose(
            summary["bertscore-precision"], 0.9699, rel_tol=0.01
        )
        assert math.isclose(summary["bertscore-f1"], 0.9044, rel_tol=0.01)

    test_BERTScoreStats()

    def test_MultiMetricStats():
        from speechbrain.utils.metric_stats import MultiMetricStats

        def metric(a, b):
            return {"sum": a + b, "diff": a - b, "sum_sq": a**2 + b**2}

        multi_metric = MultiMetricStats(metric, batch_eval=True)

        multi_metric.append(
            [f"utt1_{rank}", f"utt2_{rank}"],
            a=torch.tensor([2.0, 1.0]),
            b=torch.tensor([1.0, 2.0]),
        )

        assert multi_metric.ids == [
            "utt1_0",
            "utt2_0",
            "utt1_1",
            "utt2_1",
        ]

        summary = multi_metric.summarize()

        assert math.isclose(summary["sum"]["average"], 3.0, rel_tol=1e-5)
        assert math.isclose(summary["diff"]["average"], 0.0, rel_tol=1e-5)
        assert math.isclose(summary["sum_sq"]["average"], 5.0, rel_tol=1e-5)

    test_MultiMetricStats()

    def test_Accuracy():
        from speechbrain.utils.Accuracy import AccuracyStats

        probs = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]]).unsqueeze(0)

        stats = AccuracyStats()
        stats.append(
            torch.log(probs),
            torch.tensor([1, 1, 0]).unsqueeze(0),
            torch.tensor([2 / 3]),
        )

        acc = stats.summarize()

        assert math.isclose(acc, 0.5, rel_tol=1e-5)

    test_Accuracy()


def test_ddp_metrics(tmpdir):
    size = 2
    processes = []

    mp.set_start_method("spawn", force=True)

    for rank in range(size):
        p = mp.Process(target=_test_ddp, args=(tmpdir, rank, size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        assert p.exitcode == 0
