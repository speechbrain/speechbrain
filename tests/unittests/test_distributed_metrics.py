import math
import os

import torch
import torch.multiprocessing as mp
import torch.nn

import speechbrain as sb


def test_ddp(rank, size, backend="gloo"):
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

    sb.core.DistributedState(device="cpu", distributed_backend=backend)

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

    def test_gather_object():
        obj = [{"rank": rank}]
        gathered = sb.utils.distributed_metrics.gather_object(obj)
        assert gathered == [{"rank": i} for i in range(size)]

    test_gather_object()

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


def test_ddp_metrics():
    size = 2
    processes = []
    mp.set_start_method("spawn", force=True)
    os.makedirs("tests/tmp", exist_ok=True)
    for rank in range(size):
        p = mp.Process(target=test_ddp, args=(rank, size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        assert p.exitcode == 0


if __name__ == "__main__":
    test_ddp_metrics()
