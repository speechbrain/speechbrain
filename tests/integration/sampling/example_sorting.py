"""This minimal example checks on sampling with ascending/descending sorting and random shuffling; w/ & w/o DDP.
"""

import os
import torch
import pickle
import pathlib
import itertools
import speechbrain as sb
import torch.multiprocessing as mp
from hyperpyyaml import load_hyperpyyaml


class SamplingBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the binary probability."
        batch = batch.to(self.device)
        lens = batch.duration

        if stage == sb.Stage.TRAIN:
            self.ids_list.append(batch.id)
            if self.hparams.sorting == "ascending":
                # ignore last; non-evenly divisible data; 99 items -> last batch: 19 -> 20 items (thus, nearby sort)
                if not all(
                    [x == y for x, y in zip(lens[:-1], sorted(lens[:-1]))]
                ):  # ":-1" is specific to dummy data
                    print(lens)
                    assert False
            elif self.hparams.sorting == "descending":
                if not all(
                    [
                        x == y
                        for x, y in zip(
                            lens[:-1], sorted(lens[:-1], reverse=True)
                        )
                    ]
                ):
                    print(lens)
                    assert False
            elif self.hparams.sorting == "random":
                assert not all(
                    [x == y for x, y in zip(lens[:-1], sorted(lens[:-1]))]
                )
            else:
                raise NotImplementedError(
                    "sorting must be random, ascending or descending"
                )

        return lens

    def compute_objectives(self, predictions, batch, stage=True):
        "Given the network predictions and targets computed the binary CE"
        inputs = torch.tensor([10.0, -6.0], requires_grad=True)
        targets = torch.tensor([1, 0])
        loss = self.hparams.compute_loss(inputs, targets)
        return loss

    def on_stage_start(self, stage, epoch=None):
        "Gets called when a stage (either training, validation, test) starts."
        if stage == sb.Stage.TRAIN:
            self.ids_list = []

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of a stage."""
        if stage == sb.Stage.TRAIN:
            # check that all IDs are unique; no duplicate IDs
            batched_ids = sorted(list(itertools.chain(*self.ids_list)))
            assert batched_ids == sorted(list(set(batched_ids)))

            # write out to check later all IDs were visited
            if self.distributed_launch:
                with open(
                    f"tests/tmp/ddp_sorting_ids_{self.hparams.sorting}_{self.hparams.rank}",
                    "wb",
                ) as f:
                    pickle.dump(batched_ids, f)


def data_prep(data_folder, hparams):
    "Creates the datasets and their data processing pipelines."

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=data_folder / "annotation/dev-clean.csv",
        replacements={"data_root": data_folder},
    )

    # start: sorting impact
    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            # key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False
        # hparams["dataloader_options"]["drop_last"] = True  # drops last entry which is out of order

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            # key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False
        # hparams["dataloader_options"]["drop_last"] = True  # drops last entry which is out of order

    elif hparams["sorting"] == "random":
        # hparams["dataloader_options"]["drop_last"] = True  # reduced length from 99 to 80
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )
    # end: sorting impact

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=data_folder / "annotation/dev-clean.csv",
        replacements={"data_root": data_folder},
    )

    datasets = [train_data, valid_data]

    # 1. Define audio pipeline:
    @sb.utils.data_pipeline.takes("duration")
    @sb.utils.data_pipeline.provides("duration")
    def audio_pipeline(duration):
        return duration

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "duration"])

    return train_data, valid_data


def recipe(device="cpu", yaml_file="hyperparams.yaml", run_opts=None):
    experiment_dir = pathlib.Path(__file__).resolve().parent
    hparams_file = os.path.join(experiment_dir, yaml_file)
    data_folder = "../../samples/"
    data_folder = (experiment_dir / data_folder).resolve()
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)

    # usually here: sb.utils.distributed.ddp_init_group(run_opts)

    # Data IO creation
    train_data, valid_data = data_prep(data_folder, hparams)

    # Trainer initialization
    if run_opts is None:
        run_opts = {}
    else:
        hparams["rank"] = run_opts["local_rank"]
    run_opts["device"] = device

    ctc_brain = SamplingBrain(
        hparams["modules"], hparams["opt_class"], hparams, run_opts=run_opts,
    )

    # Training/validation loop
    ctc_brain.fit(
        range(hparams["N_epochs"]),
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )


if __name__ == "__main__":
    recipe(yaml_file="random.yaml")
    recipe(yaml_file="asc.yaml")
    recipe(yaml_file="dsc.yaml")


def test_error(device):
    recipe(device=device, yaml_file="random.yaml")
    recipe(device=device, yaml_file="asc.yaml")
    recipe(device=device, yaml_file="dsc.yaml")


def ddp_recipes(rank, size, backend="gloo"):
    """ Initialize the distributed environment. """
    os.environ["WORLD_SIZE"] = f"{size}"
    os.environ["RANK"] = f"{rank}"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    run_opts = dict()
    run_opts["distributed_launch"] = True
    run_opts["distributed_backend"] = backend
    run_opts["local_rank"] = rank

    sb.utils.distributed.ddp_init_group(run_opts)

    recipe(device="cpu", yaml_file="random.yaml", run_opts=run_opts)
    recipe(device="cpu", yaml_file="asc.yaml", run_opts=run_opts)
    recipe(device="cpu", yaml_file="dsc.yaml", run_opts=run_opts)


def test_ddp():
    size = 2
    processes = []
    mp.set_start_method("spawn", force=True)
    os.makedirs("tests/tmp", exist_ok=True)
    for rank in range(size):
        p = mp.Process(target=ddp_recipes, args=(rank, size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        assert p.exitcode == 0

    # check all
    for sorting in ["random", "ascending", "descending"]:
        ids = []
        for rank in range(2):
            idf = f"tests/tmp/ddp_sorting_ids_{sorting}_{rank}"
            with open(idf, "rb") as f:
                ids += pickle.load(f)
        assert (
            len(ids) == 100 if sorting == "random" else 99
        )  # sorted data stays within the 99; random not
        assert len(set(ids)) == 99
