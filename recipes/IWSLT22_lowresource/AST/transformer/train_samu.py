#!/usr/bin/env python3
"""Recipe for fine-tuning a wav2vec model for semantically enriching: https://arxiv.org/abs/2205.08180.

Author
 * Ha Nguyen, 2023
"""

import sys

import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


# Define training procedure
class ST(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""

        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig  # audio

        # wav2vec module
        feats = self.modules.wav2vec2(wavs)

        # self-attention pooling
        uttr_embeddings = self.modules.attn_pooling(feats)

        # norm
        uttr_embeddings = F.normalize(uttr_embeddings, p=2)

        # LaBSE
        text_embeddings = self.modules.LaBSE(batch.trans)

        return uttr_embeddings, text_embeddings

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given predictions and targets."""
        (uttr_embeddings, text_embeddings) = predictions

        B, S = uttr_embeddings.shape
        loss = 0.0
        for b in range(B):
            cosine_sim = torch.dot(
                uttr_embeddings[b].float(), text_embeddings[b].float()
            )
            loss += 1.0 - cosine_sim
        loss *= self.hparams.loss_scale
        return loss

    def init_optimizers(self):
        self.adam_optimizer = self.hparams.adam_opt_class(
            self.hparams.model.parameters()
        )

        self.optimizers_dict = {"model_optimizer": self.adam_optimizer}

        # Initializes the wav2vec2 optimizer if the model is not wav2vec2_frozen
        if not self.hparams.wav2vec2_frozen:
            self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
                self.modules.wav2vec2.parameters()
            )
            self.optimizers_dict["wav2vec_optimizer"] = self.wav2vec_optimizer

        # Initializes the labse optimizer if the model is not labse_frozen
        if not self.hparams.labse_frozen:
            self.labse_optimizer = self.hparams.labse_opt_class(
                self.modules.LaBSE.parameters()
            )
            self.optimizers_dict["labse_optimizer"] = self.labse_optimizer

    def freeze_optimizers(self, optimizers):
        """Freezes the wav2vec2 optimizer according to the warmup steps"""
        valid_optimizers = {}
        if not self.hparams.wav2vec2_frozen:
            valid_optimizers["wav2vec_optimizer"] = optimizers[
                "wav2vec_optimizer"
            ]
        if not self.hparams.labse_frozen:
            valid_optimizers["labse_optimizer"] = optimizers["labse_optimizer"]
        valid_optimizers["model_optimizer"] = optimizers["model_optimizer"]
        return valid_optimizers

    def on_stage_start(self, stage, epoch):
        """Gets called when a stage (either training, validation, test) starts."""
        return

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_loss

        else:  # valid or test
            stage_stats = {"loss": stage_loss}
            current_epoch = self.hparams.epoch_counter.current

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():
            current_epoch = self.hparams.epoch_counter.current
            old_lr_adam, new_lr_adam = self.hparams.lr_annealing_adam(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.adam_optimizer, new_lr_adam
            )

            stats_meta = {
                "epoch": current_epoch,
                "lr_adam": old_lr_adam,
            }

            if not self.hparams.wav2vec2_frozen:
                (
                    old_lr_wav2vec,
                    new_lr_wav2vec,
                ) = self.hparams.lr_annealing_wav2vec(stage_stats["loss"])
                sb.nnet.schedulers.update_learning_rate(
                    self.wav2vec_optimizer, new_lr_wav2vec
                )
                stats_meta["lr_wav2vec"] = old_lr_wav2vec

            if not self.hparams.labse_frozen:
                (old_lr_labse, new_lr_labse) = self.hparams.lr_annealing_labse(
                    stage_stats["loss"]
                )
                sb.nnet.schedulers.update_learning_rate(
                    self.labse_optimizer, new_lr_labse
                )
                stats_meta["lr_labse"] = old_lr_labse

            self.hparams.train_logger.log_stats(
                stats_meta=stats_meta,
                train_stats={"loss": self.train_stats},
                valid_stats=stage_stats,
            )

            # create checkpoint
            meta = {"loss": stage_stats["loss"], "epoch": current_epoch}
            name = "checkpoint_epoch" + str(current_epoch)

            self.checkpointer.save_and_keep_only(
                meta=meta, name=name, num_to_keep=10, min_keys=["loss"]
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )


# Define custom data procedure
def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """

    # Define audio pipeline. In this case, we simply read the path contained
    # in the variable wav with the audio reader.
    @sb.utils.data_pipeline.takes("path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the audio signal. This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    @sb.utils.data_pipeline.takes("path")
    @sb.utils.data_pipeline.provides("sig")
    def sp_audio_pipeline(wav):
        """Load the audio signal. This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio(wav)
        sig = sig.unsqueeze(0)
        sig = hparams["speed_perturb"](sig)
        sig = sig.squeeze(0)
        return sig

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("trans")
    @sb.utils.data_pipeline.provides("trans")
    def reference_text_pipeline(wrd):
        yield wrd

    datasets = {}
    data_folder = hparams["data_folder"]
    for dataset in ["train", "valid"]:
        json_path = hparams[f"{dataset}_set"]

        is_use_sp = dataset == "train" and "speed_perturb" in hparams
        audio_pipeline_func = sp_audio_pipeline if is_use_sp else audio_pipeline

        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=json_path,
            replacements={"data_root": data_folder},
            dynamic_items=[audio_pipeline_func, reference_text_pipeline],
            output_keys=["id", "sig", "duration", "trans"],
        )

    for dataset in ["test"]:
        json_path = hparams[f"{dataset}_set"]
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=json_path,
            replacements={"data_root": data_folder},
            dynamic_items=[audio_pipeline, reference_text_pipeline],
            output_keys=["id", "sig", "duration", "trans"],
        )

    # Sorting training data with ascending order makes the code  much
    # faster  because we minimize zero-padding. In most of the cases, this
    # does not harm the performance.
    if hparams["sorting"] == "ascending":
        # use smaller dataset to debug the model
        if hparams["debug"]:
            datasets["train"] = datasets["train"].filtered_sorted(
                key_min_value={"duration": 1},
                key_max_value={"duration": 5},
                sort_key="duration",
                reverse=True,
            )
            datasets["valid"] = datasets["valid"].filtered_sorted(
                key_min_value={"duration": 1},
                key_max_value={"duration": 5},
                sort_key="duration",
                reverse=True,
            )
        else:
            datasets["train"] = datasets["train"].filtered_sorted(
                sort_key="duration"
            )
            datasets["valid"] = datasets["valid"].filtered_sorted(
                sort_key="duration"
            )

        hparams["dataloader_options"]["shuffle"] = False
        hparams["dataloader_options"]["shuffle"] = False
    elif hparams["sorting"] == "descending":
        # use smaller dataset to debug the model
        if hparams["debug"]:
            datasets["train"] = datasets["train"].filtered_sorted(
                key_min_value={"duration": 1},
                key_max_value={"duration": 5},
                sort_key="duration",
                reverse=True,
            )
            datasets["valid"] = datasets["valid"].filtered_sorted(
                key_min_value={"duration": 1},
                key_max_value={"duration": 5},
                sort_key="duration",
                reverse=True,
            )
        else:
            datasets["train"] = datasets["train"].filtered_sorted(
                sort_key="duration", reverse=True
            )
            datasets["valid"] = datasets["valid"].filtered_sorted(
                sort_key="duration", reverse=True
            )

        hparams["dataloader_options"]["shuffle"] = False
        hparams["dataloader_options"]["shuffle"] = False
    elif hparams["sorting"] == "random":
        # use smaller dataset to debug the model
        if hparams["debug"]:
            datasets["train"] = datasets["train"].filtered_sorted(
                key_min_value={"duration": 3},
                key_max_value={"duration": 5},
                sort_key="duration",
            )
            datasets["valid"] = datasets["valid"].filtered_sorted(
                key_min_value={"duration": 1},
                key_max_value={"duration": 5},
            )

        hparams["dataloader_options"]["shuffle"] = True
    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    return datasets


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Create main experiment class
    st_brain = ST(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Data preparation
    import prepare_iwslt22

    if not hparams["skip_prep"]:
        run_on_main(
            prepare_iwslt22.data_proc,
            kwargs={
                "dataset_folder": hparams["root_data_folder"],
                "output_folder": hparams["data_folder"],
            },
        )

    # We can now directly create the datasets for training, valid, and test
    datasets = dataio_prepare(hparams)

    # Training
    st_brain.fit(
        st_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
    )

    # Test
    for dataset in ["valid", "test"]:
        st_brain.hparams.wer_file = (
            hparams["output_folder"] + "/wer_test" + ".txt"
        )
        st_brain.evaluate(
            datasets[dataset],
            test_loader_kwargs=hparams["test_dataloader_options"],
        )
