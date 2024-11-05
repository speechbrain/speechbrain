#!/usr/bin/env/python3
"""AISHELL-1 CTC recipe.
The system employs a wav2vec2 encoder and a CTC decoder.
Decoding is performed with greedy decoding.

To run this recipe, do the following:
> python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml

With the default hyperparameters, the system employs a pretrained wav2vec2 encoder.
The wav2vec2 model is pretrained on 10k hours Chinese data
The network is trained with CTC on characters extracted from a pretrained tokenizer.

Authors
 * Yingzhi WANG 2022
"""

import sys

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "SpeedPerturb"):
                wavs = self.hparams.speed_perturb(wavs, wav_lens)

        # Forward pass
        feats = self.modules.wav2vec2(wavs, wav_lens)

        # Add feature augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "fea_augment"):
            feats, fea_lens = self.hparams.fea_augment(feats, wav_lens)

        x = self.modules.enc(feats)
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits)

        return p_ctc, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the CTC loss given predictions and targets."""
        p_ctc, wav_lens = predictions
        ids = batch.id
        tokens, tokens_lens = batch.tokens

        # Labels must be extended if parallel augmentation or concatenated
        # augmentation was performed on the input (increasing the time dimension)
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "fea_augment"):
                tokens = self.hparams.fea_augment.replicate_labels(tokens)
                tokens_lens = self.hparams.fea_augment.replicate_labels(
                    tokens_lens
                )

        loss = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)

        if stage == sb.Stage.VALID:
            # Decode token terms to words
            sequences = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
            predicted_words_list = []

            for sequence in sequences:
                # Decode token terms to words
                predicted_tokens = self.tokenizer.convert_ids_to_tokens(
                    sequence
                )

                predicted_words = []
                for c in predicted_tokens:
                    if c == "[CLS]":
                        continue
                    elif c == "[SEP]" or c == "[PAD]":
                        break
                    else:
                        predicted_words.append(c)

                predicted_words_list.append(predicted_words)

        elif stage == sb.Stage.TEST:
            p_tokens = test_searcher(p_ctc, wav_lens)
            # select one-best
            text_hyps = [hyp[0].text for hyp in p_tokens]

            predicted_words_list = []
            preds = []
            for seq in text_hyps:
                seq = seq.replace("[CLS]", "")
                seq = seq.replace("[SEP]", "")
                seq = seq.replace("[PAD]", "")
                for c in seq:
                    preds.append(c)
                predicted_words_list.append(preds)

        if stage != sb.Stage.TRAIN:
            target_words_list = [list(wrd) for wrd in batch.wrd]
            self.cer_metric.append(
                ids=ids,
                predict=predicted_words_list,
                target=target_words_list,
            )

        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        self.batch_idx = 0
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_model, new_lr_model = self.hparams.lr_annealing_model(
                stage_stats["loss"]
            )
            old_lr_wav2vec, new_lr_wav2vec = self.hparams.lr_annealing_wav2vec(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.model_optimizer, new_lr_model
            )
            if not self.hparams.wav2vec2.freeze:
                sb.nnet.schedulers.update_learning_rate(
                    self.wav2vec_optimizer, new_lr_wav2vec
                )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_model": old_lr_model,
                    "lr_wav2vec": old_lr_wav2vec,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"CER": stage_stats["CER"]},
                min_keys=["CER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.cer_file, "w", encoding="utf-8") as w:
                self.cer_metric.write_stats(w)

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"

        # If the wav2vec encoder is unfrozen, we create the optimizer
        if not self.hparams.wav2vec2.freeze:
            self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
                self.modules.wav2vec2.parameters()
            )
            if self.checkpointer is not None:
                self.checkpointer.add_recoverable(
                    "wav2vec_opt", self.wav2vec_optimizer
                )

        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)

        if not self.hparams.wav2vec2.freeze:
            self.optimizers_dict = {
                "wav2vec_optimizer": self.wav2vec_optimizer,
                "model_optimizer": self.model_optimizer,
            }
        else:
            self.optimizers_dict = {"model_optimizer": self.model_optimizer}

    def freeze_optimizers(self, optimizers):
        """Freezes the wav2vec2 optimizer according to the warmup steps"""
        valid_optimizers = {}
        if not self.hparams.wav2vec2.freeze:
            valid_optimizers["wav2vec_optimizer"] = optimizers[
                "wav2vec_optimizer"
            ]
        valid_optimizers["model_optimizer"] = optimizers["model_optimizer"]
        return valid_optimizers


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_data"],
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_data"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_data"],
        replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    # Defining tokenizer and loading it
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("transcript")
    @sb.utils.data_pipeline.provides("wrd", "tokens_list", "tokens")
    def text_pipeline(wrd):
        wrd = "".join(wrd.split(" "))
        yield wrd
        tokens_list = tokenizer(wrd)["input_ids"]
        yield tokens_list
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "wrd", "tokens"],
    )

    # 5. If Dynamic Batching is used, we instantiate the needed samplers.
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams = hparams["dynamic_batch_sampler"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            **dynamic_hparams,
            length_func=lambda x: x["duration"],
        )

        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            **dynamic_hparams,
            length_func=lambda x: x["duration"],
        )

    return (
        train_data,
        valid_data,
        test_data,
        tokenizer,
        train_batch_sampler,
        valid_batch_sampler,
    )


if __name__ == "__main__":
    # CLI:
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

    # 1.  # Dataset prep (parsing Librispeech)
    from aishell_prepare import prepare_aishell  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_aishell,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["output_folder"],
            "skip_prep": hparams["skip_prep"],
            "remove_compressed_wavs": hparams["remove_compressed_wavs"],
        },
    )

    # here we create the datasets objects as well as tokenization and encoding
    (
        train_data,
        valid_data,
        test_data,
        tokenizer,
        train_bsampler,
        valid_bsampler,
    ) = dataio_prepare(hparams)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    asr_brain.tokenizer = tokenizer
    vocab_list = [
        tokenizer.convert_ids_to_tokens(i) for i in range(tokenizer.vocab_size)
    ]
    test_searcher = hparams["test_searcher"](
        blank_index=hparams["blank_index"],
        vocab_list=vocab_list,
        alpha=hparams["alpha"],
        beta=hparams["beta"],
        beam_size=hparams["beam_size"],
        beam_prune_logp=hparams["beam_prune_logp"],
        token_prune_min_logp=hparams["token_prune_min_logp"],
        prune_history=hparams["prune_history"],
        topk=hparams["topk"],
        kenlm_model_path=hparams.get("kenlm_model_path"),
    )

    # Changing the samplers if dynamic batching is activated
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_bsampler is not None:
        train_dataloader_opts = {
            "batch_sampler": train_bsampler,
            "num_workers": hparams["num_workers"],
        }
    if valid_bsampler is not None:
        valid_dataloader_opts = {"batch_sampler": valid_bsampler}

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )

    # Testing
    asr_brain.evaluate(
        test_data,
        min_key="CER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
