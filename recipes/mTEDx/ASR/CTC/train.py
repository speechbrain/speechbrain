#!/usr/bin/env python3
"""Recipe for training a sequence-to-sequence ASR system with mTEDx.
The system employs a wav2vec2 encoder and a CTC decoder.
Decoding is performed with greedy decoding (will be extended to beam search).

To run this recipe, do the following:
> python train.py hparams/train_wav2vec.yaml

With the default hyperparameters, the system employs a pretrained wav2vec2 encoder.
The wav2vec2 model is pretrained following the model given in the hprams file.
It may be dependent on the language.

The neural network is trained with CTC on sub-word units estimated with
character.

The experiment file is flexible enough to support a large variety of
different systems. By properly changing the parameter files, you can try
different encoders, decoders, tokens (e.g, characters instead of BPE),
training languages (all mTEDx languages), and many
other possible variations.

Authors
 * Titouan Parcollet 2021
 * Mohamed Anwar 2022
"""

import sys
import torch
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import run_on_main
from speechbrain.tokenizers.SentencePiece import SentencePiece
from recipes.mTEDx.mtedx_prepare import remove_punctuations

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.core.Brain):
    # @profile
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""

        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        # print("ids:", batch.id)
        # print("words:", batch.words)
        # print("sig, tokens shape:", batch.sig.data.shape, batch.tokens.data.shape)
        # tokens_bos, _ = batch.tokens_bos
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Forward pass
        feats = self.modules.wav2vec2(wavs)
        x = self.modules.enc(feats)
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits)

        return p_ctc, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC) given predictions and targets."""

        p_ctc, wav_lens = predictions

        ids = batch.id
        # tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        loss = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)

        if stage != sb.Stage.TRAIN:
            # Decode token terms to words
            sequence = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )

            predicted_words = self.tokenizer(sequence, task="decode_from_list")

            # Convert indices to words
            target_words = undo_padding(tokens, tokens_lens)
            target_words = self.tokenizer(target_words, task="decode_from_list")

            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        if self.auto_mix_prec:

            if not self.hparams.wav2vec2.freeze:
                self.wav2vec_optimizer.zero_grad()
            self.model_optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

            self.scaler.scale(loss).backward()
            if not self.hparams.wav2vec2.freeze:
                self.scaler.unscale_(self.wav2vec_optimizer)
            self.scaler.unscale_(self.model_optimizer)

            if self.check_gradients(loss):
                if not self.hparams.wav2vec2.freeze:
                    self.scaler.step(self.wav2vec_optimizer)
                self.scaler.step(self.model_optimizer)

            self.scaler.update()
        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)

            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            loss.backward()

            if self.check_gradients(loss):
                if not self.hparams.wav2vec2.freeze:
                    self.wav2vec_optimizer.step()
                self.model_optimizer.step()

            if not self.hparams.wav2vec2.freeze:
                self.wav2vec_optimizer.zero_grad()
            self.model_optimizer.zero_grad()

        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

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
            if not self.debug:
                self.hparams.train_logger.log_stats(
                    stats_meta={
                        "epoch": epoch,
                        "lr_model": old_lr_model,
                        "lr_wav2vec": old_lr_wav2vec,
                    },
                    train_stats=self.train_stats,
                    valid_stats=stage_stats,
                )
                self.hparams.tensorboard_logger.log_stats(
                    stats_meta={
                        "epoch": epoch,
                        "lr_model": old_lr_model,
                        "lr_wav2vec": old_lr_wav2vec,
                    },
                    train_stats=self.train_stats,
                    valid_stats=stage_stats,
                )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

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


# Define custom data procedure
def dataio_prepare(hparams, tokenizer):
    """
    This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """

    # 1. Define datasets
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["train_json"],
        replacements={"data_root": data_folder},
    )
    if hparams["sorting"] in {"ascending", "descending"}:
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=hparams["sorting"] == "descending",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False
    elif hparams["sorting"] == "random":
        pass
    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["valid_json"],
        replacements={"data_root": data_folder},
    )
    # We also sort the validation data so it is faster to validate
    valid_data = valid_data.filtered_sorted(
        sort_key="duration",
        reverse=hparams["sorting"] == "descending",
        key_max_value={"duration": hparams["avoid_if_longer_than"]},
    )

    # We also sort the test data so it is faster to test
    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_json"], replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(
        sort_key="duration",
        reverse=hparams["sorting"] == "descending",
        key_test={"lang": lambda x: x == hparams["test_lang"]},
    )

    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav_obj):
        sig = sb.dataio.dataio.read_audio(wav_obj["file"])
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("words")
    @sb.utils.data_pipeline.provides("words", "tokens_list", "tokens")
    def text_pipeline(words):
        if hparams["remove_punc_cap"]:
            words = remove_punctuations(words.lower())
        yield words
        tokens_list = tokenizer.sp.encode_as_ids(words)
        yield tokens_list
        # tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        # yield tokens_bos
        # tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        # yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "words", "tokens"],
    )
    return train_data, valid_data, test_data


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # setting seed for reproducibility
    # src: https://pytorch.org/docs/stable/notes/randomness.html
    seed = hparams["seed"]
    torch.manual_seed(seed)
    import random

    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True) #for CUDA

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Dataset preparation (parsing mTEDx)
    from recipes.mTEDx.mtedx_prepare import prepare_mtedx  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Due to DDP, we do the preparation ONLY on the main python process
    run_on_main(
        prepare_mtedx,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["data_folder"],
            "langs": sorted(hparams["langs"]),
        },
    )

    # Defining tokenizer and loading it
    tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=1000,
        annotation_train=hparams["train_json"],
        annotation_read="words",
        model_type=hparams["token_type"],
        character_coverage=hparams["character_coverage"],
        annotation_format="json",
    )

    # Create the datasets objects as well as tokenization and encoding :-D
    train_data, valid_data, test_data = dataio_prepare(hparams, tokenizer)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Adding objects to trainer.
    asr_brain.tokenizer = tokenizer

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Evaluating only test_lang
    avg_loss = asr_brain.evaluate(
        test_data,
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
    print("AVERAGE LOSS:", avg_loss)
    print("WER:", asr_brain.wer_metric.summarize("error_rate"))
    print("CER:", asr_brain.cer_metric.summarize("error_rate"))
    # write the returns predicted & true transcriptions
    import os

    test_lang = hparams["test_lang"]
    hyp_filename = (
        f"test_{test_lang}.hyp"
        if hparams["remove_punc_cap"]
        else f"test_punc_cap_{test_lang}.hyp"
    )
    hyp_filepath = os.path.join(hparams["output_folder"], hyp_filename)
    ref_filepath = os.path.join(
        hparams["output_folder"], f"test_{test_lang}.ref"
    )
    with open(hyp_filepath, "w") as hyp_fout, open(
        ref_filepath, "w"
    ) as ref_fout:
        for utt in asr_brain.wer_metric.scores:
            hyp = " ".join(utt["hyp_tokens"])
            ref = " ".join(utt["ref_tokens"])
            hyp_fout.write(hyp + "\n")
            ref_fout.write(ref + "\n")
