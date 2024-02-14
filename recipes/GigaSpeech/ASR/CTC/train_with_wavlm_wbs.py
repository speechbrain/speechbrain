"""TODO

Authors
 * Adel Moumen 2024
"""
import os
import sys
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main, if_main_process
from hyperpyyaml import load_hyperpyyaml
import webdataset as wds
import torch

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        # Downsample the inputs if specified
        if hasattr(self.modules, "downsampler"):
            wavs = self.modules.downsampler(wavs)

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)

        # Forward pass

        # Handling SpeechBrain vs HuggingFance pretrained models
        if hasattr(self.modules, "extractor"):  # SpeechBrain pretrained model
            latents = self.modules.extractor(wavs)
            feats = self.modules.encoder_wrapper(latents, wav_lens=wav_lens)[
                "embeddings"
            ]
        else:  # HuggingFace pretrained model
            feats = self.modules.wav2vec2(wavs, wav_lens)

        x = self.modules.enc(feats)

        # Compute outputs
        logits = self.modules.ctc_lin(x)

        # Upsample the inputs if they have been highly downsampled
        if hasattr(self.hparams, "upsampling") and self.hparams.upsampling:
            logits = logits.view(
                logits.shape[0], -1, self.hparams.output_neurons
            )

        p_ctc = self.hparams.log_softmax(logits)

        if stage == sb.Stage.VALID:
            p_tokens = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
        elif stage == sb.Stage.TEST:
            p_tokens = test_searcher(p_ctc, wav_lens)
        else:
            p_tokens = None

        return p_ctc, wav_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        p_ctc, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens, tokens_lens = batch.tokens

        # Label Augmentation
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            tokens = self.hparams.wav_augment.replicate_labels(tokens)
            tokens_lens = self.hparams.wav_augment.replicate_labels(tokens_lens)

        loss_ctc = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)
        loss = loss_ctc

        if stage == sb.Stage.VALID:
            # Decode token terms to words
            predicted_words = [
                "".join(self.tokenizer.decode_ndim(utt_seq)).split(" ")
                for utt_seq in predicted_tokens
            ]
        elif stage == sb.Stage.TEST:
            predicted_words = [
                hyp[0].text.split(" ") for hyp in predicted_tokens
            ]

        if stage != sb.Stage.TRAIN:
            target_words = [wrd.split(" ") for wrd in batch.text]
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

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
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(self.hparams.test_wer_file, "w") as w:
                    self.wer_metric.write_stats(w)

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        # Handling SpeechBrain vs HuggingFace pretrained models
        if hasattr(self.modules, "extractor"):  # SpeechBrain pretrained model
            self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
                self.modules.encoder_wrapper.parameters()
            )

        else:  # HuggingFace pretrained model
            self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
                self.modules.wav2vec2.parameters()
            )

        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )

        # save the optimizers in a dictionary
        # the key will be used in `freeze_optimizers()`
        self.optimizers_dict = {
            "model_optimizer": self.model_optimizer,
        }
        if not self.hparams.freeze_wav2vec:
            self.optimizers_dict["wav2vec_optimizer"] = self.wav2vec_optimizer

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec_opt", self.wav2vec_optimizer
            )
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    label_encoder = sb.dataio.encoder.CTCTextEncoder()
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    special_labels = {
        "blank_label": hparams["blank_index"],
        "unk_label": hparams["unk_label"],
    }

    # 1. Create datasets
    if hparams["use_webdataset"]:
        import json

        # load the meta info json file
        file_path = os.path.join(
            hparams["train_shards_folder_path"], "metadata.json"
        )
        with wds.gopen(file_path, "rb") as f:
            train_meta = json.load(f)

        file_path = os.path.join(
            hparams["valid_shards_folder_path"], "metadata.json"
        )
        with wds.gopen(file_path, "rb") as f:
            val_meta = json.load(f)

        file_path = os.path.join(
            hparams["test_shards_folder_path"], "metadata.json"
        )
        with wds.gopen(file_path, "rb") as f:
            test_meta = json.load(f)

        def _audio_pipeline(sample_dict):
            return sample_dict["text"]

        import glob

        train_files = glob.glob(hparams["train_shards_folder_path"] + "/*.tar")

        def _text_generator(shard_files):
            for shard_file in shard_files:
                for sample_dict in (
                    wds.WebDataset(shard_file).decode().map(_audio_pipeline)
                ):
                    yield sample_dict

        label_encoder.load_or_create(
            path=lab_enc_file,
            from_iterables=[_text_generator(train_files)],
            sequence_input=True,
            special_labels=special_labels,
        )

        def audio_pipeline(sample_dict):
            key = sample_dict["__key__"]
            audio_tensor = sample_dict["audio.pth"]
            text = sample_dict["text"]
            char_list = list(text)
            tokens_list = label_encoder.encode_sequence(char_list)
            tokens = torch.LongTensor(tokens_list)
            return {
                "id": key,
                "sig": audio_tensor,
                "text": text,
                "char_list": char_list,
                "tokens_list": tokens_list,
                "tokens": tokens,
            }

        train_data = (
            wds.WebDataset(train_files).repeat().decode().map(audio_pipeline)
        )

        valid_data = (
            wds.WebDataset(
                glob.glob(hparams["valid_shards_folder_path"] + "/*.tar")
            )
            .repeat()
            .decode()
            .map(audio_pipeline)
        )

        test_data = (
            wds.WebDataset(
                glob.glob(hparams["test_shards_folder_path"] + "/*.tar")
            )
            .repeat()
            .decode()
            .map(audio_pipeline)
        )

    else:
        print("Not implemented yet")

    return (
        train_data,
        valid_data,
        test_data,
        label_encoder,
        train_meta,
        val_meta,
        test_meta,
    )


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing Librispeech)
    from gigaspeech_prepare import prepare_gigaspeech  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_gigaspeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "splits": hparams["splits"],
            "output_train": hparams["train_shards_folder_path"],
            "output_dev": hparams["valid_shards_folder_path"],
            "output_test": hparams["test_shards_folder_path"],
            "json_file": hparams["json_file"],
            "skip_prep": hparams["skip_prep"],
            "convert_opus_to_wav": hparams["convert_opus_to_wav"],
            "use_webdataset": hparams["use_webdataset"],
            "samples_per_shard": hparams["samples_per_shard"],
            "max_size_shard": hparams.get("max_size_shard", 1e9),
        },
    )

    # here we create the datasets objects as well as tokenization and encoding
    (
        train_data,
        valid_data,
        test_data,
        label_encoder,
        train_meta,
        val_meta,
        test_meta,
    ) = dataio_prepare(hparams)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # We load the pretrained wav2vec2 model
    if "pretrainer" in hparams.keys():
        run_on_main(hparams["pretrainer"].collect_files)
        hparams["pretrainer"].load_collected()

    # We dynamicaly add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for the LM!!
    asr_brain.tokenizer = label_encoder

    ind2lab = label_encoder.ind2lab
    vocab_list = [ind2lab[x] for x in range(len(ind2lab))]
    print(vocab_list)
    print(len(vocab_list))

    from speechbrain.decoders.ctc import CTCBeamSearcher

    test_searcher = CTCBeamSearcher(
        **hparams["test_beam_search"], vocab_list=vocab_list,
    )

    hparams["train_dataloader_opts"]["collate_fn"] = sb.dataio.batch.PaddedBatch
    hparams["valid_dataloader_opts"]["collate_fn"] = sb.dataio.batch.PaddedBatch
    hparams["test_dataloader_opts"]["collate_fn"] = sb.dataio.batch.PaddedBatch

    hparams["train_dataloader_opts"]["looped_nominal_epoch"] = (
        train_meta["nb_samples"]
        // hparams["train_dataloader_opts"]["batch_size"]
    )

    hparams["valid_dataloader_opts"]["looped_nominal_epoch"] = (
        val_meta["nb_samples"] // hparams["valid_dataloader_opts"]["batch_size"]
    )

    hparams["test_dataloader_opts"]["looped_nominal_epoch"] = (
        test_meta["nb_samples"] // hparams["test_dataloader_opts"]["batch_size"]
    )

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Testing
    os.makedirs(hparams["output_wer_folder"], exist_ok=True)

    # report WER on valid data
    asr_brain.hparams.test_wer_file = os.path.join(
        hparams["output_wer_folder"], f"valid_wer.txt"
    )
    asr_brain.evaluate(
        valid_data,
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )

    # report WER on test data
    asr_brain.hparams.test_wer_file = os.path.join(
        hparams["output_wer_folder"], f"test_wer.txt"
    )
    asr_brain.evaluate(
        test_data,
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
