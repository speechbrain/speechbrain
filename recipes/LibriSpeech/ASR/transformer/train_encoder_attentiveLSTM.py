#!/usr/bin/env/python3
"""Recipe for training a sequence-to-sequence ASR system with mini-librispeech.
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with beam search coupled with a neural
language model.

To run this recipe, do the following:
> python train.py train.yaml

With the default hyperparameters, the system employs an LSTM encoder.
The decoder is based on a standard  GRU. Beam search coupled with an RNN language
model is used on the top of decoder probabilities.

The neural network is trained on both CTC and negative-log likelihood
targets and sub-word units estimated with Byte Pairwise Encoding (BPE)
are used as basic recognition tokens. Training is performed on the mini-librispeech
dataset. Note that this is a tiny dataset used here just to
provide a working example. To achieve a better performance you have to train with
larger datasets, such as the full LibriSpeech one. In this case, to allow the
model to converge, we pre-train it with a bigger one (trained on the full librispeech
with the seq2seq 1k BPE recipe).

The experiment file is flexible enough to support a large variety of
different systems. By properly changing the parameter files, you can try
different encoders, decoders, tokens (e.g, characters instead of BPE).

This recipe assumes that the tokenizer and the LM are already trained.
To avoid token mismatches, the tokenizer used for the acoustic model is
the same use for the LM.  The recipe downloads the pre-trained tokenizer
and LM.

If you would like to train a full system from scratch do the following:
1- Train a tokenizer (see ../Tokenizer)
2- Train a language model (see ../LM)
3- Train the speech recognizer (with this code).


Authors
 * Mirco Ravanelli 2020
 * Ju-Chieh Chou 2020
 * Abdel Heba 2020
 * Peter Plantinga 2020
 * Samuele Cornell 2020
"""
import os
import sys
import torch
import logging
import speechbrain as sb
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

logger = logging.getLogger(__name__)


# Brain class for speech recognition training
class ASR(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""

    def compute_forward(self, batch, stage):
        """Runs all the computation of the CTC + seq2seq ASR. It returns the
        posterior probabilities of the CTC and seq2seq networks.

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : dict
            At training time it returns predicted seq2seq log probabilities.
            If needed it also returns the ctc output log probabilities.
            At validation/test time, it returns the predicted tokens as well.
        """
        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)
        feats, self.feat_lens = self.prepare_features(stage, batch.sig)
        tokens_bos, _ = self.prepare_tokens(stage, batch.tokens_bos)

        # Running the encoder (prevent propagation to feature extraction)
        # encoded_signal = self.modules.encoder(feats.detach())
        src = self.modules.CNN(feats.detach())
        encoded_signal, _ = self.modules.transformer(
            src, tokens_bos, self.feat_lens, pad_idx=self.hparams.pad_index,
        )

        # Embed tokens and pass tokens & encoded signal to decoder
        embedded_tokens = self.modules.embedding(tokens_bos)
        decoder_outputs, _ = self.modules.decoder(
            embedded_tokens, encoded_signal, self.feat_lens
        )

        # Output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(decoder_outputs)
        predictions = {"seq_logprobs": self.hparams.log_softmax(logits)}

        if self.is_ctc_active(stage):
            # Output layer for ctc log-probabilities
            ctc_logits = self.modules.ctc_lin(encoded_signal)
            predictions["ctc_logprobs"] = self.hparams.log_softmax(ctc_logits)
        elif stage == sb.Stage.VALID:
            predictions["tokens"], _ = self.hparams.valid_search(
                encoded_signal, self.feat_lens
            )
        elif stage == sb.Stage.TEST:
            predictions["tokens"], _ = self.hparams.test_search(
                encoded_signal, self.feat_lens
            )

        return predictions

    def is_ctc_active(self, stage):
        """Check if CTC is currently active.

        Arguments
        ---------
        stage : sb.Stage
            Currently executing stage.
        """
        if stage != sb.Stage.TRAIN:
            return False
        current_epoch = self.hparams.epoch_counter.current
        return current_epoch <= self.hparams.number_of_ctc_epochs

    def prepare_features(self, stage, wavs):
        """Prepare features for computation on-the-fly

        Arguments
        ---------
        stage : sb.Stage
            Currently executing stage.
        wavs : tuple
            The input signals (tensor) and their lengths (tensor).
        """
        wavs, wav_lens = wavs

        # Add augmentation if specified. In this version of augmentation, we
        # concatenate the original and the augment batches in a single bigger
        # batch. This is more memory-demanding, but helps to improve the
        # performance. Change it if you run OOM.
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])

            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        # Feature computation and normalization
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)

        return feats, wav_lens

    def prepare_tokens(self, stage, tokens):
        """Double the tokens batch if features are doubled.

        Arguments
        ---------
        stage : sb.Stage
            Currently executing stage.
        tokens : tuple
            The tokens (tensor) and their lengths (tensor).
        """
        tokens, token_lens = tokens
        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens = torch.cat([tokens, tokens], dim=0)
            token_lens = torch.cat([token_lens, token_lens], dim=0)
        return tokens, token_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs. We here
        do multi-task learning and the loss is a weighted sum of the ctc + seq2seq
        costs.

        Arguments
        ---------
        predictions : dict
            The output dict from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        # Compute sequence loss against targets with EOS
        tokens_eos, tokens_eos_lens = self.prepare_tokens(
            stage, batch.tokens_eos
        )
        loss = sb.nnet.losses.nll_loss(
            log_probabilities=predictions["seq_logprobs"],
            targets=tokens_eos,
            length=tokens_eos_lens,
            label_smoothing=self.hparams.label_smoothing,
        )

        # Add ctc loss if necessary. The total cost is a weighted sum of
        # ctc loss + seq2seq loss
        if self.is_ctc_active(stage):
            # Load tokens without EOS as CTC targets
            tokens, tokens_lens = self.prepare_tokens(stage, batch.tokens)
            loss_ctc = self.hparams.ctc_cost(
                predictions["ctc_logprobs"], tokens, self.feat_lens, tokens_lens
            )
            loss *= 1 - self.hparams.ctc_weight
            loss += self.hparams.ctc_weight * loss_ctc

        if stage != sb.Stage.TRAIN:
            # Converted predicted tokens from indexes to words
            predicted_words = [
                self.hparams.tokenizer.decode_ids(prediction).split(" ")
                for prediction in predictions["tokens"]
            ]
            target_words = [words.split(" ") for words in batch.words]

            # Monitor word error rate and character error rated at
            # valid and test time.
            self.wer_metric.append(batch.id, predicted_words, target_words)
            self.cer_metric.append(batch.id, predicted_words, target_words)

        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        # Set up statistics trackers for this stage
        # In this case, we would like to keep track of the word error rate (wer)
        # and the character error rate (cer)
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Summarize the statistics from the stage for record-keeping.
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            # Update learning rate
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )

        # We also write statistics about test data to stdout and to the logfile.
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
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
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]
    valtest_datasets = [valid_data] + [i for k, i in test_datasets.items()]

    # We get the tokenizer as we need it to encode the labels when creating
    # mini-batches.
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(valtest_datasets, audio_pipeline)

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline_train(wav):
        # Speed Perturb is done here so it is multi-threaded with the
        # workers of the dataloader (faster).
        if hparams["speed_perturb"]:
            sig = sb.dataio.dataio.read_audio(wav)
            # factor = np.random.uniform(0.95, 1.05)
            # sig = resample(sig.numpy(), 16000, int(16000*factor))
            speed = sb.processing.speech_augmentation.SpeedPerturb(
                16000, [x for x in range(95, 105)]
            )
            sig = speed(sig.unsqueeze(0)).squeeze(0)  # torch.from_numpy(sig)
        else:
            sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item([train_data], audio_pipeline_train)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"],
    )

    # 5. If Dynamic Batching is used, we instantiate the needed samplers.
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams = hparams["dynamic_batch_sampler"]
        num_buckets = dynamic_hparams["num_buckets"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            dynamic_hparams["max_batch_len"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
        )

        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            dynamic_hparams["max_batch_len_val"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
        )

    return (
        train_data,
        valid_data,
        test_datasets,
        tokenizer,
        train_batch_sampler,
        valid_batch_sampler,
    )


if __name__ == "__main__":

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation, to be run on only one process.
    from librispeech_prepare import prepare_librispeech  # noqa

    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )

    # We can now directly create the datasets for training, valid, and test
    (
        train_data,
        valid_data,
        test_datasets,
        tokenizer,
        train_bsampler,
        valid_bsampler,
    ) = dataio_prepare(hparams)

    # In this case, pre-training is essential because mini-librispeech is not
    # big enough to train an end-to-end model from scratch. With bigger dataset
    # you can train from scratch and avoid this step.
    # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected()

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.

    # adding objects to trainer:
    asr_brain.tokenizer = hparams["tokenizer"]
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
    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        asr_brain.hparams.wer_file = os.path.join(
            hparams["output_folder"], "wer_{}.txt".format(k)
        )
        asr_brain.evaluate(
            test_datasets[k],
            max_key="ACC",
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )
