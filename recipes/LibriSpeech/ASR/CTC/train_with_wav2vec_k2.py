#!/usr/bin/env/python3
"""Recipe for training a wav2vec-based ctc ASR system with librispeech.
The system employs wav2vec as its encoder. Decoding is performed with
k2 through the use of a decoding graph and, optionally, a rescoring LM.
To run this recipe, do the following:
> python train_with_wav2vec.py hparams/train_{hf,sb}_wav2vec.yaml
The neural network is trained on CTC likelihood target and character units
are used as basic recognition tokens.

Authors
 * Pierre Champion 2023
 * Zeyu Zhao 2023
 * Georgios Karakasidis 2023
 * Rudolf A Braun 2022
 * Titouan Parcollet 2022
 * Sung-Lin Yeh 2021
 * Ju-Chieh Chou 2020
 * Mirco Ravanelli 2020
 * Abdel Heba 2020
 * Peter Plantinga 2020
 * Samuele Cornell 2020
"""

import os
import sys
from typing import List
import torch
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main, if_main_process
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path

import speechbrain.k2_integration as sbk2

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Downsample the inputs if specified
        if hasattr(self.modules, "downsampler"):
            wavs = self.modules.downsampler(wavs)

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

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
        return p_ctc, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        p_ctc, wav_lens = predictions

        ids = batch.id

        # Sort batch to be descending by length of wav files, which is demanded by k2
        if self.hparams.sorting == "ascending":
            p_ctc = torch.flip(p_ctc, (0,))
            wav_lens = torch.flip(wav_lens, (0,))
            texts = [batch.wrd[i] for i in reversed(range(len(batch.wrd)))]
        elif self.hparams.sorting == "descending":
            texts = batch.wrd
        else:
            raise NotImplementedError(
                "Only ascending or descending sorting is "
                "implemented, but got {}".format(self.hparams.sorting)
            )

        is_training = stage == sb.Stage.TRAIN
        loss = self.hparams.ctc_cost(
            log_probs=p_ctc,
            input_lens=wav_lens,
            graph_compiler=self.graph_compiler,
            texts=texts,
            is_training=is_training,
        )

        if stage == sb.Stage.VALID:
            # Decode token terms to words
            predicted_texts = self.graph_compiler.decode(
                p_ctc,
                wav_lens,
                ac_scale=self.hparams.ac_scale,
                decoding_method="1best",
            )  # list of strings
            predicted_words = [wrd.split(" ") for wrd in predicted_texts]
            target_words = [wrd.split(" ") for wrd in texts]
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)
        if stage == sb.Stage.TEST:  # Language model decoding only used for test
            lattice = sbk2.lattice_decoder.get_lattice(
                p_ctc,
                wav_lens,
                self.decoder["decoding_graph"],
                search_beam=self.hparams.test_search_beam,
                output_beam=self.hparams.test_output_beam,
                ac_scale=self.hparams.ac_scale,
                max_active_states=self.hparams.test_max_active_state,
                min_active_states=self.hparams.test_min_active_state,
            )
            key = "no_rescore"
            best_path = {
                key: sbk2.lattice_decoder.one_best_decoding(
                    lattice=lattice, use_double_scores=True
                )
            }
            decode_output = sbk2.utils.lattice_to_text(best_path[key], self.graph_compiler.lexicon.word_table)

            # best_path = 
            # return loss
            # decoding_method = self.hparams.decoding_method
            # If the decoding method is 1best then the metric stats will be
            # saved in a single file, otherwise, a new directory will be created
            # for each lm_scale used in whole lattice rescoring.
            # decode_output: List[str] = self.graph_compiler.decode(
            #     p_ctc,
            #     wav_lens,
            #     search_beam=self.hparams.test_search_beam,
            #     output_beam=self.hparams.test_output_beam,
            #     ac_scale=self.hparams.ac_scale,
            #     max_active_states=self.hparams.test_max_active_state,
            #     is_test=True,
            #     decoding_method=decoding_method,
            #     lm_scale=self.hparams.lm_scale,
            # )  # list of strings
            target_words: List[List[str]] = [wrd.split(" ") for wrd in texts]
            predicted_words: List[List[str]] = [
                snt.split(" ") for snt in decode_output
            ]
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)
        return loss

    def fit_batch(self, batch):
        should_step = self.step % self.grad_accumulation_factor == 0

        # Managing automatic mixed precision
        if self.auto_mix_prec:
            self.wav2vec_optimizer.zero_grad()
            self.model_optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                with self.no_sync():
                    outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            with self.no_sync(not should_step):
                self.scaler.scale(
                    loss / self.grad_accumulation_factor
                ).backward()
            if should_step:
                if not self.hparams.freeze_wav2vec:
                    self.scaler.unscale_(self.wav2vec_optimizer)
                self.scaler.unscale_(self.model_optimizer)
                if self.check_gradients(loss):
                    self.scaler.step(self.wav2vec_optimizer)
                    self.scaler.step(self.model_optimizer)
                self.scaler.update()
                self.optimizer_step += 1
        else:
            with self.no_sync():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            (loss / self.grad_accumulation_factor).backward()
            if should_step:
                if self.check_gradients(loss):
                    self.wav2vec_optimizer.step()
                    self.model_optimizer.step()
                self.wav2vec_optimizer.zero_grad()
                self.model_optimizer.zero_grad()
                self.optimizer_step += 1

        return loss.detach().cpu()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch. In this case,
        it initializes the wer and cer metric watchers. If the decoding
        method is whole-lattice-rescoring then a list of wer/cer metrics
        will be initialized (for each lm scale). Otherwise, a single class
        will be initialized for wer and cer, respectively.
        """
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch. During testing, its primary goal
        is to summarize the WER/CER stats and save them in a txt file.
        If the decoding method is whole-lattice-rescoring then we will
        print the WER/CER score of the best lm_scale. In addition, we will
        save the WER/CER scores of all lm_scales in separate txt files.
        If the decoding method is 1best then we will print the WER/CER score
        and save the results in a txt file.
        """
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
                with open(self.hparams.wer_file, "w") as w:
                    self.wer_metric.write_stats(w)

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        # Handling SpeechBrain vs HuggingFance pretrained models
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

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec_opt", self.wav2vec_optimizer
            )
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)

    def zero_grad(self, set_to_none=False):
        self.wav2vec_optimizer.zero_grad(set_to_none)
        self.model_optimizer.zero_grad(set_to_none)


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
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

    else:
        raise NotImplementedError("sorting must be ascending or descending")

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

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides("wrd", "char_list")
    def text_pipeline(wrd):
        yield wrd
        char_list = list(wrd)
        yield char_list

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "char_list"],
    )

    return train_data, valid_data, test_datasets


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # env_corrupt is not supported with k2 yet
    if hparams.get("env_corrupt", None):
        raise NotImplementedError("env_corrupt is not supported with k2 yet")

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing Librispeech)
    from librispeech_prepare import prepare_librispeech  # noqa

    # multi-gpu (ddp) save data preparation
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

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets = dataio_prepare(hparams)

    # Create the lexicon.txt for k2 training
    run_on_main(
        sbk2.lexicon.prepare_lexicon,
        kwargs={
            "lang_dir": hparams["lang_dir"],
            "csv_files": [hparams["output_folder"] + "/train.csv"],
            "extra_vocab_files": [hparams["vocab_file"]],
            "add_word_boundary": hparams["add_word_boundary"],
        },
    )

    # Create the lang directory for k2 training
    run_on_main(
        sbk2.prepare_lang.prepare_lang,
        kwargs={
            "lang_dir": hparams["lang_dir"],
            "sil_prob": hparams["sil_prob"],
        },
    )

    lexicon = sbk2.lexicon.Lexicon(hparams["lang_dir"])

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    graph_compiler = sbk2.graph_compiler.CtcGraphCompiler(
        lexicon=lexicon,
        device=asr_brain.device,
    )
    # Add attributes to asr_brain
    setattr(asr_brain, "graph_compiler", graph_compiler)

    # We load the pretrained wav2vec2 model
    if "pretrainer" in hparams.keys():
        run_on_main(hparams["pretrainer"].collect_files)
        hparams["pretrainer"].load_collected(asr_brain.device)

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    decoder = sbk2.lattice_decoder.get_decoding(hparams, graph_compiler, device=asr_brain.device)
    setattr(asr_brain, "decoder", decoder)

    # Testing
    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        asr_brain.hparams.wer_file = os.path.join(
            hparams["output_folder"], "wer_{}_lm_scale_{}.txt".format(k, hparams["lm_scale"])
        )
        asr_brain.evaluate(
            test_datasets[k], test_loader_kwargs=hparams["test_dataloader_opts"]
        )
