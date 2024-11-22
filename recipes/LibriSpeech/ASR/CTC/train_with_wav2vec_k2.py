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
from collections import defaultdict
from pathlib import Path

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
import speechbrain.k2_integration as sbk2
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


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

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)

        # Forward pass

        # Handling SpeechBrain vs HuggingFace pretrained models
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
        paths = None
        if stage == sb.Stage.VALID or stage == sb.Stage.TEST:
            # Decode token terms to words
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
        if stage == sb.Stage.VALID:
            # 1best decoding for fast valid
            paths = {"onebest": sbk2.lattice_decoder.one_best_decoding(lattice)}
        elif stage == sb.Stage.TEST:
            # user defined decoding for test
            paths = self.decoder["decoding_method"](lattice)

        return p_ctc, wav_lens, paths

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        p_ctc, wav_lens, paths = predictions

        # Sort batch to be descending by length of wav files, which is required
        # by `k2.intersect_dense` called in `k2.ctc_loss`
        indices = torch.argsort(wav_lens, descending=True)
        p_ctc = p_ctc[indices]
        wav_lens = wav_lens[indices]
        texts = [batch.wrd[i] for i in indices]

        is_training = stage == sb.Stage.TRAIN
        loss = self.hparams.ctc_cost(
            log_probs=p_ctc,
            input_lens=wav_lens,
            graph_compiler=self.graph_compiler,
            texts=texts,
            is_training=is_training,
        )

        if stage == sb.Stage.TEST or stage == sb.Stage.VALID:
            for k, path in paths.items():
                predicted_texts = sbk2.utils.lattice_paths_to_text(
                    path, self.lexicon.word_table
                )

                predicted_words = [wrd.split(" ") for wrd in predicted_texts]
                target_words = [wrd.split(" ") for wrd in batch.wrd]
                self.wer_metrics[k].append(
                    batch.id, predicted_words, target_words
                )
                self.cer_metrics[k].append(
                    batch.id, predicted_words, target_words
                )
            # For TEST and VALID stages, the loss value is not exact.
            # The <UNK> words have a target length (e.g., number of phones or characters) of 1.
            # As such, sentences with <UNK> have a higher loss during CTC loss 'mean' reduction mode.
            # It does not impact training.
        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch. In this case,
        it initializes the wer and cer metric watchers. If the decoding
        method is whole-lattice-rescoring then a list of wer/cer metrics
        will be initialized (for each lm scale). Otherwise, a single class
        will be initialized for wer and cer, respectively.
        """
        if stage == sb.Stage.VALID:
            logger.info("Valid stage")
        if stage == sb.Stage.TEST:
            logger.info("Test stage")
        self.cer_metrics = defaultdict(self.hparams.cer_computer)
        self.wer_metrics = defaultdict(self.hparams.error_rate_computer)

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch. During testing, its primary goal
        is to summarize the WER/CER stats and save them in a file.
        """
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            # Only report the fist config (first rescoring_lm_scale value)
            stage_stats["CER"] = list(self.cer_metrics.values())[0].summarize(
                "error_rate"
            )
            stage_stats["WER"] = list(self.wer_metrics.values())[0].summarize(
                "error_rate"
            )

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
                meta={"WER": stage_stats["WER"]},
                min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                for k, stat in self.wer_metrics.items():
                    with open(
                        self.hparams.wer_file + f"_{k}.txt",
                        "w",
                        encoding="utf-8",
                    ) as w:
                        stat.write_stats(w)

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
    It also defines the data processing pipeline through user-defined functions.
    """
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
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
        csv_path=hparams["valid_csv"],
        replacements={"data_root": data_folder},
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
        datasets,
        ["id", "sig", "wrd", "char_list"],
    )

    return train_data, valid_data, test_datasets


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file, encoding="utf-8") as fin:
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
    import librispeech_prepare

    # multi-gpu (ddp) save data preparation
    run_on_main(
        librispeech_prepare.prepare_librispeech,
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

    # Download the vocabulary file for librispeech
    librispeech_prepare.download_librispeech_vocab_text(
        destination=hparams["vocab_file"]
    )

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets = dataio_prepare(hparams)

    # Create the lexicon.txt for k2
    run_on_main(
        sbk2.lexicon.prepare_char_lexicon,
        kwargs={
            "lang_dir": hparams["lang_dir"],
            "vocab_files": [hparams["vocab_file"]],
            "extra_csv_files": (
                [hparams["output_folder"] + "/train.csv"]
                if not hparams["skip_prep"]
                else []
            ),
            "add_word_boundary": hparams["add_word_boundary"],
        },
    )

    caching = (
        {"cache": False}
        if "caching" in hparams and hparams["caching"] is False
        else {}
    )

    # Create the lang directory for k2
    run_on_main(
        sbk2.prepare_lang.prepare_lang,
        kwargs={
            "lang_dir": hparams["lang_dir"],
            "sil_prob": hparams["sil_prob"],
            **caching,
        },
    )

    # OpenSLR ngram models
    if (
        hparams["G_arpa"] + ".gz"
        in librispeech_prepare.OPEN_SLR_11_NGRAM_MODELs
        and hparams["G_rescoring_arpa"] + ".gz"
        in librispeech_prepare.OPEN_SLR_11_NGRAM_MODELs
        and (
            hparams["compose_HL_with_G"]
            or hparams["decoding_method"] == "whole-lattice-rescoring"
        )
    ):
        librispeech_prepare.download_openslr_librispeech_lm(
            destination=hparams["lm_dir"],
            rescoring_lm=(
                hparams["decoding_method"] == "whole-lattice-rescoring"
            ),
        )
    # SB ngram models
    elif (
        "sb" in hparams["G_arpa"]
        and "sb" in hparams["G_rescoring_arpa"]
        and (
            hparams["compose_HL_with_G"]
            or hparams["decoding_method"] == "whole-lattice-rescoring"
        )
    ):
        librispeech_prepare.download_sb_librispeech_lm(
            destination=hparams["lm_dir"],
            rescoring_lm=(
                hparams["decoding_method"] == "whole-lattice-rescoring"
            ),
        )

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    lexicon = sbk2.lexicon.Lexicon(hparams["lang_dir"])
    graph_compiler = sbk2.graph_compiler.CtcGraphCompiler(
        lexicon,
        device=asr_brain.device,
    )

    decoding_params = {}
    for param_name in (
        "compose_HL_with_G",
        "lm_dir",
        "decoding_method",
        "caching",
        "G_arpa",
        "G_rescoring_arpa",
        "lang_dir",
        "output_folder",
        "rescoring_lm_scale",
    ):
        if param_name in hparams:
            decoding_params[param_name] = hparams[param_name]

    decoder = sbk2.lattice_decoder.get_decoding(
        decoding_params, graph_compiler, device=asr_brain.device
    )

    # Add attributes to asr_brain
    setattr(asr_brain, "lexicon", lexicon)
    setattr(asr_brain, "graph_compiler", graph_compiler)
    setattr(asr_brain, "decoder", decoder)

    # We load the pretrained wav2vec2 model
    if "pretrainer" in hparams.keys():
        hparams["pretrainer"].collect_files()
        hparams["pretrainer"].load_collected(asr_brain.device)

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Testing
    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        wer_dir = os.path.join(hparams["output_wer_folder"], f"metric_{k}")
        os.makedirs(wer_dir, exist_ok=True)
        exp = "HLG" if hparams["compose_HL_with_G"] else "HL"
        asr_brain.hparams.wer_file = os.path.join(wer_dir, f"wer_{exp}")
        asr_brain.evaluate(
            test_datasets[k],
            test_loader_kwargs=hparams["test_dataloader_opts"],
            min_key="WER",
        )
