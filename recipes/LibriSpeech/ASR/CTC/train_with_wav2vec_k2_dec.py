#!/usr/bin/env/python3
"""Recipe for training a wav2vec-based ctc ASR system with librispeech.
The system employs wav2vec as its encoder. Decoding is performed with
k2's wfst-based decoder.
To run this recipe, do the following:
> python train_with_wav2vec_k2_dec.py hparams/train_hf_wav2vec_k2_dec.yaml
The neural network is trained on CTC likelihood target and character units
are used as basic recognition tokens.

Authors
 * Georgios Karakasidis 2023
 * Zeyu Zhao 2023
 * Rudolf A Braun 2022
 * Titouan Parcollet 2022
 * Sung-Lin Yeh 2021
 * Ju-Chieh Chou 2020
 * Mirco Ravanelli 2020
 * Abdel Heba 2020
 * Peter Plantinga 2020
 * Samuele Cornell 2020
"""

import collections
import sys
import os
import logging
from typing import List, Optional, Dict, Tuple
from pathlib import Path

import torch
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
from torch.utils.data import DataLoader
from icefall.decode import get_lattice, rescore_with_whole_lattice, rescore_with_n_best_list, one_best_decoding
from icefall.utils import get_texts


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
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])

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
        p_tokens = None
        logits = self.modules.ctc_lin(x)

        # Upsample the inputs if they have been highly downsampled
        if hasattr(self.hparams, "upsampling") and self.hparams.upsampling:
            logits = logits.view(
                logits.shape[0], -1, self.hparams.output_neurons
            )

        p_ctc = self.hparams.log_softmax(logits)
        if stage != sb.Stage.TRAIN:
            p_tokens = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
        return p_ctc, wav_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        p_ctc, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens, tokens_lens = batch.tokens  # TODO: Make sure the device is right

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens = torch.cat([tokens, tokens], dim=0)
            tokens_lens = torch.cat([tokens_lens, tokens_lens], dim=0)

        # TODO: Use k2.ctc_loss
        loss_ctc = self.hparams.ctc_cost(
            p_ctc, tokens, wav_lens, tokens_lens,
            blank_index=self.hparams.blank_index
        )
        loss = loss_ctc

        if stage == sb.Stage.VALID:
            # Decode token terms to words
            predicted_words = [
                # "".join(self.tokenizer.sp.decode_ids(utt_seq)).split(" ")
                "".join(self.tokenizer.decode_ndim(utt_seq)).split(" ")
                for utt_seq in predicted_tokens
            ]
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            self.wer_metric[0].append(ids, predicted_words, target_words)
            self.cer_metric[0].append(ids, predicted_words, target_words)
        if stage == sb.Stage.TEST:  # Language model decoding only used for test
            prediction_dict: dict = self.decode_batch(  # TODO: define decode_batch
                batch, p_ctc=p_ctc
            )
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            if self.hparams.decoding_method == "1best":
                self.wer_metrics[1].append(batch["ids"], list(prediction_dict.values())[0], target_words)
            else:
                for i, lm_scale in enumerate(self.lm_scale_list):
                    key = f"lm_scale_{lm_scale:.1f}"
                    # index 0 is for the validation set's greedy decoding 
                    self.wer_metrics[i+1].append(ids, prediction_dict[key], target_words)
                    self.cer_metrics[i+1].append(ids, prediction_dict[key], target_words)
            # Do greedy decoding for the test set, similar to the validation set
            predicted_words = [
                "".join(self.tokenizer.decode_ndim(utt_seq)).split(" ")
                # self.tokenizer.sp.decode_ids(utt_seq).split(" ")
                for utt_seq in predicted_tokens
            ]
            self.wer_metrics[-1].append(ids, predicted_words, target_words)
        return loss
    
    def get_supervision_segments(
            self, 
            n_frames: torch.Tensor, 
            seq_ids: List[int], 
            tokens: Optional[List[str]] = None
        ):
        """ Build the supervision segments which are required for building the FSA.
            NOTE: We assume that the audio does not contain segments and that all 
                  utterances start at duration 0.
            Args:
                n_frames: tensor of number of frames in each input segment
                seq_ids: a list of the sequence ids (starting from 0, up to batch size)
                tokens: A list of the transcriptions (reordered according to how the 
                        supervision_segments are sorted). If not provided, the function
                        will return only the supervision_segments.
        """
        supervision_segments = torch.stack(
            (
                seq_ids.to(self.device),
                torch.zeros(len(seq_ids), dtype=torch.int32, device=self.device),
                torch.div(
                    n_frames.to(self.device),
                    self.subsampling_factor,
                    rounding_mode="floor",
                ),
            ),
            1
        ).to(torch.int32)

        # Sort based on duration (longest to shortest) -> required by k2.DenseFsaVec
        indices = torch.argsort(supervision_segments[:, 2], descending=True)
        supervision_segments = supervision_segments[indices].to("cpu")
        if tokens is None:
            return supervision_segments
        tokens = [tokens[i] for i in indices]
        return supervision_segments, tokens

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
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metrics = []
            self.wer_metrics = []
            self.metrics_log_dir = Path(self.hparams.output_folder) / "metrics"
            os.makedirs(
                self.metrics_log_dir,
                exist_ok=True
            )
            if self.hparams.decoding_method == "1best":
                n_metrics_needs = 3
            else:
                n_metrics_needs = len(self.lm_scale_list) + 2    
            # +1 for greedy decoding in valid
            # +1 for greedy decoding in test
            for _ in range(n_metrics_needs):
                self.cer_metrics.append(self.hparams.cer_computer())
                self.wer_metrics.append(self.hparams.error_rate_computer())

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        elif stage == sb.Stage.VALID:
            stage_stats["valid WER"] = self.wer_metrics[0].summarize("error_rate")
            stage_stats["valid CER"] = self.cer_metrics[0].summarize("error_rate")
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
                meta={
                    "WER": stage_stats["valid WER"],
                    "CER": stage_stats["valid CER"],
                },
                min_keys=["valid WER"],
            )
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        elif stage == sb.Stage.TEST:
            if self.hparams.decoding_method == "1best":
                stage_stats["WER"] = self.wer_metrics[1].summarize("error_rate")
            else:
                best_wer, best_lm_scale_wer = 2000, ""
                best_cer, best_lm_scale_cer = 2000, ""
                for lm_scale, cer_metric, wer_metric in zip(
                    self.lm_scale_list, self.cer_metrics[1:-1], self.wer_metrics[1:-1]
                ):
                    wer = wer_metric.summarize(
                        "error_rate"
                    )
                    # stage_stats[f"WER_lm_scale_{lm_scale:.1f}"] = wer
                    cer = cer_metric.summarize(
                        "error_rate"
                    )
                    # stage_stats[f"CER_lm_scale_{lm_scale:.1f}"] = cer
                    if wer < best_wer:
                        best_wer = wer
                        best_lm_scale_wer = lm_scale
                    if cer < best_cer:
                        best_cer = cer
                        best_lm_scale_cer = lm_scale
                stage_stats["best_wer"] = best_wer
                stage_stats["best_cer"] = best_cer
                stage_stats["best_lm_scale_wer"] = best_lm_scale_wer
                stage_stats["best_lm_scale_cer"] = best_lm_scale_cer
            stage_stats["greedy_WER"] = self.wer_metrics[-1].summarize("error_rate")
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            # To allow multiple test set names and not override
            wer_pattern = getattr(self, "wer_file_pattern", "wer")
            if self.hparams.decoding_method == "1best":
                with open(self.metrics_log_dir / "{}_test_1best.txt".format(wer_pattern), "w") as w:
                    self.wer_metrics[1].write_stats(w)
            else:
                for lm_scale, wer_metric in zip(
                    self.lm_scale_list, self.wer_metrics[1:-1]
                ):
                    with open(self.metrics_log_dir / "{}_test_lm_scale={}.txt".format(wer_pattern, lm_scale), "w") as w:
                        wer_metric.write_stats(w)
            with open(self.metrics_log_dir / "{}_greedy.txt".format(wer_pattern), "w") as w:
                self.wer_metrics[-1].write_stats(w)

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

    @property
    def HLG(self):
        HLG = self.topology.HLG
        if not hasattr(HLG, "lm_scores"):
            HLG.lm_scores = HLG.scores.clone()
        return HLG

    @property
    def G_4gram(self):
        if self.hparams.decoding_method in ['nbest-rescoring', 'whole-lattice-rescoring']:
            return self.topology.G_4gram
        return None

    def decode(self, testset: DataLoader) -> Dict[str, List[Tuple[int, List[str], List[str]]]]:
        results = collections.defaultdict(list)
        for batch in testset:
            texts = batch["wrd"]
            ids = batch["ids"]
            hyps_dict = self.decode_batch(batch)
            for lm_scale, hyps in hyps_dict.items():
                this_batch = []
                assert len(hyps) == len(texts)
                for i, hyp_words, ref_text in zip(ids, hyps, texts):
                    ref_words = ref_text.split()
                    this_batch.append((i, ref_words, hyp_words))

                results[lm_scale].extend(this_batch)

        return results
        
    def decode_batch(self, batch, p_ctc=None) -> Dict[str, List[List[str]]]:
        """Decode a single batch assuming no forward pass has been done."""
        if p_ctc is None:
            with torch.no_grad():
                # Use train as the stage so that it won't try to do decoding
                p_ctc, _ = self.compute_forward(batch, sb.Stage.TRAIN)
        # print(p_ctc.shape)
        # print(p_ctc.topk(2))
        # p_ctc[:, :, 0] -= 20000
        # print(p_ctc.shape)
        # print(p_ctc.topk(2))
        # print("="*100)
        return self._k2_decode_from_probs(batch, p_ctc)
    
    def _k2_decode_from_probs(
        self,
        batch,
        p_ctc,
        decoding_method: Optional[str] = None
    ) -> Dict[str, List[List[str]]]:
        """Decode using k2 library."""
        # TODO: this won't work
        supervision_segments = batch["supervision_segments"]
        decoding_method = decoding_method or self.hparams.decoding_method

        # logger.info("Creating lattice...")
        try:
            lattice = get_lattice(
                nnet_output=p_ctc,
                decoding_graph=self.HLG,
                supervision_segments=supervision_segments,
                search_beam=self.hparams.search_beam,
                output_beam=self.hparams.output_beam,
                min_active_states=self.hparams.min_active_states,
                max_active_states=self.hparams.max_active_states,
                subsampling_factor=self.subsampling_factor,
            )
        except RuntimeError as e:
            msg = (
                "RuntimeError caught during lattice creation. "
                "This may be caused by a bad choice of beam size. "
                "Try to decrease the beam size."
            )
            msg += f"\n{supervision_segments=}\n\n{p_ctc.shape=}\n\n{self.HLG=}\n\n"
            raise RuntimeError(msg) from e

        # logger.info("Done creating lattice. Moving on to rescoring...")


        if decoding_method in ["1best", "nbest"]:
            if decoding_method == "1best":
                best_path = one_best_decoding(
                    lattice=lattice, use_double_scores=True
                )
                key = "no_rescore"
            # else:
            #     best_path = nbest_decoding(
            #         lattice=lattice,
            #         num_paths=params.num_paths,
            #         use_double_scores=params.use_double_scores,
            #         nbest_scale=params.nbest_scale,
            #     )
            #     key = f"no_rescore-{params.num_paths}"
            hyps = get_texts(best_path)
            hyps = [[self.lexicon.word_table[i] for i in ids] for ids in hyps]
            return {key: hyps}

        elif decoding_method == 'nbest-rescoring':
            best_path_dict = rescore_with_n_best_list(
                lattice=lattice,
                G=self.G_4gram,
                num_paths=100,
                lm_scale_list=self.lm_scale_list,
                nbest_scale=0.5  # scale for lattice.scores
            )
        elif decoding_method == 'whole-lattice-rescoring':
            best_path_dict = rescore_with_whole_lattice(
                lattice=lattice,
                G_with_epsilon_loops=self.G_4gram,
                lm_scale_list=self.lm_scale_list
            )
        else:
            raise NotImplementedError(f"Decoding method {decoding_method} not implemented.")
        
        ans = {}
        for lm_scale_str, best_path in best_path_dict.items():
            hyps = get_texts(best_path)
            hyps = [[self.lexicon.word_table[i] for i in ids] for ids in hyps]  # TODO: word_table[idx] is wrong
            ans[lm_scale_str] = hyps
        return ans


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

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    label_encoder = hparams["label_encoder"]

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "char_list", "tokens_list", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        char_list = list(wrd)
        yield char_list
        tokens_list = label_encoder.encode_sequence(char_list)
        yield tokens_list
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    special_labels = {
        "blank_label": hparams["blank_index"],
    }
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="char_list",
        special_labels=special_labels,
        sequence_input=True,
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "char_list", "tokens"],
    )

    return train_data, valid_data, test_datasets, label_encoder


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # If distributed_launch=True then
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
    train_data, valid_data, test_datasets, label_encoder = dataio_prepare(
        hparams
    )

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
        hparams["pretrainer"].load_collected(asr_brain.device)

    # We dynamicaly add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for the LM!!
    asr_brain.tokenizer = label_encoder

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
        asr_brain.hparams.wer_file = os.path.join(
            hparams["output_folder"], "wer_{}.txt".format(k)
        )
        asr_brain.evaluate(
            test_datasets[k], test_loader_kwargs=hparams["test_dataloader_opts"]
        )
