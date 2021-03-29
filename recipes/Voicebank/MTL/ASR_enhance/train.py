#!/usr/bin/env python3
"""Recipe for multi-task learning, using seq2seq and enhancement objectives.

To run this recipe, do the following:
> python train.py hparams/{config file} --data_folder /path/to/noisy-vctk

There's three provided files for three stages of training:
> python train.py hparams/pretrain_perceptual.yaml
> python train.py hparams/enhance_mimic.yaml
> python train.py hparams/robust_asr.yaml

Use your own hyperparameter file or the provided files.
The different losses can be turned on and off, and pre-trained models
can be used for enhancement or ASR models.

Authors
 * Peter Plantinga 2020, 2021
"""
import os
import sys
import torch
import torchaudio
import urllib.parse
import speechbrain as sb
from pesq import pesq
from pystoi import stoi
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.data_utils import download_file, undo_padding
from speechbrain.utils.distributed import run_on_main


def pesq_eval(pred_wav, target_wav):
    return pesq(
        fs=16000, ref=target_wav.numpy(), deg=pred_wav.numpy(), mode="wb",
    )


def estoi_eval(pred_wav, target_wav):
    return stoi(
        x=target_wav.numpy(), y=pred_wav.numpy(), fs_sig=16000, extended=True
    )


# Define training procedure
class MTLbrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """The forward pass computes enhanced feats and targets"""

        batch = batch.to(self.device)
        self.stage = stage

        predictions = {}
        if self.hparams.enhance_type is not None:
            phase_wavs, noisy_feats, lens = self.prepare_feats(batch.noisy_sig)

            # Mask with "signal approximation (SA)"
            if self.hparams.enhance_type == "masking":
                mask = self.modules.enhance_model(noisy_feats)
                m = self.hparams.mask_weight
                predictions["feats"] = m * torch.mul(mask, noisy_feats)
                predictions["feats"] += (1 - m) * noisy_feats
            elif self.hparams.enhance_type == "mapping":
                predictions["feats"] = self.modules.enhance_model(noisy_feats)
            elif self.hparams.enhance_type == "noisy":
                predictions["feats"] = noisy_feats
            elif self.hparams.enhance_type == "clean":
                phase_wavs, predictions["feats"], lens = self.prepare_feats(
                    batch.clean_sig
                )

            # Resynthesize waveforms
            enhanced_mag = torch.expm1(predictions["feats"])
            predictions["wavs"] = self.hparams.resynth(enhanced_mag, phase_wavs)

        # Generate clean features for ASR pre-training
        if self.hparams.ctc_type == "clean" or self.hparams.seq_type == "clean":
            _, clean_feats, lens = self.prepare_feats(batch.clean_sig)

        # Compute seq outputs
        if self.hparams.seq_type is not None:

            # Prepare target inputs
            tokens, token_lens = self.prepare_targets(batch.tokens_bos)
            tokens = self.modules.tgt_embedding(tokens)

            if self.hparams.seq_type == "clean":
                embed = self.modules.src_embedding(clean_feats)
            if self.hparams.seq_type == "joint":
                asr_feats = predictions["wavs"]
                if stage == sb.Stage.TRAIN:
                    asr_feats = self.hparams.augment(asr_feats, lens)
                asr_feats = self.hparams.fbank(asr_feats)
                asr_feats = self.hparams.normalizer(asr_feats, lens)
                embed = self.modules.src_embedding(asr_feats)
            dec_out = self.modules.recognizer(tokens, embed, lens)
            out = self.modules.seq_output(dec_out[0])
            predictions["seq_pout"] = self.hparams.log_softmax(out)

            if self.hparams.ctc_type is not None:
                out = self.modules.ctc_output(embed)
                predictions["ctc_pout"] = self.hparams.log_softmax(out)

            if stage != sb.Stage.TRAIN:
                predictions["hyps"], _ = self.hparams.beam_searcher(
                    embed.detach(), lens
                )

        return predictions

    def prepare_feats(self, signal, augment=True):
        """Prepare log-magnitude spectral features expected by enhance model"""
        wavs, wav_lens = signal

        if self.stage == sb.Stage.TRAIN and hasattr(self.hparams, "env_corr"):
            if augment:
                wavs_noise = self.hparams.env_corr(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
            else:
                wavs = torch.cat([wavs, wavs], dim=0)
            wav_lens = torch.cat([wav_lens, wav_lens])

        feats = self.hparams.compute_stft(wavs)
        feats = self.hparams.spectral_magnitude(feats)
        feats = torch.log1p(feats)

        return wavs, feats, wav_lens

    def prepare_targets(self, tokens):
        """Prepare target by concatenating self if "env_corr" is used"""
        tokens, token_lens = tokens

        if self.stage == sb.Stage.TRAIN and hasattr(self.hparams, "env_corr"):
            tokens = torch.cat([tokens, tokens], dim=0)
            token_lens = torch.cat([token_lens, token_lens])

        return tokens, token_lens

    def compute_objectives(self, predictions, batch, stage):
        """Compute possibly several loss terms: enhance, mimic, ctc, seq"""

        # Do not augment targets
        clean_wavs, clean_feats, lens = self.prepare_feats(
            batch.clean_sig, augment=False
        )
        loss = 0

        # Compute enhancement loss
        if self.hparams.enhance_weight > 0:
            enhance_loss = self.hparams.enhance_loss(
                predictions["feats"], clean_feats, lens
            )
            loss += self.hparams.enhance_weight * enhance_loss

            if stage != sb.Stage.TRAIN:
                self.enh_metrics.append(
                    batch.id, predictions["feats"], clean_feats, lens
                )
                self.stoi_metrics.append(
                    ids=batch.id,
                    predict=predictions["wavs"],
                    target=clean_wavs,
                    lengths=lens,
                )
                self.pesq_metrics.append(
                    ids=batch.id,
                    predict=predictions["wavs"],
                    target=clean_wavs,
                    lengths=lens,
                )

                if hasattr(self.hparams, "enh_dir"):
                    abs_lens = lens * predictions["wavs"].size(1)
                    for i, uid in enumerate(batch.id):
                        length = int(abs_lens[i])
                        wav = predictions["wavs"][i, :length].unsqueeze(0)
                        path = os.path.join(self.hparams.enh_dir, uid + ".wav")
                        torchaudio.save(path, wav.cpu(), sample_rate=16000)

        # Compute mimic loss
        if self.hparams.mimic_weight > 0:
            clean_embed = self.modules.src_embedding.CNN(clean_feats)
            enh_embed = self.modules.src_embedding.CNN(predictions["feats"])
            mimic_loss = self.hparams.mimic_loss(enh_embed, clean_embed, lens)
            loss += self.hparams.mimic_weight * mimic_loss

            if stage != sb.Stage.TRAIN:
                self.mimic_metrics.append(
                    batch.id, enh_embed, clean_embed, lens
                )

        # Compute hard ASR loss
        if self.hparams.ctc_weight > 0 and (
            not hasattr(self.hparams, "ctc_epochs")
            or self.hparams.epoch_counter.current < self.hparams.ctc_epochs
        ):
            tokens, token_lens = self.prepare_targets(batch.tokens)
            ctc_loss = self.hparams.ctc_loss(
                predictions["ctc_pout"], tokens, lens, token_lens
            )
            loss += self.hparams.ctc_weight * ctc_loss

            if stage != sb.Stage.TRAIN and self.hparams.seq_weight == 0:
                predict = sb.decoders.ctc_greedy_decode(
                    predictions["ctc_pout"], lens, blank_id=-1
                )
                self.err_rate_metrics.append(
                    ids=batch.id,
                    predict=predict,
                    target=tokens,
                    target_len=token_lens,
                    ind2lab=self.hparams.ind2lab,
                )

        # Compute nll loss for seq2seq model
        if self.hparams.seq_weight > 0:

            tokens, token_lens = self.prepare_targets(batch.tokens_eos)
            seq_loss = self.hparams.seq_loss(
                predictions["seq_pout"], tokens, token_lens
            )
            loss += self.hparams.seq_weight * seq_loss

            if stage != sb.Stage.TRAIN:
                if hasattr(self.hparams, "asr_pretrained"):
                    pred_words = [
                        self.token_encoder.decode_ids(token_seq)
                        for token_seq in predictions["hyps"]
                    ]
                    target_words = [
                        self.token_encoder.decode_ids(token_seq)
                        for token_seq in undo_padding(*batch.tokens)
                    ]
                    self.err_rate_metrics.append(
                        batch.id, pred_words, target_words
                    )
                else:
                    self.err_rate_metrics.append(
                        ids=batch.id,
                        predict=predictions["hyps"],
                        target=tokens,
                        target_len=token_lens,
                        ind2lab=self.token_encoder.decode_ndim,
                    )

        return loss

    def on_stage_start(self, stage, epoch):
        if stage != sb.Stage.TRAIN:

            if self.hparams.enhance_weight > 0:
                self.enh_metrics = self.hparams.enhance_stats()
                self.stoi_metrics = self.hparams.estoi_stats()
                self.pesq_metrics = self.hparams.pesq_stats()

            if self.hparams.mimic_weight > 0:
                self.mimic_metrics = self.hparams.mimic_stats()

            if self.hparams.ctc_weight > 0 or self.hparams.seq_weight > 0:
                self.err_rate_metrics = self.hparams.err_rate_stats()

        # Freeze models before training
        else:
            for model in self.hparams.frozen_models:
                for p in self.modules[model].parameters():
                    if (
                        hasattr(self.hparams, "unfreeze_epoch")
                        and epoch >= self.hparams.unfreeze_epoch
                        and (
                            not hasattr(self.hparams, "unfrozen_models")
                            or model in self.hparams.unfrozen_models
                        )
                    ):
                        p.requires_grad = True
                    else:
                        p.requires_grad = False

    def on_stage_end(self, stage, stage_loss, epoch):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            stage_stats = {"loss": stage_loss}
            max_keys = []
            min_keys = []

            if self.hparams.enhance_weight > 0:
                stage_stats["enhance"] = self.enh_metrics.summarize("average")
                stage_stats["stoi"] = self.stoi_metrics.summarize("average")
                stage_stats["pesq"] = self.pesq_metrics.summarize("average")
                max_keys.extend(["pesq", "stoi"])

            if self.hparams.mimic_weight > 0:
                stage_stats["mimic"] = self.mimic_metrics.summarize("average")
                min_keys.append("mimic")

            if self.hparams.ctc_weight > 0 or self.hparams.seq_weight > 0:
                err_rate = self.err_rate_metrics.summarize("error_rate")
                err_rate_type = self.hparams.target_type + "ER"
                stage_stats[err_rate_type] = err_rate
                min_keys.append(err_rate_type)

        if stage == sb.Stage.VALID:
            stats_meta = {"epoch": epoch}
            if self.hparams.ctc_weight > 0 or self.hparams.seq_weight > 0:
                old_lr, new_lr = self.hparams.lr_annealing(epoch - 1)
                sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
                stats_meta["lr"] = old_lr

            self.hparams.train_logger.log_stats(
                stats_meta=stats_meta,
                train_stats={"loss": self.train_loss},
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta=stage_stats,
                max_keys=max_keys,
                min_keys=min_keys,
                num_to_keep=self.hparams.checkpoint_avg,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.stats_file, "w") as w:
                if self.hparams.enhance_weight > 0:
                    w.write("\nstoi stats:\n")
                    self.stoi_metrics.write_stats(w)
                    w.write("\npesq stats:\n")
                    self.pesq_metrics.write_stats(w)
                if self.hparams.mimic_weight > 0:
                    w.write("\nmimic stats:\n")
                    self.mimic_metrics.write_stats(w)
                if self.hparams.seq_weight > 0:
                    self.err_rate_metrics.write_stats(w)
                print("stats written to ", self.hparams.stats_file)

    def on_evaluate_start(self, max_key=None, min_key=None):
        self.checkpointer.recover_if_possible(max_key=max_key, min_key=min_key)
        checkpoints = self.checkpointer.find_checkpoints(
            max_key=max_key,
            min_key=min_key,
            max_num_checkpoints=self.hparams.checkpoint_avg,
        )
        for model in self.modules:
            if (
                model not in self.hparams.frozen_models
                or hasattr(self.hparams, "unfrozen_models")
                and model in self.hparams.unfrozen_models
            ):
                model_state_dict = sb.utils.checkpoints.average_checkpoints(
                    checkpoints, model
                )
                self.modules[model].load_state_dict(model_state_dict)


def dataio_prep(hparams, token_encoder):
    """Creates the datasets and their data processing pipelines"""

    # Define pipelines
    @sb.utils.data_pipeline.takes("noisy_wav", "clean_wav")
    @sb.utils.data_pipeline.provides("noisy_sig", "clean_sig")
    def audio_pipeline(noisy_wav, clean_wav):
        yield sb.dataio.dataio.read_audio(noisy_wav)
        yield sb.dataio.dataio.read_audio(clean_wav)

    token_keys = ["tokens_bos", "tokens_eos", "tokens"]

    @sb.utils.data_pipeline.takes(hparams["target_type"])
    @sb.utils.data_pipeline.provides("tokens_list", *[t for t in token_keys])
    def target_pipeline(target):
        if "asr_pretrained" in hparams:
            tokens_list = token_encoder.encode_as_ids(target)
            yield tokens_list
        else:
            tokens_list = target.strip().split()
            yield tokens_list
            tokens_list = token_encoder.encode_sequence(tokens_list)
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    # Create datasets
    data = {}
    for dataset in ["train", "valid", "test"]:
        data[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, target_pipeline],
            output_keys=["id", "noisy_sig", "clean_sig"] + token_keys,
        )
        if dataset != "train":
            data[dataset] = data[dataset].filtered_sorted(sort_key="length")

    # Sort train dataset and ensure it doesn't get un-sorted
    if hparams["sorting"] == "ascending" or hparams["sorting"] == "descending":
        data["train"] = data["train"].filtered_sorted(
            sort_key="length", reverse=hparams["sorting"] == "descending",
        )
        hparams["train_loader_options"]["shuffle"] = False
    elif hparams["sorting"] != "random":
        raise NotImplementedError(
            "Sorting must be random, ascending, or descending"
        )

    # Update token_encoder
    if "asr_pretrained" not in hparams:
        token_encoder.update_from_didataset(
            data["train"], output_key="tokens_list"
        )
        token_encoder.insert_bos_eos(
            bos_label="<eos-bos>",
            eos_label="<eos-bos>",
            bos_index=hparams["bos_index"],
        )

    return data


def download_to_dir(url, directory):
    """Parse filename from url and download to directory."""
    os.makedirs(directory, exist_ok=True)
    filename = os.path.basename(urllib.parse.urlparse(url).path)
    download_file(url, os.path.join(directory, filename))
    return os.path.join(directory, filename)


# Begin Recipe!
if __name__ == "__main__":

    # Download model yaml files so we can "!include" them
    for url in [
        "https://www.dropbox.com/s/e439h7oix9m7imn/perceptual_model.yaml?dl=1",
        "https://www.dropbox.com/s/jgkw8byufw5zmco/enhance_model.yaml?dl=1",
        "https://www.dropbox.com/s/wbu3i82urhxe3in/asr_model.yaml?dl=1",
    ]:
        download_to_dir(url, os.path.join("hparams", "models"))

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Prepare data
    from voicebank_prepare import prepare_voicebank  # noqa E402

    run_on_main(
        prepare_voicebank,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["data_folder"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Load pretrained models
    for model in ["asr", "enhance", "perceptual"]:
        pretrained = f"{model}_pretrained"
        if pretrained in hparams:
            # We download the model from HuggingFace (by default).
            run_on_main(hparams[pretrained].collect_files)
            hparams[pretrained].load_collected()

    # Switch encoder based on task
    if "asr_pretrained" in hparams:
        token_encoder = hparams["tokenizer"]
    else:
        token_encoder = sb.dataio.encoder.CTCTextEncoder()

    datasets = dataio_prep(hparams, token_encoder)

    # Initialize trainer
    mtl_brain = MTLbrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        run_opts=run_opts,
        hparams=hparams,
        checkpointer=hparams["checkpointer"],
    )
    mtl_brain.token_encoder = token_encoder

    # Fit dataset
    mtl_brain.fit(
        epoch_counter=mtl_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_loader_options"],
        valid_loader_kwargs=hparams["valid_loader_options"],
    )

    # Evaluate best checkpoint, using lowest or highest value on validation
    outdir = mtl_brain.hparams.output_folder
    for dset in ["valid", "test"]:
        mtl_brain.hparams.stats_file = os.path.join(outdir, f"{dset}_stats.txt")
        mtl_brain.evaluate(
            datasets[dset],
            max_key=hparams["eval_max_key"],
            min_key=hparams["eval_min_key"],
            test_loader_kwargs=hparams["test_loader_options"],
        )
