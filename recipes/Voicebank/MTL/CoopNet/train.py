#!/usr/bin/env python3
"""Recipe for multi-task learning, using CTC and enhancement objectives.

To run this recipe, do the following:
> python train.py hparams/{config file} --data_folder /path/to/noisy-vctk

There's three provided files for three stages of training:
> python train.py hparams/pretrain_asr_and_se.yaml
> python train.py hparams/pretrain_1layer.yaml
> python train.py hparams/train_3layer.yaml

Use your own hyperparameter file or the provided files.
The different losses can be turned on and off, and pre-trained models
can be used for enhancement or ASR models.

Authors
 * Sreeramadas Sai Aravind 2021
 * Émile Dimas 2021
 * Nicolas Duchêne 2021

"""
import os
import sys
from itertools import chain
import torch
import torchaudio
import urllib.parse
import speechbrain as sb
from pesq import pesq
from pystoi import stoi
from hyperpyyaml import load_hyperpyyaml
import ruamel.yaml
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.distributed import run_on_main

from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.sampler import DynamicBatchSampler
from speechbrain.dataio.batch import PaddedBatch


def pesq_eval(pred_wav, target_wav):
    """Evaluate PESQ using pesq library"""
    return pesq(
        fs=16000, ref=target_wav.numpy(), deg=pred_wav.numpy(), mode="wb",
    )


def estoi_eval(pred_wav, target_wav):
    """Evaluate stoi using pystoi library"""
    return stoi(
        x=target_wav.numpy(), y=pred_wav.numpy(), fs_sig=16000, extended=True
    )


def get_attr(hparams, key):
    """Get an attribute if it exists"""
    if hasattr(hparams, key):
        return hparams.__getattribute__(key)
    return False


def create_folder(folder):
    if folder is not None:
        if not os.path.isdir(folder):
            os.makedirs(folder)


# Define training procedure
class CoopNetBrain(sb.Brain):
    def fuse(self, noisy_feats, layer, predictions):
        """Create attention mask to feed to a TransformerSE module"""

        # Transform the ASR output (before CTC transformation)
        out_weights = self.hparams.model[f"fuse_{layer}"](
            predictions[f"asr_out_{layer}"]
        )
        # Create attention mask
        att_mask = torch.matmul(
            out_weights, noisy_feats.permute(0, 2, 1)
        ).repeat(self.hparams.n_heads, 1, 1)

        return att_mask

    def enhance_speech(self, batch, layer, predictions):
        """Enhancement module for a given layer"""
        phase_wavs, noisy_feats, _ = self.prepare_feats(
            batch.noisy_sig, augment=False
        )

        # Mask with "signal approximation (SA)" and use ASR info
        if self.hparams.enhance_type == "masking":
            if layer > 0:  # Feed info from previous layer if applicable
                noisy_feats = predictions[f"feats_{layer-1}"]
            # Make mask
            att_mask = self.fuse(noisy_feats, layer, predictions)
            mask = self.modules.model[f"se_{layer}"](
                noisy_feats, attn_mask=att_mask
            )
            m = self.hparams.mask_weight
            predictions[f"feats_{layer}"] = m * torch.mul(mask, noisy_feats)
            predictions[f"feats_{layer}"] += (1 - m) * noisy_feats
        # Mask with "signal approximation (SA)" using clean signal
        elif self.hparams.enhance_type == "clean":
            phase_wavs, clean_feats, lens = self.prepare_feats(batch.clean_sig)

            if layer > 0:  # Feed info from previous layer
                clean_feats = predictions[f"feats_{layer-1}"]

            mask = self.modules.model[f"se_{layer}"](clean_feats)
            m = self.hparams.mask_weight
            predictions[f"feats_{layer}"] = m * torch.mul(mask, clean_feats)
            predictions[f"feats_{layer}"] += (1 - m) * clean_feats

        elif self.hparams.enhance_type == "noisy":
            phase_wavs, noisy_feats, lens = self.prepare_feats(batch.noisy_sig)

            if layer > 0:  # Feed info from previous layer
                noisy_feats = predictions[f"feats_{layer-1}"]

            mask = self.modules.model[f"se_{layer}"](noisy_feats)
            m = self.hparams.mask_weight
            predictions[f"feats_{layer}"] = m * torch.mul(mask, noisy_feats)
            predictions[f"feats_{layer}"] += (1 - m) * noisy_feats

        # Resynthesize waveforms
        enhanced_mag = torch.expm1(predictions[f"feats_{layer}"])
        predictions[f"wavs_{layer}"] = self.hparams.resynth(
            enhanced_mag, phase_wavs
        )

        return predictions

    def predict_phonemes(self, batch, layer, predictions):
        """ASR module for a given layer using CTC loss"""

        if self.hparams.ctc_type == "clean":
            wavs, _, wav_lens = self.prepare_feats(batch.clean_sig)
            asr_feats = wavs if layer == 0 else predictions[f"wavs_{layer-1}"]
        elif self.hparams.ctc_type == "joint":
            wavs, wav_lens = batch.noisy_sig
            asr_feats = wavs if layer == 0 else predictions[f"wavs_{layer-1}"]
        asr_feats = self.hparams.fbank(asr_feats)
        asr_feats = self.hparams.normalizer(asr_feats, wav_lens)
        predictions[f"asr_out_{layer}"] = self.modules.model[
            f"asr_{layer}"
        ].asr(asr_feats)

        out = self.modules.model[f"asr_{layer}"].ctc(
            predictions[f"asr_out_{layer}"]
        )
        predictions[f"ctc_pout_{layer}"] = self.hparams.log_softmax(out)

        return predictions

    def layer_forward(self, batch, stage, layer=0, predictions={}):
        """Pass through one layer of the network."""

        if self.hparams.ctc_type is not None:
            predictions = self.predict_phonemes(batch, layer, predictions)

        if self.hparams.enhance_type is not None:
            predictions = self.enhance_speech(batch, layer, predictions)

        return predictions

    def compute_forward(self, batch, stage):
        """The forward pass iterates over the layers of the model,
        accumulating enhancment and predictions at each layer.
        """
        batch = batch.to(self.device)
        self.stage = stage

        predictions = self.layer_forward(batch, stage)
        for layer in range(1, self.hparams.n_layers):
            predictions = self.layer_forward(batch, stage, layer, predictions)

        return predictions

    def prepare_feats(self, signal, augment=True, concat=False):
        """Prepare log-magnitude spectral features expected by enhance model."""
        wavs, wav_lens = signal

        if hasattr(self.hparams, "env_corr"):
            if augment:
                wavs_noise = self.hparams.env_corr(wavs, wav_lens)
                wavs = wavs_noise
                if concat:
                    torch.cat([wavs, wavs_noise], dim=0)
            else:
                if concat:
                    wavs = torch.cat([wavs, wavs], dim=0)

            if concat:
                wav_lens = torch.cat([wav_lens, wav_lens])

        feats = self.hparams.compute_stft(wavs)
        feats = self.hparams.spectral_magnitude(feats)
        feats = torch.log1p(feats)

        return wavs, feats, wav_lens

    def prepare_targets(self, tokens, concat=False):
        """Prepare target by concatenating self if "env_corr" is used"""
        tokens, token_lens = tokens

        if concat and hasattr(self.hparams, "env_corr"):
            tokens = torch.cat([tokens, tokens], dim=0)
            token_lens = torch.cat([token_lens, token_lens])

        return tokens, token_lens

    def layer_objectives(
        self,
        predictions,
        batch,
        stage,
        clean_wavs,
        clean_feats,
        lens,
        loss=0,
        layer=0,
    ):
        """Compute possibly several loss terms for one layer: enhance, ctc, phnER."""
        # Compute enhancement loss
        if self.hparams.enhance_type is not None:
            enhance_loss = self.hparams.enhance_loss(
                predictions[f"feats_{layer}"], clean_feats, lens
            )
            loss += self.hparams.enhance_weight[layer] * enhance_loss

            if stage != sb.Stage.TRAIN:
                self.metrics[f"enh_metrics_{layer}"].append(
                    batch.id, predictions[f"feats_{layer}"], clean_feats, lens
                )
                self.metrics[f"stoi_metrics_{layer}"].append(
                    ids=batch.id,
                    predict=predictions[f"wavs_{layer}"],
                    target=clean_wavs,
                    lengths=lens,
                )
                self.metrics[f"pesq_metrics_{layer}"].append(
                    ids=batch.id,
                    predict=predictions[f"wavs_{layer}"],
                    target=clean_wavs,
                    lengths=lens,
                )

                if hasattr(self.hparams, "enh_dir"):
                    abs_lens = lens * predictions[f"wavs_{layer}"].size(1)
                    for i, uid in enumerate(batch.id):
                        length = int(abs_lens[i])
                        wav = predictions[f"wavs_{layer}"][
                            i, :length
                        ].unsqueeze(0)
                        path = os.path.join(self.hparams.enh_dir, uid + ".wav")
                        torchaudio.save(path, wav.cpu(), sample_rate=16000)

        # Compute hard ASR loss
        if self.hparams.ctc_type is not None:
            tokens, token_lens = self.prepare_targets(batch.phn_encoded)
            if (
                not hasattr(self.hparams, "ctc_epochs")
                or self.hparams.epoch_counter.current <= self.hparams.ctc_epochs
            ):
                ctc_loss = self.hparams.ctc_loss(
                    predictions[f"ctc_pout_{layer}"], tokens, lens, token_lens
                )
                loss += self.hparams.ctc_weight[layer] * ctc_loss

            if stage != sb.Stage.TRAIN:
                predict = sb.decoders.ctc_greedy_decode(
                    predictions[f"ctc_pout_{layer}"],
                    lens,
                    blank_id=self.hparams.blank_index,
                )
                self.metrics[f"err_rate_metrics_{layer}"].append(
                    ids=batch.id,
                    predict=predict,
                    target=tokens,
                    target_len=token_lens,
                    ind2lab=self.label_encoder.decode_ndim,
                )

        return loss

    def compute_objectives(self, predictions, batch, stage):
        """Calculate loss terms and accumulate them across the layers."""

        # Do not augment targets
        clean_wavs, clean_feats, lens = self.prepare_feats(
            batch.clean_sig, augment=False
        )
        loss = 0
        for n in range(self.hparams.n_layers):
            loss = self.layer_objectives(
                predictions,
                batch,
                stage,
                clean_wavs,
                clean_feats,
                lens,
                loss=loss,
                layer=n,
            )
        return loss

    def init_optimizers(self):
        """Called during ``on_fit_start()``, initialize optimizers
        after parameters are fully configured (e.g. DDP, jit).

        This was rewritten to initialize the multiple optimizers used for this recipe.

        See CoopNet.py for the MultipleOptimizer class.
        """

        if self.opt_class is not None:
            if get_attr(self.hparams, "multi_optim"):
                model = self.modules.model
                # Retrieve proper parameters for each optimizer
                se_parameters = chain.from_iterable(
                    [model[k].parameters() for k in model.keys() if "se" in k]
                )
                asr_parameters = chain.from_iterable(
                    [model[k].parameters() for k in model.keys() if "asr" in k]
                    + [  # Add fuse layers to ASR optimizer
                        model[k].parameters()
                        for k in model.keys()
                        if "attention" in k
                    ]
                )
                params = {
                    "se_opt": se_parameters,
                    "asr_opt": asr_parameters,
                }

                self.optimizer = self.opt_class(params=params)
                if self.checkpointer is not None:
                    for k, opt in self.optimizer.optimizers.items():
                        self.checkpointer.add_recoverable(f"optimizer_{k}", opt)

            else:
                self.optimizer = self.opt_class(self.modules.parameters())

                if self.checkpointer is not None:
                    self.checkpointer.add_recoverable(
                        "optimizer", self.optimizer
                    )

    def on_stage_start(self, stage, epoch):
        if stage != sb.Stage.TRAIN:
            self.metrics = {}
            for layer in range(self.hparams.n_layers):
                self.metrics[
                    f"enh_metrics_{layer}"
                ] = self.hparams.enhance_stats()
                self.metrics[
                    f"stoi_metrics_{layer}"
                ] = self.hparams.estoi_stats()
                self.metrics[
                    f"pesq_metrics_{layer}"
                ] = self.hparams.pesq_stats()

                self.metrics[
                    f"err_rate_metrics_{layer}"
                ] = self.hparams.err_rate_stats()

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

            for layer in range(self.hparams.n_layers):
                if self.hparams.enhance_type is not None:
                    stage_stats[f"enhance_{layer}"] = self.metrics[
                        f"enh_metrics_{layer}"
                    ].summarize("average")
                    stage_stats[f"stoi_{layer}"] = self.metrics[
                        f"stoi_metrics_{layer}"
                    ].summarize("average")
                    stage_stats[f"pesq_{layer}"] = self.metrics[
                        f"pesq_metrics_{layer}"
                    ].summarize("average")
                    max_keys.extend(["pesq", "stoi"])

                if self.hparams.ctc_type is not None:
                    err_rate = self.metrics[
                        f"err_rate_metrics_{layer}"
                    ].summarize("error_rate")
                    err_rate_type = self.hparams.target_type + "ER"
                    stage_stats[err_rate_type + f"_{layer}"] = err_rate
                    min_keys.append(err_rate_type)

        if stage == sb.Stage.VALID:
            stats_meta = {"epoch": epoch}
            old_lr, new_lr = self.hparams.lr_annealing(epoch - 1)
            if get_attr(self.hparams, "multi_optim"):
                self.hparams.update_learning_rate(self.optimizer, new_lr)
            else:
                sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            stats_meta["lr"] = old_lr

            train_stats = {"loss": self.train_loss}
            self.hparams.train_logger.log_stats(
                stats_meta=stats_meta,
                train_stats=train_stats,
                valid_stats=stage_stats,
            )

            # Send sweep metric for bayesian optimization
            if get_attr(self.hparams, "sweep"):
                self.hparams.train_logger.run.log(
                    {
                        "final_per": stage_stats[
                            f"phnER_{self.hparams.n_layers-1}"
                        ]
                    },
                    step=epoch,
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
                for layer in range(self.hparams.n_layers):
                    w.write("\nstoi stats:\n")
                    self.metrics[f"enh_metrics_{layer}"].write_stats(w)
                    w.write("\npesq stats:\n")
                    self.metrics[f"pesq_metrics_{layer}"].write_stats(w)
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

    def load_pretrained(self):
        """Loads the specified pretrained files.
        Necessary for colab
        because pretrainer class
        uses symbolic links that do not work in colab."""

        for model_name, model_path in self.hparams.pretrained_path.items():

            # Try parsing model_path as a url first.
            try:
                print("trying to download " + model_path)
                save_dir = os.path.join(self.hparams.output_folder, "save")
                model_path = download_to_dir(model_path, save_dir)

            # If it fails, assume its a valid filepath already
            except ValueError:
                pass

            if model_name == "normalizer":
                self.hparams.normalizer._load(
                    model_path, end_of_epoch=False, device=self.device
                )
            else:
                state_dict = torch.load(model_path)
                self.modules[model_name].load_state_dict(
                    state_dict, strict=False
                )


def dataio_prep(hparams):
    """Creates the datasets and their data processing pipelines"""
    # 1. Get label encoder
    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    # 2. Define audio pipelines:
    @sb.utils.data_pipeline.takes("noisy_wav")
    @sb.utils.data_pipeline.provides("noisy_sig")
    def noisy_pipeline(wav):
        return sb.dataio.dataio.read_audio(wav)

    @sb.utils.data_pipeline.takes("clean_wav")
    @sb.utils.data_pipeline.provides("clean_sig")
    def clean_pipeline(wav):
        return sb.dataio.dataio.read_audio(wav)

    # 3. Define target pipeline:
    # @sb.utils.data_pipeline.takes("phn")
    @sb.utils.data_pipeline.takes("phones")
    @sb.utils.data_pipeline.provides("phn_list", "phn_encoded")
    def target_pipeline(target):
        phn_list = target.strip().split()
        yield phn_list
        phn_encoded = label_encoder.encode_sequence_torch(phn_list)
        yield phn_encoded

    # 4. Create datasets
    data = {}
    for dataset in ["train", "valid", "test"]:
        data[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[noisy_pipeline, clean_pipeline, target_pipeline],
            output_keys=["id", "noisy_sig", "clean_sig", "phn_encoded"],
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

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[data["train"]],
        output_key="phn_list",
        special_labels={"blank_label": hparams["blank_index"]},
        sequence_input=True,
    )

    if hparams["dynamic_batching"]:
        dynamic_hparams = hparams["dynamic_batch_sampler"]
        hope_size = dynamic_hparams["feats_hop_size"]
        for dataset in ["train", "valid", "test"]:

            batch_sampler = DynamicBatchSampler(
                data[dataset],
                dynamic_hparams["max_batch_len"],
                dynamic_hparams["left_bucket_len"],
                bucket_length_multiplier=dynamic_hparams["multiplier"],
                length_func=lambda x: x["length"] * (1 / hope_size),
                shuffle=dynamic_hparams["shuffle_ex"],
                # batch_ordering=dynamic_hparams["batch_ordering"],
            )

            data[dataset] = SaveableDataLoader(
                data[dataset],
                batch_sampler=batch_sampler,
                collate_fn=PaddedBatch,
            )

    return data, label_encoder


def download_to_dir(url, directory):
    """Parse filename from url and download to directory."""
    os.makedirs(directory, exist_ok=True)
    filename = os.path.basename(urllib.parse.urlparse(url).path)
    download_file(url, os.path.join(directory, filename))
    return os.path.join(directory, filename)


def change_yaml_values(input, new_config):
    yaml = ruamel.yaml.YAML()
    with open(input, "r") as fp:
        data = yaml.load(fp)
    for k, v in new_config.items():
        data[k] = v
    with open(input, "w") as fp:
        yaml.dump(data, fp)


# Begin Recipe!
if __name__ == "__main__":

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

    create_folder(hparams.get("enh_dir", None))
    datasets, label_encoder = dataio_prep(hparams)

    # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    if hparams.get("pretrainer", False):
        run_on_main(hparams["pretrainer"].collect_files)
        hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Initialize trainer
    CoopNetbrain = CoopNetBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        run_opts=run_opts,
        hparams=hparams,
        checkpointer=hparams["checkpointer"],
    )
    CoopNetbrain.label_encoder = label_encoder
    if hparams.get("pretrained_path", False):
        CoopNetbrain.load_pretrained()

    # Fit dataset
    CoopNetbrain.fit(
        epoch_counter=CoopNetbrain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_loader_options"],
        valid_loader_kwargs=hparams["valid_loader_options"],
    )

    # Evaluate best checkpoint, using lowest or highest value on validation
    outdir = CoopNetbrain.hparams.output_folder
    CoopNetbrain.hparams.stats_file = os.path.join(outdir, "valid_stats.txt")
    CoopNetbrain.evaluate(
        datasets["valid"],
        max_key=hparams["eval_max_key"],
        min_key=hparams["eval_min_key"],
        test_loader_kwargs=hparams["test_loader_options"],
    )
    CoopNetbrain.hparams.stats_file = os.path.join(outdir, "test_stats.txt")
    CoopNetbrain.evaluate(
        datasets["test"],
        max_key=hparams["eval_max_key"],
        min_key=hparams["eval_min_key"],
        test_loader_kwargs=hparams["test_loader_options"],
    )
