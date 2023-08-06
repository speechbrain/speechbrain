"""
 Recipe for training the speech-to-unit translation (S2UT) model, the implementation is based on the following papers:
 - Direct speech-to-speech translation with discrete units: (https://arxiv.org/abs/2006.04558)
 - Enhanced Direct Speech-to-Speech Translation Using Self-supervised Pre-training and Data Augmentation: (https://arxiv.org/abs/2204.02967)
 To run this recipe, do the following:
 # python train.py hparams/train.yaml
 Authors
 * Jarod Duret 2023
"""

import sys
import torch
import logging
import pathlib as pl
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.pretrained import UnitHIFIGAN, EncoderDecoderASR
import tqdm
import torchaudio
import numpy as np
from torch.nn.parallel import DistributedDataParallel

logger = logging.getLogger(__name__)


# Define training procedure
class S2UT(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Computes the forward pass
        Arguments
        ---------
        batch: str
            a single batch
        stage: speechbrain.Stage
            the training stage
        Returns
        -------
        the model output
        """
        batch = batch.to(self.device)
        wavs, wav_lens = batch.src_sig
        tokens_bos, _ = batch.code_bos

        # Use default padding value for wav2vec2
        wavs[wavs == self.hparams.pad_index] = 0.0

        # compute features
        enc_out = self.modules.wav2vec2(wavs, wav_lens)

        # dimensionality reduction
        enc_out = self.modules.enc(enc_out)

        if isinstance(self.modules.transformer, DistributedDataParallel):
            dec_out = self.modules.transformer.module.forward_mt_decoder_only(
                enc_out, tokens_bos, pad_idx=self.hparams.pad_index
            )
        else:
            dec_out = self.modules.transformer.forward_mt_decoder_only(
                enc_out, tokens_bos, pad_idx=self.hparams.pad_index
            )

        # logits and softmax
        pred = self.modules.seq_lin(dec_out)
        p_seq = self.hparams.log_softmax(pred)

        hyps = None
        if stage == sb.Stage.VALID:
            current_epoch = self.hparams.epoch_counter.current
            output_progress_sample = (
                self.hparams.progress_samples
                and current_epoch % self.hparams.progress_samples_interval == 0
            )
            if output_progress_sample:
                hyps, _ = self.hparams.valid_search(enc_out.detach(), wav_lens)

        elif stage == sb.Stage.TEST:
            ids = batch.id
            tgt_text = batch.tgt_text

            hyps, _ = self.hparams.test_search(enc_out.detach(), wav_lens)
            code = torch.LongTensor(hyps[0])
            wav = self.test_vocoder.decode_unit(code)
            wav_len = torch.tensor([1.0]).to(self.device)
            transcript, _ = self.test_asr.transcribe_batch(wav, wav_len)
            transcript = transcript[0].lower()
            print(transcript)
            print(tgt_text)

            self.bleu_metric.append(ids, [transcript], [tgt_text])

        return (
            enc_out.detach(),
            p_seq,
            wav_lens,
            hyps,
        )

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.
        Arguments
        ---------
        predictions : torch.Tensor
            The model generated spectrograms and other metrics from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        (enc_out, p_seq, wav_lens, hyps) = predictions
        tokens_eos, tokens_eos_lens = batch.code_eos
        ids = batch.id

        # speech translation loss
        loss = self.hparams.seq_cost(p_seq, tokens_eos, length=tokens_eos_lens)

        if stage != sb.Stage.TRAIN:
            wavs, wav_lens = batch.tgt_sig
            tgt_text = batch.tgt_text

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)

            if stage == sb.Stage.VALID:
                # Save last batch
                self.last_batch = [
                    ids,
                    enc_out,
                    (tokens_eos, tokens_eos_lens),
                    (wavs, wav_lens),
                    tgt_text,
                ]

                current_epoch = self.hparams.epoch_counter.current
                output_progress_sample = (
                    self.hparams.progress_samples
                    and current_epoch % self.hparams.progress_samples_interval
                    == 0
                )
                if output_progress_sample:
                    self.wer_metric_st_greedy.append(
                        ids, hyps, tokens_eos, target_len=tokens_eos_lens
                    )

        return loss

    def init_optimizers(self):
        """Called during ``on_fit_start()``, initialize optimizers
        after parameters are fully configured (e.g. DDP, jit).
        """
        # Initializes the wav2vec2 optimizer if the model is not wav2vec2_frozen
        if not self.hparams.wav2vec2_frozen:
            self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
                self.modules.wav2vec2.parameters()
            )
        self.model_optimizer = self.hparams.opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec_optimizer", self.wav2vec_optimizer
            )
            self.checkpointer.add_recoverable(
                "model_optimizer", self.model_optimizer
            )

    def fit_batch(self, batch):
        """Fits a single batch
        Arguments
        ---------
        batch: tuple
            a training batch
        Returns
        -------
        loss: torch.Tensor
            detached loss
        """
        if self.optimizer_step == self.hparams.wav2vec2_freeze_steps:
            logger.warning(
                "speechbrain.lobes.models.huggingface_wav2vec - wav2vec 2.0 is unfrozen."
            )

        should_step = self.step % self.grad_accumulation_factor == 0
        if self.bfloat16_mix_prec:
            with torch.autocast(
                device_type=torch.device(self.device).type,
                dtype=torch.bfloat16,
            ):
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

        with self.no_sync(not should_step):
            (loss / self.grad_accumulation_factor).backward()
        if should_step:
            if self.check_gradients(loss):
                if (
                    not self.hparams.wav2vec2_frozen
                    and self.optimizer_step
                    >= self.hparams.wav2vec2_freeze_steps
                ):  # if wav2vec2 is not frozen
                    self.wav2vec_optimizer.step()
                self.model_optimizer.step()

            if not self.hparams.wav2vec2_frozen:
                self.wav2vec_optimizer.zero_grad()
            self.model_optimizer.zero_grad()

            self.optimizer_step += 1

        self.on_fit_batch_end(batch, outputs, loss, should_step)
        return loss.detach().cpu()

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """Called after ``fit_batch()``, meant for calculating and logging metrics.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.
        outputs : list or dictionary of torch.Tensors
            Returned value of compute_forward().
        loss : torch.Tensor
            Returned value of compute_objectives().
        should_step : boolean
            Whether optimizer.step() was called or not.
        """
        if should_step:
            # anneal model lr every update
            self.hparams.noam_annealing(self.model_optimizer)

    def on_stage_start(self, stage, epoch):
        """Gets called when a stage starts.

        Arguments
        ---------
        stage : Stage
            The stage of the experiment: Stage.TRAIN, Stage.VALID, Stage.TEST
        epoch : int
            The current epoch count.
        """
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.wer_metric_st_greedy = self.hparams.error_rate_computer()
            self.bleu_metric = self.hparams.bleu_computer()
            self.last_batch = None
            self.last_epoch = 0

            if stage == sb.Stage.TEST:
                logger.info("Loading pretrained HiFi-GAN ...")
                self.test_vocoder = UnitHIFIGAN.from_hparams(
                    source=self.hparams.vocoder_source,
                    savedir=self.hparams.vocoder_download_path,
                    run_opts={"device": self.device},
                )

                logger.info("Loading pretrained ASR ...")
                self.test_asr = EncoderDecoderASR.from_hparams(
                    source=self.hparams.asr_source,
                    savedir=self.hparams.asr_download_path,
                    run_opts={"device": self.device},
                )

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
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_loss

        # At the end of validation, we can write
        elif stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():
            stage_stats = {"loss": stage_loss}
            stage_stats["ACC"] = self.acc_metric.summarize()

            output_progress_sample = (
                self.hparams.progress_samples
                and epoch % self.hparams.progress_samples_interval == 0
            )

            if output_progress_sample:
                # Compute BLEU scores
                self._run_evaluation_pipeline(epoch)

                stage_stats["BLEU"] = self.bleu_metric.summarize("BLEU")
                stage_stats["WER"] = self.wer_metric_st_greedy.summarize(
                    "error_rate"
                )

            self.last_epoch = epoch
            current_epoch = self.hparams.epoch_counter.current
            lr_model = self.hparams.noam_annealing.current_lr
            lr_wav2vec = 0.0

            if not self.hparams.wav2vec2_frozen:
                (lr_wav2vec, new_lr_wav2vec,) = self.hparams.wav2vec_annealing(
                    stage_stats["ACC"]
                )
                sb.nnet.schedulers.update_learning_rate(
                    self.wav2vec_optimizer, new_lr_wav2vec
                )

            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": current_epoch,
                    "lr_model": lr_model,
                    "lr_wav2vec": lr_wav2vec,
                },
                train_stats={"loss": self.train_stats},
                valid_stats=stage_stats,
            )

            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=10,
            )

        elif stage == sb.Stage.TEST:
            stage_stats = {"loss": stage_loss}
            stage_stats["BLEU"] = self.bleu_metric.summarize("BLEU")

            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

            logger.info(
                f"BLEU score: {round(self.bleu_metric.summarize('BLEU'), 2)}"
            )
            bleu_file = pl.Path(self.hparams.output_folder) / "bleu.txt"
            with open(bleu_file, "a+", encoding="utf-8") as w:
                self.bleu_metric.write_stats(w)

    def _run_evaluation_pipeline(self, epoch):
        """Run the complete evaluation pipeline by computing BLEU scores on transcripts extracted from synthesized speech
        using the unit-to-speech (U2S) model.
        Arguments
        ---------
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        if self.last_batch is None:
            return
        (
            ids,
            enc_out,
            (tokens_eos, tokens_eos_lens),
            (wavs, wav_lens),
            tgt_text,
        ) = self.last_batch

        save_folder = pl.Path(self.hparams.progress_sample_path) / f"{epoch}"
        save_folder.mkdir(parents=True, exist_ok=True)

        logger.info("Loading pretrained HiFi-GAN ...")
        hifi_gan = UnitHIFIGAN.from_hparams(
            source=self.hparams.vocoder_source,
            savedir=self.hparams.vocoder_download_path,
        )

        logger.info("Loading pretrained ASR ...")
        asr_model = EncoderDecoderASR.from_hparams(
            source=self.hparams.asr_source,
            savedir=self.hparams.asr_download_path,
        )

        sample_size = self.hparams.progress_batch_sample_size
        if len(ids) < sample_size:
            sample_size = len(ids)

        # Beamsearch decoding
        hyps, _ = self.hparams.test_search(
            enc_out[:sample_size], wav_lens[:sample_size]
        )

        tokens_eos = tokens_eos.cpu()
        tokens_eos_lens = tokens_eos_lens.cpu()

        tokens_eos = sb.utils.data_utils.undo_padding(
            tokens_eos, tokens_eos_lens
        )
        wavs = sb.utils.data_utils.undo_padding(wavs, wav_lens)

        transcripts = []
        for i in tqdm.tqdm(range(sample_size)):
            utt_id = ids[i]
            code = hyps[i]
            code = torch.LongTensor(code)

            wav = hifi_gan.decode_unit(code)
            sample_path = save_folder / f"{utt_id}_pred.wav"
            sb.dataio.dataio.write_audio(
                sample_path, wav.transpose(0, 1), self.hparams.sample_rate
            )

            transcript = asr_model.transcribe_file(sample_path.as_posix())
            transcript = transcript[0].lower()
            transcripts.append(transcript)

            wav = torch.FloatTensor(wavs[i])
            sample_path = save_folder / f"{utt_id}_ref.wav"
            sb.dataio.dataio.write_audio(
                sample_path, wav, self.hparams.sample_rate
            )

            sample_path = save_folder / f"{utt_id}.txt"
            with open(sample_path, "w") as file:
                file.write(f"pred: {transcript}\n")
                file.write(f"ref: {tgt_text[i]}\n")

        self.bleu_metric.append(
            ids[:sample_size], transcripts, [tgt_text[:sample_size]]
        )

        bleu_path = save_folder / "bleu.txt"
        with open(bleu_path, "w") as file:
            file.write(
                f"BLEU score: {round(self.bleu_metric.summarize('BLEU'), 2)}\n"
            )


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    codes_folder = pl.Path(hparams["codes_folder"])

    # Define audio pipeline. In this case, we simply read the audio contained
    # in the variable src_audio with the custom reader.
    @sb.utils.data_pipeline.takes("src_audio")
    @sb.utils.data_pipeline.provides("src_sig")
    def src_audio_pipeline(wav):
        """Load the audio signal. This is done on the CPU in the `collate_fn`."""
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        sig = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(sig)
        return sig

    @sb.utils.data_pipeline.takes("tgt_audio")
    @sb.utils.data_pipeline.provides("tgt_sig")
    def tgt_audio_pipeline(wav):
        """Load the audio signal. This is done on the CPU in the `collate_fn`."""
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        sig = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(sig)
        return sig

    @sb.utils.data_pipeline.takes("id")
    @sb.utils.data_pipeline.provides("code_bos", "code_eos")
    def unit_pipeline(utt_id):
        code = np.load(codes_folder / f"{utt_id}_tgt.npy")
        code = torch.LongTensor(code)
        code = torch.unique_consecutive(code)
        code_bos = torch.cat((torch.LongTensor([hparams["bos_index"]]), code))
        yield code_bos
        code_eos = torch.cat((code, torch.LongTensor([hparams["eos_index"]])))
        yield code_eos

    datasets = {}
    for split in hparams["splits"]:
        datasets[split] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{split}_json"],
            dynamic_items=[
                src_audio_pipeline,
                tgt_audio_pipeline,
                unit_pipeline,
            ],
            output_keys=[
                "id",
                "src_sig",
                "tgt_sig",
                "duration",
                "code_bos",
                "code_eos",
                "tgt_text",
            ],
        )

    # Sorting training data with ascending order makes the code  much
    # faster  because we minimize zero-padding. In most of the cases, this
    # does not harm the performance.
    if hparams["sorting"] == "ascending":
        datasets["train"] = datasets["train"].filtered_sorted(
            sort_key="duration"
        )
        datasets["valid"] = datasets["valid"].filtered_sorted(
            sort_key="duration"
        )

        hparams["train_dataloader_opts"]["shuffle"] = False
        hparams["valid_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        datasets["train"] = datasets["train"].filtered_sorted(
            sort_key="duration", reverse=True
        )
        datasets["valid"] = datasets["valid"].filtered_sorted(
            sort_key="duration", reverse=True
        )

        hparams["train_dataloader_opts"]["shuffle"] = False
        hparams["valid_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        hparams["train_dataloader_opts"]["shuffle"] = True
        hparams["valid_dataloader_opts"]["shuffle"] = False

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    # Dynamic Batching is used, we instantiate the needed samplers.
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams = hparams["dynamic_batch_sampler"]
        num_buckets = dynamic_hparams["num_buckets"]

        train_batch_sampler = DynamicBatchSampler(
            datasets["train"],
            dynamic_hparams["max_batch_len"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
            max_batch_ex=dynamic_hparams["max_batch_ex"],
        )

        valid_batch_sampler = DynamicBatchSampler(
            datasets["valid"],
            dynamic_hparams["max_batch_len_val"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"],
            shuffle=False,
            batch_ordering=dynamic_hparams["batch_ordering"],
        )

    return datasets, (train_batch_sampler, valid_batch_sampler)


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    sys.path.append("../")
    from cvss_prepare import prepare_cvss

    sb.utils.distributed.run_on_main(
        prepare_cvss,
        kwargs={
            "src_data_folder": hparams["src_data_folder"],
            "tgt_data_folder": hparams["tgt_data_folder"],
            "save_folder": hparams["save_folder"],
            "splits": hparams["splits"],
            "seed": hparams["seed"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    from extract_code import extract_cvss

    sb.utils.distributed.run_on_main(
        extract_cvss,
        kwargs={
            "data_folder": hparams["save_folder"],
            "splits": hparams["splits"],
            "kmeans_folder": hparams["kmeans_source"],
            "encoder": hparams["encoder_source"],
            "layer": hparams["layer"],
            "save_folder": hparams["save_folder"],
            "skip_extract": hparams["skip_extract"],
        },
    )

    datasets, samplers = dataio_prepare(hparams)
    (train_bsampler, valid_bsampler) = samplers

    s2ut_brain = S2UT(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_bsampler is not None:
        train_dataloader_opts = {
            "batch_sampler": train_bsampler,
            "num_workers": hparams["num_workers"],
            "collate_fn": hparams["train_dataloader_opts"]["collate_fn"],
        }

    if valid_bsampler is not None:
        valid_dataloader_opts = {
            "batch_sampler": valid_bsampler,
            "collate_fn": hparams["valid_dataloader_opts"]["collate_fn"],
        }

    s2ut_brain.fit(
        s2ut_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )

    test_dataloader_opts = {
        "batch_size": 1,
    }

    for dataset in ["test"]:
        s2ut_brain.evaluate(
            datasets[dataset],
            max_key="ACC",
            test_loader_kwargs=test_dataloader_opts,
        )
