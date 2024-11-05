"""
 Recipe for training the speech-to-unit translation (S2UT) model, the implementation is based on the following papers:
 - Direct speech-to-speech translation with discrete units: (https://arxiv.org/abs/2006.04558)
 - Enhanced Direct Speech-to-Speech Translation Using Self-supervised Pre-training and Data Augmentation: (https://arxiv.org/abs/2204.02967)
 To run this recipe, do the following:
 # python train.py hparams/train_fr-en.yaml --src_data_folder=/corpus/CommonVoice/fr --tgt_data_folder=/corpus/CVSS/fr

 Authors
 * Jarod Duret 2023
"""

import pathlib as pl
import sys

import numpy as np
import torch
import torchaudio
import tqdm
from hyperpyyaml import load_hyperpyyaml
from torch.nn.parallel import DistributedDataParallel

import speechbrain as sb
from speechbrain.inference.ASR import EncoderDecoderASR
from speechbrain.inference.vocoders import UnitHIFIGAN
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class S2UT(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Computes the forward pass.

        Arguments
        ---------
        batch : torch.Tensor or tensors
            An element from the dataloader, including inputs for processing.
        stage : Stage
            The stage of the experiment: Stage.TRAIN, Stage.VALID, Stage.TEST

        Returns
        -------
        (torch.Tensor or torch.Tensors, list of float or None, list of str or None)
            The outputs after all processing is complete.
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
        wavs = None
        transcripts = None
        if stage != sb.Stage.TRAIN:
            if (
                stage == sb.Stage.TEST
                or self.hparams.epoch_counter.current
                % self.hparams.evaluation_interval
                == 0
            ):
                ids = batch.id
                tgt_text = batch.tgt_text

                search = (
                    self.hparams.valid_search
                    if stage == sb.Stage.VALID
                    else self.hparams.test_search
                )
                hyps, _, _, _ = search(enc_out.detach(), wav_lens)

                # generate speech and transcriptions
                wavs = []
                for hyp in hyps:
                    if len(hyp) > 10:
                        code = torch.LongTensor(hyp[:-1])
                        wav = self.test_vocoder.decode_unit(code.unsqueeze(-1))
                        wavs.append(wav.squeeze(0))
                    else:
                        logger.warn(
                            f"Encountered hyp {hyp} too short for decoding, using fake blank audio for testing"
                        )
                        wavs.append(torch.zeros(40000))  # on cpu device
                if wavs:
                    wavs, wav_lens = sb.utils.data_utils.batch_pad_right(wavs)
                    transcripts, _ = self.test_asr.transcribe_batch(
                        wavs, wav_lens
                    )
                    transcripts = [
                        transcript.lower() for transcript in transcripts
                    ]

                    self.bleu_metric.append(ids, transcripts, [tgt_text])

        return (
            p_seq,
            wavs,
            transcripts,
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
        (p_seq, wavs, transcripts) = predictions
        tokens_eos, tokens_eos_lens = batch.code_eos
        ids = batch.id

        # speech translation loss
        loss = self.hparams.seq_cost(p_seq, tokens_eos, length=tokens_eos_lens)

        if stage != sb.Stage.TRAIN:
            if (
                stage == sb.Stage.TEST
                or self.hparams.epoch_counter.current
                % self.hparams.evaluation_interval
                == 0
            ):
                # compute the accuracy of the one-step-forward prediction
                self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)

                tgt_wavs, _ = batch.tgt_sig
                tgt_transcripts = batch.tgt_text

                # Save last batch
                wavs = [wav.cpu() for wav in wavs]
                tgt_wavs = [wav.cpu() for wav in tgt_wavs]
                self.last_batch = [
                    ids,
                    (wavs, transcripts),
                    (tgt_transcripts, tgt_wavs),
                ]

        return loss

    def freeze_optimizers(self, optimizers):
        """Freezes the wav2vec2 optimizer according to the warmup steps"""
        valid_optimizers = {}
        if (
            not self.hparams.wav2vec2_frozen
            and self.optimizer_step >= self.hparams.wav2vec2_freeze_steps
        ):
            valid_optimizers["wav2vec_optimizer"] = optimizers[
                "wav2vec_optimizer"
            ]
        valid_optimizers["model_optimizer"] = optimizers["model_optimizer"]
        return valid_optimizers

    def init_optimizers(self):
        """Called during ``on_fit_start()``, initialize optimizers
        after parameters are fully configured (e.g. DDP, jit).
        """
        self.optimizers_dict = {}

        # Initializes the wav2vec2 optimizer if the model is not wav2vec2_frozen
        if not self.hparams.wav2vec2_frozen:
            self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
                self.modules.wav2vec2.parameters()
            )
            self.optimizers_dict["wav2vec_optimizer"] = self.wav2vec_optimizer

        self.model_optimizer = self.hparams.opt_class(
            self.hparams.model.parameters()
        )
        self.optimizers_dict["model_optimizer"] = self.model_optimizer

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec_optimizer", self.wav2vec_optimizer
            )
            self.checkpointer.add_recoverable(
                "model_optimizer", self.model_optimizer
            )

    def on_fit_batch_start(self, batch, should_step):
        """Called at the beginning of ``fit_batch()``.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.
        should_step : boolean
            Whether optimizer.step() was called or not.
        """
        if self.optimizer_step == self.hparams.wav2vec2_freeze_steps:
            logger.warning(
                "speechbrain.lobes.models.huggingface_wav2vec - wav2vec 2.0 is unfrozen."
            )

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

        Returns
        -------
        None
        """
        if stage != sb.Stage.TRAIN:
            if (
                stage == sb.Stage.VALID
                and epoch % self.hparams.evaluation_interval != 0
            ):
                return

            self.acc_metric = self.hparams.acc_computer()
            self.bleu_metric = self.hparams.bleu_computer()
            self.last_batch = None

            logger.info("Loading pretrained HiFi-GAN ...")
            self.test_vocoder = UnitHIFIGAN.from_hparams(
                source=self.hparams.vocoder_source,
                savedir=self.hparams.vocoder_download_path,
                run_opts={"device": "cpu"},
            )

            logger.info("Loading pretrained ASR ...")
            self.test_asr = EncoderDecoderASR.from_hparams(
                source=self.hparams.asr_source,
                savedir=self.hparams.asr_download_path,
                run_opts={"device": "cpu"},
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
        elif (
            stage == sb.Stage.VALID
            and epoch % self.hparams.evaluation_interval == 0
        ):
            # delete vocoder and asr to free memory for next training epoch
            del self.test_vocoder
            del self.test_asr

            stage_stats = {"loss": stage_loss}
            stage_stats["ACC"] = self.acc_metric.summarize()
            stage_stats["BLEU"] = self.bleu_metric.summarize("BLEU")

            output_progress_sample = (
                self.hparams.progress_samples
                and epoch % self.hparams.progress_samples_interval == 0
            )

            if output_progress_sample:
                self._save_progress_sample(epoch)

            current_epoch = self.hparams.epoch_counter.current
            lr_model = self.hparams.noam_annealing.current_lr
            lr_wav2vec = 0.0

            if not self.hparams.wav2vec2_frozen:
                (lr_wav2vec, new_lr_wav2vec) = self.hparams.wav2vec_annealing(
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
                meta={
                    "ACC": stage_stats["ACC"],
                    "BLEU": stage_stats["BLEU"],
                    "epoch": epoch,
                },
                max_keys=["BLEU"],
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

    def _save_progress_sample(self, epoch):
        """Save samples and BLEU score from last batch for current epoch.

        Arguments
        ---------
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.

        Returns
        -------
        None
        """
        if self.last_batch is None:
            return

        (
            ids,
            (wavs, transcripts),
            (tgt_transcripts, tgt_wavs),
        ) = self.last_batch

        save_folder = pl.Path(self.hparams.progress_sample_path) / f"{epoch}"
        save_folder.mkdir(parents=True, exist_ok=True)

        sample_size = self.hparams.progress_batch_sample_size
        if len(ids) < sample_size:
            sample_size = len(ids)

        for i in tqdm.tqdm(range(sample_size)):
            utt_id = ids[i]
            wav = wavs[i]
            transcript = transcripts[i]
            tgt_transcript = tgt_transcripts[i]
            tgt_wav = tgt_wavs[i]

            sample_path = save_folder / f"{utt_id}_pred.wav"
            sb.dataio.dataio.write_audio(
                sample_path, wav, self.hparams.sample_rate
            )

            sample_path = save_folder / f"{utt_id}_ref.wav"
            sb.dataio.dataio.write_audio(
                sample_path, tgt_wav, self.hparams.sample_rate
            )

            sample_path = save_folder / f"{utt_id}.txt"
            with open(sample_path, "w", encoding="utf-8") as file:
                file.write(f"pred: {transcript}\n")
                file.write(f"ref: {tgt_transcript}\n")

        self.bleu_metric.append(
            ids[:sample_size],
            transcripts[:sample_size],
            [tgt_transcripts[:sample_size]],
        )

        bleu_path = save_folder / "bleu.txt"
        with open(bleu_path, "w", encoding="utf-8") as file:
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
        """Load the source language audio signal.
        This is done on the CPU in the `collate_fn`
        """
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        sig = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"]
        )(sig)
        return sig

    @sb.utils.data_pipeline.takes("tgt_audio")
    @sb.utils.data_pipeline.provides("tgt_sig")
    def tgt_audio_pipeline(wav):
        """Load the target language audio signal.
        This is done on the CPU in the `collate_fn`.
        """
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        sig = torchaudio.transforms.Resample(
            info.sample_rate,
            hparams["sample_rate"],
        )(sig)
        return sig

    @sb.utils.data_pipeline.takes("id")
    @sb.utils.data_pipeline.provides("code_bos", "code_eos")
    def unit_pipeline(utt_id):
        """Load target codes"""
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

    return datasets, train_batch_sampler


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file, encoding="utf-8") as fin:
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
            "sample_rate": hparams["sample_rate"],
            "skip_extract": hparams["skip_extract"],
        },
    )

    datasets, train_bsampler = dataio_prepare(hparams)

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

    s2ut_brain.fit(
        s2ut_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid_small"],
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )

    test_dataloader_opts = {
        "batch_size": 1,
    }

    for dataset in ["valid", "test"]:
        s2ut_brain.evaluate(
            datasets[dataset],
            max_key="BLEU",
            test_loader_kwargs=test_dataloader_opts,
        )
