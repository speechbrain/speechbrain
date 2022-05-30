"""
 Recipe for training the TransformerTTS Text-To-Speech model, an end-to-end
 neural text-to-speech (TTS) system introduced in 'Neural Speech Synthesis
 with Transformer Network' paper published in AAAI-19.
 (https://arxiv.org/abs/1809.08895)
 To run this recipe, do the following:
 # python train.py --device=cuda:0 hparams.yaml

 Authors
 * Sathvik Udupa 2021
"""
import sys
import torch
import logging
sys.path.append("../../../../")
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.lobes.models.synthesis.fastspeech import dataio_prepare

# sys.path.append("..")
from recipes.LJSpeech.TTS.common.utils import PretrainedModelMixin, ProgressSampleImageMixin

logger = logging.getLogger(__name__)

class FastSpeechBrain(sb.Brain, PretrainedModelMixin, ProgressSampleImageMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_progress_samples()

    def compute_forward(self, batch, stage):
        inputs, y = batch_to_gpu(batch)
        return self.hparams.model(*inputs)  # 1#2#

    def fit_batch(self, batch):
        result = super().fit_batch(batch)
        return result

    def compute_objectives(self, predictions, batch, stage):
        x, y = batch_to_gpu(batch)
        self._remember_sample(y, predictions)
        return criterion(predictions, y)

    def _remember_sample(self, batch, predictions):
        mel_pred, durs = predictions
        mel_target, durations, mel_length, phon_len  = batch
        # import matplotlib.pyplot as plt
        # plt.imshow(mel_target[0].detach().cpu().numpy())
        #
        # plt.savefig('gnd.png')
        # plt.imshow(mel_pred[0].detach().cpu().numpy())
        # plt.savefig('pred.png')
        # import numpy as np
        # with open('gnd.npy', 'wb') as f:
        #     np.save(f, mel_target[0].detach().cpu().numpy())
        # with open('pred.npy', 'wb') as f:
        #     np.save(f, mel_pred[0].detach().cpu().numpy())
        # import librosa
        # librosa.feature.inverse.mel_to_audio

        self.remember_progress_sample(
                                    target=self._clean_mel(mel_target, mel_length),
                                    pred=self._clean_mel(mel_pred, mel_length)
                                    )



    def _clean_mel(self, mel, len, sample_idx=0):
        assert mel.dim() == 3
        return torch.sqrt(torch.exp(mel[sample_idx][:len[sample_idx]]))


    def on_stage_end(self, stage, stage_loss, epoch):

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
            }

        # At the end of validation, we can write
        if stage == sb.Stage.VALID:
            # Update learning rate
            lr = self.optimizer.param_groups[-1]["lr"]

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(  # 1#2#
                stats_meta={"Epoch": epoch, "lr": lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )
            output_progress_sample = (
                self.hparams.progress_samples
                and epoch % self.hparams.progress_samples_interval == 0
                and epoch > self.hparams.progress_samples_min_rin
            )

            if output_progress_sample:
                print('saving')
                self.save_progress_sample(epoch)

            # Save the current checkpoint and delete previous checkpoints.
            #UNCOMMENT THIS
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["loss"])

        # We also write statistics about test data spectogramto stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )
def batch_to_gpu(batch):
    (
        text_padded,
        durations,
        input_lengths,
        mel_padded,
        output_lengths,
        len_x,
        labels,
        wavs
    ) = batch
    durations = to_gpu(durations).long()
    phonemes = to_gpu(text_padded).long()
    input_lengths = to_gpu(input_lengths).long()
    max_len = torch.max(input_lengths.data).item()
    spectogram = to_gpu(mel_padded).float()
    mel_lengths = to_gpu(output_lengths).long()
    x = (phonemes, durations)
    y = (spectogram, durations, mel_lengths, input_lengths)
    return x, y

def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x


def criterion(model_output, targets, log_scale_durations=True):
    mel_target, target_durations, mel_length, phon_len  = targets

    assert len(mel_target.shape) == 3
    mel_out, log_durations = model_output
    log_durations = log_durations.squeeze()
    if log_scale_durations:
        log_target_durations = torch.log(target_durations.float() + 1)
        durations = torch.clamp(torch.exp(log_durations) - 1, 0, 20)
    mel_loss, dur_loss = 0, 0
    for i in range(mel_target.shape[0]):
        if i == 0:
            mel_loss = torch.nn.MSELoss()(mel_out[i, :mel_length[i], :], mel_target[i, :mel_length[i], :])
            dur_loss = torch.nn.MSELoss()(log_durations[i, :phon_len[i]], log_target_durations[i, :phon_len[i]].to(torch.float32))
        else:
            mel_loss = mel_loss + torch.nn.MSELoss()(mel_out[i, :mel_length[i], :], mel_target[i, :mel_length[i], :])
            dur_loss = dur_loss + torch.nn.MSELoss()(log_durations[i, :phon_len[i]], log_target_durations[i, :phon_len[i]].to(torch.float32))
    # print(mel_loss, dur_loss)
    mel_loss = torch.div(mel_loss, len(mel_target))
    dur_loss = torch.div(dur_loss, len(mel_target))
    # print(mel_loss, dur_loss)
    return mel_loss + dur_loss


def main():
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    datasets = dataio_prepare(hparams)

    # Brain class initialization
    fastspeech_brain = FastSpeechBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    # Training
    fastspeech_brain.fit(
        fastspeech_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
    )
    if hparams.get("save_for_pretrained"):
        fastspeech_brain.save_for_pretrained()

if __name__ == "__main__":
    main()
