import sys
import torch
from tqdm.contrib import tzip
import speechbrain.data_io.wer as wer_io
import speechbrain.utils.edit_distance as edit_distance
from speechbrain.decoders.ctc import ctc_greedy_decode
from speechbrain.decoders.decoders import undo_padding
from speechbrain.core import Experiment, Brain
with open("recipes/ASR_CTC/TIMIT/VGG2_BLSTM_MLP/params.yaml") as fi:
    sb = Experiment(
        yaml_stream=fi,
        commandline_args=sys.argv[1:],
    )


class ASR(Brain):

    def forward(self, x, init_params=False):
        ids, wavs, wav_lens = x
        wavs, wav_lens = wavs.to(sb.device), wav_lens.to(sb.device)
        feats = sb.compute_features(wavs)
        feats = sb.normalize(feats, wav_lens)

        if init_params:
            sb.model.init_params(feats)
        else:
            return sb.model(feats), wav_lens

    def compute_objectives(self, predictions, targets, train=True):
        pout, pout_lens = predictions
        ids, phns, phn_lens = targets
        phns, phn_lens = phns.to(sb.device), phn_lens.to(sb.device)
        loss = sb.compute_cost(pout, phns, [pout_lens, phn_lens])

        if not train:
            sequence = ctc_greedy_decode(pout, pout_lens, blank_id=-1)
            phns = undo_padding(phns, phn_lens)
            stats = edit_distance.wer_details_for_batch(
                ids, phns, sequence, compute_alignments=True
            )
            stats = {"wer": stats}
            return loss, stats

        return loss

    def summarize(self, stats, write=False):

        # Accumulate
        accumulator = {"loss": 0.}
        if "wer" in stats[0]:
            accumulator["wer"] = []
        for stat in stats:
            for stat_type in stat:
                accumulator[stat_type] += stat[stat_type]

        # Normalize
        summary = {"loss": float(accumulator["loss"] / len(stats))}
        if "wer" in accumulator:
            wer_summary = edit_distance.wer_summary(accumulator["wer"])
            summary["wer"] = wer_summary["WER"]

            # Write test data to file
            if write:
                with open(sb.wer_file, "w") as fo:
                    wer_io.print_wer_summary(wer_summary, fo)
                    wer_io.print_alignments(accumulator["wer"], fo)

        return summary


def main():

    # Prepare the data
    sb.prepare_timit()
    train_set = sb.train_loader()
    valid_set = sb.valid_loader()
    test_set = sb.test_loader()

    # Initialize brain and learn
    asr_brain = ASR(
        models=[sb.model],
        optimizer=sb.optimizer,
        scheduler=sb.lr_annealing,
        saver=sb.saver,
    )
    asr_brain.learn(
        epoch_counter=sb.epoch_counter,
        train_set=train_set,
        valid_set=valid_set,
        min_keys=["wer"],
    )

    # Load best model, evaluate that:
    asr_brain.evaluate(test_set, min_key="wer")


if __name__ == "__main__":
    main()
