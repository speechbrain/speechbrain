import sys
import torch
import speechbrain.utils.edit_distance as edit_distance
import speechbrain.data_io.wer as wer_io
from speechbrain.decoders.ctc import ctc_greedy_decode
from speechbrain.decoders.decoders import undo_padding
from tqdm.contrib import tzip
from speechbrain.core import Experiment, Brain
with open('recipes/ASR_CTC/TIMIT/VGG2_BLSTM_MLP/params.yaml') as fi:
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
            stats = {'wer': stats}
            return loss, stats

        return loss

    def summarize(self, stats, write=False):
        loss = sum(stat['loss'] for stat in stats)
        summary = {'loss': float(loss / len(stats))}

        if 'wer' in stats[0]:
            wer_stats = [item for stat in stats for item in stat['wer']]
            wer_summary = edit_distance.wer_summary(wer_stats)
            if write:
                with open(sb.wer_file, "w") as fo:
                    wer_io.print_wer_summary(wer_summary, fo)
                    wer_io.print_alignments(wer_stats, fo)
            summary['wer'] = wer_summary['WER']

        return summary


def main():

    # Prepare the data
    sb.prepare_timit()
    train_set = sb.train_loader()
    valid_set = sb.valid_loader()
    test_set = sb.test_loader()

    # Initialize brain and learn
    asr_brain = ASR([sb.model], sb.optimizer, sb.lr_annealing, sb.saver)
    asr_brain.learn(sb.epoch_counter, train_set, valid_set, min_keys=['wer'])

    # Load best model, evaluate that:
    sb.recover_if_possible(min_key='wer')
    asr_brain.evaluate(test_set)


if __name__ == '__main__':
    main()
