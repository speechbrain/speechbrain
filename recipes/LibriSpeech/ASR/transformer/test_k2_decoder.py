
# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang & Rong Fu)
from speechbrain.pretrained import EncoderDecoderASR
import torch

import sys

from snowfall.training.ctc_graph import build_ctc_topo2
import k2
import k2.ragged as k2r

from typing import Optional, List

def load_model():
    model = EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-transformer-transformerlm-librispeech",
        savedir="pretrained_models/asr-transformer-transformerlm-librispeech",
        #  run_opts={'device': 'cuda:0'},
    )
    return model

def get_texts(best_paths: k2.Fsa) -> List[List[int]]:
    """Extract the texts (as word IDs) from the best-path FSAs.
    Args:
      best_paths:
        A k2.Fsa with best_paths.arcs.num_axes() == 3, i.e.
        containing multiple FSAs, which is expected to be the result
        of k2.shortest_path (otherwise the returned values won't
        be meaningful).
    Returns:
      Returns a list of lists of int, containing the label sequences we
      decoded.
    """
    if isinstance(best_paths.aux_labels, k2.RaggedInt):
        # remove 0's and -1's.
        aux_labels = k2r.remove_values_leq(best_paths.aux_labels, 0)
        aux_shape = k2r.compose_ragged_shapes(
            best_paths.arcs.shape(), aux_labels.shape()
        )

        # remove the states and arcs axes.
        aux_shape = k2r.remove_axis(aux_shape, 1)
        aux_shape = k2r.remove_axis(aux_shape, 1)
        aux_labels = k2.RaggedInt(aux_shape, aux_labels.values())
    else:
        # remove axis corresponding to states.
        aux_shape = k2r.remove_axis(best_paths.arcs.shape(), 1)
        aux_labels = k2.RaggedInt(aux_shape, best_paths.aux_labels)
        # remove 0's and -1's.
        aux_labels = k2r.remove_values_leq(aux_labels, 0)

    assert aux_labels.num_axes() == 2
    return k2r.to_list(aux_labels)

def k2_decoder(log_probs, wav_lens, ctc_topo):
    '''building an FSA decoder FSA decoding based on (speechbrain/nnet/RNN.py).
       Args:
           log_probs: torch.Tensor of dimension [B, T, N].
                            where, B = Batchsize,
                                T = the number of frames,
                                N = number of tokens
                    It represents the probability distribution over tokens, which
                    is the output of an encoder network.
           ctc_topo: a CTC topology fst that represents a specific topology used to
                    convert the network outputs to a sequence of phones.
       Return:
           hyps : a list of lists of int,
                This list contains batch_size number. Each inside list contains
                a list stores all the hypothesis for this sentence.
           scores : a list of float64
                This list contains the total score of each sequences.
    '''

    batchnum = log_probs.size(0)

    #
    supervisions = []
    for i in range(batchnum):
        supervisions.append([i, 0, log_probs.size(1)])
    print("supervisions list:", supervisions)
    supervision_segments = torch.tensor(supervisions, dtype=torch.int32)

    supervision_segments = torch.clamp(supervision_segments, min=0)

    dense_fsa_vec = k2.DenseFsaVec(log_probs, supervision_segments)

    lattices = k2.intersect_dense_pruned(ctc_topo, dense_fsa_vec, 20.0, 8, 30,
                                         10000)
    best_paths = k2.shortest_path(lattices, True)

    if(0):
        hyps = []
        for i in range(batchnum):
            aux_labels = best_paths[i].aux_labels
            score = best_paths[i].scores.sum()
            print("best path score:",score)
            aux_labels = aux_labels[aux_labels.nonzero().squeeze()]
            # The last entry is -1, so remove it
            aux_labels = aux_labels[:-1]
            hyps.append(aux_labels.tolist())
    else:
        hyps = get_texts(best_paths)#print(hyps)
        # sum the scores for each sequence
        scores = best_paths.get_tot_scores(True, True)

    return hyps, scores.tolist()


@torch.no_grad()
def main():
    import os

    print(os.getcwd())
    print(os.path.abspath(os.path.dirname(__file__)))

    model = load_model()

    device = model.device

    # See https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech/blob/main/example.wav
    sound_file = './example.wav'
    wav = model.load_audio(sound_file)
    print("size of wav:", wav.size())
    # wav is a 1-d tensor, e.g., [52173]

    # See https://huggingface.co/speechbrain/google_speech_command_xvector/blob/main/yes.wav
    sound_file = './yes.wav'
    wav1 = model.load_audio(sound_file)
    print("size of wav1:", wav1.size())
    wavs_ = [wav,wav1]

    # multiwav = torch.nn.utils.rnn.pad_sequence([wav,wav0,wav1],batch_first=True)


    # wavs = multiwav.float().to(device)
    # wavs = wav1.unsqueeze(0).float().to(device)
    # wavs is a 2-d tensor, e.g., [1, 52173]

    multiwav = torch.nn.utils.rnn.pad_sequence(wavs_, batch_first=True)
    # print("size of multiwav:", multiwav.size())
    wavs = multiwav.float().to(device)

    wav_lens = torch.Tensor([wav.size(), wav1.size()])#torch.ones([batchnum])#
    wav_lens = wav_lens.to(device)
    encoder_out = model.modules.encoder(wavs, wav_lens)
    # encoder_out.shape [N, T, C], e.g., [1, 82, 768]

    use_k2 = True
    if use_k2:
        print("-----------------------ctc_topo result---------------------------")

        logits = model.hparams.ctc_lin(encoder_out)
        # logits.shape [N, T, C], e.g., [1, 82, 5000]
        log_probs = model.hparams.log_softmax(logits)
        # log_probs.shape [N, T, C], e.g., [1, 82, 5000]

        device = log_probs.device
        vocab_size = log_probs.size(2)  # model.tokenizer.vocab_size()
        # print(model.tokenizer.vocab_size())
        # print(log_probs.size(2))
        ctc_topo = build_ctc_topo2(list(range(vocab_size)))
        ctc_topo = k2.create_fsa_vec([ctc_topo]).to(device)

        predicted_tokens, scores = k2_decoder(log_probs, wav_lens, ctc_topo)
        print(predicted_tokens)
        print(scores)

    else:
        predicted_tokens, scores = model.modules.decoder(encoder_out, wav_lens)
        print(predicted_tokens)
        print(scores)




    #hyp = model.tokenizer.decode(hyp)
    predicted_words = [
        model.tokenizer.decode_ids(token_seq)
        for token_seq in predicted_tokens
        ]

    print(predicted_words)


if __name__ == '__main__':
    main()
