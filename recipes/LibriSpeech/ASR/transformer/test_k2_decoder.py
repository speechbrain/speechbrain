#!/usr/bin/env python3

# Copyright (c)  2021  Xiaomi Corporation (Authors: Rong Fu, Tsinghua University)

import os

from snowfall.training.ctc_graph import build_ctc_topo2
from speechbrain.pretrained import EncoderDecoderASR

import k2
import torch
import kaldilm

from speechbrain.decoders.k2 import (
    ctc_decoding,
    read_words,
    test_read_lexicon,
    HLG_decoding,
)
from snowfall.decoding.graph import compile_HLG


def load_model():
    model = EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-transformer-transformerlm-librispeech",
        savedir="pretrained_models/asr-transformer-transformerlm-librispeech",
        #  run_opts={'device': 'cuda:0'},
    )
    return model


@torch.no_grad()
def main():

    model = load_model()

    device = model.device

    # See https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech/blob/main/example.wav
    sound_file = "./example.wav"
    wav = model.load_audio(sound_file)
    print("size of wav:", wav.size())
    # wav is a 1-d tensor, e.g., [52173]

    # See https://huggingface.co/speechbrain/google_speech_command_xvector/blob/main/yes.wav
    sound_file = "./yes.wav"
    wav1 = model.load_audio(sound_file)
    print("size of wav1:", wav1.size())
    wavs_ = [wav, wav1]

    multiwav = torch.nn.utils.rnn.pad_sequence(wavs_, batch_first=True)
    # print("size of multiwav:", multiwav.size())
    wavs = multiwav.float().to(device)

    wav_lens = torch.Tensor(
        [wav.size(), wav1.size()]
    )  # torch.ones([batchnum])#
    wav_lens = wav_lens.to(device)
    encoder_out = model.modules.encoder(wavs, wav_lens)
    # encoder_out.shape [N, T, C], e.g., [1, 82, 768]

    use_k2 = True
    use_LM = True

    if use_LM:

        print(
            "-----------------------HLG_topo result---------------------------"
        )
        logits = model.hparams.ctc_lin(encoder_out)
        # logits.shape [N, T, C], e.g., [1, 82, 5000]
        log_probs = model.hparams.log_softmax(logits)
        # log_probs.shape [N, T, C], e.g., [1, 82, 5000]

        device = log_probs.device
        vocab_size = log_probs.size(2)  # model.tokenizer.vocab_size()
        ctc_topo = build_ctc_topo2(list(range(vocab_size)))
        ctc_topo = k2.create_fsa_vec([ctc_topo]).to(device)

        if not os.path.exists("./HLG.pt"):

            """Step1: Load word list and generate corresponding lexicon from a pretainred ASR model."""
            words_txt = "../data/local/lm/librispeech-vocab.txt"
            excluded = [
                "<eps>",
                "!SIL",
                "<SPOKEN_NOISE>",
                "<UNK>",
                "#0",
                "<s>",
                "</s>",
            ]
            words = read_words(words_txt, excluded)
            lexicon = []
            for word in words[:]:
                pieces = model.tokenizer.encode_as_pieces(word.upper())
                lexicon.append((word, pieces))
            lexicon.append(("<UNK>", ["<unk>"]))
            words = ["<eps>"] + ["<UNK>"] + words + ["#0"]

            """Step2: output token list from the pretainred ASR model."""
            tokens = [
                model.tokenizer.id_to_piece(id)
                for id in range(model.tokenizer.get_piece_size())
            ]

            """Step3: generate Lexicon fst after adding pseudo-phone disambiguation symbols as well as corresponding first disambig ID of tokens and words"""
            (
                first_token_disambig_id,
                first_word_disambig_id,
                L_disambig,
            ) = test_read_lexicon(lexicon, tokens, words)

            """Step4: generate Grammar fst"""
            filename_G3gram = "G.fst.txt"
            if not os.path.exists(filename_G3gram):
                # logging.debug("Loading G.fst.txt")
                G = kaldilm.arpa2fst(
                    input_arpa="../data/local/lm/3-gram.pruned.1e-7.arpa",
                    read_symbol_table="words.txt",
                    disambig_symbol="#0",
                )
                with open(filename_G3gram, "w", encoding="utf-8") as f:
                    f.write(G)

            with open(filename_G3gram) as f:
                G = k2.Fsa.from_openfst(f.read(), acceptor=False)

            """Step5: generate HLG fst"""
            HLG = compile_HLG(
                L=L_disambig,
                G=G,
                H=ctc_topo,
                labels_disambig_id_start=first_token_disambig_id,
                aux_labels_disambig_id_start=first_word_disambig_id,
            )
            torch.save(HLG.as_dict(), "./HLG.pt")

        else:

            print("Loading pre-compiled HLG")
            d = torch.load("./HLG.pt")
            HLG = k2.Fsa.from_dict(d)
            HLG = HLG.to(device)
            HLG.aux_labels = k2.ragged.remove_values_eq(HLG.aux_labels, 0)
            HLG.requires_grad_(False)

        predicted_tokens, scores = HLG_decoding(log_probs, HLG)
        print(predicted_tokens)
        print(scores)

        word_table = k2.SymbolTable.from_file("words.txt")

        for i in range(len(predicted_tokens)):
            predicted_words = [word_table.get(x) for x in predicted_tokens[i]]
            print(predicted_words)

    else:
        if use_k2:

            print(
                "-----------------------ctc_topo result---------------------------"
            )
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

            predicted_tokens, scores = ctc_decoding(log_probs, ctc_topo)
            print(predicted_tokens)
            print(scores)

        else:
            predicted_tokens, scores = model.modules.decoder(
                encoder_out, wav_lens
            )
            print(predicted_tokens)
            print(scores)

        # hyp = model.tokenizer.decode(hyp)
        predicted_words = [
            model.tokenizer.decode_ids(token_seq)
            for token_seq in predicted_tokens
        ]
        print(predicted_words)


if __name__ == "__main__":
    main()
