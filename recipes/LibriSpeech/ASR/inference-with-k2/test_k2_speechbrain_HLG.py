#!/usr/bin/env python3

# Copyright (c)  2021  Xiaomi Corporation (authors: Mingshuang Luo)

# Apache 2.0

import os
import csv
import glob
import sys
import logging
import argparse

import torch
import torchaudio

from tqdm import tqdm
from pathlib import Path
from typing import List, Union

import k2
from k2 import Fsa, SymbolTable

from snowfall.common import get_texts
from snowfall.common import str2bool
from snowfall.common import find_first_disambig_symbol
from snowfall.common import get_phone_symbols
from snowfall.common import get_texts
from snowfall.decoding.graph import compile_HLG

from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.utils.metric_stats import ErrorRateStats

from utils.lm_rescore import rescore_with_n_best_list
from utils.lm_rescore import rescore_with_whole_lattice
from utils.create_csv import create_csv
from utils.nbest_decoding import nbest_decoding

logger = logging.getLogger(__name__)
SAMPLERATE = 16000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    model = EncoderDecoderASR.from_hparams(
            source = 'speechbrain/asr-transformer-transformerlm-librispeech',
            savedir = 'data/pretrained_models/asr-transformer-transformerlm-librispeech',)

    model.modules.to(device)
    model.hparams.ctc_lin.to(device)

    return model

def decode(sample: List, 
           model,
           output_beam_size: int,
           num_paths: int,
           HLG: k2.Fsa,
           G: None,
           lm_scale_list: List,
           symbols: SymbolTable,
           use_whole_lattice: bool
           ):

    with torch.no_grad():
        wav = sample[1]
        txt = sample[2]
        wav_lens = torch.tensor([1.0]).to(device)
        
        sig, sr = torchaudio.load(wav, channels_first=False)
        sig = model.audio_normalizer(sig, sr)
        sig = sig.unsqueeze(0).float().to(device)

        encoder_output = model.modules.encoder(sig, wav_lens)
        ctc_logits = model.hparams.ctc_lin(encoder_output)
        ctc_log_probs = model.hparams.log_softmax(ctc_logits)

        supervision_segments = torch.tensor([[0, 0, ctc_log_probs.size(1)]],
                                                    dtype=torch.int32)

        indices = torch.argsort(supervision_segments[:, 2], descending=True)

        dense_fsa_vec = k2.DenseFsaVec(ctc_log_probs, supervision_segments)

        lattices = k2.intersect_dense_pruned(HLG, dense_fsa_vec, 20.0, output_beam_size, 30, 10000)

        if G is None:
            if num_paths > 1:
                best_paths = nbest_decoding(lattices, num_paths)
                key = f'no_resocre-{num_paths}'
            else:
                key = 'no_rescore'
                best_paths = k2.shortest_path(lattices, use_double_scores=True)

            hyps = get_texts(best_paths, indices)
            hyps = ' '.join([symbols.get(x) for x in hyps[0]])
    
            return hyps, txt
        
        logging.debug('use_whole_lattice: ', use_whole_lattice)

        if use_whole_lattice:
            logging.debug(f'Using whole lattice for decoding:')
            best_paths_dict = rescore_with_whole_lattice(lattices, G, lm_scale_list)

        else:
            logging.debug(f'Using nbest paths for decoding:')
            best_paths_dict = rescore_with_n_best_list(lattices, G, num_paths, lm_scale_list)
        
        for lm_scale_str, best_paths in best_paths_dict.items():
            hyps = get_texts(best_paths, indices) 
            hyps = ' '.join([symbols.get(x) for x in hyps[0]])
            
        return hyps, txt

def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--use-lm-rescoring',
        type=str2bool,
        default=True,
        help='When enabled, it uses LM for rescoring')
    
    parser.add_argument(
        '--use-whole-lattice',
        type=str2bool,
        default=True,
        help='When enabled, it uses the whole lattice for decoding.')

    parser.add_argument(
        '--num-paths',
        type=int,
        default=-1,
        help='Number of paths for rescoring using n-best list.' \
             'If it is negative, then rescore with the whole lattice.'\
             'CAUTION: You have to reduce max_duration in case of CUDA OOM'
             )

    parser.add_argument(
        '--output-beam-size',
        type=float,
        default=8,
        help='Output beam size. Used in k2.intersect_dense_pruned.'\
             'Choose a large value (e.g., 20), for 1-best decoding '\
             'and n-best rescoring. Choose a small value (e.g., 8) for ' \
             'rescoring with the whole lattice')
    
    return parser

def locate_corpus(*corpus_dirs):
    for d in corpus_dirs:
        if os.path.exists(d):
            return d
    logging.debug(f"Please create a place on your system to put the downloaded Librispeech data "
          "and add it to `corpus_dirs`")
    sys.exit(1)

def main():
    parser = get_parser()
    args = parser.parse_args()

    num_paths = args.num_paths
    use_lm_rescoring = args.use_lm_rescoring
    use_whole_lattice = args.use_whole_lattice

    lang_dir = Path('data/lang_nosp')

    symbol_table = k2.SymbolTable.from_file(lang_dir / 'words.txt')

    logging.debug("Loading pre-compiled HLG")
    d = torch.load(lang_dir / 'HLG.pt')
    HLG = k2.Fsa.from_dict(d)

    if use_lm_rescoring:
        if use_whole_lattice:
            logging.info('Rescoring with the whole lattice')
        else:
            logging.info(f'Rescoring with n-best list, n is {num_paths}')
        first_word_disambig_id = find_first_disambig_symbol(symbol_table)

        logging.debug('Loading pre-compiled G_4_gram.pt')
        d = torch.load(lang_dir / 'G_4_gram.pt')
        G = k2.Fsa.from_dict(d).to(device)

        if use_whole_lattice:
            # Add epsilon self-loops to G as we will compose
            # it with the whole lattice later
            G = k2.add_epsilon_self_loops(G)
            G = k2.arc_sort(G)
            G = G.to(device)
        # G.lm_scores is used to replace HLG.lm_scores during
        # LM rescoring.
        G.lm_scores = G.scores.clone()
    else:
        logging.debug('Decoding without LM rescoring')
        G = None
        if num_paths > 1:
            logging.debug(f'Use n-best list decoding, n is {num_paths}')
        else:
            logging.debug('Use 1-best decoding')

    logging.debug("convert HLG to device")
    HLG = HLG.to(device)
    HLG.aux_labels = k2.ragged.remove_values_eq(HLG.aux_labels, 0)
    HLG.requires_grad_(False)

    if not hasattr(HLG, 'lm_scores'):
        HLG.lm_scores = HLG.scores.clone()

    model = load_model()

    data_dir = locate_corpus(
        '/export/corpora5/LibriSpeech',
        '/root/fangjun/data/librispeech/LibriSpeech',
        '/kome/luomingshuang/audio-data/LibriSpeech'
    )

    test_dirs = ['test-clean', 'test-other']

    lm_scale_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

    wer_metric = ErrorRateStats()

    for lm_scale in lm_scale_list:
        lm_scale_list = [lm_scale]
        print(f'Using the lm_scale {lm_scale} for decoding...')

        for test_dir in test_dirs:
            samples = []

            csv_file = os.path.join(data_dir, str(test_dir)+'.csv')

            if not os.path.exists(csv_file):
                info_lists = []
                txt_files = glob.glob(os.path.join(data_dir, test_dir, '*', '*', '*.txt'))
                for txt_file in txt_files:
                    with open(txt_file, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            items = line.strip().split(' ')
                            flac = os.path.join(data_dir, test_dir, '/'.join(items[0].split('-')[:2])+'/'+items[0]+'.flac')
                            text = ' '.join(items[1:])
                            spk_id = '-'.join(items[0].split('-')[0:2])
                            id = items[0]

                            samples.append((id, flac, text, spk_id))

                create_csv(data_dir, samples, test_dir)
            
            else:
                with open(csv_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines[1:]:
                        items = line.split(',')
                        id = items[0]
                        flac = items[1]
                        text = items[2]
                        spk_id = items[3]

                        samples.append((id, flac, text, spk_id))

            for sample in tqdm(samples):
                idx = sample[0]
                hyps, ref = decode(sample=sample, model=model, 
                       output_beam_size=args.output_beam_size, 
                       num_paths=args.num_paths, HLG=HLG, G=G, 
                       lm_scale_list=lm_scale_list, symbols=symbol_table,
                       use_whole_lattice=use_whole_lattice)

                pred = [str(hyps).split(' ')]
                grth = [str(ref).split(' ')]
                
                wer_metric.append(idx, pred, grth)

            if use_lm_rescoring:
                if test_dir == 'test-clean':
                    with open(f'results/test-clean-lm-scale-{lm_scale}.txt', 'w') as f:
                        wer_metric.write_stats(f)

                if test_dir == 'test-other':
                    with open(f'results/test-other-lm-scale-{lm_scale}.txt', 'w') as f:
                        wer_metric.write_stats(f)
            else:
                if test_dir == 'test-clean':
                    with open(f'results/test-clean-no-lm-rescoring.txt', 'w') as f:
                        wer_metric.write_stats(f)

                if test_dir == 'test-other':
                    with open(f'results/test-other-no-lm-rescoring.txt', 'w') as f:
                        wer_metric.write_stats(f)

if __name__  == '__main__':
    main()

