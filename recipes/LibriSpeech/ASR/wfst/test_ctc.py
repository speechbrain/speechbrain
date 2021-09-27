#!/usr/bin/env python3
import os
import sys
import glob
import logging
from tqdm import tqdm
from pathlib import Path

import k2

import torch
import torchaudio

from utils.utils import get_texts

from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.utils.metric_stats import ErrorRateStats

from utils.create_csv import create_csv
from utils.lexicon import Lexicon

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    model = EncoderDecoderASR.from_hparams(
            source = 'speechbrain/asr-transformer-transformerlm-librispeech',
            savedir = 'download/am',
            run_opts={'device':'cuda'}
            )
    
    return model

def locate_corpus(*corpus_dirs):
    for d in corpus_dirs:
        if os.path.exists(d):
            return d
    logging.debug(f"Please create a place on your system to put the downloaded Librispeech data "
          "and add it to `corpus_dirs`")
    sys.exit(1)

if __name__ == "__main__":
    # loading the AM 
    model = load_model()
    
    model.device = device
    
    lang_dir = 'data/lang_bpe'
    lexicon = Lexicon(lang_dir)
    max_token_id = max(lexicon.tokens)

    ctc_topo2 = k2.ctc_topo(max_token_id).to(device)

    data_dir = locate_corpus(
        '/ceph-meixu/luomingshuang/audio-data/LibriSpeech'
    )

    test_dirs = ['test-clean', 'test-other']

    wer_metric = ErrorRateStats()

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
                print('length: ', len(lines))
                for line in lines[1:]:
                    items = line.split(',')
                    #print(items)
                    id = items[0]
                    flac = items[2]
                    text = items[4]
                    spk_id = items[3]

                    samples.append((id, flac, text, spk_id))
        
        with torch.no_grad():
            for sample in tqdm(samples):
                #print(sample)
                idx = sample[0]
                #duration = sample[1]
                wav = sample[1]
                txt = sample[2]

                wav_lens = torch.tensor([1.0]).to(device)
                wav_lens = wav_lens.to(device)

                wav, sr = torchaudio.load(wav, channels_first=False)
                wav = model.audio_normalizer(wav, sr)

                wavs = wav.unsqueeze(0).float().to(device)
                encoder_out = model.modules.encoder(wavs, wav_lens)
                ctc_logits = model.hparams.ctc_lin(encoder_out)
                ctc_log_probs = model.hparams.log_softmax(ctc_logits)
                            
                vocab_size = model.tokenizer.get_piece_size()

                supervision_segments = torch.tensor([[0, 0, ctc_log_probs.size(1)]],
                                                  dtype=torch.int32)
                
                indices = torch.argsort(supervision_segments[:, 2], descending=True)

                dense_fsa_vec = k2.DenseFsaVec(ctc_log_probs, supervision_segments)

                lattices = k2.intersect_dense_pruned(ctc_topo2, dense_fsa_vec, 20.0, 8, 30, 10000)
                
                best_paths = k2.shortest_path(lattices, use_double_scores=True)
                aux_labels = best_paths[0].aux_labels

                aux_labels = aux_labels[aux_labels.nonzero().squeeze()]

                aux_labels = aux_labels[:-1]

                hyps = model.tokenizer.decode_ids(aux_labels.tolist())
                
                predicted_words = [str(hyps).split(' ')]
                target_words = [str(txt).split(' ')]
                
                wer_metric.append(idx, predicted_words, target_words)
        
        if test_dir == 'test-clean':
            with open(f'results/test-clean-ctc_topo.txt', 'w') as w:
                wer_metric.write_stats(w)
        
        if test_dir == 'test-other':
            with open(f'results/test-other-ctc_topo.txt', 'w') as w:
                wer_metric.write_stats(w)
