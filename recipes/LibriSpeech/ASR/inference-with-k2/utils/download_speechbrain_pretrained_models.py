from speechbrain.pretrained.fetching import fetch
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

hparams_file = 'hyperparams.yaml'
source = 'speechbrain/asr-transformer-transformerlm-librispeech'
savedir = 'data/pretrained_models/asr-transformer-transformerlm-librispeech' 
use_auth_token = False
overrides = {}

hparams_local_path = fetch(hparams_file, source, savedir, use_auth_token)

with open(hparams_local_path) as fin:
    hparams = load_hyperpyyaml(fin, overrides)

pretrainer = ''
pretrainer = hparams['pretrainer']

pretrainer.set_collect_in(savedir)

run_on_main(pretrainer.collect_files, kwargs={"default_source": source})

pretrainer.load_collected(device='cpu')

import os
import sentencepiece as spm


sp = spm.SentencePieceProcessor()
sp.load(os.path.join(savedir,'tokenizer.ckpt'))

sp_length = len(sp)
#sp_length = sp.get_piece_size()

lm_dir = 'data/local/lm'
dst_dir = 'data/local/dict_nosp'

if not os.path.exists(lm_dir):
    os.mkdir(lm_dir)
if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)

librispeech_lexicon = os.path.join(lm_dir, 'librispeech-lexicon.txt')
librispeech_bpe_lexicon = os.path.join(lm_dir, 'librispeech-bpe-lexicon.txt')

print('Building librispeech-bpe-lexicon.txt')

new_lines = []

with open(librispeech_lexicon, 'r') as f1:
    lines = f1.readlines()
    for line in lines:
        items = line.split(' ')
        word = line.split(' ')[0].split('\t')[0].strip('')
        tokens = sp.encode_as_pieces(str(word))
        new_line = str(word) + '  ' + ' '.join(tokens)
        new_lines.append(new_line)
f1.close()

with open(librispeech_bpe_lexicon, 'w') as f2:
    for line in new_lines:
        f2.write(line)
        f2.write('\n')
f2.close()

print('Building librispeech bpe non silence.txt')
with open(os.path.join(dst_dir, 'nonsilence_phones.txt'), 'w') as f3:
    for i in range(3, sp_length):
        bpe_unit = sp.id_to_piece(i)
        f3.write(str(bpe_unit))
        f3.write('\n')
f3.close()
