#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
# Copyright (c)  2021  Xiaomi Corporation (authors: Mingshuang Luo)

# Apache 2.0

import os
import torch
import logging
from pathlib import Path

import k2
from k2 import Fsa, SymbolTable

from snowfall.common import find_first_disambig_symbol
from snowfall.common import get_phone_symbols
from snowfall.decoding.graph import compile_HLG
from snowfall.training.ctc_graph import build_ctc_topo, build_ctc_topo2

def main():

    # load L, G, symbol_table
    lang_dir = Path('data/lang_nosp')
    #lang_dir = Path('data/lang_nosp')
    symbol_table = k2.SymbolTable.from_file(lang_dir / 'words.txt')
    phone_symbol_table = k2.SymbolTable.from_file(lang_dir / 'phones.txt')
    phone_ids = get_phone_symbols(phone_symbol_table)
    phone_ids_with_blank = [0] + phone_ids
    
    ctc_topo = k2.arc_sort(build_ctc_topo(phone_ids_with_blank))
    #ctc_topo = k2.arc_sort(build_ctc_topo2(list(range(5000))))

    if not os.path.exists(lang_dir / 'HLG.pt'):
        print("Loading L_disambig.fst.txt")
        with open(lang_dir / 'L_disambig.fst.txt') as f:
            L = k2.Fsa.from_openfst(f.read(), acceptor=False)
        print("Loading G.fst.txt")
        with open(lang_dir / 'G.fst.txt') as f:
            G = k2.Fsa.from_openfst(f.read(), acceptor=False)
        first_phone_disambig_id = find_first_disambig_symbol(phone_symbol_table)
        first_word_disambig_id = find_first_disambig_symbol(symbol_table)
        HLG = compile_HLG(L=L,
                         G=G,
                         H=ctc_topo,
                         labels_disambig_id_start=first_phone_disambig_id,
                         aux_labels_disambig_id_start=first_word_disambig_id)
        torch.save(HLG.as_dict(), lang_dir / 'HLG.pt')

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == '__main__':
    main()
