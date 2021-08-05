import k2
import os
import torch
from pathlib import Path
from snowfall.common import find_first_disambig_symbol

def main():
    lang_dir = Path('data/lang_nosp')
    if not os.path.exists(lang_dir / 'HLG.pt'):
        symbol_table = k2.SymbolTable.from_file('data/lang_nosp/words.txt')

        first_word_disambig_id = find_first_disambig_symbol(symbol_table)

        device = torch.device('cuda')

        with open('data/lang_nosp/G_4_gram.fst.txt') as f:
            G = k2.Fsa.from_openfst(f.read(), acceptor=False)
            del G.aux_labels
            G.labels[G.labels >= first_word_disambig_id] = 0
            G = k2.create_fsa_vec([G]).to(device)
            G = k2.arc_sort(G)
            torch.save(G.as_dict(), 'data/lang_nosp/G_4_gram.pt')

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == '__main__':
    main()
