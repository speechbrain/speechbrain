"""
WFST-based support for ASR

Authors
 * Abdelwahab Heba 2021
"""

import torch
import logging
import os

try:
    import k2
except ImportError:
    err_msg = "The optional dependency K2 is needed to use this module\n"
    err_msg += "Cannot import k2. To use WFSTs into autograd-based ML\n"
    err_msg += "Please follow the instructions below\n"
    err_msg += "=============================\n"
    err_msg += "pip install k2\n"
    err_msg += "for more information please refer to\n"
    err_msg += "https://github.com/k2-fsa/k2"
    raise ImportError(err_msg)

try:
    import kaldilm
except ImportError:
    err_msg = "The optional dependency kaldilm is needed to use this module\n"
    err_msg += "Cannot import kaldilm. To use arpa2fst\n"
    err_msg += "Please follow the instructions below\n"
    err_msg += "=============================\n"
    err_msg += "pip install kaldilm\n"
    raise ImportError(err_msg)

from k2 import Fsa
from kaldilm import arpa2fst

def make_TokenFst(token_symbol_table,
                keep_isymbols=False,
                keep_osymbols=False,
                arc_sort=True,
                sort_type='olabel'):
    print("T.fst")
    #T = k2.Ksa.from_str(arcs)
    #T = k2.arc_sort(T,sort_type=sort_type)

def make_LexiconFst():
    print("L.fst")


# example - OK
# TODO if G.fst.txt exist
#make_GrammarFst('/workspace/k2/update/speechbrain/recipes/LibriSpeech/ASR/inference-with-k2/data/local/lm',
#                    'G.fst.txt',
#                    '',
#                    '<s>',
#                    '#0',
#                    '</s>',
#                    True,
#                    False,
#                    30,
#                    '/workspace/k2/update/speechbrain/recipes/LibriSpeech/ASR/inference-with-k2/data/lang_nosp/words.txt',
#                    '',
#                    -1)
def make_GrammarFst(input_arpa,
             text_output_fst,
             binary_output_fst='',
             bos_symbol='<s>',
             disambig_symbol= '',
             eos_symbol='</s>',
             ilabel_sort=True,
             keep_symbols=False,
             max_arpa_warnings=30,
             read_symbol_table='',
             write_symbol_table='',
             max_order=-1):
    # check if G.fst.txt exist
    if os.path.isfile(text_output_fst):
        logging.info(f"File {text_output_fst} already exist")
        return
    # convert arpa LM 2 fst
    fst_text_format = arpa2fst(input_arpa=input_arpa,
             output_fst=binary_output_fst,
             bos_symbol=bos_symbol,
             disambig_symbol=disambig_symbol,
             eos_symbol=eos_symbol,
             ilabel_sort=ilabel_sort,
             keep_symbols=keep_symbols,
             max_arpa_warnings=max_arpa_warnings,
             read_symbol_table=read_symbol_table,
             write_symbol_table=write_symbol_table,
             max_order=max_order)
    logging.info(f"Save G at {text_output_fst}")
    with open(text_output_fst,'w') as f:
        f.write(fst_text_format)
    logging.info("To load G:\n" +
                 f"with open('{text_output_fst}') as f:\n" +
                "\tG = k2.Fsa.from_openfst(f.read(), acceptor=False)"
                )