#!/usr/bin/env python3
# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
from tqdm import tqdm
import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import torch
import torchaudio
import torch.nn as nn

from utils.decode import (
    get_lattice,
    nbest_decoding,
    one_best_decoding,
    rescore_with_n_best_list,
    rescore_with_whole_lattice,
)
from utils.lexicon import Lexicon
from utils.utils import (
    AttributeDict,
    get_texts,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=19,
        help="It specifies the checkpoint to use for decoding."
        "Note: Epoch counts from 0.",
    )
    parser.add_argument(
        "--avg",
        type=int,
        default=5,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch'. ",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="whole-lattice-rescoring",
        help="""Decoding method.
        Supported values are:
            - (1) 1best. Extract the best path from the decoding lattice as the
              decoding result.
            - (2) nbest. Extract n paths from the decoding lattice; the path
              with the highest score is the decoding result.
            - (3) nbest-rescoring. Extract n paths from the decoding lattice,
              rescore them with an n-gram LM (e.g., a 4-gram LM), the path with
              the highest score is the decoding result.
            - (4) whole-lattice-rescoring. Rescore the decoding lattice with an
              n-gram LM (e.g., a 4-gram LM), the best path of rescored lattice
              is the decoding result.
        """,
    )

    parser.add_argument(
        "--num-paths",
        type=int,
        default=100,
        help="""Number of paths for n-best based decoding method.
        Used only when "method" is one of the following values:
        nbest, nbest-rescoring
        """,
    )

    parser.add_argument(
        "--lattice-score-scale",
        type=float,
        default=0.5,
        help="""The scale to be applied to `lattice.scores`.
        It's needed if you use any kinds of n-best based rescoring.
        Used only when "method" is one of the following values:
        nbest, nbest-rescoring
        A smaller value results in more unique paths.
        """,
    )

    parser.add_argument(
        "--export",
        type=str2bool,
        default=False,
        help="""When enabled, the averaged model is saved to
        tdnn/exp/pretrained.pt. Note: only model.state_dict() is saved.
        pretrained.pt contains a dict {"model": model.state_dict()},
        which can be loaded by `icefall.checkpoint.load_checkpoint()`.
        """,
    )
    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "exp_dir": Path("results"),
            "lang_dir": Path("data/lang_bpe"),
            "lm_dir": Path("data/lm"),
            "search_beam": 20,
            "output_beam": 5,
            "min_active_states": 30,
            "max_active_states": 10000,
            "use_double_scores": True,
        }
    )
    return params

from speechbrain.pretrained import EncoderDecoderASR

def load_model():
    model = EncoderDecoderASR.from_hparams(
            source = 'speechbrain/asr-transformer-transformerlm-librispeech',
            savedir = 'download/am',
            run_opts = {'device':'cuda'})

    return model

def decode_one_batch(
    params: AttributeDict,
    HLG: k2.Fsa,
    #batch: dict,
    nnet_output: torch.tensor,
    supervision_segments: torch.tensor,
    lexicon: Lexicon,
    G: Optional[k2.Fsa] = None,
) -> Dict[str, List[List[int]]]:
    """Decode one batch and return the result in a dict. The dict has the
    following format:

        - key: It indicates the setting used for decoding. For example,
               if no rescoring is used, the key is the string `no_rescore`.
               If LM rescoring is used, the key is the string `lm_scale_xxx`,
               where `xxx` is the value of `lm_scale`. An example key is
               `lm_scale_0.7`
        - value: It contains the decoding result. `len(value)` equals to
                 batch size. `value[i]` is the decoding result for the i-th
                 utterance in the given batch.
    Args:
      params:
        It's the return value of :func:`get_params`.

        - params.method is "1best", it uses 1best decoding without LM rescoring.
        - params.method is "nbest", it uses nbest decoding without LM rescoring.
        - params.method is "nbest-rescoring", it uses nbest LM rescoring.
        - params.method is "whole-lattice-rescoring", it uses whole lattice LM
          rescoring.

      model:
        The neural model.
      HLG:
        The decoding graph.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
      lexicon:
        It contains word symbol table.
      G:
        An LM. It is not None when params.method is "nbest-rescoring"
        or "whole-lattice-rescoring". In general, the G in HLG
        is a 3-gram LM, while this G is a 4-gram LM.
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict

    device = HLG.device
    # nnet_output is (N, T, C)

    """
    lattice = get_lattice(
        nnet_output=nnet_output,
        HLG=HLG,
        supervision_segments=supervision_segments,
        search_beam=params.search_beam,
        output_beam=params.output_beam,
        min_active_states=params.min_active_states,
        max_active_states=params.max_active_states,
    )

    if params.method in ["1best", "nbest"]:
        if params.method == "1best":
            best_path = one_best_decoding(
                lattice=lattice, use_double_scores=params.use_double_scores
            )
            key = "no_rescore"
        else:
            best_path = nbest_decoding(
                lattice=lattice,
                num_paths=params.num_paths,
                use_double_scores=params.use_double_scores,
                lattice_score_scale=params.lattice_score_scale,
            )
            key = f"no_rescore-{params.num_paths}"
        hyps = get_texts(best_path)
        hyps = [[lexicon.word_table[i] for i in ids] for ids in hyps]
        return {key: hyps}

    assert params.method in ["nbest-rescoring", "whole-lattice-rescoring"]

    lm_scale_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    lm_scale_list += [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    lm_scale_list += [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

    if params.method == "nbest-rescoring":
        best_path_dict = rescore_with_n_best_list(
            lattice=lattice,
            G=G,
            num_paths=params.num_paths,
            lm_scale_list=lm_scale_list,
            lattice_score_scale=params.lattice_score_scale,
        )
    else:
        best_path_dict = rescore_with_whole_lattice(
            lattice=lattice,
            G_with_epsilon_loops=G,
            lm_scale_list=lm_scale_list,
        )

    ans = dict()
    for lm_scale_str, best_path in best_path_dict.items():
        hyps = get_texts(best_path)
        hyps = [[lexicon.word_table[i] for i in ids] for ids in hyps]
        ans[lm_scale_str] = hyps
    return ans


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    HLG: k2.Fsa,
    lexicon: Lexicon,
    G: Optional[k2.Fsa] = None,
) -> Dict[str, List[Tuple[List[int], List[int]]]]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      HLG:
        The decoding graph.
      lexicon:
        It contains word symbol table.
      G:
        An LM. It is not None when params.method is "nbest-rescoring"
        or "whole-lattice-rescoring". In general, the G in HLG
        is a 3-gram LM, while this G is a 4-gram LM.
    Returns:
      Return a dict, whose key may be "no-rescore" if no LM rescoring
      is used, or it may be "lm_scale_0.7" if LM rescoring is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the reference transcript, and the second is the
      predicted result.
    """
    results = []

    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    results = defaultdict(list)
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]

        hyps_dict = decode_one_batch(
            params=params,
            model=model,
            HLG=HLG,
            batch=batch,
            lexicon=lexicon,
            G=G,
        )

        for lm_scale, hyps in hyps_dict.items():
            this_batch = []
            assert len(hyps) == len(texts)
            for hyp_words, ref_text in zip(hyps, texts):
                ref_words = ref_text.split()
                this_batch.append((ref_words, hyp_words))

            results[lm_scale].extend(this_batch)

        num_cuts += len(batch["supervisions"]["text"])

        if batch_idx % 100 == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(
                f"batch {batch_str}, cuts processed until now is {num_cuts}"
            )
    return results


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[List[int], List[int]]]],
):
    test_set_wers = dict()
    for key, results in results_dict.items():
        recog_path = params.exp_dir / f"recogs-{test_set_name}-{key}.txt"
        store_transcripts(filename=recog_path, texts=results)
        logging.info(f"The transcripts are stored in {recog_path}")

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = params.exp_dir / f"errs-{test_set_name}-{key}.txt"
        with open(errs_filename, "w") as f:
            wer = write_error_stats(f, f"{test_set_name}-{key}", results)
            test_set_wers[key] = wer

        logging.info("Wrote detailed error stats to {}".format(errs_filename))

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
    errs_info = params.exp_dir / f"wer-summary-{test_set_name}.txt"
    with open(errs_info, "w") as f:
        print("settings\tWER", file=f)
        for key, val in test_set_wers:
            print("{}\t{}".format(key, val), file=f)

    s = "\nFor {}, WER of different settings are:\n".format(test_set_name)
    note = "\tbest for {}".format(test_set_name)
    for key, val in test_set_wers:
        s += "{}\t{}{}\n".format(key, val, note)
        note = ""
    logging.info(s)

def locate_corpus(*corpus_dirs):
    for d in corpus_dirs:
        if os.path.exists(d):
            return d
    logging.debug(f"Please create a place on your system to put the downloaded Librispeech data "
          "and add it to `corpus_dirs`")
    sys.exit(1)

@torch.no_grad()
def main():
    parser = get_parser() 
    args = parser.parse_args()

    params = get_params()
    params.update(vars(args))

    setup_logger(f"{params.exp_dir}/log/log-decode")
    logging.info("Decoding started")
    logging.info(params)

    lexicon = Lexicon(params.lang_dir)
    max_phone_id = max(lexicon.tokens)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    HLG = k2.Fsa.from_dict(
        torch.load(f"{params.lang_dir}/HLG.pt", map_location="cpu")
    )
    HLG = HLG.to(device)
    assert HLG.requires_grad is False

    if not hasattr(HLG, "lm_scores"):
        HLG.lm_scores = HLG.scores.clone()

    if params.method in ["nbest-rescoring", "whole-lattice-rescoring"]:
        if not (params.lm_dir / "G_4_gram.pt").is_file():
            logging.info("Loading G_4_gram.fst.txt")
            logging.warning("It may take 8 minutes.")
            with open(params.lm_dir / "G_4_gram.fst.txt") as f:
                first_word_disambig_id = lexicon.word_table["#0"]

                G = k2.Fsa.from_openfst(f.read(), acceptor=False)
                # G.aux_labels is not needed in later computations, so
                # remove it here.
                del G.aux_labels
                # CAUTION: The following line is crucial.
                # Arcs entering the back-off state have label equal to #0.
                # We have to change it to 0 here.
                G.labels[G.labels >= first_word_disambig_id] = 0
                G = k2.Fsa.from_fsas([G]).to(device)
                G = k2.arc_sort(G)
                torch.save(G.as_dict(), params.lm_dir / "G_4_gram.pt")
        else:
            logging.info("Loading pre-compiled G_4_gram.pt")
            d = torch.load(params.lm_dir / "G_4_gram.pt", map_location="cpu")
            G = k2.Fsa.from_dict(d).to(device)

        if params.method == "whole-lattice-rescoring":
            # Add epsilon self-loops to G as we will compose
            # it with the whole lattice later
            G = k2.add_epsilon_self_loops(G)
            G = k2.arc_sort(G)
            G = G.to(device)

        # G.lm_scores is used to replace HLG.lm_scores during
        # LM rescoring.
        G.lm_scores = G.scores.clone()
    else:
        G = None

    model = load_model()
   
    data_dir = locate_corpus(
        '/ceph-meixu/luomingshuang/audio-data/LibriSpeech',
    )

    # CAUTION: `test_sets` is for displaying only.
    # If you want to skip test-clean, you have to skip
    # it inside the for loop. That is, use
    #
    #   if test_set == 'test-clean': continue
    #
    test_dirs = ["test-clean", "test-other"]
    
    for test_dir in test_dirs:
        
        results = defaultdict(list)

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
                    flac = items[2]
                    text = items[4]

                    samples.append((id, flac, text))
        
        with torch.no_grad():
            for sample in tqdm(samples):
                idx = sample[0]
                wav = sample[1]
                txt = sample[2]
                #print('txt: ', txt) 
                txt = [txt.rstrip('\n').split(' ')]
                wav_len = torch.tensor([1.0]).to(device)
                wav, sr = torchaudio.load(wav, channels_first=False)
                feature = model.audio_normalizer(wav, sr)

                feature = feature.unsqueeze(0).float().to(device)
                encoder_output = model.modules.encoder(feature, wav_len)
                ctc_logits = model.hparams.ctc_lin(encoder_output)
                nnet_output = model.hparams.log_softmax(ctc_logits)

                supervision_segments = torch.tensor([[0, 0, nnet_output.size(1)]],
                                                      dtype=torch.int32)
                
                hyps_dict = decode_one_batch(
                        params=params, 
                        HLG=HLG,
                        nnet_output=nnet_output,
                        supervision_segments=supervision_segments,
                        lexicon=lexicon,
                        G=G,
                        )
                
                for lm_scale, hyps in hyps_dict.items():
                    this_batch = []
                    #assert len(hyps[0]) == len(text)
                    for hyp_words, ref_text in zip(hyps, txt):
                        this_batch.append((ref_text, hyp_words))

                    results[lm_scale].extend(this_batch)

        save_results(params=params,
                     test_set_name=test_dir,
                     results_dict=results
                             )

    logging.info("Done!")

if __name__ == "__main__":
    main()

