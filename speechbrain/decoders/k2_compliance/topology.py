import logging
from pathlib import Path
from typing import Optional

import k2
import torch

from speechbrain.decoders.k2_compliance.lexicon import LexiconBPE


logger = logging.getLogger(__name__)

class Topology:
    """ This class is used to build/compile HLG and LG.
    """
    def __init__(
            self,
            hparams,
            lexicon: LexiconBPE,
            device: torch.device,
            build_custom_topo: Optional[bool] = False,
        ):
        self.lm_path: str = Path(hparams.lm_path)
        self.lm_path_4gram: str = Path(hparams.lm_path_4gram)
        self.lexicon: LexiconBPE = lexicon
        self.device: torch.device = device
        self._H: k2.Fsa = None
        self._G: k2.Fsa = None
        self._G_4gram: k2.Fsa = None
        self._LG: k2.Fsa = None
        self._HLG: k2.Fsa = None
        self.build_custom_topo = build_custom_topo
    
    @property
    def H(self) -> k2.Fsa:
        """Build/Load H transducer (CTC topology)."""
        if self._H is None:
            self._H = self._get_H()
        return self._H

    @property
    def G(self) -> k2.Fsa:
        """Build/Load G transducer (3-gram language model used in HLG)."""
        if self._G is None:
            self._G = self._get_G(self.lm_path)
            self._G = self._G.to(self.device)  # NOTE: Maybe not needed in GPU.
        return self._G
    
    @property
    def G_4gram(self) -> k2.Fsa:
        """4-gram language model used for whole-lattice or n-best rescoring.
        """
        if self._G_4gram is None:
            self._G_4gram = self._get_G_4gram(self.lm_path_4gram)
        return self._G_4gram
    
    @property
    def LG(self) -> k2.Fsa:
        if self._LG is None:
            self._LG = self._get_LG()
        return self._LG

    @property
    def HLG(self) -> k2.Fsa:
        if self._HLG is None:
            # TODO: If HLG.pt exists then load it.
            self._HLG = self._get_HLG()
        return self._HLG

    def _get_H(self) -> k2.Fsa:
        # NOTE: These need to be on the CPU because 
        #       `treat_epsilons_specially` is True and it has only been implemented
        #       for CPU.
        max_token_id = max(self.lexicon.tokens)
        if self.build_custom_topo:
            H = self.lexicon.build_ctc_topo2().to('cpu')
        else:
            H = k2.ctc_topo(max_token_id).to('cpu')
        return H
    
    def _get_G(self, torch_lm_path: Path, for_rescoring: bool = False) -> k2.Fsa:
        """ Steps:
              1. Get text data
              2. Decide LM architecture
              3. Train LM (e.g. 3-gram with kaldilm)
              4. Build G in k2 format
              5. Return

             Returns:
                G (k2.Fsa): The G transducer.
        """
        # python3 -m kaldilm \
        #     --read-symbol-table="data/lang_phone/words.txt" \
        #     --disambig-symbol='#0' \
        #     --max-order=3 \
        #     $dl_dir/lm/3-gram.pruned.1e-7.arpa > data/lm/G_3_gram.fst.txt
        if not torch_lm_path.is_file():
            # python3 -m kaldilm \
            #     --read-symbol-table="data/lang_phone/words.txt" \
            #     --disambig-symbol='#0' \
            #     --max-order=4 \
            #     $dl_dir/lm/4-gram.arpa > data/lm/G_4_gram.fst.txt
            # fst_path is at the same directory as torch_lm_path and only has a different suffix (.fst.txt)
            fst_path = torch_lm_path.parent / (torch_lm_path.stem + ".fst.txt")
            if not fst_path.is_file():
                raise FileNotFoundError(f"File {fst_path} not found. You need to run the kaldilm command above.")
            with open(fst_path) as f:
                first_word_disambig_id = self.lexicon.word2id["#0"]

                G = k2.Fsa.from_openfst(f.read(), acceptor=False).to(self.device)
                if for_rescoring:
                    # G.aux_labels is not needed in later computations, so
                    # remove it here.
                    del G.aux_labels
                    # CAUTION: The following line is crucial.
                    # Arcs entering the back-off state have label equal to #0.
                    # We have to change it to 0 here.
                    G.labels[G.labels >= first_word_disambig_id] = 0
                    # See https://github.com/k2-fsa/k2/issues/874
                    # for why we need to set G.properties to None
                    G.__dict__["_properties"] = None
                    G = k2.Fsa.from_fsas([G]).to('cpu')  # only used for decoding which is done in cpu
                    G = k2.arc_sort(G)
                torch.save(G.as_dict(), torch_lm_path)
        else:
            device = self.device
            if for_rescoring:
                device = 'cpu'
            d = torch.load(torch_lm_path, map_location=device)
            G = k2.Fsa.from_dict(d).to(device)
        return G
    
    def _get_G_4gram(self, torch_lm_path: str):
        """Build a 4-gram LM.
           
           NOTE: By default returns an LM for whole-lattice-rescoring.
                 If you want to avoid that then check the decode.py icefall script.
        """
        G = self._get_G(torch_lm_path, for_rescoring=True)
        G = k2.add_epsilon_self_loops(G)
        G = k2.arc_sort(G)
        G = G.to(self.device)
        # G.lm_scores is used to replace HLG.lm_scores during
        # LM rescoring.
        G.lm_scores = G.scores.clone()
        return G

    def _get_LG(self) -> k2.Fsa:
        """ Steps:
              1. Get text data
              2. Build Lexicon and L_disambig (simple)
              3. Build G
              4. Check compile_lg in k2 for an implementation.
        """
        lg_path = Path(f"{self.lm_path.parent}/LG.pt")
        if lg_path.exists():
            return k2.Fsa.from_dict(torch.load(lg_path, map_location=self.device))
        first_token_disambig_id = self.lexicon.token_table["#0"]
        first_word_disambig_id = self.lexicon.word_table["#0"]
        # NOTE: These need to be on the CPU because 
        #       `treat_epsilons_specially` is True and it has only been implemented
        #       for CPU.
        L = k2.arc_sort(self.lexicon.L_disambig).to('cpu')
        G = k2.arc_sort(self.G).to('cpu')
        logger.info("Intersecting L and G")
        LG = k2.compose(L, G)
        logger.info(f"LG shape: {LG.shape}")

        logger.info("Connecting LG")
        LG = k2.connect(LG)
        logger.info(f"LG shape after k2.connect: {LG.shape}")

        logger.info("Determinizing LG")
        LG = k2.determinize(LG, k2.DeterminizeWeightPushingType.kLogWeightPushing)
        
        logger.info("Connecting LG after k2.determinize")
        LG = k2.connect(LG)

        logger.info("Removing disambiguation symbols on LG")
        labels = LG.labels
        labels[labels >= first_token_disambig_id] = 0
        LG.labels = labels

        assert isinstance(LG.aux_labels, k2.RaggedTensor)
        LG.aux_labels.values[LG.aux_labels.values >= first_word_disambig_id] = 0

        LG = k2.remove_epsilon(LG)
        logger.info(f"LG shape after k2.remove_epsilon: {LG.shape}")

        LG = k2.connect(LG)
        LG.aux_labels = LG.aux_labels.remove_values_eq(0)

        logger.info("Arc sorting LG")
        LG = k2.arc_sort(LG)
        torch.save(LG.as_dict(), lg_path)

        return LG.to(self.device)

    def _get_HLG(self) -> k2.Fsa:
        hlg_path = Path(f"{self.lm_path.parent}/HLG.pt")
        if hlg_path.exists():
            return k2.Fsa.from_dict(torch.load(hlg_path, map_location=self.device))

        LG = self.LG.to('cpu')
        logger.info("Composing H and LG")
        # CAUTION: The name of the inner_labels is fixed
        # to `tokens`. If you want to change it, please
        # also change other places in icefall that are using
        # it.
        HLG = k2.compose(self.H, LG, inner_labels="tokens")

        logger.info("Connecting HLG")
        HLG = k2.connect(HLG)

        logger.info("Arc sorting HLG")
        HLG = k2.arc_sort(HLG)
        logger.info(f"HLG.shape: {HLG.shape}")

        torch.save(HLG.as_dict(), hlg_path)

        return HLG