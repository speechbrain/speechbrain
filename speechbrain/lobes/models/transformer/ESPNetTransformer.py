# Borrow from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/e2e_st_transformer.py

"""Transformer speech translation model (pytorch)."""

from argparse import Namespace
import logging

import torch

from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,
)
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.st_interface import STInterface


class E2E(STInterface, torch.nn.Module):
    """E2E module.
    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options
    """

    def __init__(
        self,
        idim: int,
        odim: int,
        adim: int,
        aheads: int,
        wshare: int,
        ldconv_encoder_kernel_length: int,
        ldconv_usebias: bool,
        eunits: int,
        elayers: int,
        transformer_input_layer: str,
        transformer_encoder_selfattn_layer_type: str,
        transformer_decoder_selfattn_layer_type: str,
        ldconv_decoder_kernel_length: int,
        dunits: int,
        dlayers: int,
        dropout_rate: float = 0.1,
        transformer_attn_dropout_rate: float = 0,
        sos: int = 1,
        eos: int = 2,
        ignore_id: int = 0,
    ):
        """Construct an E2E object.
        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)

        self.encoder = Encoder(
            idim=idim,
            selfattention_layer_type=transformer_encoder_selfattn_layer_type,
            attention_dim=adim,
            attention_heads=aheads,
            conv_wshare=wshare,
            conv_kernel_length=ldconv_encoder_kernel_length,
            conv_usebias=ldconv_usebias,
            linear_units=eunits,
            num_blocks=elayers,
            input_layer=transformer_input_layer,
            dropout_rate=dropout_rate,
            positional_dropout_rate=dropout_rate,
            attention_dropout_rate=transformer_attn_dropout_rate,
        )

        self.decoder = Decoder(
            odim=odim,
            selfattention_layer_type=transformer_decoder_selfattn_layer_type,
            attention_dim=adim,
            attention_heads=aheads,
            conv_wshare=wshare,
            conv_kernel_length=ldconv_decoder_kernel_length,
            conv_usebias=ldconv_usebias,
            linear_units=dunits,
            num_blocks=dlayers,
            dropout_rate=dropout_rate,
            positional_dropout_rate=dropout_rate,
            self_attention_dropout_rate=transformer_attn_dropout_rate,
            src_attention_dropout_rate=transformer_attn_dropout_rate,
        )

        self.pad = 0  # use <blank> for padding
        self.sos = sos
        self.eos = eos
        self.odim = odim
        self.ignore_id = ignore_id

    def forward(self, xs_pad, ilens, ys_pad):
        """E2E forward.
        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        """
        # 1. forward encoder
        xs_pad = xs_pad[:, : max(ilens)]  # for data parallel
        src_mask = (
            make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        )
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)

        # 2. forward decoder
        ys_in_pad, ys_out_pad = add_sos_eos(
            ys_pad, self.sos, self.eos, self.ignore_id
        )
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)

        return hs_pad, hs_mask, pred_pad, pred_mask

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder)

    def encode(self, x):
        """Encode source acoustic features.
        :param ndarray x: source acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        x = torch.as_tensor(x).unsqueeze(0)
        enc_output, _ = self.encoder(x, None)
        return enc_output.squeeze(0)

    def translate(  # noqa: C901
        self, x, trans_args, char_list=None,
    ):
        """Translate input speech.
        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace trans_args: argment Namespace contraining options
        :param list char_list: list of characters
        :return: N-best decoding results
        :rtype: list
        """
        # preprate sos
        if getattr(trans_args, "tgt_lang", False):
            if self.replace_sos:
                y = char_list.index(trans_args.tgt_lang)
        else:
            y = self.sos
        logging.info("<sos> index: " + str(y))
        logging.info("<sos> mark: " + char_list[y])
        logging.info("input lengths: " + str(x.shape[0]))

        enc_output = self.encode(x).unsqueeze(0)

        h = enc_output

        logging.info("encoder output lengths: " + str(h.size(1)))
        # search parms
        beam = trans_args.beam_size
        penalty = trans_args.penalty

        if trans_args.maxlenratio == 0:
            maxlen = h.size(1)
        else:
            # maxlen >= 1
            maxlen = max(1, int(trans_args.maxlenratio * h.size(1)))
        minlen = int(trans_args.minlenratio * h.size(1))
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # initialize hypothesis
        hyp = {"score": 0.0, "yseq": [y]}
        hyps = [hyp]
        ended_hyps = []

        for i in range(maxlen):
            logging.debug("position " + str(i))

            # batchfy
            ys = h.new_zeros((len(hyps), i + 1), dtype=torch.int64)
            for j, hyp in enumerate(hyps):
                ys[j, :] = torch.tensor(hyp["yseq"])
            ys_mask = subsequent_mask(i + 1).unsqueeze(0).to(h.device)

            local_scores = self.decoder.forward_one_step(
                ys, ys_mask, h.repeat([len(hyps), 1, 1])
            )[0]

            hyps_best_kept = []
            for j, hyp in enumerate(hyps):
                local_best_scores, local_best_ids = torch.topk(
                    local_scores[j : j + 1], beam, dim=1
                )

                for j in range(beam):
                    new_hyp = {}
                    new_hyp["score"] = hyp["score"] + float(
                        local_best_scores[0, j]
                    )
                    new_hyp["yseq"] = [0] * (1 + len(hyp["yseq"]))
                    new_hyp["yseq"][: len(hyp["yseq"])] = hyp["yseq"]
                    new_hyp["yseq"][len(hyp["yseq"])] = int(
                        local_best_ids[0, j]
                    )
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x["score"], reverse=True
                )[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug("number of pruned hypothes: " + str(len(hyps)))
            if char_list is not None:
                logging.debug(
                    "best hypo: "
                    + "".join([char_list[int(x)] for x in hyps[0]["yseq"][1:]])
                )

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info("adding <eos> in the last position in the loop")
                for hyp in hyps:
                    hyp["yseq"].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp["yseq"][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp["yseq"]) > minlen:
                        hyp["score"] += (i + 1) * penalty
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and trans_args.maxlenratio == 0.0:
                logging.info("end detected at %d", i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug("remeined hypothes: " + str(len(hyps)))
            else:
                logging.info("no hypothesis. Finish decoding.")
                break

            if char_list is not None:
                for hyp in hyps:
                    logging.debug(
                        "hypo: "
                        + "".join([char_list[int(x)] for x in hyp["yseq"][1:]])
                    )

            logging.debug("number of ended hypothes: " + str(len(ended_hyps)))

        nbest_hyps = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[
            : min(len(ended_hyps), trans_args.nbest)
        ]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform translation "
                "again with smaller minlenratio."
            )
            # should copy becasuse Namespace will be overwritten globally
            trans_args = Namespace(**vars(trans_args))
            trans_args.minlenratio = max(0.0, trans_args.minlenratio - 0.1)
            return self.translate(x, trans_args, char_list)

        logging.info("total log probability: " + str(nbest_hyps[0]["score"]))
        logging.info(
            "normalized log probability: "
            + str(nbest_hyps[0]["score"] / len(nbest_hyps[0]["yseq"]))
        )
        return nbest_hyps

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad, ys_pad_src):
        """E2E attention calculation.
        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :param torch.Tensor ys_pad_src:
            batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights (B, H, Lmax, Tmax)
        :rtype: float ndarray
        """
        self.eval()
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad, ys_pad_src)
        ret = dict()
        for name, m in self.named_modules():
            if (
                isinstance(m, MultiHeadedAttention) and m.attn is not None
            ):  # skip MHA for submodules
                ret[name] = m.attn.cpu().numpy()
        self.train()
        return ret
