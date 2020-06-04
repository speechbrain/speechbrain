"""
Alignment code

Author
------
Elena Rastorgueva 2020
"""

import torch


def log_matrix_multiply_max(A, b):
    """
    accounts for the fact that the first dimension is batch_size
    """
    b = b.unsqueeze(1)
    x, argmax = torch.max(A + b, dim=2)
    return x, argmax


class ViterbiAligner(torch.nn.Module):
    def __init__(self, output_folder):
        super().__init__()
        self.output_folder = output_folder
        self.neg_inf = -1e5
        self.align_dict = {}

    def _make_pi_prob(self, phn_lens_abs):
        """
        create tensor of initial probabilities
        """
        batch_size = len(phn_lens_abs)
        U_max = int(phn_lens_abs.max().item())

        pi_prob = self.neg_inf * torch.ones([batch_size, U_max])
        pi_prob[:, 0] = 0

        return pi_prob

    def _make_trans_prob(self, phn_lens_abs):
        """
        create tensor of transition probabilities
        possible transitions: to same state or next state in phn sequence
        dimensions: [batcch_size, from, to]
        """
        # useful values for later
        U_max = int(phn_lens_abs.max().item())
        batch_size = len(phn_lens_abs)
        device = phn_lens_abs.device

        ## trans_prob matrix consists of 2 diagonals:
        ## (1) offset diagonal (next state) &
        ## (2) main diagonal (self-loop)
        # make offset diagonal
        trans_prob_off_diag = torch.eye(U_max - 1)
        zero_side = torch.zeros([U_max - 1, 1])
        zero_bottom = torch.zeros([1, U_max])
        trans_prob_off_diag = torch.cat((zero_side, trans_prob_off_diag), 1)
        trans_prob_off_diag = torch.cat((trans_prob_off_diag, zero_bottom), 0)

        # make main diagonal
        trans_prob_main_diag = torch.eye(U_max)

        # join the diagonals and repeat for whole batch
        trans_prob = trans_prob_off_diag + trans_prob_main_diag
        trans_prob = (
            trans_prob.reshape(1, U_max, U_max)
            .repeat(batch_size, 1, 1)
            .to(device)
        )

        # clear probabilities for too-long sequences
        mask_a = torch.arange(U_max).to(device)[None, :] < phn_lens_abs[:, None]
        mask_a = mask_a.unsqueeze(2)
        mask_a = mask_a.expand(-1, -1, U_max)
        mask_b = mask_a.permute(0, 2, 1)
        trans_prob = trans_prob * (mask_a & mask_b).float()

        ## put -infs in place of zeros:
        trans_prob = torch.where(
            trans_prob == 1, trans_prob, torch.tensor(-float("Inf")).to(device)
        )

        ## normalize
        trans_prob = torch.nn.functional.log_softmax(trans_prob, dim=2)

        ## set nans to v neg numbers
        trans_prob[trans_prob != trans_prob] = self.neg_inf
        ## set -infs to v neg numbers
        trans_prob[trans_prob == -float("Inf")] = self.neg_inf

        return trans_prob

    def _make_emiss_pred_useful(
        self, emission_pred, lens_abs, phn_lens_abs, phns
    ):
        """
        creates a 'useful' form of the posterior probabilities, rearranged into order of phoneme appearance
        """
        U_max = int(phn_lens_abs.max().item())
        fb_max_length = int(lens_abs.max().item())
        device = emission_pred.device

        # apply mask based on lens_abs
        mask_lens = (
            torch.arange(fb_max_length).to(device)[None, :] < lens_abs[:, None]
        )
        emiss_pred_acc_lens = torch.where(
            mask_lens[:, :, None],
            emission_pred,
            torch.tensor([self.neg_inf]).to(device),
        )

        # manipulate phn tensor, and then 'torch.gather'
        phns = phns.to(device)
        phns_copied = phns.unsqueeze(1).expand(-1, fb_max_length, -1)
        emiss_pred_useful = torch.gather(emiss_pred_acc_lens, 2, phns_copied)

        # apply mask based on phn_lens_abs
        mask_phn_lens = (
            torch.arange(U_max).to(device)[None, :] < phn_lens_abs[:, None]
        )
        emiss_pred_useful = torch.where(
            mask_phn_lens[:, None, :],
            emiss_pred_useful,
            torch.tensor([self.neg_inf]).to(device),
        )

        emiss_pred_useful = emiss_pred_useful.permute(0, 2, 1)

        return emiss_pred_useful

    def _calc_viterbi_alignments(
        self,
        pi_prob,
        trans_prob,
        emiss_pred_useful,
        lens_abs,
        phn_lens_abs,
        phns,
    ):
        # useful values
        batch_size = len(phn_lens_abs)
        U_max = phn_lens_abs.max()
        fb_max_length = lens_abs.max()
        device = emiss_pred_useful.device

        v_matrix = self.neg_inf * torch.ones(
            [batch_size, U_max, fb_max_length]
        ).to(device)
        backpointers = -99 * torch.ones([batch_size, U_max, fb_max_length])

        # for cropping v_matrix later
        phn_len_mask = torch.arange(U_max)[None, :].to(device) < phn_lens_abs[
            :, None
        ].to(device)

        # initialise
        v_matrix[:, :, 0] = pi_prob + emiss_pred_useful[:, :, 0]

        for t in range(2, fb_max_length + 1):  # note: t here is 1+ indexing
            x, argmax = log_matrix_multiply_max(
                trans_prob.permute(0, 2, 1), v_matrix[:, :, t - 2]
            )
            v_matrix[:, :, t - 1] = x + emiss_pred_useful[:, :, t - 1]

            # crop v_matrix
            v_matrix = torch.where(
                phn_len_mask[:, :, None],
                v_matrix,
                torch.tensor(self.neg_inf).to(device),
            )

            backpointers[:, :, t - 1] = argmax.type(torch.FloatTensor)

        z_stars = []
        z_stars_loc = []

        for utterance_in_batch in range(batch_size):
            len_abs = lens_abs[utterance_in_batch]
            U = phn_lens_abs[utterance_in_batch].long().item()

            z_star_i_loc = [U - 1]
            z_star_i = [phns[utterance_in_batch, z_star_i_loc[0]].item()]
            for time_step in range(len_abs, 1, -1):
                current_best_loc = z_star_i_loc[0]

                earlier_best_loc = (
                    backpointers[
                        utterance_in_batch, current_best_loc, time_step - 1
                    ]
                    .long()
                    .item()
                )
                earlier_z_star = phns[
                    utterance_in_batch, earlier_best_loc
                ].item()

                z_star_i_loc.insert(0, earlier_best_loc)
                z_star_i.insert(0, earlier_z_star)
            z_stars.append(z_star_i)
            z_stars_loc.append(z_star_i_loc)

        #            print("batch alignment statistics:")
        #            print("phn_lens_abs:", phn_lens_abs)
        #            print("lens_abs:", lens_abs)
        #            print("z_stars_loc:", z_stars_loc)
        #            print("z_stars:", z_stars)

        # picking out viterbi_scores
        viterbi_scores = v_matrix[
            torch.arange(batch_size), phn_lens_abs - 1, lens_abs - 1
        ]

        return z_stars, z_stars_loc, viterbi_scores

    def forward(self, emission_pred, lens, phns, phn_lens):
        lens_abs = torch.round(emission_pred.shape[1] * lens).long()
        phn_lens_abs = torch.round(phns.shape[1] * phn_lens).long()
        phns = phns.long()

        pi_prob = self._make_pi_prob(phn_lens_abs)
        trans_prob = self._make_trans_prob(phn_lens_abs)
        emiss_pred_useful = self._make_emiss_pred_useful(
            emission_pred, lens_abs, phn_lens_abs, phns
        )

        alignments, _, viterbi_scores = self._calc_viterbi_alignments(
            pi_prob, trans_prob, emiss_pred_useful, lens_abs, phn_lens_abs, phns
        )

        return alignments, viterbi_scores

    def store_alignments(self, ids, alignments):
        for i, id in enumerate(ids):
            alignment_i = alignments[i]
            self.align_dict[id] = alignment_i

    def _get_flat_start_batch(self, lens_abs, phns, phn_lens_abs):
        phns = phns.long()

        batch_size = len(lens_abs)
        fb_max_length = torch.max(lens_abs)

        flat_start_batch = torch.zeros(batch_size, fb_max_length).long()
        for i in range(batch_size):
            utter_phns = phns[i]
            utter_phns = utter_phns[: phn_lens_abs[i]]  # crop out zero padding
            repeat_amt = int(lens_abs[i] / len(utter_phns))

            utter_phns = utter_phns.repeat_interleave(repeat_amt)

            # len(utter_phns) now may not equal lens[i]
            # pad out with final phoneme to make lengths equal
            utter_phns = torch.nn.functional.pad(
                utter_phns,
                (0, int(lens_abs[i]) - len(utter_phns)),
                value=utter_phns[-1],
            )

            flat_start_batch[i, : len(utter_phns)] = utter_phns

        return flat_start_batch

    def _get_viterbi_batch(self, ids, lens_abs, phns, phn_lens_abs):
        batch_size = len(lens_abs)
        fb_max_length = torch.max(lens_abs)

        viterbi_batch = torch.zeros(batch_size, fb_max_length).long()
        for i in range(batch_size):
            viterbi_preds = self.align_dict[ids[i]]
            viterbi_preds = torch.tensor(viterbi_preds)
            viterbi_preds = torch.nn.functional.pad(
                viterbi_preds, (0, fb_max_length - len(viterbi_preds))
            )

            viterbi_batch[i] = viterbi_preds.long()

        return viterbi_batch

    def get_prev_alignments(self, ids, emission_pred, lens, phns, phn_lens):
        lens_abs = torch.round(emission_pred.shape[1] * lens).long()
        phn_lens_abs = torch.round(phns.shape[1] * phn_lens).long()

        if ids[0] in self.align_dict:
            return self._get_viterbi_batch(ids, lens_abs, phns, phn_lens_abs)
        else:
            return self._get_flat_start_batch(lens_abs, phns, phn_lens_abs)
