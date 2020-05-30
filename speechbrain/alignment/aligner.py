"""
Alignment code

Author
------
Elena Rastorgueva 2020
"""

import torch

def log_matrix_multiply_max(A, b):
    """
    also accounts for the fact that the first dimension is batch_size
    """
    inside_size = A.shape[2]
    b = b.unsqueeze(1)#.expand(-1, inside_size, -1)

    #print('A shape:', A.shape)
    #print('b shape:', b.shape)

    x, argmax = torch.max(A + b, dim = 2)
    return x, argmax

class ViterbiAligner:
    def __init__(self, output_folder):
        self.output_folder = output_folder
        self.neg_inf = -1e-5
   
    def update_from_forward(self, emission_pred, lens):
        self.emission_pred = emission_pred
        self.lens = lens

    def make_pi_prob(self, phn_lens_abs):
        """
        create tensor of initial probabilities
        """
        print('called make pi prob')
        batch_size = len(phn_lens_abs)
        U_max = int(phn_lens_abs.max().item())

        pi_prob = self.neg_inf * torch.ones([batch_size, U_max])
        pi_prob[:, 0] = 0

        return pi_prob

    def make_trans_prob(self, phn_lens_abs): 
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
        trans_prob_off_diag = torch.eye(U_max-1)
        zero_side = torch.zeros([U_max-1, 1])
        zero_bottom = torch.zeros([1, U_max])
        trans_prob_off_diag = torch.cat((zero_side, trans_prob_off_diag), 1)
        trans_prob_off_diag = torch.cat((trans_prob_off_diag, zero_bottom), 0)

        # make main diagonal
        trans_prob_main_diag = torch.eye(U_max)

        # join the diagonals and repeat for whole batch
        trans_prob = trans_prob_off_diag + trans_prob_main_diag
        trans_prob = trans_prob.reshape(1, U_max, U_max).repeat(batch_size, 1, 1).to(device)

        # clear probabilities for too-long sequences
        mask_a = torch.arange(U_max).to(device)[None, :] < phn_lens_abs[:, None] # TODO: check if U_max is correct
        mask_a = mask_a.unsqueeze(2)
        mask_a = mask_a.expand(-1, -1, U_max)
        mask_b = mask_a.permute(0, 2, 1)
        trans_prob = trans_prob * (mask_a & mask_b).float()

        ## put -infs in place of zeros:
        trans_prob = torch.where(trans_prob == 1, trans_prob,
            torch.tensor(-float("Inf")).to(device))
        #torch.tensor(torch.finfo(torch.float).min).to(device))

        # print('my trans_prob', trans_prob)

        ## normalize
        trans_prob = torch.nn.functional.log_softmax(trans_prob, dim = 2) # should be dim = 2, I think

        ## set nans to v neg numbers
        trans_prob[trans_prob != trans_prob] = -1e5
        ## set -infs to v neg numbers
        trans_prob[trans_prob == -float('Inf')] = -1e5

        return trans_prob


    def make_emiss_pred_useful(self, emission_pred, lens_abs, phn_lens_abs, phns):
        """
        creates a 'useful' form of the posterior probabilities, rearranged into order of phoneme appearance
        """
        U_max = int(phn_lens_abs.max().item())
        batch_size = len(phn_lens_abs)
        fb_max_length = int(lens_abs.max().item())
        batch_size = len(phn_lens_abs)
        device = emission_pred.device

        # make mask based on fbank_lengths
        mask = torch.arange(fb_max_length).to(device)[None, :] < lens_abs[:, None]

        emission_pred_acc_x_length = torch.where(mask[:, :, None], \
                    emission_pred, torch.tensor([-1e-38]).to(device)) # was -float("Inf"), changed & nan errors stopped (nan from logsumexpbackward)

        # create "zero_plane" for next bit:
        zero_plane = torch.unsqueeze(-1e-38*torch.ones([batch_size, fb_max_length]), dim = 2).to(device)

        # put "zero_plane" at beginning of the emission probabilities to be put in
        # the place of the 'padding' phoneme (which has the index 0)
        
        print(zero_plane.shape)
        print(emission_pred_acc_x_length.shape)
        emiss_pred_with_zeros = torch.cat((zero_plane, emission_pred_acc_x_length), 2)

        phns = phns.to(device)
        # manipulate y tensor, and then 'torch.gather'
        phns_copied = phns.unsqueeze(1).expand(-1, fb_max_length, -1)#.to(device)

        emiss_pred_useful = torch.gather(emiss_pred_with_zeros, 2, phns_copied)
        emiss_pred_useful = emiss_pred_useful.permute(0, 2, 1)
        #print('emiss_pred_useful:', emiss_pred_useful.shape)
        return emiss_pred_useful



    def _calc_viterbi_alignments(self, pi_prob, trans_prob, emiss_pred_useful, lens_abs, phn_lens_abs, phns):
        # useful values
        batch_size = len(phn_lens_abs)
        U_max = phn_lens_abs.max()
        fb_max_length = lens_abs.max()
        device = emiss_pred_useful.device

        v_matrix = self.neg_inf * torch.ones([batch_size, U_max, fb_max_length]).to(device)
        backpointers = -99 * torch.ones([batch_size, U_max, fb_max_length])

        # for cropping v_matrix later
        phn_len_mask = torch.arange(U_max)[None, :].to(device) < phn_lens_abs[:, None].to(device)
        
        # initialise
        v_matrix[:, :, 0] = pi_prob + emiss_pred_useful[:, :, 0]

        for t in range(2, fb_max_length + 1): # note: t here is 1+ indexing
            x, argmax = log_matrix_multiply_max(trans_prob.permute(0, 2, 1), v_matrix[:, :, t-2])
            v_matrix[:, :, t-1] = x + emiss_pred_useful[:, :, t-1]

            # crop v_matrix
            v_matrix = torch.where(phn_len_mask[:, :, None], v_matrix, torch.tensor(-99999.).to(device))

            backpointers[:, :, t-1] = argmax.type(torch.FloatTensor)

        
        z_stars = []

        for utterance_in_batch in range(batch_size):
            len_abs = lens_abs[utterance_in_batch]
            U = phn_lens_abs[utterance_in_batch].long().item()

            loc_of_z_star_i = [U-1]
            z_star_i = [phns[utterance_in_batch, loc_of_z_star_i[0]].item()]
            for time_step in range(len_abs, 1, -1):
                current_best_loc = loc_of_z_star_i[0]

                earlier_best_loc = backpointers[utterance_in_batch, current_best_loc, time_step-1].long().item()
                earlier_z_star = phns[utterance_in_batch, earlier_best_loc].item()

                loc_of_z_star_i.insert(0, earlier_best_loc)
                z_star_i.insert(0, earlier_z_star)
            print('loc of z star i:', loc_of_z_star_i)
            z_stars.append(z_star_i)
            
        return z_stars

    def calc_viterbi_alignments(self, phns, phn_lens):
        lens_abs = torch.round(self.emission_pred.shape[1] * self.lens).long()
        print('lens_abs', lens_abs)

        phn_lens_abs = torch.round(phns.shape[1] * phn_lens).long()
        print('phn_lens_abs', phn_lens_abs)

        phns = phns.long()

        pi_prob = self.make_pi_prob(phn_lens_abs)
        trans_prob = self.make_trans_prob(phn_lens_abs)
        emiss_pred_useful = self.make_emiss_pred_useful(self.emission_pred, lens_abs, phn_lens_abs, phns)
        
        alignments = self._calc_viterbi_alignments(pi_prob, trans_prob, emiss_pred_useful, lens_abs, phn_lens_abs, phns)    
        return alignments 
        

