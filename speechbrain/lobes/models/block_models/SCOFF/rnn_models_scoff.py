import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from speechbrain.lobes.models.block_models.utilities.attention import (
    MultiHeadAttention,
)
from speechbrain.lobes.models.block_models.utilities.set_transformer import (
    SetTransformer,
)
from .blocks_core_scoff import BlocksCore
import matplotlib.pyplot as plt
from speechbrain.lobes.models.block_models.utilities.invariant_modules import (
    PMA,
)


class Identity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input * 1.0

    def backward(ctx, grad_output):
        print(grad_output)
        return grad_output * 1.0


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(
        self,
        rnn_type,
        ntoken,
        ninp,
        nhid,
        nlayers,
        dropout=0.5,
        tie_weights=False,
        use_cudnn_version=False,
        use_adaptive_softmax=False,
        cutoffs=None,
        discrete_input=False,
        n_templates=2,
        share_inp=True,
        share_comm=True,
        memory_mlp=4,
        num_blocks=6,
        update_topk=4,
        memorytopk=4,
        use_gru=False,
        do_rel=False,
        num_modules_read_input=2,
        inp_heads=1,
        device=None,
        memory_slots=4,
        memory_head_size=16,
        num_memory_heads=4,
        attention_out=512,
        version=1,
        step_att=True,
        num_rules=0,
        rule_time_steps=0,
        perm_inv=True,
        application_option=3,
        use_dropout=True,
        rule_selection="gumble",
    ):

        super(RNNModel, self).__init__()
        self.device = device
        self.topk = update_topk
        self.memorytopk = memorytopk
        self.num_modules_read_input = num_modules_read_input
        self.inp_heads = inp_heads
        print("top k blocks, using dropput", self.topk, dropout)
        self.use_cudnn_version = use_cudnn_version
        self.drop = nn.Dropout(dropout)
        print("number of inputs, ninp", ninp)
        self.n_templates = n_templates
        self.do_rel = do_rel
        self.use_dropout = use_dropout

        self.args_to_init_blocks = {
            "ntoken": ntoken,
            "ninp": ninp,
            "use_gru": use_gru,
            "tie_weights": tie_weights,
            "do_rel": do_rel,
            "device": device,
            "memory_slots": memory_slots,
            "memory_head_size": memory_head_size,
            "num_memory_heads": num_memory_heads,
            "share_inp": share_inp,
            "share_comm": share_comm,
            "memory_mlp": memory_mlp,
            "attention_out": attention_out,
            "version": version,
            "step_att": step_att,
            "topk": update_topk,
            "memorytopk": self.memorytopk,
            "num_blocks": num_blocks,
            "n_templates": n_templates,
            "num_modules_read_input": num_modules_read_input,
            "inp_heads": inp_heads,
            "nhid": nhid,
            "perm_inv": perm_inv,
        }
        self.num_blocks = num_blocks
        self.nhid = nhid
        self.block_size = nhid // self.num_blocks
        print("number of blocks", self.num_blocks)
        self.discrete_input = discrete_input
        self.use_adaptive_softmax = use_adaptive_softmax
        self.bc_lst = None
        self.sigmoid = nn.Sigmoid()
        self.decoder = None

        self.rule_selection = rule_selection

        self.application_option = application_option

        self.perm_inv = perm_inv
        print("Dropout rate", dropout)

        self.num_rules = num_rules
        self.rule_time_steps = rule_time_steps

        self.rnn_type = rnn_type

        self.nlayers = nlayers

        self.prior_lst = None
        self.inf_ = None
        self.prior_ = None

        self.init_blocks()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.rule_emb.weight.data.uniform_(-initrange, initrange)
        if not self.use_adaptive_softmax:
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_blocks_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.rule_emb.weight.data.uniform_(-initrange, initrange)
        if not self.use_adaptive_softmax:
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_blocks(self):
        ntoken = self.args_to_init_blocks["ntoken"]
        ninp = self.args_to_init_blocks["ninp"]
        use_gru = self.args_to_init_blocks["use_gru"]
        tie_weights = self.args_to_init_blocks["tie_weights"]
        device = self.args_to_init_blocks["device"]
        do_rel = self.args_to_init_blocks["do_rel"]
        memory_slots = self.args_to_init_blocks["memory_slots"]
        memory_head_size = self.args_to_init_blocks["memory_head_size"]
        num_memory_heads = self.args_to_init_blocks["num_memory_heads"]
        share_inp = self.args_to_init_blocks["share_inp"]
        share_comm = self.args_to_init_blocks["share_comm"]
        memory_mlp = self.args_to_init_blocks["memory_mlp"]
        attention_out = self.args_to_init_blocks["attention_out"]
        version = self.args_to_init_blocks["version"]
        step_att = self.args_to_init_blocks["step_att"]
        topk = self.args_to_init_blocks["topk"]
        memorytopk = self.args_to_init_blocks["memorytopk"]
        num_blocks = self.args_to_init_blocks["num_blocks"]
        n_templates = self.args_to_init_blocks["n_templates"]
        num_modules_read_input = self.args_to_init_blocks[
            "num_modules_read_input"
        ]
        inp_heads = self.args_to_init_blocks["inp_heads"]
        nhid = self.args_to_init_blocks["nhid"]
        perm_inv = self.args_to_init_blocks["perm_inv"]

        if self.discrete_input:
            self.encoder = nn.Embedding(ntoken, ninp)
        else:
            self.encoder = nn.Linear(ntoken, ninp)

        bc_lst = []
        if True:
            if True:
                bc_lst.append(
                    BlocksCore(
                        ninp,
                        nhid,
                        1,
                        num_blocks,
                        topk,
                        memorytopk,
                        step_att,
                        num_modules_read_input,
                        inp_heads,
                        do_gru=use_gru,
                        do_rel=do_rel,
                        perm_inv=perm_inv,
                        device=device,
                        n_templates=n_templates,
                        share_inp=share_inp,
                        share_comm=share_comm,
                        memory_mlp=memory_mlp,
                        memory_slots=memory_slots,
                        num_memory_heads=num_memory_heads,
                        memory_head_size=memory_head_size,
                        attention_out=attention_out,
                        version=version,
                        num_rules=self.num_rules,
                        rule_time_steps=self.rule_time_steps,
                        application_option=self.application_option,
                        rule_selection=self.rule_selection,
                    )
                )
                self.bc_lst = nn.ModuleList(bc_lst)
            else:
                bc_lst.append(
                    BlocksCore(
                        nhid + ninp,
                        nhid,
                        1,
                        num_blocks,
                        topk,
                        memorytopk,
                        step_att,
                        num_modules_read_input,
                        inp_heads,
                        do_gru=use_gru,
                        do_rel=do_rel,
                        perm_inv=perm_inv,
                        device=device,
                        n_templates=n_templates,
                        share_inp=share_inp,
                        share_comm=share_comm,
                        memory_mlp=memory_mlp,
                        memory_slots=memory_slots,
                        num_memory_heads=num_memory_heads,
                        memory_head_size=memory_head_size,
                        attention_out=attention_out,
                        version=version,
                        num_rules=self.num_rules,
                        rule_time_steps=self.rule_time_steps,
                        rule_selection=self.rule_selection,
                    )
                )
                self.bc_lst = nn.ModuleList(bc_lst)

        if True:
            # if self.perm_inv:
            #    print('OUTPUT WILL BE PERMUTATION INVARIANT!')
            #    print('INFO: Since output is permutation invariant, do not reshape the first returned output to represent each module separately.')
            #    print('For example: the first output will be (time_steps, batch_size, per_block_size * num_blocks). Do not reshape it as (time_steps, batch_size, num_blocks, per_block_size)')
            #    self.pma = SetTransformer(self.block_size, 1, self.nhid)

            if tie_weights:
                print("tying weights!")
                if self.nhid != ninp:
                    raise ValueError(
                        "When using the tied flag, "
                        "nhid must be equal to emsize"
                    )
                self.decoder.weight = self.encoder.weight

        # self.init_blocks_weights()
        print("-------Done Initializing Module----------")

    def reparameterize(self, mu, logvar):
        if True:  # self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, input, hidden, message_to_rule_network=None):
        extra_loss = 0.0
        do_print = False

        # input_to_blocks = input.reshape(batch_size, self.block_size * self.num_blocks)
        emb = self.drop(self.encoder(input))
        timesteps, batch_size, _ = emb.shape

        hx, cx = hidden[0], hidden[1]
        if True:
            # for loop implementation with RNNCell
            layer_input = emb
            new_hidden = [[], []]
            for idx_layer in range(0, self.nlayers):
                # print('idx layer', idx_layer)
                output = []
                masklst = []
                bmasklst = []
                template_attn = []

                t0 = time.time()
                self.bc_lst[idx_layer].blockify_params()
                hx, cx = hidden[0][idx_layer], hidden[1][idx_layer]
                if self.do_rel:
                    self.bc_lst[idx_layer].reset_relational_memory(
                        input.shape[1]
                    )
                for idx_step in range(input.shape[0]):
                    hx, cx, mask, bmask, temp_attn = self.bc_lst[idx_layer](
                        layer_input[idx_step],
                        hx,
                        cx,
                        idx_step,
                        do_print=do_print,
                        message_to_rule_network=message_to_rule_network,
                    )
                    output.append(hx)
                    masklst.append(mask)
                    bmasklst.append(bmask)
                    template_attn.append(temp_attn)

                output = torch.stack(output)
                mask = torch.stack(masklst)
                bmask = torch.stack(bmasklst)
                if type(template_attn[0]) != type(None):
                    template_attn = torch.stack(template_attn)
                # print(torch.squeeze(torch.squeeze(bmask, dim=0), dim=-1).sum(dim=0))
                layer_input = output
                new_hidden[0].append(hx)
                new_hidden[1].append(cx)
            new_hidden[0] = torch.stack(new_hidden[0])
            new_hidden[1] = torch.stack(new_hidden[1])
            hidden = tuple(new_hidden)
        block_mask = bmask.squeeze(0)
        assert input.shape[1] == hx.shape[0]
        # if self.use_dropout:
        output = self.drop(output)
        dec = output.view(output.size(0) * output.size(1), self.nhid)

        # if self.perm_inv:
        #    dec = dec.view(-1, self.num_blocks, self.block_size)
        #    dec = self.pma(dec).squeeze(1)
        # dec = Identity().apply(dec)
        dec_ = dec.reshape(output.size(0), output.size(1), dec.size(1))
        return dec_, hidden, extra_loss, block_mask, template_attn

    def init_hidden(self, bsz):
        weight = next(self.bc_lst[0].block_lstm.parameters())
        if True or self.rnn_type == "LSTM" or self.rnn_type == "LSTMCell":
            return (
                weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid),
            )
            # return (weight.new(self.nlayers, bsz, self.nhid).normal_(),
            #        weight.new(self.nlayers, bsz, self.nhid).normal_())
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
