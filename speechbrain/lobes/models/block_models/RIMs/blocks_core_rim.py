import torch
import torch.nn as nn

from speechbrain.lobes.models.block_models.utilities.attention_rim import (
    MultiHeadAttention,
)
from speechbrain.lobes.models.block_models.utilities.BlockGRU import BlockGRU
from speechbrain.lobes.models.block_models.utilities.BlockLSTM import BlockLSTM
from speechbrain.lobes.models.block_models.utilities.sparse_grad_attn import (
    blocked_grad,
)
from speechbrain.lobes.models.block_models.utilities.RuleNetwork import (
    RuleNetwork,
)


"""
Core blocks module.  Takes:
    input: (ts, mb, h)
    hx: (ts, mb, h)
    cx: (ts, mb, h)

    output:
    output, hx, cx

"""


class BlocksCore(nn.Module):
    def __init__(
        self,
        ninp,
        nhid,
        num_blocks_in,
        num_blocks_out,
        topkval,
        step_att,
        do_gru,
        num_modules_read_input=2,
        device=None,
        version=0,
        attention_out=32,
        num_rules=0,
        rule_time_steps=0,
        application_option=3,
        rule_selection="gumble",
    ):
        super(BlocksCore, self).__init__()
        self.nhid = nhid
        self.num_blocks_in = num_blocks_in
        self.num_blocks_out = num_blocks_out
        self.block_size_in = nhid // num_blocks_in
        self.block_size_out = nhid // num_blocks_out
        self.ninp = ninp
        self.topkval = topkval
        self.step_att = step_att
        self.do_gru = do_gru
        self.num_modules_read_input = num_modules_read_input
        self.inp_heads = 1

        print(
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Blocks Core Initialize~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        )
        print("nhid: ", nhid)
        print("num_blocks_in: ", num_blocks_in)
        print("num_blocks_out: ", num_blocks_out)
        print("block_size_in: ", self.block_size_in)
        print("block_size_out: ", self.block_size_out)
        print("topkval: ", topkval)
        print("communication is happening", self.step_att)
        print("inp heads", self.inp_heads)

        self.mha = MultiHeadAttention(
            n_head=4,
            d_model_read=self.block_size_out,
            d_model_write=self.block_size_out,
            d_model_out=self.block_size_out,
            d_k=32,
            d_v=32,
            num_blocks_read=self.num_blocks_out,
            num_blocks_write=self.num_blocks_out,
            topk=self.num_blocks_out,
            grad_sparse=False,
        )

        self.version = version
        if self.version:
            # It supports the flexibility of each module having a sperate encoder.
            self.att_out = self.block_size_out * 1
            self.inp_att = MultiHeadAttention(
                n_head=1,
                d_model_read=self.block_size_out,
                d_model_write=int(self.nhid / self.num_blocks_out),
                d_model_out=self.att_out,
                d_k=64,
                d_v=self.att_out,
                num_blocks_read=1,
                num_blocks_write=num_blocks_in + 1,
                residual=False,
                topk=self.num_blocks_in + 1,
                grad_sparse=False,
                skip_write=True,
            )

        else:
            # this is dummy!
            self.att_out = attention_out
            print("Using version 0 att_out is", self.att_out)
            d_v = self.att_out  # //self.inp_heads
            self.inp_att = MultiHeadAttention(
                n_head=1,
                d_model_read=self.block_size_out,
                d_model_write=ninp,
                d_model_out=self.att_out,
                d_k=64,
                d_v=d_v,
                num_blocks_read=num_blocks_out,
                num_blocks_write=self.num_modules_read_input,
                residual=False,
                dropout=0.1,
                topk=self.num_blocks_in + 1,
                grad_sparse=False,
                skip_write=True,
            )

        # self.att_out = self.block_size_out*4

        # self.inp_att = MultiHeadAttention(n_head=1, d_model_read=self.block_size_out, d_model_write=ninp, d_model_out=self.att_out, d_k=64, d_v=self.att_out, num_blocks_read=num_blocks_out, num_blocks_write=num_modules_read_input,residual=False, topk=self.num_blocks_in+1, grad_sparse=False, skip_write=True)

        if do_gru:
            print("USING GRU!")
            self.block_lstm = BlockGRU(
                self.att_out * self.num_blocks_out,
                self.nhid,
                k=self.num_blocks_out,
            )
        else:
            print("USING LSTM!")
            self.block_lstm = BlockLSTM(
                self.att_out * self.num_blocks_out,
                self.nhid,
                k=self.num_blocks_out,
            )
        self.design_config = {
            "comm": True,
            "grad": False,
            "transformer": True if application_option != 5 else False,
            "application_option": application_option,
            "selection": rule_selection,
        }

        rule_config = {
            "rule_time_steps": rule_time_steps,
            "num_rules": num_rules,
            "rule_emb_dim": 64,
            "rule_query_dim": 32,
            "rule_value_dim": 64,
            "rule_key_dim": 32,
            "rule_heads": 4,
            "rule_dropout": 0.5,
        }
        self.use_rules = (
            rule_config is not None and rule_config["num_rules"] > 0
        )
        if rule_config is not None and rule_config["num_rules"] > 0:
            if True:
                print("Num Rules:" + str(num_rules))
                print("Rule Time Steps:" + str(rule_config["rule_time_steps"]))
                self.rule_network = RuleNetwork(
                    self.block_size_out,
                    num_blocks_out,
                    num_rules=rule_config["num_rules"],
                    rule_dim=rule_config["rule_emb_dim"],
                    query_dim=rule_config["rule_query_dim"],
                    value_dim=rule_config["rule_value_dim"],
                    key_dim=rule_config["rule_key_dim"],
                    num_heads=rule_config["rule_heads"],
                    dropout=rule_config["rule_dropout"],
                    design_config=self.design_config,
                ).to(device)
            # else:
            #    self.rule_network = RuleNetwork(self.block_size_out, num_blocks_out, num_rules = rule_config['num_rules'], rule_dim = rule_config['rule_emb_dim'], query_dim = rule_config['rule_query_dim'], value_dim = rule_config['rule_value_dim'], key_dim = rule_config['rule_key_dim'], num_heads = rule_config['rule_heads'], dropout = rule_config['rule_dropout']).to(device)

            self.rule_time_steps = rule_config["rule_time_steps"]

        self.device = device

    def blockify_params(self):
        self.block_lstm.blockify_params()

    def forward(
        self,
        inp,
        hx,
        cx,
        step,
        do_print=False,
        do_block=True,
        message_to_rule_network=None,
    ):

        hxl = []
        cxl = []

        inp_use = inp  # layer_input[idx_step]
        batch_size = inp.shape[0]
        sz_b = batch_size

        def _process_input(_input):
            _input = _input.unsqueeze(1)

            return torch.cat(
                [_input, torch.zeros_like(_input[:, 0:1, :])], dim=1
            )

        if self.version:
            input_to_attention = [
                _process_input(_input)
                for _input in torch.chunk(
                    inp_use, chunks=self.num_blocks_out, dim=1
                )
            ]

            split_hx = [
                chunk.unsqueeze(1)
                for chunk in torch.chunk(hx, chunks=self.num_blocks_out, dim=1)
            ]

            output = [
                self.inp_att(q=_hx, k=_inp, v=_inp)
                for _hx, _inp in zip(split_hx, input_to_attention)
            ]

            inp_use_list, iatt_list, _ = zip(*output)

            inp_use = torch.cat(inp_use_list, dim=1)
            iatt = torch.cat(iatt_list, dim=1)

            inp_use = inp_use.reshape(
                (inp_use.shape[0], self.att_out * self.num_blocks_out)
            )
        else:
            # use attention here.
            # inp_use = inp_use.reshape((inp_use.shape[0], self.num_blocks_in, self.block_size_in))

            inp_use = inp_use.reshape(
                (inp_use.shape[0], self.num_blocks_in, self.ninp)
            )

            inp_use = inp_use.repeat(1, self.num_modules_read_input - 1, 1)
            inp_use = torch.cat(
                [torch.zeros_like(inp_use[:, 0:1, :]), inp_use], dim=1
            )
            batch_size = inp.shape[0]
            inp_use, iatt, _ = self.inp_att(
                hx.reshape(
                    (hx.shape[0], self.num_blocks_out, self.block_size_out)
                ),
                inp_use,
                inp_use,
            )
            iatt = iatt.reshape(
                (self.inp_heads, sz_b, iatt.shape[1], iatt.shape[2])
            )
            iatt = iatt.mean(0)

            inp_use = inp_use.reshape(
                (inp_use.shape[0], self.att_out * self.num_blocks_out)
            )

        # inp_use = inp_use.reshape((inp_use.shape[0], self.num_blocks_in, self.ninp))
        # inp_use = inp_use.repeat(1,self.num_modules_read_input-1,1)
        # inp_use = torch.cat([torch.zeros_like(inp_use[:,0:1,:]), inp_use], dim=1)

        # inp_use, iatt, _ = self.inp_att(hx.reshape((hx.shape[0], self.num_blocks_out, self.block_size_out)), inp_use, inp_use)
        # inp_use = inp_use.reshape((inp_use.shape[0], self.att_out*self.num_blocks_out))
        # null_score = iatt.mean((0,1))[1]

        new_mask = torch.ones_like(iatt[:, :, 0])
        bottomk_indices = torch.topk(
            iatt[:, :, 0],
            dim=1,
            sorted=True,
            largest=True,
            k=self.num_blocks_out - self.topkval,
        )[1]
        new_mask.index_put_(
            (
                torch.arange(bottomk_indices.size(0)).unsqueeze(1),
                bottomk_indices,
            ),
            torch.zeros_like(bottomk_indices[0], dtype=new_mask.dtype),
        )

        mask = new_mask

        mask = (
            mask.reshape((inp_use.shape[0], self.num_blocks_out, 1))
            .repeat((1, 1, self.block_size_out))
            .reshape(
                (inp_use.shape[0], self.num_blocks_out * self.block_size_out)
            )
        )
        mask = mask.detach()

        hx_old = hx * 1.0
        cx_old = cx * 1.0

        if self.do_gru:
            hx_new, _ = self.block_lstm(inp_use, hx)
            cx_new = hx_new
        else:
            hx_new, cx_new, _ = self.block_lstm(inp_use, hx, cx)

        # Communication b/w different Blocks
        if do_block:
            if self.step_att:
                hx_new = hx_new.reshape(
                    (hx_new.shape[0], self.num_blocks_out, self.block_size_out)
                )
                hx_new_grad_mask = blocked_grad.apply(
                    hx_new,
                    mask.reshape(
                        (
                            mask.shape[0],
                            self.num_blocks_out,
                            self.block_size_out,
                        )
                    ),
                )
                hx_new_att, attn_out, extra_loss_att = self.mha(
                    hx_new_grad_mask, hx_new_grad_mask, hx_new_grad_mask
                )
                hx_new = hx_new + hx_new_att
                hx_new = hx_new.reshape((hx_new.shape[0], self.nhid))
                extra_loss = extra_loss_att

            if self.use_rules:
                hx_new = hx_new.reshape(
                    (hx_new.shape[0], self.num_blocks_out, self.block_size_out)
                )
                for r in range(self.rule_time_steps):
                    hx_new = (
                        self.rule_network(
                            hx_new,
                            message_to_rule_network=message_to_rule_network,
                        )
                        + hx_new
                    )
                hx_new = hx_new.reshape(
                    (hx_new.shape[0], self.num_blocks_out * self.block_size_out)
                )

        hx = (mask) * hx_new + (1 - mask) * hx_old
        cx = (mask) * cx_new + (1 - mask) * cx_old

        return hx, cx, mask


if __name__ == "__main__":
    bc = BlocksCore(512, 1, 4, 4)

    inp = torch.randn(10, 512)
    hx = torch.randn(10, 512)
    cx = torch.randn(10, 512)

    hx, cx = bc(inp, hx, cx)

    print("hx cx shape", hx.shape, cx.shape)
